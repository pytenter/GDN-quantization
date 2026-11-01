"""Exactly60 new trajectories on four owned TP1 lanes. No automatic GPU retry."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C
from archive_cpu import publish


def validate_jobs(jobs):
    assert len(jobs)==60 and len({x['id'] for x in jobs})==60
    assert {x['period'] for x in jobs}=={8,16} and all(x['seed']==1 and x['max_new_tokens']==81920 for x in jobs)
    assert all(x['gpu'] in range(4) and x['gpu_uuid']==C.UUIDS[x['gpu']] for x in jobs)
    for p in (8,16):assert len([x for x in jobs if x['period']==p])==30
    pairs={}
    for x in jobs:pairs.setdefault(x['problem_id'],[]).append(x)
    assert len(pairs)==30
    for rows in pairs.values():assert len(rows)==2 and len({x['gpu'] for x in rows})==1 and len({x['input_ids_sha256'] for x in rows})==1


def run(a):
    C.source(ROOT,a.source_sha)
    assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    admission=json.loads((ROOT/'validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json').read_text())
    assert admission['formal_admission'] is True and admission['new_formal_target']==60
    jobs=json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs'];validate_jobs(jobs)
    prior=[]
    if a.resume_verified:
        started=json.loads((ROOT/'FORMAL_STARTED.json').read_text())
        assert started['source_sha256']==a.source_sha and started['permit_sha256']==a.permit_sha
        try:owner=C.identity(started['owner']['pid'])
        except FileNotFoundError:owner=None
        assert owner is None or owner['starttick']!=started['owner']['starttick'],'Previous owning controller is still alive'
        C.atomic(ROOT/'runtime'/('RESUME_EPOCH_'+str(time.time_ns())+'.json'),dict(owner=C.identity(os.getpid()),original_start_sha256=C.sha(ROOT/'FORMAL_STARTED.json'),same_source=True))
        for job in jobs:
            target=ROOT/'formal'/job['id']
            if not target.exists():continue
            try:
                closed=json.loads((target/'OWNING_CLOSED.json').read_text());arc=json.loads((target/'ARCHIVE29.json').read_text())
                result=json.loads((target/'RESULT.json').read_text())
                assert closed['owning_Popen_exit']==0 and closed['release_observed'] is True and arc['status']=='PASS_29_DURABLE_NOREPLACE'
                assert closed['source_sha256']==a.source_sha and closed['permit_sha256']==a.permit_sha and closed['RESULT_sha256']==C.sha(target/'RESULT.json')
                assert result['job']==job and result['result']['seed']==1 and result['source_manifest_sha256']==a.source_sha
                payload=result.pop('payload_sha256');assert C.objsha(result)==payload
                prior.append(job['id'])
            except (FileNotFoundError,AssertionError,KeyError):
                moved=ROOT/'interrupted'/str(time.time_ns())/job['id'];moved.parent.mkdir(parents=True)
                target.rename(moved)
                C.atomic(moved/'PREFIX_CREDIT_ZERO.json',dict(original_path=str(target),retained_path=str(moved),credit=0,same_source_fresh_restart=True))
    else:
        C.atomic(ROOT/'FORMAL_STARTED.json',dict(status='UNIQUE_START',owner=C.identity(os.getpid()),source_sha256=a.source_sha,permit_sha256=a.permit_sha,monotonic=time.monotonic(),unit_count=60,reuse=0))
    initial=C.apps();assert not any(x['uuid'] in C.UUIDS for x in initial),'Authorized GPUs occupied; no preemption'
    model_receipt=ROOT/'runtime'/('CONTROLLER_MODEL17_FRESH_SHA'+('_'+str(time.time_ns()) if a.resume_verified else '')+'.json')
    C.atomic(model_receipt,dict(files=C.model_pins(ROOT,full=True)))
    source_files=[r['path'] for r in json.loads((ROOT/'FORMAL_SOURCE_MANIFEST.json').read_text())['files']]+['FORMAL_SOURCE_MANIFEST.json','GPU_PERMIT.json','validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json']
    if not a.resume_verified:
        bootstrap=publish(ROOT,source_files,'SOURCE_BOOTSTRAP');C.atomic(ROOT/'runtime/SOURCE_BOOTSTRAP29.json',bootstrap)
    queues={g:[x for x in jobs if x['gpu']==g and x['id'] not in prior] for g in range(4)}
    live={};completed=list(prior);failed=False
    def event(phase,**kw):
        row=dict(phase=phase,monotonic=time.monotonic(),**kw)
        with (ROOT/'runtime/CONTROLLER_EVENTS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    def launch(gpu):
        assert not any(x['uuid']==C.UUIDS[gpu] for x in C.apps()),'Assigned UUID occupied'
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=4*1024**3
        job=queues[gpu].pop(0);target=ROOT/'formal'/job['id'];target.mkdir()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=C.UUIDS[gpu],CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR='/dev/shm/pace48_leap_unroll_dp8_dp16_v1/triton')
        cmd=[sys.executable,'-B',str(ROOT/'scripts/formal_worker.py'),'--unit',job['id'],'--source-sha',a.source_sha,'--permit-sha',a.permit_sha]
        log=(target/'stdout.log').open('wb');child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ident=C.identity(child.pid);C.atomic(target/'DISPATCH.json',dict(job=job,owner=C.identity(os.getpid()),child=ident,source_sha256=a.source_sha,permit_sha256=a.permit_sha,start_monotonic=time.monotonic(),CVD=env['CUDA_VISIBLE_DEVICES']))
        live[gpu]=dict(child=child,identity=ident,job=job,log=log,target=target,start=time.monotonic(),last=time.monotonic(),size=0,pending_since=None)
        event('UNIT_DISPATCHED',gpu=gpu,unit=job['id'],child=ident)
    try:
        while any(queues.values()) or live:
            if (ROOT/'runtime/STOP_REQUEST.json').exists():
                raise RuntimeError('Explicit owning-parent STOP request')
            for gpu in range(4):
                if gpu not in live and queues[gpu]:launch(gpu)
            for gpu,s in list(live.items()):
                child=s['child'];rc=child.poll();now=time.monotonic()
                if rc is None:
                    try:current=C.identity(child.pid)
                    except FileNotFoundError:
                        if s['pending_since'] is None:s['pending_since']=now;event('EXIT_PENDING_MISSING_PROC',unit=s['job']['id'])
                        assert now-s['pending_since']<120,'Owning Popen exit still unresolved after proc disappearance'
                        continue
                    if current['starttick']!=s['identity']['starttick'] or current['ppid']!=os.getpid():raise RuntimeError('Live owner/starttick identity mismatch')
                    if current['cmdline']!=s['identity']['cmdline']:
                        if current['cmdline']==['']:
                            # exit_mm can clear argv before a sampled stat becomes Z.
                            # This is not an exit code: only the owning Popen can close.
                            if s['pending_since'] is None:s['pending_since']=now;event('EXIT_PENDING',unit=s['job']['id'],observation=current)
                            assert now-s['pending_since']<120,'Owning exit pending beyond120s'
                            continue
                        event('LIVE_ARGV_MISMATCH',unit=s['job']['id'],expected=s['identity'],observed=current)
                        raise RuntimeError('Live argv changed; refusing signal to unmatched process')
                    size=(s['target']/'stdout.log').stat().st_size
                    if size!=s['size']:
                        s['last']=now;s['size']=size
                        event('OWNED_WORKER_OUTPUT_ADVANCED',unit=s['job']['id'],bytes=size)
                    assert now-s['last']<1800,'True phase/token no-progress beyond1800s'
                    continue
                rc=child.wait();s['log'].close();assert rc==0,'Real owning worker exit '+str(rc)
                release_start=time.monotonic()
                while any(x['pid']==child.pid and x['uuid']==C.UUIDS[gpu] for x in C.apps()):
                    assert time.monotonic()-release_start<120,'Own NVML release beyond120s';time.sleep(1)
                result_path=s['target']/'RESULT.json';result=json.loads(result_path.read_text())
                assert result['status']=='PASS' and result['job']==s['job'] and result['source_manifest_sha256']==a.source_sha
                payload=result.pop('payload_sha256');assert C.objsha(result)==payload;result['payload_sha256']=payload
                closed=dict(status='PASS',owning_Popen_exit=rc,child=s['identity'],gpu_uuid=C.UUIDS[gpu],release_observed=True,source_sha256=a.source_sha,permit_sha256=a.permit_sha,RESULT_sha256=C.sha(result_path),dispatch_wall_s=time.monotonic()-s['start'])
                C.atomic(s['target']/'OWNING_CLOSED.json',closed)
                paths=[str(p.relative_to(ROOT)) for p in sorted(s['target'].iterdir()) if p.is_file()]
                receipt=publish(ROOT,paths,'units/'+s['job']['id']);C.atomic(s['target']/'ARCHIVE29.json',receipt)
                completed.append(s['job']['id']);event('UNIT_CLOSED_AND_ARCHIVED',unit=s['job']['id'],completed=len(completed),RESULT_sha256=closed['RESULT_sha256'])
                del live[gpu]
            time.sleep(1)
        C.atomic(ROOT/'runtime/FORMAL_SESSION_RESULT.json',dict(status='COMPLETE60_TRUE_EXIT_RELEASE_AND_ARCHIVE',completed=completed,source_sha256=a.source_sha,permit_sha256=a.permit_sha))
        return 0
    except BaseException as exc:
        failure=ROOT/'runtime/FIRST_FAILURE.json'
        if failure.exists():failure=ROOT/'runtime'/('FIRST_FAILURE_'+str(time.time_ns())+'.json')
        failed=True;C.atomic(failure,dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),completed=completed,owner=C.identity(os.getpid())))
        event('FAIL_CLOSED_STOP',error=repr(exc),completed=len(completed))
        for s in live.values():
            child=s['child']
            if child.poll() is None:
                try:current=C.identity(child.pid)
                except FileNotFoundError:current=None
                if current is not None and C.same_process(current,s['identity']):child.terminate()
        for s in live.values():
            rc=s['child'].wait(timeout=120);s['log'].close();event('OWNED_STOP_WAIT',unit=s['job']['id'],true_exit=rc)
            t=time.monotonic()
            while any(x['pid']==s['child'].pid and x['uuid']==C.UUIDS[s['job']['gpu']] for x in C.apps()):
                assert time.monotonic()-t<120,'Stopped owned child NVML release unresolved';time.sleep(1)
            event('OWNED_STOP_RELEASED',unit=s['job']['id'],partial_credit=0)
        return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--resume-verified',action='store_true');raise SystemExit(run(p.parse_args()))
