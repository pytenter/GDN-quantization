"""True owning controller/CPU-finisher chain. No second automatic GPU attempt."""
import argparse,csv,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C
from archive_cpu import publish

def owned(command,name):
    target=ROOT/'runtime'/name;target.mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    with (target/'stdout.log').open('wb') as log:
        child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ident=C.identity(child.pid);C.atomic(target/'STARTED.json',dict(owner=C.identity(os.getpid()),child=ident,command=command,start_monotonic=time.monotonic()))
        last_size=-1;last=time.monotonic()
        pending=None
        while child.poll() is None:
            try:current=C.identity(child.pid)
            except FileNotFoundError:current=None
            if current is None or current['cmdline']==['']:
                if current is not None:
                    assert current['starttick']==ident['starttick'] and current['ppid']==os.getpid(),'Empty argv identity epoch/owner changed'
                if pending is None:pending=time.monotonic()
                assert time.monotonic()-pending<120,'Owning child exit unresolved'
                time.sleep(.05);continue
            assert C.same_process(current,ident),'Live owning child identity changed; no signal to unmatched process'
            size=(target/'stdout.log').stat().st_size
            if size!=last_size:last_size=size;last=time.monotonic()
            if time.monotonic()-last>1800:
                if name=='FORMAL_CONTROLLER':
                    request=ROOT/'runtime/STOP_REQUEST.json'
                    if not request.exists():C.atomic(request,dict(reason='True1800s owning-chain phase no-progress',owner=C.identity(os.getpid())))
                elif C.same_process(C.identity(child.pid),ident):child.terminate()
                code=child.wait(timeout=125)
                C.atomic(target/'WATCHDOG_OWNING_EXIT.json',dict(child=ident,owning_Popen_exit=code))
                raise RuntimeError(name+' true phase no-progress watchdog')
            time.sleep(1)
        code=child.wait()
    C.atomic(target/'OWNING_EXIT.json',dict(status='EXITED',child=ident,owning_Popen_exit=code,observed_monotonic=time.monotonic()))
    return code

def finish(a):
    C.source(ROOT,a.source_sha)
    own=[]
    for p in (ROOT/'formal').glob('*/DISPATCH.json'):
        ident=json.loads(p.read_text())['child']
        try:current=C.identity(ident['pid'])
        except FileNotFoundError:continue
        if current['starttick']==ident['starttick']:
            own.append(ident)
    assert not own,'An exact owned worker is still live; foreign processes are not ownership evidence'
    summaries=[]
    for i in (1,2):
        output=ROOT/'runtime'/('SCORE_PASS'+str(i)+'.json')
        code=owned([sys.executable,'-B',str(ROOT/'scripts/score_cpu.py'),'--source-sha',a.source_sha,'--permit-sha',a.permit_sha,'--output',str(output)],'CPU_SCORE_FRESH'+str(i))
        assert code==0,'Real CPU scorer exit is nonzero'
        summaries.append(json.loads(output.read_text()))
    assert summaries[0]==summaries[1],'Two fresh CPU scoring passes disagree'
    score=summaries[0];C.atomic(ROOT/'SCORED_RESULTS.json',score)
    with (ROOT/'paired_results.csv').open('x',newline='') as f:
        rows=score['summary']['pairs'];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    C.atomic(ROOT/'PAIR_STATISTICS.json',score['summary'])
    C.atomic(ROOT/'runtime/FINAL_OWNED_GPU_RELEASE.json',dict(assigned_UUIDs=C.UUIDS,apps=C.apps(),all_exact_owned_workers_gone=True,per_unit_own_release_receipts_verified=True,foreign_processes_not_required_to_exit=True))
    # Scientific reports remain a CPU delivery operation; no GPU or finisher rerun.
    code=owned([sys.executable,'-B',str(ROOT/'scripts/build_reports.py'),'--terminal','--source-sha',a.source_sha,'--permit-sha',a.permit_sha],'CPU_REPORT_FINISHER')
    assert code==0,'CPU report finisher failed'
    paths=seal_manifest()
    receipt=publish(ROOT,paths,'FINAL_DELIVERY');C.atomic(ROOT/'runtime/FINAL_ARCHIVE29.json',receipt)
    C.atomic(ROOT/'DELIVERY_READY.json',dict(status='READY_FOR_INDEPENDENT_CPU_DELIVERY',source_sha256=a.source_sha,permit_sha256=a.permit_sha,all60_new=True,two_fresh_CPU_scores_equal=True,archive=receipt))

def seal_manifest():
    late={'MANIFEST.sha256','runtime/PARENT_STDOUT.log','runtime/PARENT_COMPLETE.json','runtime/BASH_WAIT.json','runtime/FINAL_ARCHIVE29.json','DELIVERY_READY.json'}
    paths=[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.') and str(p.relative_to(ROOT)) not in late and 'independent_delivery' not in p.parts]
    C.atomic(ROOT/'FINAL_MANIFEST_SCOPE.json',dict(status='MAIN_DELIVERY_SNAPSHOT',late_owning_wait_and_parent_archive_receipts='Independent immutable sibling after actual owning exit, never invented in main seal',exclusions=sorted(late)))
    paths.append('FINAL_MANIFEST_SCOPE.json')
    with (ROOT/'MANIFEST.sha256').open('x') as f:
        for name in paths:f.write(C.sha(ROOT/name)+'  '+name+'\n')
        f.flush();os.fsync(f.fileno())
    return paths+['MANIFEST.sha256']

def run(a):
    C.source(ROOT,a.source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    C.atomic(ROOT/'runtime/PARENT_STARTED.json',dict(owner=C.identity(os.getpid()),source_sha256=a.source_sha,permit_sha256=a.permit_sha))
    try:
        rc=owned([sys.executable,'-B',str(ROOT/'scripts/formal_controller.py'),'--source-sha',a.source_sha,'--permit-sha',a.permit_sha],'FORMAL_CONTROLLER')
        if rc!=0:
            C.atomic(ROOT/'runtime/PARTIAL_TERMINAL.json',dict(status='PARTIAL_WITH_VALIDATED_ARTIFACTS',controller_owning_exit=rc,full_accuracy=None,no_GPU_retry=True))
            code=owned([sys.executable,'-B',str(ROOT/'scripts/build_reports.py'),'--partial','--source-sha',a.source_sha,'--permit-sha',a.permit_sha],'CPU_REPORT_FINISHER')
            assert code==0,'Partial CPU diagnostic report finisher failed'
            receipt=publish(ROOT,seal_manifest(),'PARTIAL_DELIVERY')
            C.atomic(ROOT/'runtime/FINAL_ARCHIVE29.json',receipt)
            C.atomic(ROOT/'DELIVERY_READY.json',dict(status='READY_PARTIAL_DIAGNOSTIC',source_sha256=a.source_sha,permit_sha256=a.permit_sha,all60_new=False,full_accuracy=None,archive=receipt))
            return 1
        finish(a)
        C.atomic(ROOT/'runtime/PARENT_COMPLETE.json',dict(status='COMPLETE_TRUE_OWNING_CHAIN',source_sha256=a.source_sha,permit_sha256=a.permit_sha))
        return 0
    except BaseException as exc:
        C.atomic(ROOT/'runtime/PARENT_FIRST_FAILURE.json',dict(status='FAIL_CPU_OR_EXECUTION_DELIVERY',error=repr(exc),traceback=traceback.format_exc(),no_automatic_GPU_or_finisher_retry=True))
        return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);raise SystemExit(run(p.parse_args()))
