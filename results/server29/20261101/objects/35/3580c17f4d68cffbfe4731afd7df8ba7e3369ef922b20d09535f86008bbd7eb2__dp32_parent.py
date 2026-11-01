"""Independent CPU owner; uses only the idle GPU3 and never touches old runs."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C
UUID='GPU-30b07aba-a728-7649-4899-e09d33f8bc75'

def verify(expected):
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    p=ROOT/'SOURCE_MANIFEST.json';assert C.sha(p)==expected
    for x in json.loads(p.read_text())['files']:
        q=ROOT/x['path'];assert q.is_file() and q.stat().st_size==x['bytes'] and C.sha(q)==x['sha256'],x['path']

def telemetry():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return dict(remote_monotonic=time.monotonic(),rows=[[x.strip() for x in s.split(',')] for s in p.stdout.splitlines()],compute_apps=C.apps())

def owned(name,command,gpu,source):
    verify(source);target=ROOT/'runtime'/name;target.mkdir();pre=telemetry()
    if gpu:
        assert not any(x['uuid']==UUID for x in pre['compute_apps']),'STOP_NO_IDLE_GPU: no preemption'
        row=next(x for x in pre['rows'] if x[1]==UUID)
        assert row[0]=='3' and '3090' in row[2] and int(row[3])<64,'Assigned physical GPU not idle or identity changed'
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:cvd=next((x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            except (OSError,UnicodeDecodeError):continue
            assert cvd is None or not {x.strip() for x in cvd.split(',')}.intersection({UUID,'3'}),'STOP_NO_IDLE_GPU: another explicit CVD process is preparing this GPU'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID if gpu else '',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',TRITON_CACHE_DIR='/dev/shm/pace48_lqunroll_dp32_throughput_v1/triton')
    if gpu:Path(env['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with (target/'stdout.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        ident=C.identity(child.pid);C.atomic(target/'STARTED.json',dict(owner=C.identity(os.getpid()),child=ident,argv=command,CVD=env['CUDA_VISIBLE_DEVICES'],source_sha256=source,pre=pre,start_monotonic=start))
        size=-1;last=start;last_sample=0;pending=None
        with (target/'TELEMETRY.jsonl').open('x') as sample:
            while child.poll() is None:
                now=time.monotonic()
                try:cur=C.identity(child.pid)
                except FileNotFoundError:cur=None
                if cur is None or cur['cmdline']==['']:
                    if cur is not None:assert cur['starttick']==ident['starttick'] and cur['ppid']==os.getpid()
                    if pending is None:pending=now
                    assert now-pending<120,'Actual owning exit unresolved'
                    time.sleep(.05);continue
                assert C.same_process(cur,ident),'Live owning identity changed; no signal to unmatched process'
                current_size=(target/'stdout.log').stat().st_size
                if current_size!=size:size=current_size;last=now
                if now-last_sample>=15:
                    obs=telemetry();sample.write(json.dumps(obs)+'\n');sample.flush();last_sample=now
                    foreign=[x for x in obs['compute_apps'] if x['uuid']==UUID and x['pid']!=child.pid] if gpu else []
                    if foreign:
                        C.atomic(target/'INTERFERENCE.json',dict(status='STOP_RUNTIME_BLOCKED',foreign_gpu_apps=foreign,no_foreign_signals=True))
                        assert C.same_process(C.identity(child.pid),ident);child.terminate();break
                if now-last>1800:
                    C.atomic(target/'NO_PROGRESS.json',dict(status='STOP_RUNTIME_BLOCKED',no_progress_s=now-last))
                    assert C.same_process(C.identity(child.pid),ident);child.terminate();break
                time.sleep(1)
            rc=child.wait();sample.flush();os.fsync(sample.fileno())
    ended=time.monotonic();released=not gpu
    while gpu and time.monotonic()-ended<120:
        post=C.apps()
        if not any(x['uuid']==UUID and x['pid']==child.pid for x in post):released=True;break
        time.sleep(.2)
    receipt=dict(owning_Popen_exit=rc,child=ident,source_sha256=source,gpu_uuid=UUID if gpu else None,release_observed=released,dispatch_wall_s=time.monotonic()-start,post_NVML=C.apps(),GPU_active_seconds=None)
    C.atomic(target/'OWNING_EXIT_AND_RELEASE.json',receipt);assert released,'Own GPU release unresolved'
    print(json.dumps(dict(phase='UNIT_OWNED_EXIT',name=name,exit=rc,released=released)),flush=True)
    return rc

def main(a):
    assert not os.environ.get('CUDA_VISIBLE_DEVICES','');verify(a.source_sha)
    lock=(ROOT/'operations/PARENT.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    C.atomic(ROOT/'runtime/UNIQUE_PARENT_STARTED.json',dict(owner=C.identity(os.getpid()),bash=C.identity(os.getppid()),source_sha256=a.source_sha,initial_telemetry=telemetry()))
    status='STOP_RUNTIME_BLOCKED';error=None
    try:
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=4*1024**3,'Output space below4GiB'
        for i in [1,2]:
            rc=owned('NUMERIC_FRESH'+str(i),[sys.executable,'-B',str(ROOT/'scripts/validate_dp32.py'),str(ROOT/('validation/DP32_FRESH'+str(i)+'.json'))],True,a.source_sha)
            if rc:
                p=ROOT/('validation/DP32_FRESH'+str(i)+'.json')
                status='STOP_NUMERICAL_VALIDATION_FAILED' if p.exists() and json.loads(p.read_text()).get('status')=='STOP_NUMERICAL_VALIDATION_FAILED' else 'STOP_RUNTIME_BLOCKED'
                break
        else:
            rc=owned('THROUGHPUT_SUITE',[sys.executable,'-B',str(ROOT/'scripts/throughput_dp32.py')],True,a.source_sha)
            if rc==0:status='BENCHMARKS_COMPLETE_PENDING_CPU_REPORT'
        verify(a.source_sha)
    except BaseException as exc:
        error=repr(exc);status='STOP_NO_IDLE_GPU' if 'STOP_NO_IDLE_GPU' in error or 'not idle' in error else 'STOP_RUNTIME_BLOCKED'
        C.atomic(ROOT/'runtime/FIRST_FAILURE.json',dict(error=error,traceback=traceback.format_exc(),no_old_task_or_foreign_signals=True))
    C.atomic(ROOT/'runtime/COMPUTE_TERMINAL.json',dict(status=status,error=error,source_sha256=a.source_sha,new_AIME_samples=0))
    rc=owned('CPU_FINAL_REPORT',[sys.executable,'-B',str(ROOT/'scripts/dp32_reports.py')],False,a.source_sha)
    assert rc==0,'CPU report failed; preserve claim and all artifacts, no GPU retry'
    C.atomic(ROOT/'runtime/PARENT_COMPLETE.json',dict(status='TRUE_OWNING_EXECUTION_AND_REPORT_CLOSED',source_sha256=a.source_sha,scientific_status=json.loads((ROOT/'RESULT.json').read_text())['status']))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);raise SystemExit(main(p.parse_args()))
