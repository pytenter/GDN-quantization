"""Single-device bounded supervisor; no CUDA imports, retries, or kernel changes."""
import csv,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python';UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
PHASES=['a0_1','a0_2','a1','b0_1','b0_2','students']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(rel,obj):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def replace(rel,obj):
    p=ROOT/rel;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p)
def nv():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=list(csv.reader(raw.splitlines()));g=next(a for a in rows if int(a[0])==0)
    appsraw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    apps=[a for a in csv.reader(appsraw.splitlines()) if a and a[0].strip()==UUID]
    return dict(physical_id=int(g[0]),uuid=g[1].strip(),name=g[2].strip(),total_MiB=int(g[3]),free_MiB=int(g[4]),used_MiB=int(g[5]),utilization_percent=int(g[6]),apps=apps,raw=raw,apps_raw=appsraw,unix=time.time())
def check(ok,msg):
    if not ok:raise RuntimeError(msg)
def totals():
    files=[p for p in ROOT.rglob('*') if p.is_file()];raw=[p for p in (ROOT/'raw_snapshots').glob('*') if p.is_file()]
    return sum(p.stat().st_size for p in files),sum(p.stat().st_size for p in raw)
def main():
    check(not (ROOT/'COORDINATOR_STARTED.json').exists(),'Coordinator may only run once for execution01')
    check(read(ROOT/'execution_authorization_addendum.json')['GPU_CANARY_AUTHORIZATION']=='YES','No GPU authorization')
    check(read(ROOT/'analysis/PARENT_AND_MODEL_CONTENT_VERIFICATION.json')['status']=='PASS','Source/model prerequisite')
    check(read(ROOT/'analysis/MINIMAL_ADAPTER_CPU_REGRESSION.json')['status']=='PASS','Adapter CPU prerequisite')
    frozen=read(ROOT/'source_manifest.json')
    for a in frozen['execution_sources']:check(sha(ROOT/a['path'])==a['sha256'],'Frozen new source changed')
    probe=nv();check(probe['uuid']==UUID and '3090' in probe['name'] and probe['total_MiB']==24576,'Hardware admission mismatch')
    check(probe['free_MiB']>=23000 and probe['utilization_percent']<=1 and not probe['apps'],'GPU0 is not idle; no alternate GPU retry')
    check(os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=2*1024**3,'Insufficient disk')
    save('resource_admission.json',dict(status='PASS',probe=probe,physical_GPU_id=0,GPU_UUID=UUID,mapping='CUDA_VISIBLE_DEVICES=0 -> cuda0',other_three_GPUs='Human authorized availability; unused',model_content_identity='PASS_CURRENT_FULL_SHARD_HASHES',CPU_adapter_regression='PASS',GPUh_limit=2.0,raw_snapshot_limit_GiB=1,archive_limit_GiB=10))
    save('COORDINATOR_STARTED.json',dict(unix=time.time(),pid=os.getpid(),phases=PHASES,single_run=True))
    gates={p:'NOT_RUN' for p in PHASES};ledger=dict(GPU_HOURS_USED=0.,MAX_GPU_HOURS=2.0,GPU_UUID=UUID,physical_GPU_ID=0,phases=[],accounting='Exclusive device reservation from child dispatch through NVML-confirmed release, including imports/init/load/forward/CPU work while reserved; conservative full occupancy billing',NVML_sampling_period_seconds=.5)
    active=None;failure=None
    try:
        for phase in PHASES:
            remaining=7200-ledger['GPU_HOURS_USED']*3600
            if phase=='a0_2':predicted=ledger['phases'][0]['reservation_seconds']*1.35
            elif phase=='a1':predicted=600
            elif phase=='b0_1':predicted=max(600,ledger['phases'][0]['reservation_seconds']*3.)
            elif phase=='b0_2':predicted=max(180,ledger['phases'][0]['reservation_seconds']*.8)
            elif phase=='students':predicted=max(600,ledger['phases'][0]['reservation_seconds']*3.)
            else:predicted=600
            check(remaining>predicted+120,'Remaining aggregate budget insufficient for next gate with safe exit reserve')
            check(not (ROOT/'FIRST_FAILURE.json').exists(),'Attempt has first failure; all later GPU phases forbidden')
            for a in frozen['execution_sources']:check(sha(ROOT/a['path'])==a['sha256'],'Frozen new source changed')
            probe=nv();check(not probe['apps'] and probe['free_MiB']>=23000 and probe['utilization_percent']<=1,'Resource admission lost; stop')
            start=time.monotonic();start_wall=time.time();peak=probe['used_MiB'];samples=[]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
            log=(ROOT/f'logs/{phase}.log').open('x');active=subprocess.Popen([PY,str(ROOT/'src/gpu_worker.py'),phase],env=env,stdout=log,stderr=subprocess.STDOUT)
            entry=dict(phase=phase,pid=active.pid,reserved_unix=start_wall,prediction_seconds=predicted,budget_remaining_before_seconds=remaining)
            last_progress=start;last_progress_size=-1
            while active.poll() is None:
                time.sleep(.5);probe=nv();peak=max(peak,probe['used_MiB']);elapsed=time.monotonic()-start
                samples.append(dict(unix=probe['unix'],used_MiB=probe['used_MiB'],free_MiB=probe['free_MiB'],utilization=probe['utilization_percent']))
                check(all(int(a[1])==active.pid for a in probe['apps']),'Another process appeared on reserved GPU; stop own job')
                check(elapsed<remaining-90,'Aggregate GPU deadline reached safe-stop reserve')
                archive,raw=totals();check(archive<10*1024**3 and raw<1024**3,'Archive/snapshot budget exceeded')
                size=(ROOT/f'logs/{phase}.log').stat().st_size
                if size!=last_progress_size:last_progress=time.monotonic();last_progress_size=size
                check(time.monotonic()-last_progress<900,'15minute no-progress watchdog')
                if len(samples)%20==0:replace('analysis/GPU_RESOURCE_LEDGER_LIVE.json',dict(**ledger,active_phase=phase,active_PID=active.pid,active_reserved_seconds=elapsed,active_cumulative_GPUh=ledger['GPU_HOURS_USED']+elapsed/3600,peak_sampled_NVML_MiB=peak))
            code=active.wait();log.close();release=nv()
            deadline=time.monotonic()+20
            while any(int(a[1])==active.pid for a in release['apps']) and time.monotonic()<deadline:time.sleep(1);release=nv()
            released=not any(int(a[1])==active.pid for a in release['apps']);check(released,'Own GPU context not released')
            elapsed=time.monotonic()-start;entry.update(returncode=code,reservation_seconds=elapsed,GPU_hours=elapsed/3600,release_unix=time.time(),peak_sampled_NVML_MiB=peak,NVML_samples=samples,release_probe=release,released=True)
            ledger['phases'].append(entry);ledger['GPU_HOURS_USED']+=elapsed/3600
            gates[phase]='PASS' if code==0 else 'FAIL';replace('analysis/GPU_RESOURCE_LEDGER_LIVE.json',ledger)
            print(json.dumps(dict(phase=phase,status=gates[phase],reservation_seconds=elapsed,cumulative_GPUh=ledger['GPU_HOURS_USED'])),flush=True)
            active=None
            check(code==0,'GPU worker failed; no GPU patch/retry allowed')
            check(read(ROOT/f'analysis/RECEIPT_{phase}.json')['status']=='PASS','Worker ended without PASS receipt')
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc(),phase=locals().get('phase'),no_GPU_retry=True)
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=20)
            except subprocess.TimeoutExpired:active.kill();active.wait()
            elapsed=time.monotonic()-start
            entry.update(returncode=active.returncode,reservation_seconds=elapsed,GPU_hours=elapsed/3600,release_unix=time.time(),peak_sampled_NVML_MiB=peak,NVML_samples=samples,release_probe=nv(),released=True,terminated_for=failure['error'])
            ledger['phases'].append(entry);ledger['GPU_HOURS_USED']+=elapsed/3600;gates[phase]='FAIL'
        if not (ROOT/'FIRST_FAILURE.json').exists():save('FIRST_FAILURE.json',dict(status='BLOCKED_AND_STOP',**failure))
    save('analysis/gpu_resource_ledger.json',ledger)
    save('GATE_STATUS_FINAL.json',dict(gates=gates,coordinator_failure=failure,first_failure_file='FIRST_FAILURE.json' if (ROOT/'FIRST_FAILURE.json').exists() else None,GPU_HOURS_USED=ledger['GPU_HOURS_USED'],GPU_release_probe=nv(),no_further_GPU_work=True))
    print(json.dumps(dict(status='STOP' if failure else 'GPU_GATES_COMPLETE',gates=gates,GPU_HOURS_USED=ledger['GPU_HOURS_USED'])),flush=True)
if __name__=='__main__':main()
