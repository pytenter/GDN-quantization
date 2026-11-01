"""One bounded-roster engineering unit; true owning Popen wait and UUID release.

No fixed total GPU/wall cap. 1800s is solely output/phase no-progress watchdog.
Only the exactly owned child can be terminated; never signal another GPU user.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UUIDS=['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']
def atomic(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def proc_identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    return dict(pid=pid,ppid=int(tail[1]),starttick=int(tail[19]),cmdline=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
def nvml():
    cmd=['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']
    p=subprocess.run(cmd,capture_output=True,text=True,check=True)
    return [dict(pid=int(v[0]),uuid=v[1],memory_MiB=v[2]) for line in p.stdout.splitlines() if (v:=[s.strip() for s in line.split(',')]) and len(v)==3]
def run(a):
    assert 0<=a.gpu<4
    target=ROOT/'runtime'/a.name;target.mkdir(parents=True)
    assert not (ROOT/'FORMAL_STARTED.json').exists(),'Engineering run after formal start forbidden'
    rows=[]
    for folder in ['src','scripts','configs','vendor','protocol']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    atomic(target/'SOURCE_SNAPSHOT.json',dict(files=rows))
    assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=4*1024**3,'Output filesystem below4GiB'
    pre=nvml();assert not any(x['uuid']==UUIDS[a.gpu] for x in pre),'Assigned UUID is occupied; no preemption'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[a.gpu],PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',TRITON_CACHE_DIR='/dev/shm/qwen_gdn_hmmt26_fp_lq8_hlq8_lq16_v1/postnorm_h128_v2/triton')
    Path(env['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-B']+a.command
    start=time.monotonic()
    with (target/'stdout.log').open('wb') as log:
        child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        identity=proc_identity(child.pid)
        atomic(target/'STARTED.json',dict(owner=proc_identity(os.getpid()),child=identity,argv=cmd,CVD=env['CUDA_VISIBLE_DEVICES'],pre_NVML=pre,start_monotonic=start))
        last_size=-1;last_progress=start
        while child.poll() is None:
            n=(target/'stdout.log').stat().st_size
            if n!=last_size:last_size=n;last_progress=time.monotonic()
            if time.monotonic()-last_progress>1800:
                current=proc_identity(child.pid)
                assert current==identity,'Refuse signal after identity mismatch'
                child.terminate();raise RuntimeError('True1800s stdout/phase no-progress watchdog')
            time.sleep(1)
        rc=child.wait()
    exited=time.monotonic();released=False
    while time.monotonic()-exited<120:
        post=nvml()
        if not any(x['pid']==child.pid and x['uuid']==UUIDS[a.gpu] for x in post):released=True;break
        time.sleep(1)
    receipt=dict(status='PASS_TRUE_EXIT_AND_RELEASE' if rc==0 and released else 'FAIL',owning_Popen_exit=rc,child=identity,gpu=a.gpu,gpu_uuid=UUIDS[a.gpu],release_observed=released,post_NVML=post,dispatch_wall_s=time.monotonic()-start,generation_or_active_GPU_time_not_inferred=True)
    atomic(target/'OWNING_WAIT_AND_RELEASE.json',receipt)
    print(json.dumps(receipt),flush=True)
    if rc!=0:print((target/'stdout.log').read_text()[-8000:],flush=True)
    return 0 if rc==0 and released else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    raise SystemExit(run(a))
