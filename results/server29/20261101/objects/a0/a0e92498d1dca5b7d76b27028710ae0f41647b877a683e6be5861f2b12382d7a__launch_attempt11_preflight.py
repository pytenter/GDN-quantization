"""Outside-capsule preparation launcher. Starts at most one engineering preflight."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');C=R/'provenance_capsule_v3';A=C/'attempt_11_static_ast_qualification'
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
def cmd(*a):return subprocess.check_output(a,cwd=R,text=True).strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def main():
    assert not cmd('git','status','--porcelain=v1')
    assert not (A/'STOP.json').exists()
    assert not (A/'PREFLIGHT_LAUNCH.json').exists(),'DO_NOT_DUPLICATE_PREFLIGHT'
    head=cmd('git','rev-parse','HEAD')
    gpu=cmd('nvidia-smi','-i','6','--query-gpu=index,uuid,name,memory.used,memory.free,driver_version','--format=csv,noheader,nounits')
    assert gpu.split(',')[1].strip()==UUID
    assert float(gpu.split(',')[4])>20000
    processes=cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits')
    assert not any(UUID in line for line in processes.splitlines()),'GPU6_BUSY_NO_INTERFERENCE'
    env=os.environ.copy();profile=json.loads((C/'measurement_launcher_environment.json').read_text())
    env.update(profile['set'])
    for key in profile['unset']:env.pop(key,None)
    receipt=A/'CLEAN_ENGINEERING_LAUNCH.json'
    files=[dict(path=str((R/name).relative_to(C)),sha256=sha(R/name)) for name in cmd('git','ls-files','provenance_capsule_v3').splitlines()]
    emit(receipt,dict(commit=head,clean_worktree=True,files=files,old187_inventory='kept separately in attempt05 FAILED_LAUNCH_IDENTITY.json',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',phase='GENERATED_CODE_DISCOVERY_PREFLIGHT',gpu_query=gpu))
    argv=['/data/zypan/envs/ling-kda/bin/python','-B',str(A/'sealed_runner_attempt11.py'),'capture','preflight_01']
    with (A/'PREFLIGHT.log').open('xb') as log:
        process=subprocess.Popen(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    emit(A/'PREFLIGHT_LAUNCH.json',dict(pid=process.pid,argv=argv,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_commit=head,launch_inventory_sha256=sha(receipt),FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    print('ATTEMPT11_ENGINEERING_PREFLIGHT_LAUNCHED',process.pid,head,flush=True)
if __name__=='__main__':main()


