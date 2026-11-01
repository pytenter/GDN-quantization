"""Unique55 CPU parent: coordinator -> immutable peer collection -> one29 CPU global finisher."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
import node_rpc as RPC
C.require(C.NODE['id']=='server55_gpu0','Cross-host offline parent must55CPU');C.authorize(R,R/'GPU_PERMIT.json')
lock=(R/'operations/GLOBAL_OFFLINE_ONCE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
C.atomic_new(R/'operations/GLOBAL_OFFLINE_PARENT_STARTED.json',dict(identity=C.process_identity(os.getpid()),source55_sha256=C.sha(R/'SOURCE_FREEZE.json'),monotonic=time.monotonic()))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
with(R/'global_coordinator.log').open('x')as log:
    p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/global_coordinator.py'),'--root',str(R)],stdout=log,stderr=subprocess.STDOUT,env=env)
    ident=C.process_identity(p.pid);C.atomic_new(R/'operations/GLOBAL_COORDINATOR_LAUNCH.json',dict(identity=ident,parent=C.process_identity(os.getpid()),monotonic=time.monotonic()));exitcode=p.wait()
C.atomic_new(R/'GLOBAL_COORDINATOR_ACTUAL_EXIT.json',dict(identity=ident,exit=exitcode,controller_exact_alive=C.same_process(ident),monotonic=time.monotonic()))
with(R/'global_collector.log').open('x')as log:
    p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/collect_peer55_to29_cpu.py'),str(R)],stdout=log,stderr=subprocess.STDOUT,env=env)
    collector_exit=p.wait()
C.atomic_new(R/'operations/GLOBAL_COLLECTOR_ACTUAL_EXIT.json',dict(exit=collector_exit,monotonic=time.monotonic()))
C.require(collector_exit==0,'CPU collection failed; preserve evidence, no GPU/finisher replay')
finish=RPC.prefix()+r'''
import subprocess
assert C.NODE['id']=='server29_gpu6'and not(R/'FINALIZER_STARTED.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=C.NODE['user_site'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with(R/'finisher.log').open('x')as log:
 p=subprocess.Popen([C.NODE['runtime_python'],'-u','-B',str(R/'scripts/finalize_global_cpu.py'),str(R)],stdout=log,stderr=subprocess.STDOUT,env=env)
 identity=C.process_identity(p.pid);C.atomic_new(R/'operations/GLOBAL_FINISHER_LAUNCH.json',dict(identity=identity,monotonic=time.monotonic()));code=p.wait()
C.atomic_new(R/'operations/GLOBAL_FINISHER_ACTUAL_EXIT.json',dict(exit=code,identity=identity,monotonic=time.monotonic()))
print(json.dumps(dict(status='GLOBAL_CPU_FINISHER_ACTUAL_EXIT',exit=code,identity=identity)))
'''
fin=RPC.query29(finish,timeout=18000)
C.atomic_new(R/'operations/GLOBAL_OFFLINE_ACTUAL_EXIT.json',dict(coordinator_exit=exitcode,collector_exit=collector_exit,finisher=fin,monotonic=time.monotonic()))
raise SystemExit(0 if fin['exit']==0 else 1)
