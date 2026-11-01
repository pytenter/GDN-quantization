"""Unique parent owns controller then CPU finisher. No repeat execution."""
import os,sys,time,subprocess,fcntl
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
C.authorize(R,R/'GPU_PERMIT.json');lock=(R/'operations/OFFLINE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
C.atomic_new(R/'operations/OFFLINE_ONCE_STARTED.json',dict(identity=C.process_identity(os.getpid()),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),monotonic=time.monotonic()))
with(R/'controller.log').open('x')as log:
 p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/gpu_controller.py'),'--root',str(R),'--execute'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 C.atomic_new(R/'operations/ACTUAL_CONTROLLER_LAUNCH.json',dict(identity=C.process_identity(p.pid),parent_identity=C.process_identity(os.getpid()),monotonic=time.monotonic()));controller_exit=p.wait()
C.atomic_new(R/'operations/CONTROLLER_ACTUAL_EXIT.json',dict(exit=controller_exit,monotonic=time.monotonic()))
# The scientific finisher is invoked once regardless of true GPU success or STOP.
with(R/'finisher.log').open('x')as log:
 p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/finalize_offline.py'),str(R)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 C.atomic_new(R/'operations/ACTUAL_FINISHER_LAUNCH.json',dict(identity=C.process_identity(p.pid),monotonic=time.monotonic()));finisher_exit=p.wait()
C.atomic_new(R/'operations/OFFLINE_ACTUAL_EXIT.json',dict(controller_exit=controller_exit,finisher_exit=finisher_exit,monotonic=time.monotonic()))
raise SystemExit(0 if finisher_exit==0 else 1)
