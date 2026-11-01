"""Once-only CPU prerequisite waiter. It never retries a failed GPU pipeline."""
from pathlib import Path
import sys,time,subprocess,json,os,traceback
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
started=R/'operations/REPAIR_ADMISSION_WAITER_STARTED.json'
C.atomic_new(started,dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic(),CPU_only=True))
try:
 while not (R/'operations/PRIOR_RAM_RELEASE_AFTER_DURABLE_ARCHIVE.json').exists():
  assert not (R/'operations/PRIOR_RAM_PRESERVATION_FAILURE.json').exists(),'Engineering preservation failed: no GPU start'
  C.status(R/'operations/REPAIR_ADMISSION_WAITER_PROGRESS.json',dict(phase='WAITING_FOR_DURABLE_REFERENCE_PRESERVATION_AND_RAM_RELEASE',monotonic=time.monotonic(),GPU_dispatches=0))
  time.sleep(15)
 C.status(R/'operations/REPAIR_ADMISSION_WAITER_PROGRESS.json',dict(phase='CPU_PREFLIGHT_SOURCE_FREEZE_AND_ONCE_LAUNCH',monotonic=time.monotonic()))
 p=subprocess.run([sys.executable,'-u','-B',str(R/'operations/freeze_and_launch_repair.py'),str(R)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
 assert p.returncode==0,'Independent prelaunch failed; inspect existing evidence, no automatic retry'
 C.status(R/'operations/REPAIR_ADMISSION_WAITER_PROGRESS.json',dict(phase='ONCE_NEW_PIPELINE_STARTED',monotonic=time.monotonic()))
except BaseException as exc:
 C.atomic_new(R/'operations/REPAIR_ADMISSION_WAITER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0));raise
