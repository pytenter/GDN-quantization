"""Remote29 node parent owns its single controller once. Global finisher is separate and CPU-only."""
import fcntl,os,subprocess,sys,time
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
C.authorize(R,R/'GPU_PERMIT.json');assert C.NODE['id']=='server29_gpu6'
lock=(R/'operations/NODE_ONCE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
C.atomic_new(R/'operations/NODE_PARENT_STARTED.json',dict(identity=C.process_identity(os.getpid()),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),monotonic=time.monotonic()))
with(R/'node_controller.log').open('x')as log:
    p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/gpu_controller.py'),'--root',str(R),'--execute'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    C.atomic_new(R/'operations/NODE_CONTROLLER_LAUNCH.json',dict(identity=C.process_identity(p.pid),parent=C.process_identity(os.getpid()),monotonic=time.monotonic()));exitcode=p.wait()
C.atomic_new(R/'operations/NODE_ACTUAL_EXIT.json',dict(exit=exitcode,monotonic=time.monotonic()))
raise SystemExit(exitcode)
