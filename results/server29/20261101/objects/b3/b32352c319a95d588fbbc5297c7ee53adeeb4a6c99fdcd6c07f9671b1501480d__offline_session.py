"""One-shot tmux parent; no inference or restart logic."""
import json,sys,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import gpu_control as C
C.require(not (R/'execution/WRAPPER_STARTED.json').exists(),'No duplicate session')
C.atomic_new(R/'execution/WRAPPER_STARTED.json',dict(identity=C.process_identity(__import__('os').getpid()),monotonic=time.monotonic()))
with (R/'GPU_CONTROLLER.log').open('x') as f:
 p=subprocess.run([sys.executable,'-B',str(R/'scripts/gpu_controller.py'),'--root',str(R),'--permit',str(R/'GPU_PERMIT.json'),'--execute'],stdout=f,stderr=subprocess.STDOUT)
C.atomic_new(R/'execution/WRAPPER_RESULT.json',dict(controller_returncode=p.returncode,session_receipt_present=(R/'execution/SESSION_RESULT.json').exists(),monotonic=time.monotonic(),no_retry=True))
sys.exit(p.returncode)
