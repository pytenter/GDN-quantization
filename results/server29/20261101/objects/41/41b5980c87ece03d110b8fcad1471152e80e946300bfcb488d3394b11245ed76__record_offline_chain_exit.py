"""Record the offline shell's real owned wait statuses after child exits.

A late CPU receipt: it belongs in the independent supplement, because the
finisher's primary seal necessarily precedes the finisher's own actual exit.
No GPU calls, signals, scorer, finisher rerun, archive edits or cleanup occur.
"""
import argparse
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
p.add_argument('--controller-pid',required=True,type=int);p.add_argument('--controller-status',required=True,type=int)
p.add_argument('--finisher-pid',type=int);p.add_argument('--finisher-status',type=int)
a=p.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
C.require(str(R)==C.EXECUTION_ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Exact CPU-only offline exit observer required')
C.verify_sources(R);permit=C.read(R/'GPU_PERMIT.json');C.require(permit['execution_root']==str(R) and permit['source_freeze_sha256']==C.sha(R/'SOURCE_FREEZE.json'),'Offline exit source/permit differs')
C.require(0<=a.controller_status<=255 and (a.finisher_status is None or 0<=a.finisher_status<=255),'Invalid actual shell wait status')
def exited(name,pid,script,waitstatus):
 path=R/name
 if not path.exists():
  C.require(waitstatus!=0,'Successful child lacks actual STARTED identity')
  return dict(status='CHILD_EXIT_BEFORE_STARTED_RECEIPT',spawned_PID=pid,exact_identity='UNRESOLVED_NO_STARTED_RECEIPT')
 identity=C.read(path)['identity'];C.require(identity['pid']==pid and str(R/'scripts'/script) in identity['cmdline'],'Owned shell child/start receipt identity differs')
 C.require(not C.same_process(identity),'Exact original child still alive after claimed wait')
 return dict(status='ACTUAL_OWNED_WAIT_COMPLETED_ORIGINAL_IDENTITY_ABSENT',identity=identity)
controller=exited('GPU_RUN_STARTED.json',a.controller_pid,'gpu_controller.py',a.controller_status)
if a.finisher_pid is None:
 C.require(a.finisher_status is None and not (R/'SESSION_RESULT.json').exists(),'Finisher omission contradicts terminal session')
 finisher=None
else:
 C.require(a.finisher_status is not None,'Finisher wait status missing')
 finisher=exited('FINALIZER_STARTED.json',a.finisher_pid,'finalize_offline.py',a.finisher_status)
C.atomic_new(R/'OFFLINE_CHAIN_OBSERVED_EXIT.json',dict(status='PASS_CPU_OBSERVED_DIRECT_CHILD_SHELL_WAIT_STATUSES',controller=controller,controller_shell_wait_status=a.controller_status,finisher=finisher,finisher_shell_wait_status=a.finisher_status,shell_parent_PID=os.getppid(),observer_identity=C.process_identity(os.getpid()),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),monotonic=time.monotonic(),GPU_calls=0,signal_calls=0,finisher_or_scoring_calls=0,wait_status_semantics='Exact Bash wait return values; signal-style values>=128 are not reinterpreted as Python Popen negative returns'))
