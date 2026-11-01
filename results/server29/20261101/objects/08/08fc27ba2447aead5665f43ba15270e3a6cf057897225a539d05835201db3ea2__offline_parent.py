#!/usr/bin/env python3
"""Unique offline CPU parent: controller then one CPU finisher, exact child receipts."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pace_control as C


class Parent:
    def __init__(self, root):
        self.root=Path(root).resolve(); self.identity=C.process_identity(os.getpid())
        self.signal_number=None; self.current=None

    def signal(self, number, frame): self.signal_number=int(number)

    def child(self, name, argv):
        C.require(self.current is None, 'Only one direct child at a time')
        dispatch=dict(task_id=C.TASK,name=name,argv=argv,parent=self.identity,unix=time.time())
        C.atomic_new(self.root/('PARENT_'+name+'_DISPATCH.json'),dispatch)
        identity=None; failure=None; sent=False
        env=C.worker_environment();env['CUDA_VISIBLE_DEVICES']=''
        with (self.root/(name.lower()+'.log')).open('xb',buffering=0) as log:
            proc=subprocess.Popen(argv,cwd=self.root,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.current=proc
            try:
                until=time.monotonic()+2
                while time.monotonic()<until:
                    try:
                        identity=C.process_identity(proc.pid)
                        C.require(identity['cmdline']==argv,'Actual direct child argv differs')
                        break
                    except FileNotFoundError:
                        if proc.poll() is not None:break
                        time.sleep(.02)
                C.require(identity is not None,'Direct child identity not registered')
                C.atomic_new(self.root/('PARENT_'+name+'_ACTUAL_CHILD_IDENTITY.json'),identity)
                while proc.poll() is None:
                    if self.signal_number is not None and name=='CONTROLLER' and not sent:
                        C.terminate_owned(proc,identity);sent=True
                    # The controller handles its owned worker and watchdog. The
                    # finisher is CPU-only and is allowed to seal the true stop.
                    time.sleep(.5)
            except BaseException as exc:
                failure=dict(error=repr(exc),traceback=traceback.format_exc())
                if proc.poll() is None and identity is not None:
                    try:C.terminate_owned(proc,identity)
                    except BaseException as e:failure['cleanup_refusal']=repr(e)
                try:proc.wait(timeout=125)
                except subprocess.TimeoutExpired:
                    if identity is not None:
                        try:C.terminate_owned(proc,identity,kill=True)
                        except BaseException as e:failure['kill_refusal']=repr(e)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            finally:
                value=dict(task_id=C.TASK,name=name,parent=self.identity,identity=identity,
                           returncode=proc.poll(),actual_exited=proc.poll() is not None,
                           identity_gone=identity is not None and not C.same_process(identity),
                           dispatch_sha256=C.sha(self.root/('PARENT_'+name+'_DISPATCH.json')),
                           failure=failure,unix=time.time())
                C.atomic_new(self.root/('PARENT_'+name+'_EXIT.json'),value)
                self.current=None
        return value

    def run(self):
        C.authorize(self.root)
        C.atomic_new(self.root/'OFFLINE_PARENT_STARTED.json',dict(task_id=C.TASK,identity=self.identity,
                     source_manifest_sha256=C.sha(self.root/C.SOURCE),permit_sha256=C.sha(self.root/'GPU_PERMIT.json'),unix=time.time()))
        controller=None;finisher=None;failure=None
        try:
            C.require(self.signal_number is None,'Human stop received before any controller dispatch')
            controller=self.child('CONTROLLER',[sys.executable,str(self.root/'scripts/pace_controller.py'),
                                               '--root',str(self.root),'--execute'])
            C.require(controller['actual_exited'] and controller['identity_gone'],
                      'Controller exit unresolved: refuse any finisher/second controller')
            finisher=self.child('FINISHER',[sys.executable,str(self.root/'scripts/finalize_cpu.py'),
                                           '--root',str(self.root)])
            C.require(finisher['actual_exited'] and finisher['identity_gone'], 'CPU finisher exit unresolved')
        except BaseException as exc:
            failure=dict(error=repr(exc),traceback=traceback.format_exc())
        result=dict(task_id=C.TASK,status='PARENT_TERMINAL' if failure is None else 'PARENT_FAIL_CLOSED_STOP',
                    identity=self.identity,controller=controller,finisher=finisher,failure=failure,
                    signal_received=self.signal_number,no_automatic_restart=True,
                    final_delivery_requires_observer_confirm_this_parent_identity_gone=True,unix=time.time())
        C.atomic_new(self.root/'OFFLINE_PARENT_RESULT.json',result)
        return 0 if failure is None and finisher['returncode']==0 else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print('CPU parent entry only; --execute requires exact frozen permit. No child or GPU action taken.');return 0
    parent=Parent(a.root)
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,parent.signal)
    return parent.run()

if __name__=='__main__':raise SystemExit(main())
