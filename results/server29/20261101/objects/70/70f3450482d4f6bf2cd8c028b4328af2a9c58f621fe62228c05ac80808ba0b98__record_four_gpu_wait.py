"""Record genuine outer Bash wait; absence never invents an OS return code."""
import argparse,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pace_control as C

def record(root,mode,bash_pid,child_pid=None,returncode=None):
    root=Path(root).resolve();C.require(str(root)==C.REMOTE,'Exact fourGPU root required');C.authorize(root)
    bash=C.process_identity(bash_pid);C.require(os.getppid()==bash_pid,'Recorder must be a direct child of the owning Bash')
    pins=dict(source_manifest_sha256=C.sha(root/C.SOURCE),permit_sha256=C.sha(root/'GPU_PERMIT.json'))
    if mode=='start':
        C.atomic_new(root/'OFFLINE_CHAIN_STARTED.json',dict(status='ACTUAL_OWNING_BASH_STARTED',bash=bash,unix=time.time(),**pins));return
    start=C.read(root/'OFFLINE_CHAIN_STARTED.json');C.require(start['bash']==bash and all(start[k]==v for k,v in pins.items()),'Owning Bash identity/pins differ')
    if mode=='child':
        child=C.process_identity(child_pid)
        expected=['/data/ydai/miniconda3/envs/bitdecode/bin/python',str(root/'scripts/offline_four_gpu_parent.py'),'--root',str(root),'--execute']
        C.require(child['cmdline']==expected,'Actual outer child argv differs')
        C.atomic_new(root/'OFFLINE_CHAIN_ACTUAL_CHILD.json',dict(parent=child,bash=bash,unix=time.time(),**pins));return
    C.require(type(returncode) is int and 0<=returncode<=255,'Unmodified Bash wait integer required')
    child=C.read(root/'OFFLINE_CHAIN_ACTUAL_CHILD.json')['parent']
    C.require(not C.same_process(child),'Owning Bash wait returned while exact child remains alive')
    C.atomic_new(root/'OFFLINE_CHAIN_OBSERVED_EXIT.json',dict(status='GENUINE_BASH_OWNING_WAIT',actual_shell_wait_returncode=returncode,python_negative_signal_conversion=False,parent=child,bash=bash,child_identity_gone=True,unix=time.time(),**pins))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=['start','child','exit'],required=True);p.add_argument('--bash-pid',type=int,required=True);p.add_argument('--child-pid',type=int);p.add_argument('--returncode',type=int);a=p.parse_args()
    record(a.root,a.mode,a.bash_pid,a.child_pid,a.returncode)

if __name__=='__main__':main()
