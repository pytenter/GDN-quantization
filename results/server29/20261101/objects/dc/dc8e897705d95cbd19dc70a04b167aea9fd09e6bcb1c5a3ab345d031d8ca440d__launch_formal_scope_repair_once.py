"""Once new offline controller; validated58 origins -> fixed60 formal jobs."""
import argparse,json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
PROGRAM=r'''
import pathlib,os,sys,json,subprocess,shlex,time
r=pathlib.Path(sys.argv[1]).resolve();name='qwen-resident-formal-scope-repair-20261023'
assert r.name=='qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_formal_scope_repair_20261023'and os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(r/'src'),str(r/'scripts')]
import resident_execution_control as C,qualification_reuse as Q
assert not any(os.path.lexists(r/p)for p in ['OFFLINE_STARTED.json','CONTROLLER_STARTED.json','operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json','operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json'])
source=C.verify_sources(r);base=C.read(r/'GPU_PERMIT.json');ad=C.read(r/'operations/FINAL_CPU_ADMISSION.json')
assert base['ALLOW_GPU']=='QUALIFICATION_ONLY'and base['source_freeze_sha256']==ad['source_freeze_sha256']==source and not base['formal_permission']
assert base['human_formal_scope_repair_authorization_sha256']==C.sha(r/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json')and base['final_CPU_admission_sha256']==C.sha(r/'operations/FINAL_CPU_ADMISSION.json')
assert ad['qualification_reused']==58 and ad['qualification_remaining']==ad['current_qualification_GPU_dispatches']==ad['formal_generations']==0
assert len(Q.seed(C,r))==58 and base['qualification_reuse_admission_sha256']==C.sha(r/'QUALIFICATION_REUSE_ADMISSION.json')
C.verify_model_and_external(r,fresh_external=False);obs=C.query_gpu();assert not any(x['uuid']in C.UUIDS.values()for x in obs['apps'])
assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==1
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json',dict(status='ONE_ACTUAL_HUMAN_FORMAL_SCOPE_REPAIR_LAUNCH',tmux=name,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),monotonic=time.monotonic(),no_retry=True))
command=['tmux','new-session','-d','-s',name,shlex.join(['bash',str(r/'scripts/run_resident_offline.sh'),str(r)])]
x=subprocess.run(command,capture_output=True,text=True);assert x.returncode==0,x.stderr
for i in range(120):
 if(r/'CONTROLLER_STARTED.json').exists():break
 time.sleep(.25)
started=C.read(r/'CONTROLLER_STARTED.json');actual=C.process_identity(started['identity']['pid']);assert actual==started['identity']and C.same_process(actual)
receipt=dict(status='PASS_UNIQUE_FORMAL_SCOPE_REPAIR_CONTROLLER_ACTUALLY_STARTED',tmux=name,command=command,controller_identity=actual,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),offline=C.read(r/'OFFLINE_STARTED.json'),NVML=C.query_gpu(),monotonic=time.monotonic(),qualification_reused=58,qualification_current_remaining=0,current_engineering_dispatches=0)
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();assert a.execute
 c=json.loads((R/'ROOT.json').read_text());cmd=['ssh','-S',c['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',c['remote_python'],'-B','-c',PROGRAM,c['remote_root']])]
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (R/'operations/FORMAL_SCOPE_LAUNCH_TRANSPORT.json').open('x')as s:json.dump(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),s,indent=2)
 assert x.returncode==0,x.stderr;receipt=json.loads(x.stdout)
 with(R/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json').open('x')as s:json.dump(receipt,s,sort_keys=True,indent=2)
 print(json.dumps(dict(status=receipt['status'],identity=receipt['controller_identity'])))
if __name__=='__main__':main()
