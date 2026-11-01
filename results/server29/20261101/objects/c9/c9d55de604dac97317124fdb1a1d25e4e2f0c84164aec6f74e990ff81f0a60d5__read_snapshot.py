"""Independent SSH-only state snapshot; no response text, sampling, scoring or mutation."""
from pathlib import Path
import json,subprocess,shlex,time,sys
R=Path(__file__).resolve().parents[1];c=json.loads((R/'ROOT.json').read_text())
code='''import sys,json,subprocess,time
from pathlib import Path
R=Path(sys.argv[1]);v=dict(remote_monotonic=time.monotonic(),remote_unix=time.time())
for n in ['LAUNCH_RECEIPT.json','GPU_RUN_STARTED.json','SESSION_RESULT.json','FIRST_FAILURE.json','RESOURCE_LEDGER.json','AUTHORIZATION_FORMAL.json','FORMAL_GENERATIONS_COMPLETE.json','FINAL_TASK_STATUS.json','FINAL_SEAL_RECEIPT.json','FINALIZER_FAILURE.json','FINALIZER_STARTED.json','FORMAL_PROGRESS.json','P1_C1_S3_GPU_QUALIFICATION.json','branches/U_CPU_ELIGIBILITY.json']:
 if (R/n).exists():v[n]=json.loads((R/n).read_text())
v['units']={};identities=[]
for folder in ['execution','formal']:
 units=[]
 for p in sorted((R/folder).glob('*')):
  if not p.is_dir():continue
  row=dict(id=p.name,result_exists=(p/'RESULT.json').exists(),failure_exists=(p/'FIRST_FAILURE.json').exists())
  for n in ['STARTED.json','PROGRESS.json','DRIVER_IDENTITY.json']:
   if (p/n).exists():
    x=json.loads((p/n).read_text());row[n]=x
    if n=='STARTED.json':identities.append(x['identity'])
  units.append(row)
 v['units'][folder]=units
for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json','FORMAL_PROGRESS.json','P1_C1_S3_GPU_QUALIFICATION.json','branches/U_CPU_ELIGIBILITY.json']:
 if n in v:identities.append(v[n]['identity'])
v['actual_processes']=[]
for x in identities:
 p=Path('/proc')/str(x['pid'])
 try:
  a=dict(pid=x['pid'],starttick=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(p/'cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0'))
  if a==x:v['actual_processes'].append(a)
 except (FileNotFoundError,ProcessLookupError):pass
for name,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits']),('NVML',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),('disk',['df','-B1','/data','/dev/shm']),('tmux',['tmux','list-sessions','-F','#{session_name}'])]:
 p=subprocess.run(cmd,capture_output=True,text=True);v[name]=p.stdout.splitlines();v[name+'_returncode']=p.returncode
v['log_tails']={}
for p in [R/'controller.log',R/'finisher.log']+[R/f/u['id']/'worker.log' for f,items in v['units'].items() for u in items if not u['result_exists']]:
 if p.exists():
  with p.open('rb') as h:h.seek(max(0,p.stat().st_size-2000));v['log_tails'][str(p.relative_to(R))]=h.read().decode(errors='replace')
print(json.dumps(v))
'''
p=subprocess.run(['ssh','-S',c['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',shlex.join([c['remote_python'],'-u','-B','-c',code,c['remote_root']])],capture_output=True,timeout=60);assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);out=R/'monitoring'/('SNAPSHOT_'+time.strftime('%Y%m%dT%H%M%S')+'.json');out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v if '--brief' not in sys.argv else dict(controller=v.get('GPU_RUN_STARTED.json'),session=v.get('SESSION_RESULT.json'),failure=v.get('FIRST_FAILURE.json'),finalizer_failure=v.get('FINALIZER_FAILURE.json'),units={f:[dict(id=x['id'],passed=x['result_exists'],failed=x['failure_exists'],progress=x.get('PROGRESS.json')) for x in items] for f,items in v['units'].items()},NVML=v['NVML'],gpu=v['gpu'],actual_processes=v['actual_processes'],log_tails=v['log_tails']),indent=2))
