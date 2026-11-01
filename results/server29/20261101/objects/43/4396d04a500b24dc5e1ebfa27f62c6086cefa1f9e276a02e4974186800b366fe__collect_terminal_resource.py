#!/usr/bin/env python3
"""Read-only final stopped-run events/process/progress metadata. No GPU workload."""
from pathlib import Path
import subprocess,json,datetime,hashlib
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_REPAIR_20261019_V1'
CODE=r'''import json,time,datetime,hashlib,subprocess
from pathlib import Path
R=Path(REMOTE_ROOT)
def stable(p):
 if not p.exists():return None
 a=p.stat();raw=p.read_bytes();b=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns),'Changed during terminalread:'+str(p)
 return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'raw':raw.decode()}
def meta(p):
 x=stable(p)
 if x:x['json']=json.loads(x.pop('raw'))
 return x
def proc(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();return {'pid':pid,'starttick':int(s[s.rfind(')')+2:].split()[19]),'cmdline':[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}
 except(FileNotFoundError,PermissionError,ProcessLookupError):return None
out={'root':str(R),'remote_unix':time.time(),'remote_monotonic':time.monotonic(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shared':{},'cards':{}};pids=set()
for name in ['FIRST_FAILURE.json','SESSION_STARTED.json','SESSION_RESULT.json']+[f'COORDINATOR_{n}_{kind}.json'for n in range(4)for kind in ['SPAWN','EXIT']]:
 x=meta(R/name);out['shared'][name]=x
 if x and isinstance(x['json'].get('identity'),dict):pids.add(x['json']['identity']['pid'])
for n in range(4):
 c=R/('gpu'+str(n));v={'metadata':{},'units':[],'events':stable(c/'execution/RESOURCE_EVENTS.jsonl'),'reducer_source':stable(c/'qualification/resource_event_reducer.py')}
 for name in ['RESOURCE_LEDGER.json','FIRST_FAILURE.json','execution/CONTROLLER_STARTED.json','execution/CONTROLLER_RESULT.json','qualification/FP_NUMERICAL_VERDICT.json','qualification/PERFORMANCE_VERDICT.json']:v['metadata'][name]=meta(c/name)
 cs=v['metadata']['execution/CONTROLLER_STARTED.json']
 if cs:pids.add(cs['json']['controller_pid'])
 for line in v['events']['raw'].splitlines():
  e=json.loads(line)
  if 'pid'in e:pids.add(e['pid'])
 for d in sorted((c/'execution').glob('*')):
  if not d.is_dir():continue
  u={'unit_id':d.name,'metadata':{}}
  for name in ['DISPATCH.json','STARTED.json','RESULT.json','CONTROLLER_RECEIPT.json','EARLIEST_NUMERICAL_FAILURE.json','numeric/NUMERICAL_INDEX.json']:u['metadata'][name]=meta(d/name)
  pr=stable(d/'PROGRESS.jsonl')
  if pr:
   rows=[json.loads(s)for s in pr.pop('raw').splitlines()if s.strip()];pr.update(rows_count=len(rows),first=rows[0]if rows else None,last=rows[-1]if rows else None,phase_transitions=[x for x in rows if x.get('progress_kind')=='phase_transition'],token_commits=[x for x in rows if x.get('progress_kind')=='token_committed'])
  u['progress']=pr;u['token_receipt_filenames']=[p.name for p in sorted((d/'numeric').glob('token_*.json'))]
  log=stable(d/'stdout.log')
  if log:log['tail']='\n'.join(log.pop('raw').splitlines()[-5:])
  u['log']=log;v['units'].append(u)
 out['cards']['gpu'+str(n)]=v
out['historical_PID_current_observations']={str(p):proc(p)for p in sorted(pids)}
out['actual_current_task_processes']=[]
for p in Path('/proc').glob('[0-9]*'):
 x=proc(int(p.name))
 if x and any(str(R)in s for s in x['cmdline']):out['actual_current_task_processes'].append(x)
for name,args in [('NVML_compute_apps',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),('NVML_GPUs',['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits']),('tmux_sessions',['tmux','list-sessions','-F','#{session_name}'])]:
 p=subprocess.run(args,text=True,capture_output=True,timeout=10);out[name]={'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(out,allow_nan=False))
'''.replace('R=Path(REMOTE_ROOT)','R=Path('+repr(REMOTE)+')')
def main():
 p=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55','/data/ydai/miniconda3/envs/bitdecode/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr)
 x=json.loads(p.stdout);x['local_received_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();x['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();target=Path(__file__).parent/'RESOURCE_TERMINAL_OBSERVATIONS.json'
 with target.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'compute_apps':x['NVML_compute_apps']['stdout'],'task_processes':x['actual_current_task_processes'],'units':{k:[{'unit_id':u['unit_id'],'RESULT_status':u['metadata']['RESULT.json']['json']['status']if u['metadata']['RESULT.json']else None,'committed_receipts':len(u['token_receipt_filenames']),'STARTED':u['metadata']['STARTED.json']is not None}for u in v['units']]for k,v in x['cards'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
