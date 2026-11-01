#!/usr/bin/env python3
"""Independent read-only stopped-run metadata collector. No GPU workloads/signals."""
import datetime,hashlib,json,subprocess
from pathlib import Path
LOCAL=Path(__file__).resolve().parents[2]
REMOTE='/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_20261019_V1'
CODE=r'''import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(REMOTE_ROOT)
def stable(p):
 p=Path(p)
 if not p.exists():return None
 a=p.stat();raw=p.read_bytes();b=p.stat()
 if(a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns):raise RuntimeError('File changed during audit read:'+str(p))
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw=raw.decode())
def meta(p):
 f=stable(p)
 if f is None:return None
 f['json']=json.loads(f.pop('raw'));return f

def proc(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();return dict(pid=pid,starttick=int(s[s.rfind(')')+2:].split()[19]),cmdline=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
 except(FileNotFoundError,ProcessLookupError,PermissionError):return None
out=dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_unix=time.time(),remote_monotonic=time.monotonic(),root=str(R),shared={},cards={})
for name in ['FIRST_FAILURE.json','SESSION_STARTED.json','SESSION_RESULT.json']+[f'COORDINATOR_{n}_{kind}.json'for n in range(4)for kind in ['SPAWN','EXIT']]:out['shared'][name]=meta(R/name)
expected_pids=set()
for n in range(4):
 c=R/('gpu'+str(n));v=dict(root=str(c),metadata={},units=[])
 for name in ['FIRST_FAILURE.json','RESOURCE_LEDGER.json','execution/CONTROLLER_STARTED.json','execution/CONTROLLER_RESULT.json','qualification/FP_NUMERICAL_VERDICT.json','qualification/PERFORMANCE_VERDICT.json','qualification/resource_event_reducer.py']:
  if name.endswith('.py'):v['reducer_source']=stable(c/name)
  else:v['metadata'][name]=meta(c/name)
 ev=stable(c/'execution/RESOURCE_EVENTS.jsonl');v['resource_event_stream']=ev
 dirs=sorted(p for p in(c/'execution').glob('*')if p.is_dir())
 for d in dirs:
  unit=dict(unit_id=d.name,metadata={})
  for name in ['DISPATCH.json','STARTED.json','RESULT.json','CONTROLLER_RECEIPT.json','FIRST_FAILURE.json']:
   m=meta(d/name);unit['metadata'][name]=m
   if m and name=='STARTED.json':expected_pids.add(m['json']['pid'])
  progress=stable(d/'PROGRESS.jsonl')
  if progress:
   rows=[json.loads(s)for s in progress.pop('raw').splitlines()if s.strip()];unit['progress']=dict(progress,rows_count=len(rows),first=rows[0]if rows else None,last=rows[-1]if rows else None,last_token_committed=next((x for x in reversed(rows)if x.get('progress_kind')=='token_committed'),None),phase_transitions=[x for x in rows if x.get('progress_kind')=='phase_transition'])
  else:unit['progress']=None
  tokens=[]
  for p in sorted((d/'numeric').glob('token_*.json')):
   m=meta(p);z=m['json'];tokens.append(dict(filename=p.name,token=z.get('token'),status=z.get('status'),sha256=m['sha256'],bytes=m['bytes']))
  unit['numeric_committed_receipts']=tokens
  unit['numeric_manifest']=meta(d/'numeric/MANIFEST.json')
  log=d/'stdout.log';lf=stable(log)
  if lf:lf['tail']='\n'.join(lf.pop('raw').splitlines()[-24:]);unit['stdout_log']=lf
  v['units'].append(unit)
 cm=v['metadata']['execution/CONTROLLER_STARTED.json']
 if cm:expected_pids.add(cm['json']['controller_pid'])
 out['cards'][c.name]=v
for m in out['shared'].values():
 if not m:continue
 z=m['json']
 if isinstance(z.get('identity'),dict)and 'pid'in z['identity']:expected_pids.add(z['identity']['pid'])
out['historical_PID_current_observations']={str(pid):proc(pid)for pid in sorted(expected_pids)}
actual=[]
for p in Path('/proc').glob('[0-9]*'):
 z=proc(int(p.name))
 if z and any(str(R)in x for x in z['cmdline']):actual.append(z)
out['current_matching_processes']=actual
for key,args in [('NVML_GPU', ['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits']),('NVML_COMPUTE_APPS',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),('tmux_sessions',['tmux','list-sessions','-F','#{session_name}'])]:
 p=subprocess.run(args,text=True,capture_output=True,timeout=10);out[key]=dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out['disk_available_bytes']={name:(lambda s:s.f_bavail*s.f_frsize)(os.statvfs(name))for name in ['/data','/tmp','/home']}
print(json.dumps(out,allow_nan=False))
'''.replace('R=Path(REMOTE_ROOT)','R=Path('+repr(REMOTE)+')')
def main():
 p=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55','/data/ydai/miniconda3/envs/bitdecode/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr)
 x=json.loads(p.stdout);x['local_received_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();x['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 target=Path(__file__).parent/'RESOURCE_TERMINAL_OBSERVATIONS.json'
 with target.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),cards={k:dict(units=[dict(unit_id=u['unit_id'],committed=len(u['numeric_committed_receipts']),RESULT_status=u['metadata']['RESULT.json']['json'].get('status')if u['metadata']['RESULT.json']else None)for u in v['units']])for k,v in x['cards'].items()},NVML_COMPUTE_APPS=x['NVML_COMPUTE_APPS'],matching_processes=x['current_matching_processes']),ensure_ascii=False))
if __name__=='__main__':main()
