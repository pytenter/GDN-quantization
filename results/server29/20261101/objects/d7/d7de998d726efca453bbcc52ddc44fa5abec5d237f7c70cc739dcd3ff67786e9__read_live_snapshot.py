#!/usr/bin/env python3
"""Read-only remote status; no GPU API, forward, signal, or source mutation."""
import datetime, hashlib, json, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent.parent
REMOTE = r'''
import json,os,pathlib,time,hashlib,collections
r=pathlib.Path('/data/zypan/analyses/qwen_gdn_historical_runtime_gpu_recovery_v1/gpu_execution_20261019_v1');p=r/'payload'
o={'schema':'QWEN_RECOVERY_READ_ONLY_SNAPSHOT_V1','remote_wall_time':time.time(),'monotonic':time.monotonic(),'GPU_queries_by_snapshot':0,'GPU_dispatches_by_snapshot':0}
def read(q):
 try:return json.loads(q.read_text())
 except FileNotFoundError:return None
for rel in ['operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json','payload/FIRST_FAILURE.json','payload/execution/CONTROLLER_RESULT.json','payload/execution/RESOURCE_LEDGER.json']:
 o[rel]=read(r/rel)
q=p/'execution/RESOURCE_EVENTS.jsonl';events=[]
if q.exists():
 raw=q.read_bytes();lines=raw.splitlines()
 for i,line in enumerate(lines):
  try:events.append(json.loads(line))
  except json.JSONDecodeError:
   if i!=len(lines)-1:raise
 o['ledger']={'bytes_observed':len(raw),'prefix_sha256_observed':hashlib.sha256(raw).hexdigest(),'counts':dict(collections.Counter(x['event']for x in events)),'non_poll_events_tail':[x for x in events if x['event']not in ['NVML_OBSERVATION','REAL_PROGRESS_OBSERVED']][-12:],'latest_NVML_poll':next((x for x in reversed(events)if x['event']=='NVML_OBSERVATION'),None)}
o['units']=[]
for unit in ['NUMERIC_FRESH0','NUMERIC_FRESH1','OLD_FP_R0','OLD_FP_R1','OLD_FP_R2','NEW_FP_R0']:
 d=p/'execution'/unit
 if not d.exists():continue
 a={'unit':unit,'RESULT':read(d/'RESULT.json'),'DRIVER':read(d/'DRIVER_ORDINAL_IDENTITY.json'),'intermediate_count':len(list((d/'intermediates').glob('*.json'))),'committed_bin_bytes':sum(q.stat().st_size for q in d.rglob('*.bin')if q.is_file())}
 q=d/'PROGRESS_EVENTS.jsonl'
 if q.exists():
  rows=q.read_bytes().splitlines();a['progress_count']=len(rows)
  for z in reversed(rows):
   try:a['latest_progress']=json.loads(z);break
   except json.JSONDecodeError:pass
 q=d/'worker.log'
 if q.exists():a['worker_log']={'bytes':q.stat().st_size,'tail':q.read_text(errors='replace')[-600:]}
 o['units'].append(a)
o['processes']=[]
for q in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=q.read_bytes().rstrip(b'\0').split(b'\0');cmd=[x.decode(errors='replace')for x in argv]
  if any(str(r)in x for x in cmd)and int(q.parent.name)!=os.getpid():
   stat=(q.parent/'stat').read_text().rsplit(')',1)[1].split();o['processes'].append({'pid':int(q.parent.name),'starttick':int(stat[19]),'cmdline':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
v=os.statvfs(r);o['disk_available_bytes']=v.f_bavail*v.f_frsize
q=r/'operations/native_controller.log';o['native_log_tail']=q.read_text(errors='replace')[-1500:]if q.exists()else None
print(json.dumps(o,ensure_ascii=False,allow_nan=False))
'''

def main():
    proc = subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55','/data/ydai/miniconda3/envs/bitdecode/bin/python -B -'], input=REMOTE, text=True, capture_output=True, timeout=45)
    if proc.returncode: raise RuntimeError(proc.stderr)
    result = json.loads(proc.stdout)
    result['client_utc_observed'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = (json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    directory=HERE/'monitoring';directory.mkdir(exist_ok=True)
    out=directory/('SNAPSHOT_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with out.open('xb') as f:f.write(raw)
    units=[{'unit':x['unit'],'result':x['RESULT'],'driver_status':(x.get('DRIVER')or{}).get('status'),'intermediate_count':x['intermediate_count'],'committed_bin_bytes':x['committed_bin_bytes'],'latest_progress':x.get('latest_progress'),'worker_log':x.get('worker_log')}for x in result['units']]
    print(json.dumps({'path':str(out),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'failure':result['payload/FIRST_FAILURE.json'],'controller_result':result['payload/execution/CONTROLLER_RESULT.json'],'exit_receipt':result['operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json'],'units':units,'processes':result['processes'],'disk_available_bytes':result['disk_available_bytes'],'event_counts':result.get('ledger',{}).get('counts')},ensure_ascii=False))

if __name__=='__main__':main()
