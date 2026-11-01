#!/usr/bin/env python3
"""Read-only observation. Does not start/restart units or read scientific values."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_20261019_V1'

def main():
    code='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT)
def get(p):
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except Exception as e:return {'read_error':str(e)}
out={'remote_unix':time.time(),'root':str(root),'FIRST_FAILURE':get(root/'FIRST_FAILURE.json'),'SESSION_RESULT':get(root/'SESSION_RESULT.json'),'SESSION_STARTED':get(root/'SESSION_STARTED.json'),'cards':{}}
for n in range(4):
 c=root/('gpu'+str(n));units=[]
 for d in sorted((c/'execution').glob('*')):
  if not d.is_dir():continue
  progress=d/'PROGRESS.jsonl';last=None
  if progress.exists():
   for line in progress.read_text().splitlines()[-5:]:
    try:last=json.loads(line)
    except Exception:pass
  r=get(d/'RESULT.json');units.append({'unit_id':d.name,'started':get(d/'STARTED.json'),'result_status':r.get('status')if r else None,'progress':last,'committed_token_receipts':len(list((d/'numeric').glob('token_*.json')))})
 out['cards'][c.name]={'controller':get(c/'execution/CONTROLLER_STARTED.json'),'FIRST_FAILURE':get(c/'FIRST_FAILURE.json'),'controller_result':get(c/'execution/CONTROLLER_RESULT.json'),'ledger':get(c/'RESOURCE_LEDGER.json'),'units':units}
out['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
out['compute_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
identities=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  argv=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')]
  if not any(str(root)in x for x in argv):continue
  stat=(p/'stat').read_text();identities.append({'pid':int(p.name),'starttick':int(stat[stat.rfind(')')+2:].split()[19]),'cmdline':argv})
 except(FileNotFoundError,ProcessLookupError,PermissionError):pass
out['actual_process_identities']=identities
out['disk_available_bytes']={p:(lambda s:s.f_bavail*s.f_frsize)(os.statvfs(p))for p in ['/data','/tmp','/home']}
print(json.dumps(out,allow_nan=False))
'''.replace('root=Path(ROOT)','root=Path('+repr(REMOTE)+')')
    proc=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55','/data/ydai/miniconda3/envs/bitdecode/bin/python -B -'],input=code,text=True,capture_output=True,timeout=60)
    if proc.returncode:raise RuntimeError(proc.stderr)
    result=json.loads(proc.stdout);stamp=datetime.now(timezone.utc)
    result['local_received_utc']=stamp.isoformat()
    path=ROOT/'monitoring'/('SNAPSHOT_'+stamp.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'snapshot':str(path),'FIRST_FAILURE':result['FIRST_FAILURE'],'SESSION_RESULT':result['SESSION_RESULT'],'gpus':result['gpus'],'compute_apps':result['compute_apps'],'cards':{k:[{p:q for p,q in u.items()if p not in ['started']}for u in v['units']]for k,v in result['cards'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
