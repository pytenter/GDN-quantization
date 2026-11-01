"""Root control only: verify the completed gates, then one offline frozen formal stage."""
import base64
import json
import shlex
import subprocess
from pathlib import Path

B = Path(__file__).resolve().parents[1]
R = "/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1/repair_20261017_v1"
PY = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
CODE = r'''
from pathlib import Path
import sys,json,hashlib,base64,os,subprocess,shlex,shutil,time
r=Path(ROOT)
for name,value in json.load(sys.stdin).items():
 data=base64.b64decode(value);p=r/name
 if p.exists():assert p.read_bytes()==data,name
 else:
  with p.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
a=json.loads((r/'AUTHORIZATION_FORMAL.json').read_text());q=json.loads((r/'GPU_QUALIFICATION.json').read_text())
assert a['status']=='PASS' and a['allowed_modes']==['formal'] and a['dispatch_GPUs']==[0,1,2,3]
assert q['status']=='PASS' and len(q['gates'])==12 and all(g['status']=='PASS' for g in q['gates'])
assert not (r/'FIRST_FAILURE.json').exists() and not (r/'execution/formal/STARTED.json').exists()
ledger=json.loads((r/'GPU_RESOURCE_LEDGER.json').read_text());assert ledger['all_GPUs_released'] and ledger['status']=='STAGE_COMPLETE_RESOURCE'
assert not any(value['stage']=='formal' for value in ledger['dispatches'])
for rel,digest in a['required_gate_receipts'].items():
 p=r/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
 assert json.loads(p.read_text())['status']=='PASS',rel
for key,filename in [('prereg_sha256','PREREGISTRATION.json'),('source_manifest_sha256','SOURCE_MANIFEST.json'),('runtime_source_manifest_sha256','RUNTIME_SOURCE_MANIFEST.json'),('units_sha256','FROZEN_UNITS.json'),('model_tokenizer_identity_sha256','MODEL_TOKENIZER_FRESH_IDENTITY.json'),('plan_sha256','calibration/qwen35_9b_stepquant6_plan.pt')]:
 assert hashlib.sha256((r/filename).read_bytes()).hexdigest()==a[key],filename
m=json.loads((r/'SOURCE_MANIFEST.json').read_text());count=0
for row in m['execution_sources']+m['external_dependencies']:
 p=r/row['path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'],str(p)
 if 'bytes' in row:assert len(data)==row['bytes']
 count+=1
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert not apps.strip(),apps
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True)
for line in rows.splitlines():
 v=[s.strip() for s in line.split(',')];assert a['gpu_uuid_map'][v[0]]==v[1] and '3090' in v[2] and int(v[3])>=23000
assert shutil.disk_usage(r).free>7*1024**3
session='qwen-aime26-step6-hdualm-repair-v1';assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
py='/data/ydai/miniconda3/envs/bitdecode/bin/python'
coord=shlex.join([py,str(r/'source/task_coordinator.py'),'--root',str(r),'--mode','formal','--authorization',str(r/'AUTHORIZATION_FORMAL.json')])
finish=shlex.join([py,str(r/'source/score_complete_generations.py'),'--root',str(r),'--authorization',str(r/'AUTHORIZATION_FORMAL.json')])
script='\n'.join(['set -e','cd '+shlex.quote(str(r)),coord+' > '+shlex.quote(str(r/'formal_coordinator.log'))+' 2>&1',finish+' > '+shlex.quote(str(r/'formal_finisher.log'))+' 2>&1',''])
rec={'status':'FORMAL_OFFLINE_LAUNCH_ONCE','server_unix':time.time(),'task_id':a['task_id'],'RUN_ID':a['RUN_ID'],'authorization_sha256':hashlib.sha256((r/'AUTHORIZATION_FORMAL.json').read_bytes()).hexdigest(),'qualification_sha256':hashlib.sha256((r/'GPU_QUALIFICATION.json').read_bytes()).hexdigest(),'source_entries_verified':count,'GPU_preflight':rows,'NVML_preflight':apps,'disk_free_bytes':shutil.disk_usage(r).free,'tmux':session,'offline_shell':script,'same120_no_new_conditions':True}
p=r/'execution/FORMAL_OFFLINE_LAUNCH.json';p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as h:json.dump(rec,h,ensure_ascii=False,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
subprocess.run(['tmux','new-session','-d','-s',session,'/bin/bash','-lc',script],check=True)
print(json.dumps({'status':'FORMAL_DISPATCHED_ONCE','GPUs':[0,1,2,3],'source_entries_verified':count,'authorization_sha256':rec['authorization_sha256'],'qualification_sha256':rec['qualification_sha256']}))
'''.replace("ROOT", repr(R))


def main():
    compile(CODE, "root_control_remote_preflight", "exec")
    payload = {name: base64.b64encode((B / name).read_bytes()).decode()
               for name in ("GPU_QUALIFICATION.json", "AUTHORIZATION_FORMAL.json")}
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-S", "/tmp/qwen_gdn_replication_20261013.sock",
         "zypan@202.38.247.55", shlex.join([PY, "-c", CODE])],
        input=json.dumps(payload), capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
