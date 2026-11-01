#!/usr/bin/env python3
"""Independent read-only SSH snapshot. No response text, gold, score or dispatch.

Only current root metadata/progress, exact process identities, NVML, resources,
ledger tail and finisher state are read. Other task processes are allowed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_resident_readonly_delivery_common", ROOT / "operations/download_verified_delivery.py")
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)


def snapshot_code(config):
    return common.REMOTE_HELPERS + "\nr=Path(" + repr(config["remote_root"]) + ")\n" + r'''
v=dict(status='READ_ONLY_GOLD_BLIND_SNAPSHOT',remote_monotonic=time.monotonic(),remote_unix=time.time(),host=socket.gethostname(),remote_root=str(r))
def maybe(n):return read(r/n)if(r/n).is_file()else None
for n in('CONTROLLER_STARTED.json','PHASE_STATUS.json','SESSION_RESULT.json','FIRST_FAILURE.json','FINISHER_STARTED.json','FINISHER_RESULT.json','FINISHER_FAILURE.json','FORMAL_GENERATIONS_COMPLETE.json','AUTHORIZATION_FORMAL.json','FINAL_SEAL_RECEIPT.json'):
 x=maybe(n)
 if x is not None:v[n]=x
if(r/'SOURCE_FREEZE.json').is_file():
 f=read(r/'SOURCE_FREEZE.json');v['source']=dict(sha256=sha(r/'SOURCE_FREEZE.json'),count=len(f['files']))
keys=('phase','condition','stage','token','tokens','step','input_index','input_forwards','generated_tokens','generated_token_count','count','monotonic')
v['units']={};expected=[]
for folder in('qualification','formal'):
 rows=[]
 for p in sorted((r/folder).glob('*')):
  if not p.is_dir():continue
  row=dict(id=p.name,result_exists=(p/'RESULT.json').is_file(),failure_exists=(p/'FIRST_FAILURE.json').is_file(),exit_release_exists=(p/'RESOURCE_INTERVAL.json').is_file())
  if(p/'PROGRESS.json').is_file():
   progress=read(p/'PROGRESS.json');row['progress']={k:progress[k]for k in keys if k in progress}
  if(p/'STARTED.json').is_file():
   started=read(p/'STARTED.json');x=started.get('identity');row['identity']=x
   if isinstance(x,dict):expected.append(x)
  if(p/'RESOURCE_INTERVAL.json').is_file():
   interval=read(p/'RESOURCE_INTERVAL.json');row['exit_returncode']=interval.get('exit_returncode');row['release']=interval.get('release')
  rows.append(row)
 v['units'][folder]=rows
for name in('CONTROLLER_STARTED.json','FINISHER_STARTED.json'):
 if isinstance(v.get(name),dict)and isinstance(v[name].get('identity'),dict):expected.append(v[name]['identity'])
v['exact_process_observations']=[dict(expected=x,current=identity(x['pid']),exact_alive=identity(x['pid'])==x)for x in expected]
v['actual_own_processes']=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  x=identity(int(p.name))
  if execution_process(x,r):v['actual_own_processes'].append(x)
for label,cmd in(
 ('GPU',['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
 ('NVML',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])):
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=20);v[label]=dict(returncode=x.returncode,rows=x.stdout.splitlines(),stderr=x.stderr[-1000:])
v['NVML']['identity_rows']=[]
if v['NVML']['returncode']==0:
 for line in v['NVML']['rows']:
  if not line.strip():continue
  a=[x.strip()for x in line.split(',')];x=identity(int(a[1]));v['NVML']['identity_rows'].append(dict(uuid=a[0],pid=int(a[1]),identity=x,own_root=belongs(x,r)))
v['disk']={}
for name in('/data','/dev/shm'):
 s=os.statvfs(name);v['disk'][name]=dict(free_bytes=s.f_bavail*s.f_frsize,total_bytes=s.f_blocks*s.f_frsize)
p=r/'GPU_RESOURCE_LEDGER.jsonl';v['ledger_tail']=[]
if p.is_file():
 with p.open('rb')as f:
  offset=max(0,p.stat().st_size-512*1024);f.seek(offset)
  if offset:f.readline()
  for line in f:
   try:v['ledger_tail'].append(json.loads(line))
   except json.JSONDecodeError:pass
 v['ledger_tail']=v['ledger_tail'][-100:]
# No worker log tails, RESULT payloads, response strings, gold, scoring files,
# accuracy or answer-dependent scheduling are read by this snapshot.
print(json.dumps(v,sort_keys=True))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--brief", action="store_true")
    args = parser.parse_args(argv); config = common.read(ROOT / "ROOT.json")
    common.require(ROOT.name == common.ATTEMPT and config['task_id'] == common.TASK,
                   "Snapshot must use this exact independent task root")
    state = common.remote_json(config, "55", snapshot_code(config))
    directory = ROOT / "monitoring"; directory.mkdir(exist_ok=True)
    path = directory / ("SNAPSHOT_" + time.strftime("%Y%m%dT%H%M%S") + "_" + str(time.monotonic_ns()) + ".json")
    common.write_new(path, json.dumps(state, ensure_ascii=False, indent=2).encode() + b"\n")
    if args.brief:
        state = dict(snapshot=str(path), source=state.get("source"), phase=state.get("PHASE_STATUS.json"),
            session=state.get("SESSION_RESULT.json"), failure=state.get("FIRST_FAILURE.json"), finisher=state.get("FINISHER_RESULT.json"),
            finisher_failure=state.get("FINISHER_FAILURE.json"),
            units={folder: [dict(id=x['id'],passed=x['result_exists'],failed=x['failure_exists'],progress=x.get('progress'),release=x.get('release')) for x in rows]
                   for folder,rows in state['units'].items()}, actual_own_processes=state['actual_own_processes'],
            GPU=state['GPU'], NVML=state['NVML'], disk=state['disk'])
    print(json.dumps(state, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
