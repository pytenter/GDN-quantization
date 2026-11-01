"""Read-only remote process/progress snapshot; never read correctness or launch/retry workers."""
import json
import shlex
import subprocess
from pathlib import Path

B = Path(__file__).resolve().parents[1]
R = "/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1/repair_20261017_v1"
PY = "/data/ydai/miniconda3/envs/bitdecode/bin/python"

CODE = r'''
from pathlib import Path
import json, subprocess, time
r=Path(ROOT)
def read(p):
 if not p.exists():return None
 return json.loads(p.read_text())
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return dict(start_tick=stat[stat.rfind(')')+2:].split()[19],cmdline=cmd)
 except FileNotFoundError:return None
def cmd(args):
 x=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr)
ledger=read(r/'GPU_RESOURCE_LEDGER.json')
o=dict(server_unix=time.time(),server_monotonic=time.monotonic(),ledger_status=ledger.get('status'),
 cumulative_GPU_hours=ledger.get('cumulative_GPU_hours'),
 this_repaired_run_GPU_hours=ledger.get('cumulative_GPU_hours',0)-ledger.get('prior_failed_run_GPU_hours',0),
 all_GPUs_released=ledger.get('all_GPUs_released'),failure=read(r/'FIRST_FAILURE.json'),
 formal_stage_result=read(r/'execution/formal/RESULT.json'),scoring_status=read(r/'SCORING_STATUS.json'))
o['dispatches']=[dict(**x,actual_process=identity(x['pid'])) for x in ledger.get('dispatches',[]) if x['stage']=='formal']
o['coordinator']=read(r/'execution/formal/STARTED.json')
if o['coordinator']:o['coordinator']['actual_process']=identity(o['coordinator']['pid'])
o['workers']={};counts={'STEPQUANT6':0,'H_DUALM':0};incomplete=[]
for g in range(4):
 log=r/f'execution/formal/gpu{g}.log';events=[]
 if log.exists():
  for line in log.read_bytes().splitlines():
   try:x=json.loads(line)
   except Exception:continue
   if x.get('event')=='PROGRESS':events.append(x)
 rows=[]
 for p in sorted((r/f'formal/worker{g}').glob('*_*.json')):
  x=read(p)
  if x.get('method') not in counts or 'generated_token_ids' not in x:continue
  row={k:x.get(k) for k in ('method','canonical_index','seed','problem_id','status','output_tokens','latency_s','tokens_per_s','finish_reason','runtime_error')}
  rows.append(row)
  if x.get('status')=='COMPLETE':counts[x['method']]+=1
  else:incomplete.append(str(p.relative_to(r)))
 o['workers'][str(g)]={'last_progress':events[-1] if events else None,'recent_progress':events[-3:],'records':rows,
  'log_bytes':log.stat().st_size if log.exists() else 0,'log_mtime_unix':log.stat().st_mtime if log.exists() else None,
  'worker_result':read(r/f'formal/worker{g}/RESULT.json'),'worker_failure':read(r/f'formal/worker{g}/FIRST_FAILURE.json')}
o['complete_generations']=counts;o['incomplete_records']=incomplete
o['NVML']=cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])
o['GPU']=cmd(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'])
o['disk']=cmd(['df','-h','/data','/tmp']);o['tmux']=cmd(['tmux','list-sessions'])
print(json.dumps(o,ensure_ascii=False,allow_nan=False))
'''.replace("ROOT", repr(R))


def main():
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-S", "/tmp/qwen_gdn_replication_20261013.sock",
         "zypan@202.38.247.55", shlex.join([PY, "-c", CODE])],
        capture_output=True, text=True, check=True, timeout=60,
    )
    data = json.loads(result.stdout)
    (B / "LATEST_LIVE_SNAPSHOT.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary = {k: data[k] for k in ("server_unix", "ledger_status", "cumulative_GPU_hours", "this_repaired_run_GPU_hours", "complete_generations", "failure", "all_GPUs_released")}
    summary["workers"] = {g: {"last_progress": v["last_progress"], "completed": len(v["records"]), "failure": v["worker_failure"]} for g, v in data["workers"].items()}
    summary["NVML"] = data["NVML"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
