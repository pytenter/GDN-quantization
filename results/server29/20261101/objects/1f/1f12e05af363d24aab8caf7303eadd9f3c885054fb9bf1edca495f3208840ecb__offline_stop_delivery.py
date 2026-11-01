"""One-shot CPU-only STOP delivery in tmux; never dispatches or resumes GPU work."""
import pathlib,json,subprocess,os,time,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
status=ROOT/'analysis/OFFLINE_DELIVERY_STATUS.json'
def put(x):
 with status.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
gate=json.loads((ROOT/'GATE_STATUS_FINAL.json').read_text());ledger=json.loads((ROOT/'gpu_resource_ledger.json').read_text())
assert gate['no_further_GPU_work'] is True and gate['GPU_release_verified'] is True and ledger['final_GPU_release_verified'] is True
assert not status.exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
steps=[]
for script in ['postprocess_scores.py','finalize_report.py']:
 start=time.time();p=subprocess.run([PYTHON,str(ROOT/'analysis_scripts'/script),'--root',str(ROOT)],env=env)
 steps.append(dict(script=script,source_sha256=hashlib.sha256((ROOT/'analysis_scripts'/script).read_bytes()).hexdigest(),returncode=p.returncode,CPU_wall_seconds=time.time()-start))
 if p.returncode:
  put(dict(status='FAIL',CPU_only=True,no_GPU_dispatch=True,steps=steps));raise SystemExit(p.returncode)
put(dict(status='PASS',CPU_only=True,no_GPU_dispatch=True,tmux_session='qwen32k-delivery',steps=steps))
print('CPU STOP DELIVERY COMPLETE; no GPU restart',flush=True)
