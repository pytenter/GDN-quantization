"""One local read-only wait/download/verify/report/seal chain, never dispatches GPU."""
import argparse,json,sys,time,subprocess,os,hashlib
from pathlib import Path
from collect_terminal import R,cfg,ssh
ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');a=ap.parse_args()
manifest=json.loads((R/'operations/DELIVERY_TOOL_MANIFEST.json').read_text())
for x in manifest['files']:
 p=R/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],'Independent tool changed before starting: '+x['path']
lock=R/'operations/LOCAL_DELIVERY_PROCESS.json'
if lock.exists():
 old=json.loads(lock.read_text());raise SystemExit('Prior delivery process registration exists; inspect real process and history before any recovery: '+str(old))
with lock.open('x') as f:json.dump(dict(pid=os.getpid(),argv=sys.argv,started_unix=time.time(),GPU_dispatches=0),f,indent=2)
probe=r'''import sys,json
from pathlib import Path
r=Path(sys.argv[1]);w=r/'execution/WRAPPER_RESULT.json';s=r/'execution/SESSION_RESULT.json'
print(json.dumps(dict(wrapper=json.loads(w.read_text()) if w.exists() else None,session=s.exists())))'''
try:
 while True:
  v=json.loads(ssh(probe,[cfg['remote_root']],timeout=30))
  if v['wrapper'] is not None:
   assert v['session'],'Wrapper ended before session receipt; inspect controller log and real process, do not restart'
   break
  if not a.wait:raise RuntimeError('GPU session still active; use --wait once')
  time.sleep(30)
 tools=['collect_terminal.py','verify_terminal_archive.py','verify_post_run_dependencies.py','write_gpu_reports.py','seal_gpu_delivery.py']
 for n in tools:
  p=R/'operations'/n;print(json.dumps(dict(stage=n,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unix=time.time())),flush=True)
  subprocess.run([sys.executable,'-B',str(p)],check=True)
 result=dict(status='PASS_VERIFIED_FINAL_DELIVERY',task_result=json.loads((R/'delivery/TASK_RESULT.json').read_text()),local_verification=json.loads((R/'delivery/LOCAL_DELIVERY_VERIFICATION.json').read_text()),finished_unix=time.time())
 with (R/'operations/LOCAL_FINISHER_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(status=result['status'],scientific_status=result['task_result']['status'])),flush=True)
except BaseException as exc:
 with (R/'operations/LOCAL_FINISHER_FAILURE.json').open('x') as f:json.dump(dict(status='INDEPENDENT_DELIVERY_TOOL_STOP',error=repr(exc),unix=time.time(),GPU_retry_authorized=False),f,indent=2)
 raise
