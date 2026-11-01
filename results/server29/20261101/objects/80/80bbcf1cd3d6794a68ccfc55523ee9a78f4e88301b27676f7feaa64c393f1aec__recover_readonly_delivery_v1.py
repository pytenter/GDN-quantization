"""Resume only the previously disconnected read-only delivery; never inference."""
import json,sys,os,time,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
O=R/'operations'
assert not (R/'delivery').exists(),'Collection already exists; inspect stage before recovery'
assert (O/'LOCAL_FINISHER_FAILURE.json').exists(),'Prior delivery failure must remain'
old=json.loads((O/'LOCAL_DELIVERY_PROCESS.json').read_text())
p=subprocess.run(['ps','-p',str(old['pid']),'-o','command='],capture_output=True,text=True)
assert 'finalize_delivery.py' not in p.stdout,'Prior finisher still active'
for x in json.loads((O/'DELIVERY_TOOL_MANIFEST.json').read_text())['files']:
 p=R/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
with (O/'READONLY_RECOVERY_STARTED.json').open('x') as f:
 json.dump(dict(pid=os.getpid(),argv=sys.argv,unix=time.time(),GPU_dispatches=0,reason='Original local SSH wait timed out; remote finite GPU roster completed PASS and released. Original failure retained.'),f,indent=2)
try:
 for n in ['collect_terminal.py','verify_terminal_archive.py','verify_post_run_dependencies.py','write_gpu_reports.py','seal_gpu_delivery.py']:
  print(json.dumps(dict(stage=n,unix=time.time())),flush=True)
  subprocess.run([sys.executable,'-B',str(O/n)],check=True)
 result=dict(status='PASS_VERIFIED_FINAL_DELIVERY',task_result=json.loads((R/'delivery/TASK_RESULT.json').read_text()),local_verification=json.loads((R/'delivery/LOCAL_DELIVERY_VERIFICATION.json').read_text()),GPU_dispatches=0,finished_unix=time.time())
 with (O/'READONLY_RECOVERY_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(status=result['status'],task_status=result['task_result']['status'])),flush=True)
except BaseException as exc:
 with (O/'READONLY_RECOVERY_FAILURE.json').open('x') as f:json.dump(dict(error=repr(exc),unix=time.time(),GPU_dispatches=0),f,indent=2)
 raise
