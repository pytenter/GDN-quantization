"""Single tmux launch request; any uncertain outcome requires observation, not retry."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'ROOT.json').read_text());remote=meta['remote_root'];session=meta['tmux_session']
program=r'''
import hashlib,json,os,shlex,subprocess,time
from pathlib import Path
root=Path(REMOTE);session=SESSION
for name in ['SESSION_STARTED.json','SESSION_RESULT.json','FIRST_FAILURE.json','OFFLINE_LAUNCH_REQUEST.json']:
 if(root/name).exists():raise RuntimeError('Existing launch/work/failure marker prohibits duplicate request:'+name)
v=json.loads((root/'REMOTE_STAGE_VERIFICATION.json').read_text())
if v['status']!='PASS_EXACT_PAYLOAD_AND_FRESH_EXTERNAL_SHA_BYTES_CPU_ONLY':raise RuntimeError('Stage identity not verified')
probe=json.loads((root/'gpu0/qualification/LIVE_ARCHIVE_TRANSPORT_CPU.json').read_text())
if probe['status']!='PASS_CPU_STORAGE_TRANSPORT_ONLY':raise RuntimeError('Storage transport not verified')
if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode==0:raise RuntimeError('Existing tmux session')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
if apps.strip():raise RuntimeError('GPU compute job present before four-card launch')
freeze=json.loads((root/'SOURCE_FREEZE.json').read_text())
for name,key in [('CONTROL_SOURCE_MANIFEST.json','control_manifest_sha256'),('PAYLOAD_BUILD_MANIFEST.json','payload_build_manifest_sha256')]:
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=freeze[key]:raise RuntimeError('Frozen anchor changed')
args=['/data/ydai/miniconda3/envs/bitdecode/bin/python','-B',str(root/'scripts/run_four_card_session.py'),'--root',str(root)]
command='exec '+shlex.join(args)+' > '+shlex.quote(str(root/'session.log'))+' 2>&1'
receipt=dict(status='SINGLE_OFFLINE_LAUNCH_REQUEST',session=session,root=str(root),command=command,unix=time.time(),control_manifest_sha256=freeze['control_manifest_sha256'],no_GPU_time_limit=True,no_automatic_retry=True)
with(root/'OFFLINE_LAUNCH_REQUEST.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
print(json.dumps(receipt))
'''.replace('Path(REMOTE)','Path('+repr(remote)+')').replace('session=SESSION','session='+repr(session))
command='/data/ydai/miniconda3/envs/bitdecode/bin/python -B -c '+shlex.quote(program)
proc=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',command],capture_output=True,text=True,timeout=60)
(ROOT/'analysis/OFFLINE_LAUNCH_STDERR.txt').write_text(proc.stderr)
if proc.returncode:
 (ROOT/'analysis/OFFLINE_LAUNCH_ERROR_STDOUT.txt').write_text(proc.stdout)
 raise RuntimeError('Launch command failed/uncertain; inspect true process and markers before any action')
with(ROOT/'analysis/OFFLINE_LAUNCH_RECEIPT.json').open('x')as f:f.write(proc.stdout)
print(proc.stdout)
