"""Read-only SSH invocation of supplemental post-STOP stdlib identity audit."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parents[1]
cfg=json.loads((root/'ROOT.json').read_text())
script=here/'audit_frozen_sources_remote.py'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
cfg.update(auditor_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),receipt_name='REMOTE_SOURCE_AUDIT_'+stamp+'.json')
program=script.read_text()+'\nmain('+repr(cfg)+')\n'
command=[
    'ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
    '-p',str(cfg['remote_port']),cfg['remote_user']+'@'+cfg['remote_host'],
    shlex.quote(cfg['remote_python'])+' -B -']
result=subprocess.run(command,input=program.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
stderr=here/('SSH_STDERR_'+stamp+'.txt');stderr.write_bytes(result.stderr)
if result.returncode:
    failure=here/('AUDIT_INVOCATION_FAILURE_'+stamp+'.json')
    failure.write_text(json.dumps(dict(status='AUDIT_TOOL_FAILURE_NOT_GPU_RETRY',returncode=result.returncode,stderr=result.stderr.decode(errors='replace')),ensure_ascii=False,indent=2)+'\n')
    print(failure);sys.exit(result.returncode)
receipt=json.loads(result.stdout)
target=here/cfg['receipt_name']
with target.open('xb') as f:f.write(result.stdout)
print(json.dumps(dict(status=receipt['status'],receipt=str(target),receipt_sha256=hashlib.sha256(result.stdout).hexdigest(),external=receipt['fresh_external_dependencies']['passed'],elapsed_seconds=receipt['elapsed_seconds']),ensure_ascii=False))
