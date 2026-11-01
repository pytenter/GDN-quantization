"""Frozen CPU-only55->29 SSH transport. No reverse credentials or arbitrary foreign-root control."""
import json,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','zypan@202.38.247.29']
class ConnectionPending(RuntimeError):pass
def query29(code,timeout=55):
    try:p=subprocess.run(SSH+['env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as e:raise ConnectionPending('Read-only29 connection timeout')from e
    if p.returncode==255:raise ConnectionPending('Read-only29 SSH255: '+p.stderr[-1000:])
    if p.returncode:raise RuntimeError('Remote task CPU command failed: '+p.stderr[-5000:])
    return json.loads(p.stdout)
def prefix():
    return "import os,sys,json,time,subprocess\nfrom pathlib import Path\nR=Path("+repr(str(ROOT))+");sys.path.insert(0,str(R/'src'));import execution_control as C\n"
def state_code():
    return prefix()+'''
out=dict(node_id=C.NODE['id'],source_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),monotonic=time.monotonic(),status='RUNNING')
for name in ['GPU_RUN_STARTED.json','FIRST_FAILURE.json','SESSION_RESULT.json','FORMAL_PROGRESS.json','LOCAL_QUALIFICATION_READY.json','operations/NODE_ACTUAL_EXIT.json','operations/NODE_PARENT_STARTED.json']:
 if(R/name).is_file():out[name]=C.read(R/name)
start=out.get('GPU_RUN_STARTED.json');out['controller_alive']=bool(start and C.same_process(start['identity']))
parent=out.get('operations/NODE_PARENT_STARTED.json');out['node_parent_alive']=bool(parent and C.same_process(parent['identity']))
out['node_tmux_alive']=subprocess.run(['tmux','has-session','-t','qwen-g64-cross-v4-node29'],capture_output=True).returncode==0 if C.NODE['id']=='server29_gpu6'else False
out['active']=[]
for folder in ['execution','formal']:
 for p in(R/folder).glob('*/STARTED.json'):
  x=C.read(p)
  if C.same_process(x['identity']):out['active'].append(dict(unit=p.parent.name,identity=x['identity'],progress=C.read(p.parent/'PROGRESS.json')if(p.parent/'PROGRESS.json').exists()else None))
out['observation']=C.query_gpu()
print(json.dumps(out))
'''
