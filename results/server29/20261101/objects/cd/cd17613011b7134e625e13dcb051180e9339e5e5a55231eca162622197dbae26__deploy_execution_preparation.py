from pathlib import Path
import json,subprocess,shlex,base64
R=Path(__file__).resolve().parents[1];c=json.loads((R/'ROOT.json').read_text())
names=['ROOT.json','HUMAN_GPU_AUTHORIZATION.json']+[str(p.relative_to(R)) for folder in ['src','scripts'] for p in (R/folder).glob('*.py')]
files={n:base64.b64encode((R/n).read_bytes()).decode() for n in names}
code='''import sys,json,base64,os,subprocess
from pathlib import Path
R=Path(sys.argv[1]);assert not (R/'GPU_RUN_STARTED.json').exists(),'No hotpatch or relaunch'
for n,b in json.loads(sys.stdin.buffer.read()).items():
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for script in ['prepare_gpu_stage.py','test_execution_cpu.py']:
 log=R/'qualification'/(script+'.log')
 with log.open('xb') as f:p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts'/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 print(log.read_text(),flush=True);assert p.returncode==0,'CPU integration failed; log preserved, no GPU'
print('PASS_EXECUTION_PREPARATION_NO_GPU',flush=True)
'''
p=subprocess.run(['ssh','-S',c['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join([c['remote_python'],'-u','-B','-c',code,c['remote_root']])],input=json.dumps(files).encode(),capture_output=True,timeout=550);(R/'qualification/GPU_PREPARATION_DEPLOYMENT.log').write_bytes(p.stdout+p.stderr);print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
