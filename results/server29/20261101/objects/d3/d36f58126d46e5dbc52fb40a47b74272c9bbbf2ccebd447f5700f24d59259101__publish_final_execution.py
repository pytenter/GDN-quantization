from pathlib import Path
import json,base64,subprocess,shlex
R=Path(__file__).resolve().parents[1];c=json.loads((R/'ROOT.json').read_text());names=[str(p.relative_to(R)) for f in ['src','scripts'] for p in (R/f).iterdir() if p.is_file() and p.suffix in ['.py','.sh']]
files={n:base64.b64encode((R/n).read_bytes()).decode() for n in names}
code='''import sys,json,base64,os,subprocess
from pathlib import Path
R=Path(sys.argv[1]);assert not (R/'GPU_RUN_STARTED.json').exists() and not (R/'SOURCE_FREEZE.json').exists()
for n,b in json.loads(sys.stdin.buffer.read()).items():
 p=R/n;p.write_bytes(base64.b64decode(b))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
with (R/'qualification/FINAL_PRELAUNCH_CPU.log').open('xb') as f:p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/final_prelaunch_cpu.py')],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
print((R/'qualification/FINAL_PRELAUNCH_CPU.log').read_text(),flush=True);assert p.returncode==0,'Final CPU prelaunch failed, noGPU'
'''
p=subprocess.run(['ssh','-S',c['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join([c['remote_python'],'-u','-B','-c',code,c['remote_root']])],input=json.dumps(files).encode(),capture_output=True,timeout=950);(R/'qualification/FINAL_EXECUTION_PUBLISH.log').write_bytes(p.stdout+p.stderr);print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
