"""Deploy finite CPU preparation only; no GPU or science launch."""
from pathlib import Path
import tarfile,io,subprocess,shlex,json,hashlib
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'ROOT.json').read_text());buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for p in sorted(R.rglob('*')):
  if p.is_file() and p.suffix!='.log' and p.name!='PAYLOAD_PREPARATION.json':t.add(p,arcname=str(p.relative_to(R)),recursive=False)
payload=buf.getvalue();(R/'analysis/PAYLOAD_PREPARATION.json').write_text(json.dumps(dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),CPU_only=True),indent=2)+'\n')
code='''import sys,tarfile,io,json,subprocess,os,hashlib
from pathlib import Path
R=Path(sys.argv[1]);assert not R.exists(),'New root already exists; inspect, no blind overwrite';R.mkdir(parents=True)
b=sys.stdin.buffer.read();t=tarfile.open(fileobj=io.BytesIO(b),mode='r:gz')
for m in t.getmembers():
 p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and m.isfile(),'Unsafe deployment member'
 q=R/p;q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('xb') as f:f.write(t.extractfile(m).read())
parent=Path('/data/zypan/analyses/qwen_gdn_historical_port_int8_group32_qualification_v1/gpu_continuation_20261019_v1');manifest=json.loads((R/'evidence/group32/SOURCE_FREEZE.json').read_text());count=0
for x in manifest['files']:
 p=parent/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],'Actual final Group32 source changed:'+str(p);count+=1
with (R/'analysis/REMOTE_PARENT_500_FRESH_SHA.json').open('x') as f:json.dump(dict(status='PASS_REMOTE_ACTUAL_SOURCE_500',count=count,GPU_dispatches=0),f,indent=2)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for rep in [0,1]:
 p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/cpu_qualification.py')],cwd=R,env=env,capture_output=True,timeout=180)
 (R/'qualification'/('CPU_PROCESS_'+str(rep)+'.log')).write_bytes(p.stdout+p.stderr)
 if p.returncode:print((p.stdout+p.stderr).decode());raise RuntimeError('CPU qualification failed; original log preserved')
 v=json.loads(next(s[len('RESULT_JSON='):] for s in p.stdout.decode().splitlines() if s.startswith('RESULT_JSON=')))
 (R/'qualification'/('CPU_PROCESS_'+str(rep)+'.json')).write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(dict(repeat=rep,status=v['status'],checks=v['count'])),flush=True)
a=json.loads((R/'qualification/CPU_PROCESS_0.json').read_text());b=json.loads((R/'qualification/CPU_PROCESS_1.json').read_text());assert a==b,'Fresh CPU exact trace differs'
with (R/'CPU_CODEC_QUALIFICATION.json').open('x') as f:json.dump(dict(status='CPU_ADDITIONS_QUALIFIED',actual_checks_per_process=a['count'],fresh_CPU_processes=2,actual_full_result_equal=True,scope=a['scope'],GPU_dispatches=0,source_protocol='Original B1/B2 arithmetic and persistent writeback byte-identical; Native hook-only coordinate addition'),f,indent=2)
p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/freeze_inputs_cpu.py')],cwd=R,env=env,capture_output=True,timeout=180)
(R/'qualification/CPU_INPUT_FREEZE.log').write_bytes(p.stdout+p.stderr);print((p.stdout+p.stderr).decode(),flush=True);assert p.returncode==0,'CPU input freeze failed'
print(json.dumps(dict(status='CPU_PHASE_COMPLETE_GPU_NOT_STARTED',GPU_dispatches=0)))
'''
p=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',shlex.join([cfg['remote_python'],'-u','-B','-c',code,cfg['remote_root']])],input=payload,capture_output=True,timeout=570)
(R/'qualification/DEPLOY_CPU_STDOUT.log').write_bytes(p.stdout);(R/'qualification/DEPLOY_CPU_STDERR.log').write_bytes(p.stderr);print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
