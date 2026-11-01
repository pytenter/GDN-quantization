"""CPU execution and durable diagnostics; no GPU calls."""
import sys,subprocess,os,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
(R/'qualification').mkdir(exist_ok=True)
with (R/'qualification/CONTROL_PREPARATION_LOGGING_FAILURE.json').open('x') as f:json.dump(dict(status='PREEXECUTION_CONTROL_ERROR',error='Initial archive omitted empty qualification directory; first CPU subprocess result could not be persisted. No GPU dispatch or claimed CPU pass; durable output directories fixed in new CPU-only control source.',GPU_dispatches=0),f,indent=2)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for rep in [0,1]:
 logfile=R/'qualification'/('CPU_PROCESS_'+str(rep)+'.log')
 with logfile.open('xb') as f:p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/cpu_qualification.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 if p.returncode:print(logfile.read_text(),flush=True);raise RuntimeError('CPU qualification failed; log retained')
 v=json.loads(next(s[len('RESULT_JSON='):] for s in logfile.read_text().splitlines() if s.startswith('RESULT_JSON=')))
 (R/'qualification'/('CPU_PROCESS_'+str(rep)+'.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(repeat=rep,status=v['status'],checks=v['count'])),flush=True)
a=json.loads((R/'qualification/CPU_PROCESS_0.json').read_text());b=json.loads((R/'qualification/CPU_PROCESS_1.json').read_text());assert a==b,'Fresh CPU exact trace differs'
with (R/'CPU_CODEC_QUALIFICATION.json').open('x') as f:json.dump(dict(status='CPU_ADDITIONS_QUALIFIED',actual_checks_per_process=a['count'],fresh_CPU_processes=2,actual_full_result_equal=True,scope=a['scope'],GPU_dispatches=0,source_protocol='Original B1/B2 arithmetic and persistent writeback byte-identical; Native hook-only coordinate addition'),f,indent=2)
with (R/'qualification/CPU_INPUT_FREEZE.log').open('xb') as f:p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/freeze_inputs_cpu.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
print((R/'qualification/CPU_INPUT_FREEZE.log').read_text(),flush=True);assert p.returncode==0,'CPU input freeze failed'
print(json.dumps(dict(status='CPU_PHASE_COMPLETE_GPU_NOT_STARTED',GPU_dispatches=0)),flush=True)
