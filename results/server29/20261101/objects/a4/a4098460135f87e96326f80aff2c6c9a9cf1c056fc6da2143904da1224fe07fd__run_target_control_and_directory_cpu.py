#!/usr/bin/env python3
"""Two fresh target-Python control/actual guard tests; all bad fixtures outside science root."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
ROOT=Path(__file__).resolve().parents[1]
REMOTE=json.loads((ROOT/'ROOT.json').read_text())['remote_root']
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH=['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=20','zypan@202.38.247.55']
PROGRAM=r'''
import base64,json,os,pathlib,subprocess,sys,tempfile
r=pathlib.Path(sys.argv[1]);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
assert not any((r/x).exists()for x in ['SOURCE_FREEZE.json','CONTROLLER_STARTED.json','GPU_PERMIT.json'])
result=[]
for name,script,aggregate,status,expected in [
 ('CONTROL','test_resident_control_cpu.py','CONTROL_TARGET_FINAL_TWO_FRESH_RESULT.json','PASS_FINAL_TWO_ACTUAL_TARGET_PY310_FRESH_CPU_CONTROL_PROCESSES',79),
 ('ACTUAL_GUARD','test_directory_preparation_cpu.py','ACTUAL_GUARD_TARGET_FINAL_TWO_FRESH_RESULT.json','PASS_TWO_FRESH_REAL_GUARD_PATH_DIRECTORY_REPAIR_CPU_ONLY',74)]:
 processes=[]
 for i in (1,2):
  output=r/'analysis'/('%s_TARGET_FINAL_%d.json'%(name,i))
  with tempfile.TemporaryDirectory(prefix='resident-final-control-')as tmp:
   args=[sys.executable,'-u','-B',str(r/'scripts'/script),'--output',str(output),'--evidence-root',str(pathlib.Path(tmp)/'fixtures')]
   with (r/'operations'/('%s_TARGET_FINAL_%d.log'%(name,i))).open('xb')as log:
    child=subprocess.Popen(args,env=env,stdout=log,stderr=log);rc=child.wait()
   row=json.loads(output.read_text())if output.exists()else None
   assert rc==0 and row and row['check_count']==expected,(name,i,rc,row)
   processes.append(dict(pid=child.pid,actual_wait_returncode=rc,result=row,negative_fixture_tree_removed=True,outside_science_root=True))
  result.append(output)
 record=dict(status=status,processes=processes,CPU_only=True,CUDA_VISIBLE_DEVICES='',GPU_dispatches=0)
 output=r/'analysis'/aggregate
 with output.open('x')as f:json.dump(record,f,sort_keys=True,indent=2);f.write(chr(10))
 result.append(output)
print(json.dumps(dict(status='PASS_FOUR_FRESH_CPU_PROCESSES_NO_GPU',files={str(p.relative_to(r)):base64.b64encode(p.read_bytes()).decode()for p in result})))
'''
result=subprocess.run(SSH+[shlex.join([PYTHON,'-u','-B','-c',PROGRAM,REMOTE])],capture_output=True,timeout=600)
with (ROOT/'operations/TARGET_CONTROL_AND_DIRECTORY_CPU_TRANSPORT.json').open('x')as f:
    json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),f,indent=2);f.write(chr(10))
assert result.returncode==0,result.stderr.decode()
report=json.loads(result.stdout)
for n,b in report['files'].items():
    with (ROOT/n).open('xb')as f:f.write(base64.b64decode(b,validate=True))
print(report['status'])
