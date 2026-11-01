"""Two independent CPU processes per affected production/control path, no GPU."""
import json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cases=[('SCOPE','test_formal_profiler_scope_cpu.py',False),('DETACH','test_profiler_detach_lifecycle_cpu.py',False),('REUSE','test_qualification_reuse_cpu.py',True),('CONTROL','test_resident_control_cpu.py',False),('FINISHER','test_resident_finisher_cpu.py',True),('REUSED_FINISHER','test_reused_finisher_cpu.py',True)]
runs=[]
for i in range(2):
 for name,script,rootarg in cases:
  out=R/('analysis/FORMAL_SCOPE_CPU_V2_'+name+'_R'+str(i)+'.json');assert not out.exists()
  argv=[sys.executable,'-B',str(R/'scripts'/script)]
  if rootarg:argv+=['--root',str(R)]
  argv+=['--output',str(out)]
  p=subprocess.run(argv,capture_output=True,text=True)
  (R/('operations/FORMAL_SCOPE_CPU_V2_'+name+'_R'+str(i)+'.log')).write_text(p.stdout+p.stderr)
  row=dict(case=name,repeat=i,actual_wait_returncode=p.returncode,argv=argv,stdout=p.stdout,stderr=p.stderr)
  if out.exists():row['result']=json.loads(out.read_text())
  runs.append(row)
  print(json.dumps(dict(case=name,repeat=i,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr[-5000:])),flush=True)
  if p.returncode:
   with (R/'analysis/FORMAL_SCOPE_CPU_V2_FAILURE.json').open('x')as f:json.dump(row,f,indent=2)
   raise SystemExit(1)
with (R/'analysis/FORMAL_SCOPE_TWO_FRESH_TARGET_CPU_SUITE.json').open('x')as f:
 json.dump(dict(status='PASS_TWO_FRESH_TARGET_CPU_PROCESSES_EACH_CHANGED_PATH',runs=runs),f,sort_keys=True,indent=2)
