"""Two fresh target CPU rounds; actual Popen exits, no CUDA/model credit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (ROOT/'SESSION_STARTED.json').exists()
OUT=ROOT/'engineering/LIFECYCLE_FINAL_CPU_ADMITTED_TWO_FRESH'
OUT.mkdir(exist_ok=False)
commands=[
 ('control_4gate_35formal',ROOT/'tests/test_four_gpu_control_cpu.py',[]),
 ('finisher_and_owning_bash_wait',ROOT/'tests/test_four_gpu_finish_cpu.py',[]),
 ('owning_lifecycle',ROOT/'tests/test_owning_lifecycle_cpu.py',[]),
 ('blind_monitor',ROOT/'operations/pace_four_gpu_monitor.py',['--self-test']),
]
rows=[]
for fresh in (1,2):
 for name,script,args in commands:
  argv=[sys.executable,'-B',str(script)]+args
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
  log=OUT/('fresh%d_%s.log'%(fresh,name))
  with log.open('xb',buffering=0) as f:
   p=subprocess.Popen(argv,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   rc=p.wait(timeout=300)
  row=dict(fresh=fresh,suite=name,argv=argv,pid=p.pid,actual_returncode=rc,actual_reaped=True,
           proc_directory_gone=not Path('/proc',str(p.pid)).exists(),log=log.relative_to(ROOT).as_posix(),
           log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),GPU_calls=0,model_loads=0)
  (OUT/('fresh%d_%s.ACTUAL_EXIT.json'%(fresh,name))).write_text(json.dumps(row,sort_keys=True)+'\n')
  rows.append(row);print(json.dumps(row),flush=True)
  assert rc==0 and row['proc_directory_gone'],row
result=dict(status='PASS_TWO_FRESH_TARGET_CPU_CONTROL_4GATE_35FORMAL_EXIT_GUARD_FINISHER_MONITOR',
            children=rows,whole_controller_fixture_per_fresh=dict(actual_gate_CPU_Popen=4,actual_formal_CPU_Popen=35,synthetic_reused=25),
            GPU_calls=0,model_loads=0,CUDA_qualification_credit=0,unix=time.time())
(OUT/'RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],actual_fresh_interpreters=len(rows))),flush=True)
