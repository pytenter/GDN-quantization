"""Execute production event/queue/failure functions on CPU-only fake workloads."""
from pathlib import Path
import ast,json,os,time,tempfile,threading,types,concurrent.futures,traceback,sys
ROOT=Path(sys.argv[1]);passed=[]
def functions(path,names,ns):
 tree=ast.parse(path.read_text());body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert {x.name for x in body}==set(names)
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
def require(ok,message):
 if not ok:raise RuntimeError(message)
with tempfile.TemporaryDirectory(prefix='oscar_repair_control_') as tmp:
 R=Path(tmp);(R/'branches').mkdir();ref=R/'reference';ref.mkdir()
 event_ns=dict(Path=Path,json=json,os=os,time=time,threading=threading)
 functions(ROOT/'src/execution_control.py',['event'],event_ns)
 event=event_ns['event'];event(R/'event_test.jsonl','FAILURE',monotonic=1.25,unit='CPU_TEST')
 row=json.loads((R/'event_test.jsonl').read_text());assert row['monotonic']==1.25 and row['event']=='FAILURE' and 'unix' in row
 passed.append('actual_event_accepts_caller_monotonic_without_duplicate_key_exception')
 def atomic_new(p,x):
  with Path(p).open('x') as f:json.dump(x,f)
 C=types.SimpleNamespace(require=require,event=event,atomic_new=atomic_new,reference_root=lambda:ref,SELECTED_GPUS=(1,2,3))
 free=[125*1024**3]
 ns=dict(R=R,C=C,os=types.SimpleNamespace(statvfs=lambda p:types.SimpleNamespace(f_bavail=free[0],f_frsize=1)),time=time,threading=threading,concurrent=concurrent,traceback=traceback,
         LOCK=threading.RLock(),GLOBAL_STOP=threading.Event(),REFERENCE_PAIR_LOCK=threading.Lock(),CONDITIONS=('P1','C1','S3'),alive_branch=dict(P1=True,C1=True,S3=True),eligible=dict(P1=True,C1=True,S3=True),failure_history=[])
 functions(ROOT/'scripts/gpu_controller.py',['branches_for','record_failure','scope_alive','reference_pair_admission','queues'],ns)
 first=dict(id='GPU1_P1_512_R0',gpu=1,condition='P1',repeat=0,reference_unit='GPU1_P1_512_R0')
 free[0]=99*1024**3
 try:ns['reference_pair_admission'](first)
 except RuntimeError:passed.append('global_reference_reservation_rejects99GiB_before_worker_creation')
 else:raise AssertionError('Missing100GiB admission')
 free[0]=100*1024**3;ns['reference_pair_admission'](first);passed.append('global_reference_reservation_accepts100GiB_boundary')
 free[0]=125*1024**3
 units=[]
 for g in (1,2,3):
  name='GPU%d_P1_512_R0'%g
  units += [dict(id=name,gpu=g,condition='P1',repeat=0,reference_unit=name),dict(id='GPU%d_P1_512_R1'%g,gpu=g,condition='P1',repeat=1,reference_unit=name),dict(id='GPU%d_P1_LONG4096'%g,gpu=g,condition='P1',kind='nonAIME_persistent_long')]
 active=[None];max_active=[0];completed=[];testlock=threading.Lock()
 def run(unit,formal):
  assert not formal
  with testlock:
   if unit.get('repeat')==0:
    assert active[0] is None,'Multiple global references admitted';active[0]=unit['id'];(ref/unit['id']).mkdir();max_active[0]=max(max_active[0],1)
   elif unit.get('repeat')==1:assert active[0]==unit['reference_unit']
  time.sleep(.015)
  with testlock:
   if unit.get('repeat')==1:(ref/unit['reference_unit']).rmdir();active[0]=None
   completed.append(unit['id'])
  return True
 ns['run']=run;ns['queues'](units)
 assert max_active[0]==1 and active[0] is None and len(completed)==9 and len(set(completed))==9 and not list(ref.iterdir())
 for g in (1,2,3):assert completed.index('GPU%d_P1_512_R0'%g)<completed.index('GPU%d_P1_512_R1'%g)
 passed += ['three_actual_queue_threads_never_hold_multiple_reference_pairs','R0_R1_atomic_pair_order_and_reference_cleanup_before_next_admission','no_duplicate_unit_dispatch_with_paired_R1_skip']
 def reset():
  for p in (R/'branches').glob('*'):p.unlink()
  p=R/'FIRST_FAILURE.json'
  if p.exists():p.unlink()
  ns['alive_branch']=dict(P1=True,C1=True,S3=True);ns['eligible']=dict(P1=True,C1=True,S3=True);ns['failure_history']=[];ns['GLOBAL_STOP']=threading.Event()
 reset()
 free[0]=99*1024**3
 blocked=[dict(id='GPU1_C1_512_R0',gpu=1,condition='C1',repeat=0,reference_unit='GPU1_C1_512_R0'),dict(id='GPU1_C1_512_R1',gpu=1,condition='C1',repeat=1,reference_unit='GPU1_C1_512_R0')]
 calls=[]
 ns['run']=lambda unit,formal:calls.append(unit['id'])
 ns['queues'](blocked)
 assert not calls and not ns['GLOBAL_STOP'].is_set() and ns['alive_branch']==dict(P1=True,C1=False,S3=True) and ns['eligible']['P1']
 passed.append('reference_admission_failure_closes_own_branch_without_cancelling_other_eligible_branches')
 free[0]=125*1024**3
 reset()
 try:raise RuntimeError('CPU_TEST_U_FP_FAILURE')
 except RuntimeError as exc:ns['record_failure'](exc,dict(id='GPU1_FP_EQ',kind='fp_rotation_equivalence'))
 assert ns['alive_branch']==dict(P1=True,C1=False,S3=False) and ns['eligible']['P1']
 passed.append('actual_U_FP_failure_recording_preserves_P1_eligibility')
 reset()
 try:raise RuntimeError('CPU_TEST_P1_FAILURE')
 except RuntimeError as exc:ns['record_failure'](exc,dict(id='GPU1_P1_512_R0',condition='P1'))
 assert ns['alive_branch']==dict(P1=False,C1=True,S3=True)
 passed.append('actual_P1_failure_recording_does_not_close_U_branches')
 reset()
 def raises(unit,formal):raise RuntimeError('CPU_TEST_QUEUE_EXCEPTION')
 ns['run']=raises;ns['queues']([dict(id='CPU_SHARED_EXCEPTION',gpu=1,kind='codec_math')])
 assert ns['GLOBAL_STOP'].is_set() and not any(ns['alive_branch'].values()) and ns['failure_history'] and (R/'FIRST_FAILURE.json').exists()
 passed.append('actual_queue_exception_records_global_failure_without_error_handler_crash')
 rows=[json.loads(x) for x in (R/'GPU_RESOURCE_LEDGER.jsonl').read_text().splitlines()]
 assert sum(x['event']=='BRANCH_FAIL_CLOSED' for x in rows)==4
 passed.append('four_failure_paths_durably_append_real_production_event_records')
 result=dict(status='PASS_REPAIRED_PRODUCTION_CONTROL_CPU_TESTS',checks=passed,count=len(passed),GPU_forwards=0,model_loads=0)
 print(json.dumps(result,sort_keys=True))
