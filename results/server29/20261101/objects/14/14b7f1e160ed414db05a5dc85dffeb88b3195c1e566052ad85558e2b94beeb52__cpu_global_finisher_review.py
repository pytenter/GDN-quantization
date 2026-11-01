"""Execute actual global finisher partial/full paths on isolated CPU fixtures. No scientific GPU credit."""
import copy,ctypes,json,os,runpy,shutil,subprocess,sys,tempfile
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
checks=[];originals={n:getattr(C,n)for n in ['NODE','worker_environment','model_guard','query_gpu','same_process']};original_run=subprocess.run
C.NODE=next(n for n in C.read(R/'CROSS_HOST_PLAN.json')['nodes']if n['id']=='server29_gpu6')
C.worker_environment=lambda g:dict(os.environ,CUDA_VISIBLE_DEVICES='');C.model_guard=lambda root:None;C.query_gpu=lambda:dict(apps=[],CPU_NVML_MOCK=True);C.same_process=lambda i:False

def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
with tempfile.TemporaryDirectory(prefix='g64_GLOBAL_FINISHER_CPU_MOCK_NOT_SCIENCE_')as tmp:
 T=Path(tmp)
 for full in [False,True]:
  F=T/('full'if full else'partial');F.mkdir();archive=T/(F.name+'_archive')
  for name in ['CROSS_HOST_PLAN.json','GLOBAL_FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','AIME26_DATASET_FREEZE.json','GROUP32_HISTORY_REUSE_AND_IDENTITY.json']:
   put(F/name,C.read(R/name))
  (F/'history/group32').mkdir(parents=True);shutil.copyfile(R/'history/group32/AIME26_FROZEN_V4_SCORES.json',F/'history/group32/AIME26_FROZEN_V4_SCORES.json')
  for name in ['SOURCE_FREEZE.json','GPU_PERMIT.json']:put(F/name,dict(files=[],CPU_MOCK_FIXTURE_ONLY=True))
  (F/'peer55').mkdir();put(F/'peer55/SOURCE_FREEZE.json',dict(files=[],CPU_MOCK_FIXTURE_ONLY=True))
  put(F/'ROOT.json',dict(archive_root=str(archive)));put(F/'GLOBAL_COLLECTION_RECEIPT.json',dict(status='PASS_BOTH_NODE_TERMINALS_AND_PEER55_SNAPSHOT_SHA_BYTES',CPU_FIXTURE_ONLY=True))
  put(F/'GLOBAL_SESSION_RESULT.json',dict(status='GPU_COMPLETE_CROSS_HOST_GROUP64_120_FORMAL'if full else'FAIL_CLOSED_CROSS_HOST_STOP',CPU_FIXTURE_ONLY=True));put(F/'GLOBAL_FIRST_FAILURE.json',dict(error='CPU_INJECTED_FAILURE_NO_GPU'))
  jobs=C.read(F/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs'];old=C.read(F/'history/group32/AIME26_FROZEN_V4_SCORES.json')['scores'];oldby={(x['problem_id'],x['condition']):x for x in old};fake_scores=[]
  for j in jobs:
   source=F if j['node_id']=='server29_gpu6'else F/'peer55';d=source/'formal'/j['id']
   if full:
    x=copy.deepcopy(oldby[j['problem_id'],'Q'+j['condition'][-1]]);x.update(sample_id=j['id'],condition=j['condition'],gpu=j['gpu'],gpu_uuid=j['gpu_uuid'],node_id=j['node_id'],compute_host=j['compute_host'],wall_seconds=2.,decode_and_sampling_seconds=1.,prompt_seconds=1.,model_load_seconds=1.,input_tokens=j['input_tokens'],peak_allocated_bytes=1,peak_reserved_bytes=1,generated_token_count=2,finish_reason='eos',CPU_MOCK_FIXTURE_ONLY=True);fake_scores.append(x);put(d/'RESULT.json',dict(CPU_MOCK_FIXTURE_ONLY=True))
  calls=[]
  def fake_run(cmd,**kwargs):
   assert len(cmd)==6 and Path(cmd[3]).name=='score_global_pass_cpu.py'and kwargs['env']['CUDA_VISIBLE_DEVICES']==''
   out=Path(cmd[-1]);assert out.parent==F;put(out,dict(status='CPU_MOCK_FULL_SCORE_FIXTURE_NOT_ACTUAL_AIME',scores=fake_scores));calls.append(cmd)
   return subprocess.CompletedProcess(cmd,0,stdout=b'CPU MOCK SCORER\n',stderr=b'')
  subprocess.run=fake_run;sys.argv=[str(R/'scripts/finalize_global_cpu.py'),str(F)]
  runpy.run_path(str(R/'scripts/finalize_global_cpu.py'),run_name='__main__')
  assert C.read(F/'FINAL_SEAL_RECEIPT.json')['status']=='PASS_FINAL_SEAL'and archive.is_dir()
  for x in C.read(F/'SHA256_MANIFEST.json')['files']:
   assert C.sha(F/x['path'])==C.sha(archive/x['path'])==x['sha256']and(archive/x['path']).stat().st_size==x['bytes']
  assert (F/'FINAL_SEAL_RECEIPT.json').read_bytes()==(archive/'FINAL_SEAL_RECEIPT.json').read_bytes()
  checks.append(('Full'if full else'Zero-dispatch partial')+' production finisher actual archive SHA/bytes, real renameat2 NOREPLACE')
  if full:
   assert len(calls)==2 and len(C.read(F/'FROZEN_V4_SCORES.json')['scores'])==120
   assert C.read(F/'FINAL_TASK_STATUS.json')['status']=='FORMAL_COMPLETE_120_120_AND_STOP'
   assert C.read(F/'GROUP32_GROUP64_PAIRWISE.json')['historical_totals']=={'Q0':20,'Q1':25,'Q2':21,'Q3':25}
   assert all(x['net_correct']==0 for k,x in C.read(F/'GROUP32_GROUP64_PAIRWISE.json')['pairwise'].items()if '-Q'in k)
   checks.append('Full mocked two-fresh score agreement, original old-score byte pin, paired statistics and2x2x2 report')
  else:
   assert not calls and not(F/'FROZEN_V4_SCORES.json').exists()and not(F/'AIME26_PER_PROBLEM_RESULTS.csv').exists()
   assert sum(C.read(F/'PARTIAL_COUNTS.json')['completed_legal_counts'].values())==0
   checks.append('Partial0 no score/empty scientific CSV, no fake interval, exactGPUh unresolved')
  before=C.sha(F/'FINAL_SEAL_RECEIPT.json')
  try:runpy.run_path(str(R/'scripts/finalize_global_cpu.py'),run_name='__main__')
  except RuntimeError:pass
  else:raise AssertionError('Finisher replay admitted')
  assert C.sha(F/'FINAL_SEAL_RECEIPT.json')==before;checks.append('Actual finisher replay rejected before mutation')
 # Actual filesystem no-clobber primitive leaves both existing directories intact.
 a=T/'rename_source';b=T/'rename_existing';a.mkdir();b.mkdir();(a/'a').write_text('a');(b/'b').write_text('b')
 lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 assert fn(-100,os.fsencode(a),-100,os.fsencode(b),1)==-1 and ctypes.get_errno()==17 and(a/'a').read_text()=='a'and(b/'b').read_text()=='b';checks.append('Actual renameat2 rejects existing archive without replacing')
subprocess.run=original_run
for n,v in originals.items():setattr(C,n,v)
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_ACTUAL_GLOBAL_FINISHER_ISOLATED_CPU_FLOW',count=len(checks),checks=checks,GPU_dispatches=0,qualification_credit=0,actual_AIME_generation=0,scoring='EXPLICIT_CPU_FIXTURE_MOCK; no experiment score',archive='REAL_SHA_BYTES_AND_RENAMEAT2_ON_DISPOSABLE_CPU_FIXTURE')))
