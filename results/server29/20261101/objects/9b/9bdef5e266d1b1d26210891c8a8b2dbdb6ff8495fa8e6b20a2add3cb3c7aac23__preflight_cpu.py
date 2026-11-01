"""No CUDA or model initialization. Validate retained science, remaining roster and controls."""
import os,sys,json,copy,ast
from pathlib import Path
R=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]);sys.path.insert(0,str(R/'src'));import execution_control as C
import native_map_guards as G
checks=[]
def check(name,fn):fn();checks.append(name)
def reject(fn):
 try:fn()
 except (RuntimeError,KeyError):return
 raise RuntimeError('Expected negative CPU fixture to fail')
jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']
check('exact15remaining_seed1_cap_prompt_GPU0',lambda:C.validate_jobs(R,jobs))
for label,mutate in [('duplicate',lambda j:j.__setitem__(1,copy.deepcopy(j[0]))),('seed',lambda j:j[0].__setitem__('seed',2)),('GPU1',lambda j:j[0].__setitem__('gpu',1)),('prompt',lambda j:j[0].__setitem__('input_ids_sha256','0'*64)),('partial_resume',lambda j:j[0].__setitem__('generation_starts_at_token0',False))]:
 def negative(mutate=mutate):
  bad=copy.deepcopy(jobs);mutate(bad);reject(lambda:C.validate_jobs(R,bad))
 check('reject_'+label,negative)
rows=C.read(R/'ORIGINAL_SCIENTIFIC_SOURCE_COPY_MAP.json')['files']
for x in rows:
 if x['path']=='src/execution_control.py':continue
 p=R/x['path'];C.require(p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256'],'P1 science copied bytes changed:'+x['path'])
checks.append('all_copied_original_science_identical_except_control_adapter')
native=C.library_check.__globals__
original_assess=G.assess_native_maps;original_persist=G.persist_native_audit;original_device=C.device
try:
 G.persist_native_audit=lambda *a:dict(path='CPU_MOCK_NOT_WRITTEN');C.device=lambda *a:dict(CPU_mock=True)
 for status,fields,expected in [('PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED',dict(mapped_native=[{}],errors=[]),'PASS'),('FAIL_NATIVE_MAPPING_IDENTITY',dict(mapped_native=[{}],errors=[{}]),'FAIL')]:
  G.assess_native_maps=lambda *a,status=status,fields=fields:dict(status=status,**fields)
  C.require(C.library_check(R,0)['status']==expected,'Native result adapter altered gate semantics')
 checks.append('strict_native_status_adapter_positive_negative')
finally:G.assess_native_maps=original_assess;G.persist_native_audit=original_persist;C.device=original_device
old=Path(C.read(R/'ROOT.json')['original_root']);f=C.read(old/'SOURCE_FREEZE.json')
C.require(C.sha(old/'SOURCE_FREEZE.json')==C.read(R/'ORIGINAL_SCIENTIFIC_SOURCE_COPY_MAP.json')['original_source_freeze_sha256'],'Original cancellation source identity changed')
for x in f['files']:
 p=old/x['path'];C.require(p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256'],'Original P1 source changed:'+x['path'])
checks.append('original622_fresh_SHA')
prior=C.read(R/'provenance/P1_LEGAL_PAYLOAD_MANIFEST.json')
for x in prior['records']:
 for row in x['files']:
  p=old/row['path'];C.require(p.stat().st_size==row['bytes'] and C.sha(p)==row['sha256'],'Original15 result/release SHA changed')
checks.append('original15_payload_exit_release_provenance_fresh_SHA')
plan=C.read(old/'GPU_EXECUTION_PLAN.json')['qualification_units'];units=[u for u in plan if u.get('kind')=='codec_math' or u.get('condition')=='P1'];C.require(len(units)==27,'Prior P1 eligibility roster differs');evidence=[]
for unit in units:
 t=old/'execution'/unit['id'];result=C.read(t/'RESULT.json');interval=C.read(t/'RESOURCE_INTERVAL.json');C.require(result['status']=='PASS' and interval['exit_returncode']==0 and interval['release']=='PASS' and not C.same_process(interval['identity']),'Prior P1 qualification invalid/not released')
 evidence.append(dict(unit=unit['id'],result_sha256=C.sha(t/'RESULT.json'),interval_sha256=C.sha(t/'RESOURCE_INTERVAL.json')))
checks.append('prior27_P1_qualification_exit_release_readonly')
G.verify_model_and_external(R,fresh_external=True);checks.append('fresh61_plus293_and17stat')
obs=C.device(0);C.storage_guard(R);checks.append('actual_GPU0_idle_3090_UUID_and_storage')
for p in R.rglob('*'):
 C.require(not p.is_symlink(),'Archive catalog must contain no symlinks')
 if p.is_file() and p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
checks.append('real_root_catalog_no_symlinks_all_Python_compile')
C.require('torch' not in sys.modules and not os.environ.get('CUDA_VISIBLE_DEVICES'),'CPU preflight imported Torch/CUDA');checks.append('no_Torch_CUDA_model_or_forward')
C.atomic_new(out,dict(status='PASS_CPU_PREFLIGHT_NOT_GPU_PROOF',checks=checks,count=len(checks),original_P1_qualification=evidence,original_completed=15,new_remaining=15,original_source_fresh_count=len(f['files']),external_fresh_SHA=61,native_fresh_SHA=293,model_stat=17,observation=obs,identity=C.process_identity(os.getpid()),monotonic=__import__('time').monotonic()))
print(json.dumps(dict(status='PASS',checks=len(checks))))
