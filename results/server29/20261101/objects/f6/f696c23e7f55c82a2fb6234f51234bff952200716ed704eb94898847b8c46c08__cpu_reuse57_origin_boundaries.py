"""Read-only exact GPU5 third-origin negative cases; no GPU or scientific credit."""
import copy,json,os,sys,tempfile
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import execution_control as C
import continuation_admission as A
checks=[];p=A.binding(R);assert len(p['qualification'])==57
reuse=A.admit_local_reuse(R);assert len(reuse['qualification_ids'])==len(C.read(R/'GPU_EXECUTION_PLAN.json')['qualification_units'])
checks.append('ACTUAL_ALL_LOCAL_ORIGINAL57_BINDINGS_SOURCE_EXIT_IDENTITY_REVIEW')
u=next(k for k,v in p['qualification'].items()if v['origin_node']=='server29_gpu5')
with tempfile.TemporaryDirectory(prefix='CPU_REUSE57_THIRD_ORIGIN_NEGATIVE_')as tmp:
 F=Path(tmp)
 # Full global roster allows this pure observer to test GPU5 on either CPU host;
 # remote file access remains forbidden on55, so file-level cases run only29.
 plan=C.read(R/'GPU_EXECUTION_PLAN.json');(F/'GPU_EXECUTION_PLAN.json').write_text(json.dumps(plan))
 cases=['wrong_resume_auth','wrong_gpu5_root','wrong_gpu5_uuid','wrong_gpu5_source','wrong_gpu5_permit','move_gpu5_to_gpu6','drop_gpu5_unit','add_unknown_unit']
 if C.NODE['host']=='202.38.247.29':cases+=['wrong_gpu5_result_sha','wrong_gpu5_interval_sha','wrong_gpu5_trace_sha']
 for kind in cases:
  x=copy.deepcopy(p)
  if kind=='wrong_resume_auth':x['resume_authorization_sha256']='0'*64
  if kind=='wrong_gpu5_root':x['origins']['server29_gpu5']['root']+='_foreign'
  if kind=='wrong_gpu5_uuid':x['origins']['server29_gpu5']['node']['uuid']='GPU-foreign'
  if kind=='wrong_gpu5_source':x['origins']['server29_gpu5']['source_sha256']='0'*64
  if kind=='wrong_gpu5_permit':x['origins']['server29_gpu5']['permit_sha256']='0'*64
  if kind=='move_gpu5_to_gpu6':x['qualification'][u]['origin_node']='server29_gpu6'
  if kind=='drop_gpu5_unit':x['qualification'].pop(u)
  if kind=='add_unknown_unit':x['qualification']['unknown_duplicate_or_foreign']=copy.deepcopy(x['qualification'][u])
  if kind=='wrong_gpu5_result_sha':x['qualification'][u]['result_sha256']='0'*64
  if kind=='wrong_gpu5_interval_sha':x['qualification'][u]['interval_sha256']='0'*64
  if kind=='wrong_gpu5_trace_sha':
   target=next(k for k,v in x['qualification'].items()if v['origin_node']=='server29_gpu5'and v.get('trace_sha256')is not None)
   x['qualification'][target]['trace_sha256']='0'*64
  else:target=u
  (F/'REUSE_ADMISSION_STATIC.json').write_text(json.dumps(x))
  try:
   if kind.startswith('wrong_gpu5_')and kind.endswith('_sha'):A.qualification_source(F,target)
   else:A.binding(F)
  except(RuntimeError,KeyError):checks.append('REJECT_'+kind)
  else:raise AssertionError('Invalid third-origin proof accepted:'+kind)
 (F/'REUSE_ADMISSION_STATIC.json').write_text(json.dumps(p))
 try:A.qualification_source(F,'unknown_qualification')
 except RuntimeError:checks.append('REJECT_UNKNOWN_QUALIFICATION_ID')
 else:raise AssertionError('Unknown ID accepted')
 if C.NODE['host']=='202.38.247.55':
  try:A.origin_root(F,'server29_gpu5')
  except RuntimeError:checks.append('REJECT_UNAVAILABLE_REVERSE_HOST_ORIGIN')
  else:raise AssertionError('Reverse-host read was allowed')
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_REUSE57_STRICT_THIRD_ORIGIN_ACTUAL_FILES_AND_NEGATIVES_CPU',count=len(checks),checks=checks,GPU_calls=0,model_forwards=0,qualification_credit=0)))
