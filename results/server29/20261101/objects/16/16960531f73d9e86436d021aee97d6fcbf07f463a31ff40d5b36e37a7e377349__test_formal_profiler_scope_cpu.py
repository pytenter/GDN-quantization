"""Actual production hook and actual native-guard integration, CPU-only observers.

No CUDA APIs, library loading, model or forward. Only hardware observers inside
the real guard are stubbed; the production lifecycle selector is not replaced.
"""
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
import profiler_detach_lifecycle as L
import qualification_guards as G
import resident_execution_control as C

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch'not in sys.modules
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 def reject(n,fn):
  try:fn()
  except (RuntimeError,ValueError,FileNotFoundError,KeyError):checks.append(n)
  else:raise AssertionError(n)
 jobs=json.loads((ROOT/'FORMAL_JOB_MANIFEST.json').read_text())['jobs']
 def args(root,j,script=None):return [str(root/'scripts'/(script or 'resident_formal_worker.py')),'--root',str(root),'--unit',j['id'],'--gpu',str(j['gpu'])]
 # All60 actual frozen IDs/UUIDs travel through the exact production C->G->L
 # call chain. Native identity/NVML observers are explicitly mocked, CPU only.
 with patch.object(G,'assess_native_maps',lambda *x:dict(status='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED',mapped_native=[],errors=[])),patch.object(G,'verify_model_and_external',lambda *x,**k:[]),patch.object(G,'_own_gpu',lambda *x,**k:dict(CPU_MOCK_OBSERVER=True)),patch.object(G,'persist_native_audit',lambda *x:dict(path='CPU_MOCK_NO_FILE')):
  for j in jobs:
   with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':j['gpu_uuid']}),patch.object(sys,'argv',args(ROOT,j)):
    result=C.check_after_model(ROOT,j['gpu'],os.getpid())
    check('actual_formal_check_after_model_no_detach_'+j['id'],result['profiler_native_lifecycle']is None)
 j=jobs[0]
 with tempfile.TemporaryDirectory(prefix='formal-scope-regression-')as tmp:
  r=Path(tmp).resolve();(r/'scripts').mkdir();(r/'FORMAL_JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=jobs)))
  units=[dict(id='KNOWN_PROFILE',kind='profile_short',gpu=j['gpu'],gpu_uuid=j['gpu_uuid']),dict(id='KNOWN_32',kind='qdq32',gpu=j['gpu'],gpu_uuid=j['gpu_uuid'])]
  (r/'QUALIFICATION_JOB_MANIFEST.json').write_text(json.dumps(dict(units=units)))
  with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':j['gpu_uuid']}):
   valid=args(r,j)
   check('formal_return_before_any_trace_or_qualification_directory',L._profile_target(r,j['gpu'],valid)is None)
   with patch.object(sys,'argv',valid):check('actual_wrapper_formal_never_imports_torch',L.detach_qualification_profile_if_completed(r,j['gpu'],os.getpid())is None)
   for n,arr in [
    ('duplicate_unit',valid+['--unit',j['id']]),('missing_unit_value',[valid[0],'--unit']),
    ('duplicate_root',valid+['--root',str(r)]),('missing_gpu_value',valid[:-1]),
    ('wrong_root',[str(r/'scripts/resident_formal_worker.py'),'--root','/wrong','--unit',j['id'],'--gpu',str(j['gpu'])]),
    ('wrong_gpu_argument',valid[:-1]+['99']),
    ('unknown_formal_id',[valid[0],'--root',str(r),'--unit','UNKNOWN','--gpu',str(j['gpu'])]),
    ('foreign_script_cannot_get_formal_exclusion',args(r,j,'foreign_worker.py')),
   ]:reject(n,lambda arr=arr:L._profile_target(r,j['gpu'],arr))
   with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-WRONG'}):reject('wrong_formal_UUID',lambda:L._profile_target(r,j['gpu'],valid))
   (r/'FORMAL_JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=jobs+[j])))
   reject('duplicate_formal_manifest_ID',lambda:L._profile_target(r,j['gpu'],valid))
   for u in units:
    target=r/'qualification'/u['id'];target.mkdir(parents=True)
    arr=[str(r/'scripts/qualify_gpu.py'),'--unit',u['id']]
    check('qualification_before_trace_unchanged_'+u['id'],L._profile_target(r,j['gpu'],arr)is None)
    if u['kind'].startswith('profile_'):
     p=target/'CUDA_PROFILE.json';p.write_text('{"traceEvents":[]}')
     check('completed_qualification_profile_still_selected',L._profile_target(r,j['gpu'],arr)==(target,p,u))
     p.unlink();p.symlink_to(r/'missing')
     reject('dangling_profile_trace_still_rejected',lambda:L._profile_target(r,j['gpu'],arr))
   reject('unknown_qualification_unit_not_ignored',lambda:L._profile_target(r,j['gpu'],['x','--unit','UNKNOWN']))
 # Actual old failure is reproducible CPU-only; current fixes exactly that path.
 scope=json.loads((ROOT/'FORMAL_SCOPE_REPAIR_SOURCE_SCOPE.json').read_text())
 old=Path(scope['remote_prior_root']if str(ROOT).startswith('/data/')else scope['prior_root'])
 sp=importlib.util.spec_from_file_location('old_frozen_lifecycle',old/'src/profiler_detach_lifecycle.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':j['gpu_uuid']}):reject('old_frozen_formal_failure_exactly_reproduced',lambda:m._profile_target(old,j['gpu'],args(old,j)))
 before=ast.parse((old/'src/profiler_detach_lifecycle.py').read_text());after=ast.parse((ROOT/'src/profiler_detach_lifecycle.py').read_text())
 for name in ['_detach_exact_pinned_handle','detach_completed_profiler','detach_qualification_profile_if_completed']:
  x=next(x for x in before.body if isinstance(x,ast.FunctionDef)and x.name==name);y=next(x for x in after.body if isinstance(x,ast.FunctionDef)and x.name==name)
  check('unchanged_original_native_cleanup_'+name,ast.dump(x)==ast.dump(y))
 check('qualification_body_suffix_identical',json.loads((ROOT/'FORMAL_SCOPE_REPAIR_CONTRACT.json').read_text())['qualification_suffix_byte_identical'])
 check('no_torch_import_or_CUDA_initialization','torch'not in sys.modules)
 pins={n:sha(ROOT/n)for n in ['src/profiler_detach_lifecycle.py','src/qualification_guards.py','src/resident_execution_control.py','scripts/resident_formal_worker.py','FORMAL_JOB_MANIFEST.json']}
 result=dict(status='PASS_FORMAL_SCOPE_ACTUAL_GUARD60_AND_STRICT_NEGATIVES_CPU_ONLY',check_count=len(checks),checks=checks,source_pins=pins,GPU_dispatches=0,model_forwards=0,Torch_imported=False,native_observers_explicitly_CPU_mocked=True)
 with a.output.open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],check_count=len(checks))))
if __name__=='__main__':main()
