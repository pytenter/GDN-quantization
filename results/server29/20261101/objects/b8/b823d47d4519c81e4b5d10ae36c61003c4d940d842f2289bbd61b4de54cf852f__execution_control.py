"""Task/permit/source/device admission. Import has no torch or GPU API calls."""
import os,sys,json,hashlib,subprocess,time,ast,threading
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'inherited/group32/src'))
from inherited_device_guards import require,sha,read,atomic_new,process_identity,same_process,binding_rows,query_gpu
UUIDS=('GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75')
TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
EXECUTION_REVISION=TASK+'_CUPTI_REPAIR_20261025'
EXECUTION_ROOT='/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'
BYTEVIEW_ORIGIN_REVISION=TASK+'_BYTEVIEW_REPAIR_20261025'
BYTEVIEW_ORIGIN_ROOT='/tmp/'+BYTEVIEW_ORIGIN_REVISION
BYTEVIEW_ORIGIN_SOURCE_SHA256='05901486d72e6f62aba67e61961d6d62043dabef14340b46034a072a004ce5e1'
PARENT_EXECUTION_REVISION=TASK+'_REDUCTION_REPAIR_20261025'
PARENT_EXECUTION_ROOT='/tmp/'+PARENT_EXECUTION_REVISION
PARENT_SOURCE_SHA256='6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'
ORIGIN_SOURCE_ROOT='/tmp/'+TASK
ORIGIN_SOURCE_SHA256='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
CALIBRATION_REUSE_STATUS='PASS_REUSED_FROZEN_NONAIME_CALIBRATION_DUAL_HOST_FRESH_SHA_NO_RECOLLECT_NO_RESELECT'
HUMAN_CUPTI_AUTHORIZATION_SHA256='bd4b7397d044747e07dd4fe8ccbbdd3e6e25c1828eaa7c87db688e1acfa630a0'
AUTHORIZED_GPUS=(1,2,3)
METHODS=('W','R','RW')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def objsha(x):return hashlib.sha256(canonical(x)).hexdigest()
def status(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+str(threading.get_ident()))
 with t.open('xb') as f:f.write(canonical(obj)+b'\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def event(path,kind,**v):
 import fcntl
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(dict(event=kind,monotonic=time.monotonic(),unix=time.time(),**v),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def verify_sources(root):
 rows=read(root/'SOURCE_FREEZE.json')['files']
 for x in rows:
  p=root/x['path'];require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Frozen execution source differs:'+x['path'])
 return sha(root/'SOURCE_FREEZE.json')
def authorize(root,permit):
 root=Path(root).resolve();p=read(permit)
 require(str(root)==EXECUTION_ROOT and p.get('execution_root')==str(root),'Exact new reuse48 archive29 execution root required')
 h=read(root/'operations/HUMAN_TASK_AUTHORIZATION.json');repair=read(root/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json');byteview=read(root/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json');reuse=read(root/'CALIBRATION_REUSE_ADMISSION.json')
 require(p.get('byteview_repair_authorization_sha256')==sha(root/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json') and p.get('reduction_repair_authorization_sha256')==sha(root/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json') and p.get('calibration_reuse_admission_sha256')==sha(root/'CALIBRATION_REUSE_ADMISSION.json'),'Byteview/parent repair authorization or calibration-reuse pin differs')
 require(repair['task_id']==TASK and repair['new_remote_root']==PARENT_EXECUTION_ROOT and repair['execution_revision']==PARENT_EXECUTION_REVISION and repair['authorized_physical_GPUs']==list(AUTHORIZED_GPUS) and repair['old_qualification_credit']==0 and repair['old_formal_credit']==0,'Historical parent repair grant identity differs')
 require(byteview['task_id']==TASK and byteview['new_remote_root']==BYTEVIEW_ORIGIN_ROOT and byteview['execution_revision']==BYTEVIEW_ORIGIN_REVISION and byteview['authorized_physical_GPUs']==list(AUTHORIZED_GPUS) and byteview['old_qualification_credit']==0 and byteview['old_formal_credit']==0,'Historical byteview grant root/device/credit identity differs')
 require(byteview['prior_true_STOP_root']==PARENT_EXECUTION_ROOT and byteview['prior_true_STOP_source_sha256']==PARENT_SOURCE_SHA256 and byteview['status']=='AUTHORIZED_ONE_NEW_INDEPENDENT_BYTEVIEW_REPAIR_BEFORE_FREEZE_NO_DISPATCH' and byteview['qualification_planned']==52 and byteview['formal_planned']==90 and byteview['no_infinite_GPU_retry'] is True and byteview['no_recollection_or_reselection'] is True,'Byteview repair lineage/one-attempt scope differs')
 require(reuse['task_id']==TASK and reuse['status']==CALIBRATION_REUSE_STATUS and reuse['origin_source_root']==ORIGIN_SOURCE_ROOT and reuse['origin_source_sha256']==ORIGIN_SOURCE_SHA256,'Calibration reuse origin/status differs')
 require(reuse['selected_a_b']==[.5,.25] and reuse['new_qualification_credit']==0 and reuse['new_formal_credit']==0 and reuse['no_reselection'] is True,'Calibration reuse selection/credit differs')
 continuation=read(root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json')
 grant_sha=sha(root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json')
 require(grant_sha==HUMAN_CUPTI_AUTHORIZATION_SHA256,'Exact latest human CUPTI repair grant bytes differ')
 require(continuation['execution_revision']==EXECUTION_REVISION and continuation['new_root']==EXECUTION_ROOT and continuation['status']=='EXPLICIT_HUMAN_CUPTI_EXIT_REPAIR_AND_CONTINUE_AUTHORIZATION','New CUPTI repair grant identity differs')
 require(continuation['authorized_GPUs']==list(AUTHORIZED_GPUS) and continuation['only_UUIDs']==[UUIDS[g] for g in AUTHORIZED_GPUS] and continuation['authorized_compute_host']=='55' and continuation['archive_host']=='29_CPU_only' and continuation['PACE_GPU0_unchanged'] is True,'CUPTI repair compute/device scope differs')
 require(continuation['prior_stopped_revision']==TASK+'_REUSE48_ARCHIVE29_20261025' and continuation['prior_stop_review_sha256']=='c7cb69e48ecf9d8fef73fa80dfed0b4dc3c31c2d5034b6d5d5576bf834ccbe82','Exact preceding REUSE48 true STOP differs')
 require(continuation['reuse_qualified48_strict_original_BYTEVIEW'] is True and continuation['new_qualification_required']==4 and continuation['new_formal_after_all52_actual_exit0_release_and_strict_inspector']==90 and continuation['old_minus11_unit_credit']==0,'CUPTI repair qualification/formal counts differ')
 require(continuation['no_changes_to_model_codec_sampler_prompt_seed_caps_calibration_parameters'] is True and continuation['no_automatic_GPU_retry_loop'] is True and continuation['single_new_independent_execution_not_restart_of_old_root'] is True,'Finite CUPTI repair policy differs')
 require(p['reuse48_authorization_sha256']==grant_sha==p['cupti_repair_authorization_sha256'] and p['old_qualification_credit']==48 and p['fresh_qualification_count']==4 and p['old_formal_credit']==0,'CUPTI repair permit grant/credit differs')
 require(p['execution_revision']==EXECUTION_REVISION and p['scope']=='ONE_NEW_INDEPENDENT_CUPTI_REPAIR_REUSE48_PLUS4_THEN90_ARCHIVE29_ONCE','CUPTI repair permit revision/scope differs')
 require(p['qualification_reuse_admission_sha256']==sha(root/'QUALIFICATION_REUSE_ADMISSION.json'),'Qualification reuse admission pin differs')
 require(p['cupti_lifecycle_binding_sha256']==sha(root/'analysis/CUPTI_LIFECYCLE_BINDING.json'),'Extra CUPTI binding pin differs')
 cupti_binding(root)
 require(p['cupti_engineering_freeze_sha256']==sha(root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json') and p['cupti_engineering_result_sha256']==sha(root/'engineering/cupti_lifecycle_preflight/RESULT.json'),'Finite CUPTI engineering proof pin differs')
 admit_cupti_preflight(root)
 previous=root/'provenance/PRIOR_REUSE48_TRUE_STOP'
 require(sha(previous/'SOURCE_FREEZE.json')=='0ef900e88640bd255bdf29f6da8efe5b3d2e05b00c1c32fb74027e27a3c33d50' and sha(previous/'GPU_PERMIT.json')=='bb4174d7706c6d9fbb25963f55d3c9deead006898b587630f6eb93d742537385','Prior true STOP368 source/permit provenance differs')
 require(sha(previous/'operations/FINAL_TRUE_STOP_INDEPENDENT_REVIEW.json')==continuation['prior_stop_review_sha256'],'Prior true STOP368 closed delivery proof differs')
 require(p['task_id']==TASK and h['task_id']==TASK and p['ALLOW_GPU']=='YES','Task-specific grant missing')
 require(p['devices']==[UUIDS[g] for g in AUTHORIZED_GPUS] and p['physical_indices']==list(AUTHORIZED_GPUS),'Only55 GPU1/2/3 are authorized')
 require(p['source_freeze_sha256']==verify_sources(root) and p['plan_sha256']==sha(root/'GPU_EXECUTION_PLAN.json') and p['human_grant_sha256']==sha(root/'operations/HUMAN_TASK_AUTHORIZATION.json'),'Permit identities changed')
 require(all(p[k] is None for k in ('GPUh_hard_cap','total_wall_cap','unit_elapsed_cap')) and p['watchdog_seconds']==1800 and p['cleanup_seconds']==120,'Resource policy differs')
 require(p['max_new_tokens']==81920 and p['formal_new_samples']==90 and p['seed']==1,'Formal protocol differs')
 require(p['calibration_freeze_sha256']==sha(root/'CALIBRATION_DATA_FREEZE.json') and p['omega_manifest_sha256']==sha(root/'TRANSPORTED_QUERY_OMEGA.json') and p['parameter_selection_sha256']==sha(root/'CALIBRATION_PARAMETER_SELECTION.json'),'Frozen shared calibration/selection differs')
 return p

def admit_cupti_preflight(root):
 # The finite engineering2 profile/exit checks are not model/qualification credit.
 freeze=read(root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json')
 result=read(root/'engineering/cupti_lifecycle_preflight/RESULT.json')
 require(freeze['replicates']==['R0','R1'] and freeze['UUID']==UUIDS[1],'Finite engineering two-case scope differs')
 names={row['path'] for row in freeze['files']}
 require({'src/native_profiler_lifecycle.py','src/profiler_detach_lifecycle.py','analysis/CUPTI_LIFECYCLE_BINDING.json','operations/cupti_lifecycle_preflight_once.py'}.issubset(names),'Actual engineering helper/binding pins incomplete')
 for row in freeze['files']:
  relative=Path(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe engineering source pin')
  path=root/relative;require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Main helper differs from actual engineering source:'+row['path'])
 require(result['status']=='PASS_TWO_FRESH_ACTUAL_CUDA_PROFILE_DETACH_OS_EXIT_RELEASE' and result['freeze_sha256']==sha(root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json') and result['qualification_credit']==0 and result['AIME_generations']==0 and result['model_forward_calls']==0 and result['QDQ_calls']==0,'Engineering result/zero-scientific-credit scope differs')
 require(len(result['results'])==2 and [x['replicate'] for x in result['results']]==['R0','R1'],'Finite engineering replica order differs')
 for row in result['results']:
  require(row['actual_returncode']==0 and row['released'] is True and row['UUID']==UUIDS[1] and row['identity'] is not None,'Actual engineering exit0/release/identity missing')
  require(not same_process(row['identity']),'Original engineering child still alive')
 return dict(status='PASS_BOUND_FINITE_ENGINEERING2_WITHOUT_QUALIFICATION_CREDIT',freeze_sha256=sha(root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json'),result_sha256=sha(root/'engineering/cupti_lifecycle_preflight/RESULT.json'),qualification_credit=0,model_forward_calls=0)

def cupti_binding(root):
 # This additional existing ELF/header binding is separate from inherited386.
 # CPU SHA/stat validation only; no dlopen, Torch import, CUDA or CUPTI API.
 manifest=read(root/'analysis/CUPTI_LIFECYCLE_BINDING.json')
 require(manifest['status']=='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS' and manifest['execution_root']==EXECUTION_ROOT,'Exact extra CUPTI binding scope differs')
 require(len(manifest['files'])==1 and Path(manifest['files'][0]['path']).name=='libcupti.so.12','Exactly one canonical existing CUPTI ELF required')
 require(Path(manifest['header']['path']).name=='cupti_activity.h','Exact CUPTI activity header required')
 for row in manifest['files']+[manifest['header']]:
  path=Path(row['path']);before=path.stat();require(path.is_absolute() and path.is_file() and not path.is_symlink() and str(path.resolve())==str(path),'Canonical extra CUPTI ELF/header path differs')
  expected=(row['bytes'],row['inode'],row['device'],row['mtime_ns']);actual=(before.st_size,before.st_ino,before.st_dev,before.st_mtime_ns)
  require(actual==expected and sha(path)==row['sha256'],'Extra existing CUPTI binding SHA/stat differs:'+str(path))
  after=path.stat();require((after.st_size,after.st_ino,after.st_dev,after.st_mtime_ns)==expected,'Extra CUPTI binding changed during SHA:'+str(path))
 return manifest

def model_guard(root):
 for x in read(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  s=Path(x['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns']),'Model/tokenizer stat changed:'+x['path'])
def bindings(root):
 base=root/'inherited/group32/inherited/historical_port_payload/analysis/environment';rows=[]
 for n in ['PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json','CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json']:rows+=binding_rows(read(base/n))
 selected={str(Path(x['path']).resolve()):x for x in rows}
 for row in cupti_binding(root)['files']:
  key=str(Path(row['path']).resolve())
  require(key not in selected or (selected[key]['sha256']==row['sha256'] and selected[key]['bytes']==row['bytes']),'Existing CUPTI binding conflicts with native closure')
  selected[key]=row
 return selected
def verify_external(root):
 rows=bindings(root)
 for p,x in rows.items():require(Path(p).stat().st_size==x['bytes'] and sha(p)==x['sha256'],'External binding changed:'+p)
 return len(rows)
def device(gpu,own_pid=None,minimum_free=23000):
 require(type(gpu)==int and gpu in AUTHORIZED_GPUS,'Invalid selected physical GPU');obs=query_gpu();d=[x for x in obs['devices'] if x['physical_index']==gpu]
 require(len(d)==1 and d[0]['uuid']==UUIDS[gpu] and '3090' in d[0]['name'] and d[0]['memory_total_MiB']>=24000,'Physical device differs')
 require(not [x for x in obs['apps'] if x['uuid']==UUIDS[gpu] and x['pid']!=own_pid],'Foreign process on assigned GPU; never kill it')
 if own_pid is None:require(d[0]['memory_free_MiB']>=minimum_free,'Insufficient exclusive VRAM')
 else:
  own=[x for x in obs['apps'] if x['pid']==own_pid];require(own and all(x['uuid']==UUIDS[gpu] for x in own),'Actual worker owner-PID UUID differs or absent')
 return obs

def driver_identity(root,gpu,driver_factory=None,environment=None):
 # Same qualified driver function, only selected physical CVD/UUID admission changes.
 p=root/'inherited/group32/src/inherited_device_guards.py';tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_driver_ordinal_identity');changed=0
 for n in ast.walk(fn):
  if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='CUDA_VISIBLE_DEVICES':
   require(len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value=='0','Driver admission AST unexpected');n.comparators[0]=ast.Constant(UUIDS[gpu]);changed+=1
 require(changed==1,'Driver physical selector adapter must change exactly1 comparison')
 ns=dict(Path=Path,require=require,sha=sha,os=os,UUID=UUIDS[gpu]);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns)
 return ns[fn.name](list(bindings(root).values()),driver_factory=driver_factory,environment=environment)

def library_check(root,gpu):
 allowed=bindings(root);maps=Path('/proc/self/maps').read_text();selected={};segments=[];missing=[]
 words=('libtorch','libc10','libcudart','libcublas','libcudnn','libnvrtc','libcuda.','libtorchaudio','libcupti')
 for line in maps.splitlines():
  fields=line.split(maxsplit=5)
  if len(fields)!=6 or not fields[5].startswith('/'):continue
  p=Path(fields[5]);key=str(p.resolve());relevant=('.so' in p.name and '/site-packages/torch/' in key) or any(x in p.name for x in words)
  if not relevant:continue
  x=allowed.get(key)
  if x is None:missing.append(key);continue
  a=p.stat();require(int(fields[4])==a.st_ino,'Mapped library inode differs')
  segments.append(dict(path=key,mapped_inode=int(fields[4]),actual_inode=a.st_ino))
  if key not in selected:
   h=sha(p);z=p.stat();selected[key]=dict(path=key,sha256=h,bytes=z.st_size,match=h==x['sha256'] and z.st_size==x['bytes'] and (a.st_ino,a.st_mtime_ns)==(z.st_ino,z.st_mtime_ns))
 v=dict(status='PASS' if selected and not missing and all(x['match'] for x in selected.values()) else 'FAIL',full_maps=maps,segments=segments,selected=list(selected.values()),missing=sorted(set(missing)),identity=device(gpu,os.getpid()))
 return v

def worker_environment(gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(k,None)
 return env

def progress(target,phase,**v):status(target/'PROGRESS.json',dict(phase=phase,monotonic=time.monotonic(),**v))
def storage_guard(root):
 s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=4*1024**3,'Independent output filesystem headroom below4GiB')
 shared=os.statvfs('/data');require(shared.f_bavail*shared.f_frsize>=8*1024**3,'Shared read-only dependency filesystem headroom below8GiB')
 # Scientific output bound, excluding the immutable source and CPU preparation.
 used=sum(p.stat().st_blocks*512 for folder in ['execution','formal'] for p in (root/folder).rglob('*') if p.is_file())
 require(used<=2*1024**3,'New execution small-output2GiB bound reached; stop before exhausting disk')
 return dict(free_bytes=s.f_bavail*s.f_frsize,shared_data_free_bytes=shared.f_bavail*shared.f_frsize,allocated_output_bytes=used)
def reference_root():return Path('/dev/shm/qwen_sym8_g32_wscale_rowbal_reuse48_reference_20261025')

def admit_inherited(root):
 # Historical source/results remain outside this task's execution and are never
 # relabelled fresh. An independent fresh reader must populate this receipt.
 p=root/'HISTORICAL_G32_BASELINE_REUSE.json';r=read(p)
 require(r['status']=='PASS_READONLY_HISTORICAL_IDENTITY_AND_BASELINES','Historical independent audit is not PASS')
 require(r['task_id']==TASK and r['historical_root_readonly'] is True,'Historical provenance scope differs')
 require(r['formal_generations_reused_as_new']==0 and r['historical_scores']=={'H+G32+Sym8':21,'H+G32+Asym8':25},'Historical read-only controls differ')
 require(r['runtime_math_qualified'] is True and r['historical_formal_complete']==120,'Original qualified full runtime missing')
 return dict(status='PASS_READONLY_ORIGINAL_G32_RUNTIME_NOT_NEW_QUALIFICATION',receipt_sha256=sha(p),new_formal_credit=0)

def validate_jobs(root,jobs):
 from control_plan import validate_formal_jobs
 return validate_formal_jobs(read(root/'AIME26_DATASET_FREEZE.json'),read(root/'QUESTION_GPU_ASSIGNMENT.json'),jobs,UUIDS,AUTHORIZED_GPUS)
