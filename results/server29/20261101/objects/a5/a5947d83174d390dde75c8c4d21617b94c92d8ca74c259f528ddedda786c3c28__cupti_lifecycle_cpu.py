"""Strict lifecycle/order/entrypoint CPU fixtures; never native/GPU evidence."""
import argparse,copy,hashlib,importlib.util,json,os,sys,tempfile,types
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import native_profiler_lifecycle as N
import profiler_detach_lifecycle as D
import execution_control as C

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(b,m):
 if not b:raise AssertionError(m)
class Call:
 def __init__(self,label,events,code=0):self.label=label;self.events=events;self.code=code
 def __call__(self,*a):self.events.append((self.label,list(a)));return self.code
class Handle:
 def __init__(self,events,flush=0,finalize=0):self.cuptiActivityFlushAll=Call('native_flush',events,flush);self.cuptiFinalize=Call('native_finalize',events,finalize)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence-root',type=Path);a=ap.parse_args();base=a.evidence_root or Path(tempfile.mkdtemp(prefix='wscale_cupti_lifecycle_cpu_'));base.mkdir(parents=True,exist_ok=True);checks=[]
 original=dict(C_process=C.process_identity,C_same=C.same_process,N_CDLL=N.ctypes.CDLL,N_Path=N.Path,D_root=D.EXECUTION_ROOT,argv=sys.argv.copy(),CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=sys.modules.get('torch'))
 fake_maps=[''];events=[];loads=[];active=[False];config={'flush':0,'finalize':0};factory=lambda p,mode:(loads.append((p,mode)) or Handle(events,config['flush'],config['finalize']))
 class PathProxy:
  def read_text(self):return fake_maps[0]
 def mapped_path(p):return PathProxy() if str(p)=='/proc/self/maps' else Path(p)
 N.Path=mapped_path;N.ctypes.CDLL=factory
 fakeT=types.SimpleNamespace(autograd=types.SimpleNamespace(_profiler_enabled=lambda:active[0]),cuda=types.SimpleNamespace(synchronize=lambda:events.append(('sync',[]))))
 sys.modules['torch']=fakeT
 plan=read(R/'GPU_EXECUTION_PLAN.json');jobs=read(R/'FORMAL_JOB_MANIFEST.json');unit=next(u for u in plan['qualification_units'] if u['id']=='GPU1_base_MATCHED4096')
 counter=[0]
 def fixture():
  counter[0]+=1;f=base/('case_%03d'%counter[0]);f.mkdir();(f/'analysis').mkdir();(f/'scripts').mkdir();(f/'scripts/gpu_unit_worker.py').write_text('# CPU fixture only\n');dump(f/'GPU_EXECUTION_PLAN.json',plan);dump(f/'FORMAL_JOB_MANIFEST.json',jobs);dump(f/'SOURCE_FREEZE.json',dict(CPU_ONLY=True));dump(f/'GPU_PERMIT.json',dict(CPU_ONLY=True));elf=f/'libcupti.so.12';header=f/'cupti_activity.h';elf.write_bytes(b'CPU_MOCK_ELF_NOT_A_REAL_LIBRARY');header.write_bytes(b'CPU_MOCK_HEADER_NO_CUDA_CALLS')
  def row(p):s=p.stat();return dict(path=str(p.resolve()),bytes=s.st_size,sha256=sha(p),inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)
  dump(f/'analysis/CUPTI_LIFECYCLE_BINDING.json',dict(status='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS',execution_root=str(f.resolve()),files=[row(elf)],header=row(header),CPU_MOCK_ONLY=True,not_valid_scientific_binding=True))
  t=f/'execution'/unit['id'];t.mkdir(parents=True);identity=dict(pid=os.getpid(),starttick=1001,cmdline=['CPU_FIXTURE_WORKER_NO_GPU']);parent=dict(pid=os.getppid(),starttick=1000,cmdline=['CPU_FIXTURE_PARENT_NO_GPU']);dump(t/'STARTED.json',dict(identity=identity));dump(t/'DISPATCH.json',dict(unit=unit['id'],formal=False,gpu=1,gpu_uuid=C.UUIDS[1],permit_sha256=sha(f/'GPU_PERMIT.json'),controller_identity=parent))
  C.process_identity=lambda pid:identity if pid==os.getpid() else parent;C.same_process=lambda e:e==parent;D.EXECUTION_ROOT=str(f.resolve());os.environ['CUDA_VISIBLE_DEVICES']=C.UUIDS[1];sys.argv=[str(f/'scripts/gpu_unit_worker.py'),'--root',str(f),'--unit',unit['id']];N._HELD_LIBRARIES.clear();fake_maps[0]='';events.clear();loads.clear();active[0]=False;config.update(flush=0,finalize=0)
  return f,t,identity
 profile=dict(status='PASS_ACTUAL16TOKEN_CUDA_PROFILE',profiled_tokens=16,actual_CUDA_kernel_calls=2,quantizer_regions=1,quantizer_CUDA_milliseconds=1.,CPU_MOCK_ONLY=True,GPU_calls=0)
 try:
  f,t,i=fixture();p=D.preload_qualification_profile(f,1,os.getpid(),unit);need(len(loads)==1 and p['already_held'] is False and p['CUDA_Driver_API_calls']==p['CUPTI_API_calls']==0,'Preloaddidnativecalls');out=D.detach_qualification_profile(f,1,os.getpid(),unit,profile);j=[json.loads(l) for l in (t/'CUPTI_PROFILE_DETACH_EVENTS.jsonl').read_text().splitlines()];expected=['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN'];need([x['phase'] for x in j]==expected and events==[('sync',[]),('native_flush',[0]),('native_finalize',[])],'Documentedorderingfailed');need(out['flush_returncode']==out['finalize_returncode']==0 and out['identity']==i and out['profile_summary_sha256']==C.objsha(profile) and len(loads)==1,'Detachnotexactsamepreheldhandle');checks.append(dict(case='real_helper_preload_collection_samehandle_sync_flush_finalize_order',status='PASS_CPU_MOCK_ONLY'))
  # A second invocation fails before another native finalize/flush.
  before=len(events)
  try:D.detach_qualification_profile(f,1,os.getpid(),unit,profile)
  except (RuntimeError,FileExistsError):pass
  else:raise AssertionError('Repeated detach allowed')
  need(len(events)==before,'Repeateddetachcallednative');checks.append(dict(case='duplicate_completed_detach_refused_before_nativecalls',status='PASS_REJECTED'))
  def rejects(name,action):
   try:action()
   except (RuntimeError,FileExistsError,FileNotFoundError,KeyError):checks.append(dict(case=name,status='PASS_REJECTED'))
   else:raise AssertionError('Negativeaccepted '+name)
  for label,field,value in [('bad_manifest_status','status','UNVERIFIED'),('foreign_binding_root','execution_root','/tmp/foreign')]:
   f,t,i=fixture();b=read(f/'analysis/CUPTI_LIFECYCLE_BINDING.json');b[field]=value;dump(f/'analysis/CUPTI_LIFECYCLE_BINDING.json',b);rejects(label,lambda:N.preload_cupti_before_driver(f));need(not loads,'Badmanifestloadedlibrary')
  for key,val in [('sha256','0'*64),('inode',0),('bytes',0),('mtime_ns',0)]:
   f,t,i=fixture();b=read(f/'analysis/CUPTI_LIFECYCLE_BINDING.json');b['files'][0][key]=val;dump(f/'analysis/CUPTI_LIFECYCLE_BINDING.json',b);rejects('ELF_'+key,lambda:N.preload_cupti_before_driver(f));need(not loads,'BadELFloaded')
  f,t,i=fixture();b=read(f/'analysis/CUPTI_LIFECYCLE_BINDING.json');b['header']['sha256']='0'*64;dump(f/'analysis/CUPTI_LIFECYCLE_BINDING.json',b);rejects('header_changed',lambda:N.preload_cupti_before_driver(f))
  for lib in ('libcuda.so.535.183.01','libcudart.so.12','libcupti.so.12'):
   f,t,i=fixture();fake_maps[0]='0-1 r-xp 0 00:00 1 /CPU_MOCK/'+lib;rejects('late_preload_after_'+lib,lambda:N.preload_cupti_before_driver(f));need(not loads,'Latepreloadloadedlibrary')
  f,t,i=fixture();rejects('detach_without_preheld_handle',lambda:D.detach_completed_profiler(f,fakeT,lambda *a,**k:None));need(not events,'Missinghandlecalledflush')
  for code in (1,7):
   f,t,i=fixture();config['flush']=code;D.preload_qualification_profile(f,1,os.getpid(),unit);rejects('flush_nonzero_'+str(code),lambda:D.detach_qualification_profile(f,1,os.getpid(),unit,profile));need(not (t/'CUPTI_PROFILE_DETACH.json').exists() and not any(x[0]=='native_finalize' for x in events),'Failedflushfinalizedorpassed')
  f,t,i=fixture();config['finalize']=3;D.preload_qualification_profile(f,1,os.getpid(),unit);rejects('finalize_nonzero',lambda:D.detach_qualification_profile(f,1,os.getpid(),unit,profile));need(not (t/'CUPTI_PROFILE_DETACH.json').exists(),'Failedfinalizepassed')
  f,t,i=fixture();D.preload_qualification_profile(f,1,os.getpid(),unit);active[0]=True;rejects('active_profiler_refused',lambda:D.detach_qualification_profile(f,1,os.getpid(),unit,profile));need(not events,'Activeprofilerflushed')
  for key,value in [('status','FAIL'),('profiled_tokens',15),('actual_CUDA_kernel_calls',0),('quantizer_regions',0),('quantizer_CUDA_milliseconds',0.),('quantizer_CUDA_milliseconds',float('inf'))]:
   f,t,i=fixture();D.preload_qualification_profile(f,1,os.getpid(),unit);bad=dict(profile);bad[key]=value;rejects('profile_incomplete_'+key+'_'+str(value),lambda:D.detach_qualification_profile(f,1,os.getpid(),unit,bad));need(not events,'Incompleteprofilecallednative')
  for option in ('--root','--unit'):
   f,t,i=fixture();sys.argv += [option,str(f) if option=='--root' else unit['id']];rejects('duplicate_'+option,lambda:D.preload_qualification_profile(f,1,os.getpid(),unit))
  f,t,i=fixture();sys.argv[0]=str(f/'scripts/foreign.py');rejects('foreign_entrypoint',lambda:D.preload_qualification_profile(f,1,os.getpid(),unit))
  f,t,i=fixture();os.environ['CUDA_VISIBLE_DEVICES']=C.UUIDS[2];rejects('wrong_CVD_UUID',lambda:D.preload_qualification_profile(f,1,os.getpid(),unit))
  f,t,i=fixture();rejects('foreign_actual_PID',lambda:D.preload_qualification_profile(f,1,os.getpid()+1,unit))
  f,t,i=fixture();old=next(u for u in plan['qualification_units'] if u['kind']=='codec_fields');sys.argv[-1]=old['id'];rejects('old48_profile_dispatch_not_allowed',lambda:D.preload_qualification_profile(f,old['gpu'],os.getpid(),old))
  f,t,i=fixture();disp=read(t/'DISPATCH.json');disp['controller_identity']['pid']+=1;dump(t/'DISPATCH.json',disp);rejects('foreign_controller_parent',lambda:D.preload_qualification_profile(f,1,os.getpid(),unit))
  f,t,i=fixture();disp=read(t/'STARTED.json');disp['identity']['starttick']+=1;dump(t/'STARTED.json',disp);rejects('changed_actual_starttick',lambda:D.preload_qualification_profile(f,1,os.getpid(),unit))
  # Every formal job has no collection, detach, library load or new receipts.
  f,t,i=fixture()
  for job in jobs['jobs']:
   os.environ['CUDA_VISIBLE_DEVICES']=job['gpu_uuid'];sys.argv=[str(f/'scripts/gpu_unit_worker.py'),'--root',str(f),'--unit',job['id'],'--formal'];need(D.preload_qualification_profile(f,job['gpu'],os.getpid(),job,True) is None and D.detach_qualification_profile(f,job['gpu'],os.getpid(),job,None,True) is None,'Formalnotexcluded')
  need(not loads and not events and not list(f.rglob('CUPTI_PRELOAD_ADMISSION.json')) and not list(f.rglob('PROFILE_COLLECTION_COMPLETED.json')) and not list(f.rglob('CUPTI_PROFILE_DETACH.json')),'Formalexclusiontouchedprofiling');checks.append(dict(case='all90_exact_formal_jobs_excluded_no_ELF_API_or_receipts',status='PASS',count=90))
  # Core cleanup algorithm is copied without an AST change from historical fix.
  import ast
  old=R.parent/'QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1_FORMAL_SCOPE_REPAIR_20261023/src/profiler_detach_lifecycle.py'
  if old.is_file():
   a1=next(n for n in ast.parse(old.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_detach_exact_pinned_handle');a2=next(n for n in ast.parse((R/'src/profiler_detach_lifecycle.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==a1.name);need(ast.dump(a1,include_attributes=False)==ast.dump(a2,include_attributes=False),'OriginalcoreASTchanged');checks.append(dict(case='documented_core_cleanup_AST_identical_historical',status='PASS'))
  report=dict(status='PASS_CUPTI_LIFECYCLE_ORDER_ENTRYPOINT_FORMAL_EXCLUSION_CPU',checks=checks,count=len(checks),GPU_calls=0,model_loads=0,AIME_samples=0,actual_native_library_loads=0,explicit_mocks=['CUPTI ELF/header bytes','ctypes.CDLL+flush/finalize','torch.cuda.synchronize/profiler state','exact controller/worker identity endpoint','/proc/self/maps','test-only executionroot mapping'],actual_components=['pinnedSHA/stat/canonicalfile admission','sameheldhandle lookup','documented cleanupcore order','real receipt/journal files','strict entrypoint/argv/manifest/UUID selectors','all90formal exclusion'],source_sha256={n:sha(R/n) for n in ['src/native_profiler_lifecycle.py','src/profiler_detach_lifecycle.py','scripts/gpu_unit_worker.py','src/runtime_engine.py']},evidence_root=str(base),not_scientific_GPU_qualification=True)
  dump(base/'CUPTI_LIFECYCLE_CPU_REPORT.json',report);print(json.dumps(dict(status=report['status'],count=len(checks),report=str(base/'CUPTI_LIFECYCLE_CPU_REPORT.json'),GPU_calls=0)),flush=True)
 finally:
  C.process_identity=original['C_process'];C.same_process=original['C_same'];N.ctypes.CDLL=original['N_CDLL'];N.Path=original['N_Path'];D.EXECUTION_ROOT=original['D_root'];sys.argv=original['argv'];N._HELD_LIBRARIES.clear()
  if original['CVD'] is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=original['CVD']
  if original['torch'] is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=original['torch']
if __name__=='__main__':main()
