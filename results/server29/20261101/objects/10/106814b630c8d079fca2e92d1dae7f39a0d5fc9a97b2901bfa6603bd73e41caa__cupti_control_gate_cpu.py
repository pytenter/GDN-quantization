"""Exact added current4 lifecycle inspector: CPU receipt fixtures, no GPU proof."""
import copy,hashlib,json,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C;from control_qualification import inspect_current_profile_lifecycle

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True)+'\n')
def fixture(r):
 unit=dict(id='GPU1_base_MATCHED4096',gpu=1,gpu_uuid=C.UUIDS[1],kind='matched_performance');t=r/'execution'/unit['id'];t.mkdir(parents=True)
 for n in ('SOURCE_FREEZE.json','GPU_PERMIT.json','analysis/CUPTI_LIFECYCLE_BINDING.json'):write(r/n,dict(CPU_MOCK_ONLY=True,path=n))
 elf=dict(path='/CPU_MOCK_ONLY/libcupti.so.12',sha256='a'*64,bytes=123,inode=12345);identity=dict(pid=987654321,starttick=123,cmdline=['CPU_MOCK_ONLY']);profile=dict(status='PASS_ACTUAL16TOKEN_CUDA_PROFILE',profiled_tokens=16,actual_CUDA_kernel_calls=16,quantizer_regions=16,quantizer_CUDA_milliseconds=1.)
 common=dict(unit=unit['id'],identity=identity,gpu_uuid=C.UUIDS[1],source_freeze_sha256=C.sha(r/'SOURCE_FREEZE.json'),permit_sha256=C.sha(r/'GPU_PERMIT.json'),manifest_sha256=C.sha(r/'analysis/CUPTI_LIFECYCLE_BINDING.json'));summary=C.objsha(profile)
 preload=dict(status='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL',already_held=False,CUDA_Driver_API_calls=0,CUPTI_API_calls=0,model_forward_calls=0,QDQ_calls=0,normal_process_teardown_required=True,path=elf['path'],sha256=elf['sha256'],**common)
 collection=dict(status='PASS_ACTUAL_PROFILE_COLLECTION_COMPLETED_BEFORE_DETACH',profile_summary=profile,profile_summary_sha256=summary,**common)
 detach=dict(status='DOCUMENTED_CUPTI_FLUSH_AND_DETACH_RETURNED_SUCCESS',flush_returncode=0,finalize_returncode=0,CUPTI_API_calls=2,profiler_collection_already_completed=True,CUDA_or_CUPTI_APIs_not_disabled=True,signal_or_exit_suppression=False,actual_process_exit_still_required=True,path=elf['path'],sha256=elf['sha256'],profile_summary_sha256=summary,**common)
 write(t/'CUPTI_PRELOAD_ADMISSION.json',preload);write(t/'PROFILE_COLLECTION_COMPLETED.json',collection);write(t/'CUPTI_PROFILE_DETACH.json',detach);write(t/'RESULT.json',dict(identity=identity,result=dict(profiler=profile,profiler_native_lifecycle=detach)))
 phases=['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN'];events=[dict(phase=p,monotonic=i,pid=identity['pid']) for i,p in enumerate(phases)];events[2]['flags']=0;events[3]['returncode']=events[5]['returncode']=0;(t/'CUPTI_PROFILE_DETACH_EVENTS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
 for n in ('PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json'):write(t/n,dict(selected=[dict(path=elf['path'],sha256=elf['sha256'],bytes=elf['bytes'],match=True)],segments=[dict(path=elf['path'],mapped_inode=elf['inode'],actual_inode=elf['inode'])]))
 return unit,t,elf

def main():
 rows=[];original=C.cupti_binding
 def check(name,action=None):
  with tempfile.TemporaryDirectory(prefix='wscale-cupti-gate-cpu-') as temp:
   r=Path(temp);u,t,elf=fixture(r);C.cupti_binding=lambda root:dict(files=[elf])
   if action:action(r,u,t)
   try:result=inspect_current_profile_lifecycle(r,u)
   except (RuntimeError,KeyError,FileNotFoundError) as exc:
    assert action is not None,(name,str(exc));rows.append(dict(case=name,status='PASS_REJECTED',error=str(exc)))
   else:assert action is None,(name,result);rows.append(dict(case=name,status='PASS'))
 def mutate(t,name,key,value):
  p=t/name;x=json.loads(p.read_text());x[key]=value;write(p,x)
 try:
  check('positive exact new current profile lifecycle')
  check('missing preload refuses',lambda r,u,t:(t/'CUPTI_PRELOAD_ADMISSION.json').unlink())
  check('duplicate preload refuses',lambda r,u,t:mutate(t,'CUPTI_PRELOAD_ADMISSION.json','already_held',True))
  check('preload GPU API refuses',lambda r,u,t:mutate(t,'CUPTI_PRELOAD_ADMISSION.json','CUDA_Driver_API_calls',1))
  check('wrong manifest pin refuses',lambda r,u,t:mutate(t,'CUPTI_PRELOAD_ADMISSION.json','manifest_sha256','0'*64))
  check('wrong exact UUID refuses',lambda r,u,t:mutate(t,'CUPTI_PROFILE_DETACH.json','gpu_uuid',C.UUIDS[0]))
  check('failed finalize refuses',lambda r,u,t:mutate(t,'CUPTI_PROFILE_DETACH.json','finalize_returncode',1))
  check('exit suppression refuses',lambda r,u,t:mutate(t,'CUPTI_PROFILE_DETACH.json','signal_or_exit_suppression',True))
  check('changed profile summary refuses',lambda r,u,t:mutate(t,'PROFILE_COLLECTION_COMPLETED.json','profile_summary_sha256','0'*64))
  check('changed returned lifecycle refuses',lambda r,u,t:mutate(t,'RESULT.json','result',dict(profiler={},profiler_native_lifecycle={})))
  check('missing event refuses',lambda r,u,t:(t/'CUPTI_PROFILE_DETACH_EVENTS.jsonl').write_text(''))
  def bad_event(r,u,t):
   p=t/'CUPTI_PROFILE_DETACH_EVENTS.jsonl';x=[json.loads(l) for l in p.read_text().splitlines()];x[3]['returncode']=1;p.write_text(''.join(json.dumps(v)+'\n' for v in x))
  check('actual flush journal failure refuses',bad_event)
  def bad_native(r,u,t):
   p=t/'POSTFLIGHT_LIBRARY.json';x=json.loads(p.read_text());x['segments'][0]['actual_inode']+=1;write(p,x)
  check('mapped CUPTI inode mismatch refuses',bad_native)
  check('no current native ELF proof refuses',lambda r,u,t:mutate(t,'POSTFLIGHT_LIBRARY.json','selected',[]))
  check('old field unit not selected for lifecycle',lambda r,u,t:u.update(kind='codec_fields'))
 finally:C.cupti_binding=original
 print(json.dumps(dict(status='PASS_CURRENT4_CUPTI_CONTROL_LIFECYCLE_GATE_CPU',checks=len(rows),cases=rows,CPU_MOCK_ONLY=True,GPU_calls=0,AIME_samples=0,source_sha256=C.sha(R/'src/control_qualification.py')),sort_keys=True))
if __name__=='__main__':main()
