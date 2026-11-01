"""One admitted unit/sample, one fresh process. No retry paths."""
import argparse,os,sys,time,json,traceback,gc
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--unit',required=True);ap.add_argument('--formal',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'))
import execution_control as C
permit=R/'GPU_PERMIT.json';C.authorize(R,permit)
plan=C.read(R/('FORMAL_JOB_MANIFEST.json' if a.formal else 'GPU_EXECUTION_PLAN.json'));unit=next(x for x in plan['jobs' if a.formal else 'qualification_units'] if x['id']==a.unit);gpu=unit['gpu'];target=R/('formal' if a.formal else 'execution')/a.unit
d=C.read(target/'DISPATCH.json');C.require(d['unit']==a.unit and d['permit_sha256']==C.sha(permit) and os.getppid()==d['controller_identity']['pid'] and C.same_process(d['controller_identity']),'Actual worker parent/dispatch binding differs')
C.require(os.environ.get('CUDA_VISIBLE_DEVICES')==C.UUIDS[gpu],'Selected GPU visibility differs')
if a.formal:
 auth=C.read(R/'AUTHORIZATION_FORMAL.json');C.require(auth['source_freeze_sha256']==C.sha(R/'SOURCE_FREEZE.json') and auth['GPU_qualification_sha256']==C.sha(R/'GPU_CODEC_QUALIFICATION.json'),'Formal admission missing/changed')
C.atomic_new(target/'STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()));C.progress(target,'VERIFY_MODEL_AND_DRIVER')
engine=None;reference=None
try:
 C.model_guard(R);C.storage_guard(R);C.atomic_new(target/'DRIVER_IDENTITY.json',C.driver_identity(R,gpu))
 from runtime_engine import Engine,ByteReference
 engine=Engine(R,gpu,target);v=C.library_check(R,gpu);C.atomic_new(target/'PREFLIGHT_LIBRARY.json',v);C.require(v['status']=='PASS','Actual pre-forward library closure failed')
 if a.formal or unit['kind']=='nonAIME_free_generation_fresh':
  if a.formal:
   row=next(x for x in C.read(R/'AIME26_DATASET_FREEZE.json')['records'] if x['problem_id']==unit['problem_id']);_,ids=engine.f.render_input(engine.tokenizer,row['problem']);ids=ids[0].tolist();C.require(ids==row['input_ids'] and engine.f.ids_sha256(ids)==unit['input_ids_sha256'],'Actual prompt re-render changed');cap=81920
  else:
   row=C.read(R/'protocol/FREE_CANARY_INPUT.json');_,ids=engine.f.render_input(engine.tokenizer,row['problem']);ids=ids[0].tolist();C.require(ids==row['input_ids'],'Actual nonAIME prompt changed');cap=64;reference=ByteReference(R,target,unit)
  result=engine.generate(unit['condition'],ids,cap,unit,reference)
  if not a.formal and unit['repeat']==1:
   old=C.read(R/'execution'/unit['reference_unit']/'RESULT.json')['result']
   for key in ['generated_token_ids','input_ids_sha256','initial_generator_sha256','final_generator_sha256','finish_reason']:
    C.require(old[key]==result[key],'Fresh original sampler/RNG sequence differs:'+key)
 elif unit['kind']=='real_model_memory_pressure':result=engine.pressure()
 else:
  ids=C.read(R/'inherited/group32/inherited/historical_port_payload/candidate/benchmark_replay/fixtures/doc1_tokens.json');ids=ids if isinstance(ids,list) else ids.get('input_ids',ids.get('tokens'));C.require(len(ids)==32768,'NonAIME frozen replay identity changed')
  if unit['kind']!='nonAIME_persistent_long':reference=ByteReference(R,target,unit)
  result=engine.teacher(unit,ids,reference)
 C.model_guard(R);v=C.library_check(R,gpu);C.atomic_new(target/'POSTFLIGHT_LIBRARY.json',v);C.require(v['status']=='PASS','Actual postflight library closure failed')
 result.update(peak_allocated_bytes=engine.torch.cuda.max_memory_allocated(),peak_reserved_bytes=engine.torch.cuda.max_memory_reserved())
 record=dict(status='PASS',unit=a.unit,result=result,identity=C.process_identity(os.getpid()),gpu=gpu,gpu_uuid=C.UUIDS[gpu],permit_sha256=C.sha(permit),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),environment=engine.environment)
 record['payload_sha256']=C.objsha(record);C.atomic_new(target/'RESULT.json',record)
 if reference is not None:reference.cleanup_success()
 C.progress(target,'UNIT_PASS')
except BaseException as exc:
 if engine is not None:
  tokens=getattr(engine,'active_tokens',[])
  C.atomic_new(target/'PARTIAL_TOKENS.json',dict(generated_token_ids=tokens,count=len(tokens),technical_failure=True,score_eligible=False))
  cache=getattr(engine,'last_cache',None)
  if cache is not None:
   try:engine.diagnostic('CURRENT_STATE_FIRST_FAILURE.json',{str(i):engine.get_state(cache,i) for i in engine.layers})
   except BaseException as de:C.atomic_new(target/'DIAGNOSTIC_FAILURE.json',dict(error=repr(de)))
 failure=dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit=a.unit,identity=C.process_identity(os.getpid()))
 if hasattr(exc,'driver_identity_receipt'):failure['driver_identity_receipt']=exc.driver_identity_receipt
 C.atomic_new(target/'FIRST_FAILURE.json',failure);raise
