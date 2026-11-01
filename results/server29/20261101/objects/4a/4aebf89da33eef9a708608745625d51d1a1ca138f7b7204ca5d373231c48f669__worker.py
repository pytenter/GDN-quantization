#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,gc,json,os,subprocess,time,traceback
from pathlib import Path
import runtime as R

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--mode',choices=['canary','formal'],required=True);ap.add_argument('--repeat',type=int,default=0);a=ap.parse_args();r=a.root;manifest=R.read(r/'DATASET_AND_JOB_MANIFEST.json');uuid=manifest['gpu_uuid_map'][str(a.gpu)]
 tag=f'canary/gpu{a.gpu}_r{a.repeat}' if a.mode=='canary' else f'formal/gpu{a.gpu}'
 attempt=int(os.environ.get('WORKER_ATTEMPT','0'));wd=r/'execution'/tag/f'attempt_{attempt:02d}';wd.mkdir(parents=True,exist_ok=True)
 R.atomic_new(wd/'STARTED.json',dict(task_id=R.TASK,pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,mode=a.mode,repeat=a.repeat,unix=time.time()))
 current_job=None
 try:
  R.check(os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu),'GPU visibility mismatch');R.verify(r)
  R.check(R.read(r/'CPU_PREFLIGHT.json')['status']=='PASS','CPU GateA not PASS')
  if a.mode=='formal':
   auth=R.read(r/'AUTHORIZATION_FORMAL.json');R.check(auth['status']=='PASS' and auth['source_manifest_sha256']==R.sha(r/'SOURCE_SHA256_MANIFEST.json'),'Formal source authorization failed');R.check(auth['canary_sha256']==R.sha(r/'CANARY_QUALIFICATION.json') and R.read(r/'CANARY_QUALIFICATION.json')['status']=='PASS','Real canary not PASS')
  c=R.Components(r,a.gpu);torch=c.torch;model=c.model
  # K.load's immutable source returns model/qmod, so instantiate tokenizer CPU-only from the same identity-bound directory.
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
  actual=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  R.check(any(int(x.split(',')[0])==os.getpid() and x.split(',')[1].strip()==uuid for x in actual.splitlines() if x.strip()),'Actual PID/physical UUID mismatch')
  c.environment.update(gpu_uuid=uuid,source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json'),protocol_sha256=R.sha(r/'FROZEN_PROTOCOL.json'))
  R.atomic_new(wd/'ENVIRONMENT.json',c.environment)
  if a.mode=='canary':
   ids=R.read(r/'protocol/non_aime_fixture.json')['tokens'];R.check(len(ids)==64,'Canary fixed input changed');past=None;trace=[];cons=c.Consumption(c,model,c.qmod,True);qdq=R.HOnlyQDQ(c,model,'H_DUALM',None,True);torch.cuda.reset_peak_memory_stats();beg=time.monotonic()
   with c.f.QwenKeyHadamardPatch(model) as hp,torch.inference_mode():
    for step in range(49):
     cons.step=step;cons.details={};inp=ids[:16] if step==0 else [ids[15+step]]
     output=model(input_ids=torch.tensor([inp],device='cuda:0',dtype=torch.long),past_key_values=past,use_cache=True);past=output.past_key_values
     R.check(torch.isfinite(output.logits).all(),'Nonfinite canary logits');qdq.step_qdq(past,'prefill' if step==0 else 'decode',step);cons.after(past)
     lens=[int(past.layers[i].keys.shape[-2]) for i in range(32) if model.config.get_text_config().layer_types[i]=='full_attention'];R.check(lens==[16+step]*8,'Canary KV cursor mismatch')
     trace.append(dict(step=step,logits=c.tsha(output.logits),helper=dict(cons.details),QDQ=qdq.trace_rows[-1],KV_lengths=lens));del output
     if (step+1)%16==0:print(json.dumps(dict(event='CANARY_PROGRESS',gpu=a.gpu,repeat=a.repeat,forwards=step+1)),flush=True)
    R.check(qdq.calls==49 and qdq.writebacks==24*49 and cons.checks==24*48 and hp.calls==24*49,'Canary QDQ/consumption/H counts')
   cons.close();receipt=dict(task_id=R.TASK,status='PASS',mode='canary',gpu=a.gpu,gpu_uuid=uuid,repeat=a.repeat,input_tokens=64,prefill_tokens=16,decode_forwards=48,model_forward_calls=49,qdq_calls=49,writebacks=24*49,next_consumption_checks=24*48,trace=trace,trace_sha256=R.canonical_sha(trace),environment=c.environment,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),prefill_after_QDQ_memory=c.prefill_after_QDQ_memory,wall_seconds=time.monotonic()-beg,numeric_scope='Same strict32KReference dual() codes/row/column/dequant parity atprefill, allstepsQDQ hashes; sameGPUfresh exact; not81K numerical proof')
   if a.repeat==1:
    prior=R.read(r/'execution/canary'/f'gpu{a.gpu}_r0/attempt_00/RESULT.json');R.check(prior['status']=='PASS' and prior['gpu_uuid']==uuid and prior['trace']==trace,'Fresh-process sameGPU byte trace mismatch')
   R.atomic_new(wd/'RESULT.json',receipt);del past,trace,cons,qdq
  else:
   jobs=[j for j in manifest['jobs'] if j['gpu']==a.gpu];R.check(len(jobs)==15,'Fixed shard must contain15 jobs');completed=[]
   for job in jobs:
    current_job=job;result_path=r/'raw_results'/(job['canonical_job_id']+'.json')
    if result_path.exists():R.validate_record(R.read(result_path),job,r);completed.append(job['canonical_job_id']);continue
    if (r/'STOP_REQUESTED.json').exists():raise RuntimeError('Explicit stop request')
    jd=r/'attempts'/job['canonical_job_id']/f'attempt_{attempt:02d}';R.atomic_new(jd/'STARTED.json',dict(task_id=R.TASK,job_id=job['canonical_job_id'],pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,seed=job['seed'],unix=time.time()))
    R.atomic_status(r/'status'/f'worker{a.gpu}.json',dict(status='RUNNING',pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,job=job['canonical_job_id'],canonical_index=job['canonical_index'],attempt=attempt,start_unix=time.time(),completed=completed))
    memory=[];counter=[0]
    def metadata_hook(m,args,out):
     counter[0]+=1
     if counter[0]==1 or counter[0]%128==0:
      cache=out.past_key_values;lens=[int(cache.layers[i].keys.shape[-2]) for i in range(32) if model.config.get_text_config().layer_types[i]=='full_attention']
      mm=dict(event='MEMORY_PROGRESS',canonical_index=job['canonical_index'],forwards=counter[0],cache_length=int(cache.get_seq_length()),KV_lengths=lens,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),unix=time.time());memory.append(mm);print(json.dumps(mm),flush=True)
    handle=model.register_forward_hook(metadata_hook)
    try:row=c.generate(c,model,tok,c.qmod,job,'H_DUALM',None,a.gpu,c.environment,81920,trace=False)
    finally:handle.remove()
    row.update(task_id=R.TASK,method=R.METHOD,canonical_job_id=job['canonical_job_id'],model_identity_sha256=R.sha(r/'MODEL_AND_CODEC_IDENTITY.json'),runner_identity=R.sha(Path(__file__)),source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json'),protocol_sha256=R.sha(r/'FROZEN_PROTOCOL.json'),gpu_uuid=uuid,input_token_count=row.get('input_tokens'),input_token_ids_sha256=row.get('input_ids_hash'),generated_token_ids_sha256=R.canonical_sha(row['generated_token_ids']),raw_model_response=row['response'],eos_seen=row.get('finish_reason')=='eos',truncated=row.get('finish_reason')=='length',latency_seconds=row['latency_s'],tokens_per_second=row['tokens_per_s'],memory_observations=memory,prefill_after_QDQ_memory=getattr(c,'prefill_after_QDQ_memory',None),actual_model_forward_calls=counter[0],nonfinite_status=row['nonfinite'],scorer_version='AIME26_STRICT_V4_CANDIDATE',scorer_sha256=R.sha(r/'protocol/aime26_scorer_v4.py'),scoring_status='DEFERRED_UNTIL_ALL60_COMPLETE',top_k=20,effective_generation_config=row.get('generation_config'))
    if row.get('quantizer_audit'):row['qdq_writeback_audit']=row['quantizer_audit']
    row['output_payload_sha256']=R.canonical_sha(row);R.atomic_new(jd/'RESULT.json',row)
    if row['status']!='COMPLETE':
     R.atomic_status(r/'status'/f'worker{a.gpu}.json',dict(status='FAILED',pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,job=job['canonical_job_id'],generated_token_count=row['generated_token_count'],failure_attempt=str(jd),runtime_error=row['runtime_error'],completed=completed));raise RuntimeError('Formal technical failure; retained job attempt: '+str(jd))
    R.validate_record(row,job,r);R.check(c.f.top_k_audit_valid(row),'Original topK audit invalid');R.check(counter[0]==len(row['generated_token_ids']),'Forward count mismatch')
    R.atomic_new(result_path,row);completed.append(job['canonical_job_id']);R.atomic_status(r/'status'/f'worker{a.gpu}.json',dict(status='RUNNING',pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,job=None,completed=completed));print(json.dumps(dict(event='JOB_COMPLETE',job_id=job['canonical_job_id'],canonical_index=job['canonical_index'],tokens=row['generated_token_count'],wall_seconds=row['latency_s'],TPS=row['tokens_per_s'])),flush=True)
   R.atomic_new(wd/'RESULT.json',dict(task_id=R.TASK,status='PASS',gpu=a.gpu,gpu_uuid=uuid,completed_jobs=completed,count=len(completed),source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json')));R.atomic_status(r/'status'/f'worker{a.gpu}.json',dict(status='COMPLETE',pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,job=None,completed=completed))
  del model,c;gc.collect();torch.cuda.empty_cache();print(json.dumps(dict(event='WORKER_PASS',gpu=a.gpu,mode=a.mode)),flush=True)
 except BaseException:
  failure=dict(task_id=R.TASK,status='FAIL_CLOSED_STOP',gpu=a.gpu,gpu_uuid=uuid,mode=a.mode,repeat=a.repeat,pid=os.getpid(),job=None if current_job is None else current_job['canonical_job_id'],traceback=traceback.format_exc(),retry=False,unix=time.time())
  R.atomic_new(wd/'FAILURE.json',failure)
  try:R.atomic_new(r/'FIRST_FAILURE.json',failure)
  except FileExistsError:pass
  raise
if __name__=='__main__':main()
