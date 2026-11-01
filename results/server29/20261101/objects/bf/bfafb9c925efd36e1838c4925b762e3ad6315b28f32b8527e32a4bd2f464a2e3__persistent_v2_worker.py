"""One fresh model-resident production worker; never unload model in-process."""
import argparse,os,time,traceback
import persistent_v2_common as A
def semantic(model,g,core,bank,ident,cfg):
 import torch
 g.setup_seed(1);cache=None;outputs=[];final=[];geometry={}
 audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
 with torch.inference_mode(),g.QwenFrozenRotationPatch(model,bank,ident) as patch:
  for i,token in enumerate(cfg['input_token_ids']):
   out=model(input_ids=torch.tensor([[token]],dtype=torch.long,device='cuda:0'),past_key_values=cache,use_cache=True,output_hidden_states=True)
   cache=out.past_key_values;hidden=out.hidden_states[-1]
   A.require(bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(hidden).all()),'nonfinite semantic outputs')
   outputs.append(dict(token=i,logit_hash=g.tensor_sha256(out.logits),Final_Hidden_hash=g.tensor_sha256(hidden)))
   expected={}
   if i==31:
    for l in g.GDN_LAYERS:
     st=g.get_state(cache,l);ref=core.CAYLEY.qwen_c128_qdq(st.float())
     A.require(bool(torch.isfinite(st).all()) and bool(torch.isfinite(ref.scale).all()),'nonfinite semantic state/scale')
     expected[l]=ref.dequant.to(st.dtype)
     final.append(dict(layer=l,state_hash=g.tensor_sha256(st),scale_hash=g.tensor_sha256(ref.scale),qcode_hash=g.tensor_sha256(ref.codes)))
     geometry[str(l)]=dict(shape=list(st.shape),dtype=str(st.dtype))
   if i>0:g.quantize_c128(cache,audit)
   if i==31:
    for r in final:
     st=g.get_state(cache,r['layer']);A.require(torch.equal(st,expected[r['layer']]),'canonical final QDQ/writeback differs')
     r['post_QDQ_hash']=g.tensor_sha256(st)
 A.require(audit['calls']==31 and sorted(audit['layers'])==list(g.GDN_LAYERS),'semantic QDQ coverage')
 A.require(patch.calls==768 and patch.layers==set(g.GDN_LAYERS),'semantic rotation coverage')
 return dict(input_hash=cfg['input_token_sha256'],rotation_hash=ident['rotation_sha256'],outputs=outputs,final_layers=final,geometry=geometry,qdq_calls=31,rotation_calls=768,status='PASS')
def main(gpu,arm,key):
 import torch
 A.require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1,'worker not single GPU')
 A.verify_manifest();cfg=A.read(A.ROOT/'configs/v2_smoke_input.json');protocol=A.read(A.ROOT/'configs/v2_protocol.json')
 g=A.gmodule();core=A.mod('v2_canonical_core',A.CORE)
 A.require(g.runtime_version()==protocol['runtime'],'runtime drift')
 g.setup_seed(1);model,tok=g.load_model_and_tokenizer()
 for p in model.parameters():p.requires_grad_(False)
 A.require(next(model.parameters()).dtype==torch.bfloat16,'model dtype drift')
 torch.cuda.synchronize();model_allocated=torch.cuda.memory_allocated()
 path,manifest=A.rotation_paths(arm);A.require(A.sha(path)==A.SHAS[arm],'rotation drift')
 bank,ident=g.load_frozen_rotation(path,manifest,torch.device('cuda:0'),arm)
 torch.cuda.synchronize();bank_allocated=torch.cuda.memory_allocated()
 bank_bytes=sum(t.numel()*t.element_size() for t in bank.buffers())
 payload=torch.load(path,map_location='cpu',weights_only=False);orth=[]
 for l in g.GDN_LAYERS:
  t=payload['matrices'][str(l)];A.require(torch.equal(t,bank.layer(l).matrix().cpu()),'frozen matrix roundtrip drift')
  e=float((t.T@t-torch.eye(128)).abs().max());A.require(e<=1e-4,'orthogonality gate');orth.append(dict(layer=l,max_abs=e))
 del payload,t
 # Record-only arm aliases; same generation implementation and all sampling constants.
 g.METHODS[arm.lower()]='QWEN_INT8_C128_KEY_'+arm
 g.MAX_NEW_TOKENS=128
 item=dict(problem=cfg['production_problem'],problem_id='TRAIN_FIRST_ROW_ENGINEERING',problem_idx=0,answer=None)
 msgs,ids=g.render_input(tok,item['problem'])
 A.require(ids[0].tolist()==cfg['production_input_ids'] and msgs==cfg['production_prompt_messages'],'production prompt/input drift')
 del ids,msgs
 initial=semantic(model,g,core,bank,ident,cfg);cycles=[]
 def request(label):
  r=g.generate_one(model,tok,item,1,arm.lower(),str(gpu),gpu,0,bank,ident,A.sha(A.ROOT/'configs/v2_future_canonical_protocol.json'))
  A.require(not r['runtime_error'] and not r['nonfinite'],'production request error '+str(r['runtime_error']))
  A.require(r['generated_token_ids'] and r['rotation_sha256']==A.SHAS[arm],'production output/rotation identity')
  q=r['quantizer_audit'];A.require(q['calls']==len(r['generated_token_ids'])-1,'production QDQ call count')
  if q['calls']:A.require(q['layers']==list(g.GDN_LAYERS),'production QDQ layer coverage')
  torch.cuda.synchronize()
  m=dict(allocated_end=torch.cuda.memory_allocated(),reserved_end=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
  print(A.digest(dict(label=label,tokens=r['generated_token_count'],memory=m)),label,r['generated_token_count'],m,flush=True)
  return dict(label=label,record=r,memory=m)
 warmup=request('warmup')
 for cycle in (1,2,3):
  sem=semantic(model,g,core,bank,ident,cfg);r=request('cycle'+str(cycle));r['semantic']=sem;cycles.append(r)
 memory_pass=len({x['memory']['allocated_end'] for x in cycles})==1 and len({x['memory']['reserved_end'] for x in cycles})==1
 semantics_pass=all(x['semantic']==initial for x in cycles)
 generated=[x['record']['generated_token_ids'] for x in cycles];repeat_pass=semantics_pass and generated[0]==generated[1]==generated[2]
 result=dict(status='PASS' if memory_pass and repeat_pass else 'FAIL',pid=os.getpid(),gpu=gpu,GPU_uuid=A.UUIDS[gpu],arm=arm,key=key,rotation_identity=ident,rotation_SHA=A.SHAS[arm],runtime=g.runtime_version(),config_SHA=A.sha(A.ROOT/'configs/v2_protocol.json'),input_hash=cfg['input_token_sha256'],initial_semantic=initial,warmup=warmup,cycles=cycles,orthogonality=orth,model_allocated_bytes=model_allocated,bank_allocated_bytes=bank_allocated,additional_resident_allocated_bytes=bank_allocated-model_allocated,rotation_tensor_bytes=bank_bytes,rotation_file_bytes=path.stat().st_size,RESIDENT_WORKER_MEMORY_STABILITY_GATE='PASS' if memory_pass else 'FAIL',RESIDENT_REQUEST_REPEATABILITY_GATE='PASS' if repeat_pass else 'FAIL',model_loaded_once=True,rotation_loaded_once=True,in_process_model_unload_tested=False,production_function_SHA=A.sha(A.FORMAL),NEW_AIME_GENERATION=0,SCORING_STARTED='NO')
 A.save('analysis/'+key+'_worker_result.json',result)
 A.require(memory_pass,'resident endpoint plateau FAIL; no tolerance or retry')
 A.require(repeat_pass,'resident output/fingerprint repeatability FAIL')
 print('V2_WORKER_PASS_NORMAL_EXIT',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--arm',choices=('C6R','C7'),required=True);p.add_argument('--key',required=True);a=p.parse_args()
 try:main(a.gpu,a.arm,a.key)
 except BaseException as e:
  A.save('analysis/'+a.key+'_worker.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),pid=os.getpid(),retry=False));raise
