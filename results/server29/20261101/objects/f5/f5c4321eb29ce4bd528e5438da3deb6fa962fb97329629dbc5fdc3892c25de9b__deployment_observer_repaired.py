#!/usr/bin/env python3
"""Deployment-only CPU audits and deterministic teacher-forced formal-path smoke.

No generation/scorer/backward/optimizer entry point. Frozen numerical functions
are imported without editing; wrappers observe kernel inputs and cache writes.
"""
import argparse,ast,gc,hashlib,importlib.util,json,os,platform,subprocess,sys,time,traceback,weakref
from pathlib import Path
TASK='QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1'
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
BASE=Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OLD=BASE/'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1'
FORMAL=OLD/'scripts/run_qwen_c5_c6_formal.py'
CORE=BASE/'experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROTATION=BASE/'experiments/shared/rotation/cayley_rotation.py'
PRIOR=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1')
MODEL=Path('/data/zypan/modelscope_models/Qwen3.5-9B')
ARMS=('C6R','C7');SOURCE=ROOT/'source_evidence'
def require(v,m):
 if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def identity(p,d=None):
 p=Path(p);s=p.stat();h=sha(p);require(d is None or h==d,'SHA mismatch '+str(p))
 require(p.stat().st_size==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns,'file changed during read')
 return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h)
def cmd(x):return subprocess.check_output(x,cwd=REPO,text=True).strip()
def mod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def gmodule():return mod('deployment_unchanged_formal',FORMAL)
def prepare():
 require(not (ROOT/'analysis/prepare.error.json').exists(),'previous failure; stop')
 inv=read(ROOT/'provenance/a800_source_deployment_inventory.json');commit=inv['A800_SOURCE_COMMIT']
 require(cmd(['git','rev-parse','HEAD'])==commit,'target not exact source commit')
 require(read(SOURCE/'logs/posttrain_closure_commit.json')['new_commit_SHA']==commit,'closure commit ambiguity')
 require(read(SOURCE/'analysis/posttrain_closure_gates.json')['status']=='PASS','source closure not PASS')
 transfer=[]
 for dest,r in inv['transfer_members'].items():transfer.append(identity(ROOT/dest,r['sha256']))
 save('provenance/transfer_identity_audit.json',dict(status='PASS',files=transfer,source_commit=commit,source_package_sha256=sha(ROOT/'source_package.tar.gz'),only_frozen_R_and_required_metadata=True,C5R_deployed=False))
 selected=read(SOURCE/'deployment/selected_artifacts_pre_endpoint.json')['conditions']
 audit=read(SOURCE/'analysis/frozen_rotation_export_audit.json')['conditions']
 for arm in ARMS:
  p=ROOT/'rotations'/arm/'frozen_rotation_fp32.pt';require(sha(p)==audit[arm]['sha256']==selected[arm]['file_sha256'],'arm SHA ambiguity')
  save('configs/'+arm.lower()+'_rotation_manifest.json',selected[arm])
 # Historical source identities fixed before this deployment; no semantic edits.
 fcfg=read(OLD/'configs/formal_config.json');require(sha(OLD/'configs/formal_config.json')=='557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836','formal config drift')
 require(sha(FORMAL)==fcfg['runner_sha256']=='77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69','formal runner drift')
 prior=read(PRIOR/'protocol/canonical_endpoint_amended.json')
 require(sha(PRIOR/'protocol/canonical_endpoint_amended.json')==read(PRIOR/'protocol/canonical_endpoint_amended_freeze.json')['config_sha256'],'prior endpoint config drift')
 protected={};freeze=read(SOURCE/'provenance/trusted_environment_freeze.json');modelrows=[]
 for f in freeze['model_files']:
  p=MODEL/Path(f['path']).name
  if p.suffix in ('.json','.safetensors','.jinja') or p.name=='merges.txt':
   r=identity(p,f['sha256']);protected[str(p)]=r;modelrows.append(r)
 save('provenance/qwen_3090_model_identity.json',dict(status='PASS',model_path=str(MODEL),source_model_files=modelrows,revision_policy='exact frozen local weight/config/tokenizer bytes; no download or replacement',historical_config_expected=read(BASE/'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/configs/parent_manifest.json')['model_config_sha256']))
 # Full historical frozen source/model/protocol inventory, including scorer identity only.
 for i,(p,d) in enumerate(prior['frozen_sources'].items()):
  if p in protected:require(protected[p]['sha256']==d,'historical model digest differs')
  else:protected[p]=identity(p,d)
  if i%300==0:print(json.dumps({'event':'historical_frozen_revalidation','files':i+1}),flush=True)
 for p in (FORMAL,CORE,ROTATION,OLD/'scripts/run_aime_worker.sh',OLD/'configs/hardware_assignment.json',PRIOR/'protocol/canonical_endpoint_amended.json',PRIOR/'protocol/canonical_endpoint_amended_freeze.json',BASE/'CALIBRATION_RAW_TEXTS.jsonl'):
  protected[str(p)]=identity(p)
 save('provenance/protected_target_inventory.json',dict(status='PASS',files=list(protected.values()),count=len(protected),pre_task_HEAD=commit,pre_task_git_status=cmd(['git','status','--short']),scope='frozen historical runtime, base weights/tokenizer, calibration text, outputs and scoring evidence; scorer not executed'))
 g=gmodule();runtime=g.runtime_version();require(runtime==prior['runtime'],'canonical runtime versions changed')
 gpu=cmd(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader,nounits']).splitlines()
 require(len(gpu)==4 and all('RTX 3090' in r for r in gpu),'target hardware differs')
 import torch
 flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
 # Historical policy promises fresh-process equality, not newly enabled kernels.
 hist=read(OLD/'../QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/analysis/repeatability_and_hadamard_guard_v2.json')
 require(all(hist['conditions'][a]['repeatability']['exact'] for a in ('C5','C6')),'historical repeatability not exact')
 tree=ast.parse(FORMAL.read_text());source={x.name:ast.get_source_segment(FORMAL.read_text(),x) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
 save('analysis/qwen_3090_qdq_semantics_audit.json',dict(status='PASS',canonical_quantize_c128_source=source['quantize_c128'],source_sha256=sha(FORMAL),training_quantizer_source=identity(ROTATION),formula='unmodified original sources: symmetric round-to-nearest-even INT8 [-127,127], maxabs key-axis dim=-2 /127, EPS 1e-12; output state.copy_',dynamic_exact_reference='required in each smoke: compare original outer quantize_c128 to original CAYLEY.qwen_c128_qdq; no copied formula; inference never uses STE backward',scale_axis='Key C128',rotation='separate q/k @ H @ Delta; Value unchanged'))
 save('provenance/deployment_runtime_identity.json',dict(status='PASS',hostname=platform.node(),runtime=runtime,python_executable=sys.executable,GPUs=gpu,deterministic_flags=flags,historical_determinism_policy='same GPU/input/seed/config fresh processes: exact tensor fingerprints',four_gpu_policy='no preexisting numerical tolerance between devices; artifact/config/runtime identity exact, observed fingerprints descriptive',A800_3090_NUMERICAL_PARITY_POLICY='NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE',policy_search_scope=[str(SOURCE/'protocol/preregistration.json'),str(SOURCE/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json'),str(PRIOR/'protocol/canonical_endpoint_amended.json'),str(OLD/'../QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/analysis/repeatability_and_hadamard_guard_v2.json')],manual_hf_runtime=True,sglang=False,model_parallel=False,TP=False,radix_cache=False,mamba_radix=False,frozen_function_sources={k:hashlib.sha256(source[k].encode()).hexdigest() for k in ('load_model_and_tokenizer','load_frozen_rotation','QwenFrozenRotationPatch','quantize_c128','get_state')}))
 # Fixed historical non-AIME input BEFORE any GPU result: TRAIN:first-row 32-token prefix.
 core=mod('deployment_canonical_core',CORE);ids,_=core.tokenize(__import__('transformers').AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True),core.corpus_rows('TRAIN')[0]);ids=ids[:32]
 require(len(ids)==32 and g.tensor_sha256(torch.tensor(ids,dtype=torch.long))=='1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d','canonical smoke input mismatch')
 save('configs/frozen_smoke_protocol.json',dict(status='PASS',input_token_ids=ids,input_token_sha256=g.tensor_sha256(torch.tensor(ids,dtype=torch.long)),sample='historical TRAIN:first-row',token_count=32,seed=1,conditions=['Hadamard','C6R','C7'],C6R_C7_fresh_process_repeats=2,GPU_list=[0,1,2,3],no_teacher_cache_created=True,no_free_generation=True,engineering_deployment_smoke_only=True,formal_prefill_tokens=1,qdq_after_decode_only=True,scientific_superiority_use=False,source_sha256=sha(ROOT/'scripts/deployment_closure.py')))
 save('analysis/prepare.completion.json',dict(status='PASS',GPU_work_started=False,source_commit=commit,new_AIME_generation=0))
def smoke(gpu,repeat):
 import torch
 require(gpu==0 and repeat==1 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','only physical GPU0 repaired repeat1 authorized')
 require(cmd(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'])=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422','physical GPU0 UUID mismatch')
 verify_repair_freeze()
 save('logs/observer_repair_gpu_attempt_started.json',dict(attempt=1,gpu=0,repeat=1,pid=os.getpid(),time=time.time(),retry=False))
 require(read(ROOT/'analysis/prepare.completion.json')['status']=='PASS','CPU gate missing')
 cfg=read(ROOT/'configs/frozen_smoke_protocol.json');require(sha(ROOT/'scripts/deployment_closure.py')==cfg['source_sha256'],'prepared source changed')
 if Path(__file__).name=='deployment_closure_gpu.py':
  amendment=read(ROOT/'analysis/pre_gpu_runtime_and_cleanup_source_amendment.json')
  require(amendment['old_sha256']==cfg['source_sha256'] and amendment['new_sha256']==sha(__file__) and amendment['GPU_work_started']==False,'unfrozen GPU observer source')
 g=gmodule();require(g.runtime_version()==read(ROOT/'provenance/deployment_runtime_identity.json')['runtime'],'runtime drift')
 core=mod('deployment_canonical_core',CORE)
 require(core.GDN_LAYERS==g.GDN_LAYERS,'layer mapping differs')
 g.setup_seed(cfg['seed'])
 device=torch.device('cuda:0');memory=[]
 model_ref=observer_ref=bank_ref=None;allresults={}
 rotation_hashes={a:read(ROOT/'configs'/f'{a.lower()}_rotation_manifest.json')['file_sha256'] for a in ARMS}
 def flush_diagnostics():
  diagnostic_atomic('analysis/gpu0_repaired_smoke_runtime_diagnostics.json',dict(status='DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_PASS',memory=memory,condition_completion={a:r['status'] for a,r in allresults.items()},input_hash=cfg['input_token_sha256'],rotation_hashes=rotation_hashes,GPU_uuid='GPU-a78176ed-f97f-c54f-6201-5a01b570f422',attempt=1))
 def mark(point,condition=None):
  r=mem(point);r.update(condition=condition,model_alive=None if model_ref is None else model_ref() is not None,observer_alive=None if observer_ref is None else observer_ref() is not None,rotation_bank_alive=None if bank_ref is None else bank_ref() is not None);flush_diagnostics();return r
 def mem(point):
  torch.cuda.synchronize();r=dict(point=point,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());memory.append(r);return r
 mem('before_model');mark('M0');started=time.time();model,tok=g.load_model_and_tokenizer();model_ref=weakref.ref(model);mark('M1')
 for p in model.parameters():p.requires_grad_(False)
 del p
 require(next(model.parameters()).dtype==torch.bfloat16,'canonical dtype differs')
 before={n:g.tensor_sha256(t) for n,t in model.state_dict().items()};mem('baseline_model_loaded')
 allresults={}
 for arm in (['Hadamard','C6R','C7'] if repeat==1 else ARMS):
  g.setup_seed(cfg['seed']);torch.cuda.reset_peak_memory_stats()
  bank=None;rotation_audit=None
  if arm!='Hadamard':
   path=ROOT/'rotations'/arm/'frozen_rotation_fp32.pt';manifest=ROOT/'configs'/f'{arm.lower()}_rotation_manifest.json';bank,ident=g.load_frozen_rotation(path,manifest,device,arm)
   payload=torch.load(path,map_location='cpu',weights_only=False);rows=[]
   threshold=read(SOURCE/'analysis/frozen_rotation_export_audit.json')['conditions'][arm]['orthogonality_threshold']
   require(payload['orientation']=='q/k.float() @ H128 @ Delta; recovery Delta then H128 on key axis','rotation orientation differs')
   for l in g.GDN_LAYERS:
    cpu=payload['matrices'][str(l)];gpu_matrix=bank.layer(l).matrix();require(torch.equal(cpu,gpu_matrix.cpu()),'GPU roundtrip modified rotation')
    orth=float((cpu.T@cpu-torch.eye(128)).abs().max());require(orth<=threshold,'frozen orthogonality threshold failed')
    rows.append(dict(layer=l,sha256=g.tensor_sha256(cpu),GPU_roundtrip_exact=True,orthogonality_max_abs=orth));del gpu_matrix
   rotation_audit=dict(status='PASS',identity=ident,layers=rows,threshold=threshold,axis='Key',Value_rotation=False,layer_mapping=list(g.GDN_LAYERS),direct_frozen_load=True)
   patch=g.QwenFrozenRotationPatch(model,bank,ident)
  else:patch=g.QwenKeyHadamardPatch(model)
  import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
  step={'index':0};previous={};continuity=[];outputs=[];qdq_rows=[];kernels=[];hidden={};hook=model.lm_head.register_forward_pre_hook(lambda m,a:hidden.update(tensor=a[0]))
  quant_audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
  cache=None;out=None
  observer_ref=weakref.ref(patch);bank_ref=None if bank is None else weakref.ref(bank);mark('M2',arm)
  with torch.inference_mode(),patch:
   original=qmod.torch_recurrent_gated_delta_rule
   def observed(query,key,value,*args,**kw):
    l=int(patch.current_layer);initial=kw.get('initial_state',args[3] if len(args)>3 else None)
    if step['index']>0:
     require(initial is not None,'no recurrent initial state');require(l in previous and torch.equal(initial,previous[l]),'next token did not consume post-QDQ state')
     continuity.append({'token':step['index'],'layer':l,'exact':True})
    kernels.append({'token':step['index'],'layer':l,'Value_dtype':str(value.dtype),'initial_state_present':initial is not None})
    return original(query,key,value,*args,**kw)
   qmod.torch_recurrent_gated_delta_rule=observed
   try:
    for i,token in enumerate(cfg['input_token_ids']):
     step['index']=i;kwargs=dict(input_ids=torch.tensor([[token]],dtype=torch.long,device=device),use_cache=True)
     if cache is not None:kwargs['past_key_values']=cache
     out=model(**kwargs);cache=out.past_key_values
     require(bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(hidden['tensor']).all()),'nonfinite logit/FinalHidden')
     outputs.append(dict(token=i,logits_sha256=g.tensor_sha256(out.logits),final_hidden_sha256=g.tensor_sha256(hidden['tensor'])))
     expected={};before_states={};layerrows=[]
     if i>0:
      for l in g.GDN_LAYERS:
       st=g.get_state(cache,l);require(bool(torch.isfinite(st).all()),'nonfinite pre QDQ')
       reference=core.CAYLEY.qwen_c128_qdq(st.float());require(bool(torch.isfinite(reference.scale).all()),'nonfinite scale')
       require(int(reference.codes.min())>=-127 and int(reference.codes.max())<=127,'qcode out of range')
       expected[l]=reference.dequant.to(st.dtype);before_states[l]=g.tensor_sha256(st)
       layerrows.append(dict(layer=l,pre_sha256=before_states[l],scale_sha256=g.tensor_sha256(reference.scale),qcodes_sha256=g.tensor_sha256(reference.codes),scale_shape=list(reference.scale.shape)));del st,reference
      g.quantize_c128(cache,quant_audit)
     previous={l:g.get_state(cache,l).detach().clone() for l in g.GDN_LAYERS}
     if i>0:
      for r in layerrows:
       l=r['layer'];require(torch.equal(previous[l],expected[l]) and bool(torch.isfinite(previous[l]).all()),'formal/train QDQ mismatch')
       r.update(post_sha256=g.tensor_sha256(previous[l]),formal_vs_training_QDQ_exact=True,post_finite=True)
      qdq_rows.append(dict(token=i,layers=layerrows))
     del expected
   finally:qmod.torch_recurrent_gated_delta_rule=original
  hook.remove()
  require(len(continuity)==31*24 and quant_audit['calls']==31 and sorted(quant_audit['layers'])==list(g.GDN_LAYERS),'smoke full writeback/QDQ coverage failed')
  require(patch.calls==32*24 and patch.layers==set(g.GDN_LAYERS),'rotation mapping coverage failed')
  finalstate={str(l):g.tensor_sha256(s) for l,s in previous.items()}
  mem(arm+'_peak')
  allresults[arm]=dict(status='PASS',rotation_audit=rotation_audit,input_token_sha256=cfg['input_token_sha256'],tokens=32,outputs=outputs,final_state_sha256=finalstate,qdq=qdq_rows,writeback=continuity,quantizer_audit={k:sorted(v) if isinstance(v,set) else v for k,v in quant_audit.items()},rotation_calls=patch.calls,kernel_entry_count=len(kernels),first_state_entry=kernels[:24],no_teacher_reset=True,no_free_generation=True,smoke_only=True)
  diagnostic_atomic('analysis/gpu0_repaired_smoke_output_fingerprints.json',dict(status='DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_PASS',conditions=allresults,input_hash=cfg['input_token_sha256'],rotation_hashes=rotation_hashes,old_attempt_fingerprints_available=False,no_numeric_parity_claim_against_unsaved_old_outputs=True))
  mark({'Hadamard':'M3','C6R':'M4','C7':'M5'}[arm],arm)
  # Lifecycle-only repair: drop the wrapped callable closure after use/restoration.
  del original,observed
  del patch,bank
  mark('M6',arm)
  del cache,out,previous,qdq_rows,hidden,kwargs;hidden={}
  mark('M7',arm)
  gc.collect();torch.cuda.empty_cache();mem(arm+'_post_cache_cleanup');flush_diagnostics()
 after={n:g.tensor_sha256(t) for n,t in model.state_dict().items()};require(before==after,'base weights changed during smoke')
 del model,tok
 mark('M8')
 gc.collect();torch.cuda.empty_cache()
 mark('M9')
 end=mem('post_model_cleanup');mark('M10')
 require(end['allocated']<1024*1024,'unexpected memory retention')
 save('analysis/gpu0_repaired_smoke.json',dict(status='PASS',gpu=gpu,repeat=repeat,conditions=allresults,memory=memory,base_weight_hashes=before,base_weights_unchanged=True,runtime=g.runtime_version(),GPU_uuid=cmd(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader']),latency_s=time.time()-started,free_generation_count=0))
 print('GPU0_REPAIRED_REPEAT1=PASS_STOP',flush=True)
def diagnostic_atomic(relative,data):
 p=ROOT/relative;tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def verify_repair_freeze():
 manifest=ROOT/'manifests/observer_repair_pre_gpu_sha256.txt'
 for line in manifest.read_text().splitlines():
  digest,path=line.split('  ',1);require(sha(path)==digest,'pre-GPU repair freeze drift '+path)
 require(read(ROOT/'analysis/observer_repair_scientific_path_identity.json')['status']=='PASS','scientific path gate')
 require(read(ROOT/'analysis/observer_reference_graph_after_repair.json')['status']=='PASS','static repair gate')
 require(read(ROOT/'provenance/observer_repair_pre_gpu_identity.json')['status']=='PASS','pre-GPU protected gate')
 require(not (ROOT/'logs/observer_repair_gpu_attempt_started.json').exists(),'one attempt already consumed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('smoke',),required=True);p.add_argument('--gpu',type=int,choices=(0,),required=True);p.add_argument('--repeat',type=int,choices=(1,),required=True);a=p.parse_args()
 require(not (ROOT/'analysis/gpu0_repaired_smoke.error.json').exists() and not (ROOT/'analysis/gpu0_repaired_smoke.json').exists(),'prior repaired evidence present; no retry')
 try:smoke(a.gpu,a.repeat)
 except BaseException as e:save('analysis/gpu0_repaired_smoke.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),retry=False));raise
if __name__=='__main__':main()
