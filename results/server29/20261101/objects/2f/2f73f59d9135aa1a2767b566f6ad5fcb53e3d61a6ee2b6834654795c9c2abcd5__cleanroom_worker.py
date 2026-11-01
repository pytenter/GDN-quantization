"""Small launch envelope, actual model canary, and control-only planned segments."""
import argparse,os,sys,time,hashlib,runpy
from pathlib import Path
from cleanroom_common import D,W,read,publish,sha,stop
# Preload the new resource control API before the frozen worker prepends V4.
import gpu_admission as resource
from frozen_runtime_profile import validate_environment,validate_loaded_profile

def canary(document,result):
 import torch
 import functional_worker as f
 from prospective_common import verify,load_inputs
 cfg=verify();gpu=resource.route(cfg);runtime=[f.initialize(torch,0)]
 assert resource.device(gpu,cfg)['exclusive']
 base=f.load_module(f.PRIOR/'run_short_horizon_proxy_v2.py','cleanroom_canary_base')
 direction=f.load_module(base.DIRECTION_RUNNER,'cleanroom_canary_direction')
 direction.GPU_UUIDS[gpu]=cfg['selected_gpu_uuid'];host=direction.gpu_guard(torch,gpu)
 assert all(host[k]==cfg['runtime'][k] for k in ['kernel','driver','torch','cuda_runtime','selector'])
 _,_,capture_module,core=direction.load_assets(gpu)
 _,tokens,token_path,_,_=load_inputs(torch,cfg,document,['FP'])
 # Count peak from immediately before model construction through all forwards.
 torch.cuda.init();torch.cuda.reset_peak_memory_stats();capture=None
 try:
  capture=capture_module.Capture();capture.model.eval();cache=None;digest=hashlib.sha256()
  assert not any(p.requires_grad for p in capture.model.parameters())
  for position in range(8):
   assert not (D/'STOP.json').exists()
   cache,logits,_=direction.model_step(torch,capture,cache,tokens[position],position)
   assert logits.shape[-1]==157184 and bool(torch.isfinite(logits).all())
   assert f.engine.full_kl(torch,logits,logits)==0.0,'SELF_KL_NOT_EXACT_ZERO'
   digest.update(position.to_bytes(8,'little'));digest.update(f.engine.tensor_bytes(torch,logits));del logits
  torch.cuda.synchronize();profile=validate_loaded_profile(True)
  identity=resource.proc_identity(os.getpid());guard=resource.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,evidence=result.with_suffix('.GPU_EXCLUSIVITY.jsonl'));assert guard['exclusive'],guard['failure_conditions']
  publish(result,dict(status='PASS',kind='REAL_MODEL_8_TOKEN_CANARY_NOT_SCIENCE',document=document,UUID=cfg['selected_gpu_uuid'],model_loaded=True,token_count=8,science_trajectory_count=0,numerical_qualification=False,full8_logits_sha256=digest.hexdigest(),self_fullV_FP32_KL_exact_zero=True,profile=profile,runtime=runtime,memory=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),CUDA_peak_scope='model construction and 8-token canary, not a full32K lifetime peak',token_sha256=sha(token_path)))
 finally:
  if capture is not None:capture.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['canary','parent','fresh','formal1','formal2'],required=True);p.add_argument('--document',required=True);p.add_argument('--condition',default='FP');p.add_argument('--result',type=Path,required=True);a=p.parse_args()
 try:
  assert not (D/'STOP.json').exists();profile=validate_environment();assert sys.flags.hash_randomization==0
  if a.kind=='canary':canary(a.document,a.result)
  elif a.kind in ['parent','fresh']:
   sys.argv=[str(W/'functional_worker.py'),'--document',a.document,'--condition',a.condition,'--stage','parent' if a.kind=='parent' else 'resumed_parity']
   runpy.run_path(str(W/'functional_worker.py'),run_name='__main__')
  else:
   sys.argv=[str(W/'formal_worker.py'),'--document',a.document,'--condition',a.condition]
   from formal_checkpoint_control import execute
   execute(a.document,a.condition,1 if a.kind=='formal1' else 2)
  actual=validate_loaded_profile(True)
  import torch
  measured=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
  publish(a.result.with_suffix('.PROFILE.json'),dict(status='PASS_ACTUAL_FROZEN_PROFILE',profile=actual,initial_profile=profile,actual_environment={k:os.environ.get(k) for k in ['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID']},hash_randomization=sys.flags.hash_randomization,measured_CUDA_peak=measured,CUDA_scope='last frozen loop reset through completion including checkpoint operations; canary resets before model load',epoch=time.time()))
 except BaseException as e:
  import traceback
  stop(repr(e),stage=a.kind,document=a.document,condition=a.condition,traceback=traceback.format_exc());raise

if __name__=='__main__':main()
