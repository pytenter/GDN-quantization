"""Single authorized CUDA codec micro worker. No model or AIME generation."""
from pathlib import Path
import argparse,os,sys,time,hashlib,ctypes,traceback
from qualification_io import read,write,ref,sha,verify,wait,event

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--contract',required=True);p.add_argument('--job',required=True);p.add_argument('--device-key',required=True);a=p.parse_args()
 r=Path(a.root);j=Path(a.job);c=read(a.contract);d=c['devices'][a.device_key];stop=r/'qualification/control/FIRST_FAILURE.json'
 verify(r,c);repair=read(r/'authorization/CONTROL_REPAIR_AUTHORIZATION.json');assert repair['user_quote']=='不用管那个要求了，修正后就执行GPU资格吧' and repair['new_execution_authorized'] is True
 auth=read(r/'authorization/QUALIFICATION_AUTHORIZATION.json');assert auth['user_quote']=='授权资格，另外GPU没有使用时间的限制'
 launch=wait(j/'LAUNCH.json',job=j,stop=stop,seconds=180)
 sys.path.insert(0,str(r/'control/runtime'));from resource_monitor import read_proc_identity
 actual=read_proc_identity(os.getpid());assert all(actual[k]==launch['actual_identity'][k] for k in ('pid','uid','start_ticks','boot_id','ppid'))
 assert actual['ppid']==launch['controller']['pid'];assert os.environ['CUDA_VISIBLE_DEVICES']==d['uuid']
 write(j/'ACTUAL_ENV.json',dict(status='PASS',identity=actual,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],expected_uuid=d['uuid'],python=sys.executable,CUDA_initialized=False,contract=ref(a.contract)))
 import torch,numpy as np,importlib.metadata as md
 assert torch.__version__=='2.9.1+cu128';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.device_count()==1;torch.cuda.init()
 # Establish a real context before the controller waits for compute visibility.
 # This sentinel is initialization only, not a codec/math qualification.
 context_sentinel=torch.empty((1,),dtype=torch.uint8,device="cuda")
 torch.cuda.synchronize()
 props=torch.cuda.get_device_properties(0);u=str(props.uuid)
 if not u.startswith('GPU-'):u='GPU-'+u
 assert u==d['uuid'],('ACTUAL_CUDA_UUID_MISMATCH',u,d['uuid'])
 assert props.name==d['name'];assert (props.major,props.minor)==tuple(d['compute_capability'])
 # The live context waits for actual NVML/CSV/XML presence, not launch intent.
 write(j/'BOOTSTRAP.json',dict(status='CUDA_SENTINEL_ALLOCATED_VISIBILITY_STILL_PENDING_NOT_NUMERICAL_PASS',actual_gpu_uuid=u,device_name=props.name,compute_capability=[props.major,props.minor],identity=actual,versions={k:md.version(k) for k in ('torch','triton','numpy')},CUDA_initialized=torch.cuda.is_initialized()))
 wait(j/'RESOURCE_VISIBLE_ACK.json',job=j,stop=stop,seconds=180)
 torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
 sys.path.insert(0,str(r/'src'));from four_codec_boundary import FourCodecBoundary,ARMS
 from canonical_codec import AuthorizedCudaBackend,encode as cpu_encode
 from independent_reference import canonical_group
 from matched_baseline import build_reference_encoder
 backend=AuthorizedCudaBackend(torch,r/c['cuda_artifact']['relative_path'],gpu_stage_authorized=True,expected_library_sha256=c['cuda_artifact']['sha256'])
 reference_encoder=build_reference_encoder(torch)
 h=torch.ones((1,1),dtype=torch.float32)
 while h.shape[0]<128:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
 h=(h/(128**.5)).to('cuda');f=read(r/'fixtures/CODEC_MICRO_INPUT_BITS.json');x=torch.from_numpy(np.array(f['input_bits32'],dtype=np.uint32).view(np.float32).reshape(1,1,128,128)).to('cuda')
 assert bool(torch.isfinite(x).all());assert float((h.T@h-torch.eye(128,device='cuda')).abs().max())<=1.e-6
 rows=[];proofs=[];stage_number=0
 def durable(kind,ctx,host,stages):
  nonlocal stage_number
  stage_number+=1;stem=j/'evidence'/f'{stage_number:03d}_{kind}_{ctx["arm"]}_{ctx["position"]}';stem.parent.mkdir(parents=True,exist_ok=True)
  out={}
  def npy(label,data):
   path=Path(str(stem)+'_'+label+'.npy')
   with path.open('xb') as ff:np.save(ff,data,allow_pickle=False);ff.flush();os.fsync(ff.fileno())
   out[label]=ref(path)
  npy('S0',host.numpy())
  if stages is not None:
   for key,v in stages.items():
    if isinstance(v,dict):
     for kk,vv in v.items():npy(key+'_'+kk,vv)
    elif v is not None:npy(key,v)
  rr=write(Path(str(stem)+'.json'),dict(kind=kind,context=ctx,refs=out));proofs.append(rr)
  event(j/'PROGRESS.jsonl',dict(stage=kind,arm=ctx['arm'],position=ctx['position']))
 def compare(actual,expected):
  assert actual.stages.keys()==expected.stages.keys(),'CANONICAL_STAGE_KEYS_CHANGED'
  checks={}
  for key in expected.stages:
   av=np.ascontiguousarray(actual.stages[key]);bv=np.ascontiguousarray(expected.stages[key]);checks[key]=bool(av.dtype==bv.dtype and av.shape==bv.shape and av.tobytes()==bv.tobytes())
  if not all(checks.values()):raise AssertionError(('CANONICAL_CPU_GPU_STAGE_MISMATCH',checks))
  # Independent exact-rational oracle on every distinct input group, not float tolerances.
  groups=expected.stages['S0_bits32'];oracle_count=0;seen={}
  for g,gr in enumerate(groups):
   key=gr.tobytes()
   oracle=seen.get(key)
   if oracle is None:oracle=canonical_group(gr.tolist());seen[key]=oracle
   for name,val in oracle.items():assert np.asarray(expected.stages[name][g]).tolist()==val,('CPU_ORACLE_STAGE_MISMATCH',g,name)
   oracle_count+=1
  rows.append(dict(kind='ASYM15_STAGES_CPU_GPU_AND_FRACTION',stage_checks=checks,groups=oracle_count,unique_groups=len(seen),status='PASS_EXACT'))
 for arm in ARMS:
  b=FourCodecBoundary(torch,arm,h,gpu_authorized=True,canonical_cuda_backend=backend,strict_comparator=compare,durable_evidence=durable)
  pool=torch.zeros((1,1,128,128),device='cuda',dtype=torch.float32);idx=torch.tensor([0],device='cuda')
  transformed=b.enter_value(x)
  if not b.arm.hadamard:assert transformed is x
  roundtrip=b.restore_core(transformed);err=float((roundtrip-x).abs().max())
  assert err<=max(1.e-6,float(x.abs().max())*2.e-6),('H_MICRO_ROUNDTRIP',err)
  for pos in (0,1):
   b.before_native_kernel(pool.index_select(0,idx),0,pos)
   if pos==0:pool.copy_(transformed.transpose(-1,-2).contiguous())
   # pos1 intentionally re-encodes its own actual QDQ state; no idempotence assumption.
   host=pool.transpose(-1,-2).contiguous().to('cpu',copy=True)
   if not b.arm.asymmetric:durable('SYMMETRIC_PRE_ENCODE',dict(arm=arm,position=pos,layer=0),host,None)
   out=b.after_native_kernel(pool,idx,0,pos,probe=True)
   if not b.arm.asymmetric:
    expected=reference_encoder(host,32,False)
    checks={}
    for name,av,bv in [('codes',out.codes,expected.codes),('scales',out.scales,expected.scales),('decoded',out.decode(torch),expected.decode())]:
     checks[name]=bool(torch.equal(av.detach().cpu().contiguous().view(torch.uint8),bv.contiguous().view(torch.uint8)))
    durable('SYMMETRIC_ACTUAL_CODES_SCALES_DECODE',dict(arm=arm,position=pos,layer=0),host,dict(actual_codes=out.codes.detach().cpu().numpy(),actual_scales_bits16=out.scales.detach().cpu().view(torch.uint16).numpy(),actual_decoded_bits32=out.decode(torch).detach().cpu().view(torch.uint32).numpy(),cpu_codes=expected.codes.numpy(),cpu_scales_bits16=expected.scales.view(torch.uint16).numpy(),cpu_decoded_bits32=expected.decode().view(torch.uint32).numpy()))
    assert all(checks.values()),('SYMMETRIC_STRICT_PARITY_FAILED',checks)
    rows.append(dict(kind='SYMMETRIC_CPU_GPU',arm=arm,position=pos,status='PASS_EXACT',checks=checks))
   event(j/'PROGRESS.jsonl',dict(stage='ARM_WRITE_COMPLETE',arm=arm,position=pos))
  assert (b.writes,b.consumptions)==(2,1)
  # Deliberate corruption is a planned negative control, never a fallback.
  pool.view(torch.uint8).flatten()[0]^=1
  rejected=False
  try:b.before_native_kernel(pool.index_select(0,idx),0,2)
  except AssertionError as e:assert 'POST_QDQ_BYTES_DIFFER' in str(e);rejected=True
  assert rejected
  rows.append(dict(kind='BOUNDARY_COUNTS_AND_NEGATIVE',arm=arm,writes=b.writes,consumptions=b.consumptions,bitflip_rejected=rejected,H_roundtrip_max_absolute_error=err,status='PASS'))
  b.close();del pool
 torch.cuda.synchronize();verify(r,c)
 result=dict(status='PASS',stage='CODEC_MICRO_ONLY_NOT_REAL_MODEL_QUALIFICATION',source_verified=True,numerical_pass=True,profile_verified=True,environment_verified=True,gpu_uuid=u,device_key=a.device_key,identity=actual,source_contract=ref(a.contract),checks=rows,durable_evidence=proofs,models_loaded=0,AIME_generations=0,elapsed_seconds=time.monotonic()-t0,CUDA_peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),scope='reset_before_micro_input_allocation_through_result_record_not_model_load'),versions={k:md.version(k) for k in ('torch','triton','numpy')})
 write(j/'WORKER_RESULT.json',result)
 # Keep the actual context until controller validates result and observes it.
 wait(j/'RESULT_OBSERVED_ACK.json',job=j,stop=stop)
 print('CODEC_MICRO_COMPLETE_NOT_MODEL_QUALIFICATION',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();sys.exit(1)
