"""Comprehensive, fixed-input CPU numerical attribution. No GPU/model launch."""
import ast, hashlib, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R=Path(sys.argv[1]).resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1);torch.set_num_interop_threads(1)
U32=2.**-24;U64=2.**-53
def gamma(n,u=U32):return n*u/(1-n*u)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()
def opnorm(x):return float(np.linalg.norm(x,2))
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
def dist(x):
 a=np.asarray(x);return dict(minimum=float(a.min()),median=float(np.median(a)),P95=float(np.quantile(a,.95)),maximum=float(a.max()))
def hmatrix():
 h=np.ones((1,1),np.float64)
 while len(h)<128:h=np.block([[h,h],[h,-h]])
 return h/math.sqrt(128)
assert not (R/'analysis/CPU_EVIDENCE_COMPLETION_STARTED.json').exists()
put('analysis/CPU_EVIDENCE_COMPLETION_STARTED.json',dict(pid=os.getpid(),monotonic=time.monotonic(),scope='Complete previously unexecuted CPU helper/token22 checks only; preserve original decorator NameError and all earlier outputs',original_source_sha256=sha(R/'scripts/analyze_cpu.py'),loader_change='Strip dispatch decorator only in isolated CPU AST function extraction; mathematical function body unmodified',GPU_dispatches=0))
protocol=read(R/'CPU_ANALYSIS_PROTOCOL.json');matrices={};failures=[]
assert read(R/'M0_M3_FP32_NUMERICAL_ATTRIBUTION.json')['failures']==[]
assert read(R/'FP64_GDN_MATH_REFERENCE.json')['status'].startswith('PASS')
for row in read(R/'reuse/CANONICAL_UH_MATRICES_AND_SHA.json')['layers']:
 with np.load(R/'assets'/row['path']) as z:
  H=hmatrix()
  for h in range(16):matrices[(row['layer'],h)]=(z['U_fp64_unique'][h]@H,z['A_fp32_unique'][h].astype(np.float64))
rng=np.random.default_rng(protocol['seed'])
try:
 # Execute the *actual bound source* torch helper on CPU, compared to a separate
 # NumPy FP64 arithmetic graph. This is not a CUDA kernel parity assertion.
 src=(R/'reuse/actual_modeling_qwen3_5.py').read_text();tree=ast.parse(src)
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['l2norm','torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule']]
 
 for node in nodes:node.decorator_list=[]  # CPU sandbox only; native file remains byte-identical
 ns={'torch':torch,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<bound_actual_Qwen_GDN_CPU_functions>','exec'),ns)
 native=ns['torch_recurrent_gated_delta_rule'];chunk=ns['torch_chunk_gated_delta_rule'];norm=ns['l2norm']
 qt=torch.tensor(rng.normal(size=(1,128,32,128)).astype(np.float32)).to(torch.bfloat16)
 kt=torch.tensor(rng.normal(size=(1,128,32,128)).astype(np.float32)).to(torch.bfloat16)
 vt=torch.tensor(rng.normal(size=(1,128,32,128)).astype(np.float32)).to(torch.bfloat16)
 gt=torch.tensor(rng.uniform(-.3,-.01,size=(1,128,32)).astype(np.float32));bt=torch.tensor(rng.uniform(.05,.95,size=(1,128,32)).astype(np.float32)).to(torch.bfloat16)
 nq=norm(qt).float();nk=norm(kt).float();state=None;ref=np.zeros((32,128,128));prop=np.zeros(32);trajectory=[]
 for t in range(128):
  qq=nq[:,t:t+1];kk=nk[:,t:t+1];vv=vt[:,t:t+1];gg=gt[:,t:t+1];bb=bt[:,t:t+1]
  prev=np.zeros_like(ref) if state is None else state.numpy()[0].astype(np.float64)
  core,state=native(qq,kk,vv,g=gg,beta=bb,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=False)
  k=kk.numpy()[0,0].astype(np.float64);q=(qq/math.sqrt(128)).numpy()[0,0].astype(np.float64);v=vv.float().numpy()[0,0].astype(np.float64);a=gg.exp().numpy()[0,0].astype(np.float64);b=bb.float().numpy()[0,0].astype(np.float64)
  decay=a[:,None,None]*prev;mem=np.einsum('hkv,hk->hv',decay,k);delta=b[:,None]*(v-mem);one=decay+k[:,:,None]*delta[:,None,:]
  # Componentwise FP32 graph bound. Alpha is the already-computed FP32 exp(g).
  # Error of exp itself is separately unknown for CUDA, not assumed 1ULP.
  bd=U32*np.abs(decay)+np.finfo(np.float32).smallest_subnormal
  bm=np.einsum('hkv,hk->hv',bd,np.abs(k))+gamma(256)*np.einsum('hkv,hk->hv',np.abs(decay)+bd,np.abs(k))
  bdelta=np.abs(b[:,None])*bm+gamma(2)*np.abs(b[:,None])*(np.abs(v)+np.abs(mem)+bm)
  budget=bd+np.abs(k[:,:,None])*bdelta[:,None,:]+gamma(2)*(np.abs(decay)+bd+np.abs(k[:,:,None]*delta[:,None,:])+np.abs(k[:,:,None])*bdelta[:,None,:])
  budget+=gamma(1024,U64)*(1+np.abs(one)+np.abs(prev))
  actual=state.numpy()[0].astype(np.float64);local=np.abs(actual-one)
  assert np.all(local<=budget)
  ref=a[:,None,None]*ref;ref+=k[:,:,None]*(b[:,None]*(v-np.einsum('hkv,hk->hv',ref,k)))[:,None,:]
  L=np.abs(a)*np.maximum(1.,np.abs(1-b*np.sum(k*k,axis=-1)))
  prop=L*prop+np.linalg.norm(budget,axis=(-2,-1))
  actual_error=np.linalg.norm(actual-ref,axis=(-2,-1));assert np.all(actual_error<=prop+gamma(1024,U64))
  expected_read=np.einsum('hkv,hk->hv',actual,q);read=core.numpy()[0,0].astype(np.float64)
  readbudget=gamma(256)*np.einsum('hkv,hk->hv',np.abs(actual),np.abs(q))+gamma(1024,U64)*(1+np.abs(expected_read))
  assert np.all(np.abs(read-expected_read)<=readbudget)
  trajectory.append(dict(token=t,local_maxabs=float(local.max()),local_budget_max=float(budget.max()),max_local_componentwise_ratio=float((local/budget).max()),propagated_state_frobenius_max=float(actual_error.max()),propagated_bound_max=float(prop.max()),readout_maxabs=float(np.abs(read-expected_read).max()),status='PASS'))
 # First prompt q_length1 takes chunk path. Compare its mathematical result to
 # recurrent helper on the identical CPU inputs, allowing analytic rounding.
 args=dict(g=gt[:,:1],beta=bt[:,:1],initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=False)
 co,cs=chunk(nq[:,:1],nk[:,:1],vt[:,:1],**args);ro,rs=native(nq[:,:1],nk[:,:1],vt[:,:1],**args)
 chunkdiff=dict(state_maxabs=float((cs-rs).abs().max()),readout_maxabs=float((co-ro).abs().max()))
 assert torch.allclose(cs,rs,atol=gamma(1024),rtol=gamma(1024)) and torch.allclose(co,ro,atol=gamma(1024),rtol=gamma(1024))
 put('qualification/ACTUAL_TORCH_HELPER_CPU_ARITHMETIC_REFERENCE.json',dict(status='PASS_CPU_ONLY128_STEP_LOCAL_AND_PROPAGATED_BOUND',native_source_sha256=sha(R/'reuse/actual_modeling_qwen3_5.py'),trajectory=trajectory,first_token_chunk_vs_recurrent=chunkdiff,alpha_reference='actual FP32 exp(g) promoted to FP64; CUDA exp error NOT established',fixed_Q_K_gate_inputs=True,full_model_9B=False,CUDA_kernel_qualified=False,BF16_norm='actual source normalization before float32 cast',no_new_Query_calibration=True))

 # Recover exactly what was persisted at the old first *gate failure*.
 fm=read(R/'evidence/token22/FIRST_FULL_MODEL_FP_DIFFERENCE.json');blob=(R/'evidence/token22/FIRST_FULL_MODEL_FP_DIFFERENCE.bin').read_bytes();assert hashlib.sha256(blob).hexdigest()==fm['sha256'] and len(blob)==fm['bytes']
 arrays={f['name']:np.frombuffer(blob,dtype='<f8',count=f['bytes']//8,offset=f['offset']).reshape(f['shape']) for f in fm['tensors']}
 actual=arrays['actual'];baseline=arrays['expected_native'];d=np.abs(actual-baseline);idx=np.unravel_index(d.argmax(),d.shape);raw=read(R/'evidence/token22/FP_ROTATION_FIRST_FAILURE.json');h=raw['original_H_noise'];threshold=max(4*h['maxabs'],.02*float(np.abs(baseline).max()),1e-12)
 # Conditional BF16-grid reconstruction of the backprojected Key. The original
 # pre-rotation Key was not saved, so this is not silently labelled recovered raw.
 candidate=torch.tensor(actual).to(torch.bfloat16).double().numpy();grid_error=np.abs(actual-candidate)
 A64,Af=matrices[(14,9)];phead=int(idx[0]);Uall=np.load(R/'assets/calibration/rotations/layer_14.npz')
 roundtrip=[]
 for hidx in range(32):
  A=Uall['A_fp32_unique'][hidx//2].astype(np.float64);x=candidate[hidx]
  # Structural A32 nonorthogonality plus conventional non-TF32 dot-product
  # rounding. This is consistency of candidate with the saved inverse tensor,
  # not proof of every missing upstream intermediate or CUDA reduction path.
  abserr=np.abs(A@A.T-np.eye(128))@np.abs(x)+gamma(128)*(np.abs(A)@(np.abs(A.T)@np.abs(x)))+gamma(256,U64)*(1+np.abs(x))
  roundtrip.append(dict(recurrent_head=hidx,unique_head=hidx//2,max_distance_to_BF16_grid=float(grid_error[hidx].max()),max_candidate_roundtrip_budget=float(abserr.max()),all_consistent_with_budget=bool(np.all(grid_error[hidx]<=abserr))))
 current=read(R/'evidence/token22/CURRENT_STATE_FIRST_FAILURE.json');cb=(R/'evidence/token22/CURRENT_STATE_FIRST_FAILURE.bin').read_bytes();assert hashlib.sha256(cb).hexdigest()==current['sha256'] and len(cb)==current['bytes']
 currentstate=[]
 for z in current['tensors']:
  a=np.frombuffer(cb,dtype='<f4',count=z['bytes']//4,offset=z['offset']).reshape(z['shape']);currentstate.append(dict(layer=int(z['name']),shape=z['shape'],all_finite=bool(np.isfinite(a).all()),absolute_max=float(np.abs(a).max())))
 token=dict(status='FIRST_FAILURE_REPRODUCED_ROOT_CAUSE_NOT_UNIQUELY_RESOLVED',token_zero_based=22,token_ordinal=23,field='k/14',GDN_layer_zero_based=14,recurrent_head_at_max=phead,unique_shared_Query_Key_head=phead//2,max_index=list(map(int,idx)),saved_dtype='torch.float64',semantic_comparison='Both effective Keys backprojected to native coordinates; original normalize occurs in BF16, effective rotated Key FP32, comparison backprojection FP64',actual_at_max=float(actual[idx]),native_at_max=float(baseline[idx]),maxabs=float(d.max()),relative_L2=rel(actual,baseline),original_H=raw['original_H_noise'],original_threshold=threshold,threshold_excess=float(d.max()-threshold),original_failure_stays_FAIL=True,percentile_abs={str(q):float(np.quantile(d,q)) for q in [0,.5,.9,.95,.99,1]},elements_abs_above_one_over256=int((d>1/256).sum()),elements_total=d.size,candidate_BF16_at_max=float(candidate[idx]),native_BF16_at_max=float(baseline[idx]),candidate_difference_at_max=float(candidate[idx]-baseline[idx]),BF16_step_at_this_magnitude=1/256,steps_at_max=float((candidate[idx]-baseline[idx])/(1/256)),saved_backprojection_offset_from_candidate_at_max=float(actual[idx]-candidate[idx]),roundtrip_consistency=roundtrip,current_state_metadata=current,current_state_checks=currentstate,evidence_gaps=['Native/H matching recurrent states at token22 absent','Upstream pre/post-conv Q/K and hidden states absent','Actual normalized UH Key before rotation absent; nearestBF16 reconstruction is conditional','g/beta/value and core readout tensors at failure absent','Full logits and per-token complete error trajectory not persisted','Earliest gate violation known; earliest nonzero difference unknown','Tokens23..127 target FP run never executed'],interpretation='The roughly six BF16 steps in the Key cannot be reduced to the5.07e-8 threshold excess. Small backprojection noise explains proximity to theBF16 lattice, but cannot identify which preceding layer/token produced the full-model drift. FP32 rounding alone is NOT proved as the unique functional root cause.',GPU_replays=0)
 put('TOKEN22_FIRST_DIFFERENCE_ANALYSIS.json',token)
 put('analysis/CPU_ATTRIBUTION_RESULT.json',dict(status='PASS_STRUCTURAL_MATRIX_AND_FIXED_INPUT_REFERENCE_FUNCTIONAL_ADMISSION_UNRESOLVED',matrix_heads=384,CPU_local_steps=128,FP64_continuous_cases=3,matrix_failures=failures,token22_unique_cause='UNRESOLVED',full9B_128_run='NOT_RUN',GPU_dispatches=0,new_AIME=0,monotonic=time.monotonic()))
 print(json.dumps(dict(status='CPU_ATTRIBUTION_COMPLETE',matrix_heads=384,matrix_failures=failures,token22_max_head=phead,token22_max_unique_head=phead//2,token22_candidate_BF16_steps=token['steps_at_max'],all_candidate_backprojections_consistent=all(x['all_consistent_with_budget'] for x in roundtrip),full9B_GPU_qualified=False)),flush=True)
except BaseException as e:
 if not (R/'analysis/CPU_EVIDENCE_COMPLETION_FAILURE.json').exists():put('analysis/CPU_EVIDENCE_COMPLETION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),GPU_dispatches=0))
 raise
