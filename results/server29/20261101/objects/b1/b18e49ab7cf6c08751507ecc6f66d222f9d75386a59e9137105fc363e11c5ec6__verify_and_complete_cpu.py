"""Independent CPU verification and completion of saved-token attribution.

The original CPU analysis and its two tooling failures remain preserved.
No old numerical stage is overwritten, no GPU evaluation is performed.
"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
R=Path(sys.argv[1]).resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def load_json(p):return json.loads(Path(p).read_text())
def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as output:json.dump(value,output,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);output.write('\n')
def gm(n,u=2.**-24):return n*u/(1-n*u)
def read_tensors(name):
 metadata=load_json(R/'evidence/token22'/name)
 raw=(R/'evidence/token22'/Path(name).with_suffix('.bin')).read_bytes()
 assert len(raw)==metadata['bytes'] and hashlib.sha256(raw).hexdigest()==metadata['sha256']
 dtype={'torch.float64':'<f8','torch.float32':'<f4'}
 return metadata,{f['name']:np.frombuffer(raw,dtype=dtype[f['dtype']],count=f['bytes']//np.dtype(dtype[f['dtype']]).itemsize,offset=f['offset']).reshape(f['shape']) for f in metadata['tensors']}
assert not (R/'analysis/INDEPENDENT_CPU_COMPLETION_STARTED.json').exists()
emit('analysis/INDEPENDENT_CPU_COMPLETION_STARTED.json',dict(pid=os.getpid(),monotonic=time.monotonic(),source_sha256=file_hash(__file__),GPU_dispatches=0,original_scientific_thresholds_changed=False))
frozen=load_json(R/'CPU_ATTRIBUTION_SOURCE_FREEZE.json')
for row in frozen['files']:
 f=R/row['path'];assert file_hash(f)==row['sha256'] and f.stat().st_size==row['bytes']
matrix=load_json(R/'M0_M3_FP32_NUMERICAL_ATTRIBUTION.json');assert not matrix['failures'] and len(matrix['rows'])==384
assert load_json(R/'FP64_GDN_MATH_REFERENCE.json')['status'].startswith('PASS')
local=load_json(R/'qualification/ACTUAL_TORCH_HELPER_CPU_ARITHMETIC_REFERENCE.json')
assert len(local['trajectory'])==128 and all(t['status']=='PASS' and t['max_local_componentwise_ratio']<=1 for t in local['trajectory'])
H=np.array([[(-1.)**((i&j).bit_count()) for j in range(128)] for i in range(128)],dtype=np.float64)/math.sqrt(128)
assets=load_json(R/'reuse/CANONICAL_UH_MATRICES_AND_SHA.json');heads=0;max_quadratic_delta=0.;sharing=[]
with np.load(R/'assets/reuse/QUERY_SECOND_MOMENTS_FP64.npz') as moments:
 for layerrow in assets['layers']:
  with np.load(R/'assets'/layerrow['path']) as values:
   mapping=values['recurrent_to_unique'];assert np.array_equal(mapping,np.repeat(np.arange(16),2));c_all=moments[str(layerrow['layer'])]
   for head in range(16):
    c=c_all[2*head];U=values['U_fp64_unique'][head];A=values['A_fp32_unique'][head]
    assert c.tobytes()==c_all[2*head+1].tobytes()
    assert np.array_equal((U@H).astype(np.float32),A)
    assert hashlib.sha256(U.tobytes()).hexdigest()==layerrow['head_matrices'][head]['U64_sha256']
    assert hashlib.sha256(A.tobytes()).hexdigest()==layerrow['head_matrices'][head]['A32_sha256']
    # Independent componentwise quadratic forms, rather than GEMM diagonal.
    d=np.array([np.einsum('i,ij,j->',A[:,j].astype(np.float64),c,A[:,j].astype(np.float64),optimize=False) for j in range(128)])
    expected=matrix['rows'][heads];mu=np.trace(c)/128;den=mu if mu>1e-24 else 1.
    delta=abs(float(np.abs(d-mu).max()/den)-expected['M1']);max_quadratic_delta=max(max_quadratic_delta,delta)
    assert delta<=gm(4096,2.**-53)*max(1.,float(np.linalg.norm(c))/max(abs(mu),1e-300))
    heads+=1
   sharing.append(dict(layer=layerrow['layer'],all16_actual_pairs=True,unique_heads=16))
assert heads==384
metadata,arrays=read_tensors('FIRST_FULL_MODEL_FP_DIFFERENCE.json');actual=arrays['actual'];native=arrays['expected_native'];delta=np.abs(actual-native);position=np.unravel_index(int(delta.argmax()),delta.shape)
original=load_json(R/'evidence/token22/FP_ROTATION_FIRST_FAILURE.json');hn=original['original_H_noise'];bound=max(4*hn['maxabs'],.02*float(np.abs(native).max()),1e-12)
candidate=torch.from_numpy(actual.copy()).to(torch.bfloat16).double().numpy();grid_delta=np.abs(actual-candidate)
rows=[]
with np.load(R/'assets/calibration/rotations/layer_14.npz') as z:
 for head in range(32):
  A=z['A_fp32_unique'][head//2].astype(np.float64);value=candidate[head]
  budget=np.abs(A@A.T-np.eye(128))@np.abs(value)+gm(128)*(np.abs(A)@(np.abs(A.T)@np.abs(value)))+gm(256,2.**-53)*(1+np.abs(value))
  rows.append(dict(recurrent_head=head,unique_head=head//2,max_distance_to_BF16_grid=float(grid_delta[head].max()),maximum_candidate_roundtrip_budget=float(budget.max()),consistent=bool(np.all(grid_delta[head]<=budget))))
state_metadata,state_arrays=read_tensors('CURRENT_STATE_FIRST_FAILURE.json')
states=[dict(layer=int(name),shape=list(a.shape),dtype=str(a.dtype),all_finite=bool(np.isfinite(a).all()),absolute_max=float(np.abs(a).max())) for name,a in state_arrays.items()]
relative=float(np.linalg.norm(actual-native)/np.linalg.norm(native));dmax=float(delta.max())
assert abs(relative-original['actual_metrics']['relative'])<1e-14 and dmax==original['actual_metrics']['maxabs']
analysis=dict(status='FIRST_FAILURE_REPRODUCED_UNIQUE_UPSTREAM_CAUSE_UNRESOLVED',failure_token_zero_based=22,token_ordinal=23,layer_zero_based=14,field='k/14',comparison_shape=list(actual.shape),saved_dtype='float64',normalization_input_dtype='BF16 per actual source; pre-rotation tensor itself was not persisted',rotated_QK_dtype='FP32 per actual source',comparison_backprojection_dtype='FP64',maximum_position=list(map(int,position)),recurrent_head=int(position[0]),unique_Query_Key_head=int(position[0])//2,actual_at_maximum=float(actual[position]),native_at_maximum=float(native[position]),relative_L2=relative,max_absolute=dmax,original_max_absolute_bound=bound,original_threshold_excess=dmax-bound,actual_error_not_threshold_excess=True,old_gate_failure_retained=True,percentile_absolute={str(q):float(np.quantile(delta,q)) for q in [0,.5,.9,.95,.99,1]},maximum_element_fraction_of_squared_L2_error=float(dmax*dmax/np.sum(delta*delta)),elements_above_one_BF16_step_at_maximum_magnitude=int((delta>1/256).sum()),total_elements=delta.size,candidate_BF16_value_at_maximum=float(candidate[position]),native_value_at_maximum=float(native[position]),BF16_step_at_values_around_0_5_to_1=1/256,candidate_difference_in_BF16_steps=float((candidate[position]-native[position])/(1/256)),backprojection_offset_from_candidate_at_maximum=float(actual[position]-candidate[position]),conditional_reconstruction=True,candidate_is_not_saved_pre_rotation_tensor=True,candidate_roundtrip_checks=rows,saved_difference_sha256=metadata['sha256'],saved_current_state_sha256=state_metadata['sha256'],saved_current_state_checks=states,evidence_gaps=['No matching Native/H current recurrent state snapshots','No stored per-token original raw Q/K, normalization inputs, gate/beta/value or layer outputs','No full-vocabulary logits trace','No complete first21-token error trajectory','Earliest threshold crossing is known; first nonzero difference is not known','128-token full-model run never completed'],conclusion='Six BF16 steps in a backprojected effective Key are functional trajectory drift, not merely the5e-8 threshold excess. Candidate-BF16 consistency and small matrix representation residual do not uniquely distinguish earlier readout rounding, upstream hidden-state drift, recurrence error, or an unobserved implementation defect. The actual saved tensors cannot prove complete128-token functional equivalence.',GPU_replays=0,new_Query_calibration=0)
emit('TOKEN22_FIRST_DIFFERENCE_ANALYSIS.json',analysis)
emit('qualification/INDEPENDENT_CPU_VERIFICATION.json',dict(status='PASS_REUSED384HEADS_INDEPENDENT_QUADRATIC_FORMS_AND_SAVED_TENSOR_RECOMPUTATION',frozen_inputs=len(frozen['files']),heads=heads,max_quadratic_disagreement=max_quadratic_delta,shared_heads=sharing,actual_CPU_helper128_steps=len(local['trajectory']),local_CPU_error_ratio_max=max(t['max_local_componentwise_ratio'] for t in local['trajectory']),first_failure_metrics_reproduced=True,full9B128_qualified=False,GPU_dispatches=0))
emit('analysis/CPU_ATTRIBUTION_RESULT.json',dict(status='CPU_ATTRIBUTION_COMPLETE_C3_ACCEPTANCE_RANGE_UNRESOLVED',matrix_heads=384,matrix_failures=[],CPU_local_steps=128,FP64_continuous128_cases=3,token22_unique_cause='UNRESOLVED',full9B_128_run='NOT_RUN',GPU_dispatches=0,new_AIME=0,original_CPU_loader_failures_preserved=['analysis/CPU_ANALYSIS_FAILURE.json','analysis/CPU_EVIDENCE_COMPLETION_FAILURE.json'],no_result_overwritten=True,monotonic=time.monotonic()))
print(json.dumps(dict(status='PASS_INDEPENDENT_CPU_AUDIT',heads=heads,CPU_helper128=True,relative_L2=relative,max_absolute=dmax,maximum_position=list(map(int,position)),candidate_BF16_steps=analysis['candidate_difference_in_BF16_steps'],first_failure_source_UNRESOLVED=True,full9B_GPU_pass=False,GPU_dispatches=0)),flush=True)
