"""Actual preserved nonAIME State diagnostics; lazy GPU/native imports.

This is a finite diagnostic leaf within the admitted GPU1 codec-fields unit.
No method is selected or rejected by proxy quality. No AIME data/model loads,
new trajectories, requests, workers, or finisher dispatches occur here.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time

TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
LAYERS=tuple(i for i in range(32) if i%4!=3)
RAW='/dev/shm/qwen_sym8_g32_wscale_rowbal_calibration_v1'
PROPAGATION_ANCHOR=31
PROPAGATION_STEPS=8


def _require(ok,message):
    if not ok:raise RuntimeError(message)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _read(path):return json.loads(Path(path).read_text())


def _metric_arrays_cpu(source,reconstructed,transport,omega):
    """Independent FP64 diagnostic reductions on actual FP32 codec output."""
    import numpy as np
    x=np.asarray(source,dtype=np.float64);d=np.asarray(reconstructed,dtype=np.float64)
    g=np.asarray(transport,dtype=np.float64);w=np.asarray(omega,dtype=np.float64)
    _require(x.shape==d.shape and x.ndim>=3 and x.shape[-2:]==(128,128),'diagnostic State axes differ')
    _require(g.shape==x.shape[:-1] and w.shape==x.shape[:-1],'diagnostic actual-head transport/omega axes differ')
    _require(all(np.isfinite(z).all() for z in (x,d,g,w)) and np.all(w>0),'nonfinite diagnostic input')
    e=d-x;sse=np.sum(e*e,axis=(-2,-1));xnorm=np.sqrt(np.sum(x*x,axis=(-2,-1)))
    dnorm=np.sqrt(np.sum(d*d,axis=(-2,-1)));dot=np.sum(x*d,axis=(-2,-1))
    denom=xnorm*dnorm;cos=np.divide(dot,denom,out=np.zeros_like(dot),where=denom>0)
    cosine_error=1-np.clip(cos,-1,1);cosine_error=np.where((xnorm==0)&(dnorm==0),0,cosine_error)
    amplitude=np.abs(dnorm-xnorm)/np.maximum(xnorm,1e-30)
    readout=np.sum(e*g[...,None],axis=-2)
    result=dict(state_mse=sse/16384,weighted_state_mse=np.sum(e*e*w[...,None],axis=(-2,-1))/16384,
                one_step_transported_readout_mse=np.mean(readout*readout,axis=-1),
                relative_State_amplitude_error=amplitude,State_direction_one_minus_cosine=cosine_error,
                max_abs_State_error=np.max(np.abs(e),axis=(-2,-1)),
                max_abs_one_step_readout_error=np.max(np.abs(readout),axis=-1))
    _require(all(np.isfinite(v).all() for v in result.values()),'nonfinite diagnostic metric')
    return result


class _Statistics:
    """Exact finite-vector stats for bounded scalar metadata, no State shadow."""
    def __init__(self):self.arrays=[]
    def add(self,value):
        import numpy as np
        x=np.asarray(value,dtype=np.float64).ravel()
        _require(np.isfinite(x).all(),'nonfinite metadata diagnostic')
        self.arrays.append(x.copy())
    def finish(self):
        import numpy as np
        x=np.concatenate(self.arrays) if self.arrays else np.empty((0,),dtype=np.float64)
        if x.size==0:return dict(count=0)
        return dict(count=int(x.size),minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),
                    quantile_probabilities=[0,.01,.1,.5,.9,.99,1],quantiles=np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist(),quantiles_exact_for_all_reported_metadata_values=True)


def _raw_inventory(root):
    """Bind every captured scientific byte before any new diagnostic GPU call."""
    manifest_path=root/'calibration/FP_COLLECTION_MANIFEST.json';manifest=_read(manifest_path)
    _require(manifest['task_id']==TASK and manifest['status']=='PASS_ACTUAL_BOUNDED_NONAIME_FP_COLLECTION','actual FP collection missing')
    _require(manifest['raw_root']==RAW,'raw root differs from fixed collection')
    _require(manifest['source_freeze_sha256']==_sha(root/'CALIBRATION_SOURCE_FREEZE.json'),'PhaseA source manifest changed')
    _require(manifest['candidate_plan_sha256']==_sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),'calibration candidate plan changed')
    exit_receipt=_read(root/'calibration/FP_COLLECTION_EXIT_RELEASE.json')
    _require(exit_receipt['status']=='PASS_ACTUAL_EXIT0_AND_RELEASE' and exit_receipt['worker_identity']==manifest['identity'] and exit_receipt['exit_returncode']==0 and exit_receipt['release']=='PASS','actual FP collector exit/release missing')
    _require(exit_receipt['collection_manifest_sha256']==_sha(manifest_path),'collector exit receipt manifest SHA differs')
    plan=_read(root/'CALIBRATION_CANDIDATE_PLAN.json');docs=plan['new_collection_proposal']['documents']
    _require(len(docs)==5 and [d['role'] for d in docs]==['CAL']*4+['VAL'],'frozen CAL/VAL document roster differs')
    expected_anchor={(d['document_id'],t,l) for d in docs for t in [31,95,159,223] for l in LAYERS}
    expected_drives={(d['document_id'],t,l) for d in docs for t in list(range(32,40))+list(range(96,104))+list(range(160,168))+list(range(224,232)) for l in LAYERS}
    anchors={};drives={};seen=set();total=0
    for row in manifest['files']:
        p=Path(row['path']);identity=(row['kind'],row['relative'])
        _require(identity not in seen,'duplicate raw manifest row');seen.add(identity)
        _require(not p.is_symlink() and p.is_file() and Path(RAW).resolve() in p.resolve().parents,'unsafe raw scientific path')
        _require(p.stat().st_size==row['bytes'] and _sha(p)==row['sha256'],'raw scientific bytes/SHA differ:'+str(p));total+=row['bytes']
        if row['kind']=='FP_STATE_AND_NEXT_TRANSPORT':
            key=(row['document_id'],row['token'],row['layer']);_require(key in expected_anchor and key not in anchors and row['next_token']==row['token']+1,'raw State anchor identity differs');anchors[key]=row
        elif row['kind']=='ACTUAL_FP_GDN_DRIVE':
            key=(row['document_id'],row['token'],row['layer']);_require(key in expected_drives and key not in drives,'raw drive identity differs');drives[key]=row
        else:_require(row['kind'] in ('RAW_CAL_FP64_SUMS','FROZEN_SHARED_TRANSPORTED_OMEGA'),'unrecognized raw scientific file')
    _require(set(anchors)==expected_anchor and set(drives)==expected_drives,'incomplete frozen State/drive coverage')
    return docs,anchors,drives,dict(collection_manifest_sha256=_sha(manifest_path),files_fresh_SHA_verified=len(seen),raw_bytes_fresh_SHA_verified=total,actual_State_files=len(anchors),actual_drive_files=len(drives))


def run(root,condition):
    """Run in the already-admitted GPU1 W/R/RW FIELDS worker; returns evidence."""
    root=Path(root).resolve();_require(root.name==TASK+'_REDUCTION_REPAIR_20261025' and root==Path(__file__).resolve().parents[1],'exact reduction-repair execution root required')
    _require(condition in ('W','R','RW'),'new formal candidate required')
    sys.path.insert(0,str(root/'src'))
    import execution_control as C
    _require(os.environ.get('CUDA_VISIBLE_DEVICES')==C.UUIDS[1],'diagnostics only in admitted physicalGPU1 field worker')
    target=root/'execution'/('GPU1_'+condition+'_FIELDS')
    _require(target.is_dir(),'exact admitted codec-fields unit target missing')
    unit_start=C.read(target/'STARTED.json')
    _require(unit_start['identity']==C.process_identity(os.getpid()),'diagnostic caller is not exact owned fields process')
    docs,anchors,drives,provenance=_raw_inventory(root)
    # Codec-fields workers do not load a model, so reproduce only the original
    # canonical loader's source-path binding before importing its native helper.
    # Importing these constants does not call load_model_and_tokenizer().
    from canonical_h128 import TRANSFORMERS_SRC,STATE_QUANT_SRC
    for path in (str(STATE_QUANT_SRC),str(TRANSFORMERS_SRC)):
        if path not in sys.path:sys.path.insert(0,path)
    import numpy as np
    import torch
    from unified_codec import encode,decode
    _require(torch.cuda.is_available() and torch.cuda.device_count()==1,'actual single CUDA device required')
    C.device(1,os.getpid())
    omega_manifest=_read(root/'TRANSPORTED_QUERY_OMEGA.json');selection=_read(root/'CALIBRATION_PARAMETER_SELECTION.json')
    _require(omega_manifest['layer_ids']==list(LAYERS) and omega_manifest['shape']==[24,32,128] and omega_manifest['AIME_used'] is False,'shared sensitivity identity differs')
    _require(omega_manifest['collection_manifest_sha256']==provenance['collection_manifest_sha256'],'shared sensitivity raw origin differs')
    _require(selection['omega_manifest_sha256']==_sha(root/'TRANSPORTED_QUERY_OMEGA.json') and selection['AIME_used'] is False and selection['VAL_used_for_selection'] is False,'frozen CAL-only parameter selection differs')
    ws=np.asarray(omega_manifest['omega_normalized_fp32'],np.float32)
    _require(ws.shape==(24,32,128) and np.isfinite(ws).all() and np.all(ws>0),'actual shared omega shape/finite differs')
    omega=torch.tensor(ws,device='cuda:0',dtype=torch.float32).contiguous();a=selection['selected']['a'];b=selection['selected']['b']
    native=__import__('transformers.models.qwen3_5.modeling_qwen3_5',fromlist=['x'])
    _require(_sha(native.__file__)=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a','original actual native GDN source changed')
    helper=native.torch_recurrent_gated_delta_rule
    scalar={role:{name:_Statistics() for name in ('r','scale','base_scale','scale_ratio_vs_base','effective_step_ratio_sample_vs_base')} for role in ('CAL','VAL')}
    sums={role:{} for role in ('CAL','VAL')};counts={'CAL':0,'VAL':0};lambda_counts={role:np.zeros(5,dtype=np.int64) for role in counts}
    clipping={role:dict(code_elements=0,saturation_count_actual_codes=0,clipping_count_recomputed_FP32_levels=0,scale_floor_count=0,scale_elements=0) for role in counts}
    metadata=None;started=time.monotonic()
    def arrays(path,keys):
        with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in keys}
    with torch.inference_mode():
        for doc in docs:
            role=doc['role']
            for anchor in [31,95,159,223]:
                loaded=[arrays(anchors[(doc['document_id'],anchor,layer)]['path'],('state_h','transported_next')) for layer in LAYERS]
                x_cpu=np.stack([v['state_h'] for v in loaded]);gamma=np.stack([v['transported_next'] for v in loaded])
                x=torch.tensor(x_cpu,device='cuda:0',dtype=torch.float32).contiguous()
                encoded=encode(x,condition,omega,a=a,b=b);d=decode(encoded)
                base=encode(x,'base',omega,a=a,b=b)
                d_cpu=d.cpu().numpy();metrics=_metric_arrays_cpu(x_cpu,d_cpu,gamma,ws)
                for key,value in metrics.items():sums[role][key]=sums[role].get(key,np.zeros((24,32),np.float64))+value
                counts[role]+=1
                r=encoded.row_factors.cpu().numpy();scales=encoded.scales.cpu().numpy();base_scales=base.scales.cpu().numpy()
                scalar[role]['r'].add(r);scalar[role]['scale'].add(scales);scalar[role]['base_scale'].add(base_scales);scalar[role]['scale_ratio_vs_base'].add(scales/base_scales)
                # Exact extrema/means remain available in scale/r reports. This
                # labelled bounded sample is for effective row-step quantiles.
                # Deterministic64stride in flattened actualKey/Value order.
                effective=(r.reshape(24,32,4,32,1)*scales[...,None,:])/base_scales[...,None,:]
                scalar[role]['effective_step_ratio_sample_vs_base'].add(effective.reshape(-1)[::64])
                choices=encoded.lambda_indices.cpu().numpy();lambda_counts[role]+=np.bincount(choices.reshape(-1),minlength=5)
                codes=encoded.codes.cpu().numpy();levels=(x_cpu/r[...,None]).reshape(24,32,4,32,128)/scales[...,None,:]
                rounded=np.rint(levels.astype(np.float32));clip=clipping[role]
                clip['code_elements']+=int(codes.size);clip['saturation_count_actual_codes']+=int(np.count_nonzero(np.abs(codes.astype(np.int16))==127))
                clip['clipping_count_recomputed_FP32_levels']+=int(np.count_nonzero((rounded<-127)|(rounded>127)))
                clip['scale_floor_count']+=int(np.count_nonzero(scales<=np.float32(1e-12)*np.float32(1/127)));clip['scale_elements']+=int(scales.size)
                if metadata is None:
                    field=lambda t:t.numel()*t.element_size()
                    metadata=dict(FP32_carrier_bytes=field(x),temporary_INT8_codes_bytes=field(encoded.codes),FP32_scales_bytes=field(encoded.scales),static_omega_bytes=field(omega),allocated_dynamic_r_bytes=field(encoded.row_factors),algorithm_requires_dynamic_r=condition in ('R','RW'),lambda_choice_workspace_bytes=field(encoded.lambda_indices),negative_zero_flags_actual_bytes=field(encoded.negative_zero_flags),weighted_loss_workspace_bytes=field(encoded.weighted_losses),temporary_current_FP32_bytes=field(x),temporary_decoded_FP32_bytes=field(d),theoretical_packed_integer_codes_bytes=24*32*128*128,theoretical_latest_dynamic_scale_bytes=24*32*4*128*4,theoretical_latest_RowBal_metadata_bytes=24*32*128*4 if condition in ('R','RW') else 0,theoretical_static_omega_bytes=24*32*128*4,theoretical_negative_zero_bitmap_bytes=24*32*128*128//8,physical_packed_INT8_implemented=False,lambda_indices_required_in_packed_continuation=False)
                C.progress(target,'NONAIME_ACTUAL_STATE_DIAGNOSTIC_COMMITTED',condition=condition,document=doc['input_order'],anchor=anchor,complete_anchor_batches=counts['CAL']+counts['VAL'],actual_production_QDQ=True)
                del x,encoded,base,d,loaded,x_cpu,d_cpu,gamma,effective,levels,rounded
            C.device(1,os.getpid())
        diagnostics={}
        for role in counts:
            diagnostics[role]=dict(anchor_batches=counts[role],actual_layer_head_State_samples=counts[role]*24*32,
                aggregate={k:float((v/counts[role]).mean()) for k,v in sums[role].items()},
                per_layer_head_mean={k:(v/counts[role]).tolist() for k,v in sums[role].items()},
                metadata_distributions={k:v.finish() for k,v in scalar[role].items()},lambda_values=[.60,.75,.875,1.,1.125],lambda_counts=lambda_counts[role].tolist(),clipping_and_saturation=clipping[role])
        # Fixed-input propagation: original actual GDN helper, no model request.
        # Layer identity is represented by independent entries of batch24. No
        # transformation/re-normalization of already captured H operands occurs.
        propagated=[]
        for doc in docs:
            initial=np.stack([arrays(anchors[(doc['document_id'],PROPAGATION_ANCHOR,l)]['path'],('state_h',))['state_h'] for l in LAYERS])
            fp=torch.tensor(initial,device='cuda:0',dtype=torch.float32).contiguous()
            student=decode(encode(fp,condition,omega,a=a,b=b)).clone();steps=[]
            for step in range(1,PROPAGATION_STEPS+1):
                position=PROPAGATION_ANCHOR+step
                loaded=[arrays(drives[(doc['document_id'],position,l)]['path'],('q_H_pre_scale','key_H','value','g','alpha','beta')) for l in LAYERS]
                inputs={key:torch.tensor(np.stack([v[key] for v in loaded]),device='cuda:0',dtype=torch.float32).unsqueeze(1).contiguous() for key in loaded[0]}
                _require(torch.equal(inputs['g'].exp(),inputs['alpha']),'stored actual native alpha does not match current native exp(g) bytes')
                out_fp,fp=helper(inputs['q_H_pre_scale'],inputs['key_H'],inputs['value'],g=inputs['g'],beta=inputs['beta'],initial_state=fp,output_final_state=True,use_qk_l2norm_in_kernel=False)
                out_student,updated=helper(inputs['q_H_pre_scale'],inputs['key_H'],inputs['value'],g=inputs['g'],beta=inputs['beta'],initial_state=student,output_final_state=True,use_qk_l2norm_in_kernel=False)
                student=decode(encode(updated.contiguous(),condition,omega,a=a,b=b)).clone()
                _require(bool(torch.isfinite(out_fp).all()) and bool(torch.isfinite(out_student).all()) and bool(torch.isfinite(fp).all()) and bool(torch.isfinite(student).all()),'nonfinite native fixed-drive propagation')
                err=(out_student-out_fp).double();state_err=(student-fp).double()
                per_head_output=(err*err).mean(-1)[:,0]
                per_head_state=(state_err*state_err).mean((-2,-1))
                per_head_weighted=(state_err*state_err*omega.double()[...,None]).mean((-2,-1))
                steps.append(dict(step=step,actual_next_token=position,readout_before_current_QDQ_MSE=float(per_head_output.mean().item()),State_after_current_QDQ_MSE=float(per_head_state.mean().item()),weighted_State_after_current_QDQ_MSE=float(per_head_weighted.mean().item()),max_abs_readout_error=float(err.abs().max().item()),per_layer_head_readout_mse=per_head_output.cpu().tolist(),per_layer_head_state_mse=per_head_state.cpu().tolist()))
                C.progress(target,'NONAIME_FIXED_DRIVE_PROPAGATION_COMMITTED',condition=condition,document=doc['input_order'],anchor=PROPAGATION_ANCHOR,propagation_steps=step)
                del inputs,loaded,out_fp,out_student,updated,err,state_err
            propagated.append(dict(document_id=doc['document_id'],role=doc['role'],anchor=PROPAGATION_ANCHOR,steps=steps))
            del initial,fp,student
    C.device(1,os.getpid())
    _require(counts=={'CAL':16,'VAL':4},'all480actualState samples were not diagnosed')
    return dict(status='PASS_ACTUAL_NONAIME_DIAGNOSTICS_NO_PROXY_THRESHOLD',task_id=TASK,condition=condition,
        source_sha256=_sha(Path(__file__)),production_codec_sha256=_sha(root/'src/unified_codec.py'),native_helper_sha256=_sha(native.__file__),
        omega_manifest_sha256=_sha(root/'TRANSPORTED_QUERY_OMEGA.json'),parameter_selection_sha256=_sha(root/'CALIBRATION_PARAMETER_SELECTION.json'),
        raw_provenance=provenance,actual_State_files_consumed=480,actual_production_QDQ_state_batches=20,metrics=diagnostics,
        group_axis='Key',group_size=32,layer_ids=list(LAYERS),actual_recurrent_heads=32,metrics_reduction_dtype='independent CPU FP64 over actual FP32 reconstructed State',
        clipping_definition='nearest-even FP32 levels recomputed fromactualcurrentX/r/scales; saturation counts actualINT8codes',
        effective_step_quantile_sampling='fixedflattenedKeyValue stride64; scale andrquantiles useallvalues',
        propagation=dict(status='ACTUAL_ORIGINAL_GDN_HELPER_FROZEN_DRIVE_REPLAY',subset_rule='allfivefrozen docs inoriginalorder; anchor31only; all24layers/all32heads; exactly8nextsteps',actual_helper_calls=5*8*2,independent_layer_entries_per_helper_call=24,FP_start='saved actualunquantizedFPState X_31 inHcoordinates',student_start='actualproductionQDQ(X_31)',readout_timing='nativeupdate thenreadout beforecurrentQDQ; studentState immediatelyQDQandconsumednexthelper',query='capturedpreparedHquery beforecore1/sqrt128; normalizationFalse',inputs='samefrozenactualq/k/value/logg/beta forFP/student, nofutureAIMEquery',scope_limit='fixed-drive localGDNpropagation; notend-to-endmodelorFutureKLranking',documents=propagated),
        storage_and_metadata=metadata,all_three_formal_candidates_retained=True,proxy_quality_admission_thresholds=[],
        AIME_used=False,model_loads=0,new_model_forwards=0,new_AIME_samples=0,actual_GPU_diagnostic=True,
        same_current_exclusive_GPU1=True,wall_seconds=time.monotonic()-started)
