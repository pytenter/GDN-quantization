"""One bounded, once-only actual FP GDN capture. Never invoke on import.

A separate calibration-only permit is required; this script does not dispatch
formal samples, select AIME answers, retry, kill processes, or run a finisher.
Large scientific tensors are preserved under the permit's exact /dev/shm path.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
RAW_ROOT='/dev/shm/qwen_sym8_g32_wscale_rowbal_calibration_v1'
LAYERS=tuple(i for i in range(32) if i%4!=3)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def verify_phase_a_sources(root):
    manifest=root/'CALIBRATION_SOURCE_FREEZE.json'
    rows=read(manifest)['files'];seen=set()
    if not rows:raise RuntimeError('empty calibration source closure')
    for row in rows:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:
            raise RuntimeError('unsafe or duplicate calibration source entry')
        seen.add(str(rel));p=root/rel
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise RuntimeError('frozen calibration source differs:'+str(rel))
    for required in ('calibration/collect_fp_once.py','calibration/calibration_math.py','CALIBRATION_CANDIDATE_PLAN.json'):
        if required not in seen:raise RuntimeError('missing calibration closure entry:'+required)
    return sha(manifest)

def validate_calibration_permit(root,gpu,control):
    """All CPU/read-only admission before driver/model/CUDA initialization."""
    if root.name!=TASK or root.resolve()!=Path(__file__).resolve().parents[1]:
        raise RuntimeError('collector must execute from exact new independent root')
    permit=read(root/'CALIBRATION_PERMIT.json')
    required=dict(task_id=TASK,scope='FP_COLLECTION_ONLY',ALLOW_GPU='YES',physical_gpu=gpu,
                  gpu_uuid=control.UUIDS[gpu],raw_output_root=RAW_ROOT,
                  candidate_plan_sha256=sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),
                  source_freeze_sha256=verify_phase_a_sources(root),
                  collector_source_sha256=sha(Path(__file__)),
                  model_manifest_sha256=sha(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'),
                  human_authorization_sha256=sha(root/'operations/HUMAN_TASK_AUTHORIZATION.json'))
    for key,value in required.items():
        if permit.get(key)!=value:raise RuntimeError('calibration permit mismatch:'+key)
    if gpu not in (1,2,3) or os.environ.get('CUDA_VISIBLE_DEVICES')!=control.UUIDS[gpu]:
        raise RuntimeError('exact authorized physical UUID CVD required')
    plan=read(root/'CALIBRATION_CANDIDATE_PLAN.json')
    if plan['task_id']!=TASK or plan['AIME_used'] is not False:
        raise RuntimeError('nonAIME plan identity differs')
    proposal=plan['new_collection_proposal'];docs=proposal['documents']
    if len(docs)!=5 or proposal['total_tokens']!=1280 or proposal['CAL_next_step_transitions_per_layer_head']!=1020:
        raise RuntimeError('bounded collection size differs')
    ids=set();raws=set()
    for index,doc in enumerate(docs,1):
        if doc['input_order']!=index or doc['role']!=('CAL' if index<5 else 'VAL'):
            raise RuntimeError('document order/split differs')
        if doc['token_range']!=[0,256] or doc['State_anchors']!=[31,95,159,223]:
            raise RuntimeError('pre-frozen token range/anchors differ')
        if doc['document_id'] in ids or doc['raw_document_sha256'] in raws:
            raise RuntimeError('duplicate CAL/VAL document identity')
        ids.add(doc['document_id']);raws.add(doc['raw_document_sha256'])
        inp=root/doc['new_token_file']
        if inp.is_symlink() or sha(inp)!=doc['new_token_file_sha256']:
            raise RuntimeError('frozen nonAIME token bytes differ')
        values=read(inp)
        if len(values)!=256 or not all(type(x)==int and x>=0 for x in values):
            raise RuntimeError('frozen nonAIME token sequence invalid')
        joined=b''.join(x.to_bytes(8,'little',signed=True) for x in values)
        if hashlib.sha256(joined).hexdigest()!=doc['input_ids_int64_le_sha256']:
            raise RuntimeError('token canonical int64 identity differs')
    return permit,plan


def run(root,gpu):
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'calibration'))
    sys.path.insert(0,str(root/'inherited/group32/src'))
    import execution_control as C
    permit,plan=validate_calibration_permit(root,gpu,C)
    target=root/'calibration/execution'
    target.mkdir(parents=True,exist_ok=True)
    identity=C.process_identity(os.getpid())
    C.atomic_new(target/'STARTED.json',dict(identity=identity,monotonic=time.monotonic(),permit_sha256=sha(root/'CALIBRATION_PERMIT.json')))
    raw=Path(permit['raw_output_root'])
    try:
        C.device(gpu);C.model_guard(root);C.verify_external(root);C.storage_guard(root)
        if raw.exists() or raw.is_symlink():raise RuntimeError('raw collection already exists; no retry/reset')
        stat=os.statvfs(raw.parent)
        if stat.f_bavail*stat.f_frsize<100*1024**3:raise RuntimeError('shared RAM below frozen100GiB resource floor')
        raw.mkdir(mode=0o700,exist_ok=False)
        C.atomic_new(raw/'RAW_IDENTITY.json',dict(task_id=TASK,identity=identity,permit_sha256=sha(root/'CALIBRATION_PERMIT.json'),candidate_plan_sha256=sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),scope='new scientific nonAIME FP tensors; preserve until independently copied and SHA verified on29; never auto-delete'))
        C.atomic_new(target/'DRIVER_IDENTITY.json',C.driver_identity(root,gpu))
        import numpy as np
        import torch
        from canonical_h128 import load_model_and_tokenizer,setup_seed,QwenKeyHadamardPatch,get_state
        from calibration_math import OmegaAccumulator,transported_query
        if torch.__version__!='2.5.1+cu121' or torch.version.cuda!='12.1':raise RuntimeError('historical torch/CUDA changed')
        torch.set_num_threads(24);torch.set_num_interop_threads(24)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
        setup_seed(1);C.progress(target,'LOAD_MODEL_NONAIME_FP')
        model,tokenizer=load_model_and_tokenizer()
        if not all(p.dtype==torch.bfloat16 and p.device.type=='cuda' and p.device.index==0 for p in model.parameters()):raise RuntimeError('singleGPU fullBF16 model binding differs')
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('physical GPU runtime differs')
        native=__import__('transformers.models.qwen3_5.modeling_qwen3_5',fromlist=['x'])
        if sha(native.__file__)!='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a':raise RuntimeError('original GDN helper source differs')
        modules=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
        if {i for i,l in enumerate(modules) if getattr(l,'linear_attn',None) is not None}!=set(LAYERS):raise RuntimeError('actual GDN layer map differs')
        for layer in LAYERS:
            attn=modules[layer].linear_attn
            if (attn.num_k_heads,attn.num_v_heads,attn.head_k_dim,attn.head_v_dim)!=(16,32,128,128):raise RuntimeError('actual QK16/recurrent32 head mapping differs')
        pre=C.library_check(root,gpu)
        if pre['status']!='PASS':raise RuntimeError('actual native library preflight failed')
        C.atomic_new(target/'PREFLIGHT_LIBRARY.json',pre)
        h=QwenKeyHadamardPatch(model)
        accumulator=OmegaAccumulator(LAYERS)
        counts={i:0 for i in LAYERS};seen=set();pending={};files=[];saved=[];current_doc=None;token=-1
        def numpy_fp32(tensor):return tensor.detach().float().cpu().numpy().astype(np.float32,copy=True)
        def save_npz(relative,arrays,kind,**metadata):
            p=raw/relative;p.parent.mkdir(parents=True,exist_ok=True)
            if p.is_symlink():raise RuntimeError('scientific output symlink refused')
            with p.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
            files.append(dict(path=str(p),relative=relative,bytes=p.stat().st_size,sha256=sha(p),kind=kind,**metadata))
        def wrapper(fn):
            def call(query,key,value,*args,**kwargs):
                layer=int(h.current_layer)
                if layer not in LAYERS or layer in seen:raise RuntimeError('helper layer count/order differs')
                seen.add(layer);counts[layer]+=1
                if tuple(query.shape)!=(1,1,32,128) or query.dtype!=torch.float32 or tuple(key.shape)!=(1,1,32,128):raise RuntimeError('capture is not actual prepared H32head operands')
                if kwargs.get('use_qk_l2norm_in_kernel') is not False:raise RuntimeError('H prepared operand normalization repeated/missing')
                decay=kwargs.get('g',args[0] if len(args)>0 else None)
                beta=kwargs.get('beta',args[1] if len(args)>1 else None)
                if decay is None or beta is None or tuple(decay.shape)!=(1,1,32) or tuple(beta.shape)!=(1,1,32):raise RuntimeError('actual head-specific g/beta missing')
                q=numpy_fp32(query)[0,0];k=numpy_fp32(key)[0,0]
                lg=numpy_fp32(decay)[0,0];be=numpy_fp32(beta)[0,0]
                q_readout=numpy_fp32(query.float()*(1/math.sqrt(128)))[0,0]
                alpha=numpy_fp32(decay.float().exp())[0,0]
                gamma=transported_query(q_readout,k,lg,be,query_already_scaled=True,actual_alpha=alpha)
                if current_doc['role']=='CAL' and token>=1:accumulator.add(layer,current_doc['document_id'],token,gamma)
                if layer in pending:
                    prior_t,prior_state=pending.pop(layer)
                    if prior_t+1!=token:raise RuntimeError('State/next-step transport misaligned')
                    save_npz('doc_%02d/anchors/t%03d_l%02d.npz'%(current_doc['input_order'],prior_t,layer),
                             dict(state_h=prior_state,transported_next=gamma,q_next_H_pre_scale=q,q_next_readout_H=q_readout,k_next_H=k,g_next=lg,alpha_next=alpha,beta_next=be),
                             'FP_STATE_AND_NEXT_TRANSPORT',document_id=current_doc['document_id'],role=current_doc['role'],layer=layer,token=prior_t,next_token=token)
                if any(a<=token<b for a,b in current_doc['future_drive_capture_ranges']):
                    save_npz('doc_%02d/drives/t%03d_l%02d.npz'%(current_doc['input_order'],token,layer),
                             dict(q_H_pre_scale=q,q_readout_H=q_readout,key_H=k,value=numpy_fp32(value)[0,0],g=lg,alpha=alpha,beta=be),
                             'ACTUAL_FP_GDN_DRIVE',document_id=current_doc['document_id'],role=current_doc['role'],layer=layer,token=token)
                # Actual original native function called exactly once; no modified recurrence.
                return fn(query,key,value,*args,**kwargs)
            return call
        # Install inside the unchanged H128 wrapper: it receives prepared operands.
        for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
            fn=getattr(native,name);saved.append((name,fn));setattr(native,name,wrapper(fn))
        started=time.monotonic();fingerprints=[]
        try:
            with torch.inference_mode(),h:
                for current_doc in plan['new_collection_proposal']['documents']:
                    cache=None;tokens=read(root/current_doc['new_token_file'])
                    for token,input_id in enumerate(tokens):
                        seen.clear()
                        out=model(input_ids=torch.tensor([[input_id]],device='cuda:0'),past_key_values=cache,use_cache=True)
                        cache=out.past_key_values
                        if seen!=set(LAYERS) or not bool(torch.isfinite(out.logits).all()):raise RuntimeError('actual24layer coverage or output finite failed')
                        if token in current_doc['State_anchors']:
                            for layer in LAYERS:
                                state=get_state(cache,layer)
                                if state.dtype!=torch.float32 or tuple(state.shape)!=(1,32,128,128) or not bool(torch.isfinite(state).all()):raise RuntimeError('native FP state binding differs')
                                pending[layer]=(token,numpy_fp32(state)[0])
                            fingerprints.append(dict(document_id=current_doc['document_id'],token=token,state_sha256={str(i):hashlib.sha256(pending[i][1].tobytes()).hexdigest() for i in LAYERS}))
                        if (token+1)%32==0:
                            C.device(gpu,os.getpid());C.storage_guard(root)
                            C.progress(target,'NONAIME_FP_TOKEN_COMMITTED',document=current_doc['input_order'],document_role=current_doc['role'],tokens=token+1,total_document_tokens=256,actual_helper_calls=dict(counts),raw_files=len(files))
                    if pending:raise RuntimeError('unpaired State anchor at document boundary')
                    del out,cache;gc.collect();torch.cuda.empty_cache()
        finally:
            for name,fn in saved:setattr(native,name,fn)
        if set(counts.values())!={1280} or h.calls!=1280*24:raise RuntimeError('actual once-only FP call coverage differs')
        weights,omega_report=accumulator.finalize(expected_count=1020)
        save_npz('OMEGA_RAW_SUM_FP64.npz',{str(i):v for i,v in accumulator.sums.items()},'RAW_CAL_FP64_SUMS')
        save_npz('OMEGA_FP32.npz',{str(i):v for i,v in weights.items()},'FROZEN_SHARED_TRANSPORTED_OMEGA')
        if sum(x['kind']=='FP_STATE_AND_NEXT_TRANSPORT' for x in files)!=20*24 or sum(x['kind']=='ACTUAL_FP_GDN_DRIVE' for x in files)!=5*32*24:
            raise RuntimeError('exact bounded raw science roster differs')
        post=C.library_check(root,gpu)
        if post['status']!='PASS':raise RuntimeError('actual native postflight failed')
        C.model_guard(root);verify_phase_a_sources(root)
        C.atomic_new(target/'POSTFLIGHT_LIBRARY.json',post)
        C.atomic_new(root/'calibration/FP_COLLECTION_MANIFEST.json',dict(status='PASS_ACTUAL_BOUNDED_NONAIME_FP_COLLECTION',task_id=TASK,source_freeze_sha256=sha(root/'CALIBRATION_SOURCE_FREEZE.json'),candidate_plan_sha256=sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),permit_sha256=sha(root/'CALIBRATION_PERMIT.json'),identity=identity,gpu=gpu,gpu_uuid=C.UUIDS[gpu],raw_root=str(raw),files=files,actual_FP_tokens=1280,actual_helper_calls=counts,H128_prepared_calls=h.calls,omega=omega_report,anchor_state_fingerprints=fingerprints,query_head_mapping='actual32 recurrent heads, original16QK repeat_interleave2 before BF16 normalize; no paired-head omega copy',query_scaling='actualHquery /sqrt128 exactly once',alpha='exp(actual log_g)',quantized=False,AIME_used=False,new_AIME_samples=0,raw_preservation='preserve scientific raw inSHM until CPU copy to29 and perSHA verification; never delete automatically',seconds=time.monotonic()-started,EXACT_physical_GPUh='UNRESOLVED'))
        C.atomic_new(root/'TRANSPORTED_QUERY_OMEGA.json',dict(status='PASS_ACTUAL_NONAIME_TRANSPORTED_QUERY_OMEGA',task_id=TASK,layer_ids=list(LAYERS),omega_normalized_fp32=[weights[i].tolist() for i in LAYERS],shape=[24,32,128],dtype='FP32',head_mapping='actual32 recurrent heads with individual g/beta; adjacent Q/K shared only',omega_rules=plan['omega_rules'],source_freeze_sha256=sha(root/'CALIBRATION_SOURCE_FREEZE.json'),collection_manifest_sha256=sha(root/'calibration/FP_COLLECTION_MANIFEST.json'),raw_omega_npz_sha256=sha(raw/'OMEGA_FP32.npz'),CAL_next_step_transitions_per_layer_head=1020,AIME_used=False,VAL_used_for_omega=False))
        C.progress(target,'NONAIME_FP_COLLECTION_COMPLETE_PRESERVED_RAW')
    except BaseException as error:
        failure=dict(status='NONAIME_FP_COLLECTION_FAIL_CLOSED_STOP',error=repr(error),traceback=traceback.format_exc(),identity=identity,raw_output_root=str(raw),preserve_all_partial_raw=True,AIME_samples=0)
        C.atomic_new(target/'FIRST_FAILURE.json',failure)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--gpu',required=True,type=int)
    args=parser.parse_args();run(Path(args.root).resolve(),args.gpu)
