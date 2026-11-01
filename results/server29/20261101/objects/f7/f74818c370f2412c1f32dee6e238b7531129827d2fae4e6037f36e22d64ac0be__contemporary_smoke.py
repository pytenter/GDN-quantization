"""Prospectively frozen, current-runtime-only exact repeatability and one-step smoke.

All numerical training operations delegate to immutable historical functions.
The FP/H forward wrappers and QDQ delegate to immutable Phase2 functions.
No historical numerical outcome is read or compared.
"""
from common import *
from recovery_control import active_stop,scientific_stop
import argparse
import random
import traceback
import torch

def safe_tensor(x):
    if x.is_floating_point():assert torch.isfinite(x).all().item()
    # Serialization only: flatten permits zero-dimensional Adam step counters.
    # Tensor values, dtype, byte order and the recorded original shape are unchanged.
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=tensor_sha(x.reshape(-1)))

def base_identity(model):
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    return {n:safe_tensor(p) for n,p in model.named_parameters()}

def adapt(t,cayley,condition):
    original=cayley.PerLayerCayleyRotations
    class MatchedBank(original):
        def __init__(self,layers):
            super().__init__(layers)
            if condition=='L8':
                for l in REGIONS['early']:self.layer(l).theta.requires_grad_(False)
        def parameters(self,recurse=True):
            return (p for p in super().parameters(recurse=recurse) if p.requires_grad)
    cayley.PerLayerCayleyRotations=MatchedBank
    if condition=='L8':
        def final(bank,h,layers):
            return {l:h if l in REGIONS['early'] else h.matmul(bank.layer(l).matrix()) for l in layers}
        t.final_rotations=final

def bank_identity(bank):
    opt={id(p) for p in bank.parameters()}
    return {str(l):dict(**safe_tensor(bank.layer(l).theta),requires_grad=bank.layer(l).theta.requires_grad,
                      in_optimizer_list=id(bank.layer(l).theta) in opt) for l in LAYERS}

def flags():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        CUBLAS_WORKSPACE_CONFIG=os.getenv('CUBLAS_WORKSPACE_CONFIG'),PYTHONHASHSEED=os.getenv('PYTHONHASHSEED'),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],torch_seed=torch.initial_seed())

def fp_h_forward(full,model,h,ids):
    p=PARENT/'scripts/phase2_final_hidden_512_smoke.py'
    assert sha(p)=='74c1d06bb3eb27b1adb4e3ae566616d3f76b85388f643eba876d605e069375c4'
    p2=import_file(p,'v3_immutable_phase2_forward_components')
    probe=p2.LayerwiseProbe(full.DECAY);assert probe.install(model)==LAYERS
    cap=p2.BoundaryCapture(model);device=full.model_input_device(model);result={}
    try:
        for cond in ['FP','H']:
            cache=None;rows=[]
            for i,token in enumerate(ids):
                cap.clear();cap.enabled=True
                probe.begin(cond,'native' if cond=='FP' else 'rotated',None if cond=='FP' else {l:h for l in LAYERS})
                try:
                    with torch.inference_mode():
                        out=model(input_ids=torch.tensor([[token]],device=device),
                            attention_mask=torch.ones((1,i+1),device=device,dtype=torch.long),
                            past_key_values=cache,cache_position=torch.tensor([i],device=device),use_cache=True)
                finally:probe.end()
                cache=out.past_key_values
                quant=None
                if cond=='H':quant,_=p2.qdq_cache(cache,False)
                assert cap.final_hidden_gpu is not None and set(cap.local_gpu)==set(LAYERS)
                rows.append(dict(token_position=i,token_id=token,logits=safe_tensor(out.logits),
                    final_hidden=safe_tensor(cap.final_hidden_gpu),local={str(l):safe_tensor(v) for l,v in cap.local_gpu.items()},
                    recurrent_states={str(l):safe_tensor(p2.get_state(cache,l)) for l in LAYERS},qdq=quant))
                del out
            result[cond]=rows
            del cache;cap.clear();cap.enabled=False
    finally:cap.close();probe.close()
    return result

def main(gpu,label,condition):
    assert not active_stop()
    assert load(ROOT/'runtime/v3_runtime_identity.json')['gate']=='PASS'
    source_freeze=load(ROOT/'runtime/phase_a_source_freeze.json')
    for r in source_freeze['files']:assert sha(r['path'])==r['sha256']
    assert gpu in [0,1] and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    assert hardware()['gpus'][gpu]['gpu_uuid']==UUIDS[gpu]
    assert condition=='L7R' or (condition=='L8' and gpu==0 and label=='scientific')
    plan=load(ROOT/'protocol/contemporary_determinism_plan.json')
    trace_path=TRACES/plan['input'];assert sha(trace_path)==plan['input_sha256']
    t=trainer();cfg=load(ROOT/'configs'/('l7r.json' if condition=='L7R' else 'l8.json'))
    args=argparse.Namespace(legacy_repo=str(LEGACY),max_memory_gib=22,seed=0)
    random.seed(0);torch.manual_seed(0);os.environ['LING_MODEL_PATH']=str(MODEL)
    full,cayley,ortho,model,tok,training_probe,layers=t.load_context(args)
    training_probe.close()  # End historical hooks before independent FP/H full forwards.
    assert list(layers)==LAYERS and all(p.device.type=='cuda' for p in model.parameters())
    device=full.model_input_device(model);h=ortho.normalized_hadamard(128).to(device).float()
    trace=t.load_trace(trace_path);ids=[int(x) for x in trace['token_ids'][:4]];assert ids==plan['input_token_ids']
    base=base_identity(model)
    # FP/H tests run once in each independent L7R A/B process, not Fresh-B evaluation.
    forwards=fp_h_forward(full,model,h,ids) if condition=='L7R' else None
    adapt(t,cayley,condition)
    bank=cayley.PerLayerCayleyRotations(tuple(layers)).to(device)
    before=bank_identity(bank);rot=t.final_rotations(bank,h,layers)
    for l in layers:
        assert torch.equal(rot[l],h)
        active=condition=='L7R' or l not in REGIONS['early']
        assert before[str(l)]['requires_grad']==before[str(l)]['in_optimizer_list']==active
    initial_rot={str(l):safe_tensor(r) for l,r in rot.items()}
    records={l:t.to_device_record(trace['layer_records'][l],device) for l in layers}
    # Observers capture all fused teacher/student output/state boundaries without changing values.
    calls=[];local=[];qdq=[];orig_fused=t.fused;orig_local=t.local_output;orig_qdq=t.exact_qdq
    def observed_fused(probe,record,state,value,token):
        out,nxt=orig_fused(probe,record,state,value,token)
        calls.append(dict(call=len(calls),token=token,input_state=safe_tensor(state),value=safe_tensor(value),
                          core_output=safe_tensor(out),next_state=safe_tensor(nxt)))
        return out,nxt
    def observed_local(model,layer,core,gate):
        out=orig_local(model,layer,core,gate);local.append(dict(layer=layer,output=safe_tensor(out)));return out
    def observed_qdq(value,ste):
        out=orig_qdq(cayley,value,ste);qdq.append(dict(ste=ste,output=safe_tensor(out.dequant)));return out
    t.fused=observed_fused;t.local_output=observed_local
    t.exact_qdq=lambda c,v,s:observed_qdq(v,s)
    provenance=[]
    teacher,student=t.init_states(trace,records,rot,cayley,True)
    values=t.rollout_chunk(model=model,probe=training_probe,cayley=cayley,bank=bank,h=h,trace=trace,records=records,
        teacher=teacher,student=student,start=0,end=4,objective='L7_RECURRENT_DENSE_FUNCTIONAL',grad=True,provenance=provenance)
    assert all(torch.isfinite(v).item() for v in values[:4])
    values[0].backward();raw={}
    for l in layers:
        grad=bank.layer(l).theta.grad;active=condition=='L7R' or l not in REGIONS['early']
        if active:assert grad is not None and torch.isfinite(grad).all().item() and torch.count_nonzero(grad).item()>0
        else:assert grad is None
        raw[str(l)]=None if grad is None else safe_tensor(grad)
    optimizer=torch.optim.Adam(bank.parameters(),lr=.003,weight_decay=0.)
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all((id(bank.layer(l).theta) in optimizer_ids)==before[str(l)]['in_optimizer_list'] for l in layers)
    gradnorm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.));assert math.isfinite(gradnorm) and gradnorm>0
    clipped={str(l):None if bank.layer(l).theta.grad is None else safe_tensor(bank.layer(l).theta.grad) for l in layers}
    optimizer.step();after=bank_identity(bank);postrot=t.final_rotations(bank,h,layers)
    for l in layers:
        active=condition=='L7R' or l not in REGIONS['early']
        assert (before[str(l)]['sha256']!=after[str(l)]['sha256'])==active
        if not active:assert torch.equal(postrot[l],h) and after[str(l)]['sha256']==before[str(l)]['sha256']
    assert base_identity(model)==base and all(p.grad is None for p in model.parameters())
    writeback=t.assert_provenance(provenance)
    assert writeback['REAL_RECURRENT_WRITEBACK_GATE']==writeback['RECURRENT_STATE_PROVENANCE_GATE']=='PASS'
    orth=t.orthogonality(bank,h,tuple(layers));assert orth['status']=='PASS'
    optstate={str(l):{k:safe_tensor(v) if torch.is_tensor(v) else v for k,v in optimizer.state.get(bank.layer(l).theta,{}).items()} for l in layers}
    boundary=torch.zeros((1,128),device=device);boundary[0,0]=127.;boundary[0,1:5]=torch.tensor([.5,1.5,-.5,-1.5],device=device)
    q=cayley.ling_r128_qdq(boundary);assert torch.equal(q.dequant[0,1:5],torch.tensor([0.,2.,0.,-2.],device=device))
    x=boundary.clone().requires_grad_(True);s=cayley.ling_r128_ste(x)
    assert torch.equal(s.dequant,q.dequant);s.dequant.sum().backward();assert torch.equal(x.grad,torch.ones_like(x))
    numeric=dict(input_token_ids=ids,trace_sha256=sha(trace_path),model_base=base,H128=safe_tensor(h),
        flags=flags(),condition=condition,model_config_raw_sha256=sha(MODEL/'config.json'),
        config_sha256=sha(ROOT/'configs'/('l7r.json' if condition=='L7R' else 'l8.json')),
        FP_H_full_forwards=forwards,parameter_before=before,initial_rotations=initial_rot,
        training_fused_calls=calls,training_local_outputs=local,training_qdq=qdq,
        losses=[float(v.detach().cpu()) for v in values[:4]],raw_gradients=raw,clipped_gradients=clipped,
        global_gradient_norm=gradnorm,parameter_after=after,optimizer_state=optstate,
        final_rotations={str(l):safe_tensor(r) for l,r in postrot.items()},
        final_teacher_state={str(l):safe_tensor(v) for l,v in teacher.items()},final_student_state={str(l):safe_tensor(v) for l,v in student.items()},
        writeback=writeback,provenance=provenance,orthogonality=orth,base_before_after_equal=True,
        quantizer_ties_even=True,STE_forward_exact=True,STE_gradient_identity=True)
    receipt=dict(gate='PASS',physical_gpu=gpu,gpu_uuid=UUIDS[gpu],process_label=label,pid=os.getpid(),condition=condition,
        disposable_optimizer_updates=1,formal_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0,
        numeric=numeric,numeric_sha256=object_sha(numeric),runtime_identity_sha256=sha(ROOT/'runtime/v3_runtime_identity.json'),
        source_freeze_sha256=sha(ROOT/'runtime/phase_a_source_freeze.json'),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    emit(ROOT/f'runtime/gpu{gpu}_{label}_{condition}_smoke.json',receipt)
    print(json.dumps(dict(gate='PASS',physical_gpu=gpu,label=label,condition=condition,numeric_sha256=receipt['numeric_sha256'])),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,required=True);a.add_argument('--label',required=True)
    a.add_argument('--condition',choices=['L7R','L8'],default='L7R');x=a.parse_args()
    try:main(x.gpu,x.label,x.condition)
    except BaseException as e:
        p=ROOT/f'runtime/gpu{x.gpu}_{x.label}_{x.condition}_failure.json'
        if not p.exists():emit(p,dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),physical_gpu=x.gpu,condition=x.condition))
        scientific_stop('CONTEMPORARY_SMOKE',e);raise
