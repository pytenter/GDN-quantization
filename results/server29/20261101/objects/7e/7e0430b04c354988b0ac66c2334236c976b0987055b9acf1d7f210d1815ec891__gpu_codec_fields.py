"""Actual CUDA field qualification, exact scalar oracle, bounded failure evidence.

Synthetic non-AIME only. Import is CPU-safe; run() is admitted by the original
worker. No retry/dispatch/answer logic lives here. Original byte gates and the
preregistered 16-ULP row-factor bound are retained without relaxation.
"""
from pathlib import Path
import hashlib
import importlib.util
import math
import struct
import sys

ROW_FACTOR_MAX_ULP=16
MAX_DIFFERENCES=24


def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def _bytes(t):
    import torch
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _check(value,message):
    if not value:raise AssertionError(message)


def _ulp(a,b):
    return abs(struct.unpack('<I',struct.pack('<f',a))[0]-
               struct.unpack('<I',struct.pack('<f',b))[0])


def _finite_json(x):
    return x if not isinstance(x,float) or math.isfinite(x) else repr(x)


def _field_summary(e):
    out=dict(method=e.method,shape=list(e.shape))
    for name in ('codes','scales','negative_zero_flags','lambda_indices',
                 'weighted_losses','row_factors','invalid_flags'):
        value=getattr(e,name,None)
        if value is None:continue
        flat=value.detach().cpu().reshape(-1)
        out[name]=dict(shape=list(value.shape),dtype=str(value.dtype),
            bytes=len(_bytes(value)),sha256=hashlib.sha256(_bytes(value)).hexdigest(),
            first_values=[_finite_json(x) for x in flat[:16].tolist()])
    return out


def _exact(actual,expected,name,context):
    raw=_bytes(actual)
    if raw==expected:return
    size=actual.element_size();different=[]
    for i in range(0,max(len(raw),len(expected)),size):
        a=raw[i:i+size];b=expected[i:i+size]
        if a!=b:
            different.append(dict(flat_index=i//size,actual_hex=a.hex(),expected_hex=b.hex()))
            if len(different)>=MAX_DIFFERENCES:break
    context['mismatch']=dict(field=name,actual_bytes=len(raw),expected_bytes=len(expected),
        actual_sha256=hashlib.sha256(raw).hexdigest(),expected_sha256=hashlib.sha256(expected).hexdigest(),
        first_differences=different,first_differences_limit=MAX_DIFFERENCES)
    raise AssertionError(context['case']+': '+name+' bytes differ')


def _row_factor_evidence(observed,mathematical,distances):
    return dict(max_ulp=max(distances),preregistered_bound=ROW_FACTOR_MAX_ULP,
        worst_differences=[dict(row=i,actual=observed[i],reference=mathematical[i],ulp=distances[i]) for i in sorted(range(len(distances)),key=lambda j:distances[j],reverse=True)[:8]],
        first_differences=[dict(row=i,actual=ar,reference=br,ulp=u) for i,(ar,br,u) in enumerate(zip(observed,mathematical,distances)) if u][:MAX_DIFFERENCES])


def _case_values(name):
    if name=='zero_signed':return [-0.0 if i%2 else 0.0 for i in range(16384)]
    if name=='tiny_floor':return [((i%97)-48)*1e-17 for i in range(16384)]
    if name=='positive_negative':return [(-1 if (i//128)%2 else 1)*(.125+(i%128)/257) for i in range(16384)]
    if name=='structured_sensitive':return [math.sin(i*.731)*(.01+(i//128%29)/9) for i in range(16384)]
    raise ValueError(name)


def _candidate_evidence(reference,sx,sw,factors,expected,actual):
    """Bounded five-candidate oracle diagnostics; actual kernel saves selected loss."""
    candidates=[];a_choice=actual.lambda_indices.reshape(-1).tolist()
    a_losses=actual.weighted_losses.reshape(-1).tolist()
    chosen=[]
    for i,(a,b) in enumerate(zip(a_choice,expected.candidate_indices)):
        if a!=b:chosen.append(i)
    if not chosen:
        for i,(a,b) in enumerate(zip(a_losses,expected.losses.values)):
            if struct.pack('<f',a)!=struct.pack('<f',b):chosen.append(i)
    if not chosen:chosen=[0]
    for metadata in chosen[:8]:
        head=metadata//512;group=(metadata%512)//128;column=metadata%128
        rows=[head*16384+(group*32+k)*128+column for k in range(32)]
        x=[sx.values[i] for i in rows]
        r=list(factors.values[head*128+group*32:head*128+group*32+32])
        w=list(sw.values[head*128+group*32:head*128+group*32+32])
        y=[reference.div(v,rf) for v,rf in zip(x,r)]
        s0=reference.canonical_scale(max(map(abs,y)));losses=[]
        for lam in reference.LAMBDAS:
            scale=reference.candidate_scale(s0,lam)
            decoded=[reference.mul(reference.sym8_scalar(v,scale)[2],rf) for v,rf in zip(y,r)]
            diff=[reference.add(v,-d) for v,d in zip(x,decoded)]
            losses.append(reference.tree_sum([reference.mul(ww,reference.mul(d,d)) for ww,d in zip(w,diff)]))
        winner=reference.choose_candidate(losses);order=sorted(range(5),key=lambda i:(losses[i],reference.TIE_ORDER.index(i)))
        candidates.append(dict(head=head,group=group,column=column,
            reference_candidate_losses=losses,reference_candidate_loss_hex=[struct.pack('<f',v).hex() for v in losses],
            reference_five_candidate_argmin=winner,reference_policy_selected_index=int(expected.candidate_indices[metadata]),reference_best_runner_up_margin=reference.add(losses[order[1]],-losses[order[0]]),
            actual_selected_index=int(a_choice[metadata]),actual_selected_loss=_finite_json(a_losses[metadata]),
            actual_all_five_losses_retained=False))
    return candidates


def _verify_panel(root,target,unit,codec,reference,x,omega,a,b,name,context):
    context.clear();context.update(case=name,condition=unit['condition'],shape=list(x.shape),omega_shape=list(omega.shape),selected_parameters=dict(a=a,b=b),phase='BASELINE_BYTE_REGRESSION')
    original=_load(root/'inherited/group32/src/group_codecs.py','original_canonical_group32')
    old=original.encode(x,'B1');base=codec.encode(x,'base',omega)
    context['actual_fields_summary']=_field_summary(base.cpu())
    for field in ('codes','scales','negative_zero_flags'):
        _exact(getattr(base,field),_bytes(getattr(old,field)),'original baseline '+field,context)
    _exact(codec.decode(base),_bytes(original.decode(old)),'original baseline decoded',context)
    context['phase']='NEW_METHOD_SCALAR_ORACLE'
    actual=codec.encode(x,unit['condition'],omega,a=a,b=b);cpu=actual.cpu()
    context['actual_fields_summary']=_field_summary(cpu)
    batch,heads=x.shape[:2];isolated=[];global_ulp=0
    for batch_index in range(batch):
        c=codec.slice_encoded(cpu,batch_index);context['batch_index']=batch_index
        sx=reference.Tensor32((1,heads,128,128),x[batch_index:batch_index+1].cpu().reshape(-1).tolist())
        weights=omega[batch_index] if omega.ndim==3 else omega
        sw=reference.Tensor32((heads,128),weights.cpu().reshape(-1).tolist());kwargs={};max_ulp=0
        if unit['condition'] in ('R','RW'):
            mathematical=reference.row_factors(sx,sw,a,b);observed=c.row_factors.reshape(-1).tolist()
            distances=[_ulp(ar,br) for ar,br in zip(observed,mathematical.values)];max_ulp=max(distances)
            context['row_factor_comparison']=_row_factor_evidence(observed,mathematical.values,distances)
            _check(max_ulp<=ROW_FACTOR_MAX_ULP,name+': row factor exceeds preregistered FP32 transcendental ULP bound')
            factors=reference.Tensor32((1,heads,128),observed);kwargs['row_factors_override']=factors
        else:factors=reference.Tensor32((1,heads,128),[1.]*(heads*128))
        expected=reference.quantize(sx,sw,codec.METHODS[unit['condition']],a=a,b=b,**kwargs)
        try:
            _exact(c.codes,expected.codes.tobytes(),'selected signed codes',context)
            _exact(c.scales,expected.scales.bytes(),'selected scales',context)
            _exact(c.negative_zero_flags,expected.negative_zero,'signed-zero flags',context)
            _exact(c.lambda_indices,expected.candidate_indices,'candidate choice/tie',context)
            # New stronger gate: identical winner alone must not conceal an
            # incorrect reduction of the selected original-X objective.
            _exact(c.weighted_losses,expected.losses.bytes(),'selected weighted losses',context)
            _exact(codec.decode(c),reference.decode(expected).bytes(),'CPU decode of actual fields',context)
        except AssertionError:
            context['candidate_loss_evidence']=_candidate_evidence(reference,sx,sw,factors,expected,c)
            raise
        global_ulp=max(global_ulp,max_ulp)
        isolated.append(dict(batch_index=batch_index,actual_heads=list(range(heads)),
            status='PASS',codes_scales_sign_choice_selected_loss_decode_exact=True,max_row_factor_ulp=max_ulp))
    _exact(codec.decode(actual),_bytes(codec.decode(cpu)),'GPU decode vs CPU-field decode',context)
    _check(actual.zero_points is None,name+': nonzero-point field introduced')
    return dict(case=name,status='PASS',shape=list(x.shape),omega_shape=list(omega.shape),
        base_fields_exact=True,codes_scales_sign_choice_selected_loss_decode_exact=True,
        independent_batch_slices=isolated,max_row_factor_ulp=global_ulp,
        decoded_sha256=hashlib.sha256(_bytes(codec.decode(actual))).hexdigest())


def _run(root,target,unit,context):
    import torch
    root=Path(root);codec=_load(root/'src/unified_codec.py','qualified_unified_codec')
    reference=_load(root/'reference/math_reference.py','independent_scalar_codec')
    panels=_load(root/'reference/reduction_repair_cases.py','frozen_reduction_repair_cases')
    _check(torch.cuda.is_available(),'admitted CUDA device missing')
    _check(torch.cuda.device_count()==1,'UUID CVD must expose logical cuda0 only')
    condition=unit.get('condition') or unit.get('method');_check(condition in ('W','R','RW'),'formal candidate condition required')
    unit=dict(unit,condition=condition);selected=unit.get('row_parameters') or unit.get('selected_parameters')
    if selected is None:
        import json
        p=root/'CALIBRATION_PARAMETER_SELECTION.json'
        if not p.exists():p=root/'calibration/CALIBRATION_PARAMETER_SELECTION.json'
        if p.exists():selected=json.loads(p.read_text()).get('selected')
    if condition in ('R','RW'):
        _check(isinstance(selected,dict),'selected CAL parameters missing');a,b=float(selected['a']),float(selected['b'])
        _check((a,b) in codec.PARAMETER_GRID,'unknown CAL-selected pair')
    else:a=b=None
    old_panels=[(n,_case_values(n),[2.**((i%9)-4) for i in range(128)]) for n in ('zero_signed','tiny_floor','positive_negative','structured_sensitive')]
    result=[]
    for name,values,weights in old_panels+panels.additional_panels():
        x=torch.tensor(values,dtype=torch.float32,device='cuda:0').reshape(1,1,128,128)
        omega=torch.tensor(weights,dtype=torch.float32,device='cuda:0').reshape(1,1,128)
        result.append(_verify_panel(root,target,unit,codec,reference,x,omega,a,b,name,context))
    values=[];weights=[]
    for identity in range(4):
        values.extend(math.sin((i+identity*17)*.731)*(.01+(i//128%29)/9)*(1+identity*.25)+(identity-1.5)*.003 for i in range(16384))
        weights.extend(2.**(((row+13*identity)%9)-4) for row in range(128))
    x=torch.tensor(values,dtype=torch.float32,device='cuda:0').reshape(2,2,128,128)
    omega=torch.tensor(weights,dtype=torch.float32,device='cuda:0').reshape(2,2,128)
    result.append(_verify_panel(root,target,unit,codec,reference,x,omega,a,b,'batch2_head2_batched_omega_identity_isolation',context))
    shared=omega[0].contiguous()
    result.append(_verify_panel(root,target,unit,codec,reference,x,shared,a,b,'batch2_head2_shared_omega_identity_isolation',context))
    rejected=[];context.clear();context.update(case='INVALID_INPUT_REJECTION',condition=condition)
    for tag,bad in (('nan',float('nan')),('posinf',float('inf')),('neginf',-float('inf'))):
        x=torch.zeros((1,1,128,128),dtype=torch.float32,device='cuda:0');x[0,0,0,0]=bad
        try:codec.encode(x,condition,shared[:1].contiguous(),a=a,b=b)
        except FloatingPointError:rejected.append(tag)
        else:raise AssertionError('invalid '+tag+' was accepted')
    _check(len(rejected)==3,'invalid field cases missing')
    return dict(status='PASS',actual_cuda=True,independent_reference_pass=True,condition=condition,
        cases=result,invalid_values_rejected=rejected,original_cuda_baseline_byte_regression=True,
        independent_reference='unchanged stdlib scalar FP32; no original codec imports',
        row_factor_max_ulp_preregistered=ROW_FACTOR_MAX_ULP,
        row_factor_rule='unchanged16ULP transcendental bound, then exact codec/search/selectedloss givenactualFP32r',
        fixed_halves_tree_sensitive_groups=4,fixed_halves_tree_sensitive_column_tiles=8,
        candidate_loss_selected_bytes_exact=True,both_omega_layouts_qualified=True,
        scope='actual GPU synthetic fields only; no model/persistent/sampler/AIME claims',
        torch_version=torch.__version__,device=torch.cuda.get_device_name(0))


def run(root,target,unit):
    context={}
    try:return _run(Path(root),Path(target),unit,context)
    except BaseException as exc:
        import execution_control as C
        path=Path(target)/'CODEC_FIELDS_FIRST_FAILURE_SUMMARY.json'
        if not path.exists():
            C.atomic_new(path,dict(status='ACTUAL_GPU_FIELD_FAILURE_BOUNDED_DIAGNOSTIC',
                unit=unit['id'],condition=unit.get('condition'),error=repr(exc),
                evidence=context,raw_model_or_tensor_dump=False,maximum_examples_per_field=MAX_DIFFERENCES,
                no_retry=True,scientific_credit=False))
        raise
