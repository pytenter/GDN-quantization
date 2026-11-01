"""Bounded failure-only scale context; no model/prompt/generated-text access.

Ordinary execution does not call this module's tensor summaries. Exact known
range errors are summarized from this execution's quantizer traceback only.
Diagnostic trouble never replaces the original scientific exception.
"""
import math
from pathlib import Path

RANGE_ERRORS = frozenset({
    'extended scale requires finite raw affine scales',
    'INT8 scale exponent exceeds supported range',
    'extended FP16 mantissa remains invalid',
    'extended scale is not positive finite FP32',
    'extended decoded FP32 carrier is nonfinite',
})


def _number(value):
    value=float(value)
    if math.isnan(value):return 'NaN'
    if math.isinf(value):return '+Infinity' if value>0 else '-Infinity'
    return value


def _extended_numeric(error, frames):
    # Import only after exact error text and exact current-source traceback gates.
    import torch
    available={}
    for frame in frames:available.update(frame.f_locals)
    groups=available.get('groups');owner=available.get('self')
    bits=available.get('bits',getattr(owner,'bits',None))
    raw=available.get('raw_scale');lo=available.get('lo');hi=available.get('hi')
    mantissa=available.get('effective',getattr(owner,'scales',None))
    exponents=available.get('scale_exponents',getattr(owner,'scale_exponents',None))
    candidate=available.get('chosen',available.get('tentative'))
    effective32=available.get('scale32',available.get('effective32'))
    if error=='extended decoded FP32 carrier is nonfinite':
        mask=(~torch.isfinite(available['values'])).any(dim=-1)
    elif error=='extended scale is not positive finite FP32':
        mask=~torch.isfinite(effective32)|(effective32<=0)
    elif error=='INT8 scale exponent exceeds supported range':
        mask=(candidate>127)&available['invalid']
    elif error=='extended scale requires finite raw affine scales':
        mask=~torch.isfinite(raw)
    else:
        mask=~torch.isfinite(mantissa)|(mantissa<=0)
    indices=([[]] if bool(mask.item()) else []) if mask.ndim==0 else mask.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
    rows=[]
    tensors=dict(lo=lo,hi=hi,raw_scale=raw,FP16_mantissa=mantissa,
                 INT8_exponent=exponents,candidate_exponent=candidate,effective_FP32_scale=effective32)
    for index in indices:
        item=dict(index=index)
        for name,tensor in tensors.items():
            if tensor is not None:
                value=tensor[tuple(index)].item()
                item[name]=int(value) if name in('INT8_exponent','candidate_exponent') else _number(value)
        rows.append(item)
    return dict(status='ACTUAL_EXPONENT_FORMAT_RANGE_GUARD_FAILURE',error=error,bits=int(bits),
                group_shape=list(groups.shape)if groups is not None else list(available['values'].shape),
                input_dtype=str(groups.dtype)if groups is not None else 'UNRESOLVED_NO_INPUT_FRAME',
                input_classification='FINITE_FLOATING_INPUT_ALREADY_VALIDATED'if groups is not None else 'UNRESOLVED_NO_INPUT_FRAME',
                first_bad_group_indices=indices,groups=rows,bad_group_count=int(mask.sum().item()),
                diagnostic_group_limit=8,diagnostic_only_no_recovery=True)


def scale_failure_context(exc, root, stage, unit, identity):
    source=Path(root).resolve()/'src'
    all_frames=[];quantizer_frames=[];tb=exc.__traceback__
    while tb is not None:
        frame=tb.tb_frame;all_frames.append(frame)
        if Path(frame.f_code.co_filename).resolve()==source/'pace_quantizer.py':
            quantizer_frames.append(frame)
        tb=tb.tb_next
    numeric=getattr(exc,'diagnostics',None)
    legacy=type(exc).__name__=='FP16ScaleOverflowError'and isinstance(numeric,dict)
    extended=type(exc)is ValueError and str(exc)in RANGE_ERRORS and any(
        f.f_code.co_name in('from_tensor','dequantize')for f in quantizer_frames)
    if not legacy and not extended:return None
    if extended:
        try:numeric=_extended_numeric(str(exc),quantizer_frames)
        except BaseException as diagnostic_error:
            numeric=dict(status='RANGE_DIAGNOSTIC_UNRESOLVED_ORIGINAL_FAILURE_PRESERVED',
                         error=str(exc),diagnostic_error=repr(diagnostic_error),
                         first_bad_group_indices=[],groups=[],diagnostic_only_no_recovery=True)
    context=dict(status='ACTUAL_UNREPRESENTABLE_STORED_FP16_SCALE'if legacy else'ACTUAL_EXPONENT_FORMAT_RANGE_GUARD_FAILURE',
                 stage=stage,unit=unit,identity=identity,numeric=numeric,formal_completion_credit=0,
                 no_parameter_change=True,tensors_or_prompt_or_generated_text_recorded=False)
    heads=None
    for frame in all_frames:
        path=Path(frame.f_code.co_filename).resolve();name=frame.f_code.co_name;values=frame.f_locals
        if path==source/'pace_quantizer.py'and name=='encode':
            selected=values.get('heads')
            if isinstance(selected,tuple)and all(type(x)is int for x in selected):
                heads=selected;context['state_selected_heads']=list(selected)
        if path==source/'pace_buffer.py'and name=='_commit':
            context['role']='RECURRENT_STATE_WRITEBACK'
            if isinstance(values.get('reason'),str):context['writeback_reason']=values['reason']
        if path==source/'pace_runtime.py'and name=='call':
            if type(values.get('layer'))is int:context['layer']=values['layer']
            buffer=values.get('buffer')
            if buffer is not None:
                context['buffer_counters']={name:getattr(buffer,name)for name in
                    ('tokens','prompt_length','pending_tokens','decode_period','prefill_period','quantization_count','writeback_count')
                    if type(getattr(buffer,name,None))is int}
    context.setdefault('role','AFFINE_GROUP_ENCODING')
    context['state_group_locations']=[]
    if heads is not None:
        for index in numeric.get('first_bad_group_indices',[]):
            if isinstance(index,list)and len(index)==4 and all(type(x)is int for x in index)and 0<=index[1]<len(heads):
                context['state_group_locations'].append(dict(batch=index[0],head=heads[index[1]],value_column=index[2],key_group=index[3]))
    return context
