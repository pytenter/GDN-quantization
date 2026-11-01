"""Actual-device codec field qualification; CPU import has no GPU side effects.

run() must only be called by the task's admitted GPU worker. Synthetic cases
are not AIME, model forwards, persistent-next-consumption proof, or sampling
qualification. This module cannot dispatch another worker or repair/retry.
"""
from pathlib import Path
import hashlib
import importlib.util
import math
import struct
import sys

ROW_FACTOR_MAX_ULP=16


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
    # Row factors are positive finite. Integer bit ordering is monotonic.
    return abs(struct.unpack('<I',struct.pack('<f',a))[0]-
               struct.unpack('<I',struct.pack('<f',b))[0])


def _case_values(name):
    if name=='zero_signed':
        return [-0.0 if (i%2) else 0.0 for i in range(16384)]
    if name=='tiny_floor':
        return [((i%97)-48)*1e-17 for i in range(16384)]
    if name=='positive_negative':
        return [(-1 if (i//128)%2 else 1)*(.125+(i%128)/257) for i in range(16384)]
    if name=='structured_sensitive':
        return [math.sin(i*.731)*(.01+(i//128%29)/9) for i in range(16384)]
    raise ValueError(name)


def run(root,target,unit):
    import torch
    root=Path(root)
    codec=_load(root/'src/unified_codec.py','qualified_unified_codec')
    reference=_load(root/'reference/math_reference.py','independent_scalar_codec')
    original=_load(root/'inherited/group32/src/group_codecs.py','original_canonical_group32')
    _check(torch.cuda.is_available(),'admitted CUDA device missing')
    _check(torch.cuda.device_count()==1,'UUID CVD must expose logical cuda0 only')
    condition=unit.get('condition') or unit.get('method')
    _check(condition in ('W','R','RW'),'formal candidate condition required')
    selected=unit.get('row_parameters') or unit.get('selected_parameters')
    if selected is None:
        # Caller may pin the CAL receipt rather than repeat parameters in unit.
        import json
        p=root/'CALIBRATION_PARAMETER_SELECTION.json'
        if not p.exists():p=root/'calibration/CALIBRATION_PARAMETER_SELECTION.json'
        if p.exists():selected=json.loads(p.read_text()).get('selected')
    if condition in ('R','RW'):
        _check(isinstance(selected,dict),'selected CAL parameters missing')
        a,b=float(selected['a']),float(selected['b'])
        _check((a,b) in codec.PARAMETER_GRID,'unknown CAL-selected pair')
    else:a=b=None
    result=[]
    for name in ('zero_signed','tiny_floor','positive_negative','structured_sensitive'):
        x=torch.tensor(_case_values(name),dtype=torch.float32,device='cuda:0').reshape(1,1,128,128)
        omega=torch.tensor([2.**((i%9)-4) for i in range(128)],
                           dtype=torch.float32,device='cuda:0').reshape(1,1,128)
        # Exact old CUDA field/decode regression, including signed zeros.
        old=original.encode(x,'B1')
        base=codec.encode(x,'base',omega)
        for field in ('codes','scales','negative_zero_flags'):
            _check(_bytes(getattr(old,field))==_bytes(getattr(base,field)),
                   name+': original baseline '+field+' byte mismatch')
        _check(_bytes(original.decode(old))==_bytes(codec.decode(base)),
               name+': original baseline decode byte mismatch')
        actual=codec.encode(x,condition,omega,a=a,b=b)
        c=actual.cpu()
        sx=reference.Tensor32((1,1,128,128),x.cpu().reshape(-1).tolist())
        sw=reference.Tensor32((1,128),omega.cpu().reshape(-1).tolist())
        kwargs={}
        max_r_ulp=0
        if condition in ('R','RW'):
            mathematical_r=reference.row_factors(sx,sw,a,b)
            observed_r=c.row_factors.reshape(-1).tolist()
            max_r_ulp=max(_ulp(ar,br) for ar,br in zip(observed_r,mathematical_r.values))
            _check(max_r_ulp<=ROW_FACTOR_MAX_ULP,
                   name+': row factor exceeds preregistered FP32 transcendental ULP bound')
            # After proving r matches its formula within the frozen bound,
            # test encoding/search exactly with those observed FP32 r inputs.
            kwargs['row_factors_override']=reference.Tensor32((1,1,128),observed_r)
        expected=reference.quantize(sx,sw,codec.METHODS[condition],a=a,b=b,**kwargs)
        _check(_bytes(c.codes)==expected.codes.tobytes(),name+': selected signed codes differ')
        _check(_bytes(c.scales)==expected.scales.bytes(),name+': selected scales differ')
        _check(bytes(c.negative_zero_flags.reshape(-1).tolist())==expected.negative_zero,
               name+': signed-zero flags differ')
        _check(bytes(c.lambda_indices.reshape(-1).tolist())==expected.candidate_indices,
               name+': candidate choice/tie differs')
        _check(_bytes(codec.decode(c))==reference.decode(expected).bytes(),
               name+': CPU decode of actual fields differs from independent original-X reference')
        _check(_bytes(codec.decode(actual))==_bytes(codec.decode(c)),
               name+': fused GPU decode differs from independent CPU-field decode')
        _check(actual.zero_points is None,name+': nonzero-point field introduced')
        result.append(dict(case=name,status='PASS',base_fields_exact=True,
            candidate_codes_scales_sign_choice_decode_exact=True,max_row_factor_ulp=max_r_ulp,
            decoded_sha256=hashlib.sha256(_bytes(codec.decode(actual))).hexdigest()))
    # Exercise actual B/H addressing independently of the simple H1 panels.
    # Every batch/head has a different X and a different static omega pattern;
    # neither the quantizer nor the scalar oracle may reuse a paired head.
    values=[];weights=[]
    for identity in range(4):
        values.extend(math.sin((i+identity*17)*.731)*(.01+(i//128%29)/9)*
                      (1+identity*.25)+(identity-1.5)*.003 for i in range(16384))
        weights.extend(2.**(((row+13*identity)%9)-4) for row in range(128))
    x=torch.tensor(values,dtype=torch.float32,device='cuda:0').reshape(2,2,128,128)
    omega=torch.tensor(weights,dtype=torch.float32,device='cuda:0').reshape(2,2,128)
    old=original.encode(x,'B1');base=codec.encode(x,'base',omega)
    for field in ('codes','scales','negative_zero_flags'):
        _check(_bytes(getattr(old,field))==_bytes(getattr(base,field)),
               'B2H2 isolation: original baseline '+field+' byte mismatch')
    _check(_bytes(original.decode(old))==_bytes(codec.decode(base)),
           'B2H2 isolation: original baseline decode byte mismatch')
    actual=codec.encode(x,condition,omega,a=a,b=b);cpu=actual.cpu();isolation=[]
    for batch_index in range(2):
        c=codec.slice_encoded(cpu,batch_index)
        sx=reference.Tensor32((1,2,128,128),x[batch_index:batch_index+1].cpu().reshape(-1).tolist())
        sw=reference.Tensor32((2,128),omega[batch_index].cpu().reshape(-1).tolist())
        kwargs={};max_r_ulp=0
        if condition in ('R','RW'):
            mathematical_r=reference.row_factors(sx,sw,a,b)
            observed_r=c.row_factors.reshape(-1).tolist()
            max_r_ulp=max(_ulp(ar,br) for ar,br in zip(observed_r,mathematical_r.values))
            _check(max_r_ulp<=ROW_FACTOR_MAX_ULP,
                   'B2H2 isolation: row factor exceeds preregistered ULP bound')
            kwargs['row_factors_override']=reference.Tensor32((1,2,128),observed_r)
        expected=reference.quantize(sx,sw,codec.METHODS[condition],a=a,b=b,**kwargs)
        _check(_bytes(c.codes)==expected.codes.tobytes(),'B2H2 isolation: signed codes differ')
        _check(_bytes(c.scales)==expected.scales.bytes(),'B2H2 isolation: scales differ')
        _check(bytes(c.negative_zero_flags.reshape(-1).tolist())==expected.negative_zero,
               'B2H2 isolation: signed-zero flags differ')
        _check(bytes(c.lambda_indices.reshape(-1).tolist())==expected.candidate_indices,
               'B2H2 isolation: candidate choice/tie differs')
        _check(_bytes(codec.decode(c))==reference.decode(expected).bytes(),
               'B2H2 isolation: independent original-X decode differs')
        isolation.append(dict(batch_index=batch_index,actual_heads=[0,1],status='PASS',
            codes_scales_sign_choice_decode_exact=True,max_row_factor_ulp=max_r_ulp,
            distinct_current_state_and_static_omega_per_identity=True))
    _check(_bytes(codec.decode(actual))==_bytes(codec.decode(cpu)),
           'B2H2 isolation: GPU decode differs from independent CPU-field decode')
    result.append(dict(case='batch2_head2_identity_isolation',status='PASS',
        shape=[2,2,128,128],omega_shape=[2,2,128],base_fields_exact=True,
        independent_batch_slices=isolation))
    # Fail-closed invalid input must not produce a returned Encoded object.
    rejected=[]
    for tag,bad in (('nan',float('nan')),('posinf',float('inf')),('neginf',-float('inf'))):
        x=torch.zeros((1,1,128,128),dtype=torch.float32,device='cuda:0');x[0,0,0,0]=bad
        try:codec.encode(x,condition,omega[:1,:1].contiguous(),a=a,b=b)
        except FloatingPointError:rejected.append(tag)
        else:raise AssertionError('invalid '+tag+' was accepted')
    _check(len(rejected)==3,'invalid field cases missing')
    # These admission fields are emitted only after every actual CUDA case
    # and independent scalar-oracle comparison above has succeeded.
    return dict(status='PASS',actual_cuda=True,independent_reference_pass=True,
        condition=condition,cases=result,invalid_values_rejected=rejected,
        original_cuda_baseline_byte_regression=True,
        independent_reference='stdlib scalar FP32; no original codec imports',
        row_factor_max_ulp_preregistered=ROW_FACTOR_MAX_ULP,
        row_factor_rule='transcendental-bound checked independently, then exact codec/search givenactualFP32r',
        scope='actual GPU synthetic codec fields only; no model/persistent/sampler/AIME claims',
        torch_version=torch.__version__,device=torch.cuda.get_device_name(0))
