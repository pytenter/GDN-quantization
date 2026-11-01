import hashlib
import torch
from dual_axis_codec import encode,EncodedState,NAMES,safe_half,LOW,HIGH

def run(panel):
    checks=[]
    def ok(test,detail):checks.append(dict(test=test,detail=detail,status='PASS'))
    for name,x in panel.items():
        for codec in NAMES:
            original=x.clone();e=encode(x,codec);ref=e.decode();blob=e.to_bytes()
            x.fill_(42.)  # destroy source; encoded storage must be independent
            recovered=EncodedState.from_bytes(blob)
            assert torch.equal(ref,recovered.decode()),(name,codec,'roundtrip')
            assert len(blob)==e.byte_audit()['total_physical_payload_bytes']
            assert set(vars(e))=={'codec','shape','codes','columns','rows'}
            assert e.codes.data_ptr()!=x.data_ptr()
            for bad in (blob[:-1],blob+b'0',b'BAD!'+blob[4:]):
                try:EncodedState.from_bytes(bad)
                except ValueError:pass
                else:raise AssertionError('Malformed payload accepted')
            x.copy_(original)
            ok('roundtrip_owned_storage',{'case':name,'codec':codec,'bytes':len(blob),'wire_sha256':hashlib.sha256(blob).hexdigest()})
    for codec in NAMES:
        for value in (torch.zeros(2,3),torch.zeros(1,0,128,128),torch.zeros(1,1,128,128,dtype=torch.int32),torch.full((1,1,128,128),float('nan')),torch.full((1,1,128,128),float('inf'))):
            try:encode(value,codec)
            except ValueError:pass
            else:raise AssertionError((codec,'Invalid input accepted'))
        for label,magnitude in (('tiny',1e-30),('large',1e10),('fp32_max',torch.finfo(torch.float32).max)):
            x=torch.full((2,3,128,128),magnitude)
            x[...,::2,:]*=-1
            if codec==NAMES[0] and label=='fp32_max':
                try:encode(x,codec)
                except FloatingPointError as exc:
                    assert str(exc)=='nonfinite C128 dequant'
                    ok('deterministic_scale_stress',dict(codec=codec,case=label,behavior='fail-closed; same nonfinite guard as unchanged formal writeback'))
                    continue
                raise AssertionError('Canonical FP32_MAX must fail closed as formal reference does')
            e=encode(x,codec);assert torch.isfinite(e.decode()).all()
            ok('deterministic_scale_stress',dict(codec=codec,case=label,columns_min=e.columns.min().item(),columns_max=e.columns.max().item(),rows_min=None if e.rows is None else e.rows.min().item(),rows_max=None if e.rows is None else e.rows.max().item(),saturation_fraction=(e.codes.abs()==127).float().mean().item()))
    actual=safe_half(torch.tensor([0.,1e-40,LOW,HIGH,1e30,float('inf')]))
    assert torch.equal(actual,torch.tensor([LOW,LOW,LOW,HIGH,HIGH,HIGH],dtype=torch.float16))
    ok('half_underflow_overflow_bounds',{'lower':LOW,'upper':HIGH})
    return checks
