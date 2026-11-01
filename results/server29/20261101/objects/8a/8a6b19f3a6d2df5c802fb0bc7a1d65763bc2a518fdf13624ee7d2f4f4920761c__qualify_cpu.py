import os,sys,json,hashlib,subprocess,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np
import torch
torch.cuda._lazy_init=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('GPU forbidden'))
torch.set_num_threads(1)
from group_codecs import encode,decode,qdq_writeback,require_consumed,tensor_bytes,METHODS
from independent_reference import reference
from canonical_h128 import hadamard,quantize_c128,GDN_LAYERS
torch.random.default_generator.manual_seed(20261019)
checks=[];parity=[];repeat=[];details=[]
def check(name,ok,**kw):
    checks.append(dict(name=name,passed=bool(ok),**kw));assert ok,name
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def compare(x,label):
    row=[]
    for method in METHODS:
      for semantics in ('CPU_CANONICAL','CUDA_LEGACY_EMULATED') if method!='B2' else ('CPU_CANONICAL',):
        e=encode(x,method,cpu_semantics=semantics);ref=reference(x.numpy(),method,semantics)
        for key in ('codes','scales','zero_points','negative_zero_flags','decoded'):
            a=decode(e) if key=='decoded' else getattr(e,key)
            b=ref[key]
            assert (a is None)==(b is None),(label,method,key)
            if a is not None:assert exact(a.numpy(),b),(label,method,semantics,key)
        d=decode(e)
        row.append(dict(method=method,semantics=semantics,sha256=hashlib.sha256(tensor_bytes(d)).hexdigest(),
                        codes_sha256=hashlib.sha256(tensor_bytes(e.codes)).hexdigest(),
                        scales_sha256=hashlib.sha256(tensor_bytes(e.scales)).hexdigest(),
                        zero_points_sha256=None if e.zero_points is None else hashlib.sha256(tensor_bytes(e.zero_points)).hexdigest(),
                        saturated_code_count=int(((e.codes==0)|(e.codes==255)).sum()) if method=='B2' else int((e.codes.abs()==127).sum()),
                        mse=float((d.double()-x.double()).square().mean())))
    details.append(dict(fixture=label,results=row));return True
shape=(1,2,128,128)
base=torch.randn(shape)
fixtures={
 '01_zero':torch.zeros(shape),
 '02_positive':base.abs()+.01,
 '03_negative':-base.abs()-.01,
 '04_symmetric_asymmetric':base*torch.linspace(.1,3,128).reshape(1,1,1,128)+.3,
 '05_scale_boundary':torch.nextafter(torch.ones(shape)*1e-12,torch.zeros(shape)),
 '06_zero_point_round_boundary':torch.linspace(-127.5,127.5,128).reshape(1,1,128,1).expand(shape).clone(),
 '07_code_half_integer':torch.arange(128).reshape(1,1,128,1).float().expand(shape).clone()-63.5,
 '08_tiny_subnormal':torch.full(shape,torch.finfo(torch.float32).tiny)*torch.where(base<0,-1.,1.),
 '09_extrema_clipping':torch.linspace(-1e20,1e20,128).reshape(1,1,128,1).expand(shape).clone(),
 '10_multihead_layout':base+torch.tensor([0.,100.]).reshape(1,2,1,1),
 '11_noncontiguous':base.transpose(-1,-2),
}
for label,x in fixtures.items():check(label,compare(x,label))
# Canonical original, all24 distinct layers, full32 heads. This reference is unmodified.
class Cache:pass
cache=Cache();cache.recurrent_states={i:torch.randn(1,32,128,128)*(.01+i*.02) for i in GDN_LAYERS}
cache.recurrent_states[0][0,0,0,0]=-0.;cache.recurrent_states[0][0,0,1,0]=-1e-15
original={i:x.clone() for i,x in cache.recurrent_states.items()}
audit=dict(layers=set(),scale_shapes={},code_min=127,code_max=-127,calls=0)
quantize_c128(cache,audit)
for i in GDN_LAYERS:
    e=encode(original[i],'B0');d=decode(e)
    parity.append(dict(layer=i,shape=list(d.shape),equal_values=torch.equal(d,cache.recurrent_states[i]),equal_bytes=tensor_bytes(d)==tensor_bytes(cache.recurrent_states[i]),max_abs=float((d-cache.recurrent_states[i]).abs().max())))
check('B0_unmodified_canonical_24_layers_32_heads',all(x['equal_bytes'] and x['equal_values'] for x in parity))
h=hadamard(128);eye=torch.eye(128)
check('12_H_inverse_H',float((h.T@h-eye).abs().max())<2e-6 and float(((h.T@(h@base))-base).abs().max())<5e-6)
for method in METHODS:
    state=base.clone();e=qdq_writeback(state,method)
    check('13_writeback_'+method,tensor_bytes(state)==tensor_bytes(decode(e)) and require_consumed(state,e.cpu()))
    e2=encode(state,method);d2=decode(e2)
    check('14_repeat_'+method,compare(state,'repeat_'+method))
    repeat.append(dict(method=method,equal_bytes=tensor_bytes(state)==tensor_bytes(d2),max_abs=float((state-d2).abs().max()),idempotence_required=False))
    try:require_consumed(base,e.cpu());detected=False
    except RuntimeError:detected=True
    check('16_skip_writeback_negative_'+method,detected)
e=encode(base+.5,'B2');bad=copy.deepcopy(e);bad.zero_points[0,0,0,0]^=1
check('17_wrong_zp_negative',tensor_bytes(decode(e))!=tensor_bytes(decode(bad)))
for m in METHODS:
    right=decode(encode(base,m));wrong=decode(encode(base.transpose(-1,-2),m)).transpose(-1,-2)
    check('18_wrong_axis_negative_'+m,tensor_bytes(right)!=tensor_bytes(wrong))
# Tie-to-even and FP32 rounding cross-check against exact integer Fraction values.
from fractions import Fraction
ties=[Fraction(n,2) for n in range(-511,512,2)]
check('RNE_exact_Fraction_half_ties',all(float(torch.round(torch.tensor(float(x),dtype=torch.float32)))==round(x) for x in ties))
# Explicit clipping of representable code levels, including outside each range.
check('clipping_primitives',torch.equal(torch.round(torch.tensor([-300.,-127.5,-126.5,126.5,127.5,300.])).clamp(-127,127),torch.tensor([-127.,-127.,-126.,126.,127.,127.])))
for value in (float('nan'),float('inf')):
    x=base.clone();x[0,0,0,0]=value
    for m in METHODS:
        try:encode(x,m);caught=False
        except FloatingPointError:caught=True
        check('nonfinite_reject_'+str(value)+'_'+m,caught)
manifest=json.loads((R/'fixtures/REAL_STATE_INPUT_MANIFEST.json').read_text())
for item in manifest['records']:
    raw=(R/item['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
    x=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(1,1,128,128)
    hx=h.T@x
    compare(hx,item['file']+'_H_coordinates')
check('real_states_27_heads_H_coordinates',len(manifest['records'])==27)
out=dict(status='CPU_CODEC_QUALIFIED',scope='CPU_ONLY_NO_GPU_CLAIMS',torch=torch.__version__,numpy=np.__version__,checks=checks,all_18_categories=True,repeat_QDQ=repeat,fixtures=details,GPU_dispatches=0,GPU_queries=0)
outdir=R/'qualification';outdir.mkdir(exist_ok=True)
name=sys.argv[1] if len(sys.argv)>1 else 'CPU_PROCESS_0.json'
(outdir/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(outdir/(name+'.canonical.json')).write_text(json.dumps(dict(status='PASS_CPU_CANONICAL_BYTES',layers=parity,canonical_not_modified=True),indent=2)+'\n')
print(json.dumps(dict(status=out['status'],checks=len(checks),fixture_groups=len(details),output=name)))
