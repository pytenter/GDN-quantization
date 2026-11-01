"""Device adapter: unchanged sealed V1 encode AST; device/storage interface only."""
import ast,hashlib,json
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent;V1=PARENT/'inherited_cpu_v1'
sys.path.insert(0,str(V1/'src'))
import references
NAMES=('C128_CANONICAL','C128_FP16_SCALE','C128_LS_FP16','DUAL_M_INT8')
SYNC_COUNT=0
SCALE_TRACE=None

def require(condition,msg):
    global SYNC_COUNT
    if isinstance(condition,torch.Tensor):SYNC_COUNT+=1;condition=bool(condition.item())
    if not condition:raise RuntimeError(msg)

def source(x):
    require(isinstance(x,torch.Tensor) and x.device.type in ('cpu','cuda') and x.ndim==4 and min(x.shape)>0 and x.is_floating_point(),'invalid state interface')
    require(torch.isfinite(x).all(),'nonfinite input');x=x.detach().float()
    require(torch.isfinite(x).all(),'nonfinite FP32 conversion')
    return x

@dataclass(frozen=True)
class EncodedState:
    codec:str
    shape:tuple
    codes:torch.Tensor
    columns:torch.Tensor
    rows:torch.Tensor|None=None

    def validate(self):
        require(self.codes.dtype==torch.int8 and tuple(self.codes.shape)==self.shape,'code shape/dtype')
        require((self.codes!=-128).all(),'illegal code -128')
        b,h,k,v=self.shape
        require(tuple(self.columns.shape)==(b,h,1,v),'column shape')
        require(self.columns.dtype==(torch.float32 if self.codec==NAMES[0] else torch.float16),'column dtype')
        require(torch.isfinite(self.columns).all()&(self.columns>0).all(),'invalid columns')
        if self.codec==NAMES[3]:
            require(self.rows is not None and tuple(self.rows.shape)==(b,h,k,1) and self.rows.dtype==torch.float16,'row shape/dtype')
            require(torch.isfinite(self.rows).all()&(self.rows>0).all(),'invalid rows')
        else:require(self.rows is None,'unexpected rows')
        return self

    def decode(self):
        return (self.rows.float()*self.columns.float())*self.codes.float() if self.rows is not None else self.codes.float()*self.columns.float()

    def nbytes(self):return sum(t.numel()*t.element_size() for t in (self.codes,self.columns,self.rows) if t is not None)

_text=(V1/'src/dual_axis_codec.py').read_text();_tree=ast.parse(_text)
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('safe_half','levels','encode')]
_namespace=dict(torch=torch,CANON=references.CANON,NAMES=NAMES,LOW=2**-24,HIGH=65504.,source=source,EncodedState=EncodedState)
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(V1/'src/dual_axis_codec.py'),'exec'),_namespace)
_original_safe=_namespace['safe_half'];_original_levels=_namespace['levels']
def safe_half(x):
    global SYNC_COUNT
    SYNC_COUNT+=1
    if SCALE_TRACE is not None:SCALE_TRACE.append(x.detach().clone())
    return _original_safe(x)
def levels(x):
    global SYNC_COUNT
    SYNC_COUNT+=1
    return _original_levels(x)
_namespace.update(safe_half=safe_half,levels=levels)
encode=_namespace['encode']
ENCODE_AST_SHA256=hashlib.sha256(ast.dump(next(n for n in _nodes if n.name=='encode'),include_attributes=False).encode()).hexdigest()

def trace_encode(x,codec):
    global SCALE_TRACE
    require(SCALE_TRACE is None,'nested scale trace')
    SCALE_TRACE=[]
    try:
        result=encode(x,codec);raw=SCALE_TRACE;return result,raw
    finally:SCALE_TRACE=None

def minimal_cpu_regression():
    import dual_axis_codec as original
    generator=torch.Generator().manual_seed(20261015)
    normal=torch.randn((2,2,128,128),generator=generator)
    cases={'normal':normal,'row_imbalance':normal*torch.logspace(-3,3,128)[None,None,:,None],'zeros':torch.zeros_like(normal)}
    records=[]
    for name,x in cases.items():
        for codec in NAMES:
            old=original.encode(x,codec);new=encode(x,codec)
            require(torch.equal(old.codes,new.codes),'CPU adapter codes mismatch')
            require(torch.equal(old.columns,new.columns),'CPU adapter columns mismatch')
            require((old.rows is None and new.rows is None) or torch.equal(old.rows,new.rows),'CPU adapter rows mismatch')
            require(torch.equal(old.decode(),new.decode()),'CPU adapter reconstruction mismatch')
            records.append(dict(case=name,codec=codec,codes_scales_decode='EXACT'))
    return dict(status='PASS',checks=len(records),records=records,encode_ast_sha256=ENCODE_AST_SHA256,scope='Necessary interface adaptation only; three fixed CPU states, not full prior Phase0 repeat')

if __name__=='__main__':
    report=minimal_cpu_regression()
    with (ROOT/'analysis/MINIMAL_ADAPTER_CPU_REGRESSION.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report['status'],checks=report['checks'])))
