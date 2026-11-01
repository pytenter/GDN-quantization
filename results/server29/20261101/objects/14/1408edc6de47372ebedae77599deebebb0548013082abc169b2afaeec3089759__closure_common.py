"""New-namespace only observers/serialization. Historical numerical implementations unmodified."""
from pathlib import Path
import ast, hashlib, inspect, json, os, sys, types
import torch
D=Path(__file__).resolve().parents[1];R=D.parents[1]
V=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
sys.path.insert(0,str(V/'scripts'))
import common as c
import contemporary_smoke as s
def load(p):return json.loads(Path(p).read_text())
def emit(p,x):c.emit(D/p,x)
def m(x):
    if not torch.is_tensor(x):return str(x) if isinstance(x,torch.dtype) else x
    return dict(**s.safe_tensor(x),stride=list(x.stride()),storage_offset=x.storage_offset(),layout=str(x.layout))
def cpu(x):
    if not torch.is_tensor(x):return x
    z=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device='cpu');z.copy_(x.detach());return z
def gpu(x):
    if not torch.is_tensor(x):return x
    z=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device='cuda');z.copy_(x);return z
def verify():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and torch.cuda.device_count()==1
    assert c.hardware()['gpus'][0]['gpu_uuid']==c.UUIDS[0]
    plan=load(D/'runtime/source_freeze_attempt2.json')
    for r in plan['files']:assert c.sha(r['path'])==r['sha256'],r['path']
    bm=load(D/'protocol/prospective_call37_input_manifest.json')
    assert c.sha(bm['source'])==c.sha(bm['bundle'])==bm['bundle_sha256']
    return bm
def apply_fixed():
    hm=load(D/'runtime/historical_fixed_forward_manifest.json')
    for k,v in hm['selector'].items():assert os.environ[k]==v
    assert c.sha(hm['helper'])==hm['helper_sha256']
    assert c.sha(hm['fla_source'])==hm['fla_source_sha256']
    source=Path(hm['helper']).read_text();tree=ast.parse(source)
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==hm['function'])
    ns={'os':os};exec(compile(ast.Module(body=[fn],type_ignores=[]),hm['helper'],'exec'),ns)
    assert ns[fn.name]() is True
    import fla.modules.fused_norm_gate as f
    k=f.layer_norm_gated_fwd_kernel;seen=[]
    while k is not None:
        if hasattr(k,'configs'):
            seen.append([dict(kwargs=z.kwargs,num_warps=z.num_warps,num_stages=z.num_stages) for z in k.configs])
            assert len(k.configs)==1 and seen[-1][0]==hm['configuration']
        k=getattr(k,'fn',None)
    assert seen
    return f,hm,seen
class LaunchObserver:
    def __init__(self,kernel,name):
        self.rows=[];self.name=name
        from triton.runtime.jit import JITFunction
        while not isinstance(kernel,JITFunction):kernel=kernel.fn
        self.jit=kernel;self.original=kernel.run
        original=self.original
        def observed(_self,*args,**kwargs):
            result=original(*args,**kwargs)
            fields={k:kwargs[k] for k in ['BT','num_warps','num_stages','num_ctas','BD','BS','D','T','ACTIVATION','IS_RMS_NORM'] if k in kwargs}
            metadata=getattr(result,'metadata',None)
            compiled={k:getattr(metadata,k,None) for k in ['num_warps','num_stages','num_ctas']}
            self.rows.append(dict(kernel=name,launch=fields,compiled=compiled,warmup=bool(kwargs.get('warmup',False))))
            return result
        kernel.run=types.MethodType(observed,kernel)
    def check_forward(self,hm):
        assert self.rows and all(r['launch'].get('BT')==hm['configuration']['kwargs']['BT'] and r['launch'].get('num_warps')==hm['configuration']['num_warps'] and r['launch'].get('num_stages')==hm['configuration']['num_stages'] for r in self.rows)
        assert all(r['compiled']['num_warps']==hm['configuration']['num_warps'] and r['compiled']['num_stages']==hm['configuration']['num_stages'] for r in self.rows if not r['warmup'])
        return 'PASS'
def make_norm(bm,b):
    import fla.modules.fused_norm_gate as f
    a=bm['norm_attributes'];n=f.FusedRMSNormGated(hidden_size=a['hidden_size'],eps=a['eps'],activation=a['activation'],dtype=b['norm_weight'].dtype,device='cuda')
    n.weight.data.copy_(b['norm_weight']);n.eval();n.weight.requires_grad_(False)
    return n
