import sys
from types import SimpleNamespace
import torch
from dual_axis_codec import encode,NAMES
from references import CANON,FORMAL,STEP

def canonical(panel):
    rows=[]
    for name,x in panel.items():
        for dtype in (torch.float32,torch.float16,torch.bfloat16,torch.float64):
            value=x.to(dtype); expected=CANON['qwen_c128_qdq'](value); e=encode(value,NAMES[0])
            assert torch.equal(e.codes.float(),expected.codes),(name,dtype,'codes')
            assert torch.equal(e.columns,expected.scale),(name,dtype,'scale')
            assert torch.equal(e.decode(),expected.dequant),(name,dtype,'dequant')
        # Execute the unchanged original formal function on CPU fake cache.
        observed={}
        def trace(frame,event,arg):
            if frame.f_code is FORMAL['quantize_c128'].__code__ and event=='return':
                for key in ('codes','scale','dequant'):observed[key]=frame.f_locals[key].clone()
            return trace
        cache=SimpleNamespace(recurrent_states={i:x.clone() for i in FORMAL['GDN_LAYERS']})
        audit={'layers':set(),'scale_shapes':{},'code_min':0.,'code_max':0.,'calls':0}
        old=sys.gettrace();sys.settrace(trace)
        try:FORMAL['quantize_c128'](cache,audit)
        finally:sys.settrace(old)
        e=encode(x,NAMES[0])
        assert torch.equal(e.codes.float(),observed['codes']),(name,'formal codes')
        assert torch.equal(e.columns,observed['scale']),(name,'formal scales')
        assert torch.equal(e.decode(),observed['dequant']),(name,'formal dequant')
        assert all(torch.equal(t,e.decode()) for t in cache.recurrent_states.values()),(name,'cache writeback')
        rows.append({'case':name,'shared_dtypes':['FP32','FP16','BF16','FP64'],'codes_exact':True,'scale_exact':True,'dequant_exact':True,'formal_fake_cache_24_layers_exact':True})
    return rows

def stepquant(panel):
    rows=[]
    for name,x in panel.items():
        plan=STEP['QuantizationPlan']('gdn',torch.full(x.shape[1:3],8,dtype=torch.int64),torch.ones(x.shape[1:3]))
        z,r,c,two,recon=STEP['fit_state'](x,plan);e=encode(x,NAMES[3])
        assert two is None,(name,'unexpected INT2 rows')
        assert torch.equal(e.codes.to(torch.int16),z),(name,'STEP codes')
        assert torch.equal(e.rows.squeeze(-1),r),(name,'STEP row scales')
        assert torch.equal(e.columns.squeeze(-2),c),(name,'STEP col scales')
        assert torch.equal(e.decode(),recon),(name,'STEP reconstruction')
        rows.append(dict(case=name,codes_exact=True,rows_fp16_exact=True,columns_fp16_exact=True,decode_exact=True,pivots=0,impact='ones',bits=8))
    return rows
