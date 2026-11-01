"""Necessary AST-namespace/runtime glue regression; never initializes CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,pathlib,sys
from types import SimpleNamespace
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import gpu_worker as W
import gpu_codec as C
import torch
torch.set_num_threads(1)
F,N,arch=W.originals();checks=[]
assert N.compare.__globals__['tsha'] is N.tsha
for dtype in (torch.float32,torch.bfloat16):
    x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4).to(dtype)
    kind='model_fp32' if dtype==torch.float32 else 'model_bf16'
    same=N.compare(x,x.clone(),kind);assert same['PASS'] and same['byte_exact']
    different=N.compare(x,x+1000,kind);assert not different['PASS'] and not different['byte_exact']
    assert N.tsha(x)==W.tsha(x)
    checks.append(dict(test='unchanged_compare_namespace_and_hash',dtype=str(dtype),equal='PASS',large_difference='EXPECTED_FAIL'))
g=torch.Generator().manual_seed(20261015)
values={l:torch.randn((1,32,128,128),generator=g) for l in F.GDN_LAYERS}
cache=SimpleNamespace(recurrent_states={l:x.clone() for l,x in values.items()})
details,timing,before,after=W.quantize(cache,F,C.NAMES[0],reference=True)
assert len(details)==24 and len(after)==24 and not timing
for l,x in values.items():
    e=C.encode(x,C.NAMES[0]);decoded=C.decode_for_writeback(e,x);assert W.tsha(cache.recurrent_states[l])==W.tsha(decoded)
    assert details[str(l)]==dict(codes=W.tsha(e.codes),scale=W.tsha(e.columns),dequant=W.tsha(decoded))
assert torch.signbit(C.decode_for_writeback(C.encode(torch.tensor([[[[-0.0,-0.0001,1.0]]]]),C.NAMES[0]),torch.tensor([[[[-0.0,-0.0001,1.0]]]])))[0,0,0,0]
checks.append(dict(test='unchanged_formal_reference_trace_and_adapter',layers=24,status='EXACT_CODES_SCALE_DEQUANT_WRITEBACK'))
serialized=dict(pre_QDQ={str(l):v for l,v in before.items()},post_QDQ={str(l):v for l,v in after.items()},QDQ=details)
assert json.loads(json.dumps(serialized))==serialized
checks.append(dict(test='fresh_process_comparison_schema',status='JSON_ROUNDTRIP_EXACT'))
assert not torch.cuda.is_initialized()
result=dict(status='PASS',worker_sha256=hashlib.sha256((ROOT/'src/gpu_worker.py').read_bytes()).hexdigest(),CUDA_initialized=False,model_loaded=False,checks=checks,scope='Focused dependency/reference trace/schema regression only; prior CPU Phase0 not repeated',prior_failure_repaired='Original tsha AST and hashlib dependency recovered; canonical writeback signed-zero I/O identity restored; original compare and encode algorithm unchanged')
with (ROOT/'analysis/CPU_ENGINEERING_IMPORT_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',checks=len(checks),CUDA_initialized=False)))
