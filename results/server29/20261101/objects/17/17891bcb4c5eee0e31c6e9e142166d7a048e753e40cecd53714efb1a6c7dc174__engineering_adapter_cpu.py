"""Necessary new layer-batch/interface dependency check only; CPU, not old Phase0 rerun."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,pathlib,json,hashlib
from types import SimpleNamespace
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import torch
import core as K
import codec as C
import worker as W
import reducer
torch.set_num_threads(1);reducer.configure_cpu()
f=K.formal(False);memo=K.formal(True)
g=torch.Generator().manual_seed(20261015);x=torch.randn((24,32,128,128),generator=g)*.01
encoded=C.encode(x,'DUAL_M_INT8');plan=C.references.STEP['QuantizationPlan']('gdn',torch.full((32,128),8,dtype=torch.long),torch.ones((32,128)))
z,rows,cols,two,y=C.references.STEP['fit_state'](x,plan)
assert two is None and torch.equal(z,encoded.codes.to(torch.int16)) and torch.equal(rows,encoded.rows.squeeze(-1)) and torch.equal(cols,encoded.columns.squeeze(-2)) and torch.equal(y,encoded.decode())
for i in range(24):
 e=C.encode(x[i:i+1],'DUAL_M_INT8');assert K.tsha(e.codes)==K.tsha(encoded.codes[i:i+1]) and K.tsha(e.columns)==K.tsha(encoded.columns[i:i+1]) and K.tsha(e.rows)==K.tsha(encoded.rows[i:i+1])
a=SimpleNamespace(recurrent_states={l:x[i:i+1].clone() for i,l in enumerate(f.GDN_LAYERS)});b=SimpleNamespace(recurrent_states={l:x[i:i+1].clone() for i,l in enumerate(f.GDN_LAYERS)})
da,_=W.original_c128(a,f,True);db,_=W.original_c128(b,memo,True);assert da==db and len(da)==24
assert K.tsha(f.hadamard())==K.tsha(memo.hadamard()) and memo.hadamard() is memo.hadamard()
cache=SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros((1,4,3,256)),values=torch.zeros((1,4,3,256)),conv_states={0:torch.zeros((1,2,4))},recurrent_states={0:torch.zeros((1,32,128,128))},cursor=3)])
assert len(list(K.cache_tensors(cache)))==4
assert W.deep_hash(cache)==W.deep_hash(W.cpu_tree(cache))
assert not torch.cuda.is_initialized()
v=dict(status='PASS',CUDA_initialized=False,model_loaded=False,encode_ast_sha256=C.ENCODE_AST_SHA256,checks=['B24official all8/ones exact','B24 vs24 independent batch byteexact','Direct original C128 full24 trace namespace/writeback exact','Original H factory vs memo byteexact','dict/list GDN cache hashed','CPU snapshot tree tensor+metadata roundtrip','actual worker import/dependency invocation'],code_hashes={p.name:K.sha(p) for p in (R/'src').glob('*.py')},GPU_HOURS_USED=0)
with (R/'qualification_cpu/ENGINEERING_ADAPTER_CPU.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',CUDA_initialized=False,checks=len(v['checks']))))
