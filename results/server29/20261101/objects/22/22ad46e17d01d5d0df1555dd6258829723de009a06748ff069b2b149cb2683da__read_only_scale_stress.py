"""Describe the already preregistered G3 stress inputs; no codec modifications."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from dual_axis_codec import encode,NAMES
from panel import thash
import test_reference_parity
out=ROOT/'analysis'/'scale_stress_description'
out.mkdir(exist_ok=False)
panel={}
for name,magnitude in [('tiny',1e-30),('large',1e10),('fp32_max',torch.finfo(torch.float32).max)]:
    x=torch.full((2,3,128,128),magnitude);x[...,::2,:]*=-1;panel[name]=x
with (out/'input_manifest.json').open('x') as f:
    json.dump({n:{'sha256':thash(x),'shape':list(x.shape),'generation':'same deterministic signed constant as preregistered G3 stress'} for n,x in panel.items()},f,indent=2)
rows=[]
for name,x in panel.items():
    for codec in NAMES:
        try:
            e=encode(x,codec);y=e.decode();err=y.double()-x.double()
            rows.append(dict(case=name,codec=codec,input_sha256=thash(x),finite=True,relative_mse=float(err.square().sum()/x.double().square().sum()),max_reconstruction=y.abs().max().item(),saturation_fraction=(e.codes.abs()==127).double().mean().item()))
        except FloatingPointError as exc:
            assert codec==NAMES[0] and name=='fp32_max' and str(exc)=='nonfinite C128 dequant'
            rows.append(dict(case=name,codec=codec,input_sha256=thash(x),behavior='FAIL_CLOSED_AS_FROZEN_FORMAL',error=str(exc)))
parity=test_reference_parity.stepquant(panel)
assert not torch.cuda.is_initialized()
with (out/'results.json').open('x') as f:json.dump(dict(metrics=rows,official_stepquant_exact_parity=parity,cuda_initialized=False,GPU_HOURS_USED=0),f,indent=2,allow_nan=False)
print(json.dumps(rows))
