#!/usr/bin/env python3
"""Isolate FLA chunk_local_cumsum on the frozen decay driver."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import torch

def th(x): return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def configs(kernel):
    out=[]; cur=kernel
    for depth in range(5):
        cache=getattr(cur,"cache",None)
        if isinstance(cache,dict):
            out += [{"depth":depth,"key":str(k),"kwargs":dict(c.kwargs),"num_warps":c.num_warps,
                     "num_stages":c.num_stages,"num_ctas":getattr(c,"num_ctas",None)} for k,c in cache.items()]
        cur=getattr(cur,"fn",None)
        if cur is None: break
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--driver-bundle",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--run-id",type=int,required=True); a=p.parse_args()
    import fla.ops.utils.cumsum as cm
    d=torch.load(a.driver_bundle,map_location="cpu",weights_only=False); g=d["g"].float().cuda()
    from fla.ops.utils.constant import RCP_LN2
    with torch.inference_mode():
        y=cm.chunk_local_cumsum(g=g,scale=RCP_LN2,chunk_size=64); torch.cuda.synchronize()
    r={"task":"LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2","scope":"isolated chunk_local_cumsum",
       "run_id":a.run_id,"pid":os.getpid(),"input_sha256":th(g),"output_sha256":th(y),
       "environment":{k:os.environ.get(k) for k in ("FLA_CACHE_MODE","FLA_CONFIG_DIR","TRITON_CACHE_DIR")},
       "autotune":configs(cm.chunk_local_cumsum_vector_kernel)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    torch.save(y.detach().cpu(),a.output.with_suffix(".temporary.pt")); print(json.dumps(r,indent=2))
if __name__=="__main__": main()
