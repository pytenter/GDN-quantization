#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    ps=sorted(a.input_dir.glob("run_*.json")); rs=[json.loads(x.read_text()) for x in ps]; ts=[torch.load(x.with_suffix('.temporary.pt'),map_location='cpu',weights_only=False) for x in ps]
    if len(rs)!=5: raise RuntimeError(len(rs))
    counts=Counter(x['output_sha256'] for x in rs); by=defaultdict(list)
    for x in rs: by[json.dumps(x['autotune'],sort_keys=True)].append(x['output_sha256'])
    ref=ts[0].double(); exact=None
    for t in ts[1:]:
        d=t.double()-ref
        if bool((d!=0).any()):
            nz=(d!=0).nonzero(); exact={"max_abs":float(d.abs().max()),"relative_l2":float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ref).clamp_min(1e-12)),"different_elements":int((d!=0).sum()),"first_different_index":[int(v) for v in nz[0].tolist()]}; break
    out={"task":"LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2","runs":5,"fresh_process_repeatable":len(counts)==1,"hash_counts":dict(counts),"first_variant_exact_diff":exact,"output_hashes_by_autotune_config":dict(by),"autotune_records":[x['autotune'] for x in rs],"temporary_tensors_cleaned":True}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    for x in ps: x.with_suffix('.temporary.pt').unlink()
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
