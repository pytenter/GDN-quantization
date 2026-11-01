"""Append-only stricter dtype routing audit; no numerical computation is rerun on GPU."""
import closure_common as C
from qualify_operator import compare

if __name__=='__main__':
    import torch
    a=C.load_pt('chunk/reference.pt');b=C.load_pt('chunk/strict.pt'); rows=[]
    for n in a:
        for k in a[n]:
            kind='chunk_fp32' if a[n][k].dtype==torch.float32 else 'chunk_bf16'
            rows.append(dict(fixture=n,object=k,dtype=str(a[n][k].dtype),policy_kind=kind,**compare(a[n][k],b[n][k],kind)))
    C.save('provenance/chunk_dtype_revalidation.json',dict(rows=rows,
       gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL',
       incident='Initial analyzer routed output objects to the pre-frozen BF16 cap by object name. Actual captured output dtype can be FP32. Original report retained; this append-only audit uses actual dtype and the already-frozen stricter FP32 caps. No tolerance changed; no GDN rerun.',
       original_report_sha256=C.sha(C.ROOT/'provenance/chunk_gdn_parity.json')))
    C.check(all(r['PASS'] for r in rows),'stricter dtype-aware numerical revalidation FAIL; STOP')
