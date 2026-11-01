"""Strict operator evidence and separately marked REFERENCE-ONLY original execution."""
import argparse, sys
import closure_common as C
sys.path.insert(0,str(C.ROOT/'runtime_patch'))
from deterministic_cumsum import deterministic_cumsum_fixed_order

def main(args):
    import torch
    C.configure(reference=args.mode=='reference')
    manifest=C.read(C.ROOT/'provenance/frozen_cumsum_inputs.json'); rows=[]; outputs={}
    for fixture in manifest['fixtures']:
        C.check(C.sha(C.ROOT/fixture['path'])==fixture['sha256'],'fixture changed')
        x=C.load_pt(fixture['path'])['g'].cuda(); hashes=[]
        with torch.inference_mode():
            for _ in range(1 if args.mode=='reference' else 5):
                y=x.cumsum(-1) if args.mode=='reference' else deterministic_cumsum_fixed_order(x,-1)
                hashes.append(C.tensor_sha(y))
        rows.append(dict(name=fixture['name'],same_process_hashes=hashes,
                         same_process_exact=len(set(hashes))==1,output=C.record(y)))
        outputs[fixture['name']]=y.cpu()
    artifact=C.pt_save('operator/'+args.mode+'_'+args.run+'.pt',outputs)
    C.save('operator/'+args.mode+'_'+args.run+'.json',dict(rows=rows,artifact=artifact,
           strict=torch.are_deterministic_algorithms_enabled(),REFERENCE_ONLY_NONDETERMINISTIC_ALLOWED=args.mode=='reference',
           use_for_scientific_measurement=False))
    print('OPERATOR_PROCESS_COMPLETE '+args.mode+' '+args.run,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['strict','reference'],required=True);p.add_argument('--run',required=True)
    main(p.parse_args())
