"""Only after all equivalence gates PASS. Fixed existing-development prefill."""
import argparse, gc, statistics, sys, time
import closure_common as C
sys.path.insert(0,str(C.ROOT/'runtime_patch'))
from patched_gdn import install

def run(mode):
    import torch
    C.check(C.read(C.ROOT/'provenance/fresh_process_determinism.json')['PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE']=='PASS','benchmark ordering')
    F,core,model,tok,qmod=C.init_model(reference=mode=='reference')
    if mode=='strict': install(qmod)
    ids=C.read(C.ROOT/'protocol/frozen_probe_tokens.json')['probes']['benchmark']
    warm=torch.tensor([ids[:128]],device='cuda:0'); x=torch.tensor([ids],device='cuda:0')
    with torch.inference_mode():
        o=model(input_ids=warm,use_cache=True);torch.cuda.synchronize();del o;gc.collect();torch.cuda.empty_cache()
        rows=[]
        for i in range(3):
            F.setup_seed(C.SEED);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            out=model(input_ids=x,use_cache=True);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            rows.append(dict(seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            del out;gc.collect();torch.cuda.empty_cache()
    C.save('benchmark/'+mode+'.json',dict(mode=mode,length=len(ids),warmup_length=128,rows=rows,median_seconds=statistics.median(r['seconds'] for r in rows),
        strict=torch.are_deterministic_algorithms_enabled(),REFERENCE_ONLY_NONDETERMINISTIC_ALLOWED=mode=='reference',
        why_not_2048='The frozen existing first development document has fewer than2048 tokens; actual complete document is used without invented repetition or changing frozen input. 8192 not attempted.',
        token_sha256=C.tensor_sha(torch.tensor(ids,dtype=torch.long))))

def analyze():
    a=C.read(C.ROOT/'benchmark/reference.json');b=C.read(C.ROOT/'benchmark/strict.json')
    C.check(a['length']==b['length'] and a['token_sha256']==b['token_sha256'],'benchmark identity')
    ratio=b['median_seconds']/a['median_seconds']
    C.save('provenance/performance_benchmark.json',dict(original=a,patched=b,PATCHED_PREFILL_SLOWDOWN=ratio,
        simple_implementation_accepted=ratio<=1.5,optimization_allowed=ratio>1.5,
        decision='ACCEPT_SIMPLE_NO_OPTIMIZATION' if ratio<=1.5 else 'ONE_OPTIMIZATION_CANDIDATE_PERMITTED'))
    print('PATCHED_PREFILL_SLOWDOWN='+str(ratio),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reference','strict','analyze'],required=True);a=p.parse_args()
    analyze() if a.phase=='analyze' else run(a.phase)
