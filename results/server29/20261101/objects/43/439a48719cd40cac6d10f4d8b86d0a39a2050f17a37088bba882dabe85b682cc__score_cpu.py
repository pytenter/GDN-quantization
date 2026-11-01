"""One fresh CPU FrozenV4 extraction pass, only after60 true closed trajectories."""
import argparse,dataclasses,json,math,os,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'protocol')]
import execution_common as C
import aime26_scorer_v4 as V4


def summarize(rows):
    assert len(rows)==60
    by={p:sorted([x for x in rows if x['period']==p],key=lambda x:x['problem_id']) for p in (8,16)}
    assert all(len(x)==30 for x in by.values())
    pairs=[];counts=dict(both_correct=0,dp8_only=0,dp16_only=0,both_wrong=0)
    for a,b in zip(by[8],by[16]):
        assert a['problem_id']==b['problem_id']
        label=('both_correct' if a['correct'] and b['correct'] else 'dp8_only' if a['correct'] else 'dp16_only' if b['correct'] else 'both_wrong')
        counts[label]+=1;pairs.append(dict(problem_id=a['problem_id'],dp8_correct=a['correct'],dp16_correct=b['correct'],paired_change=label,tokens_dp8=a['generated_token_count'],tokens_dp16=b['generated_token_count']))
    n=counts['dp8_only']+counts['dp16_only'];k=min(counts['dp8_only'],counts['dp16_only'])
    pvalue=min(1.0,2*sum(math.comb(n,j) for j in range(k+1))/(2**n)) if n else 1.0
    arms={}
    for p,rr in by.items():
        lengths=[x['generated_token_count'] for x in rr]
        arms[str(p)]=dict(N=30,correct=sum(x['correct'] for x in rr),accuracy=sum(x['correct'] for x in rr)/30,
            mean_tokens=statistics.mean(lengths),median_tokens=statistics.median(lengths),
            EOS=sum(x['finish_reason']=='eos' for x in rr),length_cap=sum(x['finish_reason']=='length' for x in rr),
            nonfinite_count=0,runtime_error_count=0,mean_repeated_32gram_ratio=statistics.mean(x['repeated_32gram_ratio'] for x in rr),
            pooled_generation_TPS=sum(lengths)/sum(x['generation_wall_s'] for x in rr))
    return dict(arms=arms,paired_counts=counts,exact_McNemar_pvalue=pvalue,pairs=pairs)


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU scoring must have no visible GPU'
    C.source(ROOT,a.source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    session=json.loads((ROOT/'runtime/FORMAL_SESSION_RESULT.json').read_text());assert session['status']=='COMPLETE60_TRUE_EXIT_RELEASE_AND_ARCHIVE' and len(set(session['completed']))==60
    sys.path.insert(0,'/data/zypan/transformers-qwen35')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
    gold={x['problem_id']:x for x in map(json.loads,(ROOT/'protocol/aime26_frozen.jsonl').read_text().splitlines())}
    jobs=json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs'];rows=[]
    for job in jobs:
        target=ROOT/'formal'/job['id'];closed=json.loads((target/'OWNING_CLOSED.json').read_text());archive=json.loads((target/'ARCHIVE29.json').read_text())
        assert closed['owning_Popen_exit']==0 and closed['release_observed'] and closed['RESULT_sha256']==C.sha(target/'RESULT.json')
        assert closed['source_sha256']==a.source_sha and closed['permit_sha256']==a.permit_sha and archive['status']=='PASS_29_DURABLE_NOREPLACE'
        payload=json.loads((target/'RESULT.json').read_text());assert payload['job']==job
        assert payload['source_manifest_sha256']==a.source_sha and payload['permit_sha256']==a.permit_sha
        digest=payload.pop('payload_sha256');assert C.objsha(payload)==digest
        r=payload['result'];ids=r['generated_token_ids']
        assert r['status']=='COMPLETE' and not r['nonfinite'] and r['runtime_error'] is None and r['seed']==1
        assert r['generated_token_count']==len(ids) and 0<len(ids)<=81920 and C.objsha(ids)==r['generated_token_ids_sha256']
        text=tokenizer.decode(ids,skip_special_tokens=True);assert text==r['response'],'Same-payload CPU re-decode changed'
        extracted=V4.extract_v4(text,termination_reason=r['finish_reason'],problem_text=gold[job['problem_id']]['problem'])
        correct=V4.compare_with_gold(extracted,gold[job['problem_id']]['answer'])
        count=max(0,len(ids)-31)
        repeat=(count-len({tuple(ids[i:i+32]) for i in range(count)}))/count if count else 0.0
        rows.append(dict(unit=job['id'],problem_id=job['problem_id'],period=job['period'],seed=1,correct=correct,
            generated_token_count=len(ids),finish_reason=r['finish_reason'],generation_wall_s=r['wall_seconds'],
            repeated_32gram_ratio=repeat,extraction=dataclasses.asdict(extracted),RESULT_sha256=closed['RESULT_sha256'],raw_payload_sha256=digest))
        print(json.dumps(dict(phase='CPU_FROZENV4_PAYLOAD_VERIFIED',count=len(rows),unit=job['id'])),flush=True)
    value=dict(status='PASS_60_SAME_PAYLOAD_CPU_FROZENV4',source_sha256=a.source_sha,permit_sha256=a.permit_sha,
        scorer_sha256=C.sha(ROOT/'protocol/aime26_scorer_v4.py'),rows=rows,summary=summarize(rows),
        repetition_definition='Overlapping32token grams: (totalgrams-uniquegrams)/totalgrams,0 when fewer32tokens')
    C.atomic(a.output,value)
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--output',required=True,type=Path);raise SystemExit(run(p.parse_args()))
