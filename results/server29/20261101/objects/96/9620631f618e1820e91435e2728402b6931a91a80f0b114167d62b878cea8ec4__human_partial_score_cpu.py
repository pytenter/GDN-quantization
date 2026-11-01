"""Human-requested partial CPU score, separate from the unchanged60-unit finisher.

This helper only reads the fixed snapshot of exit0/released/29-durable outputs.
It never launches a GPU/model, changes a queue, or promotes partial accuracy to a
full30-question arm score. No sampler or FrozenV4 code is modified.
"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'protocol')]
import execution_common as C
import aime26_scorer_v4 as V4

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='' and __debug__
    snapshot=json.loads(a.snapshot.read_text())
    source=snapshot['source_sha256'];permit=snapshot['permit_sha256']
    C.source(ROOT,source);assert C.sha(ROOT/'GPU_PERMIT.json')==permit
    assert snapshot['authorization']=='EXPLICIT_HUMAN_CURRENT_PARTIAL_ACCURACY_REQUEST'
    before=C.model_pins(ROOT)
    # Recheck actual tokenizer assets, without reading or loading model weights.
    for x in json.loads((ROOT/'analysis/MODEL_TOKENIZER_FRESH_SHA.json').read_text())['files']:
        if Path(x['path']).suffix not in {'.safetensors','.bin','.pt','.pth'}:
            assert C.sha(x['path'])==x['sha256']
    sys.path.insert(0,'/data/zypan/transformers-qwen35')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
    gold={x['problem_id']:x for x in map(json.loads,(ROOT/'protocol/aime26_frozen.jsonl').read_text().splitlines())}
    jobs={x['id']:x for x in json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs']}
    rows=[]
    for record in snapshot['units']:
        unit=record['unit'];job=jobs[unit];p=ROOT/'formal'/unit
        closed=json.loads((p/'OWNING_CLOSED.json').read_text());archive=json.loads((p/'ARCHIVE29.json').read_text())
        assert C.sha(p/'RESULT.json')==record['RESULT_sha256']==closed['RESULT_sha256']
        assert C.sha(p/'OWNING_CLOSED.json')==record['OWNING_CLOSED_sha256']
        assert closed['owning_Popen_exit']==0 and closed['release_observed'] is True
        assert closed['source_sha256']==source and closed['permit_sha256']==permit
        assert archive['status']=='PASS_29_DURABLE_NOREPLACE' and archive['manifest_sha256']==record['archive_manifest_sha256']
        payload=json.loads((p/'RESULT.json').read_text());digest=payload.pop('payload_sha256');assert C.objsha(payload)==digest
        assert payload['job']==job and payload['source_manifest_sha256']==source and payload['permit_sha256']==permit
        result=payload['result'];ids=result['generated_token_ids']
        assert result['status']=='COMPLETE' and result['seed']==1 and not result['nonfinite'] and result['runtime_error'] is None
        assert len(ids)==result['generated_token_count'] and C.objsha(ids)==result['generated_token_ids_sha256'] and 0<len(ids)<=81920
        text=tokenizer.decode(ids,skip_special_tokens=True);assert text==result['response']
        extraction=V4.extract_v4(text,termination_reason=result['finish_reason'],problem_text=gold[job['problem_id']]['problem'])
        correct=V4.compare_with_gold(extraction,gold[job['problem_id']]['answer'])
        rows.append(dict(unit=unit,problem_id=job['problem_id'],period=job['period'],seed=1,
            correct=correct,generated_tokens=len(ids),finish_reason=result['finish_reason'],
            extraction_status=extraction.status,normalized_value=extraction.normalized_value,
            abstain_reason=extraction.abstain_reason,RESULT_sha256=record['RESULT_sha256'],payload_sha256=digest))
        print(json.dumps(dict(phase='HUMAN_PARTIAL_COMPLETED_PAYLOAD_CPU_VERIFIED',count=len(rows),unit=unit)),flush=True)
    assert C.model_pins(ROOT)==before
    C.source(ROOT,source)
    arms={}
    for period in (8,16):
        rr=[x for x in rows if x['period']==period];n=len(rr);correct=sum(x['correct'] for x in rr)
        arms[str(period)]=dict(completed=n,correct=correct,partial_accuracy=correct/n if n else None,
            pending=30-n,full_arm_accuracy=None,EOS=sum(x['finish_reason']=='eos' for x in rr),
            length_cap=sum(x['finish_reason']=='length' for x in rr),abstentions=sum(x['extraction_status']!='EXTRACTED' for x in rr))
    by={p:{x['problem_id']:x for x in rows if x['period']==p} for p in [8,16]}
    common=sorted(set(by[8])&set(by[16]))
    result=dict(status='PASS_HUMAN_REQUESTED_PARTIAL_CPU_FROZENV4_SAME_PAYLOAD',
        source_sha256=source,permit_sha256=permit,snapshot_sha256=C.sha(a.snapshot),
        scorer_sha256=C.sha(ROOT/'protocol/aime26_scorer_v4.py'),helper_sha256=C.sha(__file__),
        completed_units=len(rows),new_model_generations=0,actual_GPU_calls=0,
        arms=arms,common_completed_problem_ids=common,
        same_problem_subset={str(p):dict(N=len(common),correct=sum(by[p][q]['correct'] for q in common)) for p in [8,16]},
        rows=rows,full_experiment_accuracy=None,
        no_algorithm_queue_seed_prompt_or_scoring_rule_changes=True,
        warning='Partial available-case fractions only. Arms may contain different completed questions. Final60-unit two-fresh-CPU finisher remains unchanged.')
    C.atomic(a.output,result)
    print(json.dumps(dict(status=result['status'],completed_units=len(rows),arms=arms,same_problem_subset=result['same_problem_subset'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
