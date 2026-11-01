"""Read-only independently score a frozen list of actual old INT8 Asym32 payloads."""
import sys,os,json,hashlib,importlib.util
from pathlib import Path
R=Path(sys.argv[1]);out=Path(sys.argv[2]);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
P=Path('/data/zypan/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1')
sys.path.insert(0,str(P/'src'));import execution_control as C
from record_validation import validate
assert C.verify_sources(P)=='fb034cd68a0b94d511db7559042aba56d12c5b6875a6a23adee79a112eca5172';C.model_guard(P)
for n in ['AIME26_DATASET_FREEZE.json','AIME26_SEED1_PROTOCOL.json','protocol/sampler.py','protocol/aime26_scorer_v4.py','protocol/common.py','analysis/MODEL_TOKENIZER_FRESH_SHA.json']:
    assert C.sha(P/n)==C.sha(R/n),'Protocol/model mismatch:'+n
sp=importlib.util.spec_from_file_location('original_frozen_v4_history',P/'protocol/aime26_scorer_v4.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
sys.path.insert(0,'/data/zypan/transformers-qwen35/src');from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
gold={x['problem_id']:x for x in [json.loads(l) for l in (P/'protocol/aime26_frozen.jsonl').read_text().splitlines() if l.strip()]}
jobs={j['id']:j for j in C.read(P/'FORMAL_JOB_MANIFEST.json')['jobs']};scores=[]
for item in C.read(R/'analysis/INT8_COMPARISON_INPUT_MANIFEST.json')['records']:
    p=P/item['path'];assert p.stat().st_size==item['bytes'] and C.sha(p)==item['sha256']
    wrapper=C.read(p);job=jobs[wrapper['unit']];assert job['condition'] in ('Q1','Q3')
    row=validate(wrapper,job,P);assert row['response']==tok.decode(row['generated_token_ids'],skip_special_tokens=True)
    g=gold[job['problem_id']];x=m.extract_v4(row['response'],row['finish_reason'],g['problem']);correct=bool(m.compare_with_gold(x,g['answer']))
    scores.append(dict(problem_id=job['problem_id'],condition=job['condition'],correct=correct,extracted_answer=x.normalized_value,gpu_uuid=row['gpu_uuid'],input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],seed=row['seed'],generated_token_count=row['generated_token_count'],finish_reason=row['finish_reason'],wall_seconds=row['wall_seconds'],payload_sha256=wrapper['payload_sha256'],raw_sha256=item['sha256']))
C.atomic_new(out,dict(status='PASS_READONLY_ORIGINAL_INT8_MATCHED_SCORE',scores=scores,source_freeze_sha256=C.sha(P/'SOURCE_FREEZE.json'),protocols_equal=True,cross_UUID_bitwise_equivalence_claim=False))
