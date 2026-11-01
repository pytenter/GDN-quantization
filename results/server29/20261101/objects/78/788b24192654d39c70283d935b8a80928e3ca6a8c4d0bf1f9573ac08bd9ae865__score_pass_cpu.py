"""One independent CPU-only FrozenV4 pass over the exact90 legal payloads."""
import os,sys,json,importlib.util
from pathlib import Path
R=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
C.require(C.read(R/'FORMAL_GENERATIONS_COMPLETE.json')['count']==90,'All90 completion receipt required')
C.verify_sources(R);C.model_guard(R)
p=R/'protocol/aime26_scorer_v4.py';sp=importlib.util.spec_from_file_location('original_frozen_v4',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
gold={r['problem_id']:r for r in [json.loads(l) for l in (R/'protocol/aime26_frozen.jsonl').read_text().splitlines() if l.strip()]};rows=[]
# Tokenizer re-decode is CPU-only and checks the same recorded generation payload.
sys.path.insert(0,'/data/zypan/transformers-qwen35/src');from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
for job in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']:
 wrapper=C.read(R/'formal'/job['id']/'RESULT.json');row=validate(wrapper,job,R);C.require(row['response']==tok.decode(row['generated_token_ids'],skip_special_tokens=True),'Saved response differs from exact sampledtoken decode');g=gold[job['problem_id']];x=m.extract_v4(row['response'],row['finish_reason'],g['problem']);correct=bool(m.compare_with_gold(x,g['answer']))
 rows.append(dict(sample_id=job['id'],problem_id=job['problem_id'],condition=job['condition'],seed=1,gpu=job['gpu'],gpu_uuid=row['gpu_uuid'],correct=correct,extracted_answer=x.normalized_value,extraction=x.to_dict(),gold=g['answer'],generated_token_count=row['generated_token_count'],finish_reason=row['finish_reason'],finite=True,wall_seconds=row['wall_seconds'],TPS=row['TPS'],payload_sha256=wrapper['payload_sha256']))
C.require(len(rows)==90,'Incomplete score pass');C.atomic_new(out,dict(status='PASS_ORIGINAL_FROZEN_V4_90',scorer_sha256=C.sha(p),scores=rows))
