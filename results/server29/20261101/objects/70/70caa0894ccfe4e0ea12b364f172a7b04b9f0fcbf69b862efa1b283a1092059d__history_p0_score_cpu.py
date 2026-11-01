"""Fixed original P0 payload list, actual FrozenV4 re-score and CPU token decode."""
import sys,json,importlib.util
from pathlib import Path
R=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
manifest=C.read(R/'analysis/HISTORICAL_P0_INPUT_MANIFEST.json');P=Path(manifest['root']);spec=importlib.util.spec_from_file_location('original_v4',R/'protocol/aime26_scorer_v4.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
sys.path.insert(0,'/data/zypan/transformers-qwen35/src');from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
gold={r['problem_id']:r for r in [json.loads(l) for l in (R/'protocol/aime26_frozen.jsonl').read_text().splitlines() if l.strip()]};new_inputs={r['problem_id']:r for r in C.read(R/'AIME26_DATASET_FREEZE.json')['records']};jobs={j['id']:j for j in C.read(P/'FORMAL_JOB_MANIFEST.json')['jobs']};rows=[]
for x in manifest['files']:
 p=P/x['path'];C.require(p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256'],'Historical P0 payload changed');wrapper=C.read(p);job=jobs[p.parent.name];C.require(job['condition']=='Q2','Historical P0 must actual H128Sym32');row=validate(wrapper,job,P);C.require(row['response']==tok.decode(row['generated_token_ids'],skip_special_tokens=True),'Historical P0 token decode differs');C.require(row['input_ids_sha256']==new_inputs[row['problem_id']]['input_ids_sha256'],'Historical P0 prompt identity differs');g=gold[row['problem_id']];v=m.extract_v4(row['response'],row['finish_reason'],g['problem']);correct=bool(m.compare_with_gold(v,g['answer']))
 rows.append(dict(problem_id=row['problem_id'],condition='P0',original_condition='Q2',correct=correct,extracted_answer=v.normalized_value,generated_token_count=row['generated_token_count'],finish_reason=row['finish_reason'],gpu_uuid=row['gpu_uuid'],input_ids_sha256=row['input_ids_sha256'],payload_sha256=wrapper['payload_sha256']))
new=C.read(R/'AIME26_SEED1_PROTOCOL.json');old=C.read(P/'AIME26_SEED1_PROTOCOL.json');keys=['seed','enable_thinking','do_sample','temperature','top_p','top_k','min_p','presence_penalty','presence_penalty_scope','repetition_penalty','warper_order','max_new_tokens','prompt_QDQ','decode_QDQ','sampler_source_sha256','scorer_source_sha256'];diff=[k for k in keys if new[k]!=old[k]]
C.atomic_new(out,dict(status='PASS_FIXED_HISTORICAL_P0_CPU_RESCORE',scores=rows,protocol_fields_compared=keys,protocol_differences=diff,historical_observational=True,cross_device_bitwise_equivalence=False))
