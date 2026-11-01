"""One fresh CPU original FrozenV4 pass over exactly120 completed payloads."""
import os,sys,json,importlib.util
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(R/'src'));import execution_control as C
from cross_host_contract import validate_formal_record,validate_global_layout
layout=validate_global_layout(R);nodes=layout['nodes']
def source_for(job):return R if job['node_id']=='server29_gpu6' else R/'peer55'
def validate(w,j,root):return validate_formal_record(w,j,root,nodes[j['node_id']])
C.require(C.read(R/'GLOBAL_FORMAL_GENERATIONS_COMPLETE.json')['count']==120,'120 completion receipt required');C.verify_sources(R);C.verify_sources(R/'peer55');C.model_guard(R)
p=R/'protocol/aime26_scorer_v4.py';sp=importlib.util.spec_from_file_location('original_frozen_v4',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
gold={r['problem_id']:r for r in [json.loads(l)for l in (R/'protocol/aime26_frozen.jsonl').read_text().splitlines()if l.strip()]}
sys.path.insert(0,str(R/'runtime/transformers-qwen35/src'));from transformers import AutoTokenizer
import torch
C.require(not torch.cuda.is_initialized(),'CPU scorer initialized CUDA')
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True);rows=[]
for j in C.read(R/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs']:
 source=source_for(j);w=C.read(source/'formal'/j['id']/'RESULT.json');x=validate(w,j,source);interval=C.read(source/'formal'/j['id']/'RESOURCE_INTERVAL.json');C.require(interval['exit_returncode']==0 and interval['release']=='PASS' and interval['identity']==w['identity'],'Actual exit/release mismatch')
 C.require(x['response']==tok.decode(x['generated_token_ids'],skip_special_tokens=True),'Exact samepayload token decode differs');g=gold[j['problem_id']];answer=m.extract_v4(x['response'],x['finish_reason'],g['problem']);correct=bool(m.compare_with_gold(answer,g['answer']))
 rows.append(dict(sample_id=j['id'],problem_id=j['problem_id'],condition=j['condition'],seed=1,gpu=j['gpu'],gpu_uuid=x['gpu_uuid'],compute_host=j['compute_host'],node_id=j['node_id'],correct=correct,extracted_answer=answer.normalized_value,extraction=answer.to_dict(),gold=g['answer'],generated_token_count=x['generated_token_count'],finish_reason=x['finish_reason'],wall_seconds=x['wall_seconds'],TPS=x['TPS'],prompt_seconds=x['prompt_seconds'],decode_and_sampling_seconds=x['decode_and_sampling_seconds'],input_tokens=x['input_tokens'],model_load_seconds=x['environment']['model_load_seconds'],peak_allocated_bytes=x['peak_allocated_bytes'],peak_reserved_bytes=x['peak_reserved_bytes'],payload_sha256=w['payload_sha256']))
C.require(len(rows)==120 and not torch.cuda.is_initialized(),'CPU scoring cardinality/CUDA differs');C.atomic_new(out,dict(status='PASS_ORIGINAL_FROZEN_V4_120',scorer_sha256=C.sha(p),scores=rows))
