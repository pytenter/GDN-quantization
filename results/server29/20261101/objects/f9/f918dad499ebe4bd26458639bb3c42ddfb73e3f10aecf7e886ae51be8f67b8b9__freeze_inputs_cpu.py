"""Offline tokenizer/source/model identity and canonical30 freeze, zero GPU."""
import os,sys,json,hashlib,ast,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),'/data/zypan/transformers-qwen35/src']
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
torch.cuda._lazy_init=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('GPU forbidden'))
from transformers import AutoTokenizer
from protocol_binding import sampler
f,rows=sampler(R,torch)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
identity=json.loads((R/'evidence/HISTORICAL_AIME_MODEL_AND_CODEC_IDENTITY.json').read_text());modelrows=[]
for x in identity['files']:
 p=Path(x['path']);s=p.stat();assert s.st_size==x['bytes'];a=h(p);assert a==x['sha256'],str(p)
 modelrows.append(dict(path=str(p),bytes=s.st_size,sha256=a,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',local_files_only=True,trust_remote_code=True)
dataset=[json.loads(l) for l in (R/'protocol/aime26_frozen.jsonl').read_text().splitlines() if l.strip()];assert len(dataset)==30 and [int(x['problem_idx']) for x in dataset]==list(range(1,31))
old=json.loads((R/'evidence/HISTORICAL_AIME_DATASET_AND_JOB_MANIFEST.json').read_text());oldby={x['problem_id']:x for x in old['jobs'] if x['seed']==1}
uuids=['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']
freeze=[]
for row in dataset:
 pid=row['problem_id'];problem=row['problem'];messages,ids=f.render_input(tok,problem);raw=ids[0].tolist();prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
 assert f.ids_sha256(raw)==oldby[pid]['input_ids_hash'] and len(raw)==oldby[pid]['input_tokens'] and hashlib.sha256(prompt.encode()).hexdigest()==oldby[pid]['prompt_text_SHA256']
 freeze.append(dict(problem_id=pid,problem_idx=int(row['problem_idx']),problem=problem,problem_sha256=hashlib.sha256(problem.encode()).hexdigest(),prompt_text=prompt,prompt_text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids=raw,input_ids_sha256=f.ids_sha256(raw),input_tokens=len(raw)))
# Fixed salt+ID rank chooses bundles without using text, gold, answers or outcomes.
ranked=sorted(freeze,key=lambda x:hashlib.sha256(('QWEN_GDN_GROUP32_FOUR_CODEC_AIME26_SEED1_V1\0'+x['problem_id']).encode()).hexdigest());assignment=[]
for rank,x in enumerate(ranked):
 gpu=rank%4;assignment.append(dict(problem_id=x['problem_id'],hash_rank=rank,gpu=gpu,gpu_uuid=uuids[gpu],conditions=['Q0','Q1','Q2','Q3'],seed=1))
assignment.sort(key=lambda x:next(r['problem_idx'] for r in freeze if r['problem_id']==x['problem_id']))
protocol=dict(task_id=R.name,seed=1,enable_thinking=True,do_sample=True,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=1.5,repetition_penalty=1.,presence_penalty_scope='generated_output_token_boolean_presence_only',warper_order=['Temperature','TopK','TopP'],max_new_tokens=81920,total_context_required=max(x['input_tokens'] for x in freeze)+81920,prompt_QDQ='one-token forward then QDQ/writeback for every prompt token including token0',decode_QDQ='one-token forward then QDQ/writeback before next recurrent input',expected_forward_count_formula='prompt_tokens+generated_tokens-1',expected_layer_QDQ_formula='24*(prompt_tokens+generated_tokens-1)',sampler_source_sha256=h(R/'protocol/sampler.py'),sampler_AST=rows,scorer_source_sha256=h(R/'protocol/aime26_scorer_v4.py'),min_p_zero_implementation='original no-op; no new warper',runtime_protocol='HISTORICAL_PORT',deterministic_algorithms=False,sampler_exception_scope='None required under qualified HISTORICAL_PORT; same original sampler all four methods',historical_accuracy_comparability='BACKGROUND_ONLY_DIFFERENT_PREFILL_AND_RUNTIME',test_history='HISTORICALLY_OBSERVED_AIME26_NOT_VIRGIN_TEST_SET',free_generation_long_context_claim='PENDING_ACTUAL_GPU_QUALIFICATION',GPU_authorization='PENDING_THIS_TASK_EXPLICIT_GRANT')
objects={'AIME26_DATASET_FREEZE.json':dict(status='FROZEN_CPU_TOKENIZER_EXACT_CANONICAL30',dataset_sha256=h(R/'protocol/aime26_frozen.jsonl'),tokenizer_model_identity='analysis/MODEL_TOKENIZER_FRESH_SHA.json',prompt_count=30,records=freeze),'QUESTION_GPU_ASSIGNMENT.json':dict(status='FROZEN_BEFORE_ANY_GPU_OR_NEW_GENERATION',rule='SHA256(task_id+NUL+problem_id) rank, rank%4',bundle_order=['Q0','Q1','Q2','Q3'],counts={str(i):sum(a['gpu']==i for a in assignment) for i in range(4)},assignments=assignment),'AIME26_SEED1_PROTOCOL.json':protocol,'analysis/MODEL_TOKENIZER_FRESH_SHA.json':dict(status='PASS_ALL_MODEL_TOKENIZER_FRESH_SHA_CPU_READONLY',files=modelrows,GPU_dispatches=0)}
for name,obj in objects.items():
 with (R/name).open('x') as w:json.dump(obj,w,indent=2,ensure_ascii=False);w.write('\n')
print(json.dumps(dict(status='CPU_PROMPT_MODEL_FREEZE_PASS',prompts=len(freeze),max_prompt_tokens=max(x['input_tokens'] for x in freeze),context_required=protocol['total_context_required'],assignment_counts=objects['QUESTION_GPU_ASSIGNMENT.json']['counts'],GPU_dispatches=0)))
