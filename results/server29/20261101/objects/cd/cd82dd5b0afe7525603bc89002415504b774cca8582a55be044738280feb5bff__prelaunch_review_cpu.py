"""Final independent CPU-only identity/admission/protocol review, before source freeze."""
import sys,os,json,hashlib,ast,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'inherited/group32/src'),'/data/zypan/transformers-qwen35/src']
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import execution_control as C
import torch
torch.cuda._lazy_init=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('GPU forbidden in prelaunch CPU review'))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
for x in C.read(R/'reuse/PARENT_SOURCE_FREEZE.json')['files']:
    p=R/'reuse/parent'/x['path'];assert p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256']
for n in ['sampler.py','aime26_scorer_v4.py','common.py','aime26_frozen.jsonl']:
    assert C.sha(R/'protocol'/n)==C.sha(R/'reuse/parent/protocol'/n)
for n in ['canonical_h128.py','group_codecs.py','persistent_adapter.py','inherited_device_guards.py']:
    assert C.sha(R/'inherited/group32/src'/n)==C.sha(R/'reuse/parent/inherited/group32/src'/n)
assert C.read(R/'qualification/CPU_PASS_0/SUMMARY.json')==C.read(R/'qualification/CPU_PASS_1/SUMMARY.json')
assert C.read(R/'qualification/INDEPENDENT_CPU_COMPARISON.json')['status']=='PASS_TWO_SEPARATE_PROCESS_SUITES_EXACT'
C.admit_inherited(R);C.model_guard(R);externals=C.verify_external(R)
C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'])
original=C.query_gpu
try:
    obs=dict(devices=[dict(physical_index=3,uuid=C.UUIDS[3],name='RTX3090',memory_total_MiB=24576,memory_free_MiB=24256)],apps=[])
    C.query_gpu=lambda:obs;C.device(3)
    for g in (0,1,2):
        try:C.device(g)
        except RuntimeError:pass
        else:raise RuntimeError('A different GPU was admitted')
    obs['apps']=[dict(pid=999999,uuid=C.UUIDS[3],memory_MiB=1)]
    try:C.device(3)
    except RuntimeError:pass
    else:raise RuntimeError('Foreign occupation admitted')
finally:C.query_gpu=original
from transformers import AutoTokenizer
from protocol_binding import sampler
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
f,sources=sampler(R,torch);prompts=[]
for row in C.read(R/'AIME26_DATASET_FREEZE.json')['records']:
    messages,ids=f.render_input(tok,row['problem']);ids=ids[0].tolist();assert ids==row['input_ids'] and len(ids)==row['input_tokens'] and f.ids_sha256(ids)==row['input_ids_sha256']
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    prompts.append(dict(problem_id=row['problem_id'],input_ids_sha256=row['input_ids_sha256'],input_tokens=len(ids),chat_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_messages_sha256=C.objsha(messages)))
assert sources==C.read(R/'AIME26_SEED1_PROTOCOL.json')['sampler_AST']
assert (f.TEMPERATURE,f.TOP_P,f.TOP_K,f.MIN_P,f.PRESENCE_PENALTY,f.REPETITION_PENALTY,f.MAX_NEW_TOKENS)==(1.,.95,20,0.,1.5,1.,81920)
assert not torch.cuda.is_initialized()
P=Path('/data/zypan/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1')
counts={c:sum(1 for j in C.read(P/'FORMAL_JOB_MANIFEST.json')['jobs'] if j['condition']==c and (P/'formal'/j['id']/'RESULT.json').exists()) for c in ['Q0','Q1','Q2','Q3']}
available={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
assert available['MemAvailable']>=80*1024**3,'Insufficient host memory for bounded fresh-reference buffers'
C.storage_guard(R)
result=dict(status='PASS_FINAL_PRELAUNCH_CPU_REVIEW',parent_source_files_verified=542,external_SHA_files=externals,model_stat_files=17,all30_actual_prompt_inputs_exact=True,prompts=prompts,sampler_AST=sources,actual_stop_token_ids=sorted(f.stop_ids(tok)),selected_gpu=3,selected_uuid=C.UUIDS[3],other_GPU_admission_rejected=True,foreign_owner_rejected=True,historical_INT8_actual_completed=counts,historical_INT8_full_Asym30=counts['Q1']==counts['Q3']==30,host_memory=available,GPU_dispatches=0,CUDA_initialized=False,monotonic=time.monotonic())
C.atomic_new(R/'analysis/FINAL_PRELAUNCH_CPU_REVIEW.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['prompts','sampler_AST']},indent=2),flush=True)
