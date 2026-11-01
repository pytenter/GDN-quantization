"""Prepare fixed admitted identities/units on CPU; never executes a GPU."""
from pathlib import Path
import json,hashlib,sys,os
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),'/data/zypan/transformers-qwen35/src']
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from transformers import AutoTokenizer
from protocol_binding import sampler
import torch
f,_=sampler(R,torch);tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',local_files_only=True,trust_remote_code=True)
text='Explain how to verify a checksum and describe a small example of checking a file for accidental corruption. Give the reasoning before the conclusion.'
_,ids=f.render_input(tok,text)
obj=dict(problem=text,problem_id='NON_AIME_CHECKSUM_EXPLANATION_FROZEN_V1',input_ids=ids[0].tolist(),input_ids_sha256=f.ids_sha256(ids[0].tolist()),purpose='NonAIME real free sampler, stepwiseThinking prompt, legalEOS/length; not an accuracy benchmark')
with (R/'protocol/FREE_CANARY_INPUT.json').open('x') as w:json.dump(obj,w,indent=2);w.write('\n')
p=R/'GPU_EXECUTION_PLAN.json';plan=json.loads(p.read_text());(R/'evidence/GPU_EXECUTION_PLAN_PRE_GRANT.json').write_bytes(p.read_bytes());plan['status']='AUTHORIZED_FINITE_ROSTER_BEFORE_ANY_GPU';plan['GPU_execution']=True
for u in plan['qualification_units']:
 if 'repeat' in u:u['reference_unit']=u['id'].replace('_R1','_R0')
 if u['kind']=='nonAIME_free_generation_fresh':u['problem_id']=obj['problem_id']
plan['temporary_reference_contract']=dict(root='/dev/shm/qwen_group32_four_codec_aime26_seed1_v1',per_reference_cap_bytes=80*1024**3,global_admission_free_bytes=100*1024**3,ongoing_tmpfs_margin_bytes=20*1024**3,purpose='Full fresh actual-value/byte comparison buffers only, never recurrent continuation',delete_only_after='PairedPASS+durable completeSHA/layout/byte-comparison receipts',on_failure='Preserve entire temp comparison scope plus durable full first difference under2GiB failure bound; no GPU retry',permanent_large_reference_rearchive=False)
p.write_text(json.dumps(plan,indent=2)+'\n')
records={r['problem_id']:r for r in json.loads((R/'AIME26_DATASET_FREEZE.json').read_text())['records']};jobs=[]
for assignment in json.loads((R/'QUESTION_GPU_ASSIGNMENT.json').read_text())['assignments']:
 for q in ['Q0','Q1','Q2','Q3']:
  r=records[assignment['problem_id']];jobs.append(dict(id=r['problem_id']+'_'+q+'_seed1',problem_id=r['problem_id'],condition=q,seed=1,gpu=assignment['gpu'],gpu_uuid=assignment['gpu_uuid'],input_ids_sha256=r['input_ids_sha256'],input_tokens=r['input_tokens'],kind='formal_free_generation',max_new_tokens=81920))
assert len(jobs)==120
with (R/'FORMAL_JOB_MANIFEST.json').open('x') as w:json.dump(dict(task_id=R.name,jobs=jobs,count=120,question_bundle_same_UUID=True,fresh_process_each_sample=True),w,indent=2);w.write('\n')
print(json.dumps(dict(status='GPU_ROSTER_AND_NONAIME_INPUT_PREPARED_CPU_ONLY',qualification_units=len(plan['qualification_units']),formal_samples=len(jobs),nonAIME_prompt_tokens=len(obj['input_ids']),GPU_dispatches=0)))
