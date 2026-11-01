#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse,json,subprocess,sys,time
import runtime as R
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root;parent=r.parent
assert not (r/'CPU_PREFLIGHT.json').exists()
for p in (r/'source').glob('*.py'):compile(p.read_bytes(),str(p),'exec')
old=R.read(parent/'STOP_DELIVERY_MANIFEST.json');ledger=R.read(parent/'RESOURCE_LEDGER.json');assert ledger['status']=='FAIL_CLOSED_STOP' and all(x['process_exited'] and x['NVML_released'] for x in ledger['dispatches'])
for x in old['files']:
 p=parent/x['path'];assert p.stat().st_size==x['bytes'] and R.sha(p)==x['sha256']
for x in R.read(r/'EXPECTED_EXTERNAL_SOURCES.json'):
 p=Path(x['path']);assert p.stat().st_size==x['bytes'] and R.sha(p)==x['sha256']
for x in R.read(r/'MODEL_AND_CODEC_IDENTITY.json')['files']:
 st=Path(x['path']).stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns'] and st.st_ino==x['inode']
R.atomic_new(r/'PARENT_FAILURE_IDENTITY.json',dict(status='PASS_IMMUTABLE_PARENT_STOP',parent_root=str(parent),parent_stop_manifest_sha256=R.sha(parent/'STOP_DELIVERY_MANIFEST.json'),parent_source_sha256=R.sha(parent/'SOURCE_SHA256_MANIFEST.json'),parent_ledger_sha256=R.sha(parent/'RESOURCE_LEDGER.json'),GPU_seconds=ledger['cumulative_GPU_seconds'],GPU_hours=ledger['cumulative_GPU_hours'],completed=0,initial17files_full_byte_SHA_reused_with_fresh_stat_identity=True))
import torch
assert not torch.cuda.is_initialized();torch.use_deterministic_algorithms(True)
checks=[]
def observes():assert not torch.are_deterministic_algorithms_enabled();return 42
assert R.original_native_sampler_scope(torch,observes)==42 and torch.are_deterministic_algorithms_enabled();checks.append('delegation_scope_and_restore')
def fails():assert not torch.are_deterministic_algorithms_enabled();raise ValueError('injected CPU exception restoration fixture')
try:R.original_native_sampler_scope(torch,fails)
except ValueError:pass
else:raise AssertionError('Exception swallowed')
assert torch.are_deterministic_algorithms_enabled();checks.append('finally_restore_after_exception')
sys.path.insert(0,'/data/zypan/transformers-qwen35');from transformers import AutoTokenizer
ns=dict(torch=torch);R.extract(r/'protocol/sampler.py',{'sample_next','build_warpers','TEMPERATURE','TOP_P','TOP_K','MIN_P','PRESENCE_PENALTY','REPETITION_PENALTY'},ns);warpers=ns['build_warpers']();logits=torch.linspace(-3,4,257)[None,:];mask=torch.zeros_like(logits,dtype=torch.bool)
for seed in (1,2):
 g=torch.Generator(device='cpu').manual_seed(seed);ref=torch.Generator(device='cpu').manual_seed(seed);left=[];right=[]
 for _ in range(8):
  t,n=R.original_native_sampler_scope(torch,ns['sample_next'],warpers,logits,mask,g);z,k=R.original_native_sampler_scope(torch,ns['sample_next'],warpers,logits,mask,ref);assert torch.equal(t,z) and n==k;left.append(int(t.item()));right.append(int(z.item()))
 assert left==right and torch.equal(g.get_state(),ref.get_state());checks.append('original_CPU_sampler_seed'+str(seed)+'_delegation_exact')
assert not torch.cuda.is_initialized()
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',local_files_only=True,trust_remote_code=True)
common=dict(hashlib=__import__('hashlib'),json=json);R.extract(r/'protocol/common.py',{'raw_messages','ids_sha256'},common);ns=dict(torch=torch,**common);R.extract(r/'protocol/sampler.py',{'render_input'},ns)
for j in R.read(r/'DATASET_AND_JOB_MANIFEST.json')['jobs']:
 _,ids=ns['render_input'](tok,j['problem']);assert common['ids_sha256'](ids[0].tolist())==j['input_ids_hash'] and ids.shape[-1]==j['input_tokens']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True);assert not apps.strip()
R.atomic_new(r/'GPU_ADMISSION.json',dict(status='PASS',active_processes=apps,GPUs=[0,1,2,3],parent_GPU_identity_reused=True,unix=time.time()))
R.atomic_new(r/'CPU_PREFLIGHT.json',dict(task_id=R.TASK,status='PASS',repair_count=1,scope='Original native sampler preserved under bounded scope; all strict GDN/H/QDQ config/math unchanged',checks=checks,canonical30_prompts_reverified=True,source_fresh_SHA_checks=195,prior_full17model_SHA_receipt_fresh_stat_guards=True,no_CUDA_context=torch.cuda.is_initialized(),sampler_SHA=R.sha(r/'protocol/sampler.py'),GDN_core_SHA=R.sha(R.REF/'src/core.py'),codec_SHA=R.sha(R.REF/'src/codec.py')))
print(json.dumps(dict(status='PASS_CPU_REPAIR_ADMISSION',checks=checks,prior_GPUh=ledger['cumulative_GPU_hours'])))
