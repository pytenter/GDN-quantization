#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded CPU identity/prompt audit, no CUDA context or model forward."""
from pathlib import Path
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,time
import runtime as R
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
R.check(not (r/'CPU_PREFLIGHT.json').exists(),'CPU admission already exists')
expected=R.read(r/'EXPECTED_EXTERNAL_SOURCES.json');assert len(expected)==195
for x in expected:
 p=Path(x['path']);R.check(p.stat().st_size==x['bytes'] and R.sha(p)==x['sha256'],'Source SHA changed '+str(p))
for p in (r/'source').glob('*.py'):compile(p.read_bytes(),str(p),'exec')
model=[]
for x in R.read(r/'EXPECTED_MODEL_TOKENIZER_IDENTITY.json')['files']:
 p=Path(x['path']);s=p.stat();digest=R.sha(p);R.check(s.st_size==x['bytes'] and digest==x['sha256'],'Model/tokenizer SHA changed '+str(p));R.check(s.st_mtime_ns==p.stat().st_mtime_ns,'Model changed while hashing');model.append(dict(path=str(p),bytes=s.st_size,sha256=digest,mtime_ns=s.st_mtime_ns,inode=s.st_ino))
identity=dict(task_id=R.TASK,status='PASS',files=model,reference_sources={n:dict(path=str(R.REF/'src'/n),sha256=R.sha(R.REF/'src'/n)) for n in ['core.py','worker.py','codec.py']},resident='FP32 QDQ carrier, not packedINT8',reference_scope='Original 5doc32K strictReference; neither olderfastAIME nor priornativeSTEPcomparison reused as GDN baseline',known_signed_zero='Original dual parity records torch.equal and official_zero_sign_differences separately; no sign-normalization modification',raw_byte_SHA_only=True)
R.atomic_new(r/'MODEL_AND_CODEC_IDENTITY.json',identity)
sys.path.insert(0,'/data/zypan/transformers-qwen35');import torch
from transformers import AutoTokenizer
R.check(not torch.cuda.is_initialized(),'CPU-only audit unexpectedly initialized CUDA')
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
common=dict(hashlib=hashlib,json=json);R.extract(r/'protocol/common.py',{'raw_messages','ids_sha256'},common);ns=dict(torch=torch,**common);R.extract(r/'protocol/sampler.py',{'render_input'},ns)
m=R.read(r/'DATASET_AND_JOB_MANIFEST.json');ds=[json.loads(z) for z in (r/'protocol/aime26_frozen.jsonl').read_text().splitlines()];R.check(len(ds)==30 and len(m['jobs'])==60,'Roster invalid');R.check(len({j['canonical_job_id'] for j in m['jobs']})==60,'Duplicate job')
for i,d in enumerate(ds):
 msgs,ids=ns['render_input'](tok,d['problem']);h=common['ids_sha256'](ids[0].tolist())
 for seed in (1,2):
  j=m['jobs'][i*2+seed-1];R.check(all(j[k]==d[k] for k in ['problem_id','problem_idx','problem','answer']) and j['seed']==seed and j['canonical_index']==i*2+seed-1 and j['input_ids_hash']==h and j['input_tokens']==ids.shape[-1] and j['gpu']==j['canonical_index']%4,'Canonical prompt/order/seed/GPU mismatch')
R.check(R.sha(r/'protocol/aime26_scorer_v4.py')=='fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b','V4 source mismatch');s=R.load_module('cpu_frozen_v4',r/'protocol/aime26_scorer_v4.py');R.check(s.compare_with_gold(s.extract_v4('Thus the answer is \\boxed{42}.','eos','A non-AIME CPU identity fixture.'),'42'),'V4 interface changed')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu,power.limit,temperature.gpu','--format=csv,noheader,nounits'],text=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True);R.check(not apps.strip(),'No eligiblefree4GPUset: do not preempt')
for line in rows.splitlines():
 z=[x.strip() for x in line.split(',')];R.check(z[1]==m['gpu_uuid_map'][z[0]] and '3090' in z[2] and float(z[3])>23000,'3090 UUID/free memory failed')
R.check(shutil.disk_usage(r).free>4*2**30,'Insufficient disk for small raw results/logs')
R.atomic_new(r/'GPU_ADMISSION.json',dict(task_id=R.TASK,status='PASS',GPU_rows=rows,active_processes=apps,eligible_GPUs=[0,1,2,3],no_preemption=True,disk_free_bytes=shutil.disk_usage(r).free,unix=time.time()))
R.atomic_new(r/'CPU_PREFLIGHT.json',dict(task_id=R.TASK,status='PASS',source_fresh_SHA_checks=len(expected),model_tokenizer_raw_SHA_files=len(model),canonical30_re_rendered=True,unique_jobs=60,scorer_SHA_exact=True,sampler_source_SHA_exact=True,strict_reference_SHA_exact=True,no_GPU_forward=True,CUDA_initialized=torch.cuda.is_initialized(),runtime=dict(python=sys.version.split()[0],torch=torch.__version__,CUDA_build=torch.version.cuda,transformers=__import__('transformers').__version__),scope='GateA/physicalidle preflight only; GateB/C actualGPU stillNOT_RUN'))
print(json.dumps(dict(status='PASS_CPU_GATE_A_GPU_IDLE',jobs=60,model_SHA_files=len(model),source_SHA_checks=len(expected))))
