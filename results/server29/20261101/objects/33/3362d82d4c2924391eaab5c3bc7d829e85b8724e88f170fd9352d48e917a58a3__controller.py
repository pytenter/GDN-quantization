#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline lifecycle/accounting using prior mature identity/NVML contracts; one arm."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import runtime as R
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
def identity(pid):
 try:
  p=Path('/proc')/str(pid);return dict(pid=pid,start_tick=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],cmdline=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
def same(x):
 y=identity(x['pid']);return y is not None and y==x
def nvml():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=10)
 return [{'pid':int(z[0]),'uuid':z[1].strip()} for line in s.splitlines() if line.strip() for z in [line.split(',')]]
def tail(p):
 if not p.exists():return []
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-50000));s=f.read().decode(errors='replace')
 out=[]
 for line in s.splitlines():
  try:out.append(json.loads(line))
  except (ValueError,TypeError):pass
 return out
def release_owned(ds):
 for d in ds:
  if same(d['identity']):os.kill(d['pid'],signal.SIGTERM)
 until=time.monotonic()+15
 while time.monotonic()<until and any(same(d['identity']) for d in ds):time.sleep(.5)
 for d in ds:
  if same(d['identity']):os.kill(d['pid'],signal.SIGKILL)
def update(r,ledger,live,phase):
 now=time.time()
 for d in ledger['dispatches']:
  if not d.get('process_exited') and not same(d['identity']) and d['pid'] not in {x['pid'] for x in live}:
   d.update(process_exited=True,NVML_released=True,exit_and_release_unix=now,occupied_seconds=now-d['dispatch_unix'])
 seconds=sum(d.get('occupied_seconds',now-d['dispatch_unix']) for d in ledger['dispatches']);ledger.update(status=phase,cumulative_GPU_seconds=seconds,cumulative_GPU_hours=seconds/3600,updated_unix=now)
 R.atomic_status(r/'RESOURCE_LEDGER.json',ledger)
 if phase!='FORMAL_RUNNING':return
 jobs=R.read(r/'DATASET_AND_JOB_MANIFEST.json')['jobs'];done={};running=[];workers=[]
 for j in jobs:
  p=r/'raw_results'/(j['canonical_job_id']+'.json')
  if p.exists():z=R.validate_record(R.read(p),j,r);done[j['canonical_job_id']]=z
 for g in range(4):
  p=r/'status'/f'worker{g}.json';s=R.read(p) if p.exists() else dict(status='NOT_STARTED',gpu=g);workers.append(s)
  if s.get('status')=='RUNNING' and s.get('job'):
   latest=next((z for z in reversed(tail(r/'logs'/f'formal_gpu{g}_attempt_{s.get("attempt",0):02d}.log')) if z.get('event')=='PROGRESS' and z.get('canonical_index')==s.get('canonical_index')),None)
   if latest:running.append(dict(job=s['job'],gpu=g,output_tokens=latest['output_tokens'],wall_seconds=latest['wall_seconds'],tokens_per_second=latest['output_tokens']/latest['wall_seconds']))
   else:running.append(dict(job=s['job'],gpu=g,output_tokens=0))
 completed_tokens=sum(x['generated_token_count'] for x in done.values());partial_tokens=sum(x['output_tokens'] for x in running);rates=[x['tokens_per_second'] for x in running if x.get('tokens_per_second')]
 n=len(done);rate=sum(rates)
 rough=None if rate<=0 else dict(scope='Conditional only, unfinished jobs assumed30000tokens; not known outputs or promised ETA',assumed_output_tokens_per_job=30000,observed_active_prefix_TPS=sum(rates),remaining_seconds_scenario=max(0,(60-n)*30000-partial_tokens)/rate,all_remaining_hit_cap_seconds_scenario=max(0,(60-n)*81920-partial_tokens)/rate)
 R.atomic_status(r/'WORKER_STATUS.json',dict(status='FORMAL_RUNNING',workers=workers))
 R.atomic_status(r/'FORMAL_PROGRESS.json',dict(task_id=R.TASK,status='FORMAL_RUNNING',total_jobs=60,completed=n,running=len(running),pending=60-n-len(running),failed=sum(x.get('status')=='FAILED' for x in workers),seed1_completed=sum(x['seed']==1 for x in done.values()),seed2_completed=sum(x['seed']==2 for x in done.values()),running_jobs=running,active_GPU_UUIDs=[x['uuid'] for x in live if x['pid'] in {d['pid'] for d in ledger['dispatches']}],completed_generated_tokens=completed_tokens,total_generated_tokens_including_logged_partial=completed_tokens+partial_tokens,actual_GPU_hours=seconds/3600,estimated_remaining_time=rough,updated_unix=now))
def stage(r,ledger,mode,repeat=0,resume=False):
 mapping=R.read(r/'DATASET_AND_JOB_MANIFEST.json')['gpu_uuid_map'];phase='FORMAL_RUNNING' if mode=='formal' else 'CANARY_RUNNING';own=[];handles=[];procs={};logs={}
 existing=[d for d in ledger['dispatches'] if d['mode']==mode and d['repeat']==repeat]
 for g in range(4):
  older=[d for d in existing if d['gpu']==g]
  if older and same(older[-1]['identity']):own.append(older[-1]);logs[g]=r/older[-1]['log'];continue
  if older and older[-1].get('returncode')==0:continue
  if older and not resume:raise RuntimeError('Unit already dispatched; no automatic retry')
  actual=nvml();R.check(not any(x['uuid']==mapping[str(g)] for x in actual),'GPU occupied, no preemption/migration')
  attempt=len(older);log=r/'logs'/f'{mode}_gpu{g}_attempt_{attempt:02d}.log';log.parent.mkdir(exist_ok=True);h=log.open('xb');handles.append(h)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WORKER_ATTEMPT=str(attempt),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  # Canary repeat paths are separate; log filename includes repeat to avoid collisions.
  if mode=='canary':
   h.close();log.unlink();log=r/'logs'/f'canary_gpu{g}_r{repeat}.log';h=log.open('xb');handles[-1]=h
  command=[PY,'-B',str(r/'source/worker.py'),'--root',str(r),'--gpu',str(g),'--mode',mode,'--repeat',str(repeat)]
  dispatch=time.time();p=subprocess.Popen(command,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  ident=None
  for _ in range(20):
   ident=identity(p.pid)
   if ident:break
   time.sleep(.01)
  R.check(ident is not None,'No true dispatched process identity')
  d=dict(mode=mode,repeat=repeat,gpu=g,gpu_uuid=mapping[str(g)],pid=p.pid,identity=ident,attempt=attempt,log=str(log.relative_to(r)),dispatch_unix=dispatch,process_exited=False,NVML_released=False);ledger['dispatches'].append(d);own.append(d);procs[p.pid]=p;logs[g]=log
  R.atomic_status(r/'RESOURCE_LEDGER.json',ledger)
 try:
  while any(not d.get('process_exited') for d in own):
   live=nvml()
   for d in own:
    pp=procs.get(d['pid'])
    if pp is not None and pp.poll() is not None:d['returncode']=pp.returncode
    if not same(d['identity']) and d['pid'] not in {x['pid'] for x in live}:
     d.update(process_exited=True,NVML_released=True,exit_and_release_unix=time.time(),occupied_seconds=time.time()-d['dispatch_unix'])
     path=r/'execution'/(f'canary/gpu{d["gpu"]}_r{repeat}' if mode=='canary' else f'formal/gpu{d["gpu"]}')/f'attempt_{d["attempt"]:02d}'/'RESULT.json'
     if 'returncode' not in d:d['returncode']=0 if path.exists() and R.read(path)['status']=='PASS' else -1
     
     if mode=='formal' and d['returncode'] in (-9,-15,-1) and not path.exists() and not (r/'FIRST_FAILURE.json').exists():raise InterruptedError('Formal worker interrupted without a completed receipt: '+str(d['gpu']))
     R.check(d['returncode']==0 and path.exists() and R.read(path)['status']=='PASS','Worker ended without legal PASS; inspect logs, no blind retry')
    elif time.time()-logs[d['gpu']].stat().st_mtime>1800:raise RuntimeError('30minute no-progress watchdog: '+str(d['gpu']))
   update(r,ledger,live,phase)
   if (r/'FIRST_FAILURE.json').exists():raise RuntimeError('Real worker failure: stop affected strict path')
   if (r/'STOP_REQUESTED.json').exists():raise RuntimeError('Explicit stop request')
   time.sleep(5)
 finally:
  for h in handles:h.close()
 return own

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();r=a.root
 R.verify(r);lock=r/'execution/COORDINATOR.json'
 if lock.exists():R.check(not same(R.read(lock)['identity']),'Coordinator already active')
 if not a.resume:R.check(not lock.exists(),'Already started; use explicit task-level resume')
 R.check(not (r/'FIRST_FAILURE.json').exists(),'Prior true failure, not automatically retryable')
 R.atomic_status(lock,dict(task_id=R.TASK,identity=identity(os.getpid()),unix=time.time(),resume=a.resume))
 ledger=R.read(r/'RESOURCE_LEDGER.json') if (r/'RESOURCE_LEDGER.json').exists() else dict(task_id=R.TASK,dispatches=[],accounting='dispatch through trueprocess exit and NVMLrelease includingloads,warmup,canary,formal; noGPUhcap',MAX_ACTIVE_GPUS=4)
 try:
  if not (r/'CANARY_QUALIFICATION.json').exists():
   for rep in (0,1):stage(r,ledger,'canary',rep,a.resume)
   receipts=[]
   for g in range(4):
    x=R.read(r/f'execution/canary/gpu{g}_r0/attempt_00/RESULT.json');y=R.read(r/f'execution/canary/gpu{g}_r1/attempt_00/RESULT.json');R.check(x['trace']==y['trace'],'Fresh exact trace failed');receipts.append(dict(gpu=g,gpu_uuid=x['gpu_uuid'],status='PASS',same_GPU_fresh_exact=True,trace_sha256=x['trace_sha256'],peak_allocated_bytes=max(x['peak_allocated_bytes'],y['peak_allocated_bytes'])))
   R.atomic_new(r/'CANARY_QUALIFICATION.json',dict(task_id=R.TASK,status='PASS',GateA='PASS',GateB='PASS_4_ACTUAL_MODEL_LOADS',GateC='PASS_4_SAME_GPU_FRESH64INPUT_EXACT',devices=receipts,numeric_limit='64fixedinputtokens/16prefill+48cachedforwards, not81920 numericalproof; no crossGPUbitwise claim',source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json')))
  if not (r/'AUTHORIZATION_FORMAL.json').exists():R.atomic_new(r/'AUTHORIZATION_FORMAL.json',dict(task_id=R.TASK,status='PASS',source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json'),canary_sha256=R.sha(r/'CANARY_QUALIFICATION.json'),jobs_sha256=R.sha(r/'DATASET_AND_JOB_MANIFEST.json'),GPU_mapping=R.read(r/'DATASET_AND_JOB_MANIFEST.json')['gpu_uuid_map'],authorized_jobs=60))
  stage(r,ledger,'formal',0,a.resume);update(r,ledger,nvml(),'GPU_COMPLETE_AWAITING_CPU_SCORE');R.check(all(d['process_exited'] and d['NVML_released'] for d in ledger['dispatches']),'Own GPU not released')
  R.atomic_new(r/'execution/FORMAL_GPU_COMPLETE.json',dict(task_id=R.TASK,status='PASS',ledger_sha256=R.sha(r/'RESOURCE_LEDGER.json'),all_own_GPUs_released=True))
  subprocess.run([PY,'-B',str(r/'source/finish.py'),'--root',str(r)],check=True)
 except InterruptedError:
  for d in ledger['dispatches']:
   sp=r/'status'/f'worker{d["gpu"]}.json'
   if d['mode']=='formal' and not same(d['identity']) and sp.exists():
    st=R.read(sp)
    if st.get('pid')==d['pid'] and st.get('status')=='RUNNING':st.update(status='INTERRUPTED',interrupted_job=st.get('job'),job=None);R.atomic_status(sp,st)
  R.atomic_new(r/'execution'/('INTERRUPTION_'+str(time.time_ns())+'.json'),dict(task_id=R.TASK,status='INTERRUPTED',traceback=traceback.format_exc(),unix=time.time(),resume='controller.py --root ROOT --resume',healthy_workers_preserved=True))
  update(r,ledger,nvml(),'FORMAL_RUNNING');raise
 except BaseException:
  fail=dict(task_id=R.TASK,status='FAIL_CLOSED_STOP',traceback=traceback.format_exc(),unix=time.time(),retry=False)
  try:R.atomic_new(r/'FIRST_FAILURE.json',fail)
  except FileExistsError:pass
  release_owned(ledger['dispatches'])
  for _ in range(40):
   live=nvml();update(r,ledger,live,'FAIL_CLOSED_STOP')
   if all(d.get('process_exited') for d in ledger['dispatches']):break
   time.sleep(.5)
  R.atomic_status(r/'FORMAL_PROGRESS.json',dict(task_id=R.TASK,status='PARTIAL_RESULTS_PRESERVED_AND_STOP' if list((r/'raw_results').glob('*.json')) else 'FAIL_CLOSED_STOP',completed=len(list((r/'raw_results').glob('*.json'))),total_jobs=60,actual_GPU_hours=ledger['cumulative_GPU_hours'],first_failure='FIRST_FAILURE.json',retry=False))
  raise
if __name__=='__main__':main()
