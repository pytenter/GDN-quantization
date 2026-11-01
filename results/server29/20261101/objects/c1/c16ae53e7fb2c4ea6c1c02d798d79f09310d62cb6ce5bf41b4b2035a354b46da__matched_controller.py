#!/usr/bin/env python3
"""One bounded GPU trial batch. Exact own-PID cleanup, no retry or migration."""
import argparse, hashlib, json, math, os
from pathlib import Path
import signal, shutil, statistics, subprocess, sys, time, traceback
TASK='QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def new(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def replace(p,j):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);tick=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
  return dict(pid=pid,start_tick=tick,cmdline=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
def nvml():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(p[0].strip()),uuid=p[1].strip(),memory=p[2].strip()) for x in out.splitlines() if x.strip() for p in [x.split(',')]]
def telemetry():
 args=['nvidia-smi','--id=0','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,power.limit,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu','--format=csv,noheader,nounits']
 return subprocess.check_output(args,text=True).strip()
def exact_signal(proc,ident,sig):
 current=identity(proc.pid)
 if current and current['start_tick']==ident['start_tick'] and current['cmdline']==ident['cmdline']:
  os.kill(proc.pid,sig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 start=r/'execution/BATCH_STARTED.json';assert not start.exists(),'Only one GPU batch; no retry'
 plan=read(r/'GPU_DIAGNOSTIC_PLAN.json');sources=read(r/'SOURCE_SHA256_MANIFEST.json')
 for row in sources['execution_sources']:
  p=r/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 for row in sources['external_dependencies']:
  p=Path(row['path']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 assert shutil.disk_usage('/data').free>5*1024**3,'Insufficient permanent disk for bounded profiles'
 assert not [x for x in nvml() if x['uuid']==UUID],'GPU occupied'
 initial=telemetry();assert UUID in initial and '3090' in initial
 # The index/UUID check precedes the only allowed batch admission.
 new(r/'GPU_ADMISSION.json',dict(task_id=TASK,status='PASS',telemetry=initial,compute_processes=nvml(),source_manifest_sha256=sha(r/'SOURCE_SHA256_MANIFEST.json'),plan_sha256=sha(r/'GPU_DIAGNOSTIC_PLAN.json')))
 new(start,dict(task_id=TASK,status='STARTED_ONCE',pid=os.getpid(),identity=identity(os.getpid()),unix=time.time()))
 ledger=dict(task_id=TASK,status='RUNNING',MAX_TOTAL_GPU_HOURS=.75,MAX_ACTIVE_GPUS=1,MAX_GPU_TRIAL_BATCH=1,
             cumulative_GPU_seconds=0.0,cumulative_GPU_hours=0.0,dispatches=[],accounting='All child dispatch wall through actual process exit and NVML release, including loads,warmup,profiling,postprocessing; no reset')
 live=r/'GPU_RESOURCE_LEDGER.json';replace(live,ledger);completed=[];diagnostic_decision=None;proc=None;ident=None
 def persist(extra=0):
  total=sum(x.get('occupied_seconds',0) for x in ledger['dispatches'])+extra
  ledger['cumulative_GPU_seconds']=total;ledger['cumulative_GPU_hours']=total/3600;replace(live,ledger)
 def run_unit(condition,rep,profile=False):
  nonlocal proc,ident
  assert not [x for x in nvml() if x['uuid']==UUID],'Foreign or prior live GPU process'
  unit=condition+('_PROFILE' if profile else '')+'_R'+str(rep)
  assert not (r/'execution'/unit/'STARTED.json').exists(),'Unit already started'
  base=ledger['cumulative_GPU_seconds'];assert base<2450,'Protective budget admission'
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
  if not condition.startswith('OLD_'):env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
  cmd=[PY,'-B',str(r/'source/matched_worker.py'),'--root',str(r),'--condition',condition,'--repeat',str(rep)]+(['--profile'] if profile else [])
  log=r/'execution'/(unit+'.log');log.parent.mkdir(exist_ok=True);begun=time.monotonic();unix=time.time()
  with log.open('xb') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   ident=identity(proc.pid);assert ident and str(r/'source/matched_worker.py') in ident['cmdline']
   dispatch=dict(unit=unit,pid=proc.pid,identity=ident,GPU_UUID=UUID,dispatch_unix=unix,condition=condition,repeat=rep,profile=profile)
   ledger['dispatches'].append(dispatch);last_size=-1;last_progress=time.monotonic()
   try:
    while proc.poll() is None:
     elapsed=time.monotonic()-begun;total=base+elapsed
     ledger['live_unit']=unit;persist(elapsed)
     apps=nvml();foreign=[x for x in apps if x['uuid']==UUID and x['pid']!=proc.pid]
     assert not foreign,'Foreign process appeared on GPU0; stop own process only'
     with (r/'execution/RESOURCE_TELEMETRY.jsonl').open('a') as tf:
      row=dict(unit=unit,unix=time.time(),monotonic=time.monotonic(),GPU=telemetry(),NVML=apps)
      stat=Path('/proc')/str(proc.pid)/'stat'
      if stat.exists():row['worker_proc_stat']=stat.read_text()
      tf.write(json.dumps(row)+'\n')
     size=log.stat().st_size
     if size!=last_size:last_size=size;last_progress=time.monotonic()
     assert time.monotonic()-last_progress<1800,'Thirty minute no-progress watchdog'
     assert total<2450,'Protective stop before .75GPUh hard limit'
     assert elapsed<1000,'Per-unit bounded watchdog'
     time.sleep(2)
    proc.wait()
   except BaseException:
    exact_signal(proc,ident,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:exact_signal(proc,ident,signal.SIGKILL);proc.wait(timeout=15)
    raise
   finally:
    # Count until real exit and NVML release; never treat a RESULT as release.
    if proc.poll() is not None:
     deadline=time.monotonic()+90
     while any(x['pid']==proc.pid for x in nvml()):
      assert time.monotonic()<deadline,'NVML did not release actual own PID'
      time.sleep(1)
     dispatch.update(occupied_seconds=time.monotonic()-begun,returncode=proc.returncode,process_exited=True,NVML_released=True)
     ledger.pop('live_unit',None);persist()
  assert proc.returncode==0,'Worker failed; no repair/retry'
  result_path=r/'execution'/unit/'RESULT.json';result=read(result_path);assert result['status']=='PASS'
  completed.append(dict(unit=unit,path=str(result_path.relative_to(r)),bytes=result_path.stat().st_size,sha256=sha(result_path),TPS=result['tokens_per_second'],condition=condition,profile=profile))
  print(json.dumps(dict(event='UNIT_COMPLETE',**completed[-1],GPU_hours=ledger['cumulative_GPU_hours'])),flush=True)
  proc=None;ident=None
 try:
  for row in plan['main_schedule']:run_unit(row['condition'],row['repeat'])
  fp={c:[x['TPS'] for x in completed if x['condition']==c] for c in ['OLD_FP','NEW_FP']}
  ratio=statistics.median(fp['OLD_FP'])/statistics.median(fp['NEW_FP'])
  nonoverlap=min(fp['OLD_FP'])>max(fp['NEW_FP']) or min(fp['NEW_FP'])>max(fp['OLD_FP'])
  stable=nonoverlap and max(ratio,1/ratio)>=1.10
  diagnostic_decision=dict(task_id=TASK,status='PREDECLARED_RULE_EVALUATED',old_new_median_ratio=ratio,nonoverlapping_ranges=nonoverlap,stable_difference=stable,
    criterion='Three main FP repeats per path; nonoverlap and symmetric median ratio>=1.10; only resource-safe preregistered diagnostics/profile then run, never source changes')
  new(r/'execution/AUXILIARY_ADMISSION_DECISION.json',diagnostic_decision)
  if stable:
   for row in plan['auxiliary_schedule']:
    if ledger['cumulative_GPU_seconds']+90+200>2450:break
    run_unit(row['condition'],row['repeat'])
   for condition in ['OLD_FP','NEW_FP']:
    if ledger['cumulative_GPU_seconds']+450+200>2450:break
    run_unit(condition,0,True)
  ledger['status']='PASS_BATCH_COMPLETE';ledger['all_own_GPUs_released']=True;persist()
  assert ledger['cumulative_GPU_seconds']<=2700
  new(r/'execution/BATCH_RESULT.json',dict(task_id=TASK,status='PASS',completed_units=completed,auxiliary_decision=diagnostic_decision,
    ledger_sha256=sha(live),all_own_GPUs_released=True))
 except BaseException:
  ledger['status']='FAIL_CLOSED_STOP';ledger['all_own_GPUs_released']=proc is None or (proc.poll() is not None and not any(x['pid']==proc.pid for x in nvml()));persist()
  if not (r/'FIRST_FAILURE.json').exists():new(r/'FIRST_FAILURE.json',dict(task_id=TASK,status='FAIL_CLOSED_STOP',traceback=traceback.format_exc(),retry=False))
  new(r/'execution/BATCH_RESULT.json',dict(task_id=TASK,status='FAIL_CLOSED_STOP',completed_units=completed,ledger_sha256=sha(live),all_own_GPUs_released=ledger['all_own_GPUs_released']))
  raise

if __name__=='__main__':main()
