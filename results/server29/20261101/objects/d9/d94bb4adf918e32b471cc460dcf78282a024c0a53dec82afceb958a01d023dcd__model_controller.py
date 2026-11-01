"""Host CPU supervisor for this task's two direct-child codec micro workers."""
from pathlib import Path
import argparse,os,sys,time,subprocess,threading,traceback,concurrent.futures,fcntl,stat
from qualification_io import read,write,ref,sha,verify,event

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--contract',required=True);p.add_argument('--host-key',required=True);a=p.parse_args()
 r=Path(a.root);c=read(a.contract);host=c['hosts'][a.host_key];work=r/'qualification';control=work/'control';control.mkdir(parents=True,exist_ok=True)
 lock=(control/'HOST_CONTROLLER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (control/'RUN_CONTROLLER.json').exists(),'UNIQUE_CONTROLLER_ALREADY_REGISTERED'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CONTROLLER_CUDA_MUST_BE_EMPTY'
 verify(r,c)
 sys.path.insert(0,str(r/'control/runtime'));import resource_monitor as rm,desktop_graphics_policy as gp,gpu_telemetry as gt
 pairs={int(k):v for k,v in host['pairs'].items()};rm.PAIRS=pairs;gp.PAIRS=pairs;gp.TASK=c['task_id']
 policy=gp.AuthorizedGraphicsPolicy(r/host['graphics_baseline']['relative_path'],host['graphics_baseline']['sha256'])
 controller=rm.read_proc_identity(os.getpid());assert controller['status']=='PASS'
 write(control/'RUN_CONTROLLER.json',dict(task_id=c['task_id'],stage='REAL_MODEL_PERSISTENT_512',actual_identity=controller,contract=ref(a.contract),CUDA_VISIBLE_DEVICES='',GPUh_cap=None,wall_cap=None,physical_workers_fixed=8,maximum_concurrent_GPU_workers=2,formal_AIME_authorized=False))
 stop=control/'FIRST_FAILURE.json';globalstop=control/'GLOBAL_STOP_REQUEST.json';allresults={};mutex=threading.RLock(); cached_accounts={}
 def publish():
  with mutex:
   
   for key,m in monitors.items():
    if m.lock.acquire(blocking=False):
     try:cached_accounts[key]=m.accounting()
     finally:m.lock.release()
   rows=list(cached_accounts.values());value=dict(task_id=c['task_id'],current_GPUh=sum(x['current_gpu_hours'] for x in rows),prior_GPUh=host['prior_same_task_GPUh'],cumulative_GPUh=host['prior_same_task_GPUh']+sum(x['current_gpu_hours'] for x in rows),lanes=rows,GPUh_cap=None,wall_cap=None,scope='per-device spawn through strict terminal-gate observational upper bound; not exact physical residency')
   tmp=control/('RESOURCE_LEDGER.json.tmp.'+str(threading.get_ident()))
   with tmp.open('w') as f:
    import json;json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   os.replace(tmp,control/'RESOURCE_LEDGER.json')
 def global_guard():
  if globalstop.exists() and not stop.exists():write(stop,dict(status='STOP',reason='OTHER_HOST_OR_GLOBAL_CONTROL_STOP',evidence=ref(globalstop)))
  if stop.exists():raise RuntimeError('FIRST_FAILURE_LATCHED')
 monitors={key:rm.ResourceMonitor(work/'resources'/key,d['index'],d['uuid'],gt.sample,policy,stop_path=stop,initialization_timeout_seconds=180,max_adjacent_gap_seconds=15) for key,d in c['devices'].items() if d['host']==a.host_key}
 def execute(device_key,arm):
  key=device_key+'_'+arm
  d=c['devices'][device_key];m=monitors[device_key];j=work/'jobs'/key;j.mkdir(parents=True,exist_ok=False);child=None;stream=None
  try:
   global_guard();verify(r,c)
   preflight_command=[host['python'],'-B',str(r/'scripts/artifact_load_preflight.py'),'--library',str(r/c['cuda_artifact']['relative_path']),'--expected-sha256',c['cuda_artifact']['sha256'],'--expected-cudart-sha256',c['libcudart_sha256'],'--output',str(j/'CPU_ARTIFACT_ABI_PREFLIGHT.json')]
   preflight=subprocess.run(preflight_command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True)
   write(j/'CPU_ARTIFACT_ABI_PROCESS_RECEIPT.json',dict(status='CPU_ONLY_NOT_GPU_QUALIFICATION',returncode=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr,command=preflight_command))
   assert preflight.returncode==0,'CPU_ARTIFACT_ABI_PREFLIGHT_FAILED_BEFORE_GPU_SPAWN'
   global_guard();admission=m.admit(key);global_guard();verify(r,c)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   env.update(PYTHONPATH=host['runtime_source'],SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK='1',TRITON_CACHE_DIR=str(r/'runtime_cache'/device_key/'triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'runtime_cache'/device_key/'torchinductor'))
   stream=(j/'worker.closed.log').open('xb')
   with rm.DECISION_LOCK:
    global_guard();start=time.monotonic_ns();wall=time.time_ns()
    process=subprocess.Popen([host['python'],'-B',str(r/'scripts/model_worker.py'),'--root',str(r),'--contract',a.contract,'--job',str(j),'--device-key',device_key,'--arm',arm],stdout=stream,stderr=subprocess.STDOUT,env=env)
    child=m.register_child(process,key,admission,spawn_monotonic_ns=start,spawn_wall_ns=wall)
    launch=write(j/'LAUNCH.json',dict(actual_identity=child.initial_actual_identity,controller=controller,gpu_uuid=d['uuid'],contract=ref(a.contract),admission=admission,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],arm=arm,device_key=device_key,spawn_monotonic_ns=start,spawn_wall_ns=wall))
    event(control/'EVENT_LEDGER.jsonl',dict(event='ACTUAL_WORKER_SPAWN',device_key=key,launch=launch))
   last_progress=time.monotonic();last_signature=None;result_ack=False
   while True:
    global_guard();obs=m.exit_observer(process)
    if obs['status']=='EXITED_UNREAPED':break
    assert obs['status']=='LIVE_UNREAPED',obs
    observed=m.observe(child)
    if observed['status']=='RUNNING_RESOURCE_PASS' and not (j/'RESOURCE_VISIBLE_ACK.json').exists():write(j/'RESOURCE_VISIBLE_ACK.json',dict(status='ACTUAL_THREE_SOURCE_OWNER_VISIBLE',raw_ref=observed['raw_ref'],identity=child.owner.__dict__))
    signatures=[]
    for name in ('BOOTSTRAP.json','PROGRESS.jsonl','WORKER_RESULT.json'):
     pp=j/name
     if pp.exists():signatures.append((name,pp.stat().st_size,pp.stat().st_mtime_ns))
    if signatures!=last_signature:last_signature=signatures;last_progress=time.monotonic()
    assert time.monotonic()-last_progress<=1800,'NO_PROGRESS_WATCHDOG'
    used=0
    def walk_error(error):
     if not isinstance(error,FileNotFoundError):raise error
    for directory,dirs,files in os.walk(r,onerror=walk_error):
     for name in files:
      try:info=(Path(directory)/name).stat()
      except FileNotFoundError:continue
      if stat.S_ISREG(info.st_mode):used+=info.st_size
    assert used<=c['disk_limit_bytes'],'DISK_SAFETY_LIMIT'
    if (j/'WORKER_RESULT.json').exists() and not result_ack:
     result=validate(child)
     assert observed['status']=='RUNNING_RESOURCE_PASS','RESULT_ACK_REQUIRES_ACTUAL_LIVE_COMPUTE'
     write(j/'RESULT_OBSERVED_ACK.json',dict(status='ACTUAL_RESULT_SEEN_AND_OWNED_CONTEXT_VISIBLE',result=ref(j/'WORKER_RESULT.json'),raw_ref=observed['raw_ref']));result_ack=True
    publish();time.sleep(3)
   closed=m.finish(child,validate);stream.close();stream=None
   write(j/'CLOSED_WORKER_LOG_RECEIPT.json',dict(status='CLOSED',log=ref(j/'worker.closed.log')))
   event(control/'EVENT_LEDGER.jsonl',dict(event='ACTUAL_LEGAL_EXIT',device_key=key,completion=closed['completion_ref']))
   allresults[key]=closed;publish();return closed
  except BaseException as e:
   if not stop.exists():
    try:m.fail(repr(e))
    except BaseException:pass
   if child is not None and not child.closed:
    try:m.terminate_exact(child)
    except BaseException as cleanup:
     write(j/'CLEANUP_FAILURE.json',dict(error=repr(cleanup),original_error=repr(e)))
   if stream is not None:stream.close()
   event(control/'EVENT_LEDGER.jsonl',dict(event='FIRST_FAILURE_OR_PEER_STOP',device_key=key,error=repr(e)))
   allresults[key]=dict(status='STOP',error=repr(e));publish();return allresults[key]
 def validate(child):
  j=work/'jobs'/child.tag;v=read(j/'WORKER_RESULT.json');verify(r,c);assert v['status']=='PASS' and v['gpu_uuid']==child.owner.gpu_uuid
  assert all(v['identity'][k]==child.initial_actual_identity[k] for k in ('pid','uid','start_ticks','boot_id','ppid'));actualenv=read(j/'ACTUAL_ENV.json');boot=read(j/'BOOTSTRAP.json');assert boot['actual_gpu_uuid']==v['gpu_uuid'] and actualenv['expected_uuid']==v['gpu_uuid']
  assert v['stage']=='REAL_MODEL_PERSISTENT_512' and v['models_loaded']==1 and v['actual_tokens_processed']==512
  assert v['QDQ_write_count']==9216 and v['actual_next_kernel_consumption_count']==9198 and v['prefill_QDQ_write_count']==576 and v['decode_QDQ_write_count']==8640
  assert len(v['comparisons'])==72 and all(x['status']=='PASS_EXACT' for x in v['comparisons'])
  assert set(v['checkpoints'])=={'32','128','512'} and all(all(x['unchanged'].values()) for x in v['gate_probes'])
  assert v['max_total_num_tokens']>=262144 and v['model_context_len']==262144
  refs=[ref(j/n) for n in ('WORKER_RESULT.json','ACTUAL_ENV.json','BOOTSTRAP.json','LAUNCH.json','CPU_ARTIFACT_ABI_PREFLIGHT.json','CHECKPOINT_0032.json','CHECKPOINT_0128.json','CHECKPOINT_0512.json')]
  for receipt in v['numerical_evidence']:assert sha(receipt['path'])==receipt['sha256']
  return dict(status='PASS',source_verified=True,numerical_pass=True,profile_verified=True,environment_verified=True,gpu_uuid=v['gpu_uuid'],evidence_refs=refs)
 def run_lane(device_key):
  for arm in ('B0','B1','B2','B3'):
   if stop.exists():break
   result=execute(device_key,arm)
   if result['status']!='PASS':break
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(run_lane,list(monitors)))
 publish();verify(r,c)
 complete=not stop.exists() and len(allresults)==8 and all(v['status']=='PASS' for v in allresults.values())
 write(control/'CONTROLLER_TERMINAL.json',dict(status='PASS' if complete else 'STOP',stage='REAL_MODEL_PERSISTENT_512_ONLY',results=allresults,actual_identity=controller,ownworkers_reaped=all(m.active is None or m.active.closed for m in monitors.values()),closed_writer_threads_joined=True,CUDA_VISIBLE_DEVICES='',formal_AIME_dispatched=0))
 return 0 if complete else 1
if __name__=='__main__':
 try:code=main()
 except BaseException:
  traceback.print_exc();sys.exit(1)
 sys.exit(code)
