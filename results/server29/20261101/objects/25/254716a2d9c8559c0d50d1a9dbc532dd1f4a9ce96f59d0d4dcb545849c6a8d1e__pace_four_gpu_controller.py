"""Four fixed TP1 queues, owned fresh processes, first-failure STOP, no retries."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pace_control as C
from pace_four_gpu_publish import publish_unit

def progress_signature(value):
    return tuple(value.get(k) for k in ('phase','generated_tokens','input_forwards','token','tokens','total','condition','case','layer'))

class Controller:
    def __init__(self,root):
        self.root=Path(root);self.identity=C.process_identity(os.getpid());self.lock=threading.RLock()
        self.stop=threading.Event();self.first=None;self.active={};self.intervals=[];self.completed=[]
        self.reuse=C.read(self.root/'REUSE_ADMISSION.json');self.plan=C.read(self.root/'FOUR_GPU_PLAN.json')
    def fail(self,error,unit):
        with self.lock:
            if self.first is None:
                self.first=dict(status='FAIL_CLOSED_STOP',unit=unit,error=repr(error),traceback=traceback.format_exc(),unix=time.time(),no_automatic_retry=True)
                C.atomic_new(self.root/'FIRST_FAILURE.json',self.first)
            self.stop.set()
    def ledger(self):
        with self.lock:
            v=dict(task_id=C.TASK,execution_revision=C.REVISION,method_variant=C.METHOD_VARIANT,state_scale_format=C.STATE_SCALE_FORMAT,
                   controller=self.identity,intervals=list(self.intervals),completed_units=list(self.completed),active_units=dict(self.active),
                   reused_formal_count=len(self.reuse['reused_ids']),current_formal_count=sum(x['stage']=='formal' for x in self.completed),
                   formal_total_unique=len(self.reuse['reused_ids'])+sum(x['stage']=='formal' for x in self.completed),
                   formal_planned=60,physical_GPU_active_seconds='UNRESOLVED_NOT_MEASURED',no_elapsed_or_GPUh_caps=True,unix=time.time())
            seconds=sum(x['reservation_seconds'] for x in self.intervals)+sum(time.monotonic()-x['dispatch_monotonic'] for x in self.active.values())
            v.update(cumulative_GPU_seconds=seconds,cumulative_GPUh=seconds/3600,accounting_scope='Owned dispatch to actual exit/release reservation wall; historical intervals remain separate')
            C.status(self.root/'RESOURCE_LEDGER.json',v);return v
    def device(self,gpu,pid=None):
        obs=C.query_gpu();uuid=C.UUIDS[gpu];rows=[x for x in obs['devices'] if x['physical_index']==gpu]
        C.require(len(rows)==1 and rows[0]['uuid']==uuid and '3090' in rows[0]['name'] and rows[0]['memory_total_MiB']>=24000,'Assigned device identity differs')
        C.require(not any(x['uuid']==uuid and x['pid']!=pid for x in obs['apps']),'Foreign process on assigned GPU; never signal it')
        if pid is None:C.require(rows[0]['memory_free_MiB']>=23000,'Assigned free VRAM insufficient')
        C.require(pid is None or not any(x['pid']==pid and x['uuid']!=uuid for x in obs['apps']),'Worker used unauthorized UUID')
        return obs
    def run_unit(self,stage,unit,gpu):
        proc=None;identity=None;target=C.unit_target(self.root,stage,unit);log=None;started=time.monotonic();failure=None;exit_seen=None
        try:
            C.require(not self.stop.is_set(),'Shared STOP prevents dispatch')
            C.require(not (self.root/'GLOBAL_STOP_REQUEST.json').exists(),'Human stop prevents dispatch')
            C.authorize(self.root);C.model_guard(self.root);C.storage_guard(self.root);self.device(gpu)
            if stage=='formal':
                admission=C.read(self.root/'FORMAL_ADMISSION.json')
                C.require(admission['source_manifest_sha256']==C.sha(self.root/C.SOURCE) and admission['status']=='PASS_ALL_FOUR_DEVICE_GATES_ADMIT_FIXED_REMAINING_ONCE','FourGPU formal admission absent')
                job=next(x for x in self.plan['jobs'] if x['unit']==unit);C.require(job['gpu']==gpu,'Per-unit fixed assignment differs')
            C.require(not target.exists(),'Existing unit forbids retry or reset');target.mkdir(parents=True)
            C.atomic_new(target/'DISPATCH.json',dict(task_id=C.TASK,execution_revision=C.REVISION,stage=stage,unit=unit,gpu=gpu,gpu_uuid=C.UUIDS[gpu],controller=self.identity,source_manifest_sha256=C.sha(self.root/C.SOURCE),permit_sha256=C.sha(self.root/'GPU_PERMIT.json')))
            argv=[sys.executable,str(self.root/'scripts/pace_four_gpu_worker.py'),'--root',str(self.root),'--stage',stage,'--unit',unit]
            env=C.worker_environment();env.update(PACE_PHYSICAL_GPU=str(gpu),CUDA_VISIBLE_DEVICES=C.UUIDS[gpu])
            log=(target/'worker.log').open('xb',buffering=0);started=time.monotonic();dispatch_unix=time.time()
            proc=subprocess.Popen(argv,cwd=self.root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            until=time.monotonic()+2
            while time.monotonic()<until:
                try:
                    identity=C.process_identity(proc.pid)
                    if identity['cmdline']==argv:break
                except FileNotFoundError:pass
                if proc.poll() is not None:break
                time.sleep(.01)
            C.require(identity is not None and identity['cmdline']==argv,'Actual owned child identity not registered')
            C.atomic_new(target/'ACTUAL_CHILD_IDENTITY.json',identity)
            with self.lock:self.active[unit]=dict(gpu=gpu,gpu_uuid=C.UUIDS[gpu],stage=stage,identity=identity,dispatch_monotonic=started)
            self.ledger();last=time.monotonic();signature=None;next_check=0
            while proc.poll() is None:
                C.require(C.same_process(identity),'Owned child PID/starttick/argv changed; never signal replacement')
                C.require(not self.stop.is_set(),'Shared first failure stops remaining owned children')
                C.require(not (self.root/'GLOBAL_STOP_REQUEST.json').exists(),'Human requested STOP')
                p=target/'PROGRESS.json'
                try:
                    with p.open('rb') as f:value=json.load(f)
                    current=progress_signature(value)
                    if current!=signature:signature=current;last=time.monotonic()
                except FileNotFoundError:pass
                C.require(time.monotonic()-last<1800,'No real phase/token progress for1800s')
                if time.monotonic()>=next_check:self.device(gpu,proc.pid);C.storage_guard(self.root);self.ledger();next_check=time.monotonic()+10
                time.sleep(.5)
            exit_seen=time.monotonic();C.require(proc.returncode==0,'Worker actual nonzero exit: '+str(proc.returncode))
        except BaseException as e:
            failure=dict(error=repr(e),traceback=traceback.format_exc());self.fail(e,unit)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    if identity is not None and C.same_process(identity):
                        C.terminate_owned(proc,identity)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:C.terminate_owned(proc,identity,kill=True);proc.wait(timeout=15)
                    else:
                        # An unknown/replaced PID is never a cleanup target.
                        try:proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:pass
                end=time.monotonic();observation=None;released=False
                while time.monotonic()-end<120:
                    observation=C.query_gpu()
                    if C.release_state(proc.poll() is not None,observation,proc.pid,identity):released=True;break
                    time.sleep(.5)
                if log:log.close()
                interval=dict(task_id=C.TASK,stage=stage,unit=unit,gpu=gpu,gpu_uuid=C.UUIDS[gpu],pid=proc.pid,identity=identity,returncode=proc.poll(),released=released,
                              dispatch_unix=dispatch_unix,accounting_end_unix=time.time(),reservation_seconds=time.monotonic()-started,
                              physical_GPU_active_seconds='UNRESOLVED_NOT_MEASURED',NVML=observation,failure=failure,accounting_final=released,lower_bound_only=not released)
                C.atomic_new(target/'RESOURCE_INTERVAL.json',interval)
                with self.lock:self.intervals.append(interval);self.active.pop(unit,None)
                self.ledger()
                if not released:self.fail(RuntimeError('Actual own exit/release unresolved'),unit)
                if released and proc.poll() is not None:
                    try:
                        if failure is None and proc.returncode==0:
                            verified=C.verify_result(self.root,stage,unit,identity)
                            if stage=='scale_repair_gates':
                                smoke=verified['result']['paired_full_model_smoke']
                                C.require(smoke['status']=='PASS' and smoke['actual_cuda'] is True and set(smoke['configs'])=={'R2','R3'} and smoke['actual_mask_gate']['status']=='PASS' and len(smoke['actual_mask_gate']['cases'])==48,'Actual per-device paired model smoke proof missing')
                        durable=publish_unit(self.root,target,interval)
                        if failure is None and proc.returncode==0:
                            with self.lock:self.completed.append(dict(stage=stage,unit=unit,gpu=gpu,result_sha256=C.sha(target/'RESULT.json'),interval_sha256=C.sha(target/'RESOURCE_INTERVAL.json'),durable_manifest_sha256=durable['manifest_sha256']))
                            self.ledger()
                    except BaseException as e:self.fail(e,unit)
        return failure is None and proc is not None and proc.returncode==0 and not self.stop.is_set()
    def lane(self,gpu):
        for job in self.plan['jobs']:
            if job['gpu']==gpu:
                if self.stop.is_set() or not self.run_unit('formal',job['unit'],gpu):break
    def run(self):
        C.authorize(self.root);C.require(not (self.root/'SESSION_STARTED.json').exists(),'Unique launch only; no restart')
        bootstrap=C.read(self.root/'SOURCE_BOOTSTRAP_RECEIPT.json')
        C.require(bootstrap['status']=='PASS_SOURCE_BOOTSTRAP29_ALL_SHA_BYTES_NOREPLACE' and bootstrap['source_manifest_sha256']==C.sha(self.root/C.SOURCE) and bootstrap['permit_sha256']==C.sha(self.root/'GPU_PERMIT.json'),'Durable29 frozen source bootstrap required before GPU work')
        C.atomic_new(self.root/'SESSION_STARTED.json',dict(identity=self.identity,source_manifest_sha256=C.sha(self.root/C.SOURCE),permit_sha256=C.sha(self.root/'GPU_PERMIT.json'),unix=time.time()))
        try:
            for gpu in range(4):self.device(gpu)
            from pace_four_gpu_results import verify_reuse
            C.atomic_new(self.root/'INITIAL_REUSE_FRESH_VERIFICATION.json',verify_reuse(self.root,full_sources=True))
            C.atomic_new(self.root/'EXTERNAL_PRE_GPU_FRESH_SHA.json',C.verify_external(self.root,full_model=True))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda g:self.run_unit('scale_repair_gates','SCALE_REPAIR_GATES_GPU'+str(g),g),range(4)))
            C.require(not self.stop.is_set() and len(self.completed)==4,'All4 fresh device gate exit0/release/durable proofs required')
            C.freeze_science(self.root)
            C.atomic_new(self.root/'FORMAL_ADMISSION.json',dict(status='PASS_ALL_FOUR_DEVICE_GATES_ADMIT_FIXED_REMAINING_ONCE',source_manifest_sha256=C.sha(self.root/C.SOURCE),permit_sha256=C.sha(self.root/'GPU_PERMIT.json'),science_freeze_sha256=C.verify_science(self.root),gate_units=list(self.completed),reused_formal_count=len(self.reuse['reused_ids']),current_formal_planned=len(self.plan['jobs']),total_unique=60))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(self.lane,range(4)))
            C.require(not self.stop.is_set() and sum(x['stage']=='formal' for x in self.completed)==len(self.plan['jobs']),'Exact remaining formal completion missing')
        except BaseException as e:self.fail(e,'CONTROLLER')
        finally:
            ledger=self.ledger();full=self.first is None and ledger['formal_total_unique']==60 and not self.active
            C.atomic_new(self.root/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_FOURGPU_REUSE_PLUS_REMAINING60' if full else 'FAIL_CLOSED_STOP',task_id=C.TASK,execution_revision=C.REVISION,controller=self.identity,failure=self.first,ledger=ledger,completed_units=list(self.completed),formal_total_unique=ledger['formal_total_unique'],own_workers_exit_and_NVML_release=not self.active and all(x['released'] for x in self.intervals),source_manifest_sha256=C.sha(self.root/C.SOURCE),permit_sha256=C.sha(self.root/'GPU_PERMIT.json')))
        return 0 if full else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print('CPU-only entry; unique frozen fourGPU permit required');return 0
    c=Controller(a.root)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda s,f:c.fail(RuntimeError('Owned controller received human signal '+str(s)),'CONTROLLER_SIGNAL'))
    return c.run()
if __name__=='__main__':raise SystemExit(main())
