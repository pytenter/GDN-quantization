"""One bounded Ling/KDA Dual-Axis task controller; CPU process, no model import.

This is independent orchestration for the new task. Only the frozen resource
predicates, read-only telemetry and lifecycle classifiers are reused.
"""
import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import types

TASK = 'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'
CONDITIONS = ('FP_NATIVE','H_R128','H_DUAL_M','L6_R128','L6_DUAL_M')
ENV_KEYS = ('PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG',
            'FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID')
MAX_GPU_HOURS, PROTECT_GPU_HOURS = 16.0, 15.5


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.partial.{os.getpid()}.{time.monotonic_ns()}')
    try:
        with tmp.open('xb') as stream:stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if tmp.exists():tmp.unlink()


def replace_current(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    with tmp.open('wb') as stream:stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    blob=canonical(value)+b'\n'
    with path.open('ab') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);offset=stream.tell();stream.write(blob);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),offset=offset,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())


def process_identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
    status=(p/'status').read_text().splitlines()
    return dict(pid=int(pid),start_ticks=int(stat[19]),uid=int(next(x for x in status if x.startswith('Uid:')).split()[1]),
        nspid=[int(x) for x in next(x for x in status if x.startswith('NSpid:')).split()[1:]],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def exact_identity_present(identity):
    try:return process_identity(identity['pid'])==identity
    except (OSError,ValueError,StopIteration):return False


def import_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


class OriginalResourcePolicy:
    """Load only original pure resource predicates, never an old controller."""
    def __init__(self,worktree):
        worktree=Path(worktree)
        namespace=dict(Path=Path,proc_identity=process_identity)
        tree=ast.parse((worktree/'gpu_admission.py').read_text())
        selected=[]
        for node in tree.body:
            names={t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set()
            if isinstance(node,ast.FunctionDef) and node.name in ('graphics_identity','assess','assess_window'):selected.append(node)
            elif names.intersection({'GIB','HEADROOM','PEAK'}):selected.append(node)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(worktree/'gpu_admission.py'),'exec'),namespace)
        assert namespace['HEADROOM']==2*2**30 and namespace['PEAK']==20484980736
        module=types.ModuleType('gpu_admission');module.__dict__.update(namespace)
        # lifecycle_guard imports this name; it receives only pure frozen policy.
        sys.modules['gpu_admission']=module
        self.policy=module
        self.telemetry=import_file(worktree/'gpu_telemetry.py','ling_dual_axis_readonly_telemetry')
        self.guard=import_file(worktree/'lifecycle_guard.py','ling_dual_axis_lifecycle_guard')

    def sample(self,uuid):return self.telemetry.sample(uuid)
    def timing(self,s):return self.guard.validate_timing_envelope(s)
    def running(self,s,uuid,owner,baseline,before,after):
        return self.guard.assess_lifecycle_sample(s,uuid,owner,baseline,before,after,assess_fn=self.policy.assess)
    def window(self,rows,uuid,baseline):
        if baseline is None:
            for s in rows:self.guard.strict_snapshot_safety(s,uuid,None,None)
            return self.policy.assess_window(rows,uuid)
        return self.guard.assess_terminal_window(rows,uuid,baseline,assess_window_fn=self.policy.assess_window)


class TaskStopped(RuntimeError):pass


class Runner:
    def __init__(self,d,worktree,resource=None,*,poll_seconds=10.,window_seconds=5.,cleanup_timeout=60.):
        self.d=Path(d).resolve();self.w=Path(worktree).resolve();self.resource=resource
        self.poll_seconds=poll_seconds;self.window_seconds=window_seconds;self.cleanup_timeout=cleanup_timeout
        self.prereg=None;self.baseline=None;self.jobs=[];self.active=None;self.active_identity=None
        self.prereg_sha=None;self.source_sha=None;self.process=None

    def failed(self):return (self.d/'FIRST_FAILURE.json').exists()

    def first_failure(self,reason,*,evidence=None,details=None):
        value=dict(task_id=TASK,status='FAIL_CLOSED_STOP',reason=str(reason),epoch=time.time(),
            no_retry=True,active_job=self.active,owner=self.active_identity,evidence=evidence,details=details)
        try:publish(self.d/'FIRST_FAILURE.json',value)
        except FileExistsError:pass
        return read(self.d/'FIRST_FAILURE.json')

    def event(self,event,**fields):
        append(self.d/'STATE_EVENTS.jsonl',dict(event=event,epoch=time.time(),**fields))
        replace_current(self.d/'CURRENT_STATE.json',dict(event=event,epoch=time.time(),**fields))

    def used_hours(self,now=None):
        elapsed=sum(x['charged_seconds'] for x in self.jobs)
        if self.active is not None:elapsed+=(time.monotonic() if now is None else now)-self.active['launch_monotonic']
        return elapsed/3600.

    def ledger(self):
        value=dict(task_id=TASK,prior_task_gpu_hours=0.,GPU_HOURS_USED=self.used_hours(),
            charged_GPU_hours=self.used_hours(),
            completed_workers=sum(x['qualified_completion'] for x in self.jobs),
            completed_jobs=[x['tag'] for x in self.jobs if x['qualified_completion']],
            active_workers=[] if self.active_identity is None else [self.active_identity],
            max_gpu_hours=MAX_GPU_HOURS,protect_gpu_hours=PROTECT_GPU_HOURS,
            scope='Conservative charged upper bound: process launch through complete terminal resource check, including load/forward/scoring/IO/cleanup and CPU gate.',
            pure_launch_to_wait_seconds=sum(x['launch_to_wait_seconds'] for x in self.jobs),
            post_wait_gate_seconds=sum(x['terminal_gate_seconds'] for x in self.jobs),
            jobs=self.jobs,active_job=self.active,active_owner=self.active_identity,epoch=time.time())
        replace_current(self.d/'GPU_RESOURCE_LEDGER.json',value);return value

    def budget_guard(self):
        hours=self.used_hours()
        if hours>=PROTECT_GPU_HOURS:
            self.first_failure('GPU_BUDGET_PROTECTIVE_STOP',details=dict(actual_charged_gpu_hours=hours,
                protective_limit=PROTECT_GPU_HOURS,absolute_limit=MAX_GPU_HOURS))
            raise TaskStopped('GPU_BUDGET_PROTECTIVE_STOP')
        assert hours<MAX_GPU_HOURS,'ABSOLUTE_GPU_BUDGET_EXCEEDED'

    def verify_source(self):
        manifest=self.d/'SOURCE_MANIFEST.json'
        if self.source_sha is not None:assert sha(manifest)==self.source_sha,'SOURCE_MANIFEST_CHANGED'
        if self.prereg_sha is not None:assert sha(self.d/'PREREGISTRATION.json')==self.prereg_sha,'PREREGISTRATION_CHANGED'
        entries=read(manifest)['files'];assert isinstance(entries,dict) and entries
        assert str(self.w/'dual_axis_task/runner.py') in entries,'RUNNER_NOT_SOURCE_BOUND'
        for path,digest in entries.items():assert Path(path).is_absolute() and sha(path)==digest,('SOURCE_CHANGED',path)
        identity=self.prereg['identity']
        assert subprocess.check_output(['git','-C',str(self.w),'rev-parse','HEAD'],text=True).strip()==identity['source_commit'],'SOURCE_COMMIT_CHANGED'
        assert subprocess.check_output(['git','-C',str(self.w),'status','--porcelain'],text=True)=='','SOURCE_WORKTREE_DIRTY'
        return dict(status='PASS_SOURCE_IDENTITY',manifest_sha256=sha(manifest),preregistration_sha256=sha(self.d/'PREREGISTRATION.json'),source_commit=identity['source_commit'])

    def preflight(self):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CONTROLLER_MUST_BE_CPU_ONLY'
        assert not self.failed(),'PRIOR_FAILURE_NO_RETRY'
        assert not (self.d/'RUN_CONTROLLER.json').exists(),'CONTROLLER_ALREADY_LAUNCHED_NO_RETRY'
        assert not list((self.d/'jobs').glob('*/LAUNCH.json')),'PRIOR_GPU_JOB_NO_RETRY'
        self.prereg=read(self.d/'PREREGISTRATION.json');self.prereg_sha=sha(self.d/'PREREGISTRATION.json');self.source_sha=sha(self.d/'SOURCE_MANIFEST.json')
        assert self.prereg['task_id']==TASK and self.prereg['identity']['target_index']==6 and self.prereg['identity']['target_uuid']==UUID
        run=self.prereg['run'];assert (run['tokens_per_trajectory'],run['evaluation_start'],run['evaluation_count'],run['noquant_tokens'])==(8448,8192,256,128)
        budget=self.prereg['budget'];assert budget['max_gpu_hours']==16.0 and budget['protect_gpu_hours']==15.5
        assert budget['formal_safety_factor']==1.25 and budget['cleanup_reserve_gpu_hours']>=.25
        intent=read(self.d/'LAUNCH_INTENT.json')
        assert intent['task_id']==TASK and intent['source_commit']==self.prereg['identity']['source_commit']
        assert intent['preregistration_sha256']==self.prereg_sha and intent['source_manifest_sha256']==self.source_sha,'LAUNCH_INTENT_IDENTITY_MISMATCH'
        reports=self.prereg['preflight_reports'];assert len(reports)>=5,'CPU_REPORT_SET_INCOMPLETE'
        assert {x['category'] for x in reports}>={'math','codec','data','scoring','control'},'CPU_REPORT_CATEGORIES_INCOMPLETE'
        for row in reports:
            assert sha(row['path'])==row['sha256'],('CPU_REPORT_CHANGED',row['path'])
            report=read(row['path']);assert report[row.get('status_key','status')]==row['status_value'],('CPU_GATE_NOT_PASS',row['path'])
        frozen=read(self.d/'FROZEN_DOCUMENTS.json');assert len(frozen['formal_documents'])==6
        assert len({x['document'] for x in frozen['formal_documents']})==6
        assert frozen['canary_document']['document'] not in {x['document'] for x in frozen['formal_documents']}
        self.verify_source()
        return frozen

    def environment(self):
        env=os.environ.copy();env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',FLA_CACHE_MODE='default',
            FLA_CONFIG_DIR=str(self.w/'frozen_fla_configs'),PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES='6',LING_TARGET_GPU_UUID=UUID,LING_PROSPECTIVE_RESULTS=str(self.d))
        return env

    def snapshot(self,path,owner):
        try:s=self.resource.sample(UUID)
        except BaseException as exc:
            s=dict(uuid=UUID,status='UNKNOWN_SAMPLING_EXCEPTION',error=repr(exc),epoch=time.time())
        ref=append(path,dict(snapshot=s,snapshot_sha256=hashlib.sha256(canonical(s)).hexdigest(),
            owner=owner,baseline=self.baseline,epoch=time.time(),controller=process_identity(os.getpid())))
        if owner is not None and self.active is not None:
            for row in s.get('direct_NVML_processes',{}).get('compute',{}).get('processes',[]):
                if row.get('pid')==owner['pid'] and type(row.get('usedGpuMemory')) is int:
                    prior=self.active.get('NVML_owned_periodic_peak_bytes')
                    self.active['NVML_owned_periodic_peak_bytes']=max(prior or 0,row['usedGpuMemory'])
        try:
            assert s.get('status')!='UNKNOWN_SAMPLING_EXCEPTION','RESOURCE_QUERY_EXCEPTION'
            self.resource.timing(s)
            assert int(s['target'][0]['index'])==6,'GPU_INDEX_CHANGED'
        except BaseException as exc:
            self.first_failure('RESOURCE_QUERY_OR_IDENTITY_UNKNOWN',evidence=ref,details=dict(error=repr(exc),snapshot=s))
            raise TaskStopped(str(exc)) from exc
        return s,ref

    def resource_window(self,jobdir,tag):
        rows=[];references=[];tick=time.monotonic()
        for i in range(3):
            s,ref=self.snapshot(jobdir/(tag+'.RAW.jsonl'),None);rows.append(s);references.append(ref)
            # Validate each sample now, not after another 5 seconds of an unsafe window.
            try:
                if self.baseline is not None:self.resource.guard.strict_snapshot_safety(s,UUID,None,self.baseline)
                else:self.resource.guard.strict_snapshot_safety(s,UUID,None,None)
            except BaseException as exc:
                self.first_failure('RESOURCE_WINDOW_SAMPLE_FAILED',evidence=ref,details=dict(error=repr(exc),snapshot=s))
                raise TaskStopped(str(exc)) from exc
            if i<2:time.sleep(self.window_seconds)
        try:value=self.resource.window(rows,UUID,self.baseline)
        except BaseException as exc:
            self.first_failure('RESOURCE_WINDOW_FAILED',evidence=references,details=dict(error=repr(exc)))
            raise TaskStopped(str(exc)) from exc
        if self.baseline is None:self.baseline=value['graphics_baseline']
        value.update(tag=tag,seconds=time.monotonic()-tick,raw_references=references,epoch=time.time())
        publish(jobdir/(tag+'.PASS.json'),value);return value

    def validate_assets(self,result_path,kind,document,condition,expected_env):
        result=read(result_path);profile=read(result_path.with_suffix('.PROFILE.json'));source=self.verify_source()
        assert result['status']=='PASS' and result['task_id']==TASK
        assert (result['kind'],result['document'],result['condition'],result['UUID'])==(kind,document,condition,UUID),'WORKER_RESULT_IDENTITY_MISMATCH'
        assert result['binding']['preregistration_sha256']==self.prereg_sha and result['binding']['source_manifest_sha256']==self.source_sha
        assert result['binding']['source_commit']==self.prereg['identity']['source_commit']
        assert profile['status']=='PASS_ACTUAL_FROZEN_PROFILE' and profile['hash_randomization']==0
        assert profile['actual_environment']==expected_env and profile['source_binding']==result['binding'],'WORKER_PROFILE_BINDING_MISMATCH'
        actual=profile['profile'];assert actual['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE' and actual['forward_kernels_executed'] is True
        assert all(actual['environment'].get(k)==expected_env[k] for k in ENV_KEYS[:5]),'ACTUAL_FLA_PROFILE_ENVIRONMENT_MISMATCH'
        expected_endpoint=128 if kind=='noquant' else 8256 if kind=='qual_fresh' else 8448
        assert result['endpoint']==expected_endpoint and result['tokens_consumed_in_this_process']==(64 if kind=='qual_fresh' else expected_endpoint)
        if condition in CONDITIONS[1:]:
            assert result['integrity']['passed'] is True,'PERSISTENT_QDQ_INTEGRITY_FAILED'
            if kind!='qual_fresh':
                assert result['GPU_codec_reference_parity']['status'].startswith('PASS_'),'GPU_CODEC_REFERENCE_FAILED'
        if kind=='qual_fresh':assert result['fresh64_exact'] is True and result['parity64'] is not None
        if kind in ('qual_long','formal') and condition!='FP_NATIVE':assert result['score_summary']['count']==256
        for payload in result.get('FP_payloads',[]):assert sha(payload['path'])==payload['sha256'],'FP_PAYLOAD_CHANGED'
        return result,dict(status='PASS_ACTUAL_COMPLETION_ASSETS',result_path=str(result_path),
            result_sha256=sha(result_path),profile_sha256=sha(result_path.with_suffix('.PROFILE.json')),source=source)

    def stop_own_child(self,child,identity,jobdir):
        deadline=time.monotonic()+self.cleanup_timeout
        while child.poll() is None and time.monotonic()<deadline:time.sleep(min(.25,self.poll_seconds))
        actions=[]
        if child.poll() is None:
            assert exact_identity_present(identity),'CLEANUP_OWNER_IDENTITY_CHANGED'
            child.terminate();actions.append('SIGTERM_EXACT_OWN_CHILD')
            try:child.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                assert exact_identity_present(identity),'CLEANUP_OWNER_IDENTITY_CHANGED'
                child.kill();actions.append('SIGKILL_EXACT_OWN_CHILD');child.wait(timeout=10.)
        publish(jobdir/'OWN_CHILD_CLEANUP.json',dict(owner=identity,actions=actions,returncode=child.poll(),epoch=time.time()))

    def worker_command(self,kind,document,condition,result_path,parent_path=None):
        python=self.prereg.get('controller',{}).get('python_path','/data/zypan/envs/ling-kda/bin/python')
        command=[python,'-B','-u',str(self.w/'dual_axis_task/model_worker.py'),'--kind',kind,
                 '--document',document,'--condition',condition,'--result',str(result_path)]
        if parent_path is not None:command.extend(['--parent-result',str(parent_path)])
        if condition!='FP_NATIVE':
            scope='formal' if kind=='formal' else 'qualification'
            reference_kind='formal' if kind=='formal' else 'qual_long'
            command.extend(['--fp-result',str(self.d/scope/f'{document}__FP_NATIVE.{reference_kind}.json')])
        return command

    def run_job(self,kind,document,condition,parent_path=None):
        assert not self.failed(),'FIRST_FAILURE_PROHIBITS_LAUNCH'
        self.budget_guard();source=self.verify_source()
        tag=f'{document}__{condition}.{kind}';jobdir=self.d/'jobs'/tag
        jobdir.mkdir(parents=True,exist_ok=False)
        publish(jobdir/'SOURCE.json',source)
        target_dir=self.d/('formal' if kind=='formal' else 'qualification');target_dir.mkdir(exist_ok=True)
        result_path=target_dir/(tag+'.json');assert not result_path.exists(),'RESULT_EXISTS_NO_RETRY'
        self.event('ADMISSION_RUNNING',tag=tag,kind=kind,document=document,condition=condition)
        self.resource_window(jobdir,'ADMISSION');self.budget_guard();self.verify_source()
        env=self.environment();expected={k:env[k] for k in ENV_KEYS}
        command=self.worker_command(kind,document,condition,result_path,parent_path)
        publish(jobdir/'LAUNCH_ENVIRONMENT.json',dict(environment=expected,command=command,source=source,epoch=time.time()))
        launch_epoch=time.time();launch_mono=time.monotonic();child=None;wait_mono=None;rc=None;terminal=None;asset=None;result=None
        try:
            log=(jobdir/'worker.log').open('ab')
            try:child=subprocess.Popen(command,cwd=self.w,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            self.process=child;self.active=dict(tag=tag,kind=kind,document=document,condition=condition,
                launch_epoch=launch_epoch,launch_monotonic=launch_mono,result_path=str(result_path))
            self.active_identity=process_identity(child.pid)
            assert self.active_identity['uid']==os.getuid(),'WORKER_UID_MISMATCH'
            publish(jobdir/'LAUNCH.json',dict(**self.active,owner=self.active_identity,command=command,source=source))
            self.event('WORKER_RUNNING',**self.active,owner=self.active_identity)
            # Capture interpreter-start environment from /proc, not only Popen arguments.
            raw=(Path('/proc')/str(child.pid)/'environ').read_bytes();actual={}
            for item in raw.split(b'\0'):
                if b'=' in item:
                    key,value=item.split(b'=',1)
                    if key.decode(errors='replace') in ENV_KEYS:actual[key.decode()]=value.decode()
            assert actual==expected,'ACTUAL_PROCESS_ENVIRONMENT_MISMATCH'
            publish(jobdir/'ENV.json',dict(owner=self.active_identity,environment=actual,epoch=time.time()))
            while child.poll() is None:
                self.budget_guard()
                before=child.poll()
                if before is not None:break
                s,ref=self.snapshot(jobdir/'RESOURCE_RAW.jsonl',self.active_identity)
                after=child.poll()
                try:value=self.resource.running(s,UUID,self.active_identity,self.baseline,before,after)
                except BaseException as exc:
                    self.first_failure('RUNNING_RESOURCE_FAILED',evidence=ref,details=dict(error=repr(exc),snapshot=s,poll_before=before,poll_after=after))
                    raise TaskStopped(str(exc)) from exc
                append(jobdir/'LIFECYCLE.jsonl',dict(value=value,raw_reference=ref,poll_before=before,poll_after=after,epoch=time.time()))
                self.ledger()
                if self.failed():raise TaskStopped('FIRST_FAILURE')
                if after is None:
                    deadline=time.monotonic()+self.poll_seconds
                    while child.poll() is None and time.monotonic()<deadline:
                        if self.failed():raise TaskStopped('FIRST_FAILURE')
                        time.sleep(min(.25,self.poll_seconds))
            rc=child.wait();wait_mono=time.monotonic()
            if rc!=0:
                self.first_failure('WORKER_NONZERO_EXIT',details=dict(returncode=rc,worker_error_path=str(result_path.with_suffix('.ERROR.json'))))
                raise TaskStopped('WORKER_NONZERO_EXIT')
            if self.failed():raise TaskStopped('FIRST_FAILURE_CANNOT_BE_WASHED_BY_RC0')
            result,asset=self.validate_assets(result_path,kind,document,condition,expected)
            publish(jobdir/'ASSET_BINDING.json',asset)
            self.event('TERMINAL_RESOURCE_CHECK',tag=tag,owner=self.active_identity,worker_returncode=rc)
            terminal=self.resource_window(jobdir,'TERMINAL_RESOURCE')
            if self.failed():raise TaskStopped('FIRST_FAILURE_CANNOT_BE_WASHED_BY_TERMINAL_PASS')
            self.budget_guard();self.verify_source()
        except BaseException as exc:
            if not self.failed():self.first_failure('JOB_CONTROL_OR_ASSET_FAILURE',details=dict(error=repr(exc),traceback=traceback.format_exc()))
            if child is not None and child.poll() is None:self.stop_own_child(child,self.active_identity,jobdir)
            if child is not None:
                rc=child.wait();wait_mono=wait_mono or time.monotonic()
            raise
        finally:
            if child is not None:
                if child.poll() is None:
                    self.first_failure('CHILD_STILL_LIVE_DURING_FINALLY');self.stop_own_child(child,self.active_identity,jobdir)
                rc=child.wait();wait_mono=wait_mono or time.monotonic();end=time.monotonic()
                assert not exact_identity_present(self.active_identity),'OWN_WORKER_STILL_PRESENT'
                # Process gone and no matching PID can survive cleanup. A successful
                # job additionally requires the original strict three-point idle gate.
                entry=dict(tag=tag,kind=kind,document=document,condition=condition,owner=self.active_identity,
                    launch_epoch=launch_epoch,wait_epoch=launch_epoch+(wait_mono-launch_mono),end_epoch=time.time(),
                    launch_to_wait_seconds=wait_mono-launch_mono,terminal_gate_seconds=end-wait_mono,
                    charged_seconds=end-launch_mono,returncode=rc,
                    qualified_completion=bool(rc==0 and terminal is not None and asset is not None and not self.failed()),
                    terminal_resource_pass=terminal is not None,asset_binding_only=asset is not None,
                    own_process_identity_absent_after_wait=True,
                    NVML_owned_periodic_peak_bytes=self.active.get('NVML_owned_periodic_peak_bytes'),
                    NVML_peak_scope='Periodic direct NVML compute-process owned-PID samples; not a continuous peak. Missing is null.',
                    result_path=str(result_path),result_sha256=sha(result_path) if result_path.exists() else None,
                    profile_sha256=sha(result_path.with_suffix('.PROFILE.json')) if result_path.with_suffix('.PROFILE.json').exists() else None,
                    source_manifest_sha256=self.source_sha,
                    source_receipt_sha256=sha(jobdir/'SOURCE.json'),
                    environment_sha256=sha(jobdir/'ENV.json') if (jobdir/'ENV.json').exists() else None,
                    asset_binding_sha256=sha(jobdir/'ASSET_BINDING.json') if (jobdir/'ASSET_BINDING.json').exists() else None,
                    terminal_gate_sha256=sha(jobdir/'TERMINAL_RESOURCE.PASS.json') if (jobdir/'TERMINAL_RESOURCE.PASS.json').exists() else None)
                entry['status']='PASS' if entry['qualified_completion'] else 'FAIL_CLOSED_STOP'
                self.jobs.append(entry);self.active=None;self.active_identity=None;self.process=None
                publish(jobdir/'EXIT.json',entry);self.ledger()
            else:self.active=None;self.active_identity=None
        assert self.jobs[-1]['qualified_completion'] and not self.failed()
        self.event('JOB_COMPLETE',tag=tag,result_sha256=sha(result_path))
        return result_path,result

    def qualification(self,frozen):
        document=frozen['canary_document']['document'];units=[]
        for condition in CONDITIONS:
            parent_path,parent=self.run_job('qual_long',document,condition)
            fresh_path,fresh=self.run_job('qual_fresh',document,condition,parent_path)
            assert fresh['parity64']==parent['parity64'] and fresh['binding']==parent['binding'],'PARENT_FRESH64_EXACT_MISMATCH'
            assert parent['observer_enabled'] is True and fresh['observer_enabled'] is False,'OBSERVER_ON_OFF_CONTRACT_CHANGED'
            cp=parent['checkpoint'];assert cp['next_position']==8192 and sha(cp['path'])==cp['sha256']
            units.append(dict(condition=condition,status='PASS',parent_path=str(parent_path),parent_sha256=sha(parent_path),
                fresh_path=str(fresh_path),fresh_sha256=sha(fresh_path),parity64=parent['parity64'],
                GPU_codec_reference_parity=parent['GPU_codec_reference_parity'],integrity=parent['integrity']))
        noquant=[]
        for condition in ('H_NOQUANT','L6_NOQUANT'):
            path,result=self.run_job('noquant',document,condition)
            noquant.append(dict(condition=condition,path=str(path),sha256=sha(path),
                max_abs=result['noquant_max_abs'],interpretation='Recorded operational finite-precision difference; no borrowed Qwen threshold.'))
        assert len(units)==5 and not self.failed()
        qualification=dict(task_id=TASK,status='PASS_ALL_5_OWN_HISTORY_FRESH64',count=5,
            student_writeback_conditions=4,units=units,noquant=noquant,formal_trajectories_completed=0,
            scope='GPU6 actual 8448-token own histories, each fresh64, actual profiles and strict resource gates. No old-device qualification inherited.')
        publish(self.d/'GPU_QUALIFICATION.json',qualification);self.event('GPU_QUALIFICATION_COMPLETE',count=5)
        return qualification

    def forecast(self):
        values={condition:sum(x['charged_seconds'] for x in self.jobs if x['kind']=='qual_long' and x['condition']==condition)/3600. for condition in CONDITIONS}
        assert all(v>0 for v in values.values()) and len([x for x in self.jobs if x['kind']=='qual_long'])==5
        used=self.used_hours();reserve=self.prereg['budget']['cleanup_reserve_gpu_hours']
        remaining=sum(values.values())*6*1.25
        projected=used+remaining+reserve
        value=dict(task_id=TASK,status='PASS' if projected<=PROTECT_GPU_HOURS else 'BUDGET_FEASIBILITY_STOP',
            already_charged_qualification_GPUh=used,measured_8448_per_condition_GPUh=values,formal_trajectories=30,
            formal_safety_factor=1.25,formal_prediction_GPUh=remaining,cleanup_exit_reserve_GPUh=reserve,
            projected_total_GPUh=projected,protective_limit_GPUh=PROTECT_GPU_HOURS,absolute_limit_GPUh=MAX_GPU_HOURS,
            measurement_scope='Measured total 8448 process launch through actual scoring/IO/cleanup and terminal resource gate, not short-token extrapolation.')
        publish(self.d/'BUDGET_FORECAST.json',value)
        if projected>PROTECT_GPU_HOURS:
            self.first_failure('BUDGET_FEASIBILITY_STOP',details=value);raise TaskStopped('BUDGET_FEASIBILITY_STOP')
        return value

    def execute(self):
        # Duplicate execution is a read-only rejection, never a reason to alter
        # an already running or sealed task's first failure or delivery.
        assert not (self.d/'RUN_CONTROLLER.json').exists(),'CONTROLLER_ALREADY_LAUNCHED_NO_RETRY'
        assert not self.failed(),'PRIOR_FAILURE_NO_RETRY'
        try:
            frozen=self.preflight()
            publish(self.d/'RUN_CONTROLLER.json',dict(task_id=TASK,owner=process_identity(os.getpid()),
                worktree=str(self.w),output=str(self.d),started_epoch=time.time(),preregistration_sha256=self.prereg_sha,
                source_manifest_sha256=self.source_sha,source_commit=self.prereg['identity']['source_commit']))
            self.resource=self.resource or OriginalResourcePolicy(self.w)
            self.qualification(frozen);self.forecast()
            completed=[]
            for doc in frozen['formal_documents']:
                for condition in CONDITIONS:
                    path,result=self.run_job('formal',doc['document'],condition)
                    completed.append(dict(document=doc['document'],condition=condition,path=str(path),sha256=sha(path)))
            assert len(completed)==30 and not self.failed()
            publish(self.d/'FORMAL_COMPLETE.json',dict(task_id=TASK,status='PASS_30_COMPLETE_8448_TRAJECTORIES',count=30,trajectories=completed))
            self.event('FORMAL_COMPLETE',count=30)
        except BaseException as exc:
            if not self.failed():self.first_failure('CONTROLLER_FAILURE',details=dict(error=repr(exc),traceback=traceback.format_exc()))
            self.event('FAIL_CLOSED_STOP',reason=read(self.d/'FIRST_FAILURE.json')['reason'])
        finally:
            assert self.process is None or self.process.poll() is not None,'OWN_WORKER_STILL_LIVE_NO_SEAL'
            self.ledger()
            try:
                analysis=import_file(self.w/'dual_axis_task/analysis.py','ling_dual_axis_final_analysis')
                analysis.seal(self.d)
            except BaseException as exc:
                publish(self.d/'CPU_SEAL_FAILURE.json',dict(task_id=TASK,status='CPU_SEAL_FAILED',error=repr(exc),
                    traceback=traceback.format_exc(),epoch=time.time(),no_GPU_retry=True))
                raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();worktree=Path(__file__).resolve().parents[1]
    assert args.output.is_absolute() and args.output.name==TASK
    assert os.environ.get('LING_PROSPECTIVE_RESULTS')==str(args.output.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'CONTROLLER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Runner(args.output,worktree).execute()


if __name__=='__main__':main()
