#!/usr/bin/env python3
"""Real unchanged qualification-guard filesystem path, isolated CPU fixtures.

Only external P1/model/NVML/driver/process observations are explicit test mocks.
Source/permit/dispatch checks, OS flock and target.mkdir are the real production
qualification_guards.admit_single_unit body. No Torch import or GPU calls.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,sort_keys=True);f.write('\n')


def get(path):return json.loads(Path(path).read_text())


def fixture(root):
    root.mkdir(parents=True);(root/'src').mkdir();(root/'scripts').mkdir()
    guard_path=root/'src/qualification_guards.py';guard_path.write_bytes((ROOT/'src/qualification_guards.py').read_bytes())
    for name in('src/resident_execution_control.py','scripts/qualify_gpu.py','scripts/resident_controller.py','scripts/run_resident_offline.sh'):
        (root/name).write_text('# CPU SYNTHETIC FIXTURE ONLY\n')
    guards=load('_real_guard_fixture_'+root.name,guard_path)
    unit=dict(id='GPU1_CODEC',gpu=1,gpu_uuid=guards.UUIDS[1],kind='codec_math',condition=None)
    units=[unit]+[dict(unit,id='CPU_UNUSED_%02d'%i)for i in range(57)]
    jobs=[dict(id='CPU_FORMAL_%02d'%i,gpu=1,gpu_uuid=guards.UUIDS[1],condition='C1'if i%2==0else'S3')for i in range(60)]
    put(root/'QUALIFICATION_JOB_MANIFEST.json',dict(units=units));put(root/'FORMAL_JOB_MANIFEST.json',dict(jobs=jobs))
    for name in('ROOT.json','GPU_EXECUTION_PLAN.json','HUMAN_CONTINUATION_AUTHORIZATION.json','PRIOR_P1_USER_STOP_ADMISSION.json'):put(root/name,dict(CPU_SYNTHETIC_ONLY=True))
    put(root/'SOURCE_FREEZE.json',dict(CPU_SYNTHETIC_ONLY=True,files=[dict(path='src/qualification_guards.py',bytes=guard_path.stat().st_size,sha256=sha(guard_path))]))
    permit=dict(task_id=guards.TASK,ALLOW_GPU='QUALIFICATION_ONLY',
        source_freeze_sha256=sha(root/'SOURCE_FREEZE.json'),qualification_plan_sha256=sha(root/'QUALIFICATION_JOB_MANIFEST.json'),
        execution_plan_sha256=sha(root/'GPU_EXECUTION_PLAN.json'),human_authorization_sha256=sha(root/'HUMAN_CONTINUATION_AUTHORIZATION.json'),
        prior_p1_stop_admission_sha256=sha(root/'PRIOR_P1_USER_STOP_ADMISSION.json'))
    put(root/'GPU_PERMIT.json',permit)
    return guards,unit


@contextlib.contextmanager
def external_observation_mocks(guards,root,unit):
    events=[]
    parent=dict(pid=os.getppid(),starttick=101,cmdline=['CPU_SYNTHETIC_PARENT'])
    worker=dict(pid=os.getpid(),starttick=102,cmdline=[sys.executable,str(root/'scripts/qualify_gpu.py'),'--root',str(root),'--unit',unit['id'],'--gpu','1'])
    def process(pid):return parent if pid==os.getppid()else worker if pid==os.getpid()else None
    def observe(name,value):events.append(name);return value
    vendor=types.SimpleNamespace(process_identity=process,same_process=lambda x:process(x['pid'])==x,atomic_new=put)
    real_statvfs=os.statvfs
    with patch.object(guards,'vendor',return_value=vendor), \
         patch.object(guards,'p1_queue_released',side_effect=lambda *a:observe('MOCK_P1_RELEASE',dict(CPU_mock=True))), \
         patch.object(guards,'_own_gpu',side_effect=lambda *a:observe('MOCK_NVML',dict(CPU_mock=True))), \
         patch.object(guards,'verify_model_and_external',side_effect=lambda *a:observe('MOCK_MODEL_EXTERNAL',[])), \
         patch.object(guards,'driver_identity',side_effect=lambda *a:observe('MOCK_DRIVER_NO_CUDA',dict(CPU_mock=True))), \
         patch.object(guards.os,'statvfs',side_effect=lambda p:types.SimpleNamespace(f_bavail=100,f_frsize=1024**3)if str(p)=='/data'else real_statvfs(p)), \
         patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':guards.UUIDS[1]}):
        yield parent,worker,events
    for handle in guards._LOCKS:handle.close()
    guards._LOCKS.clear()


def dispatch_fixture(root,unit,parent):
    put(root/'dispatch/qualification'/(unit['id']+'.json'),dict(unit=unit['id'],unit_record=unit,controller_identity=parent,
        source_freeze_sha256=sha(root/'SOURCE_FREEZE.json'),permit_sha256=sha(root/'GPU_PERMIT.json')))


def run_suite(evidence):
    evidence=Path(evidence).resolve();evidence.mkdir(parents=True,exist_ok=False)
    controller=load('_real_directory_preparation_controller',ROOT/'scripts/resident_controller.py')
    checks=[]
    def check(name,ok,**details):
        if not ok:raise AssertionError(name)
        checks.append(dict(name=name,status='PASS',**details))
    def rejects(name,fn,types_=(RuntimeError,FileExistsError)):
        try:fn()
        except types_ as error:check(name,True,error=str(error));return
        raise AssertionError('Expected failure: '+name)
    # Observed old deployment: shell had logs/operations, controller created
    # dispatch parents, qualification was absent. Run the real unchanged guard.
    bad=evidence/'reproduce_old_parent_absence';guards,u=fixture(bad)
    (bad/'operations').mkdir();(bad/'logs').mkdir()
    with external_observation_mocks(guards,bad,u)as(parent,worker,events):
        dispatch_fixture(bad,u,parent)
        try:guards.admit_single_unit(bad,u['id'],1,os.getpid())
        except FileNotFoundError as error:
            check('actual_unchanged_guard_reproduces_missing_qualification_parent',str(bad/'qualification'/u['id'])==error.filename,error=str(error))
        else:raise AssertionError('Actual guard did not reproduce the observed missing parent failure')
        check('actual_guard_failure_reaches_real_target_mkdir_after_mock_external_boundaries',events==['MOCK_P1_RELEASE','MOCK_NVML','MOCK_MODEL_EXTERNAL','MOCK_NVML','MOCK_DRIVER_NO_CUDA'])
        check('failed_actual_guard_creates_no_target_or_STARTED',not(bad/'qualification').exists())
    # Completely fresh fixture: every runtime parent starts absent. Invoke the
    # actual new controller preparation, then the exact same guard function.
    good=evidence/'fresh_prepared_actual_guard';guards,u=fixture(good)
    check('fresh_fixture_starts_with_all_execution_parents_absent',all(not(good/name).exists()for name in controller.EXECUTION_DIRECTORIES))
    immutable={str(p.relative_to(good)):sha(p)for p in good.rglob('*')if p.is_file()}
    result=controller.prepare_execution_directories(good)
    check('actual_controller_prepares_every_required_parent',all((good/name).is_dir()and not(good/name).is_symlink()for name in controller.EXECUTION_DIRECTORIES))
    check('preparation_never_creates_unit_targets',not(good/'qualification'/u['id']).exists()and not any((good/'formal').iterdir())and result['unit_targets_created']is False)
    check('preparation_changes_no_immutable_input_bytes',all(sha(good/n)==h for n,h in immutable.items()))
    second=controller.prepare_execution_directories(good)
    check('preparation_idempotent_for_existing_ordinary_directories',second==result)
    with external_observation_mocks(guards,good,u)as(parent,worker,events):
        dispatch_fixture(good,u,parent)
        admitted=guards.admit_single_unit(good,u['id'],1,os.getpid())
        check('actual_unchanged_guard_admits_after_parent_preparation',admitted['target']==good/'qualification'/u['id'])
        check('actual_guard_writes_exact_admission_identity',get(admitted['target']/'DISPATCH_AND_ADMISSION.json')['identity']==worker)
        check('actual_guard_does_not_fake_STARTED_or_RESULT',not(admitted['target']/'STARTED.json').exists()and not(admitted['target']/'RESULT.json').exists())
        preserved=sha(admitted['target']/'DISPATCH_AND_ADMISSION.json')
        rejects('actual_guard_rejects_existing_unit_target',lambda:guards.admit_single_unit(good,u['id'],1,os.getpid()))
        check('duplicate_guard_attempt_does_not_overwrite_admission',sha(admitted['target']/'DISPATCH_AND_ADMISSION.json')==preserved)
    # Exercise the new production controller dispatcher without allowing Popen.
    control=types.SimpleNamespace(TASK=guards.TASK,UUIDS=guards.UUIDS,PYTHON=sys.executable,
        process_identity=lambda pid:dict(pid=pid,starttick=1,cmdline=['CPU_CONTROLLER_ONLY']),
        verify_sources=lambda root:sha(root/'SOURCE_FREEZE.json'),query_gpu=lambda:dict(apps=[]))
    c=controller.Controller(good,control)
    with patch.object(controller.subprocess,'Popen')as spawn:
        rejects('controller_rejects_existing_target_before_Popen',lambda:c.dispatch('qualification',u))
        check('existing_target_causes_zero_child_launches',not spawn.called)
    for name in('qualification','formal','logs','operations','dispatch','dispatch/qualification','logs/qualification','logs/formal','monitoring'):
        for kind in('file','symlink','dangling_symlink'):
            root=evidence/('bad_'+name.replace('/','_')+'_'+kind);fixture(root)
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            if kind=='file':path.write_bytes(b'KEEP_EXISTING_FILE')
            else:
                destination=evidence/('outside_'+name.replace('/','_')+'_'+kind)
                if kind=='symlink':destination.mkdir();(destination/'KEEP').write_bytes(b'KEEP_OUTSIDE')
                path.symlink_to(destination,target_is_directory=True)
            before={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and not p.is_symlink()}
            rejects('non_directory_parent_'+name+'_'+kind,lambda root=root:controller.prepare_execution_directories(root))
            check('bad_parent_preserved_'+name+'_'+kind,os.path.lexists(path)and all(sha(root/n)==h for n,h in before.items()))
    root=evidence/'existing_execution_marker';fixture(root);put(root/'CONTROLLER_STARTED.json',dict(CPU_SYNTHETIC=True))
    rejects('main_rejects_STARTED_before_any_parent_creation',lambda:controller.main(['--root',str(root),'--execute']))
    check('STARTED_refusal_makes_no_runtime_directories',all(not(root/n).exists()for n in controller.EXECUTION_DIRECTORIES))
    root=evidence/'dangling_unit_target';guards,u=fixture(root);controller.prepare_execution_directories(root)
    leaf=root/'qualification'/u['id'];leaf.symlink_to(evidence/'missing_destination',target_is_directory=True)
    c=controller.Controller(root,control)
    with patch.object(controller.subprocess,'Popen')as spawn:
        rejects('dangling_unit_target_rejected_before_Popen',lambda:c.dispatch('qualification',u))
        check('dangling_unit_not_deleted_and_no_child',leaf.is_symlink()and not spawn.called)
    check('test_did_not_import_Torch','torch'not in sys.modules)
    return dict(status='PASS_REAL_GUARD_PATH_DIRECTORY_REPAIR_CPU_ONLY',checks=checks,check_count=len(checks),
        actual_guard_sha256=sha(ROOT/'src/qualification_guards.py'),controller_sha256=sha(ROOT/'scripts/resident_controller.py'),
        source_ROOT=str(ROOT),GPU_contexts_created=0,GPU_dispatches=0,Torch_imported=False,
        external_observations_mocked=['process_identity','oldP1_release','NVML','model_external','driver_identity_no_CUDA','data_free_space'],
        actual_production_body_executed='qualification_guards.admit_single_unit: source+permit+dispatch+parent+flock+target.mkdir+admissionJSON',
        does_not_claim_real_GPU_admission_or_qualification=True,monotonic=time.monotonic())


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--evidence-root',type=Path);a=ap.parse_args()
    result=run_suite(a.evidence_root or a.output.parent/(a.output.stem+'_evidence'));a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],check_count=result['check_count'])))
if __name__=='__main__':main()
