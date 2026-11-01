"""CPU fixture tests of actual observer payload; no SSH/NVML/GPU calls."""
from contextlib import redirect_stdout
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path as RealPath
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT=RealPath(__file__).resolve().parents[1]
TOOL=ROOT/'operations/read_offline_snapshot_once.py'
spec=importlib.util.spec_from_file_location('readonly_monitor_under_test',TOOL)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='wscale_observer_cpu_')
        self.base=RealPath(self.tmp.name);self.root=self.base/M.TASK;self.root.mkdir()
        self.proc=self.base/'proc';self.proc.mkdir();self.apps=[];self.calls=[];self.opened=[]
        self.uuid=M.UUIDS[1]
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def process(self,pid,start,script,extra=(),ppid=1,state='R'):
        argv=['python3','-u','-B',str(self.root/'scripts'/script),*extra]
        p=self.proc/str(pid);p.mkdir()
        fields=[state,str(ppid)]+['0']*17+[str(start)]
        (p/'stat').write_text(str(pid)+' (python) '+' '.join(fields))
        (p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv)+b'\0')
        return dict(pid=pid,starttick=start,cmdline=argv)
    def fake_nvml(self,cmd,**kwargs):
        self.calls.append(cmd)
        self.assertEqual(cmd[0],'nvidia-smi')
        selector=next(x for x in cmd if x.startswith('--id='))
        self.assertEqual(set(selector[5:].split(',')),set(M.UUIDS.values()))
        if any(x.startswith('--query-gpu=') for x in cmd):
            out='\n'.join(f'{i}, {uuid}, NVIDIA RTX 3090, 24576, 23000' for i,uuid in M.UUIDS.items())
        else:out='\n'.join(f'{pid}, {uuid}, {memory}' for pid,uuid,memory in self.apps)
        return types.SimpleNamespace(returncode=0,stdout=out,stderr='')
    def payload(self,rotate=None):
        def path(value,*args):
            return self.proc if str(value)=='/proc' else RealPath(value,*args)
        original=RealPath.open
        rotated=False
        def opened(p,*args,**kwargs):
            nonlocal rotated
            stream=original(p,*args,**kwargs);self.opened.append(str(p))
            if rotate is not None and p==rotate and not rotated:
                temp=p.with_suffix('.new')
                with open(temp,'w') as f:json.dump(dict(phase='NEW',generated_tokens=256,monotonic=time.monotonic()),f)
                os.replace(temp,p);rotated=True
            return stream
        code=M.REMOTE_CODE.replace('REMOTE_ROOT',repr(str(self.root))).replace('EXPECTED_UUIDS',repr(M.UUIDS))
        # Inject the payload's Path binding only. Patching pathlib.Path globally
        # changes the dispatch check in Path.__new__ and breaks real fixture I/O.
        code=code.replace('from pathlib import Path','Path = __fixture_path',1)
        output=io.StringIO()
        with mock.patch('signal.alarm',lambda n:None),\
             mock.patch('subprocess.run',self.fake_nvml),mock.patch.object(RealPath,'open',opened),redirect_stdout(output):
            exec(compile(code,'actual_readonly_remote_payload','exec'),{'__fixture_path':path})
        return json.loads(output.getvalue())
    def live(self):
        c=self.process(101,1001,'gpu_controller.py',('--root',str(self.root),'--execute'))
        w=self.process(201,2001,'gpu_unit_worker.py',('--root',str(self.root),'--unit','aime26_01_W_seed1','--formal'),ppid=101)
        self.write('GPU_RUN_STARTED.json',dict(identity=c))
        self.write('SOURCE_FREEZE.json',dict(files=[]));permit=self.write('GPU_PERMIT.json',dict(task_id=M.TASK))
        self.write('FORMAL_PROGRESS.json',dict(status='FORMAL_RUNNING',completed=8,planned=90,
            completed_by_method=dict(W=3,R=3,RW=2),gold_blind=True))
        self.write('AUTHORIZATION_FORMAL.json',dict(status='PASS'))
        self.write('formal/aime26_01_W_seed1/STARTED.json',dict(identity=w))
        self.write('formal/aime26_01_W_seed1/DISPATCH.json',dict(unit='aime26_01_W_seed1',
            gpu=1,gpu_uuid=self.uuid,formal=True,permit_sha256=hashlib.sha256(permit.read_bytes()).hexdigest()))
        progress=self.write('formal/aime26_01_W_seed1/PROGRESS.json',dict(phase='OLD',generated_tokens=128,monotonic=time.monotonic()))
        self.apps=[(201,self.uuid,19000)]
        return c,w,progress
    def test_01_live_exact_root_identity_counts_and_selected_NVML(self):
        self.live();r=self.payload()
        self.assertEqual(r['status'],'PASS_READONLY_RUNNING')
        self.assertEqual(r['formal_progress']['completed'],8)
        self.assertEqual(r['controller']['status'],'EXACT_PID_STARTTICK_ARGV_ALIVE')
        self.assertTrue(r['active_units'][0]['dispatch_binding_matches'])
        self.assertEqual(len(r['nvml']['exact_owned_apps']),1)
        self.assertFalse(r['terminal_ready_observed'])
    def test_02_atomic_replace_reads_opened_old_inode_without_false_failure(self):
        _,_,progress=self.live();r=self.payload(rotate=progress)
        self.assertEqual(r['status'],'PASS_READONLY_RUNNING')
        self.assertEqual(r['active_units'][0]['progress']['generated_tokens'],128)
        self.assertEqual(json.loads(progress.read_text())['generated_tokens'],256)
        self.assertEqual(r['observer_issues'],[])
    def test_03_GPU0_foreign_never_makes_selected_scope_failure(self):
        self.live();self.apps.append((999,'GPU-a78176ed-f97f-c54f-6201-5a01b570f422',22000))
        r=self.payload();self.assertEqual(r['status'],'PASS_READONLY_RUNNING')
        self.assertEqual(len(r['nvml']['apps']),1);self.assertTrue(r['nvml']['GPU0_not_in_scope'])
    def test_04_pid_reuse_is_foreign_not_owned_liveness(self):
        c,_,_=self.live();(self.proc/'101/stat').write_text('101 (foreign) '+' '.join(['R','1']+['0']*17+['7777']))
        r=self.payload();self.assertEqual(r['controller']['status'],'PID_REUSED_FOREIGN')
        self.assertFalse(r['controller']['owned_alive'])
    def test_05_true_failure_and_gold_payloads_are_not_downloaded(self):
        self.live();self.write('FIRST_FAILURE.json',dict(status='FAIL',unit='GPU1_W_FIELDS',
            error="RuntimeError('PRIVATE_ANSWER_SENTINEL')",traceback='PRIVATE_ANSWER_SENTINEL'))
        self.write('FROZEN_V4_SCORES.json',dict(response='PRIVATE_ANSWER_SENTINEL'))
        self.write('formal/aime26_01_W_seed1/RESULT.json',dict(response='PRIVATE_ANSWER_SENTINEL'))
        r=self.payload();self.assertEqual(r['status'],'TRUE_FAILURE_OBSERVED_WAIT_ORIGINAL_TERMINAL_PIPELINE')
        self.assertNotIn('PRIVATE_ANSWER_SENTINEL',json.dumps(r))
        self.assertFalse(any(p.endswith(('RESULT.json','FROZEN_V4_SCORES.json')) for p in self.opened))
    def test_06_preparation_has_no_dispatch_or_failure_credit(self):
        self.write('calibration/PARENT_FAILURE.json',dict(error='old preserved CPU observation failure'))
        r=self.payload();self.assertEqual(r['status'],'PREPARATION_NO_GPU_DISPATCH_OBSERVED')
        self.assertEqual(r['dispatches'],0);self.assertEqual(r['finisher_calls'],0)
        self.assertFalse(any(x['exists'] for x in r['failures'].values()))
    def test_07_terminal_readiness_requires_actual_finisher_exit(self):
        c=dict(pid=101,starttick=1001,cmdline=['python3',str(self.root/'scripts/gpu_controller.py'),'--root',str(self.root),'--execute'])
        f=self.process(401,4001,'finalize_offline.py',(str(self.root),))
        self.write('GPU_RUN_STARTED.json',dict(identity=c));self.write('FINALIZER_STARTED.json',dict(identity=f))
        self.write('SESSION_RESULT.json',dict(status='GPU_COMPLETE_52_QUALIFICATION_90_FORMAL',completed=['x']*142,own_all_UUID_release=True))
        self.write('FINAL_SEAL_RECEIPT.json',dict(status='PASS_FINAL_SEAL'))
        self.write('FINAL_ARCHIVE_VERIFICATION.json',dict(status='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL'))
        r=self.payload();self.assertFalse(r['terminal_ready_observed'])
        (self.proc/'401/cmdline').unlink();(self.proc/'401/stat').unlink();(self.proc/'401').rmdir()
        r=self.payload();self.assertTrue(r['terminal_ready_observed'])
        self.assertEqual(r['status'],'TERMINAL_SEAL_READY_FOR_INDEPENDENT_AUDIT')
        self.assertFalse(r['readiness_is_independent_delivery_PASS'])
    def test_08_local_flock_skips_active_reader_without_remote_query(self):
        local=self.base/'local';(local/'operations').mkdir(parents=True)
        with (local/'operations/READONLY_OBSERVER.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with mock.patch.object(M,'ROOT',local),mock.patch.object(M,'_query_remote') as query:
                result=M.run();self.assertEqual(result['status'],'SKIP_ACTIVE_READONLY_OBSERVER');query.assert_not_called()
    def test_09_ssh255_is_saved_wait_without_retry(self):
        local=self.base/'local';(local/'operations').mkdir(parents=True)
        with mock.patch.object(M,'ROOT',local),mock.patch.object(M,'_query_remote',return_value=dict(status='WAIT_READONLY_SSH_UNAVAILABLE',ssh_returncode=255,GPU_retries=0)) as query:
            result=M.run();self.assertEqual(query.call_count,1)
        self.assertEqual(result['status'],'WAIT_READONLY_SSH_UNAVAILABLE')
        self.assertTrue(RealPath(result['observation_path']).is_file())
        self.assertFalse(result['dispatched_GPU_or_finisher'])
    def test_10_timeout_is_saved_wait_without_retry(self):
        local=self.base/'local';(local/'operations').mkdir(parents=True)
        with mock.patch.object(M,'ROOT',local),mock.patch.object(M,'_query_remote',side_effect=subprocess.TimeoutExpired('readonly_ssh',40)) as query:
            result=M.run();self.assertEqual(query.call_count,1)
        self.assertEqual(result['status'],'WAIT_READONLY_SSH_TIMEOUT')
        self.assertEqual(result['GPU_retries'],0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MonitorTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=dict(status='PASS_CPU_OBSERVER_CONTRACT' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        observer_source_sha256=hashlib.sha256(TOOL.read_bytes()).hexdigest(),
        fixture_scope='synthetic filesystem/process/NVML-output and transport mocks only; zero SSH/NVML/GPU/model calls')
    print(json.dumps(output));raise SystemExit(not result.wasSuccessful())
