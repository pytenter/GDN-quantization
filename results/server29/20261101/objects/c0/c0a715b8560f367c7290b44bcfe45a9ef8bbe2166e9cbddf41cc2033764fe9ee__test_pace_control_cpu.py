#!/usr/bin/env python3
"""Finite CPU control tests for the explicitly authorized exponent variant.

Real controller paths use owned-child mocks. No CUDA/NVML/model/SSH is called.
"""
import copy
import importlib.util
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pace_control as C
spec = importlib.util.spec_from_file_location('exponent_controller_cpu_review', ROOT/'scripts/pace_controller.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class Contracts(unittest.TestCase):
    def permit(self):
        metadata = dict(task_id=C.TASK, execution_revision=C.REVISION,
                        remote_root=C.REMOTE, archive_root=C.ARCHIVE)
        grant = C.read(ROOT/'operations/HUMAN_PACE_SCALE_EXPONENT_REPAIR_AUTHORIZATION.json')
        permit = dict(task_id=C.TASK, execution_revision=C.REVISION, ALLOW_GPU='YES',
            execution_root=C.REMOTE, physical_indices=[0], devices=[C.UUID],
            source_manifest_sha256='s', method_spec_sha256='m',
            human_authorization_sha256=C.SCALE_REPAIR_AUTHORIZATION_SHA,
            method_variant=C.METHOD_VARIANT, state_scale_format=C.STATE_SCALE_FORMAT,
            formal_new_samples=57,reused_formal_samples=3,formal_total_unique=60,
            seed=1,max_new_tokens=81920,watchdog_seconds=1800,cleanup_seconds=120,
            GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None)
        return metadata, permit, copy.deepcopy(grant)

    def test_exact_new_permit(self):
        C.validate_permit_values(*self.permit(),'s','m',C.SCALE_REPAIR_AUTHORIZATION_SHA)

    def test_permit_malformed_scope_rejected(self):
        for key,value in [('execution_root','/tmp/foreign'),('execution_revision','old'),
                          ('physical_indices',[1]),('devices',['foreign']),
                          ('source_manifest_sha256','wrong'),('human_authorization_sha256','old'),
                          ('method_variant','original_R2_R3'),('state_scale_format','FP32'),
                          ('seed',2),('formal_new_samples',60),('reused_formal_samples',4),
                          ('formal_total_unique',61),('max_new_tokens',2048),
                          ('watchdog_seconds',30),('GPUh_hard_cap',1),('total_wall_cap',1),('unit_elapsed_cap',1)]:
            with self.subTest(key=key):
                a,p,g=self.permit();p[key]=value
                with self.assertRaises(RuntimeError):C.validate_permit_values(a,p,g,'s','m',C.SCALE_REPAIR_AUTHORIZATION_SHA)

    def test_grant_format_clip_foreign_or_retry_not_authorized(self):
        for key,value in [('status','old'),('only_physical_gpu',1),('only_UUID','foreign'),
                          ('no_clipping',False),('server29_CPU_archive_only',False),
                          ('prior3_and_calibration_reuse_only_after_exact_accepted_domain_equivalence_and_provenance',False),
                          ('one_new_independent_finite_execution_no_unbounded_GPU_retry',False),
                          ('old_true_STOP_source_and_seals_readonly',False),('failed_partial_credit',1)]:
            with self.subTest(key=key):
                a,p,g=self.permit();g[key]=value
                with self.assertRaises(RuntimeError):C.validate_permit_values(a,p,g,'s','m',C.SCALE_REPAIR_AUTHORIZATION_SHA)

    def test_exact60_manifest_remains_canonical(self):
        data=C.read(ROOT/'AIME26_DATASET_FREEZE.json');value=C.build_jobs(data)
        self.assertEqual(value,C.read(ROOT/'FORMAL_JOB_MANIFEST.json'))
        self.assertEqual(len(value['jobs']),60)
        self.assertEqual([x['condition']for x in value['jobs']],['R2','R3']*30)
        self.assertEqual({x['gpu_uuid']for x in value['jobs']},{C.UUID})
        data['records'][1]=data['records'][0]
        with self.assertRaises(RuntimeError):C.build_jobs(data)

    def test_local_or_old_root_rejected_before_device_API(self):
        with mock.patch.object(C,'query_gpu',side_effect=AssertionError('device API forbidden')):
            with self.assertRaises(RuntimeError):C.authorize(ROOT)

    def test_static_source_rejects_shadow(self):
        with tempfile.TemporaryDirectory()as d:
            r=Path(d);(r/'src').mkdir();p=r/'src/entry.py';p.write_text('# bound')
            C.atomic_new(r/C.SOURCE,dict(task_id=C.TASK,count=1,files=[dict(path='src/entry.py',bytes=p.stat().st_size,sha256=C.sha(p))]))
            C.verify_sources(r)
            (r/'src/shadow.py').write_text('# not bound')
            with self.assertRaises(RuntimeError):C.verify_sources(r)

    def test_own_GPU0_filter_ignores_foreign_devices(self):
        obs=dict(devices=[dict(physical_index=0,uuid=C.UUID,name='RTX3090',memory_total_MiB=24576,memory_free_MiB=24000)],apps=[dict(pid=999,uuid='GPU_FOREIGN123')])
        with mock.patch.object(C,'query_gpu',return_value=obs):self.assertEqual(C.device(),obs)
        obs['apps'].append(dict(pid=33,uuid=C.UUID))
        with mock.patch.object(C,'query_gpu',return_value=obs):
            with self.assertRaises(RuntimeError):C.device()
            self.assertEqual(C.device(33),obs)
        obs['apps'].append(dict(pid=33,uuid='GPU_FOREIGN56'))
        with mock.patch.object(C,'query_gpu',return_value=obs):
            with self.assertRaises(RuntimeError):C.device(33)

    def test_environment_exactUUID_and_no_private_path(self):
        with mock.patch.dict(os.environ,dict(PYTHONPATH='/tmp/foreign',PYTHONHOME='bad',PYTHONUSERBASE='bad')):
            env=C.worker_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],C.UUID)
        self.assertTrue(all(k not in env for k in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE')))

    def test_release_unknown_and_PID_reuse(self):
        empty=dict(apps=[]);own=dict(apps=[dict(pid=11,uuid=C.UUID)])
        identity=dict(pid=11,starttick=22,cmdline=['ours'])
        self.assertFalse(C.release_state(False,empty,11));self.assertFalse(C.release_state(True,None,11))
        self.assertFalse(C.release_state(True,own,11,identity,reader=lambda p:identity))
        self.assertTrue(C.release_state(True,empty,11))
        foreign=dict(pid=11,starttick=23,cmdline=['foreign'])
        self.assertTrue(C.release_state(True,own,11,identity,reader=lambda p:foreign))
        self.assertFalse(C.release_state(False,own,11,identity,reader=lambda p:foreign))

    def test_termination_exact_owned_identity_only(self):
        class Child:
            pid=11
            def poll(self):return None
            def send_signal(self,s):self.signal=s
        p=Child();identity=dict(pid=11,starttick=22,cmdline=['ours'])
        C.terminate_owned(p,identity,reader=lambda p:identity);self.assertEqual(p.signal,signal.SIGTERM)
        for altered in (dict(pid=11,starttick=23,cmdline=['ours']),dict(pid=11,starttick=22,cmdline=['foreign'])):
            with self.assertRaises(RuntimeError):C.terminate_owned(p,identity,reader=lambda p:altered)
        with self.assertRaises(RuntimeError):C.terminate_owned(p,None)

    def gate_fixture(self, root):
        (root/C.SOURCE).write_text('{}');(root/'GPU_PERMIT.json').write_text('{}')
        p=root/'execution/SCALE_REPAIR_GATES';p.mkdir(parents=True)
        v=dict(status='PASS',actual_cuda=True,method_variant=C.METHOD_VARIANT,
            state_scale_format=C.STATE_SCALE_FORMAT,no_AIME_or_calibration_or_formal_credit=True,
            original_gates=dict(status='PASS',actual_cuda=True,gates={k:dict(status='PASS',actual_cuda=True)for k in 'ABC'}),
            exponent_format_tests=dict(status='PASS',actual_cuda=True,method_variant=C.METHOD_VARIANT,
                state_scale_format=C.STATE_SCALE_FORMAT,accepted_domain_exact=True,exponent_dtype='int8',
                key_buffer_extension=False,CPU_GPU_parity=dict(status='PASS',actual_cuda=True),
                physical_exponent_metadata_counted=True,no_AIME_or_calibration_or_formal_credit=True,
                fixed_mask_period_cases=[dict(layer=i,decode_period=p,status='PASS',actual_cuda=True,
                    dense_state_persistent=False,key_buffer_extension=False)for i in C.GDN_LAYERS for p in (1,8)]))
        w=dict(task_id=C.TASK,execution_revision=C.REVISION,method_variant=C.METHOD_VARIANT,
            state_scale_format=C.STATE_SCALE_FORMAT,status='PASS',stage='scale_repair_gates',
            unit='SCALE_REPAIR_GATES',gpu=0,gpu_uuid=C.UUID,identity={},
            source_manifest_sha256=C.sha(root/C.SOURCE),permit_sha256=C.sha(root/'GPU_PERMIT.json'),result=v)
        return p,w

    def save_wrapper(self,p,w):
        v=copy.deepcopy(w);v['payload_sha256']=C.objsha(v);C.status(p/'RESULT.json',v)

    def test_related_gate_exact_variant_cuda_format_contract(self):
        with tempfile.TemporaryDirectory()as d:
            r=Path(d);p,w=self.gate_fixture(r);self.save_wrapper(p,w)
            self.assertEqual(C.verify_result(r,'scale_repair_gates','SCALE_REPAIR_GATES')['result']['status'],'PASS')
            for key,value in [('actual_cuda',False),('accepted_domain_exact',False),('exponent_dtype','float32'),
                              ('key_buffer_extension',True),('method_variant','original'),('state_scale_format','FP16')]:
                bad=copy.deepcopy(w);bad['result']['exponent_format_tests'][key]=value;self.save_wrapper(p,bad)
                with self.subTest(key=key),self.assertRaises(RuntimeError):C.verify_result(r,'scale_repair_gates','SCALE_REPAIR_GATES')
            bad=copy.deepcopy(w);bad['result']['exponent_format_tests']['CPU_GPU_parity']['status']='CPU_ONLY';self.save_wrapper(p,bad)
            with self.assertRaises(RuntimeError):C.verify_result(r,'scale_repair_gates','SCALE_REPAIR_GATES')

    def test_wrapper_requires_new_revision_and_canonical_payload_hash(self):
        with tempfile.TemporaryDirectory()as d:
            r=Path(d);p,w=self.gate_fixture(r)
            for key,value in [('execution_revision','old'),('method_variant','original'),('state_scale_format','FP32')]:
                bad=copy.deepcopy(w);bad[key]=value;self.save_wrapper(p,bad)
                with self.subTest(key=key),self.assertRaises(RuntimeError):C.verify_result(r,'scale_repair_gates','SCALE_REPAIR_GATES')
            self.save_wrapper(p,w);bad=C.read(p/'RESULT.json');bad['unit']='altered';C.status(p/'RESULT.json',bad)
            with self.assertRaises(RuntimeError):C.verify_result(r,'scale_repair_gates','SCALE_REPAIR_GATES')

    def controller_fixture(self, mode):
        with tempfile.TemporaryDirectory()as d:
            r=Path(d);(r/C.SOURCE).write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
            captured={};controller=None;target=r/'execution/SCALE_REPAIR_GATES'
            def identity(pid):return dict(pid=pid,starttick=123 if pid==456 else 999,cmdline=captured.get('argv',['controller'])if pid==456 else['controller'])
            class Child:
                pid=456;returncode=None
                def poll(self):return self.returncode
                def wait(self,timeout=None):return self.returncode
                def send_signal(self,s):captured['signal']=s;self.returncode=-s
            def popen(argv,**kw):
                captured['argv']=argv;captured['env']=kw['env'];child=Child()
                child.returncode=0 if mode=='success'else 1 if mode=='failure'else None
                if mode=='success':(target/'RESULT.json').write_text('{}')
                if mode=='signal_registration_window':controller.signal(signal.SIGTERM,None)
                return child
            patches=[mock.patch.object(C,'process_identity',side_effect=identity),mock.patch.object(C,'authorize',return_value={}),
                mock.patch.object(C,'model_guard',return_value={}),mock.patch.object(C,'device',return_value={}),mock.patch.object(C,'storage_guard',return_value={}),
                mock.patch.object(C,'query_gpu',return_value=dict(apps=[])),mock.patch.object(C,'unit_target',return_value=target),
                mock.patch.object(C,'verify_result',return_value=dict(status='PASS')),mock.patch.object(M.K,'dispatch_allowed',return_value=None),
                mock.patch.object(M.subprocess,'Popen',side_effect=popen),
                mock.patch.object(C,'same_process',side_effect=lambda expected,reader=None:expected==identity(expected['pid']))]
            for p in patches:p.start()
            try:
                controller=M.Controller(r)
                if mode=='success':controller.execute_unit('scale_repair_gates','SCALE_REPAIR_GATES')
                else:
                    with self.assertRaises(RuntimeError):controller.execute_unit('scale_repair_gates','SCALE_REPAIR_GATES')
                interval=C.read(target/'RESOURCE_INTERVAL.json');self.assertTrue(interval['released']);self.assertIsNotNone(interval['returncode'])
                self.assertIsNone(controller.current);self.assertEqual(len(controller.completed),int(mode=='success'))
                self.assertEqual(C.read(target/'DISPATCH.json')['method_variant'],C.METHOD_VARIANT)
                self.assertEqual(captured['env']['CUDA_VISIBLE_DEVICES'],C.UUID)
                if mode=='signal_registration_window':self.assertEqual(captured['signal'],signal.SIGTERM)
                if mode=='success':
                    with self.assertRaises(RuntimeError):controller.execute_unit('scale_repair_gates','SCALE_REPAIR_GATES')
            finally:
                for p in reversed(patches):p.stop()

    def test_controller_own_registration_and_once_success(self):self.controller_fixture('success')
    def test_controller_nonzero_exit_no_completion(self):self.controller_fixture('failure')
    def test_controller_signal_registration_window_cleanup(self):self.controller_fixture('signal_registration_window')

    def test_actual_controller_run_finite_gate_then57_or_stop(self):
        for fail_gate in (False, True):
            with self.subTest(fail_gate=fail_gate), tempfile.TemporaryDirectory() as d:
                r=Path(d);(r/C.SOURCE).write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
                for name in ('CONTINUATION_ADMISSION.json','REUSED_CALIBRATION_ADMISSION.json'):
                    (r/name).write_text('{}')
                gate=r/'execution/SCALE_REPAIR_GATES';gate.mkdir(parents=True);(gate/'RESULT.json').write_text('{}')
                jobs=C.build_jobs(C.read(ROOT/'AIME26_DATASET_FREEZE.json'))['jobs'][3:]
                events=[];controller=None
                def execute(stage,unit):
                    events.append((stage,unit))
                    if fail_gate and stage==M.K.GATE_STAGE:raise RuntimeError('mock finite newgate failure')
                    controller.completed.append(dict(stage=stage,unit=unit))
                    return dict(status='PASS')
                patches=[mock.patch.object(C,'process_identity',return_value=dict(pid=7,starttick=8,cmdline=['controller'])),
                    mock.patch.object(C,'authorize',return_value={}),mock.patch.object(C,'device',return_value={}),
                    mock.patch.object(C,'storage_guard',return_value={}),mock.patch.object(C,'model_guard',return_value={}),
                    mock.patch.object(C,'verify_external',return_value={}),mock.patch.object(C,'validate_jobs',return_value=jobs),
                    mock.patch.object(C,'verify_head_mask',return_value={}),mock.patch.object(C,'verify_science',return_value='science'),
                    mock.patch.object(M.K,'record_admission',return_value=dict(proofs=[{},{},{}])),
                    mock.patch.object(M.K,'admit_reused_science',return_value={}),
                    mock.patch.object(M.K,'remaining_jobs',return_value=jobs),
                    mock.patch.object(M.K,'completed_rows',return_value=[None]*60)]
                for p in patches:p.start()
                try:
                    controller=M.Controller(r)
                    with mock.patch.object(controller,'execute_unit',side_effect=execute):
                        self.assertEqual(controller.run(),1 if fail_gate else 0)
                    out=C.read(r/'SESSION_RESULT.json')
                    self.assertEqual(events[0],(M.K.GATE_STAGE,M.K.GATE_UNIT))
                    self.assertEqual(len(events),1 if fail_gate else 58)
                    self.assertEqual(out['current_formal_count'],0 if fail_gate else 57)
                    self.assertEqual(out['reused_formal_count'],3)
                    self.assertEqual(out['method_variant'],C.METHOD_VARIANT)
                    self.assertEqual(out['state_scale_format'],C.STATE_SCALE_FORMAT)
                    self.assertEqual(out['status'],'FAIL_CLOSED_STOP' if fail_gate else 'GPU_COMPLETE_PACE48_REUSE3_PLUS57_FORMAL')
                    self.assertEqual((r/'FORMAL_ADMISSION.json').exists(),not fail_gate)
                    self.assertEqual((r/'FIRST_FAILURE.json').exists(),fail_gate)
                    with self.assertRaises(RuntimeError):controller.run()
                finally:
                    for p in reversed(patches):p.stop()

    def test_publisher_uses_tmux18_one_exec_command(self):
        source=(ROOT/'operations/publish_and_launch_exponent_repair_once.py').read_text()
        self.assertIn("argv=['tmux','new-session','-d','-s',TMUX,command]",source)
        self.assertIn("command='exec '+shlex.join(['/bin/sh',str(r/'scripts/run_offline.sh')])",source)
        self.assertNotIn("TMUX,'--'",source)
        self.assertIn("p=Path(ARCHIVE)",source.replace("p=Path('''+repr(ARCHIVE)+''')","p=Path(ARCHIVE)"))
        self.assertIn('no_automatic_launch_retry=True',source)


if __name__=='__main__':unittest.main(verbosity=2)
