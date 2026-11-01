#!/usr/bin/env python3
"""CPU-only control contracts and real controller-path mocks, no CUDA/model/SSH."""
import copy
import importlib.util
import json
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
spec = importlib.util.spec_from_file_location('pace_controller_cpu_review', ROOT / 'scripts/pace_controller.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def module(name, file):
    spec=importlib.util.spec_from_file_location(name, ROOT/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
F=module('pace_finisher_cpu_review','scripts/finalize_cpu.py')
P=module('pace_parent_cpu_review','scripts/offline_parent.py')


class Contracts(unittest.TestCase):
    def permit(self):
        metadata = dict(task_id=C.TASK, remote_root=C.REMOTE, archive_root=C.ARCHIVE)
        grant = dict(task_id=C.TASK, remote_root=C.REMOTE, authorized_physical_GPUs=[0],
                     authorized_UUIDs=[C.UUID], base_authorization_sha256=C.BASE_GRANT_SHA,
                     formal_new_samples=60, seed=1, max_new_tokens=81920)
        permit = dict(task_id=C.TASK, ALLOW_GPU='YES', execution_root=C.REMOTE,
                      physical_indices=[0], devices=[C.UUID], source_manifest_sha256='s',
                      method_spec_sha256='m', human_authorization_sha256='h', formal_new_samples=60,
                      seed=1, max_new_tokens=81920, watchdog_seconds=1800, cleanup_seconds=120,
                      GPUh_hard_cap=None, total_wall_cap=None, unit_elapsed_cap=None)
        return metadata, permit, grant

    def test_exact_permit_and_all_relevant_negatives(self):
        C.validate_permit_values(*self.permit(), 's', 'm', 'h')
        mutations = [('execution_root', '/tmp/foreign'), ('physical_indices', [1]),
                     ('devices', ['foreign']), ('source_manifest_sha256', 'x'), ('seed', 2),
                     ('formal_new_samples', 59), ('max_new_tokens', 2048), ('watchdog_seconds', 30),
                     ('GPUh_hard_cap', 1), ('total_wall_cap', 1), ('unit_elapsed_cap', 1)]
        for key, value in mutations:
            with self.subTest(key=key):
                a, p, g = self.permit(); p[key] = value
                with self.assertRaises(RuntimeError): C.validate_permit_values(a, p, g, 's', 'm', 'h')
        a, p, g = self.permit(); g['authorized_UUIDs'] = ['foreign']
        with self.assertRaises(RuntimeError): C.validate_permit_values(a, p, g, 's', 'm', 'h')

    def test_exact60_roster_order_no_new_questions_or_seeds(self):
        data = C.read(ROOT / 'AIME26_DATASET_FREEZE.json'); rows = C.build_jobs(data)['jobs']
        self.assertEqual(len(rows), 60)
        self.assertEqual([x['condition'] for x in rows], ['R2', 'R3'] * 30)
        self.assertEqual(len({x['id'] for x in rows}), 60)
        self.assertEqual({x['seed'] for x in rows}, {1})
        self.assertEqual({x['max_new_tokens'] for x in rows}, {81920})
        invalid = copy.deepcopy(data); invalid['records'][1] = invalid['records'][0]
        with self.assertRaises(RuntimeError): C.build_jobs(invalid)

    def test_no_old_root_execution_before_any_device_API(self):
        with mock.patch.object(C, 'query_gpu', side_effect=AssertionError('device API forbidden')):
            with self.assertRaises(RuntimeError): C.authorize(ROOT)

    def test_static_execution_directories_reject_unbound_shadow_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'src').mkdir();bound=root/'src/entry.py';bound.write_text('# bound')
            C.atomic_new(root/C.SOURCE,dict(task_id=C.TASK,count=1,files=[dict(path='src/entry.py',bytes=bound.stat().st_size,sha256=C.sha(bound))]))
            C.verify_sources(root)
            (root/'src/__pycache__').mkdir();(root/'src/__pycache__/ignored.pyc').write_bytes(b'CPU fixture ignored')
            C.verify_sources(root)
            (root/'src/shadow.py').write_text('# must reject')
            with self.assertRaises(RuntimeError):C.verify_sources(root)

    def test_only_GPU0_own_UUID_admission_ignores_other_GPU_owners(self):
        obs = dict(devices=[dict(physical_index=0, uuid=C.UUID, name='RTX3090', memory_total_MiB=24576,
                                 memory_free_MiB=24000)], apps=[dict(pid=999, uuid='GPU_FOREIGN5')])
        with mock.patch.object(C, 'query_gpu', return_value=obs): self.assertEqual(C.device(), obs)
        obs['apps'].append(dict(pid=33, uuid=C.UUID))
        with mock.patch.object(C, 'query_gpu', return_value=obs):
            with self.assertRaises(RuntimeError): C.device()
            self.assertEqual(C.device(33), obs)
        obs['apps'].append(dict(pid=33, uuid='GPU_FOREIGN6'))
        with mock.patch.object(C, 'query_gpu', return_value=obs):
            with self.assertRaises(RuntimeError): C.device(33)

    def test_environment_excludes_private_group64_paths_without_disabling_user_site(self):
        with mock.patch.dict(os.environ, {'PYTHONPATH':'/tmp/G64', 'PYTHONHOME':'bad', 'PYTHONUSERBASE':'bad'}):
            env = C.worker_environment()
        self.assertTrue(all(k not in env for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE')))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], C.UUID)
        self.assertNotIn('PYTHONNOUSERSITE', env)

    def test_release_truth_table_and_PID_reuse(self):
        empty = dict(apps=[]); own = dict(apps=[dict(pid=11, uuid=C.UUID)])
        identity = dict(pid=11, starttick=22, cmdline=['our-worker'])
        self.assertFalse(C.release_state(False, empty, 11))
        self.assertFalse(C.release_state(True, None, 11))
        self.assertFalse(C.release_state(True, own, 11, identity, reader=lambda p: identity))
        self.assertTrue(C.release_state(True, empty, 11))
        reused = dict(pid=11, starttick=99, cmdline=['foreign'])
        self.assertTrue(C.release_state(True, own, 11, identity, reader=lambda p: reused))
        self.assertFalse(C.release_state(False, own, 11, identity, reader=lambda p: reused))
        with mock.patch.object(C, 'process_identity', side_effect=FileNotFoundError):
            self.assertFalse(C.release_state(True, own, 11))

    def test_signals_require_PID_starttick_and_full_argv(self):
        class P:
            pid = 11
            def poll(self): return None
            def send_signal(self, s): self.signal = s
        p = P(); identity = dict(pid=11, starttick=22, cmdline=['worker'])
        C.terminate_owned(p, identity, reader=lambda pid: identity)
        self.assertEqual(p.signal, signal.SIGTERM)
        C.terminate_owned(p, identity, reader=lambda pid: identity, kill=True)
        self.assertEqual(p.signal, signal.SIGKILL)
        for bad in (dict(pid=11, starttick=23, cmdline=['worker']),
                    dict(pid=11, starttick=22, cmdline=['foreign'])):
            with self.assertRaises(RuntimeError): C.terminate_owned(p, identity, reader=lambda pid: bad)
        with self.assertRaises(RuntimeError): C.terminate_owned(p, None)

    def formal_fixture(self, root):
        (root / 'calibration').mkdir(); (root / 'calibration/head_mask.json').write_text('{}')
        (root / 'protocol').mkdir()
        for name in ('sampler.py','common.py','aime26_scorer_v4.py','aime26_frozen.jsonl'):
            (root / 'protocol' / name).write_text(name)
        (root / 'AIME26_DATASET_FREEZE.json').write_text('{}')
        job = dict(condition='R2', problem_id='aime26_01', seed=1, input_ids_sha256='input',
                   input_tokens=3, decode_period=8)
        tokens = [10, 11, 99]; total = 5
        store = dict(dense_state_persistent=False, fixture_only=False, tokens=total, prompt_length=3,
                     decode_tokens=2, pending_tokens=2, writeback_count=1, quantization_count=1,
                     finalization_policy='preserve')
        row = dict(status='COMPLETE', runtime_error=None, nonfinite=False, deterministic_algorithms=False,
            condition='R2', problem_id='aime26_01', seed=1, input_ids_sha256='input', input_tokens=3,
            head_mask_sha256=C.sha(root / 'calibration/head_mask.json'), generated_token_ids=tokens,
            generated_token_count=3, generated_token_ids_sha256=C.objsha(tokens), stop_token_ids=[99],
            finish_reason='eos', actual_model_forward_calls=5, sampler_calls=3,
            quantizer_audit=dict(layers=list(C.GDN_LAYERS), actual_layer_coverage=24,
                helper_calls={str(i):5 for i in C.GDN_LAYERS}, dense_recurrent_carrier=False,
                decode_period=8,prefill_period=256,output_before_state_writeback=True,marker_checks=24*4,
                storage_by_layer={str(i):copy.deepcopy(store) for i in C.GDN_LAYERS},
                native_first_chunk_compat_calls={str(i):1 for i in C.GDN_LAYERS},
                native_first_chunk_state_checked={str(i):1 for i in C.GDN_LAYERS},
                native_first_chunk_state_retained=False,native_first_chunk_state_discarded=True),
            protocol_identity={n:C.sha(root/'protocol'/n) for n in ('sampler.py','common.py','aime26_scorer_v4.py','aime26_frozen.jsonl')},
            dataset_freeze_sha256=C.sha(root / 'AIME26_DATASET_FREEZE.json'))
        return row, job

    def test_formal_contract_and_semantic_negatives(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); good, job = self.formal_fixture(root); C.validate_formal(good, job, root)
            for key, value in [('actual_model_forward_calls',6), ('seed',2), ('finish_reason','length'),
                               ('generated_token_ids_sha256','foreign'), ('head_mask_sha256','foreign')]:
                r = copy.deepcopy(good); r[key] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError): C.validate_formal(r,job,root)
            for key, value in [('dense_recurrent_carrier',True), ('decode_period',1), ('marker_checks',0)]:
                r = copy.deepcopy(good); r['quantizer_audit'][key] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError): C.validate_formal(r,job,root)
            r = copy.deepcopy(good); r['quantizer_audit']['storage_by_layer']['0']['pending_tokens'] = 0
            with self.assertRaises(RuntimeError): C.validate_formal(r,job,root)
            r = copy.deepcopy(good); r['protocol_identity'].pop('sampler.py')
            with self.assertRaises(RuntimeError): C.validate_formal(r,job,root)
            for key in ('native_first_chunk_compat_calls','native_first_chunk_state_checked'):
                r=copy.deepcopy(good);r['quantizer_audit'][key]['0']=0
                with self.assertRaises(RuntimeError):C.validate_formal(r,job,root)
            r=copy.deepcopy(good);r['quantizer_audit']['native_first_chunk_state_retained']=True
            with self.assertRaises(RuntimeError):C.validate_formal(r,job,root)

    def controller_fixture(self, mode):
        """Exercise real execute_unit registration/finally/ledger with no OS child/GPU."""
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name)
        (root / C.SOURCE).write_text('{}'); (root / 'GPU_PERMIT.json').write_text('{}')
        controller = None; captured = {}
        def identity(pid):
            return dict(pid=pid, starttick=123 if pid == 456 else 999,
                        cmdline=captured.get('argv', ['controller']) if pid == 456 else ['controller'])
        class Child:
            pid = 456
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout=None): return self.returncode
            def send_signal(self, s): captured['signal']=s; self.returncode=-s
        def popen(argv, **kwargs):
            captured['argv']=argv; captured['environment']=kwargs['env']
            child = Child()
            child.returncode = 0 if mode == 'success' else 1 if mode == 'failure' else None
            if mode == 'success': (root/'execution/GATES_ABC/RESULT.json').write_text('{}')
            if mode == 'signal_registration_window': controller.signal(signal.SIGTERM, None)
            return child
        patches = [mock.patch.object(C,'process_identity', side_effect=identity),
            mock.patch.object(C,'authorize',return_value={}),mock.patch.object(C,'model_guard',return_value={}),
            mock.patch.object(C,'device',return_value={}),mock.patch.object(C,'storage_guard',return_value={}),
            mock.patch.object(C,'query_gpu',return_value=dict(apps=[])),
            mock.patch.object(C,'unit_target',return_value=root/'execution/GATES_ABC'),
            mock.patch.object(C,'verify_result',return_value=dict(status='PASS')),
            mock.patch.object(M.subprocess,'Popen',side_effect=popen),
            mock.patch.object(C,'same_process',side_effect=lambda expected,reader=None: expected==identity(expected['pid']))]
        for p in patches: p.start()
        try:
            controller=M.Controller(root)
            if mode == 'success': controller.execute_unit('math_gpu_gates','GATES_ABC')
            else:
                with self.assertRaises(RuntimeError): controller.execute_unit('math_gpu_gates','GATES_ABC')
            interval=C.read(root/'execution/GATES_ABC/RESOURCE_INTERVAL.json')
            self.assertTrue(interval['released']); self.assertIsNotNone(interval['returncode'])
            self.assertIsNone(controller.current)
            self.assertEqual(len(controller.completed), 1 if mode=='success' else 0)
            self.assertGreaterEqual(interval['GPU_seconds'],0)
            if mode == 'signal_registration_window':
                self.assertEqual(captured['signal'], signal.SIGTERM)
                self.assertTrue((root/'execution/GATES_ABC/ACTUAL_CHILD_IDENTITY.json').exists())
            if mode=='success':
                with self.assertRaises(RuntimeError): controller.execute_unit('math_gpu_gates','GATES_ABC')
        finally:
            for p in reversed(patches): p.stop()
            temporary.cleanup()

    def test_controller_actual_success_path_and_no_rerun(self): self.controller_fixture('success')
    def test_controller_actual_nonzero_exit_no_completion_credit(self): self.controller_fixture('failure')
    def test_controller_signal_between_Popen_and_registration_cleans_only_owned_child(self):
        self.controller_fixture('signal_registration_window')

    def test_partial_excludes_invalid_exit_but_preserves_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'raw';target.mkdir();(target/'RESULT.json').write_text('raw unchanged')
            C.atomic_new(target/'RESOURCE_INTERVAL.json',dict(returncode=1,released=True,identity={}))
            with mock.patch.object(C,'validate_jobs',return_value=[dict(id='one',condition='R2')]),\
                 mock.patch.object(C,'unit_target',return_value=target):
                invalid=[];self.assertEqual(F.complete_rows(root,invalid),[])
            self.assertEqual(len(invalid),1);self.assertEqual(invalid[0]['formal_completion_credit'],0)
            self.assertEqual((target/'RESULT.json').read_text(),'raw unchanged')

    def test_CPU_parent_two_real_paths_and_no_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/C.SOURCE).write_text('{}');(root/'GPU_PERMIT.json').write_text('{}')
            argv_by_pid={};counter=[400]
            def identity(pid):return dict(pid=pid,starttick=pid+1,cmdline=argv_by_pid.get(pid,['parent']))
            class Child:
                returncode=0
                def poll(self):return 0
            def popen(argv,**kw):
                counter[0]+=1;child=Child();child.pid=counter[0];argv_by_pid[child.pid]=argv
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');return child
            with mock.patch.object(C,'authorize',return_value={}),mock.patch.object(C,'process_identity',side_effect=identity),\
                 mock.patch.object(C,'same_process',return_value=False),mock.patch.object(P.subprocess,'Popen',side_effect=popen):
                parent=P.Parent(root);self.assertEqual(parent.run(),0)
                with self.assertRaises(RuntimeError):parent.run()
            self.assertTrue(C.read(root/'PARENT_CONTROLLER_EXIT.json')['actual_exited'])
            self.assertTrue(C.read(root/'PARENT_FINISHER_EXIT.json')['identity_gone'])
            self.assertEqual(counter[0],402)
            self.assertTrue(C.read(root/'OFFLINE_PARENT_RESULT.json')['final_delivery_requires_observer_confirm_this_parent_identity_gone'])

    def test_archive_dual_SHA_NOREPLACE_non_circular_and_tensor_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);C.atomic_new(root/'ROOT.json',dict(archive_root=C.ARCHIVE))
            for name in (C.SOURCE,'GPU_PERMIT.json'): (root/name).write_text('{}')
            C.atomic_new(root/'final_verdict.json',dict(status='FAIL_CLOSED_STOP'))
            (root/'raw.pt').write_bytes(b'CPU fixture not tensor');(root/'text.json').write_text('{}')
            calls=[];proof={}
            class Result:
                returncode=0;stdout=b'{}';stderr=b''
            def run(argv,**kwargs):
                r=Result();calls.append(argv)
                if argv[0]=='rsync':
                    listing=Path(next(x.split('=',1)[1] for x in argv if x.startswith('--files-from='))).read_text()
                    self.assertIn('DELIVERY_MANIFEST.json\n',listing)
                    self.assertNotIn('DELIVERY_READY.json',listing)
                    self.assertEqual(argv[-1],F.STORAGE+':'+C.ARCHIVE+'.staging_'+C.sha(root/'DELIVERY_MANIFEST.json')+'/')
                elif '/bin/cat' in argv[-1]:r.stdout=C.canonical(proof)
                else:
                    self.assertIn(F.STORAGE,argv)
                    self.assertNotIn('nvidia-smi',argv[-1]);self.assertNotIn('SIGTERM',argv[-1])
                    if 'renameat2' in argv[-1]:
                        m=C.read(root/'DELIVERY_MANIFEST.json')
                        proof.update(status='PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES',manifest_sha256=C.sha(root/'DELIVERY_MANIFEST.json'),
                                     files=m['count'],bytes=m['total_bytes'],archive_root=C.ARCHIVE,GPU_calls=0)
                        r.stdout=C.canonical(proof)
                    else:r.stdout=b'{"status":"CREATED_NEW_EXCLUSIVE_STAGE"}'
                return r
            with mock.patch.object(F.subprocess,'run',side_effect=run):
                F.archive(root)
                with self.assertRaises(RuntimeError):F.archive(root)
            m=C.read(root/'DELIVERY_MANIFEST.json')
            self.assertFalse(next(x for x in m['files'] if x['path']=='raw.pt')['download_small'])
            self.assertNotIn('DELIVERY_MANIFEST.json',[x['path'] for x in m['files']])
            self.assertEqual(C.read(root/'DELIVERY_READY.json')['final_status'],'FAIL_CLOSED_STOP')
            self.assertEqual(len(calls),4)

    def test_finite_pipeline_full60_and_gate_failure_barrier(self):
        for fail_D in (False,True):
            with self.subTest(fail_D=fail_D),tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/C.SOURCE).write_text('{}');(root/'GPU_PERMIT.json').write_text('{}')
                jobs=C.build_jobs(C.read(ROOT/'AIME26_DATASET_FREEZE.json'))['jobs'];events=[]
                with mock.patch.object(C,'process_identity',return_value=dict(pid=123,starttick=456,cmdline=['CPU fixture'])):
                    controller=M.Controller(root)
                def execute(stage,unit):
                    events.append((stage,unit))
                    if stage=='gate_d' and fail_D:raise RuntimeError('synthetic bounded D failure')
                    folder=root/'execution'/unit;folder.mkdir(parents=True,exist_ok=True);(folder/'RESULT.json').write_text('{}')
                    controller.completed.append(dict(stage=stage,unit=unit));return dict(status='PASS')
                def freeze(r):events.append(('science_freeze','after_mask_before_D'))
                with mock.patch.object(C,'authorize',return_value={}),mock.patch.object(C,'device',return_value={}),\
                     mock.patch.object(C,'storage_guard',return_value={}),mock.patch.object(C,'model_guard',return_value={}),\
                     mock.patch.object(C,'verify_external',return_value={}),mock.patch.object(C,'validate_jobs',return_value=jobs),\
                     mock.patch.object(C,'verify_head_mask',return_value={}),mock.patch.object(C,'freeze_science',side_effect=freeze),\
                     mock.patch.object(C,'verify_science',return_value='science_sha'),\
                     mock.patch.object(controller,'execute_unit',side_effect=execute):
                    self.assertEqual(controller.run(),1 if fail_D else 0)
                self.assertEqual(events[:4],[('math_gpu_gates','GATES_ABC'),('calibration','CALIBRATION_ONCE'),
                                             ('science_freeze','after_mask_before_D'),('gate_d','FULL_MODEL_D')])
                result=C.read(root/'SESSION_RESULT.json')
                self.assertEqual(result['formal_completed'],0 if fail_D else 60)
                self.assertEqual(len(events),4 if fail_D else 64)
                self.assertEqual((root/'FORMAL_ADMISSION.json').exists(),not fail_D)
                self.assertEqual(result['status'],'FAIL_CLOSED_STOP' if fail_D else 'GPU_COMPLETE_PACE48_60_FORMAL')

if __name__=='__main__':
    unittest.main(verbosity=2)
