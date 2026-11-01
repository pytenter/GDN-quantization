"""Actual node controller with explicit CPU Popen/device/model mocks.

Checks mixed-origin seeding, no redispatch, new device qualification before formal,
dual-GPU host queues, fail-closed cleanup, and once-only restart rejection.
"""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import types

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'src'))
import execution_control as real
import continuation_admission as original_a
plan = real.read(R / 'GPU_EXECUTION_PLAN.json')
jobs = real.read(R / 'FORMAL_JOB_MANIFEST.json')['jobs']
proof = real.read(R / 'REUSE_ADMISSION_STATIC.json')
qids = [u['id'] for u in plan['qualification_units'] if u['id'] in proof['qualification']]
fids = [j['id'] for j in jobs if j['id'] in proof['formal']]
assert len(qids) == 19
checks = []
original_popen = subprocess.Popen
original_statvfs = os.statvfs
with tempfile.TemporaryDirectory(prefix='g64_v5_actual_node_cpu_mock_') as tmp:
    for scenario in ['success', 'qualification_failure']:
        F = Path(tmp) / scenario
        F.mkdir()
        (F / 'operations').mkdir()
        for n in ['CROSS_HOST_PLAN.json', 'QUESTION_GPU_ASSIGNMENT.json', 'AIME26_DATASET_FREEZE.json', 'FORMAL_JOB_MANIFEST.json', 'GPU_EXECUTION_PLAN.json']:
            (F / n).write_bytes((R / n).read_bytes())
        (F / 'GLOBAL_FORMAL_ADMISSION.json').write_text('{}')
        for n in ['SOURCE_FREEZE.json', 'GPU_PERMIT.json']:
            (F / n).write_text('{}')
        origin = F / '_CPU_ORIGIN'
        for unit in qids:
            d = origin / 'execution' / unit
            d.mkdir(parents=True)
            (d / 'RESULT.json').write_text(json.dumps(dict(result=dict(CPU_OLD_QUALIFICATION_FIXTURE=True))))
        fake = types.ModuleType('execution_control')
        for name in dir(real): setattr(fake, name, getattr(real, name))
        fake.sha = lambda p: '1' * 64 if p.name in ['SOURCE_FREEZE.json', 'GPU_PERMIT.json'] else real.sha(p)
        fake.authorize = lambda *a: dict(CPU_PERMIT_MOCK=True)
        fake.verify_sources = lambda *a: '1' * 64
        fake.model_guard = lambda *a: None
        fake.verify_external = lambda *a: 0
        fake.storage_guard = lambda *a: None
        fake.query_gpu = lambda: dict(apps=[], CPU_NVML_MOCK=True)
        fake.same_process = lambda i: False
        fake.reference_root = lambda: F / 'reference'
        fake.device = lambda g, *a, **k: real.require(g in real.UUIDS, 'Foreign CPU mock selector') or dict(CPU_DEVICE_MOCK=True)
        fake.worker_environment = lambda g: dict(os.environ, CUDA_VISIBLE_DEVICES=real.UUIDS[g], CPU_POPEN_MODEL_MOCK='1')
        children = {}
        counter = [900001]
        dispatches = []
        lock = threading.Lock()
        fake.process_identity = lambda pid: children.get(pid, dict(pid=pid, starttick=1, cmdline=['CPU_CONTROLLER_MOCK']))
        mock_a = types.ModuleType('continuation_admission')
        mock_a.admit_local_reuse = lambda root: dict(qualification_ids=qids, formal_ids=fids)
        mock_a.qualification_source = lambda root, unit: origin if unit in qids else F
        mock_a.host_reused_formal = lambda root: {i: proof['formal'][i] for i in fids}
        validator = types.ModuleType('record_validation')
        def validate(wrapper, job, root):
            assert wrapper['unit'] == job['id'] and wrapper['result']['condition'] == job['condition']
            assert wrapper['gpu_uuid'] == real.UUIDS[job['gpu']]
        validator.validate = validate
        mock_input = types.ModuleType('qualification_input_admission')
        mock_input.validate = lambda root: dict(CPU_INPUT_MOCK=True)
        mock_contract = types.ModuleType('cross_host_contract')
        def admission(root, value):
            qualified = real.read(F / 'GROUP64_GPU_CODEC_QUALIFICATION.json')
            assert set(qualified['units']) == {u['id'] for u in plan['qualification_units']}
            assert set(qualified['reused_units']) == set(qids)
        mock_contract.require_global_formal_admission = admission
        class Child:
            def __init__(self, cmd, stdout, stderr, env, start_new_session):
                assert start_new_session and Path(cmd[3]).name == 'gpu_unit_worker.py'
                unit = cmd[cmd.index('--unit') + 1]
                formal = '--formal' in cmd
                roster = jobs if formal else plan['qualification_units']
                job = next(x for x in roster if x['id'] == unit)
                assert unit not in qids + fids and env['CUDA_VISIBLE_DEVICES'] == real.UUIDS[job['gpu']]
                if formal:
                    admission(F, {})
                    assert (F / 'AUTHORIZATION_FORMAL.json').exists()
                with lock:
                    self.pid = counter[0]
                    counter[0] += 1
                    self.identity = dict(pid=self.pid, starttick=self.pid, cmdline=cmd)
                    children[self.pid] = self.identity
                    dispatches.append((unit, formal, job['gpu']))
                self.returncode = 1 if scenario == 'qualification_failure' and not formal else 0
                target = F / ('formal' if formal else 'execution') / unit
                assert (target / 'DISPATCH.json').exists()
                if self.returncode == 0:
                    row = dict(status='PASS', unit=unit, gpu=job['gpu'], gpu_uuid=real.UUIDS[job['gpu']],
                               permit_sha256='1' * 64, identity=self.identity,
                               result=dict(CPU_MODEL_FIXTURE=True, condition=job.get('condition')))
                    row['payload_sha256'] = real.objsha(row)
                    real.atomic_new(target / 'RESULT.json', row)
                    real.progress(target, 'CPU_MOCK_UNIT_PASS')
            def poll(self): return self.returncode
            def wait(self, timeout=None): return self.returncode
            def terminate(self): raise AssertionError('Already exited CPU child was signalled')
            def kill(self): raise AssertionError('Already exited CPU child was signalled')
        subprocess.Popen = Child
        os.statvfs = lambda path: types.SimpleNamespace(f_bavail=100 * 1024 ** 3, f_frsize=1) if Path(path) == F / 'reference' else original_statvfs(path)
        for name, value in [('execution_control', fake), ('continuation_admission', mock_a), ('record_validation', validator), ('qualification_input_admission', mock_input), ('cross_host_contract', mock_contract)]:
            sys.modules[name] = value
        sys.argv = [str(R / 'scripts/gpu_controller.py'), '--root', str(F), '--execute']
        expected_failure = scenario == 'qualification_failure' and len(real.UUIDS) == 2
        try: runpy.run_path(str(R / 'scripts/gpu_controller.py'), run_name='__main__')
        except RuntimeError:
            assert expected_failure
        else: assert not expected_failure
        assert len({x[0] for x in dispatches}) == len(dispatches)
        assert not any((F / 'formal' / unit / 'DISPATCH.json').exists() for unit in fids)
        assert not any((F / 'execution' / unit / 'DISPATCH.json').exists() for unit in qids)
        session = real.read(F / 'SESSION_RESULT.json')
        if expected_failure:
            assert session['status'] == 'FAIL_CLOSED_STOP' and (F / 'FIRST_FAILURE.json').exists()
            assert not any(x[1] for x in dispatches)
        else:
            assert session['status'] == 'GPU_COMPLETE_GROUP64_NODE_ALL_PLANNED'
            expected = {u['id'] for u in plan['qualification_units'] if u['id'] not in qids} | {j['id'] for j in jobs if j['id'] not in fids}
            assert {x[0] for x in dispatches} == expected
            assert set(session['completed']) == {u['id'] for u in plan['qualification_units']} | {j['id'] for j in jobs}
        intervals = real.read(F / 'RESOURCE_LEDGER.json')['intervals']
        assert len(intervals) == len(dispatches) and all(x['release'] == 'PASS' for x in intervals)
        before = len(dispatches)
        try: runpy.run_path(str(R / 'scripts/gpu_controller.py'), run_name='__main__')
        except RuntimeError: pass
        else: raise AssertionError('Controller replay accepted')
        assert len(dispatches) == before
        checks.append(dict(scenario=scenario, real_controller_source_executed=True, owned_CPU_Popen_calls=len(dispatches),
                           reused_qualification_not_dispatched=19, reused_formal_not_dispatched=len(fids),
                           unique_dispatch=True, restart_rejected=True, intervals_complete=True, expected_failure=expected_failure))
subprocess.Popen = original_popen
os.statvfs = original_statvfs
assert 'torch' not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES'] == ''
print(json.dumps(dict(status='PASS_ACTUAL_THREE_GPU_NODE_CONTROLLER_CPU_SCENARIOS', count=len(checks), checks=checks,
                     GPU_calls=0, model_forwards=0, worker_processes_devices_models_and_qualification='EXPLICIT_CPU_MOCKS',
                     qualification_credit=0, formal_generation_credit=0)))
