#!/usr/bin/env python3
"""Local stdlib mocks only; no real prepare/run/network/device operation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_delivery_tested_driver', HERE / 'seal_gpu_recovery_cpu.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
checks = []


def passed(name, ok):
    assert ok, name
    checks.append(dict(name=name, status='PASS'))


def denied(name, call, text=None):
    try:
        call()
    except RuntimeError as exc:
        assert text is None or text in str(exc), (name, str(exc))
        checks.append(dict(name=name, status='PASS'))
    else:
        raise AssertionError('Expected rejection: ' + name)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)


def row(path, name):
    b = path.read_bytes()
    return dict(path=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


with tempfile.TemporaryDirectory(prefix='synthetic_cpu_delivery_', dir=HERE) as temporary:
    p = Path(temporary)
    root = p / D.TASK
    saved = D.ROOT, D.FREEZE_SHA, D.PARENT_REVIEW_SHA, D.CONTROL_SHA
    D.ROOT = root
    try:
        for name in D.TOOLS:
            dest = root / 'delivery_tools' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((HERE / name).read_bytes())
        source_rows = []
        names = sorted(D.CONTROL_SHA) + ['inherited/file_%03d.txt' % i for i in range(405)]
        for name in names:
            dest = root / 'execution_payload' / name
            put(dest, 'synthetic immutable source ' + name + '\n')
            source_rows.append(row(dest, name))
        D.CONTROL_SHA = {name: row(root / 'execution_payload' / name, name)['sha256'] for name in D.CONTROL_SHA}
        freeze = dict(task_id=D.PARENT_TASK, ALLOW_GPU_RESTORE='NO', files=source_rows)
        put(root / 'execution_payload/SOURCE_FREEZE.json', freeze)
        D.FREEZE_SHA = hashlib.sha256((root / 'execution_payload/SOURCE_FREEZE.json').read_bytes()).hexdigest()
        proof = dict(task_id=D.TASK, status='PASS_SYNTHETIC407', files_count=407,
                     immutable_parent_sources=405, control_revision_sources=2, source_code_changes=0,
                     effective_source_freeze_sha256=D.FREEZE_SHA, files=source_rows,
                     actual_new_execution_controller_sha256=D.CONTROL_SHA['scripts/restoration_controller.py'],
                     actual_new_execution_worker_sha256=D.CONTROL_SHA['worker/restoration_worker.py'],
                     GPU_queries=0, GPU_dispatches=0)
        parent = dict(status='PASS_CPU_INDEPENDENT_PARENT_SEALS_AND_EFFECTIVE407_REVIEW', errors=[])
        for key in ['base_delivery', 'control_revision_delivery']:
            directory = p / key
            parent[key] = dict(delivery_root=str(directory))
            for filename, description in [('SHA256_MANIFEST.json', 'manifest'), ('SHA256SUMS.txt', 'SHA256SUMS'), ('FINAL_SEAL_RECEIPT.json', 'seal'), ('LOCAL_DELIVERY_VERIFICATION.json', 'local_delivery_verification')]:
                put(directory / filename, 'synthetic anchor ' + key + filename)
                parent[key][description] = row(directory / filename, filename)
        put(root / 'analysis/PARENT_SEALS_INDEPENDENT_REVIEW.json', parent)
        D.PARENT_REVIEW_SHA = hashlib.sha256((root / 'analysis/PARENT_SEALS_INDEPENDENT_REVIEW.json').read_bytes()).hexdigest()
        put(root / 'USER_REQUEST.txt', 'synthetic request explicitlyNO')
        state = dict(task_id=D.TASK, status=D.STATUS, restoration_mode=D.MODE, ALLOW_GPU_RESTORE='NO',
                     current_GPU_work_allowed=False, GPU_queries=0, GPU_dispatches=0, Driver_API_calls=0,
                     request_sha256=row(root / 'USER_REQUEST.txt', 'USER_REQUEST.txt')['sha256'],
                     effective_source_freeze_sha256=D.FREEZE_SHA,
                     parent_delivery=str(p / 'base_delivery'), control_revision_delivery=str(p / 'control_revision_delivery'),
                     remote_root='/data/qwen_gdn_historical_runtime_gpu_recovery_v1/synthetic_only',
                     archive_root='/data/qwen_historical_runtime_gpu_recovery_archive/synthetic_only',
                     current_remote_CPU_admission_receipt='qualification/REMOTE_CPU_ADMISSION_ATTEMPT02.json')
        auth = dict(task_id=D.TASK, ALLOW_GPU_RESTORE='NO', valid_execution_permit=None,
                    permit_created=False, separate_human_GPU_authorization_received=False, GPU_queries=0, GPU_dispatches=0)
        remote = dict(task_id=D.TASK, status='PASS_SYNTHETIC_REMOTE_CPU_NOT_REAL', GPU_queries=0, GPU_dispatches=0)
        put(root / 'ROOT.json', state)
        put(root / 'AUTHORIZATION_AND_PERMIT.json', auth)
        put(root / 'EFFECTIVE_SOURCE_VERIFICATION.json', proof)
        put(root / 'REMOTE_CPU_ADMISSION.json', dict(remote, status='FAILED_INITIAL_SSH_ATTEMPT_PRESERVED'))
        remote_name = state['current_remote_CPU_admission_receipt']
        put(root / remote_name, remote)
        put(root / 'FINAL_REPORT.md', 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED\n')
        put(root / 'FINAL_SCIENTIFIC_REPORT.md', 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED\n')
        review = dict(status='PASS_SYNTHETIC_FINAL_REVIEW', findings=[], checks_passed=5, checks_total=5,
                      receipts=[row(root / n, n) for n in sorted(D.REVIEW_REQUIRED | {remote_name})])
        review_path = root / 'analysis/final_review/FINAL_REPORT_REVIEW.json'
        put(review_path, review)
        passed('valid_synthetic407_NO_no_permit_admission', D.validate(root)[1]['files_verified'] == 407)
        passed('old_failed_CPU_receipt_retained_not_used_as_current', read_old_failed := json.loads((root / 'REMOTE_CPU_ADMISSION.json').read_text())['status'].startswith('FAILED_'))
        denied('foreign_or_parent_root_rejected', lambda: D.validate(p), 'independent CPU admission root')
        for key, value, message in [('ALLOW_GPU_RESTORE', 'YES', 'authorization'), ('GPU_queries', 1, 'GPU use'), ('Driver_API_calls', 1, 'device calls'), ('current_GPU_work_allowed', True, 'prohibited'), ('status', 'IN_PROGRESS', 'final CPU-only'), ('remote_root', '/data/oldtask/root', 'independent child'), ('archive_root', '/data/oldarchive/root', 'independent child')]:
            put(root / 'ROOT.json', dict(state, **{key: value}))
            denied('ROOT_' + key + '_rejected', lambda: D.validate(root), message)
            put(root / 'ROOT.json', state)
        for key, value in [('ALLOW_GPU_RESTORE', 'YES'), ('valid_execution_permit', {'forged': True}), ('permit_created', True), ('separate_human_GPU_authorization_received', True)]:
            put(root / 'AUTHORIZATION_AND_PERMIT.json', dict(auth, **{key: value}))
            denied('AUTH_' + key + '_rejected', lambda: D.validate(root), 'authorization' if key == 'ALLOW_GPU_RESTORE' else 'permit')
            put(root / 'AUTHORIZATION_AND_PERMIT.json', auth)
        for name in ['REMOTE_CPU_ADMISSION.json', '../escape.json', '/absolute.json', 'qualification/../escape.json']:
            put(root / 'ROOT.json', dict(state, current_remote_CPU_admission_receipt=name))
            denied('unsafe_current_CPU_receipt_' + name, lambda: D.validate(root))
        put(root / 'ROOT.json', state)
        link = root / 'qualification/linked_receipt.json'
        link.symlink_to(root / remote_name)
        put(root / 'ROOT.json', dict(state, current_remote_CPU_admission_receipt='qualification/linked_receipt.json'))
        denied('current_CPU_receipt_symlink_rejected', lambda: D.validate(root), 'symlink')
        put(root / 'ROOT.json', state)
        file = root / 'execution_payload' / source_rows[-1]['path']
        original = file.read_bytes()
        file.write_bytes(b'changed actual scientific source')
        denied('actual407_source_tamper_rejected', lambda: D.validate(root), 'Actual407 source')
        file.write_bytes(original)
        for value, name in [(dict(proof, files_count=406), 'count'), (dict(proof, actual_new_execution_worker_sha256='0' * 64), 'latest_worker'), (dict(proof, files=source_rows[:-1]), 'incomplete407')]:
            put(root / 'EFFECTIVE_SOURCE_VERIFICATION.json', value)
            denied('effective_proof_' + name + '_rejected', lambda: D.validate(root), 'Effective' if name != 'latest_worker' else 'control source proof')
        put(root / 'EFFECTIVE_SOURCE_VERIFICATION.json', proof)
        anchor = p / 'base_delivery/SHA256_MANIFEST.json'
        original = anchor.read_bytes()
        anchor.write_bytes(b'changed original parent manifest')
        denied('oldseal_anchor_change_rejected', lambda: D.validate(root), 'Prior sealed anchor')
        anchor.write_bytes(original)
        for value, name, message in [(dict(remote, status='FAIL'), 'FAIL', 'Remote CPU admission'), (dict(remote, GPU_dispatches=1), 'GPUuse', 'reports GPU use')]:
            put(root / remote_name, value)
            denied('remote_CPU_' + name + '_rejected', lambda: D.validate(root), message)
        put(root / remote_name, remote)
        put(root / 'FINAL_SCIENTIFIC_REPORT.md', 'different report')
        denied('report_alias_bytes_mismatch_rejected', lambda: D.validate(root), 'exact same bytes')
        put(root / 'FINAL_SCIENTIFIC_REPORT.md', 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED\n')
        for value, label in [(dict(review, findings=['unresolved']), 'findings'), (dict(review, checks_passed=4), 'counts'), (dict(review, receipts=review['receipts'][:-1]), 'missing_required_binding')]:
            put(review_path, value)
            denied('finalreview_' + label + '_rejected', lambda: D.validate(root))
        put(review_path, review)
        passed('synthetic_fixture_restored_to_PASS', D.validate(root)[1]['files_verified'] == 407)
        calls = []
        class FakeSealer:
            def source_roster(self, root):
                calls.append('source_roster')
                return [dict(path='synthetic.json', bytes=3, sha256='0' * 64)]
            def prepare(self, root, rows):
                calls.append('FAKE_prepare')
            def run(self, root, state, freeze, rows):
                calls.append('FAKE_run')
                return dict(status='PASS_FAKE_NOT_REAL_TRANSPORT')
        validator = lambda root: (state, {'files_verified': 407})
        loader = lambda root: FakeSealer()
        value = D.dispatch(root, validator=validator, loader=loader)
        passed('default_preview_does_not_prepare_run', value['status'].startswith('PREVIEW_') and calls == ['source_roster'])
        for action in ['prepare', 'run']:
            calls.clear()
            D.dispatch(root, action, validator, loader)
            passed(action + '_calls_only_fake_action', calls == ['source_roster', 'FAKE_' + action])
        calls.clear()
        denied('admission_failure_before_sealer_load', lambda: D.dispatch(root, 'run', lambda root: D.require(False, 'synthetic admission failure'), lambda root: calls.append('BAD')))
        passed('admission_failure_no_transfer_side_effect', calls == [])
    finally:
        D.ROOT, D.FREEZE_SHA, D.PARENT_REVIEW_SHA, D.CONTROL_SHA = saved

source = (HERE / 'seal_gpu_recovery_cpu.py').read_text()
tree = ast.parse(source)
passed('driver_no_runtime_engine_import', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(x in ast.unparse(n) for x in ['torch', 'pynvml', 'ctypes', 'cuda']) for n in ast.walk(tree)))
passed('driver_never_calls_old_main_validate_scope', not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'main', 'validate_scope'} for n in ast.walk(tree)))
scope_assignments = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Attribute) and ast.unparse(t).startswith('sealer.')]
passed('copied_sealer_only_TASK_relabel_not_math_or_scope_override', scope_assignments == ['sealer.TASK'])
old = (HERE / 'seal_cpu_delivery.py').read_text()
passed('inherited_one_shot_prepare_run_guards_preserved', 'Preparation already attempted: inspect, no blind retry' in old and 'Existing seal/receipt: no overwrite or reseal' in old and 'One-shot transfer state exists' in old)
passed('rawcopied_sealer_primitive_SHAs_exact', all(hashlib.sha256((HERE / n).read_bytes()).hexdigest() == h for n, h in D.TOOLS.items()))
result = dict(status='PASS_CPU_MOCK_AND_STATIC_ADMISSION_DELIVERY_DRIVER_TESTS', checks=checks,
              checks_passed=len(checks), GPU_queries=0, GPU_dispatches=0, real_prepare_calls=0, real_run_calls=0,
              network_calls=0, driver_sha256=hashlib.sha256(source.encode()).hexdigest(),
              limitations='Synthetic local407 fixture and fake sealer only. No actual driverpreview/prepare/run, model runtime, device or network executed.')
out = HERE / 'SEALER_DRIVER_CPU_TEST_RESULTS.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
