#!/usr/bin/env python3
"""CPU synthetic/static driver tests only. No real prepare/run or transport."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('revision_driver_under_test', HERE / 'seal_cpu_revision.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
checks = []


def passed(name, ok):
    assert ok, name
    checks.append(dict(name=name, status='PASS'))


def denied(name, call):
    try:
        call()
    except (RuntimeError, KeyError, ValueError):
        checks.append(dict(name=name, status='PASS'))
    else:
        raise AssertionError('Expected rejection: ' + name)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)


def describe(path, name):
    raw = path.read_bytes()
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


with tempfile.TemporaryDirectory(prefix='synthetic_revision_', dir=HERE) as tmp:
    base = Path(tmp)
    revision = base / D.REVISION
    original_root, original_parent = D.REVISION_ROOT, D.PARENT_ROOT
    D.REVISION_ROOT, D.PARENT_ROOT = revision, base
    try:
        for name in D.TOOL_SHA:
            p = revision / 'delivery_tools' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((HERE / name).read_bytes())
        for name in D.OVERLAYS | {'FINAL_REPORT.md', 'scripts/verify_effective_cpu_package.py'}:
            put(revision / name, 'cpu revision fixture\n')
        rows = [describe(revision / n, n) for n in sorted(D.OVERLAYS)]
        for i in range(405):
            name = 'inherited/source_%03d.txt' % i
            put(base / 'delivery' / name, 'immutable original source %03d\n' % i)
            rows.append(describe(base / 'delivery' / name, name))
        freeze = dict(task_id=D.TASK, ALLOW_GPU_RESTORE='NO', files=rows)
        put(revision / 'EFFECTIVE_SOURCE_FREEZE.json', freeze)
        state = dict(task_id=D.TASK, ALLOW_GPU_RESTORE='NO', status=D.STATUS,
                     GPU_queries=0, GPU_dispatches=0,
                     remote_root='/data/cpu_base/' + D.REVISION,
                     archive_root='/data/cpu_archive/' + D.REVISION,
                     effective_source_freeze_sha256=hashlib.sha256((revision / 'EFFECTIVE_SOURCE_FREEZE.json').read_bytes()).hexdigest())
        auth = dict(task_id=D.TASK, ALLOW_GPU_RESTORE='NO', GPU_queries=0, GPU_dispatches=0)
        review = dict(status='PASS_SYNTHETIC_REPORT_REVIEW', findings=[], checks_passed=3, checks_total=3,
                      receipts=[describe(revision / n, n) for n in sorted(D.OVERLAYS | {'FINAL_REPORT.md'})])
        put(revision / 'ROOT.json', state)
        put(revision / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json', auth)
        review_path = revision / 'analysis/final_review/FINAL_REPORT_REVIEW.json'
        put(review_path, review)
        vf = lambda _root: dict(result=dict(status='PASS_SYNTHETIC_LOCAL_CHECK', GPU_queries=0, GPU_dispatches=0))
        st, descriptor, _ = D.validate_revision(revision, vf)
        passed('exact407_405immutable_parent_plus2overlays', descriptor['files_verified'] == 407 and descriptor['inherited_parent_delivery_files'] == 405)
        denied('parent_original_root_rejected', lambda: D.validate_revision(base, vf))
        state_no = state.copy()
        for field, value, name in [('ALLOW_GPU_RESTORE', 'YES', 'ROOT_YES_rejected'), ('GPU_queries', 1, 'ROOT_GPUquery_rejected'), ('GPU_dispatches', 1, 'ROOT_GPUdispatch_rejected'), ('status', 'IN_PROGRESS', 'nonterminal_CPUstatus_rejected'), ('remote_root', '/data/cpu_base', 'old_remote_root_rejected'), ('archive_root', '/data/cpu_archive', 'old_archive_root_rejected'), ('effective_source_freeze_sha256', '0' * 64, 'changed_effective_freeze_binding_rejected')]:
            put(revision / 'ROOT.json', dict(state, **{field: value}))
            denied(name, lambda: D.validate_revision(revision, vf))
            put(revision / 'ROOT.json', state_no)
        put(revision / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json', dict(auth, ALLOW_GPU_RESTORE='YES'))
        denied('human_YES_rejected', lambda: D.validate_revision(revision, vf))
        put(revision / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json', auth)
        for value, name in [(dict(review, status='FAILED'), 'status'), (dict(review, findings=['issue']), 'findings'), (dict(review, checks_passed=2), 'counts'), (dict(review, receipts=review['receipts'][:-1]), 'missing_source_binding')]:
            put(review_path, value)
            denied('review_' + name + '_rejected', lambda: D.validate_revision(revision, vf))
        # The loop above uses report dictionaries only; restore the passing fixture.
        put(review_path, review)
        inherited = base / 'delivery' / rows[2]['path']
        inherited_before = inherited.read_bytes()
        inherited.write_bytes(b'changed')
        denied('changed_immutable_parent_source_rejected', lambda: D.validate_revision(revision, vf))
        inherited.write_bytes(inherited_before)
        put(revision / rows[2]['path'], 'untrusted mutable overlay')
        passed('unapproved_revision_shadow_does_not_override_original405', D.validate_revision(revision, vf)[1]['files_verified'] == 407)
        before = (revision / 'FINAL_REPORT.md').read_bytes()
        (revision / 'FINAL_REPORT.md').write_bytes(b'changed report')
        denied('changed_reviewed_FINAL_REPORT_rejected', lambda: D.validate_revision(revision, vf))
        (revision / 'FINAL_REPORT.md').write_bytes(before)
        original_tool = (revision / 'delivery_tools/seal_cpu_delivery.py').read_bytes()
        (revision / 'delivery_tools/seal_cpu_delivery.py').write_bytes(original_tool + b'\n')
        denied('changed_raw_copied_sealer_rejected', lambda: D.validate_revision(revision, vf))
        (revision / 'delivery_tools/seal_cpu_delivery.py').write_bytes(original_tool)
        for name in ['../escape', '/absolute', 'a/../escape', './noncanonical']:
            denied('unsafe_relative_' + name, lambda name=name: D.safe_relative(name))
        invocation = []
        def verifier_runner(argv, **kw):
            invocation.append((argv, kw))
            return subprocess.CompletedProcess(argv, 0, '{"status":"PASS_SYNTHETIC_CHECK","GPU_queries":0,"GPU_dispatches":0}', '')
        passed('local_verifier_PASS_JSON_accepted', D.local_verifier(revision, verifier_runner)['result']['status'].startswith('PASS_'))
        passed('verifier_CUDAhidden_NO_B_no_network_command', invocation[0][0] == [D.sys.executable, '-B', str(revision / 'scripts/verify_effective_cpu_package.py')] and invocation[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '' and invocation[0][1]['env']['ALLOW_GPU_RESTORE'] == 'NO')
        for rc, stdout, label in [(1, '{"status":"PASS_FAKE"}', 'nonzero'), (0, 'log\n{"status":"PASS_FAKE"}', 'multiple_stdout'), (0, '{"status":"FAILED"}', 'notPASS'), (0, '{"status":"PASS_FAKE","GPU_queries":1}', 'GPUreported')]:
            bad = lambda argv, rc=rc, stdout=stdout, **_kw: subprocess.CompletedProcess(argv, rc, stdout, 'retained stderr')
            denied('verifier_' + label + '_rejected', lambda bad=bad: D.local_verifier(revision, bad))
        calls = []
        class FakeSealer:
            def source_roster(self, root):
                calls.append('source_roster')
                return [dict(path='small.json', bytes=3, sha256='f' * 64)]
            def prepare(self, root, rows):
                calls.append('FAKE_prepare_only')
            def run(self, root, state, freeze, rows):
                calls.append('FAKE_run_only')
                return dict(status='PASS_SYNTHETIC_TRANSPORT_NOT_EXECUTED')
        validator = lambda root: (state, descriptor, {})
        loader = lambda root: FakeSealer()
        result = D.dispatch(revision, validator=validator, loader=loader)
        passed('default_preview_never_calls_prepare_run', calls == ['source_roster'] and result['status'].startswith('PREVIEW_'))
        calls.clear()
        D.dispatch(revision, 'prepare', validator, loader)
        passed('prepare_dispatch_only_synthetic_prepare', calls == ['source_roster', 'FAKE_prepare_only'])
        calls.clear()
        D.dispatch(revision, 'run', validator, loader)
        passed('run_dispatch_only_synthetic_run', calls == ['source_roster', 'FAKE_run_only'])
        calls.clear()
        denied('unknown_action_rejected_before_loader', lambda: D.dispatch(revision, 'unknown', validator, loader))
        passed('unknown_action_no_transfer_side_effect', calls == [])
        denied('failed_admission_cannot_load_sealer', lambda: D.dispatch(revision, 'run', lambda root: D.require(False, 'synthetic denial'), lambda root: calls.append('BAD')))
        passed('failed_admission_has_no_loader_transfer_side_effect', calls == [])
    finally:
        D.REVISION_ROOT, D.PARENT_ROOT = original_root, original_parent

tree = ast.parse((HERE / 'seal_cpu_revision.py').read_text())
called_attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
passed('no_original_main_or_validate_scope_called', 'validate_scope' not in called_attrs and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'main' for n in ast.walk(tree)))
passed('driver_no_engine_or_GPUquery_imports', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(k in ast.unparse(n) for k in ['torch', 'cuda', 'pynvml', 'nvml']) for n in ast.walk(tree)))
passed('copied_sealer_and_primitives_exact_rawSHA', all(hashlib.sha256((HERE / n).read_bytes()).hexdigest() == h for n, h in D.TOOL_SHA.items()))
result = dict(status='PASS_CPU_SYNTHETIC_AND_STATIC_REVISION_SEALER_DRIVER_TESTS',
              checks=checks, checks_passed=len(checks), GPU_queries=0, GPU_dispatches=0,
              real_prepare_calls=0, real_run_calls=0, network_calls=0,
              copied_tools=D.TOOL_SHA,
              driver_sha256=hashlib.sha256((HERE / 'seal_cpu_revision.py').read_bytes()).hexdigest(),
              limitations='Synthetic temporary local fixtures and fake sealer calls only. Actual effective407 validation/prepare/run/transport not performed by this test.')
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
