"""Actual lifecycle helper API ordering and exact qualification scope, CPU only."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import profiler_detach_lifecycle as lifecycle


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    checks = []

    def check(name, value):
        assert value, name
        checks.append(name)

    def reject(name, fn):
        try:
            fn()
        except (RuntimeError, ValueError):
            checks.append(name)
        else:
            raise AssertionError(name)

    class API:
        def __init__(self, name, code, calls):
            self.name, self.code, self.calls = name, code, calls
        def __call__(self, *params):
            self.calls.append((self.name, params))
            return self.code

    for name, active, flush, final, expected in (
            ('success', False, 0, 0, ['sync', 'flush', 'finalize']),
            ('active_rejected', True, 0, 0, []),
            ('flush_error_propagated', False, 1, 0, ['sync', 'flush']),
            ('finalize_error_propagated', False, 0, 2, ['sync', 'flush', 'finalize'])):
        calls, events = [], []
        torch = types.SimpleNamespace(autograd=types.SimpleNamespace(_profiler_enabled=lambda: active),
            cuda=types.SimpleNamespace(synchronize=lambda: calls.append(('sync', ()))))
        handle = types.SimpleNamespace(cuptiActivityFlushAll=API('flush', flush, calls),
                                       cuptiFinalize=API('finalize', final, calls))
        def execute():
            return lifecycle._detach_exact_pinned_handle(handle, torch, lambda phase, **v: events.append((phase, v)))
        if name == 'success':
            result = execute()
            check('successful_cleanup_still_requires_actual_exit', result['actual_process_exit_still_required'])
            check('successful_cleanup_does_not_suppress_signals_or_exit', not result['signal_or_exit_suppression'])
        else:
            reject(name, execute)
        check(name + '_actual_API_order', [x[0] for x in calls] == expected)

    candidate = ast.parse((ROOT / 'operations/profiler_detach_candidate.py').read_text())
    production = ast.parse((ROOT / 'src/profiler_detach_lifecycle.py').read_text())
    for name in ('_detach_exact_pinned_handle', 'detach_completed_profiler'):
        a = next(x for x in candidate.body if isinstance(x, ast.FunctionDef) and x.name == name)
        b = next(x for x in production.body if isinstance(x, ast.FunctionDef) and x.name == name)
        check(name + '_matches_two_actual_9B_exit0_engineering_path', ast.dump(a) == ast.dump(b))

    with tempfile.TemporaryDirectory(prefix='resident-profiler-lifecycle-cpu-') as temporary:
        root = Path(temporary)
        unit = dict(id='GPU1_H_PROFILE_SHORT', kind='profile_short', gpu=1, gpu_uuid='TEST-EXACT-UUID')
        (root / 'QUALIFICATION_JOB_MANIFEST.json').write_text(json.dumps(dict(units=[unit])))
        target = root / 'qualification' / unit['id']
        target.mkdir(parents=True)
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = unit['gpu_uuid']
        try:
            check('formal_or_nonunit_process_never_detaches', lifecycle._profile_target(root, 1, ['formal.py', '--job', 'aime']) is None)
            reject('duplicate_unit_argument_rejected', lambda: lifecycle._profile_target(root, 1, ['x', '--unit', unit['id'], '--unit', unit['id']]))
            reject('unknown_unit_rejected', lambda: lifecycle._profile_target(root, 1, ['x', '--unit', 'unknown']))
            reject('wrong_physical_gpu_rejected', lambda: lifecycle._profile_target(root, 2, ['x', '--unit', unit['id']]))
            check('initial_model_check_has_no_exported_trace_and_no_cleanup', lifecycle._profile_target(root, 1, ['x', '--unit', unit['id']]) is None)
            trace = target / 'CUDA_PROFILE.json'
            trace.write_text('{"traceEvents":[{"cat":"kernel"}]}')
            selected = lifecycle._profile_target(root, 1, ['x', '--unit', unit['id']])
            check('completed_profile_selects_exact_ordinary_target', selected == (target, trace, unit))
            trace.unlink()
            trace.symlink_to(root / 'never-created')
            reject('dangling_export_trace_symlink_rejected', lambda: lifecycle._profile_target(root, 1, ['x', '--unit', unit['id']]))
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous
    check('test_never_imported_Torch', 'torch' not in sys.modules)
    check('resident_math_stays_original', sha(ROOT / 'src/resident_rotation.py') == '4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95')
    check('actual_qualifier_stays_original', sha(ROOT / 'scripts/qualify_gpu.py') == 'a41b84fbecfe262b37211ba27804329955f23577ef46babade06a115ec88fff2')
    result = dict(status='PASS_ACTUAL_PROFILER_DETACH_HELPER_CPU_ORDER_ERROR_AND_SCOPE_CHECKS',
        check_count=len(checks), checks=checks, GPU_dispatches=0, model_forwards=0, Torch_imported=False,
        source_pins={n: sha(ROOT / n) for n in ['src/qualification_guards.py', 'src/profiler_detach_lifecycle.py', 'scripts/qualify_gpu.py']})
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], check_count=len(checks))))


if __name__ == '__main__':
    main()
