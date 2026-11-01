"""Independent CPU-only audit of which old qualification receipts remain applicable."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    old = Path(json.loads((ROOT / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json').read_text())['remote_prior_root'])
    sys.path[:0] = [str(old / 'src')]
    import resident_execution_control as C
    assert C.verify_sources(old) == '5be0a7471840d29579560685f98952f2a74ac9bb9b1b6ad85c6e6e0f78563092'
    assert sha(old / 'FINAL_SEAL_RECEIPT.json') == '2584b2bf29ea874494d2835997613dfb08dae81b20e70e0b21d82028a0d4f0b1'
    original = ast.parse((old / 'src/qualification_guards.py').read_text())
    changed = ast.parse((ROOT / 'src/qualification_guards.py').read_text())
    changed.body = [n for n in changed.body if not (isinstance(n, ast.Import)
        and [a.name for a in n.names] == ['importlib.util'])]
    fn = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == 'check_after_model')
    names = {'lifecycle_path', 'spec', 'lifecycle', 'profiler_lifecycle'}
    fn.body = [n for n in fn.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
        and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'exec_module')]
    returned = next(n for n in fn.body if isinstance(n, ast.Return))
    returned.value.keywords = [k for k in returned.value.keywords if k.arg != 'profiler_native_lifecycle']
    assert ast.dump(original, include_attributes=False) == ast.dump(changed, include_attributes=False), 'Guard changes exceed isolated profiler lifecycle hook'
    frozen = C.read(old / 'SOURCE_FREEZE.json')
    stable = []
    for row in frozen['files']:
        if row['path'].startswith(('src/', 'vendor/', 'assets/')) or row['path'] in [
            'scripts/qualify_gpu.py', 'QUALIFICATION_JOB_MANIFEST.json', 'FORMAL_JOB_MANIFEST.json',
            'NUMERICAL_QUALIFICATION_CONTRACT.json', 'AIME26_SEED1_PROTOCOL.json',
            'RESIDENT_ROTATION_MATRICES_SHA.json', 'H128_MATRIX_SHA.json', 'analysis/RUNTIME_NATIVE_CLOSURE.json',
            'analysis/EXTERNAL_BINDINGS.json']:
            if row['path'] == 'src/qualification_guards.py':
                continue
            p = ROOT / row['path']
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
            stable.append(row)
    old_units = C.read(old / 'QUALIFICATION_JOB_MANIFEST.json')['units']
    assert old_units == C.read(ROOT / 'QUALIFICATION_JOB_MANIFEST.json')['units']
    session = C.read(old / 'SESSION_RESULT.json')
    assert session['qualification_completed'] == 48 and session['formal_completed'] == 0 and session['own_release_all_PASS']
    by = {u['id']: u for u in old_units}
    qualified = []
    for receipt in session['qualification_receipts']:
        unit = by[receipt['unit']]
        assert not unit['kind'].startswith('profile_')
        target = old / 'qualification' / unit['id']
        result = C.validate_qualification_result(old, unit)
        identity = C.read(target / 'STARTED.json')['identity']
        interval = C._interval_ok(old, target, identity)
        for key, name in [('result_sha256', 'RESULT.json'), ('resource_interval_sha256', 'RESOURCE_INTERVAL.json'),
                          ('validation_sha256', 'CONTROLLER_VALIDATION.json')]:
            assert receipt[key] == sha(target / name)
        qualified.append(dict(unit=unit, origin_root=str(old), origin_source_freeze_sha256=sha(old/'SOURCE_FREEZE.json'),
            origin_permit_sha256=sha(old/'GPU_PERMIT.json'), controller_receipt=receipt,
            identity=identity, result_status=result['status'], actual_exit_returncode=interval['exit_returncode'],
            actual_release=interval['release'], source_guard_change_inapplicable_to_nonprofiler=True))
    assert len(qualified) == len({x['unit']['id'] for x in qualified}) == 48
    passed = {x['unit']['id'] for x in qualified}
    remaining = [u for u in old_units if u['id'] not in passed]
    assert len(remaining) == 10 and all(u['kind'].startswith(('perf_', 'profile_')) for u in remaining)
    failure = C.read(old / 'FIRST_FAILURE.json')
    assert 'GPU1_H_PROFILE_SHORT' in json.dumps(failure)
    assert 'torch' not in sys.modules
    result = dict(status='PASS_CPU_APPLICABILITY_48_REUSABLE_10_REMAIN_NOT_GPU_ADMISSION',
        human_request='资格测试可以在前面的基础上接着跑吗，不然太耗费时间了',
        reused_count=48, nonperformance_reusable=44, prior_unprofiled_performance_reusable=4,
        remaining_count=10, qualified=qualified, remaining=remaining,
        unchanged_relevant_sources=stable, guard_AST_exact_except_scoped_profiler_hook=True,
        original_H_profile_nonzero_exit_preserved_and_excluded=True,
        no_AIME_answer_or_accuracy_used=True, no_GPU_dispatch=True, original_seal_unchanged=True,
        reuse_is_origin_aware_not_new_fresh_PASS=True, monotonic=time.monotonic())
    path = ROOT / 'operations/USER_QUALIFICATION_REUSE_APPLICABILITY_AUDIT.json'
    with path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],reused=48,remaining=10,remaining_units=[u['id'] for u in remaining])))


if __name__ == '__main__':
    main()
