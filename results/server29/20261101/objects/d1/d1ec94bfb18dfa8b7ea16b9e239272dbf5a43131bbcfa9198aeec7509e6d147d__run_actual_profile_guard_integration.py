"""Bounded three-process CLI integration of repaired guard; never AIME/58 credit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'operations')]
import qualification_guards as G
from profiler_exit_probe import identity, save, sha, read


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any((ROOT / n).exists() for n in ['SOURCE_FREEZE.json', 'GPU_PERMIT.json', 'CONTROLLER_STARTED.json'])
    authorization = read(ROOT / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
    assert authorization['human_request'] == '检查为什么异常，修复后，继续完成测试'
    suite = read(ROOT / 'analysis/PROFILER_REPAIR_CPU_V2_TWO_FRESH_SUITE.json')
    assert len(suite['runs']) == 12 and all(x['actual_wait_returncode'] == 0 for x in suite['runs'])
    for row in suite['runs']:
        for p, digest in row['result'].get('source_pins', {}).items():
            assert sha(ROOT / p) == digest
    G._own_gpu(ROOT, 1)
    directory = ROOT / 'operations/actual_profile_guard_integration'
    directory.mkdir(exist_ok=False)
    mini = directory / 'engineering_only_root'
    mini.mkdir()
    prior = read(ROOT / 'PROFILER_REPAIR_SOURCE_SCOPE.json')
    for x in prior['prior_sources']:
        p = ROOT / x['path']
        q = mini / x['path']
        assert p.is_file() and not p.is_symlink()
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
    for name in ['src/profiler_detach_lifecycle.py', 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json']:
        q = mini / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, q)
    original_units = read(ROOT / 'QUALIFICATION_JOB_MANIFEST.json')['units']
    units = []
    for condition in ['H', 'C1', 'S3']:
        unit = dict(next(x for x in original_units if x['id'] == 'GPU1_' + condition + '_PROFILE_SHORT'))
        unit['id'] = 'ENGINEERING_' + condition + '_PROFILE_SHORT'
        units.append(unit)
    (mini / 'QUALIFICATION_JOB_MANIFEST.json').write_text(json.dumps(dict(units=units,
        engineering_only=True, scientific_qualification_credit=0, AIME_generations=0), sort_keys=True, indent=2) + '\n')
    config = read(mini / 'ROOT.json')
    config.update(status='ENGINEERING_ONLY_ACTUAL_QUALIFIER_CLI_INTEGRATION', remote_root=str(mini),
                  scientific_qualification_credit=0, formal_generations_allowed=0)
    (mini / 'ROOT.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
    plan = dict(status='ONLY_THREE_NEW_NONAIME_ACTUAL_CLI_PROFILER_INTEGRATIONS', units=units,
        maximum_GPU_dispatches=3, GPU0_authorized=False, gpu_uuid=G.UUIDS[1],
        human_authorization_sha256=sha(ROOT / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json'),
        actual_supervisor_sha256=sha(__file__), actual_guard_sha256=sha(ROOT / 'src/qualification_guards.py'),
        actual_qualifier_sha256=sha(ROOT / 'scripts/qualify_gpu.py'),
        scientific_qualification_credit=0, AIME_generations=0, no_automatic_retry=True)
    save(directory / 'PLAN.json', plan)
    files = []
    for p in sorted(mini.rglob('*')):
        assert not p.is_symlink()
        if p.is_file():
            files.append(dict(path=str(p.relative_to(mini)), bytes=p.stat().st_size, sha256=sha(p)))
    save(mini / 'SOURCE_FREEZE.json', dict(status='ENGINEERING_ONLY_FROZEN_ACTUAL_CLI_INPUTS', files=files,
         count=len(files), scientific_qualification_credit=0, AIME_generations=0,
         engineering_plan_sha256=sha(directory / 'PLAN.json')))
    save(mini / 'GPU_PERMIT.json', dict(task_id=G.TASK, ALLOW_GPU='QUALIFICATION_ONLY', engineering_only=True,
         scientific_qualification_credit=0, formal_permission=False, devices=[G.UUIDS[1]],
         source_freeze_sha256=sha(mini / 'SOURCE_FREEZE.json'),
         qualification_plan_sha256=sha(mini / 'QUALIFICATION_JOB_MANIFEST.json'),
         human_authorization_sha256=sha(mini / 'HUMAN_CONTINUATION_AUTHORIZATION.json'),
         prior_p1_stop_admission_sha256=sha(mini / 'PRIOR_P1_USER_STOP_ADMISSION.json'),
         latest_profiler_repair_authorization_sha256=plan['human_authorization_sha256'],
         engineering_plan_sha256=sha(directory / 'PLAN.json')))
    for name in ['qualification', 'dispatch/qualification', 'operations']:
        (mini / name).mkdir(parents=True, exist_ok=True)
    supervisor = identity(os.getpid())
    save(directory / 'STARTED.json', dict(identity=supervisor, plan_sha256=sha(directory / 'PLAN.json'), monotonic=time.monotonic()))
    results = []
    for unit in units:
        G._own_gpu(ROOT, 1)
        target = mini / 'qualification' / unit['id']
        assert not os.path.lexists(target)
        lower = time.monotonic()
        save(mini / 'dispatch/qualification' / (unit['id'] + '.json'), dict(unit=unit['id'], unit_record=unit,
             controller_identity=supervisor, source_freeze_sha256=sha(mini / 'SOURCE_FREEZE.json'),
             permit_sha256=sha(mini / 'GPU_PERMIT.json'), engineering_only=True, dispatch_monotonic_lower=lower))
        argv = [sys.executable, '-u', '-B', str(mini / 'scripts/qualify_gpu.py'), '--root', str(mini),
                '--unit', unit['id'], '--gpu', '1']
        with (directory / (unit['id'] + '.log')).open('xb') as log:
            process = subprocess.Popen(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=G.UUIDS[1],
                PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), stdout=log, stderr=log)
            expected = identity(process.pid)
            save(directory / (unit['id'] + '_DISPATCH.json'), dict(identity=expected, unit=unit,
                 dispatch_monotonic_lower=lower, scientific_qualification_credit=0))
            code = process.wait()
        observed = time.monotonic()
        deadline = observed + 120
        while any(x['uuid'] == G.UUIDS[1] for x in G.vendor(ROOT).query_gpu()['apps']):
            assert time.monotonic() < deadline, 'Integration own GPU release unresolved'
            time.sleep(1)
        row = dict(unit=unit['id'], identity=expected, actual_wait_returncode=code,
                   dispatch_monotonic_lower=lower, exit_first_observed_monotonic=observed,
                   release_first_observed_monotonic=time.monotonic(), release='PASS',
                   scientific_qualification_credit=0, AIME_generations=0, EXACT_physical_GPUh='UNRESOLVED')
        if (target / 'RESULT.json').exists():
            result = read(target / 'RESULT.json')
            row['result_sha256'] = sha(target / 'RESULT.json')
            row['actual_model_forwards'] = result['result']['continuous_QDQ_input_forwards']
        if (target / 'CUPTI_PROFILE_DETACH.json').exists():
            detach = read(target / 'CUPTI_PROFILE_DETACH.json')
            row['detach_receipt_sha256'] = sha(target / 'CUPTI_PROFILE_DETACH.json')
            row['detach_receipt'] = detach
        save(directory / (unit['id'] + '_COMPLETION.json'), row)
        results.append(row)
        assert code == 0 and not G.vendor(ROOT).same_process(expected), row
        assert row['actual_model_forwards'] == 136 and result['status'] == 'PASS'
        assert detach['flush_returncode'] == detach['finalize_returncode'] == 0
        assert detach['unit'] == unit['id'] and detach['pid'] == process.pid
    save(directory / 'COMPLETE.json', dict(status='PASS_THREE_ACTUAL_UNCHANGED_QUALIFIER_CLI_PROFILE_GUARD_EXITS',
         results=results, all_actual_exit0_and_own_release=True, new_engineering_GPU_dispatches=3,
         actual_nonAIME_model_forwards=408, scientific_qualification_credit=0, AIME_generations=0,
         GPU0_P1_untouched=True, original_qualifier_sha256=sha(ROOT / 'scripts/qualify_gpu.py')))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        p = ROOT / 'operations/ACTUAL_PROFILE_GUARD_INTEGRATION_FAILURE.json'
        if not p.exists():
            save(p, dict(error=repr(error), traceback=traceback.format_exc(), pid=os.getpid(),
                 no_automatic_retry=True, scientific_qualification_credit=0))
        raise
