"""Frozen engineering observations of profiler teardown, never qualification."""
import argparse
import atexit
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttick=int(fields[19]),
                cmdline=(p / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))


def save(p, value):
    p = Path(p)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    args = ap.parse_args()
    plan = read(ROOT / 'ENGINEERING_PROFILER_DETACH_PLAN.json')
    permit = read(ROOT / 'ENGINEERING_PROFILER_DETACH_PERMIT.json')
    freeze = read(ROOT / 'ENGINEERING_PROFILER_DETACH_FREEZE.json')
    assert permit['source_freeze_sha256'] == sha(ROOT / 'ENGINEERING_PROFILER_DETACH_FREEZE.json')
    assert permit['human_authorization_sha256'] == sha(ROOT / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
    assert permit['plan_sha256'] == sha(ROOT / 'ENGINEERING_PROFILER_DETACH_PLAN.json')
    assert not any((ROOT / n).exists() for n in ['SOURCE_FREEZE.json', 'GPU_PERMIT.json', 'CONTROLLER_STARTED.json'])
    for x in freeze['files']:
        p = ROOT / x['path']
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == x['bytes'] and sha(p) == x['sha256'], x['path']
    case = next(x for x in plan['cases'] if x['id'] == args.case)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == case['gpu_uuid'] == permit['gpu_uuid']
    target = ROOT / 'operations/profiler_exit_detach_diagnostics' / args.case
    assert target.is_dir() and not (target / 'STARTED.json').exists()
    save(target / 'STARTED.json', dict(identity=identity(os.getpid()), monotonic=time.monotonic(),
         case=case, permit_sha256=sha(ROOT / 'ENGINEERING_PROFILER_DETACH_PERMIT.json'),
         counts_as_qualification=False, AIME_generations=0))

    def stage(name, **fields):
        save(target / (name + '.json'), dict(phase=name, monotonic=time.monotonic(),
             identity=identity(os.getpid()), **fields))

    atexit.register(lambda: stage('PYTHON_ATEXIT_ENTERED'))
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor')]
    import qualification_guards as G
    stage('VERIFY_EXTERNAL_BEGIN')
    rows = G.verify_model_and_external(ROOT, True)
    G.p1_queue_released(ROOT, 1)
    G._own_gpu(ROOT, 1)
    if case['load_order'] == 'torch_first':
        import torch
        assert not torch.cuda.is_initialized()
    driver = G.driver_identity(ROOT, 1, rows)
    stage('DRIVER_ENUMERATED', driver=driver)
    import torch
    stage('TORCH_IMPORTED', torch=torch.__version__, CUDA=torch.version.cuda)
    assert torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1'
    runtime = None
    if case['kind'] == 'exact_H_profile_136':
        import resident_rotation as resident
        spec = importlib.util.spec_from_file_location('original_exact_profiler_path', ROOT / 'scripts/qualify_gpu.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.torch, module.resident = torch, resident
        unit = next(x for x in read(ROOT / 'QUALIFICATION_JOB_MANIFEST.json')['units'] if x['id'] == 'GPU1_H_PROFILE_SHORT')
        runtime = module.Runtime(ROOT, target, 1, 'H')
        G.check_after_model(ROOT, 1, os.getpid())
        stage('MODEL_LOADED')
        result = module.performance(runtime, unit)
        G.check_after_model(ROOT, 1, os.getpid())
        save(target / 'PROFILE_RESULT.json', result)
        stage('EXACT_PERFORMANCE_FUNCTION_RETURNED', actual_nonAIME_model_forwards=136,
              measured_profile_tokens=8, model_path_and_math_unmodified=True)
    else:
        value = torch.ones((128, 128), device='cuda:0')
        torch.cuda.synchronize()
        G.check_after_model(ROOT, 1, os.getpid())
        stage('CUDA_MICRO_ALLOCATED')
        if case['kind'] != 'no_profiler':
            activities = [torch.profiler.ProfilerActivity.CPU]
            if case['kind'] != 'cpu_profile':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True,
                                       profile_memory=False, with_stack=False) as profile:
                for _ in range(8):
                    value = value @ value.T * 0.001
                    torch.cuda.synchronize()
                    profile.step()
            profile.export_chrome_trace(str(target / 'CUDA_PROFILE.json'))
            events = [dict(key=x.key, count=x.count) for x in profile.key_averages(group_by_input_shape=True)]
            save(target / 'PROFILE_RESULT.json', dict(events=events, model_forwards=0, QDQ_calls=0))
            stage('MICRO_PROFILE_FUNCTION_RETURNED')
            if case.get('explicit_cleanup'):
                del profile, events
                gc.collect()
                torch.cuda.synchronize()
                stage('EXPLICIT_PROFILER_GC_COMPLETED')
        del value
        torch.cuda.synchronize()
    if case.get('explicit_detach'):
        if 'profile' in locals():
            del profile, events
        sys.path.insert(0, str(ROOT / 'operations'))
        from profiler_detach_candidate import detach_completed_profiler
        detach = detach_completed_profiler(ROOT, torch, stage)
        stage('CUPTI_EXPLICIT_DETACH_COMPLETED', receipt=detach)
    stage('NORMAL_MAIN_RETURN', AIME_generations=0, counts_as_qualification=False)
    # No os._exit, signal handler, disabled teardown, or forced successful exit.
    # The supervisor must observe the actual interpreter return code and release.


if __name__ == '__main__':
    main()
