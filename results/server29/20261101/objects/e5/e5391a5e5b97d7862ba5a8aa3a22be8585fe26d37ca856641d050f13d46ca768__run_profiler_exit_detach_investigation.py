"""Once-only bounded engineering supervisor; no qualification or formal dispatch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'operations'))
from profiler_exit_detach_probe import identity, read, save, sha


def main():
    out = ROOT / 'operations/profiler_exit_detach_diagnostics'
    prior = read(ROOT / 'operations/profiler_exit_diagnostics/INVESTIGATION_OBSERVATIONS_COMPLETE.json')
    assert prior['actual_dispatches'] == 6 and prior['all_new_probe_GPU_processes_released']
    out.mkdir(exist_ok=False)
    permit = read(ROOT / 'ENGINEERING_PROFILER_DETACH_PERMIT.json')
    plan = read(ROOT / 'ENGINEERING_PROFILER_DETACH_PLAN.json')
    assert permit['plan_sha256'] == sha(ROOT / 'ENGINEERING_PROFILER_DETACH_PLAN.json')
    assert permit['source_freeze_sha256'] == sha(ROOT / 'ENGINEERING_PROFILER_DETACH_FREEZE.json')
    assert permit['human_authorization_sha256'] == sha(ROOT / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
    assert permit['counts_as_scientific_qualification'] is False
    for row in read(ROOT / 'ENGINEERING_PROFILER_DETACH_FREEZE.json')['files']:
        p = ROOT / row['path']
        assert not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    save(out / 'SUPERVISOR_STARTED.json', dict(identity=identity(os.getpid()),
         permit_sha256=sha(ROOT / 'ENGINEERING_PROFILER_DETACH_PERMIT.json'), monotonic=time.monotonic()))
    lock = (ROOT / 'operations/PROFILER_DIAGNOSTIC_GPU1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor')]
    import qualification_guards as G
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    observations = []
    for case in plan['cases']:
        assert case['gpu_uuid'] == permit['gpu_uuid'] == G.UUIDS[1]
        G._own_gpu(ROOT, 1)
        target = out / case['id']
        target.mkdir(exist_ok=False)
        argv = [sys.executable, '-u', '-B', str(ROOT / 'operations/profiler_exit_detach_probe.py'), '--case', case['id']]
        if case['gdb']:
            argv = ['/usr/bin/gdb', '--batch', '--return-child-result', '-q',
                    '-ex', 'set pagination off', '-ex', 'set print thread-events off',
                    '-ex', 'set disable-randomization off',
                    '-ex', 'run', '-ex', 'info program', '-ex', 'thread apply all bt',
                    '-ex', 'info sharedlibrary', '--args'] + argv
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=case['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='24')
        lower = time.monotonic()
        save(target / 'DISPATCH.json', dict(case=case, argv=argv, supervisor_identity=identity(os.getpid()),
             dispatch_monotonic_lower=lower, actual_original_P1_GPU0_untouched=True))
        with (target / 'process.log').open('xb') as log:
            process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
            expected = identity(process.pid)
            save(target / 'SUBPROCESS_IDENTITY.json', expected)
            newest = lower
            while process.poll() is None:
                for p in target.glob('*.json'):
                    try:
                        value = read(p)
                        if isinstance(value.get('monotonic'), (int, float)):
                            newest = max(newest, value['monotonic'])
                    except (ValueError, AttributeError):
                        pass
                if time.monotonic() - newest > 1800:
                    save(target / 'NO_PHASE_PROGRESS_WATCHDOG.json', dict(monotonic=time.monotonic(),
                         last_progress=newest, exact_subprocess=expected, all_evidence_retained=True))
                    owned = [expected]
                    if (target / 'STARTED.json').exists():
                        owned.insert(0, read(target / 'STARTED.json')['identity'])
                    for expected_child in owned:
                        for sig in (signal.SIGTERM, signal.SIGKILL):
                            if not G.vendor(ROOT).same_process(expected_child):
                                break
                            assert str(ROOT / 'operations/profiler_exit_detach_probe.py') in expected_child['cmdline']
                            assert G.vendor(ROOT).same_process(expected_child)
                            os.kill(expected_child['pid'], sig)
                            deadline = time.monotonic() + 15
                            while G.vendor(ROOT).same_process(expected_child) and time.monotonic() < deadline:
                                time.sleep(.2)
                    raise RuntimeError('Engineering probe has no real phase progress for1800s; no automatic retry')
                time.sleep(1)
            code = process.wait()
        observed = time.monotonic()
        # A GDB-owned inferior's signal is preserved in the real debugger log;
        # it is never inferred to be exit0 from the presence of a result file.
        text = (target / 'process.log').read_text(errors='replace')
        sigsegv = 'Program received signal SIGSEGV' in text
        inferior = read(target / 'STARTED.json')['identity'] if (target / 'STARTED.json').exists() else None
        deadline = time.monotonic() + 120
        while any(x['uuid'] == case['gpu_uuid'] for x in G.vendor(ROOT).query_gpu()['apps']):
            assert time.monotonic() < deadline, 'Own120s GPU release remains unresolved; no next diagnostic dispatch'
            time.sleep(1)
        if inferior is not None:
            assert not G.vendor(ROOT).same_process(inferior)
        row = dict(case=case, supervisor_identity=identity(os.getpid()), subprocess_identity=expected,
                   inferior_identity=inferior, actual_Popen_wait_returncode=code,
                   GDB_reported_SIGSEGV=sigsegv, process_log_sha256=sha(target / 'process.log'),
                   dispatch_monotonic_lower=lower, exit_first_observed_monotonic=observed,
                   release_first_observed_monotonic=time.monotonic(), release='PASS',
                   model_forwards=136 if case['kind'] == 'exact_H_profile_136' else 0,
                   AIME_generations=0, counts_as_scientific_qualification=False,
                   EXACT_physical_GPUh='UNRESOLVED')
        save(target / 'OBSERVATION.json', row)
        observations.append(row)
    save(out / 'INVESTIGATION_OBSERVATIONS_COMPLETE.json', dict(status='BOUNDED_ENGINEERING_OBSERVATIONS_COMPLETE_NOT_SCIENTIFIC_PASS',
         cases=observations, actual_dispatches=len(observations), GPU0_dispatches=0, AIME_generations=0,
         mathematical_sources_changed=False, all_new_probe_GPU_processes_released=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        p = ROOT / 'operations/PROFILER_DETACH_SUPERVISOR_FAILURE.json'
        if not p.exists():
            save(p, dict(error=repr(error), traceback=traceback.format_exc(),
                 identity=identity(os.getpid()), monotonic=time.monotonic(), no_automatic_retry=True))
        raise
