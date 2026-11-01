"""One bounded single-GPU Phase0 dispatch; no retries and no foreign-job signals."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
HARD_SECONDS = 7200.
PROTECT_SECONDS = 6900.
HELPER = Path('/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1/repair_20261017_v1/source/task_coordinator.py')
HELPER_SHA = 'e7edb91fee5b3f32f2f8721ce1546e17be820a7976676e4f1a02045c8ae1a21f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('profile', 'baseline'))
    parser.add_argument('--authorization', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    auth = read(args.authorization)
    assert auth['status'] == 'PASS' and auth['task_id'] == TASK and args.mode in auth['allowed_modes']
    assert auth['gpu'] == 0 and auth['gpu_uuid'] == UUID
    assert sha(root / 'PLAN.json') == auth['plan_sha256']
    assert sha(root / 'REFERENCE_BINDING.json') == auth['reference_binding_sha256']
    assert sha(root / 'SOURCE_MANIFEST.json') == auth['source_manifest_sha256']
    for row in read(root / 'SOURCE_MANIFEST.json')['execution_sources']:
        path = root / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    assert sha(HELPER) == HELPER_SHA
    spec = importlib.util.spec_from_file_location('_read_only_qualified_process_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, helper.signal_stop)
    started_file = root / 'execution' / args.mode / 'CONTROLLER_STARTED.json'
    assert not started_file.exists() and not (root / 'FIRST_FAILURE.json').exists(), 'No rerun/retry'
    ledger_path = root / 'GPU_RESOURCE_LEDGER.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(task_id=TASK, dispatches=[], MAX_NEW_GPU_HOURS=2.0,
                hard_limit_seconds=HARD_SECONDS, protective_stop_seconds=PROTECT_SECONDS, cumulative_GPU_seconds=0.,
                accounting='Single GPU dispatch through real process exit and NVML release. All loads, profiling and tests charged; never reset.')
    assert all(j['process_exited'] and j['NVML_released'] for j in ledger['dispatches'])
    prior = ledger['cumulative_GPU_seconds']
    assert prior < PROTECT_SECONDS - 60, 'Insufficient budget reserve'
    apps = helper.nvml()
    if any(a['UUID'] == UUID for a in apps):
        helper.save(root / 'GPU_UNAVAILABLE_AND_STOP.json', dict(status='CPU_READY_GPU_UNAVAILABLE_AND_STOP',
                    mode=args.mode, reason='Authorized GPU0 not idle; no preemption, waiting or retry', prior_GPU_seconds=prior), new=True)
        return
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free', '--format=csv,noheader,nounits'], text=True)
    selected = [line.split(', ') for line in hardware.strip().splitlines() if line.startswith('0,')]
    assert len(selected) == 1 and selected[0][1] == UUID and '3090' in selected[0][2] and float(selected[0][3]) >= 22000
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 8 * 1024**3, 'Insufficient profile storage'
    if args.mode == 'baseline':
        assert read(root / 'execution/profile/CONTROLLER_RESULT.json')['status'] == 'PASS'
        assert auth.get('baseline_resource_admission') == 'PASS' and auth['projected_GPU_seconds'] + prior < PROTECT_SECONDS
    helper.save(started_file, dict(task_id=TASK, mode=args.mode, identity=helper.identity(os.getpid()),
                 pid=os.getpid(), authorization_sha256=sha(args.authorization), server_unix=time.time()), new=True)
    logpath = root / 'execution' / args.mode / 'worker.log'
    handle = logpath.open('xb')
    worker = root / 'source/baseline_worker.py'
    assert helper.STOP_SIGNAL is None, 'Operator stopped before dispatch'
    launch = time.monotonic()
    proc = subprocess.Popen([PYTHON, str(worker), '--root', str(root), '--authorization', str(args.authorization), '--mode', args.mode],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    job = dict(mode=args.mode, gpu=0, gpu_uuid=UUID, pid=proc.pid, identity=None, dispatch_monotonic=launch,
               dispatch_unix=time.time(), occupied_seconds=0., returncode=None, process_exited=False, NVML_released=False)
    ledger['dispatches'].append(job)
    failure = None
    def own_running():
        current = helper.identity(proc.pid)
        return proc.poll() is None and current and job['identity'] and current['start_tick'] == job['identity']['start_tick'] and str(worker) in current['cmdline']
    try:
        deadline = time.monotonic() + 2
        while job['identity'] is None and proc.poll() is None and time.monotonic() < deadline:
            job['identity'] = helper.identity(proc.pid)
            if job['identity'] is None:
                time.sleep(.01)
        assert job['identity'] and str(worker) in job['identity']['cmdline'], 'Cannot bind worker process'
        last_size = 0
        last_progress = launch
        while True:
            now = time.monotonic()
            apps = helper.nvml()
            job.update(occupied_seconds=now - launch, returncode=proc.poll(), process_exited=proc.poll() is not None,
                       NVML_released=not any(a['pid'] == proc.pid for a in apps))
            ledger.update(status='RUNNING', cumulative_GPU_seconds=prior + job['occupied_seconds'],
                          cumulative_GPU_hours=(prior + job['occupied_seconds']) / 3600, all_own_GPUs_released=False)
            helper.save(ledger_path, ledger)
            assert helper.STOP_SIGNAL is None, 'Operator stop signal'
            assert not any(a['UUID'] == UUID and a['pid'] != proc.pid for a in apps), 'Foreign GPU job appeared; stop own only'
            assert prior + now - launch < PROTECT_SECONDS, 'Cumulative protective GPU budget stop'
            size = logpath.stat().st_size
            if size != last_size:
                last_size = size
                last_progress = now
            assert now - last_progress < 1800, '30 minute no-progress watchdog'
            if proc.poll() is not None:
                assert proc.returncode == 0 and not (root / 'FIRST_FAILURE.json').exists(), 'Worker failed; no retry'
                assert read(root / 'execution' / args.mode / 'RESULT.json')['status'] == 'PASS'
                if job['NVML_released']:
                    break
            time.sleep(2)
    except BaseException:
        failure = dict(status='FAIL_CLOSED_STOP', task_id=TASK, mode=args.mode, traceback=traceback.format_exc(),
                       server_unix=time.time(), GPU_retry=False)
        if own_running():
            os.killpg(proc.pid, signal.SIGTERM)
        limit = time.monotonic() + 15
        while proc.poll() is None and time.monotonic() < limit:
            time.sleep(.25)
        if own_running():
            os.killpg(proc.pid, signal.SIGKILL)
        helper.save(root / ('CONTROLLER_FAILURE.json' if (root / 'FIRST_FAILURE.json').exists() else 'FIRST_FAILURE.json'), failure, new=True)
    finally:
        limit = time.monotonic() + 45
        while True:
            apps, error = helper.nvml_safe()
            released = proc.poll() is not None and not error and not any(a['pid'] == proc.pid for a in apps)
            if released or time.monotonic() >= limit:
                break
            time.sleep(.25)
        job.update(occupied_seconds=time.monotonic() - launch, returncode=proc.poll(), process_exited=proc.poll() is not None,
                   NVML_released=not error and not any(a['pid'] == proc.pid for a in apps), release_unix=time.time())
        total = prior + job['occupied_seconds']
        released = job['process_exited'] and job['NVML_released']
        ledger.update(status='PASS_PHASE_COMPLETE' if not failure and released else 'FAIL_CLOSED_STOP',
                      cumulative_GPU_seconds=total, cumulative_GPU_hours=total / 3600,
                      hard_limit_not_exceeded=total <= HARD_SECONDS, all_own_GPUs_released=released,
                      final_NVML_error=error)
        helper.save(ledger_path, ledger)
        helper.save(root / 'execution' / args.mode / 'CONTROLLER_RESULT.json', dict(
            status='PASS' if not failure and released and total <= HARD_SECONDS else 'FAIL_CLOSED_STOP',
            ledger_sha256=sha(ledger_path), all_own_GPUs_released=released, cumulative_GPU_seconds=total), new=True)
        handle.close()
    if failure or not released or total > HARD_SECONDS:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
