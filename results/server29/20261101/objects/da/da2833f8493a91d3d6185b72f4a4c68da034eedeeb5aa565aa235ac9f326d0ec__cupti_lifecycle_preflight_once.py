"""Exactly two fresh non-AIME CUDA lifecycle smoke checks; no qualification credit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REMOTE = '/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'
UUID = 'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3'
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
FREEZE = 'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttick=int(fields[19]), cmdline=[x.decode() for x in
                (p / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])


def apps():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    require(p.returncode == 0, 'Read-only NVML process inventory unavailable')
    rows = []
    for line in p.stdout.strip().splitlines():
        fields = [x.strip() for x in line.split(',')]
        require(len(fields) == 3, 'Malformed NVML app row')
        if fields[1] == UUID:
            rows.append(dict(pid=int(fields[0]), uuid=fields[1], used_memory_MiB=fields[2]))
    return rows


def owned_app_remains(rows, observed):
    if observed is None:
        return bool(rows)
    if not any(x['pid'] == observed['pid'] for x in rows):
        return False
    try:
        current = identity(observed['pid'])
    except FileNotFoundError:
        return True  # NVML still lists the old PID; release is not confirmed.
    return current == observed  # A replacement PID is foreign and never signalled.


def validate(root, expected):
    require(str(root) == REMOTE and not root.is_symlink(), 'Exact new RAM root required')
    require(sha(root / FREEZE) == expected, 'Engineering source freeze differs')
    manifest = read(root / FREEZE)
    require(manifest['replicates'] == ['R0', 'R1'] and manifest['UUID'] == UUID,
            'Finite two-replicate engineering scope differs')
    for row in manifest['files']:
        p = root / row['path']
        require(not p.is_symlink() and p.stat().st_size == row['bytes']
                and sha(p) == row['sha256'], 'Engineering source SHA/bytes changed')
    return manifest


def worker(root, replicate, expected):
    validate(root, expected)
    require(replicate in ['R0', 'R1'] and os.environ.get('CUDA_VISIBLE_DEVICES') == UUID,
            'Exact engineering replicate/physical UUID required')
    target = root / 'engineering/cupti_lifecycle_preflight' / replicate
    dispatch = read(target / 'DISPATCH.json')
    require(dispatch['freeze_sha256'] == expected and dispatch['UUID'] == UUID
            and dispatch['replicate'] == replicate and os.getppid() == dispatch['parent']['pid']
            and identity(os.getppid()) == dispatch['parent'], 'Engineering parent/dispatch differs')
    sys.path.insert(0, str(root / 'src'))
    from native_profiler_lifecycle import preload_cupti_before_driver
    from profiler_detach_lifecycle import detach_completed_profiler

    def record(phase, **fields):
        with (target / 'EVENTS.jsonl').open('a') as stream:
            json.dump(dict(phase=phase, monotonic=time.monotonic(), **fields), stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

    new(target / 'STARTED.json', dict(identity=identity(os.getpid()), freeze_sha256=expected))
    preload = preload_cupti_before_driver(root)
    new(target / 'CUPTI_PRELOAD.json', preload)
    record('EXACT_CUPTI_PRELOADED_BEFORE_TORCH_CUDA')
    import execution_control as C
    new(target / 'DRIVER_IDENTITY.json', C.driver_identity(root, 1))
    import torch
    require(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'Runtime differs')
    torch.set_num_threads(24)
    torch.set_num_interop_threads(24)
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 6),
            'Physical GPU visibility/capability differs')
    x = torch.ones((64, 64), device='cuda:0')
    y = torch.ones((64, 64), device='cuda:0')
    record('CUDA_INITIALIZED_NONAIME_SCALAR_FIXTURE')
    with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=True) as profiler:
        for _ in range(16):
            with torch.profiler.record_function('WSCALE_CUPTI_LIFECYCLE_ENGINEERING_ONLY'):
                z = x @ y
        torch.cuda.synchronize()
    events = profiler.events()
    kernels = [e.name for e in events if str(e.device_type).endswith('CUDA')]
    require(bool(kernels) and bool(torch.isfinite(z).all()), 'Actual CUDA profile/finite fixture missing')
    new(target / 'PROFILE_COMPLETED.json', dict(actual_CUDA_kernel_events=len(kernels),
        named_steps=16, profiler_active=torch.autograd._profiler_enabled(), kernel_names=kernels,
        model_forward_calls=0, QDQ_calls=0, AIME_generations=0, qualification_credit=0))
    del events, profiler, x, y, z
    cleanup = detach_completed_profiler(root, torch, record)
    new(target / 'CUPTI_PROFILE_DETACH.json', cleanup)
    native = C.library_check(root, 1)
    require(native['status'] == 'PASS', 'Actual mapped native SHA/inode closure failed')
    new(target / 'POSTFLIGHT_LIBRARY.json', native)
    record('DETACH_COMPLETED_NORMAL_INTERPRETER_EXIT_REQUIRED')
    new(target / 'RESULT.json', dict(status='PASS_NATIVE_LIFECYCLE_SMOKE_PAYLOAD_ONLY',
        replicate=replicate, identity=identity(os.getpid()), freeze_sha256=expected,
        actual_CUDA=True, normal_OS_exit_still_required=True, model_forward_calls=0,
        QDQ_calls=0, AIME_generations=0, qualification_credit=0, cleanup=cleanup))


def parent(root, expected):
    validate(root, expected)
    require(os.environ.get('CUDA_VISIBLE_DEVICES', '') == '', 'Engineering parent must remain CPU-only')
    base = root / 'engineering/cupti_lifecycle_preflight'
    base.mkdir(parents=True, exist_ok=False)
    mine = identity(os.getpid())
    new(base / 'PARENT_STARTED.json', dict(identity=mine, freeze_sha256=expected))
    results = []
    for replicate in ['R0', 'R1']:
        require(not apps(), 'Authorized UUID occupied; no signalling or retry')
        target = base / replicate
        target.mkdir()
        new(target / 'DISPATCH.json', dict(parent=mine, UUID=UUID, replicate=replicate,
                                           freeze_sha256=expected))
        argv = [PYTHON, '-u', '-B', str(root / 'operations/cupti_lifecycle_preflight_once.py'),
                '--root', str(root), '--worker', replicate, '--freeze-sha256', expected]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONFAULTHANDLER='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        start = time.monotonic()
        with (target / 'worker.log').open('x') as log:
            proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            observed = None
            last_progress = time.monotonic()
            event_bytes = 0
            try:
                while proc.poll() is None:
                    if observed is None:
                        try:
                            candidate = identity(proc.pid)
                            if candidate['cmdline'] == argv:
                                observed = candidate
                                new(target / 'ACTUAL_CHILD_IDENTITY.json', observed)
                        except FileNotFoundError:
                            pass
                    journal = target / 'EVENTS.jsonl'
                    size = journal.stat().st_size if journal.exists() else 0
                    if size > event_bytes:
                        event_bytes = size
                        last_progress = time.monotonic()
                    if time.monotonic() - last_progress > 1800:
                        require(observed is not None and identity(proc.pid) == observed,
                                'Watchdog cannot signal an unresolved process identity')
                        new(target / 'PROGRESS_WATCHDOG.json', dict(
                            status='NO_TRUE_PHASE_PROGRESS_FOR_1800_SECONDS', identity=observed))
                        proc.terminate()
                        break
                    time.sleep(0.1)
            except BaseException as exc:
                new(target / 'PARENT_OBSERVATION_FAILURE.json', dict(error=repr(exc)))
                if proc.poll() is None and observed is not None and identity(proc.pid) == observed:
                    proc.terminate()
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        require(identity(proc.pid) == observed, 'Own identity changed before forced cleanup')
                        proc.kill()
                        proc.wait(timeout=120)
                raise
            returncode = proc.wait()
        end = time.monotonic()
        release_start = time.monotonic()
        remaining = apps()
        while owned_app_remains(remaining, observed) and time.monotonic() - release_start < 120:
            time.sleep(0.2)
            remaining = apps()
        released = not owned_app_remains(remaining, observed)
        interval = dict(replicate=replicate, identity=observed, actual_returncode=returncode,
                        released=released, UUID=UUID, start_monotonic=start, end_monotonic=end,
                        reservation_seconds=end-start, physical_EXACT_GPUh='UNRESOLVED',
                        model_forward_calls=0, QDQ_calls=0, AIME_generations=0, qualification_credit=0)
        new(target / 'ACTUAL_EXIT_RELEASE.json', interval)
        results.append(interval)
        if returncode != 0 or not released or observed is None:
            new(base / 'FIRST_FAILURE.json', dict(status='TRUE_ENGINEERING_FAILURE_NO_RETRY', results=results))
            raise RuntimeError('Finite native lifecycle preflight failed; no retry')
        result = read(target / 'RESULT.json')
        require(result['freeze_sha256'] == expected and result['identity'] == observed,
                'Engineering result identity differs')
        require(result['cleanup']['flush_returncode'] == result['cleanup']['finalize_returncode'] == 0,
                'Explicit CUPTI cleanup did not succeed')
    out = dict(status='PASS_TWO_FRESH_ACTUAL_CUDA_PROFILE_DETACH_OS_EXIT_RELEASE', results=results,
               freeze_sha256=expected, qualification_credit=0, AIME_generations=0,
               model_forward_calls=0, QDQ_calls=0, no_exit_signal_suppression=True)
    new(base / 'RESULT.json', out)
    print(json.dumps(out, sort_keys=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--freeze-sha256', required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--parent', action='store_true')
    group.add_argument('--worker', choices=['R0', 'R1'])
    args = ap.parse_args()
    if args.parent:
        parent(args.root.resolve(), args.freeze_sha256)
    else:
        worker(args.root.resolve(), args.worker, args.freeze_sha256)
