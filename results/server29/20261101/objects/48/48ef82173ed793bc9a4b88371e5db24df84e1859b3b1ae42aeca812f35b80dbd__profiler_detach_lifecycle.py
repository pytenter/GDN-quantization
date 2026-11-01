"""Exact profiler lifecycle: documented CUPTI detach after completed trace collection.

No library replacement or exit/signal suppression. The CUDA process must still
exit normally and the external supervisor must observe its actual return code.
The installed12.1 cupti_activity.h requires CUDA synchronization and activity
buffer flush before cuptiFinalize when called outside a CUDA API-exit callback.
"""
import ctypes
import hashlib
import json
import os
import time
import gc
import sys
from pathlib import Path


def _detach_exact_pinned_handle(handle, torch_module, record):
    if torch_module.autograd._profiler_enabled():
        raise RuntimeError('Cannot detach CUPTI while a profiler scope is active')
    gc.collect()
    record('CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE')
    torch_module.cuda.synchronize()
    record('CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH')
    flush = handle.cuptiActivityFlushAll
    flush.argtypes = [ctypes.c_uint32]
    flush.restype = ctypes.c_int
    record('CUPTI_ACTIVITY_FLUSH_BEGIN', flags=0)
    status = int(flush(0))
    record('CUPTI_ACTIVITY_FLUSH_RETURN', returncode=status)
    if status != 0:
        raise RuntimeError('Exact CUPTI activity flush failed: ' + str(status))
    finalize = handle.cuptiFinalize
    finalize.argtypes = []
    finalize.restype = ctypes.c_int
    record('CUPTI_FINALIZE_BEGIN')
    status = int(finalize())
    record('CUPTI_FINALIZE_RETURN', returncode=status)
    if status != 0:
        raise RuntimeError('Exact CUPTI detach failed: ' + str(status))
    return dict(status='DOCUMENTED_CUPTI_FLUSH_AND_DETACH_RETURNED_SUCCESS',
                synchronization_before_flush=True, flush_returncode=0,
                finalize_returncode=0, CUPTI_API_calls=2,
                profiler_collection_already_completed=True,
                CUDA_or_CUPTI_APIs_not_disabled=True,
                signal_or_exit_suppression=False,
                actual_process_exit_still_required=True)


def detach_completed_profiler(root, torch_module, record):
    root = Path(root)
    sys.path.insert(0, str(root / 'src'))
    import native_profiler_lifecycle as lifecycle
    loaded = lifecycle.preload_cupti_before_driver(root)
    if not loaded['already_held']:
        raise RuntimeError('Detach must use the same exact CUPTI handle held before Driver init')
    handle = lifecycle._HELD_LIBRARIES[loaded['path']]
    result = _detach_exact_pinned_handle(handle, torch_module, record)
    result.update(path=loaded['path'], sha256=loaded['sha256'],
                  manifest_sha256=loaded['manifest_sha256'])
    return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def _profile_target(root, gpu, argv=None):
    """Identify only a completed trace of this manifest-bound qualification unit."""
    args = list(sys.argv if argv is None else argv)
    if '--unit' not in args:
        return None
    if args.count('--unit') != 1 or args.index('--unit') + 1 >= len(args):
        raise RuntimeError('Malformed exact qualification unit arguments')
    unit_id = args[args.index('--unit') + 1]
    root = Path(root).resolve()
    if args and Path(args[0]).absolute() == root / 'scripts/resident_formal_worker.py':
        for option in ('--root', '--gpu'):
            if args.count(option) != 1 or args.index(option) + 1 >= len(args):
                raise RuntimeError('Malformed exact formal profiler exclusion arguments')
        if (Path(args[args.index('--root') + 1]).resolve() != root
                or args[args.index('--gpu') + 1] != str(gpu)):
            raise RuntimeError('Formal profiler exclusion root/GPU arguments differ')
        formal = json.loads((root / 'FORMAL_JOB_MANIFEST.json').read_text())
        jobs = [x for x in formal['jobs'] if x['id'] == unit_id]
        if len(jobs) != 1:
            raise RuntimeError('Formal profiler exclusion lacks one frozen formal job')
        job = jobs[0]
        if job['gpu'] != gpu or job['gpu_uuid'] != os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('Formal profiler exclusion physical UUID binding differs')
        return None  # Formal generation never collects a qualification profile.
    manifest = json.loads((root / 'QUALIFICATION_JOB_MANIFEST.json').read_text())
    matches = [x for x in manifest['units'] if x['id'] == unit_id]
    if len(matches) != 1:
        raise RuntimeError('Profiler detach unit is not unique in the frozen manifest')
    unit = matches[0]
    if unit['kind'] not in ('profile_short', 'profile_late'):
        return None
    if unit['gpu'] != gpu or unit['gpu_uuid'] != os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Profiler detach physical UUID binding differs')
    target = root / 'qualification' / unit_id
    if not target.resolve().is_relative_to(root) or target.is_symlink() or not target.is_dir():
        raise RuntimeError('Profiler detach target is not an ordinary owned unit directory')
    trace = target / 'CUDA_PROFILE.json'
    if not os.path.lexists(trace):
        return None  # First check_after_model, before profiling has occurred.
    if trace.is_symlink() or not trace.is_file() or trace.stat().st_size <= 0:
        raise RuntimeError('Profiler detach requires the real exported ordinary trace')
    return target, trace, unit


def detach_qualification_profile_if_completed(root, gpu, pid):
    if pid != os.getpid():
        raise RuntimeError('Profiler detach can operate only the actual current process')
    selected = _profile_target(root, gpu)
    if selected is None:
        return None
    target, trace, unit = selected
    root = Path(root).resolve()
    result_path = target / 'CUPTI_PROFILE_DETACH.json'
    source_sha = _sha(root / 'SOURCE_FREEZE.json')
    permit_sha = _sha(root / 'GPU_PERMIT.json')
    if os.path.lexists(result_path):
        if result_path.is_symlink() or not result_path.is_file():
            raise RuntimeError('Nonordinary profiler detach receipt')
        result = json.loads(result_path.read_text())
        if not (result['pid'] == pid and result['source_freeze_sha256'] == source_sha
                and result['permit_sha256'] == permit_sha and result['trace_sha256'] == _sha(trace)
                and result['flush_returncode'] == result['finalize_returncode'] == 0):
            raise RuntimeError('Prior profiler detach receipt differs; no repeated detach')
        return result
    before = _sha(trace)
    journal = target / 'CUPTI_PROFILE_DETACH_EVENTS.jsonl'

    def record(phase, **fields):
        with journal.open('a') as stream:
            json.dump(dict(phase=phase, monotonic=time.monotonic(), pid=pid, **fields), stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

    # The unchanged qualifier calls its second check_after_model only after
    # performance() has completed trace export and coverage validation.
    import torch
    result = detach_completed_profiler(root, torch, record)
    if _sha(trace) != before:
        raise RuntimeError('Collected CUDA trace changed across explicit native cleanup')
    result.update(unit=unit['id'], pid=pid, gpu_uuid=unit['gpu_uuid'],
                  source_freeze_sha256=source_sha, permit_sha256=permit_sha,
                  trace_sha256=before, trace_bytes=trace.stat().st_size,
                  original_qualifier_and_model_math_unchanged=True,
                  cleanup_after_completed_collection=True, monotonic=time.monotonic())
    with result_path.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result
