"""Engineering candidate: documented CUPTI detach after completed trace collection.

No library replacement or exit/signal suppression. The CUDA process must still
exit normally and the external supervisor must observe its actual return code.
The installed12.1 cupti_activity.h requires CUDA synchronization and activity
buffer flush before cuptiFinalize when called outside a CUDA API-exit callback.
"""
import ctypes
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
