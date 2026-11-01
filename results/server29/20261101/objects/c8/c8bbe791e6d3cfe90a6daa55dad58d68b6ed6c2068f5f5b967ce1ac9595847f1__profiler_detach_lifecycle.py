"""Documented CUPTI cleanup after a real, completed exact WScale profile.

The named16-token CPU+CUDA collection and mathematical workload are unchanged.
Only fixed current matched4 units may use cleanup; formal jobs never profile.
No exit or signal suppression. External actual exit0/release remain mandatory.
"""
import ctypes,gc,json,math,os,sys,time
from pathlib import Path
import execution_control as C

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

def detach_completed_profiler(root,torch_module,record):
    import native_profiler_lifecycle as lifecycle
    binding=json.loads((Path(root)/'analysis/CUPTI_LIFECYCLE_BINDING.json').read_text())
    if len(binding['files'])!=1 or binding['files'][0]['path'] not in lifecycle._HELD_LIBRARIES:
        raise RuntimeError('Detach cannot load a new handle; exact pre-driver held handle required')
    loaded=lifecycle.preload_cupti_before_driver(root)
    if not loaded['already_held']:
        raise RuntimeError('Detach must use the same exact CUPTI handle held before Driver init')
    handle=lifecycle._HELD_LIBRARIES[loaded['path']]
    result=_detach_exact_pinned_handle(handle,torch_module,record)
    result.update(path=loaded['path'],sha256=loaded['sha256'],manifest_sha256=loaded['manifest_sha256'])
    return result

PROFILE_IDS=tuple('GPU1_'+m+'_MATCHED4096' for m in ('base','W','R','RW'))
EXECUTION_ROOT='/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'

def _profile_target(root,gpu,pid,unit,formal=False,argv=None):
    if pid!=os.getpid():raise RuntimeError('Lifecycle may operate only on its actual current process')
    root=Path(root).resolve();args=list(sys.argv if argv is None else argv)
    if str(root)!=EXECUTION_ROOT or not args or Path(args[0]).absolute()!=root/'scripts/gpu_unit_worker.py':
        raise RuntimeError('Lifecycle requires the exact new worker entrypoint/root')
    for option in ('--root','--unit'):
        if args.count(option)!=1 or args.index(option)+1>=len(args):raise RuntimeError('Malformed exact lifecycle worker arguments')
    if Path(args[args.index('--root')+1]).resolve()!=root or args[args.index('--unit')+1]!=unit['id'] or args.count('--formal')!=int(bool(formal)):
        raise RuntimeError('Lifecycle unit/root/formal arguments differ')
    if type(gpu)!=int or gpu not in C.AUTHORIZED_GPUS or unit['gpu']!=gpu or unit['gpu_uuid']!=C.UUIDS[gpu] or os.environ.get('CUDA_VISIBLE_DEVICES')!=C.UUIDS[gpu]:
        raise RuntimeError('Lifecycle actual physical UUID binding differs')
    manifest=C.read(root/('FORMAL_JOB_MANIFEST.json' if formal else 'GPU_EXECUTION_PLAN.json'))
    units=manifest['jobs' if formal else 'qualification_units'];matches=[x for x in units if x['id']==unit['id']]
    if len(matches)!=1 or matches[0]!=unit:raise RuntimeError('Lifecycle unit is not exact unique frozen definition')
    if formal:
        if unit['kind']!='formal_free_generation':raise RuntimeError('Unknown formal lifecycle exclusion')
        return None
    if unit['id'] not in PROFILE_IDS or unit['kind']!='matched_performance' or unit['gpu']!=1 or unit['tokens']!=4096:
        raise RuntimeError('Only four exact new matched-profile units may use lifecycle')
    target=root/'execution'/unit['id']
    if not target.is_dir() or target.is_symlink() or not target.resolve().is_relative_to(root):raise RuntimeError('Lifecycle target is not an ordinary owned unit')
    dispatch=C.read(target/'DISPATCH.json');start=C.read(target/'STARTED.json');identity=C.process_identity(pid)
    if dispatch['unit']!=unit['id'] or dispatch['formal'] is not False or dispatch['gpu']!=gpu or dispatch['gpu_uuid']!=C.UUIDS[gpu] or dispatch['permit_sha256']!=C.sha(root/'GPU_PERMIT.json') or dispatch['controller_identity']['pid']!=os.getppid() or not C.same_process(dispatch['controller_identity']) or start['identity']!=identity:
        raise RuntimeError('Lifecycle dispatch/current identity binding differs')
    return target,identity

def preload_qualification_profile(root,gpu,pid,unit,formal=False):
    selected=_profile_target(root,gpu,pid,unit,formal)
    if selected is None:return None
    target,identity=selected
    from native_profiler_lifecycle import preload_cupti_before_driver
    loaded=preload_cupti_before_driver(root)
    if loaded['already_held']:raise RuntimeError('Worker preload admission must be unique before Driver init')
    loaded.update(unit=unit['id'],identity=identity,gpu_uuid=unit['gpu_uuid'],source_freeze_sha256=C.sha(Path(root)/'SOURCE_FREEZE.json'),permit_sha256=C.sha(Path(root)/'GPU_PERMIT.json'))
    C.atomic_new(target/'CUPTI_PRELOAD_ADMISSION.json',loaded)
    return loaded

def detach_qualification_profile(root,gpu,pid,unit,profile,formal=False):
    selected=_profile_target(root,gpu,pid,unit,formal)
    if selected is None:
        if profile is not None:raise RuntimeError('Formal generation must never collect a qualification profile')
        return None
    target,identity=selected;root=Path(root).resolve();preload=C.read(target/'CUPTI_PRELOAD_ADMISSION.json')
    source_sha=C.sha(root/'SOURCE_FREEZE.json');permit_sha=C.sha(root/'GPU_PERMIT.json');manifest_sha=C.sha(root/'analysis/CUPTI_LIFECYCLE_BINDING.json')
    if not(preload['identity']==identity and preload['unit']==unit['id'] and preload['source_freeze_sha256']==source_sha and preload['permit_sha256']==permit_sha and preload['manifest_sha256']==manifest_sha and preload['already_held'] is False):
        raise RuntimeError('Exact preheld CUPTI admission identity differs')
    if profile['status']!='PASS_ACTUAL16TOKEN_CUDA_PROFILE' or profile['profiled_tokens']!=16 or profile['actual_CUDA_kernel_calls']<=0 or profile['quantizer_regions']<=0 or not math.isfinite(profile['quantizer_CUDA_milliseconds']) or profile['quantizer_CUDA_milliseconds']<=0:
        raise RuntimeError('Real complete CUDA profile must precede cleanup')
    summary_sha=C.objsha(profile)
    common=dict(unit=unit['id'],identity=identity,gpu_uuid=unit['gpu_uuid'],source_freeze_sha256=source_sha,permit_sha256=permit_sha,manifest_sha256=manifest_sha,profile_summary_sha256=summary_sha)
    C.atomic_new(target/'PROFILE_COLLECTION_COMPLETED.json',dict(status='PASS_ACTUAL_PROFILE_COLLECTION_COMPLETED_BEFORE_DETACH',profile_summary=profile,**common))
    journal=target/'CUPTI_PROFILE_DETACH_EVENTS.jsonl'
    with journal.open('x') as stream:stream.flush();os.fsync(stream.fileno())
    def record(phase,**fields):
        with journal.open('a') as stream:
            json.dump(dict(phase=phase,monotonic=time.monotonic(),pid=pid,**fields),stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    import torch
    result=detach_completed_profiler(root,torch,record)
    if result['manifest_sha256']!=manifest_sha or result['path']!=preload['path'] or result['sha256']!=preload['sha256'] or C.objsha(profile)!=summary_sha:
        raise RuntimeError('Profile/handle binding changed across documented native cleanup')
    result.update(common)
    C.atomic_new(target/'CUPTI_PROFILE_DETACH.json',result)
    return result
