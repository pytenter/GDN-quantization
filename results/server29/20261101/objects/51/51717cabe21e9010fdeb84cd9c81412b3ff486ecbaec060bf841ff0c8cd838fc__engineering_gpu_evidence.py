"""Strict stdlib reader of the repaired loader evidence and immutable failures."""
import hashlib
import json
from pathlib import Path

BINDING='operations/ENGINEERING_GPU_EVIDENCE_BINDING.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def bound(root,row):
    root=Path(root).resolve();rel=Path(row['path']);p=root/rel
    assert not rel.is_absolute()and '..'not in rel.parts and p.resolve().is_relative_to(root)and p.is_file()and not p.is_symlink()
    assert p.stat().st_size==row['bytes']and sha(p)==row['sha256'],row['path']
    return p

def validate(root):
    root=Path(root).resolve();binding=read(root/BINDING)
    assert binding['status']=='PASS_NATIVE_LOAD_ORDER_REPAIR_TWO_FRESH_EXITS_HISTORY_PRESERVED'
    assert sha(root/'src/resident_rotation.py')=='4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95'
    assert binding['native_exit_repair_authorization_sha256']==sha(root/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json')
    original=read(bound(root,binding['original_failed_completion']))
    assert original['status']=='FAIL_ENGINEERING_LOAD_ONLY_PROBE'and original['actual_wait_returncode']==-11
    assert original['all_exact_own_processes_exited']and original['own_GPU_UUID_released']
    old_freeze=read(root/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json')
    copy_map=read(root/binding['original_failed_probe_preserved_copy_map'])
    assert copy_map['probe_source_freeze_sha256']==sha(root/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json')
    by={x['path']:x for x in copy_map['files']};assert len(by)==len(old_freeze['files'])
    for row in old_freeze['files']:
        saved=by[row['path']];assert saved['sha256']==row['sha256']and saved['bytes']==row['bytes']
        p=root/saved['preserved_path'];assert p.resolve().is_relative_to(root)and p.is_file()and not p.is_symlink()
        assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
        if row['path']!='src/qualification_guards.py':
            assert sha(root/row['path'])==row['sha256'],'Unexpected non-lifecycle scientific input change'
    success=[]
    for receipt in binding['successful_loader_completions']:
        completion=read(bound(root,receipt))
        assert completion['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and completion['actual_wait_returncode']==0
        assert completion['all_exact_own_processes_exited']and completion['own_GPU_UUID_released']
        assert completion['model_forward_calls']==completion['QDQ_calls']==completion['AIME_generations']==0
        assert completion['counts_as_scientific_qualification']is False
        interval=completion['resource_interval'];assert interval['monotonic_same_host']is True
        assert interval['dispatch_monotonic_lower']<=interval['identity_monotonic_upper']<=interval['exit_first_observed_monotonic']<=interval['release_first_observed_monotonic']
        assert completion['probe_source_freeze_sha256']==sha(root/'ENGINEERING_CUPTI_ORDER_REPAIR_PROBE_SOURCE_FREEZE.json')
        assert completion['probe_permit_sha256']==sha(root/'ENGINEERING_CUPTI_ORDER_REPAIR_PROBE_PERMIT.json')
        result=read(bound(root,completion['result']))
        assert result['model_forward_calls']==result['QDQ_calls']==result['AIME_generations']==0
        assert result['counts_as_scientific_qualification']is False
        initialization=result['driver_identity']['native_profiler_initialization']
        assert initialization['status']=='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL'
        assert initialization['CUDA_Driver_API_calls']==initialization['CUPTI_API_calls']==0
        for observation in result['observations']:bound(root,observation)
        success.append(completion)
    assert len(success)==2 and len({(x['worker_identity']['pid'],x['worker_identity']['starttick'])for x in success})==2
    active=bound(root,binding['active_completion']);assert active==bound(root,binding['successful_loader_completions'][-1])
    review=read(bound(root,binding['active_native_review']))
    assert review['status']=='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED'
    assert review['completion_sha256']==sha(active)and review['native_manifest_sha256']==sha(root/'analysis/RUNTIME_NATIVE_CLOSURE.json')
    extra=read(bound(root,binding['all_repaired_native_reviews']))
    assert extra['status']=='PASS_BOTH_REPAIRED_LOADER_MAPS_ALL_STAGES_IDENTITY_ONLY'
    assert len(extra['loaders'])==2 and {x['completion_sha256']for x in extra['loaders']}=={x['sha256']for x in binding['successful_loader_completions']}
    for row in binding['engineering_resource_records']:bound(root,row['receipt'])
    counts=binding['dispatch_counts'];assert counts['scientific_qualification_GPU_dispatches']==0
    assert counts['total_GPU_work_dispatches']==len(binding['engineering_resource_records'])
    assert counts['engineering_native_loader_GPU_probes']==4 and counts['engineering_diagnostic_GPU_dispatches']==8
    assert counts['total_GPU_work_dispatches']==12
    return dict(status=binding['status'],binding=binding,active_completion=active,active_review=root/binding['active_native_review']['path'],
        completion=success[-1],successful_completions=success,**counts)
