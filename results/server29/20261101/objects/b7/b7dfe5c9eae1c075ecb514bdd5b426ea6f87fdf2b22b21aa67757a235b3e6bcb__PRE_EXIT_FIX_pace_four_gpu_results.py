"""Fixed closed-origin validators; original records are never relabelled fresh."""
import importlib.util,sys
from pathlib import Path
import pace_control as C
OLD='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025'
OLD_SOURCE='8c4bfa5076906bd6edbc355e4ce3373cb9e0550eb8ce285dee6f07afbd13cb9d'

def old_validators():
    root=Path(OLD)
    C.require(C.sha(root/'source_manifest.json')==OLD_SOURCE,'Fixed predecessor SOURCE changed')
    expected={x['path']:x for x in C.read(root/'source_manifest.json')['files']}
    for name in ('src/pace_control.py','src/pace_continuation.py'):
        C.require(C.sha(root/name)==expected[name]['sha256'],'Original CPU validator source changed')
    old=sys.modules.get('pace_control')
    try:
        s=importlib.util.spec_from_file_location('original_pace_control_cpu',root/'src/pace_control.py')
        ctrl=importlib.util.module_from_spec(s);sys.modules[s.name]=ctrl;s.loader.exec_module(ctrl)
        sys.modules['pace_control']=ctrl
        s=importlib.util.spec_from_file_location('original_pace_continuation_cpu',root/'src/pace_continuation.py')
        cont=importlib.util.module_from_spec(s);s.loader.exec_module(cont)
    finally:
        if old is None:sys.modules.pop('pace_control',None)
        else:sys.modules['pace_control']=old
    return ctrl,cont

def verify_reuse(root,full_sources=False):
    root=Path(root);admission=C.read(root/'REUSE_ADMISSION.json');oldc,oldk=old_validators();origin=Path(OLD)
    C.require(admission['status']=='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE' and admission['origin_root']==OLD and admission['origin_source_sha256']==OLD_SOURCE,'Exact reuse admission missing')
    C.require(admission['old_parent_controller_workers_finisher_gone'] is True,'Predecessor owning chain not closed')
    for name,row in admission['origin_terminal_controls'].items():
        C.require(C.sha(origin/name)==row['sha256'] and (origin/name).stat().st_size==row['bytes'],'Closed predecessor control bytes changed')
    if full_sources:oldc.verify_sources(origin)
    jobs={x['id']:x for x in C.validate_jobs(root)};verified=[]
    for item in admission['units']:
        unit=item['unit'];C.require(unit in jobs,'Unplanned reused unit')
        if unit in oldk.REUSED_IDS:wrapper,interval,target=oldk.verify_old_unit(origin,oldk.origin_root(origin),'formal',unit)
        else:wrapper,interval,target=oldk.verify_current_unit(origin,'formal',unit)
        C.require(wrapper['payload_sha256']==item['payload_sha256'] and C.sha(target/'RESULT.json')==item['result_sha256'] and C.sha(target/'RESOURCE_INTERVAL.json')==item['interval_sha256'],'Reused payload/actual interval binding differs')
        C.require(interval['returncode']==0 and interval['released'] is True,'Nonzero/unreleased prior sample cannot be reused')
        copied=root/'provenance/REUSED_FORMAL'/unit
        C.require(C.sha(copied/'RESULT.json')==item['result_sha256'] and C.sha(copied/'RESOURCE_INTERVAL.json')==item['interval_sha256'],'Copied provenance bytes differ')
        verified.append(unit)
    C.require(verified==admission['reused_ids'] and len(verified)==len(set(verified)),'Exact unique reuse roster differs')
    return dict(status='PASS_FRESH_ORIGINAL_VALIDATORS_EXACT_EXIT0_RELEASE_REUSE',count=len(verified),units=verified,source_sha256=OLD_SOURCE,formal_new_credit=0,model_loads=0,GPU_calls=0)

def completed_rows(root,invalid=None):
    root=Path(root);verify_reuse(root);admission=C.read(root/'REUSE_ADMISSION.json');jobs=C.validate_jobs(root);by={x['id']:x for x in jobs};reuse=set(admission['reused_ids']);rows=[]
    for job in jobs:
        unit=job['id']
        if unit in reuse:
            target=root/'provenance/REUSED_FORMAL'/unit;rows.append((job,C.read(target/'RESULT.json'),target));continue
        target=C.unit_target(root,'formal',unit)
        if not (target/'RESULT.json').exists() and not (target/'RESOURCE_INTERVAL.json').exists():continue
        try:
            interval=C.read(target/'RESOURCE_INTERVAL.json')
            C.require(interval['returncode']==0 and interval['released'] is True and interval['gpu']==job['gpu'] and interval['gpu_uuid']==job['gpu_uuid'],'Current actual exit/release/device binding differs')
            wrapper=C.verify_result(root,'formal',unit,interval['identity']);rows.append((job,wrapper,target))
        except BaseException as e:
            if invalid is None:raise
            invalid.append(dict(unit=unit,error=repr(e),completion_credit=0))
    C.require(len(rows)==len({j['id'] for j,_w,_t in rows}),'Duplicate union completion')
    return rows
