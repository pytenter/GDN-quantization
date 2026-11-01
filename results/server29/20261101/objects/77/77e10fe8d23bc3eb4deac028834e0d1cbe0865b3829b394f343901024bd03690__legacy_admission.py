"""Admit exactly one complete V2 trajectory; never admit interrupted prefixes.

The original raw RESULT bytes/source/permit and real owner exit event remain
unchanged. A new admission wrapper identifies the old execution explicitly.
No claim is made that the old process ran under the repaired source.
"""
import copy,json,shutil
from pathlib import Path
import execution_common as C

def spec(root):
    return json.loads((Path(root)/'LEGACY_COMPLETE_ADMISSION.json').read_text())
def verify_original(root):
    s=spec(root);old=Path(s['old_root']);p=old/'formal'/s['unit']/'RESULT.json'
    C.source(old,s['old_source_sha256']);assert C.sha(old/'GPU_PERMIT.json')==s['old_permit_sha256']
    assert C.sha(p)==s['original_RESULT_sha256']
    raw=json.loads(p.read_text());expected=raw.pop('payload_sha256');assert expected==s['original_saved_payload_sha256']
    assert raw['source_manifest_sha256']==s['old_source_sha256'] and raw['permit_sha256']==s['old_permit_sha256']
    assert raw['unit']==s['unit'] and raw['job']['config_id']=='C0' and raw['job']['seed']==1
    restored=copy.deepcopy(raw);calls=restored['result']['quantizer_audit']['helper_calls']
    restored['result']['quantizer_audit']['helper_calls']={int(k):v for k,v in calls.items()}
    assert C.objsha(restored)==expected,'Only the documented metadata key restoration may reproduce the old digest'
    assert C.objsha(raw)!=expected
    r=raw['result'];assert r['status']=='COMPLETE' and r['config_id']=='C0' and r['seed']==1 and not r['nonfinite'] and r['runtime_error'] is None
    assert r['finish_reason']=='eos' and r['generated_token_count']==11728 and r['generated_token_count']==len(r['generated_token_ids'])
    assert r['actual_decode_forwards']==len(r['generated_token_ids'])-1
    assert r['input_ids_sha256']==s['input_ids_sha256'] and C.objsha(r['generated_token_ids'])==s['generated_ids_sha256']
    assert r['initial_generator_sha256']==s['initial_generator_sha256']
    assert C.sha(old/'runtime/CONTROLLER_EVENTS.jsonl')==s['old_controller_events_sha256']
    events=[json.loads(x) for x in (old/'runtime/CONTROLLER_EVENTS.jsonl').read_text().splitlines()]
    exits=[x for x in events if x['phase']=='OWNED_STOP_WAIT' and x['unit']==s['unit']]
    releases=[x for x in events if x['phase']=='OWNED_STOP_RELEASED' and x['unit']==s['unit']]
    assert len(exits)==len(releases)==1 and exits[0]['true_exit']==0
    d=json.loads((p.parent/'DISPATCH.json').read_text());assert C.sha(p.parent/'DISPATCH.json')==s['original_DISPATCH_sha256']
    try:current=C.identity(d['child']['pid'])
    except FileNotFoundError:current=None
    assert current is None or current['starttick']!=d['child']['starttick']
    assert raw['job']==next(j for j in json.loads((Path(root)/'FORMAL_JOBS.json').read_text())['jobs'] if j['id']==s['unit'])
    return s,raw,exits[0],releases[0],d

def install(root,new_source,new_permit):
    root=Path(root);s,raw,exit_event,release_event,d=verify_original(root)
    target=root/'formal'/s['unit'];target.mkdir()
    old=Path(s['old_root'])/'formal'/s['unit']
    for name in ['RESULT.json','DISPATCH.json','WORKER_IDENTITY.json','LOADED_REGULAR_MAPPED_SHA.json','POSTFLIGHT_REGULAR_MAPPED_SHA.json']:
        dest=target/('ORIGINAL_'+name)
        shutil.copy2(old/name,dest)
        assert dest.read_bytes()==(old/name).read_bytes()
    value=dict(raw,execution_origin='V2_COMPLETE_VALIDATED_PRESERVED',admitted_into_source_sha256=new_source,admitted_into_permit_sha256=new_permit,
        original_RESULT_sha256=s['original_RESULT_sha256'],original_saved_payload_sha256=s['original_saved_payload_sha256'],legacy_admission_sha256=C.sha(root/'LEGACY_COMPLETE_ADMISSION.json'))
    value['payload_sha256']=C.objsha(value);C.atomic(target/'RESULT.json',value)
    C.atomic(target/'OWNING_CLOSED.json',dict(status='PASS_INDEPENDENT_ADMISSION_OF_REAL_OLD_EXIT',scope='Original owner Popen wait0 from immutable V2 journal, not a new worker/exit',
        owning_Popen_exit=exit_event['true_exit'],child=d['child'],gpu_uuid=d['gpu_uuid'],release_observed=True,source_sha256=s['old_source_sha256'],permit_sha256=s['old_permit_sha256'],
        admitted_into_source_sha256=new_source,admitted_into_permit_sha256=new_permit,RESULT_sha256=C.sha(target/'RESULT.json'),original_RESULT_sha256=s['original_RESULT_sha256'],
        original_exit_event=exit_event,original_release_event=release_event,original_controller_events_sha256=s['old_controller_events_sha256']))
    C.atomic(target/'LEGACY_RESULT_ADMISSION.json',dict(status='PASS_EXACT_COMPLETE_ONLY_NO_PREFIX_REUSE',old_source_sha256=s['old_source_sha256'],old_permit_sha256=s['old_permit_sha256'],new_source_sha256=new_source,new_permit_sha256=new_permit,
        original_RESULT_sha256=s['original_RESULT_sha256'],new_admission_wrapper_sha256=C.sha(target/'RESULT.json'),old_payload_metadata_key_cause_exact=True,old_token_payload_unchanged=True,no_old_State_RNG_cache_reuse=True))
    return s['unit'],target

def validate_installed(root,unit,new_source,new_permit):
    root=Path(root);s=spec(root);assert unit==s['unit'];target=root/'formal'/unit
    rawpath=target/'ORIGINAL_RESULT.json';assert C.sha(rawpath)==s['original_RESULT_sha256']
    original=json.loads(rawpath.read_text());orig_hash=original.pop('payload_sha256')
    restored=copy.deepcopy(original);restored['result']['quantizer_audit']['helper_calls']={int(k):v for k,v in restored['result']['quantizer_audit']['helper_calls'].items()}
    assert C.objsha(restored)==orig_hash==s['original_saved_payload_sha256']
    x=json.loads((target/'RESULT.json').read_text());digest=x.pop('payload_sha256');assert C.objsha(x)==digest
    assert x['execution_origin']=='V2_COMPLETE_VALIDATED_PRESERVED' and x['result']==original['result'] and x['job']==original['job']
    assert x['source_manifest_sha256']==s['old_source_sha256'] and x['permit_sha256']==s['old_permit_sha256']
    assert x['admitted_into_source_sha256']==new_source and x['admitted_into_permit_sha256']==new_permit
    close=json.loads((target/'OWNING_CLOSED.json').read_text());assert close['owning_Popen_exit']==0 and close['release_observed'] and close['RESULT_sha256']==C.sha(target/'RESULT.json')
    assert close['source_sha256']==s['old_source_sha256'] and close['admitted_into_source_sha256']==new_source
    return x
