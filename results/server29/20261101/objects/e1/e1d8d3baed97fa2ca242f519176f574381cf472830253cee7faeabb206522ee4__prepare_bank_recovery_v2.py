"""Append-only metadata schema recovery before any selected-document forward."""
from common import *
import ast,re,traceback

def selected_metadata_ids(path):
    x=load(path)
    if path.name=='fresh_b_selection_manifest.json':
        assert x['gate']=='PASS' and len(x['streams'])==16
        # independent_order is a candidate listing, not selected/consumed data.
        # The original selection rule expressly prohibits treating it as used.
        value={'streams':x['streams'],'exclusion_sets':x['exclusion_sets']}
    else:value=x
    return set(re.findall(r'pg19-test-\d+',json.dumps(value)))

def main():
    original=D/'scripts/prepare_bank.py'
    failure=D/'run/POST_PARSER_RECOVERY_STOP.json'
    x=load(failure)
    assert x['error']=='AssertionError()' and 'line 33, in documents' in x['traceback']
    assert not any(p.is_file() for folder in ['inputs','rotations','raw'] for p in (D/folder).rglob('*'))
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    protected=[row(original),row(D/'scripts/prepare_bank_recovery_v1.py'),row(D/'run/STOP.json'),row(failure),row(D/'provenance/engineering_parser_recovery_v1.json'),row('/tmp/ling_surrogate_v1_prepare_bank_recovery_v1.log'),row(D/'analysis/transition_parity_smoke.json'),row(D/'provenance/historical_metric_definition_audit.json'),row(D/'scripts/transition.py')]
    source=original.read_text()
    adapters=[("for doc in dd:reasons.setdefault(doc,[]).append('historical overlap '+category)","for entry in dd:reasons.setdefault(entry['document_id'],[]).append('historical overlap '+category)"),
        ("set(re.findall(r'pg19-test-\\d+',q.read_text()))","selected_metadata_ids(q)")]
    adapted=source
    for a,b in adapters:assert adapted.count(a)==1;adapted=adapted.replace(a,b)
    a=ast.parse(source);b=ast.parse(adapted)
    for fa,fb in zip(a.body,b.body):
        if isinstance(fa,ast.FunctionDef) and fa.name=='documents':continue
        assert ast.dump(fa,include_attributes=False)==ast.dump(fb,include_attributes=False)
    fields=[]
    for name in ['LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V1','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC']:
        f=P/'experiments'/name/'protocol/fresh_b_selection_manifest.json'
        fields.append(dict(source=row(f),excluded_ids=sorted(selected_metadata_ids(f)),ignored_field='independent_order',ignored_reason='candidate order is not selected/consumed data'))
    emit('provenance/engineering_parser_recovery_v2.json',dict(gate='PASS',recovery='ENGINEERING_ONLY_RECOVERY',authorization='user section37 parser/metadata recovery before first scientific measurement',adapters=adapters,field_audit=fields,protected_before=protected,SCIENTIFIC_OBJECTS_CHANGED='NO',DISCOVERY_MEASUREMENTS=0,CONFIRMATION_MEASUREMENTS=0,selection_rule_unchanged='first8 in same independent hash order after actual historical selected/excluded IDs',original_attempts_preserved=True))
    scope={'__name__':'bank_metadata_recovery_v2','__file__':str(original),'selected_metadata_ids':selected_metadata_ids}
    exec(compile(adapted,str(original)+'::metadata_adapter_v2','exec'),scope)
    scope['documents']();scope['rotations']()
    assert [row(f['path']) for f in protected]==protected
    emit('provenance/engineering_parser_recovery_v2_completion.json',dict(gate='PASS',receipt=row(D/'provenance/engineering_parser_recovery_v2.json'),protected_byte_identity='PASS',scientific_objects_changed=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        emit('run/POST_METADATA_RECOVERY_V2_STOP.json',dict(gate='FAIL',stage='FRESH_DOC_AND_BANK_PREPARATION_AFTER_METADATA_ADAPTER_V2',error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True,NEW_TRAINING='NO',NEW_AIME_GENERATION=0))
        raise
