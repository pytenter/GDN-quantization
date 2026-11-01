"""Explicit multi-artifact V1 recovery schema; no guessed aliases/default PASS."""
from common import *

REPORT_REQUIRED = ['RECOVERY_HISTORICAL_AUDIT_RESULT','SCIENTIFIC_FIELDS_EVALUATED','audit',
    'canonical_output','canonical_output_sha256','audit_source_sha256','error',
    'formal_training_started','formal_updates','fresh_b_forwards','new_AIME_generation']
VERDICT_REQUIRED = ['HISTORICAL_L7_RECIPE_IDENTITY_GATE','RECOVERY_HISTORICAL_AUDIT_RESULT','SCIENTIFIC_FIELDS_EVALUATED']

def validate(report,verdict,canonical):
    missing=['recovery_report.'+k for k in REPORT_REQUIRED if k not in report]
    missing+=['formal_verdict.'+k for k in VERDICT_REQUIRED if k not in verdict]
    if missing:raise ValueError(('V1_RECIPE_SCHEMA_GATE_FAIL',missing))
    assert isinstance(report['audit'],dict)
    for k in ['gate','fields','checked_field_count','observed_data_draw_order','historical_source_sha256']:
        if k not in report['audit']:raise ValueError(('V1_RECIPE_SCHEMA_GATE_FAIL','audit.'+k))
    assert report['audit']==canonical,'CANONICAL_RECIPE_PAYLOAD_AMBIGUITY'
    assert report['RECOVERY_HISTORICAL_AUDIT_RESULT']==verdict['RECOVERY_HISTORICAL_AUDIT_RESULT']=='PASS'
    assert report['SCIENTIFIC_FIELDS_EVALUATED']==verdict['SCIENTIFIC_FIELDS_EVALUATED']=='YES'
    assert verdict['HISTORICAL_L7_RECIPE_IDENTITY_GATE']==report['audit']['gate']=='PASS'
    assert report['error'] is None and report['formal_training_started'] is False
    assert all(report[k]==0 for k in ['formal_updates','fresh_b_forwards','new_AIME_generation'])
    fields=report['audit']['fields']
    assert len(fields)==report['audit']['checked_field_count'] and all(r['gate']=='PASS' for r in fields)
    assert report['audit']['historical_source_sha256']==SOURCE_SHA
    return dict(SCHEMA_VALIDATION='PASS',V1_RECIPE_SCHEMA_GATE='PASS',
        EXPECTED_KEYS={'recovery_report':REPORT_REQUIRED,'formal_verdict':VERDICT_REQUIRED},
        OBSERVED_KEYS={'recovery_report':sorted(report),'formal_verdict':sorted(verdict)},MISSING_KEYS=[],
        UNEXPECTED_KEYS={'recovery_report':sorted(set(report)-set(REPORT_REQUIRED)),
                         'formal_verdict':sorted(set(verdict)-set(VERDICT_REQUIRED))},
        schema='immutable V1 recovery envelope + named formal STOP verdict + exact referenced canonical audit payload',
        named_identity_gate_locator='reports/PRETRAINING_RUNTIME_STOP_VERDICT_V1.json:HISTORICAL_L7_RECIPE_IDENTITY_GATE',
        top_level_named_gate_absent_in_recovery_envelope=True,not_invented_or_defaulted=True,
        checked_scientific_fields=len(fields),HISTORICAL_L7_RECIPE_IDENTITY_GATE='PASS',
        RECOVERY_HISTORICAL_AUDIT_RESULT='PASS',SCIENTIFIC_FIELDS_EVALUATED='YES')

def actual():
    p=V1/'provenance/historical_l7_recipe_audit_recovery_v1.json';report=load(p)
    canonical_path=Path(report['canonical_output'])
    assert canonical_path==V1/'provenance/historical_l7_recipe_audit.json'
    assert sha(canonical_path)==report['canonical_output_sha256']
    verdict_path=V1/'reports/PRETRAINING_RUNTIME_STOP_VERDICT_V1.json'
    result=validate(report,load(verdict_path),load(canonical_path))
    result['evidence']=[dict(path=str(q),sha256=sha(q)) for q in [p,verdict_path,canonical_path]]
    return result

def tests():
    import copy
    report=load(V1/'provenance/historical_l7_recipe_audit_recovery_v1.json')
    verdict=load(V1/'reports/PRETRAINING_RUNTIME_STOP_VERDICT_V1.json');canonical=load(Path(report['canonical_output']))
    validate(report,verdict,canonical)
    cases=0
    for keys,source in [(REPORT_REQUIRED,'report'),(VERDICT_REQUIRED,'verdict')]:
        for key in keys:
            a,b=copy.deepcopy(report),copy.deepcopy(verdict)
            del (a if source=='report' else b)[key]
            try:validate(a,b,canonical)
            except (AssertionError,ValueError):cases+=1
            else:raise AssertionError(('MISSING_REQUIRED_FIELD_ACCEPTED',key))
    a=copy.deepcopy(report);a['synthetic_unexpected_metadata']=True
    assert validate(a,verdict,canonical)['SCHEMA_VALIDATION']=='PASS'
    b=copy.deepcopy(verdict);b['HISTORICAL_L7_RECIPE_IDENTITY_GATE']='FAIL'
    try:validate(report,b,canonical)
    except AssertionError:cases+=1
    else:raise AssertionError('AMBIGUOUS_GATE_ACCEPTED')
    return dict(gate='PASS',synthetic_schema_mutations_only=True,missing_and_ambiguity_cases=cases,
        unexpected_nonambiguous_keys_allowed=True,model_loads=0,GPU_forwards=0)
