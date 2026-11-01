"""Append-only premeasurement parser recovery, not a scientific amendment."""
from common import *
import ast, traceback

def main():
    original=D/'scripts/prepare_bank.py'
    prior=D/'run/STOP.json'
    log=Path('/tmp/ling_surrogate_v1_prepare_bank.log')
    x=load(prior)
    assert x['stage']=='FRESH_DOC_AND_BANK_PREPARATION'
    assert x['error']=='''TypeError("unhashable type: 'dict'")'''
    assert 'line 30, in documents' in x['traceback']
    assert not any(p.is_file() for folder in ['inputs','rotations','raw'] for p in (D/folder).rglob('*'))
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    assert load(D/'analysis/transition_parity_smoke.json')['gate']=='PASS'
    assert load(D/'provenance/historical_metric_definition_audit.json')['gate']=='PASS'
    audit=load(PARENTS[2]/'provenance/phase3_input_overlap_audit.json')
    schemas=[]
    for category,entries in audit['excluded_documents'].items():
        for entry in entries:
            assert isinstance(entry,dict) and isinstance(entry['document_id'],str)
        schemas.append(dict(category=category,count=len(entries),schema='object.document_id'))
    before=[row(original),row(prior),row(log),row(D/'scripts/common.py'),row(D/'scripts/transition.py'),row(D/'scripts/phase_a.py'),row(D/'analysis/transition_parity_smoke.json'),row(D/'provenance/historical_metric_definition_audit.json')]
    old="for doc in dd:reasons.setdefault(doc,[]).append('historical overlap '+category)"
    new="for entry in dd:reasons.setdefault(entry['document_id'],[]).append('historical overlap '+category)"
    source=original.read_text();assert source.count(old)==1
    adapted=source.replace(old,new)
    # Only the schema adapter is changed. Other functions and statement trees
    # remain identical; main's old STOP guard is not invoked by this receipt.
    a=ast.parse(source);b=ast.parse(adapted)
    for fa,fb in zip(a.body,b.body):
        if isinstance(fa,ast.FunctionDef) and fa.name=='documents':continue
        assert ast.dump(fa,include_attributes=False)==ast.dump(fb,include_attributes=False)
    receipt=dict(gate='PASS',recovery='ENGINEERING_ONLY_RECOVERY',authorization='User task section 37: parser recovery before first scientific measurement',change=dict(original=old,replacement=new),schemas=schemas,protected_before=before,DISCOVERY_MEASUREMENTS=0,CONFIRMATION_MEASUREMENTS=0,SCIENTIFIC_OBJECTS_CHANGED='NO',original_stop_retained=True,failed_attempt_log=row(log),source_sha256_after_in_memory_adapter=hashlib.sha256(adapted.encode()).hexdigest())
    emit('provenance/engineering_parser_recovery_v1.json',receipt)
    # A distinct execution module leaves the original script/STOP untouched.
    scope={'__name__':'bank_parser_engineering_recovery','__file__':str(original)}
    exec(compile(adapted,str(original)+'::metadata_adapter_v1','exec'),scope)
    scope['documents']();scope['rotations']()
    assert [row(f['path']) for f in before]==before
    emit('provenance/engineering_parser_recovery_v1_completion.json',dict(gate='PASS',receipt=row(D/'provenance/engineering_parser_recovery_v1.json'),protected_byte_identity='PASS',scientific_objects_changed=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        emit('run/POST_PARSER_RECOVERY_STOP.json',dict(gate='FAIL',stage='FRESH_DOC_AND_BANK_PREPARATION_AFTER_PARSER_ADAPTER',error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True,NEW_TRAINING='NO',NEW_AIME_GENERATION=0))
        raise
