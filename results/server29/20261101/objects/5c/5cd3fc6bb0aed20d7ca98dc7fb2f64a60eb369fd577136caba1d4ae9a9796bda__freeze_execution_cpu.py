"""Bind actual clean sources and unchanged registered science before GPU use."""
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import time

D=pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_01')
WT=pathlib.Path('/data/zypan/worktrees/ling-kda-p1-l7-prospective-extension-4doc-v1')
OLD=pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
BASE=pathlib.Path('/data/zypan/worktrees/ling-kda-p1-l7-mixed-outcome-resolution-v1')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def publish(p,j):
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('x') as f:json.dump(j,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()

assert read(D/'PREREGISTRATION_RECONCILIATION.json')['status']=='PASS'
assert read(D/'INPUT_ELIGIBILITY_REPRODUCTION.json')['counts']=={'HISTORICAL_MEASUREMENT_ID':20,'INSUFFICIENT_TOKENS':22,'ELIGIBLE':58}
assert not (D/'STOP.json').exists()
assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
head=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()
audit=[]
for name,functions in [('functional_worker.py',['transfer','object_fingerprint','audit_state','restore_audit','HiddenObserver','checkpoint']),('formal_worker.py',['original_state_diagnostic','observe_qdq','collect_metrics'])]:
    ta,tb=ast.parse((BASE/name).read_text()),ast.parse((WT/name).read_text())
    for function in functions:
        a=next(n for n in ast.walk(ta) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==function)
        b=next(n for n in ast.walk(tb) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==function)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),function
        audit.append(name+':'+function)
for n in ['objective.py','run_matched_l7.py','persistent_engine_v1.py','v2_runtime.py','v3_guard.py']:
    assert sha(WT/n)==sha(BASE/n),n
assert sha(WT/'next_prospective_pair_rule.py')==sha(OLD/'next_prospective_pair_rule.py')
rules=read(OLD/'DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json');cal=rules['calibration']
cfg=read(pathlib.Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/PHASE_B_CALIBRATION_CONFIG.json'))
assert cfg['conditions']==cal['rotation_order']
assert cfg['windows']==cal['windows'] and cfg['layers']==cal['layers'] and cfg['horizons']==[128] and cfg['seed']==0
cfg['documents']=read(D/'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order']
cfg['document_gpu_map']={document:6 for document in cfg['documents']}
assert cfg['gpu_uuids']['6']=='GPU-de53f243-b271-521c-d875-d96611ed8574'
publish(D/'CALIBRATION_CONFIG.json',cfg)
names=['short_worker.py','prospective_common.py','prospective_pipeline.py','functional_worker.py','formal_worker.py',
    'next_prospective_pair_rule.py','objective.py','run_matched_l7.py','persistent_engine_v1.py','v2_runtime.py','v3_guard.py']
for n in names:
    compile((WT/n).read_text(),n,'exec')
    if not (D/n).exists():(D/n).symlink_to(WT/n)
files={n:sha(WT/n) for n in names}
publish(D/'SOURCE_CODE_FREEZE.json',dict(status='FROZEN_BEFORE_FIRST_NEW_DOCUMENT_GPU_SCREEN',source_commit=head,
    source_worktree=str(WT),base_commit='80f23cd580df3bb00b9256ab8b5b3d5d221ffdad',files=files,
    config_sha256=sha(D/'CALIBRATION_CONFIG.json'),frozen_epoch=time.time()))
publish(D/'ADAPTER_DIFF_AND_AST_AUDIT.json',dict(status='PASS_CPU_SOURCE_RECONCILIATION_NOT_NEW_GPU_QUALIFICATION',
    source_commit=head,exact_critical_helper_AST_checks=audit,unchanged_full_source_modules=['objective.py','run_matched_l7.py','persistent_engine_v1.py','v2_runtime.py','v3_guard.py'],
    change_scope=['registered input/rotation routing','output routing','registered scientific-key whitelist','matching new-document FP manifest route','serial screening and stage controller'],
    historical_calibration_budget_gates_not_carried='Original2/16GPUh adapter caps omitted as explicitly required; original proxy has no deadline by default. Actual900s no-progress and OOM/storage gates retained.',
    original_P1_and_L7_score_helpers='UNMODIFIED_SHA_IDENTICAL',original_reducer_checkpoint_restore_quantizer='UNMODIFIED_HELPER_AST_OR_FULL_SOURCE_SHA',
    registered_newdoc_order_and_pair_rule='EXACT_FROZEN',GPU_numerical_contract='SAME_QUALIFIED_UUID_RUNTIME_REUSE; each own full history/fresh64 parity before formal',
    CPU_auditor_failure_provenance=dict(path=str(D.parent/'STOP.json'),sha256=sha(D.parent/'STOP.json'),correction_record_sha256=sha(D.parent/'CPU_AUDIT_CORRECTION.json'))))
publish(D/'CODE_FREEZE.json',{'functional_worker.py':files['functional_worker.py']})
deps=read(OLD/'PROSPECTIVE_FROZEN_SCORE_SOURCE_IDENTITIES.json')
publish(D/'FORMAL_CODE_FREEZE.json',dict(files={n:files[n] for n in names},dependencies=deps,source_commit=head))
with (D/'SECONDARY_SPEC.json').open('xb') as f:f.write((OLD/'SECONDARY_SPEC.json').read_bytes())
publish(D/'ANALYSIS_PLAN.json',dict(status='FROZEN_BEFORE_OUTCOMES',independent_unit='document',planned_documents=4,
    primary='original FP32 full-V KL(FP||student), sum384/384',empirical_resolution=3.3300064494019496e-6,
    sign_labels='P1 if delta<-resolution, L7 if delta>resolution, else UNRESOLVED',p_values='NO_CONFIRMATORY_P_VALUE_NO_TOKEN_P_VALUE',
    relative_KL_difference='descriptive (K_P1-K_L7)/K_L7; denominator shown, undefined if zero; not a new Primary or Secondary',
    historical_two_docs_and_posthoc_panel='SEPARATE_PRESERVE_MIXED_NOT_COMBINED',missing='no imputation, no replacement; planned denominator4 and valid count separate',
    training_authorization='NOT_AUTHORIZED',four_of_four_P1_label='DIRECTIONALLY_SUPPORTIVE_SMALL_PROSPECTIVE_EXTENSION'))
print(json.dumps(dict(status='PRE_GPU_CODE_AND_SCIENCE_BINDING_PASS',source_commit=head,source_files=len(files))))
