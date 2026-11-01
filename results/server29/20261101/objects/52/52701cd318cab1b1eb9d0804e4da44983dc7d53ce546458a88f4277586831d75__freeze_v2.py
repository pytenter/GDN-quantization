"""Freeze isolated code after actual CPU regression; qualification only."""
import subprocess
import time
from v2_common import *

def main():
    assert checked(OUT/'CPU_REGRESSION_TEST_RESULTS.json')['status']=='PASS'
    assert not list(RUNS.glob('*.LAUNCH_AUTHORIZATION.json'))
    head=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(CODE),'diff','--name-only','f1fa49883d5f1f65ed47d8ddc0b37a62cbee49d1'],text=True).splitlines()
    allowed={'prepare_v2.py','v2_common.py','v2_runtime.py','v2_worker.py','v2_controller.py','test_v2_cpu.py','freeze_v2.py'}
    assert set(changes)<=allowed
    ids=checked(OUT/'SOURCE_AND_WEIGHT_IDENTITY.json')
    code={str(p):digest(p) for p in sorted(CODE.glob('*.py'))}
    emit(FREEZE,dict(status='FROZEN',epoch=time.time(),code_commit=head,code_sha256=code,
        base_commit=ids['clean_worktree_base'],changed_files=changes,engine_byte_identical=True,
        source_identity_sha256=digest(OUT/'SOURCE_AND_WEIGHT_IDENTITY.json'),
        pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
        resource_policy_sha256=digest(OUT/'RESOURCE_POLICY_AMENDMENT_V2.json'),
        original_science_metric_model_QDQ_input_unchanged=True,
        qualification_roster='two independent fresh workers per fixed document, 512 tokens each; exact numeric fingerprint',
        qualification_no_progress_seconds=1800,micro_no_progress_seconds=1800,
        formal_launch_authorized=False,formal_requires_both_qual_micro_physical_gate_and_prefreeze=True))
    emit(OUT/'ISOLATED_REPAIR_MANIFEST.json',dict(status='COMPLETE',base_commit=ids['clean_worktree_base'],
        code_commit=head,new_worktree=str(CODE),new_results=str(OUT),changed_files=changes,
        runtime_change='strict deterministic init and flags telemetry in each child before CUDA',
        numerical_engine_unchanged=True,original_STOP_preserved=True,scientific_assets_reused_without_reselection=True,
        resource_policy_sha256=digest(OUT/'RESOURCE_POLICY_AMENDMENT_V2.json'),
        phase_status={'Stage1':'PASS','Stage2':'NOT_RUN','Stage3':'NOT_RUN','Stage4':'NOT_RUN','Stage5':'NOT_RUN'}))
    print('V2_CODE_FROZEN',head,digest(FREEZE))

if __name__=='__main__': main()
