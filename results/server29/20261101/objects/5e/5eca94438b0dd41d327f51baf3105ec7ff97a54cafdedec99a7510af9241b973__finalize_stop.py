"""Append-only stop report and independent actual-process/GPU observations."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from capsule_v2 import sha, write

C=Path(__file__).resolve().parent; A=C.parent; E=A.parents[1]; R=E.parents[1]


def main():
    get=lambda p:json.loads(Path(p).read_text())
    fail=get(C/'V2_A/STOP.json'); assert fail['FINAL_ACTION']=='FAIL_AND_STOP'
    raw=get(C/'V2_A/verify/runtime_escape_report.json')
    sem=get(C/'V2_A/verify/semantic_ingress_report.json')
    sup=get(C/'V2_A/verify/supervisor.json')
    startup=get(C/'V2_A/work/startup_verify.json')
    unknown='/usr/share/locale-langpack/zh_CN/LC_MESSAGES/git.mo'
    assert unknown in raw['error']
    locale_rows=[x for x in sem['ambient_metadata'] if x['path']=='/etc/locale.alias']
    assert len(locale_rows)==1 and locale_rows[0]['classification']=='C_AMBIENT_OS_METADATA_CONFIG'
    assert locale_rows[0]['sha256']=='c22cc401a22cb37b7f1c8766035bd4d599fa2d2da9d20bdd85be67711265d02b'
    assert locale_rows[0]['consumer']['consumer_role']=='FROZEN_GIT_IDENTITY_QUERY'
    protected=get(C/'prior_v2_protection.json')
    for row in protected['files']:
        p=A/row['path']; assert sha(p)==row['sha256'] and p.stat().st_size==row['size'],str(p)
    candidate=get(C/'candidate_architecture_freeze.json')
    for row in candidate['source_files']: assert sha(C/row['name'])==row['sha256']
    assert get(A/'FINAL_REPORT.json')['SUPERVISOR_CLEANUP_LEDGER_REMAINING']==1
    assert not (C/'V2_B').exists() and not (C/'new_premeasurement_commit.json').exists()
    assert not (C/'V2_A/work/loaded_modules_verify.json').exists()
    table=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
    names=('capsule_runner_semantic.py','launch_qualification_a.py','syscall_supervisor_v2.py','capsule_runner_v2.py')
    tasks=[x for x in table.splitlines() if any(n in x for n in names)]
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
    uuid='GPU-de53f243-b271-521c-d875-d96611ed8574'
    gpu6=[x for x in gpu.splitlines() if uuid in x]
    state=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits','-i','6'],text=True).strip()
    write(C/'post_stop_actual_state.json',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_cleanup_ledger_remaining=1,correction_cleanup_ledger_remaining=sup['cleanup_ledger_remaining'],
        actual_task_processes=tasks,actual_gpu6_compute_processes=gpu6,actual_gpu6_state=state,
        counters_not_conflated=True,no_restart_performed=True))
    assert not tasks and not gpu6,'ACTUAL_PROCESS_OR_GPU_STATE_REQUIRES_REVIEW'
    with open(unknown,'rb') as f:magic=f.read(4).hex()
    write(C/'new_unknown_readonly_diagnostic.json',dict(path=unknown,classification='D_UNKNOWN_NOT_AUTHORIZED',
        sha256=sha(unknown),size=Path(unknown).stat().st_size,first_four_bytes_hex=magic,
        observation='Read-only forensic inspection after STOP; no rerun, whitelist addition or authorization',
        probable_role='GNU gettext message-catalog candidate based on format, NOT accepted by current frozen classifier',
        successful_scientific_ingress=False,confirmed_executable_escape=False))
    gates={k:'NOT_RUN_NEW_POST_COMMIT_SUITE_NO_V2_COMMIT' for k in get(E/'preregistration.json')['blocking_gates']}
    gates.update(PREREGISTRATION_GATE='NOT_RUN_NO_NEW_V2_COMMIT',
        HISTORICAL_EXECUTION_CAPSULE_IDENTITY_GATE='NOT_FROZEN_A_INCOMPLETE',
        HISTORICAL_RUNTIME_ESCAPE_GATE='FAIL_CLOSED_UNKNOWN_INGRESS_NOT_CONFIRMED_EXECUTABLE_ESCAPE',
        FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='NOT_RUN_A_VERIFY_STOPPED')
    report=dict(EXPERIMENT='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',
        TASK='ENGINEERING_ONLY_PROVENANCE_CLASSIFIER_CORRECTION_AND_QUALIFICATION',
        ORIGINAL_V2_FAIL_STOP_REMAINS_VALID='YES',ORIGINAL_TWO_A_ATTEMPTS_BYTE_IDENTITY='PASS',
        ORIGINAL10_ARCHITECTURE_TESTS='10/10 PASS',NEW_SEMANTIC_ARCHITECTURE_TESTS='13/13 PASS',
        ORIGINAL10_SCRIPT_BYTE_IDENTITY='PASS',SEMANTIC_CLASSIFIER_ARCHITECTURE_GATE='PASS_TESTS_ONLY_RUNTIME_UNQUALIFIED',
        PATH_SPECIFIC_LOCALE_WHITELIST_ADDED='NO',LOCALE_ALIAS_CLASSIFICATION='C_AMBIENT_OS_METADATA_CONFIG',
        LOCALE_ALIAS_READ_ONLY_RECORD='PASS',LOCALE_ALIAS_SHA256=locale_rows[0]['sha256'],
        locale_environment={k:startup['environment'][k] for k in ('LANG','LC_ALL','LC_CTYPE','LC_NUMERIC','LC_COLLATE')},
        active_locale=startup['active_locale'],active_locale_categories=startup['active_locale_categories'],
        numerical_locale_configuration=startup['numerical_locale_configuration'],
        locale_end_baseline='NOT_AVAILABLE_CHILD_STOPPED_BEFORE_VERIFY_COMPLETION',
        NUMERICAL_CONFIGURATION_CHANGED='NO_BY_IMPLEMENTATION_NO_NUMERICAL_MODEL_COMPUTATION_EXECUTED',
        V2_A_QUALIFICATION='FAIL_VERIFY',V2_A_COMPLETE='NO',V2_B_STARTED='NO',
        FIRST_BLOCKING_ARTIFACT=unknown,FIRST_BLOCKING_CLASSIFICATION='D_UNKNOWN',
        NEW_UNKNOWN_INGRESS_COUNT=raw['UNKNOWN_EXTERNAL_ARTIFACT_COUNT'],
        CONTENT_HASH_MISMATCH_COUNT=raw['CONTENT_HASH_MISMATCH_COUNT'],
        NETWORK_ESCAPE_COUNT=raw['NETWORK_ESCAPE_COUNT'],
        RAW_SUPERVISOR_RUNTIME_ESCAPE_COUNTER=raw['RUNTIME_ESCAPE_COUNT'],
        CONFIRMED_EXECUTABLE_ESCAPE='NOT_ESTABLISHED',SCIENTIFIC_FAILURE='NOT_ESTABLISHED_NO_SCIENCE_RUN',
        OBSERVED_CLOSURE_ARTIFACTS=raw['OBSERVED_EXTERNAL_ARTIFACT_COUNT'],
        OBSERVED_AMBIENT_METADATA_FILES=len(sem['ambient_metadata']),
        NATIVE_ARTIFACT_COUNT=raw['NATIVE_EXTERNAL_ARTIFACTS'],
        AUTO_WHITELIST='NO',AUTO_RETRY='NO',CAPSULE_FREEZE='NOT_PERFORMED',
        EXTERNAL_ARTIFACT_MANIFEST_FREEZE='NOT_PERFORMED',V2_PREMEASUREMENT_COMMIT='NOT_CREATED',
        POST_COMMIT_PARITY_IDENTITY_SUITE='NOT_RUN',**gates,
        SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,
        CORRECTION_SUPERVISOR_CLEANUP_LEDGER_REMAINING=sup['cleanup_ledger_remaining'],
        ACTUAL_TASK_PROCESS_COUNT=len(tasks),ACTUAL_GPU6_COMPUTE_PROCESS_COUNT=len(gpu6),
        GPU6_FORMAL_AUTHORIZED='NO',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',
        cross_hardware_FP_bitwise='FAIL_PRESERVED',
        DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',FORMAL_MEASUREMENTS=0,
        L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',
        SCIENTIFIC_PROTOCOL_CHANGED='NO',PUSH_PERFORMED='NO',FINAL_ACTION='FAIL_AND_STOP',
        NEXT_AUTHORITY_REQUIRED='Explicit engineering-only review of new unknown ingress/semantic family; no automatic classifier expansion or rerun',
        candidate_architecture_sha256=sha(C/'candidate_architecture_freeze.json'),
        original_v2_final_report_sha256=sha(A/'FINAL_REPORT.md'))
    write(C/'QUALIFICATION_FINAL_REPORT.json',report)
    with (C/'QUALIFICATION_FINAL_REPORT.md').open('x',encoding='utf-8') as f:
        f.write('# Engineering-only classifier correction — FAIL_AND_STOP\n\n')
        f.write('Original V2 FAIL/STOP and both previous A receipts remain byte-identical. No path-specific locale exception was added. Original ten tests were rerun unchanged; thirteen additional semantic tests PASS. This is architecture evidence, not successful runtime qualification.\n\n')
        f.write('The real A verify run recorded `/etc/locale.alias` as read-only locale-resolution metadata consumed by a closed git status query, then stopped on unknown `/usr/share/locale-langpack/zh_CN/LC_MESSAGES/git.mo`. No auto-whitelist or retry; B, capsule freeze, new commit and post-commit suites were not performed. No model was loaded. This is not a scientific failure or a demonstrated executable escape.\n\n')
        f.write('Locale startup fields are recorded. LC_NUMERIC and LC_COLLATE are C; LC_CTYPE is zh_CN.UTF-8. No post-run locale baseline is available because verify was interrupted; do not report a completed before/after runtime gate.\n\n')
        f.write('Original cleanup ledger remains 1. New-run ledger also remains separately recorded; actual process and GPU tables are empty. The bookkeeping counter is not rewritten to zero.\n\n')
        f.write('Source historical preservation is audited separately in `post_correction_preservation.json`; absence of a new premeasurement commit means original formal blocking gates are NOT rerun/PASS-qualified.\n\n')
        f.write('```json\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\n```\n')
    print('APPEND_ONLY_QUALIFICATION_FAIL_STOP_REPORT_SAVED',flush=True)


if __name__=='__main__':main()
