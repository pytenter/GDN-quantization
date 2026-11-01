"""Append-only final engineering stop and fresh preservation audit. No retry."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from capsule_v2 import sha,write,canonical

A=Path(__file__).resolve().parent;E=A.parents[1];ROOT=E.parents[1]


def read(p):return json.loads(Path(p).read_text())


def main():
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    r=read(A/'V2_A/verify/runtime_escape_report.json')
    assert r['gate']=='FAIL' and '/etc/locale.alias' in r['error']
    before=read(A/'protected_identity_before.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['source_head']
    assert subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True).strip()==''
    s=importlib.util.spec_from_file_location('v2_readonly_preservation',A/'prepare_source_v2.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    legacy=m.legacy_check();assert legacy==before['legacy']
    for i,x in enumerate(before['files']):
        p=ROOT/x['path'];assert p.stat().st_size==x['size'] and sha(p)==x['sha256'],x['path']
        if i%200==0:print('POST_STOP_OLD_EVIDENCE_REHASH',i,len(before['files']),flush=True)
    identity=read(E/'source_identity.json')
    for kind in ('inputs','protected_historical_files'):
        for i,x in enumerate(identity[kind]):
            assert sha(x['path'])==x['sha256'],x['path']
            if i%100==0:print('POST_STOP_PROTECTED_REHASH',kind,i,len(identity[kind]),flush=True)
    frozen=read(A/'architecture_freeze_revision1.json')
    for x in frozen['prior_attempt_preserved']:
        assert sha(A/x['path'])==x['sha256'],x['path']
    write(A/'post_stop_identity.json',dict(gate='PASS',legacy=legacy,
        old_experiment_files=len(before['files']),inherited_inputs=len(identity['inputs']),
        protected_parent_files=len(identity['protected_historical_files']),
        old_V1_stops_byte_identical=True,prior_V2_attempt_byte_identical=True,
        model_and_rotations_byte_identical=True,source_head=before['source_head'],tracked_source_changes=[]))
    base=read(A/'V2_A/work/startup_verify.json')
    startup=len({x['path'] for x in base['modules']})
    observed={x['path'] for x in r['artifacts']}
    startup_observed=set(x['path'] for x in base['modules']) & observed
    tests=read(A/'engineering_tests/results.json')
    gates={x:'NOT_RUN_NO_V2_COMMIT' for x in read(E/'preregistration.json')['blocking_gates']}
    gates.update(PREREGISTRATION_GATE='NOT_RUN_NO_V2_COMMIT',
        HISTORICAL_EXECUTION_CAPSULE_IDENTITY_GATE='NOT_RUN_MANIFEST_NOT_FROZEN',
        HISTORICAL_RUNTIME_ESCAPE_GATE='FAIL_V2_A_INGRESS_CLASSIFICATION',
        HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE='FAIL',
        FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='NOT_RUN_NO_V2_COMMIT')
    report=dict(EXPERIMENT='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',
        TASK_CLASS='ENGINEERING_ONLY_PROVENANCE_ARCHITECTURE_AMENDMENT_V2',
        V1_ORIGINAL_PREMEASUREMENT_COMMIT='66966044c0e25eb31fc15ffb1724f48de734a926',
        V1_EVENT_CAPTURE_AMENDMENT_COMMIT='c9c8e3199e0d017b9712809b77d8f16cbd07f165',
        V2_PREMEASUREMENT_COMMIT='NOT_CREATED',CAPSULE_MANIFEST_ID='NOT_FROZEN',
        V1_EVENT_LEVEL_TRACING_RETIRED='YES_IN_NEW_V2_PATH_OLD_IMPLEMENTATION_PRESERVED',
        V2_EXTERNAL_ARTIFACT_SEMANTICS_FROZEN='YES_ARCHITECTURE_ONLY_NOT_QUALIFIED',
        SCIENTIFIC_PROTOCOL_CHANGED='NO',NUMERICAL_RUNTIME_CHANGED='NO',HISTORICAL_DIRTY_WORKTREE_CHANGED='NO',
        **{'V2_TEST_'+str(i):tests['V2_TEST_'+str(i)] for i in range(1,11)},
        STARTUP_FILE_BACKED_ARTIFACTS=startup,STARTUP_MODULE_RECORDS=len(base['modules']),
        STARTUP_OBSERVED_UNIQUE_PATHS=len(startup_observed),
        RUNTIME_DISCOVERED_FILE_BACKED_ARTIFACTS=len(observed-startup_observed),
        VOLATILE_EXTERNAL_ARTIFACTS=r['VOLATILE_EXTERNAL_ARTIFACTS'],NATIVE_EXTERNAL_ARTIFACTS=r['NATIVE_EXTERNAL_ARTIFACTS'],
        CAPSULE_EXTERNAL_ARTIFACT_COUNT='NOT_FROZEN',OBSERVED_EXTERNAL_ARTIFACT_COUNT=r['OBSERVED_EXTERNAL_ARTIFACT_COUNT'],
        UNKNOWN_EXTERNAL_ARTIFACT_COUNT=r['UNKNOWN_EXTERNAL_ARTIFACT_COUNT'],UNCLASSIFIED_EXTERNAL_ARTIFACT_COUNT=1,
        CONTENT_HASH_MISMATCH_COUNT=r['CONTENT_HASH_MISMATCH_COUNT'],RUNTIME_ESCAPE_COUNT=r['RUNTIME_ESCAPE_COUNT'],
        NETWORK_ESCAPE_COUNT=r['NETWORK_ESCAPE_COUNT'],**gates,
        cross_hardware_FP_bitwise='FAIL_PRESERVED',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',
        PREMEASUREMENT_GATES_ALL_PASS='NO',GPU6_FORMAL_AUTHORIZATION_READY='NO',GPU6_FORMAL_AUTHORIZED='NO',
        FORMAL_MEASUREMENTS=0,DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',
        CONFIRMATION_EXECUTED=0,AIME_EXECUTED='NO',L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',
        FINAL_ACTION='FAIL_AND_STOP',FIRST_BLOCKING_GATE='V2_A_VERIFY_EXTERNAL_INGRESS_CLASSIFICATION',
        FIRST_BLOCKING_ARTIFACT='/etc/locale.alias',
        BLOCKER_INTERPRETATION='OS locale metadata, not demonstrated new executable code. Current regular-file ingress classifier overreaches requested executable/scientific closure boundary. No scientific or numerical conclusion is authorized.',
        NUMERICAL_MODEL_LOADED='NO',V2_A_COMPLETE='NO',V2_B_STARTED='NO',PUSH_PERFORMED='NO',
        SUPERVISOR_CLEANUP_LEDGER_REMAINING=read(A/'V2_A/verify/supervisor.json')['live_tracees_after_exit'],
        NEXT_AUTHORITY_REQUIRED='Review and explicit engineering-only direction to correct ingress-classification boundary; no formal Discovery authorization requested at this failed state.',
        architecture_manifest_sha256=sha(A/'architecture_freeze_revision1.json'),
        old_V1_failure_reports_preserved=True,old_V2_attempt_report_preserved=True,
        protected_identity_evidence='post_stop_identity.json',
        scientific_asset_definition='Existing source_identity/document/rotation/historical asset manifests unchanged',
        runtime_coverage_limitation='Architecture tests pass but real runtime capsule is unqualified; no claim of complete execution closure. Anonymous code/IPC/kernel trusted-runtime boundary remains disclosed in amendment.',
        count_definition='Native count includes actual executable/library artifacts, not only Python extensions. Startup counts are module-origin paths; remaining observed paths include runtime executable/config/data ingress, not function events.')
    write(A/'FINAL_REPORT.json',report)
    with (A/'FINAL_REPORT.md').open('x') as f:
        f.write('# PROVENANCE_ENGINEERING_AMENDMENT_V2 — Final engineering STOP\n\n')
        f.write('10 architecture tests PASS, but real V2-A stopped before model loading. No frozen execution capsule, V2-B, new premeasurement commit or post-commit parity suite exists. Previous PASS is not reused.\n\n')
        f.write('The first attempt falsely rejected an absent execvp PATH probe. Its original FAIL receipts and frozen architecture remain byte-identical. A generic nonexistent-file/successful-exec distinction was corrected and tests rerun. The second A attempt stopped on `/etc/locale.alias`; this ordinary metadata ingress exposes an over-broad classifier, not evidence of a new scientific result or a confirmed code escape. It was not automatically authorized or retried.\n\n')
        f.write('The supervisor cleanup ledger retained one unreaped bookkeeping ID. Actual process/GPU status must be referenced separately; this counter is not silently reported as zero.\n\n')
        f.write('```json\n'+json.dumps(report,indent=2,sort_keys=True)+'\n```\n')
    inventory=[dict(path=str(p.relative_to(A)),size=p.stat().st_size,sha256=sha(p)) for p in sorted(A.rglob('*')) if p.is_file()]
    write(A/'final_sha256_inventory.json',dict(files=inventory,final_action='FAIL_AND_STOP',manifest_self_excluded=True))
    print('FAIL_AND_STOP_REPORT_SAVED',len(inventory),flush=True)


if __name__=='__main__':main()
