"""Administrative freeze for the authorized four-GPU schedule; no GPU arithmetic."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import parallel_stage as B
C=B.C;S=B.S;E=B.E;PAR=B.PAR;ROOT=C.ROOT;REPO=C.REPO

def preregister():
    B.verify_admin_freeze()
    for phase in S.PHASES:E.require(phase)
    E.require('cross_gpu_parity')
    assert not list((PAR/'scientific').rglob('drive_manifest.json'))
    history=S.historical_integrity()
    docs=C.read(ROOT/'protocol/frozen_discovery_documents.json');anchors=C.read(ROOT/'protocol/frozen_anchor_conditions.json')
    assert len(docs['documents'])==4 and anchors['ANCHOR_CONDITION_COUNT']==4
    for row in docs['documents']:
        assert C.sha(ROOT/row['token_path'])==row['token_sha256'] and C.sha(ROOT/row['text_path'])==row['text_sha256']
    for p,h in C.read(PAR/'protocol/extended_capture_policy.json')['production_policy']['source_sha256'].items():assert C.sha(p)==h
    budget=C.read(PAR/'engineering2048_storage_supplement.json')
    available=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    assert available>=budget['PEAK_REQUIRED_STORAGE_BYTES'],'FAIL_AND_STOP_STORAGE_CAPACITY'
    files=[]
    for folder in [ROOT/'protocol',ROOT/'inputs',PAR/'protocol',PAR/'scripts',PAR/'engineering/gates']:
        files.extend(p for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts)
    files.extend(PAR/n for n in ['long_qualification_freeze.json','extended_capture_freeze.json',
          'continuation_control_freeze.json','engineering2048_storage_supplement.json','storage_lifecycle_report.json',
          'static_parallel_source_audit.json','controller_commit_receipt.json','scheduler_handoff_intent.json',
          'scheduler_handoff_complete.json','engineering/capture_runtime_variability.json'])
    for gpu in range(4):files.extend((PAR/'engineering'/('topology_gpu%d'%gpu)/'engineering/repeatability/length_00256_run_01').glob('*.json'))
    files=sorted(set(files));assert all(p.is_file() for p in files)
    manifest=dict(role='FULL_SCIENTIFIC_PREMEASUREMENT_FREEZE; original G0-G12 plus cross-GPU exact gate PASS; zero scientific captures',
                  files=[dict(path=str(p.relative_to(REPO)),sha256=C.sha(p),bytes=p.stat().st_size) for p in files])
    schedule=C.read(PAR/'protocol/parallel_schedule.json');rules=C.read(E.RULES)
    prereg=dict(EXPERIMENT='QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1',scientific_type='PROSPECTIVE_CONTROLLED_MECHANISM_DISCOVERY',
         documents=docs,anchors=anchors,document_manifest_sha256=C.sha(ROOT/'protocol/frozen_discovery_documents.json'),
         anchor_manifest_sha256=C.sha(ROOT/'protocol/frozen_anchor_conditions.json'),
         horizons=[2048,8192,32768],metric_rules=rules,metric_window='inclusive1024..T',epsilon=1e-12,
         weighting='Equal layer/token weighting; whole32-head state Frobenius energies',persistent_trajectory=True,reset512=False,
         quantizer='Unmodified formal INT8-C128 quantize_c128; persistent writeback',
         capture_policy_sha256=C.sha(PAR/'protocol/extended_capture_policy.json'),canonical_capture='FIRST_SUCCESSFUL_VALID_CAPTURE',
         capture_strict_determinism=False,replay_strict_determinism=True,
         release_policy='Unchanged exact new drive_shards release after capture+all4anchors+all3horizons+all seals verified; no failed drive deletion',
         keep_forever='documents/tokens/tokenizer/source/model/drive manifests/per-shard hashes/runtime/anchors/metrics/results/release records/sparse FP references',
         storage_strategy='SEQUENTIAL_PER_DOCUMENT_CAPTURE_PARALLEL_ANCHOR_REPLAY_SEAL_RELEASE',
         maximum_concurrent_full_document_drives=1,maximum_concurrent_replay_workers=4,schedule=schedule,
         storage_formula='BASE+max(5GiB,20%BASE)<=AVAILABLE; margin exactly once',lifecycle_peak_required_bytes=budget['PEAK_REQUIRED_STORAGE_BYTES'],
         available_at_final_freeze=available,primary='C6R-minus-H',secondary=['C7-minus-C6R','Native-minus-H'],
         interpretation='Descriptive mechanism discovery, n_docs4; no universal surrogate rank or end-to-end reasoning causality',
         statistics=['per_doc','median_paired_difference','sign_count','range','horizon_consistency'],
         confirmation_rule='Original frozen hash ordering after four Discovery docs; NOT_EXECUTED',
         prohibited=dict(training=True,optimizer_step=True,AIME=True,free_generation=True,confirmation=True,intervention=True,rotation_search=True),
         all_engineering_gates='PASS',cross_gpu_parity_gate='PASS',any_new_failure='FAIL_AND_STOP; no retry or relaxation; retain incomplete drive',
         historical_integrity=history,previous_surgical_fix_reused=True)
    C.save(PAR/'preregistration.json',prereg);C.save(PAR/'PREMEASUREMENT_FREEZE_MANIFEST.json',manifest)
    S.write_text(PAR/'PREREGISTRATION.md','# Complete premeasurement freeze with authorized four-GPU scheduling\n\n'
        'One document capture on GPU0, shared sealed drive; Native/H/C6R/C7 on GPUs0/1/2/3 in independent processes. '
        'Join all four successful workers and verify every result before exact shard release, then advance to the next frozen document. '
        'No production mathematics, dtype, drive schema, metric, threshold, horizon, input, rotation or release rule changed.\n\n'+json.dumps(prereg,indent=2))
    sealed=files+[PAR/'preregistration.json',PAR/'PREMEASUREMENT_FREEZE_MANIFEST.json',PAR/'PREREGISTRATION.md']
    S.write_text(PAR/'PREMEASUREMENT_SHA256SUMS',''.join(C.sha(p)+'  '+str(p.relative_to(REPO))+'\n' for p in sorted(sealed)))
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode==0,'Unrelated staged changes; STOP'
    for p in sealed+[PAR/'PREMEASUREMENT_SHA256SUMS']:subprocess.run(['git','add','-f','--',str(p.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m',
        'Full premeasurement freeze: G0-G12 and four-GPU exact parity PASS; one-drive lifecycle'],cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for p in sealed:assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+str(p.relative_to(REPO))],cwd=REPO)).hexdigest()==C.sha(p)
    C.save(PAR/'scientific_execution_authorization.json',dict(DISCOVERY_EXECUTION_AUTHORIZED='YES',PREMEASUREMENT_COMMIT_SHA=commit,
           manifest_sha256=C.sha(PAR/'PREMEASUREMENT_FREEZE_MANIFEST.json'),preregistration_sha256=C.sha(PAR/'preregistration.json'),
           all_gates_pass=True,cross_gpu_exact_pass=True,scientific_capture_count_at_freeze=0,confirmation_authorized=False))
    print('PARALLEL_FULL_PREMEASUREMENT_FROZEN '+commit,flush=True)

if __name__=='__main__':preregister()
