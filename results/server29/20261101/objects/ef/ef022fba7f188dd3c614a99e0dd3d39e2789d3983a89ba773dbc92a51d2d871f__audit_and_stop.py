"""Append-only CPU metadata audit; stop on unrecoverable Discovery ordering.

No Torch/dataset imports, model calls, document contents or scientific metrics.
This is an audit commit, NOT a successful input-freeze preregistration.
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

TARGET = 'QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1'
AMENDMENT = TARGET + '_PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1'
REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1')
OLD_REPO = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1')
PREVIOUS = REPO / 'experiments' / TARGET
ROOT = PREVIOUS / 'amendments/PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1'
PARENT = OLD_REPO / 'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'
CUMSUM = OLD_REPO / 'experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1'
CORE = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def write(name, value):
    with (ROOT / name).open('x') as f:
        f.write(value)

def command(args, cwd=REPO):
    p = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return dict(command=args, cwd=str(cwd), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def seal(root, name):
    manifest = root / name
    rows = []
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.lstrip('*')
        path = root / rel
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise RuntimeError('seal path escapes root')
        actual = sha(path) if path.is_file() else None
        rows.append(dict(relative_path=rel, expected_sha256=expected, actual_sha256=actual, PASS=actual == expected))
    return dict(authoritative_path=str(root), manifest=str(manifest), manifest_sha256=sha(manifest),
                sealed_artifact_count=len(rows), all_exact=all(r['PASS'] for r in rows), rows=rows)

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('requires CUDA_VISIBLE_DEVICES empty')
    status = command(['git', 'status', '--porcelain'])
    # Only this new, isolated amendment namespace may be untracked.
    for line in status['stdout'].splitlines():
        if '/amendments/' not in line:
            raise RuntimeError('unexpected worktree changes; no edits made')
    entries = {'parent': seal(PARENT, 'FINAL_SHA256SUMS'),
               'cumsum_closure': seal(CUMSUM, 'FINAL_SHA256SUMS'),
               'previous_preflight': seal(PREVIOUS, 'SHA256SUMS')}
    if not all(v['all_exact'] for v in entries.values()):
        save('seal_failure.json', entries)
        raise RuntimeError('old seal mismatch; STOP without repairing evidence')
    pv = read(PARENT / 'analysis/final_verdict.json')
    cv = read(CUMSUM / 'provenance/final_verdict.json')['verdict']
    av = read(PREVIOUS / 'final_verdict.json')
    assert pv['TASK_STATUS'] == 'STOP_STRICT_DETERMINISTIC_RUNTIME_UNAVAILABLE'
    assert cv['DETERMINISTIC_CUMSUM_COMPATIBILITY_GATE'] == 'FAIL'
    assert av['verdict']['FROZEN_INPUT_IDENTITY_GATE'] == 'FAIL'
    assert av['verdict']['FINAL_ACTION'] == 'FAIL_AND_STOP'
    identities = {}
    for label, root, repo, verdict_path in [
        ('parent', PARENT, OLD_REPO, PARENT/'analysis/final_verdict.json'),
        ('cumsum_closure', CUMSUM, OLD_REPO, CUMSUM/'provenance/final_verdict.json'),
        ('previous_preflight', PREVIOUS, REPO, PREVIOUS/'final_verdict.json')]:
        rel = str(verdict_path.relative_to(repo))
        identities[label] = dict(repository=str(repo), branch=command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], repo)['stdout'].strip(),
            repository_HEAD=command(['git', 'rev-parse', 'HEAD'], repo)['stdout'].strip(),
            verdict_source_commit=command(['git', 'log', '-1', '--format=%H', '--', rel], repo)['stdout'].strip(),
            verdict_path=str(verdict_path), verdict_sha256=sha(verdict_path), verdict=read(verdict_path))
    save('pre_amendment_evidence_inventory.json', dict(AMENDMENT=AMENDMENT, inspected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        seals=entries, identities=identities, previous_verdicts_never_rewritten=True,
        authoritative_original_paths_used=True, git_status_before_audit=status))
    disc = read(PARENT/'analysis/discovery_summary.json')
    conf = read(PARENT/'analysis/confirmation_summary.json')
    trace = read(PARENT/'traces/trace_status.json')
    execution = read(PREVIOUS/'execution_status.json')
    phase = read(PARENT/'protocol/preregistration_repaired_001.json')
    scientific_dirs = ['discovery','confirmation','capture','captured_drives','replay','metrics']
    dir_checks = [{ 'path': str(base/name), 'exists': (base/name).exists()} for base in [PARENT, PREVIOUS] for name in scientific_dirs]
    csv_files = [str(p) for base in [PARENT, PREVIOUS] for p in base.rglob('*.csv') if ROOT not in p.parents]
    assert not any(r['exists'] for r in dir_checks) and not csv_files
    assert disc['complete']=='0/48' and not disc['rotation_conditioned_metrics_computed']
    assert pv['CONFIRMATION_COMPLETE']=='0/48' and phase['confirmation_units_started']==0
    assert trace['fresh_document_trace_count']==0 and trace['teacher_forced_recurrent_steps']==0
    assert av['scientific_measurements']==0 and execution['reducer_metrics']==0 and execution['replay_units']==0
    save('scientific_measurement_audit.json', dict(SCIENTIFIC_MEASUREMENT_COUNT=0, SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_FREEZE=0,
        DISCOVERY_COMPLETE='0/48', CONFIRMATION_DOCS_READ='NO', CONFIRMATION_DOCS_EXECUTED=0,
        PROSPECTIVE_INPUT_FREEZE_ELIGIBLE='YES', parent_discovery=disc, parent_confirmation_metadata=conf,
        parent_trace_status=trace, previous_execution_status=execution, scientific_directory_metadata=dir_checks,
        result_csv_files=csv_files, historical_development_engineering_probes_are_not_Discovery=True,
        proof_scope='Sealed authoritative protocol/status/result inventories; no scientific output directories or CSVs; no document content read.'))
    # Recover prospective rules, not scientific outcome tables.
    history = command(['git','log','--format=%H','--','experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'], OLD_REPO)
    searches = []
    for commit in history['stdout'].splitlines():
        searches.append(command(['git','grep','-n','-i','-E','PG19|deterministic.*order|selection.*rank|fresh.*doc',commit,'--',
            'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/protocol',
            'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/scripts'], OLD_REPO))
    core_lines = CORE.read_text().splitlines()
    old_task = ROOT/'source/parent_task_original.txt'
    lines = old_task.read_text(encoding='utf-8-sig').splitlines()
    rule_excerpt = '\n'.join(lines[510:591])
    assert 'deterministic pre-existing ordering' in rule_excerpt
    assert 'PG19 TEST / existing independent long-text pool' in rule_excerpt
    discovery = dict(DISCOVERY_DOCUMENT_RULE_RECOVERY_GATE='FAIL', rule_type='PARTIAL_PRE_EXISTING_RULE_NOT_UNIQUELY_EXECUTABLE',
        source_artifact=str(old_task), source_sha256=sha(old_task), source_lines_one_based=[511,591],
        source_commit='Original user task attachment not itself previously committed; prior preflight sealed its rule summary at '+identities['previous_preflight']['verdict_source_commit'],
        prior_rule_summary=dict(path=str(PREVIOUS/'preflight_parent_inventory.json'),sha256=sha(PREVIOUS/'preflight_parent_inventory.json')),
        exact_rule=rule_excerpt, candidate_pool='Preferred PG19 TEST / existing independent long-text pool; no pinned pool manifest or revision named',
        eligibility_criteria='Fresh non-AIME long text, at least32769 Qwen IDs/doc, all enumerated historical Qwen uses excluded',
        ordering_rule='deterministic pre-existing ordering (named, but comparator/order manifest/procedure not specified or recovered)',
        seed_if_any='NOT_SPECIFIED; no authoritative randomized selection procedure recovered', tie_policy='NOT_SPECIFIED',
        exclusion_rule='Prior Qwen training, validation, calibration, Shared16, mechanisms, M5, Stage-B, PG19, candidate selection, propagation; Ling only if metadata available without outcome access',
        expected_Discovery_count=4, expected_Confirmation_count=4, actual_selected_count=0,
        missing_unique_identity_fields=['authoritative candidate-pool manifest/path','pinned dataset revision/record identity','applicable pre-existing ordering/comparator','applicable tie policy if ordering keys collide'],
        history_query=history, historical_protocol_source_searches=searches,
        relevant_source=dict(path=str(CORE),sha256=sha(CORE),lines_one_based=[103,108],source='\n'.join(core_lines[102:108]),
            interpretation='corpus_rows preserves order of an existing calibration corpus and filters TRAIN/VALIDATION; not a specified fresh PG19 TEST selection policy; previously used corpus is excluded'),
        ambiguity='Dataset record order, document-ID order and hash order are all deterministic but not interchangeable. Selecting any one now would invent an unrecorded scientific input rule.',
        confirmation_contents_read=False, new_scientific_selection_rule_created=False,
        action='STOP under amendment sections8 and26, before document selection or rotation materialization')
    save('discovery_selection_rule_recovery.json', discovery)
    write('discovery_selection_rule_recovery.md', '# Discovery rule recovery: FAIL\n\n'+rule_excerpt+'\n\n'
        'The original rule names a deterministic pre-existing ordering but does not define or pin it. '
        'Parent protocol/source history does not supply a candidate-pool manifest, dataset revision, comparator or applicable tie policy. '
        'The historical calibration corpus order is not an authorized fresh TEST ordering and used calibration documents are explicitly excluded. '
        'No Discovery document selected, normalized or tokenized. No Confirmation content or token statistics read. '
        'This failure is rule ambiguity, NOT merely absence of already-materialized inputs.\n\n'
        'Section8 requires FAIL_AND_STOP; document-ID/hash/dataset order must not be substituted without a separately prospective authorization.\n')
    # Record the already-known original bank definition as documentary evidence only.
    bank_excerpt='\n'.join(lines[620:677])
    assert 'R11' in bank_excerpt
    save('rotation_bank_rule_recovery.json', dict(status='PRE_EXISTING_DEFINITION_RECORDED_ONLY',
        ROTATION_BANK_RULE_RECOVERY_GATE='NOT_EVALUATED', stop_dependency='Discovery rule recovery already failed; no bank stage executed',
        original_source=str(old_task), source_sha256=sha(old_task), source_lines_one_based=[621,677], exact_rule=bank_excerpt,
        condition_count_defined=12, RULE_ALREADY_SPECIFIED=True, NEWLY_INVENTED_RULE=False,
        interpolation='Layerwise archived Cayley/skew parameter interpolation, not Euclidean matrix interpolation',
        matrix_materialization_performed=False, new_orthogonality_calculations=0,
        recovery_evidence_reference=str(PARENT/'provenance/rotation_parameterization_identity.json'),
        recovery_evidence_sha256=sha(PARENT/'provenance/rotation_parameterization_identity.json')))
    write('rotation_bank_rule_recovery.md', '# Original rotation rule: documentary record only\n\n'+bank_excerpt+'\n\n'
        'This mathematical definition was already specified before measurement. No new coefficients were invented. '
        'Its formal recovery/materialization gates were NOT_EVALUATED in this amendment because Discovery rule recovery failed first. '
        'No checkpoint tensor was loaded and no new matrix or orthogonality metric computed.\n')
    verdict = dict(AMENDMENT=AMENDMENT, PREVIOUS_PREFLIGHT_FAIL_PRESERVED='YES', PARENT_STOP_PRESERVED='YES', OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES',
        PARENT_SEAL_INTEGRITY_GATE='PASS', CUMSUM_CLOSURE_SEAL_INTEGRITY_GATE='PASS', PREVIOUS_PREFLIGHT_SEAL_INTEGRITY_GATE='PASS',
        SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_FREEZE=0, PROSPECTIVE_INPUT_FREEZE_ELIGIBLE='YES',
        DISCOVERY_DOCUMENT_RULE_RECOVERY_GATE='FAIL', DISCOVERY_DOCUMENT_FREEZE_GATE='NOT_EVALUATED', DISCOVERY_DOC_COUNT=0,
        ROTATION_BANK_RULE_RECOVERY_GATE='NOT_EVALUATED', ROTATION_BANK_MATERIALIZATION_GATE='NOT_EVALUATED', ROTATION_BANK_ORTHOGONALITY_GATE='NOT_EVALUATED', ROTATION_BANK_COUNT=0,
        NUMERICAL_POLICY_FREEZE_GATE='NOT_EVALUATED', METRIC_DEFINITION_FREEZE_GATE='NOT_EVALUATED', HORIZON_POLICY_FREEZE_GATE='NOT_EVALUATED',
        PREMEASUREMENT_FREEZE_MANIFEST_GATE='NOT_EVALUATED', PREMEASUREMENT_COMMIT_SHA=None, PREMEASUREMENT_FREEZE_COMPLETE='NO',
        FROZEN_INPUT_IDENTITY_GATE='FAIL', SCIENTIFIC_METRIC_EXECUTION_COUNT=0, DISCOVERY_COMPLETE='0/48', CONFIRMATION_DOCS_READ='NO', CONFIRMATION_DOCS_EXECUTED=0,
        PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME='NO', NEW_C8_TRAINING_AUTHORIZED='NO', NEW_AIME_GENERATION=0, NEW_FREE_GENERATION=0,
        NEXT_AUTHORIZED_PHASE='NONE', FINAL_ACTION='FAIL_AND_STOP', NEW_TRAINING='NO', OPTIMIZER_STEP=0, PUSH_PERFORMED='NO')
    save('final_verdict.json',dict(verdict=verdict, stop_reason=discovery['missing_unique_identity_fields'],
        old_artifacts_modified=0, no_freeze_manifest_created=True, audit_commit_is_not_premeasurement_freeze_commit=True))
    save('execution_status.json',dict(old_evidence_audit='PASS', scientific_measurement_audit='PASS', discovery_rule_recovery='FAIL',
        downstream_input_preparation='NOT_STARTED', model_loaded=False, GPU_work_started=False, document_contents_read=0,
        document_tokenizations=0, rotation_bank_materializations=0, scientific_metric_execution_count=0,
        factorization_implementation_started=False, capture_attempts=0, replay_attempts=0,
        parent_metric_equivalence='NOT_EVALUATED', frozen_original_horizons=[2048,8192,32768],
        original_metric_definitions_sha256=sha(PARENT/'protocol/metric_definitions.json'),
        missing_downstream_artifacts_reason='Fail-closed at document rule recovery; no fabricated successful freeze deliverables'))
    report = '# Premeasurement input-freeze amendment: FAIL_AND_STOP\n\n'
    report += 'Input preparation is prospectively eligible (scientific measurement count0), but the old Discovery rule cannot uniquely determine four documents. '
    report += 'It names PG19 TEST / an independent long-text pool and a deterministic pre-existing ordering, without pinning the pool/revision/order. '
    report += 'Old parent protocol/source commits and the referenced calibration corpus loader do not recover an applicable unique rule. '
    report += 'Amendment sections8/26 therefore require stopping; this is not a demand that inputs already have been materialized.\n\n'
    report += 'All135 sealed predecessor artifacts (47parent+80cumsum+8Attempt0) pass SHA256 verification. All three old STOP/FAIL verdicts are permanently unchanged. '
    report += 'Discovery0/48, Confirmation0; no content read, no model/GPU work, no bank constructed, no FD metric calculation. '
    report += 'The original12-condition mathematical bank definition is recorded only; downstream gates are NOT_EVALUATED. '
    report += 'No successful PREMEASUREMENT_FREEZE_MANIFEST or preregistration commit is created. Any commit sealing this failed audit is an audit commit only. No push.\n\n'
    report += 'To proceed in a separately authorized amendment, pin a candidate dataset/pool revision and exact pre-result ordering/tie procedure, or provide an authoritative old manifest/source proving those rules already existed. '
    report += 'This report authorizes no such new rule and no next engineering/scientific phase.\n\n## Required final fields\n\n```text\n'
    report += '\n'.join(k+' = '+('null' if v is None else str(v)) for k,v in verdict.items())+'\n```\n'
    write('FINAL_AMENDMENT_REPORT.md',report)
    after={'parent':seal(PARENT,'FINAL_SHA256SUMS'),'cumsum_closure':seal(CUMSUM,'FINAL_SHA256SUMS'),'previous_preflight':seal(PREVIOUS,'SHA256SUMS')}
    assert all(v['all_exact'] for v in after.values())
    save('final_old_seal_revalidation.json',dict(all_exact=True,seals=after))
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    write('SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':
    main()
