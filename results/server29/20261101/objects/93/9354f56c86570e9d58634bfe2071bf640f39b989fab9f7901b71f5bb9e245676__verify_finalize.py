"""Read completed scientific artifacts, append final integrity closure only."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/data/zypan/worktrees/qwen-gdn-fd-path-qdq-interaction-v1')
EXP = REPO / 'experiments/QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1'
sys.path.insert(0, str(EXP / 'scripts'))
from common import CONDITIONS, GPU_MAP, LAYERS, EXPLOSION_THRESHOLD, read_json, sha256, write_json, write_text
from controller import verify_freeze


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def main():
    commit = 'dc30294816588e80e4f26727f62afbec47ff7d12'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == commit
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', 'fcd1dff6d286743732c6d20305cfdd31d9de7751', commit], cwd=REPO)
    verify_freeze()
    assert read_json(EXP / 'execution/status.json')['stage'] == 'COMPLETE'
    assert read_json(EXP / 'controller_exit.json') == {'exitcode': 0, 'DISCOVERY_COMPLETE': '16/16', 'FINAL_ACTION': 'STOP'}
    assert read_json(EXP / 'engineering/git_identity_resume_001/resume_v2_exit.json')['exitcode'] == 0
    assert all(v == 'PASS' for v in read_json(EXP / 'gate_status.json').values())
    assert not list(EXP.rglob('recovery_v2_error.json'))
    # The historical engineering failures are permanently retained and identified.
    historical = read_json(EXP / 'engineering/git_identity_resume_001/engineering_amendment.json')
    for row in historical['original_failure_preserved']:
        assert sha256(EXP / row['copy']) == row['sha256']
        if row['path'] != 'execution/status.json':
            assert sha256(EXP / row['path']) == row['sha256']

    units = {}
    expected_readpoints = {'12288', '16384', '20480', '24576', '28672', '32768'}
    released_bytes = 0
    released_count = 0
    for doc in range(1, 5):
        group = read_json(EXP / f'execution/formal_doc{doc:02d}_run1.json')
        assert group['all_pass'] and group['failed'] == []
        assert len(group['workers']) == 4
        for worker in group['workers']:
            assert worker['exitcode'] == 0 and worker['rank'] == GPU_MAP[worker['condition']]
            log = EXP / f'discovery/logs/formal_doc{doc:02d}_run1_{worker["condition"]}.log'
            assert sha256(log) == worker['log_sha256']
        doc_root = EXP / f'discovery/doc_{doc:02d}'
        seal = read_json(doc_root / 'result_seal.json')
        assert seal['RESULT_SEAL_GATE'] == 'PASS' and seal['unit_count'] == 4
        assert {r['condition'] for r in seal['results']} == set(CONDITIONS)
        for row in seal['results']:
            path = Path(row['path'])
            assert path.resolve().is_relative_to(EXP.resolve())
            assert sha256(path) == row['sha256'] and path.stat().st_size == row['bytes']
        release = read_json(doc_root / 'drive/drive_release_record.json')
        assert release['release_complete'] and release['all_anchor_results_complete']
        assert release['all_result_hashes_verified'] and release['parent_artifacts_deleted'] == 0
        assert sha256(doc_root / 'result_seal.json') == release['result_seal_sha256']
        assert sha256(doc_root / 'drive_identity.json') == release['drive_identity_sha256']
        assert sha256(doc_root / 'drive/drive_manifest.json') == release['drive_manifest_sha256']
        rows = release['released_paths'] + release['released_anchor_paths']
        assert release['released_bytes'] == sum(r['bytes'] for r in rows)
        assert len({r['path'] for r in rows}) == len(rows)
        for row in rows:
            path = Path(row['path'])
            assert path.resolve().is_relative_to(EXP.resolve()) and not path.exists()
        manifest = read_json(doc_root / 'drive/drive_manifest.json')
        assert {str(doc_root / 'drive' / row['relative_path']) for row in manifest['shards']} == {row['path'] for row in release['released_paths']}
        released_bytes += release['released_bytes']
        released_count += len(rows)
        for condition in CONDITIONS:
            path = EXP / f'discovery/unit_results/doc_{doc:02d}_{condition}.json'
            unit = read_json(path)
            finite(unit)
            assert unit['status'] == 'PASS' and unit['mode'] == 'formal'
            assert unit['document_index'] == doc and unit['condition'] == condition
            assert unit['physical_gpu'] == GPU_MAP[condition]
            assert unit['anchor_token'] == 8192 and unit['end_token'] == 32768
            assert unit['qdq_calls_after_anchor'] == 24576
            assert unit['int8_writeback_calls_after_anchor'] == 24576
            qualified_dtype = read_json(EXP / f'qualification/fresh_1_{condition}.json')['persistent_state_dtype']
            assert unit['persistent_state_dtype'] == qualified_dtype == 'torch.float32'
            assert unit['clean_FP_used_for_qparams_only'] and not unit['clean_FP_writeback_to_treatment']
            assert unit['clean_FP_replay_owner_rank'] == 0 and unit['clean_FP_replays_per_document_group'] == 1
            assert unit['runtime'] == {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'deterministic_algorithms': True, 'tf32_cudnn': False, 'tf32_matmul': False}
            assert not unit['training'] and unit['optimizer_steps'] == 0
            assert not unit['AIME'] and not unit['free_generation']
            assert unit['explosive'] == (unit['G_CLEAN_QPARAM'] >= EXPLOSION_THRESHOLD)
            assert set(unit['readpoint_diagnostics']) == expected_readpoints
            assert set(unit['state_hashes_final']) == {str(layer) for layer in LAYERS}
            for snap in unit['readpoint_diagnostics'].values():
                assert set(snap['per_layer']) == {str(layer) for layer in LAYERS}
                assert set(snap['focus_layers']) == {'6', '28'}
            units[(doc, condition)] = unit
    assert len(list((EXP / 'discovery/unit_results').glob('*.json'))) == 16
    assert len(units) == 16
    analysis = read_json(EXP / 'analysis/final_analysis.json')
    finite(analysis)
    treatment_count = sum(u['explosive'] for (_, c), u in units.items() if c != 'Native')
    native_count = sum(u['explosive'] for (_, c), u in units.items() if c == 'Native')
    prereg = read_json(EXP / 'preregistration.json')
    parent_count = sum(row['parent_explosive'] for row in prereg['parent_units'] if row['condition'] != 'Native')
    assert parent_count == 12 and treatment_count == 0 and native_count == 0
    assert analysis['formal_units'] == 16 and analysis['status'] == 'PASS'
    assert analysis['PARENT_PERSISTENT_EXPLOSIVE_COUNT'] == '12/12'
    assert analysis['CLEAN_QPARAM_EXPLOSIVE_COUNT'] == '0/12'
    assert analysis['NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT'] == '0/4'
    assert analysis['PRIMARY_LABEL'] == 'STATE_CONDITIONED_QPARAM_FEEDBACK_SUPPORTED'
    assert analysis['PRIMARY_NEXT_STEP'] == 'QPARAM_COMPONENT_LOCALIZATION'
    assert analysis['COUNTERFACTUAL_VALIDITY_GATE'] == 'PASS'
    for row in analysis['per_unit']:
        path = EXP / f'discovery/unit_results/doc_{row["document_index"]:02d}_{row["condition"]}.json'
        assert sha256(path) == row['result_sha256']
        assert row['clean_G'] == units[(row['document_index'], row['condition'])]['G_CLEAN_QPARAM']
    with (EXP / 'analysis/per_unit_parent_vs_treatment.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 16
    with (EXP / 'analysis/focus_layer_trajectories.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 192

    legacy_mismatches = []
    legacy_total = 0
    for line in (EXP / 'SHA256SUMS').read_text().splitlines():
        expected_sha, relative = line.split('  ', 1)
        actual_sha = sha256(EXP / relative)
        legacy_total += 1
        if expected_sha != actual_sha:
            legacy_mismatches.append({'path': relative, 'analysis_time_sha256': expected_sha, 'closed_run_sha256': actual_sha})
    assert {row['path'] for row in legacy_mismatches} == {
        'discovery/logs/analysis.log', 'discovery/logs/formal_controller.log', 'execution/status.json',
    }, 'unexpected scientific artifact hash mismatch'
    out = EXP / 'engineering/finalization_20261009'
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        'FINAL_ARTIFACT_INTEGRITY_GATE': 'PASS', 'DISCOVERY_COMPLETE': '16/16',
        'PREMEASUREMENT_COMMIT_SHA': commit, 'FROZEN_FILE_HASHES': 'PASS',
        'FORMAL_WORKER_EXITCODES': '16/16 zero', 'QDQ_AND_INT8_WRITEBACK': '16/16 PASS',
        'READPOINTS_AND_24_LAYERS': '16/16 PASS', 'ALL_METRICS_FINITE': True,
        'GPU_MAPPING_AND_DETERMINISTIC_RUNTIME': '16/16 PASS',
        'persistent_state_storage_dtype': 'torch.float32, unchanged from frozen parent and qualification',
        'low_bit_semantics': 'Every token selects INT8 codes and writes back their dequantized values to the canonical float32 carrier; no clean-FP state is written to treatment. This is the existing QDQ simulation, not a physically int8-resident kernel.',
        'COUNTERFACTUAL_VALIDITY_GATE': 'PASS',
        'PARENT_PERSISTENT_EXPLOSIVE_COUNT': '12/12', 'CLEAN_QPARAM_EXPLOSIVE_COUNT': '0/12',
        'NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT': '0/4',
        'PRIMARY_LABEL': analysis['PRIMARY_LABEL'], 'PRIMARY_NEXT_STEP': analysis['PRIMARY_NEXT_STEP'],
        'DRIVE_RELEASE_GATE': '4/4 PASS', 'released_files': released_count, 'released_bytes_cumulative': released_bytes,
        'parent_artifacts_deleted': 0, 'scientific_artifacts_or_original_manifest_modified': False,
        'original_manifest_sha256': sha256(EXP / 'SHA256SUMS'),
        'original_manifest_verified_entries': legacy_total - len(legacy_mismatches),
        'expected_post_analysis_runtime_updates': legacy_mismatches,
        'final_manifest': 'FINAL_SHA256SUMS', 'final_audit_script_sha256': sha256(Path(__file__)),
        'verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'FINAL_ACTION': 'STOP',
    }
    write_json(out / 'final_integrity_audit.json', audit)
    write_text(out / 'MANIFEST_CLOSURE.md', '# Append-only final manifest closure\n\nAll 16 scientific results, four result seals and drive release records, frozen source hashes, fixed GPU mapping, deterministic runtime and QDQ/writeback counts passed verification. The original SHA256SUMS remains unchanged. Its only three mismatches are analysis.log, formal_controller.log and execution/status.json, which were finalized after analysis generated that manifest. FINAL_SHA256SUMS seals the closed run including those runtime files, the original manifest, and this audit. No scientific result was recomputed or edited.\n')
    paths = sorted(p for p in EXP.rglob('*') if p.is_file() and p.name != 'FINAL_SHA256SUMS')
    write_text(EXP / 'FINAL_SHA256SUMS', ''.join(f'{sha256(p)}  {p.relative_to(EXP)}\n' for p in paths))
    for line in (EXP / 'FINAL_SHA256SUMS').read_text().splitlines():
        expected_sha, relative = line.split('  ', 1)
        assert sha256(EXP / relative) == expected_sha
    print(json.dumps({'audit': audit, 'final_manifest_entries': len(paths), 'FINAL_SHA256SUMS_SHA256': sha256(EXP / 'FINAL_SHA256SUMS'), 'condition_summaries': analysis['condition_summaries']}, indent=2))


if __name__ == '__main__':
    main()
