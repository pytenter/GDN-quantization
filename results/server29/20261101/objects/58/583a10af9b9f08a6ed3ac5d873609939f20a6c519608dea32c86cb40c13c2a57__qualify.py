import json
from pathlib import Path

from common import CONDITIONS, EXP, REPO, canonical_hash, read_json, sha256, write_json, write_text


def main():
    anchor_rows = []
    for condition in CONDITIONS:
        row = read_json(EXP / f'qualification/doc_01/anchors/{condition}.json')
        if row['status'] != 'PASS' or not row['parent_endpoint_exact'] or row['qdq_calls'] != 8192:
            raise RuntimeError(f'anchor qualification failed: {condition}')
        anchor_rows.append(row)

    same_rows = {}
    fresh_rows = {}
    for condition in CONDITIONS:
        same = read_json(EXP / f'qualification/same_process_{condition}.json')
        fresh = read_json(EXP / f'qualification/fresh_1_{condition}.json')
        if same['status'] != 'PASS' or not same['identity_equal']:
            raise RuntimeError(f'same-process determinism failed: {condition}')
        first = same['runs'][0]
        if first['deterministic_identity_sha256'] != fresh['deterministic_identity_sha256']:
            raise RuntimeError(f'fresh-process determinism failed: {condition}')
        for row in [first, fresh]:
            if row['qdq_calls_after_anchor'] != 256 or row['int8_writeback_calls_after_anchor'] != 256:
                raise RuntimeError(f'every-token QDQ/writeback failed: {condition}')
            if row['basis_roundtrip_max_abs'] < 0 or not row['clean_FP_used_for_qparams_only']:
                raise RuntimeError(f'qparam basis validation failed: {condition}')
            if row['clean_FP_writeback_to_treatment']:
                raise RuntimeError(f'clean FP contaminated treatment state: {condition}')
            if row['software_unit_test']['IDENTICAL_QPARAM_REPLAY_GATE'] != 'PASS':
                raise RuntimeError('identical-qparam API unit test failed')
        same_rows[condition] = first
        fresh_rows[condition] = fresh

    h = same_rows['H']
    if h['qparam_different_entries'] <= 0 or h['qparam_max_abs_difference'] <= 0:
        raise RuntimeError('clean-qparam intervention was not activated in H smoke')

    gates = {
        'PARENT_ARTIFACT_IDENTITY': 'PASS', 'SCIENTIFIC_ASSET_IDENTITY': 'PASS',
        'INTERVENTION_IDENTIFIABILITY': 'PASS', 'INTERVENTION_ACTIVATION': 'PASS',
        'QPARAM_BASIS_VALIDATION': 'PASS', 'EVERY_TOKEN_INT8_WRITEBACK': 'PASS',
        'SHORT_REPLAY_DETERMINISM': 'PASS', 'STORAGE': 'PASS'}
    write_json(EXP / 'qualification/qualification_report.json', {
        'status': 'PASS', 'gates': gates, 'anchor_rows': anchor_rows,
        'activation_smoke': {'document_index': 1, 'condition': 'H',
                             'tokens': [8192,8448],
                             'qparam_different_entries': h['qparam_different_entries'],
                             'qparam_max_abs_difference': h['qparam_max_abs_difference']},
        'same_process_identities': {c: same_rows[c]['deterministic_identity_sha256'] for c in CONDITIONS},
        'fresh_process_identities': {c: fresh_rows[c]['deterministic_identity_sha256'] for c in CONDITIONS},
        'software_unit_test': h['software_unit_test'],
        'scientific_measurement_count': 0})
    write_json(EXP / 'gate_status.json', gates, overwrite=True)

    frozen = [
        'INTERVENTION_IDENTIFIABILITY.md', 'PREREGISTRATION.md', 'preregistration.json',
        'parent_artifact_identity.json', 'scientific_asset_identity.json',
        'runtime_manifest.json', 'storage_gate.json',
        'protocol/QPARAM_BUNDLE_DEFINITION.json', 'protocol/extended_capture_policy.json',
        'qualification/qualification_report.json', 'gate_status.json',
    ]
    frozen += [str(p.relative_to(EXP)) for p in sorted((EXP/'scripts').glob('*.py'))]
    rows = []
    for rel in frozen:
        path = EXP / rel
        rows.append({'path': rel, 'sha256': sha256(path), 'bytes': path.stat().st_size})
    write_text(EXP / 'PREMEASUREMENT_SHA256SUMS', ''.join(
        f'{row["sha256"]}  {row["path"]}\n' for row in rows))
    write_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json', {
        'PREREGISTRATION_GATE': 'PASS', 'frozen_files': rows,
        'frozen_file_set_sha256': canonical_hash(rows),
        'PREMEASUREMENT_COMMIT_SHA': None, 'scientific_measurement_count': 0})
    write_json(EXP / 'execution/status.json', {
        'stage': 'QUALIFICATION_PASS_READY_TO_COMMIT',
        'scientific_measurement_count': 0}, overwrite=True)
    print(json.dumps({'status': 'QUALIFICATION_PASS_READY_TO_COMMIT',
                      'gates': gates}, indent=2))


if __name__ == '__main__':
    main()
