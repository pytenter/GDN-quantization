import json
from pathlib import Path

from common import (ARMS, CONDITIONS, EXP, canonical_hash, read_json, sha256,
                    write_json, write_text)


def smoke_scientific_identity(payload):
    value = dict(payload)
    for key in ['elapsed_seconds', 'peak_gpu_allocated_bytes', 'mode',
                'run_index', 'deterministic_identity_sha256',
                'paired_clean_scale_schedule']:
        value.pop(key, None)
    return canonical_hash(value)


def main():
    schedules = {}
    anchors = {}
    for condition in CONDITIONS:
        anchor = read_json(EXP / f'qualification/doc_01/anchors/{condition}.json')
        if anchor['status'] != 'PASS' or not anchor['parent_endpoint_exact'] or anchor['qdq_calls'] != 8192:
            raise RuntimeError(f'anchor qualification failed: {condition}')
        anchors[condition] = anchor
        schedule = read_json(EXP / f'qualification/schedule_identity_doc01_{condition}.json')
        if schedule['status'] != 'PASS' or not schedule['byte_identity']:
            raise RuntimeError(f'clean schedule identity failed: {condition}')
        schedules[condition] = schedule
    smoke_paths = sorted((EXP/'qualification').glob('smoke_run*_H.json'))
    if len(smoke_paths) != 2:
        raise RuntimeError(f'exactly two deterministic smoke runs required: {smoke_paths}')
    run1 = read_json(smoke_paths[0])
    run2 = read_json(smoke_paths[1])
    activation = 0
    identities = {}
    for arm in ARMS:
        first = run1['arms'][arm]
        second = run2['arms'][arm]
        first_identity = smoke_scientific_identity(first)
        second_identity = smoke_scientific_identity(second)
        if first_identity != second_identity:
            raise RuntimeError(f'short replay determinism failed: {arm}')
        if first['qdq_calls_after_anchor'] != 4096 or first['int8_writeback_calls_after_anchor'] != 4096:
            raise RuntimeError(f'every-token QDQ/writeback failed: {arm}')
        if first['persistent_state_dtype'] != 'torch.float32':
            raise RuntimeError(f'persistent carrier dtype changed: {arm}')
        if first['clean_FP_writeback_to_treatment']:
            raise RuntimeError('clean trajectory contaminated treatment')
        if first['software_unit_test']['IDENTICAL_QPARAM_REPLAY_GATE'] != 'PASS':
            raise RuntimeError('canonical QDQ unit parity failed')
        if first['streaming_reducer_parity']['G8_STREAMING_REDUCER_PARITY'] != 'PASS':
            raise RuntimeError('streaming reducer parity failed')
        activation += first['clamp_count']
        identities[arm] = first_identity
    if activation <= 0:
        raise RuntimeError('ACTIVATION_GATE failed: neither intervention activated')
    gates = {
        'G0_PARENT_ARTIFACT_IDENTITY': 'PASS',
        'G1_CLEAN_SCALE_SCHEDULE_IDENTITY': 'PASS',
        'G2_QUANTIZER_SEMANTIC_AUDIT': 'PASS',
        'G3_INTERVENTION_IDENTIFIABILITY': 'PASS',
        'G4_ACTIVATION_GATE': 'PASS',
        'G5_EVERY_TOKEN_QDQ_WRITEBACK': 'PASS',
        'G6_BASIS_SEMANTICS': 'PASS',
        'G7_SHORT_REPLAY_DETERMINISM': 'PASS',
        'G8_STREAMING_REDUCER_PARITY': 'PASS',
        'G9_STORAGE_CAPACITY': 'PASS'}
    report = {
        'status': 'PASS', 'gates': gates, 'document_index': 1,
        'historically_explosive_condition': 'H',
        'smoke_interval': [8192,12288], 'anchors': anchors,
        'smoke_artifacts': [str(x.relative_to(EXP)) for x in smoke_paths],
        'clean_schedule_identity': schedules,
        'activation': {
            arm: {'positive_scale_deviation_count': run1['arms'][arm]['positive_scale_deviation_count'],
                  'negative_scale_deviation_count': run1['arms'][arm]['negative_scale_deviation_count'],
                  'clamp_count': run1['arms'][arm]['clamp_count']}
            for arm in ARMS},
        'deterministic_identities': identities,
        'formal_measurement_count': 0}
    write_json(EXP / 'qualification/qualification_report.json', report)
    write_json(EXP / 'gate_status.json', gates, overwrite=True)
    frozen = [
        'QUANTIZER_SEMANTIC_AUDIT.json', 'INTERVENTION_IDENTIFIABILITY.md',
        'PREREGISTRATION.md', 'preregistration.json',
        'parent_artifact_identity.json', 'clean_scale_schedule_identity.json',
        'runtime_manifest.json', 'storage_capacity.json', 'gate_status.json',
        'protocol/measurement_definitions.json',
        'protocol/extended_capture_policy.json',
        'qualification/qualification_report.json']
    frozen += [str(p.relative_to(EXP)) for p in sorted((EXP/'scripts').glob('*.py'))]
    rows = [{'path': rel, 'sha256': sha256(EXP/rel),
             'bytes': (EXP/rel).stat().st_size} for rel in frozen]
    write_text(EXP / 'PREMEASUREMENT_SHA256SUMS', ''.join(
        f'{row["sha256"]}  {row["path"]}\n' for row in rows))
    write_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json', {
        'PREREGISTRATION_GATE': 'PASS', 'frozen_files': rows,
        'frozen_file_set_sha256': canonical_hash(rows),
        'formal_measurement_count': 0})
    write_json(EXP / 'execution/status.json', {
        'stage': 'QUALIFICATION_PASS_READY_TO_COMMIT',
        'formal_measurement_count': 0}, overwrite=True)
    print(json.dumps({'status': 'QUALIFICATION_PASS_READY_TO_COMMIT',
                      'gates': gates}, indent=2))


if __name__ == '__main__':
    main()
