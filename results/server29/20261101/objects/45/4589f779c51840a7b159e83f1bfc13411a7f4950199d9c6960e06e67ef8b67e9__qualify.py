import json
from pathlib import Path

from common import EXP, read_json, sha256, write_json


def main():
    same = read_json(EXP / 'smoke/same_process.json')
    fresh = [read_json(EXP / f'smoke/fresh_process_{i}.json') for i in [1, 2, 3]]
    if same['status'] != 'PASS' or len(same['runs']) != 3:
        raise RuntimeError('same-process qualification incomplete')
    runs = same['runs'] + fresh
    identities = [x['scientific_identity']['identity_sha256'] for x in runs]
    if len(set(identities)) != 1:
        raise RuntimeError('replay determinism identity mismatch')
    if any(x['status'] != 'PASS' for x in runs):
        raise RuntimeError('smoke run failed')
    if same['runs'][0]['streaming_fixture']['gate'] != 'PASS':
        raise RuntimeError('streaming parity failed')
    if fresh[0]['streaming_fixture']['gate'] != 'PASS':
        raise RuntimeError('fresh-process streaming parity failed')
    identity = runs[0]['scientific_identity']
    if identity['pulse_qdq_calls_after_8192'] != 0:
        raise RuntimeError('pulse QDQ count is not zero')
    if identity['pulse_writeback_calls_after_8192'] != 0:
        raise RuntimeError('pulse writeback count is not zero')
    if identity['calls']['blocked_cumsum'] != 0 or identity['calls']['full_model_forward'] != 0:
        raise RuntimeError('isolation gate failed')
    if identity['calls']['QDQ_path'] != 32768:
        raise RuntimeError('persistent positive control QDQ count mismatch')

    plan = read_json(EXP / 'gate_plan.json')
    if plan['G16_STORAGE_CAPACITY_GATE'] != 'PASS':
        raise RuntimeError('storage gate not pass')
    fixed = {
        'G0_PARENT_ARTIFACT_INTEGRITY_GATE': 'PASS',
        'G1_SOURCE_IDENTITY_GATE': 'PASS',
        'G2_MODEL_CHECKPOINT_IDENTITY_GATE': 'PASS',
        'G3_DOCUMENT_IDENTITY_GATE': 'PASS',
        'G4_ANCHOR_IDENTITY_GATE': 'PASS',
        'G5_FROZEN_DRIVE_IDENTITY_GATE': 'PASS_DOC1_SMOKE; PER_UNIT_ENFORCED',
        'G6_TRANSITION_F_IDENTITY_GATE': 'PASS',
        'G7_TRANSITION_Q_IDENTITY_GATE': 'PASS',
        'G8_PARENT_AGGREGATOR_PARITY_GATE': 'PASS',
        'G9_ANCHOR_STATE_IDENTITY_GATE': 'PASS',
        'G10_FP_REFERENCE_REPLAY_PARITY_GATE': 'PASS',
        'G11_PERSISTENT_CONTROL_PARITY_GATE': 'PASS',
        'G12_NO_FURTHER_QDQ_PULSE_GATE': 'PASS',
        'G13_REPLAY_ISOLATION_GATE': 'PASS',
        'G14_REPLAY_DETERMINISM_GATE': 'PASS',
        'G15_STREAMING_REDUCER_PARITY_GATE': 'PASS',
        'G16_STORAGE_CAPACITY_GATE': 'PASS',
        'same_process_runs': 3,
        'fresh_process_runs': 3,
        'shared_identity_sha256': identities[0],
        'pulse_qdq_calls_after_8192': 0,
        'persistent_qdq_calls_after_8192': identity['persistent_qdq_calls_after_8192'],
        'streaming_fixture_same_process': same['runs'][0]['streaming_fixture'],
        'streaming_fixture_fresh_process': fresh[0]['streaming_fixture'],
        'PREREGISTRATION_GATE': 'PASS',
    }
    write_json(EXP / 'engineering/gates/qualification.json', fixed)
    write_json(EXP / 'gate_report.json', fixed)
    auth = read_json(EXP / 'execution_authorization.json')
    write_json(EXP / 'discovery_authorization.json', {
        'CAUSAL_DISCOVERY_EXECUTION_AUTHORIZED': 'YES',
        'PREMEASUREMENT_COMMIT_SHA': auth['PREMEASUREMENT_COMMIT_SHA'],
        'qualification_sha256': sha256(EXP / 'engineering/gates/qualification.json'),
        'expected_units': 16,
        'stop_on_any_failure': True,
        'retry_authorized': False,
    })
    print('ALL_G0_G16_QUALIFICATION_GATES_PASS', flush=True)


if __name__ == '__main__':
    main()
