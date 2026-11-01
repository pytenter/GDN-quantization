import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import (ANCHOR_TOKEN, CHECKPOINTS, CONDITIONS, ENDPOINT, EXP,
                    EXPLOSION_THRESHOLD, FIRST_TOKEN, GPU_MAP, PARENT,
                    PARENT_ROOT, REPO, sha256, read_json, verify_sum_file,
                    write_json, write_text)


def gpu_inventory():
    query = 'index,uuid,name,memory.total,driver_version'
    raw = subprocess.check_output([
        'nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'
    ], text=True)
    rows = []
    for line in raw.strip().splitlines():
        index, uuid, name, memory, driver = [x.strip() for x in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'name': name,
                     'memory_total_mib': int(memory), 'driver': driver})
    return rows


def parent_explosive_units():
    path = PARENT / 'results/per_unit_horizon_metrics.csv'
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    values = {(int(r['document_index']), r['condition'], int(r['horizon_tokens'])):
              float(r['J_ACC_FD']) for r in rows}
    units = []
    for doc in range(1, 5):
        for condition in CONDITIONS:
            j8 = values[(doc, condition, 8192)]
            j32 = values[(doc, condition, 32768)]
            gain = j32 / j8
            units.append({'document_index': doc, 'condition': condition,
                          'parent_J_8192': j8, 'parent_J_32768': j32,
                          'G_PERSIST_PARENT_32768': gain,
                          'PERSISTENT_EXPLOSIVE_UNIT': gain >= EXPLOSION_THRESHOLD})
    return units


def main():
    if not EXP.exists():
        raise RuntimeError('experiment directory must be created with scripts first')
    for name in ['protocol', 'provenance', 'engineering/gates', 'smoke',
                 'discovery/unit_results', 'discovery/per_layer',
                 'discovery/logs', 'analysis', 'execution']:
        (EXP / name).mkdir(parents=True, exist_ok=True)

    final_rows = verify_sum_file(PARENT, PARENT / 'FINAL_SHA256SUMS')
    parallel_rows = verify_sum_file(PARENT, PARENT / 'PARALLEL_FINAL_SHA256SUMS')
    parent_verdict = read_json(PARENT / 'final_verdict.json')
    assert parent_verdict['FINAL_ACTION'] == 'DISCOVERY_COMPLETE_REPORT_ONLY'
    assert parent_verdict['DISCOVERY_COMPLETE'] == '16/16'

    parent_policy = read_json(PARENT / 'protocol/extended_capture_policy.json')
    production = parent_policy['production_policy']
    source_rows = []
    for raw_path, expected in production['source_sha256'].items():
        actual = sha256(raw_path)
        if actual != expected:
            raise RuntimeError(f'production source mismatch: {raw_path}')
        source_rows.append({'path': raw_path, 'sha256': actual})

    model_rows = []
    for row in production['model_identity']['files']:
        actual = sha256(row['path'])
        if actual != row['sha256']:
            raise RuntimeError(f'model identity mismatch: {row["path"]}')
        model_rows.append({'path': row['path'], 'sha256': actual,
                           'bytes': Path(row['path']).stat().st_size})

    documents = read_json(PARENT_ROOT / 'protocol/frozen_discovery_documents.json')
    assert documents['DISCOVERY_DOC_COUNT'] == 4
    doc_rows = []
    for i, row in enumerate(documents['documents'], 1):
        text_path = PARENT_ROOT / row['text_path']
        token_path = PARENT_ROOT / row['token_path']
        if sha256(text_path) != row['text_sha256']:
            raise RuntimeError(f'document text mismatch: {i}')
        if sha256(token_path) != row['token_sha256']:
            raise RuntimeError(f'document token mismatch: {i}')
        copied = dict(row)
        copied.update({'document_index': i, 'text_path': str(text_path),
                       'token_path': str(token_path)})
        doc_rows.append(copied)

    anchors = read_json(PARENT_ROOT / 'protocol/frozen_anchor_conditions.json')
    assert len(anchors['anchors']) == 4
    anchor_rows = []
    for row in anchors['anchors']:
        copied = dict(row)
        runtime_condition = {'Native INT8-C128': 'Native',
                             'Hadamard': 'H'}.get(copied['condition'], copied['condition'])
        export_path = copied.get('export_path')
        if export_path:
            actual = sha256(export_path)
            if actual != copied['export_file_sha256']:
                raise RuntimeError(f'anchor mismatch: {copied["condition"]}')
            copied['reverified_export_sha256'] = actual
        copied['runtime_condition'] = runtime_condition
        copied['physical_gpu'] = GPU_MAP[runtime_condition]
        anchor_rows.append(copied)

    expected_drives = []
    for doc in range(1, 5):
        base = PARENT / f'scientific/doc_{doc:02d}'
        manifest = read_json(base / 'drive_manifest.json')
        expected_drives.append({
            'document_index': doc,
            'parent_manifest_path': str(base / 'drive_manifest.json'),
            'parent_manifest_sha256': sha256(base / 'drive_manifest.json'),
            'doc_id': manifest['doc_id'],
            'token_count': manifest['token_count'],
            'shard_count': len(manifest['shards']),
            'shard_bytes': sum(x['file_size_bytes'] for x in manifest['shards']),
            'shards': [{'relative_path': x['relative_path'],
                        'sha256': x['file_sha256'],
                        'bytes': x['file_size_bytes']} for x in manifest['shards']],
            'fp_reference_checkpoints': manifest['fp_reference_checkpoints'],
            'fp_state_hashes_file_sha256': manifest['fp_state_hashes_file_sha256'],
            'parent_release_record_sha256': sha256(base / 'drive_release_record.json'),
            'parent_shards_released_by_protocol': True,
        })

    gpu_rows = gpu_inventory()
    expected_uuids = {
        0: 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422',
        1: 'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
        2: 'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
        3: 'GPU-30b07aba-a728-7649-4899-e09d33f8bc75',
    }
    if {r['index']: r['uuid'] for r in gpu_rows} != expected_uuids:
        raise RuntimeError('parent physical GPU UUID mapping is not reconstructable')

    st = os.statvfs('/data')
    available = st.f_bavail * st.f_frsize
    required = int(parent_verdict['PEAK_REQUIRED_STORAGE_BYTES'])
    if available < required:
        raise RuntimeError(f'storage capacity FAIL: {available} < {required}')

    import torch
    import transformers
    runtime = {
        'gpu_inventory': gpu_rows,
        'condition_gpu_mapping': GPU_MAP,
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'transformers': transformers.__version__,
        'triton': getattr(__import__('triton'), '__version__', 'unknown'),
        'deterministic_algorithms_for_replay': True,
        'tf32': False,
    }

    explosive = parent_explosive_units()
    prereg = {
        'EXPERIMENT': 'QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1',
        'SCIENTIFIC_TYPE': 'PROSPECTIVE_CONTROLLED_CAUSAL_DISCOVERY',
        'PARENT_EXPERIMENT': 'QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1',
        'parent_premeasurement_commit': parent_verdict['PREMEASUREMENT_COMMIT_SHA'],
        'anchor_token': ANCHOR_TOKEN,
        'endpoint': ENDPOINT,
        'first_aggregation_token': FIRST_TOKEN,
        'read_points': CHECKPOINTS,
        'conditions': CONDITIONS,
        'condition_gpu_mapping': GPU_MAP,
        'paths': {
            'A_FP_REFERENCE': 'S_A(8192)=S_FP; future F; no QDQ',
            'B_HISTORICAL_PULSE': 'S_B(8192)=physical(S_Q); future F; no QDQ/writeback',
            'C_PERSISTENT_INT8_CONTROL': 'S_C(8192)=S_Q; future Q with every-token QDQ/writeback',
        },
        'metrics': ['J_HIST_PULSE', 'J_PERSIST', 'J_QDQ_DEP',
                    'G_HIST', 'G_PERSIST', 'FINAL_ENERGY_RECOVERY_RATIO'],
        'aggregation': 'Exact parent J_ACC_FD: equal layer weight; per-layer normalized squared Frobenius energy; inclusive tokens 1024..T; FP64 accumulation; epsilon=1e-12',
        'state_identity': 'E_PERSIST = E_HIST + E_QDQ_DEP; energy includes signed 2<E_HIST,E_QDQ_DEP>',
        'explosion_threshold': EXPLOSION_THRESHOLD,
        'classification': {
            'PRESENT': '>=75% parent-persistent-explosive units history-explode AND >=3/4 Native controls do not',
            'NOT_PRESENT': '<=25% parent-persistent-explosive units history-explode',
            'MIXED': 'all other outcomes',
        },
        'layer_pattern_rule': {
            'SHARED': 'H/C6R/C7 top1 history layer identical within at least 3 of 4 documents',
            'CONDITION_SPECIFIC': 'each rotated condition has one modal top1 layer in at least 3 of 4 documents and the condition modes are not all equal',
            'DIFFUSE': 'if neither prior rule holds and median TOP5 history share across all16 units is below 0.5',
            'DOCUMENT_SPECIFIC': 'all other outcomes',
            'nonblocking': True,
        },
        'parent_persistent_unit_classification': explosive,
        'persistent_explosive_count': sum(x['PERSISTENT_EXPLOSIVE_UNIT'] for x in explosive),
        'smoke_unit': {'document_index': 1, 'condition': 'Native'},
        'determinism': {'same_process_runs': 3, 'fresh_process_runs': 3,
                        'identity_fields': ['anchor_hashes', 'J_8192', 'J_16384',
                                            'J_32768', 'per_layer_summaries']},
        'numeric_dtype': {'transition': 'parent production dtype',
                          'energy_and_reducer': 'FP64'},
        'resume_rule': 'No semantic/numerical retry. Infrastructure retry only with explicit evidence and identical unit.',
        'stop_rule': 'Any blocking gate failure => FAIL_AND_STOP.',
        'storage_lifecycle': 'Sequential per-document capture -> verify -> replay -> seal -> manifest-driven shard release',
        'alignment_causal_status': 'NOT_EVALUATED',
        'confirmation_executed': 0,
        'training_executed': False,
        'AIME_executed': False,
        'free_generation_executed': False,
        'final_action': 'STOP_AFTER_DISCOVERY',
    }

    capture_policy = json.loads(json.dumps(parent_policy))
    capture_policy['physical_gpu'] = 0
    capture_policy['minimum_runtime_free_bytes'] = 5 * 1024 ** 3
    capture_policy['documents'] = []
    for doc in doc_rows:
        i = doc['document_index']
        capture_policy['documents'].append({
            'document_index': i,
            'doc_id': doc['canonical_doc_id'],
            'text_sha256': doc['text_sha256'],
            'token_file': doc['token_path'],
            'token_file_sha256': doc['token_sha256'],
            'token_count': ENDPOINT,
            'output_directory': str(EXP / f'discovery/doc_{i:02d}/drive'),
            'required_free_bytes': required,
        })

    write_json(EXP / 'parent_artifact_identity.json', {
        'PARENT_ARTIFACT_INTEGRITY_GATE': 'PASS',
        'FINAL_SHA256SUMS_sha256': sha256(PARENT / 'FINAL_SHA256SUMS'),
        'PARALLEL_FINAL_SHA256SUMS_sha256': sha256(PARENT / 'PARALLEL_FINAL_SHA256SUMS'),
        'final_entries_verified': len(final_rows),
        'parallel_entries_verified': len(parallel_rows),
        'parent_final_verdict': parent_verdict,
    })
    write_json(EXP / 'source_identity.json', {
        'SOURCE_IDENTITY_GATE': 'PASS', 'sources': source_rows,
        'parent_replay_script_sha256': sha256(PARENT / 'scripts/persistent_anchor_replay.py'),
        'parent_replay_engine_sha256': sha256(PARENT / 'scripts/isolated_replay_engineering.py'),
        'parent_capture_script_sha256': sha256(PARENT / 'scripts/capture_extended_drive.py'),
    })
    write_json(EXP / 'provenance.json', {
        'parent_root': str(PARENT), 'parent_commit': parent_verdict['PREMEASUREMENT_COMMIT_SHA'],
        'model_checkpoint_identity_gate': 'PASS', 'model_files': model_rows,
        'production_policy_sha256': sha256(PARENT / 'protocol/extended_capture_policy.json'),
    })
    write_json(EXP / 'runtime_manifest.json', runtime)
    write_json(EXP / 'document_manifest.json', {
        'DOCUMENT_IDENTITY_GATE': 'PASS', 'documents': doc_rows,
        'source_manifest_sha256': sha256(PARENT_ROOT / 'protocol/frozen_discovery_documents.json')})
    write_json(EXP / 'anchor_manifest.json', {
        'ANCHOR_IDENTITY_GATE': 'PASS', 'anchors': anchor_rows,
        'source_manifest_sha256': sha256(PARENT_ROOT / 'protocol/frozen_anchor_conditions.json')})
    write_json(EXP / 'protocol/expected_parent_drives.json', {
        'status': 'EXPECTED_IDENTITIES_FROZEN; RECAPTURE_REQUIRED_BEFORE_REPLAY',
        'documents': expected_drives})
    write_json(EXP / 'protocol/extended_capture_policy.json', capture_policy)
    write_json(EXP / 'preregistration.json', prereg)

    md = f'''# QWEN GDN frozen-drive history-pulse causal diagnosis\n\n'''
    md += 'This is prospective controlled causal Discovery. It is not training, AIME, free generation, C8, or an inference-method proposal.\n\n'
    md += f'- Anchor token: `{ANCHOR_TOKEN}`\n- Endpoint: `{ENDPOINT}`\n- Read points: `{CHECKPOINTS}`\n'
    md += f'- Explosion threshold: `G >= {EXPLOSION_THRESHOLD:g}`\n- Units: `4 documents x 4 anchors = 16`\n'
    md += '- Primary question: does the realized 8K persistent-state error amplify under future FP recurrence with zero subsequent QDQ?\n'
    md += '- Alignment is secondary only; `ALIGNMENT_CAUSAL_STATUS=NOT_EVALUATED`.\n'
    md += '- Final action is `STOP_AFTER_DISCOVERY`.\n'
    write_text(EXP / 'preregistration.md', md)
    write_text(EXP / 'pulse_definition.md', '''# Three causal paths\n\nA starts from FP at 8192 and follows F. B starts from the physical persistent-Q state at 8192 and follows F with zero future QDQ/writeback. C starts from the same stored persistent-Q state and follows the unchanged production-equivalent condition-specific Q transition. All paths consume the exact same sealed external drive.\n''')

    gate_plan = {
        'G0_PARENT_ARTIFACT_INTEGRITY_GATE': 'PASS',
        'G1_SOURCE_IDENTITY_GATE': 'PASS',
        'G2_MODEL_CHECKPOINT_IDENTITY_GATE': 'PASS',
        'G3_DOCUMENT_IDENTITY_GATE': 'PASS',
        'G4_ANCHOR_IDENTITY_GATE': 'PASS',
        'G5_FROZEN_DRIVE_IDENTITY_GATE': 'PENDING_PER_DOCUMENT_RECAPTURE',
        'G6_TRANSITION_F_IDENTITY_GATE': 'PASS',
        'G7_TRANSITION_Q_IDENTITY_GATE': 'PASS',
        'G8_PARENT_AGGREGATOR_PARITY_GATE': 'PENDING_SMOKE',
        'G9_ANCHOR_STATE_IDENTITY_GATE': 'PENDING_SMOKE',
        'G10_FP_REFERENCE_REPLAY_PARITY_GATE': 'PENDING_SMOKE',
        'G11_PERSISTENT_CONTROL_PARITY_GATE': 'PENDING_SMOKE',
        'G12_NO_FURTHER_QDQ_PULSE_GATE': 'PENDING_SMOKE',
        'G13_REPLAY_ISOLATION_GATE': 'PENDING_SMOKE',
        'G14_REPLAY_DETERMINISM_GATE': 'PENDING_SMOKE',
        'G15_STREAMING_REDUCER_PARITY_GATE': 'PENDING_SMOKE',
        'G16_STORAGE_CAPACITY_GATE': 'PASS',
        'available_bytes': available, 'required_bytes': required,
        'headroom_bytes': available - required,
    }
    write_json(EXP / 'gate_plan.json', gate_plan)

    frozen_files = []
    for path in sorted(EXP.rglob('*')):
        if path.is_file() and path.name not in {'PREMEASUREMENT_SHA256SUMS'}:
            frozen_files.append({'path': str(path.relative_to(REPO)),
                                 'sha256': sha256(path), 'bytes': path.stat().st_size})
    lines = ''.join(f"{x['sha256']}  {x['path']}\n" for x in frozen_files)
    write_text(EXP / 'PREMEASUREMENT_SHA256SUMS', lines)
    write_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json', {
        'PREREGISTRATION_GATE': 'PASS', 'files': frozen_files,
        'measurement_count': 0, 'PREMEASUREMENT_COMMIT_SHA': None})
    print(json.dumps({'status': 'PREMEASUREMENT_READY_TO_COMMIT',
                      'available_bytes': available, 'required_bytes': required,
                      'parent_persistent_explosive_count': sum(x['PERSISTENT_EXPLOSIVE_UNIT'] for x in explosive)}, indent=2))


if __name__ == '__main__':
    main()
