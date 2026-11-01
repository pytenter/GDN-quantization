import ast
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import (ANCHOR_TOKEN, CONDITIONS, ENDPOINT, EXP,
                    EXPLOSION_THRESHOLD, FIRST_TOKEN, FOCUS_LAYERS, GPU_MAP,
                    PARENT, PARENT_HISTORY, PARENT_ROOT, READPOINTS, REPO,
                    sha256, read_json, write_json, write_text)


FORMAL = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')


def gpu_inventory():
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in raw.strip().splitlines():
        index, uuid, name, memory, driver = [x.strip() for x in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'name': name,
                     'memory_total_mib': int(memory), 'driver': driver})
    return rows


def parent_units():
    rows = list(csv.DictReader((PARENT / 'results/per_unit_horizon_metrics.csv').open(
        newline='', encoding='utf-8')))
    values = {(int(r['document_index']), r['condition'], int(r['horizon_tokens'])):
              float(r['J_ACC_FD']) for r in rows}
    result = []
    for doc in range(1, 5):
        for condition in CONDITIONS:
            j8 = values[(doc, condition, ANCHOR_TOKEN)]
            j32 = values[(doc, condition, ENDPOINT)]
            result.append({'document_index': doc, 'condition': condition,
                           'parent_J_8192': j8, 'parent_J_32768': j32,
                           'parent_G_PERSIST': j32 / j8,
                           'parent_explosive': j32 / j8 >= EXPLOSION_THRESHOLD})
    return result


def quantizer_bundle():
    source = FORMAL.read_text(encoding='utf-8')
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and
              x.name == 'quantize_c128')
    text = ast.get_source_segment(source, fn)
    assignments = {}
    for node in fn.body[0].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.get_source_segment(source, node)
    required = ['source', 'scale', 'codes', 'dequant']
    if not all(x in assignments for x in required):
        raise RuntimeError('canonical quantizer AST no longer has expected dataflow')
    if 'zero_point' in text or 'clip_threshold' in text:
        raise RuntimeError('unexpected dynamic qparam found; audit must be updated')
    return {
        'QPARAM_BUNDLE_GATE': 'PASS',
        'formal_path': str(FORMAL), 'formal_sha256': sha256(FORMAL),
        'function': 'quantize_c128',
        'dynamic_parameters_actually_used': ['scale'],
        'zero_point': 'ABSENT', 'dynamic_clip_threshold': 'ABSENT',
        'fixed_code_clamp': [-127, 127],
        'qparam_definition_source_statement': assignments['scale'],
        'implementation_rule': 'Runtime adapter compiles the original frozen AST statements for source/scale/codes/dequant; no formula is copied.',
    }


def main():
    existing = [p for p in EXP.rglob('*') if p.is_file()] if EXP.exists() else []
    if any('scripts' not in p.relative_to(EXP).parts for p in existing):
        raise RuntimeError(f'new append-only namespace is not empty: {EXP}')
    for name in ['protocol', 'provenance', 'engineering', 'qualification',
                 'discovery/unit_results', 'discovery/logs', 'analysis', 'execution']:
        (EXP / name).mkdir(parents=True, exist_ok=True)

    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                               text=True).strip() != 'fcd1dff6d286743732c6d20305cfdd31d9de7751':
        raise RuntimeError('old identifiability STOP commit is not the base')
    if not (REPO / 'experiments/QWEN_GDN_FD_PATH_QDQ_INTERACTION_V1/FAIL_AND_STOP_IDENTIFIABILITY.json').exists():
        raise RuntimeError('old identifiability STOP evidence missing')

    old_stop = read_json(REPO / 'experiments/QWEN_GDN_FD_PATH_QDQ_INTERACTION_V1/FAIL_AND_STOP_IDENTIFIABILITY.json')
    if old_stop.get('BLOCKER_CLASS') != 'PRIMARY_COUNTERFACTUAL_NONIDENTIFIABLE':
        raise RuntimeError('old STOP was altered')
    history = read_json(PARENT_HISTORY / 'analysis/historical_pulse_analysis.json')
    if history['persistent_explosive_units'] != 12 or history['history_only_explosive_units_among_parent_explosive'] != 0:
        raise RuntimeError('parent scientific outcome identity mismatch')
    parent_identity = read_json(PARENT_HISTORY / 'parent_artifact_identity.json')
    if parent_identity['PARENT_ARTIFACT_INTEGRITY_GATE'] != 'PASS':
        raise RuntimeError('parent artifact identity is not PASS')

    source_identity = read_json(PARENT_HISTORY / 'source_identity.json')
    for row in source_identity['sources']:
        if sha256(row['path']) != row['sha256']:
            raise RuntimeError(f'scientific source mismatch: {row["path"]}')
    if sha256(PARENT / 'scripts/isolated_replay_engineering.py') != source_identity['parent_replay_engine_sha256']:
        raise RuntimeError('parent replay engine mismatch')

    documents = read_json(PARENT_HISTORY / 'document_manifest.json')
    anchors = read_json(PARENT_HISTORY / 'anchor_manifest.json')
    if documents['DOCUMENT_IDENTITY_GATE'] != 'PASS' or anchors['ANCHOR_IDENTITY_GATE'] != 'PASS':
        raise RuntimeError('scientific asset identity gate failed')
    for row in documents['documents']:
        if sha256(row['text_path']) != row['text_sha256'] or sha256(row['token_path']) != row['token_sha256']:
            raise RuntimeError('document asset mismatch')
    for row in anchors['anchors']:
        if row.get('export_path') and sha256(row['export_path']) != row['export_file_sha256']:
            raise RuntimeError('rotation asset mismatch')

    bundle = quantizer_bundle()
    write_json(EXP / 'protocol/QPARAM_BUNDLE_DEFINITION.json', bundle)
    write_text(EXP / 'INTERVENTION_IDENTIFIABILITY.md', '''# Intervention identifiability\n\nBaseline P uses `theta_t^P = qparam(u_t^P)`. Treatment F uses `theta_t^clean = qparam(u_t^FP)`, while recurrence and integer-code selection consume F's own perturbed state `u_t^F`. Because the parent persistent path differs from the paired clean FP path after quantized writeback, `u_t^P` is generally not `u_t^FP`; the qparam source is therefore genuinely intervened upon. Unlike the rejected Path R, F does not share both state and qparams with P, so equality is not closed by induction.\n\n`INTERVENTION_IDENTIFIABILITY = PASS`\n''')

    parent_policy = read_json(PARENT / 'protocol/extended_capture_policy.json')
    parent_verdict = parent_identity['parent_final_verdict']
    required = int(parent_verdict['PEAK_REQUIRED_STORAGE_BYTES'])
    available = os.statvfs('/data').f_bavail * os.statvfs('/data').f_frsize
    if available < required:
        raise RuntimeError(f'STORAGE gate failed: {available} < {required}')
    policy = json.loads(json.dumps(parent_policy))
    policy['physical_gpu'] = 0
    policy['minimum_runtime_free_bytes'] = 5 * 1024 ** 3
    policy['documents'] = []
    for row in documents['documents']:
        doc = row['document_index']
        policy['documents'].append({
            'document_index': doc, 'doc_id': row['canonical_doc_id'],
            'text_sha256': row['text_sha256'], 'token_file': row['token_path'],
            'token_file_sha256': row['token_sha256'], 'token_count': ENDPOINT,
            'output_directory': str(EXP / f'discovery/doc_{doc:02d}/drive'),
            'required_free_bytes': required})
    write_json(EXP / 'protocol/extended_capture_policy.json', policy)

    units = parent_units()
    if sum(x['parent_explosive'] and x['condition'] != 'Native' for x in units) != 12:
        raise RuntimeError('parent treatment population is not frozen 12/12')
    if sum(x['parent_explosive'] and x['condition'] == 'Native' for x in units) != 0:
        raise RuntimeError('parent Native control identity mismatch')
    prereg = {
        'EXPERIMENT': 'QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1',
        'SCIENTIFIC_TYPE': 'PROSPECTIVE_CAUSAL_INTERVENTION',
        'old_failed_experiment_commit': 'fcd1dff6d286743732c6d20305cfdd31d9de7751',
        'old_fail_preserved': True,
        'parent_experiment': 'QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1',
        'anchor_token': ANCHOR_TOKEN, 'endpoint': ENDPOINT,
        'first_aggregation_token': FIRST_TOKEN, 'readpoints': READPOINTS,
        'conditions': CONDITIONS, 'condition_gpu_mapping': GPU_MAP,
        'formal_new_units': 16, 'treatment_denominator': 12,
        'parent_units': units, 'parent_persistent_explosive_count': 12,
        'intervention': 'CLEAN_QPARAM_CLAMP',
        'qparam_bundle': ['scale'],
        'clean_schedule': 'One rank-0 clean FP frozen-drive replay per document; native physical states broadcast in bounded 64-token NCCL batches; each rank maps them into its actual quantizer basis before qparam-only computation.',
        'metrics': ['J_CLEAN_QPARAM', 'G_CLEAN_QPARAM', 'state_error_norm',
                    'QDQ_residual_norm', 'qparam_deviation'],
        'focus_layers': FOCUS_LAYERS,
        'aggregation': 'Inherited exact parent J_ACC_FD: inclusive token 1024..T, equal layer weight, per-layer normalized squared Frobenius energy, FP64 accumulation, epsilon=1e-12.',
        'explosion_threshold': EXPLOSION_THRESHOLD,
        'decision': {
            '0-3': ['STATE_CONDITIONED_QPARAM_FEEDBACK_SUPPORTED', 'QPARAM_COMPONENT_LOCALIZATION'],
            '4-8': ['MIXED_QPARAM_FEEDBACK', 'STRATIFIED_QPARAM_CONFIRMATION'],
            '9-12': ['STATE_CONDITIONED_QPARAM_FEEDBACK_NOT_PRIMARY', 'STATE_TO_CODE_BIN_INTERACTION']},
        'native_validity_rule': '0/4 explosive; >=1 warning and implementation-artifact audit',
        'full_path_R_execution': 'FORBIDDEN_AS_REDUNDANT',
        'training_executed': False, 'optimizer_steps': 0,
        'new_C8_training': False, 'new_rotation_training': False,
        'AIME_executed': False, 'free_generation_executed': False,
        'final_action': 'ANALYZE_FREEZE_REPORT_STOP',
    }
    write_json(EXP / 'preregistration.json', prereg)
    write_text(EXP / 'PREREGISTRATION.md', '# Clean-qparam causal intervention\n\nThe only scientific change from frozen parent P is the source of the dynamic quantizer scale. F retains its own persistent INT8 state, every-token QDQ, and every-token low-bit writeback, but applies scale computed from the paired clean FP state in the same condition basis. Decision rules and the explosion threshold are frozen in `preregistration.json`.\n')
    write_json(EXP / 'parent_artifact_identity.json', {
        'PARENT_ARTIFACT_IDENTITY': 'PASS',
        'history_final_manifest_sha256': sha256(PARENT_HISTORY / 'sha256_manifest_final.txt'),
        'history_analysis_sha256': sha256(PARENT_HISTORY / 'analysis/historical_pulse_analysis.json'),
        'parent_identity_sha256': sha256(PARENT_HISTORY / 'parent_artifact_identity.json'),
        'parent_persistent_explosive_count': 12})
    write_json(EXP / 'scientific_asset_identity.json', {
        'SCIENTIFIC_ASSET_IDENTITY': 'PASS',
        'document_manifest_sha256': sha256(PARENT_HISTORY / 'document_manifest.json'),
        'anchor_manifest_sha256': sha256(PARENT_HISTORY / 'anchor_manifest.json'),
        'source_identity_sha256': sha256(PARENT_HISTORY / 'source_identity.json')})
    write_json(EXP / 'runtime_manifest.json', {
        'gpu_inventory': gpu_inventory(), 'condition_gpu_mapping': GPU_MAP,
        'python': sys.version, 'platform': platform.platform(),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'deterministic_algorithms': True, 'TF32': False})
    write_json(EXP / 'storage_gate.json', {
        'STORAGE': 'PASS', 'available_bytes': available,
        'required_bytes': required, 'headroom_bytes': available - required,
        'lifecycle': 'sequential capture -> replay -> seal -> release; max one full drive'})
    write_json(EXP / 'execution_authorization.json', {
        'CAPTURE_INPUT_ACQUISITION_AUTHORIZED': 'YES',
        'FORMAL_MEASUREMENT_AUTHORIZED': 'NO_PENDING_QUALIFICATION',
        'PREMEASUREMENT_COMMIT_SHA': None})
    write_json(EXP / 'scientific_execution_authorization.json', {
        'DISCOVERY_EXECUTION_AUTHORIZED': 'YES',
        'PREMEASUREMENT_COMMIT_SHA': 'fcd1dff6d286743732c6d20305cfdd31d9de7751',
        'scope': 'FROZEN_INPUT_ACQUISITION_ONLY_BEFORE_NEW_PREMEASUREMENT_FREEZE'})
    write_json(EXP / 'extended_capture_freeze.json', {
        'policy_sha256': sha256(EXP / 'protocol/extended_capture_policy.json'),
        'script_sha256': sha256(PARENT / 'scripts/capture_extended_drive.py'),
        'freeze_commit': 'fcd1dff6d286743732c6d20305cfdd31d9de7751'})
    for phase in ['factorization', 'schema', 'fp', 'invariance', 'decomposition',
                  'isolation', 'determinism', 'streaming']:
        source = PARENT / 'engineering/gates' / f'{phase}.json'
        if not source.exists() and phase == 'determinism':
            source = PARENT / 'engineering/gates/determinism_2048.json'
        value = read_json(source)
        if value['gate'] != 'PASS':
            raise RuntimeError(f'parent engineering gate not PASS: {phase}')
        write_json(EXP / f'engineering/gates/{phase}.json', {
            'gate': 'PASS', 'reused_for_frozen_input_acquisition_only': True,
            'parent_path': str(source), 'parent_sha256': sha256(source)})
    write_json(EXP / 'gate_status.json', {
        'PARENT_ARTIFACT_IDENTITY': 'PASS', 'SCIENTIFIC_ASSET_IDENTITY': 'PASS',
        'INTERVENTION_IDENTIFIABILITY': 'PASS', 'INTERVENTION_ACTIVATION': 'PENDING',
        'QPARAM_BASIS_VALIDATION': 'PENDING', 'EVERY_TOKEN_INT8_WRITEBACK': 'PENDING',
        'SHORT_REPLAY_DETERMINISM': 'PENDING', 'STORAGE': 'PASS'})
    write_json(EXP / 'execution/status.json', {'stage': 'PREPARED_CAPTURE_DOC1',
                                               'scientific_measurement_count': 0})
    print(json.dumps({'status': 'PREPARED_CAPTURE_DOC1', 'available_bytes': available,
                      'required_bytes': required, 'headroom_bytes': available-required}, indent=2))


if __name__ == '__main__':
    main()
