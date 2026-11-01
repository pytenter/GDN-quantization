import ast
import csv
import json
import os
import platform
import subprocess
import sys

from common import (ANCHOR_TOKEN, ARMS, CLEAN_EXP, CONDITIONS, ENDPOINT, EXP,
                    EXPLOSION_THRESHOLD, FOCUS_LAYERS, FORMAL, GPU_MAP,
                    HIST_BINS, HIST_MAX, HIST_MIN, PARENT, PARENT_HISTORY,
                    READPOINTS, REPO, clean_result, read_json, sha256,
                    write_json, write_text)


BASE_COMMIT = 'dc30294816588e80e4f26727f62afbec47ff7d12'
REQUIRED_STORAGE_BYTES = 27895529472


def assignment_sources():
    source = FORMAL.read_text(encoding='utf-8')
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and
              x.name == 'quantize_c128')
    loop = fn.body[0]
    rows = {}
    for node in loop.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            rows[node.targets[0].id] = ast.get_source_segment(source, node)
    required = ['source', 'scale', 'codes', 'dequant']
    if not all(x in rows for x in required):
        raise RuntimeError('canonical quantizer AST dataflow changed')
    eps_node = next(x for x in tree.body if isinstance(x, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'EPS' for t in x.targets))
    eps_text = ast.get_source_segment(source, eps_node)
    return source, ast.get_source_segment(source, fn), rows, eps_text


def audit_quantizer():
    source, fn_text, rows, eps_text = assignment_sources()
    if 'zero_point' in fn_text or 'clip_threshold' in fn_text:
        raise RuntimeError('independent dynamic qparam discovered')
    expected = {
        'source': 'source = state.detach().float()',
        'scale': 'scale = source.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0',
        'codes': 'codes = torch.round(source / scale).clamp(-127, 127)',
        'dequant': 'dequant = codes * scale'}
    if {key: rows[key] for key in expected} != expected:
        raise RuntimeError(f'quantizer exact semantics differ from frozen assumption: {rows}')
    if not eps_text.startswith('EPS ='):
        raise RuntimeError('EPS definition unavailable')
    eps_value = float(eps_text.split('=', 1)[1].strip())
    if eps_value <= 0:
        raise RuntimeError('invalid EPS floor')
    return {
        'QUANTIZER_SEMANTIC_AUDIT': 'PASS',
        'QPARAM_COMPONENT_LOCALIZATION': 'ALREADY_COLLAPSED_TO_SCALE',
        'formal_path': str(FORMAL), 'formal_sha256': sha256(FORMAL),
        'function': 'quantize_c128',
        'dynamic_qparam_count': 1, 'dynamic_qparam_names': ['scale'],
        'amax_role': 'DIRECT_INTERMEDIATE_OF_SCALE_NOT_AN_INDEPENDENT_QPARAM',
        'state_shape': ['batch', 32, 128, 128],
        'source_dtype': 'torch.float32_from_state.detach()',
        'scale_formula': rows['scale'],
        'scale_reduction_dimensions': [-2],
        'keepdim': True,
        'grouping_orientation': 'one scale per [batch, head32, column128], reducing the row128 axis',
        'scale_shape_for_batch_1': [1, 32, 1, 128],
        'epsilon_floor_symbol': 'EPS', 'epsilon_floor_value': eps_value,
        'rounding_rule': 'torch.round (nearest; ties-to-even as defined by PyTorch)',
        'integer_range': [-127, 127], 'zero_point': 0,
        'clamping_rule': 'torch.clamp(-127, 127) after rounding source/scale',
        'writeback_semantics': 'codes*scale is cast to original state dtype and copied in-place; persistent semantics are every-token INT8 QDQ with dequantized values stored in FP32 carrier',
        'independent_dynamic_clipping': False,
        'independent_dynamic_zero_point': False,
        'independent_dynamic_range': False,
        'source_statements': rows,
    }


def parse_manifest(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rows[rel.strip()] = digest
    return rows


def clean_identity():
    status = read_json(CLEAN_EXP / 'FINAL_STATUS.json')
    expected = {
        'DISCOVERY_COMPLETE': '16/16',
        'PARENT_PERSISTENT_EXPLOSIVE_COUNT': '12/12',
        'CLEAN_QPARAM_EXPLOSIVE_COUNT': '0/12',
        'NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT': '0/4',
        'PRIMARY_LABEL': 'STATE_CONDITIONED_QPARAM_FEEDBACK_SUPPORTED'}
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError(f'clean parent status mismatch: {key}')
    manifest_path = CLEAN_EXP / 'FINAL_SHA256SUMS'
    manifest = parse_manifest(manifest_path)
    schedules = []
    for doc in range(1, 5):
        for condition in CONDITIONS:
            rel = f'discovery/unit_results/doc_{doc:02d}_{condition}.json'
            path = CLEAN_EXP / rel
            if manifest.get(rel) != sha256(path):
                raise RuntimeError(f'clean unit integrity mismatch: {rel}')
            value = clean_result(doc, condition)
            schedules.append({
                'document_index': doc, 'condition': condition,
                'unit_relative_path': rel, 'unit_sha256': sha256(path),
                'qparam_schedule_sha256': value['qparam_schedule_sha256'],
                'G_CLEAN_QPARAM': value['G_CLEAN_QPARAM'],
                'explosive': value['explosive']})
    return {
        'CLEAN_PARENT_ARTIFACT_IDENTITY': 'PASS',
        'final_status': status,
        'final_manifest_path': str(manifest_path),
        'final_manifest_sha256': sha256(manifest_path),
        'schedule_materialization_retained': False,
        'identity_strategy': 'Rematerialize with the frozen exact AST/source/order; require full per-unit schedule SHA256 equality before accepting every new formal unit.',
        'frozen_schedules': schedules}


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
                           'parent_G': j32/j8,
                           'parent_explosive': j32/j8 >= EXPLOSION_THRESHOLD,
                           'clean_G': clean_result(doc, condition)['G_CLEAN_QPARAM']})
    return result


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    existing = [p for p in EXP.rglob('*') if p.is_file()]
    if any('scripts' not in p.relative_to(EXP).parts for p in existing):
        raise RuntimeError(f'new experiment namespace is not empty: {EXP}')
    for name in ['protocol', 'provenance', 'engineering', 'qualification',
                 'discovery/unit_results', 'discovery/logs', 'analysis',
                 'execution', 'reports']:
        (EXP / name).mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                   text=True).strip()
    if head != BASE_COMMIT:
        raise RuntimeError(f'independent checkout base mismatch: {head}')

    audit = audit_quantizer()
    write_json(EXP / 'QUANTIZER_SEMANTIC_AUDIT.json', audit)
    clean = clean_identity()
    write_json(EXP / 'clean_scale_schedule_identity.json', clean)
    history = read_json(PARENT_HISTORY / 'analysis/historical_pulse_analysis.json')
    if history['persistent_explosive_units'] != 12 or history['history_only_explosive_units_among_parent_explosive'] != 0:
        raise RuntimeError('history parent frozen evidence mismatch')
    parent_identity = read_json(PARENT_HISTORY / 'parent_artifact_identity.json')
    if parent_identity['PARENT_ARTIFACT_INTEGRITY_GATE'] != 'PASS':
        raise RuntimeError('parent artifact integrity failed')
    units = parent_units()
    if sum(x['parent_explosive'] and x['condition'] != 'Native' for x in units) != 12:
        raise RuntimeError('frozen parent treatment population mismatch')
    if sum(x['parent_explosive'] and x['condition'] == 'Native' for x in units) != 0:
        raise RuntimeError('frozen parent Native population mismatch')

    available = os.statvfs('/data').f_bavail * os.statvfs('/data').f_frsize
    if available < REQUIRED_STORAGE_BYTES:
        raise RuntimeError(f'STORAGE_CAPACITY gate failed: {available} < {REQUIRED_STORAGE_BYTES}')
    write_json(EXP / 'storage_capacity.json', {
        'G9_STORAGE_CAPACITY': 'PASS', 'available_bytes': available,
        'required_bytes': REQUIRED_STORAGE_BYTES,
        'headroom_bytes': available-REQUIRED_STORAGE_BYTES,
        'lifecycle': 'one document drive at a time: capture -> verify -> replay two arms -> seal -> release'})

    write_text(EXP / 'INTERVENTION_IDENTIFIABILITY.md', '''# Intervention identifiability\n\nAt every token, layer, group, and condition the frozen adaptive baseline uses `s_adapt`. `NO_SCALE_INFLATION` uses `min(s_adapt, s_clean)` and therefore differs exactly on positive deviations. `NO_SCALE_DEFLATION` uses `max(s_adapt, s_clean)` and therefore differs exactly on negative deviations. Whenever `s_adapt != s_clean`, one and only one of the two arms clamps that signed deviation to the paired clean value. Both arms preserve their own recurrent states and their own future adaptive scales, so neither trajectory is definitionally identical to the baseline.\n\n`G3_INTERVENTION_IDENTIFIABILITY = PASS`\n''')
    write_json(EXP / 'protocol/measurement_definitions.json', {
        'arms': {
            'NO_SCALE_INFLATION': 'min(s_adapt, s_clean)',
            'NO_SCALE_DEFLATION': 'max(s_adapt, s_clean)'},
        'signed_log_scale_ratio': 'log2(s_adapt/s_clean)',
        'histogram_semantics': {
            'range': [HIST_MIN, HIST_MAX], 'bins': HIST_BINS,
            'bin_width': (HIST_MAX-HIST_MIN)/HIST_BINS,
            'quantiles': ['median','p90','p99'],
            'tail_values': 'exact max positive and exact min negative are also retained'},
        'token_bins': [[ANCHOR_TOKEN+1, x] if i == 0 else [READPOINTS[i-1]+1, x]
                       for i, x in enumerate(READPOINTS)],
        'saturation': 'count/fraction of pre-QDQ source values with source/intervention_scale outside [-127,127] before rounding/clamp',
        'normalized_QDQ_residual': 'L2(post_QDQ-pre_QDQ)/max(L2(pre_QDQ),1e-12)',
        'state_error_norm': 'L2(intervention physical state - paired clean FP physical state)',
        'recovery_fraction_vs_parent': '(G_parent-G_arm)/max(G_parent-G_clean,1e-12)',
        'temporal_onset': 'first frozen readpoint whose value is strictly greater than 2x its 12288 reference; clamp activation onset is the first readpoint with cumulative clamp_count>0',
        'temporal_precedence_boundary': 'descriptive only; causal evidence is supplied by interventions'})

    documents = read_json(PARENT_HISTORY / 'document_manifest.json')
    parent_policy = read_json(PARENT / 'protocol/extended_capture_policy.json')
    policy = json.loads(json.dumps(parent_policy))
    policy['physical_gpu'] = 0
    policy['minimum_runtime_free_bytes'] = 5 * 1024**3
    policy['documents'] = []
    for row in documents['documents']:
        doc = row['document_index']
        policy['documents'].append({
            'document_index': doc, 'doc_id': row['canonical_doc_id'],
            'text_sha256': row['text_sha256'], 'token_file': row['token_path'],
            'token_file_sha256': row['token_sha256'], 'token_count': ENDPOINT,
            'output_directory': str(EXP / f'discovery/doc_{doc:02d}/drive'),
            'required_free_bytes': REQUIRED_STORAGE_BYTES})
    write_json(EXP / 'protocol/extended_capture_policy.json', policy)
    prereg = {
        'EXPERIMENT': 'QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1',
        'SCIENTIFIC_TYPE': 'PROSPECTIVE_CAUSAL_DIRECTION_LOCALIZATION',
        'base_commit': BASE_COMMIT, 'parent_artifacts_read_only': True,
        'parent_persistent_explosive_count': '12/12',
        'clean_scale_explosive_count': '0/12',
        'conditions': CONDITIONS, 'arms': ARMS, 'formal_units': 32,
        'condition_gpu_mapping': GPU_MAP, 'anchor': ANCHOR_TOKEN,
        'endpoint': ENDPOINT, 'readpoints': READPOINTS,
        'focus_layers': FOCUS_LAYERS, 'explosion_threshold': EXPLOSION_THRESHOLD,
        'parent_units': units,
        'decision_matrix': {
            'A': {'if': 'NSI<=3 and NSD>=9', 'label': 'SCALE_INFLATION_FEEDBACK_SUPPORTED', 'next': 'SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION'},
            'B': {'if': 'NSD<=3 and NSI>=9', 'label': 'SCALE_DEFLATION_FEEDBACK_SUPPORTED', 'next': 'SHORT_HORIZON_SCALE_DEFLATION_SURROGATE_VALIDATION'},
            'C': {'if': 'NSI<=3 and NSD<=3', 'label': 'BOTH_SIDED_SCALE_DEVIATION_DEPENDENCE', 'next': 'SIGNED_SCALE_DEVIATION_SURROGATE_VALIDATION'},
            'D': {'if': 'NSI>=9 and NSD>=9', 'label': 'ONE_SIDED_SCALE_CLAMPS_INSUFFICIENT', 'next': 'TEMPORAL_SCALE_TRAJECTORY_INTERACTION_DIAGNOSIS'},
            'E': {'if': 'otherwise', 'label': 'MIXED_SCALE_DIRECTION_EFFECT', 'next': 'STRATIFIED_SCALE_DIRECTION_CONFIRMATION'}},
        'rescue_bands': {'RESCUE': [0,3], 'MIXED': [4,8], 'NO_RESCUE': [9,12]},
        'persistent_semantics': 'every-token INT8 QDQ and writeback; dequantized values stored in FP32 carrier',
        'training_executed': False, 'optimizer_steps': 0,
        'new_C8_training': False, 'new_rotation_training': False,
        'AIME_executed': False, 'free_generation_executed': False,
        'final_action': 'ANALYZE_VALIDATE_FREEZE_REPORT_STOP'}
    write_json(EXP / 'preregistration.json', prereg)
    write_text(EXP / 'PREREGISTRATION.md', '# Scale feedback direction localization\n\nThe frozen two-arm experiment removes positive and negative adaptive-scale deviations separately. All population, horizons, metrics, threshold, runtime and decision rules are fixed in `preregistration.json` before formal measurement.\n')
    write_json(EXP / 'parent_artifact_identity.json', {
        'G0_PARENT_ARTIFACT_IDENTITY': 'PASS',
        'history_identity_sha256': sha256(PARENT_HISTORY/'parent_artifact_identity.json'),
        'clean_final_manifest_sha256': clean['final_manifest_sha256'],
        'clean_premeasurement_commit': BASE_COMMIT})
    write_json(EXP / 'runtime_manifest.json', {
        'python': sys.version, 'platform': platform.platform(),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'deterministic_algorithms': True, 'TF32': False,
        'condition_gpu_mapping': GPU_MAP})
    gates = {
        'G0_PARENT_ARTIFACT_IDENTITY': 'PASS',
        'G1_CLEAN_SCALE_SCHEDULE_IDENTITY': 'PENDING_FULL_DOC1_REPLAY',
        'G2_QUANTIZER_SEMANTIC_AUDIT': 'PASS',
        'G3_INTERVENTION_IDENTIFIABILITY': 'PASS',
        'G4_ACTIVATION_GATE': 'PENDING', 'G5_EVERY_TOKEN_QDQ_WRITEBACK': 'PENDING',
        'G6_BASIS_SEMANTICS': 'PENDING', 'G7_SHORT_REPLAY_DETERMINISM': 'PENDING',
        'G8_STREAMING_REDUCER_PARITY': 'PENDING', 'G9_STORAGE_CAPACITY': 'PASS'}
    write_json(EXP / 'gate_status.json', gates)
    write_json(EXP / 'execution_authorization.json', {
        'CAPTURE_INPUT_ACQUISITION_AUTHORIZED': 'YES',
        'QUALIFICATION_AUTHORIZED': 'YES',
        'FORMAL_MEASUREMENT_AUTHORIZED': 'NO_PENDING_GATES',
        'PREMEASUREMENT_COMMIT_SHA': None})
    write_json(EXP / 'execution/status.json', {
        'stage': 'PREPARED_DOC1_CAPTURE', 'formal_measurement_count': 0})
    print(json.dumps({'status': 'PREPARED_DOC1_CAPTURE', 'audit': 'PASS',
                      'available_bytes': available,
                      'headroom_bytes': available-REQUIRED_STORAGE_BYTES}, indent=2))


if __name__ == '__main__':
    main()
