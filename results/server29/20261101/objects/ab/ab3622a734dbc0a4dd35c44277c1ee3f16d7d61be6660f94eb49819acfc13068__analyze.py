import csv
import json
import math
import statistics
from pathlib import Path

from common import (CONDITIONS, EXP, EXPLOSION_THRESHOLD, FOCUS_LAYERS,
                    PARENT, canonical_hash, read_json, sha256, write_json,
                    write_text)


def main():
    prereg = read_json(EXP / 'preregistration.json')
    parent = {(x['document_index'], x['condition']): x for x in prereg['parent_units']}
    rows = []
    focus_rows = []
    for doc in range(1,5):
        for condition in CONDITIONS:
            path = EXP / f'discovery/unit_results/doc_{doc:02d}_{condition}.json'
            value = read_json(path)
            if value['status'] != 'PASS' or value['end_token'] != 32768:
                raise RuntimeError(f'incomplete formal unit: {path}')
            if value['qdq_calls_after_anchor'] != 24576 or value['int8_writeback_calls_after_anchor'] != 24576:
                raise RuntimeError(f'QDQ/writeback invariant failed: {path}')
            old = parent[(doc,condition)]
            rows.append({
                'document_index': doc, 'condition': condition,
                'parent_J_8192': old['parent_J_8192'],
                'parent_J_32768': old['parent_J_32768'],
                'parent_G_PERSIST': old['parent_G_PERSIST'],
                'parent_explosive': old['parent_explosive'],
                'clean_J_32768': value['J_CLEAN_QPARAM'],
                'clean_G': value['G_CLEAN_QPARAM'],
                'clean_explosive': value['explosive'],
                'result_sha256': sha256(path)})
            for token, snap in value['readpoint_diagnostics'].items():
                for layer in FOCUS_LAYERS:
                    metric = snap['focus_layers'][str(layer)]
                    focus_rows.append({'document_index': doc, 'condition': condition,
                                       'token': int(token), 'layer': layer, **metric})

    treatment = [r for r in rows if r['condition'] != 'Native']
    clean_count = sum(r['clean_explosive'] for r in treatment)
    native_count = sum(r['clean_explosive'] for r in rows if r['condition'] == 'Native')
    if clean_count <= 3:
        label = 'STATE_CONDITIONED_QPARAM_FEEDBACK_SUPPORTED'
        next_step = 'QPARAM_COMPONENT_LOCALIZATION'
    elif clean_count >= 9:
        label = 'STATE_CONDITIONED_QPARAM_FEEDBACK_NOT_PRIMARY'
        next_step = 'STATE_TO_CODE_BIN_INTERACTION'
    else:
        label = 'MIXED_QPARAM_FEEDBACK'
        next_step = 'STRATIFIED_QPARAM_CONFIRMATION'
    validity = 'PASS' if native_count == 0 else 'COUNTERFACTUAL_VALIDITY_WARNING'

    summaries = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition'] == condition]
        summaries[condition] = {
            'parent_median_G_PERSIST': statistics.median(r['parent_G_PERSIST'] for r in subset),
            'clean_qparam_median_G': statistics.median(r['clean_G'] for r in subset),
            'clean_explosive_count': sum(r['clean_explosive'] for r in subset),
            'unit_count': len(subset)}

    analysis = {
        'status': 'PASS', 'formal_units': 16,
        'PARENT_PERSISTENT_EXPLOSIVE_COUNT': '12/12',
        'CLEAN_QPARAM_EXPLOSIVE_COUNT': f'{clean_count}/12',
        'NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT': f'{native_count}/4',
        'COUNTERFACTUAL_VALIDITY_GATE': validity,
        'PRIMARY_LABEL': label, 'PRIMARY_NEXT_STEP': next_step,
        'condition_summaries': summaries, 'per_unit': rows,
        'interpretation_boundary': 'Frozen-drive persistent-state mechanism only; no AIME, free-generation, full closed-loop, or endpoint-causality claim.'}
    write_json(EXP / 'analysis/final_analysis.json', analysis)

    with (EXP/'analysis/per_unit_parent_vs_treatment.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (EXP/'analysis/focus_layer_trajectories.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(focus_rows[0]))
        w.writeheader(); w.writerows(focus_rows)

    lines = ['# QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1', '', '```text',
             'PARENT_PERSISTENT_EXPLOSIVE_COUNT = 12/12',
             f'CLEAN_QPARAM_EXPLOSIVE_COUNT = {clean_count}/12',
             f'NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT = {native_count}/4',
             f'PRIMARY_LABEL = {label}', f'PRIMARY_NEXT_STEP = {next_step}',
             f'COUNTERFACTUAL_VALIDITY_GATE = {validity}',
             'TRAINING_EXECUTED = NO', 'OPTIMIZER_STEPS = 0',
             'NEW_C8_TRAINING = NO', 'NEW_ROTATION_TRAINING = NO',
             'AIME_EXECUTED = NO', 'FREE_GENERATION_EXECUTED = NO',
             'FINAL_ACTION = ANALYZE_FREEZE_REPORT_STOP', '```', '',
             '## Condition summary', '',
             '| Condition | Parent median G | Clean-qparam median G | Clean explosive |',
             '|---|---:|---:|---:|']
    for condition in CONDITIONS:
        s = summaries[condition]
        lines.append(f'| {condition} | {s["parent_median_G_PERSIST"]:.9g} | {s["clean_qparam_median_G"]:.9g} | {s["clean_explosive_count"]}/4 |')
    lines += ['', '## Interpretation boundary', '',
              analysis['interpretation_boundary'], '',
              'Layer 6 and layer 28 trajectories are in `analysis/focus_layer_trajectories.csv`; all-layer readpoint summaries remain in each sealed unit JSON.']
    write_text(EXP/'FINAL_REPORT.md', '\n'.join(lines)+'\n')

    write_json(EXP/'FINAL_STATUS.json', {
        **{k: analysis[k] for k in ['PARENT_PERSISTENT_EXPLOSIVE_COUNT',
                                    'CLEAN_QPARAM_EXPLOSIVE_COUNT',
                                    'NATIVE_CLEAN_QPARAM_EXPLOSIVE_COUNT',
                                    'PRIMARY_LABEL','PRIMARY_NEXT_STEP']},
        'DISCOVERY_COMPLETE': '16/16', 'FINAL_ACTION': 'STOP'})
    files = []
    for path in sorted(EXP.rglob('*')):
        if path.is_file() and path.name not in {'SHA256SUMS'} and 'drive_shards' not in path.parts:
            files.append({'path': str(path.relative_to(EXP)), 'sha256': sha256(path),
                          'bytes': path.stat().st_size})
    write_text(EXP/'SHA256SUMS', ''.join(f'{x["sha256"]}  {x["path"]}\n' for x in files))
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
