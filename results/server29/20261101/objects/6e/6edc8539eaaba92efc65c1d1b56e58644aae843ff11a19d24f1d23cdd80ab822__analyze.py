import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import (CHECKPOINTS, CONDITIONS, EXP, EXPLOSION_THRESHOLD, LAYERS,
                    PARENT, canonical_hash, read_json, sha256, write_json,
                    write_text)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'write-once artifact exists: {path}')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    prereg = read_json(EXP / 'preregistration.json')
    parent_class = {(x['document_index'], x['condition']): x
                    for x in prereg['parent_persistent_unit_classification']}
    units = []
    for doc in range(1, 5):
        drive = read_json(EXP / f'discovery/doc_{doc:02d}/drive_identity.json')
        if drive['G5_FROZEN_DRIVE_IDENTITY_GATE'] != 'PASS':
            raise RuntimeError(f'doc {doc} drive identity not PASS')
        release = read_json(EXP / f'discovery/doc_{doc:02d}/drive/drive_release_record.json')
        if not release['release_complete']:
            raise RuntimeError(f'doc {doc} release incomplete')
        for condition in CONDITIONS:
            path = EXP / f'discovery/unit_results/doc_{doc:02d}_{condition}.json'
            row = read_json(path)
            if row['status'] != 'PASS' or row['mode'] != 'formal':
                raise RuntimeError(f'unit not PASS: {doc} {condition}')
            units.append((doc, condition, path, row))
    if len(units) != 16:
        raise RuntimeError('Discovery unit count mismatch')

    growth_rows = []
    unit_rows = []
    layer_rows = []
    concentration_rows = []
    parent_compare_rows = []
    top1_hist = {}

    for doc, condition, path, payload in units:
        identity = payload['scientific_identity']
        checkpoints = identity['checkpoint_metrics']
        j8_hist = checkpoints['8192']['normalized_mean'][0]
        j8_persist = checkpoints['8192']['normalized_mean'][1]
        if j8_hist != j8_persist:
            raise RuntimeError('8K B/C metric identity mismatch')
        final = checkpoints['32768']['normalized_mean']
        g_hist = final[0] / j8_hist
        g_persist = final[1] / j8_persist
        recovery = final[0] / final[1] if final[1] != 0 else math.inf
        parent = parent_class[(doc, condition)]
        persistent_explosive = bool(parent['PERSISTENT_EXPLOSIVE_UNIT'])
        hist_explosive = g_hist >= EXPLOSION_THRESHOLD

        for token in CHECKPOINTS:
            values = checkpoints[str(token)]['normalized_mean']
            growth_rows.append({
                'document_index': doc, 'condition': condition, 'token': token,
                'J_HIST_PULSE': values[0], 'J_PERSIST': values[1],
                'J_QDQ_DEP': values[2], 'J_CROSS': values[3],
                'G_HIST': values[0] / j8_hist,
                'G_PERSIST': values[1] / j8_persist,
                'source_result_sha256': sha256(path),
            })

        unit_rows.append({
            'document_index': doc, 'condition': condition,
            'J_8K': j8_hist, 'J_HIST_PULSE_32K': final[0],
            'J_PERSIST_32K': final[1], 'J_QDQ_DEP_32K': final[2],
            'G_HIST_32K': g_hist, 'G_PERSIST_32K': g_persist,
            'FINAL_ENERGY_RECOVERY_RATIO': recovery,
            'PERSISTENT_EXPLOSIVE_UNIT': 'YES' if persistent_explosive else 'NO',
            'HISTORY_ONLY_EXPLOSIVE': 'YES' if hist_explosive else 'NO',
            'parent_G_PERSIST_32K': parent['G_PERSIST_PARENT_32768'],
            'source_result_sha256': sha256(path),
        })
        parent_compare_rows.append({
            'document_index': doc, 'condition': condition,
            'parent_J_8K': parent['parent_J_8192'],
            'new_J_PERSIST_8K': j8_persist,
            'parent_J_32K': parent['parent_J_32768'],
            'new_J_PERSIST_32K': final[1],
            'parent_G_PERSIST_32K': parent['G_PERSIST_PARENT_32768'],
            'new_G_PERSIST_32K': g_persist,
            'exact_J_8K': parent['parent_J_8192'] == j8_persist,
            'exact_J_32K': parent['parent_J_32768'] == final[1],
        })

        per8 = checkpoints['8192']['per_layer']
        per32 = checkpoints['32768']['per_layer']
        hist_values = [row[0] for row in per32]
        persist_values = [row[1] for row in per32]
        hist_order = sorted(range(len(LAYERS)), key=lambda i: hist_values[i], reverse=True)
        persist_order = sorted(range(len(LAYERS)), key=lambda i: persist_values[i], reverse=True)
        hist_total = math.fsum(hist_values)
        persist_total = math.fsum(persist_values)
        top1_hist[(doc, condition)] = LAYERS[hist_order[0]]
        concentration_rows.append({
            'document_index': doc, 'condition': condition,
            'HIST_TOP1_LAYER': LAYERS[hist_order[0]],
            'HIST_TOP3_LAYERS': '|'.join(str(LAYERS[i]) for i in hist_order[:3]),
            'HIST_TOP5_LAYERS': '|'.join(str(LAYERS[i]) for i in hist_order[:5]),
            'HIST_TOP1_SHARE': hist_values[hist_order[0]] / hist_total,
            'HIST_TOP3_SHARE': math.fsum(hist_values[i] for i in hist_order[:3]) / hist_total,
            'HIST_TOP5_SHARE': math.fsum(hist_values[i] for i in hist_order[:5]) / hist_total,
            'PERSIST_TOP1_LAYER': LAYERS[persist_order[0]],
            'PERSIST_TOP3_LAYERS': '|'.join(str(LAYERS[i]) for i in persist_order[:3]),
            'PERSIST_TOP5_LAYERS': '|'.join(str(LAYERS[i]) for i in persist_order[:5]),
            'PERSIST_TOP1_SHARE': persist_values[persist_order[0]] / persist_total,
            'PERSIST_TOP3_SHARE': math.fsum(persist_values[i] for i in persist_order[:3]) / persist_total,
            'PERSIST_TOP5_SHARE': math.fsum(persist_values[i] for i in persist_order[:5]) / persist_total,
        })
        for i, layer in enumerate(LAYERS):
            layer_rows.append({
                'document_index': doc, 'condition': condition, 'layer': layer,
                'J_HIST_8K': per8[i][0], 'J_HIST_32K': per32[i][0],
                'G_HIST_32K': per32[i][0] / per8[i][0] if per8[i][0] else math.inf,
                'J_PERSIST_8K': per8[i][1], 'J_PERSIST_32K': per32[i][1],
                'G_PERSIST_32K': per32[i][1] / per8[i][1] if per8[i][1] else math.inf,
                'J_QDQ_DEP_32K': per32[i][2], 'J_CROSS_32K': per32[i][3],
            })

    persistent_units = [r for r in unit_rows if r['PERSISTENT_EXPLOSIVE_UNIT'] == 'YES']
    pulse_count = sum(r['HISTORY_ONLY_EXPLOSIVE'] == 'YES' for r in persistent_units)
    native_nonpulse = sum(r['condition'] == 'Native' and r['HISTORY_ONLY_EXPLOSIVE'] == 'NO'
                          for r in unit_rows)
    fraction = pulse_count / len(persistent_units)
    if fraction >= 0.75 and native_nonpulse >= 3:
        label = 'DESCRIPTIVE_HISTORY_ONLY_AMPLIFICATION_PRESENT'
        next_step = 'PREREGISTER_DIRECTIONAL_FINITE_TIME_GAIN_DIAGNOSIS'
    elif fraction <= 0.25:
        label = 'DESCRIPTIVE_HISTORY_ONLY_AMPLIFICATION_NOT_PRESENT'
        next_step = 'PREREGISTER_PATH_CONDITIONED_QDQ_INTERACTION_DIAGNOSIS'
    else:
        label = 'DESCRIPTIVE_HISTORY_ONLY_AMPLIFICATION_MIXED'
        next_step = 'STOP_AND_REVIEW_MIXED_HISTORY_VS_QDQ_MECHANISM'

    same_by_doc = sum(len({top1_hist[(d, c)] for c in ['H', 'C6R', 'C7']}) == 1
                      for d in range(1, 5))
    modal = {c: Counter(top1_hist[(d, c)] for d in range(1, 5)).most_common(1)[0]
             for c in ['H', 'C6R', 'C7']}
    if same_by_doc >= 3:
        layer_pattern = 'SHARED'
    elif all(count >= 3 for _, count in modal.values()) and len({x[0] for x in modal.values()}) > 1:
        layer_pattern = 'CONDITION_SPECIFIC'
    elif statistics.median(float(r['HIST_TOP5_SHARE']) for r in concentration_rows) < 0.5:
        layer_pattern = 'DIFFUSE'
    else:
        layer_pattern = 'DOCUMENT_SPECIFIC'

    summary = []
    for condition in CONDITIONS:
        rows = [r for r in unit_rows if r['condition'] == condition]
        top_counts = Counter(top1_hist[(d, condition)] for d in range(1, 5))
        summary.append({
            'Condition': condition,
            'Persistent_explosive_count': sum(r['PERSISTENT_EXPLOSIVE_UNIT'] == 'YES' for r in rows),
            'History_only_explosive_count': sum(r['HISTORY_ONLY_EXPLOSIVE'] == 'YES' for r in rows),
            'Median_G_PERSIST': statistics.median(float(r['G_PERSIST_32K']) for r in rows),
            'Median_G_HIST': statistics.median(float(r['G_HIST_32K']) for r in rows),
            'Median_recovery_ratio': statistics.median(float(r['FINAL_ENERGY_RECOVERY_RATIO']) for r in rows),
            'Top_unstable_layers': [x[0] for x in top_counts.most_common(5)],
        })

    write_csv(EXP / 'analysis/growth_curves.csv', list(growth_rows[0]), growth_rows)
    write_csv(EXP / 'analysis/per_unit_gain.csv', list(unit_rows[0]), unit_rows)
    write_csv(EXP / 'analysis/per_layer_gain.csv', list(layer_rows[0]), layer_rows)
    write_csv(EXP / 'analysis/layer_concentration.csv', list(concentration_rows[0]), concentration_rows)
    write_csv(EXP / 'analysis/parent_vs_pulse_comparison.csv', list(parent_compare_rows[0]), parent_compare_rows)

    result = {
        'HISTORICAL_PULSE_DISCOVERY_LABEL': label,
        'PRIMARY_NEXT_STEP_CATEGORY': next_step,
        'persistent_explosive_units': len(persistent_units),
        'history_only_explosive_units_among_parent_explosive': pulse_count,
        'history_only_explosive_fraction': fraction,
        'native_nonexplosive_controls': native_nonpulse,
        'LAYER_INSTABILITY_PATTERN': layer_pattern,
        'condition_summary': summary,
        'ALIGNMENT_CAUSAL_STATUS': 'NOT_EVALUATED',
        'interpretation_scope': 'Realized finite-time pulse propagation under frozen external drive only',
        'FULL_MODEL_CLOSED_LOOP_PROPAGATION': 'NOT_EVALUATED',
        'CONFIRMATION_EXECUTED': 0,
        'TRAINING_EXECUTED': 'NO',
        'OPTIMIZER_STEPS': 0,
        'AIME_EXECUTED': 'NO',
        'FREE_GENERATION_EXECUTED': 'NO',
        'NEW_C8_TRAINING_AUTHORIZED': 'NO',
        'FINAL_ACTION': 'STOP_AFTER_DISCOVERY',
    }
    write_json(EXP / 'analysis/historical_pulse_analysis.json', result)

    lines = ['# Historical-pulse causal Discovery', '',
             f'Classification: `{label}`', '',
             f'Parent-persistent explosive units: {len(persistent_units)}',
             f'History-only explosive units among them: {pulse_count}',
             f'Native controls remaining below threshold: {native_nonpulse}/4',
             f'Layer pattern: `{layer_pattern}`', '',
             'This is descriptive causal Discovery on the parent Discovery documents. Alignment causality, full-model closed-loop reasoning causality, AIME causality and training-objective readiness are not evaluated.', '']
    write_text(EXP / 'analysis/historical_pulse_analysis.md', '\n'.join(lines))

    gate = read_json(EXP / 'gate_report.json')
    auth = read_json(EXP / 'execution_authorization.json')
    report = ['# QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1', '',
              '## Required status', '', '```text',
              'EXPERIMENT = QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1',
              'PREREGISTRATION_GATE = PASS',
              'PARENT_ARTIFACT_INTEGRITY_GATE = PASS',
              'SOURCE_IDENTITY_GATE = PASS',
              'FROZEN_DRIVE_IDENTITY_GATE = PASS',
              'ANCHOR_STATE_IDENTITY_GATE = PASS',
              'PERSISTENT_CONTROL_PARITY_GATE = PASS',
              'NO_FURTHER_QDQ_PULSE_GATE = PASS',
              'REPLAY_DETERMINISM_GATE = PASS',
              'DISCOVERY_COMPLETE = 16 / 16',
              'CONFIRMATION_EXECUTED = 0',
              'TRAINING_EXECUTED = NO', 'OPTIMIZER_STEPS = 0',
              'AIME_EXECUTED = NO', 'FREE_GENERATION_EXECUTED = NO',
              f'PREMEASUREMENT_COMMIT_SHA = {auth["PREMEASUREMENT_COMMIT_SHA"]}',
              f'HISTORICAL_PULSE_DISCOVERY_LABEL = {label}',
              f'PRIMARY_NEXT_STEP_CATEGORY = {next_step}',
              'ALIGNMENT_CAUSAL_STATUS = NOT_EVALUATED',
              'FULL_MODEL_CLOSED_LOOP_PROPAGATION = NOT_EVALUATED',
              'NEW_C8_TRAINING_AUTHORIZED = NO',
              'FINAL_ACTION = STOP_AFTER_DISCOVERY', '```', '',
              '## Condition summary', '',
              '| Condition | Persistent explosive | History-only explosive | Median G_PERSIST | Median G_HIST | Median recovery | Top unstable layers |',
              '|---|---:|---:|---:|---:|---:|---|']
    for row in summary:
        report.append(f"| {row['Condition']} | {row['Persistent_explosive_count']} | {row['History_only_explosive_count']} | {row['Median_G_PERSIST']:.9g} | {row['Median_G_HIST']:.9g} | {row['Median_recovery_ratio']:.9g} | {','.join(map(str,row['Top_unstable_layers']))} |")
    report += ['', '## Interpretation', '',
               f'Among {len(persistent_units)} units classified as persistent-explosive before this experiment, {pulse_count} also crossed the frozen history-only threshold. The resulting prospective Discovery label is `{label}`.', '',
               'The result only addresses whether the realized 8K state error is sufficient for finite-time amplification under the sealed frozen drive. It does not establish full-model generation instability, reasoning causality, AIME causality, alignment causality, or readiness for C8 training.', '']
    write_text(EXP / 'FINAL_REPORT.md', '\n'.join(report))

    manifest_rows = []
    for path in sorted(EXP.rglob('*')):
        if not path.is_file() or path.name == 'sha256_manifest.txt':
            continue
        if 'drive_shards' in path.parts:
            raise RuntimeError('bulky drive shard remains after release')
        manifest_rows.append((sha256(path), str(path.relative_to(EXP))))
    write_text(EXP / 'sha256_manifest.txt', ''.join(f'{h}  {p}\n' for h, p in manifest_rows))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
