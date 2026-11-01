import csv
import json
import math
import statistics

from common import (ARMS, CLEAN_EXP, CONDITIONS, EXP, EXPLOSION_THRESHOLD,
                    FOCUS_LAYERS, READPOINTS, clean_result, read_json,
                    write_json, write_text)


def onset(values):
    if not values:
        return 'NOT_AVAILABLE'
    base = values[0][1]
    for token, value in values[1:]:
        if value > 2*base:
            return token
    return 'NOT_REACHED'


def main():
    prereg = read_json(EXP/'preregistration.json')
    parent_map = {(x['document_index'],x['condition']):x for x in prereg['parent_units']}
    rows, signed_rows, focus_rows = [], [], []
    counts = {arm: 0 for arm in ARMS}
    native_counts = {arm: 0 for arm in ARMS}
    temporal = []
    for doc in range(1,5):
        for condition in CONDITIONS:
            parent = parent_map[(doc,condition)]
            clean_g = clean_result(doc,condition)['G_CLEAN_QPARAM']
            for arm in ARMS:
                value = read_json(EXP/f'discovery/unit_results/doc_{doc:02d}_{condition}_{arm}.json')
                if value['status'] != 'PASS' or value['end_token'] != 32768:
                    raise RuntimeError('incomplete formal result')
                if not value['paired_clean_scale_schedule']['byte_identity']:
                    raise RuntimeError('formal clean scale schedule identity missing')
                if value['qdq_calls_after_anchor'] != 24576 or value['int8_writeback_calls_after_anchor'] != 24576:
                    raise RuntimeError('formal every-token QDQ/writeback mismatch')
                g = value['G_INTERVENTION']
                explosive = g >= EXPLOSION_THRESHOLD
                if condition == 'Native':
                    native_counts[arm] += int(explosive)
                else:
                    counts[arm] += int(explosive)
                recovery = (parent['parent_G']-g)/max(parent['parent_G']-clean_g,1e-12)
                rows.append({
                    'document_index': doc, 'condition': condition, 'arm': arm,
                    'G_parent': parent['parent_G'], 'G_clean_scale': clean_g,
                    'G_intervention': g, 'explosive': explosive,
                    'recovery_fraction_vs_parent': recovery})
                signed_rows.extend(dict(x,document_index=doc,condition=condition)
                                   for x in value['signed_scale_streaming_summary'])
                for token in READPOINTS:
                    diag = value['readpoint_diagnostics'][str(token)]['focus_layers']
                    for layer in FOCUS_LAYERS:
                        focus_rows.append(dict(diag[str(layer)], document_index=doc,
                                               condition=condition, arm=arm,
                                               token=token, layer=layer))
                for layer in FOCUS_LAYERS:
                    scale_series=[]
                    clamp_series=[]
                    for x in value['signed_scale_streaming_summary']:
                        if x['arm']==arm and x['layer']==layer:
                            scale_series.append((x['token_bin_end'],abs(x['median_signed_log_scale_ratio'])))
                            clamp_series.append((x['token_bin_end'],x['clamp_count']))
                    scale_series.sort(); clamp_series.sort()
                    residual=[(t,value['readpoint_diagnostics'][str(t)]['focus_layers'][str(layer)]['normalized_QDQ_residual']) for t in READPOINTS]
                    state=[(t,value['readpoint_diagnostics'][str(t)]['focus_layers'][str(layer)]['state_error_norm']) for t in READPOINTS]
                    temporal.append({
                        'document_index':doc,'condition':condition,'arm':arm,'layer':layer,
                        'scale_deviation_onset':next((t for t,c in clamp_series if c>0),'NOT_REACHED'),
                        'QDQ_residual_growth_onset':onset(residual),
                        'state_error_growth_onset':onset(state),
                        'interpretation':'DESCRIPTIVE_TEMPORAL_ORDERING_ONLY'})

    nsi = counts['NO_SCALE_INFLATION']; nsd = counts['NO_SCALE_DEFLATION']
    if native_counts['NO_SCALE_INFLATION'] or native_counts['NO_SCALE_DEFLATION']:
        validity='COUNTERFACTUAL_VALIDITY_REVIEW'
        label='COUNTERFACTUAL_VALIDITY_REVIEW_REQUIRED'
        next_step='IMPLEMENTATION_ARTIFACT_AUDIT'
    else:
        validity='PASS'
        if nsi <= 3 and nsd >= 9:
            label='SCALE_INFLATION_FEEDBACK_SUPPORTED'; next_step='SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION'
        elif nsd <= 3 and nsi >= 9:
            label='SCALE_DEFLATION_FEEDBACK_SUPPORTED'; next_step='SHORT_HORIZON_SCALE_DEFLATION_SURROGATE_VALIDATION'
        elif nsi <= 3 and nsd <= 3:
            label='BOTH_SIDED_SCALE_DEVIATION_DEPENDENCE'; next_step='SIGNED_SCALE_DEVIATION_SURROGATE_VALIDATION'
        elif nsi >= 9 and nsd >= 9:
            label='ONE_SIDED_SCALE_CLAMPS_INSUFFICIENT'; next_step='TEMPORAL_SCALE_TRAJECTORY_INTERACTION_DIAGNOSIS'
        else:
            label='MIXED_SCALE_DIRECTION_EFFECT'; next_step='STRATIFIED_SCALE_DIRECTION_CONFIRMATION'
    medians={}
    for condition in CONDITIONS:
        medians[condition]={}
        for arm in ARMS:
            medians[condition][arm]=statistics.median(
                x['G_intervention'] for x in rows
                if x['condition']==condition and x['arm']==arm)
    result={
        'PARENT_EXPLOSIVE_COUNT':'12/12','CLEAN_SCALE_EXPLOSIVE_COUNT':'0/12',
        'NO_SCALE_INFLATION_EXPLOSIVE_COUNT':f'{nsi}/12',
        'NO_SCALE_DEFLATION_EXPLOSIVE_COUNT':f'{nsd}/12',
        'NATIVE_NO_SCALE_INFLATION_EXPLOSIVE_COUNT':f'{native_counts["NO_SCALE_INFLATION"]}/4',
        'NATIVE_NO_SCALE_DEFLATION_EXPLOSIVE_COUNT':f'{native_counts["NO_SCALE_DEFLATION"]}/4',
        'COUNTERFACTUAL_VALIDITY_GATE':validity,
        'PRIMARY_LABEL':label,'PRIMARY_NEXT_STEP':next_step,
        'per_condition_median_G':medians,
        'DISCOVERY_COMPLETE':'32/32',
        'TRAINING_EXECUTED':'NO','OPTIMIZER_STEPS':0,
        'NEW_C8_TRAINING':'NO','NEW_ROTATION_TRAINING':'NO',
        'AIME_EXECUTED':'NO','FREE_GENERATION_EXECUTED':'NO',
        'FINAL_ACTION':'STOP'}
    write_json(EXP/'analysis/primary_result.json',result)
    write_json(EXP/'analysis/per_doc_G.json',rows)
    write_json(EXP/'analysis/signed_scale_trajectory.json',signed_rows)
    write_json(EXP/'analysis/focus_layer_trajectories.json',focus_rows)
    write_json(EXP/'analysis/temporal_ordering.json',temporal)
    def csv_write(path,data):
        with path.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    csv_write(EXP/'analysis/per_doc_G.csv',rows)
    csv_write(EXP/'analysis/signed_scale_trajectory.csv',signed_rows)
    csv_write(EXP/'analysis/focus_layer_trajectories.csv',focus_rows)
    lines=['# QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1 — Final report','',
           '## Primary result','',
           f'- `PARENT_EXPLOSIVE_COUNT = 12/12`',
           f'- `CLEAN_SCALE_EXPLOSIVE_COUNT = 0/12`',
           f'- `NO_SCALE_INFLATION_EXPLOSIVE_COUNT = {nsi}/12`',
           f'- `NO_SCALE_DEFLATION_EXPLOSIVE_COUNT = {nsd}/12`',
           f'- `NATIVE_NO_SCALE_INFLATION_EXPLOSIVE_COUNT = {native_counts["NO_SCALE_INFLATION"]}/4`',
           f'- `NATIVE_NO_SCALE_DEFLATION_EXPLOSIVE_COUNT = {native_counts["NO_SCALE_DEFLATION"]}/4`','',
           f'- `PRIMARY_LABEL = {label}`',f'- `PRIMARY_NEXT_STEP = {next_step}`','',
           '## Per-condition median G','']
    for c in CONDITIONS:
        lines.append(f'- {c}: no-inflation `{medians[c][ARMS[0]]:.9g}`, no-deflation `{medians[c][ARMS[1]]:.9g}`')
    lines += ['', '## Interpretation boundary','',
              'The two prospective one-sided interventions localize signed scale feedback under the frozen-drive persistent-QDQ setting. Temporal precedence and scale/residual correlations are descriptive; causal evidence comes from the interventions. This does not establish AIME failure, free-generation instability, endpoint accuracy gaps, or validate a new training objective.','',
              'Full per-document G values, signed scale summaries, layer-6/layer-28 trajectories, QDQ residuals, saturation and onset records are in `analysis/`.','',
              'No training, C8, new rotation, AIME, or free generation was executed. The experiment stops here.','']
    write_text(EXP/'FINAL_REPORT.md','\n'.join(lines))
    write_json(EXP/'FINAL_STATUS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

