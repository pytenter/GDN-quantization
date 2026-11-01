#!/usr/bin/env python3
"""Frozen trajectory-level statistics for GDN state-error injection."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OUT = ROOT / 'experiments/QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1'
GEOM = ROOT / 'experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1'
STAGES = ('state', 'readout', 'rmsnorm', 'gate', 'output_projection')
SEED = 20260925


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def bootstrap_median(values, rng, repeats=10000):
    x = np.asarray(values, dtype=np.float64)
    sample = rng.choice(x, size=(repeats, len(x)), replace=True)
    return [float(y) for y in np.quantile(np.median(sample, axis=1), [0.025, 0.975])]


def sign(values):
    x = np.asarray(values, dtype=np.float64)
    positive = int(np.count_nonzero(x > 0))
    negative = int(np.count_nonzero(x < 0))
    zero = len(x) - positive - negative
    return {'positive': positive, 'negative': negative, 'zero': zero,
            'two_sided_p': float(binomtest(positive, positive+negative, 0.5).pvalue)
            if positive+negative else 1.0}


def cliff(x, y):
    a = np.asarray(x)[:, None]
    b = np.asarray(y)[None, :]
    return float((np.count_nonzero(a > b) - np.count_nonzero(a < b)) / (len(x)*len(y)))


def permutation_median_difference(x, y, rng, repeats=10000):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    observed = float(np.median(x) - np.median(y))
    pooled = np.concatenate((x, y))
    extreme = 0
    for _ in range(repeats):
        shuffled = rng.permutation(pooled)
        diff = float(np.median(shuffled[:len(x)]) - np.median(shuffled[len(x):]))
        extreme += abs(diff) >= abs(observed)
    return observed, float((extreme+1)/(repeats+1))


def holm(raw):
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    adjusted = [None] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw)-rank)*raw[index]))
        adjusted[index] = running
    return adjusted


def main():
    audit = json.loads((OUT/'analysis/replay_audit.json').read_text(encoding='utf-8'))
    parity = json.loads((OUT/'analysis/zero_injection_parity.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or parity['status'] != 'PASS':
        raise RuntimeError('audit or zero-injection parity failed')
    with (GEOM/'analysis/clean59_paired.csv').open(encoding='utf-8', newline='') as stream:
        label_rows = list(csv.DictReader(stream))
    groups = {(r['problem_id'], int(r['seed'])): r['group'] for r in label_rows}
    if len(groups) != 59:
        raise RuntimeError('clean59 true-ID label pairing failed')
    paths = sorted((OUT/'analysis').glob('shard_*/*.json'))
    if len(paths) != 60:
        raise RuntimeError(f'expected 60 completed units, found {len(paths)}')
    units = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    keys = [(u['problem_id'], int(u['seed'])) for u in units]
    if len(set(keys)) != 60 or set(keys)-set(groups) != {('aime26_28', 2)}:
        raise RuntimeError('source unit IDs or frozen exclusion mismatch')
    if sum(int(u['tokens']) for u in units) != 1854391 or any(u['layers'] != 24 for u in units):
        raise RuntimeError('token/layer grid mismatch')
    if any(max(u['zero_injection_parity_max_abs'].values()) > 0.001 for u in units):
        raise RuntimeError('formal zero-injection parity failed')
    clean = [u for u in units if (u['problem_id'], int(u['seed'])) in groups]
    if len(clean) != 59:
        raise RuntimeError('clean59 incomplete')
    clean_nonfinite = {}
    floor_counts = {}
    for stage in STAGES:
        clean_nonfinite[stage] = sum(sum(u['nonfinite'][stage].values()) for u in clean)
        floor_counts[stage] = sum(sum(u['denominator_floor'][stage].values()) for u in clean)
    if any(clean_nonfinite.values()):
        raise RuntimeError(f'clean59 contains nonfinite primary local metrics: {clean_nonfinite}')
    if any(floor_counts.values()):
        raise RuntimeError(f'clean59 FP denominator pathology: {floor_counts}')
    rng = np.random.default_rng(SEED)
    statistics = {}
    group_stats = {}
    raw_ps = []
    group_p_data = []
    for stage in STAGES:
        I = np.asarray([u['metrics'][stage]['identity']['relative'] for u in clean], dtype=np.float64)
        H = np.asarray([u['metrics'][stage]['hadamard']['relative'] for u in clean], dtype=np.float64)
        B = I-H
        if not np.all(np.isfinite(np.stack((I, H, B)))):
            raise RuntimeError(f'nonfinite trajectory median at {stage}')
        statistics[stage] = {
            'identity_median': float(np.median(I)), 'hadamard_median': float(np.median(H)),
            'benefit_median': float(np.median(B)), 'benefit_bootstrap_95ci': bootstrap_median(B, rng),
            'benefit_sign_test': sign(B), 'n_trajectories': 59,
            'cliffs_delta_identity_vs_hadamard_descriptive_unpaired': cliff(I, H),
            'benefit_definition': 'Identity relative L2 minus Hadamard relative L2',
        }
        signed = statistics[stage]['benefit_sign_test']
        n_nonzero = signed['positive'] + signed['negative']
        statistics[stage]['paired_sign_effect'] = ((signed['positive'] - signed['negative']) / n_nonzero
                                                   if n_nonzero else 0.0)
        by_group = {}
        for group in ('stable_correct', 'hadamard_rescue', 'stable_failure'):
            index = [j for j, u in enumerate(clean) if groups[(u['problem_id'], int(u['seed']))] == group]
            by_group[group] = {
                'n': len(index), 'identity_median': float(np.median(I[index])),
                'hadamard_median': float(np.median(H[index])),
                'benefit_median': float(np.median(B[index])),
                'benefit_positive': int(np.count_nonzero(B[index] > 0)),
                'benefit_negative': int(np.count_nonzero(B[index] < 0)),
                'benefit_zero': int(np.count_nonzero(B[index] == 0)),
            }
        rescue = [B[j] for j, u in enumerate(clean) if groups[(u['problem_id'], int(u['seed']))] == 'hadamard_rescue']
        failure = [B[j] for j, u in enumerate(clean) if groups[(u['problem_id'], int(u['seed']))] == 'stable_failure']
        diff, p = permutation_median_difference(rescue, failure, rng)
        raw_ps.append(p)
        group_p_data.append((stage, diff, p, cliff(rescue, failure)))
        group_stats[stage] = {'groups': by_group}
    for (stage, diff, p, effect), adjusted in zip(group_p_data, holm(raw_ps)):
        group_stats[stage]['rescue_minus_stable_failure'] = {
            'median_benefit_difference': diff, 'cliffs_delta': effect,
            'two_sided_permutation_p': p, 'holm_p': adjusted, 'permutations': 10000,
        }
    primary = statistics['output_projection']
    primary_supported = (primary['benefit_bootstrap_95ci'][0] > 0 and
                         primary['benefit_sign_test']['two_sided_p'] < 0.05)
    # The audit stores the layer count; these 24 layers are frozen by the unit grid.
    frozen_layers = sorted(map(int, clean[0]['by_layer']))
    if len(frozen_layers) != 24:
        raise RuntimeError('frozen GDN layer list incomplete')
    localization = {'primary_output_projection_benefit_supported': bool(primary_supported),
                    'stage_statistics': {s: statistics[s] for s in STAGES},
                    'layer_descriptive': {str(layer): {
            stage: {'identity_median': float(np.median([u['by_layer'][str(layer)][stage]['identity'] for u in clean])),
                    'hadamard_median': float(np.median([u['by_layer'][str(layer)][stage]['hadamard'] for u in clean]))}
            for stage in STAGES} for layer in frozen_layers}}
    logits_path = OUT/'analysis/full_logits_feasibility.json'
    if not logits_path.exists():
        raise RuntimeError('full-logits feasibility gate has not been recorded')
    logits = json.loads(logits_path.read_text(encoding='utf-8'))
    save(OUT/'analysis/functional_damage_statistics.json', {
        'status': 'COMPLETE', 'primary_endpoint': 'output_projection relative L2',
        'clean59': 59, 'statistics': statistics, 'nonfinite_clean59': clean_nonfinite,
        'fp_denominator_floor_clean59': floor_counts, 'full_logits_feasibility': logits,
    })
    save(OUT/'analysis/rescue_statistics.json', {
        'status': 'COMPLETE', 'true_id_grouping': True, 'statistics': group_stats,
        'holm_family': list(STAGES),
    })
    save(OUT/'analysis/module_localization.json', localization)
    figs = OUT/'figures'
    figs.mkdir(exist_ok=True)
    I = np.asarray([u['metrics']['output_projection']['identity']['relative'] for u in clean])
    H = np.asarray([u['metrics']['output_projection']['hadamard']['relative'] for u in clean])
    labels = [groups[(u['problem_id'], int(u['seed']))] for u in clean]
    colors = {'stable_correct': '#537dbb', 'hadamard_rescue': '#e69637', 'stable_failure': '#5fa36a'}
    plt.figure(figsize=(7, 6))
    for group in colors:
        index = [i for i, label in enumerate(labels) if label == group]
        plt.scatter(I[index], H[index], label=group, color=colors[group], alpha=.8)
    low = float(min(I.min(), H.min()))
    high = float(max(I.max(), H.max()))
    plt.plot([low, high], [low, high], color='gray')
    plt.xlabel('Identity injected output-projection relative L2')
    plt.ylabel('Hadamard injected output-projection relative L2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs/'functional_damage_identity_vs_hadamard.png', dpi=160)
    plt.close()
    B = I-H
    plt.figure(figsize=(7, 5))
    order = ('stable_correct', 'hadamard_rescue', 'stable_failure')
    plt.boxplot([[B[i] for i, label in enumerate(labels) if label == group] for group in order],
                labels=order, showfliers=True)
    plt.axhline(0, color='gray')
    plt.ylabel('Functional benefit: Identity damage − Hadamard damage')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figs/'rescue_functional_benefit.png', dpi=160)
    plt.close()
    plt.figure(figsize=(8, 5))
    x = np.arange(len(STAGES))
    plt.plot(x, [statistics[s]['identity_median'] for s in STAGES], marker='o', label='Identity')
    plt.plot(x, [statistics[s]['hadamard_median'] for s in STAGES], marker='o', label='Hadamard')
    plt.xticks(x, STAGES, rotation=20)
    plt.ylabel('Trajectory-median relative L2 (clean59 median)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs/'module_error_propagation.png', dpi=160)
    plt.close()
    report = [
        '# Qwen Hadamard Error Injection Functional Analysis V1', '',
        'Deterministic teacher-forced intervention with a common FP recurrent state, FP query and FP gate. '
        'The injected error is the observed native-basis difference from each INT8 trajectory, which includes historical drift.', '',
        '## Gates and coverage', '',
        f'- Audit: PASS; zero-injection parity: PASS; source trajectories: 60; clean59: 59; GDN layers: 24.',
        '- No generation, training, model/quantizer/scorer modification, or permanent full state/logit tensors.',
        f'- Full-logits feasibility: {logits["status"]}.', '',
        '## Paired local functional damage', '',
        'Relative L2 is lower-is-better. Benefit = Identity minus Hadamard; positive favors Hadamard. '
        'Each trajectory is weighted equally.', '',
        '| Stage | Identity median | Hadamard median | Benefit median | Bootstrap 95% CI | Sign p |',
        '| --- | ---: | ---: | ---: | --- | ---: |',
    ]
    for stage in STAGES:
        item = statistics[stage]
        report.append(f'| {stage} | {item["identity_median"]:.6g} | {item["hadamard_median"]:.6g} | '
                      f'{item["benefit_median"]:.6g} | {item["benefit_bootstrap_95ci"]} | '
                      f'{item["benefit_sign_test"]["two_sided_p"]:.6g} |')
    report.extend(['', '## Rescue versus stable failure', '',
                   'True-ID groups: stable-correct 18, Hadamard-rescue 23, stable-failure 18. '
                   'Two-sided permutation p values are Holm-adjusted across five stages.', '',
                   '| Stage | Rescue benefit median | Stable-failure benefit median | Difference | Cliff delta | Holm p |',
                   '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for stage in STAGES:
        x = group_stats[stage]
        test = x['rescue_minus_stable_failure']
        report.append(f'| {stage} | {x["groups"]["hadamard_rescue"]["benefit_median"]:.6g} | '
                      f'{x["groups"]["stable_failure"]["benefit_median"]:.6g} | '
                      f'{test["median_benefit_difference"]:.6g} | {test["cliffs_delta"]:.3f} | '
                      f'{test["holm_p"]:.6g} |')
    report.extend(['', '## Interpretation', '',
                   f'Primary output-projection benefit supported under the preregistered CI/sign rule: {primary_supported}.',
                   'Local module effects are not, by themselves, proof of downstream logit benefit or reasoning causality. '
                   'No outcome-based layer, token, or outlier selection was performed.', '',
                   f'Clean59 nonfinite counts: {clean_nonfinite}. FP denominator floor counts: {floor_counts}.',
                   'The frozen excluded raw trajectory remains preserved and is not part of clean59 inference.', ''])
    (OUT/'final_report.md').write_text('\n'.join(report), encoding='utf-8')
    manifest = {
        'task': 'QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1',
        'status': 'COMPLETE', 'source_trajectories': 60, 'clean59': 59,
        'analysis_script_sha256': digest(Path(__file__)),
        'capture_script_sha256': digest(OUT/'run_injection_v2.py'),
        'unit_hashes': {str(p.relative_to(OUT)): digest(p) for p in paths},
        'outputs': {},
    }
    outputs = [OUT/'final_report.md', OUT/'analysis/replay_audit.json',
               OUT/'analysis/feasibility.json', OUT/'analysis/zero_injection_parity.json',
               OUT/'analysis/functional_damage_statistics.json', OUT/'analysis/rescue_statistics.json',
               OUT/'analysis/module_localization.json', OUT/'analysis/full_logits_feasibility.json',
               OUT/'figures/functional_damage_identity_vs_hadamard.png',
               OUT/'figures/rescue_functional_benefit.png', OUT/'figures/module_error_propagation.png',
               OUT/'preregistration.json', OUT/'analysis_amendment_v1.json',
               OUT/'analysis_amendment_v2.json']
    for path in outputs:
        if path.exists():
            manifest['outputs'][str(path.relative_to(OUT))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    save(OUT/'run_manifest.json', manifest)
    print(json.dumps({'status': 'COMPLETE', 'primary_benefit_supported': bool(primary_supported),
                      'full_logits_feasibility': logits['status'], 'source_trajectories': 60,
                      'clean59': 59, 'primary': primary}))


if __name__ == '__main__':
    main()
