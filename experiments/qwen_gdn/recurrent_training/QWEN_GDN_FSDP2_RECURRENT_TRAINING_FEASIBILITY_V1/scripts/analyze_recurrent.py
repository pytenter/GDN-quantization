#!/usr/bin/env python3
"""Analyze immutable compact recurrent probe outputs; never tune gates from results."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'analysis' / name).read_text())


def save(name, value):
    path = ROOT / 'analysis' / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def semantic():
    entries = [read('single_H1_sem_01_rank0.json')]
    entries += [read(f'fsdp_H1_sem_01_rank{rank}.json') for rank in (0, 1)]
    reference = entries[0]
    checks = []
    for entry in entries:
        checks.append({'mode': entry['mode'], 'rank': entry['rank'],
                       'sample_exact': entry['sample_token_hash'] == reference['sample_token_hash'],
                       'loss_exact': entry['loss'] == reference['loss'],
                       'qdq_all_fields_exact': entry['QDQ'] == reference['QDQ'],
                       'qdq_layer_count': len(entry['QDQ']),
                       'writeback_all_exact': all(entry['cache_writeback_exact_by_layer'].values()),
                       'theta_unchanged': entry['theta_changed_count'] == 0})
    passed = all(c['sample_exact'] and c['loss_exact'] and c['qdq_all_fields_exact']
                 and c['qdq_layer_count'] == 24 and c['writeback_all_exact']
                 and c['theta_unchanged'] for c in checks)
    save('recurrent_qdq_semantics.json', {'gate': 'PASS' if passed else 'FAIL',
                                         'condition': 'theta0, H1, frozen C5 input and C128 QDQ',
                                         'checks': checks,
                                         'sources': ['single_H1_sem_01_rank0.json',
                                                     'fsdp_H1_sem_01_rank0.json',
                                                     'fsdp_H1_sem_01_rank1.json']})
    if not passed:
        raise RuntimeError('recurrent semantics failed')


def repeatability():
    records = [[read(f'fsdp_H1_update_{i:02d}_rank{r}.json') for r in (0, 1)]
               for i in range(1, 6)]
    ref = records[0][0]
    per_process = []
    for pair in records:
        for entry in pair:
            grads = entry['gradient']
            per_process.append({'replicate': entry['replicate'], 'rank': entry['rank'],
                                'forward_exact': entry['QDQ'] == ref['QDQ'],
                                'loss_exact': entry['loss'] == ref['loss'],
                                'all_24_gradients_valid': len(grads) == 24 and all(
                                    g['exists'] and g['finite'] and g['norm'] > 0 for g in grads.values()),
                                'all_24_theta_changed': entry['theta_changed_count'] == 24,
                                'theta_update_exact': entry['theta_after'] == ref['theta_after'],
                                'model_sample_unchanged': entry['model_sample_unchanged'],
                                'writeback_all_exact': all(entry['cache_writeback_exact_by_layer'].values())})
    envelope = {}
    for name in ref['gradient']:
        values = [entry['gradient'][name]['norm'] for pair in records for entry in pair]
        envelope[name] = {'norm_min': min(values), 'norm_max': max(values),
                          'norm_range': max(values) - min(values)}
    semantic_pass = all(r['forward_exact'] and r['loss_exact'] and r['all_24_gradients_valid']
                        and r['all_24_theta_changed'] and r['model_sample_unchanged']
                        and r['writeback_all_exact'] for r in per_process)
    save('same_topology_repeatability.json', {'status': 'PASS' if semantic_pass else 'FAIL',
                                              'numeric_exact': all(r['theta_update_exact'] for r in per_process)
                                              and all(v['norm_range'] == 0 for v in envelope.values()),
                                              'records': per_process, 'gradient_norm_envelope': envelope,
                                              'source_replicates': list(range(1, 6))})
    if not semantic_pass:
        raise RuntimeError('H1 repeatability semantic gate failed')


def memory():
    rows = []
    for horizon in (1, 4, 8, 16, 32):
        for rank in (0, 1):
            entry = read(f'fsdp_H{horizon}_update_01_rank{rank}.json')
            stats = entry['memory_after_optimizer']
            rows.append({'horizon': horizon, 'rank': rank,
                         'baseline_allocated_bytes': entry['memory_before_rollout']['allocated_bytes'],
                         'baseline_reserved_bytes': entry['memory_before_rollout']['reserved_bytes'],
                         'peak_allocated_bytes': stats['peak_allocated_bytes'],
                         'peak_reserved_bytes': stats['peak_reserved_bytes'],
                         'end_allocated_bytes': stats['allocated_bytes'],
                         'end_reserved_bytes': stats['reserved_bytes'],
                         'free_margin_fraction': (stats['device_total_bytes'] - stats['peak_reserved_bytes']) /
                                                 stats['device_total_bytes'],
                         'semantics_exact_vs_H1': entry['QDQ'] == read(f'fsdp_H1_update_01_rank{rank}.json')['QDQ'],
                         'all_gradients_valid': len(entry['gradient']) == 24 and all(
                             g['exists'] and g['finite'] and g['norm'] > 0
                             for g in entry['gradient'].values()),
                         'theta_changed_count': entry['theta_changed_count'],
                         'model_sample_unchanged': entry['model_sample_unchanged']})
    passed = all(r['semantics_exact_vs_H1'] and r['all_gradients_valid']
                 and r['theta_changed_count'] == 24 and r['model_sample_unchanged'] for r in rows)
    save('memory_scaling.json', {'status': 'PASS' if passed else 'FAIL', 'rows': rows})
    if not passed:
        raise RuntimeError('memory scaling semantic gate failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['semantic', 'repeatability', 'memory'])
    globals()[p.parse_args().phase]()
