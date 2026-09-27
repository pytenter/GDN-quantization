#!/usr/bin/env python3
"""Compare frozen factorial captures without changing any model behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = ROOT.parents[1] / 'recurrent_training' / 'QWEN_GDN_DISTRIBUTED_RECURRENT_TRAINING_V2' / 'configs' / 'deterministic_full_forward_gate.json'


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {'shape_equal': False, 'bitwise_equal': False}
    exact = torch.equal(a, b)
    x, y = a.reshape(-1).double(), b.reshape(-1).double()
    d = x - y
    changed = torch.nonzero(a.reshape(-1) != b.reshape(-1), as_tuple=False)
    norm = torch.linalg.vector_norm(x).clamp_min(1e-12)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    cosine = 1.0 - float(torch.dot(x, y) / denom) if float(denom) else 0.0
    return {'shape_equal': True, 'bitwise_equal': exact,
            'max_abs': float(d.abs().max()) if d.numel() else 0.0,
            'mean_abs': float(d.abs().mean()) if d.numel() else 0.0,
            'relative_l2': float(torch.linalg.vector_norm(d) / norm),
            'cosine_error': cosine,
            'different_element_count': int(changed.numel()),
            'first_different_flat_index': int(changed[0]) if changed.numel() else None}


def compare(left: str, right: str):
    lmeta = json.loads((ROOT / 'analysis' / f'{left}_summary.json').read_text())
    rmeta = json.loads((ROOT / 'analysis' / f'{right}_summary.json').read_text())
    if lmeta['status'] != 'COMPLETE' or rmeta['status'] != 'COMPLETE':
        raise RuntimeError('both factorial conditions must complete before comparison')
    if lmeta['frozen_input'] != rmeta['frozen_input'] or lmeta['source_sha256'] != rmeta['source_sha256']:
        raise RuntimeError('frozen input/source mismatch')
    lvals = torch.load(ROOT / 'analysis' / 'temporary' / f'{left}.pt', map_location='cpu', weights_only=True)
    rvals = torch.load(ROOT / 'analysis' / 'temporary' / f'{right}.pt', map_location='cpu', weights_only=True)
    order = lmeta['capture_order']
    if order != rmeta['capture_order']:
        raise RuntimeError('ordered capture ladder differs between conditions')
    comparisons = {key: metrics(lvals[key], rvals[key]) for key in order}
    first_index = next((i for i, key in enumerate(order) if not comparisons[key]['bitwise_equal']), None)
    first = order[first_index] if first_index is not None else None
    previous = order[first_index - 1] if first_index is not None and first_index > 0 else None
    config = json.loads(V2_CONFIG.read_text())
    logits = comparisons['teacher.last_logits']
    loss = comparisons['teacher.loss']
    state = [comparisons[f'layer{layer}.pre_qdq_state'] for layer in range(32)
             if f'layer{layer}.pre_qdq_state' in comparisons]
    scales = [comparisons[f'layer{layer}.c128_scale'] for layer in range(32)
              if f'layer{layer}.c128_scale' in comparisons]
    qcodes = {str(layer): comparisons[f'layer{layer}.c128_qcodes']['different_element_count']
              for layer in range(32) if f'layer{layer}.c128_qcodes' in comparisons}
    post = [comparisons[f'layer{layer}.post_qdq_state'] for layer in range(32)
            if f'layer{layer}.post_qdq_state' in comparisons]
    loss_rel = loss['max_abs'] / max(abs(float(lvals['teacher.loss'])), 1e-12)
    criteria = {
        'logits': logits['max_abs'] <= config['logits_max_abs_threshold'] and logits['relative_l2'] <= config['logits_relative_l2_threshold'],
        'loss': loss_rel <= config['loss_relative_error_threshold'],
        'states': all(x['max_abs'] <= config['state_max_abs_threshold'] and x['relative_l2'] <= config['state_relative_l2_threshold'] for x in state),
        'scales': all(x['relative_l2'] <= config['QDQ_scale_relative_l2_threshold'] for x in scales),
        'qcodes': all(n == 0 for n in qcodes.values()),
        'post_qdq_states': all(x['max_abs'] <= config['state_max_abs_threshold'] and x['relative_l2'] <= config['state_relative_l2_threshold'] for x in post),
    }
    result = {'pair': f'{left}_vs_{right}', 'frozen_input': lmeta['frozen_input'],
              'first_divergent_node': first, 'previous_node': previous,
              'previous_node_exact': comparisons[previous]['bitwise_equal'] if previous else None,
              'prefix_input_exact': comparisons['gdn.prefix_input']['bitwise_equal'] if 'gdn.prefix_input' in comparisons else None,
              'prefix_output_exact': comparisons['gdn.prefix_output']['bitwise_equal'] if 'gdn.prefix_output' in comparisons else None,
              'logits': logits, 'loss': loss, 'loss_relative_error': loss_rel,
              'qcode_changed_count_by_layer': qcodes, 'criteria': criteria,
              'full_model_forward_semantics_gate': 'PASS' if all(criteria.values()) else 'FAIL',
              'captures': comparisons}
    path = ROOT / 'analysis' / f'{left}_vs_{right}.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('pair', 'first_divergent_node', 'previous_node', 'prefix_input_exact', 'prefix_output_exact', 'full_model_forward_semantics_gate', 'logits')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    a = p.parse_args()
    compare(a.left, a.right)
