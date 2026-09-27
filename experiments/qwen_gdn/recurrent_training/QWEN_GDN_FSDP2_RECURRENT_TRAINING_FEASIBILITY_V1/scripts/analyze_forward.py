#!/usr/bin/env python3
"""Freeze same-topology teacher-forward baseline before canonical comparison."""
from __future__ import annotations

import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'analysis'


def read(name):
    return json.loads((A / name).read_text())


def metrics(a, b):
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)
    d = x - y
    return {'bitwise_equal': bool(torch.equal(a, b)), 'max_abs': float(d.abs().max()),
            'relative_l2': float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(x).clamp_min(1e-12)),
            'different_element_count': int((x != y).sum())}


def main():
    fsdp = [read(f'fsdp{i:02d}_rank{rank}.json') for i in (1, 2, 3) for rank in (0, 1)]
    single = read('single01_rank0.json')
    if any(x['status'] != 'COMPLETE' or x['input_ids_sha256'] != single['input_ids_sha256'] or
           x['model_source_sha256'] != single['model_source_sha256'] for x in fsdp):
        raise RuntimeError('source/input/status mismatch')
    fields = ['logits', 'loss', 'layer0.state', 'layer16.state', 'layer30.state']
    fsdp_vals = [torch.load(A / 'temporary' / f'fsdp{i:02d}_rank{rank}.pt', map_location='cpu', weights_only=True)
                 for i in (1, 2, 3) for rank in (0, 1)]
    single_vals = torch.load(A / 'temporary/single01_rank0.pt', map_location='cpu', weights_only=True)
    same = {k: all(torch.equal(fsdp_vals[0][k], v[k]) for v in fsdp_vals[1:]) for k in fields}
    pair = {k: metrics(single_vals[k], fsdp_vals[0][k]) for k in fields}
    gate = 'PASS' if all(same.values()) and all(x['bitwise_equal'] for x in pair.values()) else 'FAIL'
    result = {'same_topology_fresh_process_count': 3, 'rank_outputs_per_process': 2,
              'same_topology_exact': same, 'single_vs_FSDP2': pair,
              'input_ids_sha256': single['input_ids_sha256'],
              'model_source_sha256': single['model_source_sha256'],
              'single_peak_allocated_bytes': single['memory_after_forward']['peak_allocated_bytes'],
              'fsdp_rank0_peak_allocated_bytes': fsdp[0]['memory_after_forward']['peak_allocated_bytes'],
              'fsdp_rank1_peak_allocated_bytes': fsdp[1]['memory_after_forward']['peak_allocated_bytes'],
              'CANONICAL_FORWARD_SEMANTICS_GATE': gate}
    path = A / 'forward_semantics.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'CANONICAL_FORWARD_SEMANTICS_GATE': gate, 'same_topology_exact': same,
                      'single_vs_FSDP2': pair}), flush=True)


if __name__ == '__main__':
    main()
