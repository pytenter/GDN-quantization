#!/usr/bin/env python3
"""Replay the actual layer-0 prefix input, once per fresh process."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from experiments.qwen_gdn.recurrent_training.QWEN_GDN_DISTRIBUTED_RECURRENT_TRAINING_V2.runtime.deterministic_ops.cumsum_reference import fixed_left_to_right_cumsum  # noqa: E402


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(a, b):
    x = a.detach().double().reshape(-1).cpu()
    y = b.detach().double().reshape(-1).cpu()
    delta = x - y
    return {'bitwise_equal': bool(torch.equal(a.detach().cpu(), b.detach().cpu())),
            'max_abs': float(delta.abs().max()),
            'relative_l2': float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(x).clamp_min(1e-12)),
            'different_element_count': int((x != y).sum())}


def run(index: int):
    if torch.cuda.device_count() != 1:
        raise RuntimeError('one idle GPU only')
    target = ROOT / 'analysis' / 'prefix_replay' / f'rep_{index:02d}.json'
    if target.exists():
        raise FileExistsError('fresh-process replay already exists')
    prior = torch.load(ROOT / 'analysis' / 'temporary' / 'P0R0.pt', map_location='cpu', weights_only=True)
    src = prior['gdn.prefix_input']
    actual = src.to('cuda:0')
    torch.use_deterministic_algorithms(False)
    canonical = []
    candidate = []
    for _ in range(5):
        canonical.append(actual.cumsum(dim=-1))
        candidate.append(fixed_left_to_right_cumsum(actual))
    torch.cuda.synchronize()
    # Mathematical diagnostic only: CPU FP64, never an inference/training runtime.
    reference = src.double().cumsum(dim=-1)
    result = {
        'replicate': index,
        'input': {'sha256': digest(src), 'shape': list(src.shape), 'stride': list(src.stride()),
                  'dtype': str(src.dtype), 'contiguous': src.is_contiguous(), 'dimension': -1,
                  'chunk_size': 64, 'accumulation_dtype': 'float32 for both CUDA paths',
                  'canonical_operation': 'torch.Tensor.cumsum(dim=-1)',
                  'candidate_operation': '64 sequential CUDA FP32 adds then torch.stack',
                  'launch_config': 'PyTorch default kernels; no explicit Triton/FLA config'},
        'canonical_hashes': [digest(t) for t in canonical],
        'candidate_hashes': [digest(t) for t in candidate],
        'canonical_same_process_exact': len(set(digest(t) for t in canonical)) == 1,
        'candidate_same_process_exact': len(set(digest(t) for t in candidate)) == 1,
        'canonical_vs_candidate': metrics(canonical[0], candidate[0]),
        'canonical_vs_cpu_fp64_reference': metrics(canonical[0], reference),
        'candidate_vs_cpu_fp64_reference': metrics(candidate[0], reference),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'replicate': index, 'canonical_hash': result['canonical_hashes'][0],
                      'candidate_hash': result['candidate_hashes'][0],
                      'difference': result['canonical_vs_candidate']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replicate', type=int, required=True)
    run(p.parse_args().replicate)
