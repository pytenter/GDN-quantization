#!/usr/bin/env python3
"""Opt-in GPU-only deterministic cumsum reference and bounded microdiagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/deterministic_runtime.json'


def fixed_left_to_right_cumsum(g: torch.Tensor) -> torch.Tensor:
    if not g.is_cuda or g.dtype != torch.float32 or g.shape[-1] != 64:
        raise ValueError('Only the audited CUDA FP32 chunk-64 GDN tensor is supported')
    running = torch.zeros_like(g[..., 0])
    pieces = []
    for index in range(g.shape[-1]):
        running = running + g[..., index]
        pieces.append(running)
    return torch.stack(pieces, dim=-1)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    left = a.detach().float().reshape(-1)
    right = b.detach().float().reshape(-1)
    delta = left - right
    denominator = torch.linalg.vector_norm(left).clamp_min(1e-12)
    cosine = torch.dot(left, right) / (
        torch.linalg.vector_norm(left).clamp_min(1e-12)
        * torch.linalg.vector_norm(right).clamp_min(1e-12))
    return {
        'max_abs': float(delta.abs().max().item()),
        'relative_l2': float((torch.linalg.vector_norm(delta) / denominator).item()),
        'cosine': float(cosine.item()),
        'bitwise_equal': bool(torch.equal(a, b)),
    }


def main() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected only original-server GPU0 to be visible')
    config = json.loads(CONFIG_PATH.read_text())
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    rows = []
    for shape in config['input_shapes']:
        g = torch.randn(*shape, device='cuda:0', dtype=torch.float32) * 0.1
        torch.use_deterministic_algorithms(False)
        canonical = g.cumsum(dim=-1)
        torch.use_deterministic_algorithms(True)
        candidate = fixed_left_to_right_cumsum(g)
        fp32 = metrics(canonical, candidate)
        bf16 = metrics(canonical.to(torch.bfloat16), candidate.to(torch.bfloat16))
        weights = torch.linspace(0.1, 1.0, 64, device='cuda:0')
        differentiable_input = g.detach().requires_grad_()
        gradient = torch.autograd.grad(
            (fixed_left_to_right_cumsum(differentiable_input) * weights).sum(),
            differentiable_input)[0]
        expected_gradient = fixed_left_to_right_cumsum(
            weights.flip(0).reshape(1, 1, 1, 64).contiguous()).flip(-1)
        gradient_metrics = metrics(gradient, expected_gradient.expand_as(gradient))
        rows.append({'shape': shape, 'fp32': fp32, 'bf16': bf16,
                     'candidate_gradient_vs_reverse_prefix_reference': gradient_metrics,
                     'fp32_threshold_pass': fp32['max_abs'] <= config['fp32_max_abs_threshold']
                     and fp32['relative_l2'] <= config['fp32_relative_l2_threshold']
                     and fp32['cosine'] >= config['cosine_minimum'],
                     'bf16_threshold_pass': bf16['max_abs'] <= config['bf16_max_abs_threshold']
                     and bf16['relative_l2'] <= config['bf16_relative_l2_threshold']
                     and bf16['cosine'] >= config['cosine_minimum'],
                     'gradient_reference_pass': gradient_metrics['max_abs'] <= config['fp32_max_abs_threshold']})
    result = {
        'candidate': config['candidate'], 'canonical_operation': config['canonical_operation'],
        'candidate_scope': config['candidate_scope'],
        'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        'rows': rows,
        'operator_micro_gate': 'PASS' if all(r['fp32_threshold_pass'] and r['bf16_threshold_pass']
                                             and r['gradient_reference_pass'] for r in rows) else 'FAIL',
        'recurrent_state_and_loss_comparison': 'NOT_RUN',
        'full_deterministic_op_gate': 'NOT_RUN',
    }
    target = ROOT / 'analysis/deterministic_cumsum_micro.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'operator_micro_gate': result['operator_micro_gate'], 'rows': rows}), flush=True)


if __name__ == '__main__':
    main()
