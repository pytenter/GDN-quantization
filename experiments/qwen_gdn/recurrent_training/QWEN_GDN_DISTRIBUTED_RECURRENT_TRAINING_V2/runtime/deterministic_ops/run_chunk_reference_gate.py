#!/usr/bin/env python3
"""One-shot canonical versus explicit deterministic chunk operator diagnostic."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from .qwen_chunk_reference import deterministic_torch_chunk_gated_delta_rule


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/deterministic_chunk_gate.json'
CAYLEY_SOURCE = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation')
CANONICAL_SOURCE = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')


def summarize(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a, b = reference.detach().float().reshape(-1), candidate.detach().float().reshape(-1)
    diff = a - b
    denominator = torch.linalg.vector_norm(a).clamp_min(1e-12)
    cosine = torch.dot(a, b) / (
        torch.linalg.vector_norm(a).clamp_min(1e-12)
        * torch.linalg.vector_norm(b).clamp_min(1e-12))
    return {'max_abs': float(diff.abs().max().item()),
            'relative_l2': float((torch.linalg.vector_norm(diff) / denominator).item()),
            'cosine': float(cosine.item()),
            'bitwise_equal': bool(torch.equal(reference, candidate))}


def save_once(relative: str, data: dict) -> None:
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
        raise RuntimeError('only original-server GPU0 may be visible')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = json.loads(CONFIG_PATH.read_text())
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod
    sys.path.insert(0, str(CAYLEY_SOURCE))
    import cayley_rotation

    shape = config['shape_query_key_value']
    scalar_shape = config['shape_g_beta']
    q = (torch.randn(*shape, device='cuda:0') * 0.02).to(torch.bfloat16)
    k = (torch.randn(*shape, device='cuda:0') * 0.02).to(torch.bfloat16)
    v = (torch.randn(*shape, device='cuda:0') * 0.02).to(torch.bfloat16)
    g = (-torch.rand(*scalar_shape, device='cuda:0') * 0.02).to(torch.bfloat16)
    beta = torch.rand(*scalar_shape, device='cuda:0').to(torch.bfloat16)
    initial = torch.randn(*config['shape_initial_recurrent_state'], device='cuda:0') * 0.01
    target = (torch.randn(*shape, device='cuda:0') * 0.02).to(torch.bfloat16)
    args = (q, k, v)
    kwargs = {'g': g, 'beta': beta, 'chunk_size': config['chunk_size'],
              'initial_state': initial, 'output_final_state': True,
              'use_qk_l2norm_in_kernel': config['use_qk_l2norm_in_kernel']}

    torch.use_deterministic_algorithms(False)
    hub_core, hub_state = qmod.torch_chunk_gated_delta_rule(*args, **kwargs)
    fallback = inspect.unwrap(qmod.torch_chunk_gated_delta_rule)
    canonical_core, canonical_state = fallback(*args, **kwargs)
    hub_core_metrics = summarize(canonical_core, hub_core)
    hub_state_metrics = summarize(canonical_state, hub_state)

    torch.use_deterministic_algorithms(True)
    candidate_core, candidate_state = deterministic_torch_chunk_gated_delta_rule(*args, **kwargs)
    core_metrics = summarize(canonical_core, candidate_core)
    state_metrics = summarize(canonical_state, candidate_state)
    canonical_qdq = cayley_rotation.qwen_c128_qdq(canonical_state)
    candidate_qdq = cayley_rotation.qwen_c128_qdq(candidate_state)
    qdq_scale_metrics = summarize(canonical_qdq.scale, candidate_qdq.scale)
    qdq_codes_exact = bool(torch.equal(canonical_qdq.codes, candidate_qdq.codes))
    canonical_loss = cayley_rotation.relative_mse(canonical_core.float(), target.float())
    candidate_loss = cayley_rotation.relative_mse(candidate_core.float(), target.float())
    loss_relative_error = float(((canonical_loss - candidate_loss).abs()
                                 / canonical_loss.abs().clamp_min(1e-12)).item())
    criteria = {
        'hub_wrapper_equals_fallback': hub_core_metrics['bitwise_equal'] and hub_state_metrics['bitwise_equal'],
        'core': core_metrics['max_abs'] <= config['core_max_abs_threshold']
                and core_metrics['relative_l2'] <= config['core_relative_l2_threshold'],
        'state': state_metrics['max_abs'] <= config['state_max_abs_threshold']
                 and state_metrics['relative_l2'] <= config['state_relative_l2_threshold'],
        'loss': loss_relative_error <= config['loss_relative_error_threshold'],
        'qdq_codes': qdq_codes_exact,
        'qdq_scale': qdq_scale_metrics['relative_l2'] <= config['QDQ_scale_relative_l2_threshold'],
    }
    result = {
        'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        'canonical_source_sha256': hashlib.sha256(CANONICAL_SOURCE.read_bytes()).hexdigest(),
        'hub_wrapper_vs_fallback': {'core': hub_core_metrics, 'state': hub_state_metrics},
        'candidate_vs_canonical': {'core': core_metrics, 'state': state_metrics,
                                   'qdq_scale': qdq_scale_metrics,
                                   'qdq_codes_exact': qdq_codes_exact,
                                   'loss_relative_error': loss_relative_error},
        'criteria': criteria,
        'DETERMINISTIC_CHUNK_FUNCTION_GATE': 'PASS' if all(criteria.values()) else 'FAIL',
        'full_model_recurrent_gate': 'NOT_RUN',
    }
    save_once('analysis/deterministic_chunk_function_gate.json', result)
    print(json.dumps({'gate': result['DETERMINISTIC_CHUNK_FUNCTION_GATE'],
                      'criteria': criteria, 'metrics': result['candidate_vs_canonical']}), flush=True)
    if result['DETERMINISTIC_CHUNK_FUNCTION_GATE'] != 'PASS':
        raise RuntimeError('deterministic chunk function semantic/numerical gate failed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        error = {'error_type': type(exc).__name__, 'message': str(exc),
                 'traceback': traceback.format_exc()}
        target = ROOT / 'analysis/deterministic_chunk_function_gate.error.json'
        with target.open('x', encoding='utf-8') as handle:
            json.dump(error, handle, indent=2, sort_keys=True)
            handle.write('\n')
        raise
