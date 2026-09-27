#!/usr/bin/env python3
"""One-shot model-level forward/state/QDQ check for explicit chunk candidate."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .qwen_chunk_reference import deterministic_torch_chunk_gated_delta_rule


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/deterministic_full_forward_gate.json'
V1_PATH = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')


def import_v1():
    spec = importlib.util.spec_from_file_location('v2_read_only_v1_recurrent', V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import read-only V1 implementation')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def explicit_deterministic_chunk_candidate():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod
    original = qmod.torch_chunk_gated_delta_rule
    qmod.torch_chunk_gated_delta_rule = deterministic_torch_chunk_gated_delta_rule
    try:
        yield
    finally:
        qmod.torch_chunk_gated_delta_rule = original


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.detach().float().reshape(-1)
    b = candidate.detach().float().reshape(-1)
    difference = a - b
    return {
        'max_abs': float(difference.abs().max().item()),
        'relative_l2': float((torch.linalg.vector_norm(difference)
                              / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()),
        'bitwise_equal': bool(torch.equal(reference, candidate)),
    }


def extract(model_output, gdn_layers):
    cache = model_output.past_key_values
    states = {layer: cache.layers[layer].recurrent_states[0].detach().clone()
              for layer in gdn_layers}
    logits = model_output.logits[:, -1, :].detach().clone()
    return logits, states


def save_once(relative: str, value: dict):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
        raise RuntimeError('original-server GPU0 must be the only visible GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = json.loads(CONFIG_PATH.read_text())
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    v1 = import_v1()
    row = v1.corpus_rows('TRAIN')[0]
    model, tokenizer, bank, _device = v1.load_model_bank()
    v1.freeze_model(model)
    token_ids, _positions = v1.tokenize(tokenizer, row)
    if len(token_ids) < config['prefix_tokens']:
        raise RuntimeError('fixed first training sample is too short')
    token_ids = token_ids[:config['prefix_tokens']]
    inputs = torch.tensor([token_ids], device='cuda:0', dtype=torch.long)
    target = torch.tensor([token_ids[-1]], device='cuda:0', dtype=torch.long)
    with torch.no_grad():
        torch.use_deterministic_algorithms(False)
        canonical_output = model(input_ids=inputs, use_cache=True)
        canonical_logits, canonical_states = extract(canonical_output, v1.GDN_LAYERS)
        del canonical_output
        torch.use_deterministic_algorithms(True)
        with explicit_deterministic_chunk_candidate():
            candidate_output = model(input_ids=inputs, use_cache=True)
        candidate_logits, candidate_states = extract(candidate_output, v1.GDN_LAYERS)
        del candidate_output
        logits_metrics = metrics(canonical_logits, candidate_logits)
        state_metrics = {str(layer): metrics(canonical_states[layer], candidate_states[layer])
                         for layer in v1.GDN_LAYERS}
        qdq_scale_metrics = {}
        qdq_codes_match = {}
        for layer in v1.GDN_LAYERS:
            canonical_qdq = v1.CAYLEY.qwen_c128_qdq(canonical_states[layer].float())
            candidate_qdq = v1.CAYLEY.qwen_c128_qdq(candidate_states[layer].float())
            qdq_scale_metrics[str(layer)] = metrics(canonical_qdq.scale, candidate_qdq.scale)
            qdq_codes_match[str(layer)] = bool(torch.equal(canonical_qdq.codes, candidate_qdq.codes))
        canonical_loss = F.cross_entropy(canonical_logits.float(), target)
        candidate_loss = F.cross_entropy(candidate_logits.float(), target)
        loss_relative_error = float(((canonical_loss - candidate_loss).abs()
                                     / canonical_loss.abs().clamp_min(1e-12)).item())
    criteria = {
        'logits': logits_metrics['max_abs'] <= config['logits_max_abs_threshold']
                  and logits_metrics['relative_l2'] <= config['logits_relative_l2_threshold'],
        'states': all(row['max_abs'] <= config['state_max_abs_threshold']
                      and row['relative_l2'] <= config['state_relative_l2_threshold']
                      for row in state_metrics.values()),
        'QDQ_codes': all(qdq_codes_match.values()),
        'QDQ_scale': all(row['relative_l2'] <= config['QDQ_scale_relative_l2_threshold']
                         for row in qdq_scale_metrics.values()),
        'loss': loss_relative_error <= config['loss_relative_error_threshold'],
    }
    result = {
        'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        'v1_read_only_source_sha256': hashlib.sha256(V1_PATH.read_bytes()).hexdigest(),
        'gdn_layer_count': len(v1.GDN_LAYERS), 'input_token_count': len(token_ids),
        'model_weights_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'logits': logits_metrics, 'state_per_layer': state_metrics,
        'QDQ_scale_per_layer': qdq_scale_metrics, 'QDQ_codes_exact_per_layer': qdq_codes_match,
        'loss_relative_error': loss_relative_error, 'criteria': criteria,
        'FULL_MODEL_TEACHER_FORWARD_GATE': 'PASS' if all(criteria.values()) else 'FAIL',
        'STUDENT_RECURRENT_WRITEBACK_GATE': 'NOT_RUN',
    }
    save_once('analysis/deterministic_full_model_teacher_forward_gate.json', result)
    print(json.dumps({'gate': result['FULL_MODEL_TEACHER_FORWARD_GATE'],
                      'criteria': criteria, 'logits': logits_metrics,
                      'worst_state_max_abs': max(x['max_abs'] for x in state_metrics.values()),
                      'loss_relative_error': loss_relative_error}), flush=True)
    if result['FULL_MODEL_TEACHER_FORWARD_GATE'] != 'PASS':
        raise RuntimeError('full-model teacher forward numerical gate failed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save_once('analysis/deterministic_full_model_teacher_forward_gate.error.json',
                  {'error_type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
