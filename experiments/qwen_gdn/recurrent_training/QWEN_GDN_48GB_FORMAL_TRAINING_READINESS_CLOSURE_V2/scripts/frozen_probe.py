#!/usr/bin/env python3
"""Observational non-AIME probe around the unchanged canonical recurrent patch."""
from __future__ import annotations

import hashlib
import json

import torch

import runtime

SELECTED_LAYERS = (0, 16, 30)


def fingerprint(value: torch.Tensor) -> dict:
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError('nonfinite frozen-bank functional probe tensor')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': runtime.tensor_sha256(value)}


def run_fixed_probe(core, model, tokenizer, bank) -> dict:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    ids, _ = core.tokenize(tokenizer, core.corpus_rows('TRAIN')[0])
    ids = ids[:32]
    if len(ids) != 32:
        raise RuntimeError('fixed TRAIN:first-row sample must have 32 tokens')
    sample_sha = core.tensor_sha256(torch.tensor(ids, dtype=torch.long))
    parent_sample_sha = '1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d'
    if sample_sha != parent_sample_sha:
        raise RuntimeError('fixed sample differs from V1 diagnostic')

    device = model.get_input_embeddings().weight.device
    patch = core.RecurrentExperimentPatch(model, bank, core.hadamard(device))
    state = {'token_index': None, 'selected': {}, 'post_block15': None, 'logits': None}
    originals = {}

    def key(layer: int) -> str:
        return str(layer)

    def should_capture() -> bool:
        return (patch.mode == 'student' and patch.capture and state['token_index'] == 31
                and patch.current_layer in SELECTED_LAYERS)

    def wrap_kernel(original):
        def observed(query, value_key, value, *args, **kwargs):
            if should_capture():
                row = state['selected'].setdefault(key(patch.current_layer), {})
                row['rotated_q'] = fingerprint(query)
                row['rotated_k'] = fingerprint(value_key)
            return original(query, value_key, value, *args, **kwargs)
        return observed

    for name in ('torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule'):
        if hasattr(qmod, name):
            original = getattr(qmod, name)
            originals[name] = original
            setattr(qmod, name, wrap_kernel(original))

    original_qdq = core.CAYLEY.qwen_c128_ste

    def observed_qdq(value):
        row = state['selected'].setdefault(key(patch.current_layer), {}) if should_capture() else None
        if row is not None:
            row['pre_qdq_state'] = fingerprint(value)
        output = original_qdq(value)
        if row is not None:
            row['C128_scale'] = fingerprint(output.scale)
            row['C128_qcodes'] = fingerprint(output.codes)
            row['post_qdq_state'] = fingerprint(output.dequant)
        return output

    core.CAYLEY.qwen_c128_ste = observed_qdq
    hooks = []

    def capture_block(_module, _args, output):
        if patch.mode == 'student' and patch.capture and state['token_index'] == 31:
            tensor = output[0] if isinstance(output, tuple) else output
            state['post_block15'] = fingerprint(tensor)
        return None

    def capture_logits(_module, _args, output):
        if patch.mode == 'student' and patch.capture and state['token_index'] == 31:
            state['logits'] = fingerprint(output[:, -1, :])
        return None

    try:
        with patch:
            hooks.append(model.model.layers[15].register_forward_hook(capture_block))
            hooks.append(model.lm_head.register_forward_hook(capture_logits))
            targets = core.collect_teacher_targets(model, patch, ids, [30, 31])
            hashes = {str(layer): core.tensor_sha256(targets[31]['prev'][layer])
                      for layer in core.GDN_LAYERS}
            target_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            cache = None
            post30 = {}
            writeback = {}
            losses = {}
            with torch.no_grad():
                for index, token_id in enumerate(ids):
                    state['token_index'] = index
                    capture = index in (30, 31)
                    if capture:
                        core.install_teacher_target(patch, targets[index], device)
                    cache = core.forward_student(model, patch, token_id, cache, capture)
                    if index == 30:
                        post30 = {layer: core.tensor_sha256(patch.student_post[layer])
                                  for layer in core.GDN_LAYERS}
                        patch.clear_capture()
                    if index == 31:
                        writeback = {str(layer): post30[layer] == core.tensor_sha256(patch.student_prev[layer])
                                     for layer in core.GDN_LAYERS}
                        state_loss, functional_loss, c5, c6 = patch.captured_losses()
                        losses = {'state': float(state_loss.detach().cpu()),
                                  'functional': float(functional_loss.detach().cpu()),
                                  'C5': float(c5.detach().cpu()), 'C6': float(c6.detach().cpu())}
            if not all(writeback.values()) or len(writeback) != 24:
                raise RuntimeError('fixed-probe recurrent writeback failed')
            if state['post_block15'] is None or state['logits'] is None:
                raise RuntimeError('fixed-probe block/logits capture incomplete')
            required = {'rotated_q', 'rotated_k', 'pre_qdq_state', 'C128_scale',
                        'C128_qcodes', 'post_qdq_state'}
            if set(state['selected']) != {str(layer) for layer in SELECTED_LAYERS} or any(
                    set(row) != required for row in state['selected'].values()):
                raise RuntimeError('fixed-probe selected tensor capture incomplete')
            if patch.qdq_audit['calls'] != len(core.GDN_LAYERS) * len(ids):
                raise RuntimeError('fixed-probe C128 QDQ count mismatch')
            return {
                'sample': 'TRAIN:first-row', 'token_count': len(ids),
                'sample_token_sha256': sample_sha, 'teacher_target_sha256': target_sha,
                'selected_layers': list(SELECTED_LAYERS),
                'selected': state['selected'], 'post_block15': state['post_block15'],
                'logits': state['logits'], 'losses': losses,
                'writeback_exact_by_layer': writeback,
                'C128_QDQ_calls': patch.qdq_audit['calls'],
                'C128_QDQ_code_min': patch.qdq_audit['code_min'],
                'C128_QDQ_code_max': patch.qdq_audit['code_max'],
                'no_grad': True, 'formal_inference': False,
            }
    finally:
        for handle in hooks:
            handle.remove()
        core.CAYLEY.qwen_c128_ste = original_qdq
        for name, original in originals.items():
            setattr(qmod, name, original)
