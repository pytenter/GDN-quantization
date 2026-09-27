#!/usr/bin/env python3
"""One frozen V2 teacher-forward condition, with process-local capture hooks.

This is diagnostic only. It never trains or changes the installed model package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
V1 = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
MODEL_SOURCE = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
MODEL_CONFIG = Path('/data/zypan/modelscope_models/Qwen3.5-9B/config.json')
sys.path.insert(0, str(REPO))
from experiments.qwen_gdn.recurrent_training.QWEN_GDN_DISTRIBUTED_RECURRENT_TRAINING_V2.runtime.deterministic_ops import qwen_chunk_reference as candidate  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_summary(t: torch.Tensor) -> dict:
    c = t.detach().contiguous().cpu()
    f = c.float()
    return {
        'shape': list(c.shape), 'stride': list(t.stride()), 'contiguous': t.is_contiguous(),
        'dtype': str(c.dtype), 'device': str(t.device), 'numel': c.numel(),
        'sha256': digest(tensor_bytes(c)),
        'min': float(f.min()) if f.numel() else None,
        'max': float(f.max()) if f.numel() else None,
        'mean': float(f.mean()) if f.numel() else None,
        'std': float(f.std(unbiased=False)) if f.numel() else None,
        'l2': float(torch.linalg.vector_norm(f)) if f.numel() else None,
        'nonfinite': int((~torch.isfinite(f)).sum()),
    }


class Recorder:
    def __init__(self):
        self.values: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, dict] = {}
        self.active_first_chunk = False
        self.chunk_calls = 0
        self.norm_calls = 0
        self.conv_calls = 0

    def add(self, name: str, value: torch.Tensor):
        if not isinstance(value, torch.Tensor) or name in self.values:
            return
        self.metadata[name] = tensor_summary(value)
        self.values[name] = value.detach().contiguous().cpu().clone()


def import_v1():
    spec = importlib.util.spec_from_file_location('causal_read_only_v1', V1)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_capture(model, rec: Recorder, prefix: int):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod
    handles = []
    gdn_name, gdn = next((n, m) for n, m in model.named_modules()
                         if isinstance(m, qmod.Qwen3_5GatedDeltaNet) and m.layer_idx == 0)
    decoder = model.get_submodule(gdn_name.rsplit('.linear_attn', 1)[0])

    def pre(name):
        def hook(_module, args):
            if args and isinstance(args[0], torch.Tensor):
                rec.add(name, args[0])
            if name == 'gdn.norm_input' and len(args) > 1 and isinstance(args[1], torch.Tensor):
                rec.add('gdn.gate_input_z', args[1])
        return hook

    def post(name):
        def hook(_module, _args, output):
            if isinstance(output, torch.Tensor):
                rec.add(name, output)
        return hook

    handles.append(model.get_input_embeddings().register_forward_hook(post('embedding.output')))
    for mod, name in [(decoder, 'layer0.input'), (gdn, 'gdn.input'),
                      (gdn.norm, 'gdn.norm_input'), (gdn.out_proj, 'gdn.out_proj_input')]:
        handles.append(mod.register_forward_pre_hook(pre(name)))
    for mod, name in [(decoder.input_layernorm, 'layer0.pre_norm_output'),
                      (gdn.in_proj_qkv, 'gdn.qkv_projection'),
                      (gdn.in_proj_z, 'gdn.z_projection'),
                      (gdn.in_proj_b, 'gdn.beta_projection'),
                      (gdn.in_proj_a, 'gdn.decay_projection'),
                      (gdn.norm, 'gdn.norm_output'),
                      (gdn.out_proj, 'gdn.out_proj_output'),
                      (gdn, 'gdn.output'), (decoder, 'layer0.output')]:
        handles.append(mod.register_forward_hook(post(name)))

    original_conv = qmod.causal_conv1d_fn
    def conv_capture(*args, **kwargs):
        out = original_conv(*args, **kwargs)
        if rec.conv_calls == 0:
            rec.add('gdn.short_conv_mixed', out)
            if isinstance(out, torch.Tensor):
                rec.add('gdn.q_short_conv', out[:, :gdn.key_dim])
                rec.add('gdn.k_short_conv', out[:, gdn.key_dim:2*gdn.key_dim])
                rec.add('gdn.v_short_conv', out[:, 2*gdn.key_dim:])
        rec.conv_calls += 1
        return out
    qmod.causal_conv1d_fn = conv_capture

    original_l2norm = qmod.l2norm
    candidate_l2norm = candidate.l2norm
    def norm_capture(*args, **kwargs):
        out = original_l2norm(*args, **kwargs)
        if rec.active_first_chunk and rec.norm_calls < 2:
            rec.add('gdn.normalized_q' if rec.norm_calls == 0 else 'gdn.normalized_k', out)
            rec.norm_calls += 1
        return out
    qmod.l2norm = norm_capture
    candidate.l2norm = norm_capture

    original_cumsum = torch.Tensor.cumsum
    def cumsum_capture(tensor, *args, **kwargs):
        if rec.active_first_chunk:
            rec.add('gdn.prefix_input', tensor)
        out = original_cumsum(tensor, *args, **kwargs)
        if rec.active_first_chunk:
            rec.add('gdn.prefix_output', out)
        return out
    torch.Tensor.cumsum = cumsum_capture

    original_fixed = candidate.fixed_left_to_right_cumsum
    def fixed_capture(tensor, *args, **kwargs):
        if rec.active_first_chunk:
            rec.add('gdn.prefix_input', tensor)
        out = original_fixed(tensor, *args, **kwargs)
        if rec.active_first_chunk:
            rec.add('gdn.prefix_output', out)
        return out
    candidate.fixed_left_to_right_cumsum = fixed_capture

    original_chunk = qmod.torch_chunk_gated_delta_rule
    selected = candidate.deterministic_torch_chunk_gated_delta_rule if prefix else original_chunk
    def chunk_capture(query, key, value, g, beta, *args, **kwargs):
        first = rec.chunk_calls == 0
        rec.chunk_calls += 1
        if first:
            for name, val in [('gdn.core_query', query), ('gdn.core_key', key),
                              ('gdn.core_value', value), ('gdn.core_decay_g', g),
                              ('gdn.core_beta', beta)]:
                rec.add(name, val)
        rec.active_first_chunk = first
        try:
            out = selected(query, key, value, g, beta, *args, **kwargs)
        finally:
            rec.active_first_chunk = False
        if first:
            rec.add('gdn.core_output', out[0])
            if out[1] is not None:
                rec.add('gdn.pre_qdq_state', out[1])
        return out
    qmod.torch_chunk_gated_delta_rule = chunk_capture

    def restore():
        for h in handles:
            h.remove()
        qmod.torch_chunk_gated_delta_rule = original_chunk
        qmod.causal_conv1d_fn = original_conv
        qmod.l2norm = original_l2norm
        candidate.l2norm = candidate_l2norm
        candidate.fixed_left_to_right_cumsum = original_fixed
        torch.Tensor.cumsum = original_cumsum
    return restore, {'first_gdn_module': gdn_name,
                     'canonical_chunk_callable': repr(original_chunk),
                     'candidate_chunk_callable': repr(selected)}


def run(condition: str):
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one idle GPU must be visible')
    out_path = ROOT / 'analysis' / f'{condition}_summary.json'
    tmp_path = ROOT / 'analysis' / 'temporary' / f'{condition}.pt'
    if out_path.exists() or tmp_path.exists():
        raise FileExistsError('condition artifact already exists; no automatic retry')
    prefix, strict = condition[1] == '1', condition[3] == '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(strict)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    rec = Recorder()
    result = {'task': ROOT.name, 'condition': condition, 'status': 'RUNNING',
              'prefix_candidate': prefix, 'strict_algorithms': strict,
              'source_sha256': {'modeling': digest(MODEL_SOURCE.read_bytes()),
                                'v1_c5': digest(V1.read_bytes()),
                                'config': digest(MODEL_CONFIG.read_bytes())},
              'environment': {'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES'),
                              'CUBLAS_WORKSPACE_CONFIG': os.getenv('CUBLAS_WORKSPACE_CONFIG'),
                              'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn_benchmark': torch.backends.cudnn.benchmark,
                              'cudnn_deterministic': torch.backends.cudnn.deterministic,
                              'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
              'capture_order': []}
    try:
        v1 = import_v1()
        row = v1.corpus_rows('TRAIN')[0]
        model, tokenizer, bank, device = v1.load_model_bank()
        v1.freeze_model(model)
        ids, _ = v1.tokenize(tokenizer, row)
        if len(ids) < 64:
            raise RuntimeError('frozen V2 sample has fewer than 64 tokens')
        ids = ids[:64]
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        target = torch.tensor([ids[-1]], device=device, dtype=torch.long)
        rec.add('input_ids', input_ids)
        result['frozen_input'] = {'sample_id': str(row.get('id', 'TRAIN:first-row')),
                                  'raw_text_sha256': digest(row['raw_text'].encode()),
                                  'token_ids_sha256': digest(json.dumps(ids, separators=(',', ':')).encode()),
                                  'input_ids_sha256': rec.metadata['input_ids']['sha256'],
                                  'teacher_target_sha256': digest(tensor_bytes(target)),
                                  'token_count': len(ids), 'target_token_id': ids[-1]}
        result['rotation'] = {'theta_max_abs': max(float(p.detach().abs().max()) for p in bank.parameters()),
                              'bank_parameter_count': sum(p.numel() for p in bank.parameters())}
        restore, dispatch = install_capture(model, rec, int(prefix))
        result['dispatch'] = dispatch
        try:
            with torch.no_grad():
                output = model(input_ids=input_ids, use_cache=True)
                logits = output.logits[:, -1, :].detach()
                rec.add('teacher.last_logits', logits)
                loss = F.cross_entropy(logits.float(), target)
                rec.add('teacher.loss', loss)
                for layer in v1.GDN_LAYERS:
                    state = output.past_key_values.layers[layer].recurrent_states[0].detach()
                    rec.add(f'layer{layer}.pre_qdq_state', state)
                    qdq = v1.CAYLEY.qwen_c128_qdq(state.float())
                    rec.add(f'layer{layer}.c128_scale', qdq.scale)
                    rec.add(f'layer{layer}.c128_qcodes', qdq.codes)
                    rec.add(f'layer{layer}.post_qdq_state', qdq.dequant)
            result['status'] = 'COMPLETE'
            result['loss'] = float(loss)
            result['gdn_layer_count'] = len(v1.GDN_LAYERS)
        finally:
            restore()
    except Exception as exc:
        result['status'] = 'BLOCKED_AT_CUMSUM' if 'cumsum' in str(exc).lower() else 'ERROR'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc()}
    result['captures'] = rec.metadata
    result['capture_order'] = list(rec.metadata)
    result['chunk_call_count'] = rec.chunk_calls
    result['conv_call_count'] = rec.conv_calls
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec.values, tmp_path)
    with out_path.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'condition': condition, 'status': result['status'],
                      'captures': len(rec.values), 'chunk_calls': rec.chunk_calls}), flush=True)
    if result['status'] == 'ERROR':
        raise RuntimeError(result['error']['message'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True, choices=['P0R0', 'P1R0', 'P0R1', 'P1R1'])
    run(parser.parse_args().condition)
