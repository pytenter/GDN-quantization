#!/usr/bin/env python3
"""Fixed one-token, one-layer exact full-vocabulary intervention feasibility test.

This is a resource/parity gate, never a replacement for clean59 logit statistics.
Run only after all Phase-2 scalar units have completed.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

OUT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1')
I = __import__('run_injection_v2')
P = I.P
C = I.C


class CaptureTargetPatch(I.InjectionPatch):
    def _norm_pre(self, layer, module, inputs):
        if self.capture_enabled and layer == 0:
            n = len(self.readout_samples[layer])
            readout = torch.stack(self.readout_samples[layer], dim=1).to(inputs[0].dtype)
            heads, vdim = readout.shape[2:]
            zfp = inputs[1].reshape(3, n, heads, vdim)[0]
            gate = zfp.unsqueeze(0).expand(3, -1, -1, -1)
            gated = module.forward(readout.reshape(3*n*heads, vdim),
                                   gate.reshape(3*n*heads, vdim)).reshape(3, n, heads*vdim)
            self.target_projected = self._layers()[layer].linear_attn.out_proj.forward(gated)[:, 0].detach().clone()
        super()._norm_pre(layer, module, inputs)


def run_one(model, tokenizer, patch, row, token, replacement):
    device = next(model.parameters()).device
    cache, prefill = P.prefill(model, tokenizer, patch, row, device)
    layer = patch._layers()[0].linear_attn
    handle = None
    if replacement is not None:
        def hook(_module, _inputs, output):
            modified = output.clone()
            modified[0, 0] = replacement.to(device=output.device, dtype=output.dtype)
            return modified
        handle = layer.register_forward_hook(hook)
    ids = torch.tensor([[token], [token], [token]], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    start = time.time()
    try:
        # Use the same frozen manual recurrence as the state-error capture.
        # Branch 0 stays FP; QDQ affects only branches 1/2.
        patch.quantize_enabled = True
        with torch.inference_mode():
            output = model(input_ids=ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        logits = output.logits[0, 0].detach().float().clone()
    finally:
        patch.quantize_enabled = False
        if handle is not None:
            handle.remove()
        del cache, ids
        gc.collect()
        torch.cuda.empty_cache()
    return logits, elapsed, prefill


def metrics(fp, candidate):
    logp = torch.log_softmax(fp, dim=-1)
    logq = torch.log_softmax(candidate, dim=-1)
    p = logp.exp()
    top = torch.topk(fp, 2)
    qtop = torch.topk(candidate, 2)
    winner = top.indices[0]
    competitor = qtop.values[1] if qtop.indices[0] == winner else qtop.values[0]
    return {
        'full_vocabulary_kl_fp_to_injected': float((p * (logp-logq)).sum().cpu()),
        'top1_agreement': bool(qtop.indices[0] == winner),
        'fp_margin': float((top.values[0]-top.values[1]).cpu()),
        'injected_margin_on_fp_winner': float((candidate[winner]-competitor).cpu()),
        'margin_change_fp_minus_injected': float((top.values[0]-top.values[1]-candidate[winner]+competitor).cpu()),
        'relative_logit_l2': float((torch.linalg.vector_norm(candidate-fp)/torch.linalg.vector_norm(fp).clamp_min(1e-12)).cpu()),
    }


def main():
    units = sorted((OUT/'analysis').glob('shard_*/*.json'))
    if len(units) != 60:
        raise RuntimeError('Phase 2 must complete all 60 units before full-logits feasibility')
    parity = json.loads((OUT/'analysis/zero_injection_parity.json').read_text(encoding='utf-8'))
    if parity['status'] != 'PASS':
        raise RuntimeError('Phase 1 parity gate failed')
    rows = P.load_source_rows()
    row = rows[0]
    if row['problem_id'] != 'aime26_01' or int(row['seed']) != 1:
        raise RuntimeError('preregistered first trajectory changed')
    token = int(row['generated_token_ids'][0])
    model, tokenizer = C.load_model(torch)
    patch = CaptureTargetPatch(torch, model)
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    try:
        with patch:
            cache, _ = P.prefill(model, tokenizer, patch, row, device)
            patch.capture_enabled = True
            patch.quantize_enabled = True
            ids = torch.tensor([[token], [token], [token]], dtype=torch.long, device=device)
            patch.begin_chunk()
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            target = patch.target_projected.detach().clone()
            parity_local = dict(patch.max_parity)
            del output, ids, cache
            patch.capture_enabled = False
            patch.quantize_enabled = False
            patch.begin_chunk()
            gc.collect()
            torch.cuda.empty_cache()
            base, t_base, p_base = run_one(model, tokenizer, patch, row, token, None)
            zero, t_zero, p_zero = run_one(model, tokenizer, patch, row, token, target[0])
            identity, t_i, p_i = run_one(model, tokenizer, patch, row, token, target[1])
            hadamard, t_h, p_h = run_one(model, tokenizer, patch, row, token, target[2])
            zero_max_abs = float((zero-base).abs().max().cpu())
            zero_kl = metrics(base, zero)['full_vocabulary_kl_fp_to_injected']
            times = [t_base, t_zero, t_i, t_h]
            # This lower bound omits prompt prefill, state-error capture and I/O.
            projected_hours_four_gpus = 2*44505384*min(times)/(4*3600)
            peak_mb = torch.cuda.max_memory_allocated()/2**20
            numeric_pass = zero_max_abs <= 1e-3 and zero_kl <= 1e-6
            compute_pass = projected_hours_four_gpus <= 24
            memory_pass = peak_mb <= 23000
            status = ('FULL_LOGITS_PARITY_FAILED' if not numeric_pass else
                      'FEASIBLE_FOR_FULL_CLEAN59' if compute_pass and memory_pass else
                      'NOT_FEASIBLE_FULL_CLEAN59')
            result = {
                'status': status, 'scope': 'fixed first clean59 trajectory, first generated input token, first GDN layer',
                'trajectory': {'problem_id': row['problem_id'], 'seed': int(row['seed'])},
                'token_position': 0, 'layer': 0,
                'zero_injection_logit_max_abs': zero_max_abs,
                'zero_injection_kl': zero_kl,
                'local_zero_injection_max_abs': parity_local,
                'fp_prefill_parity': [p_base, p_zero, p_i, p_h],
                'one_token_forward_seconds': times,
                'peak_gpu_memory_mb': peak_mb,
                'full_clean59_four_gpu_lower_bound_hours': projected_hours_four_gpus,
                'feasibility_thresholds': {'zero_logit_max_abs': 1e-3, 'zero_kl': 1e-6,
                                           'projected_four_gpu_hours': 24, 'peak_gpu_memory_mb': 23000},
                'pilot_only_metrics': {'identity': metrics(base, identity), 'hadamard': metrics(base, hadamard)},
                'full_clean59_logit_metrics': 'NOT_COMPUTED' if status != 'FEASIBLE_FOR_FULL_CLEAN59' else 'PENDING',
                'new_generation': False, 'training': False,
            }
            I.save_json(OUT/'analysis/full_logits_feasibility.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        del patch, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
