#!/usr/bin/env python3
"""Deterministic online native-basis state-error injection into frozen GDN modules.

No full recurrent-state or logit tensors are serialized. Existing recurrent,
quantizer, rotation, and model implementations are imported without edits.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OUT = ROOT / 'experiments/QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1'
GEOM = ROOT / 'experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1'
STABILITY = ROOT / 'experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1'
EPS = 1e-12
CHUNK = 64
STAGES = ('state', 'readout', 'rmsnorm', 'gate', 'output_projection')
CONDITIONS = ('identity', 'hadamard')


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load_file(STABILITY / 'replay_stability_v2.py', 'frozen_stability_for_injection')
P = S.P
C = S.C


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def rel_error(candidate, reference):
    difference = (candidate.float() - reference.float()).reshape(candidate.shape[0], -1)
    baseline = reference.float().reshape(reference.shape[0], -1)
    absolute = torch.linalg.vector_norm(difference, dim=-1)
    denominator = torch.linalg.vector_norm(baseline, dim=-1)
    return absolute, absolute / denominator.clamp_min(EPS), denominator


class InjectionPatch(S.StabilityPatch):
    """Observe fixed FP state, inject either observed error, and recompute one GDN block."""

    def __init__(self, torch_module, model):
        super().__init__(torch_module, model)
        self.max_parity = {'readout': 0.0, 'gate': 0.0, 'output_projection': 0.0}
        self.q_current = None
        self.current_dtype = None

    def begin_chunk(self):
        super().begin_chunk()
        self.readout_samples = {}
        self.state_norms = {}
        self.error_calls = {}
        self.fp_local_gate = {}
        self.fp_local_projection = {}
        self.norm_expected = {}

    def _remember_query(self, query, use_norm):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        q = qmod.l2norm(query, dim=-1, eps=1e-6) if use_norm else query
        self.q_current = q[0].detach().float() * (query.shape[-1] ** -0.5)
        self.current_dtype = query.dtype

    def _recurrent(self, query, key, value, g, beta, initial_state,
                   output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        if self.capture_enabled:
            self._remember_query(query, use_qk_l2norm_in_kernel)
        return super()._recurrent(query, key, value, g, beta, initial_state,
                                  output_final_state, use_qk_l2norm_in_kernel, **kwargs)

    def _chunk(self, query, key, value, g, beta, chunk_size=64, initial_state=None,
               output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if self.capture_enabled:
            self._remember_query(query, use_qk_l2norm_in_kernel)
        return super()._chunk(query, key, value, g, beta, chunk_size, initial_state,
                              output_final_state, use_qk_l2norm_in_kernel, **kwargs)

    def _condition_errors(self, state):
        en, eh, fp = super()._condition_errors(state)
        if not self.capture_enabled:
            return en, eh, fp
        layer = int(self.current_layer)
        call = self.error_calls.get(layer, 0)
        self.error_calls[layer] = call + 1
        if call % 2:
            index = call // 2
            q = self.q_current[index]
            # Recompute all three readouts with the same FP query and native basis.
            reads = torch.stack(tuple((candidate * q.unsqueeze(-1)).sum(dim=-2)
                                      for candidate in (fp, fp + en, fp + eh)), dim=0)
            self.readout_samples.setdefault(layer, []).append(reads.detach())
            nfp = torch.linalg.vector_norm(fp).clamp_min(EPS)
            norms = torch.stack((torch.linalg.vector_norm(en) / nfp,
                                 torch.linalg.vector_norm(eh) / nfp)).detach()
            self.state_norms.setdefault(layer, []).append(norms)
        return en, eh, fp

    @staticmethod
    def _rmsnorm_only(module, readout):
        h = readout.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + module.variance_epsilon)
        return module.weight * h.to(readout.dtype)

    def _norm_pre(self, layer, module, inputs):
        if not self.capture_enabled:
            return
        rec = self.pending.get(layer)
        if rec is None:
            raise RuntimeError(f'missing recurrent capture at layer {layer}')
        n = len(rec['m0']['INT8_C128'])
        if self.error_calls.get(layer) != 2*n or len(self.readout_samples.get(layer, ())) != n:
            raise RuntimeError(f'error/readout capture cadence differs at layer {layer}')
        readout = torch.stack(self.readout_samples[layer], dim=1).to(inputs[0].dtype)
        heads, vdim = readout.shape[2:]
        actual_fp = inputs[0].reshape(3, n, heads, vdim)[0]
        mismatch = (readout[0].float() - actual_fp.float()).abs().max().item()
        self.max_parity['readout'] = max(self.max_parity['readout'], mismatch)
        if mismatch > 0.001:
            raise RuntimeError(f'zero injection readout parity failed at layer {layer}: {mismatch}')
        zfp = inputs[1].reshape(3, n, heads, vdim)[0].detach()
        la = self._layers()[layer].linear_attn
        names = ('fp', 'identity', 'hadamard')
        stages = {stage: {} for stage in STAGES}
        normals = []
        gates = []
        projections = []
        for branch, name in enumerate(names):
            current = readout[branch]
            norm = self._rmsnorm_only(module, current)
            gated = module.forward(current.reshape(n*heads, vdim),
                                   zfp.reshape(n*heads, vdim)).reshape(n, heads, vdim)
            projected = la.out_proj.forward(gated.reshape(n, -1))
            normals.append(norm)
            gates.append(gated)
            projections.append(projected)
        self.fp_local_gate[layer] = gates[0].detach()
        self.fp_local_projection[layer] = projections[0].detach()
        self.norm_expected[layer] = n
        state_norm = torch.stack(self.state_norms[layer], dim=0)
        for branch, name in ((1, 'identity'), (2, 'hadamard')):
            for stage, current, baseline in (
                ('readout', readout[branch], readout[0]),
                ('rmsnorm', normals[branch], normals[0]),
                ('gate', gates[branch], gates[0]),
                ('output_projection', projections[branch], projections[0]),
            ):
                absolute, relative, denominator = rel_error(current, baseline)
                stages[stage][name] = {
                    'absolute': absolute.detach().cpu().numpy(),
                    'relative': relative.detach().cpu().numpy(),
                    'fp_norm': denominator.detach().cpu().numpy(),
                }
            stages['state'][name] = {
                'absolute': rec['m0']['INT8_C128' if branch == 1 else 'INT8_C128_KEY_HADAMARD'].detach().float().cpu().numpy(),
                'relative': state_norm[:, branch-1].float().cpu().numpy(),
                'fp_norm': np.full(n, np.nan, dtype=np.float32),
            }
        self.chunk_metrics[int(layer)] = stages
        self.layers_seen.add(int(layer))
        del self.pending[layer]
        del self.readout_samples[layer]
        del self.state_norms[layer]

    def _norm_post(self, layer, module, inputs, output):
        if not self.capture_enabled:
            return
        n = self.norm_expected[layer]
        actual = output.reshape(3, n, -1)[0]
        expected = self.fp_local_gate[layer].reshape(n, -1)
        mismatch = (actual.float() - expected.float()).abs().max().item()
        self.max_parity['gate'] = max(self.max_parity['gate'], mismatch)
        if mismatch > 0.001:
            raise RuntimeError(f'zero injection norm/gate parity failed at layer {layer}: {mismatch}')
        del self.fp_local_gate[layer]

    def _layer_post(self, layer, module, inputs, output):
        if not self.capture_enabled:
            return
        expected = self.fp_local_projection[layer]
        actual = output[0].reshape_as(expected)
        mismatch = (actual.float() - expected.float()).abs().max().item()
        self.max_parity['output_projection'] = max(self.max_parity['output_projection'], mismatch)
        if mismatch > 0.001:
            raise RuntimeError(f'zero injection projection parity failed at layer {layer}: {mismatch}')
        del self.fp_local_projection[layer]
        del self.norm_expected[layer]

    def __enter__(self):
        super().__enter__()
        for layer in C.GDN_LAYERS:
            la = self._layers()[layer].linear_attn
            def norm_post(mod, inputs, output, idx=layer):
                self._norm_post(idx, mod, inputs, output)

            def layer_post(mod, inputs, output, idx=layer):
                self._layer_post(idx, mod, inputs, output)

            self.handles.append(la.norm.register_forward_hook(norm_post))
            self.handles.append(la.register_forward_hook(layer_post))
        return self


def capture_trajectory(model, tokenizer, patch, row, limit=None):
    tokens = list(map(int, row['generated_token_ids']))
    if limit is not None:
        tokens = tokens[:limit]
    if len(tokens) < 2:
        raise RuntimeError('trajectory too short')
    device = next(model.parameters()).device
    cache, prefill_parity = P.prefill(model, tokenizer, patch, row, device)
    patch.capture_enabled = True
    patch.quantize_enabled = True
    collected = {stage: {cond: {'absolute': [], 'relative': [], 'fp_norm': []}
                         for cond in CONDITIONS} for stage in STAGES}
    by_layer = {int(layer): {stage: {cond: [] for cond in CONDITIONS} for stage in STAGES}
                for layer in C.GDN_LAYERS}
    start_time = time.time()
    try:
        for start in range(0, len(tokens)-1, CHUNK):
            chunk = tokens[start:min(start+CHUNK, len(tokens)-1)]
            ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
            patch.begin_chunk()
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if set(patch.chunk_metrics) != set(C.GDN_LAYERS):
                raise RuntimeError('GDN layer grid incomplete')
            for layer in C.GDN_LAYERS:
                for stage in STAGES:
                    for cond in CONDITIONS:
                        for kind in ('absolute', 'relative', 'fp_norm'):
                            values = patch.chunk_metrics[layer][stage][cond][kind]
                            if kind == 'relative':
                                by_layer[layer][stage][cond].append(values)
                            if stage == 'state' and kind == 'fp_norm':
                                continue
                            collected[stage][cond][kind].append(values)
            del ids, output
            if start % 4096 == 0:
                print(json.dumps({'event': 'injection_progress', 'problem_id': row['problem_id'],
                                  'seed': row['seed'], 'tokens': start+len(chunk),
                                  'total': len(tokens)-1,
                                  'tokens_per_s': (start+len(chunk))/max(time.time()-start_time, 1e-6)}), flush=True)
    finally:
        patch.capture_enabled = False
        patch.quantize_enabled = False
        patch.begin_chunk()
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    metrics = {}
    nonfinite = {}
    denominator_floor = {}
    for stage in STAGES:
        metrics[stage] = {}
        nonfinite[stage] = {}
        denominator_floor[stage] = {}
        for cond in CONDITIONS:
            metrics[stage][cond] = {}
            for kind in ('absolute', 'relative'):
                array = np.concatenate(collected[stage][cond][kind]).astype(np.float64)
                bad = int(np.count_nonzero(~np.isfinite(array)))
                nonfinite[stage][f'{cond}_{kind}'] = bad
                metrics[stage][cond][kind] = float(np.median(array[np.isfinite(array)])) if bad < len(array) else None
            if stage != 'state':
                denom = np.concatenate(collected[stage][cond]['fp_norm']).astype(np.float64)
                denominator_floor[stage][cond] = int(np.count_nonzero(denom <= EPS))
    layers = {}
    for layer in C.GDN_LAYERS:
        layers[str(layer)] = {}
        for stage in STAGES:
            layers[str(layer)][stage] = {}
            for cond in CONDITIONS:
                array = np.concatenate(by_layer[layer][stage][cond]).astype(np.float64)
                good = array[np.isfinite(array)]
                layers[str(layer)][stage][cond] = float(np.median(good)) if len(good) else None
    return {
        'problem_id': row['problem_id'], 'seed': int(row['seed']),
        'canonical_index': int(row['canonical_index']), 'tokens': len(tokens)-1,
        'layers': len(C.GDN_LAYERS), 'new_generation': False,
        'source_token_ids_sha256': hashlib.sha256(json.dumps(list(map(int, row['generated_token_ids'])), separators=(',', ':')).encode()).hexdigest(),
        'prefill_parity': prefill_parity, 'zero_injection_parity_max_abs': dict(patch.max_parity),
        'metrics': metrics, 'nonfinite': nonfinite, 'denominator_floor': denominator_floor,
        'by_layer': layers, 'elapsed_seconds': time.time()-start_time,
        'peak_gpu_memory_mb': torch.cuda.max_memory_allocated()/2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pilot', 'shard'), required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=4)
    args = parser.parse_args()
    audit = json.loads((OUT/'analysis/replay_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS':
        raise RuntimeError('audit gate not PASS')
    rows = P.load_source_rows()
    clean_id = {tuple((r['problem_id'], int(r['seed']))) for r in __import__('csv').DictReader((GEOM/'analysis/clean59_paired.csv').open(encoding='utf-8'))}
    selected = [r for r in rows if (r['problem_id'], int(r['seed'])) in clean_id]
    if len(selected) != 59:
        raise RuntimeError('clean59 source selection failed')
    if args.mode == 'shard':
        pilot = json.loads((OUT/'analysis/zero_injection_parity.json').read_text(encoding='utf-8'))
        if pilot['status'] != 'PASS':
            raise RuntimeError('zero-injection parity gate not PASS')
    model, tokenizer = C.load_model(torch)
    patch = InjectionPatch(torch, model)
    try:
        with patch:
            if args.mode == 'pilot':
                row = selected[0]
                result = capture_trajectory(model, tokenizer, patch, row, limit=128)
                status = 'PASS' if (max(result['zero_injection_parity_max_abs'].values()) <= 0.001
                                    and all(v == 0 for stage in result['nonfinite'].values() for v in stage.values())) else 'FAIL'
                feasibility = {
                    'status': status, 'sample': {'problem_id': row['problem_id'], 'seed': row['seed'], 'tokens': 127},
                    'peak_gpu_memory_mb': result['peak_gpu_memory_mb'],
                    'elapsed_seconds': result['elapsed_seconds'], 'disk_free_gib': shutil.disk_usage(OUT).free/2**30,
                    'full_logits_feasibility': 'NOT_YET_TESTED',
                }
                save_json(OUT/'analysis/feasibility.json', feasibility)
                save_json(OUT/'analysis/zero_injection_parity.json', {
                    'status': status, 'max_abs': result['zero_injection_parity_max_abs'],
                    'nonfinite': result['nonfinite'], 'sample': feasibility['sample'],
                    'source_token_ids_sha256': result['source_token_ids_sha256'],
                })
                save_json(OUT/'analysis/pilot_local_damage.json', result)
                print(json.dumps({'pilot': status, 'feasibility': feasibility, 'parity': result['zero_injection_parity_max_abs']}))
            else:
                assigned = [(i, row) for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
                unit_dir = OUT / f'analysis/shard_{args.shard_index:02d}'
                unit_dir.mkdir(parents=True, exist_ok=True)
                for i, row in assigned:
                    target = unit_dir / f'{i:03d}_{row["problem_id"]}_seed{int(row["seed"])}.json'
                    if target.exists():
                        previous = json.loads(target.read_text(encoding='utf-8'))
                        if (previous['problem_id'], int(previous['seed'])) != (row['problem_id'], int(row['seed'])):
                            raise RuntimeError(f'completed unit mismatch: {target}')
                        continue
                    if shutil.disk_usage(OUT).free < 20*2**30:
                        raise RuntimeError('disk safety stop: below 20 GiB free')
                    result = capture_trajectory(model, tokenizer, patch, row)
                    save_json(target, result)
                    print(json.dumps({'event': 'unit_complete', 'index': i, 'problem_id': row['problem_id'],
                                      'seed': row['seed'], 'tokens': result['tokens']}), flush=True)
    finally:
        del patch, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
