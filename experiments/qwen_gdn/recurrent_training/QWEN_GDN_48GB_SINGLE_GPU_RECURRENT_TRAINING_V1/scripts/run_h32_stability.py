#!/usr/bin/env python3
"""Eight bounded H32 C5 diagnostic updates; gate after update five."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import torch

import run_horizon_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path('/root/autodl-tmp/qwen_48gb_rotation_h32_8update_v1.pt')
BASELINE_SPAN_LIMIT_BYTES = 256 * 1024 * 1024


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if (ROOT / 'analysis/H32_stability.json').exists() or (ROOT / 'analysis/H32_stability.error.json').exists():
        raise RuntimeError('H32 stability has already been attempted')
    if CHECKPOINT.exists():
        raise RuntimeError('diagnostic checkpoint already exists')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA device required')
    h32 = json.loads((ROOT / 'analysis/H32.json').read_text())
    if h32['status'] != 'PASS' or not h32['free_margin_at_least_5_percent']:
        raise RuntimeError('H32 one-update and >=5% margin gates required')
    forward = json.loads((ROOT / 'analysis/forward_parity.json').read_text())
    if not forward['source_model_input_exact'] or not forward['all_finite']:
        raise RuntimeError('forward structural gate not satisfied')

    core = smoke.load_core()
    torch.manual_seed(core.TRAIN_SEED)
    model, tokenizer, bank, device = smoke.load_model_bank_without_dispatch(core)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('base model is not frozen')
    if sum(p.numel() for p in bank.parameters()) != 195072:
        raise RuntimeError('rotation parameter count mismatch')
    if any(bool(torch.count_nonzero(p).item()) for p in bank.parameters()):
        raise RuntimeError('rotation is not theta=0 initialized')
    before_weights = smoke.weight_fingerprints(model)
    ids, _ = core.tokenize(tokenizer, core.corpus_rows('TRAIN')[0])
    ids = ids[:32]
    if len(ids) != 32:
        raise RuntimeError('TRAIN:first-row shorter than 32 tokens')
    sample_hash = core.tensor_sha256(torch.tensor(ids, dtype=torch.long))
    if sample_hash != h32['sample_token_hash']:
        raise RuntimeError('diagnostic sample changed')
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    hadamard = core.hadamard(device)
    updates = []
    five_gate = 'NOT_RUN'
    with core.RecurrentExperimentPatch(model, bank, hadamard) as patch:
        targets = core.collect_teacher_targets(model, patch, ids, [30, 31])
        teacher_hashes = {str(layer): core.tensor_sha256(targets[31]['prev'][layer])
                          for layer in core.GDN_LAYERS}
        teacher_hash = hashlib.sha256(json.dumps(teacher_hashes, sort_keys=True).encode()).hexdigest()
        if teacher_hash != h32['teacher_target_hash']:
            raise RuntimeError('teacher target changed from one-update gate')
        for update in range(1, 9):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = smoke.memory(device)
            qdq_before = patch.qdq_audit['calls']
            cache = None
            post30 = {}
            loss_value = None
            writeback = {}
            for index, token_id in enumerate(ids):
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
                    state, functional, c5, c6 = patch.captured_losses()
                    if not bool(torch.isfinite(c5).item()):
                        raise FloatingPointError(f'nonfinite C5 loss update={update}')
                    loss_value = float(c5.detach().cpu())
                    memory_forward = smoke.memory(device)
                    c5.backward()
                    memory_backward = smoke.memory(device)
                    del state, functional, c5, c6
                if index == 31:
                    core.detach_cache(cache)
            if loss_value is None or not all(writeback.values()):
                raise RuntimeError(f'capture/writeback gate failed update={update}')
            norms = {}
            for layer in core.GDN_LAYERS:
                grad = bank.layer(layer).theta.grad
                if grad is None or not bool(torch.isfinite(grad).all().item()):
                    raise FloatingPointError(f'missing/nonfinite gradient update={update} layer={layer}')
                norms[str(layer)] = float(torch.linalg.vector_norm(grad).detach().cpu())
            if not any(value > 0 for value in norms.values()):
                raise FloatingPointError(f'all gradients zero update={update}')
            raw_grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            if not bool(torch.isfinite(raw_grad).item()):
                raise FloatingPointError(f'nonfinite clipped gradient update={update}')
            optimizer.step()
            memory_optimizer = smoke.memory(device)
            orthogonality = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
            if orthogonality['status'] != 'PASS':
                raise RuntimeError(f'orthogonality gate failed update={update}')
            if before_weights != smoke.weight_fingerprints(model):
                raise RuntimeError(f'base model changed update={update}')
            theta_l2 = math.sqrt(sum(float(p.detach().float().square().sum().cpu())
                                     for p in bank.parameters()))
            patch.clear_capture()
            del cache
            gc.collect()
            torch.cuda.empty_cache()
            memory_released = smoke.memory(device)
            row = {
                'update': update, 'loss': loss_value,
                'gradient_norm_by_layer': norms,
                'gradient_norm_before_clip': float(raw_grad.detach().cpu()),
                'theta_displacement_l2': theta_l2,
                'orthogonality': orthogonality,
                'recurrent_writeback_exact_by_layer': writeback,
                'C128_QDQ_calls_this_update': patch.qdq_audit['calls'] - qdq_before,
                'memory': {'before_rollout': memory_before, 'after_forward': memory_forward,
                           'after_backward': memory_backward, 'after_optimizer': memory_optimizer,
                           'after_graph_release': memory_released},
                'peak_allocated_bytes': memory_released['peak_allocated_bytes'],
                'peak_reserved_bytes': memory_released['peak_reserved_bytes'],
            }
            updates.append(row)
            print(json.dumps({'update': update, 'loss': loss_value,
                              'peak_reserved_GiB': row['peak_reserved_bytes'] / 1024**3,
                              'end_allocated_GiB': memory_released['allocated_bytes'] / 1024**3}), flush=True)
            if update == 5:
                baselines = [item['memory']['after_graph_release']['allocated_bytes'] for item in updates]
                span = max(baselines) - min(baselines)
                five_gate = 'PASS' if span <= BASELINE_SPAN_LIMIT_BYTES else 'FAIL_MEMORY_CREEP'
                if five_gate != 'PASS':
                    break

    baselines = [item['memory']['after_graph_release']['allocated_bytes'] for item in updates]
    result = {
        'condition': 'C5', 'horizon': 32, 'sample': 'TRAIN:first-row',
        'sample_token_hash': sample_hash, 'teacher_target_hash': teacher_hash,
        'updates_completed': len(updates), 'updates': updates,
        'baseline_allocated_bytes_by_update': baselines,
        'baseline_span_limit_bytes_preregistered': BASELINE_SPAN_LIMIT_BYTES,
        'baseline_span_bytes': max(baselines) - min(baselines),
        'H32_x5_sustained_gate': five_gate,
        'memory_leak': 'NO_OBSERVED' if five_gate == 'PASS' and
                       max(baselines) - min(baselines) <= BASELINE_SPAN_LIMIT_BYTES else 'SUSPECTED',
        'base_model_unchanged': True,
        'formal_C5_C6_training_started': False,
        'AIME_started_on_vgpu': False,
    }
    if len(updates) == 8 and five_gate == 'PASS':
        payload = {
            'task': 'QWEN_GDN_48GB_SINGLE_GPU_RECURRENT_TRAINING_V1',
            'diagnostic_only': True, 'condition': 'C5', 'objective': 'DENSE_STATE',
            'training_mode': 'REAL_RECURRENT_INT8_STATE_WRITEBACK',
            'seed': core.TRAIN_SEED, 'lr': 0.003, 'weight_decay': 0.0,
            'gradient_horizon': 32, 'diagnostic_updates': 8,
            'sample_token_hash': sample_hash, 'layer_ids': core.GDN_LAYERS,
            'bank': {key: value.detach().cpu().contiguous() for key, value in bank.state_dict().items()},
        }
        torch.save(payload, CHECKPOINT)
        result['rotation_checkpoint'] = {
            'absolute_path': str(CHECKPOINT), 'size_bytes': CHECKPOINT.stat().st_size,
            'sha256': sha(CHECKPOINT), 'only_rotation_parameters': True,
        }
    smoke.save_once(ROOT / 'analysis/H32_stability.json', result)
    if five_gate != 'PASS':
        raise RuntimeError('H32 sustained memory baseline exceeded preregistered span')
    print(json.dumps({'H32_x5_sustained_gate': five_gate,
                      'updates_completed': len(updates),
                      'baseline_span_bytes': result['baseline_span_bytes'],
                      'rotation_checkpoint': result.get('rotation_checkpoint')}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        detail = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        path = ROOT / 'analysis/H32_stability.error.json'
        if not path.exists():
            smoke.save_once(path, detail)
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}), flush=True)
        raise
