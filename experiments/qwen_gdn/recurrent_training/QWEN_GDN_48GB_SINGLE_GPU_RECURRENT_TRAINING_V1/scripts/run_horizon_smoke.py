#!/usr/bin/env python3
"""One canonical C5 update on a fixed non-AIME TRAIN prefix, in a fresh process."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

MIRROR = Path('/root/autodl-tmp/canonical_source')
SOURCE = MIRROR / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
TRANSFORMERS = MIRROR / 'transformers-qwen35/src'
MODEL = Path('/root/autodl-tmp/models/Qwen3.5-9B')
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('QWEN_TRANSFORMERS_SRC', str(MIRROR / 'transformers-qwen35'))
os.environ.setdefault('QWEN_MODEL_PATH', str(MODEL))
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(TRANSFORMERS))

import torch  # noqa: E402


def load_core():
    spec = importlib.util.spec_from_file_location('qwen_48gb_canonical_c5', SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError('canonical C5 source unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model_bank_without_dispatch(core):
    """Validated full-model placement; C5 patch, loss and optimizer stay canonical."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    model.to('cuda:0')
    core.freeze_model(model)
    device = model.get_input_embeddings().weight.device
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS).to(device)
    return model, tokenizer, bank, device


def digest_tensor(tensor: torch.Tensor, limit: int | None = None) -> str:
    flat = tensor.detach().flatten()
    if limit is not None:
        flat = flat[:limit]
    return hashlib.sha256(flat.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory(device: torch.device) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        'allocated_bytes': torch.cuda.memory_allocated(device),
        'reserved_bytes': torch.cuda.memory_reserved(device),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'device_free_bytes': free, 'device_total_bytes': total,
    }


def save_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def check_prerequisites(horizon: int) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError('VGPU_VISIBLE_DEVICE_COUNT must be exactly one')
    source_gate = json.loads((ROOT / 'analysis/source_parity.json').read_text())
    hardware_gate = json.loads((ROOT / 'analysis/hardware_gate.json').read_text())
    if source_gate['SOURCE_HASH_GATE'] != 'PASS' or hardware_gate['VGPU_MEMORY_GATE'] != 'PASS':
        raise RuntimeError('source/hardware gate not PASS')
    order = [1, 4, 8, 16, 32]
    if horizon not in order:
        raise RuntimeError('unregistered horizon')
    for prior in order[:order.index(horizon)]:
        prior_result = json.loads((ROOT / f'analysis/H{prior}.json').read_text())
        if prior_result['status'] != 'PASS':
            raise RuntimeError(f'H{prior} gate not PASS')
    if (ROOT / f'analysis/H{horizon}.json').exists() or (ROOT / f'analysis/H{horizon}.error.json').exists():
        raise RuntimeError('horizon already attempted; refusing retry')


def weight_fingerprints(model) -> dict:
    layers = model.model.layers
    selected = {
        'embedding': model.get_input_embeddings().weight,
        'block0': next(layers[0].parameters()),
        'block15': next(layers[15].parameters()),
        'block31': next(layers[31].parameters()),
        'lm_head': model.lm_head.weight,
    }
    return {key: digest_tensor(value, 4096) for key, value in selected.items()}


def main(horizon: int) -> None:
    check_prerequisites(horizon)
    core = load_core()
    torch.manual_seed(core.TRAIN_SEED)
    model, tokenizer, bank, device = load_model_bank_without_dispatch(core)
    if sum(p.numel() for p in bank.parameters()) != 195072:
        raise RuntimeError('TRAINABLE_PARAMETER_GATE: rotation count mismatch')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('TRAINABLE_PARAMETER_GATE: base model not frozen')
    if any(bool(torch.count_nonzero(p).item()) for p in bank.parameters()):
        raise RuntimeError('theta is not zero initialized')
    before_weights = weight_fingerprints(model)
    ids, _ = core.tokenize(tokenizer, core.corpus_rows('TRAIN')[0])
    ids = ids[:32]
    if len(ids) != 32:
        raise RuntimeError('frozen sample shorter than 32 tokens')
    sample_sha = core.tensor_sha256(torch.tensor(ids, dtype=torch.long))
    token_target_sha = core.tensor_sha256(torch.tensor(ids[31:32], dtype=torch.long))
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    hadamard = core.hadamard(device)
    snapshots = {}
    with core.RecurrentExperimentPatch(model, bank, hadamard) as patch:
        targets = core.collect_teacher_targets(model, patch, ids, [30, 31])
        target_hashes = {str(layer): core.tensor_sha256(targets[31]['prev'][layer]) for layer in core.GDN_LAYERS}
        teacher_target_sha = hashlib.sha256(json.dumps(target_hashes, sort_keys=True).encode()).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        snapshots['before_rollout'] = memory(device)
        cache = None
        post30 = {}
        writeback = {}
        final_loss = None
        loss_value = None
        backward_done = False
        for index, token_id in enumerate(ids):
            capture = index in (30, 31)
            if capture:
                core.install_teacher_target(patch, targets[index], device)
            cache = core.forward_student(model, patch, token_id, cache, capture)
            if index == 30:
                post30 = {layer: core.tensor_sha256(patch.student_post[layer]) for layer in core.GDN_LAYERS}
                patch.clear_capture()
            if index == 31:
                writeback = {str(layer): post30[layer] == core.tensor_sha256(patch.student_prev[layer])
                             for layer in core.GDN_LAYERS}
                _state, _functional, final_loss, _c6 = patch.captured_losses()
                if not bool(torch.isfinite(final_loss).item()):
                    raise FloatingPointError('nonfinite C5 loss')
                loss_value = float(final_loss.detach().cpu())
                snapshots['after_forward'] = memory(device)
            if (index + 1) % horizon == 0:
                if final_loss is not None:
                    final_loss.backward()
                    snapshots['after_backward'] = memory(device)
                    patch.clear_capture()
                    del final_loss, _state, _functional, _c6
                    final_loss = None
                    backward_done = True
                core.detach_cache(cache)
        if not backward_done or not all(writeback.values()):
            raise RuntimeError('C5 capture or recurrent writeback gate failed')
        grad_norms = {}
        for layer in core.GDN_LAYERS:
            grad = bank.layer(layer).theta.grad
            if grad is None or not bool(torch.isfinite(grad).all().item()):
                raise FloatingPointError(f'missing/nonfinite gradient layer {layer}')
            grad_norms[str(layer)] = float(torch.linalg.vector_norm(grad).detach().cpu())
        if not any(value > 0 for value in grad_norms.values()):
            raise FloatingPointError('all rotation gradients zero')
        raw_grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        if not bool(torch.isfinite(raw_grad_norm).item()):
            raise FloatingPointError('nonfinite clipped gradient norm')
        optimizer.step()
        snapshots['after_optimizer'] = memory(device)
        after_weights = weight_fingerprints(model)
        if before_weights != after_weights:
            raise RuntimeError('frozen base weight fingerprint changed')
        orthogonality = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
        if orthogonality['status'] != 'PASS':
            raise RuntimeError('orthogonality gate failed')
        theta_displacement = math.sqrt(sum(float(p.detach().float().square().sum().cpu())
                                           for p in bank.parameters()))
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total = snapshots['after_optimizer']['device_total_bytes']
        qdq = {'calls': patch.qdq_audit['calls'], 'layers': sorted(patch.qdq_audit['layers']),
               'scale_shapes': patch.qdq_audit['scale_shapes'],
               'code_min': patch.qdq_audit['code_min'], 'code_max': patch.qdq_audit['code_max']}
        result = {
            'horizon': horizon, 'status': 'PASS', 'condition': 'C5', 'updates': 1,
            'sample': 'TRAIN:first-row', 'sample_token_hash': sample_sha,
            'target_token_hash': token_target_sha, 'teacher_target_hash': teacher_target_sha,
            'capture_token_index': 31, 'loss': loss_value,
            'rotation_gradient_norm_by_layer': grad_norms,
            'rotation_gradients_present': len(grad_norms),
            'gradient_norm_before_clip': float(raw_grad_norm.detach().cpu()),
            'theta_displacement_l2': theta_displacement, 'orthogonality': orthogonality,
            'recurrent_writeback_exact_by_layer': writeback,
            'C128_QDQ': qdq, 'base_model_unchanged': True,
            'optimizer_owns_only_rotation': True,
            'memory': snapshots,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': peak_reserved,
            'free_fraction_after_peak_conservative': (total - peak_reserved) / total,
            'free_margin_at_least_5_percent': (total - peak_reserved) / total >= 0.05,
            'free_margin_at_least_10_percent': (total - peak_reserved) / total >= 0.10,
        }
        save_once(ROOT / f'analysis/H{horizon}.json', result)
        print(json.dumps({'horizon': horizon, 'status': 'PASS', 'loss': result['loss'],
                          'peak_allocated_GiB': result['peak_allocated_bytes'] / 1024**3,
                          'peak_reserved_GiB': result['peak_reserved_bytes'] / 1024**3,
                          'free_fraction_after_peak': result['free_fraction_after_peak_conservative']},
                         sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, required=True)
    args = parser.parse_args()
    try:
        main(args.horizon)
    except Exception as error:
        details = {'horizon': args.horizon,
                   'status': 'OOM' if isinstance(error, torch.cuda.OutOfMemoryError) else 'FAIL',
                   'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        if torch.cuda.is_available():
            details['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            details['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save_once(ROOT / f'analysis/H{args.horizon}.error.json', details)
        print(json.dumps({'horizon': args.horizon, 'status': details['status'], 'error': str(error)}), flush=True)
        raise
