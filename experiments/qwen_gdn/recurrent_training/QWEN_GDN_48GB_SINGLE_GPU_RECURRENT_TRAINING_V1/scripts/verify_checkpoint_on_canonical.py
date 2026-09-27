#!/usr/bin/env python3
"""Rotation-only export and non-AIME portability probe; never trains."""
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

import torch

ORIGINAL = Path('/data/zypan/modelscope_models/Qwen3.5-9B').is_dir()
BASE = Path('/data/zypan') if ORIGINAL else Path('/root/autodl-tmp/canonical_source')
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B') if ORIGINAL else Path('/root/autodl-tmp/models/Qwen3.5-9B')
SOURCE = BASE / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('QWEN_TRANSFORMERS_SRC', str(BASE / 'transformers-qwen35'))
os.environ.setdefault('QWEN_MODEL_PATH', str(MODEL))
os.environ.setdefault('HF_HOME', '/data/zypan/.cache/huggingface' if ORIGINAL else '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(BASE / 'transformers-qwen35/src'))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def save_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load_core():
    spec = importlib.util.spec_from_file_location('qwen_48gb_portable_c5', SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError('canonical C5 source unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inspect_rotation(core, checkpoint: Path) -> tuple[dict, object]:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['condition'] != 'C5' or payload['gradient_horizon'] != 32 or payload['diagnostic_updates'] != 8:
        raise RuntimeError('checkpoint protocol metadata mismatch')
    if not payload['diagnostic_only'] or tuple(payload['layer_ids']) != core.GDN_LAYERS:
        raise RuntimeError('checkpoint layer mapping mismatch')
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS)
    bank.load_state_dict(payload['bank'], strict=True)
    if sum(p.numel() for p in bank.parameters()) != 195072:
        raise RuntimeError('rotation parameter count mismatch')
    h = core.hadamard(torch.device('cpu'))
    theta_hashes = {key: sha_tensor(value) for key, value in bank.state_dict().items()}
    delta_hashes = {}
    effective_hashes = {}
    for layer in core.GDN_LAYERS:
        delta = bank.layer(layer).matrix().detach().cpu().contiguous()
        delta_hashes[str(layer)] = sha_tensor(delta)
        effective_hashes[str(layer)] = sha_tensor(h @ delta)
    orthogonality = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
    if orthogonality['status'] != 'PASS':
        raise RuntimeError('checkpoint orthogonality gate failed')
    result = {
        'checkpoint_sha256': sha_file(checkpoint),
        'checkpoint_size_bytes': checkpoint.stat().st_size,
        'canonical_source_sha256': sha_file(SOURCE),
        'model_config_sha256': sha_file(MODEL / 'config.json'),
        'model_index_sha256': sha_file(MODEL / 'model.safetensors.index.json'),
        'theta_hashes': theta_hashes,
        'cayley_delta_matrix_hashes_cpu': delta_hashes,
        'effective_hadamard_rotation_matrix_hashes_cpu': effective_hashes,
        'orthogonality_cpu': orthogonality,
        'total_rotation_parameters': 195072,
    }
    return result, bank


def fixed_forward(core, bank, expected_sample_hash: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1:
        raise RuntimeError('one CUDA device required for portability forward')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.to('cuda:0')
    core.freeze_model(model)
    bank.to('cuda:0')
    device = model.get_input_embeddings().weight.device
    ids, _ = core.tokenize(tokenizer, core.corpus_rows('TRAIN')[0])
    ids = ids[:32]
    sample_hash = core.tensor_sha256(torch.tensor(ids, dtype=torch.long))
    if len(ids) != 32 or sample_hash != expected_sample_hash:
        raise RuntimeError('fixed sample differs from checkpoint')
    with core.RecurrentExperimentPatch(model, bank, core.hadamard(device)) as patch:
        targets = core.collect_teacher_targets(model, patch, ids, [30, 31])
        cache = None
        post30 = {}
        writeback = {}
        loss_value = None
        with torch.no_grad():
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
                    _, _, c5, _ = patch.captured_losses()
                    if not bool(torch.isfinite(c5).item()):
                        raise FloatingPointError('nonfinite portability forward C5 loss')
                    loss_value = float(c5.detach().cpu())
    if loss_value is None or not all(writeback.values()):
        raise RuntimeError('portability recurrent writeback failed')
    if patch.qdq_audit['calls'] < len(core.GDN_LAYERS) * len(ids):
        raise RuntimeError('portability C128 QDQ call count incomplete')
    return {
        'sample': 'TRAIN:first-row', 'token_count': 32,
        'sample_token_hash': sample_hash, 'C5_loss': loss_value,
        'C128_QDQ_calls': patch.qdq_audit['calls'],
        'C128_QDQ_layers': sorted(patch.qdq_audit['layers']),
        'recurrent_writeback_exact_by_layer': writeback,
        'all_finite': True, 'training_performed': False,
    }


def main(args) -> None:
    checkpoint = Path(args.checkpoint)
    core = load_core()
    result, bank = inspect_rotation(core, checkpoint)
    if args.mode == 'export':
        if ORIGINAL:
            raise RuntimeError('export mode must run on vGPU')
        result['fixed_forward'] = fixed_forward(core, bank, args.sample_hash)
        save_once(Path(args.output), result)
        print(json.dumps({'mode': 'export', 'checkpoint_sha256': result['checkpoint_sha256'],
                          'C5_loss': result['fixed_forward']['C5_loss']}), flush=True)
        return
    if not ORIGINAL:
        raise RuntimeError('verify mode must run on original server')
    exported = json.loads(Path(args.export_json).read_text())
    keys = ('checkpoint_sha256', 'canonical_source_sha256', 'model_config_sha256',
            'model_index_sha256', 'theta_hashes', 'cayley_delta_matrix_hashes_cpu',
            'effective_hadamard_rotation_matrix_hashes_cpu')
    exact = {key: result[key] == exported[key] for key in keys}
    result['exact_hash_gates'] = exact
    if not all(exact.values()):
        result['VGPU_TO_CANONICAL_CHECKPOINT_PORTABILITY'] = 'FAIL_EXACT_ROTATION_MATRIX_HASH'
        result['fixed_forward'] = 'NOT_RUN_AFTER_HASH_FAILURE'
        result['exported_cayley_delta_matrix_hashes_cpu'] = exported['cayley_delta_matrix_hashes_cpu']
        result['exported_effective_hadamard_rotation_matrix_hashes_cpu'] = exported['effective_hadamard_rotation_matrix_hashes_cpu']
        save_once(Path(args.output), result)
        raise RuntimeError(f'checkpoint/source/matrix hashes differ: {exact}')
    result['fixed_forward'] = fixed_forward(core, bank, args.sample_hash)
    result['VGPU_TO_CANONICAL_CHECKPOINT_PORTABILITY'] = 'PASS'
    result['vgpu_fixed_forward_C5_loss'] = exported['fixed_forward']['C5_loss']
    result['cross_gpu_C5_loss_difference'] = (result['fixed_forward']['C5_loss'] -
                                               exported['fixed_forward']['C5_loss'])
    result['cross_gpu_tolerance'] = 'NOT_PRE_REGISTERED; difference reported, no post-hoc threshold'
    save_once(Path(args.output), result)
    print(json.dumps({'VGPU_TO_CANONICAL_CHECKPOINT_PORTABILITY': 'PASS',
                      'theta_and_matrix_hashes_exact': all(exact.values()),
                      'C5_loss_difference': result['cross_gpu_C5_loss_difference']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('export', 'verify'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--export-json')
    parser.add_argument('--sample-hash', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'type': type(exc).__name__, 'error': str(exc),
                          'traceback': traceback.format_exc()}), flush=True)
        raise
