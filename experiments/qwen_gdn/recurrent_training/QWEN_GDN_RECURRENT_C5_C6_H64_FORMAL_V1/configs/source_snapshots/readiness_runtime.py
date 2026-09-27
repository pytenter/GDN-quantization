#!/usr/bin/env python3
"""Read-only canonical runtime adapters for the V2 diagnostic; no formula changes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HOST = (Path('/data/zypan/modelscope_models/Qwen3.5-9B/config.json').is_file())
SOURCE_ROOT = Path('/data/zypan') if ORIGINAL_HOST else Path('/root/autodl-tmp/canonical_source')
MODEL = (Path('/data/zypan/modelscope_models/Qwen3.5-9B') if ORIGINAL_HOST
         else Path('/root/autodl-tmp/models/Qwen3.5-9B'))
SOURCE = SOURCE_ROOT / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROTATION = SOURCE_ROOT / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py'
TRAIN_MANIFEST = SOURCE.parent / 'manifests/training_data_manifest.json'
sys.path.insert(0, str(SOURCE_ROOT / 'transformers-qwen35/src'))
os.environ.setdefault('QWEN_TRANSFORMERS_SRC', str(SOURCE_ROOT / 'transformers-qwen35'))
os.environ.setdefault('QWEN_MODEL_PATH', str(MODEL))
os.environ.setdefault('HF_HOME', '/data/zypan/.cache/huggingface' if ORIGINAL_HOST
                      else '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def save_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def load_core():
    spec = importlib.util.spec_from_file_location('qwen_gdn_48gb_readiness_v2_canonical', SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError('canonical C5/C6 source unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_parent_sources() -> dict:
    parent = json.loads((ROOT / 'configs/parent_manifest.json').read_text())
    checks = {
        'canonical_c5_c6_source': (SOURCE, parent['canonical_c5_c6_source_sha256']),
        'rotation_source': (ROTATION, parent['rotation_source_sha256']),
        'model_config': (MODEL / 'config.json', parent['model_config_sha256']),
        'model_index': (MODEL / 'model.safetensors.index.json', parent['model_index_sha256']),
        'training_manifest': (TRAIN_MANIFEST, parent['training_manifest_sha256']),
    }
    results = {}
    for name, (path, expected) in checks.items():
        actual = file_sha256(path)
        results[name] = {'path': str(path), 'expected_sha256': expected,
                         'actual_sha256': actual, 'exact': actual == expected}
    if not all(item['exact'] for item in results.values()):
        raise RuntimeError('parent source/model/training-manifest provenance mismatch')
    return results


def load_theta_bank(core, checkpoint: Path, device: torch.device):
    parent = json.loads((ROOT / 'configs/parent_manifest.json').read_text())
    actual_sha = file_sha256(checkpoint)
    if actual_sha != parent['training_theta_checkpoint_sha256']:
        raise RuntimeError('theta checkpoint SHA differs from frozen parent')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['condition'] != 'C5' or payload['gradient_horizon'] != 32 or payload['diagnostic_updates'] != 8:
        raise RuntimeError('theta checkpoint protocol metadata mismatch')
    if not payload['diagnostic_only'] or tuple(payload['layer_ids']) != core.GDN_LAYERS:
        raise RuntimeError('theta checkpoint layer mapping mismatch')
    if len(payload['bank']) != 24 or sum(x.numel() for x in payload['bank'].values()) != 195072:
        raise RuntimeError('theta checkpoint parameter count mismatch')
    if any(x.dtype != torch.float32 for x in payload['bank'].values()):
        raise RuntimeError('theta checkpoint must contain only FP32 tensors')
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS)
    bank.load_state_dict(payload['bank'], strict=True)
    return bank.to(device), payload, actual_sha


def load_model_and_tokenizer(core):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.to('cuda:0')
    core.freeze_model(model)
    return model, tokenizer, model.get_input_embeddings().weight.device


def cuda_memory(device: torch.device) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        'allocated_bytes': torch.cuda.memory_allocated(device),
        'reserved_bytes': torch.cuda.memory_reserved(device),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'device_free_bytes': free, 'device_total_bytes': total,
    }


class FrozenMatrixLayer(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        if value.shape != (128, 128) or value.dtype != torch.float32 or value.device.type != 'cpu':
            raise RuntimeError('frozen Cayley matrix has wrong shape/dtype/device')
        self.register_buffer('frozen_matrix', value.contiguous().clone())

    def matrix(self) -> torch.Tensor:
        return self.frozen_matrix


class FrozenMatrixRotationBank(nn.Module):
    """Directly supplies the canonical patch's Cayley Delta; H remains separate."""

    def __init__(self, layer_ids, matrices: dict[str, torch.Tensor]):
        super().__init__()
        self.layer_ids = tuple(int(x) for x in layer_ids)
        if set(matrices) != {str(x) for x in self.layer_ids} or len(self.layer_ids) != 24:
            raise RuntimeError('frozen layer mapping incomplete')
        self.layers = nn.ModuleDict({str(layer): FrozenMatrixLayer(matrices[str(layer)])
                                     for layer in self.layer_ids})

    def layer(self, layer_id: int) -> FrozenMatrixLayer:
        return self.layers[str(layer_id)]


def load_frozen_artifact(path: Path, core):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = json.loads((ROOT / 'configs/deployment_checkpoint_format.json').read_text())
    parent = json.loads((ROOT / 'configs/parent_manifest.json').read_text())
    if payload['format_version'] != config['format_version'] or tuple(payload['layer_ids']) != core.GDN_LAYERS:
        raise RuntimeError('deployment checkpoint format/layers mismatch')
    if payload['theta_checkpoint_sha256'] != parent['training_theta_checkpoint_sha256']:
        raise RuntimeError('deployment theta provenance mismatch')
    if payload['rotation_source_sha256'] != parent['rotation_source_sha256'] or payload['canonical_source_sha256'] != parent['canonical_c5_c6_source_sha256']:
        raise RuntimeError('deployment source provenance mismatch')
    if payload['matrix_dtype'] != 'torch.float32' or payload['matrix_shape'] != [128, 128]:
        raise RuntimeError('deployment matrix metadata mismatch')
    if payload['orientation'] != config['runtime_orientation']:
        raise RuntimeError('deployment orientation metadata mismatch')
    bank = FrozenMatrixRotationBank(payload['layer_ids'], payload['matrices'])
    return payload, bank
