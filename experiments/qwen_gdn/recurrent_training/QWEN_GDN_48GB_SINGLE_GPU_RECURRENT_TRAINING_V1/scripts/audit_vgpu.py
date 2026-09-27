#!/usr/bin/env python3
"""Read-only hardware, environment, frozen-source and model inventory."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR = Path('/root/autodl-tmp/canonical_source')
MODEL = Path('/root/autodl-tmp/models/Qwen3.5-9B')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def save(relative: str, value: object) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write('\n')


def version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return 'NOT_INSTALLED'


def main() -> None:
    import torch
    import numpy
    import safetensors
    import tokenizers
    import transformers
    import triton

    assert str(Path(transformers.__file__).resolve()).startswith(str(MIRROR / 'transformers-qwen35/src'))
    driver = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True
    ).strip()
    gpu_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if gpu_count == 1 else None
    hardware = {
        'hostname': subprocess.check_output(['hostname'], text=True).strip(),
        'actual_gpu_name': torch.cuda.get_device_name(0) if props else None,
        'visible_cuda_device_count': gpu_count,
        'total_memory_bytes': props.total_memory if props else None,
        'total_memory_GiB': props.total_memory / 1024**3 if props else None,
        'compute_capability': [props.major, props.minor] if props else None,
        'driver_version': driver,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    save('configs/vgpu_hardware.json', hardware)
    save('analysis/hardware_gate.json', {
        'VGPU_VISIBLE_DEVICE_COUNT': gpu_count,
        'VGPU_MEMORY_GATE': 'PASS' if props and 45 <= props.total_memory / 1024**3 <= 50 else 'FAIL',
        'cuda_available': torch.cuda.is_available(),
        'hardware': hardware,
    })

    env = {
        'python': sys.version.split()[0], 'python_executable': sys.executable,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'nccl': list(torch.cuda.nccl.version()), 'driver': driver,
        'transformers': transformers.__version__, 'transformers_source': str(Path(transformers.__file__).resolve()),
        'triton': triton.__version__, 'fla': version('flash-linear-attention'),
        'numpy': numpy.__version__, 'safetensors': safetensors.__version__,
        'tokenizers': tokenizers.__version__, 'flash_attn': version('flash_attn'),
    }
    expected = {
        'python': '3.10.18', 'torch': '2.5.1+cu121', 'cuda_runtime': '12.1',
        'transformers': '5.16.0.dev0', 'triton': '3.1.0', 'numpy': '2.2.6',
        'safetensors': '0.8.0', 'tokenizers': '0.23.1',
    }
    env['ENVIRONMENT_DELTA'] = {key: {'expected': target, 'actual': env[key]}
                                for key, target in expected.items() if env[key] != target}
    save('configs/vgpu_environment.json', env)

    provenance = json.loads((ROOT / 'configs/source_provenance.json').read_text())
    checks = {}
    for entry in provenance['server_only_sources']:
        path = Path(entry['target_path'])
        actual = sha(path) if path.is_file() else None
        checks[entry['original_absolute_path']] = {
            'target_path': str(path), 'expected_sha256': entry['sha256'],
            'actual_sha256': actual, 'match': actual == entry['sha256'],
        }
    other = {
        'model_config': (MODEL / 'config.json', provenance['model_config_sha256']),
        'model_index': (MODEL / 'model.safetensors.index.json', provenance['model_safetensors_index_sha256']),
        'training_manifest': (MIRROR / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/manifests/training_data_manifest.json', provenance['frozen_manifest_sha256']['training_data']),
        'validation_manifest': (MIRROR / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/manifests/validation_data_manifest.json', provenance['frozen_manifest_sha256']['validation_data']),
        'eval_manifest': (MIRROR / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/manifests/eval_samples.jsonl', provenance['frozen_manifest_sha256']['eval_samples']),
    }
    for key, (path, target) in other.items():
        actual = sha(path) if path.is_file() else None
        checks[key] = {'target_path': str(path), 'expected_sha256': target,
                       'actual_sha256': actual, 'match': actual == target}
    shard_files = sorted(MODEL.glob('model.safetensors-*-of-*.safetensors'))
    model_size = sum(path.stat().st_size for path in MODEL.iterdir() if path.is_file())
    disk = shutil.disk_usage('/root/autodl-tmp')
    result = {
        'SOURCE_HASH_GATE': 'PASS' if all(row['match'] for row in checks.values()) else 'FAIL',
        'checks': checks, 'model_shards': [path.name for path in shard_files],
        'model_size_bytes': model_size, 'data_disk_free_bytes': disk.free,
        'data_disk_free_GiB': disk.free / 1024**3,
        'model_single_copy_path': str(MODEL),
    }
    save('analysis/source_parity.json', result)
    print(json.dumps({'VGPU_MEMORY_GATE': 'PASS' if props and 45 <= props.total_memory / 1024**3 <= 50 else 'FAIL',
                      'SOURCE_HASH_GATE': result['SOURCE_HASH_GATE'],
                      'ENVIRONMENT_DELTA': env['ENVIRONMENT_DELTA'],
                      'model_size_bytes': model_size, 'data_disk_free_GiB': result['data_disk_free_GiB']},
                     sort_keys=True), flush=True)
    if not props or not 45 <= props.total_memory / 1024**3 <= 50:
        raise RuntimeError('VGPU_MEMORY_GATE_FAIL')
    if result['SOURCE_HASH_GATE'] != 'PASS' or len(shard_files) != 4:
        raise RuntimeError('SOURCE_HASH_GATE_FAIL')


if __name__ == '__main__':
    main()
