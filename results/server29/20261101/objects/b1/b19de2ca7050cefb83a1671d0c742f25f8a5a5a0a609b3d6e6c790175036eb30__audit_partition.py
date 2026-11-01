#!/usr/bin/env python3
"""Static checkpoint metadata inventory; never materializes model tensors."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
import triton
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
V1 = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
ROT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py')
SOURCE = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
CANON = Path('/data/ydai/miniconda3/envs/bitdecode/bin/python')
if Path(sys.executable).resolve() != CANON.resolve():
    raise RuntimeError(f'Not canonical Python: {sys.executable}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


prereg = json.loads((ROOT / 'preregistration.json').read_text())
for name, path in [('canonical_c5_source_sha256', V1),
                   ('canonical_quantizer_rotation_source_sha256', ROT),
                   ('canonical_modeling_source_sha256', SOURCE),
                   ('model_config_sha256', MODEL / 'config.json')]:
    if sha(path) != prereg[name]:
        raise RuntimeError(f'Frozen source hash changed: {name}')

index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
config = json.loads((MODEL / 'config.json').read_text())['text_config']
gdn = [i for i, kind in enumerate(config['layer_types']) if kind == 'linear_attention']
if len(config['layer_types']) != 32 or len(gdn) != 24:
    raise RuntimeError('Unexpected Qwen decoder/GDN topology')
stage = {0: {'blocks': list(range(16)), 'gdn_layers': [i for i in gdn if i < 16]},
         1: {'blocks': list(range(16,32)), 'gdn_layers': [i for i in gdn if i >= 16]}}
counts = {0: Counter(), 1: Counter()}
unowned = []
file_groups = {}
for key, filename in index.items():
    file_groups.setdefault(filename, []).append(key)
for filename, keys in file_groups.items():
    with safe_open(str(MODEL / filename), framework='pt', device='cpu') as f:
        for key in keys:
            if key.startswith('model.language_model.embed_tokens.'):
                owner = 0
            elif key.startswith('model.language_model.layers.'):
                owner = 0 if int(key.split('.')[3]) < 16 else 1
            elif key.startswith('model.language_model.norm.') or key.startswith('lm_head.'):
                owner = 1
            else:
                unowned.append(key)
                continue
            shape = f.get_slice(key).get_shape()
            dtype = f.get_slice(key).get_dtype()
            width = {'BF16': 2, 'F16': 2, 'F32': 4, 'I64': 8}[dtype]
            numel = 1
            for dim in shape:
                numel *= dim
            counts[owner]['parameter_tensors'] += 1
            counts[owner]['parameters'] += numel
            counts[owner]['static_weight_bytes'] += numel * width
            counts[owner][f'dtype_{dtype}'] += 1

for owner in (0,1):
    stage[owner].update(dict(counts[owner]))
    stage[owner]['static_weight_gib'] = stage[owner]['static_weight_bytes'] / 2**30
    stage[owner]['rotation_parameter_count'] = len(stage[owner]['gdn_layers']) * 128 * 127 // 2
    stage[owner]['gdn_recurrent_state_shape_batch1'] = [1, 32, 128, 128]

total_rot = sum(stage[i]['rotation_parameter_count'] for i in (0,1))
if total_rot != 195072 or len(stage[0]['gdn_layers']) != 12 or len(stage[1]['gdn_layers']) != 12:
    raise RuntimeError('Rotation partition mismatch')
partition = {'model_path': str(MODEL), 'model_config_sha256': sha(MODEL / 'config.json'),
             'modeling_source_sha256': sha(SOURCE), 'split_after_block': 15,
             'hidden_size': config['hidden_size'], 'rank0': stage[0], 'rank1': stage[1],
             'global_rotation_parameter_count': total_rot,
             'unowned_checkpoint_keys': {'count': len(unowned), 'prefix_counts': dict(Counter(k.split('.')[0] for k in unowned)),
                                         'note': 'Vision and MTP checkpoint keys are not used by canonical text-only Qwen3_5ForCausalLM'},
             'checkpoint_index_sha256': sha(MODEL / 'model.safetensors.index.json')}
save_once(ROOT / 'configs/pp2_partition.json', partition)
save_once(ROOT / 'analysis/partition_inventory.json', partition)
save_once(ROOT / 'analysis/canonical_loss_decomposition.json',
          json.loads((ROOT / 'configs/canonical_loss_definition.json').read_text()))
save_once(ROOT / 'analysis/trainable_parameter_inventory.json', {
    'rank0_rotation_parameters': stage[0]['rotation_parameter_count'],
    'rank1_rotation_parameters': stage[1]['rotation_parameter_count'],
    'global_rotation_parameters': total_rot, 'base_model_trainable_parameters': 0,
    'status': 'STATIC_DESIGN_AUDIT_ONLY_NOT_RUNTIME_VERIFIED'})
try:
    fla = metadata.version('flash-linear-attention')
except metadata.PackageNotFoundError:
    fla = 'UNKNOWN (not registered)'
save_once(ROOT / 'configs/environment.json', {
    'python_executable': sys.executable, 'python': sys.version.split()[0],
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'nccl': list(torch.cuda.nccl.version()), 'transformers': transformers.__version__,
    'triton': triton.__version__, 'fla': fla, 'installed_or_upgraded_anything': False,
    'physical_gpu_pair_preferred': [0,1]})
print(json.dumps({'rank0_static_gib': stage[0]['static_weight_gib'],
                  'rank1_static_gib': stage[1]['static_weight_gib'],
                  'rank0_rotation': stage[0]['rotation_parameter_count'],
                  'rank1_rotation': stage[1]['rotation_parameter_count'],
                  'unowned_checkpoint_keys': len(unowned)}))
