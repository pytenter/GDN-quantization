#!/usr/bin/env python3
"""Two-rank FSDP2 load/shard-only gate. No forward, backward or optimizer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
MODEL_SOURCE = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
V1_PATH = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def import_v1():
    spec = importlib.util.spec_from_file_location('fsdp_read_only_v1', V1_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_free_bytes': free, 'device_total_bytes': total}


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two visible idle GPUs and two ranks required')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    try:
        before = memory()
        # No HF device_map: each rank loads canonical CPU weights, then FSDP2 shards.
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers') or len(model.model.layers) != 32:
            raise RuntimeError('unexpected Qwen text decoder topology')
        logical_param_count = sum(p.numel() for p in model.parameters())
        raw_logical_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model.to(torch.device('cuda', local_rank))
        full_load = memory()
        mesh = init_device_mesh('cuda', (world,))
        unit_names = [f'model.layers.{i}' for i in range(32)] + ['model.embed_tokens', 'model.norm', 'lm_head']
        unit_records = []
        for name in unit_names:
            module = model.get_submodule(name)
            logical = sum(p.numel() for p in module.parameters(recurse=False))
            if name.startswith('model.layers.'):
                logical = sum(p.numel() for p in module.parameters())
            fully_shard(module, mesh=mesh, reshard_after_forward=True)
            local = sum(p.to_local().numel() if isinstance(p, DTensor) else p.numel()
                        for p in module.parameters(recurse=False))
            if name.startswith('model.layers.'):
                local = sum(p.to_local().numel() if isinstance(p, DTensor) else p.numel()
                            for p in module.parameters())
            unit_records.append({'unit': name, 'logical_numel': logical,
                                 'local_shard_numel': local})
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        v1 = import_v1()
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(torch.device('cuda', local_rank))
        bank_params = [(n, p) for n, p in bank.named_parameters()]
        param_details = []
        for name, p in model.named_parameters():
            is_dt = isinstance(p, DTensor)
            param_details.append({'name': name, 'logical_numel': p.numel(),
                                  'local_numel': p.to_local().numel() if is_dt else p.numel(),
                                  'dtype': str(p.dtype),
                                  'placements': [str(x) for x in p.placements] if is_dt else [],
                                  'requires_grad': p.requires_grad})
        sharded = all(d['placements'] and any('Shard' in s or s.startswith('S(') for s in d['placements']) for d in param_details)
        frozen = all(not d['requires_grad'] for d in param_details)
        rotation_trainable = bool(bank_params) and all(p.requires_grad for _, p in bank_params)
        result = {'rank': rank, 'local_rank': local_rank, 'device': str(torch.cuda.current_device()),
                  'physical_visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'world_size': world, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'fsdp_api': 'torch.distributed._composable.fsdp.fully_shard',
                  'modeling_source_sha256': sha(MODEL_SOURCE), 'v1_source_sha256': sha(V1_PATH),
                  'model_config_sha256': sha(MODEL_PATH / 'config.json'),
                  'logical_model_parameters_before_shard': logical_param_count,
                  'raw_model_parameter_bytes_before_shard': raw_logical_bytes,
                  'memory_before_load': before, 'memory_full_model_before_shard': full_load,
                  'memory_after_shard_and_rotation': memory(),
                  'unit_shards': unit_records, 'parameter_shards': param_details,
                  'rotation': {'parameter_count': sum(p.numel() for _, p in bank_params),
                               'parameter_bytes': sum(p.numel() * p.element_size() for _, p in bank_params),
                               'trainable_names': [n for n, _ in bank_params],
                               'device': str(next(bank.parameters()).device)},
                  'TRAINABLE_PARAMETER_GATE': 'PASS' if frozen and rotation_trainable else 'FAIL',
                  'PARAMETER_SHARDING_GATE': 'PASS' if sharded else 'FAIL'}
        save_once(ROOT / 'analysis' / f'fsdp_load_rank{rank}.json', result)
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            other = json.loads((ROOT / 'analysis/fsdp_load_rank1.json').read_text())
            combined = {'world_size': world, 'ranks': [result, other],
                        'FSDP_LOAD_GATE': 'PASS' if all(r['memory_after_shard_and_rotation']['allocated_bytes'] < r['memory_full_model_before_shard']['allocated_bytes'] for r in (result, other)) else 'FAIL',
                        'TRAINABLE_PARAMETER_GATE': 'PASS' if all(r['TRAINABLE_PARAMETER_GATE'] == 'PASS' for r in (result, other)) else 'FAIL',
                        'PARAMETER_SHARDING_GATE': 'PASS' if all(r['PARAMETER_SHARDING_GATE'] == 'PASS' for r in (result, other)) else 'FAIL'}
            save_once(ROOT / 'analysis/fsdp_load_memory.json', combined)
            save_once(ROOT / 'analysis/trainable_parameter_inventory.json',
                      {'model_logical_parameters': logical_param_count,
                       'model_frozen_parameters': logical_param_count if frozen else None,
                       'rotation_trainable_parameters': result['rotation']['parameter_count'],
                       'rotation_trainable_names': result['rotation']['trainable_names'],
                       'rotation_placement': 'replicated independent per rank',
                       'backbone_placement': 'FSDP2 DTensor Shard(0)',
                       'TRAINABLE_PARAMETER_GATE': combined['TRAINABLE_PARAMETER_GATE']})
            save_once(ROOT / 'analysis/fsdp_parameter_shards.json',
                      {'rank0': result['parameter_shards'], 'rank1': other['parameter_shards'],
                       'gate': combined['PARAMETER_SHARDING_GATE']})
            print(json.dumps({'FSDP_LOAD_GATE': combined['FSDP_LOAD_GATE'],
                              'PARAMETER_SHARDING_GATE': combined['PARAMETER_SHARDING_GATE'],
                              'TRAINABLE_PARAMETER_GATE': combined['TRAINABLE_PARAMETER_GATE'],
                              'rank0_allocated': result['memory_after_shard_and_rotation']['allocated_bytes'],
                              'rank1_allocated': other['memory_after_shard_and_rotation']['allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'fsdp_load_rank{rank}.error.json',
                  {'rank': rank, 'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
