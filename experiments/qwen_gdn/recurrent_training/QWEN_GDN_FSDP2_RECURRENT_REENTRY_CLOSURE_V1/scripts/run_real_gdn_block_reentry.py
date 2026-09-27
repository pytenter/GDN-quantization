#!/usr/bin/env python3
"""One real frozen Qwen3.5 GDN block, repeatedly invoked before one backward."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
PREFIX = 'model.language_model.layers.0.linear_attn.'


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def mem(rank):
    torch.cuda.synchronize(rank)
    free, total = torch.cuda.mem_get_info(rank)
    return {'allocated_bytes': torch.cuda.memory_allocated(rank),
            'reserved_bytes': torch.cuda.memory_reserved(rank),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(rank),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(rank),
            'device_free_bytes': free, 'device_total_bytes': total}


def param_state(p):
    local = p.to_local() if isinstance(p, DTensor) else p
    try:
        storage_bytes = local.untyped_storage().nbytes()
        ptr = local.untyped_storage().data_ptr()
    except Exception as exc:
        storage_bytes, ptr = None, None
    return {'logical_shape': list(p.shape), 'dtype': str(p.dtype),
            'is_dtensor': isinstance(p, DTensor),
            'placement': [str(x) for x in p.placements] if isinstance(p, DTensor) else None,
            'local_shape': list(local.shape), 'local_storage_nbytes': storage_bytes,
            'local_storage_data_ptr': ptr, 'requires_grad': p.requires_grad}


def load_gdn(rank):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    config = AutoConfig.from_pretrained(str(MODEL), trust_remote_code=True,
                                        local_files_only=True).text_config
    module = Qwen3_5GatedDeltaNet(config, 0).to(dtype=torch.bfloat16)
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {name[len(PREFIX):]: filename for name, filename in mapping.items()
                if name.startswith(PREFIX)}
    loaded = {}
    for filename in sorted(set(selected.values())):
        with safe_open(str(MODEL / filename), framework='pt', device='cpu') as handle:
            for name, source in selected.items():
                if source == filename:
                    loaded[name] = handle.get_tensor(PREFIX + name)
    missing, unexpected = module.load_state_dict(loaded, strict=True)
    if missing or unexpected or len(loaded) != 9:
        raise RuntimeError(f'checkpoint GDN mismatch: {missing=}, {unexpected=}, count={len(loaded)}')
    conv = loaded['conv1d.weight']
    conv_hash = hashlib.sha256(conv.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    del loaded
    module.eval().to(device=torch.device('cuda', rank))
    for p in module.parameters():
        p.requires_grad_(False)
    return module, config, conv_hash


def run(n, reshard, cache_mode):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 or torch.cuda.device_count() != 2 or rank not in (0, 1):
        raise RuntimeError('requires exactly two visible GPUs/ranks')
    tag = f'gdn_{cache_mode}_N{n}_reshard{int(reshard)}_rank{rank}'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(20260927)
        torch.cuda.manual_seed_all(20260927)
        block, config, conv_hash = load_gdn(rank)
        mesh = init_device_mesh('cuda', (world,))
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard)
        conv = block.conv1d.weight
        if not isinstance(conv, DTensor) or conv.to_local().numel() * 2 != conv.numel():
            raise RuntimeError('frozen GDN conv parameter not half-sharded')
        device = torch.device('cuda', rank)
        hidden = torch.randn((1, 1, config.hidden_size), device=device,
                             dtype=torch.bfloat16, requires_grad=True)
        initial = hidden
        cache = None
        if cache_mode == 'recurrent':
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache(config=config)
        timeline = []
        torch.cuda.empty_cache()
        before = mem(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        for step in range(n):
            timeline.append({'event': f'forward_{step+1}_pre', 'conv_weight': param_state(block.conv1d.weight),
                             'memory': mem(rank)})
            hidden = block(hidden, cache_params=cache)
            timeline.append({'event': f'forward_{step+1}_post', 'conv_weight': param_state(block.conv1d.weight),
                             'memory': mem(rank)})
        loss = hidden.float().square().mean()
        before_backward = mem(rank)
        timeline.append({'event': 'backward_pre', 'conv_weight': param_state(block.conv1d.weight),
                         'memory': before_backward})
        loss.backward()
        after_backward = mem(rank)
        timeline.append({'event': 'backward_post', 'conv_weight': param_state(block.conv1d.weight),
                         'memory': after_backward})
        grad = initial.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.float().abs().max()) == 0:
            raise RuntimeError('missing/nonfinite/zero input gradient')
        result = {'status': 'PASS', 'rank': rank, 'world_size': world,
                  'physical_visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES'),
                  'same_module_instance': True, 'reentry_count': n,
                  'reshard_after_forward': reshard, 'cache_mode': cache_mode,
                  'checkpoint_layer0_gdn_weight_count': 9,
                  'conv_weight_checkpoint_sha256': conv_hash,
                  'conv_weight_logical_shape': list(block.conv1d.weight.shape),
                  'loss': float(loss.detach()),
                  'input_grad_max_abs': float(grad.float().abs().max()),
                  'memory_before': before, 'memory_before_backward': before_backward,
                  'memory_after_backward': after_backward, 'timeline': timeline}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'status': 'PASS', 'n': n, 'reshard': reshard,
                              'cache_mode': cache_mode,
                              'peak_allocated_bytes': after_backward['peak_allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'status': 'FAIL', 'rank': rank, 'n': n, 'reshard': reshard,
                   'cache_mode': cache_mode, 'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reentries', dest='n', type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument('--reshard', choices=['true', 'false'], required=True)
    p.add_argument('--cache-mode', choices=['none', 'recurrent'], default='none')
    a = p.parse_args()
    run(a.n, a.reshard == 'true', a.cache_mode)
