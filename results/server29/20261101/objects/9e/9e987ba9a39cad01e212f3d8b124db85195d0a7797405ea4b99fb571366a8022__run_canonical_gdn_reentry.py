#!/usr/bin/env python3
"""Reduced real-GDN reentry with the frozen V1 differentiable cache, ±C128/rotation."""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from run_real_gdn_block_reentry import ROOT, load_gdn, mem, param_state, save_once

V1_PATH = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')


def import_v1():
    spec = importlib.util.spec_from_file_location('reentry_read_only_v1', V1_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(n, reshard, mode):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two visible GPUs/ranks required')
    tag = f'canonical_{mode}_N{n}_reshard{int(reshard)}_rank{rank}'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(20260927)
        torch.cuda.manual_seed_all(20260927)
        block, config, conv_hash = load_gdn(rank)
        mesh = init_device_mesh('cuda', (world,))
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard)
        device = torch.device('cuda', rank)
        v1 = import_v1()
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        dummy = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=block)]))
        h = v1.hadamard(device)
        hidden = torch.randn((1, 1, config.hidden_size), device=device,
                             dtype=torch.bfloat16, requires_grad=True)
        initial = hidden
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=config)
        timeline = []
        torch.cuda.empty_cache()
        before = mem(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        with v1.RecurrentExperimentPatch(dummy, bank, h) as patch:
            patch.mode = 'student' if mode == 'qdq_rotation' else 'idle'
            patch.capture = False
            for step in range(n):
                timeline.append({'event': f'forward_{step+1}_pre',
                                 'conv_weight': param_state(block.conv1d.weight), 'memory': mem(rank)})
                hidden = block(hidden, cache_params=cache)
                v1.enable_differentiable_cache(cache)
                timeline.append({'event': f'forward_{step+1}_post',
                                 'conv_weight': param_state(block.conv1d.weight), 'memory': mem(rank),
                                 'recurrent_state_shape': list(cache.layers[0].recurrent_states[0].shape),
                                 'recurrent_state_requires_grad': cache.layers[0].recurrent_states[0].requires_grad})
            loss = hidden.float().square().mean()
            timeline.append({'event': 'backward_pre', 'conv_weight': param_state(block.conv1d.weight),
                             'memory': mem(rank)})
            loss.backward()
            after = mem(rank)
            timeline.append({'event': 'backward_post', 'conv_weight': param_state(block.conv1d.weight),
                             'memory': after})
        grad = initial.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.float().abs().max()) == 0:
            raise RuntimeError('missing/nonfinite/zero input gradient')
        result = {'status': 'PASS', 'rank': rank, 'same_module_instance': True,
                  'n': n, 'reshard_after_forward': reshard,
                  'mode': mode, 'cache_writeback': 'frozen V1 differentiable assignment/concat',
                  'qdq_and_rotation': mode == 'qdq_rotation',
                  'conv_weight_checkpoint_sha256': conv_hash,
                  'conv_weight_logical_shape': list(block.conv1d.weight.shape),
                  'loss': float(loss.detach()), 'input_grad_max_abs': float(grad.float().abs().max()),
                  'memory_before': before, 'memory_after_backward': after,
                  'qdq_calls': patch.qdq_audit['calls'], 'timeline': timeline}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'status': 'PASS', 'mode': mode, 'n': n,
                              'reshard': reshard, 'peak_allocated_bytes': after['peak_allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'status': 'FAIL', 'rank': rank, 'mode': mode, 'n': n,
                   'reshard': reshard, 'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reentries', type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument('--reshard', choices=['true', 'false'], required=True)
    p.add_argument('--mode', choices=['cache_only', 'qdq_rotation'], required=True)
    a = p.parse_args()
    run(a.reentries, a.reshard == 'true', a.mode)
