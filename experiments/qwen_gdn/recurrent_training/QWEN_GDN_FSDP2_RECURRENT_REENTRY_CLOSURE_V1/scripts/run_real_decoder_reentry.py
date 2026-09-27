#!/usr/bin/env python3
"""One exact-checkpoint Qwen decoder layer under the frozen V1 recurrent patch."""
import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from run_canonical_gdn_reentry import import_v1
from run_real_gdn_block_reentry import ROOT, MODEL, mem, param_state, save_once

PREFIX = 'model.language_model.layers.0.'


def load_decoder(rank):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    config = AutoConfig.from_pretrained(str(MODEL), trust_remote_code=True,
                                        local_files_only=True).text_config
    layer = Qwen3_5DecoderLayer(config, 0).to(dtype=torch.bfloat16)
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {name[len(PREFIX):]: filename for name, filename in mapping.items()
                if name.startswith(PREFIX)}
    loaded = {}
    for filename in sorted(set(selected.values())):
        with safe_open(str(MODEL / filename), framework='pt', device='cpu') as handle:
            for name, source in selected.items():
                if source == filename:
                    loaded[name] = handle.get_tensor(PREFIX + name)
    missing, unexpected = layer.load_state_dict(loaded, strict=True)
    if missing or unexpected or len(loaded) != 14:
        raise RuntimeError(f'checkpoint decoder mismatch: {missing=}, {unexpected=}, count={len(loaded)}')
    conv = loaded['linear_attn.conv1d.weight']
    conv_hash = hashlib.sha256(conv.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    del loaded
    layer.eval().to(device=torch.device('cuda', rank))
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer, config, conv_hash


def run(n, reshard):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two visible GPUs/ranks required')
    tag = f'decoder_N{n}_reshard{int(reshard)}_rank{rank}'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(20260927)
        torch.cuda.manual_seed_all(20260927)
        layer, config, conv_hash = load_decoder(rank)
        mesh = init_device_mesh('cuda', (world,))
        fully_shard(layer, mesh=mesh, reshard_after_forward=reshard)
        v1 = import_v1()
        device = torch.device('cuda', rank)
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        dummy = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
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
            patch.mode = 'student'
            patch.capture = False
            for step in range(n):
                timeline.append({'event': f'forward_{step+1}_pre',
                                 'conv_weight': param_state(layer.linear_attn.conv1d.weight),
                                 'memory': mem(rank)})
                hidden = layer(hidden, position_embeddings=(None, None), past_key_values=cache)
                v1.enable_differentiable_cache(cache)
                timeline.append({'event': f'forward_{step+1}_post',
                                 'conv_weight': param_state(layer.linear_attn.conv1d.weight),
                                 'memory': mem(rank)})
            loss = hidden.float().square().mean()
            timeline.append({'event': 'backward_pre',
                             'conv_weight': param_state(layer.linear_attn.conv1d.weight),
                             'memory': mem(rank)})
            loss.backward()
            after = mem(rank)
            timeline.append({'event': 'backward_post',
                             'conv_weight': param_state(layer.linear_attn.conv1d.weight),
                             'memory': after})
        grad = initial.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.float().abs().max()) == 0:
            raise RuntimeError('missing/nonfinite/zero input gradient')
        result = {'status': 'PASS', 'rank': rank, 'n': n,
                  'reshard_after_forward': reshard,
                  'module': 'full real Qwen3.5 layer-0 decoder block',
                  'checkpoint_weight_count': 14,
                  'conv_weight_checkpoint_sha256': conv_hash,
                  'loss': float(loss.detach()), 'input_grad_max_abs': float(grad.float().abs().max()),
                  'qdq_calls': patch.qdq_audit['calls'],
                  'memory_before': before, 'memory_after_backward': after,
                  'timeline': timeline}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'status': 'PASS', 'module': 'decoder', 'n': n,
                              'reshard': reshard, 'peak_allocated_bytes': after['peak_allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'status': 'FAIL', 'rank': rank, 'n': n, 'reshard': reshard,
                   'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reentries', type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument('--reshard', choices=['true', 'false'], required=True)
    a = p.parse_args()
    run(a.reentries, a.reshard == 'true')
