#!/usr/bin/env python3
"""Frozen non-AIME teacher forward under canonical single or FSDP2 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from run_fsdp_load import ROOT, MODEL_PATH, MODEL_SOURCE, V1_PATH, import_v1, memory, save_once, sha


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def summary(t):
    f = t.detach().float()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': tensor_sha(t),
            'min': float(f.min()), 'max': float(f.max()),
            'mean': float(f.mean()), 'l2': float(torch.linalg.vector_norm(f))}


def load_model(fsdp, rank, world):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(torch.device('cuda', rank))
    if fsdp:
        mesh = init_device_mesh('cuda', (world,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        for module in (model.model.embed_tokens, model.model.norm, model.lm_head):
            fully_shard(module, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        torch.cuda.empty_cache()  # allocator cache only; model arithmetic unchanged
    return model, tokenizer


def run(mode: str, replicate: int):
    fsdp = mode == 'fsdp'
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if (fsdp and (world != 2 or torch.cuda.device_count() != 2)) or (not fsdp and torch.cuda.device_count() != 1):
        raise RuntimeError('incorrect visible GPU count/rank topology')
    torch.cuda.set_device(rank)
    if fsdp:
        dist.init_process_group('nccl')
    tag = f'{mode}{replicate:02d}_rank{rank}'
    try:
        v1 = import_v1()
        row = v1.corpus_rows('TRAIN')[0]
        model, tokenizer = load_model(fsdp, rank, world)
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        if len(ids) != 64:
            raise RuntimeError('frozen V2 input changed')
        input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
        target = torch.tensor([ids[-1]], device='cuda', dtype=torch.long)
        before = memory()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
            logits = output.logits[:, -1, :].detach().clone()
            state = {layer: output.past_key_values.layers[layer].recurrent_states[0].detach().clone()
                     for layer in (0, 16, 30)}
            loss = F.cross_entropy(logits.float(), target)
        after = memory()
        values = {'logits': logits.cpu(), 'loss': loss.cpu(),
                  **{f'layer{layer}.state': value.cpu() for layer, value in state.items()}}
        tmp = ROOT / 'analysis/temporary' / f'{tag}.pt'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            raise FileExistsError(tmp)
        torch.save(values, tmp)
        result = {'mode': mode, 'replicate': replicate, 'rank': rank, 'world_size': world,
                  'physical_visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES'),
                  'input_ids_sha256': tensor_sha(input_ids),
                  'sample_id': str(row.get('id', 'TRAIN:first-row')),
                  'raw_text_sha256': hashlib.sha256(row['raw_text'].encode()).hexdigest(),
                  'teacher_target_sha256': tensor_sha(target),
                  'model_source_sha256': sha(MODEL_SOURCE),
                  'v1_source_sha256': sha(V1_PATH),
                  'logits': summary(logits), 'loss': float(loss),
                  'selected_states': {str(layer): summary(value) for layer, value in state.items()},
                  'memory_before_forward': before, 'memory_after_forward': after,
                  'all_backbone_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
                  'canonical_cumsum_unmodified': True,
                  'status': 'COMPLETE'}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        if fsdp:
            dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'mode': mode, 'replicate': replicate, 'status': 'COMPLETE',
                              'logits_sha256': result['logits']['sha256'],
                              'loss': result['loss'],
                              'peak_allocated_bytes': after['peak_allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'mode': mode, 'replicate': replicate, 'rank': rank,
                   'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        if fsdp:
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['single', 'fsdp'])
    parser.add_argument('--replicate', required=True, type=int)
    a = parser.parse_args()
    run(a.mode, a.replicate)
