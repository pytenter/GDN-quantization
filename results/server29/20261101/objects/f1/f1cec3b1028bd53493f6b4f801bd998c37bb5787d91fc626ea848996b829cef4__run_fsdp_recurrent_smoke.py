#!/usr/bin/env python3
"""Frozen C5 recurrent diagnostic; only the rotation bank may be updated."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from run_fsdp_load import ROOT, import_v1, memory, save_once
from run_fsdp_forward_gate import load_model, tensor_sha


def local_sample_hash(p):
    local = p.to_local() if isinstance(p, DTensor) else p
    return tensor_sha(local.detach().reshape(-1)[:4096])


def run(mode: str, horizon: int, replicate: int, update: bool):
    fsdp = mode == 'fsdp'
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if (fsdp and (world != 2 or torch.cuda.device_count() != 2)) or (not fsdp and torch.cuda.device_count() != 1):
        raise RuntimeError('incorrect GPU topology')
    if 32 % horizon:
        raise RuntimeError('this frozen 32-token diagnostic requires H dividing 32')
    torch.cuda.set_device(rank)
    if fsdp:
        dist.init_process_group('nccl')
    tag = f'{mode}_H{horizon}_{"update" if update else "sem"}_{replicate:02d}_rank{rank}'
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        v1 = import_v1()
        model, tokenizer = load_model(fsdp, rank, world)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen backbone has trainable parameters')
        device = torch.device('cuda', rank)
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        if not all(p.requires_grad for p in bank.parameters()):
            raise RuntimeError('rotation bank unexpectedly frozen')
        h = v1.hadamard(device)
        row = v1.corpus_rows('TRAIN')[0]
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        if len(ids) != 64:
            raise RuntimeError('frozen sample shorter than 64')
        sample_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        sample_keys = ['model.layers.0.linear_attn.in_proj_qkv.weight',
                       'model.layers.16.linear_attn.in_proj_qkv.weight', 'lm_head.weight']
        model_params = dict(model.named_parameters())
        before_model_hash = {name: local_sample_hash(model_params[name]) for name in sample_keys}
        theta_before = {name: tensor_sha(p) for name, p in bank.named_parameters()}
        qdq = {}
        original_qdq = v1.CAYLEY.qwen_c128_ste
        optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0) if update else None
        if optimizer is not None:
            bank_ids = {id(p) for p in bank.parameters()}
            if {id(p) for group in optimizer.param_groups for p in group['params']} != bank_ids:
                raise RuntimeError('optimizer does not exclusively own rotation')
        with v1.RecurrentExperimentPatch(model, bank, h) as patch:
            targets = v1.collect_teacher_targets(model, patch, ids[:32], [31])
            def capture_qdq(value):
                result = original_qdq(value)
                if patch.mode == 'student' and patch.capture:
                    layer = int(patch.current_layer)
                    qdq[layer] = {'pre': value.detach().cpu().clone(),
                                  'scale': result.scale.detach().cpu().clone(),
                                  'codes': result.codes.detach().cpu().clone(),
                                  'post': result.dequant.detach().cpu().clone()}
                return result
            v1.CAYLEY.qwen_c128_ste = capture_qdq
            try:
                cache = None
                before = memory()
                torch.cuda.reset_peak_memory_stats()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                for index in range(32):
                    capture = index == 31
                    if capture:
                        v1.install_teacher_target(patch, targets[31], device)
                    cache = v1.forward_student(model, patch, ids[index], cache, capture)
                    if (index + 1) % horizon == 0 and index < 31:
                        v1.detach_cache(cache)
                if sorted(qdq) != list(v1.GDN_LAYERS):
                    raise RuntimeError(f'QDQ captured layers mismatch: {sorted(qdq)}')
                state, functional, c5_loss, c6_loss = patch.captured_losses()
                if not bool(torch.isfinite(c5_loss).detach().cpu()):
                    raise FloatingPointError('nonfinite C5 diagnostic loss')
                writeback = {str(layer): bool(torch.equal(cache.layers[layer].recurrent_states[0], patch.student_post[layer]))
                             for layer in v1.GDN_LAYERS}
                if not all(writeback.values()):
                    raise RuntimeError('recurrent state writeback mismatch')
                after_forward = memory()
                grad_summary = None
                after_backward = None
                after_optimizer = None
                if update:
                    c5_loss.backward()
                    after_backward = memory()
                    grad_summary = {}
                    for name, p in bank.named_parameters():
                        g = p.grad
                        grad_summary[name] = {'exists': g is not None,
                                              'finite': bool(torch.isfinite(g).all()) if g is not None else False,
                                              'norm': float(torch.linalg.vector_norm(g.float())) if g is not None else None,
                                              'max_abs': float(g.detach().float().abs().max()) if g is not None else None,
                                              'device': str(g.device) if g is not None else None}
                    if not all(x['exists'] and x['finite'] for x in grad_summary.values()):
                        raise RuntimeError('missing/nonfinite rotation gradient')
                    optimizer.step()
                    after_optimizer = memory()
                theta_after = {name: tensor_sha(p) for name, p in bank.named_parameters()}
                after_model_hash = {name: local_sample_hash(model_params[name]) for name in sample_keys}
                values = {f'layer{layer}.{field}': tensor for layer, fields in qdq.items()
                          for field, tensor in fields.items()}
                tmp = ROOT / 'analysis/temporary' / f'{tag}.pt'
                tmp.parent.mkdir(parents=True, exist_ok=True)
                if tmp.exists():
                    raise FileExistsError(tmp)
                torch.save(values, tmp)
                result = {'mode': mode, 'horizon': horizon, 'replicate': replicate, 'rank': rank,
                          'world_size': world, 'physical_visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES'),
                          'sample_id': str(row.get('id', 'TRAIN:first-row')),
                          'sample_token_hash': sample_hash, 'target_token_index': 31,
                          'theta0_before': theta_before,
                          'theta_after': theta_after,
                          'theta_changed_count': sum(theta_before[k] != theta_after[k] for k in theta_before),
                          'model_sample_hash_before': before_model_hash,
                          'model_sample_hash_after': after_model_hash,
                          'model_sample_unchanged': before_model_hash == after_model_hash,
                          'loss': {'state': float(state.detach()), 'functional': float(functional.detach()),
                                   'C5': float(c5_loss.detach()), 'C6': float(c6_loss.detach())},
                          'QDQ': {str(layer): {field: {'sha256': tensor_sha(tensor),
                                                     'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
                                               for field, tensor in fields.items()}
                                  for layer, fields in qdq.items()},
                          'cache_writeback_exact_by_layer': writeback,
                          'gradient': grad_summary,
                          'memory_before_rollout': before,
                          'memory_after_forward': after_forward,
                          'memory_after_backward': after_backward,
                          'memory_after_optimizer': after_optimizer,
                          'status': 'COMPLETE'}
                save_once(ROOT / 'analysis' / f'{tag}.json', result)
            finally:
                v1.CAYLEY.qwen_c128_ste = original_qdq
        if fsdp:
            dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'mode': mode, 'horizon': horizon, 'update': update,
                              'replicate': replicate, 'status': 'COMPLETE',
                              'C5_loss': result['loss']['C5'],
                              'peak_allocated_bytes': after_forward['peak_allocated_bytes']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'mode': mode, 'horizon': horizon, 'replicate': replicate,
                   'rank': rank, 'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if fsdp:
            dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['single', 'fsdp'])
    p.add_argument('--horizon', required=True, type=int, choices=[1, 4, 8, 16, 32])
    p.add_argument('--replicate', required=True, type=int)
    p.add_argument('--update', action='store_true')
    a = p.parse_args()
    run(a.mode, a.horizon, a.replicate, a.update)
