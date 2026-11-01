#!/usr/bin/env python3
"""Read-only public-hook and saved-view census on a passing reduced 3-block N4 graph."""
import json
import os
import traceback
from collections import Counter

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from run_canonical_gdn_reentry import import_v1
from run_real_gdn_block_reentry import ROOT, mem, param_state, save_once
from run_small_stack_reentry import SmallStack, load_layers


def run():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two visible GPUs/ranks required')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    tag = f'lifecycle_stack3_N4_rank{rank}'
    handles = []
    try:
        torch.manual_seed(20260927)
        torch.cuda.manual_seed_all(20260927)
        layers, config, conv_hashes = load_layers(rank, 3)
        stack = SmallStack(layers)
        mesh = init_device_mesh('cuda', (world,))
        for layer in stack.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(stack, mesh=mesh, reshard_after_forward=True)
        v1 = import_v1()
        device = torch.device('cuda', rank)
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        h = v1.hadamard(device)
        hidden = torch.randn((1, 1, config.hidden_size), device=device,
                             dtype=torch.bfloat16, requires_grad=True)
        initial = hidden
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=config)
        events = []
        saved_shapes = Counter()
        special = []
        unpacked = []
        hook_counts = Counter()

        def snapshot(layer, event):
            hook_counts[event] += 1
            events.append({'event': event, 'layer': layer,
                           'conv_weight': param_state(stack.model.layers[layer].linear_attn.conv1d.weight),
                           'memory': mem(rank)})

        for layer_index, layer in enumerate(stack.model.layers):
            handles.append(layer.register_forward_pre_hook(
                lambda _m, _args, i=layer_index: snapshot(i, 'forward_pre')))
            handles.append(layer.register_forward_hook(
                lambda _m, _args, _out, i=layer_index: snapshot(i, 'forward_post')))
            handles.append(layer.register_full_backward_pre_hook(
                lambda _m, _grad, i=layer_index: snapshot(i, 'module_backward_pre')))
            handles.append(layer.register_full_backward_hook(
                lambda _m, _grad_in, _grad_out, i=layer_index: snapshot(i, 'module_backward_post')))

        def tensor_meta(t):
            try:
                storage = t.untyped_storage()
                ptr, nbytes = storage.data_ptr(), storage.nbytes()
            except Exception:
                ptr, nbytes = None, None
            base = getattr(t, '_base', None)
            try:
                base_ptr = base.untyped_storage().data_ptr() if base is not None else None
                base_nbytes = base.untyped_storage().nbytes() if base is not None else None
            except Exception:
                base_ptr, base_nbytes = None, None
            owner_matches = []
            for index, layer in enumerate(stack.model.layers):
                weight = layer.linear_attn.conv1d.weight
                try:
                    if ptr == weight.untyped_storage().data_ptr():
                        owner_matches.append(f'layer{index}.linear_attn.conv1d.weight')
                except Exception:
                    pass
            return {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device),
                    'storage_nbytes': nbytes, 'storage_data_ptr': ptr,
                    'is_view': base is not None, 'base_shape': list(base.shape) if base is not None else None,
                    'base_storage_nbytes': base_nbytes, 'base_storage_data_ptr': base_ptr,
                    'direct_conv_storage_owner_matches': owner_matches}

        def interesting(t):
            dims = list(t.shape)
            return (8192 in dims and 4 in dims and t.numel() <= 32768) or dims in (
                [8192, 1, 4], [8192, 1, 1, 4])

        def pack(t):
            saved_shapes[(str(tuple(t.shape)), str(t.dtype))] += 1
            if interesting(t) and len(special) < 200:
                special.append({'phase': 'pack', **tensor_meta(t)})
            return t

        def unpack(t):
            if interesting(t) and len(unpacked) < 200:
                unpacked.append({'phase': 'unpack', **tensor_meta(t)})
            return t

        torch.cuda.empty_cache()
        before = mem(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            with v1.RecurrentExperimentPatch(stack, bank, h) as patch:
                patch.mode = 'student'
                patch.capture = False
                for step in range(4):
                    events.append({'event': f'entry_{step+1}',
                                   'conv_weight': param_state(stack.model.layers[0].linear_attn.conv1d.weight),
                                   'memory': mem(rank)})
                    hidden = stack(hidden, cache)
                    v1.enable_differentiable_cache(cache)
                    events.append({'event': f'exit_{step+1}',
                                   'conv_weight': param_state(stack.model.layers[0].linear_attn.conv1d.weight),
                                   'memory': mem(rank)})
                loss = hidden.float().square().mean()
                events.append({'event': 'backward_pre',
                               'conv_weight': param_state(stack.model.layers[0].linear_attn.conv1d.weight),
                               'memory': mem(rank)})
                loss.backward()
                after = mem(rank)
                events.append({'event': 'backward_post',
                               'conv_weight': param_state(stack.model.layers[0].linear_attn.conv1d.weight),
                               'memory': after})
        gradient_valid = (initial.grad is not None and bool(torch.isfinite(initial.grad).all())
                          and float(initial.grad.float().abs().max()) > 0)
        if not gradient_valid:
            raise RuntimeError('instrumented input gradient missing/nonfinite/zero')
        baseline = json.loads((ROOT / 'analysis' / f'stack3_N4_reshard1_rank{rank}.json').read_text())
        loss_matches_baseline = float(loss.detach()) == baseline['loss']
        result = {'status': 'PASS', 'rank': rank, 'module': '3 real GDN decoder blocks',
                  'reentries': 4, 'reshard_after_forward': True,
                  'DIAGNOSTIC_PRIVATE_API_READ_ONLY': False,
                  'hook_perturbation_warning': 'public full backward hooks and saved_tensors_hooks may alter view/storage lifetimes; not proof of parent behavior',
                  'checkpoint_conv_hashes': conv_hashes,
                  'hook_counts': dict(hook_counts), 'events': events,
                  'saved_tensor_count': sum(saved_shapes.values()),
                  'saved_tensor_shape_counts': [{'shape': s, 'dtype': d, 'count': count}
                                                for (s, d), count in saved_shapes.most_common(50)],
                  'special_saved_pack': special, 'special_saved_unpack': unpacked,
                  'memory_before': before, 'memory_after_backward': after,
                  'input_gradient_valid': gradient_valid,
                  'loss': float(loss.detach()), 'loss_matches_uninstrumented_baseline': loss_matches_baseline}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'status': 'PASS', 'hook_counts': dict(hook_counts),
                              'special_saved': len(special), 'special_unpacked': len(unpacked)}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json',
                  {'status': 'FAIL', 'rank': rank, 'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        for handle in handles:
            handle.remove()
        dist.destroy_process_group()


if __name__ == '__main__':
    run()
