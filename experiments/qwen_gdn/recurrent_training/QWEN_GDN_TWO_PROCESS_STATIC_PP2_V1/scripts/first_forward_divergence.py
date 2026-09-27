#!/usr/bin/env python3
"""One planned first-forward-divergence audit after PP2 teacher gate failure."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from stage import StaticQwenStage
from run_single_teacher_reference import scalar_summary, sha, save_once

MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
V1 = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')


def sample(tokenizer):
    import importlib.util
    spec = importlib.util.spec_from_file_location('pp2_first_divergence_v1', V1)
    v1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v1
    spec.loader.exec_module(v1)
    ids, _ = v1.tokenize(tokenizer, v1.corpus_rows('TRAIN')[0])
    if len(ids[:64]) != 64:
        raise RuntimeError('Frozen sample shorter than 64')
    return ids[:64]


def attach_hooks(layer_map):
    captured = {}
    handles = []
    def hook(name):
        def fn(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().cpu().clone()
        return fn
    for block_id, layer in layer_map.items():
        handles.append(layer.register_forward_hook(hook(f'block.{block_id}')))
        if block_id in (0, 1):
            for name in ('input_layernorm', 'linear_attn', 'post_attention_layernorm', 'mlp'):
                module = getattr(layer, name, None)
                if module is not None:
                    handles.append(module.register_forward_hook(hook(f'block.{block_id}.{name}')))
    return captured, handles


def summarize(captured):
    return {key: scalar_summary(value) for key, value in captured.items()}


def single():
    if torch.cuda.device_count() != 1:
        raise RuntimeError('single reference needs one GPU')
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                                  local_files_only=True)
        ids = sample(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to('cuda:0')
        layers = {i: layer for i, layer in enumerate(model.model.layers)}
        captured, handles = attach_hooks(layers)
        input_ids = torch.tensor([ids], device='cuda:0')
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
        for handle in handles:
            handle.remove()
        result = {'mode': 'single', 'input_ids_sha256': sha(input_ids),
                  'outer_attn_implementation': str(model.config._attn_implementation),
                  'text_attn_implementation': str(model.model.config._attn_implementation),
                  'blocks_and_submodules': summarize(captured),
                  'logits_sha256': sha(output.logits[:, -1, :])}
        save_once(ROOT / 'analysis/first_divergence_single.json', result)
        print(json.dumps({'mode': 'single', 'block0': result['blocks_and_submodules']['block.0']['sha256'],
                          'block15': result['blocks_and_submodules']['block.15']['sha256']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis/first_divergence_single.error.json',
                  {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise


def pp2():
    if os.getenv('WORLD_SIZE') != '2' or torch.cuda.device_count() != 2:
        raise RuntimeError('pp2 needs two ranks and two visible GPUs')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        stage = StaticQwenStage.load(rank, torch.device('cuda', rank))
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                                  local_files_only=True)
        ids = sample(tokenizer)
        input_ids = torch.tensor([ids], device=f'cuda:{rank}')
        layers = {i: stage.layers[str(i)] for i in stage.block_ids}
        captured, handles = attach_hooks(layers)
        if rank == 0:
            with torch.no_grad():
                hidden, _, cache = stage.forward_stage(input_ids=input_ids,
                                                         start_position=0)
            dist.send(hidden.detach().contiguous(), dst=1)
            logits_sha = None
        else:
            wire = torch.empty((1, len(ids), stage.config.hidden_size),
                               device='cuda:1', dtype=torch.bfloat16)
            dist.recv(wire, src=0)
            with torch.no_grad():
                _, logits, cache = stage.forward_stage(boundary_hidden=wire,
                                                        start_position=0)
            logits_sha = sha(logits[:, -1, :])
        for handle in handles:
            handle.remove()
        result = {'mode': 'pp2', 'rank': rank, 'input_ids_sha256': sha(input_ids),
                  'text_attn_implementation': str(stage.config._attn_implementation),
                  'blocks_and_submodules': summarize(captured),
                  'logits_sha256': logits_sha}
        save_once(ROOT / 'analysis' / f'first_divergence_pp2_rank{rank}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({'mode': 'pp2', 'rank0_block0': result['blocks_and_submodules']['block.0']['sha256']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'first_divergence_pp2_rank{rank}.error.json',
                  {'rank': rank, 'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


def analyze():
    ref = json.loads((ROOT / 'analysis/first_divergence_single.json').read_text())
    r0 = json.loads((ROOT / 'analysis/first_divergence_pp2_rank0.json').read_text())
    r1 = json.loads((ROOT / 'analysis/first_divergence_pp2_rank1.json').read_text())
    mismatches = []
    comparison = {}
    for block_id in range(32):
        key = f'block.{block_id}'
        pp = (r0 if block_id < 16 else r1)['blocks_and_submodules'][key]
        single = ref['blocks_and_submodules'][key]
        same = pp['sha256'] == single['sha256']
        comparison[key] = {'sha256_equal': same,
                           'reference_l2': single['l2'], 'pp2_l2': pp['l2'],
                           'reference_mean': single['mean'], 'pp2_mean': pp['mean']}
        if not same:
            mismatches.append(block_id)
    early = {}
    for block_id in (0,1):
        for suffix in ('input_layernorm', 'linear_attn', 'post_attention_layernorm', 'mlp'):
            key = f'block.{block_id}.{suffix}'
            a = ref['blocks_and_submodules'][key]
            b = r0['blocks_and_submodules'][key]
            early[key] = {'sha256_equal': a['sha256'] == b['sha256'],
                          'reference_l2': a['l2'], 'pp2_l2': b['l2']}
    result = {'first_divergent_block': mismatches[0] if mismatches else None,
              'all_divergent_blocks': mismatches, 'block_comparison': comparison,
              'early_submodule_comparison': early,
              'reference_attn_implementation': ref['text_attn_implementation'],
              'pp2_attn_implementation': r0['text_attn_implementation'],
              'first_forward_divergence_localization': 'COMPLETE',
              'root_cause': 'UNRESOLVED_WITHIN_ONE_DIAGNOSTIC',
              'next_action': 'STOP_PER_FROZEN_FORWARD_FAILURE_RULE'}
    save_once(ROOT / 'analysis/first_forward_divergence_localization.json', result)
    print(json.dumps({'first_divergent_block': result['first_divergent_block'],
                      'early_submodules': early,
                      'reference_attn': result['reference_attn_implementation'],
                      'pp2_attn': result['pp2_attn_implementation']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['single', 'pp2', 'analyze'], required=True)
    phase = parser.parse_args().phase
    {'single': single, 'pp2': pp2, 'analyze': analyze}[phase]()
