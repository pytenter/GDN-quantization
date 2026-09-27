#!/usr/bin/env python3
"""Frozen Qwen ZeRO-3 load/shard gate; no model forward or optimizer step."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
MODEL_SOURCE = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
V1_PATH = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
ENV = Path('/data/zypan/worktrees/qwen-gdn-zero3-v1/.venv-qwen-zero3-recurrent-v1')


def save_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory() -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_free_bytes': free, 'device_total_bytes': total}


def import_v1():
    spec = importlib.util.spec_from_file_location('zero3_read_only_v1', V1_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_zero3(rank: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations import HfDeepSpeedConfig

    config = json.loads((ROOT / 'configs/deepspeed_zero3.json').read_text())
    hf_ds_config = HfDeepSpeedConfig(config)  # keep alive across from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    v1 = import_v1()
    # ZeRO-3 optimizer requires ds partition metadata even for this small,
    # separately-owned rotation bank. Its theta=0 construction is unchanged.
    with deepspeed.zero.Init(config_dict_or_path=config):
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS)
    bank_params = list(bank.parameters())
    optimizer = torch.optim.Adam(bank_params, lr=0.003, weight_decay=0.0)
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in bank_params}:
        raise RuntimeError('Optimizer includes a non-rotation parameter')
    engine, ds_optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=bank_params, optimizer=optimizer,
        config=config, dist_init_required=False)
    if engine.zero_optimization_stage() != 3:
        raise RuntimeError(f'Not ZeRO stage 3: {engine.zero_optimization_stage()}')
    return engine, ds_optimizer, bank, tokenizer, v1, hf_ds_config


def main():
    if Path(sys.prefix).resolve() != ENV.resolve():
        raise RuntimeError(f'Wrong isolated environment: {sys.prefix}')
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Requires exactly two visible GPUs and two ranks')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        before = memory()
        engine, ds_optimizer, bank, tokenizer, v1, hf_ds_config = load_zero3(rank)
        model = engine.module
        params = list(model.named_parameters())
        logical = sum(getattr(p, 'ds_numel', p.numel()) for _, p in params)
        local = sum(p.ds_tensor.numel() if hasattr(p, 'ds_tensor') else p.numel()
                    for _, p in params)
        zero_params = [p for _, p in params if hasattr(p, 'ds_id')]
        bank_count = sum(getattr(p, 'ds_numel', p.numel()) for p in bank.parameters())
        frozen = all(not p.requires_grad for _, p in params)
        sharded = len(zero_params) == len(params) and local < logical
        rotation_only = bank_count == 195072 and all(p.requires_grad for p in bank.parameters())
        if not (frozen and sharded and rotation_only):
            raise RuntimeError(f'Parameter gate: frozen={frozen} sharded={sharded} rotation={rotation_only}')
        result = {
            'rank': rank, 'world_size': world, 'physical_visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'torch': torch.__version__, 'deepspeed': deepspeed.__version__,
            'zero_stage': engine.zero_optimization_stage(),
            'logical_qwen_params': logical, 'local_qwen_param_elements': local,
            'zero_managed_model_params': len(zero_params), 'model_param_tensors': len(params),
            'trainable_qwen_params': sum(getattr(p, 'ds_numel', p.numel()) for _, p in params if p.requires_grad),
            'trainable_rotation_params': bank_count,
            'rotation_names': [name for name, _ in bank.named_parameters()],
            'rotation_bank_in_engine_module': any(id(p) == id(q) for p in bank.parameters() for q in model.parameters()),
            'rotation_zero_managed_count': sum(hasattr(p, 'ds_id') for p in bank.parameters()),
            'rotation_persistent_count': sum(bool(getattr(p, 'ds_persist', False)) for p in bank.parameters()),
            'optimizer_initially_rotation_only': True,
            'model_source_sha256': sha(MODEL_SOURCE),
            'model_config_sha256': sha(MODEL_PATH / 'config.json'),
            'v1_source_sha256': sha(V1_PATH),
            'quantizer_rotation_source_sha256': sha(v1.CAYLEY_SOURCE),
            'memory_before': before, 'memory_after': memory(),
            'TRAINABLE_PARAMETER_GATE': 'PASS', 'PARAMETER_SHARDING_GATE': 'PASS',
        }
        attempt = os.getenv('LOAD_ATTEMPT', '01')
        save_once(ROOT / 'analysis' / f'zero3_load_attempt{attempt}_rank{rank}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            other = json.loads((ROOT / 'analysis' / f'zero3_load_attempt{attempt}_rank1.json').read_text())
            summary = {'ranks': [result, other], 'ZERO3_LOAD_GATE': 'PASS'}
            save_once(ROOT / 'analysis' / f'zero3_load_attempt{attempt}_memory.json', summary)
            save_once(ROOT / 'analysis/trainable_parameter_inventory.json', {
                'logical_qwen_params': logical, 'frozen_qwen_params': logical,
                'trainable_qwen_params': 0, 'trainable_rotation_params': bank_count,
                'optimizer_initially_rotation_only': True, 'rotation_bank_zero_managed': True,
                'backbone_zero3_sharded': True, 'gate': 'PASS'})
            print(json.dumps({'ZERO3_LOAD_GATE': 'PASS', 'logical_qwen_params': logical,
                              'local_qwen_param_elements': local,
                              'memory_allocated_bytes': result['memory_after']['allocated_bytes']}), flush=True)
    except Exception as exc:
        attempt = os.getenv('LOAD_ATTEMPT', '01')
        save_once(ROOT / 'analysis' / f'zero3_load_attempt{attempt}_rank{rank}.error.json', {
            'rank': rank, 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
