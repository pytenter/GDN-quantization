#!/usr/bin/env python3
"""Two-rank NCCL and isolated-runtime gate; no Qwen model load."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import transformers
import triton

ROOT = Path(__file__).resolve().parents[1]
ENV = Path('/data/zypan/worktrees/qwen-gdn-zero3-v1/.venv-qwen-zero3-recurrent-v1')
if Path(sys.prefix).resolve() != ENV.resolve():
    raise RuntimeError(f'Wrong runtime prefix: {sys.prefix}')
if torch.__version__ != '2.5.1+cu121' or deepspeed.__version__ != '0.16.4':
    raise RuntimeError('Unregistered Torch/DeepSpeed version')
if os.getenv('WORLD_SIZE') != '2' or torch.cuda.device_count() != 2:
    raise RuntimeError('Exactly two CUDA devices and two ranks required')
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl')
try:
    x = torch.tensor([rank + 1.0], device='cuda')
    dist.all_reduce(x)
    result = {
        'rank': rank, 'local_rank': local_rank, 'world_size': dist.get_world_size(),
        'physical_visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'nccl': list(torch.cuda.nccl.version()), 'deepspeed': deepspeed.__version__,
        'transformers': transformers.__version__, 'triton': triton.__version__,
        'all_reduce_sum': float(x.item()),
        'gpu_name': torch.cuda.get_device_name(local_rank),
        'gate': 'PASS' if x.item() == 3.0 else 'FAIL'
    }
    target = ROOT / 'analysis' / f'environment_gate_rank{rank}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        other = json.loads((ROOT / 'analysis/environment_gate_rank1.json').read_text())
        combined = {'ranks': [result, other], 'DEEPSPEED_ENVIRONMENT_GATE':
                    'PASS' if result['gate'] == other['gate'] == 'PASS' else 'FAIL'}
        with (ROOT / 'analysis/environment_gate.json').open('x') as f:
            json.dump(combined, f, indent=2, sort_keys=True)
            f.write('\n')
        print(json.dumps({'DEEPSPEED_ENVIRONMENT_GATE': combined['DEEPSPEED_ENVIRONMENT_GATE']}), flush=True)
finally:
    dist.destroy_process_group()
