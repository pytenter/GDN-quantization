#!/usr/bin/env python3
"""Two-rank stage-local load gate; no model forward or training."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from stage import StaticQwenStage


def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved(),
            'device_free': free, 'device_total': total}


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two ranks and two visible GPUs required')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        before = memory()
        stage = StaticQwenStage.load(rank, torch.device('cuda', rank))
        after = memory()
        model_params = list(stage.named_parameters())
        count = sum(p.numel() for _, p in model_params)
        static_bytes = sum(p.numel() * p.element_size() for _, p in model_params)
        expected = json.loads((ROOT / 'configs/pp2_partition.json').read_text())[f'rank{rank}']
        if count != expected['parameters'] or static_bytes != expected['static_weight_bytes']:
            raise RuntimeError(f'Parameter size mismatch count={count} bytes={static_bytes}')
        if any(p.requires_grad for _, p in model_params):
            raise RuntimeError('Frozen backbone has trainable parameters')
        local_ids = set(stage.block_ids)
        if len(local_ids) != 16 or any((i < 16) != (rank == 0) for i in local_ids):
            raise RuntimeError('Incorrect stage ownership')
        result = {'rank': rank, 'physical_visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES'),
                  'block_ids': list(stage.block_ids), 'logical_parameters': count,
                  'static_weight_bytes': static_bytes, 'static_weight_gib': static_bytes / 2**30,
                  'all_backbone_frozen': True, 'memory_before': before, 'memory_after': after,
                  'STAGE_LOCAL_LOAD_GATE': 'PASS'}
        save_once(ROOT / 'analysis' / f'stage_load_rank{rank}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            other = json.loads((ROOT / 'analysis/stage_load_rank1.json').read_text())
            save_once(ROOT / 'analysis/stage_load_summary.json',
                      {'STAGE_LOCAL_LOAD_GATE': 'PASS', 'ranks': [result, other]})
            print(json.dumps({'STAGE_LOCAL_LOAD_GATE': 'PASS',
                              'rank0_static_gib': result['static_weight_gib'],
                              'rank1_static_gib': other['static_weight_gib']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'stage_load_rank{rank}.error.json',
                  {'rank': rank, 'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
