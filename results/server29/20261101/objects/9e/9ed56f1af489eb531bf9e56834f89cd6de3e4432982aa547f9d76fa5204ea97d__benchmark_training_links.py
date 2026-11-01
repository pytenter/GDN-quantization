#!/usr/bin/env python3
"""Bounded 6/7 copy and NCCL benchmark; never touches other GPUs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/p2p_microbenchmark_training_server.json').read_text())
PAIR = CONFIG['selected_physical_gpu_pair']


def preflight() -> None:
    if PAIR != [6, 7] or os.environ.get('CUDA_VISIBLE_DEVICES') != '6,7':
        raise RuntimeError('GPU pair or visibility differs from frozen 6/7 configuration')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expected exactly two visible GPUs')
    query = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in query.splitlines()
            if (parts := [x.strip() for x in line.split(',')])}
    for gpu in PAIR:
        used, util = rows[gpu]
        if used >= 1024 or util >= 10:
            raise RuntimeError(f'GPU_RESOURCE_GATE_BLOCKED GPU{gpu}: {used} MiB, {util}%')


def summarize(samples: list[float], bytes_transferred: int) -> dict:
    median = statistics.median(samples)
    return {'samples_seconds': samples, 'median_latency_ms': median * 1000,
            'effective_bandwidth_GBps': bytes_transferred / median / 1e9}


def write_once(relative: str, value: dict) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def copy_phase() -> None:
    preflight()
    result = {'selected_physical_gpu_pair': PAIR,
              'direct_peer_access': {
                  '6_to_7': torch.cuda.can_device_access_peer(0, 1),
                  '7_to_6': torch.cuda.can_device_access_peer(1, 0)},
              'measurements': []}
    for src, dst in ((0, 1), (1, 0)):
        for size_mib in CONFIG['sizes_mib']:
            nbytes = size_mib * 1024 * 1024
            source = torch.ones(nbytes, dtype=torch.uint8, device=f'cuda:{src}')
            target = torch.empty(nbytes, dtype=torch.uint8, device=f'cuda:{dst}')
            samples = []
            for iteration in range(CONFIG['warmup_iterations'] + CONFIG['timed_iterations']):
                torch.cuda.synchronize(src)
                torch.cuda.synchronize(dst)
                start = time.perf_counter()
                target.copy_(source, non_blocking=True)
                torch.cuda.synchronize(dst)
                elapsed = time.perf_counter() - start
                if iteration >= CONFIG['warmup_iterations']:
                    samples.append(elapsed)
            result['measurements'].append({
                'operation': 'cuda_cross_device_copy', 'source_physical_gpu': PAIR[src],
                'destination_physical_gpu': PAIR[dst], 'size_mib': size_mib,
                **summarize(samples, nbytes)})
            del source, target
    write_once('analysis/training_server_p2p_copy.json', result)
    print(json.dumps({'phase': 'copy', 'measurements': len(result['measurements'])}), flush=True)


def nccl_phase() -> None:
    if os.environ.get('NCCL_ALGO') != CONFIG['NCCL_ALGO']:
        raise RuntimeError('NCCL_ALGO differs from frozen Ring setting')
    rank = int(os.environ['LOCAL_RANK'])
    if rank not in (0, 1):
        raise RuntimeError('Expected two ranks')
    # A running rank may legitimately occupy its own GPU; the launch preflight
    # is performed separately before torchrun rather than here.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6,7':
        raise RuntimeError('Unexpected CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=2)
    measurements = []
    try:
        for size_mib in CONFIG['sizes_mib']:
            nbytes = size_mib * 1024 * 1024
            payload = torch.ones(nbytes, dtype=torch.uint8, device=f'cuda:{rank}')
            received = torch.empty_like(payload)
            for op in ('send_recv_6_to_7', 'send_recv_7_to_6', 'all_reduce'):
                samples = []
                for iteration in range(CONFIG['warmup_iterations'] + CONFIG['timed_iterations']):
                    torch.cuda.synchronize(rank)
                    dist.barrier()
                    start = time.perf_counter()
                    if op == 'send_recv_6_to_7':
                        if rank == 0:
                            dist.send(payload, dst=1)
                        else:
                            dist.recv(received, src=0)
                    elif op == 'send_recv_7_to_6':
                        if rank == 1:
                            dist.send(payload, dst=0)
                        else:
                            dist.recv(received, src=1)
                    else:
                        dist.all_reduce(payload)
                    torch.cuda.synchronize(rank)
                    elapsed = time.perf_counter() - start
                    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=f'cuda:{rank}')
                    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                    if iteration >= CONFIG['warmup_iterations']:
                        samples.append(elapsed_tensor.item())
                measurements.append({'operation': op, 'size_mib': size_mib,
                                     **summarize(samples, nbytes)})
            del payload, received
        if rank == 0:
            result = {'selected_physical_gpu_pair': PAIR, 'backend': 'nccl',
                      'NCCL_ALGO': 'Ring',
                      'copy': json.loads((ROOT / 'analysis/training_server_p2p_copy.json').read_text()),
                      'nccl_measurements': measurements,
                      'interpretation': CONFIG['interpretation']}
            write_once('analysis/training_server_p2p_bandwidth.json', result)
            print(json.dumps({'phase': 'nccl', 'measurements': len(measurements)}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('copy', 'nccl'), required=True)
    arguments = parser.parse_args()
    copy_phase() if arguments.phase == 'copy' else nccl_phase()
