#!/usr/bin/env python3
"""Bounded copy and NCCL microbenchmark on the audited GPU pair."""

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
CONFIG = json.loads((ROOT / "configs/p2p_microbenchmark.json").read_text())
PAIR = CONFIG["selected_physical_gpu_pair"]


def new_json(relative: str, value) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def preflight() -> None:
    if PAIR != [0, 1]:
        raise RuntimeError("unexpected physical pair")
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in query.splitlines() if (parts := [x.strip() for x in line.split(",")])}
    for gpu in PAIR:
        used, util = rows[gpu]
        if used >= 1024 or util >= 10:
            raise RuntimeError(f"GPU_RESOURCE_GATE_BLOCKED {gpu}: {used} MiB, {util}%")


def summary(samples: list[float], size_bytes: int) -> dict:
    middle = statistics.median(samples)
    return {"samples_seconds": samples, "median_latency_ms": middle * 1000,
            "effective_bandwidth_GBps": size_bytes / middle / 1e9}


def copies() -> None:
    preflight()
    result = {"selected_pair": PAIR,
              "direct_peer_access": {"0->1": torch.cuda.can_device_access_peer(0, 1),
                                     "1->0": torch.cuda.can_device_access_peer(1, 0)},
              "measurements": []}
    for src_gpu, dst_gpu in ((0, 1), (1, 0)):
        for size_mib in CONFIG["sizes_mib"]:
            size_bytes = size_mib * 1024 * 1024
            source = torch.ones(size_bytes, dtype=torch.uint8, device=f"cuda:{src_gpu}")
            target = torch.empty_like(source, device=f"cuda:{dst_gpu}")
            samples = []
            for iteration in range(CONFIG["warmup_iterations"] + CONFIG["timed_iterations"]):
                torch.cuda.synchronize(src_gpu)
                torch.cuda.synchronize(dst_gpu)
                start = time.perf_counter()
                target.copy_(source, non_blocking=True)
                torch.cuda.synchronize(dst_gpu)
                elapsed = time.perf_counter() - start
                if iteration >= CONFIG["warmup_iterations"]:
                    samples.append(elapsed)
            result["measurements"].append({"operation": "cuda_cross_device_copy",
                                           "source_gpu": src_gpu, "destination_gpu": dst_gpu,
                                           "size_mib": size_mib,
                                           **summary(samples, size_bytes)})
            del source, target
    new_json("analysis/p2p_copy.json", result)
    print(json.dumps({"copy_microbenchmark": "COMPLETE", "measurements": len(result["measurements"])}), flush=True)


def distributed() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    if rank not in (0, 1) or os.environ.get("NCCL_ALGO") != "Ring":
        raise RuntimeError("invalid frozen NCCL rank/environment")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    measurements = []
    try:
        for size_mib in CONFIG["sizes_mib"]:
            size_bytes = size_mib * 1024 * 1024
            payload = torch.ones(size_bytes, dtype=torch.uint8, device=f"cuda:{rank}")
            recv = torch.empty_like(payload)
            for operation in ("send_recv_0_to_1", "send_recv_1_to_0", "all_reduce"):
                samples = []
                for iteration in range(CONFIG["warmup_iterations"] + CONFIG["timed_iterations"]):
                    torch.cuda.synchronize(rank)
                    dist.barrier()
                    start = time.perf_counter()
                    if operation == "send_recv_0_to_1":
                        dist.send(payload, dst=1) if rank == 0 else dist.recv(recv, src=0)
                    elif operation == "send_recv_1_to_0":
                        dist.send(payload, dst=0) if rank == 1 else dist.recv(recv, src=1)
                    else:
                        dist.all_reduce(payload)
                    torch.cuda.synchronize(rank)
                    elapsed = time.perf_counter() - start
                    local = torch.tensor([elapsed], dtype=torch.float64, device=f"cuda:{rank}")
                    dist.all_reduce(local, op=dist.ReduceOp.MAX)
                    if iteration >= CONFIG["warmup_iterations"]:
                        samples.append(float(local.item()))
                measurements.append({"operation": operation, "size_mib": size_mib,
                                     **summary(samples, size_bytes)})
            del payload, recv
        if rank == 0:
            result = {"selected_pair": PAIR, "backend": "nccl", "NCCL_ALGO": "Ring",
                      "copy": json.loads((ROOT / "analysis/p2p_copy.json").read_text()),
                      "nccl_measurements": measurements,
                      "interpretation": "effective two-rank application bandwidth, not physical link peak"}
            new_json("analysis/p2p_bandwidth.json", result)
            print(json.dumps({"nccl_microbenchmark": "COMPLETE", "measurements": len(measurements)}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("copy", "nccl"), required=True)
    args = parser.parse_args()
    copies() if args.phase == "copy" else distributed()
