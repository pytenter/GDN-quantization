#!/usr/bin/env python3
"""Run only chunk_kda on one frozen driver bundle in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_summary(value: torch.Tensor) -> dict:
    x = value.detach().double()
    return {
        "sha256": tensor_hash(value), "dtype": str(value.dtype), "shape": list(value.shape),
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
        "norm": float(torch.linalg.vector_norm(x)), "nonfinite": int((~torch.isfinite(x)).sum()),
    }


def serialize_config(value):
    if value is None:
        return None
    return {
        "kwargs": dict(getattr(value, "kwargs", {}) or {}),
        "num_warps": getattr(value, "num_warps", None),
        "num_stages": getattr(value, "num_stages", None),
        "num_ctas": getattr(value, "num_ctas", None),
        "repr": repr(value),
    }


def autotune_snapshot() -> dict:
    import fla.ops.common.chunk_delta_h as delta_h
    import fla.ops.kda.chunk_intra as intra
    import fla.ops.kda.chunk_intra_token_parallel as token_parallel
    import fla.ops.kda.gate as gate
    import fla.ops.kda.wy_fast as wy_fast
    import fla.ops.utils.cumsum as cumsum
    import fla.modules.l2norm as l2norm
    import fla.modules.fused_norm_gate as fused_norm_gate
    import fla.ops.gla.chunk as gla_chunk
    rows = {}
    for module in (l2norm, fused_norm_gate, cumsum, gate, intra, token_parallel, wy_fast, delta_h, gla_chunk):
        for name, value in vars(module).items():
            current = value
            for depth in range(5):
                if hasattr(current, "cache"):
                    best = getattr(current, "best_config", None)
                    cache = getattr(current, "cache", None)
                    cache_rows = {}
                    if isinstance(cache, dict):
                        for key, config in cache.items():
                            cache_rows[str(key)] = serialize_config(config)
                    rows[f"{module.__name__}.{name}.depth{depth}"] = {
                        "best_config": serialize_config(best),
                        "cache": cache_rows,
                        "type": f"{type(current).__module__}.{type(current).__name__}",
                        "candidate_configs": [serialize_config(config) for config in getattr(current, "configs", [])],
                    }
                current = getattr(current, "fn", None)
                if current is None:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    from fla.ops.kda import chunk_kda
    payload = torch.load(args.driver_bundle, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    q = payload["q"].to(device)
    k = payload["k"].to(device)
    v = payload["v"].to(device)
    g = payload["g"].float().to(device)
    beta = payload["beta"].to(device)
    initial = payload.get("initial_state")
    initial = None if initial is None else initial.float().to(device)
    rows = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            before = time.time()
            output, state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=initial,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
            )
            torch.cuda.synchronize(device)
            rows.append({
                "repeat": repeat,
                "output": tensor_summary(output),
                "state": tensor_summary(state),
                "runtime_seconds": time.time() - before,
                "autotune": autotune_snapshot(),
            })
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "isolated frozen-driver chunk_kda",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "process_started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {key: os.environ.get(key) for key in (
            "FLA_CACHE_MODE", "FLA_CONFIG_DIR", "TRITON_CACHE_DIR", "CUBLAS_WORKSPACE_CONFIG",
        )},
        "driver_sha256": {key: tensor_hash(value) for key, value in payload.items() if torch.is_tensor(value)},
        "same_process_runs": rows,
        "same_process_repeatable": len({row["output"]["sha256"] + row["state"]["sha256"] for row in rows}) == 1,
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "run_id": args.run_id,
        "state_sha256": rows[0]["state"]["sha256"],
        "same_process_repeatable": result["same_process_repeatable"],
        "first_runtime_seconds": rows[0]["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
