#!/usr/bin/env python3
"""Isolate the Ling attention gated RMSNorm on frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def config_rows(kernel) -> list[dict]:
    current = kernel
    rows = []
    for depth in range(5):
        cache = getattr(current, "cache", None)
        if isinstance(cache, dict):
            rows.append({
                "depth": depth,
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "cache": [{
                    "key": str(key),
                    "kwargs": dict(config.kwargs),
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                    "num_ctas": getattr(config, "num_ctas", None),
                } for key, config in cache.items()],
            })
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
    parser.add_argument("--fixed-config", action="store_true")
    args = parser.parse_args()

    import fla.modules.fused_norm_gate as module
    if args.fixed_config:
        import triton
        current = module.layer_norm_gated_fwd_kernel
        while current is not None:
            if hasattr(current, "configs"):
                current.configs = [triton.Config({"BT": 16}, num_warps=8, num_stages=3)]
                if isinstance(getattr(current, "cache", None), dict):
                    current.cache.clear()
            current = getattr(current, "fn", None)

    payload = torch.load(args.driver_bundle, map_location="cpu", weights_only=False)
    x = payload["x"].cuda()
    g = payload["g"].cuda()
    weight = payload["weight"].cuda()
    outputs = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            output = module.rms_norm_gated(
                x, g, weight, None, activation="sigmoid", eps=float(payload["eps"])
            )
            torch.cuda.synchronize()
            outputs.append(output.detach().cpu())

    hashes = [tensor_hash(value) for value in outputs]
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "isolated frozen-input fla.modules.rms_norm_gated",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ("FLA_CACHE_MODE", "FLA_CONFIG_DIR", "TRITON_CACHE_DIR")},
        "input_sha256": tensor_hash(x),
        "gate_sha256": tensor_hash(g),
        "weight_sha256": tensor_hash(weight),
        "output_sha256": hashes[0],
        "same_process_hashes": hashes,
        "same_process_repeatable": len(set(hashes)) == 1,
        "diagnostic_fixed_config": bool(args.fixed_config),
        "autotune": config_rows(module.layer_norm_gated_fwd_kernel),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({"output": outputs[0]}, args.output.with_suffix(".temporary.pt"))
    print(json.dumps({"run_id": args.run_id, "output": hashes[0], "same_process_repeatable": result["same_process_repeatable"], "autotune": result["autotune"]}, indent=2))


if __name__ == "__main__":
    main()
