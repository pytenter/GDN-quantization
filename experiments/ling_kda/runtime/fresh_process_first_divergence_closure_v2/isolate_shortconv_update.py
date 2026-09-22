#!/usr/bin/env python3
"""Isolate the Ling decode causal-convolution update on a frozen driver."""

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


def signature(value: torch.Tensor) -> dict:
    numeric = value.detach().float().cpu()
    return {
        "sha256": tensor_hash(value),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "max_abs": float(numeric.abs().max()),
        "l2": float(torch.linalg.vector_norm(numeric)),
    }


def autotune_rows(kernel) -> list[dict]:
    current = kernel
    rows = []
    for depth in range(6):
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
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    from fla.modules.conv.triton.kernels import causal_conv1d_update_kernel
    from fla.modules.conv.triton.ops import causal_conv1d_update

    payload = torch.load(args.driver, map_location="cpu", weights_only=False)
    x = payload["x"].cuda()
    initial_cache = payload["cache"].cuda()
    weight = payload["weight"].cuda()
    bias = None if payload["bias"] is None else payload["bias"].cuda()
    residual = None if payload["residual"] is None else payload["residual"].cuda()
    outputs = []
    repeats = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            cache = initial_cache.clone()
            y, updated = causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=weight,
                bias=bias,
                activation=payload["activation"],
            )
            torch.cuda.synchronize()
            outputs.append(y.detach().cpu())
            repeats.append({
                "repeat": repeat,
                "input": signature(x),
                "cache_input": signature(initial_cache),
                "output": signature(y),
                "cache_output": signature(updated),
            })

    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "isolated frozen-driver causal_conv1d_update",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "FLA_CACHE_MODE", "FLA_CONFIG_DIR", "TRITON_CACHE_DIR")
        },
        "driver": {
            "x": signature(x),
            "cache": signature(initial_cache),
            "weight": signature(weight),
            "bias": None if bias is None else signature(bias),
            "residual": None if residual is None else signature(residual),
            "activation": payload["activation"],
            "backend": payload["backend"],
        },
        "same_process_repeatable": len({row["output"]["sha256"] for row in repeats}) == 1,
        "same_process_cache_repeatable": len({row["cache_output"]["sha256"] for row in repeats}) == 1,
        "repeats": repeats,
        "autotune": autotune_rows(causal_conv1d_update_kernel),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(outputs, args.output.with_suffix(".temporary.pt"))
    print(json.dumps({
        "run_id": args.run_id,
        "output": repeats[0]["output"]["sha256"],
        "cache": repeats[0]["cache_output"]["sha256"],
        "same_process_repeatable": result["same_process_repeatable"],
        "autotune": result["autotune"],
    }, indent=2))


if __name__ == "__main__":
    main()
