#!/usr/bin/env python3
"""Isolate FLA l2norm_fwd on frozen Q/K drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def config_rows(kernel):
    current = kernel
    rows = []
    for depth in range(5):
        cache = getattr(current, "cache", None)
        if isinstance(cache, dict):
            rows.append({
                "depth": depth,
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "cache": [{
                    "key": str(key), "kwargs": dict(config.kwargs),
                    "num_warps": config.num_warps, "num_stages": config.num_stages,
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
    args = parser.parse_args()
    import fla.modules.l2norm as lm
    payload = torch.load(args.driver_bundle, map_location="cpu", weights_only=False)
    q, k = payload["q"].cuda(), payload["k"].cuda()
    with torch.inference_mode():
        qn, qr = lm.l2norm_fwd(q)
        kn, kr = lm.l2norm_fwd(k)
        torch.cuda.synchronize()
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "isolated frozen-driver fla.modules.l2norm_fwd",
        "run_id": args.run_id, "pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ("FLA_CACHE_MODE", "FLA_CONFIG_DIR", "TRITON_CACHE_DIR")},
        "q_input_sha256": tensor_hash(q), "k_input_sha256": tensor_hash(k),
        "q_output_sha256": tensor_hash(qn), "k_output_sha256": tensor_hash(kn),
        "q_rstd_sha256": tensor_hash(qr), "k_rstd_sha256": tensor_hash(kr),
        "autotune": config_rows(lm.l2norm_fwd_kernel),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({"q": qn.detach().cpu(), "k": kn.detach().cpu()}, args.output.with_suffix(".temporary.pt"))
    print(json.dumps({"run_id": args.run_id, "q": result["q_output_sha256"], "k": result["k_output_sha256"], "autotune": result["autotune"]}, indent=2))


if __name__ == "__main__":
    main()
