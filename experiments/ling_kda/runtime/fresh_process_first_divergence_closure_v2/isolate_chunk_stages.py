#!/usr/bin/env python3
"""Fresh-process stage ladder inside FLA chunk_kda_fwd on frozen drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


ORDER = ("q_l2norm", "k_l2norm", "g_chunk_cumsum", "Aqk", "Akk", "intermediate_h", "final_state", "output")


def sig(value):
    if value is None:
        return None
    cpu = value.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    x = cpu.double()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(), "dtype": str(cpu.dtype), "shape": list(cpu.shape),
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
        "norm": float(torch.linalg.vector_norm(x)), "nonfinite": int((~torch.isfinite(x)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd
    payload = torch.load(args.driver_bundle, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    q, k, v = (payload[name].to(device) for name in ("q", "k", "v"))
    g, beta = payload["g"].float().to(device), payload["beta"].to(device)
    with torch.inference_mode():
        qn, _ = l2norm_fwd(q)
        kn, _ = l2norm_fwd(k)
        values = chunk_kda_fwd(
            q=qn, k=kn, v=v, g=g, beta=beta, scale=1.0 / math.sqrt(q.shape[-1]),
            initial_state=None, output_final_state=True, use_gate_in_kernel=False,
            chunk_size=64, return_intermediate_states=True,
        )
        torch.cuda.synchronize(device)
    o, final_state, g_cumsum, Aqk, Akk, _w, _u, _qg, _kg, _v_new, h, _initial = values
    stages = {"q_l2norm": qn, "k_l2norm": kn, "g_chunk_cumsum": g_cumsum, "Aqk": Aqk,
              "Akk": Akk, "intermediate_h": h, "final_state": final_state, "output": o}
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "frozen-driver chunk_kda internal stage ladder",
        "run_id": args.run_id, "pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ("FLA_CACHE_MODE", "FLA_CONFIG_DIR", "TRITON_CACHE_DIR")},
        "order": list(ORDER), "stages": {name: sig(stages[name]) for name in ORDER},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"run_id": args.run_id, "hashes": {name: result["stages"][name]["sha256"] for name in ORDER}}, indent=2))


if __name__ == "__main__":
    main()
