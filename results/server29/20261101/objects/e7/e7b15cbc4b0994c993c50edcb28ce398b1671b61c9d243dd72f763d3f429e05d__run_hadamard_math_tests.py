#!/usr/bin/env python3
"""Strict, model-free Hadamard algebra checks for reference-path closure."""

import argparse
import datetime as dt
import json
import math
import platform
from pathlib import Path

import torch


def sylvester_hadamard(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a positive power of two, got {n}")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.double()
    e = expected.double()
    return float(torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e).clamp_min(1e-300))


def run_case(name: str, dtype: torch.dtype, device: torch.device, limit: float) -> dict:
    generator = torch.Generator(device=device).manual_seed(20260917)
    h = sylvester_hadamard(128, device=device, dtype=dtype)
    eye = torch.eye(128, device=device, dtype=dtype)
    gram = h.transpose(0, 1) @ h
    vector = torch.randn((11, 128), generator=generator, device=device, dtype=dtype)
    matrix = torch.randn((7, 128, 128), generator=generator, device=device, dtype=dtype)
    metrics = {
        "orthogonality_max_abs": float((gram - eye).abs().max().double()),
        "orthogonality_rel_l2": rel_l2(gram, eye),
        "vector_right_roundtrip_rel_l2": rel_l2((vector @ h) @ h.transpose(0, 1), vector),
        "vector_left_roundtrip_rel_l2": rel_l2((h.transpose(0, 1) @ vector.transpose(0, 1)).transpose(0, 1) @ h, vector),
        "matrix_left_roundtrip_rel_l2": rel_l2(h.transpose(0, 1) @ (h @ matrix), matrix),
        "matrix_right_roundtrip_rel_l2": rel_l2((matrix @ h) @ h.transpose(0, 1), matrix),
    }
    maximum = max(metrics.values())
    return {
        "name": name,
        "dtype": str(dtype),
        "device": str(device),
        "threshold": limit,
        "metrics": metrics,
        "max_metric": maximum,
        "gate": "PASS" if maximum <= limit else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [
        run_case("fp64_cpu", torch.float64, torch.device("cpu"), 1e-12),
        run_case("fp32_cpu", torch.float32, torch.device("cpu"), 5e-6),
    ]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        cases.append(run_case("bf16_cuda", torch.bfloat16, torch.device("cuda"), 2e-2))
    else:
        cases.append({"name": "bf16_cuda", "gate": "UNSUPPORTED", "reason": "CUDA BF16 is unavailable"})
    fp64 = next(x for x in cases if x["name"] == "fp64_cpu")
    fp64_machine_precision_gate = fp64["max_metric"] <= 1e-12
    overall = all(x["gate"] in ("PASS", "UNSUPPORTED") for x in cases) and fp64_machine_precision_gate
    result = {
        "task": "GDN_KDA_AIME26_REFERENCE_PATH_CLOSURE_V1",
        "test": "normalized_sylvester_hadamard_algebra",
        "dimension": 128,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "formulae": {
            "orthogonality": "H^T H = I",
            "qwen_key_basis": "k_tilde = k H; S_tilde = H^T S; q_tilde = q H",
            "ling_value_basis": "v_tilde = H^T v; S_tilde = S H; o = H o_tilde",
        },
        "cases": cases,
        "fp64_near_machine_precision_gate": "PASS" if fp64_machine_precision_gate else "FAIL",
        "overall_gate": "PASS" if overall else "FAIL",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
