#!/usr/bin/env python3
"""Materialize two historical theta checkpoints as explicitly noncanonical candidates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
CANONICAL_REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
CORE_PATH = CANONICAL_REPO / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py"
EXPECTED = {
    "C5_U113": (
        ROOT / "candidate_artifacts/c5_u113_theta.pt",
        "ecdc13b64c65757b9b50c85e4a85352185564f3dd62dcb3e91758a6cf5638a11",
        "C5",
        113,
        904,
    ),
    "C6_U119": (
        ROOT / "candidate_artifacts/c6_u119_theta.pt",
        "a580b847729130faee208be0beb95a2179a18d5b7efd3cda57dadb9beec29c36",
        "C6",
        119,
        952,
    ),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: object) -> None:
    import torch

    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    started = time.time()
    core = import_file(CORE_PATH, "qwen_candidate_export_core")
    results = []
    identity = torch.eye(128, dtype=torch.float32)
    for label, (theta_path, expected_sha, family, update, exposure) in EXPECTED.items():
        theta_sha = file_sha256(theta_path)
        if theta_sha != expected_sha:
            raise RuntimeError(f"{label} theta SHA mismatch: {theta_sha}")
        payload = torch.load(theta_path, map_location="cpu", weights_only=False)
        if payload.get("condition") != family or int(payload.get("optimizer_updates")) != update:
            raise RuntimeError(f"{label} theta metadata mismatch")
        if int(payload.get("target_exposures")) != exposure or payload.get("AIME_used") is not False:
            raise RuntimeError(f"{label} exposure/AIME metadata mismatch")
        bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS).to(device)
        bank.load_state_dict(payload["bank"], strict=True)
        matrices, per_layer = {}, {}
        for layer in core.GDN_LAYERS:
            matrix = bank.layer(layer).matrix().detach().float().cpu().contiguous()
            residual = float((matrix.T @ matrix - identity).abs().max())
            sign, log_abs_det = torch.linalg.slogdet(matrix)
            if tuple(matrix.shape) != (128, 128) or not bool(torch.isfinite(matrix).all()):
                raise RuntimeError(f"{label} invalid matrix structure at layer {layer}")
            if residual > core.ORTHOGONALITY_THRESHOLD or float(sign) <= 0:
                raise RuntimeError(f"{label} invalid orthogonality/determinant at layer {layer}")
            digest = tensor_sha256(matrix)
            matrices[str(layer)] = matrix
            per_layer[str(layer)] = {
                "sha256": digest,
                "shape": [128, 128],
                "dtype": "torch.float32",
                "orthogonality_max_abs_rt_r_minus_i": residual,
                "determinant_sign": float(sign),
                "determinant_log_abs": float(log_abs_det),
            }
        out_dir = ROOT / f"candidate_artifacts/{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frozen = out_dir / "frozen_rotation_fp32.pt"
        manifest = out_dir / "deployment_manifest.json"
        if frozen.exists() or manifest.exists():
            raise RuntimeError(f"refusing to overwrite prior {label} export")
        frozen_payload = {
            "format_version": 1,
            "experiment": ROOT.name,
            "condition": label,
            "canonical": False,
            "research_candidate": True,
            "family": family,
            "update": update,
            "target_exposures": exposure,
            "layer_ids": tuple(core.GDN_LAYERS),
            "matrices": matrices,
            "matrix_dtype": "torch.float32",
            "matrix_shape": [128, 128],
            "matrix_meaning": "Noncanonical historical candidate Cayley Delta materialized from verified theta; not precomposed H@Delta.",
            "orientation": "q/k.float() @ H128 @ Delta; recovery applies Delta then H128 on key axis",
            "theta_checkpoint_sha256": theta_sha,
            "theta_payload_source_git_commit": payload.get("source_git_commit"),
            "canonical_source_sha256": payload.get("canonical_source_sha256"),
            "rotation_source_sha256": payload.get("rotation_source_sha256"),
            "export_host": {
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
        }
        atomic_torch(frozen, frozen_payload)
        frozen_sha = file_sha256(frozen)
        manifest_payload = {
            "condition": label,
            "status": "READY_NONCANONICAL_RESEARCH_CANDIDATE",
            "canonical": False,
            "loading_rule": "load frozen rotation directly; do not reconstruct R from theta on inference host",
            "frozen_rotation_file": frozen.name,
            "file_sha256": frozen_sha,
            "per_layer_sha256": {k: v["sha256"] for k, v in per_layer.items()},
            "expected_layer_ids": list(core.GDN_LAYERS),
            "expected_shape_per_layer": [128, 128],
            "expected_dtype": "torch.float32",
            "source_checkpoint_path": str(theta_path),
            "source_checkpoint_sha256": theta_sha,
            "source_checkpoint_family": family,
            "source_checkpoint_update": update,
            "source_checkpoint_exposure": exposure,
            "final_rotation_rule": "H128 @ stored candidate Delta in the canonical runtime orientation",
            "quantizer_mode": "symmetric INT8 C128; key/K axis -2; round-to-even; clamp [-127,127]",
            "per_layer_audit": per_layer,
            "AIME_EVALUATION": "NOT_STARTED",
        }
        atomic_json(manifest, manifest_payload)
        # Exact read-back through CPU serialization and per-tensor identity.
        loaded = torch.load(frozen, map_location="cpu", weights_only=False)
        if any(tensor_sha256(loaded["matrices"][str(layer)]) != per_layer[str(layer)]["sha256"] for layer in core.GDN_LAYERS):
            raise RuntimeError(f"{label} frozen tensor read-back mismatch")
        results.append(
            {
                "candidate": label,
                "status": "PASS",
                "canonical": False,
                "theta_checkpoint_sha256": theta_sha,
                "frozen_rotation_path": str(frozen),
                "frozen_rotation_sha256": frozen_sha,
                "deployment_manifest_path": str(manifest),
                "deployment_manifest_sha256": file_sha256(manifest),
                "layer_count": len(matrices),
                "maximum_orthogonality_residual": max(v["orthogonality_max_abs_rt_r_minus_i"] for v in per_layer.values()),
            }
        )
        del bank, matrices, loaded
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    atomic_json(
        ROOT / "analysis/noncanonical_candidate_export.json",
        {
            "status": "PASS",
            "labeling_rule": "noncanonical historical research candidate; never canonical deployment",
            "candidates": results,
            "runtime_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "canonical_source_path": str(CORE_PATH),
            "canonical_source_sha256": file_sha256(CORE_PATH),
            "no_training": True,
            "no_AIME": True,
        },
    )
    print(json.dumps({"status": "PASS", "gpu_hours": elapsed / 3600.0, "candidates": [x["candidate"] for x in results]}, sort_keys=True))


if __name__ == "__main__":
    main()
