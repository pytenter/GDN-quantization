#!/usr/bin/env python3
"""CPU-only preparation for QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2.

This script is deliberately incapable of loading the model or starting CUDA work.
It freezes identities, records the old-capture safety state, and computes only
offline rotation geometry from the deployment-closed FP32 matrices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
ROOT = REPO / "experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2"
OLD_ROOT = REPO / "experiments/qwen_gdn/QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1"
FORMAL_ROOT = REPO / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1"
DEPLOY_ROOT = REPO / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
SCORER = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/shared/scoring/aime26_scorer_v4.py")

EXPECTED = {
    "formal_config": "557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836",
    "hardware_assignment": "75b5df954f64f606f30143df3bf5329e238a6eabe8253bfc287141b0443c475a",
    "frozen_v4_scorer": "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b",
    "output_freeze_manifest": "064ce462e05e3f0eba0aa9718e446f7edbccc24da708bd4bc0a8ed1fa772345f",
    "artifact_hash_manifest": "f29ff354a112f0f4f339ead36169d9b5dcbb4f0699341e96c43ef50891f60277",
    "formal_runner": "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69",
    "C5_rotation": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6_rotation": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}

IDENTITY_PATHS = {
    "formal_config": FORMAL_ROOT / "configs/formal_config.json",
    "hardware_assignment": FORMAL_ROOT / "configs/hardware_assignment.json",
    "frozen_v4_scorer": SCORER,
    "output_freeze_manifest": FORMAL_ROOT / "analysis/output_freeze_manifest.json",
    "artifact_hash_manifest": FORMAL_ROOT / "hashes/artifact_sha256.txt",
    "formal_runner": FORMAL_ROOT / "scripts/run_qwen_c5_c6_formal.py",
    "C5_rotation": DEPLOY_ROOT / "C5/frozen_rotation_fp32.pt",
    "C6_rotation": DEPLOY_ROOT / "C6/frozen_rotation_fp32.pt",
}

GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def command(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return result.stdout.strip()


def hadamard(n: int = 128) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(n)


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    variance = np.mean(centered * centered)
    if variance == 0:
        return 0.0
    return float(np.mean(centered ** 4) / (variance ** 2) - 3.0)


def signed_permutation_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, list[int], list[int]]:
    correlation = reference.T @ candidate
    # Deterministic O(n^3) Hungarian assignment (min-cost form), kept local so
    # the frozen bitdecode environment needs no extra package installation.
    cost = -np.abs(correlation)
    n, m = cost.shape
    if n > m:
        raise ValueError("Hungarian implementation requires rows <= columns")
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    ref_columns = np.arange(n, dtype=np.int64)
    candidate_columns = assignment
    order = np.argsort(ref_columns)
    candidate_columns = candidate_columns[order]
    signs = np.sign(correlation[ref_columns[order], candidate_columns])
    signs[signs == 0] = 1.0
    aligned = candidate[:, candidate_columns] * signs.reshape(1, -1)
    return float(np.linalg.norm(reference - aligned, ord="fro")), candidate_columns.tolist(), signs.astype(int).tolist()


def load_rotations(condition: str, H: np.ndarray) -> dict[int, np.ndarray]:
    if condition == "H":
        return {layer: H.copy() for layer in GDN_LAYERS}
    payload = torch.load(IDENTITY_PATHS[f"{condition}_rotation"], map_location="cpu", weights_only=False)
    if tuple(payload["layer_ids"]) != GDN_LAYERS:
        raise RuntimeError(f"{condition} layer mapping mismatch")
    # Deployment payloads are Delta. Formal runtime is value.float() @ H @ Delta.
    return {layer: H @ payload["matrices"][str(layer)].double().numpy() for layer in GDN_LAYERS}


def rotation_geometry() -> tuple[list[dict], dict]:
    H = hadamard()
    rows: list[dict] = []
    full: dict[str, dict] = {}
    for condition in ("H", "C5", "C6"):
        full[condition] = {}
        for layer, matrix in load_rotations(condition, H).items():
            gram_error = matrix.T @ matrix - np.eye(128)
            raw_distance = float(np.linalg.norm(matrix - H, ord="fro"))
            aligned_distance, permutation, signs = signed_permutation_distance(H, matrix)
            relative = H.T @ matrix
            row = {
                "condition": condition,
                "layer": layer,
                "orthogonality_error_fro": float(np.linalg.norm(gram_error, ord="fro")),
                "orthogonality_error_max_abs": float(np.max(np.abs(gram_error))),
                "max_abs_entry": float(np.max(np.abs(matrix))),
                "mean_abs_entry": float(np.mean(np.abs(matrix))),
                "entry_std": float(np.std(matrix)),
                "entry_excess_kurtosis": excess_kurtosis(matrix),
                "coherence": float(math.sqrt(128) * np.max(np.abs(matrix))),
                "raw_distance_to_H_fro": raw_distance,
                "signed_perm_aligned_distance_to_H_fro": aligned_distance,
                "relative_to_H_identity_distance_fro": float(np.linalg.norm(relative - np.eye(128), ord="fro")),
            }
            for prefix, values in (("row_max_abs", np.max(np.abs(matrix), axis=1)), ("column_max_abs", np.max(np.abs(matrix), axis=0))):
                for key, value in describe(values).items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
            full[condition][str(layer)] = {**row, "signed_permutation": permutation, "signed_permutation_signs": signs}
    return rows, full


def old_capture_snapshot() -> dict:
    pgrep = command(["pgrep", "-af", "mechanism_capture.py"], check=False)
    gpu = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], check=False)
    active_lines = [line for line in pgrep.splitlines() if "QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1" in line]
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "old_v1_process_count": len(active_lines),
        "old_v1_processes": active_lines,
        "gpu_compute_processes": gpu.splitlines() if gpu else [],
        "OLD_CAPTURE_ACTIVE": "YES" if active_lines else "NO",
        "NEW_V2_GPU_EXECUTION_STARTED": "NO",
    }


def main() -> None:
    for relative in (
        "config", "provenance", "parity", "strict_fp_fanout", "recurrent", "pre_divergence",
        "layer", "analysis", "figures", "logs", "hashes", "reports", "scripts",
    ):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

    observed = {}
    for key, path in IDENTITY_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "expected": EXPECTED[key]}
        if observed[key]["sha256"] != EXPECTED[key]:
            raise RuntimeError(f"frozen identity mismatch: {key}")

    branch = command(["git", "symbolic-ref", "--short", "HEAD"])
    head = command(["git", "rev-parse", "HEAD"])
    if branch != "exp/hadamard-init-dense-orthogonal-oracle-v1-qwen" or head != "fa7918d82c144a30ba3459114abb53e70df73d98":
        raise RuntimeError(f"formal source identity mismatch: {branch} {head}")

    snapshot = old_capture_snapshot()
    identity = {
        "gate": "PASS",
        "formal_branch": branch,
        "formal_head": head,
        "training_source_branch": "exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1",
        "training_source_commit": "fe7105fbc44b49db7a5bab1bf693c25e0f9e63fa",
        "artifacts": observed,
    }
    atomic_json(ROOT / "provenance/formal_artifact_identity.json", identity)
    atomic_json(ROOT / "provenance/old_capture_safety_snapshot.json", snapshot)

    source_audit = {
        "gate": "PASS_STATIC",
        "formal_runner": str(IDENTITY_PATHS["formal_runner"]),
        "formal_runner_sha256": EXPECTED["formal_runner"],
        "rotation_path": "QwenFrozenRotationPatch._transform: value.float().matmul(H).matmul(delta)",
        "quantizer_path": "run_qwen_c5_c6_formal.quantize_c128 called directly; no V2 reimplementation is authorized",
        "writeback_path": "formal quantize_c128 state.copy_(dequant.to(state.dtype))",
        "prefill_qdq": False,
        "decode_order": ["model recurrent update", "formal quantize_c128", "next-token selection/read"],
        "instrumentation_plan": {
            "rotation": "call super()._transform and observe dispatched matmul outputs without substituting arithmetic",
            "quantizer": "call the unchanged formal quantize_c128 and capture its live locals with a scoped Python trace",
            "parity": "untouched and capture-enabled paths on one fixed non-AIME deterministic probe",
        },
        "old_v1_controlled_capture": "DESCRIPTIVE_ONLY",
        "old_v1_scientific_identity": "CONDITION_SPECIFIC_UNQUANTIZED_TRAJECTORY_ANALYSIS",
        "old_v1_formal_path_match": "PARTIAL_MISMATCH",
        "old_v1_mismatch_reasons": ["matmul association", "dtype path", "duplicated quantizer implementation"],
    }
    atomic_json(ROOT / "provenance/formal_path_source_audit.json", source_audit)

    config = {
        "task": "QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "DIAGNOSTIC_ONLY",
        "conditions": ["Native", "Hadamard", "C5", "C6"],
        "horizons": [1, 4, 8, 16, 32, 64],
        "token_source": "COMMON_FP_TEACHER_TRAJECTORY",
        "stage_order": ["A_FORMAL_REPLAY_PARITY", "B_STRICT_COMMON_FP_STATE_FANOUT", "C_FORMAL_EQUIVALENT_RECURRENT_REPLAY", "D_MULTI_HORIZON_PROPAGATION", "E_FROZEN_OUTPUT_PRE_DIVERGENCE", "F_LAYER_ATTRIBUTION", "G_TARGETED_LAYER_INTERVENTION"],
        "stage_a_hard_gate": "QCODE_PARITY=PASS_EXACT and WRITEBACK_PARITY=PASS before B/C/D",
        "no_training": True,
        "c7": "NOT_STARTED",
        "new_loss": "NOT_STARTED",
        "new_rotation_training": "NOT_STARTED",
        "new_checkpoint_sweep": "NOT_STARTED",
        "new_aime_generation": "NOT_STARTED",
        "strict_blind_scoring_sequence": "DEVIATED_BY_USER_REQUEST",
        "strict_score_once_discipline": "DEVIATED",
        "old_v1_controlled_capture": "DESCRIPTIVE_ONLY",
    }
    atomic_json(ROOT / "config/experiment_config.json", config)

    rows, full = rotation_geometry()
    columns = list(rows[0])
    with (ROOT / "analysis/rotation_geometry_per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "gate": "PASS",
        "operator_semantics": "H for Hadamard; H @ deployment Delta for C5/C6, matching formal association only as an offline matrix identity",
        "conditions": full,
        "condition_summary": {
            condition: {
                key: describe(np.array([row[key] for row in rows if row["condition"] == condition]))
                for key in ("coherence", "raw_distance_to_H_fro", "signed_perm_aligned_distance_to_H_fro", "orthogonality_error_fro")
            }
            for condition in ("H", "C5", "C6")
        },
    }
    atomic_json(ROOT / "analysis/rotation_geometry.json", summary)

    parity_pending = {
        "FORMAL_REPLAY_PARITY_CLOSURE": "NOT_RUN_OLD_CAPTURE_ACTIVE" if snapshot["OLD_CAPTURE_ACTIVE"] == "YES" else "NOT_RUN",
        "MECHANISM_REPLAY_AUTHORIZED": "NO",
        "ROTATION_INTERMEDIATE_PARITY": "NOT_RUN",
        "ROTATED_TENSOR_PARITY": "NOT_RUN",
        "SCALE_PARITY": "NOT_RUN",
        "QCODE_PARITY": "NOT_RUN",
        "DEQUANT_PARITY": "NOT_RUN",
        "WRITEBACK_PARITY": "NOT_RUN",
        "NEXT_TOKEN_STATE_PARITY": "NOT_RUN",
        "SELECTED_OUTPUT_PARITY": "NOT_RUN",
        "LOGIT_PARITY": "NOT_RUN",
    }
    atomic_json(ROOT / "parity/formal_replay_parity.json", parity_pending)

    report = f"""# QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2 — preparation status

- `FORMAL_ARTIFACT_IDENTITY_GATE = PASS`
- `OLD_CAPTURE_ACTIVE = {snapshot['OLD_CAPTURE_ACTIVE']}`
- `NEW_V2_GPU_EXECUTION_STARTED = NO`
- `FORMAL_REPLAY_PARITY_CLOSURE = {parity_pending['FORMAL_REPLAY_PARITY_CLOSURE']}`
- `MECHANISM_REPLAY_AUTHORIZED = NO`

The independent V2 directory, frozen configuration, formal-source audit, and CPU-only rotation geometry are prepared. No model was loaded and no CUDA workload was started.

The old V1 controlled capture remains `DESCRIPTIVE_ONLY`: it follows condition-specific unquantized recurrent trajectories rather than a strict per-token common-FP-state fan-out. Its formal path match is `PARTIAL_MISMATCH` because of matmul association, dtype path, and a duplicated quantizer implementation.

Stage A must use the unchanged formal rotation/quantizer/writeback path and pass exact qcode and writeback parity before any mechanism replay is authorized.
"""
    (ROOT / "reports/PREPARATION_STATUS.md").write_text(report, encoding="utf-8")

    hash_lines = []
    # Runtime logs are append-only evidence and are intentionally excluded from
    # this preparation snapshot; the final task manifest will freeze them once.
    for path in sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p != ROOT / "hashes/SHA256SUMS" and "logs" not in p.relative_to(ROOT).parts
    ):
        hash_lines.append(f"{sha256(path)}  {path.relative_to(ROOT)}")
    (ROOT / "hashes/SHA256SUMS").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": "PASS", **snapshot, "root": str(ROOT), "geometry_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
