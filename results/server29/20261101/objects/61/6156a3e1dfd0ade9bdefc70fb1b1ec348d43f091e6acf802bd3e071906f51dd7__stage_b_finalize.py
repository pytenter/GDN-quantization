#!/usr/bin/env python3
"""Finalize Stage B after its fail-closed full-run disk gate."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


TASK = "QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
DEPLOY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment")
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hadamard(n=128):
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def assignment_max(weight):
    """Exact square Hungarian assignment maximizing weight, O(n^3)."""
    weight = np.asarray(weight, dtype=np.float64)
    n = weight.shape[0]
    cost = weight.max() - weight
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
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
    out = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def vec_summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
        "max": float(x.max()),
    }


def matrix_stats(condition, layer, r, h):
    eye = np.eye(128)
    abs_entry = np.abs(r)
    centered = r - r.mean()
    variance = float(np.mean(centered ** 2))
    row = {
        "condition": condition,
        "layer": int(layer),
        "orthogonality_error_fro": float(np.linalg.norm(r.T @ r - eye)),
        "max_abs_entry": float(abs_entry.max()),
        "mean_abs_entry": float(abs_entry.mean()),
        "std_abs_entry": float(abs_entry.std()),
        "entry_kurtosis_pearson": float(np.mean(centered ** 4) / max(variance ** 2, 1e-300)),
        "coherence": float(math.sqrt(128) * abs_entry.max()),
        "row_max_abs": vec_summary(abs_entry.max(axis=1)),
        "column_max_abs": vec_summary(abs_entry.max(axis=0)),
    }
    if condition == "Hadamard":
        row.update({
            "identity_distance_h_transpose_r_fro": 0.0,
            "raw_distance_to_h_fro": 0.0,
            "signed_permutation_aligned_distance_to_h_fro": 0.0,
            "signed_permutation_objective": 128.0,
        })
    else:
        corr = h.T @ r
        permutation = assignment_max(np.abs(corr))
        signs = np.sign(corr[np.arange(128), permutation])
        signs[signs == 0] = 1.0
        aligned = r[:, permutation] * signs[None, :]
        row.update({
            "identity_distance_h_transpose_r_fro": float(np.linalg.norm(eye - corr)),
            "raw_distance_to_h_fro": float(np.linalg.norm(r - h)),
            "signed_permutation_aligned_distance_to_h_fro": float(np.linalg.norm(aligned - h)),
            "signed_permutation_objective": float(np.abs(corr[np.arange(128), permutation]).sum()),
            "signed_permutation": permutation.tolist(),
            "signed_permutation_signs": signs.astype(int).tolist(),
        })
    return row


def rotation_geometry():
    h = hadamard()
    banks = {}
    for condition in ("C5", "C6"):
        payload = torch.load(DEPLOY / condition / "frozen_rotation_fp32.pt", map_location="cpu", weights_only=False)
        banks[condition] = {int(layer): payload["matrices"][str(layer)].double().numpy() for layer in LAYERS}
    rows = []
    for layer in LAYERS:
        rows.append(matrix_stats("Hadamard", layer, h, h))
        rows.append(matrix_stats("C5", layer, h @ banks["C5"][layer], h))
        rows.append(matrix_stats("C6", layer, h @ banks["C6"][layer], h))
    dump(ROOT / "analysis/rotation_geometry.json", {
        "status": "COMPLETE_CPU_OFFLINE", "rows": rows,
        "signed_permutation_method": "exact O(n^3) Hungarian maximum-absolute-column-correlation",
    })
    flat_fields = [
        "condition", "layer", "orthogonality_error_fro", "max_abs_entry",
        "mean_abs_entry", "std_abs_entry", "entry_kurtosis_pearson", "coherence",
        "identity_distance_h_transpose_r_fro", "raw_distance_to_h_fro",
        "signed_permutation_aligned_distance_to_h_fro", "signed_permutation_objective",
        "row_max_abs_mean", "row_max_abs_median", "row_max_abs_p95", "row_max_abs_p99", "row_max_abs_max",
        "column_max_abs_mean", "column_max_abs_median", "column_max_abs_p95", "column_max_abs_p99", "column_max_abs_max",
    ]
    with (ROOT / "analysis/rotation_geometry_per_layer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in flat_fields}
            for prefix in ("row_max_abs", "column_max_abs"):
                for name, value in row[prefix].items():
                    flat[f"{prefix}_{name}"] = value
            w.writerow(flat)
    return rows


def write_stopped_tables():
    stopped = {
        "status": "NOT_RUN_FULL60_DISK_GATE_FAIL",
        "scientific_use": False,
        "reason": "Raw no-downsampling group-level table is infeasible under the frozen disk budget.",
    }
    dump(ROOT / "analysis/global_summary.json", stopped)
    for name in ("global_summary.csv", "layer_summary.csv", "sample_summary.csv", "paired_delta_summary.csv"):
        with (ROOT / "analysis" / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["status", "scientific_use", "reason"])
            w.writerow([stopped["status"], "FALSE", stopped["reason"]])
    dump(ROOT / "analysis/paired_delta_summary.json", stopped)
    dump(ROOT / "analysis/mechanism_decision.json", {
        **stopped,
        "allowed_decision_labels": [],
        "decision": "NO_SCIENTIFIC_DECISION_FROM_ENGINEERING_SMOKE",
        "NEXT_RECOMMENDED_STAGE": "REASSESS_LOCAL_METRICS",
    })
    (ROOT / "data/README.md").write_text(
        "# Full-60 scientific data not created\n\n"
        "`group_rows.parquet` and `tail_events.parquet` are intentionally absent. "
        "The no-downsampling full-60 disk gate failed before authorization. "
        "The one-token complete calibration and engineering-only tail events are under `smoke/`.\n",
        encoding="utf-8",
    )
    (ROOT / "figures/README.md").write_text(
        "# Scientific figures not created\n\n"
        "Figures 1–12 require the complete full-60 scientific dataset. The full run was not authorized after the disk gate failed.\n",
        encoding="utf-8",
    )


def report(smoke, projection):
    report = f"""# {TASK} — Final Report

## Outcome

`STAGE_B_RESULT = STOPPED_FULL_RUN_DISK_GATE_FAIL`

The strict engineering smoke passed every semantic and runtime gate.  The full-60 run was **not authorized** because the required no-downsampling per-group Parquet dataset is infeasible on the frozen `/data` budget.  No scientific mechanism ordering or Stage-B decision label is reported from smoke data.

## Hard gates

```text
V2_SOURCE_GATE = PASS
FORMAL_ARTIFACT_IDENTITY_GATE = PASS

COMMON_TOKEN_SEQUENCE_GATE = {smoke['COMMON_TOKEN_SEQUENCE_GATE']}
COMMON_FP_STATE_INPUT_GATE = {smoke['COMMON_FP_STATE_INPUT_GATE']}

STATE_RESET_TO_COMMON_FP_EACH_TOKEN = TRUE
PERSISTENT_QDQ_WRITEBACK = FALSE
NEXT_TOKEN_READS_FP_STATE = TRUE

FORMAL_ROTATION_PATH_GATE = {smoke['FORMAL_ROTATION_PATH_GATE']}
FORMAL_QUANTIZER_SHARED_PATH_GATE = {smoke['FORMAL_QUANTIZER_SHARED_PATH_GATE']}

STAGE_B_SMOKE = {smoke['STAGE_B_SMOKE']}
MEMORY_GATE = {smoke['MEMORY_GATE']}
FULL_RUN_DISK_GATE = {projection['FULL_RUN_DISK_GATE']}

EXPECTED_SAMPLES = 60
ACTUAL_SUCCESS = 0
FULL60_COMPLETENESS_GATE = NOT_RUN_DISK_GATE_FAIL

STRICT_COMMON_FP_STATE_GATE = PASS_SMOKE_ONLY
STAGE_B_RESULT = STOPPED_FULL_RUN_DISK_GATE_FAIL

RECURRENT_REPLAY_STARTED = FALSE
MULTI_HORIZON_STARTED = FALSE
PRE_DIVERGENCE_STARTED = FALSE
LAYER_INTERVENTION_STARTED = FALSE
C7_STARTED = FALSE
NEW_LOSS_STARTED = FALSE
NEW_ROTATION_TRAINING_STARTED = FALSE
NEW_AIME_GENERATION_STARTED = FALSE
```

## Engineering smoke

- Frozen units: `aime26_01_seed1`, `aime26_01_seed2`.
- Frozen prefix: 32 teacher-forced tokens per unit; smoke only, never used for a scientific conclusion.
- Peak allocated GPU memory: {smoke['peak_gpu_memory_mb']:.2f} MiB.
- Each condition executed the unchanged formal quantizer 64 times over all 24 GDN layers.
- Selected token inputs were checked on every GDN layer by actual `torch.equal`; the common tensor bytes were SHA256 hashed and the same digest recorded for Native/H/C5/C6.
- H/C5/C6 local branches were discarded after measurement. Only the Native FP state was returned to the next token.

## Decisive disk projection

- Real calibration: all {projection['calibration_rows']:,} group rows for one smoke token.
- Calibration file: {projection['calibration_parquet_bytes']:,} bytes ({projection['calibrated_bytes_per_row']:.4f} bytes/row under Zstandard Parquet).
- Frozen full trajectory: {projection['full_teacher_tokens']:,} tokens.
- Required full rows: {projection['projected_full60_rows']:,}.
- Projected artifact: {projection['PROJECTED_FULL60_ARTIFACT_GIB']:,.2f} GiB.
- Free space before smoke: {projection['data_free_gib_before_smoke']:.2f} GiB.
- Projected free space after full run: {projection['PROJECTED_FREE_AFTER_FULL60_GB']:,.2f} GB.

The gap is about three orders of magnitude beyond available storage.  Downsampling, dropping group rows, deleting old research artifacts, or silently replacing exact rows with sketches was forbidden, so the run stopped before any full-60 worker launched.

## Offline rotation geometry

The requested CPU/offline H/C5/C6 matrix audit was completed for all 24 layers, including orthogonality, coherence, entry statistics, distance to H, and exact Hungarian signed-permutation alignment.  These geometry-only results are preserved in `analysis/rotation_geometry.json` and `analysis/rotation_geometry_per_layer.csv`; they are not coupled to incomplete smoke data to make a mechanism claim.

## Scientific questions and figures

Q1–Q12 and Figures 1–12 are `NOT_EVALUATED_FULL60_DISK_GATE_FAIL`.  Reporting their ordering from the engineering smoke would violate the frozen protocol.  No Stage-B mechanism label is assigned.

`NEXT_RECOMMENDED_STAGE = REASSESS_LOCAL_METRICS`

No subsequent stage was started.
"""
    (ROOT / "reports/FINAL_REPORT.md").write_text(report, encoding="utf-8")


def hashes():
    out = ROOT / "hashes/SHA256SUMS"
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != out and "__pycache__" not in p.parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in paths), encoding="utf-8")


def main():
    smoke = load(ROOT / "smoke/smoke_summary.json")
    projection = load(ROOT / "smoke/disk_projection.json")
    if smoke["STAGE_B_SMOKE"] != "PASS":
        raise RuntimeError("smoke did not pass")
    if projection["FULL_RUN_DISK_GATE"] != "FAIL" or smoke["FULL60_AUTHORIZED"] != "NO":
        raise RuntimeError("this finalizer is only valid for the frozen disk-gate stop")
    rotation_geometry()
    write_stopped_tables()
    gate_status = {
        "V2_SOURCE_GATE": "PASS", "FORMAL_ARTIFACT_IDENTITY_GATE": "PASS",
        "COMMON_TOKEN_SEQUENCE_GATE": smoke["COMMON_TOKEN_SEQUENCE_GATE"],
        "COMMON_FP_STATE_INPUT_GATE": smoke["COMMON_FP_STATE_INPUT_GATE"],
        "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": True,
        "PERSISTENT_QDQ_WRITEBACK": False, "NEXT_TOKEN_READS_FP_STATE": True,
        "FORMAL_ROTATION_PATH_GATE": smoke["FORMAL_ROTATION_PATH_GATE"],
        "FORMAL_QUANTIZER_SHARED_PATH_GATE": smoke["FORMAL_QUANTIZER_SHARED_PATH_GATE"],
        "STAGE_B_SMOKE": smoke["STAGE_B_SMOKE"], "MEMORY_GATE": smoke["MEMORY_GATE"],
        "FULL_RUN_DISK_GATE": "FAIL", "FULL60_AUTHORIZED": "NO",
        "EXPECTED_SAMPLES": 60, "ACTUAL_SUCCESS": 0,
        "FULL60_COMPLETENESS_GATE": "NOT_RUN_DISK_GATE_FAIL",
        "STRICT_COMMON_FP_STATE_GATE": "PASS_SMOKE_ONLY",
        "STAGE_B_RESULT": "STOPPED_FULL_RUN_DISK_GATE_FAIL",
        "RECURRENT_REPLAY_STARTED": False, "MULTI_HORIZON_STARTED": False,
        "PRE_DIVERGENCE_STARTED": False, "LAYER_INTERVENTION_STARTED": False,
        "C7_STARTED": False, "NEW_LOSS_STARTED": False,
        "NEW_ROTATION_TRAINING_STARTED": False, "NEW_AIME_GENERATION_STARTED": False,
        "NEXT_RECOMMENDED_STAGE": "REASSESS_LOCAL_METRICS",
    }
    dump(ROOT / "analysis/gate_status.json", gate_status)
    report(smoke, projection)
    hashes()
    print(json.dumps(gate_status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
