#!/usr/bin/env python3
"""Strict, preregistered single-envelope analysis for eight-update trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
RAW = ROOT / "raw"
CONFIGS = ROOT / "configs"
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
EPS = 1e-12
VECTOR_METRICS = ("relative_l2", "max_abs", "cosine_error")
GRADIENT_METRICS = (*VECTOR_METRICS, "gradient_norm_absolute_difference")
ROTATION_METRICS = (*VECTOR_METRICS, "orthogonality_error_absolute_difference")


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def new_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case(mode: str, index: int) -> dict:
    path = ANALYSIS / f"{mode}_{index:02d}.json"
    value = read(path)
    if value["stem"] != path.stem or len(value["updates"]) != 8:
        raise RuntimeError(f"incomplete trajectory {path}")
    if value["protocol_sha256"] != digest(CONFIGS / "frozen_trajectory_protocol.json"):
        raise RuntimeError(f"protocol hash mismatch {path}")
    value["_json_sha256"] = digest(path)
    return value


@lru_cache(maxsize=512)
def archive(name: str, expected_sha256: str) -> dict[str, np.ndarray]:
    path = RAW / name
    if digest(path) != expected_sha256:
        raise RuntimeError(f"artifact hash mismatch {path}")
    with np.load(path, allow_pickle=False) as data:
        return {str(layer): data[str(layer)].astype(np.float64).ravel() for layer in LAYERS}


def vectors(row: dict, kind: str) -> dict[str, np.ndarray]:
    key = "gradient_artifact" if kind == "gradient" else "rotation_artifact"
    hash_key = "gradient_sha256" if kind == "gradient" else "rotation_sha256"
    return archive(row[key], row[hash_key])


def vector_metric(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("vector shape/finite mismatch")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS and nb <= EPS:
        cosine_error = None
        zero_case = "both_zero"
    elif na <= EPS or nb <= EPS:
        cosine_error = None
        zero_case = "one_zero"
    else:
        cosine = float(np.dot(a, b) / (na * nb))
        cosine_error = max(0.0, 1.0 - min(1.0, max(-1.0, cosine)))
        zero_case = "neither_zero"
    return {"relative_l2": float(np.linalg.norm(a-b)) / max(na, nb, EPS),
            "max_abs": float(np.max(np.abs(a-b))),
            "cosine_error": cosine_error,
            "norm_absolute_difference": abs(na-nb), "zero_case": zero_case}


def scalar_metric(a: float, b: float) -> dict:
    return {"absolute_difference": abs(a-b),
            "relative_difference": abs(a-b) / max(abs(a), abs(b), EPS)}


def loss_scopes(row: dict) -> dict[str, float]:
    result = {key: float(row[key]) for key in ("loss", "state_loss", "functional_loss")}
    for index, target in enumerate(row["loss_components"]):
        for key in ("state", "functional", "c5", "c6"):
            result[f"target{index}:{key}"] = float(target[key])
    return result


def pair_update_metrics(a: dict, b: dict, update: int) -> list[dict]:
    ra, rb = a["updates"][update-1], b["updates"][update-1]
    if ra["document_id"] != rb["document_id"]:
        raise RuntimeError("schedule mismatch")
    pair = (a["stem"], b["stem"])
    output = []
    la, lb = loss_scopes(ra), loss_scopes(rb)
    for scope in la:
        for metric, value in scalar_metric(la[scope], lb[scope]).items():
            output.append({"run_a": pair[0], "run_b": pair[1], "update": update,
                           "family": "loss", "scope": scope, "metric": metric,
                           "value": value, "zero_case": None})
    for kind in ("gradient", "rotation"):
        va, vb = vectors(ra, kind), vectors(rb, kind)
        scopes = [*(str(layer) for layer in LAYERS), "global"] if kind == "gradient" else [str(layer) for layer in LAYERS]
        for scope in scopes:
            vector_a = np.concatenate([va[str(layer)] for layer in LAYERS]) if scope == "global" else va[scope]
            vector_b = np.concatenate([vb[str(layer)] for layer in LAYERS]) if scope == "global" else vb[scope]
            values = vector_metric(vector_a, vector_b)
            if kind == "gradient":
                values["gradient_norm_absolute_difference"] = values.pop("norm_absolute_difference")
                names = GRADIENT_METRICS
            else:
                values["orthogonality_error_absolute_difference"] = abs(
                    ra["rotation_per_layer"][scope]["orthogonality_error_max_abs"] -
                    rb["rotation_per_layer"][scope]["orthogonality_error_max_abs"])
                names = ROTATION_METRICS
            for metric in names:
                output.append({"run_a": pair[0], "run_b": pair[1], "update": update,
                               "family": kind, "scope": scope, "metric": metric,
                               "value": values[metric], "zero_case": values["zero_case"]})
    if update == 8:
        va, vb = vectors(ra, "rotation"), vectors(rb, "rotation")
        values = vector_metric(np.concatenate([va[str(layer)] for layer in LAYERS]),
                               np.concatenate([vb[str(layer)] for layer in LAYERS]))
        for metric in VECTOR_METRICS:
            output.append({"run_a": pair[0], "run_b": pair[1], "update": 8,
                           "family": "endpoint_global_rotation", "scope": "global",
                           "metric": metric, "value": values[metric],
                           "zero_case": values["zero_case"]})
    return output


def forward_identity(a: dict, b: dict, update: int, require_student_exact: bool) -> dict:
    ra, rb = a["updates"][update-1], b["updates"][update-1]
    pa, pb = RAW / ra["forward_artifact"], RAW / rb["forward_artifact"]
    if digest(pa) != ra["forward_sha256"] or digest(pb) != rb["forward_sha256"]:
        raise RuntimeError("forward artifact hash mismatch")
    x, y = read(pa), read(pb)
    metadata_fields = ("document_id", "input_token_ids_sha256", "teacher_token_ids_sha256",
                       "target_positions_sha256", "teacher_target_tensor_hashes")
    metadata_exact = all(x[field] == y[field] for field in metadata_fields)
    gates = all(x[key] == y[key] == "PASS" for key in ("REAL_RECURRENT_WRITEBACK_GATE", "BPTT_BOUNDARY_GATE"))
    if set(x["trace"]) != set(y["trace"]):
        sampled = 0
        student_exact = False
    else:
        sampled = len(x["trace"])
        trace_fields = ("consumed_prev_state_sha256", "pre_qdq_state_sha256", "scale_sha256",
                        "qcodes_sha256", "post_qdq_state_sha256")
        student_exact = all(x["trace"][key].get(field) == y["trace"][key].get(field)
                            for key in x["trace"] for field in trace_fields)
    losses_exact = x["loss_components"] == y["loss_components"]
    return {"update": update, "pair": [a["stem"], b["stem"]],
            "metadata_exact": metadata_exact, "recurrent_gates": gates,
            "sampled_points": sampled,
            "student_state_QDQ_exact": student_exact if require_student_exact else "NOT_REQUIRED_AFTER_PARAMETER_DRIFT",
            "loss_components_exact": losses_exact if require_student_exact else "NOT_REQUIRED_AFTER_PARAMETER_DRIFT",
            "forward_gate": metadata_exact and gates and sampled == 240 and
                            (not require_student_exact or (student_exact and losses_exact))}


def all_forward_checks(runs: list[dict]) -> list[dict]:
    reference = runs[0]
    return [forward_identity(reference, run, update, update == 1)
            for run in runs[1:] for update in range(1, 9)]


def freeze_envelope(rows: list[dict], sources: list[dict]) -> dict:
    by_key = {}
    for row in rows:
        key = (row["update"], row["family"], row["scope"], row["metric"])
        item = by_key.setdefault(key, {"values": [], "zero_cases": set()})
        if row["value"] is not None:
            item["values"].append(row["value"])
        if row["zero_case"] is not None:
            item["zero_cases"].add(row["zero_case"])
    maxima = []
    for key, item in sorted(by_key.items()):
        maxima.append({"update": key[0], "family": key[1], "scope": key[2],
                       "metric": key[3], "maximum": max(item["values"]) if item["values"] else None,
                       "allowed_zero_cases": sorted(item["zero_cases"])})
    return {
        "protocol_sha256": digest(CONFIGS / "frozen_trajectory_protocol.json"),
        "num_single_runs": 10, "num_pairs": 45,
        "aggregation": "MAX_OBSERVED_NO_MULTIPLIER",
        "source_sha256": {case["stem"]: case["_json_sha256"] for case in sources},
        "maxima": maxima,
    }


def analyze_single() -> None:
    runs = [case("single", index) for index in range(1, 11)]
    checks = all_forward_checks(runs)
    if not all(check["forward_gate"] for check in checks):
        new_json(ANALYSIS / "single_gpu_10run_summary.json", {
            "SINGLE_GPU_FORWARD_GATE": "FAIL", "forward_checks": checks})
        raise RuntimeError("single forward/teacher/writeback gate failed")
    metrics = [row for a, b in itertools.combinations(runs, 2)
               for update in range(1, 9) for row in pair_update_metrics(a, b, update)]
    new_json(ANALYSIS / "single_gpu_10run_summary.json", {
        "SINGLE_GPU_FORWARD_GATE": "PASS", "forward_checks": checks,
        "runs": {run["stem"]: {"sha256": run["_json_sha256"],
                               "loss_trajectory": [value["loss"] for value in run["updates"]],
                               "gradient_norm_trajectory": [value["gradient_norm"] for value in run["updates"]]}
                 for run in runs},
        "single_pairs": 45, "updates": 8, "metric_rows": len(metrics),
    })
    path = ANALYSIS / "frozen_single_gpu_trajectory_envelope.json"
    new_json(path, freeze_envelope(metrics, runs))
    new_json(ANALYSIS / "frozen_single_gpu_trajectory_envelope_hash.json", {
        "sha256": digest(path), "protocol_sha256": digest(CONFIGS / "frozen_trajectory_protocol.json"),
        "frozen_before_dual_trajectory_runs": True})
    print(json.dumps({"SINGLE_GPU_TRAJECTORY_ENVELOPE": "FROZEN",
                      "sha256": digest(path), "metric_rows": len(metrics)}, indent=2))


def analyze_dual() -> None:
    freeze = read(ANALYSIS / "frozen_single_gpu_trajectory_envelope_hash.json")
    envelope_path = ANALYSIS / "frozen_single_gpu_trajectory_envelope.json"
    if digest(envelope_path) != freeze["sha256"]:
        raise RuntimeError("frozen envelope hash mismatch")
    envelope = read(envelope_path)
    maxima = {(row["update"], row["family"], row["scope"], row["metric"]): row for row in envelope["maxima"]}
    singles = [case("single", index) for index in range(1, 11)]
    duals = [case("dual", index) for index in range(1, 6)]
    checks = [forward_identity(single, dual, update, update == 1)
              for single, dual in itertools.product(singles, duals) for update in range(1, 9)]
    checks.extend(forward_identity(duals[0], dual, update, update == 1)
                  for dual in duals[1:] for update in range(1, 9))
    forward_pass = all(check["forward_gate"] for check in checks)
    if not forward_pass:
        new_json(ANALYSIS / "dual_gpu_5run_summary.json", {
            "DUAL_GPU_FORWARD_GATE": "FAIL", "forward_checks": checks})
        raise RuntimeError("dual forward exact gate failed")
    comparisons = []
    for single, dual in itertools.product(singles, duals):
        for update in range(1, 9):
            for row in pair_update_metrics(single, dual, update):
                key = (row["update"], row["family"], row["scope"], row["metric"])
                allowed = maxima[key]
                zero_ok = row["zero_case"] is None or row["zero_case"] in allowed["allowed_zero_cases"]
                within = zero_ok and ((row["value"] is None and allowed["maximum"] is None) or
                                      (row["value"] is not None and allowed["maximum"] is not None and
                                       row["value"] <= allowed["maximum"]))
                comparisons.append({**row, "envelope": allowed["maximum"], "within": within})
    new_csv(ANALYSIS / "trajectory_cross_comparison.csv", comparisons,
            ("run_a", "run_b", "update", "family", "scope", "metric", "value", "zero_case", "envelope", "within"))
    update_rows = []
    layer_rows = []
    for run in [*singles, *duals]:
        for row in run["updates"]:
            update_rows.append({"run": run["stem"], "mode": run["mode"], "update": row["update"],
                                "document_id": row["document_id"], "loss": row["loss"],
                                "state_loss": row["state_loss"], "functional_loss": row["functional_loss"],
                                "gradient_norm": row["gradient_norm"]})
            for layer in LAYERS:
                r = row["rotation_per_layer"][str(layer)]
                g = row["gradient_per_layer"][str(layer)]
                layer_rows.append({"run": run["stem"], "mode": run["mode"], "update": row["update"],
                                   "layer": layer, "gradient_norm": g["frobenius_norm"],
                                   "gradient_max_abs": g["max_abs"],
                                   "delta_R_frobenius_norm": r["delta_R_frobenius_norm"],
                                   "delta_R_max_abs": r["delta_R_max_abs"],
                                   "rotation_cosine_to_H": r["cosine_similarity_to_initial_H"],
                                   "orthogonality_error_max_abs": r["orthogonality_error_max_abs"]})
    new_csv(ANALYSIS / "update_level_metrics.csv", update_rows,
            ("run", "mode", "update", "document_id", "loss", "state_loss", "functional_loss", "gradient_norm"))
    new_csv(ANALYSIS / "layer_level_metrics.csv", layer_rows,
            ("run", "mode", "update", "layer", "gradient_norm", "gradient_max_abs",
             "delta_R_frobenius_norm", "delta_R_max_abs", "rotation_cosine_to_H", "orthogonality_error_max_abs"))
    failures = [row for row in comparisons if not row["within"]]
    first = min(failures, key=lambda row: (row["update"], row["run_b"], row["run_a"], row["family"], row["scope"], row["metric"])) if failures else None
    by_update = {str(update): sum(row["update"] == update for row in failures) for update in range(1, 9)}
    new_json(ANALYSIS / "dual_gpu_5run_summary.json", {
        "DUAL_GPU_FORWARD_GATE": "PASS", "DUAL_GPU_TRAJECTORY_GATE": "FAIL" if failures else "PASS",
        "frozen_envelope_sha256": digest(envelope_path),
        "source_sha256": {run["stem"]: run["_json_sha256"] for run in duals},
        "forward_checks": checks,
        "comparison_rows": len(comparisons), "failure_count": len(failures),
        "failures_by_update": by_update,
        "FIRST_OUT_OF_ENVELOPE_UPDATE": first["update"] if first else None,
        "FIRST_OUT_OF_ENVELOPE_LAYER": first["scope"] if first and first["scope"] != "global" else None,
        "FIRST_OUT_OF_ENVELOPE_METRIC_TYPE": first["family"] if first else None,
        "FIRST_OUT_OF_ENVELOPE_METRIC": first["metric"] if first else None,
        "first_failure": first,
        "all_failures": failures,
    })
    print(json.dumps({"DUAL_GPU_TRAJECTORY_GATE": "FAIL" if failures else "PASS",
                      "comparison_rows": len(comparisons), "failures": len(failures),
                      "first_failure": first}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("single", "dual"), required=True)
    args = parser.parse_args()
    if args.phase == "single":
        analyze_single()
    else:
        analyze_dual()


if __name__ == "__main__":
    main()
