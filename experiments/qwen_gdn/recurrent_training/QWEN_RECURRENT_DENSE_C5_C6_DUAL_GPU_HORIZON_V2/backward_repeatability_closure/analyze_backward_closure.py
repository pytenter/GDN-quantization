#!/usr/bin/env python3
"""Read-only numerical comparison plus append-only closure analysis outputs."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
RAW = ROOT / "raw"
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
EPSILON = 1e-12
METRICS = ("relative_l2", "max_abs", "normalized_max_abs", "cosine_error", "log_norm_ratio")


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def case(mode: str, variant: str, index: int) -> dict:
    path = ANALYSIS / f"{mode}_{variant}_{index:02d}.json"
    value = read(path)
    if value["stem"] != path.stem:
        raise RuntimeError(f"stem mismatch {path}")
    value["_json_sha256"] = digest(path)
    return value


def array(case_value: dict, kind: str, layer: int) -> np.ndarray:
    key = "gradient_artifact" if kind == "gradient" else "update_artifact"
    hash_key = "gradient_artifact_sha256" if kind == "gradient" else "update_artifact_sha256"
    archive = load_archive(case_value[key], case_value[hash_key])
    return archive[str(layer)]


@lru_cache(maxsize=64)
def load_archive(name: str, expected_sha256: str) -> dict[str, np.ndarray]:
    path = RAW / name
    if digest(path) != expected_sha256:
        raise RuntimeError(f"artifact hash mismatch {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {str(layer): archive[str(layer)].astype(np.float64).ravel() for layer in LAYERS}


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise RuntimeError("gradient/update shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("nonfinite vector")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    da = float(np.max(np.abs(a)))
    db = float(np.max(np.abs(b)))
    diff = a - b
    if na <= EPSILON and nb <= EPSILON:
        zero_case, cosine_error = "both_zero", None
    elif na <= EPSILON or nb <= EPSILON:
        zero_case, cosine_error = "one_zero", None
    else:
        zero_case = "neither_zero"
        cosine = float(np.dot(a, b) / (na * nb))
        cosine_error = max(0.0, 1.0 - min(1.0, max(-1.0, cosine)))
    return {
        "relative_l2": float(np.linalg.norm(diff)) / max(na, nb, EPSILON),
        "max_abs": float(np.max(np.abs(diff))),
        "normalized_max_abs": float(np.max(np.abs(diff))) / max(da, db, EPSILON),
        "cosine_error": cosine_error,
        "log_norm_ratio": abs(math.log((na + EPSILON) / (nb + EPSILON))),
        "zero_case": zero_case,
        "exact": bool(np.array_equal(a, b)),
    }


def pair_metrics(a: dict, b: dict, kind: str) -> dict:
    per_layer = {str(layer): metrics(array(a, kind, layer), array(b, kind, layer)) for layer in LAYERS}
    global_a = np.concatenate([array(a, kind, layer) for layer in LAYERS])
    global_b = np.concatenate([array(b, kind, layer) for layer in LAYERS])
    return {"pair": [a["stem"], b["stem"]], "per_layer": per_layer,
            "global": metrics(global_a, global_b)}


def forward_compare(a: dict, b: dict) -> dict:
    metadata = ("reference_case_sha256", "input_token_ids_sha256", "teacher_token_ids_sha256",
                "target_positions_sha256", "teacher_target_tensor_hashes")
    metadata_exact = all(a[key] == b[key] for key in metadata)
    if set(a["records"]) != set(b["records"]):
        return {"metadata_exact": metadata_exact, "trace_keys_exact": False, "forward_exact": False}
    fields = ("consumed_prev_state_sha256", "pre_qdq_state_sha256", "scale_sha256",
              "qcodes_sha256", "post_qdq_state_sha256")
    counts = {field: sum(a["records"][key].get(field) == b["records"][key].get(field)
                         for key in a["records"]) for field in fields}
    loss_exact = a["loss_components"] == b["loss_components"] and a["total_training_loss"] == b["total_training_loss"]
    return {"metadata_exact": metadata_exact, "trace_keys_exact": True,
            "sampled_points": len(a["records"]), "trace_match_counts": counts,
            "loss_exact": loss_exact,
            "forward_exact": metadata_exact and loss_exact and all(count == len(a["records"]) for count in counts.values()),
            "writeback_a": a["REAL_RECURRENT_WRITEBACK_GATE"], "writeback_b": b["REAL_RECURRENT_WRITEBACK_GATE"],
            "boundary_a": a["BPTT_BOUNDARY_GATE"], "boundary_b": b["BPTT_BOUNDARY_GATE"]}


def analyze_single5() -> None:
    runs = [case("single", "baseline", i) for i in range(1, 6)]
    pair_forward = [dict(pair=[a["stem"], b["stem"]], **forward_compare(a, b))
                    for a, b in itertools.combinations(runs, 2)]
    all_forward = all(value["forward_exact"] for value in pair_forward)
    if not all_forward:
        write_new(ANALYSIS / "single_gpu_repeatability_5run.json", {
            "SINGLE_GPU_FORWARD_REPEATABILITY": "FAIL", "pair_forward": pair_forward,
            "source_sha256": {row["stem"]: row["_json_sha256"] for row in runs}})
        raise RuntimeError("forward or loss repeatability FAIL; stop before backward interpretation")
    gradient_pairs = [pair_metrics(a, b, "gradient") for a, b in itertools.combinations(runs, 2)]
    gradient_exact = all(row["global"]["exact"] for row in gradient_pairs)
    differing_by_layer = {str(layer): sum(not row["per_layer"][str(layer)]["exact"] for row in gradient_pairs)
                          for layer in LAYERS}
    first = next((layer for layer in LAYERS if differing_by_layer[str(layer)]), None)
    prior = [layer for layer in LAYERS if first is not None and layer < first]
    write_new(ANALYSIS / "first_divergent_gradient.json", {
        "FIRST_DIVERGENT_GRADIENT_LAYER": first,
        "FIRST_DIVERGENT_GRADIENT_PARAMETER": f"bank.layer({first}).theta" if first is not None else None,
        "all_earlier_rotation_gradients_exact": all(differing_by_layer[str(layer)] == 0 for layer in prior),
        "different_pairs_by_layer": differing_by_layer,
        "divergence_pattern": "multiple_layers" if sum(bool(value) for value in differing_by_layer.values()) > 1 else "one_layer_or_none",
    })
    write_new(ANALYSIS / "single_gpu_repeatability_5run.json", {
        "num_fresh_process_runs": 5, "num_pairs": 10,
        "source_sha256": {row["stem"]: row["_json_sha256"] for row in runs},
        "SINGLE_GPU_FORWARD_REPEATABILITY": "PASS",
        "SINGLE_GPU_BACKWARD_BITWISE_REPEATABILITY": "PASS" if gradient_exact else "FAIL",
        "per_layer_gradient_stats": {row["stem"]: row["gradient_summary"] for row in runs},
        "pair_forward": pair_forward, "gradient_pair_metrics": gradient_pairs,
        "epsilon": EPSILON,
    })
    print(json.dumps({"forward": "PASS", "backward_bitwise": "PASS" if gradient_exact else "FAIL",
                      "first_divergent_layer": first, "differing_layers": sum(bool(value) for value in differing_by_layer.values())}, indent=2))


def envelope(pairs: list[dict], kind: str, source: list[dict]) -> dict:
    scopes = [*(str(layer) for layer in LAYERS), "global"]
    maxima = {}
    for scope in scopes:
        vectors = [row["global"] if scope == "global" else row["per_layer"][scope] for row in pairs]
        maxima[scope] = {
            metric: max((value[metric] for value in vectors if value[metric] is not None), default=None)
            for metric in METRICS
        }
        maxima[scope]["observed_zero_cases"] = sorted({value["zero_case"] for value in vectors})
    return {
        "kind": kind, "num_single_runs": 10, "num_pairs": 45, "epsilon": EPSILON,
        "protocol_sha256": digest(ROOT / "protocol_amendments/backward_numerical_equivalence_v1.json"),
        "source_sha256": {value["stem"]: value["_json_sha256"] for value in source},
        "aggregation": "STRICT_MAX_OF_ALL_45_PAIRS_NO_MULTIPLIER",
        "per_layer": {scope: maxima[scope] for scope in scopes if scope != "global"},
        "global": maxima["global"],
    }


def analyze_single10() -> None:
    runs = [case("single", "baseline", index) for index in range(6, 16)]
    if not all(row["optimizer_step_performed"] for row in runs):
        raise RuntimeError("single10 requires optimizer step for all runs")
    baseline = case("single", "baseline", 1)
    forward = [dict(pair=[baseline["stem"], row["stem"]], **forward_compare(baseline, row)) for row in runs]
    forward.extend(dict(pair=[a["stem"], b["stem"]], **forward_compare(a, b))
                   for a, b in itertools.combinations(runs, 2))
    if not all(row["forward_exact"] and row["writeback_a"] == row["writeback_b"] == "PASS"
               and row["boundary_a"] == row["boundary_b"] == "PASS" for row in forward):
        write_new(ANALYSIS / "single_gpu_repeatability_10run.json", {
            "SINGLE_GPU_FORWARD_REPEATABILITY": "FAIL", "pair_forward": forward})
        raise RuntimeError("forward exactness failed in single10")
    gradient_pairs = [pair_metrics(a, b, "gradient") for a, b in itertools.combinations(runs, 2)]
    update_pairs = [pair_metrics(a, b, "update") for a, b in itertools.combinations(runs, 2)]
    gradient_envelope = envelope(gradient_pairs, "raw_rotation_gradient", runs)
    update_envelope = envelope(update_pairs, "delta_theta_after_one_Adam_step", runs)
    write_new(ANALYSIS / "single_gpu_repeatability_10run.json", {
        "num_fresh_process_runs": 10, "num_pairs": 45,
        "source_sha256": {row["stem"]: row["_json_sha256"] for row in runs},
        "SINGLE_GPU_FORWARD_REPEATABILITY": "PASS",
        "SINGLE_GPU_BACKWARD_BITWISE_REPEATABILITY": "PASS" if all(row["global"]["exact"] for row in gradient_pairs) else "FAIL",
        "pair_forward": forward,
        "gradient_pair_metrics": gradient_pairs,
        "update_pair_metrics": update_pairs,
        "note": "Five earlier no-step diagnostic runs are separately preserved; these ten fresh runs are the step-capable envelope ensemble.",
    })
    gradient_path = ANALYSIS / "frozen_single_gpu_noise_envelope.json"
    update_path = ANALYSIS / "frozen_single_gpu_update_envelope.json"
    write_new(gradient_path, gradient_envelope)
    write_new(update_path, update_envelope)
    write_new(ANALYSIS / "frozen_single_gpu_envelope_manifest.json", {
        "gradient_envelope_sha256": digest(gradient_path),
        "update_envelope_sha256": digest(update_path),
        "protocol_sha256": digest(ROOT / "protocol_amendments/backward_numerical_equivalence_v1.json"),
        "frozen_before_new_dual_repeats": True,
    })
    print(json.dumps({"single10": "PASS", "gradient_envelope_sha256": digest(gradient_path),
                      "update_envelope_sha256": digest(update_path),
                      "global_gradient_relative_l2_max": gradient_envelope["global"]["relative_l2"],
                      "global_update_relative_l2_max": update_envelope["global"]["relative_l2"]}, indent=2))


def compare_to_envelope(pair: dict, frozen: dict, selected_metrics: tuple[str, ...] = METRICS) -> list[dict]:
    result = []
    for scope in [*(str(layer) for layer in LAYERS), "global"]:
        observed = pair["global"] if scope == "global" else pair["per_layer"][scope]
        allowed = frozen["global"] if scope == "global" else frozen["per_layer"][scope]
        zero_valid = observed["zero_case"] in allowed["observed_zero_cases"]
        for metric in selected_metrics:
            value = observed[metric]
            maximum = allowed[metric]
            within = zero_valid and ((value is None and maximum is None) or
                                     (value is not None and maximum is not None and value <= maximum))
            result.append({"pair_a": pair["pair"][0], "pair_b": pair["pair"][1],
                           "scope": scope, "metric": metric, "value": value,
                           "envelope": maximum, "zero_case": observed["zero_case"],
                           "within": within})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pair_a", "pair_b", "scope", "metric", "value",
                                                       "envelope", "zero_case", "within"))
        writer.writeheader()
        writer.writerows(rows)


def analyze_dual5() -> None:
    manifest = read(ANALYSIS / "frozen_single_gpu_envelope_manifest.json")
    gradient_path = ANALYSIS / "frozen_single_gpu_noise_envelope.json"
    update_path = ANALYSIS / "frozen_single_gpu_update_envelope.json"
    if digest(gradient_path) != manifest["gradient_envelope_sha256"] or digest(update_path) != manifest["update_envelope_sha256"]:
        raise RuntimeError("frozen single envelope hash mismatch")
    gradient_frozen = read(gradient_path)
    update_frozen = read(update_path)
    singles = [case("single", "baseline", index) for index in range(6, 16)]
    duals = [case("dual", "baseline", index) for index in range(11, 16)]
    if not all(row["optimizer_step_performed"] for row in duals):
        raise RuntimeError("dual5 update missing")
    forward = [dict(pair=[a["stem"], b["stem"]], **forward_compare(a, b))
               for a, b in itertools.product(singles, duals)]
    forward.extend(dict(pair=[a["stem"], b["stem"]], **forward_compare(a, b))
                   for a, b in itertools.combinations(duals, 2))
    forward_pass = all(row["forward_exact"] and row["writeback_a"] == row["writeback_b"] == "PASS"
                       and row["boundary_a"] == row["boundary_b"] == "PASS" for row in forward)
    if not forward_pass:
        write_new(ANALYSIS / "dual_gpu_repeatability_5run.json", {
            "DUAL_GPU_FORWARD_SEMANTICS": "FAIL", "pair_forward": forward,
            "frozen_envelope_manifest_sha256": digest(ANALYSIS / "frozen_single_gpu_envelope_manifest.json")})
        raise RuntimeError("dual forward/discrete gate failed")
    cross_gradient = [row for a, b in itertools.product(singles, duals)
                      for row in compare_to_envelope(pair_metrics(a, b, "gradient"), gradient_frozen)]
    dual_internal = [row for a, b in itertools.combinations(duals, 2)
                     for row in compare_to_envelope(pair_metrics(a, b, "gradient"), gradient_frozen)]
    cross_update = [row for a, b in itertools.product(singles, duals)
                    for row in compare_to_envelope(pair_metrics(a, b, "update"), update_frozen)]
    write_csv(ANALYSIS / "single_vs_dual_gradient_metrics.csv", cross_gradient)
    write_csv(ANALYSIS / "dual_vs_dual_gradient_metrics.csv", dual_internal)
    write_csv(ANALYSIS / "one_step_update_metrics.csv", cross_update)
    gates = {
        "DUAL_GPU_GRADIENT_WITHIN_SINGLE_NOISE_ENVELOPE": "PASS" if all(row["within"] for row in cross_gradient) else "FAIL",
        "DUAL_GPU_INTERNAL_REPEATABILITY": "PASS" if all(row["within"] for row in dual_internal) else "FAIL",
        "DUAL_GPU_ONE_STEP_UPDATE_WITHIN_SINGLE_NOISE_ENVELOPE": "PASS" if all(row["within"] for row in cross_update) else "FAIL",
    }
    write_new(ANALYSIS / "dual_gpu_repeatability_5run.json", {
        "DUAL_GPU_FORWARD_SEMANTICS": "PASS", "pair_forward": forward,
        **gates,
        "frozen_envelope_manifest_sha256": digest(ANALYSIS / "frozen_single_gpu_envelope_manifest.json"),
        "source_sha256": {row["stem"]: row["_json_sha256"] for row in duals},
        "cross_gradient_checks": len(cross_gradient),
        "cross_gradient_failures": [row for row in cross_gradient if not row["within"]],
        "dual_internal_checks": len(dual_internal),
        "dual_internal_failures": [row for row in dual_internal if not row["within"]],
        "cross_update_checks": len(cross_update),
        "cross_update_failures": [row for row in cross_update if not row["within"]],
    })
    print(json.dumps({**gates, "cross_gradient_failures": sum(not row["within"] for row in cross_gradient),
                      "dual_internal_failures": sum(not row["within"] for row in dual_internal),
                      "cross_update_failures": sum(not row["within"] for row in cross_update)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("single5", "single10", "dual5"), required=True)
    args = parser.parse_args()
    if args.phase == "single5":
        analyze_single5()
    elif args.phase == "single10":
        analyze_single10()
    elif args.phase == "dual5":
        analyze_dual5()


if __name__ == "__main__":
    main()
