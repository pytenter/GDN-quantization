#!/usr/bin/env python3
"""Frozen single-endpoint functional envelope and strict dual comparison."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
RAW = ROOT / "raw"
CONFIGS = ROOT / "configs"
EPS = 1e-12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def metadata(stem: str) -> dict:
    path = ANALYSIS / f"{stem}_probe.json"
    value = read(path)
    if value["stem"] != stem or value["protocol_sha256"] != sha(CONFIGS / "frozen_trajectory_protocol.json"):
        raise RuntimeError(f"invalid probe metadata {stem}")
    trajectory_path = ANALYSIS / f"{stem}.json"
    if value["trajectory_sha256"] != sha(trajectory_path):
        raise RuntimeError(f"trajectory hash mismatch {stem}")
    trajectory = read(trajectory_path)
    if value["endpoint_theta_sha256"] != trajectory["endpoint_theta_sha256"]:
        raise RuntimeError(f"endpoint hash mismatch {stem}")
    return value


@lru_cache(maxsize=15)
def arrays(stem: str) -> dict[str, np.ndarray]:
    meta = metadata(stem)
    path = RAW / meta["raw_artifact"]
    if sha(path) != meta["raw_artifact_sha256"]:
        raise RuntimeError(f"raw artifact hash mismatch {stem}")
    with np.load(path, allow_pickle=False) as loaded:
        if len(loaded.files) != 968:
            raise RuntimeError(f"probe tensor count {stem}: {len(loaded.files)}")
        return {key: loaded[key] for key in loaded.files}


def exact_metadata(metas: list[dict]) -> dict:
    baseline = metas[0]
    fields = ("input_ids_sha256", "teacher_ids_sha256", "target_positions_sha256", "teacher_target_hashes")
    mismatches = [{"stem": meta["stem"], "field": field}
                  for meta in metas[1:] for field in fields if meta[field] != baseline[field]]
    tensor_keys = set(arrays(baseline["stem"]))
    for meta in metas[1:]:
        if set(arrays(meta["stem"])) != tensor_keys:
            mismatches.append({"stem": meta["stem"], "field": "tensor_keys"})
    return {"exact": not mismatches, "mismatches": mismatches, "tensor_count": len(tensor_keys)}


def metric(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    if a.shape != b.shape:
        raise RuntimeError("probe tensor shape mismatch")
    if np.issubdtype(a.dtype, np.integer):
        return {"qcodes_mismatch_count": int(np.count_nonzero(a != b))}
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("nonfinite probe tensor")
    diff = x-y
    return {"relative_l2": float(np.linalg.norm(diff)) / max(float(np.linalg.norm(x)), float(np.linalg.norm(y)), EPS),
            "max_abs": float(np.max(np.abs(diff)))}


def family(key: str) -> str:
    return key.split("_", 1)[0] if not key.startswith("post_qdq_") else "post_qdq"


def freeze_single() -> None:
    stems = [f"single_{index:02d}" for index in range(1, 11)]
    metas = [metadata(stem) for stem in stems]
    exact = exact_metadata(metas)
    if not exact["exact"]:
        write_new(ANALYSIS / "single_gpu_functional_probe_error.json", exact)
        raise RuntimeError("single probe input/teacher mismatch")
    maxima = {}
    for a, b in itertools.combinations(stems, 2):
        for key in arrays(a):
            for name, value in metric(arrays(a)[key], arrays(b)[key]).items():
                entry = maxima.setdefault((key, name), 0)
                maxima[(key, name)] = max(entry, value)
    path = ANALYSIS / "frozen_single_gpu_functional_envelope.json"
    write_new(path, {
        "protocol_sha256": sha(CONFIGS / "frozen_trajectory_protocol.json"),
        "aggregation": "STRICT_MAX_45_SINGLE_ENDPOINT_PAIRS_NO_MULTIPLIER",
        "single_probe_source_sha256": {stem: sha(ANALYSIS / f"{stem}_probe.json") for stem in stems},
        "exact_metadata": exact,
        "maxima": [{"tensor": key, "metric": name, "maximum": value}
                   for (key, name), value in sorted(maxima.items())],
    })
    write_new(ANALYSIS / "frozen_single_gpu_functional_envelope_hash.json", {
        "sha256": sha(path), "single_pairs": 45, "frozen_before_dual_probe_analysis": True,
    })
    print(json.dumps({"SINGLE_FUNCTIONAL_ENVELOPE": "FROZEN", "sha256": sha(path),
                      "metric_cells": len(maxima)}, indent=2), flush=True)


def compare_dual() -> None:
    freeze = read(ANALYSIS / "frozen_single_gpu_functional_envelope_hash.json")
    path = ANALYSIS / "frozen_single_gpu_functional_envelope.json"
    if sha(path) != freeze["sha256"]:
        raise RuntimeError("functional envelope hash mismatch")
    envelope = read(path)
    maxima = {(row["tensor"], row["metric"]): row["maximum"] for row in envelope["maxima"]}
    singles = [f"single_{index:02d}" for index in range(1, 11)]
    duals = [f"dual_{index:02d}" for index in range(1, 6)]
    metas = [metadata(stem) for stem in [*singles, *duals]]
    exact = exact_metadata(metas)
    if not exact["exact"]:
        write_new(ANALYSIS / "update8_functional_probe.json", {
            "FUNCTIONAL_FORWARD_METADATA_GATE": "FAIL", "exact_metadata": exact,
        })
        raise RuntimeError("dual probe input/teacher mismatch")
    failures = []
    counts = {}
    family_worst = {}
    comparisons = 0
    for a, b in itertools.product(singles, duals):
        for key in arrays(a):
            for name, value in metric(arrays(a)[key], arrays(b)[key]).items():
                comparisons += 1
                upper = maxima[(key, name)]
                group = family(key)
                counts[group] = counts.get(group, 0) + 1
                excess = value - upper
                if excess > 0:
                    failures.append({"single": a, "dual": b, "tensor": key,
                                     "family": group, "metric": name, "value": value,
                                     "envelope": upper, "excess": excess})
                if group not in family_worst or excess > family_worst[group]["excess"]:
                    family_worst[group] = {"single": a, "dual": b, "tensor": key,
                                           "metric": name, "value": value,
                                           "envelope": upper, "excess": excess}
    write_new(ANALYSIS / "update8_functional_probe.json", {
        "FUNCTIONAL_FORWARD_METADATA_GATE": "PASS",
        "UPDATE8_FUNCTIONAL_PROBE_GATE": "FAIL" if failures else "PASS",
        "interpretation": "fixed non-AIME teacher-forced endpoint probe; strict 45-single-pair max envelope",
        "single_pairs": 45, "single_dual_pairs": 50, "metric_comparisons": comparisons,
        "comparison_counts_by_family": counts, "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
        "worst_excess_by_family": family_worst,
        "single_functional_envelope_sha256": sha(path),
        "probe_source_sha256": {stem: sha(ANALYSIS / f"{stem}_probe.json") for stem in [*singles, *duals]},
        "exact_metadata": exact,
    })
    print(json.dumps({"UPDATE8_FUNCTIONAL_PROBE_GATE": "FAIL" if failures else "PASS",
                      "metric_comparisons": comparisons, "failures": len(failures)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("single", "dual"), required=True)
    args = parser.parse_args()
    freeze_single() if args.phase == "single" else compare_dual()
