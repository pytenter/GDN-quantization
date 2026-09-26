#!/usr/bin/env python3
"""Freeze the selected recurrent horizon and aggregate training gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


TASK = "LING_RECURRENT_DENSE_L6_L7_V1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_horizon(args: argparse.Namespace) -> None:
    probe = read_json(args.horizon)
    if probe.get("status") != "PASS" or probe.get("HORIZON_FEASIBILITY_GATE") != "PASS":
        raise RuntimeError("horizon gate did not pass")
    selected = probe.get("selected_gradient_horizon")
    if selected not in (512, 256, 128, 64, 32):
        raise RuntimeError(f"invalid selected horizon: {selected!r}")
    config = read_json(args.config)
    config.update(
        {
            "status": "FROZEN",
            "gradient_horizon": selected,
            "detach_interval": selected,
            "selection_reason": "RESOURCE_FEASIBILITY_ONLY",
            "horizon_feasibility_sha256": sha256(args.horizon),
        }
    )
    write_json(args.config, config)
    print(selected)


def verify_smoke(args: argparse.Namespace) -> None:
    smoke = read_json(args.smoke)
    required = (
        "TRAIN_QDQ_MATCH",
        "GRADIENT_GATE",
        "REAL_RECURRENT_WRITEBACK_GATE",
        "RECURRENT_STATE_PROVENANCE_GATE",
    )
    failed = [name for name in required if smoke.get(name) != "PASS"]
    if smoke.get("status") != "PASS" or failed:
        raise RuntimeError(f"smoke gates failed: {failed}")
    write_json(args.provenance, smoke)


def aggregate(args: argparse.Namespace) -> None:
    l6_summary = read_json(args.l6_summary)
    l7_summary = read_json(args.l7_summary)
    l6_materialization = read_json(args.l6_materialization)
    l7_materialization = read_json(args.l7_materialization)
    h_materialization = read_json(args.h_materialization)
    sanity = read_json(args.sanity)
    for label, payload in (
        ("L6 training", l6_summary),
        ("L7 training", l7_summary),
        ("L6 materialization", l6_materialization),
        ("L7 materialization", l7_materialization),
        ("H materialization", h_materialization),
        ("sanity", sanity),
    ):
        if payload.get("status") != "PASS":
            raise RuntimeError(f"{label} did not pass")

    def selected_orthogonality(manifest: dict) -> dict:
        rows = manifest["layers"]
        errors = sorted(float(row["orthogonality_error"]) for row in rows)
        worst = max(rows, key=lambda row: float(row["orthogonality_error"]))
        p95_index = min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)
        return {
            "status": "PASS" if errors[-1] <= 1.0e-4 else "FAIL",
            "source": "SELECTED_CHECKPOINT_MATERIALIZED_R_FINAL",
            "median": statistics.median(errors),
            "p95": errors[p95_index],
            "max": errors[-1],
            "worst_layer": worst["layer_id"],
            "per_layer": [
                {
                    "layer_id": row["layer_id"],
                    "max_abs_rt_r_minus_i": row["orthogonality_error"],
                }
                for row in rows
            ],
        }

    orthogonality = {
        "task": TASK,
        "status": "PASS",
        "conditions": {
            "L6_RECURRENT_DENSE_STATE": selected_orthogonality(l6_materialization),
            "L7_RECURRENT_DENSE_FUNCTIONAL": selected_orthogonality(l7_materialization),
        },
    }
    if any(row.get("status") != "PASS" for row in orthogonality["conditions"].values()):
        orthogonality["status"] = "FAIL"

    materialization = {
        "task": TASK,
        "status": "PASS",
        "conditions": {
            "L2_UNIFIED_H128": h_materialization,
            "L6_RECURRENT_DENSE_STATE": l6_materialization,
            "L7_RECURRENT_DENSE_FUNCTIONAL": l7_materialization,
        },
    }
    if orthogonality["status"] != "PASS":
        raise RuntimeError("orthogonality gate failed")
    write_json(args.orthogonality, orthogonality)
    write_json(args.final_materialization, materialization)
    write_json(
        args.complete,
        {
            "task": TASK,
            "status": "PASS",
            "L6_TRAIN_COMPLETE": "PASS",
            "L7_TRAIN_COMPLETE": "PASS",
            "L6_ORTHOGONALITY_GATE": "PASS",
            "L7_ORTHOGONALITY_GATE": "PASS",
            "L6_FINAL_R_MATERIALIZATION": "PASS",
            "L7_FINAL_R_MATERIALIZATION": "PASS",
            "SANITY_PANEL": "PASS",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-horizon")
    freeze.add_argument("--horizon", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.set_defaults(function=freeze_horizon)

    smoke = subparsers.add_parser("verify-smoke")
    smoke.add_argument("--smoke", type=Path, required=True)
    smoke.add_argument("--provenance", type=Path, required=True)
    smoke.set_defaults(function=verify_smoke)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--l6-summary", type=Path, required=True)
    aggregate_parser.add_argument("--l7-summary", type=Path, required=True)
    aggregate_parser.add_argument("--l6-materialization", type=Path, required=True)
    aggregate_parser.add_argument("--l7-materialization", type=Path, required=True)
    aggregate_parser.add_argument("--h-materialization", type=Path, required=True)
    aggregate_parser.add_argument("--sanity", type=Path, required=True)
    aggregate_parser.add_argument("--orthogonality", type=Path, required=True)
    aggregate_parser.add_argument("--final-materialization", type=Path, required=True)
    aggregate_parser.add_argument("--complete", type=Path, required=True)
    aggregate_parser.set_defaults(function=aggregate)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
