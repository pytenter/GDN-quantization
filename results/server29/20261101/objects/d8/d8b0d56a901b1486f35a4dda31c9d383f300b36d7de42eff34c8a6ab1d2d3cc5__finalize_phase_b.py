#!/usr/bin/env python3
"""Finalize the shared-objective Hadamard comparison and deployment inventory."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
ANALYSIS = ROOT / "analysis"


def dominates(a: dict, b: dict) -> bool:
    return (
        a["state_mse"] <= b["state_mse"]
        and a["functional_mse"] <= b["functional_mse"]
        and (
            a["state_mse"] < b["state_mse"]
            or a["functional_mse"] < b["functional_mse"]
        )
    )


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    had = json.loads((ANALYSIS / "hadamard_shared_objective.json").read_text())
    protocol = json.loads((ANALYSIS / "shared_validation_protocol.json").read_text())
    if had.get("status") != "PASS":
        raise RuntimeError("Hadamard evaluation did not PASS")
    if protocol.get("ORIGINAL_VALIDATION_PROTOCOL_MATCH") != "YES":
        raise RuntimeError("Original validation protocol did not match")

    with (ANALYSIS / "pareto_all_checkpoints.csv").open(newline="") as handle:
        learned_raw = list(csv.DictReader(handle))
    if len(learned_raw) != 40:
        raise RuntimeError(f"expected 40 learned rows, got {len(learned_raw)}")
    learned = []
    for row in learned_raw:
        learned.append(
            {
                "condition": row["condition"],
                "update": int(row["update"]),
                "target_exposures": int(row["target_exposures"]),
                "state_mse": float(row["state_mse"]),
                "functional_mse": float(row["functional_mse"]),
                "documents": int(row["documents"]),
                "captures": int(row["captures"]),
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "source": "HISTORICAL_H64_VALIDATION",
            }
        )
    h = {
        "condition": "HADAMARD",
        "update": 0,
        "target_exposures": 0,
        "state_mse": float(had["state_relative_mse"]),
        "functional_mse": float(had["functional_relative_mse"]),
        "documents": int(had["documents"]),
        "captures": int(had["captures"]),
        "checkpoint_path": "",
        "checkpoint_sha256": "",
        "source": "CANONICAL_HADAMARD_REEVALUATED_BASELINE",
    }
    if h["documents"] != 16 or h["captures"] != 128:
        raise RuntimeError("Hadamard coverage mismatch")
    rows = learned + [h]
    for row in rows:
        row["dominated_by_count_with_hadamard"] = sum(
            dominates(other, row) for other in rows if other is not row
        )
        row["pareto_with_hadamard"] = row["dominated_by_count_with_hadamard"] == 0
    fields = [
        "condition",
        "update",
        "target_exposures",
        "state_mse",
        "functional_mse",
        "documents",
        "captures",
        "pareto_with_hadamard",
        "dominated_by_count_with_hadamard",
        "checkpoint_path",
        "checkpoint_sha256",
        "source",
    ]
    with (ANALYSIS / "h64_validation_cross_objective_with_hadamard.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def find(condition: str, update: int) -> dict:
        return next(x for x in rows if x["condition"] == condition and x["update"] == update)

    u63, u113, u119, u125 = (
        find("C5", 63),
        find("C5", 113),
        find("C6", 119),
        find("C6", 125),
    )
    h_dominates = [x for x in learned if dominates(h, x)]
    learned_dominate_h = [x for x in learned if dominates(x, h)]
    tradeoffs = [
        x
        for x in learned
        if not dominates(h, x)
        and not dominates(x, h)
        and (x["state_mse"] != h["state_mse"] or x["functional_mse"] != h["functional_mse"])
    ]
    summary = {
        "status": "PASS",
        "ORIGINAL_VALIDATION_PROTOCOL_MATCH": "YES",
        "hadamard": h,
        "hadamard_dominates_learned_count": len(h_dominates),
        "hadamard_dominates_learned_ids": [f"{x['condition']}_u{x['update']}" for x in h_dominates],
        "learned_dominates_hadamard_count": len(learned_dominate_h),
        "learned_dominates_hadamard_ids": [f"{x['condition']}_u{x['update']}" for x in learned_dominate_h],
        "tradeoff_vs_hadamard_count": len(tradeoffs),
        "tradeoff_vs_hadamard_ids": [f"{x['condition']}_u{x['update']}" for x in tradeoffs],
        "canonical_c5_pareto_optimal_with_hadamard": u63["pareto_with_hadamard"],
        "canonical_c6_pareto_optimal_with_hadamard": u125["pareto_with_hadamard"],
        "c5_u113_vs_hadamard": {
            "state_percent_change": pct(u113["state_mse"], h["state_mse"]),
            "functional_percent_change": pct(u113["functional_mse"], h["functional_mse"]),
            "better_state": u113["state_mse"] < h["state_mse"],
            "better_functional": u113["functional_mse"] < h["functional_mse"],
            "dominates_hadamard": dominates(u113, h),
            "dominated_by_hadamard": dominates(h, u113),
            "pareto_with_hadamard": u113["pareto_with_hadamard"],
        },
        "mandatory_rows": {f"{x['condition']}_u{x['update']}": x for x in (u63, u113, u119, u125)},
    }
    (ANALYSIS / "phase_b_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    inventory = [
        {
            "candidate": "Hadamard",
            "family": "HADAMARD",
            "update": 0,
            "exposure": 0,
            "canonical": "YES",
            "theta_checkpoint_path": "theta=0 initialization (no file required)",
            "theta_checkpoint_sha256": "",
            "frozen_delta_path": "identity Delta generated by audited theta=0 bank",
            "frozen_delta_sha256": "",
            "final_rotation_rule": "H128 @ I = H128 exactly",
            "deployment_status": "DEPLOYABLE_CANONICAL",
            "provenance_status": "PASS",
            "notes": "CANONICAL_HADAMARD_REEVALUATED_BASELINE; not a recovered historical update-0 row",
        },
        {
            "candidate": "C5_u63",
            "family": "C5",
            "update": 63,
            "exposure": 504,
            "canonical": "YES",
            "theta_checkpoint_path": "training-host:checkpoints/C5/candidate_update_063_exposure_0504.pt",
            "theta_checkpoint_sha256": "54416ccaf8b02cfa8d0ce5235f0b12c7db888c2ab8f07d44565eecc4c45c40d4",
            "frozen_delta_path": "training-host:artifacts/C5/frozen_rotation_fp32.pt",
            "frozen_delta_sha256": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
            "final_rotation_rule": "H128 @ stored Delta per canonical orientation",
            "deployment_status": "DEPLOYABLE_CANONICAL",
            "provenance_status": "PASS",
            "notes": "24-layer frozen artifact and deployment manifest were previously verified",
        },
        {
            "candidate": "C5_u113",
            "family": "C5",
            "update": 113,
            "exposure": 904,
            "canonical": "NO",
            "theta_checkpoint_path": "training-host:checkpoints/C5/candidate_update_113_exposure_0904.pt",
            "theta_checkpoint_sha256": "ecdc13b64c65757b9b50c85e4a85352185564f3dd62dcb3e91758a6cf5638a11",
            "frozen_delta_path": "NOT_YET_EXPORTED",
            "frozen_delta_sha256": "",
            "final_rotation_rule": "H128 @ candidate Delta after deterministic audited export",
            "deployment_status": "DEPLOYABLE_NONCANONICAL_EXPORT_REQUIRED",
            "provenance_status": "PASS_THETA_ONLY",
            "notes": "Do not call canonical; exact documented export materializes contiguous FP32 Delta from theta and validates orthogonality/hashes",
        },
        {
            "candidate": "C6_u119",
            "family": "C6",
            "update": 119,
            "exposure": 952,
            "canonical": "NO",
            "theta_checkpoint_path": "training-host:checkpoints/C6/candidate_update_119_exposure_0952.pt",
            "theta_checkpoint_sha256": "a580b847729130faee208be0beb95a2179a18d5b7efd3cda57dadb9beec29c36",
            "frozen_delta_path": "NOT_YET_EXPORTED",
            "frozen_delta_sha256": "",
            "final_rotation_rule": "H128 @ candidate Delta after deterministic audited export",
            "deployment_status": "DEPLOYABLE_NONCANONICAL_EXPORT_REQUIRED",
            "provenance_status": "PASS_THETA_ONLY",
            "notes": "Do not call canonical; exact documented export materializes contiguous FP32 Delta from theta and validates orthogonality/hashes",
        },
        {
            "candidate": "C6_u125",
            "family": "C6",
            "update": 125,
            "exposure": 1000,
            "canonical": "YES",
            "theta_checkpoint_path": "training-host:checkpoints/C6/candidate_update_125_exposure_1000.pt",
            "theta_checkpoint_sha256": "8312d59e593181e8b36de01058e492171434e3d0d9b0fb92aa6a979bdd151520",
            "frozen_delta_path": "training-host:artifacts/C6/frozen_rotation_fp32.pt",
            "frozen_delta_sha256": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
            "final_rotation_rule": "H128 @ stored Delta per canonical orientation",
            "deployment_status": "DEPLOYABLE_CANONICAL",
            "provenance_status": "PASS",
            "notes": "24-layer frozen artifact and deployment manifest were previously verified",
        },
    ]
    with (ANALYSIS / "candidate_deployment_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
