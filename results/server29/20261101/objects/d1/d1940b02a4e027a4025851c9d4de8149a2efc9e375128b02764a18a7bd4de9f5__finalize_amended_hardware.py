#!/usr/bin/env python3
"""Build actual-hardware metadata after prospectively frozen assignment amendments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_assignments() -> dict[str, dict[int, tuple[int, str]]]:
    return {
        "L6": {
            **{q: (0, "RTX4090") for q in (11, 15, 19, 23, 27, 29)},
            **{q: (1, "RTX4090") for q in (12, 16, 20, 24, 28)},
            **{q: (3, "RTX3090") for q in (13, 17, 21, 25)},
            **{q: (4, "RTX3090") for q in (14, 18, 22, 26, 30)},
        },
        "L7": {
            **{q: (0, "RTX4090") for q in (11, 15, 19, 23, 27)},
            **{q: (1, "RTX4090") for q in (12, 16, 20, 24, 25, 28, 29, 30)},
            **{q: (3, "RTX3090") for q in (13, 17, 21, 26)},
            **{q: (4, "RTX3090") for q in (14, 18, 22)},
        },
    }


def paired(name: str, treatment: str, reference: str, data: dict, ids: set[str]) -> dict:
    selected = sorted(ids & set(data[treatment]) & set(data[reference]))
    rescued = [qid for qid in selected if data[treatment][qid] and not data[reference][qid]]
    regressed = [qid for qid in selected if not data[treatment][qid] and data[reference][qid]]
    return {
        "name": name,
        "treatment": treatment,
        "reference": reference,
        "n": len(selected),
        "treatment_correct": sum(data[treatment][qid] for qid in selected),
        "reference_correct": sum(data[reference][qid] for qid in selected),
        "rescued": len(rescued),
        "regressed": len(regressed),
        "net_gain_questions": len(rescued) - len(regressed),
        "rescued_ids": rescued,
        "regressed_ids": regressed,
        "question_ids": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    expected = expected_assignments()
    assignment_rows = []
    metadata_rows = []
    correctness: dict[str, dict[str, bool]] = {"L2": {}, "L6": {}, "L7": {}}

    score_path = root / "analysis/per_sample_scores.csv"
    with score_path.open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    for row in score_rows:
        correctness[row["condition"]][row["question_id"]] = row["correct"].lower() == "true"

    actual_ids: dict[str, dict[str, set[str]]] = {
        condition: {"RTX4090": set(), "RTX3090": set()} for condition in ("L6", "L7")
    }
    row_by_key = {(row["condition"], row["question_id"]): row for row in score_rows}
    for condition in ("L6", "L7"):
        files = sorted((root / "outputs" / condition).glob("aime26_*_seed1.json"))
        if len(files) != 20:
            raise RuntimeError(f"{condition}: expected 20 samples, found {len(files)}")
        for path in files:
            item = json.loads(path.read_text(encoding="utf-8"))
            qid = item["question_id"]
            q = int(qid.rsplit("_", 1)[1])
            expected_gpu, model = expected[condition][q]
            if int(item["physical_gpu"]) != expected_gpu:
                raise RuntimeError(f"effective assignment mismatch: {condition} {qid}")
            instance_id = item["instance_id"]
            if model.removeprefix("RTX") not in instance_id:
                raise RuntimeError(f"instance/model mismatch: {condition} {qid}")
            if item.get("canonical") is False or item.get("excluded_from_scoring") is True:
                raise RuntimeError(f"noncanonical sample in main output: {condition} {qid}")
            actual_ids[condition][model].add(qid)
            row = row_by_key[(condition, qid)]
            row["gpu_architecture"] = "Ada SM 8.9" if model == "RTX4090" else "Ampere SM 8.6"
            row["gpu_model"] = model
            row["worker_or_gpu_id"] = instance_id
            assignment_rows.append(
                {
                    "condition": condition,
                    "question_id": qid,
                    "seed": 1,
                    "physical_gpu": expected_gpu,
                    "gpu_model": model,
                    "instance_id": instance_id,
                    "source_sha256": sha256(path),
                }
            )
            metadata_rows.append(
                {
                    **assignment_rows[-1],
                    "generated_tokens": int(item["generated_tokens"]),
                    "correct": bool(item["v4_correct"]),
                    "abstain": bool(item["v4_abstain"]),
                    "retry_count": int(item.get("retry_count", 0)),
                    "infrastructure_failure_count": len(item.get("infrastructure_failures", [])),
                }
            )

    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(score_rows)

    amendment_paths = [
        Path("manifests/hardware_assignment.json"),
        Path("manifests/hardware_assignment_amendment_q29_helper_v2.json"),
        Path("manifests/hardware_assignment_amendment_l7_idle_4090_helper_v3.json"),
        Path("manifests/hardware_assignment_amendment_l7_idle_helpers_v5.json"),
    ]
    amendment_hashes = {str(path): sha256(root / path) for path in amendment_paths}
    effective = {
        "status": "PASS_DERIVED_FROM_FROZEN_ASSIGNMENTS_AND_CANONICAL_OUTPUTS",
        "seed": 1,
        "assignment_sources_sha256": amendment_hashes,
        "assignments": sorted(assignment_rows, key=lambda row: (row["condition"], row["question_id"])),
    }
    canonical = json.dumps(effective["assignments"], sort_keys=True, separators=(",", ":")).encode()
    effective["effective_assignment_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(root / "manifests/effective_hardware_assignment.json", effective)
    write_json(
        root / "analysis/per_sample_gpu_metadata.json",
        {
            "status": "PASS",
            "assignment_sources_sha256": amendment_hashes,
            "samples": sorted(metadata_rows, key=lambda row: (row["condition"], row["question_id"])),
        },
    )

    strata = {}
    for model in ("RTX4090", "RTX3090"):
        condition_stats = {}
        for condition in ("L6", "L7"):
            ids = actual_ids[condition][model]
            condition_stats[condition] = {
                "n": len(ids),
                "correct": sum(correctness[condition][qid] for qid in ids),
                "question_ids": sorted(ids),
                "vs_L2": paired(f"{condition}_vs_L2", condition, "L2", correctness, ids),
            }
        same_hardware = actual_ids["L6"][model] & actual_ids["L7"][model]
        strata[model] = {
            "status": "DESCRIPTIVE_NOT_INDEPENDENT_PRIMARY_TEST",
            "condition_actual_generation": condition_stats,
            "L7_vs_L6_same_hardware_intersection": paired(
                "L7_vs_L6_same_hardware_intersection", "L7", "L6", correctness, same_hardware
            ),
        }
    cross_hardware_ids = sorted(
        (actual_ids["L6"]["RTX4090"] & actual_ids["L7"]["RTX3090"])
        | (actual_ids["L6"]["RTX3090"] & actual_ids["L7"]["RTX4090"])
    )
    write_json(
        root / "analysis/hardware_stratified.json",
        {
            "status": "COMPLETE",
            "interpretation": "descriptive actual-generation hardware strata; not independent primary tests",
            "post_hoc_subset_deletion_allowed": False,
            "assignment_sources_sha256": amendment_hashes,
            "strata": strata,
            "cross_hardware_L6_L7_question_ids": cross_hardware_ids,
        },
    )
    counts = {
        model: {
            "generated_samples": sum(row["gpu_model"] == model for row in metadata_rows),
            "retries": sum(row["retry_count"] for row in metadata_rows if row["gpu_model"] == model),
            "infrastructure_failures": sum(
                row["infrastructure_failure_count"] for row in metadata_rows if row["gpu_model"] == model
            ),
        }
        for model in ("RTX4090", "RTX3090")
    }
    print(json.dumps({"status": "PASS", "effective_assignment_sha256": effective["effective_assignment_sha256"], "hardware": counts}, indent=2))


if __name__ == "__main__":
    main()
