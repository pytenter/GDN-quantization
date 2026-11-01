#!/usr/bin/env python3
"""Apply frozen assignment-amendment metadata to the final report and hash set."""

from __future__ import annotations

import argparse
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    effective = json.loads((root / "manifests/effective_hardware_assignment.json").read_text())
    metadata = json.loads((root / "analysis/per_sample_gpu_metadata.json").read_text())
    if effective["status"] != "PASS_DERIVED_FROM_FROZEN_ASSIGNMENTS_AND_CANONICAL_OUTPUTS":
        raise RuntimeError("effective assignment audit is not PASS")
    if metadata["status"] != "PASS" or len(metadata["samples"]) != 40:
        raise RuntimeError("per-sample GPU metadata is incomplete")
    counts = {
        model: sum(row["gpu_model"] == model for row in metadata["samples"])
        for model in ("RTX4090", "RTX3090")
    }
    retries = {
        model: {
            "generated_samples": counts[model],
            "retries": sum(row["retry_count"] for row in metadata["samples"] if row["gpu_model"] == model),
            "infrastructure_failures": sum(
                row["infrastructure_failure_count"]
                for row in metadata["samples"]
                if row["gpu_model"] == model
            ),
        }
        for model in counts
    }
    report_path = root / "reports/final_report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "- scheduled/generated on RTX4090: 20\n- scheduled/generated on RTX3090: 20",
        f"- actual canonical generated on RTX4090 after frozen amendments: {counts['RTX4090']}\n"
        f"- actual canonical generated on RTX3090 after frozen amendments: {counts['RTX3090']}",
    )
    original_hash_line = "- hardware assignment manifest SHA256: `"
    insert_marker = "- cross-hardware parity artifact hashes:"
    if "effective assignment SHA256" not in report:
        report = report.replace(
            insert_marker,
            f"- effective assignment SHA256: `{effective['effective_assignment_sha256']}`\n"
            f"- original/amendment assignment hashes: `{json.dumps(effective['assignment_sources_sha256'], sort_keys=True)}`\n"
            f"- per-sample GPU metadata: `analysis/per_sample_gpu_metadata.json`\n"
            f"{insert_marker}",
        )
    report_path.write_text(report, encoding="utf-8")

    write_json(
        root / "analysis/final_hardware_audit.json",
        {
            "status": "PASS",
            "effective_assignment_sha256": effective["effective_assignment_sha256"],
            "assignment_sources_sha256": effective["assignment_sources_sha256"],
            "hardware": retries,
            "q26_canonical_sha256": next(
                row["source_sha256"]
                for row in metadata["samples"]
                if row["condition"] == "L7" and row["question_id"] == "aime26_26"
            ),
        },
    )

    excluded = {Path("manifests/artifact_manifest.json"), Path("hashes/artifact_sha256.txt")}
    artifact_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in excluded or "__pycache__" in relative.parts or relative.suffix == ".pid":
            continue
        artifact_rows.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(
        root / "manifests/artifact_manifest.json",
        {
            "task": "LING_RECURRENT_DENSE_L6_L7_V1",
            "status": "PASS",
            "assignment_amendment_overlay": "PASS",
            "files": artifact_rows,
            "exclusions": [str(path) for path in sorted(excluded)],
        },
    )
    hash_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root) != Path("hashes/artifact_sha256.txt")
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pid"
    ]
    lines = [f"{sha256(path)}  {path.relative_to(root)}" for path in hash_paths]
    (root / "hashes/artifact_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "hardware": retries, "artifact_files": len(artifact_rows), "hash_lines": len(lines)}, indent=2))


if __name__ == "__main__":
    main()
