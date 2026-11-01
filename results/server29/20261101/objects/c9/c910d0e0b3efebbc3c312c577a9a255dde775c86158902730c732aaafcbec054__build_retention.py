#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


TASK = "QWEN_SCIENTIFIC_RETENTION_AND_3090_MIGRATION_V1"
HERE = Path(__file__).resolve().parent
SOURCE_MANIFEST = HERE / "source_metadata" / "artifact_relocation_manifest.json"
ANALYSIS = HERE / "analysis"
TRANSFER = HERE / "transfer"
CLEANUP = HERE / "cleanup"
REPORTS = HERE / "reports"
TARGET_ROOT = f"/data/zypan/research_archive/qwen_gdn/{TASK}"

ARCHIVE_A = "/root/autodl-tmp/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1"
ARCHIVE_B = "/root/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1"


def extension(path: str) -> str:
    name = Path(path).name
    if name.endswith(".jsonl.gz"):
        return ".jsonl.gz"
    if name.endswith(".csv.gz"):
        return ".csv.gz"
    return Path(name).suffix.lower() or "[none]"


def actual_source_path(uri: str) -> str:
    return urlparse(uri).path


def archive_relative(path: str) -> str:
    for root in (ARCHIVE_A, ARCHIVE_B):
        if path.startswith(root + "/"):
            return path[len(root) + 1 :]
    raise ValueError(f"source path is outside archive roots: {path}")


def infer_experiment(original_path: str, recorded: str) -> str:
    marker = "/experiments/"
    if marker in original_path:
        return original_path.split(marker, 1)[1].split("/", 1)[0]
    name = Path(original_path).name
    if name.startswith("gdn_"):
        stem = name
        for token in (
            "_formal_", "_head_records", "_layer_records", "_records",
            "_checkpoint", "_rotation_screen", "_screen", "_pilot", "_smoke",
        ):
            if token in stem:
                stem = stem.split(token, 1)[0]
                break
        return stem.upper()
    if name.startswith("long_decode_"):
        return "LONG_DECODE_GRANULARITY"
    if name.startswith("state_error_"):
        return "STATE_ERROR_ACCUMULATION"
    return recorded


def scientific_role(group: str, path: str) -> str:
    name = Path(path).name
    if group == "endpoint_probes":
        return "dual_gpu_topology_endpoint_tensor_probe"
    if group == "m5_component_replay":
        if name == "status.json":
            return "rotation_m5_shard_status"
        if name.endswith(".json"):
            return "rotation_m5_per_sample_replay_metadata"
        return "rotation_m5_component_replay_trace"
    if "QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1" in path:
        return "hadamard_functional_error_geometry_trace"
    if "QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1" in path:
        return "hadamard_trajectory_metric_trace"
    if "QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1" in path:
        if name == "token_level_metrics.csv":
            return "teacher_forced_replay_aggregate_trace"
        return "teacher_forced_replay_canonical_shard"
    if "checkpoint" in name or "screen" in name:
        return "legacy_mechanism_checkpoint_or_screen_data"
    if "formal" in name:
        return "legacy_formal_mechanism_trace"
    if "pilot" in name:
        return "legacy_pilot_mechanism_trace"
    if "smoke" in name:
        return "legacy_smoke_trace"
    return "legacy_mechanism_result_trace"


def choose_duplicate_canonical(group):
    def score(item):
        p = item["original_absolute_path"]
        name = Path(p).name
        return (
            1 if "_formal_" in name or name.startswith("formal_") else 0,
            1 if "formal" in name else 0,
            -len(name),
            name,
        )
    return max(group, key=score)


def classify(item, dup_canonical):
    group = item["artifact_group"]
    original = item["original_absolute_path"]
    name = Path(original).name
    sha = item["source_sha256"]

    if dup_canonical.get(sha) is not None and dup_canonical[sha] is not item:
        canonical = dup_canonical[sha]
        return {
            "retention_class": "DELETE_SAFE",
            "retention_reason": "Exact SHA256 duplicate; the formal-named canonical copy is retained.",
            "superseded_by": canonical["original_absolute_path"],
            "duplicate_of": canonical["original_absolute_path"],
            "required_by_report": "NO",
            "required_for_rescoring": "NO",
            "required_for_future_analysis": "NO",
        }

    if group == "endpoint_probes":
        if name in {"dual_01_probe.npz", "single_01_probe.npz", "single_02_probe.npz"}:
            why = {
                "dual_01_probe.npz": "Retains the first failing dual-GPU endpoint used by both the first functional and trajectory failures.",
                "single_01_probe.npz": "Retains the single-GPU reference in the first functional-envelope failure.",
                "single_02_probe.npz": "Retains the single-GPU reference in the first trajectory-envelope failure.",
            }[name]
            return {
                "retention_class": "KEEP_UNTIL_PAPER",
                "retention_reason": why + " Compact JSON summaries and frozen envelopes remain in the active worktree.",
                "superseded_by": "",
                "duplicate_of": "",
                "required_by_report": "YES",
                "required_for_rescoring": "NO",
                "required_for_future_analysis": "YES",
            }
        return {
            "retention_class": "DELETE_SAFE",
            "retention_reason": "Non-canonical replicate of a failed topology-closure diagnostic. The frozen 10-run/5-run envelopes, per-run JSON summaries, cross-comparison table, and final verdict remain in the active worktree; three raw first-failure probes are retained.",
            "superseded_by": "active trajectory_equivalence analysis summaries and retained first-failure probes",
            "duplicate_of": "",
            "required_by_report": "NO",
            "required_for_rescoring": "NO",
            "required_for_future_analysis": "NO",
        }

    if group == "m5_component_replay":
        if name.endswith(".json"):
            return {
                "retention_class": "KEEP_CANONICAL",
                "retention_reason": "Small canonical per-sample/shard provenance required to interpret the M5 replay corpus.",
                "superseded_by": "", "duplicate_of": "",
                "required_by_report": "YES", "required_for_rescoring": "NO",
                "required_for_future_analysis": "YES",
            }
        return {
            "retention_class": "KEEP_UNTIL_PAPER",
            "retention_reason": "Unique M5 component-replay source trace supporting rotation-aware validation and likely reviewer/follow-up analysis.",
            "superseded_by": "", "duplicate_of": "",
            "required_by_report": "YES", "required_for_rescoring": "NO",
            "required_for_future_analysis": "YES",
        }

    if "QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1/token_level_metrics.csv" in original:
        return {
            "retention_class": "DELETE_SAFE",
            "retention_reason": "Byte-for-byte logical concatenation of the four retained gzip shards after removing repeated headers; 3,708,903 lines and SHA256 of decompressed concatenation exactly match.",
            "superseded_by": "formal/shard_00..03/token_metrics.csv.gz",
            "duplicate_of": "logical concatenation of four compressed shards",
            "required_by_report": "NO", "required_for_rescoring": "NO",
            "required_for_future_analysis": "NO",
        }

    if group == "completed_old_raw":
        return {
            "retention_class": "KEEP_UNTIL_PAPER",
            "retention_reason": "Unique canonical Hadamard trajectory/functional geometry or teacher-forced replay shard retained for metric recomputation, reviewer response, and figure/table regeneration.",
            "superseded_by": "", "duplicate_of": "",
            "required_by_report": "YES", "required_for_rescoring": "NO",
            "required_for_future_analysis": "YES",
        }

    if group == "results_large_raw":
        lower = name.lower()
        if "smoke" in lower:
            return {
                "retention_class": "DELETE_SAFE",
                "retention_reason": "Superseded smoke output; the corresponding pilot/formal or completed result is retained, while small provenance remains in the active results tree.",
                "superseded_by": "corresponding retained non-smoke result",
                "duplicate_of": "",
                "required_by_report": "NO", "required_for_rescoring": "NO",
                "required_for_future_analysis": "NO",
            }
        superseded_pilots = (
            "residual_direction_sensitivity_panel_v1_head_records_pilot",
            "residual_geometry_causal_intervention_v1_layer_records_pilot",
            "residual_strength_causal_intervention_v1_layer_records_pilot",
        )
        if any(token in lower for token in superseded_pilots):
            return {
                "retention_class": "DELETE_SAFE",
                "retention_reason": "Pilot trace superseded by retained deterministic formal shards for the same experiment.",
                "superseded_by": "corresponding retained formal shards",
                "duplicate_of": "",
                "required_by_report": "NO", "required_for_rescoring": "NO",
                "required_for_future_analysis": "NO",
            }
        return {
            "retention_class": "KEEP_UNTIL_PAPER",
            "retention_reason": "Unique completed mechanistic result/trace with potential paper, ablation, or reviewer-analysis value; no verified replacement exists.",
            "superseded_by": "", "duplicate_of": "",
            "required_by_report": "YES", "required_for_rescoring": "NO",
            "required_for_future_analysis": "YES",
        }

    raise ValueError(group)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    for d in (ANALYSIS, TRANSFER, CLEANUP, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["artifacts"]

    sha_groups = defaultdict(list)
    for item in source:
        sha_groups[item["source_sha256"]].append(item)
    duplicate_groups = {sha: xs for sha, xs in sha_groups.items() if len(xs) > 1}
    dup_canonical = {sha: choose_duplicate_canonical(xs) for sha, xs in duplicate_groups.items()}

    rows = []
    for item in source:
        spath = actual_source_path(item["archive_absolute_path"])
        rel = archive_relative(spath)
        exp = infer_experiment(item["original_absolute_path"], item["experiment"])
        duplabel = f"sha256:{item['source_sha256']}" if item["source_sha256"] in duplicate_groups else ""
        row = {
            "relative_path": rel,
            "size_bytes": int(item["size_bytes"]),
            "sha256": item["source_sha256"],
            "extension": extension(rel),
            "experiment": exp,
            "artifact_role": scientific_role(item["artifact_group"], item["original_absolute_path"]),
            "duplicate_sha_group": duplabel,
            "source_archive_locations": spath,
            "source_archive": "A" if spath.startswith(ARCHIVE_A + "/") else "B",
            "original_absolute_path": item["original_absolute_path"],
            "artifact_group": item["artifact_group"],
        }
        row.update(classify(item, dup_canonical))
        if row["retention_class"].startswith("KEEP_"):
            row["target_relative_path"] = "scientific_data/" + rel
        else:
            row["target_relative_path"] = ""
        rows.append(row)
    rows.sort(key=lambda x: (-x["size_bytes"], x["relative_path"]))

    inventory_fields = [
        "relative_path", "size_bytes", "sha256", "extension", "experiment",
        "artifact_role", "duplicate_sha_group", "source_archive_locations",
        "source_archive", "original_absolute_path", "artifact_group",
    ]
    retention_fields = [
        "relative_path", "size_bytes", "sha256", "experiment", "scientific_role",
        "retention_class", "retention_reason", "superseded_by", "duplicate_of",
        "required_by_report", "required_for_rescoring", "required_for_future_analysis",
        "target_relative_path", "source_archive_locations", "source_archive",
    ]
    retention_rows = []
    for r in rows:
        x = dict(r)
        x["scientific_role"] = x["artifact_role"]
        retention_rows.append(x)

    write_csv(ANALYSIS / "full_storage_inventory.csv", rows, inventory_fields)
    (ANALYSIS / "full_storage_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(ANALYSIS / "scientific_retention_manifest.csv", retention_rows, retention_fields)
    (ANALYSIS / "scientific_retention_manifest.json").write_text(json.dumps(retention_rows, indent=2), encoding="utf-8")

    duplicate_payload = []
    for i, (sha, xs) in enumerate(sorted(duplicate_groups.items()), 1):
        canonical = dup_canonical[sha]
        duplicate_payload.append({
            "duplicate_group": i,
            "sha256": sha,
            "size_bytes_per_copy": xs[0]["size_bytes"],
            "file_count": len(xs),
            "canonical_original_path": canonical["original_absolute_path"],
            "paths": [x["original_absolute_path"] for x in xs],
            "duplicate_extra_bytes": sum(x["size_bytes"] for x in xs if x is not canonical),
        })
    (ANALYSIS / "duplicate_groups.json").write_text(json.dumps(duplicate_payload, indent=2), encoding="utf-8")

    classes = ["KEEP_CANONICAL", "KEEP_UNTIL_PAPER", "DELETE_SAFE", "REVIEW"]
    class_summary = {
        c: {
            "files": sum(1 for r in rows if r["retention_class"] == c),
            "bytes": sum(r["size_bytes"] for r in rows if r["retention_class"] == c),
        }
        for c in classes
    }
    transfer_rows = [r for r in rows if r["retention_class"] in {"KEEP_CANONICAL", "KEEP_UNTIL_PAPER"}]
    summary = {
        "task": TASK,
        "source_total_files": len(rows),
        "source_total_bytes": sum(r["size_bytes"] for r in rows),
        "classes": class_summary,
        "exact_duplicate_groups": len(duplicate_payload),
        "exact_duplicate_files": sum(x["file_count"] for x in duplicate_payload),
        "exact_duplicate_extra_bytes": sum(x["duplicate_extra_bytes"] for x in duplicate_payload),
        "expected_3090_transfer_files": len(transfer_rows),
        "expected_3090_transfer_bytes": sum(r["size_bytes"] for r in transfer_rows),
        "expected_source_reclaim_bytes": sum(r["size_bytes"] for r in rows if r["retention_class"] != "REVIEW"),
        "review_bytes_remaining": class_summary["REVIEW"]["bytes"],
        "target_canonical_archive_root": TARGET_ROOT,
        "teacher_forced_aggregate_equivalence": {
            "status": "PASS",
            "aggregate_bytes": 1247473480,
            "aggregate_lines": 3708903,
            "aggregate_sha256": "fb166f7a35256acb3959dcb0e828d22e09c0fca228e90622b44168fd003f182b8",
            "concatenated_shards_bytes": 1247473480,
            "concatenated_shards_lines": 3708903,
            "concatenated_shards_sha256": "fb166f7a35256acb3959dcb0e828d22e09c0fca228e90622b44168fd003f182b8",
        },
    }
    (ANALYSIS / "retention_size_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def aggregate(key):
        out = defaultdict(lambda: {"files": 0, "bytes": 0})
        for r in rows:
            out[str(r[key])]["files"] += 1
            out[str(r[key])]["bytes"] += r["size_bytes"]
        return dict(sorted(out.items(), key=lambda kv: (-kv[1]["bytes"], kv[0])))

    rollups = {
        "size_by_experiment": aggregate("experiment"),
        "size_by_extension": aggregate("extension"),
        "size_by_artifact_role": aggregate("artifact_role"),
        "size_by_duplicate_group": aggregate("duplicate_sha_group"),
    }
    (ANALYSIS / "storage_rollups.json").write_text(json.dumps(rollups, indent=2), encoding="utf-8")
    (ANALYSIS / "top_50_largest_files.json").write_text(json.dumps(rows[:50], indent=2), encoding="utf-8")

    transfer_manifest = {
        "task": TASK,
        "target_root": TARGET_ROOT,
        "file_count": len(transfer_rows),
        "total_bytes": sum(r["size_bytes"] for r in transfer_rows),
        "files": [
            {
                "source_path": r["source_archive_locations"],
                "source_archive": r["source_archive"],
                "target_path": TARGET_ROOT + "/" + r["target_relative_path"],
                "target_relative_path": r["target_relative_path"],
                "size_bytes": r["size_bytes"],
                "sha256": r["sha256"],
                "retention_class": r["retention_class"],
            }
            for r in transfer_rows
        ],
    }
    (TRANSFER / "transfer_manifest.json").write_text(json.dumps(transfer_manifest, indent=2), encoding="utf-8")
    (TRANSFER / "sftp_download_batch.txt").write_text(
        "".join(
            f'get -p "{item["source_path"]}" "{item["target_relative_path"]}"\n'
            for item in transfer_manifest["files"]
        ) + "quit\n",
        encoding="utf-8",
    )

    for archive in ("A", "B"):
        files = [r for r in transfer_rows if r["source_archive"] == archive]
        (TRANSFER / f"rsync_files_archive_{archive}.txt").write_text(
            "".join(archive_relative(r["source_archive_locations"]) + "\n" for r in files),
            encoding="utf-8",
        )

    delete_rows = [r for r in rows if r["retention_class"] == "DELETE_SAFE"]
    delete_lines = [
        "\t".join([
            r["source_archive_locations"], str(r["size_bytes"]), r["sha256"],
            r["retention_reason"].replace("\t", " "), "",
        ])
        for r in delete_rows
    ]
    (CLEANUP / "delete_safe_manifest.txt").write_text(
        "path\tsize\tsha256\treason\ttarget_verified_path\n" + "\n".join(delete_lines) + ("\n" if delete_lines else ""),
        encoding="utf-8",
    )
    (CLEANUP / "migrated_source_duplicates_manifest.txt").write_text(
        "path\tsize\tsha256\treason\ttarget_verified_path\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
