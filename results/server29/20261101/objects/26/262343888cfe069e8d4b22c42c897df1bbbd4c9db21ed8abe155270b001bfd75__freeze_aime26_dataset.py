#!/usr/bin/env python3
"""Freeze the pinned MathArena/AIME 2026 parquet into deterministic JSONL."""

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "MathArena/aime_2026"
REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"
SOURCE_PATH = "data/train-00000-of-00001.parquet"
EXPECTED_SOURCE_SHA256 = "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rows-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256_file(args.source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"source SHA256 mismatch: {source_sha}")

    if args.rows_json is not None:
        payload = json.loads(args.rows_json.read_text(encoding="utf-8-sig"))
        if int(payload.get("num_rows_total", -1)) != 30:
            raise SystemExit("dataset-server snapshot must report 30 total rows")
        if any(item.get("truncated_cells") for item in payload.get("rows", [])):
            raise SystemExit("dataset-server snapshot contains truncated cells")
        rows = [item["row"] for item in payload["rows"]]
        column_names = list(rows[0]) if rows else []
    else:
        import pyarrow.parquet as pq

        table = pq.read_table(args.source)
        rows = table.to_pylist()
        column_names = list(table.column_names)
    if len(rows) != 30:
        raise SystemExit(f"expected 30 rows, found {len(rows)}")
    required = {"problem_idx", "problem", "answer"}
    if not required.issubset(column_names):
        raise SystemExit(f"missing columns: {sorted(required - set(column_names))}")

    frozen = []
    for row in sorted(rows, key=lambda item: int(item["problem_idx"])):
        idx = int(row["problem_idx"])
        frozen.append(
            {
                "problem_id": f"aime26_{idx:02d}",
                "problem_idx": idx,
                "problem": str(row["problem"]),
                "answer": str(int(row["answer"])),
            }
        )
    if [row["problem_idx"] for row in frozen] != list(range(1, 31)):
        raise SystemExit("problem_idx must be exactly 1..30")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "aime26_frozen.jsonl"
    with frozen_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frozen:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_repo": REPO_ID,
        "revision": REVISION,
        "source_file": SOURCE_PATH,
        "source_file_sha256": source_sha,
        "source_file_size_bytes": args.source.stat().st_size,
        "row_count": len(frozen),
        "problem_idx_min": min(row["problem_idx"] for row in frozen),
        "problem_idx_max": max(row["problem_idx"] for row in frozen),
        "columns": column_names,
        "frozen_file": frozen_path.name,
        "frozen_file_sha256": sha256_file(frozen_path),
        "freeze_status": "PASS",
    }
    if args.rows_json is not None:
        manifest["dataset_server_rows_file"] = args.rows_json.name
        manifest["dataset_server_rows_sha256"] = sha256_file(args.rows_json)
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
