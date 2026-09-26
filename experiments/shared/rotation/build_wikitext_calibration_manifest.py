#!/usr/bin/env python3
"""Freeze a local-only, non-AIME WikiText calibration/held-out corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SEED = 20260921
SCHEMA = "HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1_RAW_TEXT_V1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load_arrow(path: Path) -> list[str]:
    from datasets import Dataset

    dataset = Dataset.from_file(str(path))
    if dataset.column_names != ["text"]:
        raise RuntimeError(f"unexpected WikiText columns in {path}: {dataset.column_names}")
    return [str(row["text"]) for row in dataset]


def article_documents(rows: list[str], source_split: str, minimum_chars: int = 5000) -> list[dict]:
    """Join adjacent public WikiText paragraphs into deterministic long documents."""
    documents: list[dict] = []
    buffer: list[str] = []
    source_rows: list[int] = []
    chars = 0
    for row_id, text in enumerate(rows):
        stripped = text.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        source_rows.append(row_id)
        chars += len(stripped)
        if chars >= minimum_chars:
            raw = "\n\n".join(buffer)
            documents.append({
                "source_split": source_split,
                "source_row_start": source_rows[0],
                "source_row_end": source_rows[-1],
                "raw_text": raw,
                "raw_text_sha256": sha256_bytes(raw.encode("utf-8")),
                "raw_characters": len(raw),
            })
            buffer, source_rows, chars = [], [], 0
    return documents


def select(documents: list[dict], count: int, seed: int) -> list[dict]:
    if len(documents) < count:
        raise RuntimeError(f"need {count} documents but only {len(documents)} are available")
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    return [documents[index] for index in indices[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-arrow", required=True)
    parser.add_argument("--validation-arrow", required=True)
    parser.add_argument("--test-arrow", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--heldout-count", type=int, default=16)
    args = parser.parse_args()

    paths = {
        "train": Path(args.train_arrow).resolve(),
        "validation": Path(args.validation_arrow).resolve(),
        "test": Path(args.test_arrow).resolve(),
    }
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.heldout_count}
    role = {"train": "TRAIN", "validation": "VALIDATION", "test": "HELDOUT"}
    rows: list[dict] = []
    source_files = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        source = paths[split]
        source_bytes = source.read_bytes()
        source_files[split] = {
            "path": str(source),
            "sha256": sha256_bytes(source_bytes),
            "bytes": len(source_bytes),
        }
        documents = article_documents(load_arrow(source), split)
        picked = select(documents, counts[split], SEED + split_index)
        for local_index, document in enumerate(picked):
            document.update({"split": role[split], "document_id": f"wikitext2raw-{split}-{local_index:03d}"})
            rows.append(document)

    hashes = [row["raw_text_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("raw text hash overlap across train/validation/heldout")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "CALIBRATION_RAW_TEXTS.jsonl"
    corpus_bytes = b"".join(canonical_json(row) for row in rows)
    corpus_path.write_bytes(corpus_bytes)
    manifest = {
        "schema": SCHEMA,
        "source": "local Hugging Face cache: wikitext/wikitext-2-raw-v1 (pre-existing; no download)",
        "sampling_seed": SEED,
        "assembly": "nonempty adjacent rows joined until >=5000 Unicode characters",
        "counts": {"TRAIN": args.train_count, "VALIDATION": args.validation_count, "HELDOUT": args.heldout_count},
        "max_token_length": 1024,
        "positions_per_sequence": 8,
        "minimum_sampled_position": 128,
        "source_files": source_files,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_bytes(corpus_bytes),
        "document_ids": {split: [row["document_id"] for row in rows if row["split"] == split] for split in ("TRAIN", "VALIDATION", "HELDOUT")},
        "raw_text_hashes": {row["document_id"]: row["raw_text_sha256"] for row in rows},
        "AIME26_used": False,
        "train_validation_overlap": 0,
        "train_heldout_panel_overlap": 0,
        "validation_heldout_overlap": 0,
        "status": "PASS",
    }
    manifest_path = output / "CALIBRATION_RAW_MANIFEST.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({"manifest": str(manifest_path), "corpus": str(corpus_path), "corpus_sha256": manifest["corpus_sha256"], "counts": manifest["counts"], "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
