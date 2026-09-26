#!/usr/bin/env python3
"""Finalize recurrent trace provenance and AIME26 non-overlap manifests."""

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


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def collect_rows(trace_dir: Path, split: str) -> list[dict]:
    completions = sorted(trace_dir.glob(f"COLLECTION_{split.upper()}_SHARD*_COMPLETE.json"))
    rows = []
    for path in completions:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "PASS":
            raise RuntimeError(f"failed collection manifest: {path}")
        rows.extend(payload["rows"])
    rows.sort(key=lambda row: row["document_id"])
    expected = 64 if split == "train" else 16
    if len(rows) != expected or len({row["document_id"] for row in rows}) != expected:
        raise RuntimeError(f"{split} expected {expected} unique rows, found {len(rows)}")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"trace hash mismatch: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-samples", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    trace_dir, corpus, output_dir = Path(args.trace_dir), Path(args.corpus), Path(args.output_dir)
    corpus_rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    eval_rows = [json.loads(line) for line in Path(args.eval_samples).read_text(encoding="utf-8").splitlines() if line]
    corpus_text = {row["document_id"]: normalize(row["raw_text"]) for row in corpus_rows}
    aime_text = {row["question_id"]: normalize(row["problem"]) for row in eval_rows}
    overlap = []
    for document_id, text in corpus_text.items():
        for question_id, problem in aime_text.items():
            if hashlib.sha256(text.encode()).digest() == hashlib.sha256(problem.encode()).digest() or (
                len(problem) >= 80 and problem in text
            ):
                overlap.append({"document_id": document_id, "question_id": question_id})
    if overlap:
        raise RuntimeError(f"AIME26 overlap detected: {overlap}")
    for split in ("train", "validation"):
        rows = collect_rows(trace_dir, split)
        atomic_json(
            output_dir / f"{split}_data_manifest.json",
            {
                "task": "LING_RECURRENT_DENSE_L6_L7_V1",
                "split": split.upper(),
                "status": "PASS",
                "raw_corpus": str(corpus.resolve()),
                "raw_corpus_sha256": sha256(corpus),
                "documents": len(rows),
                "sequence_length": 512,
                "continuous_tokens_per_document": True,
                "rows": rows,
                "AIME26_used": False,
                "AIME26_overlap": "NO",
            },
        )
    atomic_json(
        output_dir / "training_data_leakage.json",
        {
            "task": "LING_RECURRENT_DENSE_L6_L7_V1",
            "status": "PASS",
            "AIME26_overlap": "NO",
            "checks": ["normalized exact equality", "AIME problem substring of at least 80 characters"],
            "overlaps": overlap,
            "eval_question_ids": sorted(aime_text),
        },
    )


if __name__ == "__main__":
    main()
