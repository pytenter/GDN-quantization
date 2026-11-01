#!/usr/bin/env python3
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

root = Path.cwd()
scorer_path = root / "experiments/shared/scoring/aime26_scorer_v4.py"
spec = importlib.util.spec_from_file_location("aime26_scorer_v4", scorer_path)
sc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sc
spec.loader.exec_module(sc)

base = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
methods = {
    "fp_state": "QWEN_FP_STATE",
    "int8_c128": "QWEN_INT8_C128",
    "int8_c128_key_h": "QWEN_INT8_C128_KEY_HADAMARD",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


rows = []
sources = []
for short_name, condition in methods.items():
    paths = sorted(base.glob(f"worker*/{short_name}.jsonl"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four shards for {short_name}, found {len(paths)}")
    for path in paths:
        stat = path.stat()
        sources.append({
            "condition": condition,
            "original_path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": sha256(path),
        })
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            result = sc.extract_v4(
                record["response"],
                record.get("finish_reason", ""),
                record.get("problem", ""),
            )
            correct = bool(sc.compare_with_gold(result, record.get("gold_answer")))
            rows.append({
                "model": "Qwen3.5-9B",
                "condition": condition,
                "problem_id": record["problem_id"],
                "seed": int(record["seed"]),
                "gold_answer": record.get("gold_answer"),
                "v4_status": result.status,
                "v4_extracted_answer": result.normalized_value,
                "v4_correct": correct,
                "v4_abstain_reason": result.abstain_reason,
                "finish_reason": record.get("finish_reason"),
                "truncated": record.get("truncated"),
                "eos_seen": record.get("eos_seen"),
                "generated_token_count": record.get("generated_token_count"),
                "runtime_error": bool(record.get("runtime_error")),
                "nonfinite": bool(record.get("nonfinite")),
                "source_file": str(path),
            })

rows.sort(key=lambda row: (row["condition"], row["problem_id"], row["seed"]))
if len(rows) != 180:
    raise RuntimeError(f"expected 180 records, found {len(rows)}")

output = root / "results/aime26/81920/qwen_gdn"
output.mkdir(parents=True, exist_ok=True)
fields = list(rows[0])
with (output / "frozen_v4_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

conditions = {}
for condition in methods.values():
    subset = [row for row in rows if row["condition"] == condition]
    correct = sum(row["v4_correct"] for row in subset)
    abstain = sum(row["v4_status"] == "ABSTAIN" for row in subset)
    conditions[condition] = {
        "samples": len(subset),
        "correct": correct,
        "wrong": len(subset) - correct,
        "abstain": abstain,
        "accuracy": correct / len(subset),
    }


def paired(baseline, treatment):
    before = {
        (row["problem_id"], row["seed"]): row["v4_correct"]
        for row in rows if row["condition"] == baseline
    }
    after = {
        (row["problem_id"], row["seed"]): row["v4_correct"]
        for row in rows if row["condition"] == treatment
    }
    keys = sorted(set(before) & set(after))
    rescued = sum((not before[key]) and after[key] for key in keys)
    lost = sum(before[key] and (not after[key]) for key in keys)
    return {
        "paired_samples": len(keys),
        "rescued": rescued,
        "lost": lost,
        "net": rescued - lost,
    }


summary = {
    "task": "AIME26_81920_FROZEN_V4_ARCHIVE",
    "server": "qwen3090",
    "scorer": sc.SCORER_NAME,
    "scorer_sha256": sha256(scorer_path),
    "max_new_tokens": 81920,
    "samples_per_condition": 60,
    "conditions": conditions,
    "pairwise": {
        "KEY_HADAMARD_VS_INT8_C128": paired(
            "QWEN_INT8_C128", "QWEN_INT8_C128_KEY_HADAMARD"
        ),
        "KEY_HADAMARD_VS_FP_STATE": paired(
            "QWEN_FP_STATE", "QWEN_INT8_C128_KEY_HADAMARD"
        ),
    },
    "source_files": sources,
}
(output / "frozen_v4_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
provenance = {
    "server": "qwen3090",
    "source_kind": "completed formal generation JSONL",
    "derivation": "Frozen scorer applied to immutable response text; no response text is committed.",
    "scorer_path": "experiments/shared/scoring/aime26_scorer_v4.py",
    "scorer_sha256": summary["scorer_sha256"],
    "derived_files": ["frozen_v4_per_sample.csv", "frozen_v4_summary.json"],
    "source_files": sources,
}
(output / "PROVENANCE.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
