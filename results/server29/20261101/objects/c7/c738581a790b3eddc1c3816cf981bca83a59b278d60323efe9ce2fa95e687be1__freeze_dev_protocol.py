#!/usr/bin/env python3
"""Freeze the independent MATH-500 screen/check panel before candidate metrics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
ANALYSIS = ROOT / "analysis"
SOURCE = Path("/data/zypan/results/gdn_end2end_bit_axis_screening_v1_math_subset.json")
SOURCE_SHA = "b197ab1779a65f7e7ca2425c5bb63b0982f4ffe8557c9ae3fdec4fee029da77a"
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
TRANSFORMERS = Path("/data/zypan/transformers-qwen35/src")
FORMAL_RUNNER = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
CAPTURES = [31, 127, 511, 1023, 2047]
MAX_STEPS = 2048


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    protocol_path = ANALYSIS / "independent_dev_protocol.json"
    panel_path = ANALYSIS / "independent_dev_panel.jsonl"
    if protocol_path.exists() or panel_path.exists():
        raise RuntimeError("independent dev panel already frozen")
    if sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source SHA mismatch")
    source = json.loads(SOURCE.read_text())
    records = list(source["records"])
    if source.get("n") != 30 or len(records) != 30 or len({x["problem_id"] for x in records}) != 30:
        raise RuntimeError("expected 30 unique MATH-500 subset records")
    if any(x.get("benchmark") != "MATH-500" or "aime" in x["problem_id"].lower() for x in records):
        raise RuntimeError("dataset identity/non-AIME gate failed")
    order = sorted(records, key=lambda x: text_sha(x["problem_id"]))
    chosen = order[:8]
    smoke = order[8]

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(TRANSFORMERS))
    spec = importlib.util.spec_from_file_location("formal_dev_protocol", str(FORMAL_RUNNER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import canonical runner")
    formal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = formal
    spec.loader.exec_module(formal)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    frozen_rows, metadata = [], []
    for index, row in enumerate(chosen):
        partition = "SCREEN" if index < 4 else "CHECK"
        _messages, prompt_ids = formal.render_input(tokenizer, row["problem"])
        teacher_tokens = tokenizer(row["solution"], add_special_tokens=False)["input_ids"]
        steps = min(MAX_STEPS, max(0, len(teacher_tokens) - 1))
        capture_positions = [x for x in CAPTURES if x < steps]
        if not capture_positions:
            raise RuntimeError(f"no valid capture for {row['problem_id']}")
        frozen = {
            "panel_index": index,
            "partition": partition,
            "document_id": row["problem_id"],
            "benchmark": row["benchmark"],
            "subject": row["subject"],
            "level": row["level"],
            "problem_index": row["problem_index"],
            "problem": row["problem"],
            "solution": row["solution"],
            "reference_answer": row["reference_answer"],
            "stable_order_key": text_sha(row["problem_id"]),
            "problem_sha256": text_sha(row["problem"]),
            "solution_sha256": text_sha(row["solution"]),
            "prompt_token_count": int(prompt_ids.shape[-1]),
            "teacher_solution_token_count": len(teacher_tokens),
            "recurrent_updates": steps,
            "capture_positions": capture_positions,
            "reference_token_source": "MATH-500 archived solution next token",
        }
        frozen_rows.append(frozen)
        metadata.append({k: v for k, v in frozen.items() if k not in ("problem", "solution")})
    tmp_panel = panel_path.with_name(panel_path.name + f".tmp.{os.getpid()}")
    with tmp_panel.open("w", encoding="utf-8") as handle:
        for row in frozen_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp_panel, panel_path)
    panel_sha = sha256(panel_path)
    total_updates = sum(x["recurrent_updates"] for x in frozen_rows)
    smoke_record = {
        "document_id": smoke["problem_id"],
        "stable_order_key": text_sha(smoke["problem_id"]),
        "excluded_from_final_panel": True,
    }
    protocol = {
        "status": "FROZEN_BEFORE_CANDIDATE_METRICS",
        "dataset": "MATH-500 archived deterministic 30-record subset",
        "dataset_role": "independent non-AIME persistent teacher-forced development screen",
        "source_path": str(SOURCE),
        "source_sha256": SOURCE_SHA,
        "source_task": source["task"],
        "source_subset_seed": source["subset_seed"],
        "source_declared_arrow": source.get("source_arrow"),
        "source_declared_arrow_present": Path(source.get("source_arrow", "")).is_file(),
        "traceability_note": "The self-contained 30-record JSON is hash-frozen; its originally declared Arrow cache path is no longer present on this host.",
        "training_data_exclusion": "C5/C6 training used only CALIBRATION_RAW_TEXTS TRAIN (WikiText-2 raw); this panel is MATH-500.",
        "shared_validation_exclusion": "The historical 16-document shared validation panel is WikiText-2 raw VALIDATION; this panel is MATH-500.",
        "AIME_exclusion": "All benchmark labels are MATH-500 and no problem_id contains AIME.",
        "previous_AIME_mechanism_panel_exclusion": True,
        "ordering_rule": "ascending SHA256(UTF-8 stable problem_id)",
        "selection_rule": "first 8 records after deterministic ordering",
        "split_rule": "ordered records 0-3 SCREEN; 4-7 CHECK",
        "screen_document_ids": [x["document_id"] for x in metadata[:4]],
        "check_document_ids": [x["document_id"] for x in metadata[4:]],
        "smoke_record": smoke_record,
        "smoke_result_path": str(ANALYSIS / "independent_dev_smoke.json"),
        "smoke_result_sha256": sha256(ANALYSIS / "independent_dev_smoke.json"),
        "panel_path": str(panel_path),
        "panel_sha256": panel_sha,
        "documents": metadata,
        "document_count": len(metadata),
        "screen_count": 4,
        "check_count": 4,
        "conditions": ["FP", "HADAMARD", "C5_U63", "C5_U113", "C6_U119", "C6_U125"],
        "candidate_set_rule": "the five mandatory provenance-passing candidates only; FP is the paired reference",
        "max_recurrent_updates_per_document": MAX_STEPS,
        "requested_capture_positions": CAPTURES,
        "actual_total_recurrent_updates_per_condition": total_updates,
        "actual_total_recurrent_updates_all_conditions": total_updates * 6,
        "teacher_path": "canonical chat-template problem prefill followed by archived solution tokens; no free generation",
        "persistent_semantics": "each condition owns independent KV/recurrent caches; every intermediate decode update is executed; every non-FP decode step performs canonical C128 QDQ and writes dequantized state back",
        "metric_reference": "FP condition on the identical token path; fixed reference token is the next archived solution token",
        "aggregation": "capture -> document -> equal-weight panel",
        "screen_check_discipline": "SCREEN hypothesis must be written before CHECK metrics are opened",
        "no_training": True,
        "no_free_generation": True,
        "AIME_used": False,
        "canonical_runner": str(FORMAL_RUNNER),
        "canonical_runner_sha256": sha256(FORMAL_RUNNER),
    }
    atomic_json(protocol_path, protocol)
    print(json.dumps({"status": protocol["status"], "documents": 8, "total_updates_all_conditions": protocol["actual_total_recurrent_updates_all_conditions"], "token_counts": [x["teacher_solution_token_count"] for x in metadata]}, sort_keys=True))


if __name__ == "__main__":
    main()
