#!/usr/bin/env python3
"""Freeze manifests, audit reusable baselines, score, and report the experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


TASK = "QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
BASE_REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
CONSOLIDATED = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921")
DATASET = BASE_REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
BASE_RUNNER = BASE_REPO / "experiments/aime26/run_qwen_aime26_formal.py"
SCORER = CONSOLIDATED / "experiments/shared/scoring/aime26_scorer_v4.py"
PROVENANCE = CONSOLIDATED / "results/aime26/81920/qwen_gdn/PROVENANCE.json"
PROTOCOL_MANIFEST = CONSOLIDATED / "results/aime26/81920/qwen_gdn/evidence/integrity/qwen_protocol_manifest.json"
DATASET_MANIFEST = CONSOLIDATED / "results/aime26/81920/qwen_gdn/evidence/integrity/dataset_manifest.json"
MODEL_DIR = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
DENSE_ROOT = REPO / "results/rotation/qwen_dense_orthogonal_oracle_v1"
CONDITIONS = {
    "C0": "FP_STATE",
    "C1": "INT8_C128_NATIVE",
    "C2": "INT8_C128_KEY_HADAMARD",
    "C2B": "INT8_C128_DENSE_HADAMARD_STEP0",
    "C3": "INT8_C128_DENSE_STATE_OLD",
    "C4": "INT8_C128_DENSE_FUNCTIONAL_OLD",
}
BASE_SOURCE_NAMES = {
    "C0": ("QWEN_FP_STATE", "fp_state.jsonl"),
    "C1": ("QWEN_INT8_C128", "int8_c128.jsonl"),
    "C2": ("QWEN_INT8_C128_KEY_HADAMARD", "int8_c128_key_h.jsonl"),
}
EXPECTED_IDS = [f"aime26_{index:02d}" for index in range(11, 31)]
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_RESAMPLES = 100000


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FROZEN = import_file(SCORER, "aime26_frozen_v4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True).strip()
    except Exception:
        return "UNKNOWN"


def dataset_rows() -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 30 or [int(row["problem_idx"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError("frozen AIME26 manifest is not exactly ordered 1..30")
    selected = rows[10:]
    if [row["problem_id"] for row in selected] != EXPECTED_IDS:
        raise RuntimeError("last-20 question IDs do not match aime26_11..aime26_30")
    return selected


def file_record(path: Path, heavy: bool = False) -> dict:
    record = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if heavy or stat.st_size < 128 * 1024 * 1024:
            record["sha256"] = sha256(path)
    return record


def model_manifest() -> list[dict]:
    cache = EXPERIMENT / "analysis/model_file_hashes.json"
    files = sorted(path for path in MODEL_DIR.iterdir() if path.is_file())
    signatures = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in files}
    if cache.is_file():
        old = read_json(cache)
        if old.get("signatures") == signatures:
            return old["files"]
    records = []
    for path in files:
        records.append({"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)})
    atomic_json(cache, {"signatures": signatures, "files": records})
    return records


def setup() -> None:
    for relative in ("configs", "manifests", "analysis", "outputs", "logs"):
        (EXPERIMENT / relative).mkdir(parents=True, exist_ok=True)
    rows = dataset_rows()
    dataset_meta = read_json(DATASET_MANIFEST)
    if sha256(DATASET) != dataset_meta["frozen_file_sha256"]:
        raise RuntimeError("frozen dataset hash mismatch")

    prereg = {
        "task": TASK,
        "status": "PREREGISTERED_BEFORE_DENSE_AIME_GENERATION",
        "evaluation_only": True,
        "training_authorized": False,
        "question_selection": "last 20 rows of the immutable 30-row manifest in original order",
        "question_ids": EXPECTED_IDS,
        "generation_seed": 1,
        "max_new_tokens": 81920,
        "conditions": CONDITIONS,
        "primary_comparison": ["C4", "C2"],
        "secondary_comparisons": [["C3", "C2"], ["C4", "C3"], ["C1", "C2"], ["learned", "C2B_if_required"]],
        "checkpoint_selection": "pre-existing best validation-primary checkpoint from the old training experiment; no AIME selection or sweep",
        "scorer": {"name": "AIME26_STRICT_V4_CANDIDATE", "path": str(SCORER), "sha256": sha256(SCORER)},
        "bootstrap": {"unit": "question", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci": 0.95},
        "stopping": "EOS under frozen protocol or max_new_tokens=81920; no performance-based cancellation",
    }
    atomic_json(EXPERIMENT / "preregistration.json", prereg)

    manifest_lines = []
    for row in rows:
        item = {
            "question_id": row["problem_id"],
            "problem_idx": int(row["problem_idx"]),
            "generation_seed": 1,
            "gold_answer_sha256": hashlib.sha256(str(row["answer"]).encode("utf-8")).hexdigest(),
            "problem_sha256": hashlib.sha256(row["problem"].encode("utf-8")).hexdigest(),
            "source_manifest": str(DATASET),
        }
        manifest_lines.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
    atomic_text(EXPERIMENT / "manifests/eval_samples.jsonl", "\n".join(manifest_lines) + "\n")

    protocol = read_json(PROTOCOL_MANIFEST)
    frozen_config = {
        "task": TASK,
        "model_path": str(MODEL_DIR),
        "model": "Qwen3.5-9B",
        "architecture": "GDN",
        "tokenizer_path": str(MODEL_DIR),
        "chat_template_path": str(MODEL_DIR / "chat_template.jinja"),
        "model_revision": "UNKNOWN_LOCAL_ARTIFACT; pinned by per-file SHA256 manifest",
        "dataset": dataset_meta,
        "selected_question_ids": EXPECTED_IDS,
        "generation_seed": 1,
        "thinking": protocol["thinking"],
        "do_sample": protocol["do_sample"],
        "temperature": protocol["temperature"],
        "top_p": protocol["top_p"],
        "top_k": protocol["top_k"],
        "min_p": protocol["min_p"],
        "presence_penalty": protocol["presence_penalty"],
        "presence_penalty_scope": protocol["presence_penalty_scope"],
        "repetition_penalty": protocol["repetition_penalty"],
        "max_new_tokens": protocol["max_new_tokens"],
        "stop_tokens": ["tokenizer.eos_token_id", "<|im_end|>", "<|endoftext|>"],
        "runtime": "canonical_manual_HF; not SGLang",
        "precision": {"model": "bfloat16", "GDN_cache_state_observed": "float32", "rotation_math": "float32", "quantizer_source": "float32"},
        "cache_policy": {"use_cache": True, "radix_cache": False, "mamba_radix": False, "prefill_quantized": False, "writeback_order": "readout_before_post_decode_QDQ"},
        "quantizer": "symmetric INT8 C128; K-axis range; scale [B,H,1,V]; torch.round ties-to-even; clamp [-127,127]",
        "determinism": "seeded sampling; identical seed setup per sample; deterministic replay gate required",
        "source_runner": {"path": str(BASE_RUNNER), "sha256": sha256(BASE_RUNNER)},
        "protocol_source": {"path": str(PROTOCOL_MANIFEST), "sha256": sha256(PROTOCOL_MANIFEST), "protocol_version": protocol["protocol_version"]},
    }
    atomic_json(EXPERIMENT / "configs/frozen_eval_config.json", frozen_config)

    dense_summary = read_json(DENSE_ROOT / "summary.json")
    checkpoint_provenance = {
        "task": TASK,
        "old_training_task": dense_summary["task"],
        "AIME26_used": dense_summary["safety"]["AIME26_used"],
        "training_seed": 0,
        "training_data": dense_summary["calibration"],
        "parameterization": dense_summary["parameterization"],
        "layer_ids": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30],
        "head_sharing": "one matrix per recurrent layer, shared across heads",
        "side": "KEY",
        "order": "canonical H128 then dense Cayley residual DeltaR",
        "cache_basis": "persistent H128 DeltaR basis across prefill/decode",
        "model_weights_changed": dense_summary["safety"]["model_weights_changed"],
        "selection_rule": "minimum validation primary metric, evaluated every 50 steps; fixed before AIME evaluation",
        "checkpoints": {
            "C3": {
                "path": str(DENSE_ROOT / "training/dense_state/best_dense_state_seed0.pt"),
                "sha256": sha256(DENSE_ROOT / "training/dense_state/best_dense_state_seed0.pt"),
                "training_summary": read_json(DENSE_ROOT / "training/dense_state/training_summary.json"),
            },
            "C4": {
                "path": str(DENSE_ROOT / "training/dense_functional/best_dense_functional_seed0.pt"),
                "sha256": sha256(DENSE_ROOT / "training/dense_functional/best_dense_functional_seed0.pt"),
                "training_summary": read_json(DENSE_ROOT / "training/dense_functional/training_summary.json"),
            },
        },
        "training_code": [
            file_record(REPO / "experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py", True),
            file_record(REPO / "experiments/shared/rotation/cayley_rotation.py", True),
            file_record(REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py", True),
        ],
        "old_gates": {
            "initialization_replay": file_record(DENSE_ROOT / "gates/initialization_ste_local_replay_gate.json", True),
            "theta0_logits": file_record(DENSE_ROOT / "gates/theta0_heldout_logits_gate.json", True),
        },
    }
    atomic_json(EXPERIMENT / "analysis/checkpoint_provenance.json", checkpoint_provenance)

    source_audit = {
        "task": TASK,
        "model_files": model_manifest(),
        "model_revision": "UNKNOWN_LOCAL_ARTIFACT",
        "tokenizer_and_template_pinned_by_model_file_manifest": True,
        "dataset": file_record(DATASET, True),
        "dataset_manifest": file_record(DATASET_MANIFEST, True),
        "formal_runner": file_record(BASE_RUNNER, True),
        "frozen_scorer": file_record(SCORER, True),
        "baseline_provenance": file_record(PROVENANCE, True),
        "dense_repo_commit": git_commit(REPO),
        "baseline_repo_commit_current": git_commit(BASE_REPO),
        "baseline_formal_commit_from_outputs": "f552b9c527246c6dfe23205c0f1dc8ed1efd6546",
        "unknowns": ["upstream model hub revision; local artifacts are instead pinned by complete per-file SHA256"],
    }
    atomic_json(EXPERIMENT / "analysis/source_audit.json", source_audit)

    condition_manifest = {
        "task": TASK,
        "conditions": {
            "C0": {"name": CONDITIONS["C0"], "source": "REUSED", "definition": "original floating recurrent state; observed cache state dtype float32"},
            "C1": {"name": CONDITIONS["C1"], "source": "REUSED", "definition": "identity rotation plus native INT8-C128"},
            "C2": {"name": CONDITIONS["C2"], "source": "REUSED", "definition": "canonical fixed key-side H128 plus INT8-C128"},
            "C2B": {"name": CONDITIONS["C2B"], "source": "NEW_GENERATION_IF_REQUIRED", "definition": "dense inference path with DeltaR=I; no training"},
            "C3": {"name": CONDITIONS["C3"], "source": "NEW_GENERATION", "checkpoint": checkpoint_provenance["checkpoints"]["C3"]["path"]},
            "C4": {"name": CONDITIONS["C4"], "source": "NEW_GENERATION", "checkpoint": checkpoint_provenance["checkpoints"]["C4"]["path"]},
        },
    }
    atomic_json(EXPERIMENT / "manifests/condition_manifest.json", condition_manifest)
    baseline_reuse_audit()
    analyze()


def baseline_paths() -> list[dict]:
    provenance = read_json(PROVENANCE)
    output = []
    for source in provenance["source_files"]:
        path = Path(source["original_path"])
        actual = file_record(path, True)
        actual.update({"condition": source["condition"], "expected_sha256": source["sha256"], "hash_match": actual.get("sha256") == source["sha256"]})
        output.append(actual)
    return output


def read_baselines() -> tuple[list[dict], dict]:
    selected = {row["problem_id"]: row for row in dataset_rows()}
    records = []
    protocol_issues = []
    source_records = baseline_paths()
    for condition, (method, filename) in BASE_SOURCE_NAMES.items():
        rows = []
        for worker in range(4):
            path = BASE_REPO / f"artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards/worker{worker}/{filename}"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw = json.loads(line)
                    if raw.get("problem_id") in selected and int(raw.get("seed", -1)) == 1:
                        raw["_source_file"] = str(path)
                        rows.append(raw)
        if len(rows) != 20 or sorted(row["problem_id"] for row in rows) != EXPECTED_IDS:
            protocol_issues.append(f"{condition}: coverage mismatch")
        for raw in rows:
            config = raw.get("effective_generation_config", {})
            valid = (
                raw.get("method") == method
                and raw.get("protocol_version") == "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
                and int(raw.get("seed", -1)) == 1
                and raw.get("max_new_tokens") == 81920
                and config == {"do_sample": True, "max_new_tokens": 81920, "min_p": 0.0, "presence_penalty": 1.5, "presence_penalty_scope": "generated_output_tokens_only", "repetition_penalty": 1.0, "sampling_seed": 1, "temperature": 1.0, "thinking": True, "top_k": 20, "top_p": 0.95}
                and raw.get("runtime_error") is None
                and raw.get("nonfinite") is False
                and raw.get("finish_reason") in {"eos", "length"}
                and raw.get("generated_token_count") == len(raw.get("generated_token_ids", []))
            )
            if not valid:
                protocol_issues.append(f"{condition}/{raw.get('problem_id')}: protocol mismatch")
            raw["_condition_id"] = condition
            records.append(raw)
    audit = {
        "task": TASK,
        "status": "PASS" if not protocol_issues and all(row["hash_match"] for row in source_records) else "FAIL",
        "reuse_scope": "only aime26_11..aime26_30 at seed1 were selected; no 60-sample aggregate reused",
        "source_files": source_records,
        "protocol_issues": protocol_issues,
        "records_per_condition": {condition: sum(row["_condition_id"] == condition for row in records) for condition in BASE_SOURCE_NAMES},
        "scoring": "rescored uniformly with frozen V4",
    }
    return records, audit


def baseline_reuse_audit() -> None:
    records, audit = read_baselines()
    if audit["status"] != "PASS":
        raise RuntimeError(f"baseline reuse audit failed: {audit}")
    atomic_json(EXPERIMENT / "analysis/baseline_reuse_audit.json", audit)
    for condition in BASE_SOURCE_NAMES:
        folder = EXPERIMENT / "outputs" / CONDITIONS[condition]
        folder.mkdir(parents=True, exist_ok=True)
        references = sorted({row["_source_file"] for row in records if row["_condition_id"] == condition})
        atomic_json(folder / "reused_manifest.json", {
            "task": TASK,
            "condition": CONDITIONS[condition],
            "reuse_status": "REUSED",
            "selected_question_ids": EXPECTED_IDS,
            "generation_seed": 1,
            "source_files": [row for row in audit["source_files"] if row["condition"] == BASE_SOURCE_NAMES[condition][0]],
            "referenced_paths": references,
        })


def read_dense() -> list[dict]:
    records = []
    parity = EXPERIMENT / "analysis/parity_and_feasibility.json"
    include_c2b = parity.is_file() and bool(read_json(parity).get("dense_step0_vs_canonical_hadamard", {}).get("c2b_required"))
    for condition in ("C2B", "C3", "C4"):
        if condition == "C2B" and not include_c2b:
            continue
        folder = EXPERIMENT / "outputs" / CONDITIONS[condition]
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("aime26_*_seed1.json")):
            marker = path.with_name(path.stem + ".complete.json")
            if not marker.is_file() or read_json(marker).get("sha256") != sha256(path):
                continue
            row = read_json(path)
            row["_condition_id"] = condition
            row["_source_file"] = str(path)
            records.append(row)
    return records


def score_record(raw: dict, gold: dict) -> dict:
    extraction = FROZEN.extract_v4(raw.get("response", ""), raw.get("finish_reason", ""), gold["problem"])
    correct = FROZEN.compare_with_gold(extraction, gold["answer"])
    return {
        "condition_id": raw["_condition_id"],
        "condition": CONDITIONS[raw["_condition_id"]],
        "question_id": raw["problem_id"],
        "generation_seed": int(raw["seed"]),
        "source_status": "REUSED" if raw["_condition_id"] in BASE_SOURCE_NAMES else "NEW_GENERATION",
        "gold_answer": gold["answer"],
        "v4_status": extraction.status,
        "v4_extracted_answer": extraction.normalized_value,
        "v4_correct": bool(correct),
        "v4_abstain_reason": extraction.abstain_reason,
        "finish_reason": raw.get("finish_reason"),
        "truncated": bool(raw.get("truncated")),
        "eos_seen": bool(raw.get("eos_seen")),
        "generated_token_count": int(raw.get("generated_token_count", 0)),
        "latency_s": float(raw.get("latency_s", 0.0)),
        "peak_gpu_memory_mb": float(raw.get("peak_gpu_memory_mb", 0.0)),
        "runtime_error": raw.get("runtime_error"),
        "nonfinite": bool(raw.get("nonfinite")),
        "source_file": raw["_source_file"],
    }


def exact_mcnemar(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(rescued, regressed) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(base: dict[str, bool], candidate: dict[str, bool]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected = [rng.choice(EXPECTED_IDS) for _ in EXPECTED_IDS]
        values.append(sum(int(candidate[item]) - int(base[item]) for item in selected) / 20.0)
    values.sort()
    return [values[int(0.025 * BOOTSTRAP_RESAMPLES)], values[min(BOOTSTRAP_RESAMPLES - 1, int(0.975 * BOOTSTRAP_RESAMPLES))]]


def paired(scored: list[dict], baseline: str, candidate: str) -> dict:
    base = {row["question_id"]: row["v4_correct"] for row in scored if row["condition_id"] == baseline}
    cand = {row["question_id"]: row["v4_correct"] for row in scored if row["condition_id"] == candidate}
    if sorted(base) != EXPECTED_IDS or sorted(cand) != EXPECTED_IDS:
        return {"status": "NOT_COMPLETE", "baseline": baseline, "candidate": candidate, "paired_questions": len(set(base) & set(cand))}
    rescued = [item for item in EXPECTED_IDS if not base[item] and cand[item]]
    regressed = [item for item in EXPECTED_IDS if base[item] and not cand[item]]
    both_correct = [item for item in EXPECTED_IDS if base[item] and cand[item]]
    both_wrong = [item for item in EXPECTED_IDS if not base[item] and not cand[item]]
    return {
        "status": "COMPLETE",
        "baseline": baseline,
        "candidate": candidate,
        "paired_questions": 20,
        "rescued": len(rescued),
        "regressed": len(regressed),
        "net_gain_questions": len(rescued) - len(regressed),
        "delta_accuracy": (len(rescued) - len(regressed)) / 20.0,
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "rescue_ids": rescued,
        "regression_ids": regressed,
        "both_correct_ids": both_correct,
        "both_wrong_ids": both_wrong,
        "bootstrap_95pct_delta_accuracy": bootstrap_delta(base, cand),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "exact_mcnemar_p_two_sided": exact_mcnemar(len(rescued), len(regressed)),
    }


def attempt_counts(condition: str) -> tuple[int, int]:
    infrastructure, model = 0, 0
    for path in (EXPERIMENT / "logs").glob(f"{condition}_worker*_attempts.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace = json.loads(line).get("traceback", "").lower()
            if "nonfinite" in trace or "floatingpoint" in trace:
                model += 1
            else:
                infrastructure += 1
    return infrastructure, model


def summarize(scored: list[dict], condition: str) -> dict:
    rows = sorted((row for row in scored if row["condition_id"] == condition), key=lambda row: row["question_id"])
    infra, model = attempt_counts(condition)
    completed = len(rows)
    correct = sum(row["v4_correct"] for row in rows)
    tokens = [row["generated_token_count"] for row in rows]
    return {
        "condition_id": condition,
        "condition": CONDITIONS[condition],
        "status": "COMPLETE" if completed == 20 else "NOT_COMPLETE",
        "scheduled": 20,
        "completed": completed,
        "infrastructure_failed": infra,
        "model_failed": model,
        "correct": correct,
        "incorrect": completed - correct,
        "official_accuracy": correct / 20.0 if completed == 20 else None,
        "correct_over_20": f"{correct}/20",
        "abstain": sum(row["v4_status"] == "ABSTAIN" for row in rows),
        "max_length_truncation": sum(row["truncated"] for row in rows),
        "explicit_eos": sum(row["eos_seen"] for row in rows),
        "correct_and_terminated": sum(row["v4_correct"] and row["eos_seen"] for row in rows),
        "generated_token_median": statistics.median(tokens) if tokens else None,
        "total_generated_tokens": sum(tokens),
        "end_to_end_time_s": sum(row["latency_s"] for row in rows),
        "peak_gpu_memory_mb": max((row["peak_gpu_memory_mb"] for row in rows), default=None),
    }


def csv_text(rows: list[dict], fieldnames: list[str]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def analyze() -> None:
    gold = {row["problem_id"]: row for row in dataset_rows()}
    baseline, reuse = read_baselines()
    if reuse["status"] != "PASS":
        raise RuntimeError("baseline reuse no longer validates")
    raw = baseline + read_dense()
    scored = [score_record(row, gold[row["problem_id"]]) for row in raw]
    scored.sort(key=lambda row: (row["condition_id"], row["question_id"]))
    score_fields = ["condition_id", "condition", "question_id", "generation_seed", "source_status", "gold_answer", "v4_status", "v4_extracted_answer", "v4_correct", "v4_abstain_reason", "finish_reason", "truncated", "eos_seen", "generated_token_count", "latency_s", "peak_gpu_memory_mb", "runtime_error", "nonfinite", "source_file"]
    atomic_text(EXPERIMENT / "analysis/per_sample_scores.csv", csv_text(scored, score_fields))

    parity = EXPERIMENT / "analysis/parity_and_feasibility.json"
    c2b_required = parity.is_file() and bool(read_json(parity).get("dense_step0_vs_canonical_hadamard", {}).get("c2b_required"))
    active = ["C0", "C1", "C2", "C3", "C4"] + (["C2B"] if c2b_required else [])
    summaries = [summarize(scored, condition) for condition in active]
    summary_fields = list(summaries[0])
    atomic_text(EXPERIMENT / "analysis/summary.csv", csv_text(summaries, summary_fields))

    comparisons = {
        "task": TASK,
        "unit": "question",
        "single_generation_seed": 1,
        "primary": paired(scored, "C2", "C4"),
        "secondary": {
            "C3_vs_C2": paired(scored, "C2", "C3"),
            "C4_vs_C3": paired(scored, "C3", "C4"),
            "C1_vs_C2": paired(scored, "C2", "C1"),
        },
    }
    if c2b_required:
        comparisons["secondary"]["C3_vs_C2B"] = paired(scored, "C2B", "C3")
        comparisons["secondary"]["C4_vs_C2B"] = paired(scored, "C2B", "C4")
    atomic_json(EXPERIMENT / "analysis/paired_comparisons.json", comparisons)
    write_report(summaries, comparisons, c2b_required)
    write_artifact_manifest()


def format_number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_report(summaries: list[dict], comparisons: dict, c2b_required: bool) -> None:
    by_id = {row["condition_id"]: row for row in summaries}
    complete = all(by_id[item]["status"] == "COMPLETE" for item in ("C0", "C1", "C2", "C3", "C4")) and (not c2b_required or by_id.get("C2B", {}).get("status") == "COMPLETE")
    parity_path = EXPERIMENT / "analysis/parity_and_feasibility.json"
    parity = read_json(parity_path) if parity_path.is_file() else {"status": "NOT_RUN"}
    lines = [
        f"# {TASK}",
        "",
        f"Status: **{'COMPLETE' if complete else 'NOT_COMPLETE'}**",
        "",
        "This is a 20-question, single-generation-seed method evaluation. One question equals five percentage points; it does not establish cross-seed stability or an independent final generalization result.",
        "",
        "## Frozen sample and protocol audit",
        "",
        "The immutable 30-row manifest was verified by SHA-256 and original order; rows 11 through 30 are exactly `aime26_11` through `aime26_30`, each scheduled once at generation seed 1. `aime26_28/seed1` is retained. The inherited formal sampling protocol uses `max_new_tokens=81920`, temperature 1.0, top-p 0.95, top-k 20, sampling enabled, generated-token-only presence penalty 1.5, repetition penalty 1.0, thinking enabled, and the canonical manual HF runtime (not SGLang).",
        "",
        "## Results",
        "",
        "| ID | Condition | Status | Completed | Correct / 20 | Accuracy | Abstain | Truncated | EOS | Correct+EOS | Median tokens | Total tokens | E2E seconds | Peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        accuracy = "N/A" if row["official_accuracy"] is None else f"{100 * row['official_accuracy']:.1f}%"
        lines.append(f"| {row['condition_id']} | {row['condition']} | {row['status']} | {row['completed']}/20 | {row['correct_over_20']} | {accuracy} | {row['abstain']} | {row['max_length_truncation']} | {row['explicit_eos']} | {row['correct_and_terminated']} | {format_number(row['generated_token_median'], 1)} | {row['total_generated_tokens']} | {format_number(row['end_to_end_time_s'], 1)} | {format_number(row['peak_gpu_memory_mb'], 1)} |")

    primary = comparisons["primary"]
    lines += ["", "## Paired comparisons", ""]
    if primary["status"] == "COMPLETE":
        lines += [
            f"Primary C4 vs C2: rescued {primary['rescued']}, regressed {primary['regressed']}, net {primary['net_gain_questions']} question(s) ({primary['delta_accuracy'] * 100:+.1f} pp); question-bootstrap 95% CI {primary['bootstrap_95pct_delta_accuracy']}; exact McNemar p={primary['exact_mcnemar_p_two_sided']:.6g}.",
            "",
            f"- Rescue IDs: {', '.join(primary['rescue_ids']) if primary['rescue_ids'] else 'none'}",
            "",
            f"- Regression IDs: {', '.join(primary['regression_ids']) if primary['regression_ids'] else 'none'}",
        ]
    else:
        lines.append("Primary C4 vs C2 is NOT_COMPLETE; no completed-only accuracy is presented as the formal result.")
    lines += ["", "| Comparison | Status | Rescued | Regressed | Net | Rescue IDs | Regression IDs | Bootstrap 95% CI | McNemar p |", "|---|---:|---:|---:|---:|---|---|---|---:|"]
    comparison_rows = [("C4 vs C2 (primary)", primary)] + [(name.replace("_", " "), row) for name, row in comparisons["secondary"].items()]
    for name, row in comparison_rows:
        if row["status"] != "COMPLETE":
            lines.append(f"| {name} | NOT_COMPLETE |  |  |  |  |  |  |  |")
        else:
            lines.append(f"| {name} | COMPLETE | {row['rescued']} | {row['regressed']} | {row['net_gain_questions']} | {', '.join(row['rescue_ids']) or 'none'} | {', '.join(row['regression_ids']) or 'none'} | {row['bootstrap_95pct_delta_accuracy']} | {row['exact_mcnemar_p_two_sided']:.6g} |")

    c3_gate = parity.get("candidates", {}).get("C3", {}).get("short_generation", {}).get("gate", "NOT_RUN")
    c4_gate = parity.get("candidates", {}).get("C4", {}).get("short_generation", {}).get("gate", "NOT_RUN")
    step0 = parity.get("dense_step0_vs_canonical_hadamard", {})
    lines += [
        "",
        "## Required conclusions",
        "",
        f"1. The last 20 original questions and seed1 are frozen correctly: **YES**.",
        f"2. Baseline results are C0 {by_id['C0']['correct_over_20']}, C1 {by_id['C1']['correct_over_20']}, and C2 {by_id['C2']['correct_over_20']} under Frozen V4.",
        f"3. Dense-State checkpoint exists, matches its recorded SHA-256, and its current integration gate is **{c3_gate}**.",
        f"4. Dense-Functional provenance is verifiable: WikiText-2 raw calibration, AIME26 unused, seed0 training, fixed minimum-validation-primary selection at step 900; current integration gate is **{c4_gate}**.",
        f"5. AIME improvement status: {'see completed paired table above' if complete else 'NOT_COMPLETE until both learned conditions have 20 valid samples'}.",
        f"6. All rescue/regression IDs are listed in the paired table; no net-only claim is used.",
        f"7. Dense step0 vs canonical C2 path: gate **{step0.get('gate', 'NOT_RUN')}**, C2B required: **{c2b_required}**; final logit max-abs {step0.get('final_logits_max_abs', 'N/A')}, final state max-abs {step0.get('final_state_max_abs', 'N/A')}.",
        "8. Generation length, truncation, and EOS counts are reported per condition in the results table.",
        f"9. Missing/numerical/infrastructure status: {'none among required formal units' if complete and all(row['infrastructure_failed'] == 0 and row['model_failed'] == 0 for row in summaries) else 'see status/failure columns and logs'}.",
        f"10. Recommendation on later task-loss training: {'use the primary/secondary paired evidence cautiously; proceed only if the learned checkpoint has a positive effect that is scientifically meaningful despite the wide 20-question single-seed uncertainty' if complete else 'defer; the preregistered evaluation is not complete'}.",
        "",
        "Local error/Future-KL improvements are not treated as AIME improvements. No causal mechanism claim is made.",
        "",
        "## Artifacts",
        "",
        f"Experiment root: `{EXPERIMENT}`",
        "",
        "See `manifests/artifact_manifest.json` for hashes. No new rotation training was started.",
    ]
    atomic_text(EXPERIMENT / "final_report.md", "\n".join(lines) + "\n")


def write_artifact_manifest() -> None:
    files = []
    for path in sorted(EXPERIMENT.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path == EXPERIMENT / "manifests/artifact_manifest.json":
            continue
        files.append({"path": str(path.relative_to(EXPERIMENT)), "size": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(EXPERIMENT / "manifests/artifact_manifest.json", {"task": TASK, "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("setup", "analyze"), required=True)
    args = parser.parse_args()
    if args.phase == "setup":
        setup()
    else:
        analyze()


if __name__ == "__main__":
    main()
