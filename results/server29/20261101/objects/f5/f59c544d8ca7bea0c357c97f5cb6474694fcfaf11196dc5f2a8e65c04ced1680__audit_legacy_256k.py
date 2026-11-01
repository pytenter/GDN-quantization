#!/usr/bin/env python3
"""Freeze and audit the completed Ling/KDA 256K AIME26 baseline panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TASK = "LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1"
EXPECTED_IDS = tuple(f"aime26_{index:02d}" for index in range(1, 31))
EVAL_IDS = EXPECTED_IDS[10:]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def condition_sources(root: Path):
    return {
        "L0_FP_STATE": [
            root
            / "artifacts/ling_256k_length_sensitivity_tp1x2_v1/records/formal256k/fp_state"
        ],
        "L1_INT8_R128_NATIVE": [
            root
            / "artifacts/ling_256k_length_sensitivity_tp1x2_v1/records/formal256k/int8_r128"
        ],
        "L2_INT8_R128_VALUE_HADAMARD_FAST": [
            root
            / "artifacts/ling_256k_length_sensitivity_gpu4_helper_migrated_v1/"
            "migration_source/4090_completed_records/formal256k/int8_r128_value_h",
            root
            / "artifacts/ling_256k_length_sensitivity_hadamard_coordinated_v1/"
            "records/formal256k/int8_r128_value_h",
        ],
    }


def locate_record(paths, problem_id: str) -> tuple[Path, list[Path]]:
    matches = [path / f"{problem_id}_seed1.json" for path in paths]
    matches = [path for path in matches if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous legacy source for {problem_id}: {matches}")
    return (matches[0] if matches else Path()), matches


def audit_record(condition: str, problem_id: str, paths, expected_row) -> tuple[dict, dict | None]:
    path, matches = locate_record(paths, problem_id)
    base = {
        "condition": condition,
        "question_id": problem_id,
        "seed": 1,
        "source_host": "nlpg-SYS-4029GP-TRT",
        "source_path": str(path) if matches else None,
        "raw_output_present": False,
        "token_metadata_present": False,
        "score_present": False,
        "finish_reason_present": False,
        "config_hash": None,
        "output_hash": None,
        "source_file_hash": None,
        "status": "MISSING_EXPECTED",
    }
    if not matches:
        return base, None
    try:
        record = read_json(path)
    except Exception as error:
        base["status"] = "PRESENT_INCOMPATIBLE"
        base["error"] = repr(error)
        return base, None

    output = record.get("decoded_output")
    generated_ids = record.get("generated_token_ids")
    generated_tokens = record.get("generated_tokens")
    base.update(
        {
            "raw_output_present": isinstance(output, str),
            "token_metadata_present": isinstance(generated_ids, list)
            and isinstance(generated_tokens, int)
            and len(generated_ids) == generated_tokens,
            "score_present": isinstance(record.get("v4_correct"), bool)
            and isinstance(record.get("v4_result"), dict),
            "finish_reason_present": record.get("finish_reason") is not None,
            "config_hash": record.get("experiment_config_hash"),
            "output_hash": sha256_bytes(output.encode("utf-8"))
            if isinstance(output, str)
            else None,
            "source_file_hash": sha256_file(path),
            "generated_tokens": generated_tokens,
            "prompt_tokens": record.get("prompt_tokens"),
            "max_new_tokens_effective": record.get("max_new_tokens_effective"),
            "finish_reason": record.get("finish_reason"),
            "v4_correct": record.get("v4_correct"),
            "v4_abstain": record.get("v4_abstain"),
            "successful_sample": record.get("successful_sample"),
            "runtime_error": record.get("runtime_error"),
        }
    )
    identity_valid = (
        record.get("problem_id") == problem_id
        and record.get("seed") == 1
        and record.get("problem") == expected_row.get("problem")
        and str(record.get("gold")) == str(expected_row.get("answer"))
    )
    protocol_valid = (
        record.get("max_context_length") == 262144
        and record.get("safety_margin") == 512
        and record.get("do_sample") is True
        and record.get("thinking") is True
        and record.get("runtime_effective_sampling_params", {}).get("sampling_seed") == 1
        and record.get("runtime_effective_sampling_params", {}).get("temperature") == 1.0
        and record.get("runtime_effective_sampling_params", {}).get("top_p") == 0.95
        and record.get("runtime_effective_sampling_params", {}).get("top_k") == 20
        and record.get("max_new_tokens_effective")
        == 262144 - int(record.get("prompt_tokens", -1)) - 512
    )
    required = all(
        base[key]
        for key in (
            "raw_output_present",
            "token_metadata_present",
            "score_present",
            "finish_reason_present",
        )
    )
    if required and identity_valid and protocol_valid and record.get("successful_sample"):
        base["status"] = "PRESENT_VALID"
    elif not base["config_hash"]:
        base["status"] = "CONFIG_UNKNOWN"
    else:
        base["status"] = "PRESENT_INCOMPATIBLE"
    base["identity_valid"] = identity_valid
    base["protocol_valid"] = protocol_valid
    return base, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()

    root = args.legacy_root.resolve()
    output = args.output_root.resolve()
    dataset_path = root / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
    source_manifest_path = args.source_manifest or (
        root / "experiments/LING_256K_LONG_HORIZON_V1/manifest.json"
    )
    dataset = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = tuple(row["problem_id"] for row in dataset)
    if ids != EXPECTED_IDS:
        raise RuntimeError(f"frozen dataset order mismatch: {ids}")
    source_manifest = read_json(source_manifest_path)

    rows = []
    records = {}
    for condition, paths in condition_sources(root).items():
        for expected in dataset:
            audit, record = audit_record(
                condition, expected["problem_id"], paths, expected
            )
            rows.append(audit)
            if record is not None:
                records[(condition, expected["problem_id"])] = record

    csv_path = output / "analysis/legacy_256k_completeness.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["status"] for row in rows)
    by_condition = {}
    for condition in condition_sources(root):
        selected = [row for row in rows if row["condition"] == condition]
        last20 = [row for row in selected if row["question_id"] in EVAL_IDS]
        by_condition[condition] = {
            "all_30": dict(Counter(row["status"] for row in selected)),
            "last_20": dict(Counter(row["status"] for row in last20)),
            "last_20_missing": [
                row["question_id"]
                for row in last20
                if row["status"] != "PRESENT_VALID"
            ],
        }
    audit = {
        "task": TASK,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_experiment": "LING_256K_LONG_HORIZON_V1",
        "source_host": "nlpg-SYS-4029GP-TRT",
        "source_root": str(root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "status_counts": dict(status_counts),
        "by_condition": by_condition,
        "protocol_conclusion": {
            "generation_seed": 1,
            "context_length": 262144,
            "safety_margin": 512,
            "max_new_tokens_definition": "262144 - prompt_tokens - 512",
            "max_new_tokens_is_per_sample": True,
            "requested_label_256k_is_total_context": True,
        },
        "rows": rows,
    }
    write_json(output / "analysis/legacy_256k_audit.json", audit)

    eval_rows = []
    for ordinal, expected in enumerate(dataset[10:], 11):
        per_condition = [records[(condition, expected["problem_id"])] for condition in condition_sources(root)]
        prompt_tokens = {record["prompt_tokens"] for record in per_condition}
        input_hashes = {record["input_ids_hash"] for record in per_condition}
        budgets = {record["max_new_tokens_effective"] for record in per_condition}
        if len(prompt_tokens) != 1 or len(input_hashes) != 1 or len(budgets) != 1:
            raise RuntimeError(f"cross-condition prompt/protocol mismatch for {expected['problem_id']}")
        eval_rows.append(
            {
                "ordinal_in_original_manifest": ordinal,
                "question_id": expected["problem_id"],
                "problem_idx": int(expected["problem_idx"]),
                "problem": expected["problem"],
                "answer": expected["answer"],
                "generation_seed": 1,
                "prompt_tokens": next(iter(prompt_tokens)),
                "input_ids_hash": next(iter(input_hashes)),
                "max_new_tokens": next(iter(budgets)),
                "context_length": 262144,
                "safety_margin": 512,
                "source_dataset": str(dataset_path),
                "source_dataset_sha256": sha256_file(dataset_path),
            }
        )
    write_jsonl(output / "manifests/eval_samples.jsonl", eval_rows)

    generation = source_manifest["generation_config"]
    tp_config = generation["tp1x2_config"]
    frozen_config = {
        "task": TASK,
        "frozen_before_new_AIME_generation": True,
        "question_ids": list(EVAL_IDS),
        "question_order_source": str(dataset_path),
        "generation_seed": 1,
        "analysis_rng_seed": 20260926,
        "model": "Ling-3.0-tiny",
        "model_revision": tp_config["model_revision"],
        "model_config_sha256": tp_config["model_sha256_config"],
        "model_index_sha256": tp_config["model_sha256_index"],
        "tokenizer_policy": "same local Ling-3.0-tiny tokenizer and chat template as legacy records",
        "thinking": True,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "repetition_penalty_source": "legacy SGLang default, made explicit for new generation",
        "eos_token_id": 156895,
        "pad_token_id": 156892,
        "stop": None,
        "context_length": 262144,
        "max_total_tokens": 262144,
        "safety_margin": 512,
        "max_new_tokens_definition": "context_length - prompt_tokens - safety_margin",
        "per_sample_max_new_tokens": {
            row["question_id"]: row["max_new_tokens"] for row in eval_rows
        },
        "dtype": "bfloat16",
        "state_rotation_compute_dtype": "float32",
        "state_storage_policy": "legacy condition-specific FP or fake INT8-R128 QDQ values in SGLang FP cache",
        "tp_size": 1,
        "parallel_instances": 2,
        "max_running_requests": 1,
        "batch_size": 1,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "uses_mamba_radix_cache": False,
        "attention_backend": "triton",
        "linear_attention_backend": "triton",
        "moe_runner_backend": "triton",
        "sampling_backend": "pytorch",
        "enable_deterministic_inference": True,
        "sglang_version": tp_config["sglang_version"],
        "sglang_source_commit": tp_config["sglang_source_commit"],
        "torch_version": tp_config["torch_version"],
        "triton_version": tp_config["triton_version"],
        "fla_version": tp_config["fla_version"],
        "scorer": tp_config["scorer"],
        "scorer_sha256": tp_config["scorer_sha256"],
        "kda_rotation_semantics_version": "CORRECTED_PREFILL_ENDPOINT_V2",
        "redundant_prefill_endpoint_rotation": False,
    }
    write_json(output / "configs/frozen_eval_config.json", frozen_config)

    conditions = [
        ("L0_FP_STATE", "fp_state", None),
        ("L1_INT8_R128_NATIVE", "int8_r128", None),
        ("L2_INT8_R128_VALUE_HADAMARD_FAST", "int8_r128_value_h", None),
        ("L3_INT8_R128_DENSE_HADAMARD_STEP0", "int8_r128_dense_h_step0", None),
        ("L4_INT8_R128_DENSE_STATE_OLD", "int8_r128_dense_state", "DENSE_STATE"),
        ("L5_INT8_R128_DENSE_FUNCTIONAL_OLD", "int8_r128_dense_functional", "DENSE_FUNCTIONAL"),
    ]
    write_json(
        output / "manifests/condition_manifest.json",
        {
            "task": TASK,
            "conditions": [
                {
                    "condition": condition,
                    "runtime_mode": mode,
                    "checkpoint_objective": objective,
                    "formal_samples": 20,
                    "generation_seed": 1,
                }
                for condition, mode, objective in conditions
            ],
        },
    )
    write_json(
        output / "preregistration.json",
        {
            "task": TASK,
            "frozen_before_new_AIME_generation": True,
            "panel": "AIME26 original frozen manifest positions 11..30",
            "question_ids": list(EVAL_IDS),
            "generation_seed": 1,
            "conditions": [condition for condition, _, _ in conditions],
            "primary_comparisons": [
                ["L4_INT8_R128_DENSE_STATE_OLD", "L3_INT8_R128_DENSE_HADAMARD_STEP0"],
                ["L5_INT8_R128_DENSE_FUNCTIONAL_OLD", "L3_INT8_R128_DENSE_HADAMARD_STEP0"],
            ],
            "primary_metric": "Frozen V4 correct / 20",
            "paired_bootstrap": {"resamples": 10000, "unit": "question", "rng_seed": 20260926},
            "multiple_testing": "Holm correction across the two primary comparisons",
            "checkpoint_selection": "pre-existing frozen best validation checkpoints; no AIME selection",
            "new_rotation_training": False,
            "model_weights_modified": False,
            "quantizer_semantics_modified": False,
            "scorer_modified": False,
        },
    )
    print(json.dumps({"status": "PASS", "rows": len(rows), "eval_samples": len(eval_rows), "counts": dict(status_counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
