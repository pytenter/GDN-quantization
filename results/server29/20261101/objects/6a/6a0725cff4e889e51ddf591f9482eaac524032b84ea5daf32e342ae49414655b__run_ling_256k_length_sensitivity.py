#!/usr/bin/env python3
"""Resumable client for LING_256K_LENGTH_SENSITIVITY_V1.

Every completed sample is committed with os.replace so a disconnect cannot
leave a partially written record.  The formal 81920 artifacts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from aime26_common import ids_sha256, load_frozen_dataset, raw_messages
from aime26_scorer_v4 import compare_with_gold, extract_v4


MODEL_NAME = "Ling-3.0-tiny"
CONTEXT_LENGTH = 262144
SAFETY_MARGIN = 512
SEED = 1
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
KDA_ROTATION_SEMANTICS_VERSION = "CORRECTED_PREFILL_ENDPOINT_V2"
METHODS = {
    "fp_state": ("LING_FP_STATE", "FP", "none"),
    "int8_r128": ("LING_INT8_R128", "INT8 symmetric canonical R128 [B,H,K,1]", "none"),
    "int8_r128_value_h": (
        "LING_INT8_R128_VALUE_HADAMARD",
        "INT8 symmetric canonical R128 [B,H,K,1]",
        "Value-side normalized H128",
    ),
}
STAGES = {
    "smoke256": {"limit": 256, "default_ids": ("aime26_01",)},
    "parity81920": {
        "limit": 81920,
        "default_ids": tuple(f"aime26_{index:02d}" for index in range(1, 7)),
    },
    "baseline81920": {"limit": 81920, "default_ids": None},
    "formal256k": {"limit": None, "default_ids": None},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def output_ids(payload, meta):
    if payload.get("output_ids") is not None:
        return [int(value) for value in payload["output_ids"]], "response.output_ids"
    values = meta.get("output_token_logprobs")
    if values is not None:
        return [int(value[1]) for value in values], "meta_info.output_token_logprobs"
    return [], None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model", default="/data/zypan/models/Ling-3.0-tiny")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--problem-ids", nargs="*")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--parallel-instance-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.stage != "smoke256" and not args.resume:
        raise RuntimeError("non-smoke execution requires --resume")
    config_path = Path(args.config_json)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    if config["context_length"] != CONTEXT_LENGTH or config["seed"] != SEED:
        raise RuntimeError("experiment configuration gate failed")
    if args.physical_gpu is not None:
        allowed_gpus = set(config.get("allowed_physical_gpus", []))
        if args.physical_gpu not in allowed_gpus:
            raise RuntimeError("physical GPU is not permitted by experiment configuration")
        if config.get("tp_size") != 1 or config.get("gpu_count_assertion") != 1:
            raise RuntimeError("TP=1 runtime metadata does not match experiment configuration")

    health = requests.get(args.base_url + "/health", timeout=30)
    health.raise_for_status()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    rows = load_frozen_dataset(args.dataset)
    requested = args.problem_ids or STAGES[args.stage]["default_ids"]
    if requested:
        requested = set(requested)
        rows = [row for row in rows if row["problem_id"] in requested]
        if {row["problem_id"] for row in rows} != requested:
            raise RuntimeError("requested problem IDs are not all in the frozen dataset")

    method, quantization, rotation = METHODS[args.method]
    records = Path(args.output_dir) / "records" / args.stage / args.method
    for index, row in enumerate(rows, 1):
        record_path = records / f"{row['problem_id']}_seed{SEED}.json"
        if args.resume and record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get("successful_sample") and existing.get("experiment_config_hash") == config_hash:
                print(f"SKIP {row['problem_id']} valid completed record", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite invalid/mismatched record: {record_path}")

        prompt = tokenizer.apply_chat_template(
            raw_messages(row["problem"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        available = CONTEXT_LENGTH - len(input_token_ids) - SAFETY_MARGIN
        if available <= 0:
            raise RuntimeError(f"no generation budget for {row['problem_id']}")
        stage_limit = STAGES[args.stage]["limit"]
        effective_limit = available if stage_limit is None else min(stage_limit, available)
        sampling = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": effective_limit,
            "sampling_seed": SEED,
        }
        started = time.time()
        runtime_error = None
        payload = None
        try:
            response = requests.post(
                args.base_url + "/generate",
                json={
                    "input_ids": input_token_ids,
                    "sampling_params": sampling,
                    "return_logprob": True,
                },
                timeout=36 * 3600,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # persisted below for forensic reproducibility
            runtime_error = repr(exc)
        elapsed = time.time() - started
        meta = (payload or {}).get("meta_info", {})
        generated_ids, token_source = output_ids(payload or {}, meta)
        text = (payload or {}).get("text", "")
        if not text and generated_ids:
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        finish = meta.get("finish_reason")
        finish_kind = finish_type(finish)
        generated_count = int(meta.get("completion_tokens", len(generated_ids)))
        eos_id = tokenizer.eos_token_id
        eos_ids = set(eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id]) if eos_id is not None else set()
        matched = finish.get("matched") if isinstance(finish, dict) else None
        eos_seen = any(token in eos_ids for token in generated_ids) or (
            finish_kind == "stop" and isinstance(matched, int) and matched in eos_ids
        )
        logprobs = meta.get("output_token_logprobs")
        nonfinite = None if logprobs is None else any(
            not math.isfinite(float(value[0])) for value in logprobs
        )
        token_provenance_valid = (
            runtime_error is None
            and token_source is not None
            and len(generated_ids) == generated_count
            and finish_kind in {"stop", "length"}
        )
        if runtime_error is None and not token_provenance_valid:
            runtime_error = "TOKEN_PROVENANCE_GATE_FAILED"
        if runtime_error is None and nonfinite is not False:
            runtime_error = "NONFINITE_GATE_FAILED_OR_MISSING_LOGPROBS"

        v4 = extract_v4(text, termination_reason=finish, problem_text=row["problem"])
        v4_correct = compare_with_gold(v4, row["answer"])
        record = {
            "experiment": config["experiment"],
            "protocol_role": args.stage,
            "formal_protocol": "81920 / 2 seeds (unchanged external baseline)",
            "length_sensitivity_protocol": "262144 total context / 1 seed",
            "model": MODEL_NAME,
            "architecture": "KDA",
            "condition": args.method,
            "method": method,
            "problem_id": row["problem_id"],
            "seed": SEED,
            "problem": row["problem"],
            "gold": row["answer"],
            "prompt_tokens": len(input_token_ids),
            "input_ids_hash": ids_sha256(input_token_ids),
            "max_context_length": CONTEXT_LENGTH,
            "safety_margin": SAFETY_MARGIN,
            "max_new_tokens_effective": effective_limit,
            "generated_tokens": generated_count,
            "generated_token_ids": generated_ids,
            "decoded_output": text,
            "finish_reason": finish,
            "hit_context_limit": finish_kind == "length" and generated_count == effective_limit,
            "eos_generated": bool(eos_seen),
            "v4_extracted_answer": v4.normalized_value,
            "v4_correct": bool(v4_correct),
            "v4_abstain": v4.status == "ABSTAIN",
            "v4_result": v4.to_dict(),
            "runtime_seconds": elapsed,
            "tokens_per_second": generated_count / elapsed if elapsed and generated_count else None,
            "gpu_ids": (
                [args.physical_gpu]
                if args.physical_gpu is not None
                else config["allowed_physical_gpus"]
            ),
            "visible_gpu_count_required": config["gpu_count_assertion"],
            "tp_size": config["tp_size"],
            "execution_layout": config.get("execution_layout", "TP2_SINGLE_INSTANCE"),
            "parallel_instance_id": args.parallel_instance_id,
            "thinking": True,
            "do_sample": True,
            "runtime_effective_sampling_params": sampling,
            "state_quantization": quantization,
            "rotation": rotation,
            "kda_rotation_semantics_version": (
                KDA_ROTATION_SEMANTICS_VERSION if args.method == "int8_r128_value_h" else None
            ),
            "redundant_prefill_endpoint_rotation": (
                False if args.method == "int8_r128_value_h" else None
            ),
            "token_ids_source": token_source,
            "token_provenance_valid": token_provenance_valid,
            "nonfinite": nonfinite,
            "runtime_error": runtime_error,
            "successful_sample": runtime_error is None,
            "experiment_config_hash": config_hash,
            "scorer_sha256": config["scorer_sha256"],
        }
        atomic_json(record_path, record)
        print(
            f"{index}/{len(rows)} {args.method} {row['problem_id']} "
            f"tokens={generated_count} finish={finish_kind} v4={v4.status} correct={v4_correct}",
            flush=True,
        )
        if runtime_error:
            raise RuntimeError(runtime_error)


if __name__ == "__main__":
    main()
