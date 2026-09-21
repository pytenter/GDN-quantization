#!/usr/bin/env python3
"""Resumable AIME26 client for one isolated, patched Ling SGLang server."""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from aime26_common import TASK, ids_sha256, load_frozen_dataset, raw_messages
from aime26_diagnostic_scorer import score_aime_layers


METHODS = {
    "fp_state": ("LING_FP_STATE", "FP", "none"),
    "int8_r128": ("LING_INT8_R128", "INT8 symmetric R128 [B,H,K,1]", "none"),
    "int8_r128_value_h": ("LING_INT8_R128_VALUE_HADAMARD", "INT8 symmetric R128 [B,H,K,1]", "Value-side normalized H128"),
}

MAX_NEW_TOKENS = 81920
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
SEEDS = (1, 2)
PROTOCOL_VERSION = "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
FORMAL_PROTOCOL_VERSION = "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
FORMAL_WORKERS = 2
KDA_ROTATION_SEMANTICS_VERSION = "CORRECTED_PREFILL_ENDPOINT_V2"


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finish_type(finish_reason):
    return finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason


def token_ids_from_response(payload, meta):
    """Return server-originated token IDs without re-tokenizing decoded text."""
    output_ids = payload.get("output_ids")
    if output_ids is not None:
        return [int(value) for value in output_ids], "response.output_ids"
    logprobs = meta.get("output_token_logprobs")
    if logprobs is not None:
        return [int(value[1]) for value in logprobs], "meta_info.output_token_logprobs"
    return [], None


def output_logprobs_nonfinite(meta):
    logprobs = meta.get("output_token_logprobs")
    if logprobs is None:
        return None, 0
    values = [float(item[0]) for item in logprobs]
    return any(not math.isfinite(value) for value in values), len(values)


def is_valid_record(row, method_key):
    method = METHODS[method_key][0]
    identity = row.get("formal_identity", {})
    params = row.get("runtime_effective_sampling_params", {})
    return (
        row.get("protocol_version") == FORMAL_PROTOCOL_VERSION
        and row.get("model") == "Ling-3.0-tiny"
        and row.get("method") == method
        and identity.get("model") == "Ling-3.0-tiny"
        and identity.get("configuration") == method
        and identity.get("problem_id") == row.get("problem_id")
        and int(identity.get("seed", -1)) == int(row.get("seed", -2))
        and identity.get("protocol_version") == FORMAL_PROTOCOL_VERSION
        and row.get("successful_sample") is True
        and row.get("runtime_error") is None
        and row.get("nonfinite") is False
        and isinstance(row.get("response"), str)
        and bool(row.get("response"))
        and isinstance(row.get("generated_token_ids"), list)
        and bool(row.get("generated_token_ids"))
        and row.get("generated_token_count") == len(row.get("generated_token_ids", []))
        and row.get("output_tokens") == len(row.get("generated_token_ids", []))
        and row.get("finish_reason") is not None
        and finish_type(row.get("finish_reason")) in {"stop", "length"}
        and isinstance(row.get("eos_seen"), bool)
        and isinstance(row.get("truncated"), bool)
        and row.get("truncated") == (finish_type(row.get("finish_reason")) == "length")
        and (finish_type(row.get("finish_reason")) != "length" or row.get("generated_token_count") == MAX_NEW_TOKENS)
        and (finish_type(row.get("finish_reason")) != "stop" or row.get("eos_seen") is True)
        and row.get("token_provenance_valid") is True
        and row.get("max_new_tokens") == MAX_NEW_TOKENS
        and row.get("ling_effective_max_new_tokens") == MAX_NEW_TOKENS
        and params.get("max_new_tokens") == MAX_NEW_TOKENS
        and params.get("temperature") == TEMPERATURE
        and params.get("top_p") == TOP_P
        and params.get("top_k") == TOP_K
        and int(params.get("sampling_seed", -1)) == int(row.get("seed", -2))
        and row.get("thinking") is True
        and (
            method_key != "int8_r128_value_h"
            or (
                row.get("kda_rotation_semantics_version") == KDA_ROTATION_SEMANTICS_VERSION
                and row.get("redundant_prefill_endpoint_rotation") is False
            )
        )
    )


def read_valid_done(path, method_key):
    if not path.exists():
        return set()
    done = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt JSON at {path}:{line_number}") from exc
        if is_valid_record(row, method_key):
            identity = (
                row["model"], row["method"], row["problem_id"], int(row["seed"]), row["protocol_version"]
            )
            if identity in done:
                raise RuntimeError(f"duplicate valid formal identity in {path}: {identity}")
            done.add(identity)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/data01/user2/models/Ling-3.0-tiny")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--num-workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens != MAX_NEW_TOKENS:
        raise RuntimeError(
            "LING_EFFECTIVE_MAX_NEW_TOKENS gate failed: "
            f"expected {MAX_NEW_TOKENS}, got {args.max_new_tokens}"
        )
    if args.stage == "formal":
        if args.num_workers != FORMAL_WORKERS or args.worker_id not in range(FORMAL_WORKERS):
            raise RuntimeError(
                f"formal requires --num-workers {FORMAL_WORKERS} and --worker-id 0..{FORMAL_WORKERS - 1}"
            )
        if not args.resume:
            raise RuntimeError("formal execution requires --resume to prevent accidental shard overwrite")
    worker_id = 0 if args.worker_id is None else args.worker_id

    response = requests.get(args.base_url + "/health", timeout=30)
    response.raise_for_status()
    rows = load_frozen_dataset(args.dataset)
    if args.stage == "smoke":
        rows = rows[:3]
    method, quantization, rotation = METHODS[args.method]
    out = (
        Path(args.output_dir) / "smoke" / f"{args.method}.jsonl"
        if args.stage == "smoke"
        else Path(args.output_dir) / "shards" / f"worker{worker_id}" / f"{args.method}.jsonl"
    )
    completed = read_valid_done(out, args.method) if args.resume and args.stage == "formal" else set()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"
    try:
        import sglang
        runtime_version = sglang.__version__
    except Exception:
        runtime_version = "0.5.19-source"

    all_jobs = [(row, seed) for row in rows for seed in SEEDS]
    method_offset = list(METHODS).index(args.method) * len(all_jobs)
    jobs = (
        [
            (row, seed, method_offset + local_index)
            for local_index, (row, seed) in enumerate(all_jobs)
            if (method_offset + local_index) % args.num_workers == worker_id
        ]
        if args.stage == "formal"
        else [(row, seed, local_index) for local_index, (row, seed) in enumerate(all_jobs)]
    )
    for job_index, (row, seed, canonical_index) in enumerate(jobs, 1):
        problem_id = row["problem_id"]
        identity_key = ("Ling-3.0-tiny", method, problem_id, seed, FORMAL_PROTOCOL_VERSION)
        if identity_key in completed:
            continue
        messages = raw_messages(row["problem"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        started = time.time()
        runtime_error = None
        payload = None
        sampling_params = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": seed,
        }
        if sampling_params["max_new_tokens"] != MAX_NEW_TOKENS:
            raise RuntimeError("LING_RUNTIME_EFFECTIVE_CONFIG_GATE failed before request")
        try:
            http = requests.post(
                args.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                },
                timeout=12 * 3600,
            )
            http.raise_for_status()
            payload = http.json()
        except Exception as exc:
            runtime_error = repr(exc)
        latency = time.time() - started
        meta = (payload or {}).get("meta_info", {})
        output_ids, token_ids_source = token_ids_from_response(payload or {}, meta)
        text = (payload or {}).get("text", "")
        if not text and output_ids:
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
        scoring = score_aime_layers(text, row["answer"])
        extracted = scoring["strict_extracted_answer"]
        correct = scoring["strict_correct"]
        finish = meta.get("finish_reason")
        termination_type = finish_type(finish)
        generated_token_count = int(meta.get("completion_tokens", len(output_ids)))
        token_provenance_valid = (
            runtime_error is None
            and token_ids_source is not None
            and len(output_ids) == generated_token_count
            and finish is not None
        )
        if runtime_error is None and not token_provenance_valid:
            runtime_error = (
                "TOKEN_PROVENANCE_GATE failed: "
                f"source={token_ids_source}, ids={len(output_ids)}, "
                f"completion_tokens={generated_token_count}, finish_reason={finish!r}"
            )
        truncated = termination_type == "length"
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_ids = set()
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = {int(value) for value in eos_token_id}
        else:
            eos_ids = {int(eos_token_id)}
        matched = finish.get("matched") if isinstance(finish, dict) else None
        eos_seen = any(token_id in eos_ids for token_id in output_ids)
        if termination_type == "stop" and isinstance(matched, int) and matched in eos_ids:
            eos_seen = True
        nonfinite, finite_checked_tokens = output_logprobs_nonfinite(meta)
        if runtime_error is None and nonfinite is None:
            runtime_error = "NONFINITE_GATE failed: runtime returned no output token logprobs"
        successful_sample = runtime_error is None and token_provenance_valid and nonfinite is False
        record = {
            "task": TASK,
            "protocol_version": FORMAL_PROTOCOL_VERSION if args.stage == "formal" else PROTOCOL_VERSION,
            "model": "Ling-3.0-tiny",
            "architecture": "KDA",
            "runtime": "sglang",
            "method": method,
            "configuration": method,
            "problem_id": problem_id,
            "seed": seed,
            "problem": row["problem"],
            "gold_answer": row["answer"],
            "input_ids_hash": ids_sha256(input_ids),
            "thinking": True,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "runtime_effective_sampling_params": sampling_params,
            "ling_effective_max_new_tokens": sampling_params["max_new_tokens"],
            "ling_effective_max_new_tokens_gate": sampling_params["max_new_tokens"] == MAX_NEW_TOKENS,
            "state_quantization": quantization,
            "rotation": rotation,
            "kda_rotation_semantics_version": (
                KDA_ROTATION_SEMANTICS_VERSION if args.method == "int8_r128_value_h" else None
            ),
            "redundant_prefill_endpoint_rotation": False if args.method == "int8_r128_value_h" else None,
            "response": text,
            "generated_token_ids": output_ids,
            "generated_token_count": generated_token_count,
            "token_ids_source": token_ids_source,
            "token_provenance_valid": token_provenance_valid,
            "extracted_answer": extracted,
            "correct": correct,
            **scoring,
            "input_tokens": len(input_ids),
            "output_tokens": generated_token_count,
            "finish_reason": finish,
            "eos_seen": eos_seen,
            "truncated": truncated,
            "runtime_error": runtime_error,
            "nonfinite": nonfinite,
            "finite_checked_output_tokens": finite_checked_tokens,
            "successful_sample": successful_sample,
            "worker_id": worker_id,
            "num_workers": FORMAL_WORKERS if args.stage == "formal" else 1,
            "canonical_index": canonical_index,
            "formal_identity": {
                "model": "Ling-3.0-tiny",
                "configuration": method,
                "problem_id": problem_id,
                "seed": seed,
                "protocol_version": FORMAL_PROTOCOL_VERSION if args.stage == "formal" else PROTOCOL_VERSION,
            },
            "latency_s": latency,
            "tokens_per_s": len(output_ids) / latency if latency and output_ids else None,
            "gpu": args.gpu,
            "git_commit": git_commit,
            "runtime_version": runtime_version,
            "sampling_seed": seed,
            "audit_jsonl": args.audit_jsonl,
        }
        if args.stage == "formal" and not is_valid_record(record, args.method):
            append_jsonl(out.with_suffix(".attempts.jsonl"), record)
            raise RuntimeError(f"formal unit failed validation: {identity_key}")
        append_jsonl(out, record)
        print(f"{job_index}/{len(jobs)} {method} {problem_id} seed={seed} tokens={record['output_tokens']} correct={correct}", flush=True)
        if runtime_error:
            raise RuntimeError(runtime_error)


if __name__ == "__main__":
    main()
