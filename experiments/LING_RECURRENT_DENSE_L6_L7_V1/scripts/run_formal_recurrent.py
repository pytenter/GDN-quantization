#!/usr/bin/env python3
"""Resumable atomic formal client for recurrent L6/L7 final-R conditions."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


CONTEXT_LENGTH = 262144
SAFETY_MARGIN = 512
SEED = 1
CONDITIONS = {
    "L6": "int8_r128_unified_final_r",
    "L7": "int8_r128_unified_final_r",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def ids_hash(values):
    payload = json.dumps([int(v) for v in values], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_ids(payload):
    if payload.get("output_ids") is not None:
        return [int(v) for v in payload["output_ids"]], "response.output_ids"
    values = payload.get("meta_info", {}).get("output_token_logprobs")
    if values is not None:
        return [int(v[1]) for v in values], "meta_info.output_token_logprobs"
    return [], None


def raw_messages(problem):
    """Frozen prompt wrapper used by the audited L2 formal client."""
    return [{"role": "user", "content": str(problem)}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--problem-ids", nargs="+", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--rotation-file", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    sys.path.insert(0, str(root / "scripts"))
    from aime26_scorer_v4 import compare_with_gold, extract_v4

    frozen = json.loads((root / "configs/frozen_eval_config.json").read_text(encoding="utf-8"))
    expected_constants = {
        "context_length": CONTEXT_LENGTH,
        "safety_margin": SAFETY_MARGIN,
        "generation_seed": SEED,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "thinking": True,
        "do_sample": True,
        "tp_size": 1,
    }
    mismatches = {
        key: {"frozen": frozen.get(key), "client": value}
        for key, value in expected_constants.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen evaluation config mismatch: {mismatches}")

    deadline = time.time() + 1200
    while True:
        try:
            health = requests.get(args.base_url + "/health", timeout=5)
            if health.ok:
                break
        except Exception:
            pass
        if time.time() >= deadline:
            raise RuntimeError("formal server did not become healthy")
        time.sleep(2)
    rows = [json.loads(line) for line in (root / "manifests/eval_samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["question_id"]: row for row in rows}
    if set(args.problem_ids) - set(by_id):
        raise RuntimeError("unknown problem ID")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for question_id in args.problem_ids:
        row = by_id[question_id]
        record_path = root / "outputs" / args.condition / f"{question_id}_seed1.json"
        if record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if args.resume and existing.get("successful_sample") and existing.get("question_id") == question_id:
                print(f"SKIP {question_id}", flush=True); continue
            raise RuntimeError(f"refusing to overwrite {record_path}")
        prompt = tokenizer.apply_chat_template(
            raw_messages(row["problem"]), tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(input_ids) != int(row["prompt_tokens"]) or ids_hash(input_ids) != row["input_ids_hash"]:
            raise RuntimeError(f"frozen prompt mismatch for {question_id}")
        max_new = CONTEXT_LENGTH - len(input_ids) - SAFETY_MARGIN
        if max_new != int(row["max_new_tokens"]):
            raise RuntimeError(f"frozen budget mismatch for {question_id}")
        sampling = {
            "temperature": frozen["temperature"],
            "top_p": frozen["top_p"],
            "top_k": frozen["top_k"],
            "repetition_penalty": frozen["repetition_penalty"],
            "max_new_tokens": max_new,
            "sampling_seed": SEED,
        }
        started = time.time()
        runtime_error, payload = None, None
        try:
            response = requests.post(
                args.base_url + "/generate",
                json={"input_ids": input_ids, "sampling_params": sampling, "return_logprob": True},
                timeout=72 * 3600,
            )
            response.raise_for_status(); payload = response.json()
        except Exception as exc:
            runtime_error = repr(exc)
        elapsed = time.time() - started
        meta = (payload or {}).get("meta_info", {})
        generated_ids, token_source = output_ids(payload or {})
        text = (payload or {}).get("text", "")
        if not text and generated_ids:
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_count = int(meta.get("completion_tokens", len(generated_ids)))
        finish = meta.get("finish_reason")
        finish_type = finish.get("type") if isinstance(finish, dict) else finish
        logprobs = meta.get("output_token_logprobs")
        nonfinite = None if logprobs is None else any(not math.isfinite(float(v[0])) for v in logprobs)
        provenance = token_source is not None and len(generated_ids) == generated_count and finish_type in {"stop", "length"}
        if runtime_error is None and not provenance:
            runtime_error = "TOKEN_PROVENANCE_GATE_FAILED"
        if runtime_error is None and nonfinite is not False:
            runtime_error = "NONFINITE_GATE_FAILED_OR_MISSING_LOGPROBS"
        score = extract_v4(text, termination_reason=finish, problem_text=row["problem"])
        correct = compare_with_gold(score, row["answer"])
        record = {
            "experiment": "LING_RECURRENT_DENSE_L6_L7_V1",
            "condition": args.condition,
            "runtime_mode": CONDITIONS[args.condition],
            "final_rotation_path": str(Path(args.rotation_file).resolve()),
            "final_rotation_sha256": file_hash(args.rotation_file),
            "question_id": question_id,
            "seed": SEED,
            "problem": row["problem"],
            "gold": row["answer"],
            "prompt_tokens": len(input_ids),
            "input_ids_hash": ids_hash(input_ids),
            "context_length": CONTEXT_LENGTH,
            "safety_margin": SAFETY_MARGIN,
            "max_new_tokens": max_new,
            "generated_tokens": generated_count,
            "generated_token_ids": generated_ids,
            "decoded_output": text,
            "finish_reason": finish,
            "hit_context_limit": finish_type == "length" and generated_count == max_new,
            "v4_extracted_answer": score.normalized_value,
            "v4_correct": bool(correct),
            "v4_abstain": score.status == "ABSTAIN",
            "v4_result": score.to_dict(),
            "runtime_seconds": elapsed,
            "tokens_per_second": generated_count / elapsed if elapsed and generated_count else None,
            "runtime_effective_sampling_params": sampling,
            "thinking": True,
            "do_sample": True,
            "state_quantization": "INT8 symmetric canonical R128 [B,H,K,1] with recurrent writeback",
            "rotation_runtime": "one entry @ R_final and one legal recovery @ R_final.T",
            "physical_gpu": args.physical_gpu,
            "instance_id": args.instance_id,
            "attempt_number": 1,
            "retry_count": 0,
            "infrastructure_failures": [],
            "tp_size": 1,
            "token_ids_source": token_source,
            "token_provenance_valid": provenance,
            "nonfinite": nonfinite,
            "runtime_error": runtime_error,
            "successful_sample": runtime_error is None,
        }
        atomic_json(record_path, record)
        print(f"DONE {args.condition} {question_id} tokens={generated_count} seconds={elapsed:.1f} success={runtime_error is None}", flush=True)
        if runtime_error is not None:
            raise RuntimeError(f"formal sample failed: {question_id}: {runtime_error}")


if __name__ == "__main__":
    main()
