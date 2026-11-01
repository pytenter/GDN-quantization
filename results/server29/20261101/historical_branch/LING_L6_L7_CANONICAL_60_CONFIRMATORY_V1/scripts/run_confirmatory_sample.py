#!/usr/bin/env python3
"""Frozen resumable client for missing canonical-60 H/L6/L7 samples."""

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


CONDITIONS = {"H", "L6", "L7"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_hash(values):
    payload = json.dumps([int(value) for value in values], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def generated_ids(payload):
    if payload.get("output_ids") is not None:
        return [int(value) for value in payload["output_ids"]], "response.output_ids"
    values = payload.get("meta_info", {}).get("output_token_logprobs")
    if values is not None:
        return [int(value[1]) for value in values], "meta_info.output_token_logprobs"
    return [], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("--sample-ids", nargs="+")
    sample_group.add_argument("--shard")
    parser.add_argument("--hardware-class", choices=("RTX4090", "RTX3090"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--rotation-file", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.experiment_root)
    source_root = Path(args.source_root)
    sys.path.insert(0, str(root / "scripts"))
    from aime26_scorer_v4 import compare_with_gold, extract_v4

    frozen = json.loads((root / "configs/frozen_eval_config.json").read_text(encoding="utf-8"))
    scorer_identity = json.loads((root / "configs/scorer_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "configs/canonical_sample_manifest.json").read_text(encoding="utf-8"))
    assignment = json.loads((root / "configs/hardware_assignment.json").read_text(encoding="utf-8"))
    stage0 = json.loads((root / "provenance/stage0_audit.json").read_text(encoding="utf-8"))

    if stage0["gates"]["PROVENANCE_GATE"] != "PASS":
        raise RuntimeError("PROVENANCE_GATE is not PASS")
    if assignment["status"] != "FROZEN_BEFORE_NEW_GENERATION":
        raise RuntimeError("hardware assignment is not frozen")
    if file_hash(root / "scripts/aime26_scorer_v4.py") != scorer_identity["expected_sha256"]:
        raise RuntimeError("SCORER_PROVENANCE_GATE failed")

    expected = {
        "context_length": 262144,
        "safety_margin": 512,
        "server_max_total_tokens": 262144,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "stop": None,
        "thinking": True,
        "do_sample": True,
        "tp_size": 1,
        "dtype": "bfloat16",
        "sglang_version": "0.5.19",
        "torch_version": "2.9.1+cu128",
        "triton_version": "3.5.1",
        "fla_version": "0.5.2",
    }
    mismatch = {key: (frozen.get(key), value) for key, value in expected.items() if frozen.get(key) != value}
    if mismatch:
        raise RuntimeError(f"frozen config mismatch: {mismatch}")

    rotation_expected = stage0["artifacts"]["H_rotation" if args.condition == "H" else f"{args.condition}_rotation"]["expected"]
    rotation_actual = file_hash(args.rotation_file)
    if rotation_actual != rotation_expected:
        raise RuntimeError(f"rotation identity mismatch for {args.condition}")

    samples = {row["sample_id"]: row for row in manifest["samples"]}
    assignments = {row["sample_id"]: row for row in assignment["assignments"]}
    sample_ids = args.sample_ids or [
        row["sample_id"]
        for row in assignment["assignments"]
        if row[f"{args.condition}_shard"] == args.shard
    ]
    if not sample_ids:
        raise RuntimeError("selected shard has no assigned samples")
    for sample_id in sample_ids:
        if sample_id not in samples or sample_id not in assignments:
            raise RuntimeError(f"unknown or non-missing sample: {sample_id}")
        row = samples[sample_id]
        assigned = assignments[sample_id]
        if row["existing_frozen_20"]:
            raise RuntimeError(f"refusing to regenerate frozen sample: {sample_id}")
        shard_key = f"{args.condition}_shard"
        expected_shard = f"{'4090' if args.hardware_class == 'RTX4090' else '3090'}_gpu{args.physical_gpu}"
        if assigned["hardware_class"] != args.hardware_class or assigned["physical_gpu"] != args.physical_gpu or assigned[shard_key] != expected_shard:
            raise RuntimeError(f"hardware assignment mismatch: {sample_id}")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for sample_id in sample_ids:
        row = samples[sample_id]
        output_path = root / "outputs" / args.condition / f"{sample_id}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if args.resume and existing.get("successful_sample") is True and existing.get("sample_id") == sample_id:
                print(f"SKIP {args.condition} {sample_id}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite {output_path}")

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(input_ids) != int(row["prompt_tokens"]) or ids_hash(input_ids) != row["input_ids_hash"]:
            raise RuntimeError(f"frozen prompt mismatch: {sample_id}")
        max_new = 262144 - len(input_ids) - 512
        if max_new != int(row["max_new_tokens"]):
            raise RuntimeError(f"frozen generation budget mismatch: {sample_id}")

        sampling = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "max_new_tokens": max_new,
            "sampling_seed": int(row["seed"]),
        }
        started = time.time()
        runtime_error = None
        payload = None
        try:
            response = requests.post(
                args.base_url + "/generate",
                json={"input_ids": input_ids, "sampling_params": sampling, "return_logprob": True},
                timeout=72 * 3600,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            runtime_error = repr(exc)

        ended = time.time()
        meta = (payload or {}).get("meta_info", {})
        token_ids, token_source = generated_ids(payload or {})
        text = (payload or {}).get("text", "")
        if not text and token_ids:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        token_count = int(meta.get("completion_tokens", len(token_ids)))
        finish_reason = meta.get("finish_reason")
        finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
        logprobs = meta.get("output_token_logprobs")
        nonfinite = None if logprobs is None else any(not math.isfinite(float(value[0])) for value in logprobs)
        provenance_valid = token_source is not None and len(token_ids) == token_count and finish_type in {"stop", "length"}
        if runtime_error is None and not provenance_valid:
            runtime_error = "TOKEN_PROVENANCE_GATE_FAILED"
        if runtime_error is None and nonfinite is not False:
            runtime_error = "NONFINITE_GATE_FAILED_OR_MISSING_LOGPROBS"
        result = extract_v4(text, termination_reason=finish_reason, problem_text=row["problem"])
        correct = compare_with_gold(result, row["gold"])
        record = {
            "experiment": "LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1",
            "sample_id": sample_id,
            "question_id": row["question_id"],
            "seed": int(row["seed"]),
            "condition": args.condition,
            "runtime_mode": "int8_r128_unified_final_r",
            "problem": row["problem"],
            "gold": row["gold"],
            "prompt_tokens": len(input_ids),
            "input_ids_hash": ids_hash(input_ids),
            "context_length": 262144,
            "safety_margin": 512,
            "max_new_tokens": max_new,
            "generated_tokens": token_count,
            "generated_token_ids": token_ids,
            "decoded_output": text,
            "finish_reason": finish_reason,
            "hit_context_limit": finish_type == "length" and token_count == max_new,
            "v4_extracted_answer": result.normalized_value,
            "v4_correct": bool(correct),
            "v4_abstain": result.status == "ABSTAIN",
            "v4_result": result.to_dict(),
            "start_time_unix": started,
            "end_time_unix": ended,
            "runtime_seconds": ended - started,
            "tokens_per_second": token_count / (ended - started) if token_count and ended > started else None,
            "runtime_effective_sampling_params": sampling,
            "thinking": True,
            "do_sample": True,
            "state_quantization": "INT8 symmetric canonical R128 [B,H,K,1] with recurrent writeback",
            "rotation_runtime": "one entry @ R_final and one legal recovery @ R_final.T",
            "final_rotation_path": str(Path(args.rotation_file).resolve()),
            "final_rotation_sha256": rotation_actual,
            "hardware_class": args.hardware_class,
            "host": assignments[sample_id]["host"],
            "physical_gpu": args.physical_gpu,
            "instance_id": args.instance_id,
            "attempt_number": 1,
            "retry_count": 0,
            "infrastructure_failures": [],
            "tp_size": 1,
            "token_ids_source": token_source,
            "token_provenance_valid": provenance_valid,
            "nonfinite": nonfinite,
            "runtime_error": runtime_error,
            "successful_sample": runtime_error is None,
            "source_root": str(source_root.resolve()),
            "frozen_config_sha256": file_hash(root / "configs/frozen_eval_config.json"),
            "hardware_assignment_sha256": file_hash(root / "configs/hardware_assignment.json"),
            "scorer_sha256": file_hash(root / "scripts/aime26_scorer_v4.py"),
        }
        atomic_json(output_path, record)
        print(f"DONE {args.condition} {sample_id} tokens={token_count} seconds={ended-started:.1f} success={runtime_error is None}", flush=True)
        if runtime_error is not None:
            raise RuntimeError(f"formal sample failed: {sample_id}: {runtime_error}")


if __name__ == "__main__":
    main()
