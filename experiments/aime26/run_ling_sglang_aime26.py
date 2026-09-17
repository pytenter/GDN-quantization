#!/usr/bin/env python3
"""Resumable AIME26 client for one isolated, patched Ling SGLang server."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from aime26_common import TASK, ids_sha256, load_frozen_dataset, raw_messages, score_aime


METHODS = {
    "fp_state": ("LING_FP_STATE", "FP", "none"),
    "int8_r128": ("LING_INT8_R128", "INT8 symmetric R128 [B,H,K,1]", "none"),
    "int8_r128_value_h": ("LING_INT8_R128_VALUE_HADAMARD", "INT8 symmetric R128 [B,H,K,1]", "Value-side normalized H128"),
}


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    response = requests.get(args.base_url + "/health", timeout=30)
    response.raise_for_status()
    rows = load_frozen_dataset(args.dataset)
    if args.stage == "smoke":
        rows = rows[:3]
    method, quantization, rotation = METHODS[args.method]
    out = Path(args.output_dir) / ("smoke" if args.stage == "smoke" else "raw") / f"{args.method}.jsonl"
    completed = set()
    if args.resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add((item["problem_id"], int(item["seed"])))
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

    jobs = [(row, seed) for row in rows for seed in (1, 2)]
    for job_index, (row, seed) in enumerate(jobs, 1):
        problem_id = row["problem_id"]
        if (problem_id, seed) in completed:
            continue
        messages = raw_messages(row["problem"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        started = time.time()
        runtime_error = None
        payload = None
        try:
            http = requests.post(
                args.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "max_new_tokens": 32768,
                        "sampling_seed": seed,
                    },
                },
                timeout=12 * 3600,
            )
            http.raise_for_status()
            payload = http.json()
        except Exception as exc:
            runtime_error = repr(exc)
        latency = time.time() - started
        meta = (payload or {}).get("meta_info", {})
        output_ids = (payload or {}).get("output_ids", [])
        text = (payload or {}).get("text", "")
        if not text and output_ids:
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
        extracted, correct = score_aime(text, row["answer"])
        finish = meta.get("finish_reason")
        finish_type = finish.get("type") if isinstance(finish, dict) else finish
        truncated = finish_type == "length"
        record = {
            "task": TASK,
            "model": "Ling-3.0-tiny",
            "architecture": "KDA",
            "runtime": "sglang",
            "method": method,
            "problem_id": problem_id,
            "seed": seed,
            "problem": row["problem"],
            "gold_answer": row["answer"],
            "input_ids_hash": ids_sha256(input_ids),
            "thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 32768,
            "state_quantization": quantization,
            "rotation": rotation,
            "response": text,
            "extracted_answer": extracted,
            "correct": correct,
            "input_tokens": len(input_ids),
            "output_tokens": meta.get("completion_tokens", len(output_ids)),
            "finish_reason": finish,
            "truncated": truncated,
            "runtime_error": runtime_error,
            "nonfinite": False,
            "latency_s": latency,
            "tokens_per_s": len(output_ids) / latency if latency and output_ids else None,
            "gpu": args.gpu,
            "git_commit": git_commit,
            "runtime_version": runtime_version,
            "sampling_seed": seed,
            "audit_jsonl": args.audit_jsonl,
        }
        append_jsonl(out, record)
        print(f"{job_index}/{len(jobs)} {method} {problem_id} seed={seed} tokens={record['output_tokens']} correct={correct}", flush=True)
        if runtime_error:
            raise RuntimeError(runtime_error)


if __name__ == "__main__":
    main()
