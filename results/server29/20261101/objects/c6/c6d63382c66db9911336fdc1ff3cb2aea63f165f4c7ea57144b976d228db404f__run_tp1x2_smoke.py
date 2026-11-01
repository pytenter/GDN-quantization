#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from aime26_common import load_frozen_dataset, raw_messages
from aime26_scorer_v4 import compare_with_gold, extract_v4


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_one(port: int, input_ids: list[int], gpu: int) -> dict:
    sampling = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 256,
        "sampling_seed": 1,
    }
    started = time.time()
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"input_ids": input_ids, "sampling_params": sampling, "return_logprob": True},
        timeout=3600,
    )
    response.raise_for_status()
    payload = response.json()
    meta = payload.get("meta_info", {})
    ids = payload.get("output_ids")
    source = "response.output_ids"
    if ids is None:
        ids = [int(value[1]) for value in meta.get("output_token_logprobs", [])]
        source = "meta_info.output_token_logprobs"
    ids = [int(value) for value in ids]
    count = int(meta.get("completion_tokens", len(ids)))
    logprobs = meta.get("output_token_logprobs")
    nonfinite = None if logprobs is None else any(not math.isfinite(float(v[0])) for v in logprobs)
    return {
        "physical_gpu": gpu,
        "port": port,
        "runtime_seconds": time.time() - started,
        "generated_tokens": count,
        "generated_token_ids": ids,
        "generated_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "text": payload.get("text", ""),
        "finish_reason": meta.get("finish_reason"),
        "token_ids_source": source,
        "token_provenance_valid": len(ids) == count,
        "nonfinite": nonfinite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    row = load_frozen_dataset(args.dataset)[0]
    prompt = tokenizer.apply_chat_template(
        raw_messages(row["problem"]), tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    jobs = [(31002, 2), (31003, 3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, port, input_ids, gpu) for port, gpu in jobs]
        results = [future.result() for future in futures]
    for result in results:
        result["decoded_output"] = result.pop("text") or tokenizer.decode(
            result["generated_token_ids"], skip_special_tokens=True
        )
        v4 = extract_v4(
            result["decoded_output"], termination_reason=result["finish_reason"], problem_text=row["problem"]
        )
        result["v4_status"] = v4.status
        result["v4_extracted_answer"] = v4.normalized_value
        result["v4_correct"] = compare_with_gold(v4, row["answer"])
    exact = results[0]["generated_token_ids"] == results[1]["generated_token_ids"]
    errors = []
    for result in results:
        if result["generated_tokens"] != 256:
            errors.append(f"GPU{result['physical_gpu']}_TOKEN_COUNT")
        if not result["token_provenance_valid"]:
            errors.append(f"GPU{result['physical_gpu']}_TOKEN_PROVENANCE")
        if result["nonfinite"] is not False:
            errors.append(f"GPU{result['physical_gpu']}_NONFINITE")
        if not result["decoded_output"]:
            errors.append(f"GPU{result['physical_gpu']}_EMPTY_OUTPUT")
    if not exact:
        errors.append("CROSS_GPU_TOKEN_MISMATCH")
    summary = {
        "trial": "LING_256K_TP1X2_SMOKE_V1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "context_length": 262144,
        "tp_size_per_instance": 1,
        "parallel_instances": 2,
        "physical_gpus": [2, 3],
        "problem_id": row["problem_id"],
        "seed": 1,
        "cross_gpu_exact_generated_ids": exact,
        "results": results,
    }
    atomic_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

