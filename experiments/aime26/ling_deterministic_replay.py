#!/usr/bin/env python3
"""Short SGLang deterministic replay gate using server-originated token IDs."""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer

from aime26_common import ids_sha256, load_frozen_dataset, raw_messages


def token_ids(payload):
    if payload.get("output_ids") is not None:
        return [int(value) for value in payload["output_ids"]], "response.output_ids"
    values = payload.get("meta_info", {}).get("output_token_logprobs")
    if values is not None:
        return [int(value[1]) for value in values], "meta_info.output_token_logprobs"
    return [], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="/data01/user2/models/Ling-3.0-tiny")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    requests.get(args.base_url + "/health", timeout=30).raise_for_status()
    row = load_frozen_dataset(args.dataset)[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    prompt = tokenizer.apply_chat_template(raw_messages(row["problem"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    request = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                            "max_new_tokens": args.tokens, "sampling_seed": 1},
        "return_logprob": True,
    }
    payloads = []
    for _ in range(2):
        response = requests.post(args.base_url + "/generate", json=request, timeout=3600)
        response.raise_for_status()
        payloads.append(response.json())
    first, source1 = token_ids(payloads[0])
    second, source2 = token_ids(payloads[1])
    report = {
        "gate": "PASS" if first == second and bool(first) and source1 is not None and source2 is not None else "FAIL",
        "model": "Ling-3.0-tiny", "configuration": "LING_FP_STATE", "problem_id": row["problem_id"], "seed": 1,
        "diagnostic_token_limit": args.tokens,
        "formal_generation_config": {"thinking": True, "do_sample": True, "temperature": 1.0,
                                     "top_p": 0.95, "top_k": 20, "max_new_tokens": 81920},
        "input_ids_hash": ids_sha256(input_ids), "run1_token_ids": first, "run2_token_ids": second,
        "run1_token_source": source1, "run2_token_source": source2, "exact_token_id_match": first == second,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["gate"] == "PASS" else 2)


if __name__ == "__main__":
    main()
