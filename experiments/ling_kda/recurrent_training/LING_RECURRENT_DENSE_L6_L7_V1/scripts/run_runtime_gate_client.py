#!/usr/bin/env python3
"""Run fixed non-AIME forced-token requests against one SGLang gate server."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


PROMPTS = (
    "Explain why the sum of the first n odd integers equals n squared.",
    "Give a concise proof that the square root of two is irrational.",
    "Describe a deterministic algorithm for reversing a singly linked list.",
)


def ids_sha256(values):
    return hashlib.sha256(",".join(str(int(v)) for v in values).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def output_ids(payload):
    if payload.get("output_ids") is not None:
        return [int(value) for value in payload["output_ids"]]
    values = payload.get("meta_info", {}).get("output_token_logprobs") or []
    return [int(value[1]) for value in values]


def wait_ready(base_url, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(base_url + "/health", timeout=5)
            if response.ok:
                return
            last = f"HTTP {response.status_code}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {last}")


def request(base_url, input_ids, max_new_tokens):
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "sampling_seed": 20260926,
            },
            "return_logprob": True,
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-json")
    parser.add_argument("--create-reference", action="store_true")
    parser.add_argument("--dump-group")
    parser.add_argument("--result-name", default="client_result.json")
    parser.add_argument("--natural-short", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args()

    if bool(args.reference_json) == bool(args.create_reference):
        raise RuntimeError("choose exactly one of --reference-json or --create-reference")
    run_dir = Path(args.run_dir).resolve()
    force_path = run_dir / "force.json"
    dump_root = run_dir / "forced_dumps"
    if args.dump_group:
        dump_root = dump_root / args.dump_group
    wait_ready(args.base_url, args.wait_seconds)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    inputs = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs.append(tokenizer(text, add_special_tokens=False).input_ids)

    reference = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
        if reference.get("prompts") != list(PROMPTS):
            raise RuntimeError("reference prompt freeze mismatch")

    rows = []
    for index, (prompt, input_ids) in enumerate(zip(PROMPTS, inputs), 1):
        natural_tokens = None
        if args.natural_short:
            natural_tokens = output_ids(request(args.base_url, input_ids, 8))
            if len(natural_tokens) != 8:
                raise RuntimeError(f"natural request {index} returned {len(natural_tokens)} tokens")
        if args.create_reference:
            natural = request(args.base_url, input_ids, 8)
            forced_tokens = output_ids(natural)
            if len(forced_tokens) != 8:
                raise RuntimeError(f"reference request {index} returned {len(forced_tokens)} tokens")
        else:
            forced_tokens = [int(v) for v in reference["rows"][index - 1]["forced_tokens"]]

        request_id = f"{args.mode}_prompt{index:02d}_{time.time_ns()}"
        dump_dir = dump_root / f"prompt{index:02d}"
        atomic_json(
            force_path,
            {
                "request_id": request_id,
                "tokens": forced_tokens,
                "dump_dir": str(dump_dir),
            },
        )
        started = time.time()
        payload = request(args.base_url, input_ids, len(forced_tokens))
        elapsed = time.time() - started
        actual = output_ids(payload)
        if actual != forced_tokens:
            raise RuntimeError(f"forced token mismatch for prompt {index}: {actual} != {forced_tokens}")
        rows.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "prompt_tokens": len(input_ids),
                "input_ids_sha256": ids_sha256(input_ids),
                "forced_tokens": forced_tokens,
                "forced_tokens_sha256": ids_sha256(forced_tokens),
                "natural_short_tokens": natural_tokens,
                "natural_short_tokens_sha256": ids_sha256(natural_tokens) if natural_tokens else None,
                "elapsed_seconds": elapsed,
                "finish_reason": payload.get("meta_info", {}).get("finish_reason"),
                "dump_dir": str(dump_dir),
            }
        )

    if force_path.exists():
        force_path.unlink()
    output = {
        "status": "PASS",
        "mode": args.mode,
        "prompts": list(PROMPTS),
        "forced_steps": 8,
        "rows": rows,
    }
    atomic_json(run_dir / args.result_name, output)
    if args.create_reference:
        atomic_json(run_dir.parent / "forced_reference.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
