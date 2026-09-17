#!/usr/bin/env python3
"""Drive a patched SGLang server through prompt and teacher-forced diagnostics."""

import argparse
import json
import os
import time
from pathlib import Path

import requests


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/data01/user2/models/Ling-3.0-tiny")
    parser.add_argument("--manual-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force-json", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    manual_dir, output_dir = Path(args.manual_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    force_path = Path(args.force_json)

    health = requests.get(args.base_url + "/health", timeout=30)
    health.raise_for_status()
    gate_a_rows, request_rows = [], []
    prompt_paths = sorted(manual_dir.glob("prompt_*/prompt.json"))
    for prompt_path in prompt_paths:
        prompt = json.loads(prompt_path.read_text())
        tok = requests.post(
            args.base_url + "/tokenize",
            json={
                "model": args.model,
                "messages": prompt["messages"],
                "chat_template_kwargs": {"enable_thinking": True},
                "add_special_tokens": False,
            },
            timeout=120,
        )
        tok.raise_for_status()
        sglang_ids = tok.json()["tokens"]
        gate_a_rows.append({
            "prompt_index": prompt["prompt_index"],
            "problem_id": prompt["problem_id"],
            "manual_count": len(prompt["input_ids"]),
            "sglang_count": len(sglang_ids),
            "input_ids_exact": sglang_ids == prompt["input_ids"],
            "eos_token_id": prompt["eos_token_id"],
            "pad_token_id": prompt["pad_token_id"],
            "messages_exact": True,
            "enable_thinking": True,
        })
    gate_a = {
        "gate": "PASS" if all(r["input_ids_exact"] for r in gate_a_rows) else "FAIL",
        "manual": "canonical HF tokenizer/chat template",
        "sglang": "SGLang /tokenize endpoint",
        "rows": gate_a_rows,
    }
    save_json(output_dir / "gate_a_prompt.json", gate_a)
    if gate_a["gate"] != "PASS":
        raise SystemExit("Gate A failed")

    for prompt_path in prompt_paths:
        prompt = json.loads(prompt_path.read_text())
        prompt_index = int(prompt["prompt_index"])
        dump_dir = output_dir / f"prompt_{prompt_index:02d}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        force_payload = {
            "request_id": f"{args.mode}-p{prompt_index}-{time.time_ns()}",
            "tokens": prompt["teacher_token_ids"],
            "dump_dir": str(dump_dir.resolve()),
        }
        save_json(force_path, force_payload)
        started = time.time()
        response = requests.post(
            args.base_url + "/generate",
            json={
                "input_ids": prompt["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": len(prompt["teacher_token_ids"]),
                    "ignore_eos": True,
                },
            },
            timeout=3600,
        )
        response.raise_for_status()
        payload = response.json()
        force_path.unlink(missing_ok=True)
        output_ids = payload.get("output_ids", [])
        request_rows.append({
            "prompt_index": prompt_index,
            "problem_id": prompt["problem_id"],
            "latency_s": time.time() - started,
            "completion_tokens": payload.get("meta_info", {}).get("completion_tokens"),
            "output_ids_exact": output_ids == prompt["teacher_token_ids"],
            "n_logits_dumps": len(list(dump_dir.glob("logits_step*.pt"))),
            "response_meta_info": payload.get("meta_info", {}),
        })
        save_json(dump_dir / "request.json", request_rows[-1])
        print(f"SGLang {args.mode} diagnostic prompt {prompt_index} complete", flush=True)
    save_json(output_dir / "requests.json", {"mode": args.mode, "requests": request_rows})


if __name__ == "__main__":
    main()
