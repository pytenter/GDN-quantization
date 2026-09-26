#!/usr/bin/env python3
"""Short, real-model deterministic replay gate for the Qwen manual runtime."""

import argparse
import json
from pathlib import Path

import torch

import run_qwen_aime26_formal as runner


def generate(model, tokenizer, item, seed, token_limit):
    runner.setup_seed(seed)
    device = next(model.parameters()).device
    _, input_ids = runner.render_input(tokenizer, item["problem"])
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    warpers = runner.build_warpers()
    stops = runner.stop_ids(tokenizer)
    tokens = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        if not bool(torch.isfinite(output.logits).all().item()):
            raise FloatingPointError("nonfinite prefill logits")
        past = output.past_key_values
        generated_token_mask = torch.zeros_like(output.logits[:, -1, :], dtype=torch.bool)
        current, _ = runner.sample_next(warpers, output.logits[:, -1, :], generated_token_mask, generator)
        generated_token_mask.scatter_(1, current, True)
        tokens.append(int(current.item()))
        while len(tokens) < token_limit and int(current.item()) not in stops:
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            if not bool(torch.isfinite(output.logits).all().item()):
                raise FloatingPointError("nonfinite decode logits")
            past = output.past_key_values
            current, _ = runner.sample_next(warpers, output.logits[:, -1, :], generated_token_mask, generator)
            generated_token_mask.scatter_(1, current, True)
            tokens.append(int(current.item()))
    return {
        "input_ids_hash": runner.ids_sha256(input_ids[0].tolist()),
        "generated_token_ids": tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    rows = runner.load_frozen_dataset(runner.DATASET)
    model, tokenizer = runner.load_model_and_tokenizer()
    first = generate(model, tokenizer, rows[0], 1, args.tokens)
    second = generate(model, tokenizer, rows[0], 1, args.tokens)
    report = {
        "gate": "PASS" if first == second and bool(first["generated_token_ids"]) else "FAIL",
        "model": "Qwen3.5-9B",
        "configuration": "QWEN_FP_STATE",
        "problem_id": rows[0]["problem_id"],
        "seed": 1,
        "diagnostic_token_limit": args.tokens,
        "formal_generation_config": {
            "thinking": True,
            "do_sample": True,
            "temperature": runner.TEMPERATURE,
            "top_p": runner.TOP_P,
            "top_k": runner.TOP_K,
            "min_p": runner.MIN_P,
            "presence_penalty": runner.PRESENCE_PENALTY,
            "repetition_penalty": runner.REPETITION_PENALTY,
            "presence_penalty_scope": "generated_output_tokens_only",
            "max_new_tokens": runner.MAX_NEW_TOKENS,
        },
        "input_ids_hash": first["input_ids_hash"],
        "run1_token_ids": first["generated_token_ids"],
        "run2_token_ids": second["generated_token_ids"],
        "exact_token_id_match": first["generated_token_ids"] == second["generated_token_ids"],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["gate"] == "PASS" else 2)


if __name__ == "__main__":
    main()
