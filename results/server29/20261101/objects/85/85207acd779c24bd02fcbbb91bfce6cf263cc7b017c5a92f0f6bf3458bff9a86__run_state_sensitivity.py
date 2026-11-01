#!/usr/bin/env python3
"""Run a small FP32/INT state fake-quant decode sensitivity experiment."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import quantize_cache_recurrent_states


def parse_bits(value):
    out = []
    for part in value.split(","):
        part = part.strip().upper()
        if part in ("FP32", "32"):
            out.append(32)
        else:
            out.append(int(part))
    return out


def run_decode(torch, model, tokenizer, cfg, prompt, max_new_tokens, bits):
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = list(getattr(text_cfg, "layer_types", []))
    first_device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(first_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(first_device)
    past = None
    generated = []
    step_records = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for token_idx in range(max_new_tokens + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            quant_records = quantize_cache_recurrent_states(past, bits=bits, layer_types=layer_types)
            for rec in quant_records:
                rec["token_index"] = token_idx
            step_records.extend(quant_records)
            if token_idx == max_new_tokens:
                break
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(first_device)
            attention_mask = None
    rel = [r["relative_error"] for r in step_records if r.get("relative_error") is not None]
    abs_err = [r["quant_error_l2"] for r in step_records if r.get("quant_error_l2") is not None]
    return {
        "bits": "FP32" if bits >= 32 else bits,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "num_state_records": len(step_records),
        "relative_error_mean": sum(rel) / len(rel) if rel else None,
        "relative_error_max": max(rel) if rel else None,
        "absolute_error_mean": sum(abs_err) / len(abs_err) if abs_err else None,
        "absolute_error_max": max(abs_err) if abs_err else None,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "records": step_records,
    }


def main():
    parser = build_arg_parser("Run GDN recurrent state fake quantization sensitivity")
    parser.add_argument("--prompt", default="State quantization test:")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--bits", default="FP32,8,4")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    experiments = []
    for bits in parse_bits(args.bits):
        experiments.append(run_decode(torch, model, tokenizer, cfg, args.prompt, args.max_new_tokens, bits))
    fp32_text = next((e["generated_text"] for e in experiments if e["bits"] == "FP32"), None)
    for exp in experiments:
        exp["matches_fp32_generation"] = None if fp32_text is None else exp["generated_text"] == fp32_text
    result = {
        "run": timestamp_record(),
        "model_path": cfg["model_path"],
        "model_class": model.__class__.__name__,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "experiments": experiments,
    }
    save_json(Path(cfg["output_root"]) / "results" / "state_sensitivity.json", result)
    print("Wrote results/state_sensitivity.json")
    for exp in experiments:
        print("bits=%s match_fp32=%s rel_mean=%s rel_max=%s text=%r" % (
            exp["bits"], exp["matches_fp32_generation"], exp["relative_error_mean"], exp["relative_error_max"], exp["generated_text"]
        ))


if __name__ == "__main__":
    main()
