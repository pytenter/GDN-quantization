#!/usr/bin/env python3
"""Run GDN recurrent state quantization granularity ablations."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import VALID_STATE_GRANULARITIES, quantize_cache_recurrent_states


def parse_bits(value):
    bits = []
    for part in value.split(","):
        item = part.strip().upper()
        if item in ("FP32", "32"):
            bits.append(32)
        else:
            bits.append(int(item))
    return bits


def parse_granularities(value):
    granularities = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [g for g in granularities if g not in VALID_STATE_GRANULARITIES]
    if unknown:
        raise ValueError("unknown granularities %r, expected subset of %r" % (unknown, VALID_STATE_GRANULARITIES))
    return granularities


def agreement(a, b):
    total = max(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return {
        "same_tokens": same,
        "total_tokens": total,
        "agreement": same / total if total else 1.0,
        "length_match": len(a) == len(b),
    }


def finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


def summarize_records(records):
    rel = [r["relative_error"] for r in records if finite(r.get("relative_error"))]
    abs_err = [r["quant_error_l2"] for r in records if finite(r.get("quant_error_l2"))]
    nonfinite = sum(1 for r in records if not finite(r.get("relative_error")))
    by_layer = {}
    scale_shapes = {}
    for rec in records:
        key = str(rec["layer_idx"])
        by_layer.setdefault(key, {"relative_errors": [], "nonfinite_records": 0})
        if finite(rec.get("relative_error")):
            by_layer[key]["relative_errors"].append(rec["relative_error"])
        else:
            by_layer[key]["nonfinite_records"] += 1
        scale_shapes[key] = rec.get("scale_shape")
    layer_summary = {}
    for key, data in by_layer.items():
        vals = data["relative_errors"]
        layer_summary[key] = {
            "mean_relative_error": sum(vals) / len(vals) if vals else None,
            "max_relative_error": max(vals) if vals else None,
            "finite_records": len(vals),
            "nonfinite_records": data["nonfinite_records"],
            "scale_shape": scale_shapes.get(key),
        }
    return {
        "mean_relative_error": sum(rel) / len(rel) if rel else None,
        "max_relative_error": max(rel) if rel else None,
        "mean_absolute_error": sum(abs_err) / len(abs_err) if abs_err else None,
        "max_absolute_error": max(abs_err) if abs_err else None,
        "finite_records": len(rel),
        "nonfinite_records": nonfinite,
        "layer_summary": layer_summary,
    }


def run_decode(torch, model, tokenizer, prompt, max_new_tokens, bits, granularity, layer_types):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    generated = []
    records = []
    with torch.inference_mode():
        for token_idx in range(max_new_tokens):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            step_records = quantize_cache_recurrent_states(
                past,
                bits=bits,
                layer_types=layer_types,
                granularity=granularity,
            )
            for rec in step_records:
                rec["token_index"] = token_idx
            records.extend(step_records)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(device)
            attention_mask = None
    return {
        "bits": "FP32" if bits >= 32 else bits,
        "granularity": "none" if bits >= 32 else granularity,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "state_error": summarize_records(records),
    }


def main():
    parser = build_arg_parser("Run GDN recurrent state quantization granularity ablations")
    parser.add_argument("--prompt", default="Answer briefly: What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bits", default="8,4")
    parser.add_argument("--granularities", default="per-layer,per-head,per-row,per-column")
    parser.add_argument("--output", default="state_granularity_ablation.json")
    args = parser.parse_args()

    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = list(getattr(text_cfg, "layer_types", []))

    fp32 = run_decode(torch, model, tokenizer, args.prompt, args.max_new_tokens, 32, "per-layer", layer_types)
    experiments = []
    for bits in parse_bits(args.bits):
        if bits >= 32:
            continue
        for granularity in parse_granularities(args.granularities):
            print("running bits=%s granularity=%s" % (bits, granularity), flush=True)
            exp = run_decode(torch, model, tokenizer, args.prompt, args.max_new_tokens, bits, granularity, layer_types)
            exp["token_agreement_with_fp32"] = agreement(fp32["generated_token_ids"], exp["generated_token_ids"])
            experiments.append(exp)

    result = {
        "run": timestamp_record(),
        "model_path": cfg["model_path"],
        "model_class": model.__class__.__name__,
        "state_quantization": "GDN recurrent state fake quantization only",
        "weight_quantization": "none",
        "activation_quantization": "none",
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "granularities": parse_granularities(args.granularities),
        "fp32": fp32,
        "experiments": experiments,
    }
    out_path = Path(cfg["output_root"]) / "results" / args.output
    save_json(out_path, result)
    print("Wrote %s" % out_path)
    for exp in experiments:
        err = exp["state_error"]
        agree = exp["token_agreement_with_fp32"]["agreement"]
        print(
            "bits=%s granularity=%s rel_mean=%s rel_max=%s nonfinite=%s token_agreement=%.6f"
            % (
                exp["bits"],
                exp["granularity"],
                err["mean_relative_error"],
                err["max_relative_error"],
                err["nonfinite_records"],
                agree,
            )
        )


if __name__ == "__main__":
    main()
