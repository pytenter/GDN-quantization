#!/usr/bin/env python3
"""Single-prompt GDN recurrent state granularity ablation."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import (
    STATE_GRANULARITY_ALIASES,
    normalize_granularity,
    quantize_cache_recurrent_states,
)


def parse_bits(value):
    bits = []
    for part in value.split(","):
        item = part.strip().upper()
        if item in ("FP32", "32"):
            bits.append(32)
        else:
            bit = int(item)
            if bit not in (8, 4):
                raise ValueError("supported quantized bits are 8 and 4, got %s" % bit)
            bits.append(bit)
    return bits


def parse_granularities(value):
    granularities = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        normalize_granularity(item)
        granularities.append(item)
    return granularities


def is_finite_number(value):
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def token_agreement(fp32_tokens, quant_tokens):
    total = max(len(fp32_tokens), len(quant_tokens))
    same = sum(1 for a, b in zip(fp32_tokens, quant_tokens) if a == b)
    return {
        "same_tokens": same,
        "total_tokens": total,
        "agreement": same / total if total else 1.0,
        "length_match": len(fp32_tokens) == len(quant_tokens),
    }


def summarize_layer_records(records):
    by_layer = {}
    total_nan = 0
    total_inf = 0
    shape_mismatch = 0
    dtype_mismatch = 0
    for rec in records:
        key = str(rec["layer_idx"])
        item = by_layer.setdefault(key, {
            "relative_errors": [],
            "state_norms": [],
            "nan_records": 0,
            "inf_records": 0,
            "shape_mismatch_records": 0,
            "dtype_mismatch_records": 0,
            "scale_shape": rec.get("scale_shape", []),
            "broadcast_scale_shape": rec.get("broadcast_scale_shape", []),
            "state_shape": rec.get("state", {}).get("shape"),
            "state_dtype": rec.get("state", {}).get("dtype"),
        })
        rel = rec.get("relative_error")
        norm = rec.get("state_norm_l2")
        if is_finite_number(rel):
            item["relative_errors"].append(rel)
        elif rel != rel:
            item["nan_records"] += 1
            total_nan += 1
        else:
            item["inf_records"] += 1
            total_inf += 1
        if is_finite_number(norm):
            item["state_norms"].append(norm)
        if not rec.get("output_shape_unchanged", True):
            item["shape_mismatch_records"] += 1
            shape_mismatch += 1
        if not rec.get("dtype_preserved", True):
            item["dtype_mismatch_records"] += 1
            dtype_mismatch += 1

    layers = {}
    all_rel = []
    for key, item in by_layer.items():
        rels = item.pop("relative_errors")
        norms = item.pop("state_norms")
        all_rel.extend(rels)
        layers[key] = {
            **item,
            "mean_relative_error": sum(rels) / len(rels) if rels else None,
            "max_relative_error": max(rels) if rels else None,
            "mean_state_norm": sum(norms) / len(norms) if norms else None,
            "records": len(rels) + item["nan_records"] + item["inf_records"],
            "finite_records": len(rels),
        }
    return {
        "mean_relative_error": sum(all_rel) / len(all_rel) if all_rel else None,
        "max_relative_error": max(all_rel) if all_rel else None,
        "finite_records": len(all_rel),
        "nan_records": total_nan,
        "inf_records": total_inf,
        "shape_mismatch_records": shape_mismatch,
        "dtype_mismatch_records": dtype_mismatch,
        "layers": layers,
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
                rec["output_shape_unchanged"] = rec.get("state", {}).get("shape") == [1, 32, 128, 128]
                rec["dtype_preserved"] = rec.get("state", {}).get("dtype") == "torch.float32"
            records.extend(step_records)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(device)
            attention_mask = None

    return {
        "bits": "FP32" if bits >= 32 else bits,
        "granularity": "none" if bits >= 32 else normalize_granularity(granularity),
        "granularity_arg": "none" if bits >= 32 else granularity,
        "generated_tokens": generated,
        "output_text": tokenizer.decode(generated, skip_special_tokens=True),
        "state_error": summarize_layer_records(records),
    }


def main():
    parser = build_arg_parser("Run GDN recurrent state quantization granularity ablation")
    parser.add_argument("--bits", default="8,4")
    parser.add_argument("--granularity", default="layer,head,row,column")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="Answer briefly: What is the capital of France?")
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

    granularities = parse_granularities(args.granularity)
    bits_list = parse_bits(args.bits)
    fp32 = run_decode(torch, model, tokenizer, args.prompt, args.max_new_tokens, 32, "layer", layer_types)
    experiments = []
    for bits in bits_list:
        for granularity in granularities:
            print("running bits=%s granularity=%s" % (bits, granularity), flush=True)
            exp = run_decode(torch, model, tokenizer, args.prompt, args.max_new_tokens, bits, granularity, layer_types)
            exp["token_agreement_with_fp32"] = token_agreement(fp32["generated_tokens"], exp["generated_tokens"])
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
        "requested_bits": bits_list,
        "requested_granularity": granularities,
        "granularity_aliases": STATE_GRANULARITY_ALIASES,
        "expected_state_shape": [1, 32, 128, 128],
        "expected_state_dtype": "torch.float32",
        "fp32": fp32,
        "experiments": experiments,
    }
    output_path = Path(cfg["output_root"]) / "results" / args.output
    save_json(output_path, result)
    print("Wrote %s" % output_path, flush=True)
    for exp in experiments:
        err = exp["state_error"]
        agree = exp["token_agreement_with_fp32"]["agreement"]
        print(
            "bits=%s granularity=%s rel_mean=%s rel_max=%s nan=%s inf=%s shape_bad=%s dtype_bad=%s agreement=%.6f"
            % (
                exp["bits"],
                exp["granularity_arg"],
                err["mean_relative_error"],
                err["max_relative_error"],
                err["nan_records"],
                err["inf_records"],
                err["shape_mismatch_records"],
                err["dtype_mismatch_records"],
                agree,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
