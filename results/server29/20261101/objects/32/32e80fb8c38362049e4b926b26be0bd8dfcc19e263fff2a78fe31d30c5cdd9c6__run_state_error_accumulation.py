#!/usr/bin/env python3
"""Long decode recurrent state error accumulation experiment."""
import json
import time
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from quant_utils import error_record, symmetric_quant_dequant
from state_fake_quant import iter_recurrent_state_tensors


DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]


def parse_bits(value):
    out = []
    for part in value.split(","):
        part = part.strip().upper()
        out.append(32 if part in ("FP32", "32") else int(part))
    return out


def update_bucket(bucket, value):
    bucket["count"] += 1
    bucket["sum"] += value
    bucket["max"] = max(bucket["max"], value)


def finalize_bucket(bucket):
    if bucket["count"] == 0:
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": bucket["count"],
        "mean": bucket["sum"] / bucket["count"],
        "max": bucket["max"],
    }


def run_one_setting(torch, model, tokenizer, prompt, max_new_tokens, bits, layer_types, records_fh):
    first_device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(first_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(first_device)
    past = None
    generated = []
    token_buckets = {}
    layer_buckets = {}
    started = time.time()
    with torch.inference_mode():
        for token_idx in range(max_new_tokens + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            token_bucket = token_buckets.setdefault(token_idx, {"count": 0, "sum": 0.0, "max": 0.0})
            for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(past, layer_types):
                if bits >= 32:
                    rel = 0.0
                    abs_err = 0.0
                    state_norm = float(torch.linalg.vector_norm(state.detach().float()).item())
                else:
                    qdq, _ = symmetric_quant_dequant(state, bits=bits)
                    rec_err = error_record(state, qdq)
                    rel = rec_err["relative_error"]
                    abs_err = rec_err["quant_error_l2"]
                    state_norm = rec_err["state_norm_l2"]
                    state.copy_(qdq.to(state.dtype))
                update_bucket(token_bucket, rel)
                layer_bucket = layer_buckets.setdefault(layer_idx, {"count": 0, "sum": 0.0, "max": 0.0})
                update_bucket(layer_bucket, rel)
                records_fh.write(json.dumps({
                    "bits": "FP32" if bits >= 32 else bits,
                    "token_index": token_idx,
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "state_idx": state_idx,
                    "relative_error": rel,
                    "absolute_error_l2": abs_err,
                    "state_norm_l2": state_norm,
                    "shape": list(state.shape),
                    "dtype": str(state.dtype),
                    "device": str(state.device),
                }, sort_keys=True) + "\n")
            if token_idx == max_new_tokens:
                break
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(first_device)
            attention_mask = None
            if token_idx and token_idx % 64 == 0:
                elapsed = time.time() - started
                print("bits=%s token=%d/%d elapsed=%.1fs" % ("FP32" if bits >= 32 else bits, token_idx, max_new_tokens, elapsed), flush=True)
    return {
        "bits": "FP32" if bits >= 32 else bits,
        "max_new_tokens": max_new_tokens,
        "generated_token_count": len(generated),
        "generated_text_prefix": tokenizer.decode(generated[:64], skip_special_tokens=True),
        "token_error": [
            {"token_index": token_idx, **finalize_bucket(bucket)}
            for token_idx, bucket in sorted(token_buckets.items())
        ],
        "layer_error": [
            {"layer_idx": layer_idx, **finalize_bucket(bucket)}
            for layer_idx, bucket in sorted(layer_buckets.items())
        ],
    }


def prefix_summary(setting, lengths):
    token_error = setting["token_error"]
    out = []
    for length in lengths:
        subset = [p for p in token_error if p["token_index"] <= length]
        vals = [p["mean"] for p in subset if p["mean"] is not None]
        out.append({
            "decode_length": length,
            "mean_relative_error": sum(vals) / len(vals) if vals else None,
            "max_relative_error": max((p["max"] for p in subset if p["max"] is not None), default=None),
            "num_token_points": len(subset),
        })
    return out


def plot(summary, root):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)
    plt.figure(figsize=(8, 4.5))
    for setting in summary["settings"]:
        if setting["bits"] == "FP32":
            continue
        xs = [p["token_index"] for p in setting["token_error"]]
        ys = [p["mean"] for p in setting["token_error"]]
        plt.plot(xs, ys, linewidth=1.4, label="INT%s" % setting["bits"])
    plt.xlabel("Decode token position")
    plt.ylabel("Mean relative state error")
    plt.title("Qwen3.5 GDN recurrent state error vs decode length")
    plt.legend()
    plt.tight_layout()
    fig = root / "figures" / "state_error_vs_length.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig, dpi=160)
    return None


def main():
    parser = build_arg_parser("Run long decode recurrent state error accumulation")
    parser.add_argument("--prompt", default="Qwen recurrent state long decode test:")
    parser.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    parser.add_argument("--bits", default="FP32,8,4")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    max_len = max(lengths)
    bits_list = parse_bits(args.bits)
    torch = require("torch")
    transformers = require("transformers")
    root = Path(cfg["output_root"])
    records_path = root / "results" / "state_error_accumulation_records.jsonl"
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = list(getattr(text_cfg, "layer_types", []))
    settings = []
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as records_fh:
        for bits in bits_list:
            print("starting bits=%s max_len=%d" % ("FP32" if bits >= 32 else bits, max_len), flush=True)
            settings.append(run_one_setting(torch, model, tokenizer, args.prompt, max_len, bits, layer_types, records_fh))
    for setting in settings:
        setting["length_prefix_summary"] = prefix_summary(setting, lengths)
    summary = {
        "run": timestamp_record(),
        "model_path": cfg["model_path"],
        "model_class": model.__class__.__name__,
        "cache_path": "DynamicCache.layers[layer_idx].recurrent_states[0]",
        "lengths": lengths,
        "bits": ["FP32" if b >= 32 else b for b in bits_list],
        "records_path": str(records_path),
        "settings": settings,
    }
    plot_error = plot(summary, root)
    if plot_error:
        summary["plot_error"] = plot_error
    save_json(root / "results" / "state_error_accumulation_summary.json", summary)
    layer = {}
    for setting in settings:
        layer[str(setting["bits"])] = setting["layer_error"]
    save_json(root / "results" / "layer_state_sensitivity.json", {
        "run": timestamp_record(),
        "source": str(root / "results" / "state_error_accumulation_summary.json"),
        "layer_error": layer,
    })
    print("Wrote results/state_error_accumulation_summary.json")
    print("Wrote results/layer_state_sensitivity.json")
    print("Wrote figures/state_error_vs_length.png")


if __name__ == "__main__":
    main()
