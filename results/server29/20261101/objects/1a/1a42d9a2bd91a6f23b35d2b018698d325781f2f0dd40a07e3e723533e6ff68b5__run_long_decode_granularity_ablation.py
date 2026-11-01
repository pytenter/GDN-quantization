#!/usr/bin/env python3
"""Long decode stability ablation for GDN recurrent state quantization granularity."""
import json
import math
import time
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import normalize_granularity, quantize_cache_recurrent_states


DEFAULT_LENGTHS = [512, 1024, 2048, 4096, 8192]
DEFAULT_SETTINGS = [
    (32, "none"),
    (8, "layer"),
    (8, "column"),
    (4, "layer"),
    (4, "row"),
    (4, "column"),
]


def is_finite_number(value):
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def parse_lengths(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_settings(value):
    if not value:
        return DEFAULT_SETTINGS
    settings = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.upper() in ("FP32", "32", "FP32:none"):
            settings.append((32, "none"))
            continue
        bits_s, granularity = item.split(":", 1)
        bits = int(bits_s)
        if bits not in (8, 4):
            raise ValueError("only INT8 and INT4 are supported for quantized settings")
        normalize_granularity(granularity)
        settings.append((bits, granularity))
    return settings


def setting_name(bits, granularity):
    if bits >= 32:
        return "FP32"
    return "INT%d-%s" % (bits, granularity)


def new_bucket():
    return {
        "count": 0,
        "finite_count": 0,
        "relative_error_sum": 0.0,
        "relative_error_max": 0.0,
        "state_norm_sum": 0.0,
        "state_norm_max": 0.0,
        "nan_count": 0,
        "inf_count": 0,
    }


def update_bucket(bucket, relative_error, state_norm, nan_count, inf_count):
    bucket["count"] += 1
    bucket["nan_count"] += nan_count
    bucket["inf_count"] += inf_count
    if is_finite_number(relative_error) and is_finite_number(state_norm):
        bucket["finite_count"] += 1
        bucket["relative_error_sum"] += relative_error
        bucket["relative_error_max"] = max(bucket["relative_error_max"], relative_error)
        bucket["state_norm_sum"] += state_norm
        bucket["state_norm_max"] = max(bucket["state_norm_max"], state_norm)


def finalize_bucket(bucket):
    finite_count = bucket["finite_count"]
    return {
        "count": bucket["count"],
        "finite_count": finite_count,
        "nan_count": bucket["nan_count"],
        "inf_count": bucket["inf_count"],
        "mean_relative_error": bucket["relative_error_sum"] / finite_count if finite_count else None,
        "max_relative_error": bucket["relative_error_max"] if finite_count else None,
        "mean_state_norm": bucket["state_norm_sum"] / finite_count if finite_count else None,
        "max_state_norm": bucket["state_norm_max"] if finite_count else None,
    }


def nonfinite_state_counts(torch, state):
    x = state.detach()
    nan_count = int(torch.isnan(x).sum().item())
    inf_count = int(torch.isinf(x).sum().item())
    return nan_count, inf_count


def first_nonfinite_head(torch, state):
    mask = ~torch.isfinite(state.detach())
    if not bool(mask.any().item()):
        return None
    idx = mask.nonzero(as_tuple=False)[0].tolist()
    if len(idx) >= 4:
        return {
            "batch_id": int(idx[-4]),
            "head_id": int(idx[-3]),
            "key_index": int(idx[-2]),
            "value_index": int(idx[-1]),
        }
    return {"index": [int(x) for x in idx]}


def inspect_cache_nonfinite(torch, cache, layer_types):
    from state_fake_quant import iter_recurrent_state_tensors

    out = {}
    for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(cache, layer_types):
        nan_count, inf_count = nonfinite_state_counts(torch, state)
        out[(layer_idx, state_idx)] = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "head_location": first_nonfinite_head(torch, state) if nan_count or inf_count else None,
        }
    return out


def run_one_setting(torch, model, tokenizer, prompt, max_new_tokens, bits, granularity, layer_types, records_fh):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    token_buckets = {}
    layer_buckets = {}
    first_failure = None
    generated = []
    past = None
    started = time.time()
    norm_granularity = "none" if bits >= 32 else normalize_granularity(granularity)

    with torch.inference_mode():
        for token_idx in range(max_new_tokens + 1):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            records = quantize_cache_recurrent_states(
                past,
                bits=bits,
                layer_types=layer_types,
                granularity="layer" if bits >= 32 else granularity,
            )
            nonfinite_by_state = inspect_cache_nonfinite(torch, past, layer_types)
            token_bucket = token_buckets.setdefault(token_idx, new_bucket())
            for rec in records:
                key = (rec["layer_idx"], rec["state_idx"])
                nf = nonfinite_by_state.get(key, {"nan_count": 0, "inf_count": 0, "head_location": None})
                nan_count = nf["nan_count"]
                inf_count = nf["inf_count"]
                relative_error = rec.get("relative_error")
                state_norm = rec.get("state_norm_l2")
                if first_failure is None and (nan_count or inf_count or not is_finite_number(relative_error) or not is_finite_number(state_norm)):
                    first_failure = {
                        "token_index": token_idx,
                        "layer_idx": rec["layer_idx"],
                        "state_idx": rec["state_idx"],
                        "head": nf["head_location"],
                        "relative_error": relative_error,
                        "state_norm_l2": state_norm,
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                    }
                update_bucket(token_bucket, relative_error, state_norm, nan_count, inf_count)
                layer_bucket = layer_buckets.setdefault(str(rec["layer_idx"]), new_bucket())
                update_bucket(layer_bucket, relative_error, state_norm, nan_count, inf_count)
                records_fh.write(json.dumps({
                    "setting": setting_name(bits, granularity),
                    "bits": "FP32" if bits >= 32 else bits,
                    "granularity": norm_granularity,
                    "token_index": token_idx,
                    "layer_idx": rec["layer_idx"],
                    "layer_type": rec["layer_type"],
                    "state_idx": rec["state_idx"],
                    "relative_error": relative_error,
                    "state_norm_l2": state_norm,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "scale_shape": rec.get("scale_shape", []),
                    "broadcast_scale_shape": rec.get("broadcast_scale_shape", []),
                    "shape": rec.get("state", {}).get("shape"),
                    "dtype": rec.get("state", {}).get("dtype"),
                }, sort_keys=True) + "\n")

            if token_idx == max_new_tokens:
                break
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(device)
            attention_mask = None
            if token_idx and token_idx % 256 == 0:
                print(
                    "%s token=%d/%d elapsed=%.1fs"
                    % (setting_name(bits, granularity), token_idx, max_new_tokens, time.time() - started),
                    flush=True,
                )

    token_error = [
        {"token_index": token_idx, **finalize_bucket(bucket)}
        for token_idx, bucket in sorted(token_buckets.items())
    ]
    layer_error = [
        {"layer_idx": int(layer_idx), **finalize_bucket(bucket)}
        for layer_idx, bucket in sorted(layer_buckets.items(), key=lambda kv: int(kv[0]))
    ]
    return {
        "setting": setting_name(bits, granularity),
        "bits": "FP32" if bits >= 32 else bits,
        "granularity": norm_granularity,
        "granularity_arg": "none" if bits >= 32 else granularity,
        "max_new_tokens": max_new_tokens,
        "generated_token_count": len(generated),
        "generated_tokens_prefix": generated[:128],
        "generated_text_prefix": tokenizer.decode(generated[:128], skip_special_tokens=True),
        "first_failure": first_failure,
        "token_error": token_error,
        "layer_error": layer_error,
    }


def prefix_summaries(setting, lengths):
    out = []
    token_error = setting["token_error"]
    for length in lengths:
        bucket = new_bucket()
        first_failure = None
        for point in token_error:
            if point["token_index"] > length:
                continue
            bucket["count"] += point["count"]
            bucket["finite_count"] += point["finite_count"]
            bucket["relative_error_sum"] += (point["mean_relative_error"] or 0.0) * point["finite_count"]
            bucket["state_norm_sum"] += (point["mean_state_norm"] or 0.0) * point["finite_count"]
            if point["max_relative_error"] is not None:
                bucket["relative_error_max"] = max(bucket["relative_error_max"], point["max_relative_error"])
            if point["max_state_norm"] is not None and is_finite_number(point["max_state_norm"]):
                bucket["state_norm_max"] = max(bucket["state_norm_max"], point["max_state_norm"])
            bucket["nan_count"] += point["nan_count"]
            bucket["inf_count"] += point["inf_count"]
        if setting["first_failure"] and setting["first_failure"]["token_index"] <= length:
            first_failure = setting["first_failure"]
        out.append({
            "bits": setting["bits"],
            "granularity": setting["granularity"],
            "length": length,
            **finalize_bucket(bucket),
            "first_failure": first_failure,
        })
    return out


def downsample(points, max_points=1200):
    if len(points) <= max_points:
        return points
    stride = int(math.ceil(len(points) / max_points))
    return points[::stride]


def plot_results(settings, root):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for setting in settings:
        if setting["bits"] == "FP32":
            continue
        points = downsample(setting["token_error"])
        xs = [p["token_index"] for p in points]
        ys = [p["mean_relative_error"] for p in points]
        plt.plot(xs, ys, linewidth=1.2, label=setting["setting"])
    plt.xlabel("Decode token position")
    plt.ylabel("Mean relative state error")
    plt.title("GDN state error vs length by granularity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "state_error_vs_length_granularity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for setting in settings:
        points = downsample(setting["token_error"])
        xs = [p["token_index"] for p in points]
        ys = [p["mean_state_norm"] for p in points]
        plt.plot(xs, ys, linewidth=1.2, label=setting["setting"])
    plt.xlabel("Decode token position")
    plt.ylabel("Mean finite state norm")
    plt.title("GDN state norm stability by granularity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "state_norm_stability.png", dpi=160)
    plt.close()
    return None


def main():
    parser = build_arg_parser("Run long decode GDN state granularity stability ablation")
    parser.add_argument("--prompt", default="Qwen recurrent state long decode granularity test:")
    parser.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    parser.add_argument("--settings", default=None, help="Comma list like FP32,8:layer,8:column,4:layer,4:row,4:column")
    parser.add_argument("--output", default="long_decode_granularity_ablation.json")
    parser.add_argument("--records-output", default="long_decode_granularity_records.jsonl")
    args = parser.parse_args()

    cfg = resolve_cfg(args)
    root = Path(cfg["output_root"])
    lengths = parse_lengths(args.lengths)
    max_len = max(lengths)
    settings_spec = parse_settings(args.settings)
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

    records_path = root / "results" / args.records_output
    records_path.parent.mkdir(parents=True, exist_ok=True)
    settings = []
    with records_path.open("w", encoding="utf-8") as records_fh:
        for bits, granularity in settings_spec:
            print("starting %s max_len=%d" % (setting_name(bits, granularity), max_len), flush=True)
            settings.append(
                run_one_setting(
                    torch, model, tokenizer, args.prompt, max_len, bits, granularity, layer_types, records_fh
                )
            )

    length_summaries = []
    for setting in settings:
        length_summaries.extend(prefix_summaries(setting, lengths))

    plot_error = plot_results(settings, root)
    result = {
        "run": timestamp_record(),
        "model_path": cfg["model_path"],
        "model_class": model.__class__.__name__,
        "cache_path": "DynamicCache.layers[layer_idx].recurrent_states[0]",
        "state_shape": [1, 32, 128, 128],
        "state_dtype": "torch.float32",
        "lengths": lengths,
        "settings_requested": [
            {"bits": "FP32" if bits >= 32 else bits, "granularity": "none" if bits >= 32 else normalize_granularity(granularity)}
            for bits, granularity in settings_spec
        ],
        "records_path": str(records_path),
        "figures": [
            str(root / "figures" / "state_error_vs_length_granularity.png"),
            str(root / "figures" / "state_norm_stability.png"),
        ],
        "length_summaries": length_summaries,
        "settings": settings,
    }
    if plot_error:
        result["plot_error"] = plot_error
    save_json(root / "results" / args.output, result)
    print("Wrote %s" % (root / "results" / args.output), flush=True)
    for item in length_summaries:
        if item["length"] == max_len:
            print(
                "%s-%s length=%d mean_rel=%s nan=%s inf=%s first_failure=%s"
                % (
                    item["bits"],
                    item["granularity"],
                    item["length"],
                    item["mean_relative_error"],
                    item["nan_count"],
                    item["inf_count"],
                    item["first_failure"],
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
