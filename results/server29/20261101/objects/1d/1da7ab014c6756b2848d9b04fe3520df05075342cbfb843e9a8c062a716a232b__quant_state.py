#!/usr/bin/env python3
"""Recurrent state quantization and tracing for GDN/KDA cache states."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from quant_utils import error_record, symmetric_quant_dequant, tensor_stats


def collect_recurrent_states(cache):
    states = []
    if cache is None:
        return states
    candidates = []
    if hasattr(cache, "layers"):
        for layer_idx, layer in enumerate(cache.layers):
            recurrent_states = getattr(layer, "recurrent_states", None)
            if recurrent_states is None:
                continue
            if isinstance(recurrent_states, dict):
                iterator = recurrent_states.items()
            else:
                iterator = enumerate(recurrent_states)
            for state_idx, value in iterator:
                if hasattr(value, "shape"):
                    states.append({"name": "layers.%s.recurrent_states.%s" % (layer_idx, state_idx), "tensor": value})
    if hasattr(cache, "recurrent_states"):
        candidates.append(("recurrent_states", getattr(cache, "recurrent_states")))
    if isinstance(cache, (tuple, list)):
        candidates.append(("tuple_cache", cache))
    for root_name, obj in candidates:
        if isinstance(obj, dict):
            iterator = obj.items()
        else:
            iterator = enumerate(obj)
        for key, value in iterator:
            if hasattr(value, "shape"):
                states.append({"name": "%s.%s" % (root_name, key), "tensor": value})
            elif isinstance(value, (tuple, list)):
                for j, sub in enumerate(value):
                    if hasattr(sub, "shape") and len(sub.shape) >= 3:
                        states.append({"name": "%s.%s.%s" % (root_name, key, j), "tensor": sub})
    return states


def main():
    parser = build_arg_parser("Inspect or quantize GDN recurrent state")
    parser.add_argument("--bits", type=int, default=8, choices=[16, 8, 6, 4, 3, 2])
    parser.add_argument("--mode", choices=["inspect-state", "trace"], default="trace")
    parser.add_argument("--prompt", default="Trace recurrent state quantization over decoding.")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    records = []
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        states = collect_recurrent_states(getattr(out, "past_key_values", None))
        for token_idx, item in enumerate(states):
            state = item["tensor"]
            qdq, _ = symmetric_quant_dequant(state, args.bits)
            rec = error_record(state, qdq)
            rec.update({"token_index": token_idx, "state_name": item["name"], "bits": args.bits})
            rec["state"] = tensor_stats(state)
            records.append(rec)
    result = {
        "run": timestamp_record(),
        "mode": args.mode,
        "setting": "S%d" % args.bits,
        "records": records,
        "note": "Exact token-by-token cache rewriting depends on the installed Qwen3.5 cache implementation.",
    }
    name = "state_error_trace.json" if args.bits == 8 else "state_error_trace_s%d.json" % args.bits
    save_json(Path(cfg["output_root"]) / "results" / name, result)
    print("Wrote", name)


if __name__ == "__main__":
    main()
