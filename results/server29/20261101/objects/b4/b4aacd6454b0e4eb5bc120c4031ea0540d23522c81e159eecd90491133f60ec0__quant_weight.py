#!/usr/bin/env python3
"""Simple PTQ weight sensitivity for W8 per-channel and W4 group-wise."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from quant_utils import groupwise_weight_quant_dequant, symmetric_quant_dequant, tensor_stats


def quantize_linear_weights(model, mode):
    torch = require("torch")
    records = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            w = module.weight.data
            before = tensor_stats(w)
            if mode == "w8":
                q, _ = symmetric_quant_dequant(w, bits=8, axis=1)
                module.weight.data.copy_(q.to(w.dtype))
            elif mode == "w4":
                module.weight.data.copy_(groupwise_weight_quant_dequant(w, bits=4, group_size=128).to(w.dtype))
            after = tensor_stats(module.weight.data)
            records.append({"module": name, "before": before, "after": after})
    return records


def main():
    parser = build_arg_parser("Run simple PTQ weight sensitivity")
    parser.add_argument("--mode", choices=["w8", "w4"], required=True)
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    records = quantize_linear_weights(model, args.mode)
    result = {
        "run": timestamp_record(),
        "mode": args.mode,
        "setting": "W8A16S16" if args.mode == "w8" else "W4A16S16",
        "records": records,
        "quality": None,
        "latency_ms": None,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save_json(Path(cfg["output_root"]) / "results" / "weight_sensitivity_%s.json" % args.mode, result)
    print("Wrote weight sensitivity result for", args.mode)


if __name__ == "__main__":
    main()
