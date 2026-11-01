#!/usr/bin/env python3
"""Activation sensitivity hooks for GDN/KDA-related tensors."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from quant_utils import error_record, symmetric_quant_dequant


def is_target_module(name, targets):
    lowered = name.lower()
    return any(t in lowered for t in targets)


def main():
    parser = build_arg_parser("Run activation quantization sensitivity")
    parser.add_argument("--bits", type=int, choices=[8, 4], required=True)
    parser.add_argument("--prompt", default="Measure activation sensitivity.")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    targets = cfg["activation_quantization"]["target_names"]
    records = []

    def hook(name):
        def _hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return output
            qdq, _ = symmetric_quant_dequant(output, args.bits)
            rec = error_record(output, qdq)
            rec["module"] = name
            rec["bits"] = args.bits
            records.append(rec)
            return qdq.to(output.dtype)
        return _hook

    handles = []
    for name, module in model.named_modules():
        if is_target_module(name, targets):
            handles.append(module.register_forward_hook(hook(name)))
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model(**inputs, use_cache=True)
    for h in handles:
        h.remove()
    result = {
        "run": timestamp_record(),
        "setting": "W16A%dS16" % args.bits,
        "target_names": targets,
        "records": records,
        "quality_drop": None,
    }
    save_json(Path(cfg["output_root"]) / "results" / "activation_sensitivity_a%d.json" % args.bits, result)
    print("Wrote activation sensitivity result for A%d" % args.bits)


if __name__ == "__main__":
    main()
