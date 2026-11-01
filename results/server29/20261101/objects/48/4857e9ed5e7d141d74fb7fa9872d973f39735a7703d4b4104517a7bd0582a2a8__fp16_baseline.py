#!/usr/bin/env python3
"""FP16/BF16 baseline for prompt generation and optional perplexity."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record


def main():
    parser = build_arg_parser("Run Qwen3.5 FP16/BF16 baseline")
    parser.add_argument("--prompt", default="Summarize the purpose of recurrent state in linear attention.")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    model_path = cfg["model_path"]
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start:
        start.record()
    with torch.inference_mode():
        out = model.generate(**inputs, **cfg["generation"])
    if end:
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end)
    else:
        latency_ms = None
    result = {
        "run": timestamp_record(),
        "model_path": model_path,
        "setting": "W16A16S16",
        "batch_size": cfg["batch_size"],
        "max_sequence_length": cfg["max_sequence_length"],
        "generation": cfg["generation"],
        "latency_ms": latency_ms,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "prompt_tokens": int(inputs["input_ids"].numel()),
        "output_tokens": int(out.numel()),
        "decoded_sample": tokenizer.decode(out[0], skip_special_tokens=True),
        "accuracy": None,
        "perplexity": None,
        "note": "LongBench/RULER/perplexity fields are placeholders until datasets are configured.",
    }
    save_json(Path(cfg["output_root"]) / "results" / "fp16_baseline.json", result)
    print("Wrote results/fp16_baseline.json")


if __name__ == "__main__":
    main()
