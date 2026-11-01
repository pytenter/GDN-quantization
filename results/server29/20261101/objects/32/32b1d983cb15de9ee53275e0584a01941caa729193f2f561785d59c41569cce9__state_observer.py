#!/usr/bin/env python3
"""Record FP32 GDN recurrent state norms during a short decode."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import iter_recurrent_state_tensors
from quant_utils import tensor_stats


def main():
    parser = build_arg_parser("Observe Qwen3.5 GDN recurrent state during decode")
    parser.add_argument("--prompt", default="State quantization test:")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = list(getattr(text_cfg, "layer_types", []))
    inputs = tokenizer(args.prompt, return_tensors="pt")
    first_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(first_device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(first_device)
    past = None
    records = []
    generated = []
    with torch.inference_mode():
        for token_idx in range(args.max_new_tokens + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(past, layer_types):
                rec = tensor_stats(state)
                rec.update({"token_index": token_idx, "layer_idx": layer_idx, "layer_type": layer_type, "state_idx": state_idx})
                records.append(rec)
            if token_idx == args.max_new_tokens:
                break
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(first_device)
            attention_mask = None
    result = {
        "run": timestamp_record(),
        "mode": "fp32_observation",
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "model_class": model.__class__.__name__,
        "cache_class": past.__class__.__name__ if past is not None else None,
        "gdn_layer_count": sum(1 for x in layer_types if x == "linear_attention"),
        "records": records,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    save_json(Path(cfg["output_root"]) / "results" / "state_trace_fp32.json", result)
    print("Wrote results/state_trace_fp32.json with %d records" % len(records))


if __name__ == "__main__":
    main()
