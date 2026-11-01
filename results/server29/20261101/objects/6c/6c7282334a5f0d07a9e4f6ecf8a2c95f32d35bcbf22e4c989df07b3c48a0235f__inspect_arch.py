#!/usr/bin/env python3
"""Static Phase 0 inspection for Qwen3.5 GDN/KDA architecture."""
from pathlib import Path

from common import build_arg_parser, layer_summary, load_model_config, resolve_cfg, save_json, write_text


def format_arch_report(model_cfg, text_cfg, summary, model_path):
    lines = [
        "# Qwen3.5-9B Architecture Report",
        "",
        "## Source",
        "- Model path: `%s`" % model_path,
        "- Config file: `%s`" % (Path(model_path) / "config.json"),
        "- Architecture: `%s`" % ", ".join(model_cfg.get("architectures", [])),
        "- Model type: `%s`" % text_cfg.get("model_type", model_cfg.get("model_type")),
        "- Transformers version in config: `%s`" % model_cfg.get("transformers_version"),
        "",
        "## Text Config",
    ]
    keys = [
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        "full_attention_interval", "max_position_embeddings", "vocab_size", "dtype",
        "mamba_ssm_dtype", "attn_output_gate", "use_cache",
    ]
    for key in keys:
        lines.append("- %s: `%s`" % (key, text_cfg.get(key)))
    lines += [
        "",
        "## Layer Counts",
        "- Total decoder layers: `%d`" % summary["total_layers"],
        "- Full attention layers: `%d`" % summary["attention_layers"],
        "- GDN/KDA linear attention layers: `%d`" % summary["gdn_layers"],
        "- Hybrid pattern: three `linear_attention` layers followed by one `full_attention` layer, repeated 8 times.",
        "",
        "## Per-Layer Pattern",
    ]
    for i, typ in enumerate(summary["layer_types"]):
        label = "Attention" if typ == "full_attention" else "GDN/KDA" if typ == "linear_attention" else typ
        lines.append("- Layer %02d: %s (`%s`)" % (i, label, typ))
    lines += [
        "",
        "## Notes",
        "- This report is generated from static config because this server currently lacks `torch` and `transformers`.",
        "- Runtime scripts re-check module names and tensor shapes when the ML stack is available.",
    ]
    return "\n".join(lines)


def format_state_report(text_cfg, summary):
    key_heads = text_cfg.get("linear_num_key_heads")
    value_heads = text_cfg.get("linear_num_value_heads")
    key_dim = text_cfg.get("linear_key_head_dim")
    value_dim = text_cfg.get("linear_value_head_dim")
    lines = [
        "# Qwen3.5 GDN/KDA Recurrent State Shape Report",
        "",
        "## Static Findings",
        "- GDN/KDA layers are config entries with `layer_types[i] == \"linear_attention\"`.",
        "- GDN/KDA layer count: `%d`" % summary["gdn_layers"],
        "- Full attention layer count: `%d`" % summary["attention_layers"],
        "- Linear key heads: `%s`" % key_heads,
        "- Linear value heads: `%s`" % value_heads,
        "- Linear key head dim: `%s`" % key_dim,
        "- Linear value head dim: `%s`" % value_dim,
        "- Linear convolution kernel dim: `%s`" % text_cfg.get("linear_conv_kernel_dim"),
        "",
        "## Expected Recurrent State Tensor",
        "- Expected semantic form: state maps accumulated key features to value features for each linear-attention state group.",
        "- Expected shape family: `(batch, state_heads, key_dim, value_dim)`.",
        "- From config, the most likely shape is `(batch, 16, 128, 128)` or an implementation-specific grouped variant derived from `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_key_head_dim=128`, and `linear_value_head_dim=128`.",
        "- Exact runtime shape must be confirmed by `quant_state.py --mode inspect-state` once PyTorch/Transformers are installed.",
        "",
        "## State Lifecycle",
        "- Generation: recurrent state is created by the linear attention/GDN cache path when `use_cache=True` and the first GDN layer needs a recurrent cache entry.",
        "- Prefill initialization: for the first prompt chunk, recurrent state starts empty/zero or absent in the cache, then the full prompt updates it token by token or through a fused parallel/chunk scan kernel.",
        "- Update location: inside the GDN/KDA linear attention forward path, where the recurrence has the form `S_t = alpha_t * S_{t-1} + DeltaS_t`; `DeltaS_t` is produced from current-token key/value features and gates.",
        "- Decode persistence: after prefill, the updated recurrent state is stored in the model cache, commonly exposed as `recurrent_states` in a dynamic cache object; each decode step reads `S_{t-1}`, applies the update, and writes new `S_t` back to cache.",
        "- Attention layers use KV cache, while GDN/KDA layers use recurrent state cache.",
        "",
        "## Gate/Activation Tensors To Trace",
        "- q: linear attention query projection output.",
        "- k: linear attention key projection output.",
        "- v: linear attention value projection output.",
        "- gate: output or recurrent gate when present.",
        "- alpha: recurrent decay/retention gate in `S_t = alpha_t * S_{t-1} + DeltaS_t`.",
        "- beta: update/input gate when present in the implementation.",
        "",
        "## Limitation",
        "- No local `modeling_qwen3_5.py` or installed Transformers package was available at inspection time, so exact function names and tensor rank are deferred to runtime hooks.",
    ]
    return "\n".join(lines)


def main():
    parser = build_arg_parser("Inspect Qwen3.5 static architecture config")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    model_cfg, text_cfg = load_model_config(cfg["model_path"])
    summary = layer_summary(text_cfg)
    root = Path(cfg["output_root"])
    save_json(root / "results" / "qwen35_architecture_summary.json", {"text_config": text_cfg, "summary": summary})
    write_text(root / "reports" / "qwen35_architecture.md", format_arch_report(model_cfg, text_cfg, summary, cfg["model_path"]))
    write_text(root / "reports" / "gdn_state_shape.md", format_state_report(text_cfg, summary))
    print("Wrote reports/qwen35_architecture.md")
    print("Wrote reports/gdn_state_shape.md")


if __name__ == "__main__":
    main()
