# Qwen3.5-9B Architecture Report

## Source
- Model path: `/data/zypan/modelscope_models/Qwen3.5-9B`
- Config file: `/data/zypan/modelscope_models/Qwen3.5-9B/config.json`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Model type: `qwen3_5_text`
- Transformers version in config: `4.57.0.dev0`

## Text Config
- hidden_size: `4096`
- intermediate_size: `12288`
- num_hidden_layers: `32`
- num_attention_heads: `16`
- num_key_value_heads: `4`
- head_dim: `256`
- linear_num_key_heads: `16`
- linear_num_value_heads: `32`
- linear_key_head_dim: `128`
- linear_value_head_dim: `128`
- linear_conv_kernel_dim: `4`
- full_attention_interval: `4`
- max_position_embeddings: `262144`
- vocab_size: `248320`
- dtype: `bfloat16`
- mamba_ssm_dtype: `float32`
- attn_output_gate: `True`
- use_cache: `True`

## Layer Counts
- Total decoder layers: `32`
- Full attention layers: `8`
- GDN/KDA linear attention layers: `24`
- Hybrid pattern: three `linear_attention` layers followed by one `full_attention` layer, repeated 8 times.

## Per-Layer Pattern
- Layer 00: GDN/KDA (`linear_attention`)
- Layer 01: GDN/KDA (`linear_attention`)
- Layer 02: GDN/KDA (`linear_attention`)
- Layer 03: Attention (`full_attention`)
- Layer 04: GDN/KDA (`linear_attention`)
- Layer 05: GDN/KDA (`linear_attention`)
- Layer 06: GDN/KDA (`linear_attention`)
- Layer 07: Attention (`full_attention`)
- Layer 08: GDN/KDA (`linear_attention`)
- Layer 09: GDN/KDA (`linear_attention`)
- Layer 10: GDN/KDA (`linear_attention`)
- Layer 11: Attention (`full_attention`)
- Layer 12: GDN/KDA (`linear_attention`)
- Layer 13: GDN/KDA (`linear_attention`)
- Layer 14: GDN/KDA (`linear_attention`)
- Layer 15: Attention (`full_attention`)
- Layer 16: GDN/KDA (`linear_attention`)
- Layer 17: GDN/KDA (`linear_attention`)
- Layer 18: GDN/KDA (`linear_attention`)
- Layer 19: Attention (`full_attention`)
- Layer 20: GDN/KDA (`linear_attention`)
- Layer 21: GDN/KDA (`linear_attention`)
- Layer 22: GDN/KDA (`linear_attention`)
- Layer 23: Attention (`full_attention`)
- Layer 24: GDN/KDA (`linear_attention`)
- Layer 25: GDN/KDA (`linear_attention`)
- Layer 26: GDN/KDA (`linear_attention`)
- Layer 27: Attention (`full_attention`)
- Layer 28: GDN/KDA (`linear_attention`)
- Layer 29: GDN/KDA (`linear_attention`)
- Layer 30: GDN/KDA (`linear_attention`)
- Layer 31: Attention (`full_attention`)

## Notes
- This report is generated from static config because this server currently lacks `torch` and `transformers`.
- Runtime scripts re-check module names and tensor shapes when the ML stack is available.
