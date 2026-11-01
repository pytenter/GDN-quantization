# Qwen3.5 Transformers Diagnosis

## Current Runtime

- Env python: `/data/ydai/miniconda3/envs/bitdecode/bin/python3.10`
- Transformers version: `4.57.3`
- Transformers location: `/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/transformers`

## Current Package Contents

`transformers/models/qwen3_5/` exists: `False`

Available Qwen-family packages in this install:

- colqwen2
- qwen2
- qwen2_5_omni
- qwen2_5_vl
- qwen2_audio
- qwen2_moe
- qwen2_vl
- qwen3
- qwen3_moe
- qwen3_next
- qwen3_omni_moe
- qwen3_vl
- qwen3_vl_moe

## Model Config

File: `/data/zypan/modelscope_models/Qwen3.5-9B/config.json`

- top-level `model_type`: `qwen3_5`
- top-level `architectures`: `Qwen3_5ForConditionalGeneration`
- top-level `auto_map`: absent
- `text_config.model_type`: `qwen3_5_text`
- `text_config.auto_map`: absent

## Diagnosis

Current Transformers cannot load this checkpoint because the installed package has no built-in `qwen3_5` implementation and the model snapshot does not provide `auto_map` remote-code metadata or local `modeling_qwen3_5.py` files.
