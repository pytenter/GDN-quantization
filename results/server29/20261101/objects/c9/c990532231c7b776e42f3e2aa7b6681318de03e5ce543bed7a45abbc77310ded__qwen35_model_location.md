# Qwen3.5-9B Model Location

## Confirmed Local Model

- Path: `/data/zypan/modelscope_models/Qwen3.5-9B`
- Size: 19G
- Download was not attempted.

## Key Files

| file | size bytes |
|---|---:|
| config.json | 3126 |
| configuration.json | 51 |
| model.safetensors-00001-of-00004.safetensors | 5276436216 |
| model.safetensors-00002-of-00004.safetensors | 5335161512 |
| model.safetensors-00003-of-00004.safetensors | 5368717440 |
| model.safetensors-00004-of-00004.safetensors | 3325995712 |
| model.safetensors.index.json | 79657 |
| tokenizer.json | 12807982 |
| tokenizer_config.json | 16710 |
| vocab.json | 6722759 |
| merges.txt | 3353259 |
| chat_template.jinja | 7756 |
| preprocessor_config.json | 390 |
| video_preprocessor_config.json | 385 |

## Model Config Summary

- architecture: Qwen3_5ForConditionalGeneration
- model_type: qwen3_5
- text model_type: qwen3_5_text
- text layers: 32
- GDN / linear attention layers: 24
- full attention layers: 8
- layer pattern: 3 linear_attention + 1 full_attention repeated 8 times
- linear_num_key_heads: 16
- linear_num_value_heads: 32
- linear_key_head_dim: 128
- linear_value_head_dim: 128

## Local Code Availability

Only `configuration.json` was found as an extra configuration file.
No local Python implementation files were found in the model directory.

Missing local implementation files include:

- modeling_qwen3_5.py
- configuration_qwen3_5.py
- tokenization_qwen3_5.py
