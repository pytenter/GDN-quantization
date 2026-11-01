# Qwen3.5 Transformers Compatibility

## Root Cause

The local checkpoint uses:

- `model_type = "qwen3_5"`
- `architectures = ["Qwen3_5ForConditionalGeneration"]`

The original runtime had Transformers 4.57.3, but that install did not contain
`transformers/models/qwen3_5/`. The checkpoint also has no `auto_map` metadata
and no local remote-code files such as `modeling_qwen3_5.py`, so
`AutoConfig.from_pretrained()` could not map the checkpoint to a model class.

## Official Support Check

A separate upstream source tree was placed at:

`/data/zypan/transformers-qwen35`

Revision:

`a353632607c59463e6ced86a44c2de3c2cd62d5e`

Confirmed files:

- `src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `src/transformers/models/qwen3_5/tokenization_qwen3_5.py`

Confirmed auto mappings include `qwen3_5`, `qwen3_5_text`, and
`Qwen3_5ForConditionalGeneration`.

## Exact Fix

Installed Transformers from the separate source tree into the existing
`bitdecode` environment as editable source:

```bash
cd /data/zypan/transformers-qwen35
/data/ydai/miniconda3/envs/bitdecode/bin/pip install -e . --no-deps
```

The source install required a few Transformers runtime dependencies that were
missing or below the source tree's minimum versions. These were changed only at
the Python package level and did not touch torch, CUDA, driver, system Python, or
model files:

- `httpx==0.28.1`
- `anyio==4.14.2`
- `httpcore==1.0.9`
- `h11==0.16.0`
- `exceptiongroup==1.3.1`
- `huggingface-hub==1.28.0`
- `tokenizers==0.23.1`
- `safetensors==0.8.0`

Torch and CUDA were unchanged.

## Final Working Environment

- Python: 3.10.18
- torch: 2.5.1+cu121
- torch CUDA: 12.1
- CUDA available: True
- NVIDIA driver: 535.183.01
- Transformers: 5.16.0.dev0
- Transformers path: `/data/zypan/transformers-qwen35/src/transformers`
- huggingface-hub: 1.28.0
- tokenizers: 0.23.1
- safetensors: 0.8.0

## Validation Commands

```bash
cd /data/zypan
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 \
  experiments/qwen35_gdn_quant/inspect_arch.py

/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 \
  experiments/qwen35_gdn_quant/quant_state.py --mode inspect-state
```

Both commands completed successfully.

## State Inspect Result

Model loading:

- model class: `Qwen3_5ForCausalLM`
- cache class: `DynamicCache`
- cache layout: `DynamicCache.layers[layer_idx].recurrent_states[state_idx]`
- GDN / linear attention layers: 24
- full attention layers: 8

Observed GDN recurrent state:

- number of recurrent states: 24
- one recurrent state per GDN / linear attention layer
- GDN layers: 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30
- shape for every state: `[1, 32, 128, 128]`
- dtype for every state: `torch.float32`
- memory per state: 2.0 MiB
- total recurrent state memory for batch 1: 48.0 MiB

Device placement:

- `cuda:0`: 2 states
- `cuda:1`: 8 states
- `cuda:2`: 8 states
- `cuda:3`: 6 states

This revises the earlier static expectation from likely `[1, 16, 128, 128]` to
the runtime-confirmed `[1, 32, 128, 128]`.

## Output Files

- `/data/zypan/results/gdn_state_runtime_inspect.json`
- `/data/zypan/results/state_error_trace.json`
- `/data/zypan/reports/qwen35_transformers_diagnosis.md`
- `/data/zypan/reports/qwen35_transformers_compatibility.md`

No benchmark, LongBench run, quantization sweep, CUDA change, model-file change,
or torch reinstall was performed.
