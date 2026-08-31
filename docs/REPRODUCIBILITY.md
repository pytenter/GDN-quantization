# Reproducibility

This repository stores research scripts and compact evidence. It does not store model checkpoints, benchmark datasets, raw token/layer/head traces, or server logs.

## Required Local Assets

- Qwen3.5-9B model weights, available through a local model path.
- A compatible Qwen3.5-capable Transformers checkout or installation.
- Benchmark datasets obtained independently under their own licenses.

## Environment Variables

The copied scripts retain the original experiment structure, but the public repository uses configurable roots:

```bash
export GDN_DATA_ROOT=/path/to/gdn_data_root
export GDN_TRANSFORMERS_SRC=/path/to/transformers-qwen35
export GDN_MATH500_ARROW=/path/to/math-500-test.arrow
export GDN_AIME2024_ARROW=/path/to/aime_2024-train.arrow
```

Expected layout under `GDN_DATA_ROOT`:

```text
experiments/qwen35_gdn_quant/
results/
reports/
transformers-qwen35/
```

## What Is Not Included

- Model checkpoints or tokenizer payloads.
- Raw `*_records.jsonl` files.
- Raw token/layer/head `*.jsonl.gz` traces.
- Dataset copies or full benchmark prompt text.
- SSH keys, API keys, tokens, or server credentials.

## Continuation Provenance

Several mechanism experiments use teacher-forced continuations obtained by retokenizing decoded FP_STATE responses:

```text
continuation_source = P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED
exact_original_generation_token_ids_available = false
exact_replay_claimed = false
```

These runs should not be described as exact replay of original generation token IDs.
