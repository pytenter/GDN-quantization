# Ling/KDA Canonical 256K Evaluation Protocol

Status: **CURRENT FROZEN PROTOCOL**

This document freezes the generation protocol for current Ling-3.0-tiny/KDA canonical evaluations. It is documentation only and does not change code, configuration, checkpoints, or frozen outputs.

The machine-readable source of truth is `experiments/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/configs/frozen_eval_config.json` on branch `exp/ling-l6-l7-canonical-60-confirmatory-v1`, frozen at commit `69b23bd999dc23a9c852497b98df0f56abdbdff0`.

## Overview

```text
Model:
Ling-3.0-tiny / KDA

Purpose:
Long-horizon recurrent-state INT8 evaluation
```

The state path uses INT8-R128 with the corrected `CORRECTED_PREFILL_ENDPOINT_V2` semantics. This document freezes generation and runtime identity; it does not redefine quantizer or rotation artifacts.

## Context

```ini
context_length = 262144
server_max_total_tokens = 262144
safety_margin = 512
```

The per-sample generation budget is dynamic:

```text
max_new_tokens = 262144 - prompt_tokens - 512
```

The prompt token count is measured for the exact frozen prompt. A fixed value of 81,920 is not valid for current canonical Ling generation.

## Decoding

```ini
thinking = true
do_sample = true
temperature = 1.0
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
sampling_seed = 1 or 2
```

Additional frozen request behavior:

```ini
stop = null
sampling_backend = pytorch
batch_size = 1
max_running_requests = 1
```

Parameters not explicitly present in the frozen evaluation configuration must not be inferred from historical runs or library defaults.

## Runtime

```makefile
SGLang 0.5.19
Torch 2.9.1+cu128
Triton 3.5.1
FLA 0.5.2
dtype=bfloat16
radix_cache=false
cuda_graph=false
tp_size=1
```

The runtime additionally freezes deterministic inference and Triton attention, linear-attention, and MoE runner backends. Runtime upgrades require a new protocol identity; they must not silently inherit this protocol label.

## Sample and scoring identity

- Dataset: all 30 AIME26 problems.
- Sampling seeds: `1` and `2`.
- Canonical sample count: 60 per condition.
- Scorer: exact Frozen V4 (`AIME26_STRICT_V4_CANDIDATE`).
- Scorer SHA256: `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`.

## Protocol Compatibility Warning

Do not compare or merge outputs generated under:

- the historical 81,920-token fixed budget; and
- the current 256K dynamic-budget protocol.

unless the analysis is explicitly labeled as a historical cross-protocol comparison. Canonical Ling experiments after this freeze must use the 256K protocol.

The earlier 81,920-token Ling AIME26 result remains immutable historical evidence for comparison and regression analysis. It is not a current canonical output source. The earlier seed-1 fixed-cap 256K length-sensitivity study is also separate precursor evidence and must not be pooled with the current two-seed dynamic-budget evaluation.
