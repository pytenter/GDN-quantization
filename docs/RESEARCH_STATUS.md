# Research Status

Snapshot date: 2026-09-21

## Project lines

| Line | Current condition | Rotation line | Status |
|---|---|---|---|
| Qwen3.5-9B / GDN | INT8_C128 | Key-side Hadamard | AIME26 81,920 complete; evidence belongs to qwen3090 inventory |
| Ling-3.0-tiny / KDA | INT8_R128 | Value-side Hadamard | AIME26 81,920 complete; compact evidence archived here |

The current project scope is recurrent-state INT8 quantization. Historical INT4 work is not a current main line.

## AIME26 81,920

`AIME26_STRICT_V4_CANDIDATE` is the frozen scoring implementation for the archived final comparison. Each condition has 60 samples (30 questions x 2 seeds).

| Condition | Correct | Wrong | Abstain | Accuracy |
|---|---:|---:|---:|---:|
| Qwen FP_STATE | 52 | 8 | 3 | 86.67% |
| Qwen INT8_C128 | 18 | 42 | 39 | 30.00% |
| Qwen INT8_C128 + Key-Hadamard | 41 | 19 | 6 | 68.33% |
| Ling FP_STATE | 44 | 16 | 11 | 73.33% |
| Ling INT8_R128 | 23 | 37 | 28 | 38.33% |
| Ling INT8_R128 + Value-Hadamard | 28 | 32 | 28 | 46.67% |

`Abstain` is a subset of incorrect samples where the strict scorer does not accept a final answer extraction; therefore correct + wrong = samples, while abstain is not a third mutually exclusive denominator category.

## Ling corrected rotation semantics

```text
KDA_ROTATION_SEMANTICS_VERSION = CORRECTED_PREFILL_ENDPOINT_V2
PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY = PASS
REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO
HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION = YES
```

The corrected recurrent state remains in rotated Value coordinates from KDA kernel return through cache write and first decode consumption. Only the KDA readout is mapped back to native Value coordinates before downstream normalization/gating/merge/projection.

## Ling 256K

`LING_256K_LENGTH_SENSITIVITY_4090_HELPER_V1` is in progress on `ling4090`. It uses two independent TP=1 instances on GPUs 0 and 1, odd/even problem partitioning, seed 1, deterministic inference, no radix cache, no CUDA graph, and corrected rotation semantics. No final accuracy is claimed.

## Publication gate

This consolidation is not yet safe to push because the qwen3090 server evidence has not been cross-audited into this branch, the active 256K run is incomplete, and several historical KDA artifact families remain `NEEDS_REVIEW` or legacy.
