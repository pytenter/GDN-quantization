# GDN Quantization

Mechanism-oriented study of low-bit recurrent-state quantization in Gated DeltaNet / linear-attention models.

## Current Model

Qwen3.5-9B

## Research Question

How does recurrent-state quantization error enter, propagate through, and accumulate in GDN recurrence, and which errors actually determine long-horizon model fidelity?

## Current Status

Project stage: `MECHANISM_VALIDATION`.

`MECHANISM_CLOSURE_CANDIDATE = NO`

`METHOD_DESIGN_READY = NO`

No final quantization method is proposed yet.

## Protocol Summary

The experiments use FP32 prefill followed by quantized or intervened recurrent continuation. The main recurrent state has shape:

```text
[B,H,K,V] = [1,32,128,128]
```

Axis naming in this repository:

```text
row    = Key-axis row group
column = Value-axis column group

R128 = full 128-element Value-axis row group
C128 = full 128-element Key-axis column group
```

## Key Findings So Far

- R128/C128 orientation remains a first-order variable.
- Canonical R128 poor quantizability is strongly representation dependent.
- Cross-Value range and scale contamination are the strongest current source-side explanation.
- Repeated R128 quantization causes formal cadence-dependent trajectory accumulation.
- V-space source rescue formally attenuates repeated trajectory and KL accumulation in 18/18 formal units.
- Residual magnitude is a causal contributor, but magnitude alone is insufficient.
- Same-norm R/C residual structure produces systematically different behavior.
- A more harmful R residual can be less persistent: persistence is not relevance.
- Future-key interaction is a path signal, not a proven standalone causal variable.
- Local operator coupling is real: R has larger U/E and state-error consumption than C.
- One-step update transduction does not explain the downstream behavioral gap.
- Single-head downstream operator feedback is not supported.
- Single-layer multi-head scope amplification remains inconclusive.
- Head-wise S8 robustness is only partial and shows strong head-set heterogeneity.

Do not read this repository as claiming V-Hadamard is a finished method, C128 is a final method, J_key is the metric, U is the final mechanism, or method design is ready.

## Repository Layout

```text
experiments/          Canonical experiment scripts
results/              Compact final JSON and protocol/stage0 JSON
reports/              Final Markdown reports
docs/                 Experiment index, evidence map, status, prompt manifest
```

Large raw token/layer/head traces, model checkpoints, logs, worker files, checkpoints, and dataset copies are intentionally excluded.

## Reproducibility Note

The repository contains research scripts and compact experimental evidence from an active mechanism-validation project. Some scripts retain assumptions from the original experimental environment and may require local model/data path configuration.
