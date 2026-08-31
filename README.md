# GDN Quantization

Mechanism-oriented study of low-bit recurrent-state quantization in Gated DeltaNet / linear-attention models.

## Current Model

Qwen3.5-9B

## Research Question

How does recurrent-state quantization error enter, propagate through, and accumulate in GDN recurrence, and which errors actually determine long-horizon model fidelity?

## Current Status

Mechanism validation.

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

- INT8 row/column orientation is a first-order variable.
- R128 degradation is substantially stronger than C128 under the tested protocol.
- Finer grouping rescues KL, but same-codebook lost-update metrics do not explain the rescue.
- Runtime effective-update metrics track R16/C16 KL rescue across 6 canonical prompts.
- Residual-strength attenuation supports a causal dose response.
- Same-norm residual geometry causally changes model fidelity.
- Same-norm single-pulse perturbations exhibit direction-dependent recurrent propagation.
- Natural R128 orthogonal residual direction is low-gain biased in the direction sensitivity panel.
- Readout-aware single-pulse pilot improves correlation over raw state persistence, but remains inconclusive.

No final method is proposed yet.

## Repository Layout

```text
experiments/          Canonical experiment scripts
results/              Compact final JSON and protocol/stage0 JSON
reports/              Final Markdown reports
docs/                 Experiment index, evidence map, status, prompt manifest
```

Large raw token/layer/head traces, model checkpoints, logs, and dataset copies are intentionally excluded.

## Reproducibility Note

The repository currently contains research scripts and compact experimental evidence from an active mechanism-validation project. Some scripts retain assumptions from the original experimental environment and may require local model/data path configuration.

Large raw token/layer/head traces and model checkpoints are not stored in Git.
