# Protocol

This document records the shared protocol assumptions used across the current GDN recurrent-state quantization experiments.

## Model

```text
Qwen3.5-9B
```

The relevant GDN recurrent state has shape:

```text
[B,H,K,V] = [1,32,128,128]
```

## Axis Naming

```text
row    = Key-axis row group
column = Value-axis column group

R128 = full 128-element Value-axis row group
C128 = full 128-element Key-axis column group
```

## Execution Pattern

The common mechanism protocol is:

```text
FP32 prefill
+
quantized or intervened recurrent continuation
```

Formal intervention experiments use teacher-forced continuation so that model-state changes can be compared under the same continuation tokens.

## Quantization And Intervention Scope

- State quantization targets GDN recurrent state tensors, not model weights.
- Causal interventions alter recurrent-state residuals or pulse geometry at specified token indices.
- Single-pulse experiments inject one state perturbation at `t0`, then continue without repeated injection.
- Method design, mixed precision, and new quantizer construction are not part of the current mechanism-validation phase.

## Trajectory Source

The canonical 6-prompt mechanism experiments currently use:

```text
continuation_source = P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED
exact_original_generation_token_ids_available = false
exact_replay_claimed = false
```

This is a valid controlled continuation source for mechanism diagnostics, but it is not exact replay of original generation token IDs.

## Current Gate Policy

The project remains in mechanism validation:

```text
MECHANISM_CLOSURE_CANDIDATE = NO
METHOD_DESIGN_READY_CANDIDATE = NO
METHOD_DESIGN_READY = NO
```
