# RECURRENT_FUTURE_AWARE_OBJECTIVE_FEASIBILITY_V1 Plan

This is a design-only artifact. It does not authorize or execute training.

## Candidate A: MULTI_STEP_LOCAL_TRAJECTORY

Use cached or streamed recurrent drivers and roll each frozen local replay forward for 16 tokens. Evaluate losses at horizons 1, 4, 8, and 16, with post-out-projection relative MSE as the primary target. Only rotation parameters would receive gradients in a future approved phase; the language model remains frozen. Stream one sequence at a time, retain only the active trajectory, and record VRAM, wall time, gradient norm, orthogonality residual, and numerical finiteness. This avoids full 9B-model BPTT.

## Candidate B: TRUE_FUTURE_KL_SHORT_BPTT

Run a feasibility-only probe at horizons 1 and 4 before any formal training. Measure peak VRAM, step time, whether a finite gradient reaches the rotation parameterization, OOM status, and numerical stability. Use teacher forcing, frozen model weights, FP32 master rotation parameters, and the model-specific canonical rotation semantics. No checkpoint selection or scientific claim is allowed from the feasibility probe.

## Decision gate

Prefer Candidate A when it preserves the intended functional boundary and gives stable finite gradients with bounded memory. Consider Candidate B only if the short-BPTT probe fits with safety margin and demonstrates a materially different, usable gradient signal. If neither is stable and affordable, retain Hadamard or the current frozen dense method; do not escalate to HARP, Butterfly, or per-head training without a separate protocol.
