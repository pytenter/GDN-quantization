# HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1 Plan

Status: `READY_WITH_FP32_ROTATION_POLICY`. This document is a design only; no optimization or training is authorized by `ORTHOGONAL_NUMERICAL_CLOSURE_V1`.

## Objective

Measure learnable orthogonal headroom beyond the fixed H128 baseline while keeping every model weight frozen.

The candidate rotation is

`R_theta = DeltaR_theta @ H128`,

with `DeltaR_theta(initial) = I`, so `R_theta(initial) = H128` exactly within the selected numerical representation.

## Frozen and trainable state

- Freeze all Qwen and Ling model parameters.
- Freeze tokenizer, prompts, teacher-forced tokens, quantization settings, and rotation placement.
- Train only the 128x128 correction constrained to `SO(128)`.
- Do not introduce HARP, Butterfly, Householder stacks, or weight reparameterization in this oracle.

## Precision policy

- Store the trainable/master rotation parameter in FP32.
- Construct and apply `R_theta` and its explicit transpose in FP32.
- Qwen: preserve the current FP32 q/k rotation path.
- Ling: keep Value forward rotation and functional readout recovery in FP32, with an explicit cast only at the established native-model boundary.
- Do not silently change recurrent cache/storage dtype. Evaluate any FP32-state variant as a separately named ablation because numerical closure did not show uniformly better behavioral metrics from that change.
- Record the actual matrix, input, output, cache, and observable accumulation dtypes.

## Initialization gate

Before optimization, require the initialized oracle to reproduce canonical H128 within the same-harness numerical envelope:

1. FP64 operator identity passes.
2. Matrix determinant is positive and orthogonality residual is recorded.
3. Correct Qwen Key-side and Ling Value-side axes are unchanged.
4. Ling prefill endpoint continuity remains exact and redundant endpoint rotation remains `NO`.
5. Audit path remains side-effect free.
6. 3x128 and 1x512 teacher-forced errors are no worse than the numerical-closure H baseline by an order of magnitude.

## Optimization protocol to review later

- Use a mathematically explicit `SO(128)` parameterization or retraction; select it only after a separate implementation review.
- Log objective, orthogonality residual, determinant, gradient norm, and distance from H128.
- Maintain fixed validation prompts and a held-out teacher-forced trajectory.
- Compare H128, untrained oracle initialization, trained dense oracle, and identity.
- Stop on nonfinite values, determinant sign change, basis discontinuity, or cache semantic failure.

## Deliverables for the future stage

- A reviewed optimizer/parameterization design.
- Initialization-parity report.
- Compact train/validation metrics and rotation manifests.
- A final answer to: “How much learnable orthogonal headroom remains after Hadamard?”

No training is performed by this plan.
