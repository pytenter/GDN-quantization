# Evidence Map

This document is a research evidence map, not a paper abstract and not a method proposal.

## A. Orientation / Representation

| Claim | Level | Evidence |
|---|---|---|
| R128/C128 orientation is a first-order variable. | OBSERVATION / SUPPORTED | Orientation and end-to-end screening results. |
| Canonical R128 quantizability is representation dependent. | SUPPORTED | V-space scale-contamination screen and behavioral pilot. |
| Cross-Value range / scale contamination is the strongest source-side explanation. | SUPPORTED | V-Hadamard and PCA-spread reduce residuals and rescue KL. |
| K-space coordinate rotation does not cleanly decouple future-key interaction. | NEGATIVE | `ROTATION_MANIFOLD_DOES_NOT_CLEANLY_DECOUPLE_KEY`. |

## B. Residual Magnitude

| Claim | Level | Evidence |
|---|---|---|
| Residual magnitude is causal. | CAUSAL | Natural R/C norm-swap: R shrink improves KL in 9/9 units. |
| Magnitude alone is insufficient. | CAUSAL / SUPPORTED | Same-norm R structure remains worse than C in 9/9 norm-swap units. |

## C. Temporal Accumulation

| Claim | Level | Evidence |
|---|---|---|
| Repeated recurrent quantization accumulates trajectory error. | FORMAL | Repeated accumulation formal: 18/18 units. |
| V-space source rescue attenuates temporal accumulation. | FORMAL | Source-to-temporal bridge: SOURCE_FORMAL, TEMPORAL_RESCUE_FORMAL, SOURCE_TO_TEMPORAL_BRIDGE all SUPPORTED. |

## D. Same-Norm Residual Structure

| Claim | Level | Evidence |
|---|---|---|
| Same-norm residual structure changes behavior. | CAUSAL / SUPPORTED | Natural residual norm-swap and residual-geometry causal experiments. |
| More harmful R residual can be less persistent. | SUPPORTED / PARADOX | Frozen observability: J_state R/C < 1 while J_key and full KL R/C > 1. |
| Head-wise S8 shows a weak positive sign but not strong robustness. | INCONCLUSIVE / CANDIDATE | FAST S8 8/9; subset audit PARTIAL with strong head-set heterogeneity. |

## E. Operator-Conditioned Diagnostics

| Claim | Level | Evidence |
|---|---|---|
| Native replay semantics are fixed. | SUPPORTED | Core-output replay root cause is dtype semantics; next-state and core-output replay errors are zero after correction. |
| Local operator coupling is real. | SUPPORTED | R U/E and consumed-energy fraction exceed C in 8/9 units. |
| One-step update transduction is the downstream causal mechanism. | NEGATIVE | R_STATE_ONLY improves KL vs R_FULL only 3/9; UPDATE_TRANSDUCTION_CAUSAL = NOT_SUPPORTED. |

## F. Important Negative Results

| Claim | Level | Evidence |
|---|---|---|
| J_key alone is a proven causal scalar. | NEGATIVE | Future-key causal construction is `CONSTRUCTION_NOT_CLEAN_ENOUGH`. |
| Single-head downstream operator feedback explains the gap. | NEGATIVE | Feedback path signal = NOT_SUPPORTED; R hidden/operator/state drift > C is approximately 0/9. |
| Simple multi-head scope amplification closes the mechanism. | INCONCLUSIVE / NEGATIVE | FAST scope growth is PARTIAL and primary interpretation is INCONCLUSIVE. |

## G. Open Mechanism Questions

- Which functional head identities make same-norm R structure harmful?
- Is the weak distributed R-bias approximately additive or direction-specific?
- Does the unresolved same-norm structure effect require cross-layer composition?
- How should any future method preserve the supported source-to-temporal pathway without overfitting to a diagnostic scalar?
