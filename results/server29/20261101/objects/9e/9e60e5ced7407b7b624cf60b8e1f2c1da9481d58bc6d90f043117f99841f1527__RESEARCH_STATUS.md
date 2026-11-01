# Research Status

PROJECT_STAGE = `MECHANISM_VALIDATION`

Current model: `Qwen3.5-9B`

Latest completed canonical task: `GDN_INT8_HEADWISE_S8_SUBSET_ROBUSTNESS_AUDIT_V1`

MECHANISM_CLOSURE_CANDIDATE = `NO`

METHOD_DESIGN_READY = `NO`

## High-Level Status

| Signal | Status |
|---|---|
| SOURCE_TO_TEMPORAL_BRIDGE | SUPPORTED |
| R128_REPEATED_ACCUMULATION_FORMAL | SUPPORTED |
| RESIDUAL_STRENGTH_CAUSAL_SUPPORT | YES |
| RESIDUAL_GEOMETRY_CAUSAL_SUPPORT | YES |
| FROZEN_PATH_SIGNAL | FUTURE_KEY_INTERACTION_SIGNAL |
| FUTURE_KEY_CAUSAL_INTERVENTION | CONSTRUCTION_NOT_CLEAN_ENOUGH |
| KSPACE_ROTATION_COUNTERFACTUAL | ROTATION_MANIFOLD_DOES_NOT_CLEANLY_DECOUPLE_KEY |
| LOCAL_OPERATOR_COUPLING_SIGNAL | SUPPORTED |
| UPDATE_TRANSDUCTION_CAUSAL | NOT_SUPPORTED |
| FEEDBACK_PATH_SIGNAL | NOT_SUPPORTED |
| FAST_MULTIHEAD_SCOPE_GROWTH | PARTIAL / INCONCLUSIVE |
| S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL | PARTIAL |
| DISTRIBUTED_WEAK_R_BIAS | CANDIDATE |
| HEAD_SET_HETEROGENEITY | STRONG |

## Strongest Supported Pathway

```text
representation-dependent source error
-> per-step residual dose
-> repeated recurrent exposure
-> trajectory accumulation
-> behavioral degradation
```

## Core Unresolved Phenomenon

At the same residual norm, R-like residual structure can be less persistent but more behaviorally harmful. Current evidence does not close the functional mechanism behind that structure effect.

## Eliminated Or Insufficient Simple Explanations

- Persistence alone is insufficient.
- J_key is not a proven standalone causal scalar.
- One-step update transduction U does not explain the behavioral gap.
- Single-head downstream operator feedback is not supported.
- Simple single-layer multi-head scope amplification remains inconclusive.

No method design is ready.
