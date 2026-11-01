# Result Summary

This summary is a compact guide to the evidence chain. Full compact JSON and Markdown reports are stored under `results/` and `reports/`.

| Area | Experiment | Status | Key Result | Limitation |
|---|---|---|---|---|
| Phenomenon | INT8 row E2E completion | COMPLETE | INT8 orientation is a first-order variable; R128 degradation is much stronger than C128 in the tested setting. | Small public evidence bundle excludes raw traces and benchmark text. |
| Mechanism | Orientation state change | COMPLETE | State-change metrics separate row and column behavior under the canonical continuation protocol. | Uses retokenized FP_STATE decoded responses, not exact original generation IDs. |
| Negative result | Axis geometry rescue | NEGATIVE/INCONCLUSIVE | Finer grouping rescues KL, but same-codebook lost-update metrics do not explain the rescue. | Does not establish a final causal chain from dynamic range to quality. |
| Metric | Effective-update audit | COMPLETE | Effective-update metrics are better aligned with KL rescue than raw same-codebook metrics. | Metric audit only; not a method proposal. |
| Validation | Effective-update multi-prompt validation | COMPLETE | R128 to R16/C16 improves KL and effective-update metrics across 6 prompts. | Prompt set is canonical and small. |
| Causal | Residual strength | SUPPORTED | Residual-strength attenuation produces a KL dose response. | Intervention is controlled and diagnostic, not a production quantizer. |
| Causal | Residual geometry | SUPPORTED | Same residual magnitude with different direction changes model fidelity. | Geometry intervention does not by itself explain natural long-horizon accumulation. |
| Dynamics | Single-pulse residual propagation | SUPPORTED | Same-norm pulse perturbations show direction-dependent damping, persistence, and amplification. | Single-pulse setting only. |
| Dynamics | Direction sensitivity panel | SUPPORTED | Random orthogonal directions show broad recurrent directional sensitivity. | `PROPAGATION_FIDELITY_LINK = INCONCLUSIVE`; natural R128 direction is low-gain biased. |
| Readout | Readout-aware propagation audit | INCONCLUSIVE | Readout-aware metrics improve over raw state persistence in pilot but do not meet the formal-run gate. | Formal readout-aware run was not started because pilot threshold was not met. |

## Latest Readout-Aware Pilot

```text
rho_state           = 0.0882
rho_frozen_readout  = 0.3500
rho_actual_readout  = 0.3000
rho_postproj        = 0.2353
rho_residual_stream = 0.1059
```

The result is inconclusive, not positive.

## Current Status

```text
Current phase = Mechanism Validation
MECHANISM_CLOSURE_CANDIDATE = NO
METHOD_DESIGN_READY_CANDIDATE = NO
METHOD_DESIGN_READY = NO
```
