# GDN INT8 K-Space Compute-Invariant Rotation Audit V1

## 1. Scientific Question
Can FP-equivalent K-space basis changes naturally generate matched R128 residual counterfactuals with different future-key interaction?

## 2. Prior Evidence
Frozen path signal is FUTURE_KEY_INTERACTION_SIGNAL; direct key surgery was CONSTRUCTION_NOT_CLEAN_ENOUGH.

## 3. Why Direct Residual Surgery Failed
Previous controlled residual surgery preserved controls but could not manipulate J_key enough.

## 4. Compute-Invariant Quantization-Basis Probe
Phase A: 22-rotation shadow + frozen screening only

## 5. Rotation Definitions
{
  "IDENTITY": "U=I canonical R128",
  "HADAMARD": "normalized Sylvester Hadamard on K axis",
  "RANDOM_ORTHOGONAL": "16 deterministic QR orthogonal matrices, seeds 1000..1015",
  "KEY_PCA_ALIGNED": "per-layer/head future FP key covariance eigenbasis",
  "KEY_PCA_SPREAD": "KEY_PCA_ALIGNED @ HADAMARD",
  "STATE_PCA_ALIGNED": "per-layer/head snapshot state K covariance eigenbasis",
  "STATE_PCA_SPREAD": "STATE_PCA_ALIGNED @ HADAMARD"
}

## 6. Stage-0 Gates
```json
{
  "FP_ROTATION_IDENTITY_GATE": "PASS",
  "FROZEN_DRIVER_REUSE_GATE": "PASS",
  "FROZEN_METRIC_REPRODUCTION_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "ORTHOGONALITY_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "R128_QUANTIZER_IDENTITY_GATE": "PASS",
  "ROTATION_AXIS_GATE": "PASS",
  "STATE_SEMANTICS_GATE": "PASS"
}
```

## 7. Smoke
PASS

## 8. Rotation-Induced Residual Distribution
Median residual-norm spread: 1.2218485666585448

## 9. Rotation-Induced Scale Statistics
Scale statistics are stored per rotation/layer/head in JSON.

## 10. Frozen State Persistence
Median J_state spread: 1.002879953626822

## 11. Future-Key Interaction
Median J_key spread: 1.0016693819673486

## 12. Future-Query Observability
Median J_query spread: 1.0050183537896247

## 13. Rotation-Manifold Coupling
Pair search tests whether norm, J_state, and value profile can be matched while J_key differs.

## 14. Counterfactual Pair Search
Primary matched pairs found: 0 / 9; secondary: 0 / 9.

## 15. Counterfactual Feasibility Gate
FAIL

## 16. Behavioral Pilot
NOT_RUN_DUE_TO_FEASIBILITY_GATE

## 17. Matched-Pair KL Results
Not run unless Phase B is explicitly reached.

## 18. Structured vs Random Rotations
See per-unit rotation table in JSON.

## 19. R128 vs C128 Reference
C128 reference metrics are stored per unit.

## 20. Layer/Head Heterogeneity
Per-layer/head rotation statistics are stored in JSON.

## 21. Pilot Classification
ROTATION_MANIFOLD_DOES_NOT_CLEANLY_DECOUPLE_KEY

## 22. What Is Supported
KSPACE_QUANTIZATION_BASIS_SENSITIVITY = INCONCLUSIVE

## 23. What Is NOT Supported
- No behavioral KL was run before counterfactual feasibility gate.
- No method design or formal expansion was run.

## 24. Negative / Corrective Results
No deployable rotation method is implied.

## 25. Next Recommended Experiment
FP_DRIVER_CLAMP_VS_FULL_FEEDBACK or mechanism synthesis
