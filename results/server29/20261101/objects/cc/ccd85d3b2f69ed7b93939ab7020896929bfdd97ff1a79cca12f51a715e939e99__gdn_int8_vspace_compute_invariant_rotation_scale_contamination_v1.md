# GDN INT8 V-Space Compute-Invariant Rotation Scale-Contamination V1

## 1. Scientific Question
Does FP-identity Value-space rotation reduce R128 row range imbalance, residual magnitude, and potentially behavioral damage?

## 2. Prior Evidence
R128 residual magnitude and repeated accumulation are supported; K-space rotation did not cleanly decouple key geometry.

## 3. K-Space Negative Result
K-space rotation produced only small frozen metric variation and no matched counterfactual pairs.

## 4. Cross-Value Scale-Contamination Hypothesis
R128 rows share one scale across Value coordinates.

## 5. V-Space Compute-Identity Intervention
Phase A source-side V-space shadow quantization screening; no behavioral branch unless SOURCE_RESCUE_GATE PASS

## 6. Rotation Definitions
{
  "IDENTITY": "canonical R128",
  "V_PERMUTATION": "deterministic reversed Value permutation negative control",
  "V_HADAMARD": "normalized Hadamard on Value axis",
  "RANDOM_V_ORTHOGONAL": "16 deterministic QR orthogonal rotations, seeds 2000..2015",
  "V_PCA_ALIGNED": "per-layer/head Value covariance eigenbasis",
  "V_PCA_SPREAD": "V_PCA_ALIGNED @ V_HADAMARD"
}

## 7. Stage-0 Gates
```json
{
  "C128_REFERENCE_GATE": "PASS",
  "FP_V_ROTATION_IDENTITY_GATE": "PASS",
  "FROZEN_PATH_REPRODUCTION_GATE": "PASS",
  "IDENTITY_RESIDUAL_REPRODUCTION_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "ORTHOGONALITY_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "R128_QUANTIZER_IDENTITY_GATE": "PASS",
  "STATE_SEMANTICS_GATE": "PASS",
  "V_PERMUTATION_EQUIVARIANCE_GATE": "PASS",
  "V_ROTATION_AXIS_GATE": "PASS"
}
```

## 8. Smoke
PASS

## 9. Row Peak / Range Statistics
Identity median peak/RMS: 7.435234069824219; V-Hadamard: 2.4671576023101807; V-PCA-Spread: 1.771932601928711

## 10. Scale Distribution
Scale summaries are stored per rotation.

## 11. Natural R128 Residual
Identity median residual: 4.708525001935389; V-Hadamard: 0.7395477365965671; V-PCA-Spread: 0.5868494900015059

## 12. C128 Reference
C128 median residual: 2.5293681024721377

## 13. R128-to-C128 Gap Closure
Hadamard: 1.807197515177012; PCA-Spread: 1.8763870399303753

## 14. Normalized Frozen-Path Diagnostics
Hadamard G_key/Identity: 41.02815341381757; PCA-Spread: 65.31027678438025

## 15. Layer / Head Heterogeneity
Stored in JSON.

## 16. Original-Value Victim Analysis
Per-original-Value energy is stored after inverse rotation.

## 17. Source Rescue Gate
VSPACE_SOURCE_RESCUE=SUPPORTED; SOURCE_RESCUE_GATE=PASS

## 18. Behavioral Single-Pulse Pilot
NOT_RUN_YET

## 19. Norm-Restored Mediation Control
Only run if Phase B runs.

## 20. Per-Prompt / Per-t0 Results
See JSON.

## 21. Pilot Classification
SOURCE_RESCUE_READY_FOR_BEHAVIORAL_PILOT

## 22. What Is Supported
VSPACE_SOURCE_RESCUE = SUPPORTED in Phase A; behavioral mediation pilot is allowed.

## 23. What Is NOT Supported
- No formal expansion was run.
- No method design was started.
- Hadamard/PCA rotations are oracle mechanism controls, not proposed methods.

## 24. Negative / Corrective Results
No method design is implied.

## 25. Next Recommended Experiment
Phase B single-pulse behavioral mediation pilot
