# GDN INT8 Value Geometry Transfer Validity and Retest V1

## 1. Executive Summary

{
  "NUMERICAL_ROOT_CAUSE": "FP32/GPU SVD reconstruction error was previously measured as if it were pure SVD identity; CPU FP64 pure SVD separates the mathematical identity from precision/cast error.",
  "PURE_SVD_RECONSTRUCTION_GATE": "PASS",
  "BEHAVIORAL_SELF_IDENTITY_GATE": "RAW_REPORTED_NO_PREEXISTING_TOLERANCE",
  "SVD_TRANSFER_IDENTIFIABILITY_GATE": "PASS",
  "TRANSFER_NUMERICAL_VALIDITY_GATE": "PASS",
  "TRANSFER_CONSTRUCTION_USED": "SVD_VALUE_TRANSFER",
  "C2_VALID_PILOT_UNITS": 3,
  "R_TO_CV_SAFER_COUNT": 3,
  "C_TO_RV_MORE_HARMFUL_COUNT": 3,
  "BIDIRECTIONAL_POSITIVE_COUNT": 3,
  "C2_PILOT_GATE": "PASS",
  "C2_FORMAL_RUN": "YES",
  "R_TO_CV_FORMAL_COUNT": 9,
  "C_TO_RV_FORMAL_COUNT": 9,
  "BIDIRECTIONAL_FORMAL_COUNT": 9,
  "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "FORMAL_SUPPORTED",
  "VALUE_AXIS_ORIENTATION_TRANSFER_SIGNAL": "NOT_RUN",
  "FINAL_MECHANISM_CLASSIFICATION": "SPECIFIC_VALUE_GEOMETRY_HARMFULNESS_TRANSFER_SUPPORTED",
  "MECHANISM_CLOSURE_CANDIDATE": "YES_CANDIDATE",
  "METHOD_DESIGN_READY": "NO",
  "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
  "output_dir": "/data/zypan/runs/gdn_int8_value_geometry_transfer_validity_and_retest_v1",
  "git_commit_start": "122cc30fd299fb433f47f21e17eac75b48909015"
}

## 2. Prior Formal Value-Side Evidence

C1 remains canonical: VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL = FORMAL_SUPPORTED.

## 3. Why Previous C2 Was Invalid

Previous C2 counted FP32/GPU SVD reconstruction error against the pure mathematical SVD identity gate, producing 0/3 valid units despite raw 3/3 direction consistency.

## 4. Numerical Root-Cause Audit

See `numerical_audit_per_unit.csv` and `numerical_audit_summary.json`.

## 5. Dtype / SVD / Vh Audit

PyTorch `torch.linalg.svd` returns `Vh = V^H`; reconstruction uses `U @ diag(S) @ Vh`. FP64 CPU and FP32 paths are reported separately.

## 6. Tensor Self-Reconstruction Identity

See `self_reconstruction_tensor_identity.csv`.

## 7. Behavioral Self-Reconstruction Identity

See `self_reconstruction_behavior_identity.csv`; no new loose tolerance was invented.

## 8. Singular Spectrum / Identifiability Audit

See `svd_spectrum_audit.json` and `svd_identifiability_summary.json`.

## 9. Transfer Construction

Construction used: `SVD_VALUE_TRANSFER`.

## 10. Transfer Invariance Checks

See `transfer_construction_audit.csv` or `procrustes_audit.csv`.

## 11. SAME 3-Unit Pilot Retest

Same pilot units from the prior run were reused without reselection.

## 12. R -> C Value Transfer

Pilot safer count: 3

## 13. C -> R Value Transfer

Pilot more harmful count: 3

## 14. Bidirectional Pilot Gate

PASS

## 15. 9-Unit Formal Transfer

YES

## 16. Basis-Invariant Procrustes Fallback

NOT_RUN

## 17. Positive Results

Valid positive results are reported only from the selected construction after numerical gates.

## 18. Negative / Corrective Results

Previous raw 3/3 is retained as a debugging observation, not scientific evidence.

## 19. What Previous Invalid Raw 3/3 Means

It motivated the audit but is not combined into the retest counts.

## 20. Final Mechanism Classification

SPECIFIC_VALUE_GEOMETRY_HARMFULNESS_TRANSFER_SUPPORTED

## 21. What Is Still Not Proven

No downstream-sensitive Value directions are identified; no method design is ready.

## 22. Recommended Next Experiment

Only if transfer is strongly supported: `GDN_INT8_VALUE_DIRECTION_DOWNSTREAM_SENSITIVITY_AUDIT_V1`. Not run here.
