# GDN INT8 Value-Side Functional Direction Formal and Transfer V1

## 1. Executive Summary

{
  "STAGE0": "PASS",
  "C1_VALID_UNITS": 9,
  "C1_VALID_RUNS": 36,
  "V_ROTATION_GAP_DECREASE_RUNS": 35,
  "V_ROTATION_SIGN_INVERSIONS": 34,
  "V_ROTATION_POSITIVE_UNITS": 9,
  "C1_MEDIAN_RESCUE": 2.8416468042625542,
  "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
  "STAGE_C2_RUN": "YES",
  "C2_PILOT_STATUS": "NOT_SUPPORTED_OR_INCONCLUSIVE",
  "C2_FORMAL_RUN": "NO",
  "R_TO_CV_SAFETY_TRANSFER": "PILOT_ONLY",
  "C_TO_RV_HARM_TRANSFER": "PILOT_ONLY",
  "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": 0,
  "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "NOT_SUPPORTED_OR_INCONCLUSIVE",
  "FINAL_MECHANISM_CLASSIFICATION": "VALUE_SIDE_DIRECTION_FORMAL_SUPPORTED_TRANSFER_NOT_CLOSED",
  "MECHANISM_CLOSURE_CANDIDATE": "NO",
  "METHOD_DESIGN_READY": "NO",
  "git_commit_start": "25e97ad3fa827c13c8579f851c31c94cf5156013",
  "output_dir": "/data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1"
}

## 2. Prior Evidence

Prior evidence is inherited from `GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1`; Stage0 PASS, observability gate PASS, and V-axis pilot SUPPORTED.

## 3. Scientific Question

Does canonical same-norm R harmfulness travel through Value-side residual geometry, and can that geometry transfer harmfulness bidirectionally between R and C?

## 4. Repository Audit

Reused previous canonical 9 units, residual construction, full-model KL metric, exact injection hook, QR rotation construction, and replay identity infrastructure.

## 5. Tensor / Axis Semantics

State is `[B,H,K,V]`; row is Key axis and column is Value axis. C1 applies right-side orthogonal Value rotations `E Q_V` with the same `Q_V` for R and C.

## 6. Stage 0

{
  "STAGE0": "PASS",
  "canonical_units": [
    "test/algebra/1332.json|64",
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|256",
    "test/counting_and_probability/119.json|64",
    "test/counting_and_probability/119.json|128",
    "test/counting_and_probability/119.json|256",
    "test/geometry/477.json|64",
    "test/geometry/477.json|128",
    "test/geometry/477.json|256"
  ],
  "config": {
    "c1_gate": {
      "formal_supported": "UNIT_V_RESCUE_POSITIVE >= 8/9, median unit rescue > 0, no systematic invariance failure",
      "not_supported": "<=5/9 positive units",
      "partial": "6/9 or 7/9 positive units"
    },
    "c2_gate": {
      "formal_supported": "R_to_CV safer >= 8/9 and C_to_RV more harmful >= 8/9",
      "pilot_pass": "bidirectional positive >= 2/3 and no serious numerical confound"
    },
    "git_commit_start": "25e97ad3fa827c13c8579f851c31c94cf5156013",
    "horizon": 128,
    "invariance_thresholds": {
      "observability_rot_over_original_bounds": [
        0.5,
        2.0
      ],
      "persistence_rot_over_original_bounds": [
        0.8,
        1.25
      ],
      "singular_value_rel_error": 0.0001,
      "source_norm_rel_error": 0.0001
    },
    "output_dir": "/data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1",
    "previous_commit": "25e97ad3fa827c13c8579f851c31c94cf5156013",
    "previous_run_dir": "/data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1",
    "previous_task": "GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1",
    "rotation": {
      "axis": "V",
      "construction": "seeded Gaussian -> QR -> deterministic diagonal-sign convention",
      "same_Q_for_R_and_C": true,
      "seeds": [
        1101,
        1102,
        1103,
        1104
      ]
    },
    "task": "GDN_INT8_VALUE_SIDE_FUNCTIONAL_DIRECTION_FORMAL_AND_TRANSFER_V1",
    "tensor_semantics": {
      "column": "Value axis",
      "row": "Key axis",
      "state": "[B,H,K,V]"
    },
    "timestamp": "2026-09-03 15:09:02 +0800"
  },
  "gates": {
    "CANONICAL_9_UNITS_LOADED": true,
    "INJECTION_HOOK_REUSED_FROM_PREVIOUS_EXPERIMENT": true,
    "ORIGINAL_R_C_NORM_MATCH": true,
    "ORIGINAL_R_GT_C_9_OF_9": true,
    "PREVIOUS_OBSERVABILITY_GATE_PASS": true,
    "PREVIOUS_STAGE0_PASS": true,
    "PREVIOUS_V_PILOT_SUPPORTED": true,
    "V_ROTATION_ORTHOGONALITY_SMOKE": true
  },
  "max_original_norm_rel_error": 2.82423914692438e-09,
  "orthogonality_error_seed0": 5.6794556257955264e-06,
  "timestamp": "2026-09-03 15:09:05 +0800"
}

## 7. Stage C1 Protocol

All 9 canonical units were evaluated with four fixed V-rotation seeds: 1101, 1102, 1103, 1104.

## 8. V-Rotation Invariance Audit

{
  "invalid_runs": 0,
  "confounded_runs": 2,
  "systematic_invariance_failure": false
}

## 9. Stage C1 Formal Results

{
  "C1_MEDIAN_RESCUE": 2.8416468042625542,
  "C1_VALID_RUNS": 36,
  "C1_VALID_UNITS": 9,
  "STAGE_C1_RUN": "YES",
  "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
  "V_ROTATION_GAP_DECREASE_RUNS": 35,
  "V_ROTATION_POSITIVE_UNITS": 9,
  "V_ROTATION_SIGN_INVERSIONS": 34,
  "completed_runs": 36,
  "confounded_runs": 2,
  "expected_runs": 36,
  "failed": {},
  "invalid_runs": 0,
  "systematic_invariance_failure": false
}

## 10. Unit-Level Rescue Analysis

See `stageC1_per_unit.csv`.

## 11. Sign-Inversion Analysis

Sign inversions are secondary evidence and are not used as the formal gate.

## 12. Stage C1 Gate

FORMAL_SUPPORTED

## 13. Stage C2 SVD / Subspace Audit

SVD reconstruction and near-degenerate spectrum diagnostics are saved in `stageC2_*_svd_audit.json` when C2 runs.

## 14. Stage C2 Pilot

{
  "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": 0,
  "C2_FORMAL_RUN": "NO",
  "C2_PILOT_STATUS": "NOT_SUPPORTED_OR_INCONCLUSIVE",
  "C_TO_RV_HARM_TRANSFER": "PILOT_ONLY",
  "C_to_RV_more_harmful_count": 0,
  "R_TO_CV_SAFETY_TRANSFER": "PILOT_ONLY",
  "R_to_CV_safer_count": 0,
  "STAGE_C2_RUN": "YES",
  "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "NOT_SUPPORTED_OR_INCONCLUSIVE",
  "bidirectional_positive_count": 0,
  "units": [
    "test/counting_and_probability/119.json|256",
    "test/counting_and_probability/119.json|64",
    "test/algebra/1332.json|256"
  ],
  "valid_units": 0
}

## 15. Stage C2 Formal Results

Not run.

## 16. R->C Value Geometry Transfer

PILOT_ONLY

## 17. C->R Value Geometry Transfer

PILOT_ONLY

## 18. Bidirectional Transfer Analysis

0

## 19. Positive Results

C1 positive units and C2 transfer counts are reported only at unit level.

## 20. Negative / Corrective Results

Any C1 confounding or asymmetric C2 transfer is preserved as a scientific result.

## 21. Mechanism Interpretation

VALUE_SIDE_DIRECTION_FORMAL_SUPPORTED_TRANSFER_NOT_CLOSED

## 22. What Is NOT Proven

No downstream sensitive Value directions are identified here. No method design, bit allocation, quantizer, or deployment path is claimed.

## 23. Mechanism Readiness

MECHANISM_CLOSURE_CANDIDATE = NO; METHOD_DESIGN_READY = NO.

## 24. Recommended Next Experiment

Only if C2 is strongly positive, consider `GDN_INT8_VALUE_DIRECTION_DOWNSTREAM_SENSITIVITY_AUDIT_V1`; it is not run in this task.
