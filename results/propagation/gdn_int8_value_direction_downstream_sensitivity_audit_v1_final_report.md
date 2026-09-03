# GDN INT8 Value Direction Downstream Sensitivity Audit V1

## 1. Executive Summary

{
  "STAGE_D0": "PASS",
  "DIRECTIONAL_SENSITIVITY_PILOT": "PASS",
  "D2_FORMAL_RUN": "YES",
  "R_DIRECTION_MORE_SENSITIVE_UNITS": 3,
  "DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL": "NOT_SUPPORTED",
  "D3_RUN": "NOT_RUN_DUE_TO_GATE",
  "R_TO_CV_SENSITIVITY_RESCUE_UNITS": 0,
  "C_TO_RV_SENSITIVITY_GAIN_UNITS": 0,
  "BIDIRECTIONAL_SENSITIVITY_TRANSFER_UNITS": 0,
  "TRANSFER_SENSITIVITY_MEDIATION_SIGNAL": "NOT_RUN_DUE_TO_GATE",
  "OBSERVABILITY_ONLY_EXPLANATION": "INCONCLUSIVE",
  "FINAL_MECHANISM_CLASSIFICATION": "VALUE_GEOMETRY_TRANSFER_SUPPORTED_BUT_LOCAL_DIRECTIONAL_SENSITIVITY_NOT_SUPPORTED",
  "MECHANISM_CLOSURE_CANDIDATE": "YES_CANDIDATE",
  "METHOD_DESIGN_READY": "NO",
  "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
  "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "FORMAL_SUPPORTED",
  "output_dir": "/data/zypan/runs/gdn_int8_value_direction_downstream_sensitivity_audit_v1",
  "git_commit_start": "40cf04b7cca9658ff1c8052106167b9a409f4634"
}

## 2. Current Formal Mechanism Evidence

Value-side functional direction and specific value-geometry transfer remain FORMAL_SUPPORTED.

## 3. Scientific Question

Does matched-amplitude normalized R readout direction cause larger downstream KL than C, and does validated value-geometry transfer move that sensitivity?

## 4. Hook / Tensor Semantics

{
  "batch_dimension": 0,
  "head_dimension": 2,
  "head_scope": "all heads, preserving per-head Value dimension",
  "layer_scope": "all canonical GDN layers",
  "location": "Qwen3.5 linear_attn recurrent core output immediately after recurrent-state readout and before gated RMSNorm and out_proj",
  "relative_to_projection": "before head concatenation/out_proj",
  "tensor_shape": "[B, T=1, H=32, V=128] per GDN layer",
  "token_dimension": 1,
  "value_dimension": 3
}

## 5. Stage D0 Identity

{
  "STAGE_D0": "PASS",
  "clean_KL_zero_perturb": 0.0,
  "direction_norm_smoke": 0.999999995017799,
  "epsilon_smoke": 0.03811708787779006,
  "hook_semantics_file": "stageD0_hook_semantics.json",
  "selected_offset_smoke": 1
}

## 6. Readout-Direction Construction

`z_t(E)=E_t^T q_t`; directions are normalized before applying the shared epsilon.

## 7. Token Selection

Top-3 offsets by pooled original readout energy only.

## 8. Matched-Amplitude Perturbation Protocol

Each selected token uses `epsilon=sqrt((||z_R||^2+||z_C||^2)/2)` and symmetric +/- injection.

## 9. Stage D1 Pilot

{
  "DIRECTIONAL_SENSITIVITY_PILOT": "PASS",
  "R_DIRECTION_MORE_SENSITIVE_UNITS": 2,
  "median_directional_sensitivity_R_over_C": 1.0868305300614383,
  "valid_units": 3
}

## 10. Pilot Gate

PASS

## 11. Stage D2 Formal Original R vs C

{
  "D2_FORMAL_RUN": "YES",
  "DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL": "NOT_SUPPORTED",
  "R_DIRECTION_MORE_SENSITIVE_UNITS": 3,
  "median_directional_sensitivity_R_over_C": 0.9296820235722136,
  "valid_units": 9
}

## 12. Downstream Directional Sensitivity Result

NOT_SUPPORTED

## 13. Stage D3 Value-Geometry Transfer Sensitivity

{
  "BIDIRECTIONAL_SENSITIVITY_TRANSFER_UNITS": 0,
  "C_TO_RV_SENSITIVITY_GAIN_UNITS": 0,
  "D3_RUN": "NOT_RUN_DUE_TO_GATE",
  "R_TO_CV_SENSITIVITY_RESCUE_UNITS": 0,
  "TRANSFER_SENSITIVITY_MEDIATION_SIGNAL": "NOT_RUN_DUE_TO_GATE",
  "gate_block": "DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL != FORMAL_SUPPORTED"
}

## 14. R -> C_V Sensitivity Rescue

0

## 15. C -> R_V Sensitivity Gain

0

## 16. Behavioral Transfer vs Sensitivity Transfer

D3 tests whether the already validated behavioral transfer is mirrored by matched-amplitude directional sensitivity.

## 17. Observability x Direction Sensitivity

See `secondary_observability_sensitivity_analysis.csv`.

## 18. Positive Results

Reported at unit level only.

## 19. Negative / Corrective Results

No random direction search, Jacobian, Hessian, or method design was run.

## 20. Mechanism Interpretation

VALUE_GEOMETRY_TRANSFER_SUPPORTED_BUT_LOCAL_DIRECTIONAL_SENSITIVITY_NOT_SUPPORTED

## 21. What Is Still Not Proven

No generalizable downstream-sensitive Value subspace has been established.

## 22. Mechanism Readiness

METHOD_DESIGN_READY = NO.

## 23. Recommended Next Experiment

If continuing: `GDN_INT8_VALUE_SENSITIVE_SUBSPACE_GENERALIZATION_AUDIT_V1`. Not run here.
