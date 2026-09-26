# GDN INT8 Same-Norm Functional Geometry Audit V1

## 1. Executive Summary

{
  "STAGE0": "PASS",
  "OBSERVABILITY_GATE": "PASS",
  "RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL": "SUPPORTED",
  "ADDRESSABILITY_DISSIPATION_INVERSION": "OBSERVED",
  "KEY_MEDIATED_DISSIPATION_SIGNAL": "INCONCLUSIVE",
  "STAGE_B_RUN": "YES",
  "KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL": "NOT_SUPPORTED",
  "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "SUPPORTED",
  "FINAL_MECHANISM_CLASSIFICATION": "VALUE_SIDE_FUNCTIONAL_DIRECTION_DOMINANT_PILOT_SIGNAL",
  "MECHANISM_CLOSURE_CANDIDATE": "NO",
  "METHOD_DESIGN_READY": "NO"
}

## 2. Scientific Question

Why can a same-norm R residual, which is less persistent in raw state space, produce larger behavioral damage than C?

## 3. Prior Evidence Used

Frozen exact-replay pilot: `/data/zypan/results/gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json`. Same-norm behavioral KL: `/data/zypan/results/gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_pilot.json`.

## 4. Repository / Implementation Audit

Reused Qwen3.5 GDN exact replay helpers from the frozen-observability and norm-swap experiments.

## 5. Tensor Semantics

State is treated as `[B,H,K,V]`; query observability uses exact core_readout_error from implementation replay rather than a paper-equation substitute.

## 6. Stage 0 - Replay Identity

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
  "corrected_native_core_output_replay_error": 0.0,
  "corrected_native_next_state_replay_error": 0.0,
  "gates": {
    "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    "PROTOCOL_GATE": "PASS",
    "REPLAY_IDENTITY_GATE": "PASS",
    "SOURCE_NORM_GATE": "PASS",
    "TENSOR_SEMANTICS_GATE": "PASS"
  },
  "git_commit": "c502b15a050814219c3d8bc0e9b1d2cb97f5c3a4",
  "max_prior_frozen_replay_identity_error": 0.0033362211907986103,
  "max_source_norm_match_error": 2.3186139603256868e-07,
  "model_path": "/data/zypan/modelscope_models/Qwen3.5-9B",
  "source_files": {
    "frozen": "/data/zypan/results/gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json",
    "native_stage0": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_stage0.json",
    "normswap": "/data/zypan/results/gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_pilot.json",
    "v2_stage0": "/data/zypan/results/gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_stage0.json"
  },
  "task": "GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1",
  "timestamp": "2026-09-03 13:14:04 +0800"
}

## 7. Stage A - Raw State Persistence

P_state R>C = 0/9; median R/C = 0.6664534169081278

## 8. Stage A - Whole-Horizon Query Observability

O_q R>C = 8/9; median R/C = 1.291245838715231

## 9. Stage A - Observability Efficiency

eta_q R>C = 9/9; median R/C = 2.0053845990687615

## 10. Paired R/C Analysis

{
  "Spearman_log_O_ratio_vs_log_KL_ratio": -0.2,
  "Spearman_log_P_ratio_vs_log_KL_ratio": 0.1,
  "Spearman_log_eta_ratio_vs_log_KL_ratio": -0.35
}

## 11. Stage A Gate Decision

{
  "OBSERVABILITY_GATE": "PASS",
  "RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL": "SUPPORTED"
}

## 12. Stage A2 - Addressability / Dissipation

{
  "ADDRESSABILITY_DISSIPATION_INVERSION": "OBSERVED",
  "J_key_R_gt_C": "9/9",
  "KEY_MEDIATED_DISSIPATION_SIGNAL": "INCONCLUSIVE",
  "inversion_count": "8/9",
  "median_J_key_R_over_C": 2.473200378223244,
  "note": "A2 reports signed exact-replay path energies. It is not used as standalone J_key causal proof."
}

## 13. Stage B - K-Axis Rotation

{
  "runs": 12,
  "median_G_original": 0.00016225074899729448,
  "median_G_rot": 0.0009312386980139108,
  "median_absolute_gap_change": 0.0006645732271441135,
  "median_rescue": -4.754866773579373,
  "gap_decreased_count": 0,
  "max_norm_error": 1.6644300887256335e-07,
  "max_sv_error": 1.1247917523814083e-05
}

## 14. Stage B - V-Axis Rotation

{
  "runs": 12,
  "median_G_original": 0.00016225074899729448,
  "median_G_rot": -0.00022266065115103255,
  "median_absolute_gap_change": -0.00041900359371621767,
  "median_rescue": 1.6648012792409284,
  "gap_decreased_count": 11,
  "max_norm_error": 1.500715653769838e-07,
  "max_sv_error": 2.6500376959423678e-05
}

## 15. Behavioral KL Rescue

Stored in `stageB_per_run.csv` if Stage B ran.

## 16. Positive Results

Stage A supports recurrent functional observability if O_q and eta_q satisfy the pre-registered paired sign criteria.

## 17. Negative / Corrective Results

Stage B pilot is treated only as pilot causal evidence; negative rotation results do not erase the Stage A observability signal.

## 18. Mechanism Classification

VALUE_SIDE_FUNCTIONAL_DIRECTION_DOMINANT_PILOT_SIGNAL

## 19. What Is NOT Proven

No final method, no additive mechanism, no standalone J_key causal scalar, and no mechanism closure are claimed.

## 20. Recommended Next Experiment

If continuing, use a larger pre-registered rotation/SVD-factor audit; do not start it automatically.
