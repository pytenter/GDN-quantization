# GDN INT8 Value-Basis Symmetry-Breaking Localization V1

## 1. Executive Summary

{
  "STAGE_F0": "PASS",
  "CORE_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "READOUT_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "F1_VALID_UNITS": 9,
  "F1_VALID_ROTATIONS": 36,
  "F1_MAX_CORE_EQUIVARIANCE_ERROR": 1.2508366628227047e-15,
  "F1_MAX_READOUT_EQUIVARIANCE_ERROR": 1.7275758027374494e-15,
  "F1_ACTUAL_DTYPE_RAW_CORE_EQUIVARIANCE_ERROR": 0.0001250745114876695,
  "F1_ACTUAL_DTYPE_RAW_READOUT_EQUIVARIANCE_ERROR": 2.4715985004651433,
  "POSTCORE_VALID_STAGES": 6,
  "EARLIEST_TRANSFER_CONSISTENT_STAGE": "rms_normalized_core",
  "PRIMARY_BREAKING_OPERATOR": "rms_normalized_core",
  "R_TRANSFER_RESCUE_AT_EARLIEST_STAGE": 9,
  "C_TRANSFER_GAIN_AT_EARLIEST_STAGE": 9,
  "BIDIRECTIONAL_AT_EARLIEST_STAGE": 9,
  "VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION": "FORMAL_SUPPORTED",
  "SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY": "NOT_SUPPORTED",
  "FINAL_MECHANISM_CLASSIFICATION": "VALUE_GEOMETRY_TRANSFER_SUPPORTED_AND_POSTCORE_VALUE_BASIS_BREAKING_LOCALIZED",
  "MECHANISM_CLOSURE_CANDIDATE": "STRONG_YES_CANDIDATE",
  "METHOD_DESIGN_READY": "NO",
  "output_dir": "/data/zypan/runs/gdn_int8_value_basis_symmetry_breaking_localization_v1",
  "git_commit_start": "b18503e19b3bb3515a5e99d2111be07429890c06"
}

## 2. Prior Formal Evidence

Value-side functional direction and Value-geometry bidirectional transfer are inherited as FORMAL_SUPPORTED. The previous single-token matched-amplitude downstream sensitivity result remains NOT_SUPPORTED.

## 3. Core GDN/KDA Formula Motivation

The frozen recurrent core should carry right-side Value rotations equivariantly: `E_Q(t) = E(t)Q`; under the row-vector implementation convention readout obeys `z_Q(t) = z(t)Q`.

## 4. Actual Qwen3.5 Operator Path

See `postcore_stage_map.json`.

## 5. Stage F0 Identity

{
  "CORE_HOOK_IDENTITY": "PASS",
  "POSTCORE_DECOMPOSITION_IDENTITY": "PASS",
  "STAGE_F0": "PASS",
  "TENSOR_SEMANTICS_GATE": "PASS",
  "config_file": "config.json",
  "gates": {
    "CANONICAL_9_UNITS_RECOVERED": true,
    "CORE_HOOK_IDENTITY": true,
    "INSTRUMENTATION_NONINTERFERENCE": true,
    "POSTCORE_DECOMPOSITION_IDENTITY": true,
    "PRIOR_DOWNSTREAM_SINGLE_TOKEN_NEGATIVE_RECOVERED": true,
    "PRIOR_TRANSFER_FORMAL_SUPPORTED": true,
    "TENSOR_SEMANTICS_GATE": true
  },
  "max_core_hook_identity_rel_error": 0.0,
  "max_postcore_output_identity_rel_error": 0.0,
  "max_shadow_norm_identity_rel_error": 0.0
}

## 6. Stage F1 Core Value Equivariance

{
  "CORE_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "F1_VALID_ROTATIONS": 36,
  "F1_VALID_UNITS": 9,
  "READOUT_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "actual_dtype_raw_core_equivariance_error": 0.0001250745114876695,
  "actual_dtype_raw_readout_equivariance_error": 2.4715985004651433,
  "expected_rotations": 36,
  "expected_units": 9,
  "max_core_equivariance_error": 0.0001250745114876695,
  "max_pure_fp64_core_equivariance_error": 1.2508366628227047e-15,
  "max_pure_fp64_readout_equivariance_error": 1.7275758027374494e-15,
  "max_readout_equivariance_error": 2.4715985004651433,
  "max_replay_identity_error": 0.0
}

## 7. Readout Equivariance

Reported in `stageF1_core_equivariance_per_run.csv`; max readout error is in final summary.

## 8. Post-Core Prefix Stage Map

raw core -> RMS normalize -> learned norm weight -> dynamic SiLU(z) gate -> head merge -> out_proj.

## 9. Whole-Trajectory Matched-Amplitude Protocol

For every valid future token, four directions use the same `epsilon_t` from original R/C and symmetric `+/-` prefix perturbation energy.

## 10. Stagewise R/C Comparison

|idx|stage|R>C|R->C_V rescue|C->R_V gain|bidirectional|
|-:|-|-:|-:|-:|-:|
|0|raw_core_output|5/9|0/9|5/9|0/9|
|1|rms_normalized_core|0/9|9/9|9/9|9/9|
|2|learned_norm_weight|0/9|9/9|9/9|9/9|
|3|dynamic_silu_z_gate|0/9|9/9|9/9|9/9|
|4|head_merge_flatten|0/9|9/9|9/9|9/9|
|5|out_proj_hidden_contribution|0/9|9/9|9/9|9/9|

## 11. Value-Geometry Transfer Localization

{
  "POSTCORE_VALID_STAGES": 6,
  "EARLIEST_TRANSFER_CONSISTENT_STAGE": "rms_normalized_core",
  "PRIMARY_BREAKING_OPERATOR": "rms_normalized_core",
  "R_TRANSFER_RESCUE_AT_EARLIEST_STAGE": 9,
  "C_TRANSFER_GAIN_AT_EARLIEST_STAGE": 9,
  "BIDIRECTIONAL_AT_EARLIEST_STAGE": 9,
  "VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION": "FORMAL_SUPPORTED",
  "stage_summary": [
    {
      "stage_index": 0,
      "stage_name": "raw_core_output",
      "R_GT_C_units": 5,
      "median_R_over_C": 1.0168600945176376,
      "R_to_CV_rescue_units": 0,
      "C_to_RV_gain_units": 5,
      "bidirectional_units": 0,
      "median_R_transfer_rescue": -0.3962529390307028,
      "median_C_transfer_gain": 0.015587219256684937
    },
    {
      "stage_index": 1,
      "stage_name": "rms_normalized_core",
      "R_GT_C_units": 0,
      "median_R_over_C": 0.14843857625888796,
      "R_to_CV_rescue_units": 9,
      "C_to_RV_gain_units": 9,
      "bidirectional_units": 9,
      "median_R_transfer_rescue": 426.68842660591963,
      "median_C_transfer_gain": 15404.941758442643
    },
    {
      "stage_index": 2,
      "stage_name": "learned_norm_weight",
      "R_GT_C_units": 0,
      "median_R_over_C": 0.22485431962734065,
      "R_to_CV_rescue_units": 9,
      "C_to_RV_gain_units": 9,
      "bidirectional_units": 9,
      "median_R_transfer_rescue": 416.43811839685236,
      "median_C_transfer_gain": 14868.142000013833
    },
    {
      "stage_index": 3,
      "stage_name": "dynamic_silu_z_gate",
      "R_GT_C_units": 0,
      "median_R_over_C": 0.39433517743457835,
      "R_to_CV_rescue_units": 9,
      "C_to_RV_gain_units": 9,
      "bidirectional_units": 9,
      "median_R_transfer_rescue": 576.9746908669082,
      "median_C_transfer_gain": 14540.882135307038
    },
    {
      "stage_index": 4,
      "stage_name": "head_merge_flatten",
      "R_GT_C_units": 0,
      "median_R_over_C": 0.39433517743457835,
      "R_to_CV_rescue_units": 9,
      "C_to_RV_gain_units": 9,
      "bidirectional_units": 9,
      "median_R_transfer_rescue": 576.9746908669082,
      "median_C_transfer_gain": 14540.882135307038
    },
    {
      "stage_index": 5,
      "stage_name": "out_proj_hidden_contribution",
      "R_GT_C_units": 0,
      "median_R_over_C": 0.600896510978606,
      "R_to_CV_rescue_units": 9,
      "C_to_RV_gain_units": 9,
      "bidirectional_units": 9,
      "median_R_transfer_rescue": 619.307206740071,
      "median_C_transfer_gain": 14060.50388618791
    }
  ]
}

## 12. Earliest Transfer-Consistent Stage

rms_normalized_core

## 13. Incremental Operator Contribution

See `stageF2_per_unit_stage.csv` incremental gain columns.

## 14. Dynamic Gate / Norm Weight Analysis

See `secondary_gate_weight_analysis.csv`. These are secondary descriptive metrics only.

## 15. Out-Projection Analysis

The out-projection prefix uses the actual merged head layout and actual `out_proj` module, not a standalone weight-norm proxy.

## 16. Relation to Previous Single-Token Negative

This audit does not revise the previous result: `SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY=NOT_SUPPORTED`.

## 17. Positive Results

Core and readout Value equivariance are reported above if F1 passes.

## 18. Negative / Corrective Results

No random directions, Jacobians, Hessians, new behavioral interventions, or method design were run.

## 19. Mechanism Interpretation

VALUE_GEOMETRY_TRANSFER_SUPPORTED_AND_POSTCORE_VALUE_BASIS_BREAKING_LOCALIZED

## 20. What Is NOT Proven

No quantization method, bit allocation rule, or causal neutralization has been established.

## 21. Mechanism Readiness

MECHANISM_CLOSURE_CANDIDATE = `STRONG_YES_CANDIDATE`; METHOD_DESIGN_READY = `NO`.

## 22. Recommended Next Experiment

If localization is positive, run but do not start here: `GDN_INT8_VALUE_BASIS_BREAKER_CAUSAL_NEUTRALIZATION_V1`.
