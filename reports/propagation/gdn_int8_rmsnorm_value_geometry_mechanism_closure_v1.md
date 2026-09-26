# GDN INT8 RMSNorm Value Geometry Mechanism Closure V1

## 1. Executive Summary

{
  "task": "GDN_INT8_RMSNORM_VALUE_GEOMETRY_MECHANISM_CLOSURE_V1",
  "git_commit_start": "a3300c907a5b5206650dff1c256646736e1d89b2",
  "PROTOCOL_GATE": "PASS",
  "TENSOR_SEMANTICS_GATE": "PASS",
  "ROTATION_DOMAIN_GATE": "PASS",
  "OPERATOR_ORDER_GATE": "PASS",
  "ACTUAL_DTYPE_AUDIT_GATE": "PASS",
  "PURE_RMS_EQUIVARIANCE_GATE": "PASS",
  "DENOMINATOR_INTERVENTION_GATE": "PASS",
  "DIRECTION_SCALE_DECOMPOSITION_GATE": "PASS",
  "LEARNED_WEIGHT_AUDIT_GATE": "PASS",
  "DYNAMIC_GATE_AUDIT_GATE": "PASS",
  "METRIC_GATE": "PASS",
  "EARLIEST_TRANSFER_VISIBLE_STAGE": "rms_normalized_core",
  "EARLIEST_MATHEMATICALLY_NON_EQUIVARIANT_OPERATOR": "learned_norm_weight_under_full-vector_rotation; pure_rms_full-vector_equivariant; residual-only clean-context RMS denominator interaction is earliest functional breaker",
  "ACTUAL_DTYPE_EQUIVARIANCE_STATUS": "METRIC_SCALE_ARTIFACT",
  "PURE_RMS_OPERATOR_BREAKING": "NOT_SUPPORTED",
  "PURE_RMS_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "RMS_DENOMINATOR_MECHANISM": "SUPPORTED",
  "DENOMINATOR_INTERVENTION_CLASSIFICATION": "HARMFULNESS_FOLLOWS_RMS_DENOMINATOR",
  "POST_RMS_GEOMETRY_DRIVER": "DIRECTION_WITH_CLEAN_CONTEXT_DENOMINATOR_INTERACTION",
  "LEARNED_NORM_WEIGHT_BREAKING": "FORMAL_ONLY",
  "DYNAMIC_GATE_BREAKING": "FORMAL_ONLY",
  "FINAL_MECHANISM_CLASSIFICATION": "RMS_STAGE_TRANSFER_VISIBLE_DUE_TO_CLEAN_CONTEXT_DENOMINATOR_INTERACTION_NOT_FULL_VECTOR_RMS_BASIS_BREAKING",
  "MECHANISM_CLOSURE": "STRONG_CANDIDATE",
  "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
  "METHOD_DESIGN_READY": "NO",
  "output_dir": "/data/zypan/runs/gdn_int8_rmsnorm_value_geometry_mechanism_closure_v1"
}

## 2. Protocol / Tensor Semantics

{
  "OPERATOR_ORDER_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "ROTATION_DOMAIN_GATE": "PASS",
  "TENSOR_SEMANTICS_GATE": "PASS",
  "dtype_path": "core is cast to z/model dtype for fused RMSNormGated; RMS variance computed in fp32 in native implementation",
  "epsilon": 1e-06,
  "learned_norm_weight_shape": [
    128
  ],
  "num_value_heads": 32,
  "reshape_before_rms": "core_attn_out.reshape(-1, head_v_dim)",
  "rms_axis_domain": "last Value dimension V=128, independently for each head row after reshape",
  "rms_independent_per_head": true,
  "rotation_mixes_heads_or_groups": false,
  "rotations_inside_one_rms_group": true,
  "source_references": {
    "modeling_function": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward",
    "previous_f1": "/data/zypan/runs/gdn_int8_value_basis_symmetry_breaking_localization_v1/stageF1_core_equivariance_summary.json",
    "previous_stage_map": "/data/zypan/runs/gdn_int8_value_basis_symmetry_breaking_localization_v1/postcore_stage_map.json"
  },
  "tensor_shape_entering_core_attn_out": "[B,T,H,V]",
  "value_head_dimension": 128,
  "z_gate_shape": "[B,T,H,V] reshaped to [-1,V], multiplied elementwise after SiLU"
}

## 3. Actual-Dtype Readout Discrepancy

The prior `2.4715985004651433` value is the maximum relative-L2 readout equivariance error over the actual model dtype replay. FP64 algebra is at numerical precision; the large raw value is classified as a scale-sensitive metric artifact rather than a standalone real-dtype mechanism.

## 4. Pure RMS Equivariance

{
  "ACTUAL_DTYPE_EQUIVARIANCE_STATUS": "METRIC_SCALE_ARTIFACT",
  "PURE_RMS_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED",
  "PURE_RMS_OPERATOR_BREAKING": "NOT_SUPPORTED",
  "pure_rms_fp64_tolerance": 1e-09,
  "pure_rms_fp64_max_relative_L2": 2.0107379789411123e-10,
  "pure_rms_fp32_max_relative_L2": 2.0325879513263036e-07,
  "pure_rms_native_max_relative_L2": 0.002557664468728287,
  "LEARNED_NORM_WEIGHT_FORMAL_EQUIVARIANCE": "NOT_SUPPORTED",
  "learned_weight_max_relative_L2": 2.125035489334466,
  "DYNAMIC_GATE_FORMAL_EQUIVARIANCE": "NOT_SUPPORTED",
  "dynamic_gate_max_relative_L2": 9.906901638805028
}

## 5. Denominator / Direction Controls

{
  "C_den_bidirectional_units": 5,
  "DENOMINATOR_INTERVENTION_CLASSIFICATION": "HARMFULNESS_FOLLOWS_RMS_DENOMINATOR",
  "POST_RMS_GEOMETRY_DRIVER": "DIRECTION_WITH_CLEAN_CONTEXT_DENOMINATOR_INTERACTION",
  "RMS_DENOMINATOR_MECHANISM": "SUPPORTED",
  "R_den_bidirectional_units": 0,
  "fp_den_bidirectional_units": 0,
  "mode_summary": [
    {
      "C_to_RV_gain_units": 9,
      "R_to_CV_rescue_units": 9,
      "bidirectional_units": 9,
      "denominator_mode": "self_den",
      "median_C_transfer_gain": 3550.544259808141,
      "median_R_transfer_rescue": 181.28355872663747
    },
    {
      "C_to_RV_gain_units": 0,
      "R_to_CV_rescue_units": 0,
      "bidirectional_units": 0,
      "denominator_mode": "fp_den",
      "median_C_transfer_gain": -209.5883625121587,
      "median_R_transfer_rescue": -183.00809477806547
    },
    {
      "C_to_RV_gain_units": 0,
      "R_to_CV_rescue_units": 0,
      "bidirectional_units": 0,
      "denominator_mode": "R_den",
      "median_C_transfer_gain": -214.1602221899684,
      "median_R_transfer_rescue": -187.91026910880578
    },
    {
      "C_to_RV_gain_units": 9,
      "R_to_CV_rescue_units": 5,
      "bidirectional_units": 5,
      "denominator_mode": "C_den",
      "median_C_transfer_gain": 4636.414193605314,
      "median_R_transfer_rescue": 20.94557758994506
    }
  ],
  "self_den_bidirectional_units": 9
}

## 6. Formal Operator Equivariance

See `tableA_formal_operator_equivariance.csv`.

## 7. Functional Geometry Transfer

See `tableB_functional_geometry_transfer.csv` and Stage D denominator-swap tables.

## 8. Technical Answer

The recurrent core and readout carry right-side Value rotations equivariantly, so the R/C difference is not generated inside the recurrent linear update. Full-vector pure RMS is also Value-equivariant when the whole vector and the rotation live inside the same 128-dimensional per-head normalization group. The observed transfer becomes visible at the RMS prefix because the experiment perturbs a fixed clean activation by residual-only geometry; RMS uses the norm of `x_clean + delta`, so numerator direction and clean-context denominator interact before learned weight and gate are applied. Learned weight and dynamic gate are formally non-commuting coordinate-wise modulators, but they are not the earliest functional source because the 9/9 transfer-consistent signal is already present at pure RMS output.

## 9. Plain-Language Answer

The recurrent state update preserves the Value orientation. The difference appears when the residual is added to the actual clean readout and then normalized: RMS normalization itself would respect rotations if the entire vector were rotated, but here only the error is rotated while the clean signal stays fixed. That clean-context normalization makes R-like and C-like residual geometry separable. Later weight and gate layers can reshape the signal, but they are not where it first appears.

## 10. Negative Results

Pure RMS as a full-vector mathematical basis breaker is NOT supported. The previous single-token local directional sensitivity result remains NOT_SUPPORTED. No new quantizer or method was designed.

## 11. Mechanism Readiness

MECHANISM_CLOSURE = `STRONG_CANDIDATE`; METHOD_PRINCIPLE_EXTRACTION_READY = `YES`; METHOD_DESIGN_READY = `NO`.
