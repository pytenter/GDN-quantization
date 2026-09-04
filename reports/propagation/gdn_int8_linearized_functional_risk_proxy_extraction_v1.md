# GDN_INT8_LINEARIZED_FUNCTIONAL_RISK_PROXY_EXTRACTION_V1

## 1. Research Question

Can a mechanism-derived, linearized functional risk proxy predict future full-logit KL better than raw recurrent-state reconstruction error?

## 2. Prior Causal Evidence

The previous canonical chain supports qTE readout relevance, RMS tangential survival, and out-proj directional amplification. This run treats those as priors and tests proxy utility on independent natural REAL_R/REAL_C residuals.

## 3. Why Proxy Extraction Is Needed

The mechanism explains damage, but method readiness requires prediction without rollout and without future-KL leakage.

## 4. Data Lineage

See `data_inventory.json`. Primary proxy evidence uses naturally produced REAL_R/REAL_C residuals for 9 canonical units. Controlled high/low out-proj directions are reference-only.

## 5. Tensor Semantics

`E` is handled as `[batch, head, d_k, d_v]`; readout residual is `E^T q`, equivalent to `q^T E`, producing `d_v` per value head.

## 6. Linearized Functional Operator

The implemented local operator is `W_O D_g D_w J_RMS(o) E^T q`, using the real `Qwen3_5RMSNormGated` order: RMS, learned weight, SiLU gate, head merge, out_proj.

## 7. Metric Definitions

M0-M5 and state-space variants are written in `natural_residual_metrics.csv`, `state_space_risk_results.csv`, and `query_covariance_results.csv`.

## 8. Linearization Accuracy

Real INT8 median cosine: `0.8528156482027733`. Real INT8 median relative error: `0.613826915519402`.

## 9. Natural Quantization Panel

Primary rows: `18`. Valid units: `9`.

## 10. Controlled Reference Panel

The previous out-proj high/low panel is recorded as mechanism reference only and is not pooled into primary proxy evidence.

## 11. R/C Ranking

|metric|predicts_R_gt_C|total|accuracy|
|---|---|---|---|
|M0_raw_state_error|3.0|9.0|0.3333333333333333|
|M0_raw_state_error_rel|3.0|9.0|0.3333333333333333|
|M1_qTE_norm|9.0|9.0|1.0|
|M2_tangent_norm|9.0|9.0|1.0|
|M2_tangent_frac|9.0|9.0|1.0|
|M3_RMS_linearized_norm|0.0|9.0|0.0|
|M4_post_gate_linearized_norm|9.0|9.0|1.0|
|M5_out_proj_linearized_norm|9.0|9.0|1.0|
|M5_gain|0.0|9.0|0.0|
|M5_state|9.0|9.0|1.0|
|R_current_q_G0|9.0|9.0|1.0|
|R_current_q_G1|0.0|9.0|0.0|
|R_current_q_G2|9.0|9.0|1.0|
|R_current_q_G3|9.0|9.0|1.0|
|R_past16_G3|9.0|9.0|1.0|
|R_past64_G3|9.0|9.0|1.0|
|R_past256_G3|9.0|9.0|1.0|
|R_pastEMA_G3|9.0|9.0|1.0|

## 12. Multi-Config Ranking

No additional protocol-compatible independent natural multi-config panel was found beyond REAL_R/REAL_C. Norm-swapped/constructed panels are not treated as natural configs.

## 13. Query-Covariance Analysis

Current, past16, past64, past256, EMA0.95, and future-oracle G3 variants were computed. Future oracle is `ORACLE_ONLY_NOT_DEPLOYABLE`.

## 14. Deployable vs Oracle Metrics

|metric|information_class|spearman_future_KL_AUC|pearson_future_KL_AUC|
|---|---|---|---|
|M0_raw_state_error|DEPLOYABLE|-0.14963880288957687|-0.05026300513189065|
|M0_raw_state_error_rel|DEPLOYABLE|0.007223942208462332|-0.05140505873713726|
|M1_qTE_norm|DEPLOYABLE|0.56656346749226|0.5506030813159992|
|M2_tangent_norm|DEPLOYABLE|0.5562435500515995|0.5442856819628706|
|M2_tangent_frac|DEPLOYABLE|0.5624355005159959|0.5095618659543003|
|M3_RMS_linearized_norm|DEPLOYABLE|-0.28586171310629516|-0.37604811811665484|
|M4_post_gate_linearized_norm|DEPLOYABLE|0.8163054695562435|0.8309979214698616|
|M5_out_proj_linearized_norm|DEPLOYABLE|0.7956656346749226|0.7918669119913309|
|M5_gain|DEPLOYABLE|-0.24871001031991744|-0.28231250225504145|
|M5_state|DEPLOYABLE|0.7956656346749226|0.7782412114914938|
|actual_out_proj_distortion_norm|ORACLE_ONLY_NOT_DEPLOYABLE|0.7708978328173375|0.754789942407438|
|R_current_q_G0|DEPLOYABLE|0.56656346749226|0.5579815159765714|
|R_current_q_G1|DEPLOYABLE|-0.28586171310629516|-0.3472542865282354|
|R_current_q_G2|DEPLOYABLE|0.8163054695562435|0.8500387313292694|
|R_current_q_G3|DEPLOYABLE|0.7956656346749226|0.8025785043501003|
|R_past16_G3|DEPLOYABLE|0.7853457172342622|0.7799543512715676|
|R_past64_G3|DEPLOYABLE|0.7853457172342622|0.774833767171214|
|R_past256_G3|DEPLOYABLE|0.7853457172342622|0.7779882002717665|
|R_pastEMA_G3|DEPLOYABLE|0.8018575851393189|0.7831214294139218|
|R_future_oracle_G3|ORACLE_ONLY_NOT_DEPLOYABLE|0.7853457172342622|0.7616621200802386|

## 15. Candidate-Selection Regret

|metric|selection_hit_rate|mean_selection_regret|
|---|---|---|
|M0_raw_state_error|0.3333333333333333|0.0001620353704795271|
|M5_out_proj_linearized_norm|1.0|0.0|
|M4_post_gate_linearized_norm|1.0|0.0|

## 16. Compute Feasibility

See `compute_feasibility.json`. This is an audit only; no kernel or quantizer is designed.

## 17. Gate Summary

```json
{
  "COMPUTE_FEASIBILITY_GATE": "PASS",
  "CONTROLLED_PANEL_GATE": "PARTIAL",
  "DATA_LINEAGE_GATE": "PASS",
  "FUTURE_KL_PROXY_GATE": "PASS",
  "GENERALIZATION_GATE": "PARTIAL",
  "HOOK_TARGET_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "LINEARIZATION_IDENTITY_GATE": "PASS",
  "LINEAR_OPERATOR_GATE": "PASS",
  "LOCAL_PROXY_GATE": "PASS",
  "METHOD_DESIGN_GATE": "PARTIAL",
  "METHOD_PRINCIPLE_GATE": "PASS",
  "METRIC_IMPLEMENTATION_GATE": "PASS",
  "MULTICONFIG_RANKING_GATE": "NOT_RUN",
  "NATURAL_PANEL_GATE": "PASS",
  "NO_FUTURE_LEAKAGE_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "R_C_RANKING_GATE": "PASS",
  "TEMPORAL_PROXY_GATE": "PASS",
  "TENSOR_SEMANTICS_GATE": "PASS"
}
```

## 18. Supported Conclusions

Natural R/C proxy evaluation completed with predefined M0-M5 and deployable/oracle state-space variants; controlled synthetic directions are excluded from primary generalization evidence.

## 19. Negative / Partial Results

Functional metrics strongly improve R/C prediction, but only REAL_R/REAL_C are protocol-compatible independent natural configs here; >=4-config ranking is NOT_RUN, so method design remains NO.

## 20. Final Scientific Classification

`FUNCTIONAL_PROXY_SUPPORTED_FOR_RC_BUT_GENERALIZATION_PARTIAL`

## 21. Method-Readiness Decision

`METHOD_PRINCIPLE_EXTRACTION_READY=YES` and `METHOD_DESIGN_READY=NO`.

## 22. Next Recommended Experiment

build a protocol-compatible natural multi-config residual panel before designing a final quantizer
