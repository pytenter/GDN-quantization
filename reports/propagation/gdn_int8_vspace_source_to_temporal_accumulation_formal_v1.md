# GDN INT8 V-Space Source To Temporal Accumulation Formal V1

## 1. Task
GDN_INT8_VSPACE_SOURCE_TO_TEMPORAL_ACCUMULATION_FORMAL_V1

## 2. Scientific question
Does a compute-invariant V-space representation intervention that lowers R128 source residual also attenuate repeated temporal accumulation?

## 3. Prior evidence
R128 repeated accumulation is formally supported; V_HADAMARD source rescue and single-pulse KL rescue are supported in prior pilot evidence.

## 4. Protocol
FP32 prompt prefill + teacher-forced continuation + only GDN recurrent-state intervention.

## 5. State / axis semantics
DynamicCache.layers[layer_idx].recurrent_states[0], shape [1,32,128,128], axis 2 Key, axis 3 Value.

## 6. Quantizer semantics
INT8 symmetric, zero point 0, qrange [-127,127], torch.round; R128 uses [B,H,K,1] scales.

## 7. V-space transform semantics
S_rot=S@H; S_rot_q=Q_R128(S_rot); S_q=S_rot_q@H.T. V_HADAMARD remains a mechanism control, not a proposed method.

## 8. Stage 0 gates
```json
{
  "INJECTION_POINT_GATE": "PASS",
  "METRIC_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "QUANTIZER_IDENTITY_GATE": "PASS",
  "REPRODUCIBILITY_GATE": "PASS",
  "SHADOW_NONINTERFERENCE_GATE": "PASS",
  "STATE_SEMANTICS_GATE": "PASS",
  "VSPACE_IDENTITY_GATE": "PASS"
}
```

## 9. Source formal results
{
  "bootstrap_95ci_median_log_ratio": [
    -1.2657735795401157,
    -1.2544263653254162
  ],
  "iqr_ratio": [
    0.2814447999181524,
    0.285544152494523
  ],
  "median_R128": 0.019356100355562095,
  "median_R128_VH": 0.005489653645186305,
  "median_ratio": 0.2837622917906929,
  "paired_win_count": 18
}

## 10. Pilot results
{
  "R128_VH_KL_AUC_improves": 3,
  "R128_VH_KL_excess_improves": 3,
  "R128_VH_STATE_AUC_improves": 3,
  "R128_VH_STATE_excess_improves": 3,
  "bootstrap_95ci_median_log_ratio_E1_KL_AUC": [
    -3.2244646375399744,
    -2.1903506626570217
  ],
  "bootstrap_95ci_median_log_ratio_E1_STATE_AUC": [
    -0.8142124863369041,
    -0.7928192692358048
  ],
  "bootstrap_95ci_median_log_ratio_KL_excess": [
    -3.442203341841331,
    -2.3197087123326665
  ],
  "bootstrap_95ci_median_log_ratio_STATE_excess": [
    -0.48780491671814163,
    -0.39709377189373174
  ],
  "bootstrap_seed": 20260901,
  "median_E1_KL_AUC_R128": 0.013891563203434296,
  "median_E1_KL_AUC_R128_VH": 0.0010609855229468587,
  "median_E1_STATE_AUC_R128": 0.03477216603843417,
  "median_E1_STATE_AUC_R128_VH": 0.015403653998330148,
  "median_KL_accumulation_excess_R128": 0.013313630304830334,
  "median_KL_accumulation_excess_R128_VH": 0.000804818305277509,
  "median_STATE_accumulation_excess_R128": 0.01982621254428367,
  "median_STATE_accumulation_excess_R128_VH": 0.01234747961004711,
  "median_ratio_E1_KL_AUC_VH_over_R": 0.08337731469441366,
  "median_ratio_E1_STATE_AUC_VH_over_R": 0.4501378301413237,
  "median_ratio_KL_excess_VH_over_R": 0.05993359091599472,
  "median_ratio_STATE_excess_VH_over_R": 0.622785596686462,
  "median_rho_KL_AUC_R128": 0.9999999999994997,
  "median_rho_KL_AUC_R128_VH": 0.9999999999994997,
  "median_rho_STATE_AUC_R128": 0.9999999999994997,
  "median_rho_STATE_AUC_R128_VH": 0.9999999999994997,
  "valid_units": 3
}

## 11. Formal temporal results
{
  "R128_VH_KL_AUC_improves": 18,
  "R128_VH_KL_excess_improves": 18,
  "R128_VH_STATE_AUC_improves": 18,
  "R128_VH_STATE_excess_improves": 18,
  "bootstrap_95ci_median_log_ratio_E1_KL_AUC": [
    -2.7736251214874743,
    -2.094927195467645
  ],
  "bootstrap_95ci_median_log_ratio_E1_STATE_AUC": [
    -0.8477753767651691,
    -0.7899992751486635
  ],
  "bootstrap_95ci_median_log_ratio_KL_excess": [
    -2.968900988896789,
    -2.211292240779125
  ],
  "bootstrap_95ci_median_log_ratio_STATE_excess": [
    -0.4904576053199463,
    -0.39876443674690093
  ],
  "bootstrap_seed": 20260901,
  "median_E1_KL_AUC_R128": 0.009938979224258236,
  "median_E1_KL_AUC_R128_VH": 0.0007673474115168793,
  "median_E1_STATE_AUC_R128": 0.03478033254715926,
  "median_E1_STATE_AUC_R128_VH": 0.015486944934291973,
  "median_KL_accumulation_excess_R128": 0.009144931748869137,
  "median_KL_accumulation_excess_R128_VH": 0.0006670139773945535,
  "median_KL_excess_E1_vs_E32_R128": 0.008921390965318583,
  "median_KL_excess_E1_vs_E32_R128_VH": 0.0006266938442485591,
  "median_STATE_accumulation_excess_R128": 0.019461878334849626,
  "median_STATE_accumulation_excess_R128_VH": 0.012302545240145983,
  "median_STATE_excess_E1_vs_E32_R128": 0.01348932869305457,
  "median_STATE_excess_E1_vs_E32_R128_VH": 0.01070142928400054,
  "median_ratio_E1_KL_AUC_VH_over_R": 0.1036817604676569,
  "median_ratio_E1_STATE_AUC_VH_over_R": 0.44401205285951884,
  "median_ratio_KL_excess_VH_over_R": 0.0864933183665494,
  "median_ratio_STATE_excess_VH_over_R": 0.6322764920106787,
  "median_rho_KL_AUC_R128": 0.9999999999999429,
  "median_rho_KL_AUC_R128_VH": 0.9428571428570891,
  "median_rho_STATE_AUC_R128": 0.9999999999999429,
  "median_rho_STATE_AUC_R128_VH": 0.9999999999999429,
  "valid_units": 18
}

## 12. Cadence dose response
Reported separately for R128 and R128_VH as Spearman rho(freq, AUC).

## 13. Accumulation-amplitude comparison
Primary bridge endpoint is E1-vs-SINGLE excess for KL and state AUC.

## 14. Shadow-source vs live-dynamics comparison
Shadow source measurements are computed on untouched FP trajectories; live dynamics are separate intervention branches.

## 15. C128 contextual comparison
C128 is included in source formal as context only, not as the primary causal gate.

## 16. Negative / corrective findings
Magnitude alone is not claimed as a complete explanation; prior same-norm structure evidence remains preserved.

## 17. Allowed conclusion
A compute-invariant V-space intervention can link representation-dependent source disturbance to temporal accumulation if formal gates pass.

## 18. Claims NOT supported
No final method, no Hadamard novelty claim, no mechanism closure, no METHOD_DESIGN_READY=YES.

## 19. Mechanism interpretation
SOURCE -> TEMPORAL ACCUMULATION is tested while preserving operator-conditioned geometry as a separate mechanism.

## 20. Recommended next scientific step
Do not launch automatically; inspect residual geometry/operator-conditioned pathway after this bridge result.

## 21. Artifact paths
- /data/zypan/experiments/qwen35_gdn_quant/run_int8_vspace_source_to_temporal_accumulation_formal.py
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_source_formal.json
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_pilot.json
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_formal.json
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_checkpoint.json
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_raw.npz
- /data/zypan/reports/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1.md
- /data/zypan/results/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_figures
