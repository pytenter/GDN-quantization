# GDN INT8 R128 vs C128 Operator-Aligned Residual Audit V1

## 1. Scientific Question
Why is C128 more stable than R128 under matched INT8 bit-width, group size, and scale count?

## 2. Current Evidence
R128 repeated accumulation formal is SUPPORTED; WHY_C128_BEATS_R128 remains not explained.

## 3. Hypotheses
Magnitude dominated, operator-structure dominated, or mixed.

## 4. GDN Operator Semantics
{
  "beta_shape_after_transpose": "[B,H,T] and beta_t [B,H,1]",
  "decay_shape": "g [B,H,T], gamma=exp(g_t) broadcast to [B,H,1,1]",
  "formula": "S_decay=gamma*S; memory=k^T S_decay; delta=beta*(v-memory); S_next=S_decay+k*delta^T; core=q^T S_next",
  "modeling_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
  "qk_normalized_status": "use_qk_l2norm_in_kernel=True in decode path; q is additionally scaled by 1/sqrt(K)",
  "reduction_dimension_for_k": "K axis via `(last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)`",
  "reduction_dimension_for_q": "K axis via `(last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)`",
  "state_axis_order": "[B,H,K,V] where K is key/read dimension and V is value/output dimension"
}

## 5. R128 / C128 Definitions
{
  "R128": {
    "fixed": "Key row",
    "scale_shape": [
      1,
      32,
      128,
      1
    ],
    "shared_scale_over": "128 Value-axis elements"
  },
  "C128": {
    "fixed": "Value column",
    "scale_shape": [
      1,
      32,
      1,
      128
    ],
    "shared_scale_over": "128 Key-axis elements"
  }
}

## 6. Shadow-Quantization Protocol
FP trajectory only; R128/C128 shadow quantization on the same FP recurrent state; no cache write-back.

## 7. Stage-0 Gates
{
  "ANALYTIC_REPLAY_IDENTITY_GATE": "PASS",
  "C128_QUANTIZER_IDENTITY_GATE": "PASS",
  "NEXT_TOKEN_DRIVER_HOOK_GATE": "PASS",
  "R128_QUANTIZER_IDENTITY_GATE": "PASS",
  "SCALE_COUNT_MATCH_GATE": "PASS",
  "SHADOW_NONINTERFERENCE_GATE": "PASS",
  "SOURCE_SEMANTICS_GATE": "PASS",
  "STATE_IDENTITY_GATE": "PASS"
}

## 8. Analytic vs Actual Replay Identity
{
  "max_core_readout_relative_error": 0.0001084559737424396,
  "max_delta_relative_error": 0.0005671976263020272,
  "max_memory_read_relative_error": 5.331238124096907e-05,
  "max_next_state_relative_error": 8.07379718446045e-05,
  "tolerance": 0.005
}

## 9. Raw Residual Magnitude

## 10. Per-Value-Channel Residual Structure

## 11. Memory-Read Error

## 12. Effective Delta Error

## 13. One-Step Next-State Error

## 14. Core Readout Error

## 15. Normalized Functional Gain

## 16. R128 Scale-Setter Concentration

## 17. Cross-Value Resolution Contamination

## 18. Layer / Head Heterogeneity
{
  "fraction_E_R_gt_E_C": 0.8754340277777778,
  "fraction_mem_gain_R_gt_C": 0.9256365740740741,
  "fraction_core_gain_R_gt_C": 0.2591145833333333,
  "raw_residual_ratio_stats": {
    "median": 1.5711883189835407,
    "p25": 1.2110783641245286,
    "p75": 1.9573956079460049,
    "p90": 2.3577391955554368,
    "p95": 2.571709308094927,
    "p99": 2.972511444857024,
    "max": 3.92776115185767,
    "mean": 1.6129395227750156
  },
  "mem_gain_ratio_stats": {
    "median": 2.3359775786005557,
    "p25": 1.5534784769674457,
    "p75": 3.870473826195105,
    "p90": 6.580990877537816,
    "p95": 8.967257178973455,
    "p99": 17.59946902060361,
    "max": 186.9000921853801,
    "mean": 3.4476147531537005
  },
  "core_gain_ratio_stats": {
    "median": 0.586893705608204,
    "p25": 0.3540479303929304,
    "p75": 1.0243159722751807,
    "p90": 1.7154881068661543,
    "p95": 2.4772915096703043,
    "p99": 5.995462047027048,
    "max": 94.79929197214551,
    "mean": 1.0281037271281557
  },
  "layer_head_count": 6912
}

## 19. Per-Prompt / Per-Token Results
| Snapshot | Raw ||E_R||/||E_C|| | Mem Gain R/C | Delta Gain R/C | Next-State Gain R/C | Core-Readout Gain R/C | Classification |
|---|---:|---:|---:|---:|---:|---|
| test/algebra/1332.json|t=64 | 1.6036221461338602 | 2.284388534858672 | 2.2843884882186343 | 0.9947508935511055 | 0.5618238359709828 | descriptive |
| test/algebra/1332.json|t=128 | 1.578520904514646 | 2.3377372291620944 | 2.3377372286667795 | 0.9940748749799732 | 0.5980273731822003 | descriptive |
| test/algebra/1332.json|t=256 | 1.5537632157762382 | 2.12209588479287 | 2.1220959677894595 | 0.9952748425509774 | 0.6478214174663993 | descriptive |
| test/counting_and_probability/119.json|t=64 | 1.552611679223886 | 2.5082544202010464 | 2.5082544421395983 | 0.9926437257628131 | 0.5287003386222996 | descriptive |
| test/counting_and_probability/119.json|t=128 | 1.536752184487458 | 2.2620272769066077 | 2.262027079443773 | 0.9932706360550826 | 0.5527762592551912 | descriptive |
| test/counting_and_probability/119.json|t=256 | 1.5449313203024246 | 2.363745899843332 | 2.363745970899228 | 0.9944236238712376 | 0.6027185682783693 | descriptive |
| test/geometry/477.json|t=64 | 1.593467421667707 | 2.435003999934023 | 2.435003934184423 | 0.9937317346745802 | 0.5847947444665562 | descriptive |
| test/geometry/477.json|t=128 | 1.5846544240612945 | 2.3503164646244032 | 2.3503165695512487 | 0.9926373461961489 | 0.6017404659154628 | descriptive |
| test/geometry/477.json|t=256 | 1.5819794589852165 | 2.367760407731496 | 2.3677604941496826 | 0.9945609859020959 | 0.6440362490122982 | descriptive |

| Metric | R128 Median | C128 Median | R>C Units | Median R/C Ratio |
|---|---:|---:|---:|---:|
| relative residual norm | 0.018402949673986344 | 0.011478073867354097 | 9 / 9 | 1.578520904514646 |
| memory error gain | 0.15256029526000411 | 0.06753711808593121 | 0 / 9 | 2.3503164646244032 |
| delta error gain | 0.055159421260419145 | 0.021407337004334274 | 0 / 9 | 2.3503165695512487 |
| next-state gain | 0.9759263083343049 | 0.9877570499554048 | 0 / 9 | 0.9940748749799732 |
| core-readout gain | 0.0035528068843841028 | 0.00595555219849824 | 0 / 9 | 0.5980273731822003 |

| Prompt | Token | R128 top-1 setter share | R128 top-5 setter share | Setter entropy | Median resolution ratio | P95 resolution ratio | Victim-channel count |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/algebra/1332.json | 64 | 0.734375 | 0.984375 | 0.18636916324065878 | 2.4213797450065613 | 17.43572039604186 | 11.0 |
| test/algebra/1332.json | 128 | 0.734375 | 0.9765625 | 0.18468804026197372 | 2.3742905855178833 | 17.39701814651486 | 11.0 |
| test/algebra/1332.json | 256 | 0.7109375 | 0.9765625 | 0.19368852666873793 | 2.4443633556365967 | 17.90400085449216 | 11.0 |
| test/counting_and_probability/119.json | 64 | 0.71875 | 0.9765625 | 0.19433278644879262 | 2.374225080013275 | 16.58409152030944 | 11.0 |
| test/counting_and_probability/119.json | 128 | 0.703125 | 0.9765625 | 0.19812666917881383 | 2.415007174015045 | 17.212866306304925 | 11.0 |
| test/counting_and_probability/119.json | 256 | 0.73828125 | 0.984375 | 0.17871258776861373 | 2.477772295475006 | 18.419645929336532 | 11.0 |
| test/geometry/477.json | 64 | 0.734375 | 0.984375 | 0.1856759092715624 | 2.4197773933410645 | 17.70263915061946 | 11.0 |
| test/geometry/477.json | 128 | 0.765625 | 0.984375 | 0.16246413063883341 | 2.4868125915527344 | 18.907195377349836 | 11.0 |
| test/geometry/477.json | 256 | 0.75 | 0.984375 | 0.17820751025228976 | 2.4993501901626587 | 18.56820120811461 | 11.0 |

## 20. Pilot Classification
`MAGNITUDE_DOMINATED_SIGNAL`

## 21. What Is Supported
Pilot supports magnitude-dominated candidate: R128 residuals are larger, while normalized operator gains do not robustly favor R128.

## 22. What Is NOT Supported
- No cadence or repeated quantization comparison is run in this task.
- No trajectory write-back, feedback difference, norm-swap causality, or method readiness is established.
- Scale contamination is observational here, not causal.

## 23. Negative / Corrective Results
This audit is shadow-only. No cadence, repeated write-back, norm-swap, feedback decomposition, or method design was run.

## 24. Next Recommended Experiment
Inspect operator-aligned audit heterogeneity before any causal norm-swap.
