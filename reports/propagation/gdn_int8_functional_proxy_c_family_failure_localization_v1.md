# GDN_INT8_FUNCTIONAL_PROXY_C_FAMILY_FAILURE_LOCALIZATION_V1

## 1. Task
Localize why frozen M4/M5 rank R-family configs well but fail within C-family.

## 2. Data Gate
{
  "EXISTING_DATA_GATE": "PASS",
  "n_cases": 72,
  "n_configs": 8,
  "n_units": 9,
  "required_fields_present": true,
  "source_commit": "d75f290b356926a19d358b074cbe117bbc319ef6",
  "source_raw": "/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl",
  "source_summary": "/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_final_summary.json"
}

## 3. Family Separability
{
  "C": {
    "KL_range_mean": 5.578561187161473e-05,
    "KL_range_median": 5.992694783932436e-05,
    "best_second_gap_mean": 1.9604129640804667e-05,
    "best_second_gap_median": 1.5566697003796374e-05,
    "family": "C",
    "max_pairwise_gap": 0.00010758705776820572,
    "median_pairwise_gap": 2.4190752262533357e-05,
    "n_units": 9,
    "pair_gap_fraction_lt_1e-4": 0.9814814814814815,
    "pair_gap_fraction_lt_1e-5": 0.18518518518518517,
    "pair_gap_fraction_lt_1e-6": 0.018518518518518517
  },
  "C_over_R_KL_range_median": 0.14502198722787182,
  "C_over_R_best_second_gap_median": 0.3173670578935056,
  "R": {
    "KL_range_mean": 0.0005304161184940254,
    "KL_range_median": 0.00041322663438709927,
    "best_second_gap_mean": 9.953325657838409e-05,
    "best_second_gap_median": 4.904950371898022e-05,
    "family": "R",
    "max_pairwise_gap": 0.0010977815666740423,
    "median_pairwise_gap": 0.00022599487052322163,
    "n_units": 9,
    "pair_gap_fraction_lt_1e-4": 0.25925925925925924,
    "pair_gap_fraction_lt_1e-5": 0.0,
    "pair_gap_fraction_lt_1e-6": 0.0
  }
}

## 4. Winner Distribution
{
  "C_family": {
    "M0_argmin": {
      "C16": 9
    },
    "M4_argmin": {
      "C16": 9
    },
    "M5_argmin": {
      "C16": 9
    },
    "true_KL": {
      "C128": 4,
      "C16": 2,
      "C32": 1,
      "C64": 2
    }
  },
  "R_family": {
    "M0_argmin": {
      "R16": 9
    },
    "M4_argmin": {
      "R16": 9
    },
    "M5_argmin": {
      "R16": 9
    },
    "true_KL": {
      "R16": 5,
      "R32": 4
    }
  },
  "overall_true_winner_family": {
    "C": 9
  }
}

## 5. Granularity Monotonicity
{
  "C": {
    "adjacent_decrease_fraction": 0.5185185185185185,
    "adjacent_increase_fraction": 0.48148148148148145,
    "family": "C",
    "strict_order_units_128_ge_64_ge_32_ge_16": 1,
    "strict_order_units_128_le_64_le_32_le_16": 0,
    "unit_count": 9
  },
  "R": {
    "adjacent_decrease_fraction": 0.8518518518518519,
    "adjacent_increase_fraction": 0.14814814814814814,
    "family": "R",
    "strict_order_units_128_ge_64_ge_32_ge_16": 5,
    "strict_order_units_128_le_64_le_32_le_16": 0,
    "unit_count": 9
  }
}

## 6. Metric Ladder By Family
{
  "C": {
    "M0": {
      "correct_pairs": 26,
      "family": "C",
      "kendall_tau": -0.09841269841269841,
      "metric": "M0",
      "pairwise_concordance": 0.48148148148148145,
      "spearman": -0.1552123552123552,
      "total_pairs": 54
    },
    "M1": {
      "correct_pairs": 25,
      "family": "C",
      "kendall_tau": 0.06031746031746032,
      "metric": "M1",
      "pairwise_concordance": 0.46296296296296297,
      "spearman": 0.10527670527670528,
      "total_pairs": 54
    },
    "M2": {
      "correct_pairs": 26,
      "family": "C",
      "kendall_tau": -0.047619047619047616,
      "metric": "M2",
      "pairwise_concordance": 0.48148148148148145,
      "spearman": -0.06666666666666667,
      "total_pairs": 54
    },
    "M3": {
      "correct_pairs": 26,
      "family": "C",
      "kendall_tau": 0.13015873015873017,
      "metric": "M3",
      "pairwise_concordance": 0.48148148148148145,
      "spearman": 0.21698841698841698,
      "total_pairs": 54
    },
    "M4": {
      "correct_pairs": 26,
      "family": "C",
      "kendall_tau": 0.3142857142857143,
      "metric": "M4",
      "pairwise_concordance": 0.48148148148148145,
      "spearman": 0.41312741312741313,
      "total_pairs": 54
    },
    "M5": {
      "correct_pairs": 26,
      "family": "C",
      "kendall_tau": 0.30793650793650795,
      "metric": "M5",
      "pairwise_concordance": 0.48148148148148145,
      "spearman": 0.418018018018018,
      "total_pairs": 54
    }
  },
  "R": {
    "M0": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.3492063492063492,
      "metric": "M0",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.5137709137709138,
      "total_pairs": 54
    },
    "M1": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.4507936507936508,
      "metric": "M1",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.6406692406692407,
      "total_pairs": 54
    },
    "M2": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.4507936507936508,
      "metric": "M2",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.6406692406692407,
      "total_pairs": 54
    },
    "M3": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.4158730158730159,
      "metric": "M3",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.5938223938223938,
      "total_pairs": 54
    },
    "M4": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.5523809523809524,
      "metric": "M4",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.7374517374517374,
      "total_pairs": 54
    },
    "M5": {
      "correct_pairs": 49,
      "family": "R",
      "kendall_tau": 0.5428571428571428,
      "metric": "M5",
      "pairwise_concordance": 0.9074074074074074,
      "spearman": 0.7328185328185328,
      "total_pairs": 54
    }
  }
}

## 7. M4 vs M5 Increment
{
  "C-C": {
    "M4_correct": 26,
    "M4_incorrect": 28,
    "M4_pairwise": 0.48148148148148145,
    "M5_correct": 26,
    "M5_incorrect": 28,
    "M5_pairwise": 0.48148148148148145,
    "right_to_wrong": 0,
    "same_correct": 26,
    "same_wrong": 28,
    "total": 54,
    "wrong_to_right": 0
  },
  "R-C": {
    "M4_correct": 115,
    "M4_incorrect": 29,
    "M4_pairwise": 0.7986111111111112,
    "M5_correct": 126,
    "M5_incorrect": 18,
    "M5_pairwise": 0.875,
    "right_to_wrong": 1,
    "same_correct": 114,
    "same_wrong": 17,
    "total": 144,
    "wrong_to_right": 12
  },
  "R-R": {
    "M4_correct": 49,
    "M4_incorrect": 5,
    "M4_pairwise": 0.9074074074074074,
    "M5_correct": 49,
    "M5_incorrect": 5,
    "M5_pairwise": 0.9074074074074074,
    "right_to_wrong": 0,
    "same_correct": 49,
    "same_wrong": 5,
    "total": 54,
    "wrong_to_right": 0
  },
  "overall": {
    "M4_correct": 190,
    "M4_incorrect": 62,
    "M4_pairwise": 0.753968253968254,
    "M5_correct": 201,
    "M5_incorrect": 51,
    "M5_pairwise": 0.7976190476190477,
    "right_to_wrong": 1,
    "same_correct": 189,
    "same_wrong": 50,
    "total": 252,
    "wrong_to_right": 12
  }
}

## 8. Candidate Selection Failure
See `candidate_failure_map.json` and `regret_margin_analysis.json`.

## 9. Unit Failure Map
See `unit_failure_map.json`.

## 10. Phase A Decision
{
  "C_FAMILY_LOW_SEPARATION": true,
  "C_FAMILY_MEANINGFUL_SEPARATION": false,
  "case": "C_FAMILY_LOW_RISK_LOW_SEPARATION",
  "reason": "C separability is compared directly with R and M5 C-C pairwise remains below 0.6."
}

## 11. Phase B
{
  "PHASE_B_REQUIRED": false,
  "PHASE_B_STATUS": "NOT_RUN_PHASE_A_SUFFICIENT"
}

## 12. Final Scientific Classification
`C_FAMILY_LOW_RISK_LOW_SEPARATION`

## 13. Method Design
METHOD_PRINCIPLE_EXTRACTION_READY=`YES`; METHOD_DESIGN_READY=`NO`

## 14. Interpretation
Phase A uses only the previous 72 formal cases. It does not modify M4/M5, refit normalization, introduce persistence, or define a new proxy. The key question is whether C-family failures are near-ties or a real missing damage source.

## 15. Next Recommended Task
report C-family low-separation failure localization; do not design quantizer yet
