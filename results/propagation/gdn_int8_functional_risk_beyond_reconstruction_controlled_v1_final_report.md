# GDN_INT8_FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_CONTROLLED_V1

## 1. TASK
Test whether frozen M5 predicts future KL direction when raw M0 reconstruction error is approximately matched.

## 2. Scientific motivation
This is a controlled incremental-value test: same reconstruction, different functional risk, different future damage?

## 3. Previous evidence
{
  "c_family_localization_commit": "d0ac45087ff0d845a34abafd81782f6e987edce9",
  "natural_config_commit": "0059f8bc3f0f5c25138e200350a17174768b68aa"
}

## 4. Frozen metric definitions
{
  "M0": "raw state reconstruction error",
  "M5": "|| W_O D_g D_w J_RMS(o) E^T q ||_2",
  "modified": false
}

## 5. Protocol audit
{
  "future_KL_join_after_blind_selection": true,
  "same_configs": true,
  "same_units": true,
  "source_rows": 72
}

## 6. Blind pair-selection rule
{
  "created_before_future_KL_join": true,
  "future_KL_fields_included": false,
  "primary_min_R_C_pairs": 5,
  "primary_min_total": 10,
  "primary_order": [
    [
      0.05,
      0.5
    ],
    [
      0.05,
      0.25
    ],
    [
      0.1,
      0.5
    ],
    [
      0.1,
      0.25
    ],
    [
      0.2,
      0.5
    ],
    [
      0.2,
      0.25
    ]
  ],
  "primary_rule": {
    "M0_relative_gap_max": 0.2,
    "M5_relative_gap_min": 0.25,
    "reason": "first blind threshold in predeclared order satisfying minimum total and R-C pair density"
  },
  "selection_inputs": [
    "unit_id",
    "config_A",
    "config_B",
    "M0",
    "M5",
    "protocol-compatible metadata"
  ],
  "selection_rule_sha256": "125917956579d903074ad3316d562b84a16a8eb3c31f59ec8338d7de5ac4e620",
  "strong_control_rule": {
    "M0_relative_gap_max": 0.05,
    "M5_relative_gap_min": 0.5
  },
  "task": "GDN_INT8_FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_CONTROLLED_V1",
  "threshold_grid": {
    "M0_relative_gap_max": [
      0.05,
      0.1,
      0.2
    ],
    "M5_relative_gap_min": [
      0.5,
      0.25
    ]
  }
}

## 7. Pair-count audit
[
  {
    "M0_relative_gap_max": 0.05,
    "M5_relative_gap_min": 0.5,
    "median_M0_relative_gap": null,
    "median_M5_relative_gap": null,
    "n_CC": 0,
    "n_RC": 0,
    "n_RR": 0,
    "n_pairs": 0
  },
  {
    "M0_relative_gap_max": 0.05,
    "M5_relative_gap_min": 0.25,
    "median_M0_relative_gap": 0.037634835033338226,
    "median_M5_relative_gap": 0.2745669693028492,
    "n_CC": 0,
    "n_RC": 4,
    "n_RR": 0,
    "n_pairs": 4
  },
  {
    "M0_relative_gap_max": 0.1,
    "M5_relative_gap_min": 0.5,
    "median_M0_relative_gap": 0.05573702860325182,
    "median_M5_relative_gap": 0.5427353609646618,
    "n_CC": 0,
    "n_RC": 1,
    "n_RR": 0,
    "n_pairs": 1
  },
  {
    "M0_relative_gap_max": 0.1,
    "M5_relative_gap_min": 0.25,
    "median_M0_relative_gap": 0.04198035625377183,
    "median_M5_relative_gap": 0.3275054045623254,
    "n_CC": 0,
    "n_RC": 6,
    "n_RR": 0,
    "n_pairs": 6
  },
  {
    "M0_relative_gap_max": 0.2,
    "M5_relative_gap_min": 0.5,
    "median_M0_relative_gap": 0.14941155261983743,
    "median_M5_relative_gap": 0.589807132248387,
    "n_CC": 0,
    "n_RC": 3,
    "n_RR": 0,
    "n_pairs": 3
  },
  {
    "M0_relative_gap_max": 0.2,
    "M5_relative_gap_min": 0.25,
    "median_M0_relative_gap": 0.1413704870710999,
    "median_M5_relative_gap": 0.3297698366640097,
    "n_CC": 0,
    "n_RC": 14,
    "n_RR": 0,
    "n_pairs": 14
  }
]

## 8. M0 matching quality
{
  "binomial_p_two_sided_vs_0_5": 0.0001220703125,
  "concordance": 1.0,
  "confidence_interval_95_clopper_pearson": [
    0.7683642383498834,
    1.0
  ],
  "correct": 14,
  "label": "all",
  "mean_KL_absolute_gap": 0.0001430270702993493,
  "median_KL_absolute_gap": 8.21974281182201e-05,
  "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
  "median_M0_relative_gap": 0.1413704870710999,
  "median_M5_relative_gap": 0.3297698366640097,
  "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
  "total": 14
}

## 9. M5 separation quality
{
  "binomial_p_two_sided_vs_0_5": 0.0001220703125,
  "concordance": 1.0,
  "confidence_interval_95_clopper_pearson": [
    0.7683642383498834,
    1.0
  ],
  "correct": 14,
  "label": "all",
  "mean_KL_absolute_gap": 0.0001430270702993493,
  "median_KL_absolute_gap": 8.21974281182201e-05,
  "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
  "median_M0_relative_gap": 0.1413704870710999,
  "median_M5_relative_gap": 0.3297698366640097,
  "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
  "total": 14
}

## 10. Overall matched-pair result
{
  "binomial_p_two_sided_vs_0_5": 0.0001220703125,
  "concordance": 1.0,
  "confidence_interval_95_clopper_pearson": [
    0.7683642383498834,
    1.0
  ],
  "correct": 14,
  "label": "all",
  "mean_KL_absolute_gap": 0.0001430270702993493,
  "median_KL_absolute_gap": 8.21974281182201e-05,
  "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
  "median_M0_relative_gap": 0.1413704870710999,
  "median_M5_relative_gap": 0.3297698366640097,
  "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
  "total": 14
}

## 11. R-C controlled result
{
  "binomial_p_two_sided_vs_0_5": 0.0001220703125,
  "concordance": 1.0,
  "confidence_interval_95_clopper_pearson": [
    0.7683642383498834,
    1.0
  ],
  "correct": 14,
  "label": "R-C",
  "mean_KL_absolute_gap": 0.0001430270702993493,
  "median_KL_absolute_gap": 8.21974281182201e-05,
  "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
  "median_M0_relative_gap": 0.1413704870710999,
  "median_M5_relative_gap": 0.3297698366640097,
  "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
  "total": 14
}

## 12. R-R control
{
  "binomial_p_two_sided_vs_0_5": null,
  "concordance": null,
  "confidence_interval_95_clopper_pearson": null,
  "correct": 0,
  "label": "R-R",
  "mean_KL_absolute_gap": null,
  "median_KL_absolute_gap": null,
  "median_KL_high_M5_minus_low_M5": null,
  "median_M0_relative_gap": null,
  "median_M5_relative_gap": null,
  "median_normalized_KL_high_M5_minus_low_M5": null,
  "total": 0
}

## 13. C-C low-separation control
{
  "binomial_p_two_sided_vs_0_5": null,
  "concordance": null,
  "confidence_interval_95_clopper_pearson": null,
  "correct": 0,
  "label": "C-C",
  "mean_KL_absolute_gap": null,
  "median_KL_absolute_gap": null,
  "median_KL_high_M5_minus_low_M5": null,
  "median_M0_relative_gap": null,
  "median_M5_relative_gap": null,
  "median_normalized_KL_high_M5_minus_low_M5": null,
  "total": 0
}

## 14. KL-gap analysis
{
  "C-C": {
    "binomial_p_two_sided_vs_0_5": null,
    "concordance": null,
    "confidence_interval_95_clopper_pearson": null,
    "correct": 0,
    "label": "C-C",
    "mean_KL_absolute_gap": null,
    "median_KL_absolute_gap": null,
    "median_KL_high_M5_minus_low_M5": null,
    "median_M0_relative_gap": null,
    "median_M5_relative_gap": null,
    "median_normalized_KL_high_M5_minus_low_M5": null,
    "total": 0
  },
  "R-C": {
    "binomial_p_two_sided_vs_0_5": 0.0001220703125,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.7683642383498834,
      1.0
    ],
    "correct": 14,
    "label": "R-C",
    "mean_KL_absolute_gap": 0.0001430270702993493,
    "median_KL_absolute_gap": 8.21974281182201e-05,
    "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
    "median_M0_relative_gap": 0.1413704870710999,
    "median_M5_relative_gap": 0.3297698366640097,
    "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
    "total": 14
  },
  "R-R": {
    "binomial_p_two_sided_vs_0_5": null,
    "concordance": null,
    "confidence_interval_95_clopper_pearson": null,
    "correct": 0,
    "label": "R-R",
    "mean_KL_absolute_gap": null,
    "median_KL_absolute_gap": null,
    "median_KL_high_M5_minus_low_M5": null,
    "median_M0_relative_gap": null,
    "median_M5_relative_gap": null,
    "median_normalized_KL_high_M5_minus_low_M5": null,
    "total": 0
  },
  "all": {
    "binomial_p_two_sided_vs_0_5": 0.0001220703125,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.7683642383498834,
      1.0
    ],
    "correct": 14,
    "label": "all",
    "mean_KL_absolute_gap": 0.0001430270702993493,
    "median_KL_absolute_gap": 8.21974281182201e-05,
    "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
    "median_M0_relative_gap": 0.1413704870710999,
    "median_M5_relative_gap": 0.3297698366640097,
    "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
    "total": 14
  },
  "top_100pct_largest_KL_gap_descriptive": {
    "binomial_p_two_sided_vs_0_5": 0.0001220703125,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.7683642383498834,
      1.0
    ],
    "correct": 14,
    "label": "top_100pct_largest_KL_gap_descriptive",
    "mean_KL_absolute_gap": 0.0001430270702993493,
    "median_KL_absolute_gap": 8.21974281182201e-05,
    "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
    "median_M0_relative_gap": 0.1413704870710999,
    "median_M5_relative_gap": 0.3297698366640097,
    "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
    "total": 14
  },
  "top_25pct_largest_KL_gap_descriptive": {
    "binomial_p_two_sided_vs_0_5": 0.125,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.3976353643835253,
      1.0
    ],
    "correct": 4,
    "label": "top_25pct_largest_KL_gap_descriptive",
    "mean_KL_absolute_gap": 0.00030415004753407937,
    "median_KL_absolute_gap": 0.00030254628905188117,
    "median_KL_high_M5_minus_low_M5": 0.00030254628905188117,
    "median_M0_relative_gap": 0.1424244022795763,
    "median_M5_relative_gap": 0.40938941890226144,
    "median_normalized_KL_high_M5_minus_low_M5": 0.9415725828267136,
    "total": 4
  },
  "top_50pct_largest_KL_gap_descriptive": {
    "binomial_p_two_sided_vs_0_5": 0.015625,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.5903836027749965,
      1.0
    ],
    "correct": 7,
    "label": "top_50pct_largest_KL_gap_descriptive",
    "mean_KL_absolute_gap": 0.00024063120225460447,
    "median_KL_absolute_gap": 0.0002744407868871957,
    "median_KL_high_M5_minus_low_M5": 0.0002744407868871957,
    "median_M0_relative_gap": 0.13332942152236238,
    "median_M5_relative_gap": 0.2966172379043895,
    "median_normalized_KL_high_M5_minus_low_M5": 0.6332928866088121,
    "total": 7
  },
  "wrong_prediction_gap_profile": {
    "wrong_count": 0,
    "wrong_fraction_below_1e-4": null,
    "wrong_fraction_below_1e-5": null,
    "wrong_median_KL_gap": null
  }
}

## 15. Unit-level consistency
{
  "median_unit_concordance": 1.0,
  "units": 6,
  "units_above_chance": 6,
  "units_perfect": 6
}

## 16. Statistical confidence
{
  "by_pair_class": {
    "C-C": {
      "binomial_p_two_sided_vs_0_5": null,
      "concordance": null,
      "confidence_interval_95_clopper_pearson": null,
      "correct": 0,
      "label": "C-C",
      "mean_KL_absolute_gap": null,
      "median_KL_absolute_gap": null,
      "median_KL_high_M5_minus_low_M5": null,
      "median_M0_relative_gap": null,
      "median_M5_relative_gap": null,
      "median_normalized_KL_high_M5_minus_low_M5": null,
      "total": 0
    },
    "R-C": {
      "binomial_p_two_sided_vs_0_5": 0.0001220703125,
      "concordance": 1.0,
      "confidence_interval_95_clopper_pearson": [
        0.7683642383498834,
        1.0
      ],
      "correct": 14,
      "label": "R-C",
      "mean_KL_absolute_gap": 0.0001430270702993493,
      "median_KL_absolute_gap": 8.21974281182201e-05,
      "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
      "median_M0_relative_gap": 0.1413704870710999,
      "median_M5_relative_gap": 0.3297698366640097,
      "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
      "total": 14
    },
    "R-R": {
      "binomial_p_two_sided_vs_0_5": null,
      "concordance": null,
      "confidence_interval_95_clopper_pearson": null,
      "correct": 0,
      "label": "R-R",
      "mean_KL_absolute_gap": null,
      "median_KL_absolute_gap": null,
      "median_KL_high_M5_minus_low_M5": null,
      "median_M0_relative_gap": null,
      "median_M5_relative_gap": null,
      "median_normalized_KL_high_M5_minus_low_M5": null,
      "total": 0
    }
  },
  "primary": {
    "binomial_p_two_sided_vs_0_5": 0.0001220703125,
    "concordance": 1.0,
    "confidence_interval_95_clopper_pearson": [
      0.7683642383498834,
      1.0
    ],
    "correct": 14,
    "label": "all",
    "mean_KL_absolute_gap": 0.0001430270702993493,
    "median_KL_absolute_gap": 8.21974281182201e-05,
    "median_KL_high_M5_minus_low_M5": 8.21974281182201e-05,
    "median_M0_relative_gap": 0.1413704870710999,
    "median_M5_relative_gap": 0.3297698366640097,
    "median_normalized_KL_high_M5_minus_low_M5": 0.3966154673840838,
    "total": 14
  },
  "strong_control": {
    "binomial_p_two_sided_vs_0_5": null,
    "concordance": null,
    "confidence_interval_95_clopper_pearson": null,
    "correct": 0,
    "label": "strong_control",
    "mean_KL_absolute_gap": null,
    "median_KL_absolute_gap": null,
    "median_KL_high_M5_minus_low_M5": null,
    "median_M0_relative_gap": null,
    "median_M5_relative_gap": null,
    "median_normalized_KL_high_M5_minus_low_M5": null,
    "total": 0
  }
}

## 17. Phase C decision
{
  "PHASE_C_REQUIRED": false,
  "reason": "existing blind-matched R-C pair density is sufficient for the orientation-difference controlled question; strict strong-control subset is unavailable and reported as a limitation"
}

## 18. Phase C result if run
NOT_RUN_EXISTING_PANEL_SUFFICIENT

## 19. Negative results
No M4/M5 modification, threshold tuning on KL, regression, persistence, decay, temporal weighting, or fresh GPU experiment was used.

## 20. Limitations
Pairs are drawn from the existing 8-config natural panel; pair-level observations within a unit are not fully independent, so unit-level summaries are reported separately. The blind grid found no strict M0<=5% and M5>=50% strong-control pairs, and no R-R/C-C matched-M0/M5-separated pairs under the selected rule.

## 21. Final scientific classification
`FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_SUPPORTED_FOR_ORIENTATION_DIFFERENCES`

## 22. Method-design readiness
METHOD_PRINCIPLE_EXTRACTION_READY=`YES`; METHOD_DESIGN_READY=`NO`

## 23. Next recommended task
formalize functional-risk orientation principle before any quantizer design
