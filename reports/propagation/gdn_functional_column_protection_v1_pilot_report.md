# GDN_FUNCTIONAL_COLUMN_PROTECTION_INTERVENTION_V1

## Protocol Gates
{
  "CALIBRATION_EVAL_DISJOINT": "YES",
  "DIAG_M5_DECOMPOSITION_GATE": "PASS",
  "HYBRID_PROTECTION_SEMANTICS_GATE": "PASS",
  "PROTOCOL_GATE": "PASS"
}

## Selector Score Audit
{
  "FUNCTIONAL_INCREMENTAL_SIGNAL_WEAK": false,
  "error_only_score": {
    "max": 0.900965670744578,
    "mean": 5.918790476445316e-05,
    "median": 8.811116606703232e-08,
    "min": 4.8598283961901835e-11
  },
  "favor_score": {
    "max": 0.050992329950275916,
    "mean": 6.976017836195105e-06,
    "median": 1.9921772216746855e-07,
    "min": 1.6719251193325378e-12
  },
  "functional_only_score": {
    "max": 87989657.33333333,
    "mean": 439217.018035388,
    "median": 33610.546223958336,
    "min": 0.002219898965752994
  },
  "inf_count": 0,
  "magnitude_score": {
    "max": 9666.721028645834,
    "mean": 0.525112494436379,
    "median": 0.0007540430384930612,
    "min": 3.8736931410691494e-07
  },
  "n_rows": 98304,
  "nan_count": 0,
  "score_columns": [
    "magnitude_score",
    "error_only_score",
    "functional_only_score",
    "favor_score"
  ],
  "spearman": {
    "error_only_score_vs_favor_score_spearman": 0.47178883906183944,
    "error_only_score_vs_functional_only_score_spearman": -0.46831341898509227,
    "functional_only_score_vs_favor_score_spearman": 0.387197600108866,
    "magnitude_score_vs_error_only_score_spearman": 0.9829294541386862,
    "magnitude_score_vs_favor_score_spearman": 0.45977857173111936,
    "magnitude_score_vs_functional_only_score_spearman": -0.47300107155017584
  },
  "zero_variance": []
}

## Pilot Results
[
  {
    "future_KL_mean": 0.0007413350425258906,
    "future_KL_median": 0.0008688037252671532,
    "n": 3,
    "protection_ratio": 0.0,
    "relative_rescue_vs_C128_median": 0.0,
    "selector": "C128_INT8",
    "state_error_rel_mean": 0.08159772998874974,
    "top10_overlap_mean": 0.9760416666666666,
    "top1_agreement_mean": 0.9895833333333334
  },
  {
    "future_KL_mean": 0.00043231276908138605,
    "future_KL_median": 0.00045394484324422224,
    "n": 3,
    "protection_ratio": 0.25,
    "relative_rescue_vs_C128_median": 0.4775058732832411,
    "selector": "ERROR_ONLY",
    "state_error_rel_mean": 0.010463649496076545,
    "top10_overlap_mean": 0.9812499999999998,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.0003857406952454447,
    "future_KL_median": 0.00042159835029526027,
    "n": 3,
    "protection_ratio": 0.25,
    "relative_rescue_vs_C128_median": 0.514736943973902,
    "selector": "FAVOR",
    "state_error_rel_mean": 0.015746636942107237,
    "top10_overlap_mean": 0.9812499999999998,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.0,
    "future_KL_median": 0.0,
    "n": 3,
    "protection_ratio": 1.0,
    "relative_rescue_vs_C128_median": 0.999999998848992,
    "selector": "FP_STATE",
    "state_error_rel_mean": 0.0,
    "top10_overlap_mean": 1.0,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.0006045584165825952,
    "future_KL_median": 0.0008210940978354841,
    "n": 3,
    "protection_ratio": 0.25,
    "relative_rescue_vs_C128_median": 0.2513304692404794,
    "selector": "FUNCTIONAL_ONLY",
    "state_error_rel_mean": 0.08119715467852284,
    "top10_overlap_mean": 0.9833333333333333,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.0007413350425258906,
    "future_KL_median": 0.0008688037252671532,
    "n": 3,
    "protection_ratio": 0.0,
    "relative_rescue_vs_C128_median": 0.0,
    "selector": "HYBRID_0_IDENTITY",
    "state_error_rel_mean": 0.08159772998874974,
    "top10_overlap_mean": 0.9760416666666666,
    "top1_agreement_mean": 0.9895833333333334
  },
  {
    "future_KL_mean": 0.0,
    "future_KL_median": 0.0,
    "n": 3,
    "protection_ratio": 1.0,
    "relative_rescue_vs_C128_median": 0.999999998848992,
    "selector": "HYBRID_100_IDENTITY",
    "state_error_rel_mean": 0.0,
    "top10_overlap_mean": 1.0,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.00039411059611936874,
    "future_KL_median": 0.00033873167982667596,
    "n": 3,
    "protection_ratio": 0.25,
    "relative_rescue_vs_C128_median": 0.40550832221380345,
    "selector": "MAGNITUDE",
    "state_error_rel_mean": 0.010567135888637851,
    "top10_overlap_mean": 0.9833333333333333,
    "top1_agreement_mean": 1.0
  },
  {
    "future_KL_mean": 0.0006934958189424638,
    "future_KL_median": 0.0007572786971801193,
    "n": 9,
    "protection_ratio": 0.25,
    "relative_rescue_vs_C128_median": 0.02056203763569319,
    "selector": "RANDOM",
    "state_error_rel_mean": 0.0695392521191743,
    "top10_overlap_mean": 0.9815972222222222,
    "top1_agreement_mean": 0.9895833333333334
  }
]

## Pairwise Comparisons
{
  "FAVOR_vs_ERROR_ONLY": {
    "comparison": "FAVOR_vs_ERROR_ONLY",
    "losses_for_first_lower_KL": 1,
    "mean_gap_second_minus_first": 4.657207383594134e-05,
    "median_gap_second_minus_first": 3.234649294896197e-05,
    "paired_units": 3,
    "ties": 0,
    "wins_for_first_lower_KL": 2
  },
  "FAVOR_vs_FUNCTIONAL_ONLY": {
    "comparison": "FAVOR_vs_FUNCTIONAL_ONLY",
    "losses_for_first_lower_KL": 1,
    "mean_gap_second_minus_first": 0.00021881772133715052,
    "median_gap_second_minus_first": 0.0003914395381601557,
    "paired_units": 3,
    "ties": 0,
    "wins_for_first_lower_KL": 2
  },
  "FAVOR_vs_MAGNITUDE": {
    "comparison": "FAVOR_vs_MAGNITUDE",
    "losses_for_first_lower_KL": 2,
    "mean_gap_second_minus_first": 8.369900873924039e-06,
    "median_gap_second_minus_first": -8.286667046858431e-05,
    "paired_units": 3,
    "ties": 0,
    "wins_for_first_lower_KL": 1
  },
  "FAVOR_vs_RANDOM": {
    "comparison": "FAVOR_vs_RANDOM",
    "losses_for_first_lower_KL": 1,
    "mean_gap_second_minus_first": 0.0003077551236970191,
    "median_gap_second_minus_first": 0.0003979736202688865,
    "paired_units": 3,
    "ties": 0,
    "wins_for_first_lower_KL": 2
  }
}

## Efficiency Audit
{
  "DYNAMIC_GATHER_SCATTER": "NO in static packed layout; YES in prototype mask implementation",
  "EXTRA_FULL_STATE_PASS": 0,
  "EXTRA_STATE_READ_BYTES": "prototype reads FP branch for causal intervention; algorithm would keep protected segment natively",
  "FUSABLE_SINGLE_PASS": "YES_CONDITIONAL",
  "ONLINE_FUNCTIONAL_SCORE": "NO",
  "ONLINE_RANKING": "NO",
  "STATIC_LAYOUT_FEASIBLE": "CONDITIONAL",
  "algorithmic_runtime_overhead": "same recurrent update plus fixed mixed-precision state write; no online scoring",
  "future_optimized_kernel_feasibility": "STATIC_LAYOUT_CONDITIONAL_FEASIBLE",
  "prototype_implementation_overhead": "Python branch replay and torch.where masks are not kernel latency",
  "rows": [
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 16,
      "protection_ratio": 0.125,
      "raw_data_bytes_per_full_gdn_state_estimate": 14155776.0,
      "raw_state_bits_per_value": 9.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.0009765625
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 32,
      "protection_ratio": 0.25,
      "raw_data_bytes_per_full_gdn_state_estimate": 15728640.0,
      "raw_state_bits_per_value": 10.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.001953125
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 48,
      "protection_ratio": 0.375,
      "raw_data_bytes_per_full_gdn_state_estimate": 17301504.0,
      "raw_state_bits_per_value": 11.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.0029296875
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 64,
      "protection_ratio": 0.5,
      "raw_data_bytes_per_full_gdn_state_estimate": 18874368.0,
      "raw_state_bits_per_value": 12.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.00390625
    }
  ]
}

## Static Reorder Feasibility
{
  "STATIC_REORDER_FEASIBLE": "CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED",
  "must_permute": [
    "value projection output/value state V axis",
    "recurrent state Value dimension",
    "out_proj input slices aligned by head/value"
  ],
  "note": "Formal task audited static packing feasibility but did not implement CUDA kernel or final quantizer.",
  "permutation_identity_test": "NOT_RUN_NO_KERNEL_IMPLEMENTATION"
}

## Terminal Summary
```text
TASK =
GDN_FUNCTIONAL_COLUMN_PROTECTION_INTERVENTION_V1

FORMAL_STATUS =
COMPLETE

PROTOCOL_GATE =
PASS

DIAG_M5_DECOMPOSITION_GATE =
PASS

HYBRID_PROTECTION_SEMANTICS_GATE =
PASS

CALIBRATION_EVAL_DISJOINT =
YES

PILOT_CLASSIFICATION =
PILOT_POSITIVE

FORMAL_CLASSIFICATION =
FUNCTIONAL_COLUMN_PROTECTION_PARTIALLY_SUPPORTED

FAVOR_VS_ERROR_ONLY =
{'comparison': 'FAVOR_vs_ERROR_ONLY', 'losses_for_first_lower_KL': 1, 'mean_gap_second_minus_first': 4.657207383594134e-05, 'median_gap_second_minus_first': 3.234649294896197e-05, 'paired_units': 3, 'ties': 0, 'wins_for_first_lower_KL': 2}

FAVOR_VS_FUNCTIONAL_ONLY =
{'comparison': 'FAVOR_vs_FUNCTIONAL_ONLY', 'losses_for_first_lower_KL': 1, 'mean_gap_second_minus_first': 0.00021881772133715052, 'median_gap_second_minus_first': 0.0003914395381601557, 'paired_units': 3, 'ties': 0, 'wins_for_first_lower_KL': 2}

BEST_SELECTOR =
MAGNITUDE

BEST_PROTECTION_RATIO =
0.375

BEST_FUTURE_KL =
0.0003329350385091745

RELATIVE_RESCUE_VS_C128 =
0.4622163143286948

RAW_BITS_PER_VALUE =
11.0

ONLINE_FUNCTIONAL_SCORE =
NO

ONLINE_RANKING =
NO

EXTRA_FULL_STATE_PASS =
0

STATIC_REORDER_FEASIBLE =
CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED

FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY =
PARTIAL

METHOD_DESIGN_READY =
CONDITIONAL_YES

NEXT_RECOMMENDED_TASK =
diagnose selector-vs-baseline failures before rate-distortion design
```

