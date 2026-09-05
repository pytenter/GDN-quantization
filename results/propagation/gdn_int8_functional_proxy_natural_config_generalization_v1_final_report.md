# GDN_INT8_FUNCTIONAL_PROXY_NATURAL_CONFIG_GENERALIZATION_V1

## 1. TASK
Frozen M0-M5 proxy generalization across R128/R64/R32/R16/C128/C64/C32/C16.

## 2. Git / environment
{
  "HEAD": "d75f290b356926a19d358b074cbe117bbc319ef6",
  "branch": "research-sync-2026-09-02",
  "git_status": "?? experiments/propagation/run_int8_functional_proxy_natural_config_generalization.py\n?? reports/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1.md\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_candidate_selection.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_config_compatibility.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_cross_unit_stability.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_failure_analysis.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_final_report.md\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_final_summary.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_future_kl_token_curves.csv\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_pairwise_metrics.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_protocol_audit.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_ranking_metrics.json\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_smoke_raw_results.csv\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_smoke_raw_results.jsonl\n?? results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_smoke_shard0.jsonl"
}

## 3. Protocol audit
{
  "M0_to_M5_source": "/data/zypan/experiments/qwen35_gdn_quant/run_int8_linearized_functional_risk_proxy_extraction.py",
  "current_HEAD_matches_freeze": true,
  "future_KL_source": "single t0 intervention, teacher-forced FP continuation, full-logit KL",
  "layer_head_unit_selection": "frozen canonical 9 units, all canonical GDN layers and all heads",
  "model_revision": "Qwen3.5-9B via p1.setup_model()",
  "previous_frozen_commit": "d75f290",
  "previous_frozen_task": "GDN_INT8_LINEARIZED_FUNCTIONAL_RISK_PROXY_EXTRACTION_V1",
  "quantizer_rounding_scale_grouping": "axis.grouped_quant torch.round symmetric int8 qrange [-127,127]"
}

## 4. Frozen proxy verification
{
  "M0": "sqrt(sum ||E||_F^2)",
  "M1": "sqrt(sum ||E^T q||^2)",
  "M2": "sqrt(sum tangential(E^T q, clean readout)^2)",
  "M3": "sqrt(sum ||J_RMS(o) E^T q||^2)",
  "M4": "sqrt(sum ||D_g D_w J_RMS(o) E^T q||^2)",
  "M5": "sqrt(sum ||W_O D_g D_w J_RMS(o) E^T q||^2)",
  "definitions_modified": false,
  "future_KL_used_in_proxy_phase": false,
  "normalization_refit": false
}

## 5. Natural config panel
[
  {
    "bit_width": 8,
    "config": "R128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  }
]

## 6. Compatibility audit
{
  "C128": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "C128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  "C16": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "C16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  },
  "C32": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "C32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  "C64": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "C64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  },
  "R128": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "R128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  "R16": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "R16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  },
  "R32": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "R32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  "R64": {
    "bit_width": 8,
    "compatibility": "COMPATIBLE",
    "config": "R64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "same_int8_quantizer_definition": true,
    "same_model_revision": true,
    "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": true,
    "same_tensor_semantics": true,
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  }
}

## 7. Stage0 gates
{
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "QUANTIZER_SEMANTICS_GATE": "PASS",
  "TENSOR_SEMANTICS_GATE": "PASS"
}

## 8. Pilot
{
  "status": "see pilot_results.json",
  "used_for_metric_tuning": false
}

## 9. Formal status
`COMPLETE`; raw table rows: `72`.

## 10. Raw config x unit table
See `formal_raw_results.jsonl` and `formal_raw_results.csv`.

## 11. Spearman results
{
  "M0": {
    "aggregate_kendall_tau": 0.2809076682316119,
    "aggregate_spearman": 0.42083092160267543,
    "metric": "M0",
    "per_unit_kendall_mean": 0.46825396825396826,
    "per_unit_kendall_median": 0.5,
    "per_unit_spearman_max": 0.8095238095238095,
    "per_unit_spearman_mean": 0.5846560846560847,
    "per_unit_spearman_median": 0.6190476190476191,
    "per_unit_spearman_min": 0.3333333333333333
  },
  "M1": {
    "aggregate_kendall_tau": 0.4522691705790297,
    "aggregate_spearman": 0.6397195961155058,
    "metric": "M1",
    "per_unit_kendall_mean": 0.7063492063492064,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 1.0,
    "per_unit_spearman_mean": 0.8306878306878305,
    "per_unit_spearman_median": 0.8333333333333334,
    "per_unit_spearman_min": 0.6666666666666666
  },
  "M2": {
    "aggregate_kendall_tau": 0.42644757433489827,
    "aggregate_spearman": 0.6053122387291787,
    "metric": "M2",
    "per_unit_kendall_mean": 0.7142857142857143,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 1.0,
    "per_unit_spearman_mean": 0.8359788359788358,
    "per_unit_spearman_median": 0.8333333333333334,
    "per_unit_spearman_min": 0.6666666666666666
  },
  "M3": {
    "aggregate_kendall_tau": 0.01486697965571205,
    "aggregate_spearman": 0.028169014084507043,
    "metric": "M3",
    "per_unit_kendall_mean": -0.15873015873015872,
    "per_unit_kendall_median": -0.21428571428571427,
    "per_unit_spearman_max": -0.047619047619047616,
    "per_unit_spearman_mean": -0.2645502645502645,
    "per_unit_spearman_median": -0.23809523809523808,
    "per_unit_spearman_min": -0.5476190476190477
  },
  "M4": {
    "aggregate_kendall_tau": 0.5031298904538342,
    "aggregate_spearman": 0.6748022380860506,
    "metric": "M4",
    "per_unit_kendall_mean": 0.5079365079365079,
    "per_unit_kendall_median": 0.5714285714285714,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.6084656084656084,
    "per_unit_spearman_median": 0.6666666666666666,
    "per_unit_spearman_min": 0.23809523809523808
  },
  "M5": {
    "aggregate_kendall_tau": 0.5305164319248826,
    "aggregate_spearman": 0.7158338156794649,
    "metric": "M5",
    "per_unit_kendall_mean": 0.5952380952380952,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.7037037037037037,
    "per_unit_spearman_median": 0.7619047619047619,
    "per_unit_spearman_min": 0.42857142857142855
  }
}

## 12. Kendall tau results
{
  "M0": {
    "aggregate_kendall_tau": 0.2809076682316119,
    "aggregate_spearman": 0.42083092160267543,
    "metric": "M0",
    "per_unit_kendall_mean": 0.46825396825396826,
    "per_unit_kendall_median": 0.5,
    "per_unit_spearman_max": 0.8095238095238095,
    "per_unit_spearman_mean": 0.5846560846560847,
    "per_unit_spearman_median": 0.6190476190476191,
    "per_unit_spearman_min": 0.3333333333333333
  },
  "M1": {
    "aggregate_kendall_tau": 0.4522691705790297,
    "aggregate_spearman": 0.6397195961155058,
    "metric": "M1",
    "per_unit_kendall_mean": 0.7063492063492064,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 1.0,
    "per_unit_spearman_mean": 0.8306878306878305,
    "per_unit_spearman_median": 0.8333333333333334,
    "per_unit_spearman_min": 0.6666666666666666
  },
  "M2": {
    "aggregate_kendall_tau": 0.42644757433489827,
    "aggregate_spearman": 0.6053122387291787,
    "metric": "M2",
    "per_unit_kendall_mean": 0.7142857142857143,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 1.0,
    "per_unit_spearman_mean": 0.8359788359788358,
    "per_unit_spearman_median": 0.8333333333333334,
    "per_unit_spearman_min": 0.6666666666666666
  },
  "M3": {
    "aggregate_kendall_tau": 0.01486697965571205,
    "aggregate_spearman": 0.028169014084507043,
    "metric": "M3",
    "per_unit_kendall_mean": -0.15873015873015872,
    "per_unit_kendall_median": -0.21428571428571427,
    "per_unit_spearman_max": -0.047619047619047616,
    "per_unit_spearman_mean": -0.2645502645502645,
    "per_unit_spearman_median": -0.23809523809523808,
    "per_unit_spearman_min": -0.5476190476190477
  },
  "M4": {
    "aggregate_kendall_tau": 0.5031298904538342,
    "aggregate_spearman": 0.6748022380860506,
    "metric": "M4",
    "per_unit_kendall_mean": 0.5079365079365079,
    "per_unit_kendall_median": 0.5714285714285714,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.6084656084656084,
    "per_unit_spearman_median": 0.6666666666666666,
    "per_unit_spearman_min": 0.23809523809523808
  },
  "M5": {
    "aggregate_kendall_tau": 0.5305164319248826,
    "aggregate_spearman": 0.7158338156794649,
    "metric": "M5",
    "per_unit_kendall_mean": 0.5952380952380952,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.7037037037037037,
    "per_unit_spearman_median": 0.7619047619047619,
    "per_unit_spearman_min": 0.42857142857142855
  }
}

## 13. Pairwise concordance
{
  "C-vs-C": {
    "M0": {
      "correct_pairs": 26,
      "metric": "M0",
      "pair_class": "C-vs-C",
      "pairwise_concordance": 0.48148148148148145,
      "total_pairs": 54
    },
    "M4": {
      "correct_pairs": 26,
      "metric": "M4",
      "pair_class": "C-vs-C",
      "pairwise_concordance": 0.48148148148148145,
      "total_pairs": 54
    },
    "M5": {
      "correct_pairs": 26,
      "metric": "M5",
      "pair_class": "C-vs-C",
      "pairwise_concordance": 0.48148148148148145,
      "total_pairs": 54
    }
  },
  "R-vs-C": {
    "M0": {
      "correct_pairs": 110,
      "metric": "M0",
      "pair_class": "R-vs-C",
      "pairwise_concordance": 0.7638888888888888,
      "total_pairs": 144
    },
    "M4": {
      "correct_pairs": 115,
      "metric": "M4",
      "pair_class": "R-vs-C",
      "pairwise_concordance": 0.7986111111111112,
      "total_pairs": 144
    },
    "M5": {
      "correct_pairs": 126,
      "metric": "M5",
      "pair_class": "R-vs-C",
      "pairwise_concordance": 0.875,
      "total_pairs": 144
    }
  },
  "R-vs-R": {
    "M0": {
      "correct_pairs": 49,
      "metric": "M0",
      "pair_class": "R-vs-R",
      "pairwise_concordance": 0.9074074074074074,
      "total_pairs": 54
    },
    "M4": {
      "correct_pairs": 49,
      "metric": "M4",
      "pair_class": "R-vs-R",
      "pairwise_concordance": 0.9074074074074074,
      "total_pairs": 54
    },
    "M5": {
      "correct_pairs": 49,
      "metric": "M5",
      "pair_class": "R-vs-R",
      "pairwise_concordance": 0.9074074074074074,
      "total_pairs": 54
    }
  },
  "overall": {
    "M0": {
      "correct_pairs": 185,
      "metric": "M0",
      "pair_class": "overall",
      "pairwise_concordance": 0.7341269841269841,
      "total_pairs": 252
    },
    "M4": {
      "correct_pairs": 190,
      "metric": "M4",
      "pair_class": "overall",
      "pairwise_concordance": 0.753968253968254,
      "total_pairs": 252
    },
    "M5": {
      "correct_pairs": 201,
      "metric": "M5",
      "pair_class": "overall",
      "pairwise_concordance": 0.7976190476190477,
      "total_pairs": 252
    }
  }
}

## 14. Within-R ranking
{
  "M0": {
    "kendall_tau": 0.3492063492063492,
    "metric": "M0",
    "n": 36,
    "orientation": "R",
    "spearman": 0.5137709137709138
  },
  "M4": {
    "kendall_tau": 0.5523809523809524,
    "metric": "M4",
    "n": 36,
    "orientation": "R",
    "spearman": 0.7374517374517374
  },
  "M5": {
    "kendall_tau": 0.5428571428571428,
    "metric": "M5",
    "n": 36,
    "orientation": "R",
    "spearman": 0.7328185328185328
  }
}

## 15. Within-C ranking
{
  "M0": {
    "kendall_tau": -0.09841269841269841,
    "metric": "M0",
    "n": 36,
    "orientation": "C",
    "spearman": -0.1552123552123552
  },
  "M4": {
    "kendall_tau": 0.3142857142857143,
    "metric": "M4",
    "n": 36,
    "orientation": "C",
    "spearman": 0.41312741312741313
  },
  "M5": {
    "kendall_tau": 0.30793650793650795,
    "metric": "M5",
    "n": 36,
    "orientation": "C",
    "spearman": 0.418018018018018
  }
}

## 16. Cross R/C ranking
{
  "M0": {
    "correct_pairs": 110,
    "metric": "M0",
    "pair_class": "R-vs-C",
    "pairwise_concordance": 0.7638888888888888,
    "total_pairs": 144
  },
  "M4": {
    "correct_pairs": 115,
    "metric": "M4",
    "pair_class": "R-vs-C",
    "pairwise_concordance": 0.7986111111111112,
    "total_pairs": 144
  },
  "M5": {
    "correct_pairs": 126,
    "metric": "M5",
    "pair_class": "R-vs-C",
    "pairwise_concordance": 0.875,
    "total_pairs": 144
  }
}

## 17. Candidate-selection hit rate
{
  "M0": {
    "hit_rate": 0.2222222222222222,
    "max_regret": 6.517462510054813e-05,
    "mean_regret": 2.905965011737633e-05,
    "median_regret": 2.3834137128729517e-05,
    "metric": "M0"
  },
  "M2": {
    "hit_rate": 0.2222222222222222,
    "max_regret": 6.517462510054813e-05,
    "mean_regret": 2.905965011737633e-05,
    "median_regret": 2.3834137128729517e-05,
    "metric": "M2"
  },
  "M4": {
    "hit_rate": 0.1111111111111111,
    "max_regret": 0.00013480143442748445,
    "mean_regret": 4.9321806252810484e-05,
    "median_regret": 4.660021847439276e-05,
    "metric": "M4"
  },
  "M5": {
    "hit_rate": 0.1111111111111111,
    "max_regret": 7.334694997126601e-05,
    "mean_regret": 3.720931122529477e-05,
    "median_regret": 4.660021847439276e-05,
    "metric": "M5"
  }
}

## 18. Selection regret
{
  "M0": {
    "hit_rate": 0.2222222222222222,
    "max_regret": 6.517462510054813e-05,
    "mean_regret": 2.905965011737633e-05,
    "median_regret": 2.3834137128729517e-05,
    "metric": "M0"
  },
  "M2": {
    "hit_rate": 0.2222222222222222,
    "max_regret": 6.517462510054813e-05,
    "mean_regret": 2.905965011737633e-05,
    "median_regret": 2.3834137128729517e-05,
    "metric": "M2"
  },
  "M4": {
    "hit_rate": 0.1111111111111111,
    "max_regret": 0.00013480143442748445,
    "mean_regret": 4.9321806252810484e-05,
    "median_regret": 4.660021847439276e-05,
    "metric": "M4"
  },
  "M5": {
    "hit_rate": 0.1111111111111111,
    "max_regret": 7.334694997126601e-05,
    "mean_regret": 3.720931122529477e-05,
    "median_regret": 4.660021847439276e-05,
    "metric": "M5"
  }
}

## 19. Cross-unit stability
{
  "layer_head": "not expanded; frozen canonical units aggregate all canonical GDN layers and heads",
  "prompt_id": [
    {
      "group": "test/algebra/1332.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.34057971014492755,
      "metric": "M0",
      "n": 24,
      "spearman": 0.48
    },
    {
      "group": "test/algebra/1332.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.6014492753623188,
      "metric": "M4",
      "n": 24,
      "spearman": 0.7652173913043478
    },
    {
      "group": "test/algebra/1332.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.6304347826086957,
      "metric": "M5",
      "n": 24,
      "spearman": 0.808695652173913
    },
    {
      "group": "test/counting_and_probability/119.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.26811594202898553,
      "metric": "M0",
      "n": 24,
      "spearman": 0.3982608695652174
    },
    {
      "group": "test/counting_and_probability/119.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.2971014492753623,
      "metric": "M4",
      "n": 24,
      "spearman": 0.3834782608695652
    },
    {
      "group": "test/counting_and_probability/119.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.38405797101449274,
      "metric": "M5",
      "n": 24,
      "spearman": 0.48695652173913045
    },
    {
      "group": "test/geometry/477.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.5869565217391305,
      "metric": "M0",
      "n": 24,
      "spearman": 0.7426086956521739
    },
    {
      "group": "test/geometry/477.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.6521739130434783,
      "metric": "M4",
      "n": 24,
      "spearman": 0.8173913043478261
    },
    {
      "group": "test/geometry/477.json",
      "group_by": "prompt_id",
      "kendall_tau": 0.6666666666666666,
      "metric": "M5",
      "n": 24,
      "spearman": 0.8486956521739131
    }
  ],
  "t0": [
    {
      "group": 128,
      "group_by": "t0",
      "kendall_tau": 0.18115942028985507,
      "metric": "M0",
      "n": 24,
      "spearman": 0.30869565217391304
    },
    {
      "group": 128,
      "group_by": "t0",
      "kendall_tau": 0.5072463768115942,
      "metric": "M4",
      "n": 24,
      "spearman": 0.6921739130434783
    },
    {
      "group": 128,
      "group_by": "t0",
      "kendall_tau": 0.5217391304347826,
      "metric": "M5",
      "n": 24,
      "spearman": 0.711304347826087
    },
    {
      "group": 256,
      "group_by": "t0",
      "kendall_tau": 0.30434782608695654,
      "metric": "M0",
      "n": 24,
      "spearman": 0.41043478260869565
    },
    {
      "group": 256,
      "group_by": "t0",
      "kendall_tau": 0.4855072463768116,
      "metric": "M4",
      "n": 24,
      "spearman": 0.6330434782608696
    },
    {
      "group": 256,
      "group_by": "t0",
      "kendall_tau": 0.5217391304347826,
      "metric": "M5",
      "n": 24,
      "spearman": 0.691304347826087
    },
    {
      "group": 64,
      "group_by": "t0",
      "kendall_tau": 0.4855072463768116,
      "metric": "M0",
      "n": 24,
      "spearman": 0.6660869565217391
    },
    {
      "group": 64,
      "group_by": "t0",
      "kendall_tau": 0.5072463768115942,
      "metric": "M4",
      "n": 24,
      "spearman": 0.6660869565217391
    },
    {
      "group": 64,
      "group_by": "t0",
      "kendall_tau": 0.5869565217391305,
      "metric": "M5",
      "n": 24,
      "spearman": 0.7452173913043478
    }
  ]
}

## 20. Failure cases
{
  "orientation_shortcut_check": "reported separately via R-vs-R, C-vs-C, and R-vs-C pairwise concordance",
  "prompt_or_t0_specific_failure": "see cross_unit_stability.json",
  "protocol_failure": "NO",
  "proxy_tuning_after_KL": "NO",
  "same_orientation_granularity_failure": "NO_OR_PARTIAL"
}

## 21. M0 vs M4 vs M5 comparison
{
  "M0": {
    "aggregate_kendall_tau": 0.2809076682316119,
    "aggregate_spearman": 0.42083092160267543,
    "metric": "M0",
    "per_unit_kendall_mean": 0.46825396825396826,
    "per_unit_kendall_median": 0.5,
    "per_unit_spearman_max": 0.8095238095238095,
    "per_unit_spearman_mean": 0.5846560846560847,
    "per_unit_spearman_median": 0.6190476190476191,
    "per_unit_spearman_min": 0.3333333333333333
  },
  "M4": {
    "aggregate_kendall_tau": 0.5031298904538342,
    "aggregate_spearman": 0.6748022380860506,
    "metric": "M4",
    "per_unit_kendall_mean": 0.5079365079365079,
    "per_unit_kendall_median": 0.5714285714285714,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.6084656084656084,
    "per_unit_spearman_median": 0.6666666666666666,
    "per_unit_spearman_min": 0.23809523809523808
  },
  "M5": {
    "aggregate_kendall_tau": 0.5305164319248826,
    "aggregate_spearman": 0.7158338156794649,
    "metric": "M5",
    "per_unit_kendall_mean": 0.5952380952380952,
    "per_unit_kendall_median": 0.7142857142857143,
    "per_unit_spearman_max": 0.9047619047619048,
    "per_unit_spearman_mean": 0.7037037037037037,
    "per_unit_spearman_median": 0.7619047619047619,
    "per_unit_spearman_min": 0.42857142857142855
  }
}

## 22. Scientific interpretation
The frozen proxy is evaluated only as a deployable pre-rollout ranking signal. M0 is retained as the reconstruction baseline, and M3 is retained as a negative/ablation metric. R-vs-R and C-vs-C concordance are separated to test whether any success is only an orientation split.

## 23. Negative results
No metric definition, normalization, coefficient, future-oracle query, persistence factor, or new M6+ proxy was introduced.

## 24. Limitations
The panel reuses the frozen canonical unit set; per-layer and per-head formal expansion is intentionally not added in this task.

## 25. Final scientific classification
`FUNCTIONAL_PROXY_MULTICONFIG_GENERALIZATION_PARTIAL`

## 26. METHOD_DESIGN_READY
`NO`

## 27. Recommended next step
failure analysis before functional-aware quantizer design
