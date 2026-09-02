# GDN INT8 Same-Norm Downstream Operator Feedback Propagation V1

## 1. Task

GDN_INT8_SAME_NORM_DOWNSTREAM_OPERATOR_FEEDBACK_PROPAGATION_V1

## 2. Previous Positive And Negative Evidence

Stage A operator coupling is supported; Stage B one-step U causal pilot was negative or inconclusive.

## 3. Updated Scientific Hypothesis

Residual-operator feedback dynamics may emerge downstream through future hidden/operator drift.

## 4. Protocol

9 canonical units; FP/R_LOCAL/C_LOCAL; live future model dynamics; online drift metrics only.

## 5. Clean Local Residual Seed

Primary pair R_STRUCT_C_NORM vs REAL_C, layer/head-local, same-norm.

## 6. Stage0 Gates

{
  "PROTOCOL_GATE": "PASS",
  "NORM_MATCH_GATE": "PASS",
  "OPERATOR_INVARIANCE_GATE": "PASS",
  "REPLAY_REGRESSION_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "METRIC_GATE": "PASS",
  "max next-state replay relative error": 0.0,
  "max core-output replay relative error": 0.0,
  "STAGE0": "PASS"
}

## 7. Hidden-State Propagation

{}

## 8. q/k/v/g/beta Operator Drift

{
  "query": 0,
  "key": 0,
  "value": 0,
  "g": 1,
  "beta": 0
}

## 9. Recurrent-State Propagation

{
  "R state-drift AUC > C": "0 / 9"
}

## 10. Readout Propagation

{
  "readout_core": 0,
  "readout_actual": 0
}

## 11. Behavioral KL Propagation

{
  "R_LOCAL > C_LOCAL KL_AUC": "6 / 9",
  "Median R KL_AUC": 9.713330123747247e-05,
  "Median C KL_AUC": 7.720005348361434e-05
}

## 12. Cross-Layer Analysis

Layer x token heatmaps are stored as figures 5 and 6 for a deterministic representative unit.

## 13. Cross-Token Analysis

{
  "hidden_out": {
    "values": [
      13,
      31,
      null,
      null,
      4,
      26,
      null,
      null,
      null
    ],
    "median": 19.5
  },
  "query": {
    "values": [
      13,
      7,
      8,
      24,
      13,
      21,
      4,
      4,
      null
    ],
    "median": 10.5
  },
  "key": {
    "values": [
      5,
      15,
      6,
      1,
      23,
      3,
      1,
      10,
      null
    ],
    "median": 5.5
  },
  "value": {
    "values": [
      0,
      15,
      72,
      null,
      22,
      3,
      3,
      1,
      null
    ],
    "median": 3.0
  },
  "g": {
    "values": [
      6,
      14,
      13,
      22,
      26,
      4,
      57,
      9,
      null
    ],
    "median": 13.5
  },
  "beta": {
    "values": [
      23,
      22,
      55,
      2,
      0,
      6,
      1,
      1,
      null
    ],
    "median": 4.0
  },
  "state": {
    "values": [
      0,
      null,
      null,
      null,
      1,
      31,
      null,
      null,
      0
    ],
    "median": 0.5
  },
  "readout_core": {
    "values": [
      10,
      38,
      10,
      52,
      3,
      13,
      3,
      5,
      null
    ],
    "median": 10.0
  },
  "readout_actual": {
    "values": [
      121,
      31,
      34,
      null,
      36,
      8,
      null,
      null,
      null
    ],
    "median": 34.0
  },
  "KL": {
    "values": [
      13,
      9,
      1,
      18,
      3,
      48,
      16,
      11,
      3
    ],
    "median": 11.0
  }
}

## 14. Earliest Crossover Analysis

{
  "EARLIEST_PERSISTENT_R_GT_C_STAGE": "KL",
  "median token offset": 11.0,
  "median layer offset": null
}

## 15. Live-vs-Frozen Diagnostic

NOT_RUN

## 16. Feedback-Path Classification

NOT_SUPPORTED

## 17. Negative / Corrective Findings

The one-step U pathway remains unsupported as the main behavioral explanation; this task tests downstream feedback instead.

## 18. Allowed Conclusion

This tracing audit localizes downstream R/C drift after a clean same-norm local seed. It is pathway evidence only and does not establish operator drift as causal.

## 19. Claims Not Supported

Operator drift is tracing evidence only, not causal proof. U remains not supported as main behavioral path.

## 20. Recommended Next Task

controlled multi-head/layer composition study

## 21. Artifact Paths

{
  "script": "/data/zypan/experiments/qwen35_gdn_quant/run_int8_same_norm_downstream_operator_feedback_propagation.py",
  "stage0": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_stage0.json",
  "trace": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_trace.json",
  "summary": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_summary.json",
  "checkpoint": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_checkpoint.json",
  "raw": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_raw.npz",
  "report": "/data/zypan/reports/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1.md",
  "figures": "/data/zypan/results/gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_figures"
}
