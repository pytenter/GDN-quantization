# Natural rotation ranking report

Final scientific decision: `SHORT_HORIZON_READOUT_SURROGATE_SUPPORTED_FOR_VALIDATION`. Best validated candidate: `P1`.

Actual coverage: two documents, 16 natural rotations each, two early calibration windows each, and four historical document-anchor targets. These are **two independent documents**, not 64 independent replications. No p-values, in-sample R-squared claims or post-hoc window/layer selection are used.

The table reports Spearman rho / pairwise concordance. Higher proxy means greater damage; pairwise ties are excluded. Residual association is Pearson correlation of the rank residuals after an intercept plus rank(JPROP), i.e. partial Spearman, reported in PANEL_STATISTICS.csv.

| Metric | 37702 Primary | 37702 Secondary | 5396 Primary | 5396 Secondary |
|---|---:|---:|---:|---:|
| P1 | 0.726471 / 0.783333 | 0.870588 / 0.883333 | 0.858824 / 0.841667 | 0.794118 / 0.833333 |
| P2_W128 | 0.697059 / 0.766667 | 0.858824 / 0.883333 | 0.885294 / 0.866667 | 0.714706 / 0.775000 |
| P3_H32 | 0.697059 / 0.766667 | 0.858824 / 0.883333 | 0.888235 / 0.866667 | 0.726471 / 0.775000 |
| P3_H128 | 0.697059 / 0.766667 | 0.858824 / 0.883333 | 0.900000 / 0.875000 | 0.732353 / 0.783333 |
| B0 | 0.567647 / 0.691667 | 0.697059 / 0.775000 | 0.738235 / 0.800000 | 0.600000 / 0.725000 |
| B1_H32 | 0.567647 / 0.691667 | 0.697059 / 0.775000 | 0.729412 / 0.791667 | 0.588235 / 0.716667 |
| B1_H128 | 0.570588 / 0.700000 | 0.700000 / 0.783333 | 0.761765 / 0.800000 | 0.614706 / 0.725000 |
| B2_H32 | 0.458824 / 0.666667 | 0.538235 / 0.716667 | 0.388235 / 0.666667 | 0.232353 / 0.575000 |
| B2_H128 | 0.644118 / 0.741667 | 0.664706 / 0.758333 | 0.679412 / 0.750000 | 0.623529 / 0.725000 |
| B3_JPROP | 0.697059 / 0.791667 | 0.702941 / 0.800000 | 0.914706 / 0.925000 | 0.626471 / 0.725000 |
| B3_POR | 0.776471 / 0.808333 | 0.841176 / 0.850000 | 0.861765 / 0.866667 | 0.761765 / 0.800000 |
| B3_HISTORICAL_J_PROP | 0.429412 / 0.616667 | 0.588235 / 0.750000 | 0.647059 / 0.758333 | 0.523529 / 0.700000 |

Frozen stability/incremental rules and their complete results:

```json
{
  "stability": {
    "P1": true,
    "P2_W128": true,
    "P3_H32": true,
    "P3_H128": true
  },
  "incremental": {
    "P1": {
      "beats_each_baseline_panels": {
        "B0": 4,
        "B1_H32": 4,
        "B1_H128": 4,
        "B2_H32": 4,
        "B2_H128": 4
      },
      "positive_residual_panels": 3,
      "passed": true
    },
    "P2_W128": {
      "beats_each_baseline_panels": {
        "B0": 4,
        "B1_H32": 4,
        "B1_H128": 4,
        "B2_H32": 4,
        "B2_H128": 4
      },
      "positive_residual_panels": 4,
      "passed": true
    },
    "P3_H32": {
      "beats_each_baseline_panels": {
        "B0": 4,
        "B1_H32": 4,
        "B1_H128": 4,
        "B2_H32": 4,
        "B2_H128": 4
      },
      "positive_residual_panels": 4,
      "passed": true
    },
    "P3_H128": {
      "beats_each_baseline_panels": {
        "B0": 4,
        "B1_H32": 4,
        "B1_H128": 4,
        "B2_H32": 4,
        "B2_H128": 4
      },
      "positive_residual_panels": 4,
      "passed": true
    }
  },
  "cheaper": {
    "P3_H32": 2,
    "P3_H128": 2
  }
}
```

All per-condition rankings are in ALL_ROTATION_RANKS.csv; every discordant pair is in ALL_RANK_REVERSALS.csv. No counterexample was dropped. All B0-relative-gap <=10% pairs are enumerated in NATURAL_MAGNITUDE_MATCHED_CONTRASTS.csv (294 metric-panel rows). Natural directional discrimination is `DISCOVERY_ONLY`; there was no frozen numerical threshold for a 'significant' P2/P3 contrast, so no such threshold was invented after label inspection.

The subgroup H/L6/L7 can be read directly from the complete rank CSV. Layer20 values are retained in the score CSV only as auxiliary diagnostics, never substituted for all-layer rankings.

Historical JPROP/POR use realized future states/queries and are retrospective benchmarks, not deployable short-window objectives. The historical target is full-vocabulary FP||pulse KL AUC after a natural persistent-QDQ prefix to anchor, with no further QDQ in the continuation. The source error can influence future recurrent propagation and logits, but this target does not measure continued INT8 quantizer feedback through 32K.

Proxies use driver 580.178.04 and labels 580.173.02: **EXPLORATORY_ASSOCIATION_ONLY**, not cross-runtime trajectory equivalence, independent validation, or an Attempt05 merge. Shared calibration scores are reused for both anchors; the anchors are robustness checks, not new independent samples.

Artificial-direction auxiliary data preserve the known future-query sensitivity contrast, but cannot supply the early P1/P2/P3 tensors cheaply. CAUSAL_AUXILIARY_REUSE.json marks this boundary; no 8K continuation was repeated. Extreme VISIBLE/LOW separation is not substituted for natural-rotation evidence.

At most one next mechanism question is proposed in NEXT_EXPERIMENT_DECISION.json. No objective training, optimizer update, new rotation or prospective 32K validation was run.
