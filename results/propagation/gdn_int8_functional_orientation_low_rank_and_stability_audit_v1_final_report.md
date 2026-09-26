# GDN_INT8_FUNCTIONAL_ORIENTATION_LOW_RANK_AND_STABILITY_AUDIT_V1

## 1. TASK
Audit whether frozen M5 geometry is low-rank, top-k sufficient, and stable/reusable.

## 2. Scientific motivation
How much of M5 do we actually need?

## 3. Previous evidence
{
  "beyond_reconstruction_commit": "6fab2921c92f72249d05b052575b1561fedd3206"
}

## 4. Frozen M5 definition
{
  "formula": "|| W_O D_g D_w J_RMS(o) E^T q ||_2",
  "modified": false
}

## 5. Tensor semantics audit
{
  "e_space": "[head,value]=[32,128] flattened to 4096",
  "operator": "A=W_O blockdiag(D_g D_w J_RMS_head)",
  "out_proj": "4096x4096 full projection"
}

## 6. M5 quadratic identity
{
  "M5_QUADRATIC_IDENTITY_GATE": "PASS",
  "max_relative_error": 5.680098418632273e-07,
  "n_checks": 1728
}

## 7. Functional spectrum
See `functional_spectra.json`.

## 8. Cumulative energy
{
  "top16_energy_mean": 0.028164348916475925,
  "top16_energy_median": 0.027590589575361577,
  "top1_energy_mean": 0.003032784425690865,
  "top1_energy_median": 0.003088736906490309,
  "top2_energy_mean": 0.005420277858408335,
  "top2_energy_median": 0.005684347790870528,
  "top32_energy_mean": 0.04599045436520493,
  "top32_energy_median": 0.0429501509970239,
  "top4_energy_mean": 0.009519603669604451,
  "top4_energy_median": 0.009666478358676721,
  "top64_energy_mean": 0.07333257180929909,
  "top64_energy_median": 0.06903782991507584,
  "top8_energy_mean": 0.01661388682856398,
  "top8_energy_median": 0.01688940899088772
}

## 9. Effective rank
{
  "max": 10219.14190629482,
  "median": 9078.402715242251,
  "min": 6841.1831438258905
}

## 10. Eigenvalue gaps
{
  "k1_normalized_gap_median": 0.13816718286892393,
  "k2_normalized_gap_median": 0.054634332080207224,
  "k4_normalized_gap_median": 0.028421416558982827,
  "k8_normalized_gap_median": 0.02216897037074125
}

## 11. Top-k approximation
[
  {
    "k": 1,
    "median_relative_error": 0.9896998044648517,
    "pairwise_vs_full": 0.6626984126984127,
    "spearman_vs_full": 0.3710206444144318
  },
  {
    "k": 2,
    "median_relative_error": 0.9831394860783849,
    "pairwise_vs_full": 0.7103174603174603,
    "spearman_vs_full": 0.5947327802431025
  },
  {
    "k": 4,
    "median_relative_error": 0.9716262583801681,
    "pairwise_vs_full": 0.746031746031746,
    "spearman_vs_full": 0.5920959547237764
  },
  {
    "k": 8,
    "median_relative_error": 0.9520250494648932,
    "pairwise_vs_full": 0.7777777777777778,
    "spearman_vs_full": 0.6672454820245675
  },
  {
    "k": 16,
    "median_relative_error": 0.9303641758965857,
    "pairwise_vs_full": 0.9007936507936508,
    "spearman_vs_full": 0.7872210431539006
  },
  {
    "k": 32,
    "median_relative_error": 0.9025430554763137,
    "pairwise_vs_full": 0.8968253968253969,
    "spearman_vs_full": 0.7690848286063412
  },
  {
    "k": 64,
    "median_relative_error": 0.8563141310750234,
    "pairwise_vs_full": 0.9246031746031746,
    "spearman_vs_full": 0.7418161939674577
  }
]

## 12. Top-k vs full M5
See `topk_m5_approximation.json`.

## 13. Top-k 72-case ranking retention
{
  "full": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.875,
    "R-R_pairwise": 0.9074074074074074,
    "correct_pairs": 201,
    "kendall": 0.5305164319248826,
    "pairwise": 0.7976190476190477,
    "spearman": 0.7158338156794649,
    "total_pairs": 252
  },
  "top1": {
    "C-C_pairwise": 0.5185185185185185,
    "R-C_pairwise": 0.6319444444444444,
    "R-R_pairwise": 0.6481481481481481,
    "correct_pairs": 154,
    "kendall": 0.09467918622848201,
    "pairwise": 0.6111111111111112,
    "spearman": 0.15010611614894848,
    "total_pairs": 252
  },
  "top16": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.8958333333333334,
    "R-R_pairwise": 0.8333333333333334,
    "correct_pairs": 200,
    "kendall": 0.3075117370892019,
    "pairwise": 0.7936507936507936,
    "spearman": 0.45025403562930094,
    "total_pairs": 252
  },
  "top2": {
    "C-C_pairwise": 0.5185185185185185,
    "R-C_pairwise": 0.6805555555555556,
    "R-R_pairwise": 0.7407407407407407,
    "correct_pairs": 166,
    "kendall": 0.23317683881064163,
    "pairwise": 0.6587301587301587,
    "spearman": 0.36011962184063284,
    "total_pairs": 252
  },
  "top32": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.9583333333333334,
    "R-R_pairwise": 0.8518518518518519,
    "correct_pairs": 209,
    "kendall": 0.3435054773082942,
    "pairwise": 0.8293650793650794,
    "spearman": 0.4943404720560808,
    "total_pairs": 252
  },
  "top4": {
    "C-C_pairwise": 0.5370370370370371,
    "R-C_pairwise": 0.7152777777777778,
    "R-R_pairwise": 0.7592592592592593,
    "correct_pairs": 173,
    "kendall": 0.22065727699530516,
    "pairwise": 0.6865079365079365,
    "spearman": 0.3185735417068622,
    "total_pairs": 252
  },
  "top64": {
    "C-C_pairwise": 0.3888888888888889,
    "R-C_pairwise": 0.9236111111111112,
    "R-R_pairwise": 0.8518518518518519,
    "correct_pairs": 200,
    "kendall": 0.34820031298904536,
    "pairwise": 0.7936507936507936,
    "spearman": 0.49112483117885397,
    "total_pairs": 252
  },
  "top8": {
    "C-C_pairwise": 0.5185185185185185,
    "R-C_pairwise": 0.7708333333333334,
    "R-R_pairwise": 0.7777777777777778,
    "correct_pairs": 181,
    "kendall": 0.2652582159624413,
    "pairwise": 0.7182539682539683,
    "spearman": 0.3787381825197762,
    "total_pairs": 252
  }
}

## 14. Top-k matched-M0 14-pair validation
{
  "full": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "top1": {
    "concordance": 0.42857142857142855,
    "correct": 6,
    "total": 14
  },
  "top16": {
    "concordance": 0.9285714285714286,
    "correct": 13,
    "total": 14
  },
  "top2": {
    "concordance": 0.5714285714285714,
    "correct": 8,
    "total": 14
  },
  "top32": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "top4": {
    "concordance": 0.7142857142857143,
    "correct": 10,
    "total": 14
  },
  "top64": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "top8": {
    "concordance": 0.8571428571428571,
    "correct": 12,
    "total": 14
  }
}

## 15. Functional subspace stability
{
  "top16_projection_overlap_median": 0.28816920171895316,
  "top1_cosine_median": 0.019919468089938164,
  "top2_projection_overlap_median": 0.002396432988301212,
  "top4_projection_overlap_median": 0.21581054576523767,
  "top8_projection_overlap_median": 0.24636308904499188
}

## 16. Context dependence
Subspace similarity is computed across the available 9 canonical units; no new prompt/t0 panel was created.

## 17. Static calibration protocol
{
  "G_bar": "arithmetic mean over calibration units only",
  "split": "leave-one-unit-out",
  "static_k": 32
}

## 18. Held-out static validation
{
  "dynamic_top32_72case": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.9583333333333334,
    "R-R_pairwise": 0.8518518518518519,
    "correct_pairs": 209,
    "kendall": 0.3435054773082942,
    "pairwise": 0.8293650793650794,
    "spearman": 0.4943404720560808,
    "total_pairs": 252
  },
  "dynamic_top32_matched": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "static_top32_72case": {
    "C-C_pairwise": 0.3888888888888889,
    "R-C_pairwise": 0.9097222222222222,
    "R-R_pairwise": 0.8148148148148148,
    "correct_pairs": 196,
    "kendall": 0.25586854460093894,
    "pairwise": 0.7777777777777778,
    "spearman": 0.4134349475850537,
    "total_pairs": 252
  },
  "static_top32_matched": {
    "concordance": 0.9285714285714286,
    "correct": 13,
    "total": 14
  }
}

## 19. Static vs dynamic comparison
{
  "dynamic_top32_72case": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.9583333333333334,
    "R-R_pairwise": 0.8518518518518519,
    "correct_pairs": 209,
    "kendall": 0.3435054773082942,
    "pairwise": 0.8293650793650794,
    "spearman": 0.4943404720560808,
    "total_pairs": 252
  },
  "dynamic_top32_matched": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "static_top32_72case": {
    "C-C_pairwise": 0.3888888888888889,
    "R-C_pairwise": 0.9097222222222222,
    "R-R_pairwise": 0.8148148148148148,
    "correct_pairs": 196,
    "kendall": 0.25586854460093894,
    "pairwise": 0.7777777777777778,
    "spearman": 0.4134349475850537,
    "total_pairs": 252
  },
  "static_top32_matched": {
    "concordance": 0.9285714285714286,
    "correct": 13,
    "total": 14
  }
}

## 20. Basis-only vs weighted-basis ablation if run
NOT_RUN; weighted averaged-G top-k is the primary static definition.

## 21. Negative results
No M5 modification, new metric, persistence, decay, learned coefficient, KL-weighted calibration, or future rollout was introduced.

## 22. Limitations
The operator is large; spectra use the true merged-head 4096-dimensional per-layer M5 geometry and a block view across layers. Stability is limited to the existing 9 canonical units.

## 23. Final scientific classification
`FUNCTIONAL_ORIENTATION_NOT_LOW_RANK`

## 24. METHOD_DESIGN_READY
`NO`

## 25. Simplest next method hypothesis
inspect dynamic functional subspace variation before quantizer design
