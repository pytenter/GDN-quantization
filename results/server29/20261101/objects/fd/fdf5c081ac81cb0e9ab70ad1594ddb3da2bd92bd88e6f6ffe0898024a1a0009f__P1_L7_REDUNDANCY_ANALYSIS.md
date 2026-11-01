# P1 and original L7: actual Phase A analysis

Status: `PHASE_A_COMPLETE`. Route: **A, P1 versus L7_TOTAL_ORIGINAL**.
This is a matched-forward scoring result, not a training or 32K persistent-validation result.

## Evidence and methods

Both historical documents completed all 16 frozen rotations on their fixed GPUs. Every metric was recomputed against a fresh shared same-GPU FP reference. The old score CSV was not joined because it lacks the required FP intermediate tensor identities. Calibration starts are 256 and 1024, H128, all 18 KDA layers and all 16 heads. The frozen equal-two-window score mean is the route/pair-selection input. Individual windows are reported, not cherry-picked for the route.

L7 Functional, State and Total use the recovered historical helper definitions. Total is Functional + 0.1 × State. Applying those helpers to full-model deployment tensors is not a replay of the historical 512-token FP-driver-conditioned training sampling or H128 backward protocol. See `L7_OBJECTIVE_IDENTITY_AND_PARITY.md` for the exact distinction and real-GPU parity evidence.

All scores are lower-is-safer. Spearman uses exact-tie average ranks. Concordance excludes tied pairs and reports their counts separately. All displayed panels here have 120 untied rotation pairs and no ties. Partial association correlates the residuals of predictor and target midranks after separate OLS regressions on an intercept and L7_TOTAL midrank. A zero residual variance is unavailable, not zero association. Leave-one-rotation-out recalculates ranks after each omission. No p-value, new correlation cutoff, gradient-equivalence claim or training-advantage claim is introduced.

## P1_L7_RANKING_REDUNDANCY

Conclusion: **HIGH_OVERLAP_WITH_INFORMATIVE_NONIDENTICAL_RANKINGS**. This descriptive label is not a new numerical decision threshold.

| Document | Calibration view | rho P1/Total | rho P1/Functional | rho Total/B0 | rho P3/Total | P1/Total concordance | Discordant pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 37702 | 256 | 0.882353 | 0.873529 | 0.588235 | 0.947059 | 106/120 | 14 |
| 37702 | 1024 | 0.873529 | 0.870588 | 0.820588 | 0.920588 | 105/120 | 15 |
| 37702 | Equal mean | 0.900000 | 0.873529 | 0.800000 | 0.944118 | 107/120 | 13 |
| 5396 | 256 | 0.941176 | 0.929412 | 0.655882 | 0.935294 | 108/120 | 12 |
| 5396 | 1024 | 0.891176 | 0.894118 | 0.750000 | 0.905882 | 107/120 | 13 |
| 5396 | Equal mean | 0.852941 | 0.861765 | 0.714706 | 0.920588 | 101/120 | 19 |

For the fixed aggregate, P1/Total leave-one-rotation-out rho ranges are [0.878571, 0.935714] on 37702 and [0.821429, 0.896429] on 5396. High overlap is robust in these rosters. It does not imply identical rankings, equal gradients, equivalent objectives over other distributions, or equivalent optimization.

## Frozen Route A decision

The route was computed and frozen from complete short scores before this analysis execution opened historical Future-KL labels. Both full-roster midrank gaps must be at least 3. Priority is A before B before C, without considering any old or new long-horizon outcome.

There are two eligible A pairs on 37702 and five on 5396. The first deterministic pair in each historical document is:

| Document | Pair | P1 winner | Total winner | P1 rank gap | Total rank gap |
| --- | --- | --- | --- | ---: | ---: |
| 37702 | H_to_L6_a50 / H_to_L7_a50 | H_to_L6_a50 | H_to_L7_a50 | 5 | 3 |
| 5396 | H_to_L7R_a50 / H_to_L7_a50 | H_to_L7R_a50 | H_to_L7_a50 | 5 | 6 |

These historical pairs establish route eligibility only. They are **not** the two prospective pairs and will not be launched as independent validation. The complete opposite-pair lists, including eligible B pairs, are retained in `phase_a_analysis_20261011/PHASE_A_ROUTE.json`. B does not replace the higher-priority eligible A comparison.

## P1_INCREMENTAL_INFORMATION_OVER_L7

Conclusion: **POSITIVE_EXPLORATORY_RESIDUAL_RANK_ASSOCIATION; PERSISTENT_INCREMENTAL_VALUE_UNCONFIRMED**.

The following labels are historical pulse-prefix followed by **no further QDQ**, from a different runtime. Every statistic is `DISCOVERY_ONLY`. Four anchor panels are two independent documents, not four documents. Neither these targets nor the residual associations selected the route, calibration positions, layer weighting, metric scaling or any prospective document.

| Document | Old anchor | rho P1/Future-KL | rho Total/Future-KL | rho Functional/Future-KL | rho B0/Future-KL | rho P3/Future-KL | Partial P1/Future-KL controlling Total | Partial LOO range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 37702 | 8192 | 0.726471 | 0.726471 | 0.735294 | 0.567647 | 0.697059 | 0.242527 | [0.098766, 0.406352] |
| 37702 | 2048 | 0.870588 | 0.885294 | 0.864706 | 0.697059 | 0.858824 | 0.364196 | [0.101318, 0.812586] |
| 5396 | 8192 | 0.844118 | 0.820588 | 0.791176 | 0.741176 | 0.888235 | 0.483359 | [0.109310, 0.602197] |
| 5396 | 2048 | 0.752941 | 0.735294 | 0.705882 | 0.588235 | 0.726471 | 0.355518 | [0.241615, 0.621980] |

These are the pre-fixed equal-window aggregates. All 12 document/window/anchor P1 partial associations are positive, ranging from 0.147488 to 0.526931. Aggregate LOO ranges remain positive but their magnitudes vary appreciably. The exact per-window statistics, residuals and all omissions are delivered separately. This is exploratory evidence of some nonredundant rank information in these natural rotations. It is not a test of genuine persistent 32K predictive superiority, an independent replication, or a causal explanation of historical training performance. The 37702 secondary panel actually has higher marginal rho for Total than P1, so this report does not claim uniform P1 superiority even on old labels.

## Complete tables and validity

`MATCHED_SHORT_HORIZON_SCORES.csv`: 96 rows = 2 documents × 16 conditions × (256, 1024, equal mean), seven scientific metrics with input/rotation/FP/runner/GPU identities.

`ROTATION_MIDRANKS.csv`: 672 rows. `PAIRWISE_RANKINGS.csv`: 15,120 rows for every pair of seven metrics, all 120 rotation pairs, all six scoring panels. `RANKING_REDUNDANCY_STATISTICS.csv`: 126 rows. `RANKING_REDUNDANCY_LEAVE_ONE_OUT.csv`: 2,016 rows.

`DISCOVERY_ONLY_FUTURE_KL_ASSOCIATIONS.csv`: 84 rows. `DISCOVERY_ONLY_RANK_RESIDUALS.csv`: 1,344 rows. `DISCOVERY_ONLY_LEAVE_ONE_OUT.csv`: 1,344 rows. Blank conditional associations mean unavailable residual variance; they are not numeric zero.

Every formal trajectory performed 20,736 actual layer/token QDQ writebacks and 20,718 verified next-kernel consumptions. Both documents passed same-GPU FP replay parity, all 72 real helper checks, finite results, full-roster coverage, canonical/sidecar SHA verification and unchanged frozen scoring code. The historical 80 canonicals, 80 sidecars and permanent Attempt04/05 STOPs were rehashed unchanged. Independent CPU verification checked every exported cell and recalculated all 126 correlations with pandas midranks and NumPy.

Phase A cost: 0.999650581743982 summed GPU-hours, a conservative launch-to-completion bound including both GPUs, smoke, loading and FP. The runner-accounted sum is 0.999577368696531 GPU-hours. Waiting, CPU analysis and unrelated experiments are excluded. The initial zero ledger remains frozen; the actual Phase A ledger is published in `phase_a_deliverables_20261011/GPU_COST_LEDGER.json`.

## Next authorized step and unresolved questions

Route A admits independent-document audit and calibration-only selection. It does not itself authorize any 32K launch. Two genuinely new documents must be selected in fixed seed/hash order excluding all relevant prior documents, including 48693 and 51410. Each must independently yield an eligible opposite pair from all 16 rotations, with unchanged gap/min-gap/scientific-key rules. Both pairs, winners and primary metric must be frozen before any new long outcome.

Full-model persistent QDQ qualification, hybrid-cache correctness, same-GPU FP reference, actual microbenchmark/storage/continuity checks, and a conservative complete four-unit-plus-reference total ≤16 GPU-hours are still required. None is reported as completed by this Phase A report. Q2 is unresolved. Q3 optimization explanations remain hypotheses; no training, gradients, optimizer update or temporal causal experiment has been executed.
