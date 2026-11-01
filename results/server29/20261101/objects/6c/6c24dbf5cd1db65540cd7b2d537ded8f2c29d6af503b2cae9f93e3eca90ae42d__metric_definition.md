# Metric definitions

All metric directions and aggregations below are frozen before any future capture is joined to AIME labels.

## Representation fidelity

For clean recurrent state `S_t` and quantized-condition state `S_hat_t`:

`state_error_t = ||S_t - S_hat_t||_2 / (||S_t||_2 + eps)`.

Aggregate over layer × timestep with mean, median, p95, and max. Also retain max absolute value, p50/p95/p99/p99.9, clipping ratio, channel max/RMS imbalance, and excess kurtosis. Existing AIME artifacts do not contain the required tensors or summaries.

## Functional sensitivity (M5)

Frozen existing definition:

`M5_t = || W_O D_g D_w J_RMS(o) E_t^T q_t ||_2`.

Retain per-layer and per-timestep values; aggregate mean, median, p95, and max. No normalization may be refit using AIME labels. Existing independent M5 panels are reported separately; AIME-aligned M5 is absent.

## Persistence / recurrent amplification

For fixed `alpha` and horizon set `K` declared in capture metadata:

`P_t = sum_{k in K} alpha^k ||E_{t+k}||_2 / (||E_t||_2 + eps)`.

Primary aggregation is p95; mean and max are secondary. The clean and perturbed paths must share the same teacher-forced token suffix so persistence is not confounded by sampled-token divergence. Existing AIME artifacts do not contain state-error trajectories.

## Decision-boundary sensitivity

Let `m_t = z_(1),t - z_(2),t` be the FP top-1/top-2 margin. Let `delta_z_t` be the change in the FP-winning token's logit under quantization (and additionally store the worst top-2 contrast shift). The frozen primary statistic is:

`D_t = |delta_z_t| / (m_t + eps)`.

Retain mean, p95, max, and `count(D_t > 1)`. Logits must be taken on the same teacher-forced prefix. Existing AIME artifacts do not contain logits.

## Outcomes and populations

- INT8 failure: `FP correct AND INT8_C128 wrong`; primary population is the 52 FP-correct trajectories.
- Hadamard rescue: `FP correct AND INT8_C128 wrong AND INT8_C128+Key-H correct`; primary population is the 34 FP-correct INT8 failures.
- Group A/B/C/D assignments follow these exact Boolean rules and use labels only as outcomes, never inside a metric.

## Statistics

For each captured metric, report raw-direction Spearman, Pearson when nondegenerate, and ROC AUC with higher metric value predeclared as higher failure risk. No orientation flipping, threshold selection, normalization fitting, or metric redesign is allowed after labels are inspected. Tail statistics are predeclared because rare recurrent excursions are the scientific hypothesis.

## Non-mechanism diagnostics

Common-prefix length, generated-token count, repetition, and self-correction are post-hoc trajectory descriptors. FP-only behavioral descriptors are evaluated exploratorily but are ineligible as rotation optimization objectives because they require an expensive generated reasoning trajectory and do not directly measure recurrent quantization quality.
