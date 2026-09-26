# Metric definitions — QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1

Metrics were frozen before joining mechanism values to outcome labels.

## M0 state error

After each teacher-forced token updates the recurrent state and the existing C128 QDQ is applied:

`E_t = S_t(condition, recovered to FP basis) - S_t(FP)` and `M0_t,l = ||E_t,l||_F`.

Hadamard states are recovered with `H128 @ S_hadamard`. Absolute and FP-relative errors are retained. Aggregates: mean, median, p95, max.

## M5 functional error

Frozen existing definition: `M5 = ||W_O D_g D_w J_RMS(o) E^T q||_2`.

Timing is explicit: cached state error `E_(t-1)` is evaluated against FP query `q_t`, then passed through the FP RMS/gate/out-projection linearization. Values are computed for every timestep and GDN layer. Aggregates: mean, median, p95, max.

## Persistence

With alpha=0.9 and K=[1, 4, 8, 16, 32]:

`P_t = sum_(k in K) alpha^k * ||E_(t+k)|| / (||E_t|| + 1e-12)`.

The state norm combines all GDN layers by L2. Tail positions without any future horizon are missing, not zero. Aggregates: mean, p95, max.

## Decision sensitivity

For FP margin `m_t = z1-z2`, and condition shift of the FP-winning token `delta_z_t`:

`D_t = |delta_z_t| / max(m_t, 1e-06)`.

The report includes mean, p95, max, count/fraction above 1, exact/near-tie counts, and actual top-1 changes.

## Statistical populations

- Failure: 52 FP-correct trajectories; positive means Native INT8 failed (34 positives).
- Rescue: 34 FP-correct Native-INT8 failures; positive means Key-Hadamard rescued the trajectory (23 positives).
- Rescue risk uses lower Hadamard risk as positive direction. Risk-reduction analysis uses `Native - Hadamard` with higher reduction as positive.
- AUROC, Spearman, Pearson, and paired-trajectory bootstrap 95% confidence intervals use 2000 resamples.
- Equal-weight composites use unsupervised z-scores; no label-fitted weights or thresholds are used.

## Non-finite values

The raw CSV preserves numerical overflow as `inf`/`nan`. Trajectory summaries report finite-prefix statistics and expose exact overflow counts/onsets in `token_level_metrics.json`. Only one Group D trajectory is affected, so primary A/B/C failure and rescue statistics require no censoring or imputation.
