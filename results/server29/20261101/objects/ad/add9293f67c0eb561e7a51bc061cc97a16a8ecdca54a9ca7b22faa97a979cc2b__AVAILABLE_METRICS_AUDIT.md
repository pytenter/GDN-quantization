# Available metrics audit

## Canonical inventory

- Verified canonical + sidecar bytes: **80/80**
- Unique `(document, condition, anchor, endpoint)` keys: **80**; duplicates: **0**
- Primary: **41/64**; Secondary: **39/64**; complete pairs: **39**; unpaired Primary: **2**
- Core complete documents: `pg19-test-37702`, `pg19-test-5396` (64 units, 32 pairs)
- Auxiliary incomplete documents: `pg19-test-28988`, `pg19-test-26493` (16 units, 7 pairs + 2 unpaired Primary)
- All scalar analysis fields are present, finite, and strictly positive where log sensitivity is used. No imputation was performed.

## Actual metric definitions

- `RAW_ACC_ERROR`: squared Frobenius norm of the anchor error `E = quantized_state - FP_state`.
- `LOCAL_K3_ACCUMULATED`: squared norm of historical K3 query/readout applied to anchor error.
- `HISTORICAL_J_PROP`: unchanged historical replay-step statistic, averaged over its historical support; retained separately and **not** used as the primary propagation-magnitude variable.
- `J_INJ`, `J_ACC`: unchanged historical replay-step summaries.
- `JPROP` in this analysis: canonical `PULSE_PROP_ENERGY_AUC`, the sum over future tokens of squared Frobenius norm between the FP path and the pulse-injected path.
- `POR`: sum over the same future tokens of squared K3 readout difference between those two paths.
- `POR_GAIN`: `POR / RAW_ACC_ERROR`, with the original zero-denominator rule.
- `FutureKL`: canonical `TARGET_FUTURE_KL_AUC`, the accumulated full-logit KL between the FP reference path and the pulse-injected path.

`POR` is calculated in `bridge.py::replay` from state-path differences and future K3 query/readout records. `FutureKL` is calculated separately in `bridge.py::pulse_target` from full logits. POR does **not** read the Future-KL scalar or logit target, so there is no direct label leakage. It does use the realized future token/query/readout sequence, making it a retrospective future-aware diagnostic rather than a presently deployable training objective.

## Alignment

For each unit, JPROP, POR and FutureKL use the same document, condition, anchor, endpoint=32768 and future window. Primary and Secondary use identical metric definitions but different pulse anchors and hence different future-token counts. All units preserve same-GPU FP pairing, initial pulse identity and no-further-QDQ gates.

## Rotation comparability and missingness

Each complete document has the full frozen 16-condition rotation panel at both anchors. Auxiliary documents are analyzed only on observed conditions. Their missingness follows scheduler execution order after the reboot rather than random sampling, so auxiliary correlations are selection-biased descriptive checks only.
