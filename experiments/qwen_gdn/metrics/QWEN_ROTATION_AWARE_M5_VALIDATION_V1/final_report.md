# Qwen Rotation-Aware M5 Validation V1

## Executive conclusion

`ROTATION_AWARE_M5_GATE = FAIL`. Promising variants: `NONE`. No persistent-KL replay was executed.

## Coordinate and transformation audit

The prior “raw” M5 is already `M5_canonical`: Hadamard recurrent state is pulled back with the normalized symmetric orthogonal H128 before the error is formed, and the functional map uses the shared native FP query, RMS point, gate, norm weight and output projection. In exact arithmetic, `S_native=H S_rotated` and `q_rotated=q_native H` preserve the GDN readout. Consequently `M5_canonical` is not a newly repaired metric; it is an explicit name for the existing implementation.

The replay parity gate reproduced prior M0/M5 p95 to ~1e-15 and logit-derived scalars exactly. In the preregistered four-sample, 512-token FP-equivalence audit, Identity residual is zero. Hadamard finite-precision residuals and their ratios to the same-point INT8 risks are recorded in `fp_equivalent_coordinate_audit.json`. The shared FP-signal denominator has fraction <1e-8 = 0.000%, <1e-6 = 0.000%, and <1e-4 = 0.000%; denominator pathology = `False`.

## Primitive scale source

The largest median absolute Hadamard/Identity log10 shifts are:

[
  {
    "component": "rms_jvp_norm",
    "p50": 0.24462904781103134,
    "p95": 2.2398349881172184,
    "max": 15.666481971740723
  },
  {
    "component": "m5_canonical",
    "p50": 0.22926175594329834,
    "p95": 1.9619288086891178,
    "max": 16.215219497680664
  },
  {
    "component": "m5_over_fp_signal",
    "p50": 0.22926174849271774,
    "p95": 1.9619288086891178,
    "max": 16.215219497680664
  },
  {
    "component": "weighted_jvp_norm",
    "p50": 0.22186394780874252,
    "p95": 1.9726143717765812,
    "max": 16.210914611816406
  },
  {
    "component": "m5_over_m0",
    "p50": 0.17041727155447006,
    "p95": 0.8305992394685746,
    "max": 12.621748924255371
  }
]

M0/M5 shared-scale evidence: log10(M0) versus log10(M5) Pearson is 0.302 in Identity and 0.880 in Hadamard; paired log10(H/I) M0 versus M5 Pearson is 0.939. Layer, sample, token and head-localization tables are included without selecting favorable subsets.

## Candidate metrics and four gates

| Metric | Identity median trajectory p95 | Hadamard median trajectory p95 | Coordinate | Within both | Direction | Rescue | Classification |
|---|---:|---:|---|---|---|---|---|
| raw_M5 | 6.83016 | 29.642 | FAIL | PASS | INCORRECT | FAIL | DIAGNOSTIC_ONLY |
| M5_canonical | 6.83016 | 29.642 | FAIL | PASS | INCORRECT | FAIL | FAILED_COORDINATE_SANITY |
| FP_signal_normalized_M5 | 0.383352 | 2.2004 | FAIL | PASS | INCORRECT | FAIL | FAILED_COORDINATE_SANITY |
| M5_over_M0 | 1.2144 | 1.84627 | FAIL | FAIL | INCORRECT | FAIL | FAILED_COORDINATE_SANITY |

`Delta_M5` is `DIAGNOSTIC_ONLY` because the exact reference is zero and it collapses to M5_canonical. `relative_M5_reference` is `NOT_APPLICABLE`: dividing by zero or by a tiny finite-precision coordinate residual would manufacture a denominator pathology.

## Within-rotation diagnosis

Best descriptive within-rotation metric among the preregistered variants: `FP_signal_normalized_M5`. Both rotations are reported independently with AUROC, 5000-sample bootstrap CI, Spearman and Cliff's delta in `within_rotation_analysis/`. Raw M5 is compared directly rather than replaced by a favorable subset.

## Rescue association

Groups were reconstructed from true IDs: `{"hadamard_rescue": 23, "stable_correct": 18, "stable_failure": 18}`. Best descriptive rescue association: `FP_signal_normalized_M5`. `paired_metric_deltas.csv` contains every clean pair; `rescue_statistics.json` reports P(delta<0), Wilson interval, sign test, one-sided 10,000-permutation comparison against stable failure, and Cliff's delta.

## Decision

`ROTATION_AWARE_M5_GATE = FAIL`. No mathematically justified M5 variant clears all four gates. The preserved persistent-functional pipeline is now eligible as a fallback, but this task stops here as instructed and does not launch it.

No model, quantizer, rotation, kernel semantics or scorer was modified; no new token was generated.
