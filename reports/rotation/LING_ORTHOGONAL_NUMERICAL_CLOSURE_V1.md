# LING_ORTHOGONAL_NUMERICAL_CLOSURE_V1

## Verdict

`ORTHOGONAL_NUMERICAL_CLOSURE = PASS_WITH_PRECISION_POLICY`

`RECOMMENDED_ROTATION_DTYPE = FP32`

The ideal FP32 Value rotation roundtrip is approximately `1.57e-7`. Storing the forward-rotated Value intermediate in BF16 raises the recovered error by approximately `1.66e-3`, identifying the BF16 intermediate boundary as the dominant first numerical error. The FP32-reference end-to-end path remains in the Hadamard numerical envelope but is not uniformly better on every max-error or top-1 statistic, so the result supports an FP32 rotation policy—not an unreviewed full-model or cache dtype change.

## Provenance

- H: analytic normalized H128.
- R0: `256b08edba5e03fef9a9c189e850ce02d9ee8fd6176afbfa65ee2c91eae78f59`.
- R1: `729a7c8bc492b33041fe4587801e64caea0dd8584cc6d56659b4af3506add441`.
- R2: `c5fe7f092829177abff315ab2c45d56ebcd1094b20a1248d12d660ac5395b548`.
- Previous-gate hash match: PASS.

## Matrix-only roundtrip

Values are relative L2 for the representative real layer-0 Value tensor.

| Rotation | FP64 | FP32 | BF16 storage + FP32, pre-final-cast | post-final-cast | BF16 intermediate + FP32 rotation | native BF16 diagnostic |
|---|---:|---:|---:|---:|---:|---:|
| H | 4.31e-16 | 1.28e-7 | 1.28e-7 | 4.29e-10 | 1.663e-3 | 3.558e-3 |
| R0 | 1.02e-15 | 1.66e-7 | 1.66e-7 | 4.34e-9 | 1.661e-3 | 4.161e-3 |
| R1 | 1.02e-15 | 1.68e-7 | 1.68e-7 | 1.92e-10 | 1.628e-3 | 4.145e-3 |
| R2 | 1.02e-15 | 1.67e-7 | 1.67e-7 | 6.15e-9 | 1.682e-3 | 4.156e-3 |

The native-BF16 matmul accumulator is not observable from the public runtime interface and is recorded as `ACCUMULATION_DTYPE_UNKNOWN`.

Mean representative decomposition:

- ideal FP32 roundtrip: `1.570042e-7`;
- BF16 intermediate-storage contribution: `1.658337e-3`;
- signed final BF16 recast delta: `-2.015071e-5` (grid snapping, not negative physical noise).

`LING_DOMINANT_ERROR_SOURCE = BF16_INTERMEDIATE_CAST`

## Corrected semantics and side effects

- `KDA_ROTATION_SEMANTICS_VERSION = CORRECTED_PREFILL_ENDPOINT_V2`.
- recurrent state returned by prefill stays in rotated Value coordinates.
- `REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO`.
- the functional `R.T` is applied only to KDA core/readout output before RMSNorm/gate/o_proj.
- state and Value recoveries used for comparison are detached audit-only tensors.

`AUDIT_PATH_SIDE_EFFECT = NO` (bitwise-equal logits with audit recovery enabled/disabled).

## Layerwise amplification

- Earliest nonzero error: layer-0 Value BF16 intermediate storage, H=`0.00175679`, R0=`0.00170679`, R1=`0.00168028`, R2=`0.00165663` recovered relative L2.
- Layer-0 recovered core output: approximately `0.00218–0.00249`.
- Largest observed amplification: R0, layer 17, out_proj, relative L2 `0.315960`; the immediately preceding RMSNorm value is `0.311316`.
- H and random-R traces show the same amplification pattern; no random-R-only structural stage appears.

## Teacher-forced comparison

R0 is the representative dense random rotation for the FP32-reference trajectory.

| Path | 3x128 relL2 median/p95/max | top-1 | top-20 | 1x512 relL2 median/p95/max | top-1 |
|---|---|---:|---:|---|---:|
| H current | .014291/.043408/.128260 | 385/387 | .97610 | .008102/.025433/.096328 | 509/513 |
| R0 current | .013828/.046908/.138533 | 383/387 | .97726 | .007880/.024524/.084404 | 508/513 |
| R0 FP32 reference | .013161/.039207/.252902 | 382/387 | .97558 | .007982/.028427/.125996 | 507/513 |

The FP32 reference improves primary median by about 4.8% and p95 by about 16.4%, but worsens the isolated maximum and loses one additional top-1 position in both primary and stress sets. All paths have zero nonfinite values. The correct conclusion is numerical closure with an explicit precision policy, not that converting the whole recurrent path to FP32 is automatically superior.

## Scientific classification

- Mathematical equivalence: PASS.
- Structural equivalence: PASS.
- Numerical stability: PASS_WITH_PRECISION_POLICY.
- Precision policy: FP32 master rotation and FP32 forward/inverse matmuls; any change to recurrent cache/storage dtype must remain a separately reviewed ablation.
- Safe for dense orthogonal oracle planning: YES, subject to the documented FP32 policy and initialization parity gate.
