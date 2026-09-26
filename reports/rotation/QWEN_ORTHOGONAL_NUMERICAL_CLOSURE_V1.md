# QWEN_ORTHOGONAL_NUMERICAL_CLOSURE_V1

## Verdict

`ORTHOGONAL_NUMERICAL_CLOSURE = PASS`

`RECOMMENDED_ROTATION_DTYPE = FP32`

Qwen's canonical q/k rotation already uses FP32 matrix multiplication and does not recast q/k to BF16 before the GDN core. The remaining finite-precision seed is an approximately `1.5e-7` FP32 dense-matmul roundtrip error, which is amplified through later recurrent and normalization stages. No structural equation error or audit side effect was found.

## Provenance

- H: analytic normalized H128.
- R0: `14a637c7f94026d70b23d02149f8ca599dd6558c97a9c954956b8640ff2c0595`.
- R1: `5d58a163eeb257ff829be138abbb70b22b529ffcb352f9f81d0239fa9960895b`.
- R2: `6633a10c6e9522be8068d2bd570d80255256ad6131d237c77f3079a478d20093`.
- Previous-gate hash match: PASS.

## Matrix-only roundtrip

Values are the maximum q/k relative L2 for the representative real layer-0 tensors.

| Rotation | FP64 | FP32 | BF16 storage + FP32, pre-final-cast | post-final-cast | native BF16 diagnostic |
|---|---:|---:|---:|---:|---:|
| H | 3.90e-16 | 1.42e-7 | 1.42e-7 | 1.24e-7 | 1.68e-3 |
| R0 | 9.34e-16 | 1.56e-7 | 1.56e-7 | 7.68e-9 | 2.89e-3 |
| R1 | 9.35e-16 | 1.50e-7 | 1.50e-7 | 1.86e-9 | 3.01e-3 |
| R2 | 9.81e-16 | 1.61e-7 | 1.61e-7 | 7.51e-9 | 2.87e-3 |

The native-BF16 diagnostic is not the production Qwen path. Its internal accumulation dtype could not be observed reliably and is recorded as `ACCUMULATION_DTYPE_UNKNOWN`.

Mean representative decomposition:

- FP32 rotation roundtrip: `1.506886e-7`.
- BF16 intermediate-storage contribution in production: `0` (there is no q/k BF16 recast before GDN).
- Signed final BF16 recast delta in the simulation: `-1.303508e-7`; this grid-snapping delta is diagnostic only.

`QWEN_DOMINANT_ERROR_SOURCE = ROTATION_MATMUL`

## Interface and side effects

- q/k forward rotations: functional inference operations.
- recurrent state: remains in rotated Key coordinates.
- q/k and state inverse transformations: audit-only detached recoveries.
- functional Key recovery after GDN: not required because the Value/readout dimension stays native.

`AUDIT_PATH_SIDE_EFFECT = NO` (bitwise-equal logits with recovery enabled/disabled).

## Layerwise amplification

- Earliest nonzero error: layer-0 q/k FP32 rotation roundtrip, `2.43e-7` worst observed for dense R.
- Layer-0 core relative L2: approximately `2.36e-3` for H and all R seeds.
- Largest instrumented amplification: R0, layer 16, post-norm/gate, relative L2 `0.0770162`.
- Largest core-only late-layer values remain on the same scale for H and random R; there is no random-R-specific order-of-magnitude jump.

## Teacher-forced comparison

The FP32-reference path is byte-for-byte the current canonical Qwen rotation implementation.

| Path | 3x128 relL2 median/p95/max | top-1 | 1x512 relL2 median/p95/max | top-1 |
|---|---|---:|---|---:|
| H current | .009166/.019339/.027327 | 387/387 | .008886/.019382/.220018 | 511/513 |
| R0 current = FP32 reference | .009657/.020377/.054551 | 386/387 | .009031/.021845/.176902 | 513/513 |
| R1 current = FP32 reference | .009534/.021968/.048071 | 387/387 | .009146/.021619/.241449 | 513/513 |
| R2 current = FP32 reference | .009508/.021020/.044376 | 386/387 | .009090/.020789/.131690 | 512/513 |

All conditions have zero nonfinite values. Random-R errors and behavioral stability remain in the Hadamard envelope.

## Scientific classification

- Mathematical equivalence: PASS.
- Structural equivalence: PASS.
- Numerical stability: PASS.
- Precision policy: retain FP32 q/k rotation and explicit `R.T` only for detached audit recovery.
- Safe for dense orthogonal oracle planning: YES.
