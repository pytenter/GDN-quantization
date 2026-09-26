# QWEN_ARBITRARY_ORTHOGONAL_EQUIVALENCE_GATE_V1

## Verdict

`status = GENERAL_ORTHOGONAL_EQUIVALENCE_PASS_WITH_NUMERICAL_CAVEAT`

The FP64 GDN operator identity passes for Identity, H128, and all three dense SO(128) matrices. The full-model strict gate does not pass because a few finite-precision top-1 flips occur. Random-R error is the same scale as the Hadamard control, so this is not a general-orthogonal algebra failure.

`STRICT_TOP1_GATE = FAIL`

`SAFE_TO_START_DENSE_ORTHOGONAL_ORACLE = NO`

## Matrix sanity

Runtime: PyTorch 2.5.1+cu121, CPU FP64 Gaussian QR, diagonal-sign canonicalization, determinant forced positive.

| Matrix | det(R) | max abs(RT R - I) | Frobenius residual |
|---|---:|---:|---:|
| R0 | 1.0000000000000056 | 1.1102230246251565e-15 | 9.87607388949762e-15 |
| R1 | 1.0000000000000027 | 8.881784197001252e-16 | 9.56564669127333e-15 |
| R2 | 0.9999999999999992 | 6.661338147750939e-16 | 9.81578595651885e-15 |

All Stage-0 matrix gates pass the `1e-12` threshold. Exact hashes are in `rotation_manifest.json`.

## Controls and stages

- Identity: PASS; 3 prompts × 128 tokens, zero logit and state error.
- FP64 operator: PASS for H/R0/R1/R2.
- Prefill top-1: 100% for all conditions (four prefill comparisons per non-identity condition, including the stress prompt).
- First decode: no structural equation break. The first nonzero difference is FP32 q/k rotation roundtrip at GDN layer 0 (~2e-7 relative); the layer-0 core readout amplifies it to ~2.36e-3 relative.
- Nonfinite values: 0.

| Condition | 3×128 top-1 | 1×512 top-1 | primary logit relL2 median / p95 / max | primary recovered-state relL2 median / p95 / max |
|---|---:|---:|---:|---:|
| H128 | 387/387 | 511/513 | 0.00916593 / 0.0193385 / 0.0273266 | 0.00978438 / 0.0124034 / 0.0143259 |
| R0 | 386/387 | 513/513 | 0.00965721 / 0.0203770 / 0.0545513 | 0.0100621 / 0.0133701 / 0.0154326 |
| R1 | 387/387 | 513/513 | 0.00953408 / 0.0219685 / 0.0480706 | 0.0100844 / 0.0132395 / 0.0153407 |
| R2 | 386/387 | 512/513 | 0.00950811 / 0.0210200 / 0.0443760 | 0.0100163 / 0.0124830 / 0.0141404 |

Primary median random/H error ratios are R0=1.0536, R1=1.0402, R2=1.0373. There is no order-of-magnitude random-R degradation.

Top-1 divergences:

- H: stress token 332 and 367 on `test/geometry/702.json`.
- R0: primary token 109 on `test/algebra/1332.json`.
- R1: none.
- R2: primary token 86 on `test/counting_and_probability/119.json`; stress token 332 on `test/geometry/702.json`.

## Rotation semantics audit

The historical Hadamard interface relied on H being symmetric/self-inverse in audit recovery. The generalized implementation uses explicit forward `x @ R`, inverse `x @ R.T`, and Qwen state recovery `R @ S_rot` over the Key axis.

`HADAMARD_SELF_INVERSE_DEPENDENCY_FOUND = YES_IN_HISTORICAL_INTERFACE`

`ROOT_CAUSE_CLASSIFICATION = FINITE_PRECISION_BASIS_ROUNDTRIP_AND_LAYERWISE_AMPLIFICATION`

`STRUCTURAL_EQUATION_BREAK = NONE_OBSERVED`

## Safety

- INT8 and state quantization were disabled.
- No AIME26 generation was run.
- Model weights were not modified.
- Existing formal outputs were not modified.
