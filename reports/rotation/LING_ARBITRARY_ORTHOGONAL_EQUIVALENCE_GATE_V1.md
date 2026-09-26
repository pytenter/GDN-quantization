# LING_ARBITRARY_ORTHOGONAL_EQUIVALENCE_GATE_V1

## Verdict

`status = GENERAL_ORTHOGONAL_EQUIVALENCE_PASS_WITH_NUMERICAL_CAVEAT`

The FP64 KDA operator identity passes for Identity, H128, and all three dense SO(128) matrices. Corrected prefill endpoint semantics remain intact. The full-model strict gate does not pass because finite-precision Value-basis roundtrips cause a small number of top-1 flips. Random-R error remains on the same scale as the Hadamard control, so this is not a general-orthogonal algebra failure.

`STRICT_TOP1_GATE = FAIL`

`SAFE_TO_START_DENSE_ORTHOGONAL_ORACLE = NO`

## Matrix sanity

Runtime: PyTorch 2.7.1+cu128, CPU FP64 Gaussian QR, diagonal-sign canonicalization, determinant forced positive.

| Matrix | det(R) | max abs(RT R - I) | Frobenius residual |
|---|---:|---:|---:|
| R0 | 1.0000000000000044 | 1.33226762955019e-15 | 9.89175255021083e-15 |
| R1 | 1.0000000000000013 | 9.99200722162641e-16 | 9.93772289417777e-15 |
| R2 | 1.0000000000000044 | 1.11022302462516e-15 | 1.04882158754192e-14 |

All Stage-0 matrix gates pass the `1e-12` threshold. Exact hashes are in `rotation_manifest.json`.

## Controls and stages

- Identity: PASS; 3 prompts x 128 tokens, zero logit and state error.
- FP64 operator: PASS for H/R0/R1/R2.
- Prefill top-1: 100% for all conditions (four prefill comparisons per non-identity condition, including the stress prompt).
- First decode: no structural equation break. The first nonzero difference is the layer-0 Value rotation plus BF16 boundary cast; the recovered Value relative L2 is 0.00175679 for H and 0.00170679 for R0.
- Prefill endpoint continuity: PASS, maximum relative L2 = 0 for every condition.
- Nonfinite values: 0.

| Condition | 3x128 top-1 | 1x512 top-1 | primary logit relL2 median / p95 / max | primary recovered-state relL2 median / p95 / max |
|---|---:|---:|---:|---:|
| H128 | 385/387 | 509/513 | 0.0142915 / 0.0434080 / 0.128260 | 0.0169731 / 0.0275925 / 0.0392142 |
| R0 | 383/387 | 508/513 | 0.0138281 / 0.0469078 / 0.138533 | 0.0174086 / 0.0275555 / 0.0400444 |
| R1 | 383/387 | 509/513 | 0.0140333 / 0.0442155 / 0.132248 | 0.0164413 / 0.0270533 / 0.0376033 |
| R2 | 380/387 | 505/513 | 0.0155367 / 0.0446215 / 0.131431 | 0.0203700 / 0.0256793 / 0.0307979 |

Primary median random/H error ratios are R0=0.9676, R1=0.9819, R2=1.0871. There is no order-of-magnitude random-R degradation.

First top-1 divergence tokens:

- H: primary token 34; stress token 29.
- R0: primary token 30; stress token 30.
- R1: primary token 34; stress token 29.
- R2: primary token 34; stress token 63.

## Corrected KDA semantics

`KDA_ROTATION_SEMANTICS_VERSION = CORRECTED_PREFILL_ENDPOINT_V2`

`LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE = PASS`

`REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO`

The recurrent state returned by the prefill kernel is written directly to cache in the rotated Value basis and consumed by the first decode step without an extra H/R or transpose application.

## Rotation semantics audit

The historical Hadamard interface relied on H being symmetric/self-inverse in audit recovery. The generalized implementation uses explicit forward `x @ R`, inverse `x @ R.T`, and explicit Value-axis state recovery.

`HADAMARD_SELF_INVERSE_DEPENDENCY_FOUND = YES_IN_HISTORICAL_INTERFACE`

`ROOT_CAUSE_CLASSIFICATION = FINITE_PRECISION_VALUE_BASIS_ROUNDTRIP_AND_LAYERWISE_AMPLIFICATION`

`STRUCTURAL_EQUATION_BREAK = NONE_OBSERVED`

## Safety

- INT8 and state quantization were disabled.
- No AIME26 generation was run by this gate.
- Model weights were not modified.
- Existing formal outputs were not modified.
