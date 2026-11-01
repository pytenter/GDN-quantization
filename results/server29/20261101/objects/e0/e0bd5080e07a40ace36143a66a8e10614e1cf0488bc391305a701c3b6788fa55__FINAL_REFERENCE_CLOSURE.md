# FINAL REFERENCE CLOSURE

`REFERENCE_CLOSURE_STATUS = PASS`

`QWEN_HISTORICAL_DATASET = HuggingFaceH4/MATH-500 test split, fixed 30-item subset`

`QWEN_HADAMARD_OPERATOR = PASS`

`QWEN_HADAMARD_FULLPATH = PASS`

`LING_HADAMARD_OPERATOR = PASS`

`LING_HADAMARD_FULLPATH = PASS`

`NUMERICAL_PRECISION_FINDING = BF16 / optimized-kernel numerical floor confirmed; FP32 reference/manual paths close near 1e-6`

`SGLANG_READY = YES` (reference-equivalence gate only; installation was not performed, and Server A persistent storage remains blocked)

## Final conclusions

1. **Provenance conflict:** the Qwen 23/30, 7/30, 18/30 results are MATH-500, not AIME-2024. Exact and normalized matching are 30/30 against MATH-500 and 0/30 against AIME-2024.
2. **Qwen old failure:** not a basis or axis error. FP64 operator error is 0; FP32 operator state/readout errors are 2.902e-07/2.988e-07. BF16 transform/cast perturbations compound through deep layers. The repaired FP32 model/reference path applies H after q/k normalization, before both prefill and decode GDN kernels, never rotates convolution state, and never double-rotates cache state.
3. **Ling old failure:** not a value-axis or inverse-placement error. BF16 FLA gives percent-level drift; FP32 values cannot be mixed with BF16 operands in the Triton chunk kernel, and even an all-FP32 optimized Triton smoke retains 1.764e-02 max logit rel-L2. The repaired reference path uses the same manual FP32 KDA recurrence for native and rotated branches, rotates final post-convolution/SiLU `v`, keeps cache in `S H`, and applies exactly one inverse before `o_norm`/dynamic gate/merge/`o_proj`.
4. **What changed:** immutable snapshots were committed first; new reference-only test drivers were added in isolated trees. No original source file was edited. No SGLang package was installed and no AIME-2026 generation was started.

## Strict 3 prompts × 128 teacher-forced tokens

| Metric | Qwen Key-Hadamard FP32 | Ling Value-Hadamard manual FP32 |
|---|---:|---:|
| Compared steps | 387 | 387 |
| Logit rel-L2 median | 6.971742e-07 | 2.802551e-07 |
| Logit rel-L2 p95 | 1.580396e-06 | 4.552240e-07 |
| Logit rel-L2 max | 1.829874e-05 | 1.007773e-06 |
| State rel-L2 median | 1.072105e-06 | 6.614209e-07 |
| State rel-L2 p95 | 3.327516e-06 | 1.071911e-06 |
| State rel-L2 max | 4.740057e-06 | 1.536326e-06 |
| Top-1 agreement | 100.0% | 100.0% |
| KL max | 1.519203e-07 | 1.868162e-07 |

Ling recovered-core/post-norm-input rel-L2 max: `2.330829e-06` / `2.330829e-06`. Prefill and decode basis checks pass for both architectures; `DOUBLE_ROTATION = NO`; Ling inverse count is exactly one per KDA call.

## Precision ablation

| Architecture | FP64 operator state/readout | FP32 operator state/readout | Native BF16 operator state/readout | Old/BF16 fullpath max | Repaired formal max |
|---|---:|---:|---:|---:|---:|
| Qwen | 0.000e+00 / 0.000e+00 | 2.902e-07 / 2.988e-07 | 2.847e-03 / 2.506e-03 | 2.795e-02 | 1.830e-05 |
| Ling | 0.000e+00 / 0.000e+00 | 3.193e-07 / 2.603e-07 | 2.810e-03 / 2.482e-03 | 7.275e-02 | 1.008e-06 |

The normalized H128 algebra gate passes on both servers. FP64 max error is at most `5.730e-16`; FP32 max is at most `2.896e-07`; BF16 max is at most `2.522e-03`.

## Commits

- Server A provenance-only: `5eb3cd50cb3ae3b4f58140c678e86cd4d77bff0f`
- Server A Qwen closure: `7167908a529be9a919fff8c67cd8bb550dba0209`
- Server A precision diagnostics: `c959e2d67656e8dffe8562494298440992241d32`
- Server B provenance-only: `7fd931e414cff3f769689d2d24a17c9a3b949f62`
- Server B Ling closure and combined evidence: `6f4a075a9dc10a2d11932e866875158eb2eed4f5`

## Resource state

At final audit there are no active compute processes. Server A GPUs 0–3 show 0% utilization and 3 MiB each; Server B GPUs 0–1 show 0% utilization (driver/context memory only). `GPU_IDLE = YES`.
