# LING_RUNTIME_REPEATABILITY_CLOSURE_V1

## Verdict

- `LING_REPEATABILITY_VERDICT = REPEATABILITY_NOT_CLOSED`
- `ROOT_CAUSE_CLASSIFICATION = PROCESS_INITIALIZATION_NONDETERMINISM`
- `PERSISTENT_EVALUATION_NOT_SCIENTIFICALLY_RESOLVED = YES`
- `16-doc rerun authorized = NO`
- `PREVIOUS_16DOC_REPLICATION = NOT_RERUN`
- `LING_PERSISTENT_EVALUATION_READY = NO`

The runtime noise is larger than the paired method effects. Therefore neither the failed earlier 16-document replication nor a 64-document persistent conclusion is rerun or reinterpreted here.

## Frozen inputs and provenance

- Deterministic selection: manifest indices 0, 7, and 15
- Manifest SHA256: `d3ffc64f53d80046343e81f8457279b753b8e068341028ac939fa76e1457d18f`
- Each document has 1,024 frozen token IDs; raw-text, token-ID, tokenizer, corpus, and source-manifest hashes are stored in `LING_REPEATABILITY_3DOC_MANIFEST.json`.
- Dense-State checkpoint: `dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318`
- Dense-Functional checkpoint: `65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c`
- Checkpoint provenance gate: PASS
- Corrected KDA semantics: `CORRECTED_PREFILL_ENDPOINT_V2`; redundant endpoint rotation: NO

## Same-process stage

One model process on GPU1 evaluated 3 documents × 4 conditions × 5 repeats. Every doc-condition had AUC range 0, AUC standard deviation 0, and identical byte hashes between repeats at prefill cache pre/post quantization, first decode input/writeback, horizon logits, and horizon caches.

| Condition | Result | AUC range across every doc-condition | Key-checkpoint bitwise repeatability |
|---|---|---:|---|
| Native INT8 | PASS | 0 | PASS |
| Hadamard | PASS | 0 | PASS |
| Dense-State | PASS | 0 | PASS |
| Dense-Functional | PASS | 0 | PASS |

`SAME_PROCESS_REPEATABILITY = PASS`.

## Fresh-process stage

The fixed document was `wikitext2raw-test-000`. Hadamard and Dense-State were each run five times in completely new Python processes without seed intervention and five times with Python, NumPy, Torch, and Torch CUDA seeds all fixed to 20260922.

| Condition / seed mode | Five AUCs | Range | Std. dev. | Bitwise |
|---|---|---:|---:|---|
| Hadamard / untouched | 0.024702, 0.049413, 0.024702, 0.025364, 0.064713 | 0.040011 | 0.016474 | FAIL |
| Hadamard / fixed | 0.024702, 0.025588, 0.048706, 0.025364, 0.031125 | 0.024004 | 0.009101 | FAIL |
| Dense-State / untouched | 0.027562, 0.022500, 0.034389, 0.037327, 0.037327 | 0.014827 | 0.005869 | FAIL |
| Dense-State / fixed | 0.034389, 0.037327, 0.022500, 0.037327, 0.037327 | 0.014827 | 0.005751 | FAIL |

`FRESH_PROCESS_REPEATABILITY = FAIL`. Fixed seeds reduce neither the qualitative failure nor the Dense-State range; random seeding is not the root cause.

## Earliest observed divergence

The first compact checkpoint already differed at the recurrent state returned by KDA prefill, before state quantization. Per-layer hashes locate the earliest affected KDA layer at layer 0. A separate two-process temporary capture quantified and then deleted the layer-0 tensors:

- `FIRST_DIVERGENT_LAYER = 0`
- `FIRST_DIVERGENT_MODULE = KDA prefill recurrent state/cache before quantization`
- shape/dtype: `[1, 16, 128, 128]`, FP32
- different elements: 18,026
- maximum absolute difference: `2.3484230e-05`
- relative L2 difference: `5.1008208e-05`
- first differing index: `[0, 0, 1, 1]`
- temporary tensor bytes: 2,100,576; cleaned: YES

This is a fresh-process divergence. Fine-grained same-process A7 tracing was not triggered because Stage 1 was exactly repeatable.

## Runtime-factor audit

Canonical settings were deterministic algorithms OFF, CUDA graph OFF, radix cache OFF, Mamba radix cache OFF, Torch compile OFF, CUDA matmul TF32 OFF, cuDNN benchmark OFF, and fused FLA/Triton KDA ON. `FLA_CACHE_MODE` and `FLA_CONFIG_DIR` were unset; the installed FLA code therefore uses its documented default-disabled config cache and contains Triton autotuning decorators for KDA kernels. No explicit atomic operation was found in the audited KDA forward source; workspace reuse is not controlled by the canonical harness.

One-factor findings:

1. Fixing every exposed random seed did not close the fresh-process variation.
2. Enabling `torch.use_deterministic_algorithms(True)` alone stopped at the first CuBLAS linear/GEMM operation because `CUBLAS_WORKSPACE_CONFIG` was not preconfigured. The operation and error were recorded; the environment was not modified to force the experiment through.

The evidence bounds the class to process initialization and localizes the first observed numerical difference to layer-0 KDA prefill state. It does not prove whether Triton autotune choice, workspace initialization/reuse, or another process-level initialization detail is the single low-level cause.

## Reference path

The installed FLA package includes `naive_recurrent_kda`. On the frozen document, exact captured driver tensors from the first KDA layer were replayed three times through both the canonical fast `chunk_kda` path and the naive FP32 recurrence for Hadamard and Dense-State.

- Fast operator: 3/3 output and state hashes identical for both conditions
- Naive reference: 3/3 output and state hashes identical for both conditions
- `REFERENCE_REPEATABILITY = PASS`
- `FAST_OPERATOR_REPEATABILITY = PASS`
- `FAST_PATH_ONLY_NONDETERMINISM = INCONCLUSIVE`

Fast-versus-naive relative L2 differences were approximately 0.00232–0.00238 for output and 0.00190–0.00194 for final state. This is a repeatability reference, not a parity claim. Since both operators are repeatable for fixed captured inputs while the full fresh process is not, the audit does not support blaming repeated invocation of the isolated fast operator alone.

## Noise versus method effect

- `AUC_NOISE_P95 = 0.02312851`
- `AUC_NOISE_MAX = 0.04001055`
- Median absolute same-process Hadamard-minus-Dense-State effect: 0.00902718
- Median absolute same-process Hadamard-minus-Dense-Functional effect: 0.01586185
- `METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO` (Dense-State): 0.2256
- `METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO` (Dense-Functional): 0.3964

Both ratios are below 1. Runtime variation can dominate the measured method effect, so the persistent comparison is not scientifically resolved.

## GPU comparison

GPU1 supplied the same-process and fresh-process observations above. GPU0 remained occupied by an unrelated external job throughout the allowed window. In accordance with the protocol, it was not preempted or shared.

- GPU0 repeatability: `NOT_RUN_EXTERNAL_GPU_JOB_ACTIVE`
- GPU1 repeatability: `same-process PASS; fresh-process FAIL`
- cross-GPU variation: `NOT_RUN_GPU0_BUSY`

## Safety and storage

No AIME generation/output, model weight, checkpoint, dependency, environment, or existing job was modified. Durable results are about 1.1 MB. No tensor trace was retained; the only temporary tensors were deleted immediately after the exact diff.
