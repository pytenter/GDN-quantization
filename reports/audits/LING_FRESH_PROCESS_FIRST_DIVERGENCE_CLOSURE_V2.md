# LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2

## Scope and safety

- Model/server: Ling-3.0-tiny / KDA on the designated 2x RTX 4090 host; diagnostics used GPU1 only.
- Qwen/MATH-500: not inspected, modified, restarted, or stopped after the user override.
- Existing Ling/256K/AIME outputs: not modified.
- Model weights and Python/CUDA/FLA packages: not modified.
- Numerical changes: none in the default path. All closure settings are process-scoped diagnostics enabled by `FLA_CONFIG_DIR` and `LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`.
- Large persistent tensor traces: none retained. Small frozen operator inputs were used for isolation and removed after summaries were produced.

## Fixed diagnostic sample and provenance

- Diagnostic sample: `wikitext2raw-test-000`, condition `Hadamard`.
- Raw-text SHA256: `28ffb08172879785121bd1ed6338f65e6dee58c4165140f078e368ce3bb5abb95`.
- Token-ID SHA256: `87bf6cb42f3f86f78c6e81b653848fb8e2292ef8ba905742def5bd2bb101344ff5`.
- Model-config-file SHA256: `9750d847957913f665a13c0b5a6537199e33c6f3ec970d9fcb55a00e5076d4012`.
- Tokenizer-config-file SHA256: `2456b0372956cd3e82f17e33372148b115a94970bfd4878ba55e7e60cd3204f74`.
- Dense-State checkpoint SHA256: `dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318`.
- Dense-Functional checkpoint SHA256: `65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c`.
- `MODEL_LOAD_PROVENANCE_MATCH = PASS` across five fresh processes. Layer-0 and all key projection/norm/conv parameter hashes matched.

## First-divergence localization

The ordered layer-0 capture ladder was:

`tokens -> embedding -> layer input -> pre-norm -> q/k/v/beta/decay projections -> q/k/v short-conv -> normalized KDA drivers -> chunk_kda -> prefill recurrent cache`.

The first baseline-visible divergence was the `chunk_kda` final state, but frozen-driver decomposition moved the causal boundary upstream. The earliest confirmed nondeterministic operator was:

`FIRST_NONDETERMINISTIC_OPERATOR = fla.modules.l2norm_fwd`

Its fixed-input baseline selected different Triton configurations across processes and produced two q-normalization hashes (4:1):

- dtype: BF16 output (FP32 reciprocal-norm/accumulation path)
- shape: `[1, 128, 16, 128]`
- max absolute difference: `4.8828125e-4`
- relative L2: `1.618452628958716e-5`
- different elements: `3`
- same-process repeats: stable
- diagnostic fix: `BT=8, warps=2, stages=3`
- fixed result: 5/5 fresh processes bitwise equal

This exposed additional downstream configuration-sensitive sites that had been masked by the first divergence:

1. `chunk_local_cumsum_vector_kernel`: diagnostic fixed configuration `BS=32, warps=4, stages=3`; layer-0 recurrent state/cache then became 5/5 exact.
2. `FusedRMSNormGated`: the raw Triton autotuner selected a `BT=16, warps=16` variant that changed one BF16 element (`max_abs=1.220703125e-4`, `relL2=3.1516298087705284e-5`). The FLA JSON cache does not control this raw autotuner, so the process-scoped gate `LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8` narrows it to the verified configuration before first call. Fixed result: 5 fresh processes x 5 same-process calls exact.
3. Decode `causal_conv1d_update_kernel`: at step 2, layer 9 q short-conv was the first divergent reference event. Layer-9 input, pre-norm, q projection, and the incoming convolution cache were byte-identical. The anomalous `BD=64, warps=4` configuration changed exactly one BF16 element (`max_abs=1.220703125e-4`, `relL2=9.834845695877448e-5`). Fixed configuration `BD=8, warps=8, stages=3` produced 5 fresh processes x 5 same-process calls exactly equal, including cache writeback.

`PROCESS_INITIALIZATION_KERNEL_STATE_SIGNAL = PRESENT`: each process is stable after its first autotune choice, while fresh-process first calls can select numerically different configurations.

## End-to-end validation

- Fixed reference decode ladder: 5 fresh processes x 8 decode steps.
- `REFERENCE_DECODE_REPEATABLE = true`.
- `FIRST_DIVERGENT_REFERENCE_DECODE_EVENT = null`.
- Full closure matrix: 3 fixed documents x {Hadamard, Dense-State} x 5 fresh processes = 30 runs.

Final matrix values:

- `AUC_NOISE_P95_BEFORE = 0.023128505481597707`
- `AUC_NOISE_P95_AFTER = 0.0`
- `AUC_NOISE_MAX_BEFORE = 0.0400105529926776`
- `AUC_NOISE_MAX_AFTER = 0.0`
- `METHOD_EFFECT = 0.009027184397862181`
- `METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO = 9027184397.86218` (the analyzer's `1e-12` denominator floor; mathematically unbounded because measured noise is zero)
- `all_key_checkpoints_bitwise = true`
- `LING_REPEATABILITY_VERDICT = REPEATABILITY_CLOSED`
- `16_doc_authorized = YES`
- `64_doc_authorized = NO`

## Classification

`ROOT_CAUSE_CLASSIFICATION = PROCESS_LOCAL_TRITON_AUTOTUNE_CONFIG_NUMERICAL_VARIANCE`

The issue is not model-load drift, seed drift, state quantization, the corrected KDA value-basis semantics, or a redundant prefill-endpoint rotation. It is a chain of process-local Triton autotune choices whose small BF16 differences can enter recurrent decoding and amplify downstream.

The fixed-config path is diagnostic evidence only; it is not automatically promoted to the canonical Ling runtime.

## Required status

- `KDA_ROTATION_SEMANTICS_VERSION = CORRECTED_PREFILL_ENDPOINT_V2`
- `REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO`
- `LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE = PASS`
- `LING_PERSISTENT_EVALUATION_READY = YES`, under the audited diagnostic fixed-config path; canonical promotion remains an explicit follow-up decision.

## Recommended next step

Run the authorized 16-document Ling replication with the audited fixed-config path, after explicit approval. Do not train a new rotation objective from this audit.
