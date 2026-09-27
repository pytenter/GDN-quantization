# LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1 preregistration

Status: frozen before any new generation.

## Scope

This confirmatory evaluation uses Ling-3.0-tiny/KDA only. It reuses the frozen 20 samples `aime26_11_seed1` through `aime26_30_seed1` and prospectively generates the 40 missing members of the AIME26 30-problem × seeds `[1,2]` sample set. The 20 frozen samples must not be regenerated.

The historical 60-sample protocol used `max_new_tokens=81920`. It supplies only the canonical sample-ID and seed definition here. Its outputs are not mixed into this experiment. To preserve exact compatibility with the mandatory reused 20-case L6/L7 closure, every new sample uses the frozen 262,144-context budget:

`max_new_tokens = 262144 - prompt_tokens - 512`.

## Frozen hypotheses and endpoints

The primary endpoint is Frozen V4 correctness. The primary paired comparison is L7 versus static Hadamard. Secondary paired comparisons are L6 versus Hadamard and L7 versus L6. FP is reference-only; no new FP generation is authorized.

All endpoint tests are paired, two-sided exact McNemar/exact paired-binomial tests on discordant samples. Holm correction is frozen across the three endpoint comparisons. No accuracy threshold is preregistered.

Trajectory metrics are secondary or exploratory only: generated tokens, length ceiling, termination type, explicit stop, abstention, first committed claim, stable final commitment, answer changes, retractions, and repeated 4-gram fraction. Global shortening or repetition reduction is not a preregistered success criterion.

## Frozen execution

New H, L6, and L7 outputs use identical prompts, sample seeds, per-sample token budgets, sampling parameters, INT8-R128 quantization, deterministic SGLang runtime, and the archived H/L6/L7 rotation identities. H, L6, and L7 for any one missing sample use the same hardware class and physical GPU.

The prospective assignment is deterministic round-robin over the ordered missing sample list: 4090 GPU0, 4090 GPU1, 3090 GPU3, 3090 GPU4. Each shard receives 10 samples. Conditions run serially in the frozen order H, L6, L7; the four shards may run concurrently within a condition.

Only 3090 GPU3 and GPU4 are authorized. Qwen/GDN GPUs and the 48GB vGPU are forbidden.

## Frozen retry and stop rules

Scientific failures such as a wrong answer, abstention, long loop, or length ceiling are never retried. A deterministic retry is allowed only for a proven infrastructure failure that produced no complete canonical output, with unchanged sample, seed, hardware assignment, runtime, and configuration; every retry must be audited.

Any manifest ambiguity, scorer/artifact hash mismatch, runtime or quantizer drift, post-freeze assignment change, unsafe disk condition, source corruption, or FP mixed-hardware pooling stops the experiment.

Machine-readable details are in `preregistration.json`, `configs/canonical_sample_manifest.json`, `configs/hardware_assignment.json`, `configs/runtime_identity.json`, `configs/scorer_identity.json`, and `provenance/stage0_audit.json`.
