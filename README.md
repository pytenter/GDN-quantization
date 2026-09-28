# GDN/KDA recurrent-state quantization

This repository studies INT8 recurrent-state quantization in recurrent and linear-attention language models:

- Qwen3.5-9B / Gated DeltaNet (GDN), where C128 is the preferred tested grouping and the fixed rotation is Key-side Hadamard.
- Ling-3.0-tiny / KDA, where R128 is the preferred tested grouping and the fixed rotation is Value-side Hadamard.

Model weights are not quantized by these experiments. The repository contains executable research code, compact evidence, frozen protocols, and provenance—not checkpoints, datasets, caches, logs, or raw generation traces.

## Historical AIME26 results at 81,920 tokens

`Incorrect` includes `Abstain`; therefore `Correct + Incorrect = 60` for every row.

| Model | Condition | Correct | Incorrect | of which Abstain | Accuracy |
|---|---|---:|---:|---:|---:|
| Qwen | FP_STATE | 52 | 8 | 3 | 86.67% |
| Qwen | INT8_C128 | 18 | 42 | 39 | 30.00% |
| Qwen | INT8_C128 + Key-Hadamard | 41 | 19 | 6 | 68.33% |
| Ling | FP_STATE | 44 | 16 | 11 | 73.33% |
| Ling | INT8_R128 | 23 | 37 | 28 | 38.33% |
| Ling | INT8_R128 + Value-Hadamard | 28 | 32 | 28 | 46.67% |

For Ling/KDA, these 81,920-token results are a frozen historical baseline only. They are retained for historical comparison and regression analysis and must not be merged with current 256K canonical outputs. The completed seed-1 256K offline study (FP_STATE 21/30, INT8_R128 9/30, INT8_R128 + Value-Hadamard 17/30) is also retained as precursor evidence; it is not itself the current 60-sample confirmatory result.

## Ling/KDA Current Canonical Evaluation Protocol

Ling-3.0-tiny / KDA currently uses the 256K-token long-horizon protocol as the canonical evaluation setting.

The canonical context budget is:

```text
context_length = 262144
max_total_tokens = 262144
```

`max_new_tokens` is dynamically computed:

```text
262144 - prompt_tokens - safety_margin
```

with:

```text
safety_margin = 512
```

The previous 81,920-token Ling AIME26 evaluation is historical reference only. It must not be mixed with the current 256K canonical evaluation. The 81,920-token results are retained for historical comparison and regression analysis.

The complete frozen Ling/KDA runtime and decoding definition is in [`docs/LING_CANONICAL_256K_PROTOCOL.md`](docs/LING_CANONICAL_256K_PROTOCOL.md).

## Protocol Compatibility Warning

Do not compare or merge outputs generated under:

- 81,920-token fixed budget
- 256K dynamic-budget protocol

unless explicitly marked as historical comparison.

Canonical Ling experiments after this update should use the 256K protocol.

## Rotation status

- Qwen Key-Hadamard: fixed canonical implementation and formal evidence archived.
- Ling Value-Hadamard: corrected prefill-endpoint V2 semantics; the KDA kernel-returned recurrent state is already in the rotated Value basis and is written directly to cache without an extra endpoint rotation.
- Recurrent-aware learnable rotation: implementation and validation are in progress. Qwen C5/C6 semantic gates pass, but resource and numerical-equivalence closure still block formal evaluation; Ling L6/L7 training is complete, L6 formal evaluation is running/partial, and L7 evaluation is pending.
- ButterflyQuant and HARP: **NOT EXECUTED**.
- Qwen `gdn_rotation_headroom_v1` and `gdn_postconv_structured_rotation_v1`: retained only as failed/closed prototypes.

Earlier Dense-State and Dense-Functional runs used `FP_STATE_RESET_EACH_TOKEN`: each training example saw an FP teacher state, a one-step INT8 perturbation, and then a reset. Formal inference instead writes the quantized student state back and accumulates error across tokens. The earlier E2E shortfall is therefore not a clean test of recurrent-aware learnable rotation.

The current phase is **recurrent-aware learnable rotation design and validation**. The main scientific question is whether a learnable orthogonal state rotation, trained through real multi-step INT8 writeback, can outperform fixed Hadamard on long-horizon reasoning.

## Repository map

```text
experiments/   Canonical AIME26, rotation, and mechanism code
results/       Compact immutable evidence, frozen reasoning results, and finalized offline analyses
reports/       Scientific reports and consolidation audits
tests/         Scorer and rotation regressions
docs/          Protocols, experiment index, provenance, and server inventories
```

Start with `docs/HANDOFF_2026-09-26.md`, `docs/RESEARCH_STATUS_2026-09-26.md`, `docs/EXPERIMENT_INDEX.md`, `docs/HANDOFF_ARTIFACT_MATRIX.md`, and `docs/SERVER_PROVENANCE.md`. Frozen protocol details remain in `docs/PROTOCOL.md` and `docs/SCORER_PROTOCOL.md`.
