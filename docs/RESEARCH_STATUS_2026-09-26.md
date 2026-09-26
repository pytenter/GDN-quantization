# Research status at the 2026-09-26 handoff

## One-sentence status

Fixed Hadamard substantially recovers recurrent-state INT8 reasoning, while the old Dense rotations were trained with single-step FP-state resets; the active phase is to establish reliable recurrent-aware multi-step training and test whether a learnable orthogonal rotation can outperform Hadamard.

## Canonical configuration and result

| Model | State quantization | Fixed rotation | FP | Native INT8 | Fixed rotation |
|---|---|---|---:|---:|---:|
| Qwen3.5-9B / GDN | INT8-C128 | Key-side Hadamard | 52/60 | 18/60 | 41/60 |
| Ling-3.0-tiny / KDA | INT8-R128 | Value-side Hadamard | 44/60 | 23/60 | 28/60 |

These are the frozen AIME26 Strict V4 results under `GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2`: all 30 problems are evaluated with seeds 1 and 2, giving 60 samples per condition, with `max_new_tokens = 81920`. The scorer SHA256 is `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`. Qwen's C128 grouping follows the Key-oriented GDN state geometry. Ling's R128 grouping and Value rotation follow KDA's Value basis; the corrected implementation writes the kernel-returned state directly because it is already in the rotated Value basis.

## What the metric work established

Teacher-forced replay covers 60 paired trajectories, 24 GDN layers, and 3,708,902 token-conditions. M5 p95 predicts failure *within a fixed rotation* (AUROC 0.851; bootstrap CI 0.7315--0.9462), but raw M5 does not rank different rotations: Hadamard can have worse raw M5 while yielding much better reasoning. Rotation-aware transforms and rescue tests did not close that mismatch (`ROTATION_AWARE_M5_GATE = FAIL`). Local replay metrics are diagnostics, not a validated cross-rotation optimization target.

## Why the old Dense result is superseded

Old C3/C4 and L4/L5 training used `FP_STATE_RESET_EACH_TOKEN`. Train-time QDQ is numerically aligned with formal QDQ, but recurrent writeback is not (`REAL_RECURRENT_TRAINING_GATE = FAIL`, `TRAIN_VS_FORMAL_WRITEBACK_MATCH = FAIL`). The resulting last-20 Qwen scores were FP 17/20, Native 4/20, Hadamard 13/20, Dense-State 12/20, and Dense-Functional 8/20. These remain valid measurements of those checkpoints, but they do not falsify recurrent-aware learnable rotation.

## Current Qwen state

- C5/C6 code passes overlap, step-0, exact-QDQ, recurrent-writeback, provenance, gradient, and BPTT-boundary gates.
- Single-GPU H32 is not sustainable. The memory audit found no persistent leak: the classification is `NO_LEAK_H32_TOO_LARGE`, meaning per-rollout peak/resource pressure.
- Dual-GPU forward/state/QDQ/loss/writeback/BPTT checks pass. Gradient and update hashes differ, but fresh single-GPU repeats are themselves not bitwise stable, and strict deterministic mode reaches an unsupported cumsum path. The frozen same-topology numerical envelope currently yields `DUAL_GPU_TRAINING_TOPOLOGY = FAIL`; this is not evidence of a forward semantic error.
- Trajectory-equivalence closure is **RUNNING**. Completed immutable analyses `single_01` through `single_08` are archived; the active ninth run was not copied. C5/C6 formal AIME evaluation has not started.

## Current Ling state

- Runtime reproducibility progressed from same-process pass/fresh-process failure to localization at process-local Triton autotune variance and then deterministic V2 replication pass. V2 was run on the third 8x3090 host, not the 2x4090 Ling host; fresh-process noise is zero in the archived evidence.
- Deterministic 16-doc Future-KL AUC mean/median: Native 0.031744/0.016550; Hadamard 0.018873/0.009969; Dense-State 0.028116/0.009978; Dense-Functional 0.016269/0.009924. It shows no credible persistent Dense headroom.
- L6/L7 recurrent training and final-R materialization passed. L6 formal evaluation is **RUNNING/PARTIAL** on the third server; seven sample outputs were present at the read-only snapshot and no completion marker existed. L7 evaluation is **PENDING**.
- The Ling recurrent-memory audit remains **PARTIAL**. Loss/history graph retention and state detach pass; H128 peaked at about 19.805 GiB. Cache lifetime, QDQ autograd lifetime, validation return-to-baseline, and sustained stability remain unresolved.

## 256K length sensitivity

The completed offline Ling 256K analysis reports FP 21/30, Native INT8_R128 9/30, and Value-Hadamard 17/30. Compact per-sample metadata, taxonomy, stability analysis, scoring, manifests, and reports are archived. Full responses remain on the source server and are excluded from Git. This result is a separate length-sensitivity study; the frozen canonical comparison remains 81,920 tokens.

## Current blockers and next decision

1. Let the running Qwen trajectory closure and Ling L6 evaluation finish naturally; verify completion markers before importing any new output.
2. Decide Qwen training topology using a preregistered numerical/functional equivalence envelope rather than impossible bitwise-backward equality, then establish a sustainable multi-step horizon.
3. Close the remaining Ling memory-lifetime probes without disturbing formal evaluation, then score L6 completely and evaluate L7 under the frozen deterministic runtime.
4. Only after those gates close, compare recurrent-aware learned rotations with fixed Hadamard on the frozen reasoning protocol. Do not optimize directly against unvalidated cross-rotation raw M5.

See `HANDOFF_2026-09-26.md` for the full narrative, `EXPERIMENT_INDEX.md` for status classification, and `HANDOFF_ARTIFACT_MATRIX.md` plus `artifact_registry/` for provenance and exclusions.
