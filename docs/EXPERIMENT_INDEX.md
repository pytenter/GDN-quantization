# Experiment index

Classification is semantic, not chronological:

- `CANONICAL_CURRENT`: frozen baseline/protocol or evidence needed for the current recurrent-aware program.
- `HISTORICAL_MECHANISM`: useful mechanistic or runtime history that does not define the present result.
- `NEGATIVE_OR_SUPERSEDED`: retained negative evidence, invalidated semantics, or an older implementation replaced by a better-defined one.

## Canonical current

| Experiment / area | Model | Status | Primary archived path | Scientific role |
|---|---|---|---|---|
| AIME26 81920 Frozen V4 | Qwen | COMPLETE/FROZEN | `results/aime26/81920/qwen_gdn/` | FP 52/60; INT8-C128 18/60; Key-Hadamard 41/60 |
| AIME26 81920 Frozen V4 | Ling | COMPLETE/FROZEN | `results/aime26/81920/ling_kda/` | FP 44/60; INT8-R128 23/60; Value-Hadamard 28/60 |
| Strict V4 scorer | Shared | COMPLETE/FROZEN | `experiments/shared/scoring/aime26_scorer_v4.py` | Canonical scoring; hash in scorer protocol |
| Key-Hadamard | Qwen | COMPLETE/FROZEN | `experiments/qwen_gdn/rotation/key_hadamard/` | Strong fixed C128 Key-side baseline |
| Corrected Value-Hadamard V2 | Ling | COMPLETE/FROZEN | `experiments/ling_kda/rotation/value_hadamard/` | Strong fixed R128 Value-side baseline; no redundant endpoint rotation |
| QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1 | Qwen | COMPLETE | `experiments/qwen_gdn/replay/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1/` | Within-rotation M5 diagnostic evidence |
| QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1 | Qwen | COMPLETE | `experiments/qwen_gdn/dense/QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1/` | Establishes old Dense training/writeback mismatch |
| QWEN_RECURRENT_DENSE_C5_C6_V1 | Qwen | PARTIAL/GATED | `experiments/qwen_gdn/recurrent_training/QWEN_RECURRENT_DENSE_C5_C6_V1/` | Current recurrent-aware implementation; formal AIME not started |
| QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1 | Qwen | COMPLETE NEGATIVE RESOURCE GATE | `experiments/qwen_gdn/runtime/QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1/` | `NO_LEAK_H32_TOO_LARGE` |
| QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2 | Qwen | PARTIAL/GATE FAIL | `experiments/qwen_gdn/recurrent_training/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2/` | Forward topology passes; frozen backward/topology envelope fails |
| Qwen trajectory-equivalence closure | Qwen | RUNNING | `experiments/qwen_gdn/recurrent_training/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2/backward_repeatability_closure/trajectory_equivalence/` | Eight completed single-run analyses archived; active run excluded |
| LING_DETERMINISTIC_16DOC_REPLICATION_V2 | Ling | COMPLETE/PASS | `experiments/ling_kda/runtime/LING_DETERMINISTIC_16DOC_REPLICATION_V2/` | Current deterministic replay evidence; executed on third server |
| LING_DENSE_TRAINING_SGLANG_PROVENANCE_AUDIT_V1 | Ling | COMPLETE | `experiments/ling_kda/dense/LING_DENSE_TRAINING_SGLANG_PROVENANCE_AUDIT_V1/` | Establishes L4/L5 single-step semantics |
| LING_RECURRENT_DENSE_L6_L7_V1 | Ling | RUNNING/PARTIAL | `experiments/ling_kda/recurrent_training/LING_RECURRENT_DENSE_L6_L7_V1/` | Training/final-R pass; L6 eval running, L7 eval pending |
| LING_RECURRENT_MEMORY_LIFETIME_AUDIT_V2 | Ling | PARTIAL | `experiments/ling_kda/runtime/LING_RECURRENT_MEMORY_LIFETIME_AUDIT_V2/` | Graph retention/detach pass; lifetime and stability probes unresolved |
| LING_256K_LONG_HORIZON_V1 | Ling | COMPLETE/OFFLINE | `experiments/ling_kda/long_horizon/LING_256K_LONG_HORIZON_V1/` | 256K sensitivity: FP 21/30, Native 9/30, Hadamard 17/30 |

## Negative or superseded

| Experiment / area | Classification | Evidence / reason |
|---|---|---|
| QWEN_ROTATION_SURROGATE_VALIDATION_V1 | NEGATIVE_OR_SUPERSEDED / CURRENT NEGATIVE EVIDENCE | Raw within-rotation M5 does not rank rotations |
| QWEN_ROTATION_AWARE_M5_VALIDATION_V1 | NEGATIVE_OR_SUPERSEDED / CURRENT NEGATIVE EVIDENCE | `ROTATION_AWARE_M5_GATE = FAIL`; no validated variant |
| QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1 | NEGATIVE_OR_SUPERSEDED | Real E2E result for old C3/C4 single-step checkpoints |
| Qwen C3/C4 and Ling L4/L5 Dense | SUPERSEDED_TRAINING_SEMANTICS | `FP_STATE_RESET_EACH_TOKEN`; no true recurrent INT8 history |
| Ling redundant prefill-endpoint rotation | INVALID_OLD_SEMANTICS | Kernel-returned state already occupies rotated Value basis |
| Ling runtime repeatability V1 | HISTORICAL_DIAGNOSTIC | Same-process pass/fresh-process fail, resolved by later closure |
| Ling fresh-process first-divergence V2 | HISTORICAL_DIAGNOSTIC | Localized Triton configuration variance; precursor to deterministic V2 |
| Qwen gdn_rotation_headroom_v1 | NEGATIVE_OR_SUPERSEDED | Failed/closed structured prototype |
| Qwen gdn_postconv_structured_rotation_v1 | NEGATIVE_OR_SUPERSEDED | Failed/closed structured prototype |
| Strict scorers V1/V2/V3 | SUPERSEDED | Replaced by frozen Strict V4 |
| 65,536-token runtime smoke | LEGACY | Replaced by frozen 81,920-token protocol |

## Historical mechanism archive

| Area | Model | Primary paths | Classification |
|---|---|---|---|
| Orientation and axis geometry | Qwen/GDN | `experiments/orientation/`, `results/orientation/`, `reports/orientation/` | HISTORICAL_MECHANISM |
| Residual/RMSNorm/out-projection/readout | Qwen/GDN | `experiments/propagation/`, `results/propagation/`, `reports/propagation/` | HISTORICAL_MECHANISM |
| Causal interventions and state-space risk | Qwen/GDN | `experiments/causal_intervention/`, `results/causal_intervention/`, `reports/causal_intervention/` | HISTORICAL_MECHANISM |
| Effective-update metrics | Qwen/GDN | `experiments/effective_update/`, `results/effective_update/`, `reports/effective_update/` | HISTORICAL_MECHANISM |
| Architecture/state semantics | Ling/KDA | `experiments/ling/`, `results/ling/`, `reports/ling/` | HISTORICAL_MECHANISM |
| KDA commutator/decay/error decomposition | Ling/KDA | `experiments/ling_kda/mechanism/`, `results/mechanism/ling_kda/` | HISTORICAL_MECHANISM |

For source-server paths, inclusion flags, raw exclusions, and provenance, see `HANDOFF_ARTIFACT_MATRIX.md` and `artifact_registry/`.
