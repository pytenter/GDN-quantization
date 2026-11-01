# QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2

```text
EXPERIMENT = QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2
PARENT_V1 = QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1
V1_IN_PROCESS_FULL_UNLOAD_GATE = FAIL
V1_FINAL_ALLOCATED_BYTES = 8519680
V1_ROOT_CAUSE = UNRESOLVED
V1_ROOT_CAUSE_DETAIL = UNRESOLVED_AFTER_OBSERVER_HYPOTHESIS_REJECTED
V2_PROTOCOL_FREEZE_GATE = PASS
PRODUCTION_RUNNER_SCIENTIFIC_PATH_IDENTITY_GATE = PASS
SOURCE_ARTIFACT_REVALIDATION_GATE = PASS
C6R_ROTATION_SHA = a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6
C7_ROTATION_SHA = 57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8
GPU0_C6R_RESIDENT_MEMORY_GATE = PASS
GPU0_C6R_WORKER_READY = YES
GPU0_C7_RESIDENT_MEMORY_GATE = PASS
GPU0_C7_WORKER_READY = YES
GPU1_C6R_RESIDENT_MEMORY_GATE = PASS
GPU1_C6R_WORKER_READY = YES
GPU1_C7_RESIDENT_MEMORY_GATE = PASS
GPU1_C7_WORKER_READY = YES
GPU2_C6R_RESIDENT_MEMORY_GATE = PASS
GPU2_C6R_WORKER_READY = YES
GPU2_C7_RESIDENT_MEMORY_GATE = PASS
GPU2_C7_WORKER_READY = YES
GPU3_C6R_RESIDENT_MEMORY_GATE = PASS
GPU3_C6R_WORKER_READY = YES
GPU3_C7_RESIDENT_MEMORY_GATE = PASS
GPU3_C7_WORKER_READY = YES
C6R_FRESH_PROCESS_DETERMINISM_GATE = PASS
C7_FRESH_PROCESS_DETERMINISM_GATE = PASS
PROCESS_EXIT_RECLAIM_GATE = PASS
FOUR_GPU_PERSISTENT_WORKER_GATE = PASS
HISTORICAL_81920_RUNTIME_COMPATIBILITY_GATE = PASS
ROTATION_DOES_NOT_CHANGE_CACHE_GEOMETRY_GATE = PASS
CANONICAL60_WORKER_ASSIGNMENT = FROZEN
CANONICAL60_CONDITION_ORDER = PREEXISTING_FROZEN_ORDER_C6R_THEN_C7
V2_PERSISTENT_WORKER_READINESS = PASS
C6R_3090_DEPLOYMENT_READY = YES
C7_3090_DEPLOYMENT_READY = YES
READY_FOR_C6R_C7_CANONICAL60 = YES
CROSS_GPU_NUMERICAL_EQUALITY_GATE = NOT_REQUIRED
A800_3090_NUMERICAL_PARITY_POLICY = NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE
NEW_TRAINING_STARTED = NO
TRAINING_RERUN = NO
C6R_AIME_STARTED = NO
C7_AIME_STARTED = NO
NEW_AIME_GENERATION = 0
SCORING_STARTED = NO
PUSH_PERFORMED = NO
```

V2 DOES NOT OVERRIDE V1. V1 and V2 test different lifecycle requirements. V1 residual owner remains unresolved; no claim of a cuBLAS or other root cause.

## Measured resident endpoints

| Worker | Cycle | Allocated bytes | Reserved bytes | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| gpu0_c6r | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c6r | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c6r | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7 | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7 | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7 | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c6r_repeat_b | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c6r_repeat_b | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c6r_repeat_b | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7_repeat_b | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7_repeat_b | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu0_c7_repeat_b | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c6r | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c6r | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c6r | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c7 | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c7 | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu1_c7 | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c6r | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c6r | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c6r | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c7 | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c7 | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu2_c7 | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c6r | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c6r | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c6r | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c7 | 1 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c7 | 2 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |
| gpu3_c7 | 3 | 17917727744 | 20868759552 | 20390115328 | 20868759552 |

## Interpretation and scope

Production requests directly call the frozen manual-HF generate_one function, with a prospective non-AIME TRAIN:first-row prompt and max_new_tokens=128. One model/one frozen condition per process; one warmup and three identical measured requests. Semantic smoke repeats in each cycle using the same frozen32 input. Public output_hidden_states capture is semantic-only, with no extra diagnostic hooks. Original required scientific rotation patch remains unchanged. No extra gc/empty_cache, no full model deletion test.

Generated IDs and compact semantic fingerprints are exact within each worker; GPU0 A/B comparison requires exact equality. Cross-GPU hashes are descriptive only, no invented tolerance. The original source/weights/tokenizer/runtime and all protected V1 evidence were rehashed.

128-token smoke tests request lifetime, NOT 81920-token memory feasibility. Historical long-generation capability is inherited only via exact source/runtime/cache compatibility; bank changes do not alter cache geometry. Rotation tensor payload is 1,572,864 bytes per condition; actual allocated increment is recorded separately.

All workers were sequential fresh subprocesses to enforce first-failure stop without interrupting any currently running work. Every measured process has its exact NVML byte baseline, PID/descendant audit, exit code and <=10-second reclaim history in analysis/*_worker.json. Incomplete rows are NOT_RUN, not evidence of success. No semantic/model/answer retries.

## Frozen protocol and reproducibility

Premeasurement commit: 2bd52228c016dec64997354b56dd0694d2c5da9a

```sh
cd /data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1
/data/ydai/miniconda3/envs/bitdecode/bin/python /data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/experiments/QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2/scripts/persistent_v2_controller.py
# Single-use controller refuses existing evidence; do not rerun.
```

Future Canonical60 plan is same problem/seed→same physical GPU, rank%4, earlier frozen C6R_THEN_C7 order; 60/arm, C5R=0. Readiness activation is recorded independently from immutable premeasurement plans. No AIME/scoring run was started.

STOP: await explicit user authorization for QWEN_GDN_C6R_C7_CANONICAL60_V1.
