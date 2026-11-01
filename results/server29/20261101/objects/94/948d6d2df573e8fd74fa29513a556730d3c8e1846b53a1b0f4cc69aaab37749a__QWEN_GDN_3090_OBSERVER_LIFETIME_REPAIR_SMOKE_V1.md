# QWEN_GDN_3090_OBSERVER_LIFETIME_REPAIR_SMOKE_V1

```text
EXPERIMENT = QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1
SUBPHASE = QWEN_GDN_3090_OBSERVER_LIFETIME_REPAIR_SMOKE_V1
PREVIOUS_FAILURE = GPU0_REPEAT1_FINAL_CLEANUP_MEMORY_RETENTION
REPAIR_SCOPE = OBSERVER_REFERENCE_LIFETIME_ONLY
GPU_MODEL_ATTEMPTS = 1
GPU_USED = PHYSICAL_RTX3090_GPU0
GPU_UUID = GPU-a78176ed-f97f-c54f-6201-5a01b570f422
OBSERVER_REPAIR_SCIENTIFIC_PATH_IDENTITY_GATE = PASS
STATIC_OBSERVER_LIFETIME_REPAIR_GATE = PASS
PRE_GPU_PROTECTED_IDENTITY_GATE = PASS
C6R_ROTATION_LOAD_GATE = PASS
C7_ROTATION_LOAD_GATE = PASS
C6R_ORTHOGONALITY_GATE = PASS
C7_ORTHOGONALITY_GATE = PASS
ROTATION_MAPPING_GATE = PASS
QDQ_SEMANTICS_GATE = PASS
REAL_RECURRENT_WRITEBACK_GATE = PASS
DEPLOYMENT_MEMORY_GATE = FAIL
OBSERVER_LIFETIME_ROOT_CAUSE = NOT_SUPPORTED
REPAIRED_GPU0_SMOKE_GATE = FAIL
C6R_3090_DETERMINISM_GATE = NOT_RUN
C7_3090_DETERMINISM_GATE = NOT_RUN
FOUR_GPU_CONSISTENCY_GATE = NOT_RUN
C6R_3090_DEPLOYMENT_READY = NO
C7_3090_DEPLOYMENT_READY = NO
READY_FOR_C6R_C7_CANONICAL60 = NO
NEW_TRAINING_STARTED = NO
TRAINING_RERUN = NO
C6R_AIME_STARTED = NO
C7_AIME_STARTED = NO
NEW_AIME_GENERATION = 0
SCORING_STARTED = NO
SMOKE_AUTOMATIC_RETRY = NO
PUSH_PERFORMED = NO
PROTECTED_ARTIFACT_REVALIDATION_GATE = PASS
FINAL_ACTION = STOP
```

## Repair and causal evidence

Only a new runner was added; the original failed runner and artifacts remain byte-identical. Lifecycle repair: `del original,observed` after each condition capture, followed by existing patch/bank deletion. Existing tensor deletion, gc.collect(), empty_cache(), threshold and numerical bodies are unchanged. Weakrefs and atomic diagnostic writes add no strong model/tensor references.

Static AST/body audit and exact allowlisted source diff are frozen before the sole GPU attempt. No old successful output fingerprint exists, so numerical equality with the unsaved old run cannot be measured; scientific-source identity, actual finite outputs and exact QDQ/writeback checks supply the valid evidence.

Root-cause verdict: NOT_SUPPORTED. Old cleanup FAIL; repaired cleanup FAIL. Weakrefs at final check: {'model_alive': False, 'observer_alive': False, 'rotation_bank_alive': False}.

## Memory timeline

| Point | Condition | Allocated bytes | Reserved bytes | Peak allocated | Peak reserved | Model alive | Observer alive | Bank alive |
|---|---|---:|---:|---:|---:|---|---|---|
| before_model |  | 0 | 0 | 0 | 0 |  |  |  |
| M0 | None | 0 | 0 | 0 | 0 | None | None | None |
| M1 | None | 17907635200 | 17926455296 | 17907635712 | 17926455296 | True | None | None |
| baseline_model_loaded |  | 17907635200 | 17926455296 | 17907635712 | 17926455296 |  |  |  |
| M2 | Hadamard | 17907635200 | 17926455296 | 17907635200 | 17926455296 | True | True | None |
| Hadamard_peak |  | 18019944960 | 18152947712 | 18120608256 | 18152947712 |  |  |  |
| M3 | Hadamard | 18019944960 | 18152947712 | 18120608256 | 18152947712 | True | True | None |
| M6 | Hadamard | 18019944960 | 18152947712 | 18120608256 | 18152947712 | True | False | None |
| M7 | Hadamard | 17916154880 | 18152947712 | 18120608256 | 18152947712 | True | False | None |
| Hadamard_post_cache_cleanup |  | 17916154880 | 17926455296 | 18120608256 | 18152947712 |  |  |  |
| M2 | C6R | 17917727744 | 17926455296 | 17917727744 | 17926455296 | True | True | True |
| C6R_peak |  | 18021517824 | 18152947712 | 18122181120 | 18152947712 |  |  |  |
| M4 | C6R | 18021517824 | 18152947712 | 18122181120 | 18152947712 | True | True | True |
| M6 | C6R | 18019944960 | 18152947712 | 18122181120 | 18152947712 | True | False | False |
| M7 | C6R | 17916154880 | 18152947712 | 18122181120 | 18152947712 | True | False | False |
| C6R_post_cache_cleanup |  | 17916154880 | 17926455296 | 18122181120 | 18152947712 |  |  |  |
| M2 | C7 | 17917727744 | 17926455296 | 17917727744 | 17926455296 | True | True | True |
| C7_peak |  | 18021517824 | 18152947712 | 18122181120 | 18152947712 |  |  |  |
| M5 | C7 | 18021517824 | 18152947712 | 18122181120 | 18152947712 | True | True | True |
| M6 | C7 | 18019944960 | 18152947712 | 18122181120 | 18152947712 | True | False | False |
| M7 | C7 | 17916154880 | 18152947712 | 18122181120 | 18152947712 | True | False | False |
| C7_post_cache_cleanup |  | 17916154880 | 17926455296 | 18122181120 | 18152947712 |  |  |  |
| M8 | None | 8519680 | 17926455296 | 18122181120 | 18152947712 | False | False | False |
| M9 | None | 8519680 | 17909678080 | 18122181120 | 18152947712 | False | False | False |
| post_model_cleanup |  | 8519680 | 17909678080 | 18122181120 | 18152947712 |  |  |  |
| M10 | None | 8519680 | 17909678080 | 18122181120 | 18152947712 | False | False | False |

Final allocated bytes: 8519680. Final reserved bytes: 17909678080. Threshold strictly remains <1,048,576 allocated bytes. This is not long-generation memory proof.

## Dynamic gates and fingerprint boundary

Hadamard/C6R/C7 each use the unchanged 32-token list, seed1, physical GPU0. QDQ calls/condition:31; layer coverage24; rotation calls768; writeback continuity744 exact transitions. These counts are validated by unchanged assertions before condition PASS is recorded, not extrapolated to other GPUs. Raw diagnostic details and all 32 logit/FinalHidden hashes, 31x24 scale/qcode/post-QDQ hashes, final states are in gpu0_repaired_smoke_output_fingerprints.json. No scientific ranking is performed.

## Provenance and final stop

Branch: exp/qwen-gdn-c6r-c7-4x3090-deployment-v1. Pre-task HEAD: ff6895f4129f1889ad96eedbeb771febd56dbdc1. Old failures and all 3136 target protected files unchanged; all51 A800 source files unchanged. New commit record will be logs/observer_repair_commit.json; no push.

Exactly one fresh GPU0 attempt. No GPU0 repeat2, no GPU1-3 work, no four-GPU consistency, future config freeze, AIME, training or scoring. Readiness remains NO even if this smoke passes. FINAL_ACTION=STOP.
