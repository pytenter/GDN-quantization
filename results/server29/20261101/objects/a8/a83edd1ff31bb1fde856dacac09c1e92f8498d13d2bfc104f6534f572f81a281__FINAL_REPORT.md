# QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1 — Final Report

## Outcome

`STAGE_B_RESULT = STOPPED_FULL_RUN_DISK_GATE_FAIL`

The strict engineering smoke passed every semantic and runtime gate.  The full-60 run was **not authorized** because the required no-downsampling per-group Parquet dataset is infeasible on the frozen `/data` budget.  No scientific mechanism ordering or Stage-B decision label is reported from smoke data.

## Hard gates

```text
V2_SOURCE_GATE = PASS
FORMAL_ARTIFACT_IDENTITY_GATE = PASS

COMMON_TOKEN_SEQUENCE_GATE = PASS
COMMON_FP_STATE_INPUT_GATE = PASS

STATE_RESET_TO_COMMON_FP_EACH_TOKEN = TRUE
PERSISTENT_QDQ_WRITEBACK = FALSE
NEXT_TOKEN_READS_FP_STATE = TRUE

FORMAL_ROTATION_PATH_GATE = PASS
FORMAL_QUANTIZER_SHARED_PATH_GATE = PASS

STAGE_B_SMOKE = PASS
MEMORY_GATE = PASS
FULL_RUN_DISK_GATE = FAIL

EXPECTED_SAMPLES = 60
ACTUAL_SUCCESS = 0
FULL60_COMPLETENESS_GATE = NOT_RUN_DISK_GATE_FAIL

STRICT_COMMON_FP_STATE_GATE = PASS_SMOKE_ONLY
STAGE_B_RESULT = STOPPED_FULL_RUN_DISK_GATE_FAIL

RECURRENT_REPLAY_STARTED = FALSE
MULTI_HORIZON_STARTED = FALSE
PRE_DIVERGENCE_STARTED = FALSE
LAYER_INTERVENTION_STARTED = FALSE
C7_STARTED = FALSE
NEW_LOSS_STARTED = FALSE
NEW_ROTATION_TRAINING_STARTED = FALSE
NEW_AIME_GENERATION_STARTED = FALSE
```

## Engineering smoke

- Frozen units: `aime26_01_seed1`, `aime26_01_seed2`.
- Frozen prefix: 32 teacher-forced tokens per unit; smoke only, never used for a scientific conclusion.
- Peak allocated GPU memory: 17351.27 MiB.
- Each condition executed the unchanged formal quantizer 64 times over all 24 GDN layers.
- Selected token inputs were checked on every GDN layer by actual `torch.equal`; the common tensor bytes were SHA256 hashed and the same digest recorded for Native/H/C5/C6.
- H/C5/C6 local branches were discarded after measurement. Only the Native FP state was returned to the next token.

## Decisive disk projection

- Real calibration: all 393,216 group rows for one smoke token.
- Calibration file: 17,438,417 bytes (44.3482 bytes/row under Zstandard Parquet).
- Frozen full trajectory: 1,854,451 tokens.
- Required full rows: 729,199,804,416.
- Projected artifact: 30,117.75 GiB.
- Free space before smoke: 31.06 GiB.
- Projected free space after full run: -32,305.34 GB.

The gap is about three orders of magnitude beyond available storage.  Downsampling, dropping group rows, deleting old research artifacts, or silently replacing exact rows with sketches was forbidden, so the run stopped before any full-60 worker launched.

## Offline rotation geometry

The requested CPU/offline H/C5/C6 matrix audit was completed for all 24 layers, including orthogonality, coherence, entry statistics, distance to H, and exact Hungarian signed-permutation alignment.  These geometry-only results are preserved in `analysis/rotation_geometry.json` and `analysis/rotation_geometry_per_layer.csv`; they are not coupled to incomplete smoke data to make a mechanism claim.

## Scientific questions and figures

Q1–Q12 and Figures 1–12 are `NOT_EVALUATED_FULL60_DISK_GATE_FAIL`.  Reporting their ordering from the engineering smoke would violate the frozen protocol.  No Stage-B mechanism label is assigned.

`NEXT_RECOMMENDED_STAGE = REASSESS_LOCAL_METRICS`

No subsequent stage was started.
