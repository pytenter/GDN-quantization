# PROVENANCE_EXECUTION_CAPSULE_V3 — Final qualification report

Experiment: `LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1`

Task: `ENGINEERING_ONLY_PROVENANCE_RECOVERY`

V3 qualification **did not pass**. The first sealed verify attempt stopped before model loading, CUDA measurement, or scientific computation. No retry or source correction was performed after STOP.

## Blocking evidence

The engineering adapter at `/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/provenance_capsule_v3/sealed_runner.py:174` contains an incorrectly transcribed expected hash. The original historical bridge, its source-server snapshot, and its target-server snapshot agree with the frozen source manifest. This is an adapter implementation error, not a demonstrated mutation of historical scientific source or a numerical/scientific failure.

```text
ACTUAL_AND_MANIFEST_SHA256 = e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd
ADAPTER_EXPECTED_LITERAL = e4baf3f7e3ecc19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd
EXPECTED_LITERAL_LENGTH = 65
FIRST_ATTEMPT = verify
FINAL_ACTION = FAIL_AND_STOP
```

Immutable first STOP: `/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/provenance_capsule_v3/STOP.json`

STOP SHA256: `d2330b426afbea3e443bc44fc0381b9d4e045fcfa3a18739bd6d5570a85e2fe8`

## Gates

| Gate | Evidence-backed state |
| --- | --- |
| V3_CLEAN_WORKTREE_GATE | PASS (both candidates clean before launch) |
| V3_SOURCE_SNAPSHOT_BYTE_IDENTITY_GATE | PASS (125 exact source copies; separate adapter expected-hash assertion FAILED) |
| V3_PROJECT_LOCAL_DEPENDENCY_CLOSURE_GATE | NOT_COMPLETED (qualification stopped during entrypoint binding) |
| V3_NO_DIRTY_WORKTREE_EXECUTION_GATE | NOT_QUALIFIED (no original dirty source executed before STOP) |
| V3_RUNTIME_ENVIRONMENT_IDENTITY_GATE | PREPARATION_PASS; SEALED_RUNTIME_NOT_QUALIFIED |
| V3_SCIENTIFIC_ASSET_IDENTITY_GATE | PREPARATION_PASS (406 assets); SEALED_REVALIDATION_NOT_RUN |
| V3_NO_GIT_DURING_MEASUREMENT_GATE | NOT_QUALIFIED (failed engineering attempt Git count=0; formal phase not started) |
| V3_HISTORICAL_FUNCTIONAL_PARITY_SUITE_GATE | NOT_RUN |
| V3_FRESH_PROCESS_DETERMINISM_GATE | NOT_RUN (0/3) |
| V3_HARDWARE_AND_STORAGE_GATE | PREPARATION_PASS; NUMERICAL_HARDWARE_PAIRING_NOT_QUALIFIED |

Preparation PASS is not an overall sealed-runtime qualification PASS. The source-snapshot identity gate passed; the independent adapter expected-hash assertion failed. All remaining numerical qualification and post-commit gates are unexecuted.

## Frozen candidate, not qualified

```text
UNQUALIFIED_SCAFFOLDING_COMMIT = a2b434c0efac572d7cd83f393c7c8b331012e151
UNQUALIFIED_INPUTS_COMMIT = 7df9bef0a713a73f723b496c5709a0ebd8fdfbc4
CAPSULE_ID = NOT_CREATED
V3_QUALIFICATION_COMMIT = NOT_CREATED
V3_PREMEASUREMENT_COMMIT = NOT_CREATED
POST_COMMIT_GATES_ALL_PASS = NO
GPU6_FORMAL_AUTHORIZED = NO
```

Execution roots:

- `/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3`
- `/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3` (exact sparse candidate checkout; not a full-history/full-repository checkout)

## Historical integrity and process state

```text
V1_STATUS = FAIL_PRESERVED
V2_STATUS = FAIL_PRESERVED
V2_RUNTIME_INGRESS_ARCHITECTURE = RETIRED_AS_OVERBROAD_FOR_FORMAL_EXECUTION
HISTORICAL_SOURCE_PRESERVATION = PASS
HISTORICAL_TARGET_PRESERVATION = PASS
SUPERVISOR_CLEANUP_LEDGER_REMAINING = 1
ACTUAL_TASK_PROCESS_COUNT = 0
GPU6_COMPUTE_PROCESS_COUNT = 0
GIT_SUBPROCESS_COUNT_DURING_FAILED_QUALIFICATION = 0
FP_REFERENCES_MIXED_ACROSS_HARDWARE = NO
```

Original V1 STOP and all three V2 A-attempt failure receipts are retained in place. 1476 source-side and 1475 target-side historical files were rehashed after STOP. Legacy HEAD/status and source snapshot bytes remain unchanged.

V2 retirement means runtime-wide regular-file ingress closure was too broad for practical scientific execution; it does not rewrite or invalidate V2 receipts. V3 installs no general ordinary-file/OS metadata classifier and makes no `NO_EXTERNAL_FILES_READ` claim. The claim that all scientific/executable dependencies are qualified cannot yet be made.

## Runtime and preparation evidence

52 historically frozen package versions matched exactly; Python 3.11.16, Torch 2.7.1+cu128, Transformers 4.57.6, Triton 3.3.1, FLA-core 0.5.2, NumPy 2.4.6; SciPy is not installed (not an expected historical package). 10,179 environment source/extension identities and 156 explicitly identified native library identities were recorded. Third-party source was not copied into the project snapshot.

GPU6: `GPU-de53f243-b271-521c-d875-d96611ed8574, NVIDIA GeForce RTX 3090, 24576 MiB, 18 MiB, 24107 MiB, 580.173.02`

Free storage after STOP: 850136092672 bytes. Historical storage estimate: 2 GiB non-trace overhead plus unchanged 20 GiB reserve; all four frozen same-GPU6 FP drives are reused, not regenerated.

Decisive receipts:

- `candidate_launch_identity.json`
- `historical_source_snapshot_manifest.json`
- `runtime_environment_manifest.json`
- `scientific_asset_manifest.json`
- `shared_library_manifest.json`
- `storage_capacity_identity.json`
- `runs/qualification/source_post_stop_preservation.json`
- `runs/qualification/STOP_AUDIT.json`
- `runs/qualification/GATE_STATUS_AT_STOP.json`

## Scientific state and stop boundary

```text
FORMAL_MEASUREMENTS = 0
DISCOVERY_PRIMARY_COMPLETE = 0/64
DISCOVERY_SECONDARY_COMPLETE = 0/64
FRESH_PROCESS_SMOKE_COMPLETE = 0/3
L9_TRAINING_EXECUTED = NO
NEW_ROTATION_TRAINING_EXECUTED = NO
AIME_EXECUTED = NO
FINAL_ACTION = FAIL_AND_STOP
```

No conclusion about the observability hypothesis is authorized. A further explicit engineering correction authorization is required; the first STOP must remain immutable. No automatic retry, Discovery, training, or AIME is permitted from this failed attempt.
