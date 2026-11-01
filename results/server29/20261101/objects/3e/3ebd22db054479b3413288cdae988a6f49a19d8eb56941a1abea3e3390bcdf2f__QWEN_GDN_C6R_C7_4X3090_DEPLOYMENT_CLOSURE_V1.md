# QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1

```text
EXPERIMENT = QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1
PARENT_EXPERIMENT = QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
C5R_REPRODUCTION = INCONCLUSIVE
C5R_AIME = SKIP
C6R_SOURCE_EXPORT_GATE = PASS
C7_SOURCE_EXPORT_GATE = PASS
C6R_TRANSFER_SHA_GATE = PASS
C7_TRANSFER_SHA_GATE = PASS
QWEN_3090_MODEL_IDENTITY_GATE = PASS
C6R_ROTATION_LOAD_GATE = FAIL
C7_ROTATION_LOAD_GATE = FAIL
C6R_ORTHOGONALITY_GATE = FAIL
C7_ORTHOGONALITY_GATE = FAIL
ROTATION_MAPPING_GATE = FAIL
QDQ_SEMANTICS_GATE = FAIL
REAL_RECURRENT_WRITEBACK_GATE = FAIL
A800_3090_NUMERICAL_PARITY_POLICY = NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE
C6R_DEPLOYMENT_SMOKE_GATE = FAIL
C7_DEPLOYMENT_SMOKE_GATE = FAIL
C6R_3090_DETERMINISM_GATE = NOT_RUN
C7_3090_DETERMINISM_GATE = NOT_RUN
FOUR_GPU_CONSISTENCY_GATE = NOT_RUN
DEPLOYMENT_MEMORY_GATE = FAIL
PROTECTED_ARTIFACT_REVALIDATION_GATE = PASS
C6R_3090_DEPLOYMENT_READY = NO
C7_3090_DEPLOYMENT_READY = NO
READY_FOR_C6R_C7_CANONICAL60 = NO
C6R_AIME_STARTED = NO
C7_AIME_STARTED = NO
NEW_AIME_GENERATION = 0
SCORING_STARTED = NO
PUSH_PERFORMED = NO
TRAINING_RERUN = NO
C5R_DEPLOYED = NO
SMOKE_RETRY_STARTED = NO
FUTURE_CONFIG_FREEZE = NOT_PERFORMED
FINAL_STATUS = STOPPED_FAIL_CLOSED
```

## Outcome

Stopped after GPU0 repeat1 failed at final model cleanup with `RuntimeError: unexpected memory retention`. No smoke retry, second repeat, GPU1–3 smoke, future config freeze, AIME generation or scoring was started.

Dynamic load/orthogonality/mapping/QDQ/writeback FAIL entries mean closure is unverified because the unit did not persist results. They do not establish numerical failure of the frozen rotations. The original CPU-only QDQ audit is static PASS only.

## Failure evidence and limits

The exception is at the final allocated-memory cleanup assertion (<1 MiB). Earlier condition loops and base-weight rehash occur before it, but results and memory points are written only after it. Exact in-process allocated/reserved/peak bytes and output fingerprints are unavailable; no values are fabricated.

Static inspection finds a retained reference chain: observer local `original` -> formal `_wrap.wrapped` closure -> patch `self` -> `self.model` (and rotation bank). The local `observed` also retains `original`. This is a plausible observer-lifetime defect, not a GPU-verified root cause. No post-failure patch or rerun was applied. Frozen scientific code was not modified.

Files: `analysis/gpu0_repeat1_smoke.error.json`, `logs/gpu0_repeat1.log`, `logs/gpu0_repeat1.exitcode`, `logs/smoke_controller.error.json`, `analysis/failure_static_reference_audit.json`.

## Passed identity and preservation gates

A800 selected exports and exact transfer SHA PASS. Base model/config/tokenizer bytes match the A800 trusted freeze and historical canonical runtime. Source final revalidation: 51 files PASS. Target protected final revalidation: 3080 files PASS (size, mtime, SHA); transferred members also unchanged.

C6R rotation SHA256: `a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6`.
C7 rotation SHA256: `57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8`.

## Hardware and provenance

Hostname: `hostname`. Source A800 closure commit and target pre-task HEAD: `cfae4815f5d157a2b7a771adae12dd487afb7ed2`. Target branch: `exp/qwen-gdn-c6r-c7-4x3090-deployment-v1`.

```json
{
  "cuda": "12.1",
  "python": "3.10.18",
  "torch": "2.5.1+cu121",
  "transformers": "5.16.0.dev0"
}
```

```text
0, NVIDIA GeForce RTX 3090, GPU-a78176ed-f97f-c54f-6201-5a01b570f422, 535.183.01, 24576
1, NVIDIA GeForce RTX 3090, GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3, 535.183.01, 24576
2, NVIDIA GeForce RTX 3090, GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31, 535.183.01, 24576
3, NVIDIA GeForce RTX 3090, GPU-30b07aba-a728-7649-4899-e09d33f8bc75, 535.183.01, 24576
```

Architecture: manual HF, independent single-GPU workers; no TP/FSDP/ZeRO/SGLang. Cross-hardware policy: no preexisting A800–3090 numerical tolerance; none invented. Fixed historical non-AIME smoke: seed1, TRAIN:first-row 32-token prefix, Hadamard/C6R/C7, no free generation, engineering-only.

## Launch incidents and code review

An initial tmux redirection failed because the logs directory did not exist, before Python/GPU work. This wrapper-I/O-only incident was recorded in logs/tmux_launch_incident_001.json. Directory creation was corrected before the sole GPU attempt; it was not a model/sample retry.

Original prepare source preserved. A prospective observer amendment before GPU execution removed an unused thread override and deleted a local parameter reference; its exact diff and source hashes are in analysis/pre_gpu_runtime_and_cleanup_source_amendment.json. The later wrapped-operator reference issue was not modified.

## Stop and next authority

READY_FOR_C6R_C7_CANONICAL60 = NO. All four GPUs released their deployment process. Proposed next action, not executed: separately authorize a minimal observer-reference-lifetime repair with audit and identical fixed smoke; do not change model, quantizer, rotation, numerical tolerances or protocol. AIME requires a later explicit authorization even after readiness.

## Commit and integrity

JSON parse and SHA manifest are verified before one local diagnostic deployment-closure commit. No push. The exact commit SHA and reviewed staged scope are saved after commit in logs/deployment_commit.json (excluded from the pre-commit manifest to avoid a self-referential commit). Mutable finalizer console log is excluded; all earlier failure logs are included.
