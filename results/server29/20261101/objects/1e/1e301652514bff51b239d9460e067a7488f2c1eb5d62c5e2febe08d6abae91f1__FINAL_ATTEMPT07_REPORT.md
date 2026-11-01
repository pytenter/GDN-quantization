# Attempt 07 Generated Execution Identity — FAIL AND STOP

```ini
EXPERIMENT = LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1
ENGINEERING_ATTEMPT = GENERATED_CODE_IDENTITY_ATTEMPT_07
ATTEMPT_06 = FAIL_PRESERVED
OLD_EVIDENCE_INTEGRITY_GATE = PASS
ATTEMPT_07 = FAIL_AND_STOP
GENERATED_CODE_PREFLIGHT_GATE = NOT_RUN
GENERATED_EVENT_SEQUENCE_DETERMINISM_GATE = NOT_RUN
GENERATED_MODULE_BYTE_DETERMINISM_GATE = NOT_RUN
GENERATED_PRODUCER_CLOSURE_GATE = NOT_COMPLETED
GENERATED_FORMAL_VERIFY_GATE = NOT_RUN
GENERATED_EXECUTABLE_CAPSULE_ID = NOT_CREATED
V3_QUALIFICATION_COMMIT = NOT_CREATED
V3_PREMEASUREMENT_COMMIT = NOT_CREATED
GPU6_FORMAL_AUTHORIZED = NO
FORMAL_MEASUREMENTS = 0
DISCOVERY_PRIMARY_COMPLETE = 0/64
DISCOVERY_SECONDARY_COMPLETE = 0/64
L9_TRAINING_EXECUTED = NO
NEW_ROTATION_TRAINING_EXECUTED = NO
AIME_EXECUTED = NO
FREE_GENERATION_EXECUTED = NO
FP_REFERENCES_MIXED_ACROSS_HARDWARE = NO
SUPERVISOR_CLEANUP_LEDGER_REMAINING = 1
ACTUAL_TASK_PROCESS_COUNT = 0
GPU6_COMPUTE_PROCESS_COUNT = 0
FINAL_ACTION = FAIL_AND_STOP
UNCHANGED_ATTEMPT06_14_TESTS = PASS
OLD12_ARCHITECTURE_TESTS = NOT_RUN
ATTEMPT07_IDENTITY_TEST_SUITE = FAIL
TORCH_FLA_MODEL_PREFLIGHT = NOT_STARTED
```

## Blocking evidence

RuntimeError: UNKNOWN_GENERATED_PRODUCER: /data/zypan/envs/ling-kda/lib/python3.11/pathlib.py

T_IDENTITY_01 through T_IDENTITY_05 CAPTURE passed, together with changed-byte and reordered-execution VERIFY denial cases. T_IDENTITY_06 stopped while its fixture wrote a generated file with pathlib.write_bytes: the synthetic static identity map froze the test file but omitted pathlib.py. The guard rejected that unlisted writer frame. This is a defect in the newly prepared synthetic fixture closure, not an observed unknown producer in the real Torch/FLA runtime, not a virtual-filename collision, and not a scientific failure. T_IDENTITY_07/08 and later semantics cases were not run; old12 execution was not reached. All unchanged Attempt06 fourteen tests passed using their unchanged historical implementation, not as qualification of the new guard.

Full traceback and all successful invocation lines remain in SYNTHETIC_TEST_OUTPUT.log and STOP.json. Synthetic TemporaryDirectory capture files were cleaned up by the test harness and are unavailable; they were not recreated as observed evidence. No actual model-generated executable capture was attempted.

STOP SHA256: 004f0f5a81208c9ece71fe3d6ef9fbe9b6f8e7e30d9119963314e45a9928029b

No automatic retry or manifest expansion. Historical 214 protected files, including all Attempt06 code, captured bytes, STOP and reports, remain byte-identical. This engineering failure does not imply a scientific outcome. No Discovery or formal GPU authorization was issued. Cleanup ledger remains 1, independent of actual process/GPU counts.
