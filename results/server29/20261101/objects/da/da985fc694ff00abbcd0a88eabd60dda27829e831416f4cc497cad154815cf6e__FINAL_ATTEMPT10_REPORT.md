# Attempt 10 — preserved preparation engineering FAIL

```ini
ATTEMPT_10 = FAIL_PRESERVED
FAILURE_CLASS = PREPARATION_EXPECTED_HASH_LITERAL_TRANSCRIPTION_BUG
PROTECTED_SOURCE_ACTUAL_BYTE_MISMATCH = NO
FORMAL_MEASUREMENTS = 0
GPU6_FORMAL_AUTHORIZED = NO
DISCOVERY_PRIMARY_COMPLETE = 0/64
DISCOVERY_SECONDARY_COMPLETE = 0/64
SCIENTIFIC_PROTOCOL_CHANGED = NO
RUNTIME_SEMANTICS_CHANGED = NO
MODEL_LOADED = NO
TESTS = NOT_RUN
REAL_PREFLIGHT = NOT_RUN
SUPERVISOR_CLEANUP_LEDGER_REMAINING = 1
ACTUAL_TASK_PROCESS_COUNT = 0
GPU6_COMPUTE_PROCESS_COUNT = 0
```

The hard-coded expected Attempt09 STOP SHA256 had an extra character. Current bytes equal the frozen Git blob exactly; all 10,730 protected files remain byte-identical. The preparation gate correctly rejected the wrong literal. No scientific asset inconsistency, numerical failure or runtime escape was observed; full asset qualification and tests were not reached. The failed preparation input package is retained exactly, and Attempt11 uses immutable prior failure inventories for expected SHA256 rather than hand transcription.
