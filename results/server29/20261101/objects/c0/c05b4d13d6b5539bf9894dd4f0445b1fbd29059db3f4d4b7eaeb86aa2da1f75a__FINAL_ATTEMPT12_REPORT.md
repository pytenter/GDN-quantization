# Attempt 12 — unknown native binary FAIL preserved

```ini
ATTEMPT_12 = FAIL_PRESERVED
FAILURE_CLASS = UNKNOWN_NONADMITTED_NATIVE_CACHE_BINARY
PROTECTED_SOURCE_ACTUAL_BYTE_MISMATCH = NO
FORMAL_MEASUREMENTS = 0
DISCOVERY_PRIMARY_COMPLETE = 0/64
DISCOVERY_SECONDARY_COMPLETE = 0/64
GPU6_FORMAL_AUTHORIZED = NO
SCIENTIFIC_PROTOCOL_CHANGED = NO
MODEL_SEMANTICS_CHANGED = NO
QDQ_SEMANTICS_CHANGED = NO
KDA_RECURRENCE_CHANGED = NO
BASIS_SEMANTICS_CHANGED = NO
DOCUMENT_SET_CHANGED = NO
ROTATION_SET_CHANGED = NO
HORIZON_CHANGED = NO
METRIC_CHANGED = NO
TARGET_CHANGED = NO
TOLERANCE_RELAXED = NO
ADVANCEMENT_RULE_CHANGED = NO
TESTS = 30_NEW_PLUS_14_OLD_CONTROL_PLUS_12_OLD_ARCHITECTURE_PASS
HISTORICAL_NUMERICAL_PARITY_SUITE = NOT_COMPLETED
NATIVE_BINARY_IDENTITY_GATE = FAIL
NATIVE_BINARY_LOADER_ADMISSION = DENIED_BEFORE_EXTENSION_CREATE_MODULE
NATIVE_BINARY_SOURCE_TOOLCHAIN_ATTRIBUTION = NOT_ESTABLISHED
AUTO_WHITELIST = NO
AUTO_RETRY = NO
REQUEST_EXPLICIT_USER_AUTHORITY = YES
SUPERVISOR_CLEANUP_LEDGER_REMAINING = 1
ACTUAL_TASK_PROCESS_COUNT = 0
GPU6_COMPUTE_PROCESS_COUNT = 0
L9_TRAINING_EXECUTED = NO
NEW_ROTATION_TRAINING_EXECUTED = NO
AIME_EXECUTED = NO
FREE_GENERATION_EXECUTED = NO
FINAL_ACTION = FAIL_AND_STOP
```

T_IDENTITY_06 closure and all synthetic/old regressions passed. Actual smoke reached the model FLA/Triton convolution launcher. The extension loader denied a pre-existing cached launcher binary before its create_module call. The requesting Triton producer Python source matches the existing frozen runtime manifest, but the cached binary has no frozen identity or recorded source-to-binary/toolchain provenance. Its name/cache path alone does not establish admission. No native whitelist, scientific change or numerical tolerance relaxation was performed. Smoke and parity did not complete: this is not a confirmed numerical parity FAIL, numerical nondeterminism or scientific conclusion. All previous protected files and historical STOP receipts remain byte-identical. The blocked .so is copied as nonexecuted forensic failure evidence only, never as an accepted runtime asset.
