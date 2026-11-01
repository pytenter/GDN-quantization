# Attempt 06 Generated-Code Qualification — FAIL AND STOP

```ini
EXPERIMENT = LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1
ENGINEERING_ATTEMPT = PROVENANCE_EXECUTION_CAPSULE_V3_GENERATED_CODE_QUALIFICATION_ATTEMPT_06
ATTEMPT_05 = FAIL_PRESERVED
OLD_EVIDENCE_INTEGRITY_GATE = PASS
TRACKED_LAUNCH_IDENTITY_GATE = PASS
GENERATED_CODE_PREFLIGHT_GATE = FAIL
GENERATED_PRODUCER_CLOSURE_GATE = NOT_COMPLETED
GENERATED_MODULE_SET_DETERMINISM_GATE = NOT_RUN
GENERATED_MODULE_BYTE_DETERMINISM_GATE = NOT_RUN
GENERATED_PREEXEC_CAPTURE_GATE = INCOMPLETE
GENERATED_FORMAL_VERIFY_GATE = NOT_RUN
GENERATED_MODULE_COUNT = NOT_ESTABLISHED
GENERATED_CODE_CANDIDATES_CAPTURED = 1
GENERATED_FILE_MODULES_CAPTURED = 0
GENERATED_EXECUTABLE_CAPSULE_ID = NOT_CREATED
HISTORICAL_K3_PARITY_GATE = NOT_RUN
HISTORICAL_JPROP_PARITY_GATE = NOT_RUN
FUTURE_KL_HARNESS_PARITY_GATE = NOT_RUN
BASIS_SEMANTICS_GATE = NOT_RUN
FRESH_PROCESS_NUMERIC_DETERMINISM_GATE = NOT_RUN
V3_QUALIFICATION_COMMIT = NOT_CREATED
V3_PREMEASUREMENT_COMMIT = NOT_CREATED
POST_COMMIT_GATES_ALL_PASS = NO
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
```

## What failed

The new guard incorrectly conflates distinct generated expressions that share the virtual filename `<string>`. During the import of the frozen historical bridge, Python statistics imports decimal and later constructs LinearRegression via collections.namedtuple. The first generated expression was captured before execution (79 exact bytes); the second legitimate expression reused `<string>` and hit GENERATED_COMPILE_INPUT_MISMATCH before execution. This is an instrumentation identity-association defect, NOT evidence of cross-process byte nondeterminism, unknown malicious producer, FP mismatch or numerical/scientific failure. Torch/FLA and model initialization were not reached.

## Preserved evidence

Attempt05 remains FAIL_PRESERVED; its unavailable temporary bytes were never recreated or relabeled. Old 187 files and all 203 frozen inputs are unchanged. Gate0 separately rehashed 125 source snapshots, 406 scientific assets, 10,179 runtime files, 531 historical inputs and 1,475 protected historical files. Old12 and new14 synthetic tests passed, but they did not cover repeated distinct dynamic expressions under the same virtual filename; that missing fixture does not turn runtime preflight into PASS. The one actual captured candidate, its source SHA and call stacks remain in runs/preflight_01/generated_capture. No complete generated module set, three-repeat qualification, frozen generated manifest, numerical parity, qualification commit or scientific premeasurement commit exists.

## Boundaries

No retry, code correction, allowlist expansion, scientific change, AIME, training or Discovery was performed after STOP. Cleanup ledger remains 1 independently of actual task/GPU process counts, both 0. Monitoring remains paused. Any future recovery must be append-only and distinguish real source-file identity from virtual-filename generation events without normalizing bytes or permitting unknown code.

Engineering-input-only commit: d4adb6e8da161dcc28990ac40ee8b4ef9a385598

STOP SHA256: b4e34043d1a97c24f0a9f27ffa6b4c27de042e16a018bbdf975350d95286907c

Preflight log SHA256: ff201b2af7e47799dd69f0804469193cbdbdabba5c8251a07b1d0c4f524ba264
