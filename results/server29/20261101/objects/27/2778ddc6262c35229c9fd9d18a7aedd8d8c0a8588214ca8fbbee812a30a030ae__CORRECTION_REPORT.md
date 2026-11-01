# V3 expected-hash correction — attempt 02

The approved hash-only correction is complete in a separate `sealed_runner_attempt02.py` entrypoint. The original runner, first STOP, first report and all original candidate source/input manifests remain byte-identical. No change was made to scientific functions, fixtures, documents, rotations, metrics, thresholds or tolerances.

Static checks: syntax, exact allowed diff, corrected 64-character hash matching original source manifest, all 125 source snapshots and 157 previous candidate files: PASS. This is not numerical/runtime qualification.

GPU6 snapshot: `GPU-de53f243-b271-521c-d875-d96611ed8574, NVIDIA GeForce RTX 3090, 17891 MiB, 6234 MiB, 580.173.02`

Concurrent occupant: account `ychu`, PID `2391698`, `GPU-de53f243-b271-521c-d875-d96611ed8574, 2391698, /data/ychu/miniconda3/envs/forcingkv/bin/python, 17868 MiB`.

```ini
HASH_CORRECTION = PASS
STATIC_CORRECTION_SCOPE_GATE = PASS
FIRST_V3_STOP_PRESERVED = YES
NUMERICAL_QUALIFICATION_LAUNCH = NOT_STARTED
FRESH_PROCESS_COMPLETE = 0/3
GPU6_FORMAL_AUTHORIZED = NO
V3_QUALIFICATION_COMMIT = NOT_CREATED
V3_PREMEASUREMENT_COMMIT = NOT_CREATED
FORMAL_MEASUREMENTS = 0
DISCOVERY_PRIMARY_COMPLETE = 0/64
DISCOVERY_SECONDARY_COMPLETE = 0/64
L9_TRAINING_EXECUTED = NO
NEW_ROTATION_TRAINING_EXECUTED = NO
AIME_EXECUTED = NO
SUPERVISOR_CLEANUP_LEDGER_REMAINING = 1
ACTUAL_TASK_PROCESS_COUNT = 0
GPU6_COMPUTE_PROCESS_COUNT = 1
HARDWARE_ASSIGNMENT_GATE = FAIL
FINAL_ACTION = FAIL_AND_STOP
```

No unrelated GPU process was signalled or terminated, no GPU reassignment was performed, and no model was loaded. Numerical parity, three fresh processes and post-commit gates remain unexecuted. The original first-attempt report remains valid for that attempt; this report is append-only.
