# QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2 — preparation status

- `FORMAL_ARTIFACT_IDENTITY_GATE = PASS`
- `OLD_CAPTURE_ACTIVE = YES`
- `NEW_V2_GPU_EXECUTION_STARTED = NO`
- `FORMAL_REPLAY_PARITY_CLOSURE = NOT_RUN_OLD_CAPTURE_ACTIVE`
- `MECHANISM_REPLAY_AUTHORIZED = NO`

The independent V2 directory, frozen configuration, formal-source audit, and CPU-only rotation geometry are prepared. No model was loaded and no CUDA workload was started.

The old V1 controlled capture remains `DESCRIPTIVE_ONLY`: it follows condition-specific unquantized recurrent trajectories rather than a strict per-token common-FP-state fan-out. Its formal path match is `PARTIAL_MISMATCH` because of matmul association, dtype path, and a duplicated quantizer implementation.

Stage A must use the unchanged formal rotation/quantizer/writeback path and pass exact qcode and writeback parity before any mechanism replay is authorized.
