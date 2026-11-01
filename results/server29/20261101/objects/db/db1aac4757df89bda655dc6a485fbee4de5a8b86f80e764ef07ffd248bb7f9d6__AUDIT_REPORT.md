# Acceleration audit report

## Source and scope

- Repository: `/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2`
- Branch: `exp/qwen-gdn-stage-b-streaming-acceleration-v1`
- Profiling baseline commit: `255a41b2217b7044926c0b4d41dc89eaeed6d509`
- Optimization root: `experiments/QWEN_GDN_STAGE_B_STREAMING_ACCELERATION_V1`
- Frozen V3 oracle imported unchanged.
- Frozen common-FP fan-out, rotations, formal quantizer, teacher tokens, metric
  specifications, tail specifications, histogram specifications, and TopK
  ordering imported unchanged.

## Change boundary

Only the reducer data plane was changed:

1. Batched NumPy histogram candidate extraction with scalar boundary fallback.
2. Deterministic local TopK candidate extraction and oracle-key CPU merge.
3. A gated parity and smoke harness with utilization/memory measurement.

No model, loss, checkpoint, training, rotation, recurrence, generation, or
mechanism-analysis protocol code was modified.

## Evidence

- P0 synthetic adversarial parity: 53,248 values, PASS.
- P1 synthetic tie/merge parity: 262,144 events, PASS.
- P0 formal parity: two samples / 64 tokens, byte-exact state PASS.
- P1 formal parity: two samples / 64 tokens, byte-exact state PASS.
- Final accelerated smoke: two samples / 64 tokens, PASS.
- Atomic write/hash/readback: PASS.
- Memory creep: absent.
- Forbidden-stage flags: all false.

The pre-compute smoke identity incident and its narrowly scoped repair are
preserved in `audit/smoke_failed_launch_001.json` and the corresponding logs.
