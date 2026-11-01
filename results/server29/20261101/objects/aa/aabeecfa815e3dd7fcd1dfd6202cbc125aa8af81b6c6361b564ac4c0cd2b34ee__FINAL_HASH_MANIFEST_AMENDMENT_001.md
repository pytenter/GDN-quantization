# Final hash-manifest amendment 001

Classification: `ENGINEERING_ONLY_FINALIZATION_ORDER`

The original `SHA256SUMS` is preserved unchanged. It was generated immediately before the final analyzer and controller processes appended their terminal records, so exactly three append-only log files no longer match their pre-finalization hashes:

- `logs/analyze.log`
- `logs/controller_events.jsonl`
- `logs/resume_controller_001.log`

No scientific result, frozen pair identity, runner, metric, trajectory, decision rule, or completed unit was modified. After all controller processes exited, `SHA256SUMS_FINAL` was generated over the stable experiment tree (excluding itself) and verified in full.

Final hashes of the three append-only logs:

```text
0957e73fa2087ffc2d45f3be66d2d5e269ec39678a005ff8c0ba55b2ec9f6fb1  logs/analyze.log
f9f320645f4e248544991b2a85a374bd8e988bf2e623b8fb570df41d9e44ca8f  logs/controller_events.jsonl
ff1f31d593f3a8c87f0981e8f4be4f377b9801330476cc76ea5b41e2c95174e3  logs/resume_controller_001.log
```

The historical mismatch and the original manifest remain available as provenance evidence.
