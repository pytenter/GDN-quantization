# Post-final hash-manifest amendment 001

Classification: `ENGINEERING_ONLY_FINALIZATION_ORDER`

The original `SHA256SUMS` and `FINAL_SHA256SUMS` are preserved unchanged. The final manifest was generated immediately before the finalizer and controller appended their terminal log records, so exactly two append-only log files no longer match their pre-terminal hashes:

- `discovery/logs/finalize.log`
- `execution/controller_recovery003.log`

No scientific result, frozen input, model, checkpoint, quantizer, rotation, metric, threshold, deterministic configuration, replay unit, or decision rule was modified. After the controller exited successfully, `POSTFINAL_SHA256SUMS` was generated over the stable experiment tree (excluding itself) and verified in full.

Final hashes of the two append-only logs:

```text
606d7fe0f420f17b232f9cdc9950165347e55003994f28cd03c76b73b21573ec  discovery/logs/finalize.log
271ff1f0b9cc0f19a657bb7504bb1ffe8c97a6106f4f0e7dc2a8a252353eceb6  execution/controller_recovery003.log
```

The historical manifests and their finalization-order mismatch remain available as provenance evidence.
