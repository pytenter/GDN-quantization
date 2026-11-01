# Runtime and horizon scope analysis

## Five-level assessment

### Level 1 — identity consistency: CONFIRMED

Checkpoint/source, documents, rotations, tokenizer-related assets, quantizer/recurrent implementation and frozen runtime assets have auditable byte identities. This confirms configuration continuity only; it cannot establish numerical equivalence between drivers.

### Level 2 — same-runtime determinism: CONFIRMED FOR OLD RUNTIME

Three fresh processes matched exactly on both assigned GPUs under driver 580.173.02. This establishes repeatability conditional on the same GPU, driver and runtime. It does not authorize cross-driver merging.

### Level 3 — cross-runtime anchored comparison: INSUFFICIENT FOR MERGE

A post-reboot GPU6 short probe observed exact equality to the frozen short historical reference, but no applicable pre-frozen rule declares this short fixture sufficient to qualify a full 32K panel. GPU3 has no completed same-UUID cross-driver comparison in this audit. Driver-induced divergence at the relevant horizon remains `NOT_EVALUATED`.

### Level 4 — long-horizon applicability: INSUFFICIENT EVIDENCE

GPU3's FP-only reference replay is bitwise for 32768 tokens, but the scientific units add persistent every-token INT8 QDQ/writeback, recurrent feedback, error accumulation, K3/POR and Future-KL. Historical K3/JPROP/Future-KL gates use local, 32-transition, 128-token, or 4-token fixtures. No frozen implication from these fixtures to the full 32768-token trajectory exists.

### Level 5 — merge-level validity: FAIL

The historical finalizer validates a single frozen panel identity and does not encode a mixed-driver admission rule. Therefore the proposed `80 old + 48 new` panel has no established numerical merge qualification.

## Risk register

- `driver-induced numerical divergence`: `NOT_EVALUATED` at 32K; short GPU6 fixture exact PASS does not close the risk.
- `FP reference mismatch`: old per-GPU references are `CONFIRMED` and must remain per UUID; new-driver full-horizon pairing is `NOT_EVALUATED`.
- `long-horizon error propagation sensitivity`: `INSUFFICIENT_EVIDENCE` across drivers.
- `Future-KL / observability comparability`: `INSUFFICIENT_EVIDENCE` across drivers at 32K.
- `cross-GPU provenance mismatch`: `CONFIRMED AND CONTROLLED` historically by per-GPU references; cross-hardware FP bitwise failure remains preserved and is not the current cross-driver answer.
- `scientific hypothesis`: `NOT_EVALUATED`; no inference is permitted from this engineering/qualification stop.

## Acceptance-rule boundary

Exact equality is frozen for several short/local checks. The only numerical tolerances found (`1e-4`, `1e-6`) belong to basis semantics. Neither exact short-fixture equality nor those basis tolerances have a frozen sufficiency claim for mixed-driver full-horizon measurements. No tolerance was invented or widened in this audit.
