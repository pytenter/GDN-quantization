# Prospective preregistration: QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1

Status: **PROTOCOL FROZEN; C5/C6 FORMAL TRAINING NOT STARTED**

## Canonical-definition audit

- Unique canonical C5/C6 source: `/root/autodl-tmp/canonical_source/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py`
- Canonical source SHA256: `48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e`
- Rotation/quantizer source SHA256: `62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067`
- Selection rule: minimum fixed held-out local primary loss; exact ties retain the earliest candidate.
- C5 optimized objective: recurrent state relative-MSE.
- C6 optimized objective: local out-proj relative-MSE + 0.1 x state relative-MSE; selection primary is local out-proj relative-MSE alone.

## Frozen matched protocol

- One 48GB vGPU; no distributed training.
- H=64; 1024 tokens/document; 8 canonical deterministic captures/document.
- 1000 target exposures = 125 document-level Adam updates.
- C5 LR 0.003; C6 LR 0.001; weight decay 0; global-norm clip 1.0.
- Real recurrent INT8-C128 writeback; graph detached only at H64 boundaries; one optimizer step per document.
- C5 and C6 use the identical frozen document order and validation panel, and independently start at theta=0 / Hadamard.
- Validation uses 16 documents x 8 captures at the 20 frozen boundaries listed in `configs/formal_training_protocol.json`.

## Provenance and erratum

Readiness V2 at `40c06322d5f9ad092fbfb908fabb29a107a0446f` selected H64. Its baseline parent is `d3f1ab61925dc91d7215de0c505cbc01c9043d65`. The historical invalid 65-character first-document digest remains preserved; the explicit one-character erratum is referenced, and the corrected canonical digest is `0e5f826ac7e3f7e5bcd4dd889ccf512a1b216a140522d9fc1993310da0af3caf`.

## Stop and exclusion policy

All stop rules are frozen in `preregistration.json`. AIME is not part of this task, is not used for checkpoint selection, and remains NOT_STARTED. H128 is not retried. No objective, loss, LR, update count, precision, batch, capture count, or architecture may be changed post hoc.
