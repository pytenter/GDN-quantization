# Ling Persistent Headroom Confirmation V1

## Scope and safety

- Model/runtime: Ling-3.0-tiny / KDA on the two-RTX4090 server.
- Evaluation only. No optimizer, backward pass, checkpoint update, rotation retraining, model-weight change, environment upgrade, AIME data, 256K output, HARP, Butterfly, or future-aware training was used.
- The third eight-RTX3090 server was not used.
- Existing `kda_closed_loop_v1` and `kda_decay_closure_v1` tmux sessions were not changed.
- During the final run, an unrelated `user14` process began using about 4.15 GiB on GPU0. The Ling process remained healthy and completed; this concurrent load is recorded as a timing caveat, not a numerical correction.

## Frozen provenance and semantics

The previous aggregate summary contained one-character-long SHA strings. Direct hashing of the frozen checkpoints matched the original training summaries, so this was corrected as a metadata erratum without changing either checkpoint:

- Dense-State: `dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318`
- Dense-Functional: `65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c`

The formal evaluator used the frozen checkpoints and the corrected KDA path:

```text
KDA_ROTATION_SEMANTICS_VERSION = CORRECTED_PREFILL_ENDPOINT_V2
REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO
PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY = PASS
```

## Dataset and streaming execution

- Source: pre-existing WikiText-2 raw test cache.
- Panel: the original 16 documents in identical order plus 48 new held-out documents, 64 total.
- Shared raw-text corpus SHA-256: `80133307bb1fecba49ddc8929368a532e083e9c8314a9b9d3b4b05adc41ceb40`.
- Frozen Ling manifest SHA-256: `233c711b634d69e31a0b3bc8cb44c48cf871634a74f37415fec5843b45ebbd61`.
- Qwen/Ling raw-text match: PASS.
- Train, validation, previous calibration, AIME24 test, and AIME26 overlap: 0.
- Internal duplicates: 0.
- Two contiguous formal shards covered indices 0–31 and 32–63. Both completed 32/32; 64 unique document IDs and indices 0–63 were verified.
- Nonfinite metrics: 0.
- New raw tensor traces: 0 bytes. Compact streaming evidence: 405,910 bytes; complete experiment directory including audits and ignored diagnostics: 1,054,728 bytes.

## Runtime repeatability audit

An early modulo-sharded attempt was invalidated and excluded because it changed execution order for the original 16 documents. Repeated one-document checks using the legacy warmup path still showed runtime nondeterminism, especially for learned rotations. The invalid run and smoke diagnostics are retained only as ignored audit material and are not mixed into the formal 64-document files.

The predeclared Ling replication gate operates on each method's frozen 16-document aggregate, with tolerance `max(0.001, 10% × previous AUC)`.

| Method | Previous 16-doc AUC | New 16-doc AUC | Absolute difference | Tolerance | Gate |
|---|---:|---:|---:|---:|---|
| Native INT8 | 0.03792503 | 0.03702756 | 0.00089747 | 0.00379250 | PASS |
| Hadamard | 0.02627500 | 0.02261284 | 0.00366216 | 0.00262750 | FAIL |
| Dense-State | 0.02513393 | 0.03042467 | 0.00529073 | 0.00251339 | FAIL |
| Dense-Functional | 0.02723232 | 0.02428122 | 0.00295110 | 0.00272323 | FAIL |

```text
PREVIOUS_16DOC_REPLICATION = FAIL
SCIENTIFIC_CONCLUSION_AUTHORIZED = false
```

Per protocol, the replication failure blocks a formal 64-document scientific conclusion. All results below are diagnostic only.

## Diagnostic 64-document result

| Method | Mean AUC | Median AUC | Q25 | Q75 |
|---|---:|---:|---:|---:|
| Native INT8 | 0.03086828 | 0.01417374 | 0.00712517 | 0.02588728 |
| Hadamard | 0.02498716 | 0.01230396 | 0.00639075 | 0.03120301 |
| Dense-State | 0.02452921 | 0.01291912 | 0.00673174 | 0.03212534 |
| Dense-Functional | 0.02579457 | 0.01222049 | 0.00606143 | 0.03011248 |

Hadamard minus Dense-State:

- Mean effect: `+0.000457953`; paired bootstrap 95% CI `[-0.00593859, +0.00825807]`.
- Median effect: `-0.0000618154`; paired bootstrap 95% CI `[-0.00125933, +0.00207536]`.
- Relative median reduction: `-0.5024%`.
- Wins/ties/losses: `31/0/33`.

Hadamard minus Dense-Functional:

- Mean effect: `-0.000807416`; paired bootstrap 95% CI `[-0.00718494, +0.00722758]`.
- Median effect: `-0.000564373`; paired bootstrap 95% CI `[-0.00211510, +0.000411727]`.
- Relative median reduction: `-4.5869%`.
- Wins/ties/losses: `28/0/36`.

Both bootstrap analyses used 10,000 paired document-level resamples. Dense-State has the lowest diagnostic mean AUC, but neither learned method has a positive median effect or a CI excluding zero.

## Interpretation

```text
LING_LOCAL_GAIN_PERSISTS = YES
LING_PERSISTENT_GAIN_64DOC = DIAGNOSTIC_ONLY
LING_OBJECTIVE_TRANSFER_GAP = INCONCLUSIVE_REPLICATION_FAIL
LING_PERSISTENT_HEADROOM = INCONCLUSIVE
```

The 64-document diagnostic trend does not establish persistent dense-rotation headroom. It also cannot formally confirm the previous objective-transfer-gap claim because the required original-16 replication gate failed. The appropriate next action is to resolve Ling runtime reproducibility/protocol drift before spending compute on a future-aware objective. The future-aware feasibility plan is documented but was not executed.

## Evidence

- `results/rotation/ling_persistent_headroom_confirmation_v1/final/summary.json`
- `results/rotation/ling_persistent_headroom_confirmation_v1/final/per_document_auc.csv`
- `results/rotation/ling_persistent_headroom_confirmation_v1/final/PERSISTENT_HEADROOM_64DOC_MANIFEST.json`
- `results/rotation/ling_persistent_headroom_confirmation_v1/LING_PREVIOUS_16DOC_REPLICATION_GATE.json`
- `results/rotation/ling_persistent_headroom_confirmation_v1/LING_RUNTIME_REPEATABILITY_AUDIT.json`
- `results/rotation/ling_persistent_headroom_confirmation_v1/LING_CHECKPOINT_PROVENANCE_AUDIT.json`
- `results/rotation/ling_persistent_headroom_confirmation_v1/provenance/PREVIOUS_16DOC_PERSISTENT_FUTURE_KL.json`
