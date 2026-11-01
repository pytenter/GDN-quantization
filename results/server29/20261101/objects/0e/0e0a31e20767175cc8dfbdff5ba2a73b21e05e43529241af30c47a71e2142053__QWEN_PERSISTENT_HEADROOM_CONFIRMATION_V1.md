# QWEN_PERSISTENT_HEADROOM_CONFIRMATION_V1

## Safety

- Server: `zypan@202.38.247.55`, 4 x RTX 3090.
- Pre-existing running jobs impacted: NO.
- AIME26 used: NO.
- Rotation retrained: NO.
- Model weights changed: NO.
- Environment changed: NO.
- Third 8 x RTX 3090 server used: NO.

## Disk cleanup

`QWEN_DENSE_ORACLE_TRACE_REPRODUCIBILITY_GATE = PASS`: all A–N evidence checks passed. Runtime checkpoint bytes match their original runtime `training_summary.json` SHA256 values. The prior aggregate summary contained one extra `0` in the Dense-Functional hash; this branch corrects that transcription error and preserves the detailed erratum in the gate JSON.

Before deletion, `/data` had 12,168,261,632 bytes available (about 12 GiB). The approved manifest contained 99 `.pt` files: 39,926,989,994 bytes of formal traces and 1,247,832,879 bytes across three smoke directories, totaling 41,174,797,650 bytes (38.347 GiB). Every path, category, suffix, size, mtime, and parent directory was independently revalidated immediately before deletion.

After deletion, `/data` had 53,344,317,440 bytes available (about 50 GiB). All 99 approved files were deleted, zero approved `.pt` files remained, and `DELETE_SCOPE_MISMATCH = false`. Checkpoints, learned matrices, source, manifests, provenance, compact evaluation results, summaries, and reports were preserved.

## Dataset and execution

- Source: pre-existing local `wikitext/wikitext-2-raw-v1` test cache.
- Panel: original 16 held-out documents in original order plus 48 new documents, 64 total.
- Raw panel SHA256: `80133307bb1fecba49ddc8929368a532e083e9c8314a9b9d3b4b05adc41ceb40`.
- TRAIN overlap: 0. VALIDATION overlap: 0. Previous calibration overlap: 0. Internal duplicates: 0. AIME24/AIME26 overlap: 0.
- Four modulo shards ran independently on GPUs 0–3 under tmux.
- Permanent output was compact JSONL/JSON/CSV only. No tensor, logit, recurrent-state, or layer trace was written.

## Previous 16-document replication

The new streaming evaluator reproduced every prior per-document AUC exactly (`max absolute difference = 0`). Mean AUCs were Native 0.0011326862, Hadamard 0.0035879625, Dense-State 0.0030362050, and Dense-Functional 0.0027485769. `PREVIOUS_16DOC_REPLICATION = PASS`.

## 64-document result

| Method | Mean AUC | Median AUC | Q25 | Q75 |
|---|---:|---:|---:|---:|
| Native INT8 | 0.00134640 | 0.00117283 | 0.00080604 | 0.00158083 |
| Hadamard | 0.00330097 | 0.00250999 | 0.00165651 | 0.00442614 |
| Dense-State | 0.00346788 | 0.00266844 | 0.00155757 | 0.00413628 |
| Dense-Functional | 0.00320486 | 0.00256067 | 0.00161815 | 0.00374430 |

Dense-Functional is the best learned method by mean AUC. For paired `Hadamard - Dense-Functional`, the mean effect is 0.0000961090 with 95% bootstrap CI [-0.0004239155, 0.0006492788]. The median effect is 0.0000578612 with 95% bootstrap CI [-0.0002346494, 0.0002548895]. Relative median reduction is 2.31%, with 36 wins, 0 ties, and 28 losses.

Dense-State has median effect -0.0001396465, 95% CI [-0.0003919438, 0.0001007638], and 28/0/36 wins/ties/losses. It shows no persistent headroom.

The Dense-Functional median-CI width shrank from 0.00136861 at 16 documents to 0.000489539 at 64 documents, but the point effect also shrank from an 11.45% relative trend to 2.31% and the CI still crosses zero. Therefore:

- `QWEN_HEADROOM_16DOC = PROMISING_NOT_SIGNIFICANT (11.45%)`
- `QWEN_HEADROOM_64DOC = PROMISING_NOT_SIGNIFICANT (2.31%)`
- `QWEN_PREVIOUS_UNCERTAINTY_PRIMARILY_SAMPLE_SIZE = NO`
- `QWEN_PERSISTENT_HEADROOM = PROMISING_NOT_SIGNIFICANT`

The larger panel gives a substantially narrower interval but does not reveal a stable >=10% dense-over-Hadamard effect. Native INT8 again has lower Future-KL than every rotated condition; this experiment only tests incremental dense-over-Hadamard headroom.

## Storage policy

The complete formal streaming directory is about 400 KiB and the entire experiment directory is about 1.1 MiB, versus a 4 GiB hard limit. `/data` remained around 50 GiB free. `QWEN_STREAMING_STORAGE_POLICY = PASS` and new raw trace size is 0.

## Recommendation

Do not claim clear dense-rotation headroom for Qwen. If another method phase is authorized, a future-aware objective is better motivated than further local dense fitting; otherwise retain Hadamard or Native INT8 according to the deployment objective. This phase does not authorize or execute future-aware training, per-head training, HARP, or Butterfly.
