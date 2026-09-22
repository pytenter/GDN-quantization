# PERSISTENT_HEADROOM_CONFIRMATION_V1 Protocol

## Scope and safety

This phase evaluates frozen Hadamard, Dense-State, and Dense-Functional rotations on an expanded held-out panel. It performs inference only: no optimizer, backward pass, Cayley update, checkpoint update, model-weight change, environment upgrade, AIME input, HARP, Butterfly, or future-aware training is permitted. The third 8 x RTX 3090 server is out of scope.

Qwen raw Dense Oracle traces may be deleted only after the A–N reproducibility gate passes and only through the reviewed `QWEN_TRACE_DELETE_MANIFEST.json`. Checkpoints, learned rotations, source, manifests, provenance, compact metrics, summaries, and reports are permanent evidence.

## Frozen panel

The panel contains the original 16 held-out WikiText-2 raw test documents in their original order plus the next 48 entries from the same deterministic shuffled document pool (`seed = 20260923`). All documents are non-AIME general text. Raw texts are identical across Qwen and Ling; each model uses its own frozen tokenizer. Required checks are zero overlap with TRAIN, VALIDATION, previous training calibration, AIME24 test, and AIME26, plus zero internal duplicates.

## Frozen methods and semantics

- Native INT8 is a reference condition.
- Hadamard is the canonical fixed rotation.
- Dense-State and Dense-Functional load the prior phase's frozen best checkpoints. Checkpoint SHA256 must match the runtime training summary before model loading.
- Qwen uses Key-side `INT8_C128`, canonical Hadamard followed by FP32 learned `DeltaR`, and the existing numerical closure policy.
- Ling uses Value-side `INT8_R128`, canonical H128 followed by FP32 learned `DeltaR`, canonical BF16 functional boundaries, and `CORRECTED_PREFILL_ENDPOINT_V2`. The prefill-returned state is cached directly; redundant endpoint rotation is forbidden.

## Streaming evaluation

Each document is evaluated independently with teacher forcing and the same future tokens on FP and INT8 paths. Horizons are 1, 4, 8, 16, 32, 64, and 128. Future-KL AUC reuses the prior normalized trapezoidal definition. The evaluator writes one compact JSONL row per document and immediately releases all recurrent states, logits, and cache tensors. No raw tensor trace is written.

Qwen runs four deterministic modulo shards, one per RTX 3090. Ling runs two modulo shards, one per RTX 4090. Runs are resumable by document ID. Qwen checks `/data` free space and experiment storage at least every eight completed documents; it stops below 8 GiB free or above 4 GiB experiment storage.

## Gates and statistics

The original 16-document subset must reproduce the previous per-document AUCs within `1e-6`; otherwise the 64-document conclusion is blocked. The primary unit is the document. For each learned method, report Hadamard-minus-learned mean and median effects, paired 10,000-resample document bootstrap CIs for both, relative reduction, and win/tie/loss counts (`1e-12` tie tolerance).

Verdicts are `CLEAR_PERSISTENT_HEADROOM`, `SMALL_BUT_SIGNIFICANT_HEADROOM`, `PROMISING_NOT_SIGNIFICANT`, or `NO_PERSISTENT_HEADROOM` under the task-defined median-effect, CI, and 10% relative-reduction rules. Ling may additionally be classified `OBJECTIVE_TRANSFER_GAP_CONFIRMED` when local gains still fail to transfer persistently.
