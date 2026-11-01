# Original L7 objective identity and executed forward parity

Status: original mathematical definition recovered; fixed-input CPU checks and real matched-window GPU smoke on both authorized UUIDs PASS. This is not a replication of the historical 512-token training distribution, a completed natural-rotation ranking analysis, or a 32K persistent validation result.

## Recovered source and exact objective

The unchanged historical trainer is `recurrent_dense_train.py` in the read-only cleanroom handoff. Its SHA256 is `d1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae`. The retained historical copy is byte-identical. Training-config SHA256: `fe9d889bae525e3359ed7906417aa7e4721ca177f4e44c69bb5caefb1e5fd4a7`. The scorer extracts the original `local_output`, `relative_mse`, `relative_state_loss`, `exact_qdq` and EPS assignment by AST; it does not import or execute the training entry point.

For each token and KDA layer, Functional is the sum of squared differences of student and teacher **post-o_proj outputs**, evaluated in FP32, divided by the teacher output energy clamped from below at `1e-12`. The original helper applies the real gated `o_norm` to native KDA core output, reshapes heads, and applies the layer's `o_proj`. The target is not a pre-projection core, K3 error, state MSE, or full-logit KL. Heads are mixed by the actual projection; Functional does not average independently normalized head-output losses.

State computes squared native-state error summed across the K and V axes for each batch/head, divides each head's FP state energy clamped at `1e-12`, and then averages batch and heads. Layers and tokens receive equal weight. It is not B0 raw state MSE or one global relative-MSE. Total is exactly `Functional + 0.1 * State`. No loss scaling was fitted to any correlation or outcome.

The current token's Functional output is obtained from the recurrence kernel before that token's QDQ, but already reflects prior quantized writebacks. State is evaluated after current-token QDQ and inverse rotation into native physical coordinates. This timing is retained; post-QDQ state is not used to invent a replacement current-token output.

## Historical protocol versus this measurement

Historical L7 used continuous 512-token WikiText2 captures, shared FP q/k/v/beta/decay/gates for local student and teacher rollouts, every-token INT8-R128 QDQ/writeback with BF16 quantization boundaries, and H128 truncated BPTT. Graphs detach at H128 boundaries without resetting numerical recurrent state. The model is frozen, layer rotations are shared over heads, captured q/k/v/gate/projected outputs are BF16, beta/log-decay are FP32, and attention masks are ones. Validation weights tokens; equal-length documents therefore receive equal weight. The source audit records the original Adam/STE/Cayley training details, but this task invokes no backward or optimizer update.

This task applies those exact loss definitions to actual full-model deployment tensors on fixed PG19 windows `[256,384)` and `[1024,1152)`, all 18 KDA layers and all 16 heads. It uses a fresh shared FP teacher/reference for all metrics. Student upstream drivers, native FP32 post-INT8-QDQ physical cache storage, and sampling distribution differ from the historical FP-conditioned local training rollout. The historical mathematical objective is faithfully applied; historical training capture and optimization are **NOT_REPLAYED**. FP32 physical storage does not remove the real INT8 QDQ/writeback boundary.

Metadata-relative historical `best.pt` and `L7_final_rotation.pt` are absent from the cleanroom handoff. Loss recovery is supported by source/config evidence; full historical checkpoint reconstruction is not claimed. Deployed rotation-bank hashes are tracked separately and are not substituted for raw historical training checkpoint hashes.

## Actual CPU and GPU checks

The retained CPU audit passed FP32/BF16 fixed-input identity, ordinary/tiny/zero teacher energy, exact denominator clamp, per-head State versus global-MSE counterexample, 0.1 weighting, finite values, repeated calculation, original R128 V-axis grouping and ties-to-even. CPU PASS alone was not accepted as GPU qualification.

Both real GPU smoke runs used the fixed document map, seed0 and the unchanged measurement runner. Each ran two full persistent IDENTITY trajectories to token384 with H128 scoring at start256. Actual `o_proj` outputs were hooked read-only. Reusing the original helper with real module inputs at tokens256 and257 gave 36/36 bitwise-identical post-projection comparisons per document, with zero identity loss. FP replay checks at positions255,287,383 were finite and had max-absolute and relative-L2 errors exactly zero.

Each repeated trajectory performed `384 * 18 = 6912` actual per-token/layer QDQ writebacks. Its next-kernel consumption checks matched all `383 * 18 = 6894` expected post-QDQ states. The two repetitions produced identical window scores. The runner rejected nonfinite outputs and result trees. These are executed short-window qualifications, not a claim of cache/storage/continuity correctness through32768; Phase B requires its own full-persistent qualification and benchmark.

| Document | GPU UUID | Smoke canonical SHA256 |
|---|---|---|
| pg19-test-37702 | GPU-de53f243-b271-521c-d875-d96611ed8574 | 42e74a8a70cb9cb9d8d364df6f8a5a2e015890fe61d2beef4904b2dc6c00fa82 |
| pg19-test-5396 | GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae | 0872b0074fa050d4f2ae42659259e73244d9f0364d6197ba0ab820c8b2d78f9f |

The corresponding sidecar files were rehashed and matched each canonical. Smoke FP tensor-identity digests are `49896e13fb3d46dfe66cbcc3facd006da1bbd37dd08784de2f1c157cb0fcfc66` and `7e7d1a22f85c2e18bf7103e4d930c00757b4571c23cb0187835bced781086bd7`. These identify actual post-o_proj outputs, full logits, replay drivers and checkpoint states. JSON readback restores the frozen runner's integer token/layer keys before rehashing, preserving numeric ordering rather than silently changing the identity digest.

Runtime: driver580.178.04, kernel7.0.0-38-generic, torch2.7.1+cu128, CUDA12.8, selectorBT16_W8. Measurement commit: `a74255b1b173d27a3e51f16dbc4cb26b46104d08`. Runner SHA256: `6f1b66553a4a30d8e39e1cd144f8a1f77fa98cec25027198cb600834c8179d1c`. Appended CPU analysis/accounting commit: `fb2c838a01bcbc09d2c2aef9ad4e166bb9c42c48`. Frozen measurement files remain byte-identical.

## Interpretation boundary

Real-module parity establishes that the reused loss helper measures the intended post-projection tensor under this matched deployment protocol. It does not establish gradient equivalence, faithful historical training-distribution replay, a new objective's optimization advantage, or long-horizon predictive superiority. The separately frozen Phase A roster is running; all sixteen conditions per document are required before scientific ranking analysis.

Historical validation Functional and Total minima coincide at step100 among the eleven recorded evaluations. Selection by Functional rather than Total did not pick a different recorded checkpoint. STE bias, truncation, data shift, state-weight tradeoffs, insufficient optimization and local-objective/reasoning mismatch remain untested explanations, not conclusions.
