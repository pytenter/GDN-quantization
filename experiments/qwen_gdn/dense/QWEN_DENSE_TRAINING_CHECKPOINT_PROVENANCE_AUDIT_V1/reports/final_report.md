# QWEN Dense Training Checkpoint Provenance Audit V1

## 1. Executive verdict

`QWEN_DENSE_TRAINING_PROVENANCE_AUDIT = FAIL` for the narrow claim that the old checkpoints were trained with real recurrent INT8 history. Both were trained as documented local oracles with `FP_STATE_RESET_EACH_TOKEN`: every optimizer step starts from a detached FP teacher state and no quantized updated state is written back into a later training token.

This is not evidence that Dense/Cayley is intrinsically invalid. The checkpoints, rotation reconstruction, Key-side basis, exact QDQ formula, orthogonality, data separation, checkpoint selection, and current C3/C4 loader all validate. Accordingly both checkpoints are `CONDITIONALLY_VALID` as local one-step objectives, while their E2E interpretation is `CONDITIONAL`, not FULL.

## 2. Current formal-task context

The audited objects are exactly C3 `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt` (`a3597986bf9fd57e8af8dfcd13cbb8b443e1e641459444510a41ee556955074c`) and C4 `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt` (`758a5ab138d23fa5dae5666cd51039550ec662b565d9b4f4a3dcbb50ca61cc4c`). No AIME outcome was used as audit evidence. The formal run was not interrupted and no GPU workload was added.

## 3. Checkpoint lineage

Both checkpoint hashes match the current formal loader, training summaries, and original report. Training logs and calibration manifests exist. Lineage is nevertheless PARTIAL: no exact full launch command, optimizer state, intermediate checkpoints, or trace tensor shards remain; the training runner was an uncommitted working-tree variant and the checkpoint embeds no source hash.

## 4. Exact rotation parameterization

Each checkpoint stores 24 FP32 tensors `rotations.<layer>.theta`, each shape `[8128]`, one layer-shared-across-heads Cayley free parameter. The code constructs skew `A`, then `DeltaR=(I-A)(I+A)^-1`. The deployed row-vector transform is `q'=q H DeltaR`, `k'=k H DeltaR`; state coordinates are `S'=(H DeltaR)^T S`. No reorthogonalization is applied. All matrices are finite, full rank, and pass the frozen `1e-4` orthogonality threshold in saved FP32 and recomputed FP64.

## 5. Exact Dense-State objective

The target is the detached FP32 native-basis teacher `state_input`, not a recurrently rolled quantized state. Per layer it computes relative squared L2 after rotate -> exact C128 QDQ/STE -> native recovery, sums K/V, averages batch/head, and averages all 24 layers. Epsilon is `1e-12`.

## 6. Exact Dense-Functional objective

This is `LOCAL_BLOCK_OUTPROJ_OBJECTIVE`, not logits or full-block loss. It injects QDQ of the FP teacher state into one local recurrence using teacher q/k/v/g/beta, then compares local RMSNorm/gate/`out_proj` output against the detached FP teacher `out_proj_output`. Optimization uses `L_outproj + 0.1 L_state`; checkpoint selection uses `L_outproj` alone.

## 7. Recurrent-training semantics

Both modes are `FP_STATE_RESET_EACH_TOKEN`, unroll/BPTT length 1. Teacher-forced tokens and FP recurrent-state teacher forcing are both present. Quantized history never propagates to token `t+1`, and layer objectives do not propagate quantized effects upstream/downstream. Therefore `REAL_RECURRENT_TRAINING_GATE = FAIL`.

## 8. QDQ/writeback comparison

The isolated QDQ operator matches formal numerically: FP32 symmetric INT8, K-axis C128 (`-2`), scale `[B,H,1,V]`, `max(abs)/127`, floor `1e-12`, `torch.round`, clamp `[-127,127]`, zero point 0, dequant `codes*scale`. Precision boundaries match. Writeback does not: training performs no recurrent writeback, while formal evaluation QDQs each updated decode state in place for the next token. Thus QDQ is `NUMERICALLY_EQUIVALENT`, precision PASS, but writeback FAIL and `OLD_DENSE_RUNTIME_SEMANTICS = MISMATCH`.

## 9. FP/basis equivalence

PASS. Orthogonal q/k rotation and persistent K-axis state basis preserve the FP recurrence/readout algebra; no value-axis rotation or illegal elementwise commute was found, and all 24 target layers map exactly.

## 10. Dense-Hadamard step0 parity

PASS using the already-existing non-AIME current-formal-path gate. Dense theta=0 and the canonical fast Hadamard path had exact token IDs, logit hashes, and post-QDQ state hashes over prefill plus four decode steps; final max-absolute logit/state differences were 0. No wider tolerance was invented. Scale/qcode arrays were not saved, so those subfields are explicitly `NOT_SAVED`.

## 11. Gradient/trainability

`GRADIENT_RUNTIME_AUDIT = NOT_RUN`: all GPUs were occupied by formal C4 and saved trace tensors are absent. Static code shows identity STE, trainable theta, frozen model parameters, and hard-QDQ forward. Archived training logs record finite nonzero gradients (C3 extremely small; C4 materially larger), but this is not represented as a new independent runtime smoke.

## 12. Training-data leakage

`AIME26_TRAINING_OVERLAP = NO`. The 96-record corpus is locally cached WikiText-2 raw (64 train, 16 validation, 16 heldout). All raw-text hashes recompute, no exact AIME problem hash matches, and no normalized full AIME problem string—including the current last 20—occurs in the corpus. Answers are unused.

## 13. Checkpoint-selection rule

Both checkpoints use `MIN_HELDOUT_LOCAL_LOSS` on the frozen WikiText VALIDATION panel (128 local samples). C3 selected step 100; C4 selected step 900. LR probes selected C3 `3e-3` and C4 `1e-3` before full runs. No AIME selection evidence exists; `CHECKPOINT_SELECTION_LEAKAGE = PASS`.

## 14. Current C3/C4 loader verification

PASS for both. The loader pins checkpoint SHA256, schema, task/model/objective/seed/layers, strictly loads all 24 theta tensors, reconstructs DeltaR exactly once, and applies `H` then `DeltaR`. There is no silent identity fallback, C3/C4 swap, extra transpose, missing H, or double H.

## 15. Gate table

```text
QWEN_DENSE_TRAINING_PROVENANCE_AUDIT = FAIL
DENSE_STATE_LINEAGE = PARTIAL
DENSE_FUNCTIONAL_LINEAGE = PARTIAL
DENSE_STATE_OBJECTIVE_RECONSTRUCTION = PASS
DENSE_FUNCTIONAL_OBJECTIVE_RECONSTRUCTION = PASS
DENSE_STATE_RECURRENT_MODE = FP_STATE_RESET_EACH_TOKEN
DENSE_FUNCTIONAL_RECURRENT_MODE = FP_STATE_RESET_EACH_TOKEN
REAL_RECURRENT_TRAINING_GATE = FAIL
TRAIN_VS_FORMAL_QDQ_MATCH = NUMERICALLY_EQUIVALENT
TRAIN_VS_FORMAL_WRITEBACK_MATCH = FAIL
TRAIN_VS_FORMAL_PRECISION_MATCH = PASS
DENSE_FP_EQUIVALENCE_SEMANTICS = PASS
FAST_H_VS_DENSE_H_STEP0 = PASS
GRADIENT_RUNTIME_AUDIT = NOT_RUN
AIME26_TRAINING_OVERLAP = NO
CHECKPOINT_SELECTION_LEAKAGE = PASS
C3_CURRENT_LOAD_GATE = PASS
C4_CURRENT_LOAD_GATE = PASS
CHECKPOINT_FORMAT_LOADER_MATCH = PASS
```

## 16. C3 validity verdict

`C3_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID`. It is a correctly reconstructed local state-QDQ oracle with an almost-identity learned correction, but it is not verified—and is affirmatively not implemented—as real recurrent INT8 training.

## 17. C4 validity verdict

`C4_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID`. It is a correctly reconstructed local out-proj objective with valid Cayley/STE machinery, but its FP-state one-step injection is not the formal long-horizon writeback process.

## 18. E2E interpretability

`C3_E2E_RESULT_INTERPRETABILITY = CONDITIONAL` and `C4_E2E_RESULT_INTERPRETABILITY = CONDITIONAL`. Runtime differences from C2 do isolate learned DeltaR versus identity DeltaR in the same implementation path, but the checkpoints must be described as local teacher-state rotations, not rotations learned through recurrent quantized rollout.

## 19. Remaining blockers

- Exact training source commit and exact full launch commands are unavailable.
- Optimizer/intermediate checkpoint state and training trace tensors are unavailable.
- A new independent gradient smoke was not run; the missing trace tensors prevent exact replay without regenerating artifacts.
- Current-path parity did not retain per-step scale/qcode arrays, although final state/logit hashes were exact.

| Audit item | C3 Dense-State | C4 Dense-Functional |
|---|---|---|
| Provenance | PARTIAL | PARTIAL |
| Exact objective | PASS: local state | PASS: local out_proj + 0.1 state |
| Orthogonal R | PASS | PASS |
| Correct Key-side basis | PASS | PASS |
| Real recurrent training | FAIL: FP reset/token | FAIL: FP reset/token |
| QDQ matches formal | NUMERICALLY_EQUIVALENT | NUMERICALLY_EQUIVALENT |
| Writeback matches formal | FAIL | FAIL |
| No AIME26 leakage | PASS | PASS |
| Checkpoint selection clean | PASS | PASS |
| Loader correct | PASS | PASS |
| Step0 H parity | PASS | PASS |
| Final validity | CONDITIONALLY_VALID | CONDITIONALLY_VALID |
