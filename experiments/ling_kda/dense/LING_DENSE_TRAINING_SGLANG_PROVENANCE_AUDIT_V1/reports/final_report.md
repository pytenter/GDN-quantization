# LING_DENSE_TRAINING_SGLANG_PROVENANCE_AUDIT_V1

## 1. Executive verdict

`LING_DENSE_TRAINING_SGLANG_AUDIT = PARTIAL`.

The three questions have separate answers:

- **A — old L4/L5 training:** `CONDITIONALLY_VALID`, not fully verified. The checkpoints reconstruct correctly, use corrected Value-side semantics, clean WikiText validation selection, and compatible R128 QDQ. However, training is **FP_STATE_RESET_EACH_TOKEN**, not real recurrent INT8 rollout; L5 is a local attention out-projection objective, not a full-model/logit objective; source lineage is partial; a fresh gradient smoke is deferred.
- **B — L3 versus L2:** matrices are exactly the same normalized H128, but runtime paths are not exact parity. The audited L2 branch itself uses a dense `@ H` despite the “Fast-H” label, while L3 adds an extra `@ I` GEMM. This is `CONTROLLED_NUMERICAL_PATH_DIFFERENCE`, with non-bitwise FP/INT8 states and logits but no axis, transpose, endpoint, loader, or TP semantic error.
- **C — current SGLang bridge:** L3/L4/L5 loader and Value-basis lifecycle pass for the frozen TP=1 runtime. L3 is live on both formal instances with matching H hashes; prior non-AIME gate servers loaded L4/L5 with the expected checkpoint hashes/objectives and all 18 layers.

## 2. Formal runtime snapshot

Commit `d19f68bd74485fc902f3f541bbef8c43a6c4a50a`, patch `806708a37accb71279c3efa90581b229e43398e23cbea2962af9fd67a8930dc3`, model `e3a47d5b986e7141b6efd62597d598ebb392060d`, SGLang 0.5.19, torch 2.9.1+cu128, Triton 3.5.1, FLA 0.5.2. TP=1, two single-GPU instances on physical GPUs 0/1, BF16 model path, deterministic inference, context 262144. Seed is **1**. Per-sample `max_new_tokens` is `262144 - prompt_tokens - 512`, not a fixed 262144.

## 3. Historical Value-side timeline

The corrected endpoint commit (`641d03e…`) predates both trainer commit and training run. The actual trace collector explicitly used native prefix prefill followed by one canonical first-decode step, and its three completion markers record `CORRECTED_PREFILL_ENDPOINT_V2`, zero continuity error, and no redundant endpoint rotation. Therefore both `L4_TRAINED_ON_CORRECTED_VALUE_RUNTIME` and `L5_TRAINED_ON_CORRECTED_VALUE_RUNTIME` are `YES`.

## 4. L4/L5 checkpoint lineage

Both lineages are `PARTIAL`: checkpoint/run/config/data/log hashes and the launch recipe exist, but the trainer had an uncommitted collection fix, the launch/evaluation scripts were untracked, and optimizer/intermediate checkpoints were not retained.

## 5. Rotation parameterization

For row-vector tensors, the actual total matrix is **`R = H128 @ delta`**, not `delta @ H128`. `delta=(I-A)(I+A)^-1`, with strict-upper FP32 theta and antisymmetric A. Theta=0 gives delta=I. Each checkpoint stores 18 independent 8128-element Cayley parameters, one 128×128 matrix per KDA layer shared across all heads. Loader reconstructs H once and does not reinterpret theta as final R.

## 6. Orthogonality

L4 max `|R.T R-I|` = 1.1920929e-06 (median 8.64267349e-07, p95 1.14142895e-06, worst layer 5); L5 max = 8.34465027e-07 (median 6.55651093e-07, p95 7.33137131e-07, worst layer 17). All matrices are finite, 128×128, full-rank by Cayley construction, and have positive determinant. Both gates pass the 1e-4 threshold.

## 7. Exact Dense-State objective

`L4_EXACT_OBJECTIVE` is per-layer relative squared reconstruction error of the native FP-prefix recurrent state after H→delta→BF16→R128-QDQ→BF16→delta.T→H.T. The denominator is the native state squared norm with epsilon 1e-12. It averages batch/head/key rows and then uniformly averages 18 layers. It is not a recurrent rollout loss.

## 8. Exact Dense-Functional objective

`L5_EXACT_OBJECTIVE = LOCAL_BLOCK_OUTPROJ_OBJECTIVE`: one token is replayed through KDA from the locally quantized FP-prefix state; the rotated KDA readout is mapped back before `o_norm`/gate and `o_proj`; relative MSE is computed against the native attention `out_proj` output. Training minimizes this plus 0.1×Dense-State loss, while checkpoint selection uses the local out-proj term alone. It is not a model-hidden or logits objective.

## 9. Recurrent training semantics

Both are `FP_STATE_RESET_EACH_TOKEN`: unroll=1, reset/detach/BPTT interval=1. Quantized state is not written back into the next training sample. `REAL_RECURRENT_TRAINING_GATE = FAIL`; this is a scope limitation, not evidence of wrong local QDQ math.

## 10. Corrected Value-basis lifecycle

Training and current SGLang rotate Value before KDA, retain recurrent state in that basis, and invert before RMSNorm/learned weight/gate/merge/o_proj. The inverse is not pushed through nonlinear operations. L4/L5/current SGLang lifecycle gates pass.

## 11. Train vs formal INT8-R128 semantics

Both use symmetric [-127,127], ties-to-even `torch.round`, `amax/127`, epsilon 1e-12, and one scale per fixed Key row over 128 Value entries. Training canonical `[B,H,K,V]` axis -1 equals SGLang physical `[B,H,V,K]` axis -2. Operator semantics are numerically equivalent at the local state boundary; the material difference is recurrence (FP reset in training versus QDQ writeback in formal runtime).

## 12. Precision policy

FP32 theta/rotation and the explicit BF16→FP32-QDQ boundaries match. Exact internal KDA accumulation dtype was not freshly emitted/re-audited, so precision match is `PARTIAL_MATCH` rather than full.

## 13. L3 matrix = H audit

`PASS`: max abs 0, relative L2 0, identical canonical FP32 hash `6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68`, delta=I on all 18 layers.

## 14. FP Fast-H vs Dense-H parity

Not exact. Across 24 fixed non-AIME forced-token logits, relative L2 median=0.0200579, p95=0.0463101, max=0.0496509; top-1 was 100% in the frozen gate. Across 81 selected state dumps, median=0.0211176, p95=0.0483996, max=0.0565905. Classification: `NUMERICALLY_EQUIVALENT` under the frozen drift policy, not bitwise parity.

## 15. INT8 Fast-H vs Dense-H parity

Across 24 logits, relative L2 median=0.0212791, p95=0.0570389, max=0.0591829. The 36 first-event head-0 QDQ dumps have qcode exact-match median=0.116974, p95=0.645386, range [0.0144043,0.959717]. Divergence compounds recurrently. QDQ operator/axis/timing itself matches.

## 16. First divergence

The first observed divergence is layer 0 at the prefill-end state dump. Exact localization to rotated Value versus KDA core is unavailable because those intermediate tensors were not dumped. The first **confirmed implementation** difference is source-level: L3 adds an identity FP32 GEMM after the common dense H GEMM. A new finer-grained GPU trace is deferred.

## 17. Prefill/decode endpoint audit

`PREFILL_ENDPOINT_DOUBLE_ROTATION = NO`; `PREFILL_DECODE_BASIS_CONTINUITY = PASS`. In FP, kernel return equals cache; in INT8 the expected QDQ changes kernel-return to cache, and the cached tensor then equals first-decode input bitwise. A dedicated new chunk-layout GPU test is deferred, while static chunked `extend` uses the same no-extra-rotation endpoint code.

## 18. SGLang loader audit

L3/L4/L5 load gates pass. Objective, exact layer tuple, missing tensors, and orthogonality have hard errors; there is no silent identity fallback for learned modes. Runtime audit hashes distinguish L4 from L5 and match the frozen checkpoint hashes.

## 19. TP mapping

Pass for formal TP=1: full 128×128 R is unsharded on the only rank and applied along Value dimension 128. This audit makes no claim for TP>1.

## 20. Gradient audit

`DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN`. Historical logs show finite nonzero norms (L4 step1 1.29518617e-06; L5 step1 0.00903241523), zero trainable model parameters, and exact-QDQ-forward/identity-STE-backward, but these are not promoted to a fresh runtime PASS.

## 21. Training-data leakage

No AIME26 identifier or full competition name was found in the 96-row WikiText-2 corpus, and every run manifest says AIME26 unused. L4/L5 overlap = `NO`.

## 22. Checkpoint selection

Both use `MIN_VALIDATION_LOCAL_LOSS`, 21 validation candidates (step 1 and every 50 through 1000), selected step 1000 on 128 WikiText validation samples. Recomputed minima match summaries. Heldout/AIME were not selection data.

## 23. Gate table

| Audit item | L3 H-step0 | L4 Dense-State | L5 Dense-Functional |
|---|---|---|---|
| Checkpoint lineage | N/A | PARTIAL | PARTIAL |
| Exact objective | N/A | VERIFIED local state objective | VERIFIED local out-proj objective |
| Orthogonal R | PASS | PASS | PASS |
| Correct Value-side | PASS | PASS | PASS |
| Correct endpoint | PASS | PASS | PASS |
| R128 semantics | PASS | NUMERICALLY_EQUIVALENT | NUMERICALLY_EQUIVALENT |
| Real recurrence | N/A | FAIL: FP reset | FAIL: FP reset |
| FP parity | NUMERICALLY_EQUIVALENT | runtime bridge PASS | runtime bridge PASS |
| INT8 parity | NUMERICALLY_EQUIVALENT | runtime bridge PASS | runtime bridge PASS |
| TP mapping | PASS (TP=1) | PASS (TP=1) | PASS (TP=1) |
| Loader correct | PASS | PASS | PASS |
| No AIME leakage | N/A | PASS | PASS |
| Final validity | CONDITIONALLY_VALID | CONDITIONALLY_VALID | CONDITIONALLY_VALID |

Overall `LING_DENSE_TRAINING_SGLANG_AUDIT = PARTIAL`.

## 24. L3 validity

`L3_DENSE_HADAMARD_STEP0_VALIDITY = CONDITIONALLY_VALID`: R=H and all semantic gates pass, but the extra identity GEMM makes it a controlled numerical path rather than exact L2 parity, and required intermediate-stage/chunked GPU traces remain deferred.

## 25. L4 validity

`L4_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID`: correct local objective/QDQ/value basis/checkpoint/loader and no leakage, limited by partial source lineage, FP-state reset training, partial precision evidence, and deferred gradient smoke.

## 26. L5 validity

`L5_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID` for the same reasons, with the additional interpretation constraint that the optimized functional target is local attention out-proj error, not full-model/logit preservation.

## 27. E2E interpretation permissions

L3/L4/L5 E2E interpretability is `CONDITIONAL`. L4-vs-L3 and L5-vs-L3 are the primary learned-rotation comparisons within the same two-stage dense implementation. L2-vs-L3 measures implementation-path effect. The observed free-generation difference must not be described as accuracy gain or bug evidence from this audit alone.

## 28. Remaining unknowns

- Fresh gradient smoke, full intermediate tensor parity, and dedicated chunked-prefill parity: `DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN`.
- Exact process roles were not logged for each patch-install pid; TP rank is unambiguous because TP=1.
- Exact KDA internal accumulation dtype was not freshly emitted.
- Uncommitted/untracked training sources prevent full lineage verification.

### Mandatory final values

```text
L4_LINEAGE = PARTIAL
L5_LINEAGE = PARTIAL
L4_TRAINED_ON_CORRECTED_VALUE_RUNTIME = YES
L5_TRAINED_ON_CORRECTED_VALUE_RUNTIME = YES
L4_EXACT_OBJECTIVE = LOCAL_RELATIVE_STATE_QDQ_RECONSTRUCTION
L5_EXACT_OBJECTIVE = LOCAL_BLOCK_OUTPROJ_OBJECTIVE
L4_RECURRENT_TRAINING_MODE = FP_STATE_RESET_EACH_TOKEN
L5_RECURRENT_TRAINING_MODE = FP_STATE_RESET_EACH_TOKEN
L4_TRAIN_VS_FORMAL_QDQ = NUMERICALLY_EQUIVALENT
L5_TRAIN_VS_FORMAL_QDQ = NUMERICALLY_EQUIVALENT
L4_VALUE_BASIS_LIFECYCLE = PASS
L5_VALUE_BASIS_LIFECYCLE = PASS
L3_MATRIX_EQUALS_HADAMARD = PASS
FP_FAST_H_VS_DENSE_H = NUMERICALLY_EQUIVALENT
INT8_FAST_H_VS_DENSE_H = NUMERICALLY_EQUIVALENT
FAST_H_VS_DENSE_H_CLASSIFICATION = CONTROLLED_NUMERICAL_PATH_DIFFERENCE
PREFILL_ENDPOINT_DOUBLE_ROTATION = NO
PREFILL_DECODE_BASIS_CONTINUITY = PASS
TP_ROTATION_MAPPING_GATE = PASS
L3_LOAD_GATE = PASS
L4_LOAD_GATE = PASS
L5_LOAD_GATE = PASS
CHECKPOINT_FORMAT_LOADER_MATCH = PASS
L4_AIME26_OVERLAP = NO
L5_AIME26_OVERLAP = NO
L4_CHECKPOINT_SELECTION_CLEAN = PASS
L5_CHECKPOINT_SELECTION_CLEAN = PASS
L3_DENSE_HADAMARD_STEP0_VALIDITY = CONDITIONALLY_VALID
L4_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID
L5_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID
L3_E2E_INTERPRETABILITY = CONDITIONAL
L4_E2E_INTERPRETABILITY = CONDITIONAL
L5_E2E_INTERPRETABILITY = CONDITIONAL
FAST_H_VS_DENSE_H_E2E_DIFFERENCE_OBSERVED = YES
```
