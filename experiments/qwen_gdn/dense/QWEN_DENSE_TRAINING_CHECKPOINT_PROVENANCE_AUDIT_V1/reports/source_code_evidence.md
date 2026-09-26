# Source-code evidence

This audit used implementation and artifact evidence only. It did not read AIME responses or correctness labels.

## Training trace and objective

- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:117-155`: captures detached q/k/v/g/beta, FP32 `state_input`, local norm/gate and `out_proj_output`.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:210-251`: runs an FP teacher cache to isolated sampled token positions.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:258-276`: applies `(H@Delta)^T` to state, exact C128 QDQ/STE, then `H@Delta` recovery.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:293-316`: one local recurrence and local RMSNorm/gate/`out_proj` replay.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:384-401`: exact losses; Functional is local out-proj relative MSE plus `0.1 * state_loss`.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py:419-461`: random one-trace optimization and minimum validation-primary checkpoint selection.

## Cayley and QDQ

- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py:47-64`: skew construction and `Delta=(I-A)(I+A)^-1` via `solve`.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py:140-165`: symmetric C128, K-axis `-2`, FP32, `round`, `[-127,127]`, and identity STE.

## Formal runtime

- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1/run_dense_eval.py:103-148`: strict hash/schema/objective/layer checks and theta loading.
- `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1/run_dense_eval.py:151-165`: current q/k path is `q/k @ H128 @ DeltaR`.
- `/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py:151-167`: exact cache-state C128 QDQ and in-place writeback.
- `/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py:289-310`: formal prefill, decode, then post-decode writeback for the next token.

## Provenance limitation

The current training runner SHA256 is `943856325333b63f3f15827159a3b42dee1bccafdbaa2816c8b12848d7e9a553` and has an uncommitted FP32 `state_input` fix relative to local HEAD `fa7918d82c144a30ba3459114abb53e70df73d98`. Its mtime precedes trace collection and checkpoint creation, and the manifest/log schema agrees with this variant, but the checkpoint does not embed the source hash. The report's claimed implementation tip `0b6d598` is not present in the local object database. Therefore both lineages are PARTIAL, not VERIFIED.
