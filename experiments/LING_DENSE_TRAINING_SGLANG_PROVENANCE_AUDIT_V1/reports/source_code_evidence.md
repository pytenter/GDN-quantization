# Source-code evidence

This audit is result-blind: AIME correctness was not used in any gate.

| Claim | Evidence |
|---|---|
| Trainer worktree | `/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling` at `506071224db4d091c398a0fc96b0c58c67b296f4`; dirty/untracked provenance is recorded rather than hidden. |
| Corrected endpoint commit | `641d03e0224d6733039a027295ad3ab5f61d5a7b`, 2026-09-17 18:08:26 +08:00. |
| Training collection | `/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling/experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_ling_dense_oracle.py:119-168`; independent native prefix, canonical first decode, zero endpoint rotation. |
| Rotation order | trainer lines 171-194 and runtime patch lines 160-210, 265-299: row-vector `H @ delta`, inverse `delta.T @ H.T`. |
| Dense-State objective | trainer lines 184-194 and 262-279. |
| Dense-Functional objective | trainer lines 262-279; comparison is local attention `o_proj` output, not logits/full model. |
| Cayley and QDQ | `/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling/experiments/shared/rotation/cayley_rotation.py:32-76,140-170`. |
| Runtime loader guards | `/data01/user2/worktrees/ling-sglang-dense-aime26-last20-256k-singleseed-v1/experiments/aime26/sglang_kda_runtime_patch.py:133-226`. |
| Runtime placement | same patch lines 265-299, 338-430, 449-488. |
| L2/L3 implementation difference | same patch lines 265-280: L2 executes `source @ H`; L3 executes `(source @ H) @ I`. |
| Runtime QDQ | same patch lines 344-390: physical axis -2 corresponds to canonical Value axis -1. |
| Checkpoint hashes | L4 `dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318`; L5 `65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c`. |

Important provenance limitation: the exact trainer used for trace collection contains an uncommitted fix and the launch/evaluation scripts are untracked. Their hashes and timestamps are frozen in this audit, but this prevents `VERIFIED` lineage.
