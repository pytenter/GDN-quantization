# Final three-server consolidation

Consolidation date: 2026-09-21. Final branch: `codex/final-consolidation-2026-09-21`, selectively based on `origin/main` at `78b566659776a32183fcc0d6271fe2219bcf5c58`. No server branch was merged wholesale.

## 1. Server map

| Label | Actual host/hardware | Role | Read-only live snapshot |
|---|---|---|---|
| `ling4090` | `lthpc1`, 2 x RTX 4090 | Ling/KDA formal provenance; 256K Value-Hadamard helper | `ling256_4090_helper`, PIDs 2566162/2566163, 10/30 at 2026-09-21T16:07:03+08:00 |
| `qwen3090` | 4 x RTX 3090 | Qwen/GDN formal provenance | all four GPUs idle; no tmux/screen job |
| `new4090` | `nlpg-SYS-4029GP-TRT`, 8 x RTX 3090 | later Ling/KDA post-fix and 256K FP/INT8 | `ling256_tp1x2_formal`, PIDs 3920155/3920156/3921171/3921172 on GPUs 2/3; unrelated user work was untouched |

`new4090` is a historical audit alias, not a hardware description.

## 2. Canonical source map

| Asset | Canonical source |
|---|---|
| Qwen AIME26 81920 code/results | SERVER_B / `qwen3090` |
| Ling AIME26 81920 formal results | SERVER_A / `ling4090` |
| Ling corrected Value-Hadamard code semantics | SERVER_C late post-fix chain; imported SERVER_A copies are byte-identical |
| Frozen V4 scorer | three-server exact match; one shared repository copy |
| Common AIME26 utility | SERVER_B canonical version |
| Ling/KDA mechanism additions | SERVER_C / `new4090` |
| Qwen/GDN mechanism additions | SERVER_B / `qwen3090` |

Audit branches and commits:

- `codex/server-consolidation-2026-09-21-ling4090` at `ba639ccfb0eada0bc3fced2f5ad7eda17756eb4c`.
- `codex/server-consolidation-2026-09-21-qwen3090` at `11580bcb50a12e58f77096703e29b9bfd0a1bf71`.
- `codex/server-consolidation-2026-09-21-new4090` at `0a06faeee7027fe773b354210a378ad395d96f75`.

## 3. Canonical hashes

- Frozen V4 scorer: `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`.
- Dataset manifest: `5cdda617c7bbf73a0277de75a0f25ed52976e385b01e1264b806085d33649b0f`.
- Common utility selected: `34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7`.
- Common utility rejected legacy: `9676ac18472ff662dca04c8a30bc4b5a1b910cc60b775a2cdb698b99e0bff30e`.
- Qwen canonical runner: `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`.
- Ling writeback / kernel-equivariance / first-divergence / post-fix scripts: `7cdec7a9e9ec84166ea31ceddad7674aa9e672db66bffe7bea57eef30a7be65c`, `062a9224ed0f5acd5bdd68a951f03417d8e07214b1104a6bd33a7d7227d64ade`, `3f04f8b9ba089e023d2167998692315dcc005449512a7d6ac263047f26bd323d`, `e8682ad197c291bbe84db3a3096790dcc66b4ae84cbfa2d94c636212cd04154f`.

## 4. Scorer closure

The scorer is `AIME26_STRICT_V4_CANDIDATE` with design `RIGHTMOST_COMMITTED_FINAL_CLAIM`. All three servers match the frozen hash. The repository retains one canonical scorer and a focused regression suite. Gold is not used for candidate selection. `Abstain` is a subset of `Incorrect`.

## 5. Dataset closure

SERVER_A live manifest, SERVER_B staged canonical manifest, and SERVER_C live/inventory manifest all hash to `5cdda617c7bbf73a0277de75a0f25ed52976e385b01e1264b806085d33649b0f`. Dataset content is not committed.

## 6. Qwen AIME26 provenance

Canonical protocol: `GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2`, manual Hugging Face runtime, seeds 1 and 2.

| Condition | Correct | Incorrect | of which Abstain | Accuracy |
|---|---:|---:|---:|---:|
| FP_STATE | 52 | 8 | 3 | 86.67% |
| INT8_C128 | 18 | 42 | 39 | 30.00% |
| INT8_C128 + Key-Hadamard | 41 | 19 | 6 | 68.33% |

Paired Key-Hadamard comparison: rescued 23, lost 0, net +23.

## 7. Ling AIME26 provenance

Canonical formal source: SERVER_A / `ling4090`, 81,920-token protocol, seeds 1 and 2.

| Condition | Correct | Incorrect | of which Abstain | Accuracy |
|---|---:|---:|---:|---:|
| FP_STATE | 44 | 16 | 11 | 73.33% |
| INT8_R128 | 23 | 37 | 28 | 38.33% |
| INT8_R128 + Value-Hadamard | 28 | 32 | 28 | 46.67% |

Paired Value-Hadamard comparison: rescued 7, lost 2, net +5.

## 8. Rotation canonical resolution

The four Ling implementation scripts, four regression tests, and twelve compact post-fix result files compared byte-for-byte between SERVER_A and SERVER_C. They match. Canonical code provenance is assigned to SERVER_C's later post-fix chain; the organized imported copies are from SERVER_A; formal AIME26 result provenance remains SERVER_A. There is no unresolved canonical conflict.

The corrected result records buggy Hadamard AUC `0.12175164`, corrected Hadamard AUC `0.00150127`, causal closure `1.010718`, 18 formal units, and state-error reduction `28.34%`. Formal status is complete and method-design readiness for this fixed correction is yes.

## 9. Duplicate resolution

Twenty Ling rotation artifacts (4 scripts, 4 tests, 12 compact results) were exact cross-server duplicates; only one organized copy of each was retained. Two extra server copies of the exact Frozen V4 scorer were omitted in favor of one shared copy. Runtime-local common-utility mirrors are identical to the selected canonical hash. Count convention: 22 explicitly verified duplicate file copies excluded.

## 10. Legacy exclusions

Legacy families excluded from canonical results are: 65,536-token smoke outputs; V1/V2/V3 and diagnostic scorer outputs; and the `9676ac18...` common utility. Failed Qwen structured-rotation prototypes are retained only under the clearly labelled `prototypes/` path. Count convention: 3 legacy families excluded.

## 11. Invalid-semantics exclusions

The historical KDA double prefill-endpoint state rotation is `INVALID_OLD_ROTATION_SEMANTICS`. It is not present in canonical implementation. It appears only in labelled reinterpretation/bug-provenance evidence. Count convention: 1 invalid semantic family excluded.

## 12. Mechanism evidence map

Qwen compact chains cover orientation, RMSNorm geometry, out-projection anisotropy, recurrent mediation, residual geometry, persistent source error, magnitude equalization, and magnitude-matched residual closure. Ling compact chains cover persistent-error decomposition, commutator, decay scalarization, prefill writeback, kernel equivariance, first divergence, and corrected rotation closure.

Per-chain status is authoritative. Partial/observational chains remain partial/observational; in particular the Ling commutator, decay scalarization, and persistent-error decomposition chains retain `METHOD_DESIGN_READY=NO`.

## 13. Model provenance

- Ling: Hugging Face `inclusionAI/Ling-3.0-tiny`, immutable revision `e3a47d5b986e7141b6efd62597d598ebb392060d`, confidence HIGH, with config/tokenizer/index/modeling/configuration hashes archived.
- Qwen: ModelScope `Qwen/Qwen3.5-9B@master`; immutable revision not recovered, confidence MEDIUM, with config/tokenizer/index fingerprints archived.

## 14. 256K current status

Snapshot timestamp: `2026-09-21T16:07:03+08:00`.

- FP_STATE: 30/30 on SERVER_C, `COMPLETED_GENERATION_PENDING_FINAL_FREEZE`.
- INT8_R128: 13/30 on SERVER_C, `IN_PROGRESS`.
- INT8_R128 + Value-Hadamard: 10/30 on SERVER_A, `IN_PROGRESS`.

No 256K accuracy is asserted and no condition is promoted to `FINAL`.

## 15. Environment provenance

SERVER_A uses 2 x RTX 4090 for Ling/KDA and the 256K helper. SERVER_B uses 4 x RTX 3090 and the Qwen manual Hugging Face path. SERVER_C is actually 8 x RTX 3090; its Ling 256K workers use SGLang/Triton on GPUs 2/3 with YaRN factor 2 and original maximum positions 131072. Full environment notes are in `docs/server_inventory/new4090_environment.md` and the three inventory records.

## 16. Excluded raw artifact manifest

The exact 12 Qwen and 6 Ling formal JSONL paths, sizes, mtimes, hashes, conditions, and protocols are recorded in their `PROVENANCE.json` files. Four additional entries—two growing 256K output roots, one 16,233,574-byte free-running trace, and one 635,124,582-byte horizon CSV—are recorded in `reports/audits/EXCLUDED_RAW_ARTIFACTS.json`. Total explicit entries: 22. No raw response text, dataset, checkpoint, model cache, or large trace is tracked.

## 17. Unresolved NEEDS_REVIEW

- Qwen's immutable upstream ModelScope revision was not recoverable; only `@master` provenance is available.
- The authoritative physical hostname for SERVER_B was not reliably recovered from its shell output.
- All 256K conditions require completion/frozen scoring review before any final claim.
- Historical source repositories were not normalized or cleaned; only isolated audit clones and the clean consolidation clone were committed.
- Partial/observational mechanism chains are not method-design closure.

## 18. Learned-rotation current status

Adaptive/learnable structured state rotation: `NOT YET EXECUTED`. ButterflyQuant: `NOT EXECUTED`. HARP: `NOT EXECUTED`. No Givens, Householder, Cayley, Stiefel, dense trainable orthogonal, Butterfly, or HARP training result is claimed. The next stage is structured / learnable recurrent-state rotation design.

Safety outcome: no running experiment was changed; no server original was deleted; `main` was not modified; no force push was used.
