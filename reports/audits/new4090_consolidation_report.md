# SERVER CONSOLIDATION REPORT

SERVER: new4090 (`nlpg-SYS-4029GP-TRT`; hardware is 8×RTX 3090, despite the server label)

## RUNNING JOBS

- `tmux: ling256_tp1x2_formal` remains active.
- Two SGLang servers and two odd/even Ling INT8_R128 workers use physical GPUs 2 and 3.
- Final read-only snapshot: FP 30/30 records present; INT8_R128 13/30 records present.
- At the final snapshot, GPUs 0, 1, 5, 6, and 7 were idle. GPU 4 had acquired an unrelated process owned by `jtzhou`; the audit did not inspect beyond process metadata or interfere with it. No process was killed, restarted, reconfigured, or moved.

## CANONICAL ASSETS FOUND

- Corrected KDA Value-Hadamard post-fix scientific rebaseline, including code, regression test, semantic contract, compact stage summaries, and formal summary.
- Corrected result: `CORRECTED_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION=NO`, endpoint basis continuity `PASS`, formal status `COMPLETE`, method design ready `YES`.

## SCORER LINEAGE

- V2 FOUND: NO standalone source or regression suite recovered.
- V3 FOUND: NO standalone source or regression suite recovered.
- V4 path: `/data/zypan/stage0_incoming_ling256/aime26_scorer_v4.py` (two additional byte-identical copies exist in the running worktree).
- V4 SHA256: `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`.
- Cross-server match: PASS.
- Version/design strings: `AIME26_STRICT_V4_CANDIDATE`; `RIGHTMOST_COMMITTED_FINAL_CLAIM`.
- Referencing scripts: Ling 256K runner, TP1×2 smoke runner, and experiment configuration files.
- A standalone V4-specific regression test was not found; older diagnostic scorer tests exist.

## ROTATION AND MECHANISM ASSETS

- Learned/trainable rotation implementation: NOT_FOUND.
- Butterfly / ButterflyQuant / HARP: NOT_FOUND.
- Givens / Householder / Cayley / Stiefel parameterizations: NOT_FOUND.
- Same-name Qwen failed prototypes `gdn_rotation_headroom_v1` and `gdn_postconv_structured_rotation_v1`: NOT_FOUND in the source repository, so no semantic-diff claim is made.
- Later canonical Ling asset: KDA Value-Hadamard post-fix rebaseline (2026-09-17), closing the historical redundant prefill-endpoint state rotation bug.
- Useful provenance retained: writeback localization, kernel equivariance, first-divergence localization, commutator chain, decay scalarization, and persistent-error decomposition.
- The later Qwen granularity phase-diagram aggregate is not promoted: manifest gate failed, formal N=0, status `INCOMPLETE_INHERITED_L1_ONLY`.
- No separate new FLA/Triton kernel implementation was found. The kernel-equivalence audit is staged as evidence, not as a deployable kernel patch.

## CROSS_SERVER_KEYS

- `scorer_v4_sha256 = fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`
- `aime26_common_sha256 = 9676ac18472ff662dca04c8a30bc4b5a1b910cc60b775a2cdb698b99e0bff30e`
- `dataset_manifest_sha256 = 5cdda617c7bbf73a0277de75a0f25ed52976e385b01e1264b806085d33649b0f`
- `shared_quantizer_sha256 = NOT_FOUND` (no standalone shared quantizer utility)
- `shared_hadamard_sha256 = NOT_FOUND` (logic is experiment-local; no standalone shared utility)
- `shared_rotation_utils_sha256 = NOT_FOUND` (logic is experiment-local; no standalone shared utility)
- `shared_test_utils_sha256 = NOT_FOUND`

## DUPLICATES AND CONFLICTS

- Frozen V4 is a duplicate by SHA and is intentionally not staged.
- Dataset copy and manifest are intentionally not staged.
- No conflicting V4 version was found.
- Absolute local paths remain in some historical scripts/results as provenance/defaults; they are not credentials and can be overridden where the script exposes environment variables.

## DOC SEMANTICS

- `DOC_SEMANTICS_NEEDS_FIX = NO` for the compact reports selected here.
- No third-server report was found that totals `Correct + Incorrect + Abstain` as three mutually exclusive classes.
- Canonical interpretation remains: `Correct + Incorrect = N`; Abstain is a subset of Incorrect.

## EXCLUSIONS

- Large/raw exclusions include the 635,124,582-byte post-fix horizon CSV, 16,233,574-byte free-running trace, mechanism raw JSONL/CSV traces up to 220,677,497 bytes, generation records, logs, caches, and model weights.
- Sensitive-file scan of candidate staged assets found no credential/private-key/token pattern. No sensitive file is staged.

## VALIDATION

- All selected Python scripts and tests passed `py_compile` in this audit.
- A fresh CPU-only pytest run was not performed because the exact Ling runtime lacks the `pytest` module; no package was installed. Existing source-result evidence records 142, 168, and 208 passing tests for the three localization stages, and `PYTEST=PASS` for the post-fix rebaseline.

## GIT

- Current branch: `codex/server-consolidation-2026-09-21-new4090`.
- Files to add: unique Ling/KDA mechanism and corrected-rotation code/tests/compact evidence, model/environment provenance, inventories, and this report.
- Files to modify: none from repository HEAD; all staged candidates are additions.
- Files not to commit: scorer duplicate, datasets, checkpoints/weights, HF/ModelScope caches, active generation outputs, raw traces, large CSV/JSONL, logs, and incomplete Qwen phase-diagram assets.
- Safe to push: NO. Audit protocol requires review; no commit or push was performed.
