# Provenance closure

- `PROVENANCE_STATUS = PASS`
- `QWEN_HISTORICAL_DATASET = HuggingFaceH4/MATH-500 test split, fixed seed-20260827 30-item subset`
- `MATCH_MATH500 = true` (30/30 exact; 30/30 normalized)
- `MATCH_AIME24 = false` (0/30 exact; 0/30 normalized)
- `N_EXACT_MATCH = 30`
- Ordered problem-text list SHA256: `1530f65df60fa54318b2f1ea527ca99d3fdb67b506281ac28eebe8d0daddbe4d`

The raw 150-record artifact contains only `MATH-500` rows: five configurations × 30. Recomputed correct counts are FP_STATE 23/30, INT8-row/R128 7/30, INT8-column/C128 18/30, and they match the summary. `AIME24_RESULTS` is empty. The historical AIME24 label was therefore incorrect.

Canonical external/untracked sources were copied byte-for-byte into `reference_snapshot/`. Both server manifests report copied SHA256 equal to original SHA256. The first commit on each branch is provenance-only: Server A `5eb3cd50...`; Server B `7fd931e414...`.

Server A runs Git 1.8.3.1, which predates `git worktree`; it uses a shared-object isolated local clone at `/data/zypan/worktrees/aime26-sglang-rotation-v1`. This is explicitly recorded as an infrastructure deviation; the original working tree was not modified. Server B uses native `git worktree`.

Storage audit: `QWEN_SGLANG_STORAGE_CANDIDATE = NONE`. No persistent writable mount has more than 80 GiB free (`/data`: 23.6 GB; `/home`: 63.5 GB and no writable `/home/zypan`; `/scratch`/`/work` missing). `/dev/shm` has capacity but is volatile tmpfs and rejected. No cache, environment, model, or result was deleted.
