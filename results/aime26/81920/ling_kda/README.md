# Ling/KDA AIME26 81,920 archive

This directory contains compact evidence derived from completed formal generation files on `ling4090`.

- 30 questions x 2 seeds = 60 samples per condition.
- Frozen scorer: `AIME26_STRICT_V4_CANDIDATE`.
- `frozen_v4_per_sample.csv` contains correctness/extraction metadata but no response text.
- `frozen_v4_summary.json` contains condition and matched-pair summaries.
- `PROVENANCE.json` records the immutable source JSONL paths, sizes, mtimes, and SHA256 hashes.
- `evidence/` contains small preflight, replay, integrity, and gate records.

The six raw generation JSONLs total about 69 MiB and include response text; they are intentionally not committed.
