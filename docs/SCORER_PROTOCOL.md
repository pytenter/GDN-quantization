# AIME26 Scorer Protocol

## Frozen scorer

The archived 81,920-token final comparison uses:

```text
name = AIME26_STRICT_V4_CANDIDATE
sha256 = fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b
path = experiments/shared/scoring/aime26_scorer_v4.py
```

Scores in `results/aime26/81920/ling_kda/` were derived by running this scorer over completed formal JSONL response records. The response text is intentionally not committed. `PROVENANCE.json` records each source file path, byte size, modification time, and SHA256.

## Counting rules

- `correct`: extracted/accepted final answer equals ground truth.
- `wrong`: every non-correct sample, including extraction abstentions.
- `abstain`: scorer did not accept a final answer extraction; this is a subset of `wrong`.
- accuracy denominator: all 60 samples per condition.
- matched comparisons pair the same question and seed.

## Version boundary

V1, V2, V3, diagnostic extractors, and development corpora are historical or audit assets. They must not silently rescore or overwrite Frozen V4 results. Any future scorer change requires a new versioned output directory and cross-audit.

## 256K boundary

The active Ling 256K helper references the same V4 scorer implementation, but its growing outputs are not final and are not archived or declared as final accuracy in this consolidation.
