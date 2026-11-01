# LING L6/L7 paired trajectory closure

## Purpose

This directory is the frozen, read-only closure analysis for the 20 paired AIME26 questions used to compare FP, Hadamard, L6, and L7. It archives the saved evidence; it does not retrain, regenerate, rescore, or reinterpret the experiment.

## Inputs and conditions

- 20 paired questions, AIME26 q11-q30
- 80 condition outcomes: FP, Hadamard, L6, L7
- seed 1
- Ling-3.0-tiny, frozen 262144-token context configuration
- Frozen V4 scorer: AIME26_STRICT_V4_CANDIDATE
- L6/L7 use the frozen recurrent INT8-R128 configuration and final rotations
- Upstream canonical artifacts: `experiments/LING_RECURRENT_DENSE_L6_L7_V1/`

## Canonical result

| Condition | Correct |
|---|---:|
| FP | 14/20 |
| Hadamard | 9/20 |
| L6 | 9/20 |
| L7 | 10/20 |

The 80/80 saved scorer comparisons are consistent with Frozen V4.

## Paired switches

- L6 vs Hadamard: rescue q23; regress q26; net 0; Holm p = 1.0.
- L7 vs Hadamard: rescue q11 and q12; regress q26; net +1; Holm p = 1.0.
- L7 vs L6: rescue q11 and q12; regress q23; net +1; exact p = 1.0.

For L7 versus Hadamard, q11 and q12 change from Hadamard length-limit failures to shorter normal-stop correct outputs. q26 changes from a correct Hadamard normal-stop output to an L7 length-limit abstention. The q26 chain--trajectory instability, no stable final commitment, continued generation, length limit, abstain--is descriptive evidence and is not a demonstrated internal causal mechanism.

L7 does not globally shorten generation. Relative to Hadamard, the mean length delta is approximately +754 tokens and the median is approximately +1261 tokens; the interval is wide and there is no globally consistent repetition improvement.

## Frozen interpretation

```text
PROVENANCE_GATE = PASS
SCORER_ARTIFACT_EXPLANATION = NOT_SUPPORTED
L7_STRUCTURED_TRAJECTORY_SIGNAL = SUPPORTIVE
L6_STRUCTURED_TRAJECTORY_SIGNAL = INCONCLUSIVE
CURRENT_RESULT_INTERPRETATION = STRUCTURED_BUT_UNDERPOWERED
NEXT_ACTION = CANONICAL_60_EVALUATION
```

This archive does not claim that L7 beats Hadamard.

## Code entry points

- `scripts/extract_baselines.py`: exact read-only FP/H baseline extraction source.
- `scripts/run_analysis.py`: closure reconstruction, paired statistics, trajectory diagnostics, switch forensics, rotation checks, and integrity inventory.
- `provenance/code_snapshots/analyze_long_generations.py`: exact established metric-source snapshot.
- `provenance/CODE_PROVENANCE.md`: all evidenced project-owned training, runtime, parity, evaluation, scoring, finalization, and analysis code.

## Artifact layout

- `analysis/per_sample_outcomes.csv`: 80 canonical outcome rows.
- `analysis/trajectory_metrics.csv`: 80 trajectory rows.
- `analysis/pairwise_deltas.csv`: 20 paired-delta rows.
- `analysis/statistics.json`: paired tests and corrected p-values.
- `analysis/rotation_sanity.json`: L6/L7 finite and orthogonality checks.
- `analysis/switch_case_forensics.md`: q11/q12/q26 descriptive evidence.
- `analysis/final_report.md`: frozen final report.
- `ARCHIVE_MANIFEST.json`: archive inventory and scientific status.
- `provenance/LARGE_ARTIFACT_MANIFEST.json`: excluded raw/binary artifacts and hashes.

## Integrity verification

From this directory:

```bash
sha256sum -c SHA256SUMS
sha256sum SHA256SUMS
sha256sum -c provenance/ARCHIVE_SHA256SUMS
```

The frozen `SHA256SUMS` file itself must hash to `42b6831fb46410e8cd1c8d5a08232408ac78a38094ae0e73788d803d954dab49`. It is intentionally unchanged. `provenance/ARCHIVE_SHA256SUMS` covers the archival additions.

## Upstream dependencies

The frozen provenance records SGLang 0.5.19 at source commit `0bcd822377da7b5718e674eaf9c870d349424dd1`, PyTorch 2.9.1+cu128, Triton 3.5.1, and FLA 0.5.2. Project-owned modifications and launch glue are archived in the upstream experiment and enumerated in the code-provenance map.
