# GDN/KDA Recurrent-State Quantization

This repository studies INT8 quantization of recurrent states in two model families:

- Qwen3.5-9B / Gated DeltaNet (GDN)
- Ling-3.0-tiny / Kimi Delta Attention (KDA)

The current scope is recurrent-state quantization only. Model-weight quantization and the historical INT4 line are not current project claims.

## Main result lines

The two architectures reverse the favorable grouping/rotation orientation:

| Model | Native INT8 condition | Rotation condition |
|---|---|---|
| Qwen/GDN | `INT8_C128` | Key-side Hadamard |
| Ling/KDA | `INT8_R128` | Value-side Hadamard |

For Ling/KDA, the corrected Value-Hadamard semantics keep the recurrent state in rotated Value coordinates across kernel return, cache write, and first decode consumption. The prefill endpoint must not rotate that state again.

## AIME26 Frozen V4 at 81,920 tokens

Each condition contains 30 questions x 2 seeds = 60 samples. Accuracy uses the frozen `AIME26_STRICT_V4_CANDIDATE` scorer.

| Model | Condition | Correct | Accuracy |
|---|---|---:|---:|
| Qwen/GDN | FP_STATE | 52/60 | 86.67% |
| Qwen/GDN | INT8_C128 | 18/60 | 30.00% |
| Qwen/GDN | INT8_C128 + Key-Hadamard | 41/60 | 68.33% |
| Ling/KDA | FP_STATE | 44/60 | 73.33% |
| Ling/KDA | INT8_R128 | 23/60 | 38.33% |
| Ling/KDA | INT8_R128 + Value-Hadamard | 28/60 | 46.67% |

Matched question/seed comparison:

- Qwen Key-Hadamard vs INT8_C128: 23 rescued, 0 lost, net +23 (+38.33 percentage points).
- Ling Value-Hadamard vs INT8_R128: 7 rescued, 2 lost, net +5 (+8.34 percentage points after display rounding).
- Relative to FP_STATE, Qwen Hadamard loses 11 correct samples and Ling Hadamard loses 16.

This server-consolidation branch contains raw-derived compact Ling evidence and source hashes. Qwen final evidence must be cross-checked against the qwen3090 server inventory before the branch is publishable.

## Current status

- AIME26 81,920-token generation is complete and archived as compact evidence.
- Ling/KDA post-fix Value-Hadamard rebaseline is complete; the historical redundant prefill-endpoint rotation path is invalid for current comparisons.
- A Ling 256K length-sensitivity helper run is in progress on `ling4090`. It is not a final result and is not scored here as final.
- Historical mechanism experiments remain valuable, but failed smokes, pre-metric-fix runs, and pre-fix rotation artifacts are retained as legacy rather than merged into canonical results.

## Repository layout

```text
experiments/   Canonical scripts, launchers, scorers, and tests
results/       Compact results and provenance; no large response text
reports/       Scientific and audit reports
docs/          Protocol, status, experiment index, and server inventories
```

Model checkpoints, caches, complete dataset copies, raw token/layer/head traces, generation logs, and large response outputs are intentionally excluded.

## Reproducibility boundary

Scripts retain environment-specific assumptions and require local model/data paths. Compact results include hashes and provenance where source artifacts are not committed. See `docs/PROTOCOL.md`, `docs/SCORER_PROTOCOL.md`, and `docs/server_inventory/` before reproducing or comparing runs.
