# NEXT_SINGLE_FACTOR_EXPERIMENT

## Decision

Modify exactly one primary factor: replace the local GDN State/out-projection training surrogate with a persistent downstream final-hidden representation fidelity objective.

This is a proposal only. No training, C7 run, new AIME generation, or additional screen was started.

## Evidence

- C5 u113 dominates canonical Hadamard on both frozen shared objectives, yet its independent behavior is split- and metric-dependent.
- C5 u63 reproduced a State advantage on CHECK (4/4 documents) but its downstream final-hidden advantage reversed from 4/4 on SCREEN to only 1/4 on CHECK.
- Fixed-reference NLL and FP-reference margin often disagree with State, Functional, hidden, and KL metrics.
- C6 u119/u125 show large document-specific downstream outliers. Therefore changing checkpoint selection alone is not justified.

## Frozen factors

Hold fixed: model, canonical C128 QDQ/writeback, H64 persistent recurrence semantics, rotation parameterization and initialization, training corpus and ordering, optimizer, learning rate, exposure count, seed, precision, and evaluation panel. Do not add a weighted multi-metric score.

## Expected effect

The objective should make improvements survive beyond local GDN State/out-projection error into the full-model representation while retaining Hadamard-like stability across documents.

## Falsification criterion

Reject the objective change if a single preregistered pilot fails either condition on the already frozen independent panel:

1. the final-hidden error improvement versus Hadamard has the same direction on both SCREEN and CHECK with at least 3/4 documents improved in each split; or
2. the maximum positive final-hidden regression is not lower than the current candidates' corresponding split-specific maximum.

NLL, KL, or margin disagreement must be reported separately and must not be hidden in an aggregate score.

## Compute

Estimate: at most 4 GPU-hours for one small H64 pilot plus the already frozen non-AIME teacher-forced validation. This estimate is prospective and is not part of the current task's 0.9907508527570301 GPU-hours.

## Why this factor

The shared objectives were measurably optimized, so exposure or generic optimizer tuning is less directly motivated. The first repeatable boundary where the SCREEN hypothesis failed was the persistent downstream final-hidden representation. A downstream objective is therefore a narrower, more falsifiable intervention than a checkpoint sweep, data expansion, layer rollback, or post-hoc metric combination.
