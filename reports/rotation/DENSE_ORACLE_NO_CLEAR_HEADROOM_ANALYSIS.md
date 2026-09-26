# DENSE_ORACLE_NO_CLEAR_HEADROOM_ANALYSIS

Neither model met the preregistered `CLEAR_HEADROOM` criterion for layer-shared dense orthogonal corrections.

| Model | Local | Persistent statistical result | Protocol classification |
|---|---|---|---|
| Qwen3.5-9B/GDN | PRESENT | PROMISING_HEADROOM; 11.45% paired median reduction, CI crosses 0 | PROMISING_HEADROOM |
| Ling-3.0-tiny/KDA | PRESENT | Median effect <=0 and CI crosses 0 | OBJECTIVE_TRANSFER_GAP |

## A. Hadamard may already be near the layer-shared optimum

This is not established. Both models show held-out local improvements over Hadamard, and both have a learned variant with a lower mean future-KL AUC. The uncertainty or paired-document inconsistency prevents a persistent claim, but the local results argue against declaring the layer-shared search space exhausted.

`NO_CLEAR_LAYER_SHARED_DENSE_HEADROOM = YES` under the strict persistent criterion.

`NO_ORTHOGONAL_HEADROOM = NOT_ESTABLISHED`.

## B. The local objective may not be recurrent-aware enough

This is the strongest explanation supported by the current evidence. Ling reduces local state/out-proj error substantially, yet its paired median long-horizon effect is slightly negative. Qwen transfers directionally on average and in the 512-token stress, but its 16-document confidence interval crosses zero. Local single-step replay losses therefore do not reliably rank long-history trajectories.

## C. Per-head capacity remains untested

`PER_HEAD_ORACLE_NOT_TESTED = YES` for both models. A negative layer-shared result cannot exclude useful head-specific rotations. However, increasing capacity before fixing objective transfer risks fitting the same local surrogate more strongly without improving recurrent fidelity.

## D. Calibration-distribution uncertainty remains

The experiment uses a clean, frozen, non-AIME generic WikiText-2 panel with 64/16/16 documents and no split overlap. It is scientifically valid for this V1 question, but 16 held-out documents leave wide paired intervals and visible heavy-tailed KL behavior. A larger independent generic-text panel would improve precision without touching AIME26.

## Recommendation

1. Develop a future-aware/recurrent-aware objective that directly optimizes short unrolled future-KL or recurrent trajectory error while keeping model weights frozen.
2. Validate it on a larger non-AIME held-out panel with the same strict paired bootstrap.
3. Only if transfer improves, run a per-head dense oracle to separate objective limitations from layer-shared capacity limitations.
4. Do not start Butterfly, HARP, or another structured-rotation comparison from the current evidence.

Final cross-model fields:

- `HADAMARD_NEAR_LAYER_SHARED_OPTIMUM_QWEN = INCONCLUSIVE`
- `HADAMARD_NEAR_LAYER_SHARED_OPTIMUM_LING = INCONCLUSIVE`
- `STRUCTURED_ROTATION_RESEARCH_JUSTIFIED = INCONCLUSIVE`
- `SAFE_TO_START_BUTTERFLY_HARP_COMPARISON = NO`
