# L7 objective redundancy and persistent validation

Experiment: `LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1`.
This independent directory contains no replacement historical canonical, training,
optimizer update, new rotation, AIME, free generation, or Qwen mutation.

## Ordered gates

1. Recover the original trainer, configuration, and manifests; reuse its exact loss helpers by AST, without importing a training entry point.
2. CPU fixed-input formula/call-site checks. GPU matched-window forward parity and actual next-kernel QDQ consumption are separate, mandatory gates.
3. Two historical documents × all 16 frozen natural rotations; starts 256/1024, H128, all 18 KDA layers and 16 heads. The document–GPU map remains 37702→GPU6 and 5396→GPU3.
4. Report per-window and equal-window aggregate original Functional/State/Total, P1, P2_W128, P3_H128, and B0; all lower-is-safer. Keep all pairwise ranks and gaps, not only averaged rho.
5. Analyze old pulse/no-further-QDQ targets only as discovery. Four anchors are two documents, not four independent samples. No old/new runtime merge eligibility claim.
6. Select Route A if an opposite P1/Total ranking pair with both midrank gaps ≥3 exists in the historical panels; otherwise Route B if an opposite Total/B0 pair meets that same criterion; otherwise Route C. No new rho cutoff or post-outcome threshold. Redundancy and incremental association are separately reported.
7. Only an informative route admits independent-document selection/calibration. Exclude all previously used documents, including the currently active persistent-surrogate experiment's 48693/51410. Use deterministic hash/seed ordering, no target access. Freeze one eligible pair per new document using the user's gap≥3/min-gap/tie-break rule; do not relax it.
8. Full-model every-token INT8-R128/QDQ/writeback qualification, same-GPU FP references, microbenchmark, continuity/storage checks, total conservative budget ≤16 GPU-hours; only then four 32768-token units. Anchor does not disable QDQ.
9. Fixed primary mean KL(FP‖student) on [8192,32768); secondary [2048,32768) from the same trajectory. Actual outcomes cannot alter comparator, selection, or primary window.

## Definition versus training-distribution fidelity

The recovered helpers operate on actual post-o_proj outputs and post-QDQ native-coordinate states from matched full-model deployment trajectories. This is a faithful application of the historical loss *mathematical definition* to the requested windows; it is not a replay of the original 512-token WikiText2 FP-driver-conditioned local training rollouts. The old trainer kept rotated BF16-boundary/dequant states and shared FP q/k/v/gates; the frozen deployed scorer keeps physical native FP32 post-INT8-QDQ caches and actual student upstream drivers. Neither difference will be hidden or silently changed to improve correlations.

The previous P1 CSV lacks persisted FP intermediate tensor hashes. New L7 and P1/B0/P2/P3 will therefore be measured on one fresh, shared FP reference and checked against prior scores. All matched metrics use the newly recomputed set if reference equivalence cannot be established; no partial joining across unmatched inputs.

## Cost and scheduling

Total hard cap: 16 GPU-hours, summed across UUIDs. Phase A target cap: 2 GPU-hours including smoke/reference/loading. Waiting on a GPU consumes no experiment GPU-hours. GPU3 UUID `GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`; GPU6 UUID `GPU-de53f243-b271-521c-d875-d96611ed8574`. Verify UUID, idle process/device state, host maintenance, source hashes, and budget immediately before any launch. No sharing, signals, killing, migration, or use of other GPUs.

At preparation, both authorized devices are occupied by another persistent experiment. No new GPU measurement is launched. A free device is necessary for Phase A3 matched-forward parity before any formal scores. No automatic restart after science/runtime/hardware/parity failures. Result files are emitted only for actually executed stages; awaiting hardware is not a scientific failure or a completed Phase A.

Phase C is a proposal only. No temporal causal measurement or training is authorized in this run.
