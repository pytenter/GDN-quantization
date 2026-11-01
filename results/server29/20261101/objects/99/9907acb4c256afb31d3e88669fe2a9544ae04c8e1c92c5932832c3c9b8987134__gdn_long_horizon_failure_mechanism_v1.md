# GDN_LONG_HORIZON_FAILURE_MECHANISM_CAUSAL_AUDIT_V1

- FINAL_CLASSIFICATION: `LONG_HORIZON_MECHANISM_INCONCLUSIVE`
- METHOD_DESIGN_READY: `NO`
- MECHANISM_CLOSURE_STATUS: `OFFLINE_FORENSIC_PARTIAL; GPU_CAUSAL_INTERVENTIONS_NOT_STARTED`
- METRIC_GATE: `BLOCKED_FOR_CAUSAL_SWAMPING_CLAIMS`
- OLD_LOST_UPDATE_RATIO_ROOT_CAUSE: `METRIC_NAMING_ERROR`

## Metric Audit

Previous lost_update_energy_ratio is update reconstruction error ratio.
It is not a bounded same-grid lost-update fraction.
Old metric count: 26749
Old metric > 1 count: 4016
Old metric max: 15.824380963997399
Corrected bounded metrics are not recomputable from prior records.

## Failure Forensics

INT4-column failures: 12
L25 H14/H15 failures: 11
L25 H14/H15 fraction: 0.9166666666666666
Failure-aligned rows: 120
Exact aligned rows: 0

## Causal Status

L25 H14/H15 is strong correlational evidence, not causal closure.
Rescue, rounding, scale, and residual interventions were not run offline.
