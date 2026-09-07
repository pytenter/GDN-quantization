# Efficiency Audit

```json
{
  "DYNAMIC_SELECTOR": "NO",
  "EXTRA_FULL_STATE_PASS": 0,
  "FUNCTIONAL_REFINEMENT_COST_BENEFIT_WEAK": "YES",
  "ONLINE_FUNCTIONAL_SCORING": "NO",
  "ONLINE_MAGNITUDE_SCORING": "NO",
  "ONLINE_RANKING": "NO",
  "STATIC_BIT_ASSIGNMENT": "YES",
  "STATIC_PACKING_FEASIBLE": "YES_CONDITIONAL",
  "extra_calibration_FLOPs": "offline only; roughly one q^T residual contraction per layer/head/value column/calibration token",
  "extra_calibration_memory": "selector_scores.csv plus M5 diagonal tensor reads; no inference memory increase",
  "magnitude_only_calibration_cost": "state column squared norms from calibration trajectory",
  "magnitude_plus_functional_calibration_cost": "adds q-visible C128 residual contraction and diagonal G weighting from existing M5 artifacts"
}
```
