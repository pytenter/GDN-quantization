# GDN INT8 R128/C128 Future-Key Addressability Causal V1

## 1. Scientific Question
Can future-key addressability be causally manipulated while preserving residual norm, frozen state persistence, value profile, and layer/head budgets?

## 2. Prior Evidence
Frozen path decomposition found FUTURE_KEY_INTERACTION_SIGNAL.

## 3. Why Correlation Is Insufficient
This task stopped before behavioral KL because the preregistered construction gate failed.

## 4. Causal Intervention Design
single-pulse same-norm key-addressability causal intervention; stopped before behavioral pilot at construction gate

## 5. Frozen Key-Risk Operator
{
  "J_key": "sum_t ||k_t^T E_t||_2^2 under frozen FP recurrent drivers",
  "J_state": "sum_t ||E_t||_F^2 under frozen FP recurrent drivers",
  "construction": "per layer/head orthogonal redistribution in key dimension with per-Value-column norm preservation",
  "neutral_control": "deterministic orthogonal-complement redistribution with J_key target kept near base when feasible"
}

## 6. Constraint Definitions
Norm <=1e-4, J_state drift <=5%, J_key change >=25% in the correct direction, value-profile cosine >=0.99, neutral J_key change <=5%.

## 7. Intervention Construction
Per layer/head key-dimension orthogonal redistribution with per-Value-column norm preservation.

## 8. Neutral Control
Neutral controls remained near the baseline J_key.

## 9. Stage-0 Gates
```json
{
  "FROZEN_TRAJECTORY_IDENTITY_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "INTERVENTION_CONSTRUCTION_GATE": "FAIL",
  "KEY_MANIPULATION_GATE": "FAIL",
  "KEY_RISK_DECOMPOSITION_GATE": "PASS",
  "NEUTRAL_CONTROL_GATE": "PASS",
  "NORM_CONTROL_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "RISK_OPERATOR_IDENTITY_GATE": "PASS",
  "SAME_NORM_BASELINE_GATE": "PASS",
  "SINGLE_PULSE_GATE": "PASS",
  "STATE_PERSISTENCE_CONTROL_GATE": "PASS",
  "VALUE_PROFILE_CONTROL_GATE": "PASS"
}
```

## 10. Construction Feasibility
FAIL; valid construction units 0 / 9.

## 11. R-Key Suppression Results
R_KEY_SUPPRESSED produced negligible median J_key change, so it did not satisfy the manipulation gate.

## 12. C-Key Enhancement Results
C_KEY_ENHANCED also produced negligible median J_key change, despite acceptable state and value-profile controls.

## 13. State-Persistence Control
State persistence was effectively preserved; this was not the blocking failure.

## 14. Behavioral KL
Not run because construction feasibility failed.

## 15. Targeted vs Neutral Effect
Not run.

## 16. Layer / Head Analysis
Per-layer/head construction quality is stored in the checkpoint JSON.

## 17. Per-Prompt / Per-t0 Results
| Prompt | t0 | Unit Valid | R J_key Ratio | C J_key Ratio | R J_state Ratio | C J_state Ratio |
|---|---:|---|---:|---:|---:|---:|
| test/algebra/1332.json | 64 | False | 0.9999999997351968 | 0.999755436693304 | 1.0000000000613418 | 0.9997360997013752 |
| test/algebra/1332.json | 128 | False | 1.0000000000885327 | 1.0005520281243754 | 1.0000000003745377 | 1.00029069123455 |
| test/algebra/1332.json | 256 | False | 1.0000000001797702 | 1.0008046172043767 | 0.9999999993510496 | 1.00045297004653 |
| test/counting_and_probability/119.json | 64 | False | 0.9999999997286613 | 0.9995041213933646 | 1.0000000000938714 | 0.9995907666386075 |
| test/counting_and_probability/119.json | 128 | False | 1.0000000002687024 | 1.0000510000821323 | 0.9999999999015293 | 1.0000948517110473 |
| test/counting_and_probability/119.json | 256 | False | 0.9999999997171616 | 0.9997592143058286 | 1.0000000000615517 | 0.9997951231713212 |
| test/geometry/477.json | 64 | False | 0.9999999999890523 | 1.000387183673608 | 0.9999999999161657 | 1.0002455433495103 |
| test/geometry/477.json | 128 | False | 0.9999999996711793 | 1.0008633062081556 | 1.000000000135738 | 1.0013709846931387 |
| test/geometry/477.json | 256 | False | 1.000000000227924 | 1.0000030872328196 | 1.0000000003191285 | 1.0007097189036471 |

## 18. Pilot Classification
`CONSTRUCTION_NOT_CLEAN_ENOUGH`

## 19. What Is Supported
The attempted clean manipulation could not change J_key enough under the registered controls.

## 20. What Is NOT Supported
- FUTURE_KEY_ADDRESSABILITY_CAUSAL_CONTRIBUTOR is not tested by behavioral KL in this run.
- KEY_PATH_NOT_CAUSAL is not supported, because the key manipulation gate failed before behavioral pilot.
- METHOD_DESIGN_READY remains NO.

## 21. Negative / Corrective Results
The straightforward key-subspace redistribution was too constrained or ineffective for this residual/operator geometry.

## 22. Next Recommended Experiment
revise intervention construction without relaxing controls
