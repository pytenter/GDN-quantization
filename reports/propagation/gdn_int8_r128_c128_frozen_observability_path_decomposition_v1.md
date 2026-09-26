# GDN INT8 R128/C128 Frozen Observability Path Decomposition V1

## 1. Scientific Question
Where does same-norm R-structure extra behavioral damage first appear along the frozen FP path?

## 2. Prior Causal Evidence
R128 repeated accumulation is formal-supported, and natural residual norm-swap remains no-clear-decomposition.

## 3. Current Contradiction
Same-norm R structure produces larger future KL, but one-step normalized core readout was lower for R in the previous audit.

## 4. Competing Path Hypotheses
Pure recurrence, future-key interaction, future-query observability, local output path, feedback-required, mixed, or no clear path.

## 5. Protocol
SAME-NORM STRUCTURE-ONLY frozen FP trajectory replay; no cadence; no repeated quantization; no method design

## 6. Same-Norm Residual Pairs
Primary: R_STRUCT_C_NORM vs REAL_C. Secondary: REAL_R vs C_STRUCT_R_NORM.

## 7. FP Driver Capture
Drivers are captured only from the pure FP teacher-forced trajectory.

## 8. Source Semantics Audit
See JSON `source_semantics`.

## 9. Replay Identity Gates
```json
{
  "FP_DRIVER_CAPTURE_GATE": "PASS",
  "FROZEN_MEMORY_READ_IDENTITY_GATE": "PASS",
  "FROZEN_NEXT_STATE_IDENTITY_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "LOCAL_OUTPUT_REPLAY_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "READOUT_IDENTITY_GATE": "PASS",
  "SAME_NORM_REPRODUCTION_GATE": "PASS",
  "SOURCE_SEMANTICS_GATE": "PASS"
}
```

## 10. Pure Recurrent-State Propagation
Primary median R/C J_state ratio: 0.6664534169081335

## 11. Future-Key Interaction
Primary median R/C J_key ratio: 2.473200378223244

## 12. Future-Query Observability
Primary median R/C J_query ratio: 1.291245838823943

## 13. Observability Crossover
Primary crossover units: 9 / 9

## 14. GDN Local Output Path
Primary median R/C J_local ratio: 0.7682056171717685

## 15. Full-Model KL Reference
Primary R full KL > C: 9 / 9

## 16. Path Waterfall
| Prompt | t0 | Initial Norm R/C | State R/C | Key R/C | Query R/C | Local Output R/C | Full KL R/C | Crossover Offset | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| test/algebra/1332.json | 64 | 1 | 0.651414 | 2.93926 | 1.17481 | 0.95508 | 2.21409 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/algebra/1332.json | 128 | 1 | 0.666453 | 2.43224 | 1.49477 | 0.956255 | 1.70424 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/algebra/1332.json | 256 | 1 | 0.79472 | 2.46598 | 1.83083 | 1.16875 | 4.5876 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/counting_and_probability/119.json | 64 | 1 | 0.621971 | 2.39603 | 1.00165 | 0.540333 | 1.97687 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/counting_and_probability/119.json | 128 | 1 | 0.742824 | 2.4732 | 1.19366 | 0.768206 | 2.75114 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/counting_and_probability/119.json | 256 | 1 | 0.746711 | 2.31043 | 1.49744 | 0.765095 | 1.56214 | 5 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/geometry/477.json | 64 | 1 | 0.605437 | 3.1773 | 0.976872 | 0.750033 | 2.122 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/geometry/477.json | 128 | 1 | 0.63926 | 2.77283 | 1.29125 | 0.71343 | 2.0473 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |
| test/geometry/477.json | 256 | 1 | 0.704015 | 2.70993 | 1.67291 | 0.802112 | 1.44872 | 2 | FUTURE_KEY_INTERACTION_SIGNAL |

## 17. Layer / Head Attribution
Per-layer and per-head rows are stored in the pilot JSON.

## 18. Value-Channel Attribution
Per-value-channel summary statistics are stored in the pilot JSON.

## 19. Per-Prompt / Per-t0 Results
See path waterfall and JSON curves.

## 20. Path-vs-KL Association
```json
{
  "rho_state_ratio_vs_KL_ratio": 0.09999999999999831,
  "rho_key_ratio_vs_KL_ratio": 0.3499999999999941,
  "rho_query_ratio_vs_KL_ratio": -0.19999999999999662,
  "rho_local_ratio_vs_KL_ratio": 0.2499999999999958
}
```

## 21. Pilot Classification
`FUTURE_KEY_INTERACTION_SIGNAL`

## 22. What Is Supported
Pilot path classification: FUTURE_KEY_INTERACTION_SIGNAL

## 23. What Is NOT Supported
- This is not a formal multi-prompt validation.
- This does not prove feedback if classification is FEEDBACK_REQUIRED_CANDIDATE.
- No cadence, repeated quantization, new quantizer, bit allocation, scale redesign, Hadamard method, or method design was run.

## 24. Negative / Corrective Results
Readout is computed from frozen FP queries and cloned recurrent replay, not from the previous parallel-branch hook timing signal.

## 25. Next Recommended Experiment
multi-prompt formal observability validation
