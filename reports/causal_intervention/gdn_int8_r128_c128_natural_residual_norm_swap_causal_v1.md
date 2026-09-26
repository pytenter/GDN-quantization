# GDN INT8 R128/C128 Natural Residual Norm-Swap Causal V1

## 1. Scientific Question
Does the natural R128-vs-C128 single-pulse damage gap mainly come from residual magnitude rather than residual structure?

## 2. Prior Evidence
R128 repeated accumulation formal is SUPPORTED; operator-aligned shadow audit found MAGNITUDE_DOMINATED_SIGNAL.

## 3. Competing Hypotheses
Magnitude causal dominant, mixed, structure causal dominant, or unclear.

## 4. Experimental Design
3 prompts x t0={64,128,256}; conditions FP, REAL_R, REAL_C, R_STRUCT_C_NORM, C_STRUCT_R_NORM.

## 5. Residual Definitions
E_R=Q_R128(S_t)-S_t; E_C=Q_C128(S_t)-S_t.

## 6. Norm-Swap Construction
Per layer/head Frobenius norms are swapped; no global scalar is used.

## 7. Protocol
single-pulse residual injection at t0 from the same FP state; teacher-forced continuation; no repeated quantization

## 8. Stage-0 Gates
{
  "C128_RESIDUAL_IDENTITY_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "NORM_SWAP_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "R128_RESIDUAL_IDENTITY_GATE": "PASS",
  "SINGLE_PULSE_GATE": "PASS",
  "STRUCTURE_DIRECTION_GATE": "PASS"
}

## 9. Natural R128 vs C128 Damage
{
  "REAL_R > REAL_C KL": "9 / 9",
  "Median REAL_R KL AUC": 0.0005797154165073414,
  "Median REAL_C KL AUC": 0.00016010373975861025,
  "Median R_STRUCT_C_NORM KL AUC": 0.00032834268730479177,
  "Median C_STRUCT_R_NORM KL AUC": 0.00015945278576565082,
  "Median natural R-C KL gap": 0.0005013377777409897,
  "R shrink improves KL": "9 / 9",
  "Median R shrink rescue fraction": 0.729979764751469,
  "C enlargement worsens KL": "6 / 9",
  "At C norm R structure worse": "9 / 9",
  "Median structure effect at C norm": 0.00016225074899729448,
  "At R norm R structure worse": "9 / 9",
  "Median structure effect at R norm": 0.0004946689004350837
}

## 10. R-Structure at C-Norm

## 11. C-Structure at R-Norm

## 12. Same-Norm Structure Comparisons

## 13. State Propagation

## 14. Readout-Aware Diagnostics

## 15. Behavioral KL

## 16. Layer / Head Heterogeneity
Layer/head norm summaries are stored in JSON per unit.

## 17. Per-Prompt / Per-t0 Results
| Prompt | t0 | REAL_R KL | REAL_C KL | R_STRUCT_C_NORM KL | C_STRUCT_R_NORM KL | Natural R-C Gap | R Shrink Rescue | C Enlarge Damage | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| test/algebra/1332.json | 64 | 0.0003555873760452669 | 0.00011306085905189798 | 0.0002503266632394594 | 8.85029935998711e-05 | 0.00024252651699336892 | 0.00010526071280580752 | -2.4557865452026886e-05 | descriptive |
| test/algebra/1332.json | 128 | 0.0005514973365127371 | 0.00016010373975861025 | 0.0002728551723764287 | 0.0001347802392901157 | 0.0003913935967541269 | 0.00027864216413630845 | -2.5323500468494556e-05 | descriptive |
| test/algebra/1332.json | 256 | 0.0005487653861917414 | 0.00010580278440220769 | 0.0004853808461824319 | 0.00015945278576565082 | 0.00044296260178953376 | 6.338454000930955e-05 | 5.3650001363443134e-05 | descriptive |
| test/counting_and_probability/119.json | 64 | 0.0010365976142654599 | 0.00016609193830749729 | 0.00032834268730479177 | 0.0002535748510744387 | 0.0008705056759579626 | 0.0007082549269606681 | 8.74829127669414e-05 | descriptive |
| test/counting_and_probability/119.json | 128 | 0.0005779328986039627 | 7.659512086297301e-05 | 0.0002107239836072007 | 0.00015533594572300473 | 0.0005013377777409897 | 0.000367208914996762 | 7.874082486003171e-05 | descriptive |
| test/counting_and_probability/119.json | 256 | 0.0005797154165073414 | 0.0001267737564188637 | 0.0001980385848757696 | 8.504651607225771e-05 | 0.0004529416600884777 | 0.0003816768316315718 | -4.172724034660598e-05 | descriptive |
| test/geometry/477.json | 64 | 0.0015831390617447004 | 0.00021612386996426255 | 0.0004586142724585481 | 0.0003400912850036325 | 0.001367015191780438 | 0.0011245247892861523 | 0.00012396741503936995 | descriptive |
| test/geometry/477.json | 128 | 0.0010467427923719665 | 0.000358217984099674 | 0.0007333794043785985 | 0.0004000065784839923 | 0.0006885248082722924 | 0.000313363387993368 | 4.178859438431831e-05 | descriptive |
| test/geometry/477.json | 256 | 0.0013499596585535905 | 0.0005071603297231891 | 0.0007347332020312984 | 0.0005257989447973272 | 0.0008427993288304015 | 0.0006152264565222922 | 1.863861507413818e-05 | descriptive |

## 18. Causal Decomposition
`NO_CLEAR_CAUSAL_DECOMPOSITION`

## 19. Pilot Classification
`NO_CLEAR_CAUSAL_DECOMPOSITION`

## 20. What Is Supported
No clean magnitude-vs-structure causal decomposition is supported.

## 21. What Is NOT Supported
- No cadence experiment, repeated quantization, scale intervention, frozen-driver decomposition, or method design was run.
- METHOD_DESIGN_READY remains NO.

## 22. Negative / Corrective Results
Report preserves same-norm reversals and small denominators; no positive story is forced.

## 23. Next Recommended Experiment
inspect heterogeneity before next causal intervention
