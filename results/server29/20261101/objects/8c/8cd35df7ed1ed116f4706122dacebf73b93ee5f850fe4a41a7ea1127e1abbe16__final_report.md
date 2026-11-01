# Qwen Hadamard Error Injection Functional Analysis V1

Deterministic teacher-forced intervention with a common FP recurrent state, FP query and FP gate. The injected error is the observed native-basis difference from each INT8 trajectory, which includes historical drift.

## Gates and coverage

- Audit: PASS; zero-injection parity: PASS; source trajectories: 60; clean59: 59; GDN layers: 24.
- No generation, training, model/quantizer/scorer modification, or permanent full state/logit tensors.
- Full-logits feasibility: NOT_FEASIBLE_FULL_CLEAN59.

## Paired local functional damage

Relative L2 is lower-is-better. Benefit = Identity minus Hadamard; positive favors Hadamard. Each trajectory is weighted equally.

| Stage | Identity median | Hadamard median | Benefit median | Bootstrap 95% CI | Sign p |
| --- | ---: | ---: | ---: | --- | ---: |
| state | 0.391514 | 0.366796 | 0.0329513 | [0.013291329145431519, 0.03979501128196716] | 0.00379371 |
| readout | 0.166405 | 0.221941 | -0.0493482 | [-0.05765761435031891, -0.03657527267932892] | 3.46945e-18 |
| rmsnorm | 0.14126 | 0.220203 | -0.0764389 | [-0.08145973086357117, -0.05999520421028137] | 3.46945e-18 |
| gate | 0.187837 | 0.281502 | -0.0893868 | [-0.1046159639954567, -0.07519406080245972] | 3.46945e-18 |
| output_projection | 0.184305 | 0.285364 | -0.0998916 | [-0.1149105578660965, -0.08437404036521912] | 3.46945e-18 |

## Rescue versus stable failure

True-ID groups: stable-correct 18, Hadamard-rescue 23, stable-failure 18. Two-sided permutation p values are Holm-adjusted across five stages.

| Stage | Rescue benefit median | Stable-failure benefit median | Difference | Cliff delta | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| state | 0.0316719 | -0.0200784 | 0.0517503 | 0.614 | 0.00439956 |
| readout | -0.0500358 | -0.10152 | 0.0514844 | 0.609 | 0.00349965 |
| rmsnorm | -0.0764389 | -0.113881 | 0.037442 | 0.628 | 0.00449955 |
| gate | -0.0893868 | -0.135344 | 0.0459575 | 0.599 | 0.00449955 |
| output_projection | -0.0998916 | -0.148761 | 0.0488698 | 0.594 | 0.00449955 |

## Interpretation

Primary output-projection benefit supported under the preregistered CI/sign rule: False.
Local module effects are not, by themselves, proof of downstream logit benefit or reasoning causality. No outcome-based layer, token, or outlier selection was performed.

Clean59 nonfinite counts: {'state': 0, 'readout': 0, 'rmsnorm': 0, 'gate': 0, 'output_projection': 0}. FP denominator floor counts: {'state': 0, 'readout': 0, 'rmsnorm': 0, 'gate': 0, 'output_projection': 0}.
The frozen excluded raw trajectory remains preserved and is not part of clean59 inference.
