# Causal Mechanism Report

Status: **COMPLETE_AND_VALIDATED**.

## Result

The frozen directional intervention produced a positive primary contrast on development and replicated prospectively on both validation documents:

| Document | Role | GPU | D: visible minus low Future-KL AUC | D(+) | D(-) |
|---|---|---|---:|---:|---:|
| pg19-test-37702 | development | GPU6 | 70.268568 | 90.849051 | 49.688085 |
| pg19-test-28988 | validation | GPU6 | 141.175802 | 96.593700 | 185.757905 |
| pg19-test-26493 | validation | GPU3 | 52.382955 | 73.885682 | 30.880227 |

Every document passed fork parity, initial post-QDQ magnitude matching, signed-direction alignment, visible/low sensitivity separation, branch identity, finite-value, and sidecar-integrity gates. Both signs were positive in every document; the finding is therefore not carried by a single sign choice.

## Causal interpretation and boundary

At matched initial post-QDQ error magnitude, alignment with the independently frozen future-query-sensitive direction caused greater long-horizon Future-KL AUC than alignment with the low-sensitivity direction. This is a replicated pilot causal signal that initial error direction matters for downstream functional damage.

The effect is not explained by larger final propagated error magnitude: mean endpoint target-state norms were visible/low 26.715/68.003, 30.568/87.631, and 27.263/84.262, respectively. However, the branch intervention changes the recurrent state trajectory, and every later step includes the real L6 quantizer/writeback. Therefore the evidence supports a direction-mediated trajectory effect, not a claim that direct readout visibility is the sole mediator or that propagation and quantizer feedback are irrelevant.

This is a three-document mechanistic pilot, not a population-level statistical estimate. It does not validate a training-time surrogate, authorize a new rotation, or establish generality beyond the frozen L6 setting.

## Integrity and budget

- Frozen runner SHA256: `fb49e377b6b2786deedd694c4489115f08d4cb0bb95cb7f45a93b0a44914ed20`.
- Frozen L6 SHA256: `51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964`.
- Formal GPU-hours: 10.129063970844.
- Completed GPU-hours including the passing micro: 10.137294944525.
- Conservative total including upper-bounded failed probes: 10.613125083414 / 16.0 (PASS).
- Historical micro attempts 01–04 remain engineering failures/disconnect evidence and were not retroactively relabeled. Attempt04/05 experiment history remained isolated and unmodified.

## Scientific conclusion

`DIRECTIONAL_CAUSAL_SIGNAL_REPLICATED_IN_PILOT`.
