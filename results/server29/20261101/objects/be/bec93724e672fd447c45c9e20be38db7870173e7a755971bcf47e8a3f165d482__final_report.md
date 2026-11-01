# LING_RECURRENT_DENSE_L6_L7_V1 final report

Status: **COMPLETE**

## Frozen formal results

- L2: 9/20, abstain=6, length_limit=8, EOS=12
- L6: 9/20, abstain=4, length_limit=4, EOS=16
- L7: 10/20, abstain=6, length_limit=9, EOS=11

FP cross-hardware bitwise parity failed. Audited INT8 condition-specific parity passed before mixed-hardware formal generation.

- hardware assignment manifest SHA256: `407d17fead6088b1c7418c9e57093172444eba7fb1e1b61bfb88ab15c25b6c10`
- effective assignment SHA256: `e2781d2ecaf81f09d0c36b78438bb4d88c8236cebb5bd7a55c72c53a85bb3cf3`
- original/amendment assignment hashes: `{"manifests/hardware_assignment.json": "407d17fead6088b1c7418c9e57093172444eba7fb1e1b61bfb88ab15c25b6c10", "manifests/hardware_assignment_amendment_l7_idle_4090_helper_v3.json": "7b61e1c92be0b00fc6b401fb96ddac0634f6b17dfa98968d642304256ddca890", "manifests/hardware_assignment_amendment_l7_idle_helpers_v5.json": "98e2d7c0f68e315320c890501c58d2074198a44b694e289950fdbcfb1d9a6514", "manifests/hardware_assignment_amendment_q29_helper_v2.json": "72e41986f79c1e6ba82b86910fba232bd7b9cf265952159155e450dadae30765"}`
- per-sample GPU metadata: `analysis/per_sample_gpu_metadata.json`
- cross-hardware parity artifact hashes: `{"FP_cross_hardware_failure": "e8168de0acd6270f6dadf1ad5aa2e2a8ef3f19ab53f504e1c076455d436409eb", "L6_INT8_condition_gate": "b41352758a510bacb3c8b853db6c7e85c690a4912beecd2423c29467f32a4536", "L7_INT8_condition_gate": "4e7c659176b01ce109dc90feaf7d70e0824050a301eac7c71d6df8fbe59e911d"}`
- actual canonical generated on RTX4090 after frozen amendments: 24
- actual canonical generated on RTX3090 after frozen amendments: 16
- retries and infrastructure failures by hardware: `{"RTX3090": {"generated_samples": 16, "infrastructure_failures": 0, "retries": 0}, "RTX4090": {"generated_samples": 24, "infrastructure_failures": 0, "retries": 0}}`

Hardware-stratified descriptive results (not independent primary tests) are in `analysis/hardware_stratified.json`; no hardware subset was deleted post hoc.

Primary comparisons use the preregistered two-comparison Holm correction. Full rescued/regressed IDs, exact McNemar p-values, bootstrap intervals, token counts, and termination metadata are in `analysis/paired_comparisons.json`, `analysis/per_sample_scores.csv`, and `analysis/summary.csv`.

L6 vs L2: rescued=1, regressed=1, net=0, Holm-adjusted p=1.0.

L7 vs L2: rescued=2, regressed=1, net=1, Holm-adjusted p=1.0.

L7 vs L6 (secondary): rescued=2, regressed=1, net=1, exact p=1.0.

## Training semantics proof

```text
Old L4/L5: FP teacher state_t -> one token -> QDQ -> local loss -> RESET

New L6/L7: student INT8 state_t -> recurrent update -> QDQ -> student INT8 state_t+1
                                                                  |
                                                                  +-> WRITE BACK
                                                                        |
                                                                        +-> token t+1 consumes identical hash
```

`REAL_RECURRENT_WRITEBACK_GATE` and `RECURRENT_STATE_PROVENANCE_GATE` are PASS. The recorded next-token-consumed hashes equal the previous post-QDQ hashes and diverge from teacher-state hashes after quantization error appears.

## Scientific questions

**Q1 — Did old local L4/L5 training only improve local metrics?** The old artifacts establish local/single-step improvements but do not expose the optimizer to accumulated quantization history. The sanity panel reports both local and persistent metrics; task-level claims about L4/L5 are not made because this amendment deliberately did not launch their 20×256K formal runs.

### Non-AIME sanity panel

| condition | single-step state | recurrent state | local functional | persistent functional |
|---|---:|---:|---:|---:|
| H_fixed | 6.5917002e-05 | 0.0051791397 | 7.4315308e-05 | 0.0031478057 |
| L4_old_single_step_Dense_State | 5.1177033e-05 | 0.0043219052 | 5.5051684e-05 | 0.0022675367 |
| L5_old_single_step_Dense_Functional | 4.8872909e-05 | 0.0041319494 | 5.2243432e-05 | 0.0021153298 |
| L6_recurrent_Dense_State | 4.4968326e-05 | 0.0035746222 | 5.0471331e-05 | 0.0019231318 |
| L7_recurrent_Dense_Functional | 5.2411268e-05 | 0.0043340508 | 5.731583e-05 | 0.0023076979 |

Lower is better for all four relative-error metrics. The panel was not used for checkpoint, horizon, or method selection.

**Q2 — Does real recurrent exposure change long-horizon quantization behavior?** Yes, the tested rotations produce measurably different persistent trajectories. Relative to the matched old single-step objectives, the observed deltas are `{"L6_minus_L4_persistent_functional_error": -0.0003444048730457303, "L6_minus_L4_recurrent_state_error": -0.0007472829505776385, "L7_minus_L5_persistent_functional_error": 0.00019236810911473987, "L7_minus_L5_recurrent_state_error": 0.00020210135470360383}` (negative means lower error). This is a non-AIME mechanistic sanity result, not by itself an end-to-end accuracy claim.

**Q3 — Does L6 exceed L2?** The tested recurrent-aware Dense/Cayley state objective did not outperform fixed H under this protocol. This does not reject learnable rotation in general; recurrent exposure plus this state objective was insufficient here.

**Q4 — Does L7 exceed L2?** L7 has a positive observed net gain over L2 under this panel. The result supports this recurrent-aware local out-projection objective in the tested setting, without establishing it as the unique mechanism.

**Q5 — L7 vs L6.** The secondary paired comparison reports net=1 and exact p=1.0. It indicates which tested objective is more favorable after recurrent semantics are held fixed, but it is not evidence that one objective is universally preferable.

## Resource amendment

L3 is `EARLY_STOPPED_FOR_RESOURCE_REALLOCATION` with completed_count=7. Completed outputs and hashes are retained; the partial panel is not called a final accuracy. No new L4/L5 formal generation was launched. Resources were reassigned because the audit showed L3 and L2 use the same H128 mathematical rotation with only a controlled numerical-path difference.

No second seed, extra loss, alternative rotation structure, or outcome-driven retraining was performed.
