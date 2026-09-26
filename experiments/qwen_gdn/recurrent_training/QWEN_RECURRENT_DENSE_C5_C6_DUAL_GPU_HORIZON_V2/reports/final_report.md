# QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2 — stopped at H32 parity

The 4-GPU resource amendment was registered prospectively. Its topology/horizon search cannot start because the mandatory 2-GPU H32 gradient and optimizer-update parity gates failed. No H128/H64/H32 sustained screen, formal C5/C6 training, or AIME generation was started.

| Gate | Result |
|---|---|
| Available GPUs | 4 × RTX3090 24GB |
| Model partition / untied embedding–head | PASS / PASS |
| Single-GPU H32 forward repeat | PASS |
| Single-GPU H32 gradient bitwise repeat | FAIL |
| 2-GPU H32 forward parity | PASS |
| 2-GPU H32 gradient parity | FAIL |
| 2-GPU H32 one-step update parity | FAIL |
| Recurrent writeback / BPTT boundary | PASS / PASS |
| 2-GPU training semantics | FAIL |
| 4-GPU topology | NOT RUN, blocked by 2-GPU foundation gate |

All 240 sampled layer/token traces match bitwise for consumed state, pre-QDQ state, scale, qcodes, and post-QDQ state. Inputs, target positions, all eight per-target loss components, and the total C5 training loss (0.0036859801330138) also match. The raw and clipped rotation gradient hashes differ across both single-GPU repeats and the dual-GPU run; the post-Adam theta and rotation hashes likewise differ. Gradient norms were 0.071152754128, 0.0711315870285 (single repeats) and 0.0711613073945 (dual). This establishes a bitwise mismatch, not that the dual topology alone caused it. No applicable pre-existing frozen numerical tolerance was found, and none was invented after seeing the dual result.

Resource decision table:

| GPUs | H | C5 | C6 | Verdict |
|---:|---:|---|---|---|
| 2 | 128 | NOT RUN | NOT RUN | blocked at parity |
| 4 | 128 | NOT RUN | NOT RUN | blocked at parity |
| 2 | 64 | NOT RUN | NOT RUN | blocked at parity |
| 4 | 64 | NOT RUN | NOT RUN | blocked at parity |
| 2 | 32 | NOT RUN | NOT RUN | blocked at parity |
| 4 | 32 | NOT RUN | NOT RUN | blocked at parity |

FINAL_TRAIN_GPU_COUNT = NOT_SELECTED; FINAL_GRADIENT_HORIZON = NOT_SELECTED; FINAL_PARTITION = NOT_SELECTED. Selection rule remains MAXIMIZE_HORIZON_THEN_MINIMIZE_GPU_COUNT.

Stop condition: 2GPU H32 semantic parity FAIL. Await manual direction; do not proceed to four GPUs, formal training, or AIME. Preserved inputs: `analysis/one_step_c5_single_repeat1.json`, `analysis/one_step_c5_single_repeat2.json`, `analysis/one_step_c5_dual_split16.json` and their logs.
