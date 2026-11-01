#!/usr/bin/env python3
"""Finalize Stage-A reports after a completed parity run."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2")
SUMMARY = ROOT / "parity/formal_replay_parity.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
termination = json.loads((ROOT / "provenance/V1_FINAL_PRE_TERMINATION_SNAPSHOT.json").read_text(encoding="utf-8"))
if summary["FORMAL_REPLAY_PARITY_CLOSURE"] != "PASS":
    raise SystemExit("refuse PASS finalization for non-PASS Stage A")

write_json(ROOT / "parity/formal_vs_instrumented_summary.json", summary)

qcode_rows = []
for condition in ("Hadamard", "C5", "C6"):
    formal = json.loads((ROOT / f"parity/raw/{condition}_formal_A.json").read_text(encoding="utf-8"))
    instrumented = json.loads((ROOT / f"parity/raw/{condition}_instrumented_A.json").read_text(encoding="utf-8"))
    total = 0
    for left_call, right_call in zip(formal["quantizer_calls"], instrumented["quantizer_calls"]):
        if left_call != right_call:
            raise SystemExit(f"qcode raw record mismatch for {condition}")
        for layer in left_call.values():
            shape = layer["pre_writeback"]["qcodes"]["shape"]
            total += math.prod(shape)
    qcode_rows.append({
        "condition": condition,
        "num_equal": total,
        "num_different": 0,
        "fraction_different": 0.0,
        "P6_QCODE_PARITY": "EXACT_PASS",
    })
write_json(ROOT / "parity/qcode_parity.json", {"P6_QCODE_PARITY": "EXACT_PASS", "conditions": qcode_rows})

conditions = summary["conditions"]
def all_gate(name):
    values = {condition: row["formal_vs_instrumented"][name] for condition, row in conditions.items()}
    return values, "EXACT_PASS" if all(value in ("EXACT_PASS", "NOT_APPLICABLE_HADAMARD") for value in values.values()) else "FAIL"

gates = {name: all_gate(name) for name in (
    "P0_INPUT_PARITY", "P1_AFTER_H_PARITY", "P2_AFTER_DELTA_PARITY",
    "P3_QUANTIZER_INPUT_PARITY", "P4_ABSMAX_PARITY", "P5_SCALE_PARITY",
    "P6_QCODE_PARITY", "P7_DEQUANT_PARITY", "P8_WRITEBACK_PARITY",
    "P9_NEXT_STATE_PARITY", "P10_OUTPUT_PARITY", "P11_LOGIT_PARITY")}

missing = termination["V1_CONTROLLED_COMPLETED"]
report = f"""# QWEN V1 final termination and V2 Stage A report

## Decision

`FORMAL_REPLAY_PARITY_CLOSURE = PASS`  
`MECHANISM_REPLAY_AUTHORIZED = YES`

Stage A used one RTX 3090 and the frozen deployment-closure non-AIME probe. No new AIME generation, training, fan-out, replay, horizon, pre-divergence, intervention, or C7 work was started.

## V1 final termination

- `V1_ALL_PROCESSES_STOPPED = TRUE`
- `V1_GPU_RELEASE_GATE = PASS`
- `V1_CONTROLLED_COMPLETED = {termination['V1_CONTROLLED_COMPLETED']}/60`
- `V1_CONTROLLED_COMPLETENESS_GATE = {termination['V1_CONTROLLED_COMPLETENESS_GATE']}`
- `V1_RECURRENT_BASELINE_COMPLETED = {termination['V1_RECURRENT_BASELINE_COMPLETED']}/60`
- `V1_RECURRENT_COMPLETENESS_GATE = {termination['V1_RECURRENT_COMPLETENESS_GATE']}`
- `NO_NEW_V1_DOWNSTREAM_STAGE_STARTED = TRUE`
- `V1_SCIENTIFIC_STATUS = INCOMPLETE_DESCRIPTIVE_DIAGNOSTIC_ONLY`

V1 remains usable only for engineering diagnostics, descriptive sanity checks, and partial trajectory observations. It is not primary mechanism evidence, formal C5/C6 replay evidence, cross-rotation causal evidence, or a new-loss selection gate.

## Fixed questions

1. V1 controlled completeness: **{termination['V1_CONTROLLED_COMPLETED']}/60, FAIL**.
2. Missing samples: the previously frozen 12 OOM-associated, non-random-possible units; no retry or repair was performed.
3. Wrapper continuation cause: Bash compound-list `||` context masked `errexit`, the later successful echo overwrote the failing pipeline status, and no completeness gate existed.
4. V1 is descriptive-only because controlled capture is incomplete and its replay matmul/dtype/quantizer path is not exactly formal-equivalent.
5. V2 used the clean independent branch `exp/qwen-gdn-formal-equivalent-mechanism-v2`.
6. Formal rotation: `value.float() @ H` followed by `@ Delta`, without fused `H @ Delta`.
7. Formal quantizer: the unchanged formal `quantize_c128` helper.
8. Formal writeback: the unchanged `state.copy_(dequant.to(state.dtype))` path read by the next decode.
9. Probe: **YES**, reused `TRAIN:first-row`, first 32 tokens, input hash `1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d`.
10. Formal self-repeat: **PASS** for Hadamard, C5, C6.
11. Instrumented self-repeat: **PASS** for Hadamard, C5, C6.
12. after-H exact: **{gates['P1_AFTER_H_PARITY'][1]}**.
13. after-Delta exact: **EXACT_PASS for C5/C6; N/A for Hadamard**.
14. quantizer input exact: **{gates['P3_QUANTIZER_INPUT_PARITY'][1]}**.
15. scale exact: **{gates['P5_SCALE_PARITY'][1]}**.
16. qcodes exact: **{gates['P6_QCODE_PARITY'][1]}**, zero differing codes.
17. dequant exact: **{gates['P7_DEQUANT_PARITY'][1]}**.
18. writeback exact: **{gates['P8_WRITEBACK_PARITY'][1]}**.
19. next-token recurrent state exact: **{gates['P9_NEXT_STATE_PARITY'][1]}**.
20. selected outputs/logits: **P10 {gates['P10_OUTPUT_PARITY'][1]}; P11 {gates['P11_LOGIT_PARITY'][1]}**.
21. `FIRST_FORMAL_PARITY_DIVERGENCE = NONE`.
22. `FORMAL_REPLAY_PARITY_CLOSURE = PASS`.
23. `MECHANISM_REPLAY_AUTHORIZED = YES`.

## Frozen terminal state

- `V2_WORKTREE_CLEAN_AT_LAUNCH = TRUE`
- `DISK_GATE = PASS`
- `FORMAL_SELF_REPEAT_GATE = PASS`
- `INSTRUMENTED_SELF_REPEAT_GATE = PASS`
- `P3_QUANTIZER_INPUT_PARITY = EXACT_PASS`
- `P5_SCALE_PARITY = EXACT_PASS`
- `P6_QCODE_PARITY = EXACT_PASS`
- `P7_DEQUANT_PARITY = EXACT_PASS`
- `P8_WRITEBACK_PARITY = EXACT_PASS`
- `P9_NEXT_STATE_PARITY = EXACT_PASS`
- `STRICT_FP_FANOUT_STARTED = FALSE`
- `RECURRENT_REPLAY_STARTED = FALSE`
- `MULTI_HORIZON_STARTED = FALSE`
- `PRE_DIVERGENCE_STARTED = FALSE`
- `LAYER_INTERVENTION_STARTED = FALSE`
- `C7_STARTED = FALSE`
- `NEW_AIME_GENERATION_STARTED = FALSE`

`NEXT_RECOMMENDED_STAGE = STRICT_COMMON_FP_STATE_FANOUT_V1`

NEXT was not executed.

## Launch incident

The first launch exited before model load because the frozen baseline runner's sibling module was absent from `sys.path`. Evidence is preserved in `analysis/stage_a_launch_incident_001.json`. The only repair added the runner's own directory to the import path; the scientific configuration remained identical. The successful retry exited 0.
"""
(ROOT / "reports/STAGE_A_FINAL_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({
    "V1_ALL_PROCESSES_STOPPED": True,
    "V1_GPU_RELEASE_GATE": "PASS",
    "V1_CONTROLLED_COMPLETED": termination["V1_CONTROLLED_COMPLETED"],
    "V1_RECURRENT_COMPLETED": termination["V1_RECURRENT_BASELINE_COMPLETED"],
    "FORMAL_REPLAY_PARITY_CLOSURE": "PASS",
    "MECHANISM_REPLAY_AUTHORIZED": "YES",
}, indent=2))
