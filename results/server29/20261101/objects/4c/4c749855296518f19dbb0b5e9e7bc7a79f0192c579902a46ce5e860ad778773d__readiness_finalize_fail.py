#!/usr/bin/env python3
"""Finalize the readiness task after the preregistered offline parity failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_READINESS_V1")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()


def main():
    parity=load(ROOT/"validation/exact_parity.json")
    quant=load(ROOT/"validation/quantile_validation.json")
    bad=[x for x in quant["rows"] if not x["pass"]]
    diagnosis={
        "status":"READINESS_STOPPED_OFFLINE_PARITY_FAIL",
        "model_or_formal_semantics_failure":False,
        "exact_sections":{
            "group_moments":parity["sections"]["group_moments"]["status"],
            "token_local_tail_moments":parity["sections"]["token_local_tail_moments"]["status"],
            "paired_scale_ratios":parity["sections"]["paired_scale_ratios"]["status"],
            "key_coverage_exact":parity["key_coverage_exact"],
        },
        "top_k_event_identity":"FAIL",
        "top_k_mismatch_count_after_keyed_comparison":2944,
        "top_k_root_cause":"equal-value ties (especially exact zeros) were selected by argpartition without the frozen deterministic group-id tie break",
        "quantile_sketch_validation":"FAIL",
        "preregistered_relative_tolerance_p999":0.003,
        "failed_quantiles":bad,
        "post_hoc_tolerance_change_permitted":False,
        "gpu_validation_authorized":False,
        "prospective_recovery_required":[
            "new reducer version with deterministic full tie ordering before top-K truncation",
            "new sketch/percentile comparison contract and tolerance preregistered before looking at rerun results",
        ],
    }
    dump(ROOT/"validation/failure_diagnosis.json",diagnosis)
    dump(ROOT/"smoke/gpu_streaming_smoke_summary.json",{"status":"NOT_RUN_STREAMING_REDUCER_PARITY_FAIL","scientific_conclusion":False})
    dump(ROOT/"smoke/memory_summary.json",{"status":"NOT_REACHED_GPU_SMOKE_NOT_AUTHORIZED"})
    dump(ROOT/"smoke/artifact_size_summary.json",{"status":"NOT_REACHED_GPU_SMOKE_NOT_AUTHORIZED"})
    dump(ROOT/"projection/full60_disk_projection.json",{
        "status":"NOT_REACHED_GPU_STREAMING_ARTIFACT_SIZE_UNAVAILABLE",
        "PROJECTED_FULL60_FINAL_ARTIFACT_GB":None,
        "PROJECTED_FULL60_TEMPORARY_PEAK_GB":None,
        "PROJECTED_FREE_AFTER_FULL60_GB":None,
        "FULL60_RESOURCE_READINESS":"NOT_REACHED",
    })
    dump(ROOT/"projection/full60_runtime_projection.json",{"status":"NOT_REACHED_GPU_SMOKE_NOT_AUTHORIZED"})
    gates={
        "SOURCE_GATE":"PASS","GROUP_CARDINALITY_GATE":"PASS","CURRENT_ROW_SCHEMA_CORRECT":"YES",
        "STATISTICAL_UNIT_FROZEN":"PASS","PRIMARY_METRICS_FROZEN":"PASS",
        "TAIL_DEFINITION_FROZEN":"PASS","PERCENTILE_DEFINITION_FROZEN":"PASS",
        "STREAMING_REDUCER_IMPLEMENTED":"PASS","OFFLINE_REFERENCE_GATE":"PASS",
        "STREAMING_REDUCER_PARITY":"FAIL",
        "QUANTILE_SKETCH_TOLERANCE_PREREGISTERED":"PASS","QUANTILE_SKETCH_VALIDATION":"FAIL",
        "GPU_STREAMING_SMOKE":"NOT_RUN","STREAMING_MEMORY_GATE":"NOT_REACHED",
        "ATOMIC_ARTIFACT_GATE":"PASS","FAIL_CLOSED_ORCHESTRATION_GATE":"PASS","RESUME_POLICY_GATE":"PASS",
        "PROJECTED_FULL60_FINAL_ARTIFACT_GB":"NOT_REACHED",
        "PROJECTED_FULL60_TEMPORARY_PEAK_GB":"NOT_REACHED",
        "PROJECTED_FREE_AFTER_FULL60_GB":"NOT_REACHED",
        "FULL60_RESOURCE_READINESS":"NOT_REACHED",
        "STAGE_B_STREAMING_READINESS":"FAIL","FIRST_FAILED_GATE":"STREAMING_REDUCER_PARITY",
        "FULL60_AUTHORIZED_BY_THIS_TASK":False,"FULL60_STARTED":False,
        "RECURRENT_REPLAY_STARTED":False,"MULTI_HORIZON_STARTED":False,
        "PRE_DIVERGENCE_STARTED":False,"LAYER_INTERVENTION_STARTED":False,
        "C7_STARTED":False,"NEW_LOSS_STARTED":False,"NEW_ROTATION_TRAINING_STARTED":False,
        "NEW_AIME_GENERATION_STARTED":False,
        "NEXT_RECOMMENDED_ACTION":"PROSPECTIVE_STREAMING_REDUCER_V2_PROTOCOL_REQUIRED",
    }
    dump(ROOT/"validation/final_gate_status.json",gates)
    report=f"""# QWEN_GDN_STAGE_B_STREAMING_READINESS_V1 — Final Readiness Report

## Final status

```text
STAGE_B_STREAMING_READINESS = FAIL
FIRST_FAILED_GATE = STREAMING_REDUCER_PARITY
FULL60_RESOURCE_READINESS = NOT_REACHED
FULL60_AUTHORIZED_BY_THIS_TASK = FALSE
FULL60_STARTED = FALSE
NEXT_RECOMMENDED_ACTION = PROSPECTIVE_STREAMING_REDUCER_V2_PROTOCOL_REQUIRED
```

No GPU validation smoke and no full-60 workload were launched.

## Gate summary

```text
SOURCE_GATE = PASS
GROUP_CARDINALITY_GATE = PASS
CURRENT_ROW_SCHEMA_CORRECT = YES
STATISTICAL_UNIT_FROZEN = PASS
PRIMARY_METRICS_FROZEN = PASS
TAIL_DEFINITION_FROZEN = PASS
PERCENTILE_DEFINITION_FROZEN = PASS
STREAMING_REDUCER_IMPLEMENTED = PASS
OFFLINE_REFERENCE_GATE = PASS
STREAMING_REDUCER_PARITY = FAIL
QUANTILE_SKETCH_TOLERANCE_PREREGISTERED = PASS
QUANTILE_SKETCH_VALIDATION = FAIL
GPU_STREAMING_SMOKE = NOT_RUN
STREAMING_MEMORY_GATE = NOT_REACHED
ATOMIC_ARTIFACT_GATE = PASS
FAIL_CLOSED_ORCHESTRATION_GATE = PASS
RESUME_POLICY_GATE = PASS
PROJECTED_FULL60_FINAL_ARTIFACT_GB = NOT_REACHED
PROJECTED_FULL60_TEMPORARY_PEAK_GB = NOT_REACHED
PROJECTED_FREE_AFTER_FULL60_GB = NOT_REACHED
FULL60_RESOURCE_READINESS = NOT_REACHED
```

## Q1–Q4: true group cardinality and old schema

1. `393,216 records/token` is `24 layers × 4 conditions × 4,096 true C128 groups`.
2. Yes. The authoritative true group count is `scale.numel()`.
3. No. The old serializer reduced K (`dim=-2`) and flattened `[1,32,128]`; it did not expand the 128 elements inside each group.
4. `TRUE_GROUPS_PER_TOKEN = 98,304` per condition, or `393,216` across Native/H/C5/C6. State and qcode each have 524,288 elements/layer; scale has 4,096 elements/layer.

`ROW_SCHEMA_BUG_FOUND = NO`. The former model forward and serializer semantics remain valid; serializing all groups is simply operationally infeasible.

## Q5–Q7: frozen science and tails

5. Primary metrics are actual runtime scale, absmax, `PAPR=absmax/(RMS+eps)`, and QDQ relative L2. Range and the other diagnostics remain secondary.
6. `STATISTICAL_UNIT = SAMPLE`, `N=60`. Token/layer/head/group observations are hierarchical mechanism data, not independent statistical n.
7. Token-local median/p95/p99/CVaR95/CVaR99/max are exact, with NumPy/torch linear interpolation and top-`ceil(tail*N)` CVaR. Global group tails use a deterministic 0.25%-relative-accuracy DDSketch-style histogram and are approximate.

## Q8–Q9: offline parity

8. Exact float64 moments, token-local tails, paired threshold counts, and key coverage passed. Overall parity failed because top-32 event identity did not implement the frozen deterministic tie rule: 2,944 keyed event positions disagreed among equal values, mainly exact zeros.
9. Approximate global quantile validation failed the preregistered 0.3% p99.9 tolerance in two cells:
   - C6 absmax p99.9: 0.326056% relative error.
   - Native scale p99.9: 0.304377% relative error.

The tolerance cannot be widened after observing these results. A prospective V2 protocol must freeze a corrected tie algorithm and sketch/percentile error contract before rerun.

## Q10–Q13: GPU/runtime/resource validation

10. GPU streaming smoke was not authorized because offline reducer parity failed.
11. Streaming GPU memory stability was therefore not measured.
12. Streaming per-sample and full-60 artifact sizes were not measured.
13. Projected post-full60 free space was not calculated; resource readiness is `NOT_REACHED`.

## Q14–Q16: orchestration

14. Atomic artifact protocol passed: fsynced temp payload/checksum, atomic rename, done manifest, and hash validation.
15. Fail-closed tests passed for worker nonzero, 59/60 completeness, OOM, and emergency disk. Every case blocked downstream launch.
16. Resume policy passed: only valid `sample.done + hash + identical identity` was accepted; partial, corrupted, and identity-mismatched artifacts were rejected.

## Q17–Q18: readiness and execution

17. No, the current implementation is not ready to launch full-60.
18. No. `FULL60_STARTED = FALSE`.

## Stopping confirmation

```text
FULL60_STARTED = FALSE
RECURRENT_REPLAY_STARTED = FALSE
MULTI_HORIZON_STARTED = FALSE
PRE_DIVERGENCE_STARTED = FALSE
LAYER_INTERVENTION_STARTED = FALSE
C7_STARTED = FALSE
NEW_LOSS_STARTED = FALSE
NEW_ROTATION_TRAINING_STARTED = FALSE
NEW_AIME_GENERATION_STARTED = FALSE
```
"""
    (ROOT/"reports/FINAL_READINESS_REPORT.md").write_text(report,encoding="utf-8")
    out=ROOT/"hashes/SHA256SUMS"; out.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in ROOT.rglob("*") if p.is_file() and p!=out and "__pycache__" not in p.parts)
    out.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files),encoding="utf-8")
    print(json.dumps(gates,indent=2,sort_keys=True))


if __name__=="__main__": main()
