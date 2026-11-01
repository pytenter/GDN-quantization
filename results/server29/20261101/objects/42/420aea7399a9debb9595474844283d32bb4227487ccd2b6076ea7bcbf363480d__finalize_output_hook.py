#!/usr/bin/env python3
"""Finalize strictly from preserved rank evidence; no GPU execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
R = ROOT / "reports"
H = ROOT / "hashes"


def read(name):
    return json.loads((A / name).read_text())


def write_json(name, obj):
    with (A / name).open("x") as out:
        json.dump(obj, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def write_report(name, body):
    with (R / name).open("x") as out:
        out.write(body.strip() + "\n")


def main():
    parent = json.loads((ROOT / "configs/parent_manifest.json").read_text())
    single = read("zero_anchor_single_semantics.json")
    if single["ZERO_ANCHOR_SINGLE_GPU_SEMANTICS_GATE"] != "PASS" or not all(single["comparisons"].values()):
        raise RuntimeError("single-GPU exact neutrality not established")
    h4 = [read(f"h4_anchor_rank{i}.error.json") for i in (0, 1)]
    if any(e["phase"] != "H4_anchored_backward" or "setStorage" not in e["message"]
           or "storage of size 0" not in e["message"] or "[1024, 4096]" not in e["message"]
           for e in h4):
        raise RuntimeError("unexpected H4 evidence; do not auto-classify")
    h4_summary = {
        "baseline": "frozen parent: both ranks setStorage [8192,1,1,4], storage size 0; original H4 not rerun",
        "baseline_error_sha256_by_rank": [parent[f"parent_h4_rank{i}_error_sha256"] for i in (0, 1)],
        "intervention": "C5 + 0.0 * finite ordinary final-token model logits sum; same model call/cache semantics, with logits retained solely to form anchor",
        "result": "NEW_FAILURE_MODE",
        "ZERO_ANCHOR_H4_RESCUE": "NO",
        "rank_evidence": [{
            "rank": i, "file": f"h4_anchor_rank{i}.error.json",
            "phase": e["phase"], "error_type": e["type"],
            "message_first_line": e["message"].splitlines()[0],
            "memory_before_backward": e["memory_before_backward"],
            "memory_at_failure": e["memory_at_failure"],
        } for i, e in enumerate(h4)],
        "old_failed_storage_owner": "UNKNOWN",
        "new_failed_storage_owner": "UNKNOWN",
        "subsequent_H4_interventions": "NOT_RUN_STOP_RULE",
    }
    write_json("h4_zero_anchor_result.json", h4_summary)
    h1 = []
    for rank in (0, 1):
        plain = read(f"hook_h1_v2_plain_rank{rank}.json")
        hooked = read(f"hook_h1_v2_instrumented_rank{rank}.json")
        checks = {
            "loss_exact": plain["loss_sha256"] == hooked["loss_sha256"],
            "input_gradient_exact": plain["input_gradient_sha256"] == hooked["input_gradient_sha256"],
            "rotation_gradients_exact": plain["rotation_gradient_sha256"] == hooked["rotation_gradient_sha256"],
            "theta_before_exact": plain["theta_before_sha256"] == hooked["theta_before_sha256"],
            "theta_after_Adam_exact": plain["theta_after_sha256"] == hooked["theta_after_sha256"],
            "checkpoint_conv_hashes_equal": plain["checkpoint_conv_hashes"] == hooked["checkpoint_conv_hashes"],
        }
        peak_delta = (hooked["memory_after_backward"]["peak_allocated_bytes"] -
                      plain["memory_after_backward"]["peak_allocated_bytes"])
        pre_backward = [name for name in hooked["fsdp_profiler_event_names"]
                        if name.startswith("FSDP::pre_backward")]
        if not all(checks.values()) or peak_delta > 500 * 1024 * 1024 or len(pre_backward) != 2:
            raise RuntimeError(f"hook noninterference/visibility failed on rank {rank}")
        h1.append({
            "rank": rank, "plain_file": f"hook_h1_v2_plain_rank{rank}.json",
            "instrumented_file": f"hook_h1_v2_instrumented_rank{rank}.json",
            "checks": checks, "peak_allocated_delta_bytes": peak_delta,
            "pre_backward_profiler_events": pre_backward,
            "public_hook_counts": hooked["hook_counts"],
        })
    write_json("hook_noninterference.json", {
        "HOOK_INSTRUMENTATION_NONINTERFERENCE": "PASS",
        "scope": "reduced passing two-real-GDN-decoder, nested FSDP2 H1, fresh two-rank processes",
        "ranks": h1,
        "v1_diagnostic_serializer_error_preserved": [f"hook_h1_plain_rank{i}.error.json" for i in (0, 1)],
        "first_v1_error_explanation": "Backward completed; only 0-dimensional scalar-to-byte hash serialization failed. v2 changed only diagnostic hashing and wrote new filenames.",
        "full_H4_profiler_trace": "NOT_RUN_NEW_FAILURE_MODE_STOP_RULE",
    })
    verdict = {
        "experiment": ROOT.name,
        "status": "BLOCKED",
        "root_cause_confidence": "UNRESOLVED",
        "C5_LOSS_STANDARD_OUTPUT_DEPENDENCY": "PARTIAL",
        "ZERO_ANCHOR_SINGLE_GPU_SEMANTICS_GATE": "PASS",
        "ZERO_ANCHOR_H4_RESCUE": "NO",
        "H4_anchor_result": "NEW_FAILURE_MODE_SETSTORAGE_1024_4096_SIZE_0_BOTH_RANKS",
        "HOOK_INSTRUMENTATION_NONINTERFERENCE": "PASS_REDUCED_H1_ONLY",
        "FSDP_PRE_BACKWARD_HOOK_ANOMALY": "UNKNOWN_FULL_H4_NOT_TRACED",
        "baseline_pre_backward_hook_trace": "NOT_CAPTURED_FROZEN_PARENT",
        "anchor_pre_backward_hook_trace": "NOT_CAPTURED_STOP_RULE",
        "FIRST_MISSING_PRE_BACKWARD_UNIT": "UNKNOWN",
        "FAILED_STORAGE_OWNER": "UNKNOWN",
        "OUTPUT_VIEW_INPLACE_SIGNAL": "NOT_ESTABLISHED",
        "OUTPUT_HOOK_BYPASS_CAUSAL_SIGNAL": "UNRESOLVED",
        "FSDP_OUTPUT_HOOK_HYPOTHESIS": "UNRESOLVED",
        "DEEPSPEED_ZERO3_NEXT_STAGE_RECOMMENDED": "NO_NOT_YET_JUSTIFIED_BY_THIS_PROTOCOL",
        "DEEPSPEED_INSTALL": "NO",
        "DEEPSPEED_RUNTIME_TEST": "NOT_RUN",
        "H4_COMPATIBILITY_GATE": "BLOCKED",
        "H8_started": False, "H16_started": False, "H32_started": False,
        "formal_C5_C6_started": False, "AIME_started": False,
        "other_user_tasks_interrupted": False,
        "parent_H4_backward": "FAIL", "parent_H32_feasibility": "NOT_DETERMINED",
    }
    write_json("final_verdict.json", verdict)
    write_report("ZERO_OUTPUT_ANCHOR_TEST.md", """
# Zero-output-anchor causal test

The anchored scalar was exactly `C5 + 0.0 * final_captured_token_model_logits.float().sum()`. The ordinary logits were a finite, grad-connected `[1,1,248320]` tensor, retained by an experiment-local helper that makes the same model call and cache writeback as frozen V1. Original and anchored conditions used that *same* helper. No epsilon, detach, sanitization, or production source edit was used.

In one single-GPU full-Qwen H1 process, original and anchored C5 loss hashes matched exactly; all 24 rotation gradient hashes, all 24 one-step Adam theta hashes, final-token QDQ hashes and recurrent writeback checks matched. The exact gate **PASS** is in `analysis/zero_anchor_single_semantics.json`.

The frozen parent H4 failure was reused, not rerun: both ranks failed with `setStorage [8192,1,1,4] ... storage size 0` during C5 backward. One and only one two-rank H4 anchor intervention was run. Both ranks again failed within the same zero-storage error family, but the failing shape was **`[1024,4096]`**, a different tensor shape. The protocol's different-error stop rule classifies this as `NEW_FAILURE_MODE`, not a rescue. At the failure each rank had peak allocated **14,201,228,288 bytes** and peak reserved **17,190,354,944 bytes**; no OOM occurred. See both raw error JSONs and `analysis/h4_zero_anchor_result.json`.

The changed error shape shows altered execution, not identification of either failed storage owner or proof of a missing FSDP hook. The protocol required stopping H4 intervention at this new failure; no anchor variants or repeat runs were attempted. `ZERO_ANCHOR_H4_RESCUE=NO`, `H4_COMPATIBILITY_GATE=BLOCKED`.
""")
    write_report("FSDP_PRE_BACKWARD_HOOK_AUDIT.md", """
# FSDP pre-backward hook audit

Installed PyTorch 2.5.1 FSDP2 `_fsdp_state.py:316-323` registers a pre-backward hook on grad-requiring tensors in a wrapped module's ordinary output. Its `_fsdp_param_group.py:321-329` records `FSDP::pre_backward (<module FQN>)` while unsharding. These files were inspected read-only; no private state or hooks were mutated.

The first reduced H1 audit finished backward but its diagnostic-only scalar hash serializer rejected a 0-dimensional tensor. Its two error JSON files were preserved. The corrected v2 used a flattened scalar before byte reinterpretation and new evidence filenames. In fresh two-rank, two-real-GDN-decoder H1 comparisons, public output hooks plus CPU profiler changed **none** of the loss, input gradient, rotation gradient, or Adam theta hashes and added **0 bytes** to peak allocated GPU memory. Each rank recorded one forward entry/exit and one output-gradient hook per decoder and root; profiler recorded `FSDP::pre_backward (model.layers.1)` then `(model.layers.0)`. `HOOK_INSTRUMENTATION_NONINTERFERENCE=PASS` for this reduced passing graph only.

Neither the frozen parent H4 baseline nor the anchored H4 run has an FSDP pre-backward trace. After the anchored run produced a new backward failure shape, the frozen stop rule prohibited any hook-only H4 rerun. Consequently baseline-vs-anchor missing/restored hook comparison, first missing unit, and H4 hook anomaly are all **UNKNOWN**. The passing reduced H1 trace cannot be projected onto the failing full 9B H4 graph.
""")
    write_report("FINAL_REPORT.md", """
# QWEN_GDN_FSDP2_OUTPUT_HOOK_CAUSAL_CLOSURE_V1 — final report

**Status: BLOCKED; root cause UNRESOLVED.** The finite zero-output anchor was exactly neutral on single-GPU H1, but did not rescue the full 9B FSDP2 H4 backward. Both ranks failed on a **new** zero-storage tensor shape. This task stopped H4 testing at the required new-failure-mode gate. The old parent remains BLOCKED, with H4 backward FAIL and H32 feasibility NOT_DETERMINED.

## Provenance and scope

- Branch: `exp/qwen-gdn-fsdp2-output-hook-causal-closure-v1`, based on re-entry closure `f730e3dca7589d8cc272de839d6e5b3a77cd1eaa`; original FSDP feasibility parent `ad5007de643f55d92fccdb37562318f22a2e734b`.
- Parent H4 error hashes are in `configs/parent_manifest.json`. Runtime is PyTorch 2.5.1+cu121 and the composable `torch.distributed._composable.fsdp.fully_shard` API.
- No old experiment, production Qwen operator, C128 QDQ, recurrent writeback, C5 objective, rotation parameterization, frozen backbone, Torch/CUDA installation, or other user's process was changed.
- The current branch HEAD is the commit containing this file; use `git rev-parse HEAD` or the final handoff commit SHA to identify it without creating a self-referential hash.

## Required gates and observations

| Gate / question | Result |
|---|---|
| C5 depends on ordinary model output? | **PARTIAL**: no direct path through final logits or same block's returned output; upstream decoder outputs can feed downstream internal states |
| Exact zero anchor tensor | finite final captured-token model logits `[1,1,248320]`, grad-connected; `0.0 * logits.float().sum()` |
| Single-GPU original vs anchored H1 | loss exact; 24/24 gradients exact; 24/24 one-step Adam theta exact; QDQ/writeback exact — **PASS** |
| Frozen baseline full 9B H4 | both ranks `setStorage [8192,1,1,4] ... storage size 0` at C5 backward; not rerun |
| Anchored full 9B H4 | both ranks `setStorage [1024,4096] ... storage size 0` during backward — **NEW_FAILURE_MODE** |
| `ZERO_ANCHOR_H4_RESCUE` | **NO** |
| Anchored H4 peak allocated / reserved | 14,201,228,288 / 17,190,354,944 bytes per rank; no OOM |
| H1 reduced hook noninterference | **PASS**: exact loss/gradient/update; 0-byte GPU peak increase; two named FSDP pre-backward events/rank |
| Baseline / anchored H4 pre-backward trace | **NOT_CAPTURED / NOT_CAPTURED**; H4 hook-only rerun forbidden by stop rule |
| Baseline missing / anchor restored hook; first missing unit | **UNKNOWN / UNKNOWN** |
| `FSDP_PRE_BACKWARD_HOOK_ANOMALY`; `FAILED_STORAGE_OWNER` | **UNKNOWN / UNKNOWN** |
| Output-view/in-place signal | **NOT_ESTABLISHED**; conv weight and cache paths contain views/mutations, not proven to mutate a wrapped decoder output |
| `OUTPUT_HOOK_BYPASS_CAUSAL_SIGNAL`; output-hook hypothesis | **UNRESOLVED / UNRESOLVED** |
| DeepSpeed ZeRO-3 recommended next? | **NO, not justified by this protocol's evidence**; no DeepSpeed install or runtime test |
| H4 compatibility after intervention | **BLOCKED** |
| H8 / H16 / H32 / formal C5-C6 / AIME started | **NO / NO / NO / NO / NO** |
| Other user tasks interrupted | **NO** |

Static C5 graph and output/view audits are in their own reports; exact rank and hook evidence are in `analysis/`. The [related PyTorch FSDP2 output-view issue](https://github.com/pytorch/pytorch/issues/181832) illustrates a possible mechanism, but it is **not** proof that this Qwen graph has that bug. With no H4 hook trace and no anchor rescue, no failed tensor owner or causal hook bypass can be asserted. A future separately approved full-graph localization diagnostic would be more discriminating than installing another distributed runtime now.
""")
    H.mkdir(exist_ok=True)
    files = sorted(p for sub in (A, ROOT / "configs", ROOT / "scripts", R)
                   for p in sub.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix in (".json", ".py", ".md"))
    files.append(ROOT / "preregistration.json")
    with (H / "artifact_sha256.txt").open("x") as out:
        for path in sorted(files):
            out.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n")
    print(json.dumps({"verdict": verdict["status"], "root_cause": verdict["root_cause_confidence"],
                      "hashed_files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
