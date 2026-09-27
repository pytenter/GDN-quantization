#!/usr/bin/env python3
"""Summarize immutable diagnostic JSON; never infer a parent storage owner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
HASHES = ROOT / "hashes"


def read(name: str) -> dict:
    return json.loads((ANALYSIS / name).read_text())


def write_json(name: str, data: dict) -> None:
    path = ANALYSIS / name
    with path.open("x") as out:
        json.dump(data, out, indent=2, sort_keys=True)
        out.write("\n")


def write_report(name: str, body: str) -> None:
    with (REPORTS / name).open("x") as out:
        out.write(body.strip() + "\n")


def cell(prefix: str, n: int, reshard: int) -> dict:
    stem = f"{prefix}_N{n}_reshard{reshard}_rank"
    paths = []
    records = []
    for rank in (0, 1):
        options = [ANALYSIS / f"{stem}{rank}.json", ANALYSIS / f"{stem}{rank}.error.json"]
        found = [p for p in options if p.exists()]
        if len(found) != 1:
            raise RuntimeError(f"expected one record per rank: {stem}{rank}, got {found}")
        paths.append(found[0].name)
        records.append(json.loads(found[0].read_text()))
    statuses = [r.get("status") for r in records]
    if statuses[0] != statuses[1]:
        raise RuntimeError(f"rank mismatch: {paths}")
    return {
        "rank_files": paths,
        "status": statuses[0],
        "error_type": [r.get("type") for r in records] if statuses[0] != "PASS" else None,
        "error_message_first_line": [r.get("message", "").splitlines()[0] for r in records]
        if statuses[0] != "PASS" else None,
        "peak_allocated_bytes_by_rank": [r.get("memory_after_backward", {}).get("peak_allocated_bytes") for r in records]
        if statuses[0] == "PASS" else None,
        "peak_reserved_bytes_by_rank": [r.get("memory_after_backward", {}).get("peak_reserved_bytes") for r in records]
        if statuses[0] == "PASS" else None,
    }


def main() -> None:
    HASHES.mkdir(exist_ok=True)
    primary = {f"N{n}_reshard{r}": cell("gdn_none", n, r) for r in (1, 0) for n in (1, 2, 3, 4)}
    if not all(primary[f"N{n}_reshard{r}"]["status"] == "PASS" for r in (1, 0) for n in (1, 2, 3)):
        raise RuntimeError("unexpected primary control outcome")
    if any("storage" in " ".join(primary[f"N4_reshard{r}"]["error_message_first_line"]).lower() for r in (1, 0)):
        raise RuntimeError("N4 failure must not be misclassified as parent storage failure")
    canonical = {
        mode: {f"N{n}_reshard{r}": cell(f"canonical_{mode}", n, r)
               for n, r in ((1, 1), (2, 1), (4, 1), (4, 0))}
        for mode in ("cache_only", "qdq_rotation")
    }
    decoder = {f"N{n}_reshard{r}": cell("decoder", n, r)
               for n, r in ((1, 1), (2, 1), (4, 1), (4, 0))}
    stacks = {f"stack{blocks}_N{n}_reshard{r}": cell(f"stack{blocks}", n, r)
              for blocks, n, r in ((2, 1, 1), (2, 4, 1), (2, 4, 0),
                                   (3, 1, 1), (3, 4, 1), (3, 4, 0))}
    if any(x["status"] != "PASS" for group in (canonical["cache_only"], canonical["qdq_rotation"], decoder, stacks)
           for x in group.values()):
        raise RuntimeError("unexpected canonical/decoder/stack failure")
    raw_cache = {
        "N1_reshard1": cell("gdn_recurrent", 1, 1),
        "N2_reshard1": cell("gdn_recurrent", 2, 1),
        "N2_reshard0": cell("gdn_recurrent", 2, 0),
        "interpretation": "Raw DynamicCache copy_ is not the frozen V1 differentiable writeback; N2 is an autograd version-counter error, not the parent zero-storage failure.",
    }
    write_json("minimal_reentry_matrix.json", {
        "model": "real checkpoint Qwen3.5 layer0 GDN, identical synthetic BF16 input and scalar loss in fresh processes",
        "same_module_instance": True,
        "one_backward_per_reentry_chain": True,
        "primary_no_recurrent_state": primary,
        "raw_cache_diagnostic": raw_cache,
        "canonical_recurrent_writeback": canonical,
        "real_decoder": decoder,
        "nested_real_block_stacks": stacks,
        "minimum_reentry_count_with_parent_storage_failure": "NOT_FOUND_IN_REDUCED_TESTS",
        "primary_N4_failure": "input-gradient zero under synthetic scalar loss on both reshard controls; backward completed; distinct from parent setStorage failure",
    })

    traces = [read(f"lifecycle_stack3_N4_rank{rank}.json") for rank in (0, 1)]
    for trace in traces:
        if trace["status"] != "PASS" or not trace["loss_matches_uninstrumented_baseline"]:
            raise RuntimeError("lifecycle trace invalid")
    trace_summary = []
    saved_summary = []
    for trace in traces:
        events = trace["events"]
        zero = [e.get("event") for e in events if e.get("conv_weight", {}).get("local_storage_nbytes") == 0]
        hooks = trace["hook_counts"]
        if hooks != {"forward_pre": 12, "forward_post": 12, "module_backward_pre": 12, "module_backward_post": 12}:
            raise RuntimeError("hook counts unexpected")
        pack = trace["special_saved_pack"]
        unpack = trace["special_saved_unpack"]
        conv_pack = [v for v in pack if v["shape"] == [8192, 1, 4]]
        conv_unpack = [v for v in unpack if v["shape"] == [8192, 1, 4]]
        if len(conv_pack) != 12 or len(conv_unpack) != 12:
            raise RuntimeError("conv view count unexpected")
        trace_summary.append({
            "rank": trace["rank"], "source": f"lifecycle_stack3_N4_rank{trace['rank']}.json",
            "event_count": len(events), "hook_counts": hooks,
            "zero_storage_at_public_hook_or_outer_boundary": zero,
            "first_event": events[0]["event"], "last_event": events[-1]["event"],
            "conv_weight_storage_nbytes_at_boundaries": sorted(set(
                e.get("conv_weight", {}).get("local_storage_nbytes") for e in events if "conv_weight" in e)),
            "peak_allocated_bytes": trace["memory_after_backward"]["peak_allocated_bytes"],
        })
        saved_summary.append({
            "rank": trace["rank"], "all_saved_tensor_count": trace["saved_tensor_count"],
            "conv_weight_shape_8192_1_4_pack_count": len(conv_pack),
            "conv_weight_shape_8192_1_4_unpack_count": len(conv_unpack),
            "conv_pack_all_views": all(v["is_view"] for v in conv_pack),
            "conv_unpack_all_views": all(v["is_view"] for v in conv_unpack),
            "conv_pack_storage_nbytes": sorted(set(v["storage_nbytes"] for v in conv_pack)),
            "conv_unpack_storage_nbytes": sorted(set(v["storage_nbytes"] for v in conv_unpack)),
            "parent_failing_shape_8192_1_1_4_seen": any(
                v["shape"] == [8192, 1, 1, 4] for v in pack + unpack),
            "storage_match_to_named_conv_weight_in_passing_reduced_graph": all(
                bool(v["direct_conv_storage_owner_matches"]) for v in conv_pack),
        })
    write_json("fsdp_lifecycle_trace_summary.json", {
        "condition": "3 real GDN decoder blocks x4; nested FSDP2; reshard_after_forward=True; backward PASS",
        "instrumentation": "public module full-backward hooks + saved_tensors_hooks; can perturb view/storage lifetimes",
        "ranks": trace_summary,
        "parent_failure_lifecycle_captured": False,
    })
    write_json("pre_backward_hook_audit.json", {
        "condition": "passing reduced 3-block N4 reshard=True only",
        "public_module_backward_pre_hooks_per_rank": 12,
        "public_module_backward_post_hooks_per_rank": 12,
        "internal_fsdp_pre_backward_unshard_hook_count": "UNKNOWN",
        "parent_h4_internal_hook_count": "UNKNOWN",
        "interpretation": "Public module hooks are not FSDP's private unshard hooks; counts cannot prove parent failure's unshard sequence.",
    })
    write_json("saved_tensor_view_audit.json", {
        "condition": "passing reduced 3-block N4 reshard=True only",
        "ranks": saved_summary,
        "parent_failed_saved_tensor_identity": "UNKNOWN",
        "parent_saved_tensor_is_parameter_view": "UNKNOWN",
        "caveat": "saved_tensors_hooks may change tensor/view lifetime; positive storage in a passing reduced graph does not diagnose parent H4.",
    })
    write_json("failure_storage_owner.json", {
        "FAILED_STORAGE_OWNER": "UNKNOWN", "failed_tensor_parameter_or_view_name": "UNKNOWN",
        "parent_error_shape": [8192, 1, 1, 4],
        "candidate": "linear_attn.conv1d.weight view only: exact element-count match and Qwen squeeze/unsqueeze source path",
        "evidence_against_overclaim": "Parent traceback ends at autograd engine; no failing model/FSDP frame or saved-tensor identity; reduced passing graph shows conv weight views but not parent failing shape.",
        "first_storage_invalid_lifecycle_event": "UNKNOWN",
    })
    write_json("conv1d_weight_view_audit.json", {
        "checkpoint_weight_shape": [8192, 1, 4],
        "model_source": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "model_source_lines": {"weight_squeeze": [461, 473], "weight_unsqueeze": [212, 232]},
        "passing_reduced_graph_saved_conv_view_count_per_rank": 12,
        "failing_parent_tensor_same_as_this_view": "UNKNOWN",
    })
    write_json("reshard_intervention.json", {
        "reduced_N4_canonical_cache_qdq_rotation": {str(r): canonical["qdq_rotation"][f"N4_reshard{r}"] for r in (1, 0)},
        "reduced_N4_decoder": {str(r): decoder[f"N4_reshard{r}"] for r in (1, 0)},
        "reduced_N4_stack3": {str(r): stacks[f"stack3_N4_reshard{r}"] for r in (1, 0)},
        "RESHARD_REENTRY_CAUSAL_SIGNAL": "NO_IN_TESTED_REDUCED_CONDITIONS",
        "parent_H4_causality": "UNRESOLVED",
        "full_9B_no_reshard_diagnostic": "MEMORY_UNSAFE_NOT_RUN; no reduced causal signal, no legal intervention, full 9B retention likely defeats sharding memory objective",
    })
    write_json("wrap_variant_results.json", {
        "tested": "single GDN, one decoder FSDP unit, nested per-block+root stack of 2 and 3 real GDN decoders; both reshard controls at N4",
        "all_tested_canonical_N4": "PASS",
        "root_only": "NOT_RUN_NO_CAUSAL_SIGNAL",
        "grouped_blocks": "NOT_RUN_NO_CAUSAL_SIGNAL",
        "parent_32_block_wrap_plan": "not rerun or modified",
    })
    write_json("h4_confirmation.json", {
        "H4_COMPATIBILITY_GATE": "NOT_RUN", "H4_peak_memory": None,
        "reason": "No legal intervention found; full-model H4 control preregistered only after positive reduced reshard signal and memory safety.",
        "parent_H4_backward": "FAIL_SETSTORAGE_ZERO_STORAGE_BOTH_RANKS",
    })
    verdict = {
        "experiment": ROOT.name, "status": "BLOCKED_ROOT_CAUSE_UNRESOLVED",
        "root_cause_level": "UNRESOLVED", "FAILED_STORAGE_OWNER": "UNKNOWN",
        "minimal_parent_storage_failure_reproducer_found": False,
        "minimum_reentry_count_with_parent_storage_failure": "NOT_FOUND_IN_REDUCED_TESTS",
        "recurrent_state_required_for_parent_failure": "UNKNOWN", "QDQ_REQUIRED_FOR_FAILURE": "UNKNOWN",
        "rotation_graph_required_for_parent_failure": "UNKNOWN",
        "first_storage_invalid_lifecycle_event": "UNKNOWN",
        "parent_pre_backward_unshard_observed": "UNKNOWN",
        "parent_failed_saved_tensor_is_parameter_view": "UNKNOWN",
        "conv1d_involved_in_parent_failure": "UNKNOWN",
        "RESHARD_REENTRY_CAUSAL_SIGNAL": "NO_IN_TESTED_REDUCED_CONDITIONS",
        "PYTORCH_FSDP2_VERSION_COMPATIBILITY_SUSPECTED": "NO_POSITIVE_EVIDENCE",
        "legal_fix_found": False, "fix_preserves_sharding_memory_benefit": "NOT_APPLICABLE",
        "H4_COMPATIBILITY_GATE": "NOT_RUN", "H4_RETRY_AUTHORIZED": False,
        "H8_started": False, "H16_started": False, "H32_started": False,
        "formal_C5_C6_started": False, "AIME_started": False, "DeepSpeed_started": False,
        "other_user_tasks_interrupted": False,
        "parent_status_retained": "BLOCKED", "parent_H4_backward": "FAIL",
        "parent_H32_feasibility": "NOT_DETERMINED",
        "FSDP2_recurrent_training_feasibility": "NOT_ESTABLISHED",
    }
    write_json("final_verdict.json", verdict)
    REPORTS.mkdir(exist_ok=True)
    write_report("FSDP_REENTRY_MINIMAL_REPRODUCER.md", """
# Minimal real-path re-entry diagnostics

All tests used two ranks on physical GPUs 0/1, exact Qwen3.5 checkpoint tensors, one fresh process per condition and one backward after N invocations of the same FSDP2-wrapped module. Inputs were synthetic, not AIME. `analysis/minimal_reentry_matrix.json` links all 60 rank-level records.

The preregistered real GDN, no-cache factorial passed N1/N2/N3 with either reshard setting. N4 completed backward but failed the diagnostic nonzero input-gradient gate under the synthetic squared-output scalar loss on **both** settings; this is not the parent's zero-storage error. An exploratory raw `DynamicCache` N2 failed a different autograd in-place version check and is not the frozen V1 differentiable writeback. With the exact V1 differentiable cache, N1/N2/N4 passed. Adding canonical C128 QDQ and Hadamard theta=0 Key-side rotation also passed N1/N2/N4. Real full decoder layer 0 passed N1/N2/N4. Nested FSDP2 stacks of two and three real GDN decoder blocks passed N1/N4. N4 reshard=True and reshard=False both passed in every canonical cache/decoder/stack comparison.

No reduced condition reproduced `setStorage ... storage size 0`; the minimum parent-style failing re-entry count is **not found**. This negative result cannot establish that the full 9B/H4 parent failure was not an FSDP lifecycle issue.
""")
    write_report("STORAGE_LIFECYCLE_AUDIT.md", """
# Storage and backward lifecycle audit

The parent traceback captures `c5_loss.backward()` on both ranks but exposes no failing Python model frame, FSDP frame or tensor identity. Therefore `FAILED_STORAGE_OWNER=UNKNOWN`; `linear_attn.conv1d.weight` is only a candidate based on `[8192,1,4]` checkpoint shape and Qwen source `squeeze(1)` → causal-convolution `unsqueeze(1)`.

In a **passing** three-real-decoder N4, reshard=True graph, public module hooks on each rank counted 12 forward entries, 12 forward exits, 12 module backward-pre and 12 module backward-post events. The sharded conv-weight local storage remained nonzero (32,768 bytes) at outer boundaries; forward entry/registered hook observations also saw full 65,536-byte storage. The saved-tensor hook recorded 12 `[8192,1,4]` BF16 conv-weight views per rank, all with 65,536-byte storage at pack and unpack. No `[8192,1,1,4]` saved tensor was seen in that passing reduced graph.

The public module hooks are **not** PyTorch FSDP's internal pre-backward unshard hooks. Their count does not establish internal hook count or first invalidation in the failed parent. `saved_tensors_hooks` and full-backward hooks may themselves perturb view/storage lifetime. No production operator or framework code was modified. Parent first invalid storage event, actual failed view/base and internal unshard behavior remain unknown. Raw traces are `analysis/lifecycle_stack3_N4_rank{0,1}.json`.
""")
    write_report("RESHARD_CAUSAL_TEST.md", """
# Reshard single-factor intervention

In the same reduced canonical N4 model, both `reshard_after_forward=True` and `False` passed backward. The three-block nested stack peaked at 2,117,376,000 allocated bytes with reshard=True and 2,907,156,480 with reshard=False on rank 0: retaining full parameters increased peak allocation by 789,780,480 bytes (about 37%). This shows a memory cost without a correctness rescue in the tested reduced graph.

The preregistered positive signal (N1 True pass, N4 True parent-style fail, N4 False pass) did not occur. `RESHARD_REENTRY_CAUSAL_SIGNAL=NO_IN_TESTED_REDUCED_CONDITIONS`; parent H4 causality remains unresolved. Root-only/grouped wrapping and full 9B H4/no-reshard were not run because the reduced signal was absent, no legal fix existed, and full-model parameter retention would risk the 24GB memory budget. This is not a proof that PyTorch 2.5.1 is bug-free or that parent H4 can train.
""")
    write_report("FINAL_REPORT.md", """
# QWEN_GDN_FSDP2_RECURRENT_REENTRY_CLOSURE_V1 — final report

**Verdict: BLOCKED; root cause UNRESOLVED.** Parent H4 zero-storage failure remains real on both ranks; it was not an OOM. This task found no minimal reduced reproduction and no legal fix. Consequently H4 compatibility was not rerun and H32 feasibility remains **NOT_DETERMINED**. Do not start formal training on this evidence.

## Provenance

- Branch: `exp/qwen-gdn-fsdp2-recurrent-reentry-closure-v1`; parent `ad5007de643f55d92fccdb37562318f22a2e734b`. The current branch HEAD and commit list are given by Git; the report does not hard-code its own future commit ID.
- Parent status: BLOCKED. H1 QDQ/writeback, gradient/update and same-topology repeatability passed; H4 backward failed; FSDP2 recurrent training feasibility NOT_ESTABLISHED.
- Parent rank error SHA256 and exact wrap plan/script hashes: `configs/parent_failure_manifest.json`. Parent error has `setStorage sizes [8192,1,1,4] ... storage size 0` at `c5_loss.backward()` on both ranks, without a deeper Python model/FSDP frame.
- Runtime: PyTorch 2.5.1+cu121, CUDA 12.1, NCCL 2.21.5, Transformers 5.16.0.dev0, Triton 3.1.0. FLA version is unavailable, not guessed. API: `torch.distributed._composable.fsdp.fully_shard`, two ranks on physical GPUs 0/1. Installed FSDP source hashes are in the analysis manifest.

## Direct answers

| Question | Result |
|---|---|
| `FAILED_STORAGE_OWNER`; failed parameter/view | **UNKNOWN**; `linear_attn.conv1d.weight` is a shape/source candidate only |
| Parent-style minimal reproducer | NO, in tested real reduced paths |
| Minimum re-entry count causing parent-style storage failure | NOT_FOUND_IN_REDUCED_TESTS |
| No-cache GDN N1/N2/N4, reshard=True | PASS / PASS / nonzero-gradient gate FAIL, **not storage** |
| No-cache GDN N1/N2/N4, reshard=False | PASS / PASS / same nonzero-gradient gate FAIL, **not storage** |
| Canonical V1 cache + QDQ/rotation N1/N2/N4, reshard=True | PASS / PASS / PASS |
| Canonical V1 cache + QDQ/rotation N4, reshard=False | PASS |
| One real decoder N1/N2/N4 reshard=True; N4 reshard=False | PASS / PASS / PASS; PASS |
| Nested 2/3 real GDN decoders N1/N4, reshard=True; N4 reshard=False | All PASS |
| Recurrent state, QDQ or rotation *required for parent failure* | UNKNOWN / UNKNOWN / UNKNOWN |
| First storage-invalid lifecycle event | UNKNOWN |
| FSDP internal pre-backward unshard in failed parent | UNKNOWN; passing reduced graph only has public module hook counts |
| Failed parent saved tensor a parameter view? | UNKNOWN; passing reduced graph contains conv-weight views |
| Conv1d involved in failed parent? | UNKNOWN; candidate, not identified |
| Reshard/re-entry causal signal | NO in tested reduced conditions; parent unresolved |
| PyTorch-version compatibility suspected? | NO positive evidence; not excluded |
| Legal fix / preserves parameter-sharding memory benefit | NO / not applicable |
| H4 after fix / peak memory | NOT_RUN / not available |
| H8 / H16 / H32 / formal C5-C6 / AIME / DeepSpeed started | NO / NO / NO / NO / NO / NO |
| Other user tasks interrupted | NO |

The three-block N4 peak allocated memory increased from 2,117,376,000 bytes (reshard=True) to 2,907,156,480 bytes (False) on rank 0, without changing the PASS outcome. Direct lifecycle evidence and caveats are in `reports/STORAGE_LIFECYCLE_AUDIT.md`; exact condition/rank files in `analysis/minimal_reentry_matrix.json`.

## Stop point

The parent failure owner and first invalidation remain unobserved; the critical parent 9B/H4 backward and reduced synthetic loss/graph are not equivalent. There is no authorized H4 retry or full-model no-reshard control. No production Qwen, FSDP, QDQ, rotation or cache semantics were changed. New research would require a separately authorized diagnostic capable of observing the parent failing graph without violating the 24GB memory gate. This task stops here.
""")
    write_json("installed_fsdp_source_manifest.json", {
        "torch": "2.5.1+cu121", "read_only": True,
        "files_sha256": {
            "fully_shard.py": "8ebe40ba29ece84fd40ea2ced42358132eb641ae137fb68ce985dc68e06f4bc4",
            "_fsdp_state.py": "a6c39cf091faefa5940c4e4b52ad8a7272e8a436420d788e13a70c08267290fc",
            "_fsdp_param_group.py": "d91b33eea3e6276a670bb729f84d8f825d2cb443763262fa43ce73b2af8d73fa",
        },
        "upgrade_test": "NOT_RUN",
    })
    paths = sorted(p for folder in (ROOT / "configs", ROOT / "scripts", ANALYSIS, REPORTS)
                   for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix in (".json", ".py", ".md"))
    paths.append(ROOT / "preregistration.json")
    with (HASHES / "artifact_sha256.txt").open("x") as out:
        for p in sorted(paths):
            out.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n")
    print(json.dumps({"status": verdict["status"], "root_cause_level": verdict["root_cause_level"],
                      "evidence_files_hashed": len(paths), "rank_json_count": 60}, sort_keys=True))


if __name__ == "__main__":
    main()
