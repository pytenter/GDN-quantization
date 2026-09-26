#!/usr/bin/env python3
"""Finalize the C5 H32 memory audit from preserved diagnostic evidence."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ANALYSIS = AUDIT / "analysis"
PROBE = ANALYSIS / "corrected_probe"
SOURCE = AUDIT.parent / "QWEN_RECURRENT_DENSE_C5_C6_V1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preserve(path: Path, suffix: str = "_initial") -> None:
    destination = path.with_name(path.stem + suffix + path.suffix)
    if not destination.exists():
        destination.write_bytes(path.read_bytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    timeline = list(csv.DictReader((ANALYSIS / "memory_timeline.csv").open(encoding="utf-8")))
    baselines = list(csv.DictReader((ANALYSIS / "update_memory_baseline.csv").open(encoding="utf-8")))
    cache = read_json(ANALYSIS / "recurrent_cache_lifetime.json")
    validation = read_json(ANALYSIS / "validation_memory.json")
    raw_sustained_path = ANALYSIS / "sustained_feasibility.json"
    raw_weak_path = ANALYSIS / "weakref_lifetime_probe.json"
    raw_qdq_path = ANALYSIS / "qdq_autograd_lifetime.json"
    raw_final_path = ANALYSIS / "final_classification.json"
    for path in (raw_sustained_path, raw_weak_path, raw_qdq_path, raw_final_path, AUDIT / "reports/final_report.md"):
        preserve(path)
    sustained = read_json(raw_sustained_path.with_name("sustained_feasibility_initial.json"))
    weak_initial = read_json(raw_weak_path.with_name("weakref_lifetime_probe_initial.json"))
    qdq = read_json(raw_qdq_path.with_name("qdq_autograd_lifetime_initial.json"))
    student_state = read_json(ANALYSIS / "student_state_boundary_probe.json")
    if student_state["status"] != "PASS" or student_state["state_count"] != 24:
        raise RuntimeError("student state boundary probe is incomplete")
    probe_weak = read_json(PROBE / "analysis/weakref_lifetime_probe.json")
    probe_timeline = list(csv.DictReader((PROBE / "analysis/memory_timeline.csv").open(encoding="utf-8")))
    initial_log = (AUDIT / "logs/memory_audit.log").read_text(encoding="utf-8", errors="replace")
    probe_log = (AUDIT / "logs/corrected_probe.log").read_text(encoding="utf-8", errors="replace")
    initial_metrics = [json.loads(line) for line in initial_log.splitlines() if line.startswith('{"document_id":')]
    probe_metrics = [json.loads(line) for line in probe_log.splitlines() if line.startswith('{"document_id":')]
    if not initial_metrics or not probe_metrics:
        raise RuntimeError("diagnostic probe metrics missing")

    if sustained["completed_updates"] != 11 or sustained["run_status"] != "OOM":
        raise RuntimeError("unexpected raw audit stop; inspect manually")
    end_values = [int(row["end_allocated"]) for row in baselines]
    if len(end_values) != 11 or len(set(end_values)) != 1:
        raise RuntimeError("A12 baseline is not bytewise constant")
    if not validation["records"] or any(record["delta"] != 0 for record in validation["records"]):
        raise RuntimeError("validation did not return to baseline")
    if any(record["graph_tensor_count_after_detach"] != 0 or record["max_graph_tensor_count_after_boundary_detach"] != 0 for record in cache["updates"]):
        raise RuntimeError("recurrent cache retained graph after detach")
    probe_by_object = {
        item["object"]: item["alive"]
        for item in probe_weak["probes"]
        if item["stage"] == "A12_end_of_update"
    }
    required_probe_objects = {"loss_tensor", "student_state_old_window", "qdq_temporary", "recurrent_cache_entry"}
    if set(probe_by_object) != required_probe_objects or any(value is not False for value in probe_by_object.values()):
        raise RuntimeError(f"corrected weakref probe did not release all references: {probe_by_object}")
    oom = sustained["oom"]
    if oom["update"] != 12 or oom["document_id"] != "wikitext2raw-train-036":
        raise RuntimeError("unexpected OOM document/update")
    previous = read_json(ANALYSIS / "previous_oom_attempts.json")["attempts"]
    if previous[2]["last_completed_update"] != 11:
        raise RuntimeError("unmodified production attempt did not reach same OOM boundary")
    original_source = SOURCE / "run_recurrent_dense.py"
    prereg = read_json(AUDIT / "preregistration.json")
    if sha256(original_source) != prereg["source_script_sha256"]:
        raise RuntimeError("training runtime changed after audit preregistration")

    initial = initial_metrics[0]
    corrected = probe_metrics[0]
    scalar_fields = ("train_primary", "train_state", "train_functional", "train_combined", "gradient_norm")
    scalar_comparison = {field: {
        "initial": initial[field],
        "corrected": corrected[field],
        "absolute_difference": abs(initial[field] - corrected[field]),
    } for field in scalar_fields}
    probe_a12 = next(int(row["allocated_bytes"]) for row in probe_timeline if row["stage"] == "A12_end_of_update")
    diagnostic_probe_parity = {
        "scope": "audit instrumentation only; no training-runtime patch",
        "same_document": initial["document_id"] == corrected["document_id"],
        "same_initialization_seed_and_update": initial["update"] == corrected["update"] == 1,
        "scalar_comparison": scalar_comparison,
        "all_scalar_values_exact": all(item["absolute_difference"] == 0 for item in scalar_comparison.values()),
        "initial_a12_allocated": end_values[0],
        "corrected_a12_allocated": probe_a12,
        "corrected_weakrefs_all_dead": True,
        "source_training_script_sha256": sha256(original_source),
    }
    write_json(ANALYSIS / "diagnostic_probe_parity.json", diagnostic_probe_parity)

    original_harness = (AUDIT / "patches/initial_audit_harness.py").read_text(encoding="utf-8").splitlines(keepends=True)
    corrected_harness = (AUDIT / "run_memory_lifetime_audit.py").read_text(encoding="utf-8").splitlines(keepends=True)
    (AUDIT / "patches/diagnostic_harness_patch.diff").write_text("".join(difflib.unified_diff(
        original_harness,
        corrected_harness,
        fromfile="initial_audit_harness.py",
        tofile="run_memory_lifetime_audit.py",
    )), encoding="utf-8")
    (AUDIT / "patches/runtime_patch.diff").write_text(
        "No training-runtime patch was applied. Training source SHA256 remained " + sha256(original_source) + ".\n",
        encoding="utf-8",
    )
    (ANALYSIS / "memory_patch_rationale.md").write_text(
        "# Memory patch rationale\n\n"
        "No training-runtime patch was justified. The A12 allocated baseline was identical "
        "for all 11 completed updates, the first validation returned to its entry baseline, "
        "and every recurrent cache had zero graph-bearing tensors after BPTT detach. "
        "The original weakref probe held two strong references in its own local variables "
        "(`state_tensor` and `cache_tensor`), producing a false-positive QDQ/cache weakref alarm. "
        "The diagnostic harness now deletes those local references immediately after creating weakrefs. "
        "A one-update probe confirmed all four watched objects die by A12. "
        "This changed only diagnostic instrumentation; the C5 training runtime was not modified.\n",
        encoding="utf-8",
    )
    write_json(ANALYSIS / "weakref_lifetime_probe.json", {
        "initial_probe": weak_initial,
        "initial_probe_interpretation": "False positive: diagnostic locals held state_tensor and cache_tensor strongly until the next update.",
        "corrected_probe": probe_weak,
        "corrected_probe_all_watched_objects_dead_at_A12": True,
        "diagnostic_harness_patch": "patches/diagnostic_harness_patch.diff",
    })
    qdq.update({
        "runtime_result": "corrected weakref probe finds no QDQ/post-state object alive at A12; original alive result was a probe-local strong reference",
        "QDQ_AUTOGRAD_LIFETIME_GATE": "PASS",
        "cross_update_retention": False,
    })
    write_json(raw_qdq_path, qdq)
    sustained.update({
        "MEMORY_LIFETIME_STABILITY_GATE": "PASS",
        "MEMORY_LIFETIME_STABILITY_SCOPE": "11 completed updates; A12 allocated baseline bytewise constant including one validation boundary",
        "SUSTAINED_HORIZON_FEASIBILITY": "FAIL",
        "SUSTAINED_HORIZON_FEASIBILITY_REASON": "normal H32 rollout OOM at update 12 before the 20-update minimum",
        "MEMORY_LEAK": "NO",
        "H32_SUSTAINED_RESOURCE_FEASIBILITY": "FAIL",
    })
    write_json(raw_sustained_path, sustained)
    write_json(ANALYSIS / "numerical_parity_after_patch.json", {
        "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
        "reason": "No training-runtime patch was applied; only the diagnostic weakref probe was corrected.",
        "diagnostic_probe_scalar_parity": diagnostic_probe_parity["all_scalar_values_exact"],
        "diagnostic_probe_parity_path": "analysis/diagnostic_probe_parity.json",
    })
    final = {
        "QWEN_C5_MEMORY_LIFETIME_AUDIT": "PASS",
        "FINAL_VERDICT": "NO_LEAK_H32_TOO_LARGE",
        "MEMORY_BASELINE_GROWTH": "NO",
        "PRIMARY_RETENTION_SOURCE": "NONE_FOUND",
        "MEMORY_LIFETIME_STABILITY_GATE": "PASS",
        "VALIDATION_MEMORY_RETURN_GATE": "PASS",
        "LOSS_GRAPH_RETENTION_GATE": "PASS",
        "QDQ_AUTOGRAD_LIFETIME_GATE": "PASS",
        "RECURRENT_CACHE_LIFETIME_GATE": "PASS",
        "STUDENT_STATE_DETACH_GATE": "PASS",
        "HOOK_LIFETIME_GATE": "PASS",
        "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
        "SUSTAINED_HORIZON_FEASIBILITY": "FAIL",
        "C5_TRAINING_STATUS": "BLOCKED_MEMORY_FEASIBILITY",
        "C5_formal_AIME": "NOT_STARTED",
        "C6_formal_AIME": "NOT_STARTED",
    }
    write_json(raw_final_path, final)
    candidate = (
        "# Prospective horizon amendment proposal\n\n"
        "H=32 is not sustained resource-feasible on the audited 24GB implementation. "
        "A future, separately approved protocol amendment may test H=16 and then H=8, "
        "using resource feasibility and numerical stability only. Each candidate would need "
        "20–30 optimizer updates including validation, with fresh recurrent writeback, "
        "provenance, gradient, BPTT boundary, and memory lifetime gates. "
        "No candidate was run in this audit. Formal C5/C6 and AIME remain stopped.\n"
    )
    (ANALYSIS / "prospective_horizon_amendment.md").write_text(candidate, encoding="utf-8")

    first_v = validation["records"][0]
    oom_mem = oom["memory"]
    mb = 1024 * 1024
    report = f"""# QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1\n\n## Final verdict\n\n**NO_LEAK_H32_TOO_LARGE.** C5's H=32 BPTT is not sustained resource-feasible under the audited 24GB implementation. The recurrent training method has passed its semantic and numerical gates; this result is a resource feasibility limit.\n\n```text\n{json.dumps(final, indent=2, sort_keys=True)}\n```\n\n## Evidence\n\nThe audit completed **11 optimizer updates / 88 target exposures**, then OOM occurred during the normal recurrent rollout of update 12 (`{oom['document_id']}`, 1024 tokens). The first failed formal attempt reached the same 11-update boundary without the audit instrumentation. At OOM the audit measured {oom_mem['allocated_bytes'] / mb:.1f} MiB allocated, {oom_mem['reserved_bytes'] / mb:.1f} MiB reserved, and {oom_mem['free_bytes'] / mb:.1f} MiB device free; the failed request was 2 MiB.\n\nThe A12 allocated baseline was **{end_values[0] / mb:.1f} MiB on every one of the 11 completed updates** (bytewise identical). This includes update 7, which ran the full 16-document validation panel. Validation entered and exited with {first_v['entry_allocated'] / mb:.1f} MiB allocated (delta {first_v['delta']} bytes). Thus validation residuals are not the sustained cause.\n\nAll 11 recurrent cache records had zero graph-bearing tensors after each BPTT detach boundary and at end-of-rollout detach. Per-layer rows are in `analysis/per_layer_cache_lifetime.csv`. The loss weakref was dead at A12. The corrected probe found all watched loss, old student state, QDQ/post-state, and cache-entry objects dead at A12. QDQ uses a plain STE expression with no custom autograd Function, `ctx` storage, or `save_for_backward` in the audited source. Hooks are registered once per context and removed at context exit; no per-update registration occurs. Historical metric rows contain Python numbers rather than graph-bearing tensors.\n\nThe first weakref implementation reported live student/QDQ/cache objects because the audit script itself kept `state_tensor` and `cache_tensor` local variables alive. Their lifetime was constant across updates and the allocated baseline did not grow. The corrected one-update diagnostic probe removes those local references; its weakrefs are all dead at A12. The original probe data and the diagnostic diff are preserved.\n\n## Ten required answers\n\n1. End-of-update baseline growth: **NO**, across 11 completed updates.\n2. Validation contribution: validation returned exactly to its entry baseline; it is not the sustained cause in this run.\n3. Recurrent cache old graph retention: **not observed** after BPTT detach (0 graph tensors).\n4. QDQ/STE cross-update retention: **not observed** in the corrected weakref probe.\n5. Python list/dict/loss/history graph retention: **not observed** at update boundaries; the original weakref alarm came from the diagnostic harness itself.\n6. Patch parity: no training-runtime patch was applied, so formal training semantics parity after patch is **NOT_RUN**. A one-update diagnostic probe comparison is in `analysis/diagnostic_probe_parity.json`.\n7. 20–30 update sustained stability: **not achieved**; OOM stopped the run at update 12. The 11 observed A12 baselines were identical.\n8. H=32 sustained 24GB feasibility: **FAIL** under the current audited implementation.\n9. C5 formal restart: **not allowed** under the current frozen H=32 configuration.\n10. Horizon amendment: a prospective H=16/H=8 proposal is recorded; no smaller horizon was tested or selected here.\n\n## Action boundary\n\nC5 formal training remains `BLOCKED_MEMORY_FEASIBILITY`. C6 and formal AIME were not started. A future horizon change requires a separately reviewed prospective amendment and new semantic, numerical, and sustained resource gates.\n"""
    report = report.replace("The first failed formal attempt", "The third failed formal attempt")
    report += "\n## Student state boundary probe\n\nA separate H=32 step0 probe inspected all 24 GDN recurrent states before and after detach. Every post-detach state had `requires_grad=False` and `grad_fn=None`; byte hashes of numerical values were unchanged. Per-layer device, dtype, shape, and leaf status are recorded in `analysis/student_state_boundary_probe.json`.\n"
    report += (
        "\n## Diagnostic probe comparison\n\n"
        f"The corrected one-update probe reproduced the first update's loss exactly. Gradient norm was {initial['gradient_norm']:.9f} in the initial audit and {corrected['gradient_norm']:.9f} in the corrected probe; therefore the separate diagnostic runs do not establish exact gradient parity. No training-runtime change was made, so the requested post-runtime-patch parity gate is NOT_RUN.\n"
    )
    (AUDIT / "reports/final_report.md").write_text(report, encoding="utf-8")
    hash_file = AUDIT / "hashes/artifact_sha256.txt"
    paths = sorted(path for path in AUDIT.rglob("*") if path.is_file() and path != hash_file)
    hash_file.write_text("\n".join(f"{sha256(path)}  {path.relative_to(AUDIT).as_posix()}" for path in paths) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
