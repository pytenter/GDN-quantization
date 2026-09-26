"""Finalize the V2 H32 topology gate from preserved one-step evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
HASHES = ROOT / "hashes"


def read(name: str) -> dict:
    return json.loads((ANALYSIS / name).read_text())


def write(name: str, value: dict) -> None:
    (ANALYSIS / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_exact(a: dict, b: dict, field: str) -> tuple[int, int]:
    if set(a["records"]) != set(b["records"]):
        raise RuntimeError("trace keys differ")
    keys = sorted(a["records"])
    return sum(a["records"][key].get(field) == b["records"][key].get(field) for key in keys), len(keys)


def main() -> None:
    if any((ANALYSIS / name).exists() for name in (
        "single_gpu_h32_reference.json", "dual_gpu_h32_parity.json",
        "gradient_parity.json", "one_step_update_parity.json",
        "topology_horizon_selection.json")):
        raise RuntimeError("refusing to overwrite finalized gate artifacts")
    a = read("one_step_c5_single_repeat1.json")
    b = read("one_step_c5_single_repeat2.json")
    c = read("one_step_c5_dual_split16.json")
    for item in (a, b, c):
        if item["condition"] != "C5":
            raise RuntimeError("unexpected condition")
    discrete_fields = ["source_document_id", "input_token_ids", "target_positions", "trace_token_indices"]
    discrete = all(a[key] == b[key] == c[key] for key in discrete_fields)
    trace_fields = ["consumed_prev_state_sha256", "pre_qdq_state_sha256", "scale_sha256",
                    "qcodes_sha256", "post_qdq_state_sha256"]
    trace_counts = {field: {
        "single_repeat": count_exact(a, b, field),
        "single_vs_dual": count_exact(a, c, field),
    } for field in trace_fields}
    forward_exact = (
        discrete
        and all(value["single_repeat"][0] == value["single_repeat"][1]
                and value["single_vs_dual"][0] == value["single_vs_dual"][1]
                for value in trace_counts.values())
        and a["loss_components"] == b["loss_components"] == c["loss_components"]
        and a["total_training_loss"] == b["total_training_loss"] == c["total_training_loss"]
        and a["total_primary_loss"] == b["total_primary_loss"] == c["total_primary_loss"]
    )
    gradient_fields = ["raw_rotation_gradient_sha256", "clipped_rotation_gradient_sha256"]
    update_fields = ["post_optimizer_theta_sha256", "post_optimizer_rotation_sha256"]
    gradient_counts = {field: {
        "single_repeat_matching_layers": sum(a[field][key] == b[field][key] for key in a[field]),
        "single_vs_dual_matching_layers": sum(a[field][key] == c[field][key] for key in a[field]),
        "total_layers": len(a[field]),
    } for field in gradient_fields + update_fields}
    gradient_exact = all(gradient_counts[field]["single_vs_dual_matching_layers"] ==
                         gradient_counts[field]["total_layers"] for field in gradient_fields)
    update_exact = all(gradient_counts[field]["single_vs_dual_matching_layers"] ==
                       gradient_counts[field]["total_layers"] for field in update_fields)
    single_repeat_gradient_exact = all(gradient_counts[field]["single_repeat_matching_layers"] ==
                                       gradient_counts[field]["total_layers"] for field in gradient_fields)
    prior_tolerance = "NONE_FOUND_IN_PREEXISTING_V2_V1_OR_MEMORY_AUDIT_PROTOCOL_ARTIFACTS"
    source_files = ["one_step_c5_single_repeat1.json", "one_step_c5_single_repeat2.json",
                    "one_step_c5_dual_split16.json"]
    sources = {name: digest(ANALYSIS / name) for name in source_files}
    write("single_gpu_h32_reference.json", {
        "condition": "C5", "reference_files": source_files[:2],
        "reference_sha256": {key: sources[key] for key in source_files[:2]},
        "forward_repeatability": "PASS" if forward_exact else "FAIL",
        "gradient_repeatability": "PASS" if single_repeat_gradient_exact else "FAIL",
        "gradient_norms": [a["gradient_norm_after_clip"], b["gradient_norm_after_clip"]],
        "interpretation": "One-step single-GPU gradient is not bitwise reproducible; this does not license a new tolerance.",
    })
    write("dual_gpu_h32_parity.json", {
        "condition": "C5", "split_block": 16, "source_sha256": sources,
        "discrete_exact": discrete, "trace_match_counts": trace_counts,
        "loss_components_exact": a["loss_components"] == c["loss_components"],
        "training_loss": {"single": a["total_training_loss"], "dual": c["total_training_loss"]},
        "H32_SINGLE_VS_DUAL_FORWARD_PARITY": "PASS" if forward_exact else "FAIL",
        "DUAL_GPU_REAL_RECURRENT_WRITEBACK_GATE": c["REAL_RECURRENT_WRITEBACK_GATE"],
        "DUAL_GPU_BPTT_BOUNDARY_GATE": c["BPTT_BOUNDARY_GATE"],
        "DUAL_GPU_OPTIMIZER_COLOCATION_GATE": c["OPTIMIZER_COLOCATION_GATE"],
    })
    write("gradient_parity.json", {
        "condition": "C5", "source_sha256": sources,
        "gradient_match_counts": {key: gradient_counts[key] for key in gradient_fields},
        "gradient_norms": {"single_repeat1": a["gradient_norm_after_clip"],
                           "single_repeat2": b["gradient_norm_after_clip"],
                           "dual": c["gradient_norm_after_clip"]},
        "preexisting_frozen_numerical_tolerance": prior_tolerance,
        "H32_SINGLE_VS_DUAL_GRADIENT_PARITY": "PASS" if gradient_exact else "FAIL",
    })
    write("one_step_update_parity.json", {
        "condition": "C5", "source_sha256": sources,
        "update_match_counts": {key: gradient_counts[key] for key in update_fields},
        "preexisting_frozen_numerical_tolerance": prior_tolerance,
        "H32_SINGLE_VS_DUAL_UPDATE_PARITY": "PASS" if update_exact else "FAIL",
    })
    if gradient_exact or update_exact:
        raise RuntimeError("unexpected exact gradient/update result: review before verdict")
    order = [(2, 128), (4, 128), (2, 64), (4, 64), (2, 32), (4, 32)]
    write("topology_horizon_selection.json", {
        "selection_rule": "MAXIMIZE_HORIZON_THEN_MINIMIZE_GPU_COUNT",
        "decision_order": [{"gpu_count": g, "gradient_horizon": h,
                            "C5": "NOT_RUN", "C6": "NOT_RUN",
                            "verdict": "NOT_RUN_PARITY_GATE_FAILED"} for g, h in order],
        "final_gpu_count": None, "final_gradient_horizon": None,
        "DUAL_GPU_TRAINING_SEMANTICS_GATE": "FAIL",
        "C6_PARITY": "NOT_RUN_AFTER_C5_SEMANTICS_GATE_FAILED",
        "4GPU_TOPOLOGY_GATE": "NOT_RUN_2GPU_FOUNDATION_GATE_FAILED",
        "status": "BLOCKED_WAITING_FOR_MANUAL_DECISION",
    })
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "final_report.md").write_text(
        "# QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2 — stopped at H32 parity\n\n"
        "The 4-GPU resource amendment was registered prospectively. Its topology/horizon search "
        "cannot start because the mandatory 2-GPU H32 gradient and optimizer-update parity gates failed. "
        "No H128/H64/H32 sustained screen, formal C5/C6 training, or AIME generation was started.\n\n"
        "| Gate | Result |\n|---|---|\n"
        "| Available GPUs | 4 × RTX3090 24GB |\n"
        "| Model partition / untied embedding–head | PASS / PASS |\n"
        f"| Single-GPU H32 forward repeat | {'PASS' if forward_exact else 'FAIL'} |\n"
        f"| Single-GPU H32 gradient bitwise repeat | {'PASS' if single_repeat_gradient_exact else 'FAIL'} |\n"
        f"| 2-GPU H32 forward parity | {'PASS' if forward_exact else 'FAIL'} |\n"
        "| 2-GPU H32 gradient parity | FAIL |\n"
        "| 2-GPU H32 one-step update parity | FAIL |\n"
        f"| Recurrent writeback / BPTT boundary | {c['REAL_RECURRENT_WRITEBACK_GATE']} / {c['BPTT_BOUNDARY_GATE']} |\n"
        "| 2-GPU training semantics | FAIL |\n"
        "| 4-GPU topology | NOT RUN, blocked by 2-GPU foundation gate |\n\n"
        f"All {len(a['records'])} sampled layer/token traces match bitwise for consumed state, pre-QDQ state, "
        "scale, qcodes, and post-QDQ state. Inputs, target positions, all eight per-target loss components, "
        f"and the total C5 training loss ({a['total_training_loss']:.16g}) also match. "
        "The raw and clipped rotation gradient hashes differ across both single-GPU repeats and the dual-GPU run; "
        "the post-Adam theta and rotation hashes likewise differ. Gradient norms were "
        f"{a['gradient_norm_after_clip']:.12g}, {b['gradient_norm_after_clip']:.12g} (single repeats) "
        f"and {c['gradient_norm_after_clip']:.12g} (dual). This establishes a bitwise mismatch, "
        "not that the dual topology alone caused it. No applicable pre-existing frozen numerical tolerance "
        "was found, and none was invented after seeing the dual result.\n\n"
        "Resource decision table:\n\n"
        "| GPUs | H | C5 | C6 | Verdict |\n|---:|---:|---|---|---|\n"
        + "".join(f"| {g} | {h} | NOT RUN | NOT RUN | blocked at parity |\n" for g, h in order)
        + "\nFINAL_TRAIN_GPU_COUNT = NOT_SELECTED; FINAL_GRADIENT_HORIZON = NOT_SELECTED; "
          "FINAL_PARTITION = NOT_SELECTED. Selection rule remains "
          "MAXIMIZE_HORIZON_THEN_MINIMIZE_GPU_COUNT.\n\n"
          "Stop condition: 2GPU H32 semantic parity FAIL. Await manual direction; do not proceed to four GPUs, "
          "formal training, or AIME. Preserved inputs: `analysis/one_step_c5_single_repeat1.json`, "
          "`analysis/one_step_c5_single_repeat2.json`, `analysis/one_step_c5_dual_split16.json` and their logs.\n"
    )
    HASHES.mkdir(exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and path != HASHES / "artifact_sha256.txt" and "__pycache__" not in path.parts)
    (HASHES / "artifact_sha256.txt").write_text(
        "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files))
    print(json.dumps({"status": "BLOCKED_WAITING_FOR_MANUAL_DECISION",
                      "forward": "PASS" if forward_exact else "FAIL",
                      "gradient": "FAIL", "update": "FAIL",
                      "single_repeat_gradient_exact": single_repeat_gradient_exact,
                      "trace_points": len(a["records"])}, indent=2))


if __name__ == "__main__":
    main()
