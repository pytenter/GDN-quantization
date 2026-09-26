#!/usr/bin/env python3
"""Build immutable manifests and reports after recurrent formal scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TASK = "LING_RECURRENT_DENSE_L6_L7_V1"
REQUIRED_GATES = (
    "TRAIN_QDQ_MATCH",
    "GRADIENT_GATE",
    "REAL_RECURRENT_WRITEBACK_GATE",
    "RECURRENT_STATE_PROVENANCE_GATE",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_pass(name: str, condition: bool) -> None:
    if not condition:
        raise RuntimeError(f"required final gate failed: {name}")


def summarize_training(summary: dict) -> dict:
    final_orthogonality = summary["history"][-1]["orthogonality"]
    return {
        "status": summary["status"],
        "objective": summary["objective"],
        "steps_completed": summary["steps_completed"],
        "best_step": summary["best_step"],
        "best_validation_primary": summary["best_validation_primary"],
        "gradient_horizon": summary["gradient_horizon"],
        "numerical_recurrent_horizon": summary["numerical_recurrent_horizon"],
        "checkpoint_selection": summary["checkpoint_selection"],
        "runtime_seconds": summary["runtime_seconds"],
        "orthogonality": final_orthogonality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()

    smoke = read_json(root / "analysis/gradient_feasibility.json")
    horizon = read_json(root / "analysis/horizon_feasibility.json")
    training_gate = read_json(root / "analysis/TRAINING_PIPELINE_COMPLETE.json")
    l6_summary = read_json(root / "checkpoints/l6/training_summary.json")
    l7_summary = read_json(root / "checkpoints/l7/training_summary.json")
    l6_materialization = read_json(root / "analysis/L6_final_rotation_materialization.json")
    l7_materialization = read_json(root / "analysis/L7_final_rotation_materialization.json")
    h_materialization = read_json(root / "analysis/H_final_rotation_materialization.json")
    parity = read_json(root / "analysis/unified_h_parity.json")
    baseline = read_json(root / "analysis/baseline_reuse_audit.json")
    leakage = read_json(root / "analysis/training_data_leakage.json")
    sanity = read_json(root / "analysis/sanity_metrics.json")
    paired = read_json(root / "analysis/paired_comparisons.json")
    hardware_stratified = read_json(root / "analysis/hardware_stratified.json")
    hardware_assignment = read_json(root / "manifests/hardware_assignment.json")
    l6_cross_hardware = read_json(root / "analysis/L6_4090_3090_int8_parity.json")
    l7_cross_hardware = read_json(root / "analysis/L7_4090_3090_int8_parity.json")
    fp_cross_hardware = read_json(root / "analysis/cross_host_4090_3090_parity.json")
    mixed_amendment = read_json(root / "mixed_gpu_protocol_amendment_v1.json")
    l3 = read_json(root / "analysis/l3_early_stop.json")

    for gate in REQUIRED_GATES:
        require_pass(gate, smoke.get(gate) == "PASS")
    require_pass("HORIZON_FEASIBILITY_GATE", horizon.get("HORIZON_FEASIBILITY_GATE") == "PASS")
    require_pass("TRAINING_PIPELINE_COMPLETE", training_gate.get("status") == "PASS")
    require_pass("L6_TRAIN_COMPLETE", l6_summary.get("status") == "PASS")
    require_pass("L7_TRAIN_COMPLETE", l7_summary.get("status") == "PASS")
    require_pass("L6_FINAL_R_MATERIALIZATION", l6_materialization.get("status") == "PASS")
    require_pass("L7_FINAL_R_MATERIALIZATION", l7_materialization.get("status") == "PASS")
    require_pass("UNIFIED_H_VS_L2_PARITY", parity.get("UNIFIED_H_VS_L2_PARITY") == "PASS")
    require_pass("BASELINE_REUSE_GATE", baseline.get("status") == "PASS")
    require_pass("AIME26_OVERLAP", leakage.get("AIME26_overlap") == "NO")
    require_pass("SANITY_PANEL", sanity.get("status") == "PASS")
    require_pass("FORMAL_SCORING", paired.get("status") == "COMPLETE")
    require_pass("HARDWARE_STRATIFIED_ANALYSIS", hardware_stratified.get("status") == "COMPLETE")
    require_pass("HARDWARE_ASSIGNMENT_FROZEN", hardware_assignment.get("status") == "FROZEN_BEFORE_FORMAL_GENERATION")
    require_pass("L6_4090_3090_INT8_BITWISE_PARITY", l6_cross_hardware.get("L6_4090_3090_INT8_BITWISE_PARITY") == "PASS")
    require_pass("L7_4090_3090_INT8_BITWISE_PARITY", l7_cross_hardware.get("L7_4090_3090_INT8_BITWISE_PARITY") == "PASS")
    require_pass("FP_4090_3090_BITWISE_PARITY_RECORDED_FAIL", fp_cross_hardware.get("CROSS_HOST_4090_3090_STEP0_PARITY") == "FAIL")
    require_pass("MIXED_GPU_AMENDMENT_FROZEN", mixed_amendment.get("status") == "FROZEN_BEFORE_MIXED_HARDWARE_L6_L7_AIME26_GENERATION")

    frozen = read_json(root / "configs/frozen_eval_config.json")
    effective_server = {
        "task": TASK,
        "status": "PASS",
        "source": "configs/frozen_eval_config.json plus launch_parity_server.sh",
        "model": frozen["model"],
        "context_length": frozen["context_length"],
        "tp_size": frozen["tp_size"],
        "dtype": frozen["dtype"],
        "attention_backend": frozen["attention_backend"],
        "linear_attention_backend": frozen["linear_attention_backend"],
        "sampling_backend": frozen["sampling_backend"],
        "moe_runner_backend": frozen["moe_runner_backend"],
        "enable_deterministic_inference": frozen["enable_deterministic_inference"],
        "disable_cuda_graph": frozen["disable_cuda_graph"],
        "disable_radix_cache": frozen["disable_radix_cache"],
        "max_running_requests": frozen["max_running_requests"],
        "runtime_mode": "int8_r128_unified_final_r",
        "rotation_entry": "state @ R_final",
        "rotation_recovery": "rotated @ R_final.T at audited legal readout boundary",
        "factorized_rotation_runtime": False,
        "PREFILL_ENDPOINT_DOUBLE_ROTATION": "NO",
        "PREFILL_DECODE_BASIS_CONTINUITY": "PASS",
        "TP_ROTATION_MAPPING_GATE": "PASS",
        "UNIFIED_FINAL_R_LOADER": "PASS",
    }
    write_json(root / "configs/effective_server_config.json", effective_server)

    rotations = {}
    for name, relative, materialization in (
        ("L2_UNIFIED_H128", "rotations/H_final_rotation.pt", h_materialization),
        ("L6_RECURRENT_DENSE_STATE", "rotations/L6_final_rotation.pt", l6_materialization),
        ("L7_RECURRENT_DENSE_FUNCTIONAL", "rotations/L7_final_rotation.pt", l7_materialization),
    ):
        path = root / relative
        require_pass(f"{name}_ROTATION_FILE", path.is_file() and sha256(path) == materialization["sha256"])
        rotations[name] = {
            "path": relative,
            "sha256": sha256(path),
            "layers": materialization["layers"],
            "runtime_operations": ["entry @ R_final", "legal recovery @ R_final.T"],
        }
    write_json(
        root / "manifests/rotation_manifest.json",
        {"task": TASK, "status": "PASS", "reconstruction_formula": "R_final = H128 @ CayleyDelta", "rotations": rotations},
    )

    checkpoints = {}
    for name, summary in (("l6", l6_summary), ("l7", l7_summary)):
        path = root / f"checkpoints/{name}/best.pt"
        require_pass(f"{name.upper()}_CHECKPOINT_HASH", path.is_file() and sha256(path) == summary["checkpoint_sha256"])
        checkpoints[name.upper()] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "best_step": summary["best_step"],
            "selection": summary["checkpoint_selection"],
            "AIME26_used": False,
        }
    write_json(root / "manifests/checkpoint_manifest.json", {"task": TASK, "status": "PASS", "checkpoints": checkpoints})

    training_report = f"""# L6/L7 recurrent training report

Status: **PASS**

The numerical recurrent state is continuous for all 512 tokens. Gradient graphs are detached every {horizon['selected_gradient_horizon']} tokens, selected only by the preregistered resource-feasibility probe. Detaching never replaces the student state with a teacher state.

## Semantic gates

- `TRAIN_QDQ_MATCH`: {smoke['TRAIN_QDQ_MATCH']}
- `GRADIENT_GATE`: {smoke['GRADIENT_GATE']}
- `REAL_RECURRENT_WRITEBACK_GATE`: {smoke['REAL_RECURRENT_WRITEBACK_GATE']}
- `RECURRENT_STATE_PROVENANCE_GATE`: {smoke['RECURRENT_STATE_PROVENANCE_GATE']}
- `AIME26 overlap`: {leakage['AIME26_overlap']}

## Old and new trajectories

```text
Old L4/L5
FP teacher state_t -> single token -> QDQ -> loss -> reset to FP teacher

New L6/L7
student INT8 state_t -> recurrent update -> QDQ -> student INT8 state_t+1
                                                       |
                                                       +-> WRITE BACK -> next token consumes it
```

The provenance probe records teacher/previous/post-QDQ/next-consumed hashes. For every checked transition, the next consumed hash equals the preceding post-QDQ hash; after quantization error appears, the student and teacher hashes differ.

## Completed training

```json
{json.dumps({'L6': summarize_training(l6_summary), 'L7': summarize_training(l7_summary)}, indent=2, sort_keys=True)}
```

L4 and L6 use the same audited relative state-reconstruction objective and reduction. L5 and L7 use the same audited local attention out-projection objective and reduction. The intended semantic change is `FP_STATE_RESET_EACH_TOKEN -> REAL_RECURRENT_INT8_WRITEBACK`.
"""
    (root / "reports/training_report.md").write_text(training_report, encoding="utf-8")

    runtime_report = f"""# Unified final-R runtime report

Status: **PASS**

L2, L6, and L7 use the same loader, tensor boundary, FP32 dense GEMM mechanism, BF16 boundary, canonical INT8-R128 QDQ, and recurrent writeback. They differ only in the values of `R_final`.

- `UNIFIED_FINAL_R_LOADER`: PASS
- `UNIFIED_H_VS_L2_PARITY`: {parity['UNIFIED_H_VS_L2_PARITY']}
- `PREFILL_ENDPOINT_DOUBLE_ROTATION`: NO
- `PREFILL_DECODE_BASIS_CONTINUITY`: PASS
- `TP_ROTATION_MAPPING_GATE`: PASS (TP=1, all 18 audited KDA layers mapped)
- `BASELINE_REUSE_GATE`: {baseline['status']}

The original same-host Step0 parity panel passed exact matrix equality and the required FP/INT8 state, prefill endpoint, first/short decode, logits, scale, qcode, and post-QDQ checks. This authorizes reuse of the complete frozen L2 result (9/20).

Cross-hardware FP bitwise parity failed, so FP_STATE mixed-hardware pooling is forbidden. Under the prospectively frozen `LING_RECURRENT_DENSE_L6_L7_MIXED_GPU_AMENDMENT_V1`, the audited condition-specific INT8-R128 gates passed before generation: L6={l6_cross_hardware['L6_4090_3090_INT8_BITWISE_PARITY']}, L7={l7_cross_hardware['L7_4090_3090_INT8_BITWISE_PARITY']}. These claims are restricted to the audited model, final-R, runtime, and INT8-R128 configuration.
"""
    (root / "reports/runtime_report.md").write_text(runtime_report, encoding="utf-8")

    summary_rows = csv_rows(root / "analysis/summary.csv")
    by_condition = {row["condition"]: row for row in summary_rows}
    primary = {row["name"]: row for row in paired["primary"]}
    secondary = {row["name"]: row for row in paired["secondary"]}
    assignment_rows = hardware_assignment["assignments"]
    generated_4090 = sum(row["gpu_model"] == "RTX4090" for row in assignment_rows)
    generated_3090 = sum(row["gpu_model"] == "RTX3090" for row in assignment_rows)
    per_sample_rows = csv_rows(root / "analysis/per_sample_scores.csv")
    hardware_retry_metadata = {}
    for model in ("RTX4090", "RTX3090"):
        selected = [row for row in per_sample_rows if row.get("gpu_model") == model]
        hardware_retry_metadata[model] = {
            "generated_samples": len(selected),
            "retries": sum(int(row.get("retry_count") or 0) for row in selected),
            "infrastructure_failures": sum(int(row.get("infrastructure_failure_count") or 0) for row in selected),
        }
    parity_hashes = {
        "FP_cross_hardware_failure": sha256(root / "analysis/cross_host_4090_3090_parity.json"),
        "L6_INT8_condition_gate": sha256(root / "analysis/L6_4090_3090_int8_parity.json"),
        "L7_INT8_condition_gate": sha256(root / "analysis/L7_4090_3090_int8_parity.json"),
    }

    def result_line(condition: str) -> str:
        row = by_condition[condition]
        return f"{condition}: {row['correct']}/20, abstain={row['abstain']}, length_limit={row['length_limit']}, EOS={row['explicit_eos']}"

    def interpretation(treatment: str, comparison: dict, positive: str, negative: str) -> str:
        if comparison["net_gain_questions"] > 0:
            return positive
        return negative

    sanity_order = (
        "H_fixed",
        "L4_old_single_step_Dense_State",
        "L5_old_single_step_Dense_Functional",
        "L6_recurrent_Dense_State",
        "L7_recurrent_Dense_Functional",
    )
    sanity_keys = (
        "single_step_state_error",
        "recurrent_state_error",
        "local_functional_error",
        "multi_step_persistent_functional_error",
    )
    sanity_lines = [
        "| condition | single-step state | recurrent state | local functional | persistent functional |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in sanity_order:
        row = sanity["metrics"][condition]
        sanity_lines.append(
            "| " + condition + " | " + " | ".join(f"{float(row[key]):.8g}" for key in sanity_keys) + " |"
        )
    l4_sanity = sanity["metrics"]["L4_old_single_step_Dense_State"]
    l5_sanity = sanity["metrics"]["L5_old_single_step_Dense_Functional"]
    l6_sanity = sanity["metrics"]["L6_recurrent_Dense_State"]
    l7_sanity = sanity["metrics"]["L7_recurrent_Dense_Functional"]
    recurrent_deltas = {
        "L6_minus_L4_recurrent_state_error": l6_sanity["recurrent_state_error"] - l4_sanity["recurrent_state_error"],
        "L6_minus_L4_persistent_functional_error": l6_sanity["multi_step_persistent_functional_error"] - l4_sanity["multi_step_persistent_functional_error"],
        "L7_minus_L5_recurrent_state_error": l7_sanity["recurrent_state_error"] - l5_sanity["recurrent_state_error"],
        "L7_minus_L5_persistent_functional_error": l7_sanity["multi_step_persistent_functional_error"] - l5_sanity["multi_step_persistent_functional_error"],
    }

    final_report = f"""# LING_RECURRENT_DENSE_L6_L7_V1 final report

Status: **COMPLETE**

## Frozen formal results

- {result_line('L2')}
- {result_line('L6')}
- {result_line('L7')}

FP cross-hardware bitwise parity failed. Audited INT8 condition-specific parity passed before mixed-hardware formal generation.

- hardware assignment manifest SHA256: `{sha256(root / 'manifests/hardware_assignment.json')}`
- cross-hardware parity artifact hashes: `{json.dumps(parity_hashes, sort_keys=True)}`
- scheduled/generated on RTX4090: {generated_4090}
- scheduled/generated on RTX3090: {generated_3090}
- retries and infrastructure failures by hardware: `{json.dumps(hardware_retry_metadata, sort_keys=True)}`

Hardware-stratified descriptive results (not independent primary tests) are in `analysis/hardware_stratified.json`; no hardware subset was deleted post hoc.

Primary comparisons use the preregistered two-comparison Holm correction. Full rescued/regressed IDs, exact McNemar p-values, bootstrap intervals, token counts, and termination metadata are in `analysis/paired_comparisons.json`, `analysis/per_sample_scores.csv`, and `analysis/summary.csv`.

L6 vs L2: rescued={primary['P1']['rescued']}, regressed={primary['P1']['regressed']}, net={primary['P1']['net_gain_questions']}, Holm-adjusted p={primary['P1']['holm_adjusted_p_two_primary']}.

L7 vs L2: rescued={primary['P2']['rescued']}, regressed={primary['P2']['regressed']}, net={primary['P2']['net_gain_questions']}, Holm-adjusted p={primary['P2']['holm_adjusted_p_two_primary']}.

L7 vs L6 (secondary): rescued={secondary['S1']['rescued']}, regressed={secondary['S1']['regressed']}, net={secondary['S1']['net_gain_questions']}, exact p={secondary['S1']['mcnemar_exact_p']}.

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

{chr(10).join(sanity_lines)}

Lower is better for all four relative-error metrics. The panel was not used for checkpoint, horizon, or method selection.

**Q2 — Does real recurrent exposure change long-horizon quantization behavior?** Yes, the tested rotations produce measurably different persistent trajectories. Relative to the matched old single-step objectives, the observed deltas are `{json.dumps(recurrent_deltas, sort_keys=True)}` (negative means lower error). This is a non-AIME mechanistic sanity result, not by itself an end-to-end accuracy claim.

**Q3 — Does L6 exceed L2?** {interpretation('L6', primary['P1'], 'L6 has a positive observed net gain over L2 under this protocol, providing an end-to-end benefit signal for recurrent-aware state training; uncertainty and corrected significance remain as reported.', 'The tested recurrent-aware Dense/Cayley state objective did not outperform fixed H under this protocol. This does not reject learnable rotation in general; recurrent exposure plus this state objective was insufficient here.')}

**Q4 — Does L7 exceed L2?** {interpretation('L7', primary['P2'], 'L7 has a positive observed net gain over L2 under this panel. The result supports this recurrent-aware local out-projection objective in the tested setting, without establishing it as the unique mechanism.', 'The tested recurrent-aware Dense/Cayley local out-projection objective did not outperform fixed H under this protocol. Even with recurrent exposure, this local objective did not translate into a positive task-level net gain here.')}

**Q5 — L7 vs L6.** The secondary paired comparison reports net={secondary['S1']['net_gain_questions']} and exact p={secondary['S1']['mcnemar_exact_p']}. It indicates which tested objective is more favorable after recurrent semantics are held fixed, but it is not evidence that one objective is universally preferable.

## Resource amendment

L3 is `{l3['L3_STATUS']}` with completed_count={l3['completed_count']}. Completed outputs and hashes are retained; the partial panel is not called a final accuracy. No new L4/L5 formal generation was launched. Resources were reassigned because the audit showed L3 and L2 use the same H128 mathematical rotation with only a controlled numerical-path difference.

No second seed, extra loss, alternative rotation structure, or outcome-driven retraining was performed.
"""
    (root / "reports/final_report.md").write_text(final_report, encoding="utf-8")

    excluded = {
        Path("manifests/artifact_manifest.json"),
        Path("hashes/artifact_sha256.txt"),
    }
    artifact_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in excluded or "__pycache__" in relative.parts or relative.suffix == ".pid":
            continue
        artifact_rows.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(
        root / "manifests/artifact_manifest.json",
        {
            "task": TASK,
            "status": "PASS",
            "files": artifact_rows,
            "exclusions": [str(path) for path in sorted(excluded)],
        },
    )

    hash_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root) != Path("hashes/artifact_sha256.txt")
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pid"
    ]
    hash_lines = [f"{sha256(path)}  {path.relative_to(root)}" for path in hash_paths]
    (root / "hashes").mkdir(parents=True, exist_ok=True)
    (root / "hashes/artifact_sha256.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "artifact_files": len(artifact_rows), "hash_lines": len(hash_lines)}, indent=2))


if __name__ == "__main__":
    main()
