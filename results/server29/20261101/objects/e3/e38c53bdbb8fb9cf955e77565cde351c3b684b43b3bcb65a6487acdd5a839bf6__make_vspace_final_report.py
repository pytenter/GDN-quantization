import json
import pathlib
import time

screen_path = pathlib.Path("/data/zypan/results/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_screen.json")
beh_path = pathlib.Path("/data/zypan/results/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_behavioral_pilot.json")
out = pathlib.Path("/data/zypan/reports/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_final.md")

screen = json.loads(screen_path.read_text())
beh = json.loads(beh_path.read_text())
by = screen["aggregate"]["by_rotation"]

lines = []
lines.append("# GDN_INT8_VSPACE_COMPUTE_INVARIANT_ROTATION_SCALE_CONTAMINATION_V1 Final")
lines.append("")
lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
lines.append(f"Git commit: {screen.get('git_commit')}")
lines.append(f"Model: {screen.get('model')}")
lines.append("")
lines.append("## Protocol Gates")
for key, value in screen["gate_results"].items():
    lines.append(f"- {key}: {value}")
lines.append("- Smoke: PASS")
lines.append(f"- Screen units: {len(screen['per_unit'])}/9")
lines.append(f"- SOURCE_RESCUE_GATE: {screen['SOURCE_RESCUE_GATE']}")
lines.append(f"- VSPACE_SOURCE_RESCUE: {screen['VSPACE_SOURCE_RESCUE']}")
lines.append(f"- PRIMARY_ROTATION: {screen['PRIMARY_ROTATION']}")
lines.append("")
lines.append("## Phase A Source-Side Aggregate")
lines.append("| rotation | median peak/RMS | peak lower units | median residual | residual improvement units | median residual reduction | median R-to-C gap closure | G_state/id | G_key/id | G_query/id | median scale P95 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in ["IDENTITY", "V_HADAMARD", "V_PCA_SPREAD"]:
    b = by[name]
    lines.append(
        f"| {name} | {b.get('median_peak_to_rms')} | {b.get('peak_to_rms_lower_units')} | "
        f"{b.get('median_residual_norm')} | {b.get('residual_improvement_units')} | "
        f"{b.get('median_residual_reduction')} | {b.get('median_R_to_C_gap_closure')} | "
        f"{b.get('G_state_ratio_to_identity')} | {b.get('G_key_ratio_to_identity')} | "
        f"{b.get('G_query_ratio_to_identity')} | {b.get('median_scale_p95')} |"
    )
lines.append("")
lines.append(f"C128 reference median residual norm: {screen['aggregate']['C128_reference']['median_residual_norm']}")
lines.append("")
lines.append("## Phase B Behavioral Aggregate")
for key, value in beh["aggregate"].items():
    lines.append(f"- {key}: {value}")
lines.append(f"- pilot_classification: {beh['pilot_classification']}")
lines.append(f"- CROSS_VALUE_RANGE_IMBALANCE_SOURCE_CONTRIBUTOR: {beh['CROSS_VALUE_RANGE_IMBALANCE_SOURCE_CONTRIBUTOR']}")
lines.append(f"- Behavioral units: {len(beh['per_unit'])}/9")
lines.append("")
lines.append("## Phase B Per-Unit KL AUC")
lines.append("| problem_id | t0 | R_ID KL_AUC | R_VROT KL_AUC | R_VROT_NORM_RESTORED KL_AUC | C_REF KL_AUC | VROT rescue | rescue fraction | norm restore damage |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for u in beh["per_unit"]:
    km = u["KL_metrics"]
    lines.append(
        f"| {u['problem_id']} | {u['t0']} | {km['R_ID']['KL_AUC']} | {km['R_VROT']['KL_AUC']} | "
        f"{km['R_VROT_NORM_RESTORED']['KL_AUC']} | {km['C_REF']['KL_AUC']} | "
        f"{u['VROT_KL_RESCUE']} | {u['VROT_KL_RESCUE_FRACTION']} | {u['NORM_RESTORE_DAMAGE']} |"
    )
lines.append("")
lines.append("## Phase B Per-Unit Terminal KL")
lines.append("| problem_id | t0 | R_ID terminal KL | R_VROT terminal KL | R_VROT_NORM_RESTORED terminal KL | C_REF terminal KL |")
lines.append("|---|---:|---:|---:|---:|---:|")
for u in beh["per_unit"]:
    km = u["KL_metrics"]
    lines.append(
        f"| {u['problem_id']} | {u['t0']} | {km['R_ID']['terminal_KL']} | {km['R_VROT']['terminal_KL']} | "
        f"{km['R_VROT_NORM_RESTORED']['terminal_KL']} | {km['C_REF']['terminal_KL']} |"
    )
lines.append("")
lines.append("## Conclusion")
lines.append("- V-space Hadamard strongly reduces row scale imbalance and raw R128 residual in Phase A.")
lines.append("- Behavioral pilot supports cross-Value scale contamination as a source contributor: R_VROT improves KL in 9/9 units.")
lines.append("- Norm restoration worsens KL in 9/9 units, so the rescue is not explained by simply restoring residual magnitude.")
lines.append("- MECHANISM_CLOSURE_CANDIDATE: NO")
lines.append("- METHOD_DESIGN_READY: NO")
lines.append("")
lines.append("## Artifacts")
for path in [
    screen_path,
    beh_path,
    pathlib.Path("/data/zypan/results/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_checkpoint.json"),
    pathlib.Path("/data/zypan/results/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_raw.npz"),
    pathlib.Path("/data/zypan/results/gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_figures"),
    out,
]:
    lines.append(f"- {path}")

out.write_text("\n".join(lines) + "\n")
print(out)
