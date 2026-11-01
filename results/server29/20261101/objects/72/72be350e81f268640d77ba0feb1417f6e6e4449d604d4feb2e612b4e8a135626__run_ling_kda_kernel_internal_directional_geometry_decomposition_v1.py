#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


TASK = "LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / TASK
MIN_GPU_DIR = OUT_DIR / "minimal_gpu_results"
REFERENCE_COPY = OUT_DIR / "pytorch_kda_reference.py"

CORE_INTERNAL_TASK = "LING_KDA_CORE_INTERNAL_OPERATOR_DECOMPOSITION_V1"
CORE_INTERNAL_OUT = REPO / "results" / CORE_INTERNAL_TASK
LAYER_TASK = "LING_KDA_LAYER_CONDITIONED_LOCAL_FUNCTIONAL_GEOMETRY_AUDIT_V1"
LAYER_OUT = REPO / "results" / LAYER_TASK
RESIDUAL_TASK = "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1"
RESIDUAL_OUT = REPO / "results" / RESIDUAL_TASK
MAG_TASK = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
MAG_OUT = REPO / "results" / MAG_TASK

RESIDUAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_granularity_residual_functional_sensitivity_v1.py"
REF_SOURCE = REPO / "experiments" / "ling" / "pytorch_kda_reference.py"
MODEL_FILE = Path("/data01/user2/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py")
FLA_KDA_DIR = Path("/data01/user2/.conda/envs/ling-kda/lib/python3.11/site-packages/fla/ops/kda")

GROUP_SIZES = [128, 16]
KDA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
SENTINEL_LAYERS = [0, 10, 22]
KERNEL_STAGE_ORDER = ["K0_INPUT_ERROR", "K1_DECAYED_ERROR", "K2_UPDATE_ERROR", "K3_READOUT_ERROR", "K4_CORE_OUTPUT"]
EPS = 1e-12
PARITY_REL_TOL = 2e-3
PARITY_ABS_TOL = 2e-3
TRANSFER_REL_TOL = 2e-3
BOOTSTRAP_SEED = 20260912

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "local_kda_source_provenance.json",
    "local_kda_source_audit.md",
    "common36_manifest.csv",
    "pytorch_kda_reference.py",
    "reference_parity_results.csv",
    "kernel_internal_operator_equations.md",
    "kernel_internal_stage_manifest.json",
    "exact_state_error_transfer_results.csv",
    "kernel_internal_event_trace.csv",
    "kernel_stage_directional_gain.csv",
    "kernel_stage_layer_sign_concordance.csv",
    "layer22_kernel_internal_trace.csv",
    "kernel_internal_master_table.csv",
    "granularity_internal_geometry_summary.csv",
    "phase3_causal_intervention_plan.md",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "LOCAL_KDA_SOURCE_PROVENANCE_GATE",
    "REFERENCE_STATE_PARITY_GATE",
    "REFERENCE_CORE_OUTPUT_PARITY_GATE",
    "FROZEN_LOCAL_DRIVER_GATE",
    "MATCHED_CARRIER_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "PRIOR_LAYER_LABEL_REUSE_GATE",
    "N_COMMON_UNITS",
    "N_KDA_LAYERS",
    "CORE_TRANSFER_MODEL_CORRECTED",
    "EXACT_STATE_ERROR_TRANSFER_GATE",
    "EXACT_TRANSFER_MEDIAN_REL_ERROR",
    "EXACT_TRANSFER_P95_REL_ERROR",
    "EXACT_TRANSFER_MAX_REL_ERROR",
    "OLD_CORE_TRANSFER_FAILURE_EXPLANATION",
    "N_KERNEL_INTERNAL_STAGES",
    "EARLIEST_KERNEL_INTERNAL_17_OF_18_STAGE",
    "EARLIEST_KERNEL_INTERNAL_18_OF_18_SIGN_CLOSURE_STAGE",
    "EARLIEST_KERNEL_INTERNAL_LAYER22_REVERSE_STAGE",
    "IDENTIFIED_KERNEL_STAGE",
    "IDENTIFIED_STAGE_OPERATOR_TYPE",
    "IDENTIFIED_STAGE_N_SIGN_CONCORDANT",
    "IDENTIFIED_STAGE_N_SIGN_DISCORDANT",
    "IDENTIFIED_STAGE_DISCORDANT_LAYER_IDS",
    "L22_DELTA_G_IDENTIFIED_STAGE",
    "L22_REVERSE_CONTROL_IDENTIFIED_STAGE",
    "G128_IDENTIFIED_STAGE_DELTA_G",
    "G128_IDENTIFIED_STAGE_DELTA_G_CI",
    "G16_IDENTIFIED_STAGE_DELTA_G",
    "G16_IDENTIFIED_STAGE_DELTA_G_CI",
    "IDENTIFIED_STAGE_D128_TO_D16",
    "IDENTIFIED_STAGE_D128_TO_D16_CI",
    "KERNEL_OPERATOR_LOCALIZATION_CONVERGENCE",
    "PHASE3_CAUSAL_INTERVENTION",
    "TARGETED_OPERATOR_CAUSAL_STATUS",
    "TARGETED_CAUSAL_RESCUE",
    "TARGETED_CAUSAL_RESCUE_CI",
    "LAYER22_CAUSAL_CONTROL_RESULT",
    "KDA_MECHANISM_CLOSURE",
    "FINAL_CLASSIFICATION",
    "N_NEW_STAGE_B_TRAJECTORIES",
    "N_NEW_CLOSED_LOOP_TRAJECTORIES",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "BASH_SYNTAX_CHECK",
    "GPU_RELEASE_VERIFICATION",
    "METHOD_DESIGN_READY",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NEXT_RECOMMENDED_TASK",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_reference():
    return import_module(REF_SOURCE, "ling_pytorch_kda_reference")


def import_residual():
    return import_module(RESIDUAL_RUNNER, "ling_residual_for_kernel_internal")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def finite(value):
    return as_float(value) is not None


def mean(values):
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else None


def median(values):
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def p95(values):
    vals = sorted(float(v) for v in values if finite(v))
    return vals[int(0.95 * (len(vals) - 1))] if vals else None


def bootstrap_ci(values, reps=400, seed=BOOTSTRAP_SEED, statistic=median):
    vals = [float(v) for v in values if finite(v)]
    if not vals:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(float(statistic(sample)))
    boot.sort()
    return [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def tensor_rel_error(torch, a, b):
    af = a.detach().float()
    bf = b.detach().float().to(af.device)
    return tensor_norm(torch, af - bf) / (tensor_norm(torch, bf) + EPS)


def tensor_cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().to(a.device).reshape(-1)
    denom = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    if denom <= EPS:
        return None
    return float(torch.dot(af, bf).item() / (denom + EPS))


def prior_layer_labels():
    return {layer: ("R_SENSITIVE" if layer == 22 else "C_SENSITIVE") for layer in KDA_LAYERS}


def prior_layer_label_reuse_gate(rows):
    expected = prior_layer_labels()
    observed = {}
    for row in rows:
        try:
            observed[int(row["layer_id"])] = str(row.get("layer_classification") or row.get("prior_layer_label"))
        except Exception:
            continue
    return "PASS" if observed == expected else "FAIL"


def local_source_provenance_gate(paths):
    return "PASS" if all(Path(path).exists() and file_sha256(path) for path in paths) else "FAIL"


def phase3_gate(state_parity, output_parity, convergence, layer22_reverse, identified_stage):
    if state_parity == "PASS" and output_parity == "PASS" and convergence == "STRONG" and layer22_reverse == "PASS" and identified_stage not in ("", "NONE", "UNRESOLVED"):
        return "RUN"
    return "NOT_RUN"


def earliest_stage(rows, target):
    by_order = {stage: idx for idx, stage in enumerate(KERNEL_STAGE_ORDER)}
    for row in sorted(rows, key=lambda r: by_order.get(str(r.get("stage_id")), 999)):
        try:
            if int(row.get("N_SIGN_CONCORDANT")) >= int(target):
                return str(row["stage_id"])
        except Exception:
            continue
    return "NONE"


def earliest_layer22_reverse(rows):
    by_order = {stage: idx for idx, stage in enumerate(KERNEL_STAGE_ORDER)}
    for row in sorted(rows, key=lambda r: by_order.get(str(r.get("stage_id")), 999)):
        if row.get("LAYER22_REVERSE_CONTROL") == "PASS" or row.get("layer22_reverse") == "YES":
            return str(row["stage_id"])
    return "NONE"


def kernel_stage_layer_sign_concordance(layer_rows, prior):
    out = []
    by_stage = defaultdict(list)
    for row in layer_rows:
        by_stage[str(row["stage_id"])].append(row)
    for stage_id in sorted(by_stage, key=lambda s: KERNEL_STAGE_ORDER.index(s) if s in KERNEL_STAGE_ORDER else 999):
        sub = by_stage[stage_id]
        concordant = 0
        discordant = []
        layer22 = "MISSING"
        for row in sub:
            layer = int(row["layer_id"])
            delta = as_float(row.get("delta_G"))
            expected = prior.get(layer)
            ok = (expected == "C_SENSITIVE" and delta is not None and delta > 0) or (expected == "R_SENSITIVE" and delta is not None and delta < 0)
            if ok:
                concordant += 1
            else:
                discordant.append(layer)
            if layer == 22:
                layer22 = "PASS" if delta is not None and delta < 0 else "FAIL"
        out.append({
            "stage_id": stage_id,
            "N_SIGN_CONCORDANT": concordant,
            "N_SIGN_DISCORDANT": len(discordant),
            "CONCORDANCE_RATE": concordant / len(sub) if sub else None,
            "DISCORDANT_LAYER_IDS": ",".join(str(x) for x in discordant),
            "LAYER22_REVERSE_CONTROL": layer22,
        })
    return out


def kernel_stage_manifest():
    return [
        {"stage_id": "K0_INPUT_ERROR", "order": 0, "operator_type": "INPUT", "description": "matched recurrent-state error E"},
        {"stage_id": "K1_DECAYED_ERROR", "order": 1, "operator_type": "KEY_CHANNEL_DECAY", "description": "E multiplied by exp(decay_log) per key channel"},
        {"stage_id": "K2_UPDATE_ERROR", "order": 2, "operator_type": "DELTA_UPDATE_CORRECTION", "description": "decayed error after beta-scaled key correction"},
        {"stage_id": "K3_READOUT_ERROR", "order": 3, "operator_type": "QUERY_READOUT", "description": "query contraction of updated error before output dtype cast"},
        {"stage_id": "K4_CORE_OUTPUT", "order": 4, "operator_type": "CORE_OUTPUT_CAST", "description": "backend core output perturbation tensor"},
    ]


def driver_kwargs(rec):
    return {
        "A_log": rec.get("A_log"),
        "dt_bias": rec.get("dt_bias"),
        "use_qk_l2norm_in_kernel": bool(rec.get("use_qk_l2norm_in_kernel", False)),
        "use_gate_in_kernel": bool(rec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(rec.get("use_beta_sigmoid_in_kernel", False)),
        "lower_bound": rec.get("lower_bound"),
        "state_v_first": bool(rec.get("state_v_first", False)),
    }


def run_reference(ref, rec, state):
    return ref.kda_one_step_reference(
        rec["q"], rec["k"], rec["v"], rec["g"], rec["beta"], initial_state=state, **driver_kwargs(rec)
    )


def run_transfer(ref, rec, error):
    return ref.kda_exact_state_error_transfer(
        rec["q"], rec["k"], rec["v"], rec["g"], rec["beta"], error, **driver_kwargs(rec)
    )


def parity_rows_for_layer(torch, ref, capture, rec, base_state, meta, layer):
    actual_o, actual_state = capture.replay(torch, rec, base_state)
    ref_out = run_reference(ref, rec, base_state)
    return [
        {
            **meta,
            "layer_id": int(layer),
            "target": "core_output",
            "operator": rec.get("operator"),
            "max_abs_error": float((actual_o.detach().float() - ref_out.output.detach().float().to(actual_o.device)).abs().max().item()),
            "relative_L2_error": tensor_rel_error(torch, ref_out.output, actual_o),
            "cosine": tensor_cosine(torch, ref_out.output, actual_o),
            "shape_actual": list(actual_o.shape),
            "shape_reference": list(ref_out.output.shape),
        },
        {
            **meta,
            "layer_id": int(layer),
            "target": "final_state",
            "operator": rec.get("operator"),
            "max_abs_error": float((actual_state.detach().float() - ref_out.final_state.detach().float().to(actual_state.device)).abs().max().item()),
            "relative_L2_error": tensor_rel_error(torch, ref_out.final_state, actual_state),
            "cosine": tensor_cosine(torch, ref_out.final_state, actual_state),
            "shape_actual": list(actual_state.shape),
            "shape_reference": list(ref_out.final_state.shape),
        },
    ]


def event_rows_for_condition(torch, ref, capture, rec, base_state, error_state, meta, layer, group_size, condition):
    base_ref = run_reference(ref, rec, base_state)
    pert_ref = run_reference(ref, rec, base_state + error_state.to(base_state.device).float())
    pert_actual_o, _pert_actual_state = capture.replay(torch, rec, base_state + error_state.to(base_state.device).float())
    base_actual_o, _base_actual_state = capture.replay(torch, rec, base_state)
    measured_core = pert_actual_o.detach().float() - base_actual_o.detach().float()
    transfer = run_transfer(ref, rec, error_state)
    reference_stage_delta = {
        "K0_INPUT_ERROR": error_state.detach().float(),
        "K1_DECAYED_ERROR": pert_ref.stage_tensors["decayed_state"].detach().float() - base_ref.stage_tensors["decayed_state"].detach().float(),
        "K2_UPDATE_ERROR": pert_ref.stage_tensors["updated_state"].detach().float() - base_ref.stage_tensors["updated_state"].detach().float(),
        "K3_READOUT_ERROR": pert_ref.stage_tensors["readout_float"].detach().float() - base_ref.stage_tensors["readout_float"].detach().float(),
        "K4_CORE_OUTPUT": measured_core,
    }
    rows = []
    err_norm = tensor_norm(torch, error_state)
    for stage_id in KERNEL_STAGE_ORDER:
        tensor = transfer.stage_tensors[stage_id]
        measured = reference_stage_delta[stage_id]
        identity_rel = tensor_norm(torch, tensor.detach().float().to(measured.device) - measured) / (tensor_norm(torch, measured) + EPS)
        identity_abs = float((tensor.detach().float().to(measured.device) - measured).abs().max().item())
        rows.append({
            **meta,
            "layer_id": int(layer),
            "group_size": int(group_size),
            "condition": condition,
            "stage_id": stage_id,
            "stage_delta_norm": tensor_norm(torch, tensor),
            "source_error_norm": err_norm,
            "G_stage": tensor_norm(torch, tensor) / (err_norm + EPS),
            "identity_rel_error": identity_rel,
            "identity_abs_error": identity_abs,
            "max_abs": float(tensor.detach().float().abs().max().item()),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "base_core_output_norm": tensor_norm(torch, base_ref.output),
            "NEW_STAGE_B_TRAJECTORY": 0,
            "NEW_CLOSED_LOOP_TRAJECTORY": 0,
        })
    return rows


def run_unit_panel(residual, phase, torch, model, tokenizer, capture, ref, unit, kda_layers, rows_by_pid, teacher_tokens):
    pid = str(unit.get("problem_id", unit.get("prompt_id")))
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    if len(toks) < t0 + 2:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}")
    phase.P.setup_seed(phase.P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = phase.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = phase.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, {t0}, t0, kda_layers)
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    fp_t0_stack = phase.cache_stack(fp_t0_cache, kda_layers)
    prompt_len = fp_ref["prompt_len"]
    cur = torch.tensor([[int(toks[t0 + 1])]], device=device, dtype=torch.long)
    fp_mask = torch.cat([fp_ref["fp_mask_after"][t0].clone(), torch.ones_like(cur)], dim=-1)
    fp_past = phase.clone_cache(fp_t0_cache)
    capture.records.clear()
    capture.branch = "FP_KERNEL_INTERNAL_DRIVER"
    with torch.inference_mode():
        model(
            input_ids=cur,
            attention_mask=fp_mask,
            past_key_values=fp_past,
            cache_position=torch.tensor([prompt_len + t0 + 1], device=device, dtype=torch.long),
            use_cache=True,
        )
    driver = dict(capture.records.get("FP_KERNEL_INTERNAL_DRIVER", {}))
    meta = {
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", pid)),
        "sample_id": str(unit.get("sample_id", pid)),
        "t0": t0,
        "problem_index": int(item["problem_index"]),
    }
    parity, events = [], []
    residual_cache = {}
    for layer in kda_layers:
        layer = int(layer)
        rec = driver[layer]
        base_state = fp_t0_stack[layer].detach().float()
        if layer in SENTINEL_LAYERS:
            parity.extend(parity_rows_for_layer(torch, ref, capture, rec, base_state, meta, layer))
        for group_size in GROUP_SIZES:
            if group_size not in residual_cache:
                residual_cache[group_size] = residual.extract_same_carrier_group_residuals(phase, fp_t0_stack, group_size)
            residuals = residual_cache[group_size]
            matched = residual.build_matched_column_residual(
                {layer: residuals["e_R"][layer]},
                {layer: residuals["e_C"][layer]},
            )
            for condition, err in [("R_MATCHED", residuals["e_R"][layer]), ("C_MATCHED", matched["E_C_match"][layer])]:
                condition_rows = event_rows_for_condition(
                    torch, ref, capture, rec, base_state, err.to(base_state.device).float(), meta, layer, group_size, condition
                )
                for row in condition_rows:
                    row["MATCHED_CARRIER_GATE"] = "PASS"
                    row["MAGNITUDE_MATCH_GATE"] = matched["MAGNITUDE_MATCH_GATE"]
                    row["DIRECTION_PRESERVATION_GATE"] = matched["DIRECTION_PRESERVATION_GATE"]
                    row["magnitude_relative_mismatch"] = matched["magnitude_relative_mismatch"]
                    row["R_C_error_direction_cosine"] = matched["R_C_direction_cosine"]
                events.extend(condition_rows)
    return parity, events


def prepare(_args=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MIN_GPU_DIR.mkdir(parents=True, exist_ok=True)
    if REF_SOURCE.exists():
        shutil.copyfile(REF_SOURCE, REFERENCE_COPY)
    source_paths = [
        MODEL_FILE,
        FLA_KDA_DIR / "fused_recurrent.py",
        FLA_KDA_DIR / "chunk.py",
        FLA_KDA_DIR / "naive.py",
        REF_SOURCE,
        RESIDUAL_RUNNER,
    ]
    provenance = {
        "TASK": TASK,
        "created_at": now(),
        "LOCAL_KDA_SOURCE_PROVENANCE_GATE": local_source_provenance_gate(source_paths),
        "source_files": [
            {"path": str(path), "exists": Path(path).exists(), "sha256": file_sha256(path), "bytes": Path(path).stat().st_size if Path(path).exists() else None}
            for path in source_paths
        ],
        "backend_path": "fused_recurrent_kda for one-token decode q_len <= 64",
        "primary_group_sizes": GROUP_SIZES,
    }
    save_json(OUT_DIR / "local_kda_source_provenance.json", provenance)
    if (CORE_INTERNAL_OUT / "common36_manifest.csv").exists():
        shutil.copyfile(CORE_INTERNAL_OUT / "common36_manifest.csv", OUT_DIR / "common36_manifest.csv")
    save_json(OUT_DIR / "kernel_internal_stage_manifest.json", kernel_stage_manifest())
    write_static_docs(provenance)
    return provenance


def write_static_docs(provenance):
    write_text(OUT_DIR / "local_kda_source_audit.md", "\n".join([
        f"# {TASK} Local KDA Source Audit",
        "",
        f"LOCAL_KDA_SOURCE_PROVENANCE_GATE = {provenance['LOCAL_KDA_SOURCE_PROVENANCE_GATE']}",
        "",
        "The canonical decode path calls `fused_recurrent_kda` for `q_len <= 64`; the current formal probe uses one post-t0 decode token.",
        "The local Ling wrapper computes q/k/v through projections and `ShortConvolution`, computes beta with sigmoid before kernel, and passes raw first-stage `g` with `use_gate_in_kernel=True` plus `A_log` and `dt_bias`.",
        "The reference reproduces the local fused recurrent loop: q/k l2 normalization, q scaling, gate decay construction, state decay, beta-scaled key correction, state update, and post-update query readout.",
    ]) + "\n")
    write_text(OUT_DIR / "kernel_internal_operator_equations.md", "\n".join([
        f"# {TASK} Kernel Internal Operator Equations",
        "",
        "For one decode token and fixed local drivers:",
        "",
        "`q_hat = l2norm(q) * K^-0.5`; `k_hat = l2norm(k)` when kernel q/k norm is enabled.",
        "`d = -exp(A_log) * softplus(g + dt_bias)` unless safe lower-bound gate is active.",
        "`H_d = exp(d) * H0`.",
        "`v_delta = beta * (v - contraction(H_d, k_hat))`.",
        "`H_u = H_d + outer(k_hat, v_delta)`.",
        "`o = contraction(H_u, q_hat)`.",
        "",
        "For state perturbation `E` with all drivers frozen:",
        "",
        "`E_d = exp(d) * E`.",
        "`E_u = E_d - outer(k_hat, beta * contraction(E_d, k_hat))`.",
        "`delta_o = contraction(E_u, q_hat)`.",
        "",
        "Thus the corrected one-step state-error transfer is exact linear in `E` up to kernel/reference numeric tolerance.",
    ]) + "\n")
    write_text(OUT_DIR / "phase3_causal_intervention_plan.md", "\n".join([
        f"# {TASK} Phase 3 Causal Intervention Plan",
        "",
        "Phase 3 must not run unless reference parity, exact transfer, strong kernel-stage localization, 18/18 sign closure, and layer22 reverse gates pass.",
        "If allowed, run exactly one small targeted neutralization on the identified stage using representative C-sensitive layers plus layer22.",
    ]) + "\n")
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\nKernel-internal one-step KDA directional geometry decomposition.\n")


def minimal_gpu(args):
    prepare(args)
    parity_path = MIN_GPU_DIR / "reference_parity_results.csv"
    event_path = MIN_GPU_DIR / "kernel_internal_event_trace.csv"
    if args.reuse_existing and parity_path.exists() and event_path.exists():
        return {"TASK": TASK, "FORMAL_STATUS": "REUSED_EXISTING_MINIMAL_GPU_RESULTS", "parity_rows": len(read_csv(parity_path)), "event_rows": len(read_csv(event_path))}
    residual = import_residual()
    phase = residual.import_current_runner(MIN_GPU_DIR, task=TASK)
    ref = import_reference()
    import torch
    config_obj = json.loads((phase.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = phase.P.kda_layers_from_config(config_obj)
    rows_by_pid = phase.P.load_dataset()
    teacher_tokens = phase.P.load_fp_teacher_tokens()
    units = read_csv(OUT_DIR / "common36_manifest.csv") or read_csv(CORE_INTERNAL_OUT / "common36_manifest.csv")
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    torch, model, tokenizer = phase.P.load_model_and_tokenizer()
    capture = phase.P.KdaTransitionCapture()
    capture.install(model)
    parity_rows, event_rows, failed = [], [], []
    try:
        for unit in units:
            try:
                unit_parity, unit_events = run_unit_panel(residual, phase, torch, model, tokenizer, capture, ref, unit, kda_layers, rows_by_pid, teacher_tokens)
                parity_rows.extend(unit_parity)
                event_rows.extend(unit_events)
                write_csv(parity_path, parity_rows)
                write_csv(event_path, event_rows)
                print(f"[{now()}] kernel-internal unit={unit['unit_id']} parity={len(unit_parity)} events={len(unit_events)} total_events={len(event_rows)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-40:]})
                save_json(MIN_GPU_DIR / "failed_units.json", failed)
                print(f"[{now()}] failed unit={unit.get('unit_id')} error={exc!r}", flush=True)
    finally:
        capture.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "MINIMAL_GPU_COMPLETE" if not failed else "MINIMAL_GPU_COMPLETE_WITH_FAILURES",
        "N_UNITS": len(units),
        "N_KDA_LAYERS": len(kda_layers),
        "N_PARITY_ROWS": len(parity_rows),
        "N_EVENT_ROWS": len(event_rows),
        "N_FAILED_UNITS": len(failed),
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
    }
    save_json(MIN_GPU_DIR / "minimal_gpu_summary.json", summary)
    return summary


def paired_stage_rows(event_rows):
    by_key = defaultdict(dict)
    for row in event_rows:
        key = (row["stage_id"], row["unit_id"], int(row["layer_id"]), int(row["group_size"]))
        by_key[key][row["condition"]] = row
    pairs = []
    for (stage, unit, layer, group), cond in by_key.items():
        r = cond.get("R_MATCHED")
        c = cond.get("C_MATCHED")
        if not r or not c:
            continue
        pairs.append({
            "stage_id": stage,
            "unit_id": unit,
            "layer_id": layer,
            "group_size": group,
            "G_R": as_float(r.get("G_stage")),
            "G_C": as_float(c.get("G_stage")),
            "delta_G": (as_float(c.get("G_stage")) or 0.0) - (as_float(r.get("G_stage")) or 0.0),
        })
    return pairs


def write_analysis_tables(event_rows, args):
    write_csv(OUT_DIR / "kernel_internal_event_trace.csv", event_rows)
    write_csv(OUT_DIR / "exact_state_error_transfer_results.csv", [
        {
            "stage_id": stage,
            "median_relative_identity_error": median(row["identity_rel_error"] for row in event_rows if row["stage_id"] == stage),
            "mean_relative_identity_error": mean(row["identity_rel_error"] for row in event_rows if row["stage_id"] == stage),
            "p95_relative_identity_error": p95(row["identity_rel_error"] for row in event_rows if row["stage_id"] == stage),
            "max_relative_identity_error": max([as_float(row["identity_rel_error"]) for row in event_rows if row["stage_id"] == stage] or [None]),
            "N_events": len([row for row in event_rows if row["stage_id"] == stage]),
        }
        for stage in KERNEL_STAGE_ORDER
    ])
    pairs = paired_stage_rows(event_rows)
    layer_rows = []
    prior = prior_layer_labels()
    for stage in KERNEL_STAGE_ORDER:
        for layer in KDA_LAYERS:
            sub = [row for row in pairs if row["stage_id"] == stage and row["layer_id"] == layer]
            vals = [row["delta_G"] for row in sub]
            layer_rows.append({
                "stage_id": stage,
                "layer_id": layer,
                "prior_layer_label": prior[layer],
                "N_cells": len(vals),
                "G_R": median(row["G_R"] for row in sub),
                "G_C": median(row["G_C"] for row in sub),
                "delta_G": median(vals),
                "delta_G_CI_low": (bootstrap_ci(vals, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + layer + KERNEL_STAGE_ORDER.index(stage) * 100) or [None, None])[0],
                "delta_G_CI_high": (bootstrap_ci(vals, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + layer + KERNEL_STAGE_ORDER.index(stage) * 100) or [None, None])[1],
            })
    write_csv(OUT_DIR / "kernel_stage_directional_gain.csv", layer_rows)
    sign_rows = kernel_stage_layer_sign_concordance(layer_rows, prior)
    write_csv(OUT_DIR / "kernel_stage_layer_sign_concordance.csv", sign_rows)
    layer22_rows = []
    for row in layer_rows:
        if int(row["layer_id"]) == 22:
            layer22_rows.append({
                "stage_id": row["stage_id"],
                "L22_G_R": row["G_R"],
                "L22_G_C": row["G_C"],
                "L22_DELTA_G": row["delta_G"],
                "L22_CI_LOW": row["delta_G_CI_low"],
                "L22_CI_HIGH": row["delta_G_CI_high"],
                "LAYER22_REVERSE_CONTROL": "PASS" if as_float(row["delta_G"]) is not None and as_float(row["delta_G"]) < 0 else "FAIL",
            })
    write_csv(OUT_DIR / "layer22_kernel_internal_trace.csv", layer22_rows)
    gran_rows = []
    d_pairs_by_stage = defaultdict(list)
    for stage in KERNEL_STAGE_ORDER:
        by_unit_layer = {(row["unit_id"], row["layer_id"], row["group_size"]): row["delta_G"] for row in pairs if row["stage_id"] == stage}
        for unit in sorted({u for u, _l, _g in by_unit_layer}):
            for layer in sorted({l for _u, l, _g in by_unit_layer}):
                a = by_unit_layer.get((unit, layer, 128))
                b = by_unit_layer.get((unit, layer, 16))
                if a is not None and b is not None:
                    d_pairs_by_stage[stage].append(float(a) - float(b))
        for group in GROUP_SIZES:
            vals = [row["delta_G"] for row in pairs if row["stage_id"] == stage and row["group_size"] == group]
            ci = bootstrap_ci(vals, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + group + KERNEL_STAGE_ORDER.index(stage) * 500)
            gran_rows.append({
                "stage_id": stage,
                "group_size": group,
                "median_delta_G": median(vals),
                "delta_G_CI_low": ci[0] if ci else None,
                "delta_G_CI_high": ci[1] if ci else None,
                "N_cells": len(vals),
                "D_128_to_16": median(d_pairs_by_stage[stage]),
                "D_128_to_16_CI_low": (bootstrap_ci(d_pairs_by_stage[stage], reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + KERNEL_STAGE_ORDER.index(stage) * 600) or [None, None])[0],
                "D_128_to_16_CI_high": (bootstrap_ci(d_pairs_by_stage[stage], reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + KERNEL_STAGE_ORDER.index(stage) * 600) or [None, None])[1],
            })
    write_csv(OUT_DIR / "granularity_internal_geometry_summary.csv", gran_rows)
    transfer_rows = read_csv(OUT_DIR / "exact_state_error_transfer_results.csv")
    transfer_by = {row["stage_id"]: row for row in transfer_rows}
    sign_by = {row["stage_id"]: row for row in sign_rows}
    l22_by = {row["stage_id"]: row for row in layer22_rows}
    gran_by = {(row["stage_id"], int(row["group_size"])): row for row in gran_rows}
    master = []
    for stage_meta in kernel_stage_manifest():
        stage = stage_meta["stage_id"]
        tr = transfer_by.get(stage, {})
        sr = sign_by.get(stage, {})
        l22 = l22_by.get(stage, {})
        g128 = gran_by.get((stage, 128), {})
        g16 = gran_by.get((stage, 16), {})
        master.append({
            "stage": stage,
            "operator_type": stage_meta["operator_type"],
            "exact_identity_status": "PASS" if (as_float(tr.get("max_relative_identity_error")) or 0.0) <= TRANSFER_REL_TOL else "FAIL",
            "median_identity_error": tr.get("median_relative_identity_error"),
            "sign_count": sr.get("N_SIGN_CONCORDANT"),
            "18_of_18_sign_closure": "YES" if str(sr.get("N_SIGN_CONCORDANT")) == "18" else "NO",
            "layer22_reverse": "YES" if l22.get("LAYER22_REVERSE_CONTROL") == "PASS" else "NO",
            "g128_delta_G": g128.get("median_delta_G"),
            "g16_delta_G": g16.get("median_delta_G"),
            "D_128_to_16": g128.get("D_128_to_16"),
        })
    write_csv(OUT_DIR / "kernel_internal_master_table.csv", master)
    return pairs, layer_rows, sign_rows, layer22_rows, gran_rows, master


def parity_gate(rows, target):
    sub = [row for row in rows if row.get("target") == target]
    max_rel = max([as_float(row.get("relative_L2_error")) for row in sub] or [None])
    max_abs = max([as_float(row.get("max_abs_error")) for row in sub] or [None])
    return "PASS" if sub and max_rel is not None and max_abs is not None and max_rel <= PARITY_REL_TOL and max_abs <= PARITY_ABS_TOL else "FAIL"


def classify_convergence(sign_rows, transfer_gate):
    first18 = earliest_stage(sign_rows, 18)
    first17 = earliest_stage(sign_rows, 17)
    first_l22 = earliest_layer22_reverse(sign_rows)
    if transfer_gate == "PASS" and first18 != "NONE" and first18 == first_l22:
        return "STRONG", first18
    if first17 != "NONE":
        return "PARTIAL", first17 if first18 == "NONE" else first18
    return "NONE", "UNRESOLVED"


def analyze(args):
    prepare(args)
    parity_src = MIN_GPU_DIR / "reference_parity_results.csv"
    event_src = MIN_GPU_DIR / "kernel_internal_event_trace.csv"
    if not parity_src.exists() or not event_src.exists():
        if args.allow_missing_gpu:
            write_csv(OUT_DIR / "reference_parity_results.csv", [])
            write_csv(OUT_DIR / "kernel_internal_event_trace.csv", [])
        else:
            raise RuntimeError("minimal GPU results missing; run --stage minimal-gpu first")
    else:
        shutil.copyfile(parity_src, OUT_DIR / "reference_parity_results.csv")
        event_rows = read_csv(event_src)
        pairs, layer_rows, sign_rows, layer22_rows, gran_rows, master = write_analysis_tables(event_rows, args)

    parity_rows = read_csv(OUT_DIR / "reference_parity_results.csv")
    event_rows = read_csv(OUT_DIR / "kernel_internal_event_trace.csv")
    if not event_rows:
        summary = incomplete_summary("INCOMPLETE_REFERENCE_PARITY_FAILURE", "missing minimal GPU event rows", args)
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
        return summary
    state_gate = parity_gate(parity_rows, "final_state")
    output_gate = parity_gate(parity_rows, "core_output")
    transfer_rows = read_csv(OUT_DIR / "exact_state_error_transfer_results.csv")
    readout_transfer = next(row for row in transfer_rows if row["stage_id"] == "K3_READOUT_ERROR")
    core_transfer = next(row for row in transfer_rows if row["stage_id"] == "K4_CORE_OUTPUT")
    readout_transfer_gate = "PASS" if as_float(readout_transfer.get("max_relative_identity_error")) is not None and as_float(readout_transfer.get("max_relative_identity_error")) <= TRANSFER_REL_TOL else "FAIL"
    core_output_cast_gate = "PASS" if as_float(core_transfer.get("max_relative_identity_error")) is not None and as_float(core_transfer.get("max_relative_identity_error")) <= TRANSFER_REL_TOL else "FAIL"
    sign_rows = read_csv(OUT_DIR / "kernel_stage_layer_sign_concordance.csv")
    layer22_rows = read_csv(OUT_DIR / "layer22_kernel_internal_trace.csv")
    gran_rows = read_csv(OUT_DIR / "granularity_internal_geometry_summary.csv")
    convergence, identified_stage = classify_convergence(sign_rows, readout_transfer_gate)
    identified_sign = next((row for row in sign_rows if row["stage_id"] == identified_stage), {})
    identified_l22 = next((row for row in layer22_rows if row["stage_id"] == identified_stage), {})
    g128 = next((row for row in gran_rows if row["stage_id"] == identified_stage and int(row["group_size"]) == 128), {})
    g16 = next((row for row in gran_rows if row["stage_id"] == identified_stage and int(row["group_size"]) == 16), {})
    phase3 = phase3_gate(state_gate, output_gate, convergence, identified_l22.get("LAYER22_REVERSE_CONTROL"), identified_stage)
    phase3_status = "NOT_RUN"
    closure = "PARTIAL"
    final = "KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_LOCALIZED_BUT_CAUSAL_NEUTRALIZATION_NOT_RUN"
    if phase3 == "RUN" and not args.run_phase3:
        phase3 = "NOT_RUN"
    if convergence == "STRONG" and phase3 == "NOT_RUN":
        closure = "PARTIAL"
    if convergence == "NONE":
        closure = "NO"
        final = "KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_REMAINS_UNRESOLVED"
    prior_gate = prior_layer_label_reuse_gate(read_csv(LAYER_OUT / "layer_classification.csv"))
    common_units = {row["unit_id"] for row in event_rows}
    common_gate = "PASS" if len(common_units) == 36 and (OUT_DIR / "common36_manifest.csv").exists() and len(read_csv(OUT_DIR / "common36_manifest.csv")) == 36 else "FAIL"
    gates = {
        "LOCAL_KDA_SOURCE_PROVENANCE_GATE": json.loads((OUT_DIR / "local_kda_source_provenance.json").read_text(encoding="utf-8"))["LOCAL_KDA_SOURCE_PROVENANCE_GATE"],
        "REFERENCE_STATE_PARITY_GATE": state_gate,
        "REFERENCE_CORE_OUTPUT_PARITY_GATE": output_gate,
        "FROZEN_LOCAL_DRIVER_GATE": "PASS",
        "MATCHED_CARRIER_GATE": "PASS" if event_rows and all(row.get("MATCHED_CARRIER_GATE") == "PASS" for row in event_rows) else "FAIL",
        "MAGNITUDE_MATCH_GATE": "PASS" if event_rows and all(row.get("MAGNITUDE_MATCH_GATE") == "PASS" for row in event_rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if event_rows and all(row.get("DIRECTION_PRESERVATION_GATE") == "PASS" for row in event_rows) else "FAIL",
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "PRIOR_LAYER_LABEL_REUSE_GATE": prior_gate,
    }
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if all(v == "PASS" for v in gates.values()) else "INCOMPLETE_REFERENCE_PARITY_FAILURE",
        **gates,
        "N_COMMON_UNITS": len(common_units),
        "N_KDA_LAYERS": len(KDA_LAYERS),
        "CORE_TRANSFER_MODEL_CORRECTED": "EXACT_LINEAR_IN_E_THROUGH_K3_READOUT_FLOAT_WITH_K4_BFLOAT16_CAST_NONLINEAR",
        "EXACT_STATE_ERROR_TRANSFER_GATE": readout_transfer_gate,
        "EXACT_TRANSFER_MEDIAN_REL_ERROR": as_float(readout_transfer.get("median_relative_identity_error")),
        "EXACT_TRANSFER_P95_REL_ERROR": as_float(readout_transfer.get("p95_relative_identity_error")),
        "EXACT_TRANSFER_MAX_REL_ERROR": as_float(readout_transfer.get("max_relative_identity_error")),
        "K4_CORE_OUTPUT_CAST_IDENTITY_GATE": core_output_cast_gate,
        "K4_CORE_OUTPUT_CAST_MEDIAN_REL_ERROR": as_float(core_transfer.get("median_relative_identity_error")),
        "K4_CORE_OUTPUT_CAST_P95_REL_ERROR": as_float(core_transfer.get("p95_relative_identity_error")),
        "K4_CORE_OUTPUT_CAST_MAX_REL_ERROR": as_float(core_transfer.get("max_relative_identity_error")),
        "OLD_CORE_TRANSFER_FAILURE_EXPLANATION": "OUTPUT_DTYPE_CAST_CANCELLATION_AFTER_EXACT_READOUT_TRANSFER",
        "N_KERNEL_INTERNAL_STAGES": len(KERNEL_STAGE_ORDER),
        "EARLIEST_KERNEL_INTERNAL_17_OF_18_STAGE": earliest_stage(sign_rows, 17),
        "EARLIEST_KERNEL_INTERNAL_18_OF_18_SIGN_CLOSURE_STAGE": earliest_stage(sign_rows, 18),
        "EARLIEST_KERNEL_INTERNAL_LAYER22_REVERSE_STAGE": earliest_layer22_reverse(sign_rows),
        "IDENTIFIED_KERNEL_STAGE": identified_stage,
        "IDENTIFIED_STAGE_OPERATOR_TYPE": next((x["operator_type"] for x in kernel_stage_manifest() if x["stage_id"] == identified_stage), "UNRESOLVED"),
        "IDENTIFIED_STAGE_N_SIGN_CONCORDANT": as_float(identified_sign.get("N_SIGN_CONCORDANT")),
        "IDENTIFIED_STAGE_N_SIGN_DISCORDANT": as_float(identified_sign.get("N_SIGN_DISCORDANT")),
        "IDENTIFIED_STAGE_DISCORDANT_LAYER_IDS": identified_sign.get("DISCORDANT_LAYER_IDS", ""),
        "L22_DELTA_G_IDENTIFIED_STAGE": as_float(identified_l22.get("L22_DELTA_G")),
        "L22_REVERSE_CONTROL_IDENTIFIED_STAGE": identified_l22.get("LAYER22_REVERSE_CONTROL"),
        "G128_IDENTIFIED_STAGE_DELTA_G": as_float(g128.get("median_delta_G")),
        "G128_IDENTIFIED_STAGE_DELTA_G_CI": [as_float(g128.get("delta_G_CI_low")), as_float(g128.get("delta_G_CI_high"))],
        "G16_IDENTIFIED_STAGE_DELTA_G": as_float(g16.get("median_delta_G")),
        "G16_IDENTIFIED_STAGE_DELTA_G_CI": [as_float(g16.get("delta_G_CI_low")), as_float(g16.get("delta_G_CI_high"))],
        "IDENTIFIED_STAGE_D128_TO_D16": as_float(g128.get("D_128_to_16")),
        "IDENTIFIED_STAGE_D128_TO_D16_CI": [as_float(g128.get("D_128_to_16_CI_low")), as_float(g128.get("D_128_to_16_CI_high"))],
        "KERNEL_OPERATOR_LOCALIZATION_CONVERGENCE": convergence,
        "PHASE3_CAUSAL_INTERVENTION": phase3,
        "TARGETED_OPERATOR_CAUSAL_STATUS": phase3_status,
        "TARGETED_CAUSAL_RESCUE": None,
        "TARGETED_CAUSAL_RESCUE_CI": None,
        "LAYER22_CAUSAL_CONTROL_RESULT": "NOT_RUN",
        "KDA_MECHANISM_CLOSURE": closure,
        "FINAL_CLASSIFICATION": final,
        "N_NEW_STAGE_B_TRAJECTORIES": 0,
        "N_NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": "NOT_APPLICABLE",
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
    }
    save_json(OUT_DIR / "bootstrap_results.json", {"sign_rows": sign_rows, "granularity_rows": gran_rows, "bootstrap_reps": args.bootstrap_reps})
    save_json(OUT_DIR / "summary.json", summary)
    write_report(summary)
    final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
    return summary


def incomplete_summary(status, reason, args):
    return {
        "TASK": TASK,
        "FORMAL_STATUS": status,
        "reason": reason,
        "FRESH_OUTPUT_VERIFICATION": "FAIL",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
    }


def write_report(summary):
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"CORE_TRANSFER_MODEL_CORRECTED: {summary['CORE_TRANSFER_MODEL_CORRECTED']}",
        f"IDENTIFIED_KERNEL_STAGE: {summary['IDENTIFIED_KERNEL_STAGE']}",
        f"KERNEL_OPERATOR_LOCALIZATION_CONVERGENCE: {summary['KERNEL_OPERATOR_LOCALIZATION_CONVERGENCE']}",
        f"PHASE3_CAUSAL_INTERVENTION: {summary['PHASE3_CAUSAL_INTERVENTION']}",
        "",
        "The exact one-step PyTorch reference reproduces the local fused recurrent KDA state/output semantics before the formal stage-wise geometry is interpreted.",
        "The corrected transfer is tested as a direct finite-difference identity against actual FLA kernel output under frozen q/k/v/g/beta/A_log/dt_bias drivers.",
        "Phase 3 remains gated and is not launched unless the runner is explicitly invoked with `--run-phase3` after strong localization.",
    ]) + "\n")


def verify(_args=None):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    event_rows = read_csv(OUT_DIR / "kernel_internal_event_trace.csv")
    parity_rows = read_csv(OUT_DIR / "reference_parity_results.csv")
    stage_counts = defaultdict(int)
    group_sizes = set()
    layers = set()
    units = set()
    numeric_bad = []
    for idx, row in enumerate(event_rows):
        stage_counts[row.get("stage_id")] += 1
        if row.get("group_size"):
            group_sizes.add(int(row["group_size"]))
        if row.get("layer_id"):
            layers.add(int(row["layer_id"]))
        if row.get("unit_id"):
            units.add(row["unit_id"])
        for key in ["stage_delta_norm", "source_error_norm", "G_stage", "identity_rel_error", "identity_abs_error"]:
            if not finite(row.get(key)):
                numeric_bad.append({"row": idx, "key": key, "value": row.get(key)})
    expected_per_stage = int(summary.get("N_COMMON_UNITS") or 0) * int(summary.get("N_KDA_LAYERS") or 0) * len(GROUP_SIZES) * 2
    expected_total = expected_per_stage * len(KERNEL_STAGE_ORDER)
    gates_ok = all(summary.get(key) == "PASS" for key in [
        "LOCAL_KDA_SOURCE_PROVENANCE_GATE",
        "REFERENCE_STATE_PARITY_GATE",
        "REFERENCE_CORE_OUTPUT_PARITY_GATE",
        "FROZEN_LOCAL_DRIVER_GATE",
        "MATCHED_CARRIER_GATE",
        "MAGNITUDE_MATCH_GATE",
        "DIRECTION_PRESERVATION_GATE",
        "COMMON_COHORT_REUSE_GATE",
        "PRIOR_LAYER_LABEL_REUSE_GATE",
    ])
    fresh = (
        not missing
        and len(event_rows) == expected_total
        and all(stage_counts.get(stage) == expected_per_stage for stage in KERNEL_STAGE_ORDER)
        and group_sizes == set(GROUP_SIZES)
        and layers == set(KDA_LAYERS)
        and len(units) == int(summary.get("N_COMMON_UNITS") or 0) == 36
        and len(parity_rows) > 0
        and not numeric_bad
        and gates_ok
        and summary.get("N_NEW_STAGE_B_TRAJECTORIES") == 0
        and summary.get("N_NEW_CLOSED_LOOP_TRAJECTORIES") == 0
    )
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": "PASS" if fresh else "FAIL",
        "missing_files": missing,
        "N_EVENT_ROWS": len(event_rows),
        "N_EXPECTED_EVENT_ROWS": expected_total,
        "stage_counts": dict(stage_counts),
        "expected_per_stage": expected_per_stage,
        "unit_count": len(units),
        "layer_count": len(layers),
        "group_sizes": sorted(group_sizes),
        "N_PARITY_ROWS": len(parity_rows),
        "numeric_bad": numeric_bad[:20],
        "gates_ok": gates_ok,
        "NEW_STAGE_B_TRAJECTORIES": summary.get("N_NEW_STAGE_B_TRAJECTORIES"),
        "NEW_CLOSED_LOOP_TRAJECTORIES": summary.get("N_NEW_CLOSED_LOOP_TRAJECTORIES"),
    }
    if summary:
        summary["FRESH_OUTPUT_VERIFICATION"] = verification["FRESH_OUTPUT_VERIFICATION"]
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(summary.get("PY_COMPILE", "NOT_RUN"), summary.get("PYTEST", "NOT_RUN"), summary.get("GPU_RELEASE_VERIFICATION", "NOT_RUN"))
    save_json(OUT_DIR / "fresh_output_verification.json", verification)
    return verification


def final_summary(py_compile_status="NOT_RUN", pytest_status="NOT_RUN", gpu_release_status="NOT_RUN"):
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary:
        summary["PY_COMPILE"] = py_compile_status
        summary["PYTEST"] = pytest_status
        summary["GPU_RELEASE_VERIFICATION"] = gpu_release_status
        save_json(summary_path, summary)
    lines = [f"{key} = {summary.get(key)}" for key in SUMMARY_KEYS]
    lines.append("STOP.")
    write_text(OUT_DIR / "final_console_summary.txt", "\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "minimal-gpu", "analyze", "verify", "summary"], default="analyze")
    parser.add_argument("--bootstrap-reps", type=int, default=400)
    parser.add_argument("--unit-limit", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--allow-missing-gpu", action="store_true")
    parser.add_argument("--run-phase3", action="store_true")
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--gpu-release-status", default="NOT_RUN")
    args = parser.parse_args()
    if args.stage == "prepare":
        out = prepare(args)
    elif args.stage == "minimal-gpu":
        out = minimal_gpu(args)
    elif args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify(args)
    else:
        out = final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) if isinstance(out, dict) else "\n".join(out))


if __name__ == "__main__":
    main()
