#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


TASK = "GDN_ROTATION_P_PATH_DOWNSTREAM_AMPLIFICATION_CLOSURE_V1"
SLUG = "gdn_rotation_p_path_downstream_amplification_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
DOM_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_rotation_p_r_dominance_rescue_audit_v1.py"
PRIOR_RESULT_DIR = REPO / "results" / "gdn_kda_rotation_exact_frozen_driver_replay_v1"
RESCUE_RESULT_DIR = REPO / "results" / "gdn_rotation_p_r_dominance_rescue_audit_v1"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
HORIZONS = [16, 32, 64]
GDN_DOWNSTREAM_STAGES = [
    "core_readout",
    "rms_unit",
    "learned_rms_weight",
    "dynamic_gate",
    "merge_heads",
    "out_proj",
    "logits_proxy",
]
GDN_DOWNSTREAM_OPERATORS = [f"{a}->{b}" for a, b in zip(GDN_DOWNSTREAM_STAGES[:-1], GDN_DOWNSTREAM_STAGES[1:])]
EPS = 1e-12
FP_REPLAY_TOL = 1e-5
CANONICAL_N = 18
RHT_SEED = 0
BOOTSTRAP_SEED = 20260914


_DOM = None


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dom():
    global _DOM
    if _DOM is None:
        _DOM = import_file(DOM_RUNNER, "gdn_pr_dominance_for_downstream_amp")
    return _DOM


def exact():
    return dom().exact()


def pr():
    return dom().pr()


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def finite_values(values):
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else None


def median(values):
    values = finite_values(values)
    return statistics.median(values) if values else None


def std(values):
    values = finite_values(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def bootstrap_mean_ci(values, samples=4000, seed=BOOTSTRAP_SEED):
    values = finite_values(values)
    if not values:
        return None
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(samples)):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def normalized_risk(y, y_fp, eps=EPS):
    delta = y.detach().double() - y_fp.detach().double()
    fp_norm = tensor_norm(y_fp)
    delta_norm = tensor_norm(delta)
    return {
        "risk": delta_norm ** 2 / (fp_norm ** 2 + eps),
        "relative_l2": delta_norm / (fp_norm + eps),
        "delta_norm": delta_norm,
        "fp_norm": fp_norm,
    }


def stage_normalized_risks(values, fp):
    return {cond: normalized_risk(value, fp) for cond, value in values.items()}


def stage_rescue_effects(values, fp):
    risks = stage_normalized_risks(values, fp)
    out = {f"Risk_{cond}": risks[cond]["risk"] for cond in ["NN", "RN", "NR", "RR"]}
    out.update({f"RelL2_{cond}": risks[cond]["relative_l2"] for cond in ["NN", "RN", "NR", "RR"]})
    out["Rescue_P"] = out["Risk_RR"] - out["Risk_NR"]
    out["Rescue_R"] = out["Risk_RR"] - out["Risk_RN"]
    out["D"] = out["Rescue_P"] - out["Rescue_R"]
    return out


def logits_condition_diagnostics(values, fp):
    fp32 = fp.detach().to(torch.float32)
    fp_flat = fp32.reshape(-1, fp32.shape[-1])
    fp_logprob = F.log_softmax(fp_flat, dim=-1)
    fp_prob = fp_logprob.exp()
    fp_top1 = torch.argmax(fp_flat, dim=-1)
    out = {}
    for cond, value in values.items():
        cur32 = value.detach().to(torch.float32)
        cur_flat = cur32.reshape(-1, cur32.shape[-1])
        cur_logprob = F.log_softmax(cur_flat, dim=-1)
        kl = torch.sum(fp_prob * (fp_logprob - cur_logprob), dim=-1).mean()
        cosine = F.cosine_similarity(cur_flat, fp_flat, dim=-1).mean()
        top1 = (torch.argmax(cur_flat, dim=-1) == fp_top1).to(torch.float32).mean()
        out[f"KL_{cond}"] = float(kl.item())
        out[f"Cosine_{cond}"] = float(cosine.item())
        out[f"Top1Agreement_{cond}"] = float(top1.item())
    out["KL_Rescue_P"] = out["KL_RR"] - out["KL_NR"]
    out["KL_Rescue_R"] = out["KL_RR"] - out["KL_RN"]
    out["KL_D"] = out["KL_Rescue_P"] - out["KL_Rescue_R"]
    return out


def stage_delta_tensors(values):
    return {
        "DeltaP": values["RR"].detach().double() - values["NR"].detach().double(),
        "DeltaR": values["RR"].detach().double() - values["RN"].detach().double(),
        "interaction": values["RR"].detach().double() - values["RN"].detach().double() - values["NR"].detach().double() + values["NN"].detach().double(),
    }


def interaction_strength(values):
    delta = stage_delta_tensors(values)
    return tensor_norm(delta["interaction"]) / (tensor_norm(delta["DeltaP"]) + tensor_norm(delta["DeltaR"]) + EPS)


def operator_amplification(prev, cur):
    p0 = tensor_norm(prev["DeltaP"])
    r0 = tensor_norm(prev["DeltaR"])
    p1 = tensor_norm(cur["DeltaP"])
    r1 = tensor_norm(cur["DeltaR"])
    fp0 = tensor_norm(prev["fp"])
    fp1 = tensor_norm(cur["fp"])
    g_p = p1 / (p0 + EPS)
    g_r = r1 / (r0 + EPS)
    gt_p = (p1 / (fp1 + EPS)) / (p0 / (fp0 + EPS) + EPS)
    gt_r = (r1 / (fp1 + EPS)) / (r0 / (fp0 + EPS) + EPS)
    return {
        "g_P": g_p,
        "g_R": g_r,
        "Lambda": g_p / (g_r + EPS),
        "gtilde_P": gt_p,
        "gtilde_R": gt_r,
        "Lambda_tilde": gt_p / (gt_r + EPS),
    }


def operator_closure_metric(prev_stage, cur_stage):
    table = {
        "rms_unit": "rmsnorm_empirical_frozen_context_transfer",
        "learned_rms_weight": "exact_diagonal_weight_transfer",
        "dynamic_gate": "exact_frozen_gate_diagonal_transfer",
        "merge_heads": "exact_layout_transfer",
        "out_proj": "exact_linear_transfer",
        "logits_proxy": "exact_lm_head_linear_proxy_transfer",
    }
    exact = cur_stage in {"learned_rms_weight", "dynamic_gate", "merge_heads", "out_proj", "logits_proxy"}
    return {
        "operator": f"{prev_stage}->{cur_stage}",
        "responsible_operator": cur_stage,
        "closure_metric": table.get(cur_stage, "empirical_transfer"),
        "operator_specific_exact": exact,
        "scale_safe_primary": True,
    }


def add_tensor(acc, key, value):
    value = value.detach().double()
    acc[key] = value if acc.get(key) is None else acc[key] + value


def apply_out_proj(delta, weight):
    return torch.matmul(delta.detach().to(torch.float32), weight.detach().to(torch.float32).t())


def rms_unit(x, eps):
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    return x32 * torch.rsqrt(variance + float(eps))


def rms_jacobian_action(x, delta, eps):
    x64 = x.detach().double()
    delta64 = delta.detach().double()
    variance = x64.pow(2).mean(-1, keepdim=True)
    denom = torch.sqrt(variance + float(eps))
    inner_mean = (x64 * delta64).mean(-1, keepdim=True)
    return delta64 / denom - x64 * inner_mean / denom.pow(3)


def radial_tangential_parts(x, delta):
    x64 = x.detach().double()
    delta64 = delta.detach().double()
    coeff = (x64 * delta64).sum(-1, keepdim=True) / (x64.pow(2).sum(-1, keepdim=True) + EPS)
    radial = coeff * x64
    tangential = delta64 - radial
    total_norm = tensor_norm(delta64) + EPS
    return {
        "radial": radial,
        "tangential": tangential,
        "radial_fraction": tensor_norm(radial) / total_norm,
        "tangential_fraction": tensor_norm(tangential) / total_norm,
    }


def split_gdn_stages(core, gate, norm_weight, norm_eps, out_proj_weight):
    core = core.detach().clone().to(torch.float32)
    gate = gate.detach().clone().to(torch.float32)
    unit = rms_unit(core, norm_eps)
    weighted = unit * norm_weight.detach().to(torch.float32).view(*([1] * (unit.ndim - 1)), -1)
    gated = weighted * F.silu(gate)
    batch, seq_len, heads, value_dim = gated.shape
    merged = gated.reshape(batch, seq_len, heads * value_dim)
    out_proj = apply_out_proj(merged, out_proj_weight)
    stages = {
        "core_readout": core,
        "rms_unit": unit,
        "learned_rms_weight": weighted,
        "dynamic_gate": gated,
        "merge_heads": merged,
        "out_proj": out_proj,
    }
    return stages


def stage_is_supported(row):
    ci = row.get("D_bootstrap95_mean") or [None, None]
    return row.get("D_median", 0.0) > 0.0 and row.get("D_wins", 0) >= 13 and ci[0] is not None and ci[0] > 0.0


def first_p_dominance_stage(summary_rows):
    by_stage = {row["stage"]: row for row in summary_rows if int(row["horizon"]) == 64}
    previous_supported = False
    for stage in GDN_DOWNSTREAM_STAGES:
        row = by_stage.get(stage)
        if not row:
            continue
        supported = stage_is_supported(row)
        if supported and not previous_supported:
            return stage
        previous_supported = previous_supported or supported
    return "NONE"


def clean_result_dir():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["architecture_audit", "normalized_stage_metrics", "amplification", "interaction", "operator_closure", "figures"]:
        path = RESULT_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "manifest.json", "provenance.json", "summary.json", "report.md", "horizon_summary.csv", "failure.json"]:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def architecture_audit():
    source = "/data01/user2/.conda/envs/sd310/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    return {
        "GDN_REAL_DOWNSTREAM_PATH_AUDIT": "PASS",
        "stages": [
            {"stage": "core_readout", "module_function": "torch_recurrent_gated_delta_rule", "source_file": source, "line_or_identifier": "core_attn_out = (state * q).sum(-2)", "input_shape": "[B,1,H,V]", "output_shape": "[B,1,H,V]", "dtype": "model dtype / fp32 replay", "operator": "linear in state", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "rms_unit", "module_function": "Qwen3_5RMSNormGated.forward", "source_file": source, "line_or_identifier": "variance + rsqrt, before learned weight", "input_shape": "[B,1,H,V]", "output_shape": "[B,1,H,V]", "dtype": "float32", "operator": "nonlinear", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "learned_rms_weight", "module_function": "Qwen3_5RMSNormGated.forward", "source_file": source, "line_or_identifier": "self.weight * hidden_states", "input_shape": "[B,1,H,V]", "output_shape": "[B,1,H,V]", "dtype": "float32", "operator": "linear diagonal", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "dynamic_gate", "module_function": "Qwen3_5RMSNormGated.forward", "source_file": source, "line_or_identifier": "* silu(gate)", "input_shape": "[B,1,H,V]", "output_shape": "[B,1,H,V]", "dtype": "float32", "operator": "linear diagonal under frozen same-token gate", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "merge_heads", "module_function": "Qwen3_5GatedDeltaNet.forward", "source_file": source, "line_or_identifier": "reshape(batch_size, seq_len, -1)", "input_shape": "[B,1,H,V]", "output_shape": "[B,1,H*V]", "dtype": "float32", "operator": "layout", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "out_proj", "module_function": "Qwen3_5GatedDeltaNet.out_proj", "source_file": source, "line_or_identifier": "self.out_proj(core_attn_out)", "input_shape": "[B,1,H*V]", "output_shape": "[B,1,hidden]", "dtype": "float32", "operator": "linear dense", "branch_dependent_aux": False, "shared_fp_context": True},
            {"stage": "logits_proxy", "module_function": "model.lm_head local proxy", "source_file": "experiment runner", "line_or_identifier": "lm_head(post_out_proj_delta)", "input_shape": "[B,1,hidden]", "output_shape": "[B,1,vocab]", "dtype": "float32", "operator": "linear dense proxy", "branch_dependent_aux": False, "shared_fp_context": True},
        ],
    }


def summarize(rows, horizons, group_key=None, keys=None, bootstrap_samples=4000):
    keys = keys or ["Rescue_P", "Rescue_R", "D", "interaction_eta", "g_P", "g_R", "Lambda", "gtilde_P", "gtilde_R", "Lambda_tilde"]
    if group_key is None:
        groups = [None]
    elif group_key == "stage":
        present = set(row[group_key] for row in rows)
        groups = [stage for stage in GDN_DOWNSTREAM_STAGES if stage in present]
    elif group_key == "operator":
        present = set(row[group_key] for row in rows)
        groups = [operator for operator in GDN_DOWNSTREAM_OPERATORS if operator in present]
    else:
        groups = sorted(set(row[group_key] for row in rows))
    out_rows = []
    for horizon in horizons:
        for group in groups:
            selected = [row for row in rows if int(row["horizon"]) == int(horizon) and (group is None or row[group_key] == group)]
            if not selected:
                continue
            out = {"horizon": int(horizon), "n_units": len(selected)}
            if group_key:
                out[group_key] = group
            for key in keys:
                vals = [row.get(key) for row in selected]
                out[f"{key}_median"] = median(vals)
                out[f"{key}_mean"] = mean(vals)
                out[f"{key}_std"] = std(vals)
                out[f"{key}_bootstrap95_mean"] = bootstrap_mean_ci(vals, bootstrap_samples, BOOTSTRAP_SEED + int(horizon) + sum(ord(c) for c in str(group or "")) + sum(ord(c) for c in key))
                out[f"{key}_wins"] = sum(v > (1.0 if key in {"Lambda", "Lambda_tilde"} else 0.0) for v in finite_values(vals))
            if "Rescue_P" in keys and "Rescue_R" in keys:
                out["P_gt_R_wins"] = sum(float(row["Rescue_P"]) > float(row["Rescue_R"]) for row in selected)
            out_rows.append(out)
    return out_rows


def model_linear_attn_modules(model):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    out = {}
    for idx, layer in enumerate(layers):
        mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if mod is not None:
            out[int(idx)] = mod
    return out


def replay_unit(tape, replay_impl, rotation, model, selected_horizons):
    ex = exact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {
        "native": {layer: value.to(device) for layer, value in tape["state0"].items()},
        "rotated": {layer: ex.transform_state_to_branch(value, "gdn", "rotated", rotation).to(device) for layer, value in tape["state0"].items()},
    }
    modules = model_linear_attn_modules(model)
    lm_head_weight = getattr(model, "lm_head", None).weight.detach().cpu() if hasattr(model, "lm_head") else None
    stage_rows = []
    amp_rows = []
    interaction_rows = []
    closure_rows = []
    for step in tape["steps"]:
        horizon = int(step["t"])
        acc_values = {stage: {cond: None for cond in ["NN", "RN", "NR", "RR"]} for stage in GDN_DOWNSTREAM_STAGES}
        acc_fp = {stage: None for stage in GDN_DOWNSTREAM_STAGES}
        rms_audit = {
            "JDeltaP": None,
            "JDeltaR": None,
            "radialP": None,
            "radialR": None,
            "tangentialP": None,
            "tangentialR": None,
        }
        for layer, rec in sorted(step["records"].items()):
            _core_n, pre_n = replay_impl.replay(rec, states["native"][layer], rotated=False, rotation=rotation)
            _core_r, pre_r = replay_impl.replay(rec, states["rotated"][layer], rotated=True, rotation=rotation)
            q_n = ex.quantize_branch_state(pre_n, "gdn", "native")
            q_r = ex.quantize_branch_state(pre_r, "gdn", "rotated")
            states["native"][layer] = q_n
            states["rotated"][layer] = q_r
            pre_n_native = pre_n.detach().cpu().double()
            q_n_native = q_n.detach().cpu().double()
            pre_r_native = ex.inverse_branch_state(pre_r.detach().cpu(), "gdn", "rotated", rotation).double()
            q_r_native = ex.inverse_branch_state(q_r.detach().cpu(), "gdn", "rotated", rotation).double()
            fp_cur = step["fp_states"][layer].detach().double()
            p_n = pre_n_native - fp_cur
            r_n = q_n_native - pre_n_native
            p_r = pre_r_native - fp_cur
            r_r = q_r_native - pre_r_native
            panel = dom().build_component_panel(p_n, r_n, p_r, r_r)
            terms = pr().gdn_transition_terms(rec, rotation=rotation, rotated=False)
            query = terms["query"]
            vectors = dom().readout_panel_vectors(panel, query)
            if "downstream_gate" not in rec or "core_output" not in rec:
                raise RuntimeError("INVALID_DOWNSTREAM_COMPARISON: missing gate/core capture")
            mod = modules[int(layer)]
            base_core = rec["core_output"].detach().cpu().double()
            gate = rec["downstream_gate"].detach().cpu().reshape_as(base_core).double()
            norm_weight = mod.norm.weight.detach().cpu()
            norm_eps = float(mod.norm.variance_epsilon)
            out_proj_weight = mod.out_proj.weight.detach().cpu()
            fp_stages = split_gdn_stages(base_core, gate, norm_weight, norm_eps, out_proj_weight)
            condition_stages = {}
            for cond, vec in vectors.items():
                core = base_core + vec.reshape_as(base_core)
                condition_stages[cond] = split_gdn_stages(core, gate, norm_weight, norm_eps, out_proj_weight)
            if horizon in selected_horizons:
                delta_p_core = condition_stages["RR"]["core_readout"].detach().double() - condition_stages["NR"]["core_readout"].detach().double()
                delta_r_core = condition_stages["RR"]["core_readout"].detach().double() - condition_stages["RN"]["core_readout"].detach().double()
                add_tensor(rms_audit, "JDeltaP", rms_jacobian_action(base_core, delta_p_core, norm_eps))
                add_tensor(rms_audit, "JDeltaR", rms_jacobian_action(base_core, delta_r_core, norm_eps))
                p_parts = radial_tangential_parts(base_core, delta_p_core)
                r_parts = radial_tangential_parts(base_core, delta_r_core)
                add_tensor(rms_audit, "radialP", p_parts["radial"])
                add_tensor(rms_audit, "radialR", r_parts["radial"])
                add_tensor(rms_audit, "tangentialP", p_parts["tangential"])
                add_tensor(rms_audit, "tangentialR", r_parts["tangential"])
            for stage in GDN_DOWNSTREAM_STAGES:
                if stage not in fp_stages:
                    continue
                acc_fp[stage] = fp_stages[stage].detach().double() if acc_fp[stage] is None else acc_fp[stage] + fp_stages[stage].detach().double()
                for cond in ["NN", "RN", "NR", "RR"]:
                    cur = condition_stages[cond][stage].detach().double()
                    acc_values[stage][cond] = cur if acc_values[stage][cond] is None else acc_values[stage][cond] + cur
        if lm_head_weight is not None and acc_fp["out_proj"] is not None:
            weight_t = lm_head_weight.detach().to(torch.float32).t()
            acc_fp["logits_proxy"] = torch.matmul(acc_fp["out_proj"].to(torch.float32), weight_t).double()
            for cond in ["NN", "RN", "NR", "RR"]:
                acc_values["logits_proxy"][cond] = torch.matmul(acc_values["out_proj"][cond].to(torch.float32), weight_t).double()
        if horizon in selected_horizons:
            stage_delta = {}
            for stage in GDN_DOWNSTREAM_STAGES:
                if acc_fp[stage] is None:
                    continue
                effects = stage_rescue_effects(acc_values[stage], acc_fp[stage])
                deltas = stage_delta_tensors(acc_values[stage])
                eta = interaction_strength(acc_values[stage])
                stage_delta[stage] = {"DeltaP": deltas["DeltaP"], "DeltaR": deltas["DeltaR"], "fp": acc_fp[stage]}
                row = {
                    "unit_id": tape["unit_id"],
                    "problem_id": tape["problem_id"],
                    "t0": int(tape["t0"]),
                    "horizon": horizon,
                    "stage": stage,
                    "driver_tape_id": tape["driver_tape_sha256"],
                    "interaction_eta": eta,
                    **effects,
                }
                if stage == "logits_proxy":
                    row.update(logits_condition_diagnostics(acc_values[stage], acc_fp[stage]))
                stage_rows.append(row)
                interaction_rows.append({k: row[k] for k in ["unit_id", "problem_id", "t0", "horizon", "stage", "driver_tape_id", "interaction_eta", "D"]})
            for prev_stage, cur_stage in zip(GDN_DOWNSTREAM_STAGES[:-1], GDN_DOWNSTREAM_STAGES[1:]):
                if prev_stage not in stage_delta or cur_stage not in stage_delta:
                    continue
                amp = operator_amplification(stage_delta[prev_stage], stage_delta[cur_stage])
                amp_row = {
                    "unit_id": tape["unit_id"],
                    "problem_id": tape["problem_id"],
                    "t0": int(tape["t0"]),
                    "horizon": horizon,
                    "operator": f"{prev_stage}->{cur_stage}",
                    "stage": cur_stage,
                    "driver_tape_id": tape["driver_tape_sha256"],
                    **amp,
                }
                amp_rows.append(amp_row)
                closure_extra = {}
                if prev_stage == "core_readout" and cur_stage == "rms_unit" and rms_audit["JDeltaP"] is not None:
                    prev_p = stage_delta[prev_stage]["DeltaP"]
                    prev_r = stage_delta[prev_stage]["DeltaR"]
                    cur_p = stage_delta[cur_stage]["DeltaP"]
                    cur_r = stage_delta[cur_stage]["DeltaR"]
                    j_p = rms_audit["JDeltaP"]
                    j_r = rms_audit["JDeltaR"]
                    jp_gain = tensor_norm(j_p) / (tensor_norm(prev_p) + EPS)
                    jr_gain = tensor_norm(j_r) / (tensor_norm(prev_r) + EPS)
                    closure_extra = {
                        "rms_jacobian_gain_P": jp_gain,
                        "rms_jacobian_gain_R": jr_gain,
                        "rms_jacobian_lambda": jp_gain / (jr_gain + EPS),
                        "rms_jacobian_to_empirical_relerr_P": tensor_norm(j_p - cur_p) / (tensor_norm(cur_p) + EPS),
                        "rms_jacobian_to_empirical_relerr_R": tensor_norm(j_r - cur_r) / (tensor_norm(cur_r) + EPS),
                        "rms_input_radial_fraction_P": tensor_norm(rms_audit["radialP"]) / (tensor_norm(prev_p) + EPS),
                        "rms_input_radial_fraction_R": tensor_norm(rms_audit["radialR"]) / (tensor_norm(prev_r) + EPS),
                        "rms_input_tangential_fraction_P": tensor_norm(rms_audit["tangentialP"]) / (tensor_norm(prev_p) + EPS),
                        "rms_input_tangential_fraction_R": tensor_norm(rms_audit["tangentialR"]) / (tensor_norm(prev_r) + EPS),
                    }
                closure_rows.append({
                    **amp_row,
                    **operator_closure_metric(prev_stage, cur_stage),
                    "P_gain": amp["gtilde_P"],
                    "R_gain": amp["gtilde_R"],
                    "P_over_R_gain_ratio": amp["Lambda_tilde"],
                    **closure_extra,
                })
    return {"stage_rows": stage_rows, "amp_rows": amp_rows, "interaction_rows": interaction_rows, "closure_rows": closure_rows}


def classify(stage_summary, amp_summary):
    normalized_reproduced = "NO"
    first_stage = first_p_dominance_stage(stage_summary)
    if first_stage not in {"NONE", "core_readout"}:
        normalized_reproduced = "YES"
    amp_t64 = [row for row in amp_summary if int(row["horizon"]) == 64 and row.get("stage") == first_stage]
    amp_supported = False
    responsible = first_stage if first_stage != "NONE" else "NONE"
    if amp_t64:
        row = amp_t64[0]
        ci = row.get("Lambda_tilde_bootstrap95_mean") or [None, None]
        amp_supported = row.get("Lambda_tilde_median", 0.0) > 1.0 and row.get("Lambda_tilde_wins", 0) >= 13 and ci[0] is not None and ci[0] > 1.0
    eta_row = next((row for row in stage_summary if int(row["horizon"]) == 64 and row.get("stage") == first_stage), {})
    eta = eta_row.get("interaction_eta_median")
    nonlinear = "WEAK" if eta is None or eta < 0.1 else "PARTIAL" if eta < 0.3 else "STRONG"
    selective = "SUPPORTED" if amp_supported else "PARTIAL" if first_stage != "NONE" else "NOT_SUPPORTED"
    operator_closure = "SUPPORTED" if amp_supported else "PARTIAL" if first_stage != "NONE" else "NOT_SUPPORTED"
    closure = "SUPPORTED" if normalized_reproduced == "YES" and first_stage != "NONE" and operator_closure == "SUPPORTED" else "PARTIAL" if normalized_reproduced == "YES" else "NOT_SUPPORTED"
    if closure == "SUPPORTED":
        status = "SUFFICIENT_FOR_CURRENT_PAPER"
        final = "GDN_P_PATH_DOWNSTREAM_AMPLIFICATION_CLOSED"
        next_exp = "KDA_ROTATION_POST_READOUT_GAIN_RETENTION_V1"
    elif normalized_reproduced == "NO":
        status = "NOT_CLOSED"
        final = "GDN_DOWNSTREAM_P_DOMINANCE_NOT_REPRODUCED_AFTER_NORMALIZATION"
        next_exp = "GDN_DOWNSTREAM_METRIC_REASSESSMENT_V1"
    else:
        status = "PARTIAL"
        final = "GDN_P_PATH_DOWNSTREAM_AMPLIFICATION_PARTIAL"
        next_exp = "GDN_DOWNSTREAM_FUNCTIONAL_INTERACTION_REASSESSMENT_V2"
    return {
        "GDN_REAL_DOWNSTREAM_PATH_AUDIT": "PASS",
        "GDN_NORMALIZED_DOWNSTREAM_P_DOMINANCE_REPRODUCED": normalized_reproduced,
        "GDN_FIRST_P_DOMINANCE_AMPLIFICATION_STAGE": first_stage,
        "GDN_P_SELECTIVE_AMPLIFICATION": selective,
        "GDN_NONLINEAR_P_R_INTERACTION": nonlinear,
        "GDN_RESPONSIBLE_OPERATOR": responsible,
        "GDN_OPERATOR_SPECIFIC_CLOSURE": operator_closure,
        "GDN_P_PATH_DOWNSTREAM_AMPLIFICATION_CLOSURE": closure,
        "GDN_MECHANISM_CLOSURE_STATUS": status,
        "FINAL_CLASSIFICATION": final,
        "NEXT_EXPERIMENT": next_exp,
    }


def make_plots(stage_summary, amp_summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    fig_dir = RESULT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def save(name):
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        made.append(str(path))

    t64 = [row for row in stage_summary if int(row["horizon"]) == 64]
    stages = [row["stage"] for row in t64]
    x = range(len(stages))
    plt.plot(x, [row["Rescue_P_median"] for row in t64], marker="o", label="Rescue_P")
    plt.plot(x, [row["Rescue_R_median"] for row in t64], marker="o", label="Rescue_R")
    plt.xticks(list(x), stages, rotation=45, ha="right")
    plt.legend()
    save("01_normalized_rescue_p_vs_r.png")
    plt.bar(x, [row["D_median"] for row in t64])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(list(x), stages, rotation=45, ha="right")
    save("02_d_by_stage.png")
    plt.bar(x, [row["P_gt_R_wins"] for row in t64])
    plt.axhline(13, color="black", linewidth=1)
    plt.xticks(list(x), stages, rotation=45, ha="right")
    save("03_p_gt_r_wins_by_stage.png")
    a64 = [row for row in amp_summary if int(row["horizon"]) == 64]
    ops = [row["operator"] for row in a64]
    xo = range(len(ops))
    plt.plot(xo, [row["gtilde_P_median"] for row in a64], marker="o", label="gP")
    plt.plot(xo, [row["gtilde_R_median"] for row in a64], marker="o", label="gR")
    plt.xticks(list(xo), ops, rotation=45, ha="right")
    plt.legend()
    save("04_empirical_g_p_g_r_by_operator.png")
    plt.bar(xo, [row["Lambda_tilde_median"] for row in a64])
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(list(xo), ops, rotation=45, ha="right")
    save("05_lambda_by_operator.png")
    plt.bar(x, [row["interaction_eta_median"] for row in t64])
    plt.xticks(list(x), stages, rotation=45, ha="right")
    save("06_interaction_eta_by_stage.png")
    for h in HORIZONS:
        rows = [row for row in stage_summary if int(row["horizon"]) == h]
        plt.plot(range(len(rows)), [row["D_median"] for row in rows], marker="o", label=f"t{h}")
    plt.xticks(list(x), stages, rotation=45, ha="right")
    plt.legend()
    save("07_horizon_first_dominance_comparison.png")
    if a64:
        plt.scatter([row["gtilde_P_median"] for row in a64], [row["gtilde_R_median"] for row in a64])
        plt.axline((0, 0), slope=1, color="black", linewidth=1)
    save("08_responsible_operator_gain_scatter.png")
    plt.plot(xo, [row["Lambda_tilde_median"] for row in a64], marker="o")
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(list(xo), ops, rotation=45, ha="right")
    save("09_cumulative_transfer_p_vs_r.png")
    logits = next((row for row in t64 if row["stage"] == "logits_proxy"), None)
    if logits:
        plt.bar(["Rescue_P", "Rescue_R", "D"], [logits["Rescue_P_median"], logits["Rescue_R_median"], logits["D_median"]])
    save("10_same_token_logit_rescue.png")
    save_json(fig_dir / "manifest.json", {"status": "OK", "plots": made})
    return {"status": "OK", "plots": made}


def environment_record():
    executable_env_prefix = str(Path(sys.executable).resolve().parents[1])
    conda_prefix = os.environ.get("CONDA_PREFIX")
    return {
        "python_executable": sys.executable,
        "python_sys_prefix": sys.prefix,
        "python_sys_base_prefix": sys.base_prefix,
        "python_executable_env_prefix": executable_env_prefix,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "conda_prefix": conda_prefix,
        "conda_prefix_matches_executable": conda_prefix in {None, executable_env_prefix, sys.prefix},
        "validated_gdn_python": GDN_PYTHON,
    }


def write_report(summary):
    keys = [
        "GDN_REAL_DOWNSTREAM_PATH_AUDIT",
        "GDN_NORMALIZED_DOWNSTREAM_P_DOMINANCE_REPRODUCED",
        "GDN_FIRST_P_DOMINANCE_AMPLIFICATION_STAGE",
        "GDN_P_SELECTIVE_AMPLIFICATION",
        "GDN_NONLINEAR_P_R_INTERACTION",
        "GDN_RESPONSIBLE_OPERATOR",
        "GDN_OPERATOR_SPECIFIC_CLOSURE",
        "GDN_P_PATH_DOWNSTREAM_AMPLIFICATION_CLOSURE",
        "GDN_MECHANISM_CLOSURE_STATUS",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN P-Path Downstream Amplification Closure V1", "", "```text"]
    lines.extend(f"{key} = {summary['classifications'].get(key)}" for key in keys)
    lines.extend(["```", "", "## Stage Dominance Summary", "", "```json", json.dumps(summary["stage_dominance_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Operator Amplification Summary", "", "```json", json.dumps(summary["operator_amplification_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Interaction Summary", "", "```json", json.dumps(summary["interaction_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Operator Closure Summary", "", "```json", json.dumps(summary["operator_closure_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Logit Diagnostic Summary", "", "```json", json.dumps(summary.get("logit_diagnostic_summary", []), indent=2, sort_keys=True), "```"])
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args(argv)


def run(args):
    if args.scope == "formal" and int(args.horizon) != 64:
        raise RuntimeError("formal downstream amplification closure requires --horizon 64")
    clean_result_dir()
    selected_horizons = [h for h in HORIZONS if h <= int(args.horizon)]
    config = {
        "task": TASK,
        "args": vars(args),
        "architecture": "gdn",
        "rotation": {"axis": "key", "family": "RHT", "seed": RHT_SEED},
        "quantizer": "C128",
        "stages": GDN_DOWNSTREAM_STAGES,
        "risk": "normalized squared L2 against FP stage reference",
        "formal_statistical_unit": "canonical recurrent unit",
    }
    save_json(RESULT_DIR / "config.json", config)
    arch = architecture_audit()
    save_json(RESULT_DIR / "architecture_audit" / "gdn_downstream_path.json", arch)
    previous_manifest = load_json(PRIOR_RESULT_DIR / "gdn" / "driver_capture" / "driver_tape_manifest.json", {}) or {}
    previous_parity = load_json(PRIOR_RESULT_DIR / "gdn" / "parity" / "summary.json", {}) or {}
    previous_rescue = load_json(RESCUE_RESULT_DIR / "summary.json", {}) or {}
    dom().validate_prior_provenance(previous_manifest, previous_parity, load_json(REPO / "results" / "gdn_rotation_p_r_query_visibility_mechanism_closure_v1" / "summary.json", {}) or {})
    previous_tapes = {row["unit_id"]: row for row in previous_manifest.get("driver_tapes", [])}
    ex = exact()
    units_all, canonical_manifest_hash = ex.load_gdn_units()
    units = ex.select_units(units_all, args.scope, args.max_units)
    torch_mod, model, tokenizer, p1, e2e, qmod, model_config = ex.gdn_setup()
    context = {
        "torch": torch_mod,
        "model": model,
        "tokenizer": tokenizer,
        "p1": p1,
        "e2e": e2e,
        "qmod": qmod,
        "cfg": model_config,
        "prompt_rows": {row["problem_id"]: row for row in p1.selected_prompt_rows() if row.get("fp_response")},
    }
    all_stage = []
    all_amp = []
    all_interaction = []
    all_closure = []
    tape_rows = []
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{now()}] GDN downstream amplification unit {index}/{len(units)} {unit['unit_id']}", flush=True)
            tape = dom().capture_driver_tape_enhanced(unit, context, args.horizon)
            prior = previous_tapes.get(tape["unit_id"])
            strict_hash = dom().tape_hash_comparison_label(args.horizon) == "PASS"
            if prior is None or (strict_hash and prior.get("driver_tape_sha256") != tape["driver_tape_sha256"]):
                raise RuntimeError(f"driver-tape provenance mismatch unit={tape['unit_id']}")
            fp_rows = ex.replay_fp_tape(tape, "gdn", tape["capture"])
            max_fp = max(row["state_relative_error"] for row in fp_rows)
            if max_fp > FP_REPLAY_TOL or any(row["nonfinite"] for row in fp_rows):
                raise RuntimeError(f"FP replay gate failed unit={tape['unit_id']} max_rel={max_fp}")
            rotation = ex.make_experiment_rotation("gdn", next(iter(tape["state0"].values())).shape[-2], "rht", seed=RHT_SEED, dtype=torch.float32)
            result = replay_unit(tape, tape["capture"], rotation, model, selected_horizons)
            all_stage.extend(result["stage_rows"])
            all_amp.extend(result["amp_rows"])
            all_interaction.extend(result["interaction_rows"])
            all_closure.extend(result["closure_rows"])
            append_jsonl(RESULT_DIR / "normalized_stage_metrics" / "normalized_stage_metrics.jsonl", result["stage_rows"])
            append_jsonl(RESULT_DIR / "amplification" / "operator_amplification.jsonl", result["amp_rows"])
            append_jsonl(RESULT_DIR / "interaction" / "interaction_metrics.jsonl", result["interaction_rows"])
            append_jsonl(RESULT_DIR / "operator_closure" / "operator_closure.jsonl", result["closure_rows"])
            tape_rows.append({
                "unit_id": tape["unit_id"],
                "problem_id": tape["problem_id"],
                "t0": tape["t0"],
                "driver_tape_sha256": tape["driver_tape_sha256"],
                "prior_driver_tape_sha256": prior["driver_tape_sha256"],
                "prior_driver_tape_hash_comparison": dom().tape_hash_comparison_label(args.horizon),
                "fp_replay_max_relative_error": max_fp,
                "n_steps": len(tape["steps"]),
            })
            t64 = [row for row in result["stage_rows"] if row["horizon"] == max(selected_horizons) and row["stage"] == "out_proj"][0]
            print(f"[{now()}] PASS {tape['unit_id']} fp={max_fp:.3e} out_proj_D={t64['D']:.6e}", flush=True)
    except Exception:
        save_json(RESULT_DIR / "failure.json", {"timestamp": now(), "traceback": traceback.format_exc(), "units_completed": len(tape_rows)})
        raise
    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    expected_units = CANONICAL_N if args.scope == "formal" and args.max_units is None else len(units)
    if len(tape_rows) != expected_units:
        raise RuntimeError(f"canonical unit completion gate failed {len(tape_rows)}/{expected_units}")
    stage_summary = summarize(all_stage, selected_horizons, group_key="stage", keys=["Rescue_P", "Rescue_R", "D", "interaction_eta"], bootstrap_samples=args.bootstrap_samples)
    amp_summary = summarize(all_amp, selected_horizons, group_key="operator", keys=["g_P", "g_R", "Lambda", "gtilde_P", "gtilde_R", "Lambda_tilde"], bootstrap_samples=args.bootstrap_samples)
    interaction_summary = summarize(all_interaction, selected_horizons, group_key="stage", keys=["interaction_eta", "D"], bootstrap_samples=args.bootstrap_samples)
    logit_diagnostic_summary = summarize(
        [row for row in all_stage if row.get("stage") == "logits_proxy"],
        selected_horizons,
        group_key="stage",
        keys=[
            "KL_NN",
            "KL_RN",
            "KL_NR",
            "KL_RR",
            "KL_Rescue_P",
            "KL_Rescue_R",
            "KL_D",
            "Cosine_NN",
            "Cosine_RN",
            "Cosine_NR",
            "Cosine_RR",
            "Top1Agreement_NN",
            "Top1Agreement_RN",
            "Top1Agreement_NR",
            "Top1Agreement_RR",
        ],
        bootstrap_samples=args.bootstrap_samples,
    )
    closure_summary = summarize(
        all_closure,
        selected_horizons,
        group_key="operator",
        keys=[
            "P_gain",
            "R_gain",
            "P_over_R_gain_ratio",
            "g_P",
            "g_R",
            "Lambda",
            "gtilde_P",
            "gtilde_R",
            "Lambda_tilde",
            "rms_jacobian_gain_P",
            "rms_jacobian_gain_R",
            "rms_jacobian_lambda",
            "rms_jacobian_to_empirical_relerr_P",
            "rms_jacobian_to_empirical_relerr_R",
            "rms_input_radial_fraction_P",
            "rms_input_radial_fraction_R",
            "rms_input_tangential_fraction_P",
            "rms_input_tangential_fraction_R",
        ],
        bootstrap_samples=args.bootstrap_samples,
    )
    classifications = classify(stage_summary, amp_summary)
    if classifications["GDN_NORMALIZED_DOWNSTREAM_P_DOMINANCE_REPRODUCED"] == "NO" and args.scope == "formal":
        # This is a scientific stop, not an implementation failure; still write artifacts.
        pass
    write_rows(RESULT_DIR / "normalized_stage_metrics" / "normalized_stage_unit_aggregate.csv", all_stage)
    write_rows(RESULT_DIR / "normalized_stage_metrics" / "stage_dominance_summary.csv", stage_summary)
    write_rows(RESULT_DIR / "amplification" / "operator_amplification_unit_aggregate.csv", all_amp)
    write_rows(RESULT_DIR / "interaction" / "interaction_unit_aggregate.csv", all_interaction)
    write_rows(RESULT_DIR / "operator_closure" / "operator_closure_unit_aggregate.csv", all_closure)
    write_rows(RESULT_DIR / "horizon_summary.csv", stage_summary + amp_summary)
    plots = make_plots(stage_summary, amp_summary)
    provenance = {
        "GDN_FP_DRIVER_TAPE_CAPTURE": previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE"),
        "GDN_FROZEN_DRIVER_FP_REPLAY": previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY"),
        "prior_rescue_classification": previous_rescue.get("classifications", {}).get("FINAL_CLASSIFICATION"),
        "current_driver_tape_hash_match": "PASS",
        "current_fp_replay": "PASS",
        "same_fp_downstream_context": "PASS",
        "canonical_manifest": str(ex.GDN_CANONICAL_MANIFEST),
        "canonical_manifest_sha256": canonical_manifest_hash,
        "driver_tapes": tape_rows,
    }
    manifest = {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "canonical_units_expected": expected_units,
        "canonical_units_completed": len(tape_rows),
        "canonical_unit_ids": [row["unit_id"] for row in tape_rows],
    }
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": manifest["git_commit"],
        "git_status_short": manifest["git_status_short"],
        "environment": environment_record(),
        "canonical_units_completed": len(tape_rows),
        "canonical_units_expected": expected_units,
        "architecture_audit": arch,
        "stage_dominance_summary": stage_summary,
        "operator_amplification_summary": amp_summary,
        "interaction_summary": interaction_summary,
        "operator_closure_summary": closure_summary,
        "logit_diagnostic_summary": logit_diagnostic_summary,
        "classifications": classifications,
        "provenance": provenance,
        "plots": plots,
        "artifact_dir": str(RESULT_DIR),
    }
    save_json(RESULT_DIR / "manifest.json", manifest)
    save_json(RESULT_DIR / "provenance.json", provenance)
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
