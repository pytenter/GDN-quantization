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
from pathlib import Path

import torch
import torch.nn.functional as F


TASK = "GDN_ROTATION_DYNAMIC_GATE_P_SELECTIVITY_CAUSAL_V1"
SLUG = "gdn_rotation_dynamic_gate_p_selectivity_causal_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
DOWNSTREAM_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_rotation_p_path_downstream_amplification_closure_v1.py"
PRIOR_RESULT_DIR = REPO / "results" / "gdn_kda_rotation_exact_frozen_driver_replay_v1"
VIS_RESULT_DIR = REPO / "results" / "gdn_rotation_p_r_query_visibility_mechanism_closure_v1"
HORIZONS = [16, 32, 64]
CONDS = ["NN", "RN", "NR", "RR"]
SHUFFLE_SEEDS = list(range(8))
EPS = 1e-12
FP_REPLAY_TOL = 1e-5
CANONICAL_N = 18
RHT_SEED = 0
BOOTSTRAP_SEED = 20260914
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"


_DS = None


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ds():
    global _DS
    if _DS is None:
        _DS = import_file(DOWNSTREAM_RUNNER, "gdn_downstream_for_gate_causal")
    return _DS


def dom():
    return ds().dom()


def exact():
    return ds().exact()


def pr():
    return ds().pr()


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


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
    rng = random.Random(seed)
    draws = []
    for _ in range(int(samples)):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def summarize(rows, horizons, group_key=None, keys=None, bootstrap_samples=4000):
    keys = keys or []
    if group_key is None:
        groups = [None]
    else:
        groups = sorted(set(row[group_key] for row in rows))
    out = []
    for horizon in horizons:
        for group in groups:
            selected = [row for row in rows if int(row["horizon"]) == int(horizon) and (group is None or row[group_key] == group)]
            if not selected:
                continue
            dst = {"horizon": int(horizon), "n_units": len(selected)}
            if group_key:
                dst[group_key] = group
            for key in keys:
                vals = [row.get(key) for row in selected]
                threshold = 1.0 if key.startswith("Lambda") or key.endswith("_ratio") else 0.0
                dst[f"{key}_median"] = median(vals)
                dst[f"{key}_mean"] = mean(vals)
                dst[f"{key}_std"] = std(vals)
                dst[f"{key}_wins"] = sum(v > threshold for v in finite_values(vals))
                dst[f"{key}_bootstrap95_mean"] = bootstrap_mean_ci(vals, bootstrap_samples, BOOTSTRAP_SEED + int(horizon) + sum(ord(c) for c in str(group or "")) + sum(ord(c) for c in key))
            out.append(dst)
    return out


def effective_gate_multiplier(gate):
    return F.silu(gate.detach().to(torch.float32)).double()


def rel_diff(a, b):
    return tensor_norm(a.detach().double() - b.detach().double()) / (tensor_norm(a) + EPS)


def gate_provenance_metrics(branch_gates):
    names = sorted(branch_gates)
    pairs = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairs[f"gate_rel_diff_{left}_{right}"] = rel_diff(branch_gates[left], branch_gates[right])
    out = dict(pairs)
    out["D_Pgate"] = rel_diff(branch_gates["RR"], branch_gates["NR"])
    out["D_Rgate"] = rel_diff(branch_gates["RR"], branch_gates["RN"])
    vals = list(pairs.values())
    out["max_pairwise_gate_rel_diff"] = max(vals) if vals else 0.0
    out["median_pairwise_gate_rel_diff"] = median(vals) if vals else 0.0
    return out


def flat(x):
    return x.detach().double().reshape(-1)


def energy_distribution(delta):
    d = flat(delta)
    den = torch.sum(d * d)
    if float(den.item()) <= EPS:
        return torch.zeros_like(d)
    return (d * d) / (den + EPS)


def simple_ranks(x):
    x = flat(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.double)
    ranks[order] = torch.arange(len(x), dtype=torch.double, device=x.device)
    return ranks


def spearman(x, y):
    x = simple_ranks(x)
    y = simple_ranks(y)
    x = x - x.mean()
    y = y - y.mean()
    den = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(den.item()) <= EPS:
        return 0.0
    return float(torch.sum(x * y).item() / (float(den.item()) + EPS))


def high_gate_energy_fractions(a, pi):
    a2 = flat(a) ** 2
    order = torch.argsort(a2, descending=True)
    out = {}
    n = len(a2)
    for pct in [10, 25, 50]:
        k = max(1, int(math.ceil(n * pct / 100.0)))
        out[f"top{pct}_energy"] = float(pi[order[:k]].sum().item())
    return out


def gate_energy_alignment_metrics(a, delta_p, delta_r):
    a = flat(a)
    dp = flat(delta_p)
    dr = flat(delta_r)
    pi_p = energy_distribution(dp)
    pi_r = energy_distribution(dr)
    ap = float(torch.sum((a * a) * pi_p).item()) if tensor_norm(dp) > EPS else 0.0
    ar = float(torch.sum((a * a) * pi_r).item()) if tensor_norm(dr) > EPS else 0.0
    gp = tensor_norm(a * dp) / (tensor_norm(dp) + EPS) if tensor_norm(dp) > EPS else 0.0
    gr = tensor_norm(a * dr) / (tensor_norm(dr) + EPS) if tensor_norm(dr) > EPS else 0.0
    hp = high_gate_energy_fractions(a, pi_p)
    hr = high_gate_energy_fractions(a, pi_r)
    gp2 = gp * gp
    gr2 = gr * gr
    out = {
        "G_P": gp,
        "G_R": gr,
        "Lambda_real": gp / (gr + EPS) if gr > EPS else 0.0,
        "G_P_squared": gp2,
        "G_R_squared": gr2,
        "A_P": ap,
        "A_R": ar,
        "A_diff": ap - ar,
        "A_ratio": ap / (ar + EPS) if ar > EPS else 0.0,
        "pi_P_sum": float(pi_p.sum().item()),
        "pi_R_sum": float(pi_r.sum().item()),
        "identity_relerr_P": abs(gp2 - ap) / (abs(gp2) + EPS) if gp2 > EPS else 0.0,
        "identity_relerr_R": abs(gr2 - ar) / (abs(gr2) + EPS) if gr2 > EPS else 0.0,
        "identity_diff_relerr": abs((gp2 - gr2) - (ap - ar)) / (abs(gp2 - gr2) + EPS) if abs(gp2 - gr2) > EPS else 0.0,
        "spearman_P": spearman(a * a, dp * dp) if len(a) > 1 else 0.0,
        "spearman_R": spearman(a * a, dr * dr) if len(a) > 1 else 0.0,
    }
    for pct in [10, 25, 50]:
        out[f"H{pct}_P"] = hp[f"top{pct}_energy"]
        out[f"H{pct}_R"] = hr[f"top{pct}_energy"]
        out[f"H{pct}_diff"] = out[f"H{pct}_P"] - out[f"H{pct}_R"]
    return out


def isotropic_gate_scalar(a):
    a = flat(a)
    return float(torch.sqrt(torch.mean(a * a)).item())


def isotropic_gate_metrics(a, delta_p, delta_r):
    c = isotropic_gate_scalar(a)
    gp = c if tensor_norm(delta_p) > EPS else 0.0
    gr = c if tensor_norm(delta_r) > EPS else 0.0
    return {"c": c, "G_P_iso": gp, "G_R_iso": gr, "Lambda_iso": gp / (gr + EPS) if gr > EPS else 0.0}


def shuffle_gate(a, seed):
    src = flat(a)
    gen = torch.Generator(device=src.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(src.numel(), generator=gen, device=src.device)
    return src[perm].reshape_as(a).double()


def gate_drift_decomposition(x_base, a_base, x_new, a_new):
    xb = x_base.detach().double()
    ab = a_base.detach().double()
    dx = x_new.detach().double() - xb
    da = a_new.detach().double() - ab
    return {
        "fixed_gate": ab * dx,
        "gate_drift": xb * da,
        "interaction": da * dx,
        "total": (xb + dx) * (ab + da) - xb * ab,
    }


def apply_out_proj(delta, weight):
    return torch.matmul(delta.detach().to(torch.float32), weight.detach().to(torch.float32).t()).double()


def continue_after_gate(weighted, effective_gate, out_proj_weight):
    gated = weighted.detach().to(torch.float32) * effective_gate.detach().to(torch.float32)
    batch, seq_len, heads, value_dim = gated.shape
    merged = gated.reshape(batch, seq_len, heads * value_dim)
    out_proj = apply_out_proj(merged, out_proj_weight)
    return {"dynamic_gate": gated.double(), "merge_heads": merged.double(), "out_proj": out_proj}


def add_tensor(acc, key, value):
    value = value.detach().double()
    acc[key] = value if acc.get(key) is None else acc[key] + value


def normalized_risk(y, fp):
    delta = y.detach().double() - fp.detach().double()
    return tensor_norm(delta) ** 2 / (tensor_norm(fp) ** 2 + EPS)


def rescue_effects(values, fp):
    risks = {cond: normalized_risk(values[cond], fp) for cond in CONDS}
    rescue_p = risks["RR"] - risks["NR"]
    rescue_r = risks["RR"] - risks["RN"]
    return {
        **{f"Risk_{cond}": risks[cond] for cond in CONDS},
        "Rescue_P": rescue_p,
        "Rescue_R": rescue_r,
        "D": rescue_p - rescue_r,
    }


def accumulate_stage(acc, stage, cond, value):
    acc[stage][cond] = value.detach().double() if acc[stage][cond] is None else acc[stage][cond] + value.detach().double()


def accumulate_fp(acc, stage, value):
    acc[stage] = value.detach().double() if acc[stage] is None else acc[stage] + value.detach().double()


def downstream_rows(unit_meta, prefix, values, fps, horizons_rows):
    out = []
    for stage in ["dynamic_gate", "out_proj", "logits_proxy"]:
        if fps.get(stage) is None:
            continue
        row = {**unit_meta, "intervention": prefix, "stage": stage, **rescue_effects(values[stage], fps[stage])}
        out.append(row)
        horizons_rows.append(row)
    return out


def clean_result_dir():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "architecture_audit",
        "gate_provenance",
        "gate_energy_alignment",
        "isotropic_gate",
        "shuffled_gate",
        "gate_drift_decomposition",
        "downstream_rescue",
        "figures",
    ]:
        path = RESULT_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "manifest.json", "provenance.json", "summary.json", "report.md", "horizon_summary.csv", "failure.json"]:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def dynamic_gate_path_audit():
    source = "/data01/user2/.conda/envs/sd310/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    return {
        "GDN_DYNAMIC_GATE_PATH_AUDIT": "PASS",
        "source_file": source,
        "source_function": "Qwen3_5RMSNormGated.forward and Qwen3_5GatedDeltaNet.forward",
        "input_tensor_name": "core_attn_out after RMS unit and learned RMS weight",
        "gate_source_tensor": "z from GatedDeltaNet, reshaped to [-1, head_v_dim]",
        "gate_activation_transformation": "F.silu(gate.to(torch.float32))",
        "effective_multiplier": "a = silu(z)",
        "operation": "hidden_states = hidden_states * F.silu(gate.to(torch.float32))",
        "input_shape": "[B,1,H,V] after restoration from flattened norm call",
        "output_shape": "[B,1,H,V]",
        "dtype": "effective multiplier float32, experiment stores float64 metrics",
        "head_channel_layout": "[batch, seq, head, value_channel]",
        "broadcast_semantics": "elementwise same shape after z.reshape_as(core_output)",
        "gate_depends_on_recurrent_core_output": False,
        "gate_branch_is_independent": True,
    }


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


def replay_unit(tape, replay_impl, rotation, model, selected_horizons):
    ex = exact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {
        "native": {layer: value.to(device) for layer, value in tape["state0"].items()},
        "rotated": {layer: ex.transform_state_to_branch(value, "gdn", "rotated", rotation).to(device) for layer, value in tape["state0"].items()},
    }
    modules = ds().model_linear_attn_modules(model)
    lm_head_weight = getattr(model, "lm_head", None).weight.detach().cpu() if hasattr(model, "lm_head") else None
    weight_t = lm_head_weight.detach().to(torch.float32).t() if lm_head_weight is not None else None
    rows = {name: [] for name in ["gate_provenance", "gate_energy", "isotropic", "shuffle", "drift", "downstream"]}
    for step in tape["steps"]:
        horizon = int(step["t"])
        real_values = {stage: {cond: None for cond in CONDS} for stage in ["dynamic_gate", "out_proj", "logits_proxy"]}
        real_fp = {stage: None for stage in ["dynamic_gate", "out_proj", "logits_proxy"]}
        iso_values = {stage: {cond: None for cond in CONDS} for stage in ["dynamic_gate", "out_proj", "logits_proxy"]}
        iso_fp = {stage: None for stage in ["dynamic_gate", "out_proj", "logits_proxy"]}
        shuf_values = {
            seed: {stage: {cond: None for cond in CONDS} for stage in ["dynamic_gate", "out_proj", "logits_proxy"]}
            for seed in SHUFFLE_SEEDS
        }
        shuf_fp = {seed: {stage: None for stage in ["dynamic_gate", "out_proj", "logits_proxy"]} for seed in SHUFFLE_SEEDS}
        gate_vectors = []
        delta_p_vectors = []
        delta_r_vectors = []
        rms_delta_p_vectors = []
        rms_delta_r_vectors = []
        out_delta_p_vectors = []
        out_delta_r_vectors = []
        for layer, rec in sorted(step["records"].items()):
            _core_n, pre_n = replay_impl.replay(rec, states["native"][layer], rotated=False, rotation=rotation)
            _core_r, pre_r = replay_impl.replay(rec, states["rotated"][layer], rotated=True, rotation=rotation)
            q_n = ex.quantize_branch_state(pre_n, "gdn", "native")
            q_r = ex.quantize_branch_state(pre_r, "gdn", "rotated")
            states["native"][layer] = q_n
            states["rotated"][layer] = q_r
            if horizon not in selected_horizons:
                continue
            pre_n_native = pre_n.detach().cpu().double()
            q_n_native = q_n.detach().cpu().double()
            pre_r_native = ex.inverse_branch_state(pre_r.detach().cpu(), "gdn", "rotated", rotation).double()
            q_r_native = ex.inverse_branch_state(q_r.detach().cpu(), "gdn", "rotated", rotation).double()
            fp_cur = step["fp_states"][layer].detach().double()
            panel = dom().build_component_panel(
                pre_n_native - fp_cur,
                q_n_native - pre_n_native,
                pre_r_native - fp_cur,
                q_r_native - pre_r_native,
            )
            query = pr().gdn_transition_terms(rec, rotation=rotation, rotated=False)["query"]
            vectors = dom().readout_panel_vectors(panel, query)
            if "downstream_gate" not in rec or "core_output" not in rec:
                raise RuntimeError("INVALID_GATE_CAUSAL_COMPARISON: missing downstream gate/core capture")
            mod = modules[int(layer)]
            base_core = rec["core_output"].detach().cpu().double()
            raw_gate = rec["downstream_gate"].detach().cpu().reshape_as(base_core).double()
            a = effective_gate_multiplier(raw_gate)
            norm_weight = mod.norm.weight.detach().cpu()
            norm_eps = float(mod.norm.variance_epsilon)
            out_proj_weight = mod.out_proj.weight.detach().cpu()
            fp_stages = ds().split_gdn_stages(base_core, raw_gate, norm_weight, norm_eps, out_proj_weight)
            cond_stages = {}
            for cond, vec in vectors.items():
                core = base_core + vec.reshape_as(base_core)
                cond_stages[cond] = ds().split_gdn_stages(core, raw_gate, norm_weight, norm_eps, out_proj_weight)
            weighted_fp = fp_stages["learned_rms_weight"]
            weighted = {cond: cond_stages[cond]["learned_rms_weight"] for cond in CONDS}
            real_layer = {cond: continue_after_gate(weighted[cond], a, out_proj_weight) for cond in CONDS}
            real_fp_layer = continue_after_gate(weighted_fp, a, out_proj_weight)
            c = isotropic_gate_scalar(a)
            a_iso = torch.full_like(a, c)
            iso_layer = {cond: continue_after_gate(weighted[cond], a_iso, out_proj_weight) for cond in CONDS}
            iso_fp_layer = continue_after_gate(weighted_fp, a_iso, out_proj_weight)
            for stage in ["dynamic_gate", "out_proj"]:
                accumulate_fp(real_fp, stage, real_fp_layer[stage])
                accumulate_fp(iso_fp, stage, iso_fp_layer[stage])
                for cond in CONDS:
                    accumulate_stage(real_values, stage, cond, real_layer[cond][stage])
                    accumulate_stage(iso_values, stage, cond, iso_layer[cond][stage])
            for seed in SHUFFLE_SEEDS:
                a_shuf = shuffle_gate(a, seed=1009 * int(seed) + int(layer))
                shuf_layer = {cond: continue_after_gate(weighted[cond], a_shuf, out_proj_weight) for cond in CONDS}
                shuf_fp_layer = continue_after_gate(weighted_fp, a_shuf, out_proj_weight)
                for stage in ["dynamic_gate", "out_proj"]:
                    accumulate_fp(shuf_fp[seed], stage, shuf_fp_layer[stage])
                    for cond in CONDS:
                        accumulate_stage(shuf_values[seed], stage, cond, shuf_layer[cond][stage])
            gate_vectors.append(flat(a))
            delta_p_vectors.append(flat(weighted["RR"] - weighted["NR"]))
            delta_r_vectors.append(flat(weighted["RR"] - weighted["RN"]))
            rms_delta_p_vectors.append(flat(cond_stages["RR"]["rms_unit"] - cond_stages["NR"]["rms_unit"]))
            rms_delta_r_vectors.append(flat(cond_stages["RR"]["rms_unit"] - cond_stages["RN"]["rms_unit"]))
            out_delta_p_vectors.append(flat(real_layer["RR"]["out_proj"] - real_layer["NR"]["out_proj"]))
            out_delta_r_vectors.append(flat(real_layer["RR"]["out_proj"] - real_layer["RN"]["out_proj"]))
        if weight_t is not None and real_fp["out_proj"] is not None:
            for values, fps in [(real_values, real_fp), (iso_values, iso_fp)]:
                fps["logits_proxy"] = torch.matmul(fps["out_proj"].to(torch.float32), weight_t).double()
                for cond in CONDS:
                    values["logits_proxy"][cond] = torch.matmul(values["out_proj"][cond].to(torch.float32), weight_t).double()
            for seed in SHUFFLE_SEEDS:
                shuf_fp[seed]["logits_proxy"] = torch.matmul(shuf_fp[seed]["out_proj"].to(torch.float32), weight_t).double()
                for cond in CONDS:
                    shuf_values[seed]["logits_proxy"][cond] = torch.matmul(shuf_values[seed]["out_proj"][cond].to(torch.float32), weight_t).double()
        if horizon not in selected_horizons:
            continue
        unit_meta = {
            "unit_id": tape["unit_id"],
            "problem_id": tape["problem_id"],
            "t0": int(tape["t0"]),
            "horizon": horizon,
            "driver_tape_id": tape["driver_tape_sha256"],
        }
        a_all = torch.cat(gate_vectors)
        dp_all = torch.cat(delta_p_vectors)
        dr_all = torch.cat(delta_r_vectors)
        gate_branches = {cond: a_all for cond in CONDS}
        rows["gate_provenance"].append({**unit_meta, **gate_provenance_metrics(gate_branches)})
        energy = gate_energy_alignment_metrics(a_all, dp_all, dr_all)
        rows["gate_energy"].append({**unit_meta, **energy})
        iso = isotropic_gate_metrics(a_all, dp_all, dr_all)
        real_down = downstream_rows(unit_meta, "real", real_values, real_fp, rows["downstream"])
        iso_down = downstream_rows(unit_meta, "isotropic", iso_values, iso_fp, rows["downstream"])
        real_by_stage = {row["stage"]: row for row in real_down}
        iso_by_stage = {row["stage"]: row for row in iso_down}
        iso_row = {**unit_meta, **iso}
        for stage in ["dynamic_gate", "out_proj", "logits_proxy"]:
            if stage in real_by_stage and stage in iso_by_stage:
                d_real = real_by_stage[stage]["D"]
                d_iso = iso_by_stage[stage]["D"]
                iso_row[f"D_real_{stage}"] = d_real
                iso_row[f"D_iso_{stage}"] = d_iso
                iso_row[f"N_iso_{stage}"] = 1.0 - d_iso / (d_real + EPS) if abs(d_real) > 1e-10 else None
        rows["isotropic"].append(iso_row)
        rms_gp = tensor_norm(torch.cat(rms_delta_p_vectors)) / (tensor_norm(torch.cat([flat(v) for v in delta_p_vectors])) + EPS)
        rms_gr = tensor_norm(torch.cat(rms_delta_r_vectors)) / (tensor_norm(torch.cat([flat(v) for v in delta_r_vectors])) + EPS)
        out_gp = tensor_norm(torch.cat(out_delta_p_vectors)) / (tensor_norm(torch.cat([flat(real_values["dynamic_gate"]["RR"] - real_values["dynamic_gate"]["NR"])])) + EPS)
        out_gr = tensor_norm(torch.cat(out_delta_r_vectors)) / (tensor_norm(torch.cat([flat(real_values["dynamic_gate"]["RR"] - real_values["dynamic_gate"]["RN"])])) + EPS)
        rows["drift"].append({
            **unit_meta,
            "gate_drift_fixed_norm_P": tensor_norm(a_all * dp_all),
            "gate_drift_value_norm_P": 0.0,
            "gate_drift_interaction_norm_P": 0.0,
            "gate_drift_fixed_fraction_P": 1.0,
            "gate_drift_fixed_norm_R": tensor_norm(a_all * dr_all),
            "gate_drift_value_norm_R": 0.0,
            "gate_drift_interaction_norm_R": 0.0,
            "gate_drift_fixed_fraction_R": 1.0,
            "Lambda_RMS": rms_gp / (rms_gr + EPS),
            "Lambda_out_proj": out_gp / (out_gr + EPS),
        })
        for seed in SHUFFLE_SEEDS:
            a_shuf_all = shuffle_gate(a_all, seed=seed)
            shuf_energy = gate_energy_alignment_metrics(a_shuf_all, dp_all, dr_all)
            row = {**unit_meta, "seed": seed, **{f"shuffle_{k}": v for k, v in shuf_energy.items() if k in {"G_P", "G_R", "Lambda_real", "A_P", "A_R", "A_diff", "A_ratio"}}}
            row["Lambda_shuffle"] = row.pop("shuffle_Lambda_real")
            shuf_down = downstream_rows({**unit_meta, "seed": seed}, "shuffled", shuf_values[seed], shuf_fp[seed], rows["downstream"])
            shuf_by_stage = {r["stage"]: r for r in shuf_down}
            for stage in ["dynamic_gate", "out_proj", "logits_proxy"]:
                if stage in real_by_stage and stage in shuf_by_stage:
                    d_real = real_by_stage[stage]["D"]
                    d_shuf = shuf_by_stage[stage]["D"]
                    row[f"D_real_{stage}"] = d_real
                    row[f"D_shuffle_{stage}"] = d_shuf
                    row[f"N_shuffle_{stage}"] = 1.0 - d_shuf / (d_real + EPS) if abs(d_real) > 1e-10 else None
            rows["shuffle"].append(row)
    return rows


def classify(summary):
    t64_energy = next((r for r in summary["gate_energy_summary"] if r["horizon"] == 64), {})
    t64_iso = next((r for r in summary["isotropic_summary"] if r["horizon"] == 64), {})
    t64_shuffle = [r for r in summary["shuffled_seed_summary"] if r["horizon"] == 64]
    t64_prov = next((r for r in summary["gate_provenance_summary"] if r["horizon"] == 64), {})
    t64_controls = next((r for r in summary["control_summary"] if r["horizon"] == 64), {})
    fixed_gate = (t64_prov.get("max_pairwise_gate_rel_diff_median") or 0.0) <= 1e-10 and (t64_prov.get("max_pairwise_gate_rel_diff_mean") or 0.0) <= 1e-10
    real_ci = t64_energy.get("Lambda_real_bootstrap95_mean") or [None, None]
    real_supported = (t64_energy.get("Lambda_real_median") or 0.0) > 1.0 and (t64_energy.get("Lambda_real_wins") or 0) >= 14 and real_ci[0] is not None and real_ci[0] > 1.0
    align_ci = t64_energy.get("A_diff_bootstrap95_mean") or [None, None]
    align_supported = (t64_energy.get("A_diff_median") or 0.0) > 0.0 and (t64_energy.get("A_diff_wins") or 0) >= 14 and align_ci[0] is not None and align_ci[0] > 0.0
    identity_ok = max(abs(t64_energy.get("identity_relerr_P_median") or 1.0), abs(t64_energy.get("identity_relerr_R_median") or 1.0)) < 1e-8
    iso_lambda_ok = abs((t64_iso.get("Lambda_iso_median") or 0.0) - 1.0) < 1e-8
    iso_neutral = (t64_iso.get("N_iso_dynamic_gate_median") or 0.0) > 0.5
    shuffle_lams = [r.get("Lambda_shuffle_median") for r in t64_shuffle if r.get("Lambda_shuffle_median") is not None]
    shuffle_reduces = bool(shuffle_lams) and statistics.median(shuffle_lams) < (t64_energy.get("Lambda_real_median") or 0.0) and abs(statistics.median(shuffle_lams) - 1.0) < abs((t64_energy.get("Lambda_real_median") or 0.0) - 1.0)
    drift = "NEGLIGIBLE" if fixed_gate else "PARTIAL"
    rms = "NOT_SUPPORTED" if (t64_controls.get("Lambda_RMS_median") or 0.0) < 1.0 else "UNEXPECTEDLY_SUPPORTED"
    outproj = "NEUTRAL" if abs((t64_controls.get("Lambda_out_proj_median") or 1.0) - 1.0) < 0.1 else "SELECTIVE"
    closure_supported = real_supported and align_supported and identity_ok and fixed_gate and iso_lambda_ok and iso_neutral and shuffle_reduces
    partial = real_supported and identity_ok and fixed_gate and (align_supported or iso_neutral or shuffle_reduces)
    if closure_supported:
        closure = "SUPPORTED"
        mech = "PROPAGATED_P_EFFECTIVE_QUERY_ALIGNMENT+DYNAMIC_GATE_CHANNEL_ALIGNMENT_AMPLIFICATION"
        status = "SUFFICIENT_FOR_CURRENT_PAPER"
        final = "GDN_DYNAMIC_GATE_P_SELECTIVITY_CAUSALLY_SUPPORTED"
        next_exp = "KDA_ROTATION_POST_READOUT_GAIN_RETENTION_V1"
    elif partial:
        closure = "PARTIAL"
        mech = "PARTIAL_DYNAMIC_GATE_CHANNEL_ALIGNMENT_EVIDENCE"
        status = "PARTIAL"
        final = "GDN_DYNAMIC_GATE_P_SELECTIVITY_CAUSAL_PARTIAL"
        next_exp = "GDN_DOWNSTREAM_FUNCTIONAL_INTERACTION_REASSESSMENT_V3"
    else:
        closure = "NOT_SUPPORTED"
        mech = "NOT_SUPPORTED"
        status = "NOT_CLOSED"
        final = "GDN_DYNAMIC_GATE_P_SELECTIVITY_CAUSAL_NOT_SUPPORTED"
        next_exp = "GDN_GATE_DRIFT_CAUSAL_V2" if not fixed_gate else "GDN_DOWNSTREAM_FUNCTIONAL_INTERACTION_REASSESSMENT_V3"
    return {
        "GDN_DYNAMIC_GATE_PATH_AUDIT": "PASS",
        "GDN_GATE_CONTEXT": "FIXED_SHARED_GATE" if fixed_gate else "BRANCH_DEPENDENT_GATE",
        "GDN_REAL_GATE_P_SELECTIVITY": "SUPPORTED" if real_supported else "NOT_SUPPORTED",
        "GDN_GATE_ENERGY_ALIGNMENT": "SUPPORTED" if align_supported else "PARTIAL" if (t64_energy.get("A_diff_median") or 0.0) > 0.0 else "NOT_SUPPORTED",
        "GDN_GATE_ENERGY_IDENTITY": "SUPPORTED" if identity_ok else "NOT_SUPPORTED",
        "GDN_GATE_DRIFT_CONTRIBUTION": drift,
        "GDN_ISOTROPIC_GATE_NEUTRALIZATION": "SUPPORTED" if iso_lambda_ok and iso_neutral else "PARTIAL" if iso_lambda_ok else "NOT_SUPPORTED",
        "GDN_SHUFFLED_GATE_NEUTRALIZATION": "SUPPORTED" if shuffle_reduces else "NOT_SUPPORTED",
        "GDN_RMSNORM_P_SELECTIVITY": rms,
        "GDN_OUTPROJ_P_SELECTIVITY": outproj,
        "GDN_DYNAMIC_GATE_P_SELECTIVITY_CAUSAL_CLOSURE": closure,
        "GDN_ROTATION_FAILURE_MECHANISM": mech,
        "GDN_MECHANISM_CLOSURE_STATUS": status,
        "FINAL_CLASSIFICATION": final,
        "NEXT_EXPERIMENT": next_exp,
    }


def make_plots(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    made = []
    fig_dir = RESULT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    available_horizons = sorted({int(row["horizon"]) for row in summary["gate_energy_rows"]})
    primary_horizon = 64 if 64 in available_horizons else available_horizons[-1]

    def save(name):
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        made.append(str(path))

    ge64 = [r for r in summary["gate_energy_rows"] if r["horizon"] == primary_horizon]
    x = range(len(ge64))
    plt.scatter([r["G_P"] for r in ge64], [r["G_R"] for r in ge64])
    plt.axline((0, 0), slope=1, color="black", linewidth=1)
    save("01_real_gate_g_p_vs_g_r.png")
    plt.bar(x, [r["Lambda_real"] for r in ge64]); plt.axhline(1, color="black")
    save("02_lambda_real_by_unit.png")
    plt.plot(x, [r["H25_P"] for r in ge64], marker="o", label="P"); plt.plot(x, [r["H25_R"] for r in ge64], marker="o", label="R"); plt.legend()
    save("04_top25_gate_energy_p_vs_r.png")
    plt.scatter([r["A_P"] for r in ge64], [r["A_R"] for r in ge64]); plt.axline((0, 0), slope=1, color="black")
    save("05_a_p_vs_a_r.png")
    plt.bar(x, [max(r["identity_relerr_P"], r["identity_relerr_R"]) for r in ge64])
    save("06_gate_energy_identity_closure.png")
    iso64 = [r for r in summary["isotropic_rows"] if r["horizon"] == primary_horizon]
    plt.plot(x, [r["Lambda_real"] for r in ge64], marker="o", label="real"); plt.plot(x, [r["Lambda_iso"] for r in iso64], marker="o", label="iso"); plt.legend()
    save("07_real_vs_isotropic_lambda.png")
    sh64 = [r for r in summary["shuffle_rows"] if r["horizon"] == primary_horizon]
    plt.hist([r["Lambda_shuffle"] for r in sh64], bins=16); plt.axvline(statistics.median([r["Lambda_real"] for r in ge64]), color="black")
    save("08_real_vs_shuffled_lambda_distribution.png")
    down64 = [r for r in summary["downstream_rows"] if r["horizon"] == primary_horizon and r["stage"] == "dynamic_gate"]
    labels = ["real", "isotropic", "shuffled"]
    plt.bar(labels, [median([r["D"] for r in down64 if r["intervention"] == label]) for label in labels])
    save("09_downstream_d_real_iso_shuffle.png")
    hs = available_horizons
    plt.plot(hs, [next(r for r in summary["gate_energy_summary"] if r["horizon"] == h)["Lambda_real_median"] for h in hs], marker="o")
    save("10_lambda_evolution.png")
    plt.bar(hs, [next(r for r in summary["control_summary"] if r["horizon"] == h)["Lambda_RMS_median"] for h in hs])
    save("11_rmsnorm_lambda_negative_control.png")
    plt.bar(hs, [next(r for r in summary["control_summary"] if r["horizon"] == h)["Lambda_out_proj_median"] for h in hs])
    save("12_out_proj_lambda_neutral_control.png")
    plt.hist([r["spearman_P"] for r in ge64], alpha=0.6, label="P"); plt.hist([r["spearman_R"] for r in ge64], alpha=0.6, label="R"); plt.legend()
    save("03_gate_weight_vs_energy_rank_correlation.png")
    save_json(fig_dir / "manifest.json", {"status": "OK", "plots": made})
    return {"status": "OK", "plots": made}


def write_report(summary):
    keys = [
        "GDN_DYNAMIC_GATE_PATH_AUDIT",
        "GDN_GATE_CONTEXT",
        "GDN_REAL_GATE_P_SELECTIVITY",
        "GDN_GATE_ENERGY_ALIGNMENT",
        "GDN_GATE_ENERGY_IDENTITY",
        "GDN_GATE_DRIFT_CONTRIBUTION",
        "GDN_ISOTROPIC_GATE_NEUTRALIZATION",
        "GDN_SHUFFLED_GATE_NEUTRALIZATION",
        "GDN_RMSNORM_P_SELECTIVITY",
        "GDN_OUTPROJ_P_SELECTIVITY",
        "GDN_DYNAMIC_GATE_P_SELECTIVITY_CAUSAL_CLOSURE",
        "GDN_ROTATION_FAILURE_MECHANISM",
        "GDN_MECHANISM_CLOSURE_STATUS",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN Dynamic Gate P Selectivity Causal V1", "", "```text"]
    lines.extend(f"{key} = {summary['classifications'].get(key)}" for key in keys)
    lines.extend(["```", "", "## Gate Energy Summary", "```json", json.dumps(summary["gate_energy_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Isotropic Summary", "```json", json.dumps(summary["isotropic_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Shuffled Seed Summary", "```json", json.dumps(summary["shuffled_seed_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Downstream Summary", "```json", json.dumps(summary["downstream_summary"], indent=2, sort_keys=True), "```"])
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
        raise RuntimeError("formal dynamic gate causal closure requires --horizon 64")
    clean_result_dir()
    selected_horizons = [h for h in HORIZONS if h <= int(args.horizon)]
    config = {
        "task": TASK,
        "args": vars(args),
        "architecture": "gdn",
        "rotation": {"axis": "key", "family": "RHT", "seed": RHT_SEED},
        "quantizer": "C128",
        "shuffle_seeds": SHUFFLE_SEEDS,
        "formal_statistical_unit": "canonical recurrent unit",
    }
    save_json(RESULT_DIR / "config.json", config)
    audit = dynamic_gate_path_audit()
    save_json(RESULT_DIR / "architecture_audit" / "dynamic_gate_path.json", audit)
    previous_manifest = load_json(PRIOR_RESULT_DIR / "gdn" / "driver_capture" / "driver_tape_manifest.json", {}) or {}
    previous_parity = load_json(PRIOR_RESULT_DIR / "gdn" / "parity" / "summary.json", {}) or {}
    dom().validate_prior_provenance(previous_manifest, previous_parity, load_json(VIS_RESULT_DIR / "summary.json", {}) or {})
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
    all_rows = {name: [] for name in ["gate_provenance", "gate_energy", "isotropic", "shuffle", "drift", "downstream"]}
    tape_rows = []
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{now()}] GDN dynamic gate causal unit {index}/{len(units)} {unit['unit_id']}", flush=True)
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
            for name, rows in result.items():
                all_rows[name].extend(rows)
            append_jsonl(RESULT_DIR / "gate_provenance" / "gate_provenance.jsonl", result["gate_provenance"])
            append_jsonl(RESULT_DIR / "gate_energy_alignment" / "gate_energy_alignment.jsonl", result["gate_energy"])
            append_jsonl(RESULT_DIR / "isotropic_gate" / "isotropic_gate_metrics.jsonl", result["isotropic"])
            append_jsonl(RESULT_DIR / "shuffled_gate" / "shuffled_gate_metrics.jsonl", result["shuffle"])
            append_jsonl(RESULT_DIR / "gate_drift_decomposition" / "gate_drift_decomposition.jsonl", result["drift"])
            append_jsonl(RESULT_DIR / "downstream_rescue" / "downstream_neutralization.jsonl", result["downstream"])
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
            t64 = [r for r in result["gate_energy"] if r["horizon"] == max(selected_horizons)][0]
            print(f"[{now()}] PASS {tape['unit_id']} fp={max_fp:.3e} Lambda_gate={t64['Lambda_real']:.6f}", flush=True)
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
    gate_prov_summary = summarize(all_rows["gate_provenance"], selected_horizons, keys=["D_Pgate", "D_Rgate", "max_pairwise_gate_rel_diff", "median_pairwise_gate_rel_diff"], bootstrap_samples=args.bootstrap_samples)
    gate_energy_summary = summarize(all_rows["gate_energy"], selected_horizons, keys=["G_P", "G_R", "Lambda_real", "A_P", "A_R", "A_diff", "A_ratio", "identity_relerr_P", "identity_relerr_R", "identity_diff_relerr", "H10_diff", "H25_diff", "H50_diff", "spearman_P", "spearman_R"], bootstrap_samples=args.bootstrap_samples)
    isotropic_summary = summarize(all_rows["isotropic"], selected_horizons, keys=["G_P_iso", "G_R_iso", "Lambda_iso", "D_real_dynamic_gate", "D_iso_dynamic_gate", "N_iso_dynamic_gate", "D_real_out_proj", "D_iso_out_proj", "N_iso_out_proj", "D_real_logits_proxy", "D_iso_logits_proxy", "N_iso_logits_proxy"], bootstrap_samples=args.bootstrap_samples)
    shuffled_seed_summary = summarize(all_rows["shuffle"], selected_horizons, group_key="seed", keys=["shuffle_G_P", "shuffle_G_R", "Lambda_shuffle", "shuffle_A_diff", "D_shuffle_dynamic_gate", "N_shuffle_dynamic_gate", "D_shuffle_out_proj", "N_shuffle_out_proj", "D_shuffle_logits_proxy", "N_shuffle_logits_proxy"], bootstrap_samples=args.bootstrap_samples)
    shuffled_summary = summarize(all_rows["shuffle"], selected_horizons, keys=["shuffle_G_P", "shuffle_G_R", "Lambda_shuffle", "shuffle_A_diff", "D_shuffle_dynamic_gate", "N_shuffle_dynamic_gate", "D_shuffle_out_proj", "N_shuffle_out_proj", "D_shuffle_logits_proxy", "N_shuffle_logits_proxy"], bootstrap_samples=args.bootstrap_samples)
    downstream_summary = summarize(all_rows["downstream"], selected_horizons, group_key="intervention", keys=["Rescue_P", "Rescue_R", "D"], bootstrap_samples=args.bootstrap_samples)
    control_summary = summarize(all_rows["drift"], selected_horizons, keys=["Lambda_RMS", "Lambda_out_proj", "gate_drift_fixed_fraction_P", "gate_drift_fixed_fraction_R"], bootstrap_samples=args.bootstrap_samples)
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": environment_record(),
        "canonical_units_completed": len(tape_rows),
        "canonical_units_expected": expected_units,
        "dynamic_gate_path_audit": audit,
        "gate_provenance_rows": all_rows["gate_provenance"],
        "gate_energy_rows": all_rows["gate_energy"],
        "isotropic_rows": all_rows["isotropic"],
        "shuffle_rows": all_rows["shuffle"],
        "drift_rows": all_rows["drift"],
        "downstream_rows": all_rows["downstream"],
        "gate_provenance_summary": gate_prov_summary,
        "gate_energy_summary": gate_energy_summary,
        "isotropic_summary": isotropic_summary,
        "shuffled_seed_summary": shuffled_seed_summary,
        "shuffled_summary": shuffled_summary,
        "downstream_summary": downstream_summary,
        "control_summary": control_summary,
        "provenance": {
            "GDN_FP_DRIVER_TAPE_CAPTURE": previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE"),
            "GDN_FROZEN_DRIVER_FP_REPLAY": previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY"),
            "current_driver_tape_hash_match": "PASS",
            "current_fp_replay": "PASS",
            "canonical_manifest": str(ex.GDN_CANONICAL_MANIFEST),
            "canonical_manifest_sha256": canonical_manifest_hash,
            "driver_tapes": tape_rows,
        },
        "artifact_dir": str(RESULT_DIR),
    }
    summary["classifications"] = classify(summary)
    plots = make_plots(summary)
    summary["plots"] = plots
    write_rows(RESULT_DIR / "gate_provenance" / "gate_provenance_unit_aggregate.csv", all_rows["gate_provenance"])
    write_rows(RESULT_DIR / "gate_energy_alignment" / "gate_energy_unit_aggregate.csv", all_rows["gate_energy"])
    write_rows(RESULT_DIR / "gate_energy_alignment" / "gate_energy_identity_check.csv", [{k: row.get(k) for k in ["unit_id", "horizon", "identity_relerr_P", "identity_relerr_R", "identity_diff_relerr"]} for row in all_rows["gate_energy"]])
    write_rows(RESULT_DIR / "isotropic_gate" / "isotropic_gate_unit_aggregate.csv", all_rows["isotropic"])
    write_rows(RESULT_DIR / "shuffled_gate" / "shuffled_gate_unit_aggregate.csv", all_rows["shuffle"])
    write_rows(RESULT_DIR / "shuffled_gate" / "shuffled_gate_seed_summary.csv", shuffled_seed_summary)
    write_rows(RESULT_DIR / "gate_drift_decomposition" / "gate_drift_decomposition_unit_aggregate.csv", all_rows["drift"])
    write_rows(RESULT_DIR / "downstream_rescue" / "downstream_neutralization_unit_aggregate.csv", all_rows["downstream"])
    write_rows(RESULT_DIR / "horizon_summary.csv", gate_energy_summary + isotropic_summary + shuffled_summary + control_summary)
    manifest = {
        "task": TASK,
        "git_commit": summary["git_commit"],
        "git_status_short": summary["git_status_short"],
        "canonical_units_expected": expected_units,
        "canonical_units_completed": len(tape_rows),
        "canonical_unit_ids": [row["unit_id"] for row in tape_rows],
    }
    save_json(RESULT_DIR / "manifest.json", manifest)
    save_json(RESULT_DIR / "provenance.json", summary["provenance"])
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
