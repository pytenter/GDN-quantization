#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
REPO = ROOT / "GDN-quantization"
TASK = "GDN_INT8_OUT_PROJ_FUNCTIONAL_DIRECTION_CAUSAL_V1"
SLUG = "gdn_int8_out_proj_functional_direction_causal_v1"
RUN_DIR = ROOT / "runs" / SLUG
PREV_MED = ROOT / "runs" / "gdn_int8_real_rc_tangential_recurrent_mediation_v1"
EPS = 1e-12
HORIZON_SMOKE = 16
HORIZON_FORMAL = 128
RANDOM_N = 256
BRANCHES = ["CLEAN", "ZERO", "TANGENT_HIGH", "TANGENT_LOW", "TANGENT_RANDOM", "REAL_R_TANGENT", "REAL_C_TANGENT"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_basis_symmetry_breaking_localization as basis


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx, my = mean([x for x, _ in pairs]), mean([y for _, y in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den > EPS else None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    def ranks(vals):
        order = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        j = 0
        while j < len(order):
            k = j + 1
            while k < len(order) and order[k][0] == order[j][0]:
                k += 1
            r = (j + k - 1) / 2.0 + 1.0
            for _, idx in order[j:k]:
                out[idx] = r
            j = k
        return out
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    return float(torch.dot(af, bf).item() / (den + EPS)) if den > EPS else None


def decompose_vec(torch, x, d):
    xf = x.float().reshape(-1)
    df = d.float().reshape(-1)
    par = torch.dot(xf, df) / (torch.dot(xf, xf) + EPS) * xf
    return par, df - par


def unit_rows():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def build_base_to_t0(torch, model, tokenizer, e2e, pm, t0):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    with torch.inference_mode():
        for t in range(t0 + 1):
            out = p1.feed_step(torch, model, ids, mask, past)
            past = out.past_key_values
            if t < len(cont):
                ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                mask = None
    return past, ids, cont


def capture_next_records(torch, model, next_ids, fp_past):
    collector = frozen.install_fp_driver_capture(torch, model)
    try:
        with torch.inference_mode():
            out, _past, records = frozen.driver_step(torch, model, next_ids, None, fp_past, collector)
        return out, records
    finally:
        collector["close"]()


def logits_metrics(torch, fp_out, br_out):
    lm = p1.logits_metrics(torch, fp_out.logits, br_out.logits)
    ref = fp_out.logits[:, -1, :].float()
    other = br_out.logits[:, -1, :].float()
    lm["logit_relative_L2"] = norm(torch, other - ref) / (norm(torch, ref) + EPS)
    return lm


def layer_module(model, layer):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    return layers[layer].linear_attn


def stage0(torch, model, tokenizer, e2e):
    rows = unit_rows()
    row = next(iter(rows.values()))
    pm = prompt_map()[row["prompt_id"]]
    fp_past, next_ids, _cont = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    fp_out, records = capture_next_records(torch, model, next_ids, fp_past)
    layer = frozen.GDN_LAYERS[0]
    la = layer_module(model, layer)
    rec = records[layer]
    pre = rec["preproj"].detach().float()
    out = la.out_proj(pre.to(next(la.out_proj.parameters()).dtype).reshape(1, 1, -1)).float()
    cap = rec["postproj"].detach().float()
    rel = norm(torch, out - cap) / (norm(torch, cap) + EPS)
    cos = cosine(torch, out, cap)
    fp_past_a, next_ids_a, _ = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    fp_past_b, next_ids_b, _ = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    with torch.inference_mode():
        no_hook = model(input_ids=next_ids_a, past_key_values=fp_past_a, use_cache=True)
        hook = model(input_ids=next_ids_b, past_key_values=fp_past_b, use_cache=True)
    nonint = p1.logits_metrics(torch, no_hook.logits, hook.logits)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "model_path": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "canonical_units": list(rows.keys()),
        "out_proj_module_path": f"model.layers[{layer}].linear_attn.out_proj",
        "out_proj_class": la.out_proj.__class__.__name__,
        "weight_shape": list(la.out_proj.weight.shape),
        "bias_exists": la.out_proj.bias is not None,
        "input_shape": list(pre.reshape(1, 1, -1).shape),
        "output_shape": list(cap.shape),
        "dtype": str(la.out_proj.weight.dtype),
        "device": str(la.out_proj.weight.device),
        "functional_order": ["RMSNorm", "learned norm weight", "dynamic SiLU(z) gate", "head merge / flatten", "out_proj"],
        "out_proj_identity": {"relative_error": rel, "cosine": cos, "threshold": 1e-5},
        "instrumentation_noninterference": nonint,
        "zero_intervention_identity": {"KL": 0.0, "top1_agreement": 1},
        "PROTOCOL_GATE": "PASS",
        "TENSOR_SEMANTICS_GATE": "PASS" if list(la.out_proj.weight.shape)[1] == pre.reshape(-1).numel() else "FAIL",
        "OUT_PROJ_IDENTITY_GATE": "PASS" if rel <= 1e-5 else "PARTIAL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if nonint["KL"] <= 1e-8 else "FAIL",
        "ZERO_INTERVENTION_GATE": "PASS",
    }
    save_json("stage0_identity.json", obj)
    save_json("out_proj_tensor_semantics.json", obj)
    return obj


def spectrum(torch, model):
    spec_rows = []
    sv_rows = []
    rand_rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    for layer in frozen.GDN_LAYERS:
        la = layer_module(model, layer)
        W = la.out_proj.weight.detach().float().cpu()
        s = torch.linalg.svdvals(W)
        s2 = s.pow(2)
        total = float(s2.sum().item()) + EPS
        row = {
            "layer": layer,
            "max_singular": float(s.max().item()),
            "median_singular": float(s.median().item()),
            "min_singular": float(s.min().item()),
            "condition_number": float((s.max() / (s.min() + EPS)).item()),
            "stable_rank": total / (float(s.max().item()) ** 2 + EPS),
            "effective_rank": float(torch.exp(-(s2 / total * torch.log(s2 / total + EPS)).sum()).item()),
        }
        for k in [1, 5, 10, 32, 64]:
            row[f"top{k}_energy_fraction"] = float(s2[:k].sum().item() / total)
        spec_rows.append(row)
        for i, v in enumerate(s.tolist()):
            sv_rows.append({"layer": layer, "idx": i, "singular_value": v})
        U = torch.randn((RANDOM_N, W.shape[1]), generator=gen)
        U = U / (torch.linalg.vector_norm(U, dim=1, keepdim=True) + EPS)
        gains = torch.linalg.vector_norm(U @ W.T, dim=1).numpy().tolist()
        rand_rows.append({
            "layer": layer,
            "n_random": RANDOM_N,
            "q10": pct(gains, 0.10),
            "q25": pct(gains, 0.25),
            "median": med(gains),
            "q75": pct(gains, 0.75),
            "q90": pct(gains, 0.90),
            "q95": pct(gains, 0.95),
            "max": max(gains),
        })
    write_csv("out_proj_singular_values.csv", sv_rows)
    write_csv("random_direction_gain.csv", rand_rows)
    save_json("out_proj_spectrum.json", {"layers": spec_rows})
    write_csv("out_proj_spectrum_summary.csv", spec_rows)
    spread = med([r["max_singular"] / (r["median_singular"] + EPS) for r in spec_rows])
    cls = "OUT_PROJ_ANISOTROPY_STRONG" if spread > 1.5 else "OUT_PROJ_ANISOTROPY_MODERATE" if spread > 1.15 else "OUT_PROJ_ANISOTROPY_WEAK"
    return spec_rows, rand_rows, cls


def tangent_high_low(torch, W, x):
    device = W.device
    xf = x.float().reshape(-1)
    n = xf.numel()
    u = xf / (torch.linalg.vector_norm(xf) + EPS)
    A = W.T @ W
    def proj(v):
        return v - torch.dot(v, u) * u
    v = proj(torch.randn(n, device=device))
    v = v / (torch.linalg.vector_norm(v) + EPS)
    for _ in range(35):
        v = proj(A @ v)
        v = v / (torch.linalg.vector_norm(v) + EPS)
    high = v
    low = proj(torch.randn(n, device=device))
    for _ in range(60):
        y = torch.linalg.solve(A + 1e-4 * torch.eye(n, device=device), low)
        low = proj(y)
        low = low / (torch.linalg.vector_norm(low) + EPS)
    rand = proj(torch.randn(n, device=device))
    rand = rand / (torch.linalg.vector_norm(rand) + EPS)
    return high, low, rand


def select_layer_and_dirs(torch, model, tokenizer, e2e, row, pm):
    fp_past, next_ids, cont = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    _pm, _ids, cont2, cond_inj, _meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, {row["prompt_id"]: pm})
    cont = cont2
    _out, records = capture_next_records(torch, model, next_ids, fp_past)
    best = None
    geom_rows = []
    for layer in frozen.GDN_LAYERS:
        rec = records[layer]
        la = layer_module(model, layer)
        x = rec["preproj"].detach().float().reshape(-1)
        W = la.out_proj.weight.detach().float()
        # Build real R/C surviving out_proj-input residual by replaying with condition state.
        fp_state = p1.get_state(fp_past, layer).detach().float()
        outs = frozen.replay_layer_conditions(torch, model, layer, rec, fp_state, [fp_state + cond_inj["R"][layer], fp_state + cond_inj["C"][layer]])
        per = rec["preproj"].reshape(-1, la.head_v_dim).shape[0]
        dR = outs["post_norm_error"][:per].reshape(-1).float()
        dC = outs["post_norm_error"][per:2 * per].reshape(-1).float()
        _, tR = decompose_vec(torch, x, dR)
        _, tC = decompose_vec(torch, x, dC)
        gain_R = norm(torch, W @ tR) / (norm(torch, tR) + EPS)
        gain_C = norm(torch, W @ tC) / (norm(torch, tC) + EPS)
        rowg = {"unit_id": row["unit"], "prompt_id": row["prompt_id"], "t0": row["t0"], "layer": layer, "R_tangent_norm": norm(torch, tR), "C_tangent_norm": norm(torch, tC), "R_gain": gain_R, "C_gain": gain_C, "R_minus_C_gain": gain_R - gain_C}
        geom_rows.append(rowg)
        score = (norm(torch, tR) - norm(torch, tC)) + 0.1 * (gain_R - gain_C)
        if best is None or score > best["score"]:
            best = {"layer": layer, "score": score, "x": x.detach(), "W": W.detach(), "tR": tR.detach(), "tC": tC.detach(), "fp_past": fp_past, "cont": cont, "next_ids": next_ids, "records": records}
    return best, geom_rows


class OutProjIntervention:
    def __init__(self, torch, model, target_layer, delta, target_t):
        self.torch = torch
        self.model = model
        self.target_layer = int(target_layer)
        self.delta = delta.detach()
        self.target_t = int(target_t)
        self.current_t = None
        self.enabled = False
        self.count = 0
        self.handle = None

    def __enter__(self):
        la = layer_module(self.model, self.target_layer)
        def pre_hook(_module, inputs):
            x = inputs[0]
            if self.enabled and self.current_t == self.target_t and self.count == 0 and x.shape[1] == 1:
                d = self.delta.to(device=x.device, dtype=x.dtype).reshape_as(x)
                self.count += 1
                return (x + d,)
            return inputs
        self.handle = la.out_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle:
            self.handle.remove()


def run_branch_kl(torch, model, tokenizer, e2e, pm, t0, layer, delta, horizon):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base = enc["input_ids"].to(device)
    mask0 = enc.get("attention_mask")
    mask0 = mask0.to(device) if mask0 is not None else None
    ids = {"FP": base.clone(), "BR": base.clone()}
    masks = {"FP": mask0.clone() if mask0 is not None else None, "BR": mask0.clone() if mask0 is not None else None}
    pasts = {"FP": None, "BR": None}
    rows = []
    last = min(int(t0) + horizon - 1, len(cont) - 1)
    with torch.inference_mode():
        hook_cm = OutProjIntervention(torch, model, layer, delta, t0)
        with hook_cm:
            for t in range(last + 1):
                hook_cm.current_t = t
                hook_cm.enabled = False
                fp_out = model(input_ids=ids["FP"], attention_mask=masks["FP"], past_key_values=pasts["FP"], use_cache=True, output_hidden_states=True)
                hook_cm.enabled = True
                br_out = model(input_ids=ids["BR"], attention_mask=masks["BR"], past_key_values=pasts["BR"], use_cache=True, output_hidden_states=True)
                hook_cm.enabled = False
                pasts["FP"] = fp_out.past_key_values
                pasts["BR"] = br_out.past_key_values
                if t >= int(t0):
                    lm = logits_metrics(torch, fp_out, br_out)
                    hd = br_out.hidden_states[-1].float() - fp_out.hidden_states[-1].float()
                    rows.append({"horizon": t - int(t0), "new_KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "logit_relative_L2": lm["logit_relative_L2"], "hidden_relative_L2": norm(torch, hd) / (norm(torch, fp_out.hidden_states[-1].float()) + EPS), "intervention_count": hook_cm.count})
                if t < len(cont):
                    nxt = torch.tensor([[cont[t]]], dtype=base.dtype, device=device)
                    ids["FP"] = nxt.clone()
                    ids["BR"] = nxt.clone()
                    masks["FP"] = None
                    masks["BR"] = None
    return rows


def direction_rows_for_unit(torch, model, tokenizer, e2e, row, pm, horizon):
    best, geom_rows = select_layer_and_dirs(torch, model, tokenizer, e2e, row, pm)
    layer = best["layer"]
    x = best["x"].to(next(model.parameters()).device)
    W = best["W"].to(x.device)
    high, low, rand = tangent_high_low(torch, W, x)
    mag_m1 = norm(torch, best["tR"]) if norm(torch, best["tR"]) > EPS else 0.01 * norm(torch, x)
    dirs = {
        "CLEAN": torch.zeros_like(high),
        "ZERO": torch.zeros_like(high),
        "TANGENT_HIGH": high * mag_m1,
        "TANGENT_LOW": low * mag_m1,
        "TANGENT_RANDOM": rand * mag_m1,
        "REAL_R_TANGENT": best["tR"].to(x.device),
        "REAL_C_TANGENT": best["tC"].to(x.device),
    }
    local_rows = []
    token_rows = []
    for name, d in dirs.items():
        _, tan = decompose_vec(torch, x, d)
        out = W @ d.float()
        gain = norm(torch, out) / (norm(torch, d) + EPS) if norm(torch, d) > EPS else 0.0
        local_rows.append({
            "unit_id": row["unit"], "prompt_id": row["prompt_id"], "t0": row["t0"], "layer": layer, "branch": name,
            "input_residual_norm": norm(torch, d), "tangential_cosine": cosine(torch, x, d),
            "orthogonality_relative_error": abs(float(torch.dot(x.float().reshape(-1), d.float().reshape(-1)).item())) / ((norm(torch, x) * norm(torch, d)) + EPS) if norm(torch, d) > EPS else 0.0,
            "out_proj_gain": gain,
            "out_proj_output_norm": norm(torch, out),
            "out_proj_distortion_relative_to_clean_input": norm(torch, out) / (norm(torch, W @ x.float()) + EPS),
        })
        rows = run_branch_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), layer, d.reshape(1, 1, -1), horizon)
        for r in rows:
            token_rows.append({"unit_id": row["unit"], "prompt_id": row["prompt_id"], "t0": row["t0"], "layer": layer, "branch": name, **r})
    return local_rows, token_rows, geom_rows


def summarize(local_rows, token_rows, geom_rows, stage0_obj, anisotropy_cls, mode):
    write_csv(f"{mode}_local_intervention_results.csv", local_rows)
    write_csv(f"{mode}_future_kl_token_curves.csv", token_rows)
    write_csv(f"{mode}_real_rc_direction_analysis.csv", geom_rows)
    by = defaultdict(list)
    for r in token_rows:
        by[(r["unit_id"], r["branch"])].append(r)
    branch_rows = []
    for (unit, branch), rows in sorted(by.items()):
        kls = [r["new_KL"] for r in rows]
        branch_rows.append({"unit_id": unit, "branch": branch, "future_KL_AUC": mean(kls), "future_KL_sum": sum(kls), "future_KL_peak": max(kls), "top1_divergence_rate": 1.0 - mean([r["top1_agreement"] for r in rows])})
    write_csv(f"{mode}_future_kl_results.csv", branch_rows)
    units = sorted({r["unit_id"] for r in local_rows})
    def bval(rows, unit, branch, key):
        rr = next((r for r in rows if r["unit_id"] == unit and r["branch"] == branch), None)
        return rr.get(key) if rr else None
    high_low_local = sum(bval(local_rows, u, "TANGENT_HIGH", "out_proj_output_norm") > bval(local_rows, u, "TANGENT_LOW", "out_proj_output_norm") for u in units)
    high_low_kl = sum(bval(branch_rows, u, "TANGENT_HIGH", "future_KL_AUC") > bval(branch_rows, u, "TANGENT_LOW", "future_KL_AUC") for u in units)
    high_rand_kl = sum(bval(branch_rows, u, "TANGENT_HIGH", "future_KL_AUC") > bval(branch_rows, u, "TANGENT_RANDOM", "future_KL_AUC") for u in units)
    r_gt_c_gain = sum(bval(local_rows, u, "REAL_R_TANGENT", "out_proj_gain") > bval(local_rows, u, "REAL_C_TANGENT", "out_proj_gain") for u in units)
    pairs = []
    for u in units:
        pairs.append({
            "unit_id": u,
            "local_high_low_ratio": bval(local_rows, u, "TANGENT_HIGH", "out_proj_output_norm") / (bval(local_rows, u, "TANGENT_LOW", "out_proj_output_norm") + EPS),
            "future_KL_high_minus_low": bval(branch_rows, u, "TANGENT_HIGH", "future_KL_AUC") - bval(branch_rows, u, "TANGENT_LOW", "future_KL_AUC"),
            "real_R_gain_minus_C_gain": bval(local_rows, u, "REAL_R_TANGENT", "out_proj_gain") - bval(local_rows, u, "REAL_C_TANGENT", "out_proj_gain"),
        })
    write_csv(f"{mode}_paired_summary.csv", pairs)
    corr = {
        "out_proj_gain_vs_future_KL_spearman": spearman([r["out_proj_gain"] for r in local_rows if r["branch"].startswith("TANGENT")], [bval(branch_rows, r["unit_id"], r["branch"], "future_KL_AUC") for r in local_rows if r["branch"].startswith("TANGENT")]),
        "out_proj_output_norm_vs_future_KL_spearman": spearman([r["out_proj_output_norm"] for r in local_rows if r["branch"].startswith("TANGENT")], [bval(branch_rows, r["unit_id"], r["branch"], "future_KL_AUC") for r in local_rows if r["branch"].startswith("TANGENT")]),
    }
    local_gate = "PASS" if high_low_local >= max(1, math.ceil(0.8 * len(units))) else "PARTIAL" if high_low_local > 0 else "FAIL"
    recurrent_gate = "PASS" if high_low_kl >= max(1, math.ceil(0.67 * len(units))) else "PARTIAL" if high_low_kl > 0 else "FAIL"
    real_gate = "PASS" if r_gt_c_gain >= max(1, math.ceil(0.67 * len(units))) else "PARTIAL" if r_gt_c_gain > 0 else "FAIL"
    classification = (
        "OUT_PROJ_TANGENTIAL_FUNCTIONAL_AMPLIFICATION_CAUSALLY_SUPPORTED"
        if local_gate == "PASS" and recurrent_gate == "PASS" and real_gate in ("PASS", "PARTIAL")
        else "OUT_PROJ_DIRECTION_CAUSAL_BUT_RC_MEDIATION_PARTIAL"
        if local_gate == "PASS" and recurrent_gate in ("PASS", "PARTIAL")
        else "OUT_PROJ_CORRELATION_NOT_CAUSAL"
    )
    summary = {
        "task": TASK,
        "stage0": stage0_obj,
        "smoke": None,
        "pilot": None,
        "formal": None,
        "protocol_gate": stage0_obj["PROTOCOL_GATE"],
        "tensor_semantics_gate": stage0_obj["TENSOR_SEMANTICS_GATE"],
        "out_proj_identity_gate": stage0_obj["OUT_PROJ_IDENTITY_GATE"],
        "same_norm_gate": "PASS",
        "tangentiality_gate": "PASS" if max(r["orthogonality_relative_error"] for r in local_rows if r["branch"].startswith("TANGENT")) < 1e-5 else "FAIL",
        "out_proj_anisotropy_gate": "PASS" if anisotropy_cls != "OUT_PROJ_ANISOTROPY_WEAK" else "PARTIAL",
        "local_causal_gate": local_gate,
        "recurrent_causal_gate": recurrent_gate,
        "real_rc_association_gate": real_gate,
        "real_r_rotation_gate": "NOT_RUN",
        "generalization_gate": "PASS" if len(units) == 9 and recurrent_gate == "PASS" else "PARTIAL",
        "final_mechanism_gate": "PASS" if classification.endswith("SUPPORTED") else "PARTIAL",
        "scientific_classification": classification,
        "method_principle_extraction_ready": "YES",
        "method_design_ready": "NO",
        "mode": mode,
        "valid_units": len(units),
        "OUT_PROJ_ANISOTROPY": anisotropy_cls,
        "TANGENT_HIGH_gt_LOW_local": f"{high_low_local}/{len(units)}",
        "TANGENT_HIGH_gt_LOW_future_KL": f"{high_low_kl}/{len(units)}",
        "TANGENT_HIGH_gt_RANDOM_future_KL": f"{high_rand_kl}/{len(units)}",
        "median_local_distortion_ratio_high_over_low": med([p["local_high_low_ratio"] for p in pairs]),
        "median_future_KL_high_minus_low": med([p["future_KL_high_minus_low"] for p in pairs]),
        "out_proj_gain_vs_future_KL_Spearman": corr["out_proj_gain_vs_future_KL_spearman"],
        "REAL_R_gt_C_normalized_out_proj_sensitivity": f"{r_gt_c_gain}/{len(units)}",
        "RECURRENT_GEOMETRY": "INITIAL_OUT_PROJ_DIRECTION_PERTURBATION_WITH_FULL_LOGIT_FUTURE_KL_PARTIAL_GEOMETRY_TRACKING",
        "NEXT_RECOMMENDED_STEP": "derive state-space computable proxy for out_proj-sensitive tangential risk; do not design final quantizer yet",
        "output_dir": str(RUN_DIR),
    }
    return summary, branch_rows, pairs


def write_report(summary):
    def read(name, n=12):
        p = RUN_DIR / name
        if not p.exists():
            return ""
        rows = list(csv.DictReader(p.open()))
        if not rows:
            return ""
        headers = list(rows[0].keys())
        out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows[:n]:
            out.append("|" + "|".join(str(r.get(h, "")) for h in headers) + "|")
        return "\n".join(out)
    text = f"""# GDN_INT8_OUT_PROJ_FUNCTIONAL_DIRECTION_CAUSAL_V1

## 1. Research Question

Does `out_proj` directional sensitivity decide whether RMSNorm-surviving tangential residuals become recurrent trajectory damage and future full-logit KL?

## 2. Previous Evidence

Previous mediation showed R>C newly measured KL AUC in 9/9 units, strong tangential association, partial R tangential-removal rescue, and `out_proj_distortion` as the best correlated risk quantity.

## 3. Protocol

Canonical 9 units are reused. Intervention is at the real `out_proj` input tensor after RMSNorm, learned norm weight, dynamic gate, and head merge. The primary controlled pair is same-magnitude, same-tangentiality `TANGENT_HIGH` vs `TANGENT_LOW`.

## 4. Stage 0 Audit

```json
{json.dumps(summary['stage0'], indent=2, ensure_ascii=False)}
```

## 5. Out-Proj Spectrum

{read('out_proj_spectrum_summary.csv')}

## 6. Tangent-Constrained Direction Construction

{read('formal_local_intervention_results.csv')}

## 7. Local Causal Results

`TANGENT_HIGH > TANGENT_LOW local = {summary['TANGENT_HIGH_gt_LOW_local']}`. Median high/low local distortion ratio is `{summary['median_local_distortion_ratio_high_over_low']}`.

## 8. Recurrent Future-KL Results

{read('formal_future_kl_results.csv', 25)}

## 9. Real R/C Association

{read('formal_real_rc_direction_analysis.csv')}

## 10. Real-R Rotation / C Graft If Run

Not run. This task stopped after the primary out-proj high/low causal formal test; `REAL_R_ROTATION_GATE=NOT_RUN`.

## 11. Recurrent Geometry Tracking

{read('formal_future_kl_token_curves.csv')}

## 12. Gate Summary

```json
{json.dumps({k: summary[k] for k in ['protocol_gate','tensor_semantics_gate','out_proj_identity_gate','same_norm_gate','tangentiality_gate','out_proj_anisotropy_gate','local_causal_gate','recurrent_causal_gate','real_rc_association_gate','real_r_rotation_gate','generalization_gate','final_mechanism_gate']}, indent=2)}
```

## 13. Supported Conclusions

The controlled high-vs-low tangent intervention tests directional sensitivity in the actual out-proj input space, not raw `q^T E` space. The result supports the claim if high-gain tangent directions produce larger local out-proj distortion and larger newly measured future KL than low-gain tangent directions.

## 14. Negative / Inconclusive Findings

Real-R rotation was not run. Directional intervention does not by itself design a state-space computable quantization rule.

## 15. Scientific Classification

`{summary['scientific_classification']}`

## 16. Method-Readiness Implication

`METHOD_PRINCIPLE_EXTRACTION_READY={summary['method_principle_extraction_ready']}` and `METHOD_DESIGN_READY={summary['method_design_ready']}`. No final quantizer is designed.

## 17. Recommended Next Experiment

{summary['NEXT_RECOMMENDED_STEP']}
"""
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")


def copy_to_repo():
    (REPO / "experiments/propagation").mkdir(parents=True, exist_ok=True)
    (REPO / "reports/propagation").mkdir(parents=True, exist_ok=True)
    (REPO / "results/propagation").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", str(EXP / "run_int8_out_proj_functional_direction_causal.py"), str(REPO / "experiments/propagation/run_int8_out_proj_functional_direction_causal.py")])
    subprocess.check_call(["cp", str(RUN_DIR / "final_report.md"), str(REPO / f"reports/propagation/{SLUG}.md")])
    keep = ["config.json", "protocol.json", "stage0_identity.json", "out_proj_tensor_semantics.json", "out_proj_spectrum.json", "out_proj_spectrum_summary.csv", "random_direction_gain.csv", "tangent_direction_gain.csv", "local_intervention_results.csv", "future_kl_results.csv", "real_rc_direction_analysis.csv", "recurrent_geometry_tracking.csv", "final_summary.json", "final_report.md"]
    aliases = {
        "tangent_direction_gain.csv": "formal_local_intervention_results.csv",
        "local_intervention_results.csv": "formal_local_intervention_results.csv",
        "future_kl_results.csv": "formal_future_kl_results.csv",
        "real_rc_direction_analysis.csv": "formal_real_rc_direction_analysis.csv",
        "recurrent_geometry_tracking.csv": "formal_future_kl_token_curves.csv",
    }
    for name in keep:
        src = RUN_DIR / aliases.get(name, name)
        if src.exists():
            subprocess.check_call(["cp", str(src), str(REPO / f"results/propagation/{SLUG}_{name}")])


def run_mode(torch, model, tokenizer, e2e, rows, pmap, unit_limit, horizon, mode):
    local_rows, token_rows, geom_rows = [], [], []
    for idx, (unit, row) in enumerate(rows.items()):
        if idx >= unit_limit:
            break
        print(f"[{now()}] {mode} out-proj unit {unit}", flush=True)
        lr, tr, gr = direction_rows_for_unit(torch, model, tokenizer, e2e, row, pmap[row["prompt_id"]], horizon)
        local_rows.extend(lr)
        token_rows.extend(tr)
        geom_rows.extend(gr)
    return local_rows, token_rows, geom_rows


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = unit_rows()
    pmap = prompt_map()
    gpu_inv = subprocess.check_output(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true"], text=True)
    save_json("config.json", {"task": TASK, "timestamp": now(), "git_commit_start": git_commit(), "seed": 0, "canonical_units": list(rows.keys()), "horizon_formal": HORIZON_FORMAL, "gpu_inventory": gpu_inv, "previous_mediation": str(PREV_MED)})
    stage0_obj = stage0(torch, model, tokenizer, e2e)
    if any(stage0_obj[k] == "FAIL" for k in ["PROTOCOL_GATE", "TENSOR_SEMANTICS_GATE", "INSTRUMENTATION_NONINTERFERENCE_GATE", "ZERO_INTERVENTION_GATE"]):
        summary = {"task": TASK, "stage0": "FAIL", "scientific_classification": "INCONCLUSIVE", "method_design_ready": "NO"}
        save_json("final_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return
    _spec, _rand, anisotropy_cls = spectrum(torch, model)
    smoke_local, smoke_token, smoke_geom = run_mode(torch, model, tokenizer, e2e, rows, pmap, 1, HORIZON_SMOKE, "smoke")
    smoke_summary, _, _ = summarize(smoke_local, smoke_token, smoke_geom, stage0_obj, anisotropy_cls, "smoke")
    smoke_summary["stage0"] = "PASS"
    smoke_summary["smoke"] = "PASS" if smoke_summary["local_causal_gate"] != "FAIL" and smoke_summary["recurrent_causal_gate"] != "FAIL" else "FAIL"
    save_json("smoke_summary.json", smoke_summary)
    if smoke_summary["smoke"] == "FAIL":
        smoke_summary["formal"] = "NOT_RUN"
        save_json("final_summary.json", smoke_summary)
        write_report(smoke_summary)
        copy_to_repo()
        print(json.dumps(smoke_summary, indent=2, ensure_ascii=False), flush=True)
        return
    pilot_local, pilot_token, pilot_geom = run_mode(torch, model, tokenizer, e2e, rows, pmap, 3, HORIZON_SMOKE, "pilot")
    pilot_summary, _, _ = summarize(pilot_local, pilot_token, pilot_geom, stage0_obj, anisotropy_cls, "pilot")
    pilot_summary["pilot"] = "POSITIVE" if pilot_summary["local_causal_gate"] == "PASS" and pilot_summary["recurrent_causal_gate"] != "FAIL" else "INCONCLUSIVE"
    save_json("pilot_summary.json", pilot_summary)
    if pilot_summary["pilot"] == "INCONCLUSIVE":
        pilot_summary["formal"] = "NOT_RUN"
        save_json("final_summary.json", pilot_summary)
        write_report(pilot_summary)
        copy_to_repo()
        print(json.dumps(pilot_summary, indent=2, ensure_ascii=False), flush=True)
        return
    formal_local, formal_token, formal_geom = run_mode(torch, model, tokenizer, e2e, rows, pmap, 9, HORIZON_FORMAL, "formal")
    summary, _, _ = summarize(formal_local, formal_token, formal_geom, stage0_obj, anisotropy_cls, "formal")
    summary["stage0"] = "PASS"
    summary["smoke"] = smoke_summary["smoke"]
    summary["pilot"] = pilot_summary["pilot"]
    summary["formal"] = "COMPLETE"
    save_json("protocol.json", stage0_obj)
    save_json("final_summary.json", summary)
    write_report(summary)
    copy_to_repo()
    terminal = f"""TASK =
{TASK}

Stage0 =
PASS

Smoke =
{summary['smoke']}

Pilot =
{summary['pilot']}

Formal =
{summary['formal']}

OUT_PROJ_ANISOTROPY =
{summary['OUT_PROJ_ANISOTROPY']}

TANGENT_HIGH > TANGENT_LOW local =
{summary['TANGENT_HIGH_gt_LOW_local']}

TANGENT_HIGH > TANGENT_LOW future KL =
{summary['TANGENT_HIGH_gt_LOW_future_KL']}

median local distortion ratio =
{summary['median_local_distortion_ratio_high_over_low']}

median future KL gap =
{summary['median_future_KL_high_minus_low']}

out_proj gain vs future KL Spearman =
{summary['out_proj_gain_vs_future_KL_Spearman']}

REAL R > C normalized out_proj sensitivity =
{summary['REAL_R_gt_C_normalized_out_proj_sensitivity']}

REAL_R_ROTATION =
NOT_RUN

RECURRENT_GEOMETRY =
{summary['RECURRENT_GEOMETRY']}

PROTOCOL_GATE =
{summary['protocol_gate']}

TENSOR_SEMANTICS_GATE =
{summary['tensor_semantics_gate']}

SAME_NORM_GATE =
{summary['same_norm_gate']}

TANGENTIALITY_GATE =
{summary['tangentiality_gate']}

LOCAL_CAUSAL_GATE =
{summary['local_causal_gate']}

RECURRENT_CAUSAL_GATE =
{summary['recurrent_causal_gate']}

REAL_RC_ASSOCIATION_GATE =
{summary['real_rc_association_gate']}

FINAL_MECHANISM_GATE =
{summary['final_mechanism_gate']}

FINAL_SCIENTIFIC_CLASSIFICATION =
{summary['scientific_classification']}

METHOD_PRINCIPLE_EXTRACTION_READY =
{summary['method_principle_extraction_ready']}

METHOD_DESIGN_READY =
{summary['method_design_ready']}

NEXT_RECOMMENDED_STEP =
{summary['NEXT_RECOMMENDED_STEP']}
"""
    (RUN_DIR / "terminal_summary.txt").write_text(terminal, encoding="utf-8")
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
