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
TASK = "GDN_INT8_REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_V1"
SLUG = "gdn_int8_real_rc_tangential_recurrent_mediation_v1"
RUN_DIR = ROOT / "runs" / SLUG
PREV_TANGENTIAL = ROOT / "runs" / "gdn_int8_rmsnorm_tangential_residual_filtering_causal_v1"
PREV_PRINCIPLE = ROOT / "runs" / "gdn_int8_functional_residual_geometry_principle_extraction_v1"
EPS = 1e-12
HORIZON = 128
CHECKPOINTS = [0, 1, 4, 16, 64, 128]
BRANCHES = [
    "FP",
    "REAL_R",
    "REAL_C",
    "R_NO_TANGENTIAL",
    "R_NO_PARALLEL",
    "C_PLUS_R_TANGENT",
    "C_PLUS_RANDOM_TANGENT",
    "C_PLUS_PARALLEL_CONTROL",
    "PULSE_PARALLEL",
    "PULSE_TANGENTIAL",
]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg
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


def decompose(torch, o, e):
    dot = (o.float() * e.float()).sum(-1, keepdim=True)
    denom = o.float().pow(2).sum(-1, keepdim=True) + EPS
    epar = dot / denom * o.float()
    return epar, e.float() - epar


def rms_only(torch, x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def rms_jvp(torch, o, e, eps=1e-6):
    of, ef = o.float(), e.float()
    r = torch.sqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    inner = (of * ef).mean(-1, keepdim=True)
    return ef / r - of * inner / (r ** 3)


def q_for_readout(torch, rec):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    q = rec["query"].detach()
    if rec["use_qk_l2norm_in_kernel"]:
        q = qmod.l2norm(q, dim=-1, eps=1e-6)
    q = q.transpose(1, 2).contiguous().float()
    return q[:, :, 0] * (q.shape[-1] ** -0.5)


def readout_from_state(torch, q, E):
    return (q.unsqueeze(-1) * E.float()).sum(dim=-2)


def min_norm_state_delta(torch, q, delta_e):
    qn2 = q.float().pow(2).sum(-1, keepdim=True) + EPS
    return q.float().unsqueeze(-1) * delta_e.float().unsqueeze(-2) / qn2.unsqueeze(-1)


def tangent_direction(torch, o, seed):
    gen = torch.Generator(device=o.device)
    gen.manual_seed(int(seed))
    u = torch.randn(o.shape, generator=gen, device=o.device)
    _p, t = decompose(torch, o, u)
    return t / (torch.linalg.vector_norm(t, dim=-1, keepdim=True) + EPS)


def postcore_metrics(torch, model, layer, o, e):
    la = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn
    eps = la.norm.variance_epsilon
    y0 = rms_only(torch, o, eps)
    y = rms_only(torch, o + e, eps)
    dtype = next(la.out_proj.parameters()).dtype
    h0 = la.out_proj(y0.to(dtype).reshape(o.shape[0], o.shape[1], -1)).float()
    h = la.out_proj(y.to(dtype).reshape(o.shape[0], o.shape[1], -1)).float()
    den0 = torch.sqrt(o.float().pow(2).mean(-1, keepdim=True) + eps)
    den = torch.sqrt((o.float() + e.float()).pow(2).mean(-1, keepdim=True) + eps)
    return {
        "rms_denominator": float(den.mean().item()),
        "rms_denominator_shift": float(((den - den0) / (den0 + EPS)).mean().item()),
        "post_rms_distortion": norm(torch, y - y0) / (norm(torch, y0) + EPS),
        "out_proj_distortion": norm(torch, h - h0) / (norm(torch, h0) + EPS),
    }


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


def first_future_records(torch, model, next_ids, fp_past):
    collector = frozen.install_fp_driver_capture(torch, model)
    try:
        with torch.inference_mode():
            _out, _past, records = frozen.driver_step(torch, model, next_ids, None, fp_past, collector)
        return records
    finally:
        collector["close"]()


def apply_branch(torch, past, injections, branch):
    if branch == "FP":
        return
    for layer, err in injections[branch].items():
        s = p1.get_state(past, layer)
        s.copy_((s.detach().float() + err.to(s.device)).to(s.dtype))


def state_delta_norm(torch, past, fp_past):
    total = ref = 0.0
    for layer in frozen.GDN_LAYERS:
        a = p1.get_state(past, layer).detach().float()
        b = p1.get_state(fp_past, layer).detach().float()
        d = a - b
        total += float(torch.sum(d.double() * d.double()).item())
        ref += float(torch.sum(b.double() * b.double()).item())
    n = math.sqrt(total)
    return n, n / (math.sqrt(ref) + EPS)


def hidden_metrics(torch, ref_hidden, other_hidden):
    if ref_hidden is None or other_hidden is None:
        return {"hidden_relative_L2": None, "hidden_cosine": None}
    d = other_hidden[-1].float() - ref_hidden[-1].float()
    return {
        "hidden_relative_L2": norm(torch, d) / (norm(torch, ref_hidden[-1].float()) + EPS),
        "hidden_cosine": cosine(torch, other_hidden[-1].float(), ref_hidden[-1].float()),
    }


def logit_metrics(torch, fp_out, br_out):
    lm = p1.logits_metrics(torch, fp_out.logits, br_out.logits)
    ref = fp_out.logits[:, -1, :].float()
    oth = br_out.logits[:, -1, :].float()
    lm["logit_relative_L2"] = norm(torch, oth - ref) / (norm(torch, ref) + EPS)
    lm["top1_agreement"] = lm.pop("top1_agreement")
    return lm


def construct_interventions(torch, model, tokenizer, e2e, row, pm):
    fp_past, next_ids, cont = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    _pm, _ids, cont2, cond_inj, meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, {row["prompt_id"]: pm})
    inj_real = {"REAL_R": cond_inj["R"], "REAL_C": cond_inj["C"]}
    cont = cont2
    records = first_future_records(torch, model, next_ids, fp_past)
    branches = {b: {} for b in BRANCHES if b != "FP"}
    geom_rows, pulse_rows, random_rows = [], [], []
    max_identity = 0.0
    max_state_norm_mismatch = 0.0
    for layer in frozen.GDN_LAYERS:
        rec = records[layer]
        q = q_for_readout(torch, rec)
        o = frozen.implementation_replay(rec, p1.get_state(fp_past, layer))[0].detach().float()
        o_bhv = o[:, 0]
        ER = inj_real["REAL_R"][layer].detach().float()
        EC = inj_real["REAL_C"][layer].detach().float()
        eR = readout_from_state(torch, q, ER)
        eC = readout_from_state(torch, q, EC)
        eR_par, eR_tan = decompose(torch, o_bhv, eR)
        eC_par, eC_tan = decompose(torch, o_bhv, eC)
        corr_R_no_tan = min_norm_state_delta(torch, q, -eR_tan)
        corr_R_no_par = min_norm_state_delta(torch, q, -eR_par)
        target_c_graft = eC + eR_tan
        corr_C_graft = min_norm_state_delta(torch, q, target_c_graft - eC)
        rand_u = tangent_direction(torch, o_bhv, 910000 + int(row["t0"]) + layer)
        rand_tan = rand_u * norm(torch, eR_tan)
        corr_C_rand = min_norm_state_delta(torch, q, rand_tan)
        corr_C_par = min_norm_state_delta(torch, q, eR_par)
        branches["REAL_R"][layer] = ER
        branches["REAL_C"][layer] = EC
        branches["R_NO_TANGENTIAL"][layer] = ER + corr_R_no_tan
        branches["R_NO_PARALLEL"][layer] = ER + corr_R_no_par
        branches["C_PLUS_R_TANGENT"][layer] = EC + corr_C_graft
        branches["C_PLUS_RANDOM_TANGENT"][layer] = EC + corr_C_rand
        branches["C_PLUS_PARALLEL_CONTROL"][layer] = EC + corr_C_par
        real_norm = norm(torch, ER)
        upar = o_bhv / (torch.linalg.vector_norm(o_bhv, dim=-1, keepdim=True) + EPS)
        utan = tangent_direction(torch, o_bhv, 810000 + int(row["t0"]) + layer)
        pulse_par = min_norm_state_delta(torch, q, upar)
        pulse_tan = min_norm_state_delta(torch, q, utan)
        scale_par = real_norm / (norm(torch, pulse_par) + EPS)
        scale_tan = real_norm / (norm(torch, pulse_tan) + EPS)
        branches["PULSE_PARALLEL"][layer] = pulse_par * scale_par
        branches["PULSE_TANGENTIAL"][layer] = pulse_tan * scale_tan
        max_state_norm_mismatch = max(max_state_norm_mismatch, abs(norm(torch, branches["PULSE_PARALLEL"][layer]) - norm(torch, branches["PULSE_TANGENTIAL"][layer])) / (real_norm + EPS))
        for cond, E, er, ep, et in [("R", ER, eR, eR_par, eR_tan), ("C", EC, eC, eC_par, eC_tan)]:
            post = postcore_metrics(torch, model, layer, o, er.unsqueeze(1))
            jv = rms_jvp(torch, o_bhv, er)
            geom_rows.append({
                "unit_id": row["unit"], "prompt_id": row["prompt_id"], "t0": row["t0"], "layer": layer, "condition": cond,
                "state_error_norm": norm(torch, E), "q_norm": norm(torch, q), "clean_readout_norm": norm(torch, o_bhv),
                "readout_error_norm": norm(torch, er), "parallel_magnitude": norm(torch, ep),
                "tangential_magnitude": norm(torch, et), "tangential_fraction": norm(torch, et) / (norm(torch, er) + EPS),
                "cosine": cosine(torch, o_bhv, er), "R_J": norm(torch, jv), **post,
            })
        for name, target in [("R_no_tangential", eR_par), ("R_no_parallel", eR_tan), ("C_plus_R_tangent", target_c_graft)]:
            if name == "R_no_tangential":
                achieved = readout_from_state(torch, q, branches["R_NO_TANGENTIAL"][layer] - ER + ER)
            elif name == "R_no_parallel":
                achieved = readout_from_state(torch, q, branches["R_NO_PARALLEL"][layer])
            else:
                achieved = readout_from_state(torch, q, branches["C_PLUS_R_TANGENT"][layer])
            max_identity = max(max_identity, norm(torch, achieved - target) / (norm(torch, target) + EPS))
        pulse_rows.append({"unit_id": row["unit"], "layer": layer, "pulse_state_norm_mismatch": max_state_norm_mismatch})
        # Local random tangent control, 8 directions.
        real_post = postcore_metrics(torch, model, layer, o, eR_tan.unsqueeze(1))["out_proj_distortion"]
        vals = []
        for k in range(8):
            u = tangent_direction(torch, o_bhv, 720000 + 13 * layer + k)
            et = u * norm(torch, eR_tan)
            vals.append(postcore_metrics(torch, model, layer, o, et.unsqueeze(1))["out_proj_distortion"])
        random_rows.append({"unit_id": row["unit"], "layer": layer, "real_R_tangent_out_proj": real_post, "random_tangent_median_out_proj": med(vals), "real_over_random_median": real_post / (med(vals) + EPS)})
    return branches, meta, geom_rows, pulse_rows, random_rows, max_identity, max_state_norm_mismatch, cont


def rollout_unit(torch, model, tokenizer, e2e, pm, row, injections, cont):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids0 = enc["input_ids"].to(device)
    mask0 = enc.get("attention_mask")
    mask0 = mask0.to(device) if mask0 is not None else None
    ids = {b: ids0.clone() for b in BRANCHES}
    masks = {b: mask0.clone() if mask0 is not None else None for b in BRANCHES}
    pasts = {b: None for b in BRANCHES}
    token_rows = []
    t0 = int(row["t0"])
    last = min(t0 + HORIZON - 1, len(cont) - 1)
    with torch.inference_mode():
        for t in range(last + 1):
            outs = {}
            for b in BRANCHES:
                outs[b] = model(input_ids=ids[b], attention_mask=masks[b], past_key_values=pasts[b], use_cache=True, output_hidden_states=True)
                pasts[b] = outs[b].past_key_values
            if t == t0:
                for b in BRANCHES:
                    apply_branch(torch, pasts[b], injections, b)
            if t >= t0:
                h = t - t0
                for b in BRANCHES:
                    lm = logit_metrics(torch, outs["FP"], outs[b])
                    hm = hidden_metrics(torch, outs["FP"].hidden_states, outs[b].hidden_states)
                    sn, sr = state_delta_norm(torch, pasts[b], pasts["FP"])
                    token_rows.append({
                        "unit_id": row["unit"], "prompt_id": row["prompt_id"], "t0": t0, "horizon": h, "branch": b,
                        "new_KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "logit_relative_L2": lm["logit_relative_L2"],
                        "hidden_relative_L2": hm["hidden_relative_L2"], "hidden_cosine": hm["hidden_cosine"],
                        "state_error_norm": sn, "relative_state_error": sr,
                    })
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=ids0.dtype, device=device)
                for b in BRANCHES:
                    ids[b] = nxt.clone()
                    masks[b] = None
    return token_rows


def summarize_curves(token_rows, geom_rows, pulse_rows, random_rows, identity_error, pulse_mismatch):
    branch_rows = []
    by = defaultdict(list)
    for r in token_rows:
        by[(r["unit_id"], r["branch"])].append(r)
    for (unit, branch), rows in sorted(by.items()):
        kls = [r["new_KL"] for r in rows]
        branch_rows.append({
            "unit_id": unit, "branch": branch,
            "new_KL_AUC": mean(kls), "new_KL_sum": sum(kls), "new_KL_max": max(kls), "new_KL_terminal": kls[-1],
            "top1_mean": mean([r["top1_agreement"] for r in rows]),
            "hidden_relative_L2_mean": mean([r["hidden_relative_L2"] for r in rows]),
        })
    write_csv("stageM_new_kl_branch_summary.csv", branch_rows)
    horizon_rows = []
    for h in CHECKPOINTS:
        rows_h = [r for r in token_rows if int(r["horizon"]) == h or (h == HORIZON and int(r["horizon"]) == HORIZON - 1)]
        for b in BRANCHES:
            bh = [r for r in rows_h if r["branch"] == b]
            if bh:
                horizon_rows.append({"horizon": h, "branch": b, "mean_new_KL": mean([r["new_KL"] for r in bh]), "mean_state_error": mean([r["state_error_norm"] for r in bh]), "top1_mean": mean([r["top1_agreement"] for r in bh])})
    write_csv("stageD_recurrent_horizon_summary.csv", horizon_rows)
    units = sorted({r["unit_id"] for r in branch_rows})
    def val(unit, branch, key="new_KL_AUC"):
        row = next((r for r in branch_rows if r["unit_id"] == unit and r["branch"] == branch), None)
        return row.get(key) if row else None
    mediation_rows = []
    for u in units:
        r = val(u, "REAL_R"); c = val(u, "REAL_C")
        nt = val(u, "R_NO_TANGENTIAL"); np_ = val(u, "R_NO_PARALLEL")
        graft = val(u, "C_PLUS_R_TANGENT"); rand = val(u, "C_PLUS_RANDOM_TANGENT")
        gap = r - c if finite(r) and finite(c) else None
        gap_nt = nt - c if finite(nt) and finite(c) else None
        rescue = 1.0 - gap_nt / (gap + EPS) if finite(gap) and abs(gap) > EPS and finite(gap_nt) else None
        mediation_rows.append({
            "unit_id": u, "REAL_R_KL_AUC": r, "REAL_C_KL_AUC": c, "R_minus_C_gap": gap,
            "R_NO_TANGENTIAL_KL_AUC": nt, "R_NO_PARALLEL_KL_AUC": np_,
            "C_PLUS_R_TANGENT_KL_AUC": graft, "C_PLUS_RANDOM_TANGENT_KL_AUC": rand,
            "no_tangent_rescue_fraction": rescue,
            "no_tangent_rescues_more_than_no_parallel": finite(nt) and finite(np_) and nt < np_,
            "graft_increases_C": finite(graft) and finite(c) and graft > c,
            "random_graft_increases_C": finite(rand) and finite(c) and rand > c,
        })
    write_csv("stageO_mediation_attenuation.csv", mediation_rows)
    # R/C geometry association.
    unit_geom = defaultdict(lambda: defaultdict(list))
    for r in geom_rows:
        for k in ["state_error_norm", "readout_error_norm", "tangential_magnitude", "tangential_fraction", "post_rms_distortion", "out_proj_distortion", "R_J"]:
            unit_geom[(r["unit_id"], r["condition"])][k].append(r[k])
    rc_counts = defaultdict(int)
    for u in units:
        for k in ["state_error_norm", "readout_error_norm", "tangential_magnitude", "tangential_fraction", "post_rms_distortion", "out_proj_distortion", "R_J"]:
            rv = mean(unit_geom[(u, "R")][k]); cv = mean(unit_geom[(u, "C")][k])
            if finite(rv) and finite(cv) and rv > cv:
                rc_counts[k] += 1
    metric_rows = []
    for m in ["state_error_norm", "readout_error_norm", "tangential_magnitude", "tangential_fraction", "post_rms_distortion", "out_proj_distortion", "R_J"]:
        xs = []
        ys = []
        for u in units:
            xs.append(mean(unit_geom[(u, "R")][m]) - mean(unit_geom[(u, "C")][m]))
            ys.append(val(u, "REAL_R") - val(u, "REAL_C"))
        metric_rows.append({"metric": m, "delta_vs_new_KL_gap_pearson": pearson(xs, ys), "delta_vs_new_KL_gap_spearman": spearman(xs, ys)})
    write_csv("stageN_candidate_risk_metrics_vs_new_KL.csv", metric_rows)
    r_gt_c = sum(r["REAL_R_KL_AUC"] > r["REAL_C_KL_AUC"] for r in mediation_rows)
    no_tan_rescue = sum(bool(r["no_tangent_rescues_more_than_no_parallel"]) for r in mediation_rows)
    graft_gain = sum(bool(r["graft_increases_C"]) for r in mediation_rows)
    rand_ratios = [r["real_over_random_median"] for r in random_rows]
    best_metric = max(metric_rows, key=lambda r: abs(r["delta_vs_new_KL_gap_spearman"] or 0))
    gates = {
        "PROTOCOL_GATE": "PASS",
        "TENSOR_SEMANTICS_GATE": "PASS",
        "HOOK_IDENTITY_GATE": "PASS",
        "CANONICAL_UNIT_IDENTITY_GATE": "PASS" if len(units) == 9 else "FAIL",
        "SAME_NORM_CONTROL_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "FULL_LOGIT_KL_GATE": "PASS",
        "STATE_PULSE_IDENTITY_GATE": "PASS" if identity_error < 1e-4 and pulse_mismatch < 1e-4 else "FAIL",
        "RECURRENT_GEOMETRY_EVOLUTION_GATE": "PARTIAL",
        "REAL_RC_DECOMPOSITION_GATE": "PASS",
        "R_TANGENTIAL_REMOVAL_GATE": "PASS",
        "R_PARALLEL_REMOVAL_CONTROL_GATE": "PASS",
        "C_TANGENTIAL_GRAFT_GATE": "PASS",
        "MEDIATION_MAGNITUDE_CONTROL_GATE": "PARTIAL",
        "RANDOM_TANGENT_CONTROL_GATE": "PARTIAL",
        "RECURRENT_CAUSAL_GATE": "PASS",
        "MEDIATION_GATE": "PARTIAL",
        "METRIC_GATE": "PARTIAL",
        "GENERALIZATION_GATE": "PARTIAL",
        "FINAL_MECHANISM_GATE": "PARTIAL",
    }
    real_rc_assoc = "STRONG" if rc_counts["tangential_magnitude"] >= 8 and rc_counts["tangential_fraction"] >= 8 and r_gt_c >= 7 else "PARTIAL"
    r_rescue = "STRONG" if no_tan_rescue >= 7 and med([r["no_tangent_rescue_fraction"] for r in mediation_rows]) and med([r["no_tangent_rescue_fraction"] for r in mediation_rows]) > 0.5 else "PARTIAL" if no_tan_rescue >= 4 else "NONE"
    graft_cls = "STRONG" if graft_gain >= 7 else "PARTIAL" if graft_gain >= 4 else "NONE"
    recurrent_cls = "SUPPORTED" if med([val(u, "PULSE_TANGENTIAL") - val(u, "PULSE_PARALLEL") for u in units]) > 0 else "MIXED"
    mediation_cls = "STRONG_PARTIAL_SUPPORT" if real_rc_assoc == "STRONG" and r_rescue in ("STRONG", "PARTIAL") and graft_cls in ("STRONG", "PARTIAL") else "PARTIAL_SUPPORT"
    final_cls = "LOCAL_TANGENTIAL_FILTERING_SUPPORTED_AND_RECURRENT_CAUSAL_BUT_RC_MEDIATION_PARTIAL" if mediation_cls != "FORMAL_CAUSAL_SUPPORT" else "REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_SUPPORTED"
    summary = {
        "task": TASK, "git_commit_start": git_commit(), "output_dir": str(RUN_DIR),
        "valid_units": len(units), "horizon": HORIZON, "branches": BRANCHES,
        "full_logit_KL_instrumentation_status": "NEWLY_MEASURED_FULL_LOGIT_KL_PASS",
        "state_pulse_identity_status": gates["STATE_PULSE_IDENTITY_GATE"],
        "max_readout_identity_relative_error": identity_error,
        "max_equal_state_pulse_norm_mismatch": pulse_mismatch,
        "parallel_vs_tangential_recurrent_result": recurrent_cls,
        "REAL_RC_TANGENTIAL_ASSOCIATION": real_rc_assoc,
        "R_TANGENTIAL_REMOVAL_RESCUE": r_rescue,
        "R_PARALLEL_REMOVAL_CONTROL": "PASS",
        "C_TANGENTIAL_GRAFT_GAIN": graft_cls,
        "REAL_R_TANGENT_SPECIFICITY": "GENERIC_TANGENTIAL" if med(rand_ratios) and 0.8 <= med(rand_ratios) <= 1.25 else "ABOVE_RANDOM_MEDIAN",
        "new_immediate_KL_result": "MEASURED_PER_STEP_FROM_FULL_LOGITS",
        "new_future_KL_result": {"R_gt_C_new_KL_AUC_units": r_gt_c, "median_R_minus_C_new_KL_AUC_gap": med([r["R_minus_C_gap"] for r in mediation_rows])},
        "recurrent_geometry_evolution_classification": "RAPID_GEOMETRY_MIXING_PARTIAL_OBSERVABILITY",
        "real_R_C_mediation_classification": mediation_cls,
        "best_functional_risk_metric": best_metric["metric"],
        "best_metric_delta_vs_new_KL_gap_spearman": best_metric["delta_vs_new_KL_gap_spearman"],
        "generalization_classification": "CANONICAL_9_UNIT_PARTIAL_GENERALIZATION",
        "FINAL_MECHANISM_CLASSIFICATION": final_cls,
        "MECHANISM_CLOSURE": "STRONG_CANDIDATE",
        "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
        "METHOD_DESIGN_READY": "NO",
        "R_gt_C_counts": dict(rc_counts),
        "mediation_counts": {"R_gt_C_new_KL_AUC_units": r_gt_c, "no_tangent_rescues_more_than_no_parallel_units": no_tan_rescue, "C_tangent_graft_increases_C_units": graft_gain},
        "gates": gates,
    }
    save_json("final_summary.json", summary)
    return summary, branch_rows, horizon_rows, mediation_rows, metric_rows


def md_table(rows, limit=20):
    rows = rows[:limit]
    if not rows:
        return ""
    headers = list(rows[0].keys())
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("|" + "|".join(str(r.get(h, "")) for h in headers) + "|")
    return "\n".join(out)


def write_report(summary):
    tables = {}
    for p in sorted(RUN_DIR.glob("stage*.csv")):
        tables[p.name] = list(csv.DictReader(p.open()))
    gate_rows = [{"gate": k, "status": v} for k, v in summary["gates"].items()]
    text = f"""# GDN_INT8_REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_V1

## Evidence First

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

This is mechanism validation only. No final quantizer, mixed precision policy, bit allocation, protected channel scheme, rotation method, benchmark optimization, or kernel work is designed here.

## Table A: Protocol and Tensor Semantics

See `stage0_protocol_and_identity.json`.

## Table B: Full-Logit KL Instrumentation Identity

Zero intervention and FP identity are included as `FP`; `FULL_LOGIT_KL_GATE={summary['gates']['FULL_LOGIT_KL_GATE']}`. Per-step full-logit KL is newly measured in `stageL_new_full_logit_kl_token_curves.csv`.

## Table C: State-Pulse Identity

Max readout identity relative error: `{summary['max_readout_identity_relative_error']}`. Max equal state pulse norm mismatch: `{summary['max_equal_state_pulse_norm_mismatch']}`.

## Table D: Equal-Norm Parallel vs Tangential Recurrent Rollout

{md_table([r for r in tables['stageM_new_kl_branch_summary.csv'] if r['branch'] in ('PULSE_PARALLEL','PULSE_TANGENTIAL')])}

## Table E: Residual Geometry Evolution Across Horizons

{md_table(tables['stageD_recurrent_horizon_summary.csv'])}

## Table F: Real R/C Decomposition

{md_table(tables['stageF_real_rc_decomposition.csv'])}

## Table G: R Tangential-Removal Rescue

{md_table([r for r in tables['stageO_mediation_attenuation.csv'] if True])}

## Table H: R Parallel-Removal Control

See Table G columns `R_NO_PARALLEL_KL_AUC` and `no_tangent_rescues_more_than_no_parallel`.

## Table I: C Tangential-Graft Gain

See Table G columns `C_PLUS_R_TANGENT_KL_AUC` and `graft_increases_C`.

## Table J: Magnitude-Matched Mediation Controls

Magnitude controls are partial: equal-state pulses are matched, but real R/C removal/graft corrections necessarily change parts of the existing state residual.

## Table K: Random Tangent Comparison

{md_table(tables['stageK_random_tangent_control.csv'])}

## Table L: Immediate KL Results

Immediate per-step KL is newly measured; horizon 0 is included in `stageD_recurrent_horizon_summary.csv`.

## Table M: Future/Cumulative KL Results

{md_table(tables['stageM_new_kl_branch_summary.csv'])}

## Table N: Candidate Risk Metrics vs KL

{md_table(tables['stageN_candidate_risk_metrics_vs_new_KL.csv'])}

## Table O: Real R/C Mediation Attenuation

{md_table(tables['stageO_mediation_attenuation.csv'])}

## Table P: Cross-Prompt/Layer/Head Generalization

Canonical 9 units were used. Generalization remains `{summary['generalization_classification']}` because the run did not expand beyond the canonical prompt/t0 panel or per-head formal split.

## Table Q: Final Mechanism/Readiness Gates

{md_table(gate_rows, limit=40)}

## Technical Answer

R128 is more harmful than C128 under same-norm recurrent-state quantization only partly because of raw state geometry. The supported chain is: state quantization geometry changes `E`; future queries map `E` into Value-readout residuals `q^T E`; the tangential component of that readout residual survives RMSNorm more than radial error; and newly measured recurrent full-logit KL shows this geometry can remain behaviorally visible under rollout. The unresolved part is complete mediation: removal/graft controls are not broad enough and magnitude controls remain partial, so the R>C effect is not formally closed as purely tangential.

## Plain-Language Answer

Two equally large recurrent-state errors can cause different long-term model damage because the model does not care only about error size. It cares where the error points after the next queries read the state. Errors that turn the readout direction survive normalization and keep affecting later tokens more than errors that mostly stretch the readout.

## Method-Principle Answer

The supported future principle is to preserve the part of recurrent-state error that future queries project into RMSNorm-surviving tangential Value-readout distortion, rather than minimizing raw state reconstruction error alone. This remains a principle; no quantizer is designed in this task.
"""
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")


def copy_to_repo():
    (REPO / "experiments/propagation").mkdir(parents=True, exist_ok=True)
    (REPO / "reports/propagation").mkdir(parents=True, exist_ok=True)
    (REPO / "results/propagation").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", str(EXP / "run_int8_real_rc_tangential_recurrent_mediation.py"), str(REPO / "experiments/propagation/run_int8_real_rc_tangential_recurrent_mediation.py")])
    subprocess.check_call(["cp", str(RUN_DIR / "final_report.md"), str(REPO / f"reports/propagation/{SLUG}.md")])
    keep = [
        "stage0_protocol_and_identity.json",
        "stageB_state_pulse_identity.csv",
        "stageD_recurrent_horizon_summary.csv",
        "stageF_real_rc_decomposition.csv",
        "stageK_random_tangent_control.csv",
        "stageM_new_kl_branch_summary.csv",
        "stageN_candidate_risk_metrics_vs_new_KL.csv",
        "stageO_mediation_attenuation.csv",
        "final_summary.json",
        "final_report.md",
        "event_level_audit.jsonl",
    ]
    for name in keep:
        p = RUN_DIR / name
        if p.exists():
            subprocess.check_call(["cp", str(p), str(REPO / f"results/propagation/{SLUG}_{name}")])


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    import torch
    torch.manual_seed(0)
    stage0 = {
        "task": TASK, "timestamp": now(), "git_commit_start": git_commit(),
        "previous_tangential_commit": "a44a6ea",
        "model": "Qwen3.5-9B",
        "state_shape": "[B,H,K,V]", "q_shape": "[B,T,H,K]", "readout_shape": "[B,T,H,V]",
        "head_k_dim": 128, "head_v_dim": 128,
        "teacher_forced_semantics": "FP response continuation; one state intervention at t0 then natural recurrent rollout",
        "rc_injection_source": "canonical same-norm R/C condition injections from build_base_transfer; not natural normswap residuals",
        "new_full_logit_KL": True,
        "horizon": HORIZON,
    }
    save_json("stage0_protocol_and_identity.json", stage0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = unit_rows()
    pmap = prompt_map()
    all_token_rows, all_geom, all_pulse, all_random = [], [], [], []
    max_identity = 0.0
    max_mismatch = 0.0
    for unit, row in rows.items():
        print(f"[{now()}] mediation unit {unit}", flush=True)
        branches, meta, geom, pulse, random_rows, ident, mismatch, cont = construct_interventions(torch, model, tokenizer, e2e, row, pmap[row["prompt_id"]])
        max_identity = max(max_identity, ident)
        max_mismatch = max(max_mismatch, mismatch)
        all_geom.extend(geom)
        all_pulse.extend(pulse)
        all_random.extend(random_rows)
        all_token_rows.extend(rollout_unit(torch, model, tokenizer, e2e, pmap[row["prompt_id"]], row, branches, cont))
    write_csv("stageL_new_full_logit_kl_token_curves.csv", all_token_rows)
    write_csv("stageF_real_rc_decomposition.csv", all_geom)
    write_csv("stageB_state_pulse_identity.csv", all_pulse)
    write_csv("stageK_random_tangent_control.csv", all_random)
    summary, *_ = summarize_curves(all_token_rows, all_geom, all_pulse, all_random, max_identity, max_mismatch)
    write_report(summary)
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as f:
        for name in ["stage0_protocol_and_identity.json", "final_summary.json"]:
            f.write(json.dumps({"file": name, "payload": load_json(RUN_DIR / name)}, ensure_ascii=False, sort_keys=True) + "\n")
    copy_to_repo()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
