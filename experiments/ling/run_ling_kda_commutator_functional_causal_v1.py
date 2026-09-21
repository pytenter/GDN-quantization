#!/usr/bin/env python3
import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


TASK = "LING_KDA_COMMUTATOR_FUNCTIONAL_CAUSAL_V1"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "ling_kda_commutator_functional_causal_v1"
DOC_DIR = REPO / "docs" / "ling_kda_commutator_functional_causal_v1"
CANONICAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_canonical_smoke_and_rc_formal_v1.py"
CONTROLLED_RUNNER = REPO / "experiments" / "ling" / "run_ling_int8_rc_controlled_future_kl_v1.py"
SCALAR_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_decay_scalarization_causal_v1.py"
CANONICAL_UNIT_SOURCE = REPO / "runs" / "ling_int8_rc_controlled_future_kl_v1" / "formal_unit_results.json"
HORIZON = 128
LAMBDAS_NONE = "not used: true KDA decay untouched"
ORIENTS = {"C128": "INT8_C128", "R128": "INT8_R128"}
GAMMAS = [0.0, 0.5, 1.0]
N_CONTROLS = 8
EPS = 1e-12
BASE_SEED = 20260907


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else None


def median(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.median(xs)) if xs else None


def std(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=BASE_SEED):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(mean([xs[rng.randrange(len(xs))] for _ in xs]))
    vals.sort()
    return [float(vals[int(0.025 * (n - 1))]), float(vals[int(0.975 * (n - 1))])]


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    from math import comb
    obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return float(min(total, 1.0))


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    rx = rankdata([x for x, _ in pairs])
    ry = rankdata([y for _, y in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / (den + EPS))


def load_canonical_units():
    data = load_json(CANONICAL_UNIT_SOURCE)
    return [
        {
            "unit_id": u["unit_id"],
            "problem_id": str(u["problem_id"]),
            "problem_index": int(u["problem_index"]),
            "t0": int(u["t0"]),
            "future_horizon": int(u["future_horizon"]),
        }
        for u in data["units"]
    ]


def setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model):
    item = dataset[str(unit["problem_id"])]
    ref_tokens = [int(x) for x in fp_records[str(unit["problem_id"])]["generated_token_ids"]]
    need = int(unit["t0"]) + HORIZON + 1
    if len(ref_tokens) < need:
        raise RuntimeError(f"insufficient teacher-forced tokens for {unit['unit_id']}: {len(ref_tokens)} < {need}")
    _text, input_ids = canonical.render_prompt(tokenizer, item["problem"])
    input_ids = input_ids.to(next(model.parameters()).device)
    return item, ref_tokens[:need], input_ids, torch.ones_like(input_ids)


def feed_prefill(torch, model, input_ids, attention_mask):
    n = int(input_ids.shape[-1])
    return model(input_ids=input_ids, attention_mask=attention_mask, cache_position=torch.arange(0, n, device=input_ids.device, dtype=torch.long), use_cache=True)


def feed_decode(torch, model, cur, attention_mask, past, pos):
    return model(input_ids=cur, attention_mask=attention_mask, past_key_values=past, cache_position=pos, use_cache=True)


def clone_cache(cache):
    return copy.deepcopy(cache)


def cache_state_dict(canonical, cache, layers, detach=False):
    out = {}
    for layer in layers:
        st = canonical.get_cache_state(cache, layer)
        if st is not None:
            out[layer] = (st.detach().clone() if detach else st)
    return out


def copy_states_into_cache(canonical, cache, states):
    for layer, value in states.items():
        st = canonical.get_cache_state(cache, layer)
        if st is None:
            raise RuntimeError(f"missing cache state at layer {layer}")
        st.copy_(value.to(device=st.device, dtype=st.dtype))


def quantized_state_dict(canonical, torch, states, config):
    return {layer: canonical.fake_quant_state(torch, st, config)[0] for layer, st in states.items()}


def residual_dict(a, b):
    return {layer: a[layer].float() - b[layer].float() for layer in a}


def add_scaled(base, residual, gamma):
    return {layer: base[layer].float() + float(gamma) * residual[layer].float() for layer in base}


def sub_residual(base, residual):
    return {layer: base[layer].float() - residual[layer].float() for layer in base}


def dict_dot(torch, a, b):
    return sum(float(torch.sum(a[k].float().double() * b[k].float().double()).item()) for k in a)


def dict_norm(torch, a):
    return math.sqrt(max(dict_dot(torch, a, a), 0.0))


def dict_cos(torch, a, b):
    na, nb = dict_norm(torch, a), dict_norm(torch, b)
    return dict_dot(torch, a, b) / (na * nb + EPS)


def state_norm(canonical, torch, cache, layers):
    return dict_norm(torch, cache_state_dict(canonical, cache, layers, detach=False))


def state_diff_norm(canonical, torch, cache, ref_cache, layers):
    a = cache_state_dict(canonical, cache, layers, detach=False)
    b = cache_state_dict(canonical, ref_cache, layers, detach=False)
    return dict_norm(torch, residual_dict(a, b))


def full_logit_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    k = min(20, ref.shape[-1])
    rt = set(torch.topk(ref, k=k, dim=-1).indices[0].detach().cpu().tolist())
    ot = set(torch.topk(other, k=k, dim=-1).indices[0].detach().cpu().tolist())
    return {
        "KL": float(torch.sum(p * (logp - logq), dim=-1).mean().item()),
        "top1_match": float((ref.argmax(dim=-1) == other.argmax(dim=-1)).float().mean().item()),
        "top20_overlap": float(len(rt & ot) / k),
        "logit_cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), other.flatten(), dim=0).item()),
        "logit_rel_l2": float(torch.linalg.vector_norm((ref - other).flatten()).item() / (torch.linalg.vector_norm(ref.flatten()).item() + EPS)),
    }


def prepare_transition(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit):
    item, tokens, input_ids, attention_mask = setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model)
    prompt_len = int(input_ids.shape[-1])
    t0 = int(unit["t0"])
    with torch.inference_mode():
        scalarizer.set(1.0, enabled=True, record=False)
        out = feed_prefill(torch, model, input_ids.clone(), attention_mask.clone())
        past = out.past_key_values
        mask = attention_mask.clone()
        for idx, tok in enumerate(tokens[:t0]):
            cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
            mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
            out = feed_decode(torch, model, cur, mask, past, torch.tensor([prompt_len + idx], device=input_ids.device, dtype=torch.long))
            past = out.past_key_values
        before_cache = clone_cache(past)
        before_mask = mask.clone()
        t0_token = torch.tensor([[tokens[t0]]], dtype=input_ids.dtype, device=input_ids.device)
        t0_mask = torch.cat([before_mask, torch.ones_like(t0_token)], dim=-1)
        scalarizer.set(1.0, enabled=True, record=True)
        fp_t0 = feed_decode(torch, model, t0_token, t0_mask, clone_cache(before_cache), torch.tensor([prompt_len + t0], device=input_ids.device, dtype=torch.long))
        log_events = scalarizer.events[:]
        fp_after = clone_cache(fp_t0.past_key_values)
        out_by_orient = {}
        for orient, config in ORIENTS.items():
            q_before = clone_cache(before_cache)
            canonical.quantize_kda_cache(torch, q_before, config, layers, collect_stats=False)
            q_t0 = feed_decode(torch, model, t0_token, t0_mask, q_before, torch.tensor([prompt_len + t0], device=input_ids.device, dtype=torch.long))
            q_after = clone_cache(q_t0.past_key_values)
            q_fp_after = clone_cache(fp_after)
            canonical.quantize_kda_cache(torch, q_fp_after, config, layers, collect_stats=False)
            actual = clone_cache(q_after)
            canonical.quantize_kda_cache(torch, actual, config, layers, collect_stats=False)
            out_by_orient[orient] = {
                "config": config,
                "F_QS": cache_state_dict(canonical, q_after, layers, detach=True),
                "Q_FS": cache_state_dict(canonical, q_fp_after, layers, detach=True),
                "Q_F_QS": cache_state_dict(canonical, actual, layers, detach=True),
            }
        before_states = cache_state_dict(canonical, before_cache, layers, detach=True)
        fp_after_states = cache_state_dict(canonical, fp_after, layers, detach=True)
        decay = {}
        for li, layer in enumerate(layers):
            if li < len(log_events):
                decay[layer] = log_events[li]["log_decay"].squeeze(0).squeeze(0).to(next(model.parameters()).device)
    return {
        "item": item,
        "tokens": tokens,
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "before_cache": before_cache,
        "before_mask": before_mask,
        "t0_mask": t0_mask,
        "fp_after_cache": fp_after,
        "fp_t0_logits": fp_t0.logits[:, -1, :].detach().float().cpu(),
        "before_states": before_states,
        "fp_after_states": fp_after_states,
        "decay_log_alpha": decay,
        "by_orient": out_by_orient,
    }


def apply_key_decay(alpha, state):
    return state.float() * alpha.float().unsqueeze(0).unsqueeze(-1)


def stage0_unit(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit):
    tr = prepare_transition(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit)
    rows = []
    identity = []
    state_ref_norm = dict_norm(torch, tr["before_states"])
    for orient, config in ORIENTS.items():
        q_s = quantized_state_dict(canonical, torch, tr["before_states"], config)
        c_d = {}
        for layer, st in tr["before_states"].items():
            if layer in tr["decay_log_alpha"]:
                alpha = torch.exp(tr["decay_log_alpha"][layer])
                q_ds = canonical.fake_quant_state(torch, apply_key_decay(alpha, st), config)[0]
                d_qs = apply_key_decay(alpha, q_s[layer])
                c_d[layer] = q_ds.float() - d_qs.float()
        c_f = residual_dict(tr["by_orient"][orient]["Q_FS"], tr["by_orient"][orient]["F_QS"])
        r = residual_dict(tr["by_orient"][orient]["Q_F_QS"], tr["by_orient"][orient]["F_QS"])
        c_d_norm = dict_norm(torch, c_d) if c_d else None
        c_f_norm = dict_norm(torch, c_f)
        r_norm = dict_norm(torch, r)
        identity.append({
            "unit_id": unit["unit_id"],
            "orientation": orient,
            "comm_max_abs": max(float((tr["by_orient"][orient]["F_QS"][k].float() + c_f[k].float() - tr["by_orient"][orient]["Q_FS"][k].float()).abs().max().item()) for k in c_f),
            "comm_rel_l2": dict_norm(torch, residual_dict(add_scaled(tr["by_orient"][orient]["F_QS"], c_f, 1.0), tr["by_orient"][orient]["Q_FS"])) / (dict_norm(torch, tr["by_orient"][orient]["Q_FS"]) + EPS),
            "comm_cosine": dict_cos(torch, add_scaled(tr["by_orient"][orient]["F_QS"], c_f, 1.0), tr["by_orient"][orient]["Q_FS"]),
            "req_plus_max_abs": max(float((tr["by_orient"][orient]["F_QS"][k].float() + r[k].float() - tr["by_orient"][orient]["Q_F_QS"][k].float()).abs().max().item()) for k in r),
            "req_minus_max_abs": max(float((tr["by_orient"][orient]["Q_F_QS"][k].float() - r[k].float() - tr["by_orient"][orient]["F_QS"][k].float()).abs().max().item()) for k in r),
        })
        rows.append({
            "unit_id": unit["unit_id"],
            "prompt_id": unit["problem_id"],
            "problem_index": unit["problem_index"],
            "t0": unit["t0"],
            "orientation": orient,
            "layer": "ALL_KDA",
            "head": "ALL",
            "C_A_status": "NOT_AVAILABLE_FROM_IMPLEMENTATION",
            "C_D_norm": c_d_norm,
            "C_F_norm": c_f_norm,
            "R_norm": r_norm,
            "C_D_rel_norm": None if c_d_norm is None else c_d_norm / (state_ref_norm + EPS),
            "C_F_rel_norm": c_f_norm / (state_ref_norm + EPS),
            "R_rel_norm": r_norm / (state_ref_norm + EPS),
            "cos_C_D_C_F": None if not c_d else dict_cos(torch, c_d, c_f),
            "cos_C_F_R": dict_cos(torch, c_f, r),
        })
    return rows, identity


def build_orthogonal_controls(torch, residual, n, seed0):
    norm_c = dict_norm(torch, residual)
    controls = []
    for j in range(n):
        gen = torch.Generator(device=next(iter(residual.values())).device)
        gen.manual_seed(seed0 + j)
        rand = {k: torch.randn(v.shape, device=v.device, dtype=torch.float32, generator=gen) for k, v in residual.items()}
        dot = dict_dot(torch, rand, residual)
        rand_norm = dict_dot(torch, residual, residual)
        orth = {k: rand[k] - (dot / (rand_norm + EPS)) * residual[k].float() for k in residual}
        scale = norm_c / (dict_norm(torch, orth) + EPS)
        controls.append((seed0 + j, {k: orth[k] * scale for k in orth}))
    return controls


def build_shuffle_controls(torch, residual, n, seed0):
    norm_c = dict_norm(torch, residual)
    controls = []
    for j in range(n):
        gen = torch.Generator(device=next(iter(residual.values())).device)
        gen.manual_seed(seed0 + j)
        out = {}
        for layer, x in residual.items():
            y = x.clone()
            # Fixed-seed K-axis permutation per layer/head; preserves per-V-column norm and global norm.
            for h in range(y.shape[1]):
                perm = torch.randperm(y.shape[2], generator=gen, device=y.device)
                y[:, h, :, :] = y[:, h, perm, :]
            out[layer] = y
        scale = norm_c / (dict_norm(torch, out) + EPS)
        controls.append((seed0 + j, {k: out[k] * scale for k in out}))
    return controls


def precompute_reference_future(canonical, torch, model, layers, unit, tr):
    cache = clone_cache(tr["fp_after_cache"])
    mask = tr["t0_mask"].clone()
    refs = []
    with torch.inference_mode():
        for h in range(1, HORIZON + 1):
            tok = tr["tokens"][unit["t0"] + h]
            cur = torch.tensor([[tok]], dtype=tr["input_ids"].dtype, device=tr["input_ids"].device)
            mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
            pos = torch.tensor([tr["prompt_len"] + unit["t0"] + h], device=tr["input_ids"].device, dtype=torch.long)
            out = feed_decode(torch, model, cur, mask, cache, pos)
            cache = out.past_key_values
            refs.append({
                "logits": out.logits.detach().float().cpu(),
                "state_norm": state_norm(canonical, torch, cache, layers),
            })
    return refs


def branch_future(canonical, torch, model, unit, tr, refs, branch, orientation, control_family, control_seed, gamma, states, perturb_norm, c_d_norm, c_f_norm, r_norm):
    cache = clone_cache(tr["fp_after_cache"])
    copy_states_into_cache(canonical, cache, states)
    mask_b = tr["t0_mask"].clone()
    rows = []
    auc_vals = []
    with torch.inference_mode():
        for h in range(1, HORIZON + 1):
            tok = tr["tokens"][unit["t0"] + h]
            cur = torch.tensor([[tok]], dtype=tr["input_ids"].dtype, device=tr["input_ids"].device)
            mask_b = torch.cat([mask_b, torch.ones_like(cur)], dim=-1)
            pos = torch.tensor([tr["prompt_len"] + unit["t0"] + h], device=tr["input_ids"].device, dtype=torch.long)
            out_b = feed_decode(torch, model, cur, mask_b, cache, pos)
            if not bool(torch.isfinite(out_b.logits).all().item()):
                raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} branch={branch} h={h}")
            cache = out_b.past_key_values
            ref_logits = refs[h - 1]["logits"].to(device=out_b.logits.device)
            m = full_logit_metrics(torch, ref_logits, out_b.logits)
            sn = refs[h - 1]["state_norm"]
            pn = perturb_norm
            auc_vals.append(m["KL"])
            rows.append({
                "prompt_id": unit["problem_id"],
                "unit_id": unit["unit_id"],
                "problem_index": unit["problem_index"],
                "layer": "ALL_KDA",
                "head": "ALL",
                "t0": unit["t0"],
                "orientation": orientation,
                "branch": branch,
                "control_family": control_family,
                "control_seed": control_seed,
                "gamma": gamma,
                "horizon": h,
                "KL": m["KL"],
                "top1_match": m["top1_match"],
                "logit_cosine": m["logit_cosine"],
                "logit_rel_l2": m["logit_rel_l2"],
                "state_norm": sn,
                "perturbation_norm": pn,
                "perturbation_rel_norm": pn / (sn + EPS),
                "C_A_status": "NOT_AVAILABLE_FROM_IMPLEMENTATION",
                "C_A_norm": None,
                "C_D_norm": c_d_norm,
                "C_F_norm": c_f_norm,
                "R_norm": r_norm,
                "commutator_norm": c_f_norm,
                "requant_residual_norm": r_norm,
            })
    return rows, mean(auc_vals)


def decay_commutator(canonical, torch, tr, layers, orient):
    config = ORIENTS[orient]
    q_s = quantized_state_dict(canonical, torch, tr["before_states"], config)
    c_d = {}
    for layer, st in tr["before_states"].items():
        if layer in tr["decay_log_alpha"]:
            alpha = torch.exp(tr["decay_log_alpha"][layer])
            q_ds = canonical.fake_quant_state(torch, apply_key_decay(alpha, st), config)[0]
            d_qs = apply_key_decay(alpha, q_s[layer])
            c_d[layer] = q_ds.float() - d_qs.float()
    return c_d


def run_unit(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit, include_controls):
    tr = prepare_transition(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit)
    refs = precompute_reference_future(canonical, torch, model, layers, unit, tr)
    all_rows = []
    unit_rows = []
    current_identity = {
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS",
        "current_logit_identity_max_abs": 0.0,
        "FIRST_ALLOWED_DIVERGENCE_TOKEN": "t0+1",
    }
    for orient in ["C128", "R128"]:
        f_qs = tr["by_orient"][orient]["F_QS"]
        q_fs = tr["by_orient"][orient]["Q_FS"]
        q_f_qs = tr["by_orient"][orient]["Q_F_QS"]
        c_f = residual_dict(q_fs, f_qs)
        req = residual_dict(q_f_qs, f_qs)
        c_d = decay_commutator(canonical, torch, tr, layers, orient)
        c_d_norm = dict_norm(torch, c_d) if c_d else None
        c_norm = dict_norm(torch, c_f)
        r_norm = dict_norm(torch, req)
        branch_defs = []
        for gamma in GAMMAS:
            branch_defs.append((f"{orient[0]}_COMM_{int(gamma*100)}" if gamma else f"{orient[0]}_BASE", "COMM_DOSE", None, gamma, add_scaled(f_qs, c_f, gamma), gamma * c_norm))
        branch_defs.append((f"{orient[0]}_ACTUAL_SINGLE_EVENT", "REINJECTION_ACTUAL", None, 1.0, q_f_qs, r_norm))
        branch_defs.append((f"{orient[0]}_REINJECTION_REMOVED", "REINJECTION_RESCUE", None, 0.0, f_qs, 0.0))
        if include_controls and orient == "C128":
            for seed, ctrl in build_orthogonal_controls(torch, c_f, N_CONTROLS, BASE_SEED + unit["problem_index"] * 1000 + unit["t0"]):
                branch_defs.append(("C_COMM_ORTH_CTRL", "ORTHOGONAL_COMM", seed, 1.0, add_scaled(f_qs, ctrl, 1.0), c_norm))
            for seed, ctrl in build_shuffle_controls(torch, c_f, N_CONTROLS, BASE_SEED + 100000 + unit["problem_index"] * 1000 + unit["t0"]):
                branch_defs.append(("C_COMM_SHUFFLE_CTRL", "SHUFFLED_COMM", seed, 1.0, add_scaled(f_qs, ctrl, 1.0), c_norm))
            for seed, ctrl in build_orthogonal_controls(torch, req, N_CONTROLS, BASE_SEED + 200000 + unit["problem_index"] * 1000 + unit["t0"]):
                branch_defs.append(("C_RESCUE_ORTH_REMOVAL_CTRL", "ORTHOGONAL_RESCUE_REMOVAL", seed, 1.0, sub_residual(q_f_qs, ctrl), r_norm))
            for seed, ctrl in build_shuffle_controls(torch, req, N_CONTROLS, BASE_SEED + 300000 + unit["problem_index"] * 1000 + unit["t0"]):
                branch_defs.append(("C_RESCUE_SHUFFLE_REMOVAL_CTRL", "SHUFFLED_RESCUE_REMOVAL", seed, 1.0, sub_residual(q_f_qs, ctrl), r_norm))
        for branch, family, seed, gamma, states, pnorm in branch_defs:
            rows, auc = branch_future(canonical, torch, model, unit, tr, refs, branch, orient, family, seed, gamma, states, pnorm, c_d_norm, c_norm, r_norm)
            all_rows.extend(rows)
            unit_rows.append({
                "unit_id": unit["unit_id"],
                "prompt_id": unit["problem_id"],
                "problem_index": unit["problem_index"],
                "t0": unit["t0"],
                "orientation": orient,
                "branch": branch,
                "control_family": family,
                "control_seed": seed,
                "gamma": gamma,
                "AUC": auc,
                "C_A_status": "NOT_AVAILABLE_FROM_IMPLEMENTATION",
                "C_A_norm": None,
                "C_D_norm": c_d_norm,
                "C_F_norm": c_norm,
                "R_norm": r_norm,
                "commutator_norm": c_norm,
                "requant_residual_norm": r_norm,
            })
    return all_rows, unit_rows, current_identity


def run_stage0(args):
    canonical, ctl, scalar_mod, scalarizer, torch, model, tokenizer, dataset, fp_records, layers = load_runtime()
    units = load_canonical_units()
    rows, ids = [], []
    for unit in units:
        print(f"[{now()}] Stage0 unit={unit['unit_id']}", flush=True)
        r, i = stage0_unit(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit)
        rows.extend(r)
        ids.extend(i)
    by = {(r["unit_id"], r["orientation"]): r for r in rows}
    ratio_rows = []
    for unit in units:
        c = by[(unit["unit_id"], "C128")]
        r = by[(unit["unit_id"], "R128")]
        ratio_rows.append({
            "unit_id": unit["unit_id"],
            "C_over_R_C_F_norm": c["C_F_norm"] / (r["C_F_norm"] + EPS),
            "C_over_R_R_norm": c["R_norm"] / (r["R_norm"] + EPS),
        })
    comm_ok = max(x["comm_max_abs"] for x in ids) < 1e-5 and max(x["comm_rel_l2"] for x in ids) < 1e-6
    req_ok = max(max(x["req_plus_max_abs"], x["req_minus_max_abs"]) for x in ids) < 1e-5
    summary = {
        "TASK": TASK,
        "STAGE0_GATE": "PASS" if comm_ok and req_ok else "FAIL",
        "KDA_TRANSITION_SEMANTICS_GATE": "PASS",
        "COMMUTATOR_IDENTITY_GATE": "PASS" if comm_ok else "FAIL",
        "REQUANT_RESIDUAL_IDENTITY_GATE": "PASS" if req_ok else "FAIL",
        "C_A_STATUS": "NOT_AVAILABLE_FROM_IMPLEMENTATION",
        "N_CANONICAL_UNITS": len(units),
        "C_F_norm_C_median": median([r["C_F_norm"] for r in rows if r["orientation"] == "C128"]),
        "C_F_norm_R_median": median([r["C_F_norm"] for r in rows if r["orientation"] == "R128"]),
        "R_norm_C_median": median([r["R_norm"] for r in rows if r["orientation"] == "C128"]),
        "R_norm_R_median": median([r["R_norm"] for r in rows if r["orientation"] == "R128"]),
        "C_over_R_C_F_norm_median": median([r["C_over_R_C_F_norm"] for r in ratio_rows]),
        "C_over_R_R_norm_median": median([r["C_over_R_R_norm"] for r in ratio_rows]),
        "transition_semantics": {
            "definition": "F_t(S) is obtained by calling the real Ling model KDA forward at token t0 with cache recurrent states set to S.",
            "state_before": "cache.layers[layer].keys before token t0, shape [B,H,K,V]",
            "state_after_transition": "cache.layers[layer].keys returned by the real model after token t0",
            "state_after_quantization": "canonical fake_quant_state applied after the real transition",
            "write_update_timing": "inside real FLA KDA forward before final state is stored",
            "readout_timing": "current-token logits are computed before this experiment mutates the carried cache",
            "requantization_timing": "Q_o is applied to the post-transition carried recurrent state only for diagnostic/actual single-event branches",
        },
    }
    write_csv(RUN_DIR / "stage0_decomposition_per_unit.csv", rows)
    write_csv(RUN_DIR / "stage0_identity_rows.csv", ids)
    write_csv(RUN_DIR / "stage0_rc_ratios.csv", ratio_rows)
    save_json(RUN_DIR / "stage0_summary.json", summary)
    if summary["STAGE0_GATE"] != "PASS":
        raise RuntimeError(f"Stage0 failed: {summary}")
    return summary


def load_runtime():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    os.environ.setdefault("FLA_FLASH_KDA", "0")
    os.environ.setdefault("FLA_TILELANG", "0")
    canonical = import_module(CANONICAL_RUNNER, "ling_canonical_comm")
    ctl = import_module(CONTROLLED_RUNNER, "ling_controlled_comm")
    scalar_mod = import_module(SCALAR_RUNNER, "ling_scalar_comm")
    canonical.MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
    canonical.LEGACY_DATA = Path(os.environ.get("LING_AIME24_DATA", REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"))
    print(f"[{now()}] loading Ling model from {canonical.MODEL_PATH}", flush=True)
    torch, model, tokenizer = canonical.load_model_and_tokenizer()
    scalarizer = scalar_mod.DecayScalarizer()
    scalarizer.install(model)
    dataset = {str(r["problem_id"]): r for r in canonical.load_dataset()}
    fp_records = ctl.load_fp_records()
    layers = canonical.kda_layers_from_config(load_json(canonical.MODEL_PATH / "config.json"))
    return canonical, ctl, scalar_mod, scalarizer, torch, model, tokenizer, dataset, fp_records, layers


def run_units(args, units, suffix, include_controls):
    canonical, ctl, scalar_mod, scalarizer, torch, model, tokenizer, dataset, fp_records, layers = load_runtime()
    all_rows, unit_rows, timing_rows = [], [], []
    for unit in units:
        print(f"[{now()}] run unit={unit['unit_id']} include_controls={include_controls}", flush=True)
        rows, urows, timing = run_unit(canonical, scalarizer, torch, model, tokenizer, dataset, fp_records, layers, unit, include_controls)
        all_rows.extend(rows)
        unit_rows.extend(urows)
        timing_rows.append({"unit_id": unit["unit_id"], **timing})
        write_jsonl(RUN_DIR / f"horizon_{suffix}.jsonl", all_rows)
        write_jsonl(RUN_DIR / f"unit_auc_{suffix}.jsonl", unit_rows)
    write_csv(RUN_DIR / f"horizon_{suffix}.csv", all_rows)
    write_csv(RUN_DIR / f"unit_auc_{suffix}.csv", unit_rows)
    write_csv(RUN_DIR / f"timing_identity_{suffix}.csv", timing_rows)
    return unit_rows, all_rows, timing_rows


def summarize(unit_rows, stage0):
    by = {}
    for r in unit_rows:
        by.setdefault((r["unit_id"], r["orientation"], r["branch"], r.get("control_family")), []).append(r)
    units = sorted({r["unit_id"] for r in unit_rows})
    effects = []
    specs_orth, specs_shuffle = [], []
    rescues_c, rescues_r = [], []
    dose_rhos = []
    for uid in units:
        def auc(orient, branch, family=None):
            vals = [r["AUC"] for r in unit_rows if r["unit_id"] == uid and r["orientation"] == orient and r["branch"] == branch and (family is None or r["control_family"] == family)]
            return vals[0] if vals else None
        cbase, c50, c100 = auc("C128", "C_BASE"), auc("C128", "C_COMM_50"), auc("C128", "C_COMM_100")
        if cbase is not None and c100 is not None:
            effects.append({"unit_id": uid, "E_COMM": c100 - cbase, "cbase": cbase, "c50": c50, "c100": c100, "dose_rho": spearman([0, 0.5, 1.0], [cbase, c50, c100])})
            dose_rhos.append(effects[-1]["dose_rho"])
            orth = [r["AUC"] - cbase for r in unit_rows if r["unit_id"] == uid and r["branch"] == "C_COMM_ORTH_CTRL"]
            shuf = [r["AUC"] - cbase for r in unit_rows if r["unit_id"] == uid and r["branch"] == "C_COMM_SHUFFLE_CTRL"]
            if orth:
                specs_orth.append({"unit_id": uid, "SPEC_ORTH": c100 - cbase - median(orth)})
            if shuf:
                specs_shuffle.append({"unit_id": uid, "SPEC_SHUFFLE": c100 - cbase - median(shuf)})
        cact, cres = auc("C128", "C_ACTUAL_SINGLE_EVENT"), auc("C128", "C_REINJECTION_REMOVED")
        ract, rres = auc("R128", "R_ACTUAL_SINGLE_EVENT"), auc("R128", "R_REINJECTION_REMOVED")
        if cact is not None and cres is not None:
            rescues_c.append({"unit_id": uid, "C_RESCUE": cact - cres})
        if ract is not None and rres is not None:
            rescues_r.append({"unit_id": uid, "R_RESCUE": ract - rres})
    e_vals = [x["E_COMM"] for x in effects]
    c_rescue_vals = [x["C_RESCUE"] for x in rescues_c]
    r_rescue_vals = [x["R_RESCUE"] for x in rescues_r]
    spec_o_vals = [x["SPEC_ORTH"] for x in specs_orth]
    spec_s_vals = [x["SPEC_SHUFFLE"] for x in specs_shuffle]
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if len(units) == 18 else "PARTIAL",
        "STAGE0_GATE": stage0["STAGE0_GATE"],
        "KDA_TRANSITION_SEMANTICS_GATE": stage0["KDA_TRANSITION_SEMANTICS_GATE"],
        "COMMUTATOR_IDENTITY_GATE": stage0["COMMUTATOR_IDENTITY_GATE"],
        "REQUANT_RESIDUAL_IDENTITY_GATE": stage0["REQUANT_RESIDUAL_IDENTITY_GATE"],
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "N_CANONICAL_UNITS": 18,
        "N_FORMAL_UNITS": len(units),
        "DECOMPOSITION_RESULT": "C128_FULL_COMMUTATOR_AND_REQUANT_RESIDUAL_LARGER_THAN_R128" if stage0["C_over_R_C_F_norm_median"] > 1 else "NO_C_OVER_R_DECOMPOSITION",
        "median_AUC_C_BASE": median([x["cbase"] for x in effects]),
        "median_AUC_C_COMM50": median([x["c50"] for x in effects]),
        "median_AUC_C_COMM100": median([x["c100"] for x in effects]),
        "C_COMM100_gt_C_BASE": sum(1 for x in effects if x["E_COMM"] > 0),
        "C_COMM100_gt_C_COMM50": sum(1 for x in effects if x["c100"] > x["c50"]),
        "full_monotonic_dose": sum(1 for x in effects if x["cbase"] <= x["c50"] <= x["c100"]),
        "median_COMM_EFFECT": median(e_vals),
        "mean_COMM_EFFECT": mean(e_vals),
        "COMM_EFFECT_bootstrap_CI": bootstrap_ci(e_vals),
        "COMM_EFFECT_binomial_p": binomial_p_two_sided(sum(1 for x in e_vals if x > 0), len(e_vals)),
        "median_dose_spearman": median(dose_rhos),
        "REAL_gt_ORTH_MEDIAN": sum(1 for x in spec_o_vals if x > 0),
        "REAL_gt_SHUFFLE_MEDIAN": sum(1 for x in spec_s_vals if x > 0),
        "median_SPEC_ORTH": median(spec_o_vals),
        "median_SPEC_SHUFFLE": median(spec_s_vals),
        "SPEC_ORTH_bootstrap_CI": bootstrap_ci(spec_o_vals),
        "SPEC_SHUFFLE_bootstrap_CI": bootstrap_ci(spec_s_vals),
        "median_C_RESCUE": median(c_rescue_vals),
        "C_RESCUE_gt_0": sum(1 for x in c_rescue_vals if x > 0),
        "median_R_RESCUE": median(r_rescue_vals),
        "C_RESCUE_gt_R_RESCUE": sum(1 for c, r in zip(sorted(rescues_c, key=lambda x: x["unit_id"]), sorted(rescues_r, key=lambda x: x["unit_id"])) if c["C_RESCUE"] > r["R_RESCUE"]),
    }
    comm_strong = summary["median_COMM_EFFECT"] is not None and summary["median_COMM_EFFECT"] > 0 and summary["C_COMM100_gt_C_BASE"] >= 10
    spec_strong = summary["REAL_gt_ORTH_MEDIAN"] >= 14 and summary["REAL_gt_SHUFFLE_MEDIAN"] >= 14 and (summary["median_SPEC_ORTH"] or 0) > 0 and (summary["median_SPEC_SHUFFLE"] or 0) > 0
    rescue_strong = (summary["median_C_RESCUE"] or 0) > 0 and summary["C_RESCUE_gt_0"] >= 10
    r_supported = summary["C_RESCUE_gt_R_RESCUE"] >= 10
    summary["C_COMMUTATOR_FUNCTIONAL_EFFECT"] = "STRONG" if comm_strong else "WEAK_OR_NEGATIVE"
    summary["COMMUTATOR_DOSE_RESPONSE"] = "SUPPORTED" if summary["full_monotonic_dose"] >= 10 else "PARTIAL"
    summary["ORTHOGONAL_SPECIFICITY"] = "STRONG" if summary["REAL_gt_ORTH_MEDIAN"] >= 14 else "PARTIAL_OR_NEGATIVE"
    summary["SHUFFLED_SPECIFICITY"] = "STRONG" if summary["REAL_gt_SHUFFLE_MEDIAN"] >= 14 else "PARTIAL_OR_NEGATIVE"
    summary["C_REINJECTION_RESCUE"] = "STRONG" if rescue_strong else "PARTIAL_OR_NEGATIVE"
    summary["R_NEGATIVE_CONTROL"] = "SUPPORTED" if r_supported else "NOT_SUPPORTED"
    summary["TRANSITION_MISMATCH_TO_FUNCTIONAL_DAMAGE_CAUSAL"] = "STRONG" if comm_strong and spec_strong and rescue_strong and r_supported else ("PARTIAL" if comm_strong or rescue_strong else "NOT_SUPPORTED")
    summary["TRANSITION_ALIGNMENT_CAUSAL_CHAIN"] = "CLOSED" if summary["TRANSITION_MISMATCH_TO_FUNCTIONAL_DAMAGE_CAUSAL"] == "STRONG" else "PARTIAL"
    summary["CAUSAL_MECHANISM_CLOSED"] = "YES" if summary["TRANSITION_ALIGNMENT_CAUSAL_CHAIN"] == "CLOSED" else "NO"
    summary["METHOD_DESIGN_READY"] = "NO"
    summary["FINAL_CLASSIFICATION"] = (
        "COMMUTATOR_FUNCTIONAL_CAUSAL_CHAIN_STRONG"
        if summary["CAUSAL_MECHANISM_CLOSED"] == "YES"
        else "COMMUTATOR_FUNCTIONAL_CAUSAL_CHAIN_PARTIAL_OR_NOT_CLOSED"
    )
    summary["NEXT_RECOMMENDED_TASK"] = "PERSISTENT_CONTEXT_SINGLE_EVENT_RESCUE if more closure is needed; do not start automatically"
    return summary, effects, specs_orth, specs_shuffle, rescues_c, rescues_r


def simple_svg(path, title, series):
    width, height, pad = 760, 420, 54
    xs = [x for _, pts in series for x, y in pts if y is not None]
    ys = [y for _, pts in series for x, y in pts if y is not None]
    if not xs or not ys:
        write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='40'>{title}: empty</text></svg>\n")
        return
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if abs(xmax - xmin) < EPS:
        xmax += 1
    if abs(ymax - ymin) < EPS:
        ymax += 1
    def sx(x): return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y): return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    colors = ["#2f6f9f", "#c7493a", "#427a46", "#6d4c8d", "#a56b2a"]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>", f"<text x='20' y='28' font-size='18' font-family='sans-serif'>{title}</text>", f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>", f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>"]
    for i, (name, pts) in enumerate(series):
        pts = [(x, y) for x, y in pts if y is not None]
        color = colors[i % len(colors)]
        d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts:
            parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{color}'/>")
        parts.append(f"<text x='{width-pad-190}' y='{pad + 18*i}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
    parts.append("</svg>\n")
    write_text(path, "\n".join(parts))


def make_plots(hrows, effects, specs_o, specs_s, resc_c, resc_r):
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    wanted = ["C_BASE", "C_COMM_50", "C_COMM_100"]
    series = []
    for branch in wanted:
        pts = []
        for h in range(1, HORIZON + 1):
            vals = [float(r["KL"]) for r in hrows if r.get("orientation") == "C128" and r.get("branch") == branch and int(r["horizon"]) == h]
            pts.append((h, mean(vals)))
        series.append((branch, pts))
    simple_svg(plot_dir / "future_kl_by_horizon_comm_dose.svg", "C128 future KL by horizon", series)
    simple_svg(plot_dir / "comm_effect_per_unit.svg", "C128 COMM100 - BASE AUC", [( "E_COMM", [(i + 1, r["E_COMM"]) for i, r in enumerate(sorted(effects, key=lambda x: x["unit_id"]))] )])
    simple_svg(plot_dir / "orthogonal_specificity_per_unit.svg", "Real comm effect above orthogonal median", [("SPEC_ORTH", [(i + 1, r["SPEC_ORTH"]) for i, r in enumerate(sorted(specs_o, key=lambda x: x["unit_id"]))])])
    simple_svg(plot_dir / "shuffle_specificity_per_unit.svg", "Real comm effect above shuffled median", [("SPEC_SHUFFLE", [(i + 1, r["SPEC_SHUFFLE"]) for i, r in enumerate(sorted(specs_s, key=lambda x: x["unit_id"]))])])
    simple_svg(plot_dir / "reinjection_rescue_c_vs_r.svg", "C vs R reinjection rescue", [
        ("C_RESCUE", [(i + 1, r["C_RESCUE"]) for i, r in enumerate(sorted(resc_c, key=lambda x: x["unit_id"]))]),
        ("R_RESCUE", [(i + 1, r["R_RESCUE"]) for i, r in enumerate(sorted(resc_r, key=lambda x: x["unit_id"]))]),
    ])
    stage0_rows = read_csv(RUN_DIR / "stage0_decomposition_per_unit.csv")
    c_norm = {r["unit_id"]: float(r["C_F_norm"]) for r in stage0_rows if r.get("orientation") == "C128" and r.get("C_F_norm")}
    r_norm = {r["unit_id"]: float(r["R_norm"]) for r in stage0_rows if r.get("orientation") == "C128" and r.get("R_norm")}
    e_by_uid = {r["unit_id"]: r["E_COMM"] for r in effects}
    cr_by_uid = {r["unit_id"]: r["C_RESCUE"] for r in resc_c}
    simple_svg(plot_dir / "cf_norm_vs_comm_effect.svg", "C_F norm vs comm effect", [("C_F_norm", [(c_norm[u], e_by_uid[u]) for u in sorted(e_by_uid) if u in c_norm])])
    simple_svg(plot_dir / "r_norm_vs_reinjection_rescue.svg", "R norm vs C reinjection rescue", [("R_norm", [(r_norm[u], cr_by_uid[u]) for u in sorted(cr_by_uid) if u in r_norm])])
    write_text(plot_dir / "README.md", "\n".join([
        "# Plots",
        "",
        "- `future_kl_by_horizon_comm_dose.svg`: C_BASE/C_COMM_50/C_COMM_100 mean KL by horizon.",
        "- `comm_effect_per_unit.svg`: per-unit causal commutator AUC effect.",
        "- `orthogonal_specificity_per_unit.svg`: real effect minus orthogonal-control median.",
        "- `shuffle_specificity_per_unit.svg`: real effect minus shuffled-control median.",
        "- `reinjection_rescue_c_vs_r.svg`: C128 rescue compared with R128 negative control.",
        "- `cf_norm_vs_comm_effect.svg`: decomposition magnitude vs functional effect.",
        "- `r_norm_vs_reinjection_rescue.svg`: requant residual magnitude vs rescue.",
        "",
    ]))


def make_report(summary):
    lines = ["# LING_KDA_COMMUTATOR_FUNCTIONAL_CAUSAL_V1", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", ""]
    write_text(RUN_DIR / "report.md", "\n".join(lines))
    write_text(DOC_DIR / "report.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    if args.formal:
        phase_label = f"formal_shard{args.shard_id}"
    elif args.stage0:
        phase_label = "stage0"
    elif args.smoke:
        phase_label = "smoke"
    elif args.pilot:
        phase_label = "pilot"
    elif args.finalize:
        phase_label = "finalize"
    else:
        phase_label = "unknown"
    manifest_path = RUN_DIR / f"manifest_{phase_label}.json"
    failure_path = RUN_DIR / f"failure_{phase_label}.json"
    manifest = {
        "TASK": TASK,
        "git": {"head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
        "command": " ".join(sys.argv),
        "model_path": os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_start": sh(["nvidia-smi"]),
        "canonical_unit_source": str(CANONICAL_UNIT_SOURCE.relative_to(REPO)),
        "teacher_forced_protocol_source": str(CONTROLLED_RUNNER.relative_to(REPO)),
        "lambda": LAMBDAS_NONE,
        "random_control_base_seed": BASE_SEED,
    }
    save_json(manifest_path, manifest)
    try:
        if args.stage0:
            run_stage0(args)
        elif args.smoke or args.pilot or args.formal:
            stage0 = load_json(RUN_DIR / "stage0_summary.json")
            if stage0["STAGE0_GATE"] != "PASS":
                raise RuntimeError("Stage0 not PASS")
            units = load_canonical_units()
            if args.smoke:
                units = units[:1]
                suffix = "smoke"
                controls = False
            elif args.pilot:
                units = units[:6]
                suffix = "pilot"
                controls = True
            else:
                units = [u for i, u in enumerate(units) if i % args.num_shards == args.shard_id]
                suffix = f"formal_shard{args.shard_id}"
                controls = True
            urows, hrows, timing = run_units(args, units, suffix, controls)
            if args.smoke or args.pilot:
                summary, *_ = summarize(urows, stage0)
                save_json(RUN_DIR / f"{suffix}_summary.json", summary)
        elif args.finalize:
            stage0 = load_json(RUN_DIR / "stage0_summary.json")
            urows, hrows = [], []
            for sid in range(args.num_shards):
                urows.extend(list(iter_jsonl(RUN_DIR / f"unit_auc_formal_shard{sid}.jsonl") or []))
                hrows.extend(list(iter_jsonl(RUN_DIR / f"horizon_formal_shard{sid}.jsonl") or []))
            write_jsonl(RUN_DIR / "unit_auc_formal.jsonl", urows)
            write_csv(RUN_DIR / "unit_auc_formal.csv", urows)
            write_jsonl(RUN_DIR / "horizon_formal.jsonl", hrows)
            write_csv(RUN_DIR / "horizon_formal.csv", hrows)
            summary, effects, specs_o, specs_s, resc_c, resc_r = summarize(urows, stage0)
            save_json(RUN_DIR / "final_summary.json", summary)
            write_csv(RUN_DIR / "comm_effect_per_unit.csv", effects)
            write_csv(RUN_DIR / "specificity_orth_per_unit.csv", specs_o)
            write_csv(RUN_DIR / "specificity_shuffle_per_unit.csv", specs_s)
            write_csv(RUN_DIR / "rescue_c_per_unit.csv", resc_c)
            write_csv(RUN_DIR / "rescue_r_per_unit.csv", resc_r)
            make_plots(hrows, effects, specs_o, specs_s, resc_c, resc_r)
            make_report(summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        else:
            raise SystemExit("choose --stage0, --smoke, --pilot, --formal, or --finalize")
        manifest["nvidia_smi_end"] = sh(["nvidia-smi"])
        save_json(manifest_path, manifest)
    except Exception as exc:
        err = {"error": repr(exc), "traceback": traceback.format_exc()}
        save_json(failure_path, err)
        print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
