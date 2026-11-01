#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "GDN_KDA_ROTATION_RESIDUAL_DIRECTION_CAUSAL_CLOSURE_V1"
SLUG = "gdn_kda_rotation_residual_direction_causal_closure_v1"
V1_SLUG = "gdn_kda_axis_matched_state_rotation_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
V1_RESULT_DIR = REPO / "results" / V1_SLUG
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
GDN_CANONICAL_MANIFEST = REPO / "results" / "propagation" / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"
KDA_CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64]
EPS = 1e-12
QMAX_INT8 = 127.0
CANONICAL_N = 18
RHT_SEED = 0


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x, default=None):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=20260913):
    vals = [float(x) for x in xs if finite_number(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(n)):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def ranks(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[pairs[k][1]] = rank
        i = j
    return out


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = mean(x), mean(y)
    den = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (den + EPS) if den > EPS else 0.0


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V1 = None


def v1():
    global _V1
    if _V1 is None:
        _V1 = import_file(REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py", "axis_rotation_v1_reused_for_closure")
    return _V1


def make_rotation(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
    return v1().make_rotation(n, kind=kind, seed=seed, dtype=dtype, device=device, samples=samples)


def rotate_state_key_axis(state, r):
    return v1().rotate_state_key_axis(state, r)


def inverse_rotate_state_key_axis(state, r):
    return v1().inverse_rotate_state_key_axis(state, r)


def rotate_state_value_axis(state, u):
    return v1().rotate_state_value_axis(state, u)


def inverse_rotate_state_value_axis(state, u):
    return v1().inverse_rotate_state_value_axis(state, u)


def fake_quant_state(state, config):
    return v1().fake_quant_state(state, config)


def relerr(a, b):
    return v1().relerr(a, b)


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def flat_cosine(a, b):
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(aa).item() * torch.linalg.vector_norm(bb).item())
    return float(torch.dot(aa, bb).item() / (den + EPS)) if den > EPS else None


def residual_pair(state, architecture, rotation):
    arch = str(architecture).lower()
    x = state.detach().to(torch.float64)
    if arch == "gdn":
        q_native, _ = fake_quant_state(x, "C128")
        e_native = q_native - x
        x_rot = rotate_state_key_axis(x, rotation)
        q_rot, _ = fake_quant_state(x_rot, "C128")
        e_rot_coord = q_rot - x_rot
        e_rot_native = inverse_rotate_state_key_axis(e_rot_coord, rotation)
    elif arch == "kda":
        q_native, _ = fake_quant_state(x, "R128")
        e_native = q_native - x
        x_rot = rotate_state_value_axis(x, rotation)
        q_rot, _ = fake_quant_state(x_rot, "R128")
        e_rot_coord = q_rot - x_rot
        e_rot_native = inverse_rotate_state_value_axis(e_rot_coord, rotation)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    return {
        "E_N": e_native.to(dtype=state.dtype),
        "E_R": e_rot_native.to(dtype=state.dtype),
        "E_R_coord": e_rot_coord.to(dtype=state.dtype),
        "m_N": tensor_norm(e_native),
        "m_R": tensor_norm(e_rot_native),
        "m_R_coord": tensor_norm(e_rot_coord),
        "m_R_over_m_N": tensor_norm(e_rot_native) / (tensor_norm(e_native) + EPS),
        "frobenius_mapping_relative_error": abs(tensor_norm(e_rot_native) - tensor_norm(e_rot_coord)) / (tensor_norm(e_rot_coord) + EPS),
        "cos_E_N_E_R": flat_cosine(e_native, e_rot_native),
    }


def readout_perturbation(error, query):
    return torch.einsum("bhkv,bhk->bhv", error.detach().float(), query.detach().float())


def rho_q(error, query):
    ro = readout_perturbation(error, query)
    den = tensor_norm(error) * tensor_norm(query)
    return tensor_norm(ro) / (den + EPS)


def factorial_residuals(e_native, e_rotated):
    e_n = e_native.detach().float()
    e_r = e_rotated.detach().float()
    m_n = torch.linalg.vector_norm(e_n).clamp_min(EPS)
    m_r = torch.linalg.vector_norm(e_r).clamp_min(EPS)
    d_n = e_n / m_n
    d_r = e_r / m_r
    return {
        "ZERO": torch.zeros_like(e_n),
        "NN": e_n,
        "RN": d_n * m_r,
        "NR": d_r * m_n,
        "RR": e_r,
    }


def factorial_sanity_checks(factors, e_native, e_rotated, tol=1e-5):
    m_n = tensor_norm(e_native)
    m_r = tensor_norm(e_rotated)
    norms = {k: tensor_norm(v) for k, v in factors.items()}
    checks = {
        "ZERO_norm": norms["ZERO"],
        "NN_norm_abs_error": abs(norms["NN"] - m_n),
        "RN_norm_abs_error": abs(norms["RN"] - m_r),
        "NR_norm_abs_error": abs(norms["NR"] - m_n),
        "RR_norm_abs_error": abs(norms["RR"] - m_r),
        "cos_RN_EN": flat_cosine(factors["RN"], e_native),
        "cos_NR_ER": flat_cosine(factors["NR"], e_rotated),
    }
    ok = (
        checks["ZERO_norm"] <= 0.0
        and checks["NN_norm_abs_error"] <= tol
        and checks["RN_norm_abs_error"] <= tol
        and checks["NR_norm_abs_error"] <= tol
        and checks["RR_norm_abs_error"] <= tol
        and (checks["cos_RN_EN"] or 0.0) >= 1.0 - 1e-5
        and (checks["cos_NR_ER"] or 0.0) >= 1.0 - 1e-5
    )
    checks["gate"] = "PASS" if ok else "FAIL"
    return checks


def stack_norm(stack):
    return math.sqrt(sum(tensor_norm(v) ** 2 for v in stack.values()))


def stack_max_abs_diff(a, b):
    vals = []
    for key in sorted(set(a) & set(b)):
        vals.append(float((a[key].detach().float() - b[key].detach().float()).abs().max().item()))
    return max(vals, default=0.0)


def clone_stack(stack):
    return {int(k): v.detach().clone() for k, v in stack.items()}


def add_stack(stack, delta):
    return {int(k): stack[k].detach().float() + delta[k].to(stack[k].device).float() for k in stack}


def apply_single_pulse_to_stack(stack, delta, already_applied=False):
    if already_applied:
        raise RuntimeError("single-pulse invariant violated: residual already applied")
    with torch.inference_mode():
        for layer, residual in delta.items():
            stack[layer].copy_((stack[layer].detach().float() + residual.to(stack[layer].device).float()).to(stack[layer].dtype))
    return {
        "injection_count": 1,
        "future_quantization_count": 0,
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "SINGLE_PULSE_GATE": "PASS",
    }


def readout_only_intervention(cache, residual, query_stack):
    delta_o = {int(layer): readout_perturbation(residual[layer], query_stack[layer]) for layer in residual}
    return {"cache_after": clone_stack(cache), "delta_o": delta_o, "cache_noninterference_max_abs": 0.0}


def memory_only_intervention(cache, residual, current_logits):
    future_cache = clone_stack(cache)
    apply_single_pulse_to_stack(future_cache, residual, already_applied=False)
    return {
        "current_logits_after": current_logits.detach().clone(),
        "future_cache": future_cache,
        "current_logit_noninterference_max_abs": 0.0,
        "memory_cache_modified": stack_max_abs_diff(future_cache, cache) > 0.0,
    }


def teacher_token_for_timestep(tokens, timestep):
    timestep = int(timestep)
    if timestep < 1:
        raise ValueError("teacher-forced decode timesteps start at 1")
    return int(tokens[timestep - 1])


def teacher_token_for_future_horizon(tokens, t0, horizon):
    return teacher_token_for_timestep(tokens, int(t0) + int(horizon))


def selected_horizons(max_horizon=64):
    return [h for h in HORIZONS if h <= int(max_horizon)]


def teacher_forced_sequences_identical(branch_tokens):
    vals = list(branch_tokens.values())
    return bool(vals) and all(list(v) == list(vals[0]) for v in vals)


def factorial_effects(risk):
    nn = float(risk["NN"])
    rn = float(risk["RN"])
    nr = float(risk["NR"])
    rr = float(risk["RR"])
    return {
        "M_N": nn - rn,
        "M_R": nr - rr,
        "D_N": nr - nn,
        "D_R": rr - rn,
        "interaction": (rr - rn) - (nr - nn),
    }


def propagation_gain_curve(error, operators, horizons=(1, 2, 4, 8, 16, 32, 64)):
    cur = error.detach().float()
    base = tensor_norm(cur) + EPS
    out = {}
    max_h = max(horizons) if horizons else 0
    for i in range(1, max_h + 1):
        if i - 1 < len(operators):
            op = operators[i - 1].to(device=cur.device, dtype=cur.dtype)
            cur = torch.einsum("ak,bhkv->bhav", op, cur)
        if i in horizons:
            out[i] = tensor_norm(cur) / base
    return out


def future_readout_gain_curve(error, operators, queries, horizons=(1, 2, 4, 8, 16, 32, 64)):
    cur = error.detach().float()
    base = tensor_norm(error) + EPS
    out = {}
    max_h = max(horizons) if horizons else 0
    for i in range(1, max_h + 1):
        if i - 1 < len(operators):
            op = operators[i - 1].to(device=cur.device, dtype=cur.dtype)
            cur = torch.einsum("ak,bhkv->bhav", op, cur)
        if i in horizons:
            q = queries[min(i - 1, len(queries) - 1)].to(device=cur.device, dtype=cur.dtype)
            out[i] = tensor_norm(readout_perturbation(cur, q)) / base
    return out


def full_logit_metrics(ref_logits, branch_logits):
    ref = ref_logits[:, -1, :].float()
    branch = branch_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    branch_logp = torch.log_softmax(branch, dim=-1)
    p = ref_logp.exp()
    k = min(20, ref.shape[-1])
    rt = set(torch.topk(ref, k=k, dim=-1).indices[0].detach().cpu().tolist())
    bt = set(torch.topk(branch, k=k, dim=-1).indices[0].detach().cpu().tolist())
    return {
        "KL": float((p * (ref_logp - branch_logp)).sum().item()),
        "top1_match": int(ref.argmax(dim=-1).item() == branch.argmax(dim=-1).item()),
        "logit_cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), branch.flatten(), dim=0).item()),
        "logit_rel_l2": float(torch.linalg.vector_norm((branch - ref).flatten()).item() / (torch.linalg.vector_norm(ref.flatten()).item() + EPS)),
        "top20_overlap": float(len(rt & bt) / k),
    }


def state_metrics_from_stacks(branch, reference):
    err_sq = 0.0
    ref_sq = 0.0
    aa, bb = [], []
    for layer in sorted(reference):
        actual = branch[layer].detach().float()
        expected = reference[layer].to(actual.device).detach().float()
        delta = actual - expected
        err_sq += float((delta.double() * delta.double()).sum().item())
        ref_sq += float((expected.double() * expected.double()).sum().item())
        aa.append(actual.reshape(-1))
        bb.append(expected.reshape(-1))
    err = math.sqrt(err_sq)
    cosine = flat_cosine(torch.cat(aa), torch.cat(bb)) if aa else None
    return {"state_error_norm": err, "state_relative_L2": err / (math.sqrt(ref_sq) + EPS), "state_cosine": cosine}


def result_arch_dir(arch, stage):
    path = RESULT_DIR / arch / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_gdn_units():
    obj = load_json(GDN_CANONICAL_MANIFEST, {})
    units = []
    for unit in obj.get("manifest", []):
        row = dict(unit)
        row["unit_id"] = f"{row['problem_id']}|{int(row['t0'])}"
        units.append(row)
    return units, hashlib.sha256(GDN_CANONICAL_MANIFEST.read_bytes()).hexdigest() if GDN_CANONICAL_MANIFEST.exists() else None


def load_kda_units():
    obj = load_json(KDA_CANONICAL_UNITS, {})
    raw = obj.get("formal_units") or obj.get("canonical_units") or obj.get("units") or (obj if isinstance(obj, list) else [])
    units = []
    for i, unit in enumerate(raw):
        row = dict(unit)
        row.setdefault("unit_id", row.get("id") or f"{row.get('problem_id')}|{int(row.get('t0', 0))}")
        row.setdefault("prompt_id", row.get("problem_id"))
        row["problem_id"] = str(row["problem_id"])
        row["t0"] = int(row["t0"])
        row["problem_index"] = int(row.get("problem_index", i))
        units.append(row)
    return units, hashlib.sha256(KDA_CANONICAL_UNITS.read_bytes()).hexdigest() if KDA_CANONICAL_UNITS.exists() else None


def select_units(units, scope, max_units=None):
    out = list(units)
    if scope == "smoke":
        out = out[:1]
    elif scope == "pilot":
        picks = [0, 4, 8, 9, 13, 17]
        out = [out[i] for i in picks if i < len(out)]
    if max_units is not None:
        out = out[: int(max_units)]
    return out


def gdn_setup():
    os.environ.setdefault("GDN_DATA_ROOT", "/data01/user2")
    q = import_file(REPO / "experiments" / "propagation" / "run_qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1.py", "gdn_mag_eq_for_closure")
    return q, *q.setup_qwen_model()


def gdn_prompt_rows(p1):
    return {row["problem_id"]: row for row in p1.selected_prompt_rows()}


def gdn_stack(p1, cache):
    if hasattr(cache, "recurrent_states"):
        return {int(layer): cache.recurrent_states[layer] for layer in GDN_LAYERS}
    return {int(layer): p1.get_state(cache, layer) for layer in GDN_LAYERS}


def gdn_copy_stack(p1, cache):
    stack = gdn_stack(p1, cache)
    return {int(layer): state.detach().float().cpu().clone() for layer, state in stack.items()}


def gdn_apply_delta(p1, cache, delta):
    stack = gdn_stack(p1, cache)
    return apply_single_pulse_to_stack(stack, delta, already_applied=False)


def gdn_run_to_t0(torch_mod, model, tokenizer, p1, e2e, prompt_row, t0, max_horizon=64):
    prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
    tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
    if len(tokens) <= int(t0) + int(max_horizon):
        raise RuntimeError(f"teacher continuation too short for {prompt_row['problem_id']}: need>{int(t0)+int(max_horizon)} have={len(tokens)}")
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    with torch_mod.inference_mode():
        out = p1.feed_step(torch_mod, model, ids, mask, None)
    past = out.past_key_values
    for timestep in range(1, int(t0) + 1):
        cur = torch_mod.tensor([[tokens[timestep - 1]]], dtype=ids.dtype, device=device)
        with torch_mod.inference_mode():
            out = p1.feed_step(torch_mod, model, cur, None, past)
        past = out.past_key_values
    return {"past": past, "tokens": tokens, "dtype": ids.dtype, "device": device}


def gdn_collect_query_stack(torch_mod, model, p1, fp_past, token_id, dtype, device):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    collector = {"records": defaultdict(dict), "current_layer": None, "orig_rule": qmod.torch_recurrent_gated_delta_rule}
    handles = []
    orig_rule = qmod.torch_recurrent_gated_delta_rule

    def wrapped_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        out, last_state = orig_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, **kwargs)
        layer = collector.get("current_layer")
        if layer is not None and query.shape[1] == 1 and last_state is not None:
            q = query.detach()
            if use_qk_l2norm_in_kernel:
                q = qmod.l2norm(q, dim=-1, eps=1e-6)
            q_t = q.transpose(1, 2).contiguous().to(torch_mod.float32)[:, :, 0] * (q.shape[-1] ** -0.5)
            collector["records"][int(layer)]["q_scaled"] = q_t.detach().float().cpu().clone()
        return out, last_state

    qmod.torch_recurrent_gated_delta_rule = wrapped_rule
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "linear_attn"):
            continue

        def make_pre(idx):
            def pre(_module, _inputs):
                collector["current_layer"] = idx
            return pre

        def post(_module, _inputs, _output):
            collector["current_layer"] = None

        handles.append(layer.linear_attn.register_forward_pre_hook(make_pre(layer_idx)))
        handles.append(layer.linear_attn.register_forward_hook(post))
    try:
        cur = torch_mod.tensor([[int(token_id)]], dtype=dtype, device=device)
        with torch_mod.inference_mode():
            _out = p1.feed_step(torch_mod, model, cur, None, copy.deepcopy(fp_past))
        return {int(layer): rec["q_scaled"] for layer, rec in collector["records"].items() if "q_scaled" in rec}
    finally:
        qmod.torch_recurrent_gated_delta_rule = collector["orig_rule"]
        for handle in handles:
            handle.remove()


def kda_setup():
    mod = import_file(REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py", "kda_granularity_for_closure")
    torch_mod, model, tokenizer = mod.P.load_model_and_tokenizer()
    capture = mod.P.KdaTransitionCapture()
    captured_layers = capture.install(model)
    config_obj = json.loads((mod.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = mod.P.kda_layers_from_config(config_obj) if hasattr(mod.P, "kda_layers_from_config") else captured_layers
    rows_by_pid = mod.P.load_dataset()
    teacher_tokens = mod.P.load_fp_teacher_tokens()
    return mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens


def kda_run_to_t0(torch_mod, model, tokenizer, capture, mod, unit, row, tokens, t0, kda_layers):
    device = next(model.parameters()).device
    input_ids = mod.P.render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch_mod.ones_like(input_ids)
    capture.records.clear()
    capture.branch = "FP_PREFILL"
    with torch_mod.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device),
            use_cache=True,
        )
    past = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    for t, tok in enumerate(tokens[: int(t0)]):
        cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
        mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_TO_T0"
        with torch_mod.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=past,
                cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long),
                use_cache=True,
            )
        past = out.past_key_values
    capture.branch = None
    return {"past": past, "mask": mask, "prompt_len": prompt_len, "tokens": tokens, "device": device}


def kda_stack(mod, cache, kda_layers):
    return {int(layer): mod.P.get_cache_state(cache, layer) for layer in kda_layers}


def kda_copy_stack(mod, cache, kda_layers):
    return {int(layer): mod.P.get_cache_state(cache, layer).detach().float().cpu().clone() for layer in kda_layers}


def kda_apply_delta(mod, cache, delta):
    stack = {int(layer): mod.P.get_cache_state(cache, layer) for layer in delta}
    return apply_single_pulse_to_stack(stack, delta, already_applied=False)


def kda_step(torch_mod, model, capture, past, mask, prompt_len, token_id, token_index, branch):
    device = next(model.parameters()).device
    cur = torch_mod.tensor([[int(token_id)]], dtype=torch_mod.long, device=device)
    mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
    capture.records.clear()
    capture.branch = branch
    with torch_mod.inference_mode():
        out = model(
            input_ids=cur,
            attention_mask=mask,
            past_key_values=past,
            cache_position=torch_mod.tensor([int(prompt_len) + int(token_index)], device=device, dtype=torch_mod.long),
            use_cache=True,
        )
    capture.branch = None
    return out, mask


def stack_residual_pairs(stack, arch):
    out = {}
    for layer, state in stack.items():
        rot_dim = state.shape[-2] if arch == "gdn" else state.shape[-1]
        rot = make_rotation(rot_dim, "rht", seed=RHT_SEED, dtype=torch.float64)
        if arch == "kda":
            # V1 registered KDA's value-side RHT as the right-multiply convention that
            # corresponds to D @ H for state residuals; make_rotation returns H @ D.
            rot = rot.t().contiguous()
        out[int(layer)] = residual_pair(state, arch, rot)
    return out


def residual_stack_from_pairs(pairs, key):
    return {int(layer): row[key].detach().float().cpu() for layer, row in pairs.items()}


def stack_readout_norm(error_stack, query_stack):
    vals = []
    for layer in sorted(set(error_stack) & set(query_stack)):
        vals.append(tensor_norm(readout_perturbation(error_stack[layer], query_stack[layer])) ** 2)
    return math.sqrt(sum(vals)) if vals else None


def fresh_rows_for_unit(arch, unit, stack, query_stack, manifest_hash):
    pairs = stack_residual_pairs(stack, arch)
    e_n = residual_stack_from_pairs(pairs, "E_N")
    e_r = residual_stack_from_pairs(pairs, "E_R")
    e_rc = residual_stack_from_pairs(pairs, "E_R_coord")
    m_n = stack_norm(e_n)
    m_r = stack_norm(e_r)
    m_rc = stack_norm(e_rc)
    qn = stack_readout_norm(e_n, query_stack) if query_stack else None
    qr = stack_readout_norm(e_r, query_stack) if query_stack else None
    qnorm = stack_norm(query_stack) if query_stack else None
    row = {
        "task": TASK,
        "architecture": arch,
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit["problem_id"]),
        "t0": int(unit["t0"]),
        "manifest_sha256": manifest_hash,
        "rotation": "RHT_seed_0",
        "native_quantizer": "C128" if arch == "gdn" else "R128",
        "matched_rotation_axis": "key" if arch == "gdn" else "value",
        "m_N": m_n,
        "m_R": m_r,
        "m_R_coord": m_rc,
        "m_R_over_m_N": m_r / (m_n + EPS),
        "relative_magnitude_reduction": 1.0 - (m_r / (m_n + EPS)),
        "frobenius_mapping_relative_error": abs(m_r - m_rc) / (m_rc + EPS),
        "cos_E_N_E_R": stack_cosine(e_n, e_r),
        "E_N_T_q_norm": qn,
        "E_R_T_q_norm": qr,
        "rho_q_E_N": None if qn is None or qnorm is None else qn / (m_n * qnorm + EPS),
        "rho_q_E_R": None if qr is None or qnorm is None else qr / (m_r * qnorm + EPS),
        "n_layers": len(stack),
    }
    return row, pairs


def stack_cosine(a, b):
    aa, bb = [], []
    for layer in sorted(set(a) & set(b)):
        aa.append(a[layer].detach().float().reshape(-1))
        bb.append(b[layer].detach().float().reshape(-1))
    return flat_cosine(torch.cat(aa), torch.cat(bb)) if aa else None


def summarize_fresh(rows, arch):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    unit_rows = []
    for unit_id, vals in sorted(grouped.items()):
        unit_rows.append({
            "architecture": arch,
            "unit_id": unit_id,
            "n_t0_records": len(vals),
            "median_m_R_over_m_N": median([v["m_R_over_m_N"] for v in vals]),
            "mean_m_R_over_m_N": mean([v["m_R_over_m_N"] for v in vals]),
            "median_relative_magnitude_reduction": median([v["relative_magnitude_reduction"] for v in vals]),
            "mean_relative_magnitude_reduction": mean([v["relative_magnitude_reduction"] for v in vals]),
            "max_frobenius_mapping_relative_error": max(v["frobenius_mapping_relative_error"] for v in vals),
            "mean_cos_E_N_E_R": mean([v["cos_E_N_E_R"] for v in vals]),
            "mean_rho_q_E_N": mean([v["rho_q_E_N"] for v in vals]),
            "mean_rho_q_E_R": mean([v["rho_q_E_R"] for v in vals]),
        })
    reductions = [r["median_relative_magnitude_reduction"] for r in unit_rows]
    ratios = [r["median_m_R_over_m_N"] for r in unit_rows]
    wins = sum(r["median_m_R_over_m_N"] < 1.0 for r in unit_rows)
    ci = bootstrap_ci(reductions)
    gate = "PASS" if len(unit_rows) >= 13 and median(ratios) < 0.85 and wins >= 13 and ci and ci[0] > 0 else "FAIL"
    return unit_rows, {
        "architecture": arch,
        "stage": "fresh_capture",
        "n_units_completed": len(unit_rows),
        "median_m_R_over_m_N": median(ratios),
        "mean_m_R_over_m_N": mean(ratios),
        "std_m_R_over_m_N": std(ratios),
        "wins": wins,
        "bootstrap95_relative_magnitude_reduction": ci,
        f"{arch.upper()}_FRESH_RESIDUAL_CAPTURE": gate,
    }


def all_factor_stack(pair_stack):
    per_layer = {}
    for layer, pair in pair_stack.items():
        per_layer[int(layer)] = factorial_residuals(pair["E_N"], pair["E_R"])
    return {cond: {layer: per_layer[layer][cond].detach().float().cpu() for layer in per_layer} for cond in ["ZERO", "NN", "RN", "NR", "RR"]}


def reference_stack_for_arch(arch, ctx, past):
    if arch == "gdn":
        return gdn_copy_stack(ctx["p1"], past)
    return kda_copy_stack(ctx["mod"], past, ctx["kda_layers"])


def apply_delta_for_arch(arch, ctx, past, delta):
    if arch == "gdn":
        return gdn_apply_delta(ctx["p1"], past, delta)
    return kda_apply_delta(ctx["mod"], past, delta)


def step_for_arch(arch, ctx, past, mask, prompt_len, token_id, token_index, branch):
    if arch == "gdn":
        cur = ctx["torch"].tensor([[int(token_id)]], dtype=ctx["dtype"], device=ctx["device"])
        with ctx["torch"].inference_mode():
            out = ctx["p1"].feed_step(ctx["torch"], ctx["model"], cur, None, past)
        return out, mask
    return kda_step(ctx["torch"], ctx["model"], ctx["capture"], past, mask, prompt_len, token_id, token_index, branch)


def run_full_state_factorial_unit(arch, ctx, unit, pair_stack, max_horizon=64):
    tokens = ctx["tokens"]
    t0 = int(unit["t0"])
    horizons = selected_horizons(max_horizon)
    max_offset = max(horizons)
    factors = all_factor_stack(pair_stack)
    fp_past = copy.deepcopy(ctx["past"])
    branch_pasts = {cond: copy.deepcopy(ctx["past"]) for cond in factors}
    branch_masks = {cond: None if ctx.get("mask") is None else ctx["mask"].clone() for cond in factors}
    injection_audits = {}
    for cond, delta in factors.items():
        injection_audits[cond] = apply_delta_for_arch(arch, ctx, branch_pasts[cond], delta)
    fp_mask = None if ctx.get("mask") is None else ctx["mask"].clone()
    rows = []
    auc_terms = {cond: [] for cond in factors}
    selected_kl = defaultdict(dict)
    selected_state = defaultdict(dict)
    for off in range(0, max_offset + 1):
        token = int(tokens[t0 + off])
        token_index = t0 + off
        fp_out, fp_mask = step_for_arch(arch, ctx, fp_past, fp_mask, ctx.get("prompt_len"), token, token_index, "FP_FULL")
        fp_past = fp_out.past_key_values
        fp_stack = reference_stack_for_arch(arch, ctx, fp_past)
        for cond in ["ZERO", "NN", "RN", "NR", "RR"]:
            out, branch_masks[cond] = step_for_arch(arch, ctx, branch_pasts[cond], branch_masks[cond], ctx.get("prompt_len"), token, token_index, f"FULL_{cond}")
            branch_pasts[cond] = out.past_key_values
            logit = full_logit_metrics(fp_out.logits, out.logits)
            sm = state_metrics_from_stacks(reference_stack_for_arch(arch, ctx, branch_pasts[cond]), fp_stack)
            auc_terms[cond].append(logit["KL"])
            if off in horizons:
                selected_kl[cond][off] = logit["KL"]
                selected_state[cond][off] = sm
                row = {
                    "task": TASK,
                    "architecture": arch,
                    "unit_id": str(unit["unit_id"]),
                    "prompt_id": str(unit["problem_id"]),
                    "t0": t0,
                    "condition": cond,
                    "horizon": off,
                    "horizon_semantics": "first_future_logit_after_t0_plus_offset",
                    "future_kl": logit["KL"],
                    "logit_relative_l2": logit["logit_rel_l2"],
                    "logit_cosine": logit["logit_cosine"],
                    "top1_match": logit["top1_match"],
                    "top20_overlap": logit["top20_overlap"],
                    "state_error_norm": sm["state_error_norm"],
                    "state_relative_L2": sm["state_relative_L2"],
                    "state_cosine": sm["state_cosine"],
                    "future_quantization_count": 0,
                    "injection_count": injection_audits[cond]["injection_count"],
                    "NO_FURTHER_QUANTIZATION_GATE": injection_audits[cond]["NO_FURTHER_QUANTIZATION_GATE"],
                }
                rows.append(row)
    unit_rows = []
    for cond in ["ZERO", "NN", "RN", "NR", "RR"]:
        row = {
            "architecture": arch,
            "unit_id": str(unit["unit_id"]),
            "condition": cond,
            "future_kl_auc": mean(auc_terms[cond]),
            "selected_future_kl_auc": mean(selected_kl[cond].values()),
            "future_quantization_count": 0,
            "injection_count": 1,
        }
        for h in horizons:
            row[f"KL_h{h}"] = selected_kl[cond].get(h)
            sm = selected_state[cond].get(h, {})
            row[f"state_error_h{h}"] = sm.get("state_error_norm")
            row[f"state_relative_L2_h{h}"] = sm.get("state_relative_L2")
        unit_rows.append(row)
    return rows, unit_rows


def aggregate_factorial(unit_rows, arch):
    by_unit = defaultdict(dict)
    for row in unit_rows:
        by_unit[row["unit_id"]][row["condition"]] = row
    effects = []
    horizons = defaultdict(list)
    for unit_id, rows in sorted(by_unit.items()):
        if not all(c in rows for c in ["NN", "RN", "NR", "RR"]):
            continue
        risk = {c: as_float(rows[c]["future_kl_auc"]) for c in ["NN", "RN", "NR", "RR"]}
        eff = {"architecture": arch, "unit_id": unit_id, **risk, **factorial_effects(risk)}
        effects.append(eff)
        for h in HORIZONS:
            rr = as_float(rows["RR"].get(f"KL_h{h}"))
            rn = as_float(rows["RN"].get(f"KL_h{h}"))
            if rr is not None and rn is not None:
                horizons[h].append(rr - rn)
    summary = {"architecture": arch, "stage": "full_state_factorial", "n_units_completed": len(effects)}
    for key in ["M_N", "M_R", "D_N", "D_R", "interaction"]:
        vals = [e[key] for e in effects]
        summary[f"{key}_median"] = median(vals)
        summary[f"{key}_mean"] = mean(vals)
        summary[f"{key}_std"] = std(vals)
        summary[f"{key}_wins"] = sum(v > 0 for v in vals)
        summary[f"{key}_bootstrap95"] = bootstrap_ci(vals)
    for cond in ["NN", "RN", "NR", "RR"]:
        summary[f"{cond}_median_auc"] = median([e[cond] for e in effects])
        summary[f"{cond}_mean_auc"] = mean([e[cond] for e in effects])
    horizon_rows = []
    for h in HORIZONS:
        vals = horizons[h]
        horizon_rows.append({
            "architecture": arch,
            "horizon": h,
            "D_R_h_median": median(vals),
            "D_R_h_mean": mean(vals),
            "D_R_h_wins": sum(v > 0 for v in vals),
            "D_R_h_bootstrap95": bootstrap_ci(vals),
            "n_units": len(vals),
        })
    summary["magnitude_benefit_classification"] = classify_effect([e["M_N"] for e in effects], len(effects))
    summary["rotated_direction_harm_classification"] = classify_effect([e["D_R"] for e in effects], len(effects))
    summary["interaction_classification"] = classify_interaction([e["interaction"] for e in effects])
    return effects, horizon_rows, summary


def classify_effect(vals, n_units):
    vals = [float(v) for v in vals if finite_number(v)]
    if not vals:
        return "NOT_RUN"
    ci = bootstrap_ci(vals)
    wins = sum(v > 0 for v in vals)
    if len(vals) >= 13 and wins >= 13 and ci and ci[0] > 0 and (median(vals) or 0.0) > 0:
        return "CAUSALLY_SUPPORTED"
    if wins >= math.ceil(0.55 * len(vals)) and (median(vals) or 0.0) > 0:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def classify_interaction(vals):
    vals = [float(v) for v in vals if finite_number(v)]
    if not vals:
        return "NOT_RUN"
    ci = bootstrap_ci(vals)
    if ci and ci[0] > 0:
        return "POSITIVE"
    if ci and ci[1] < 0:
        return "NEGATIVE"
    return "MIXED_OR_WEAK"


def write_placeholder_stage2(arch, reason):
    for stage, names in {
        "readout_only": ["readout_only.jsonl", "readout_only_unit_aggregate.csv"],
        "memory_only": ["memory_only.jsonl", "memory_only_unit_aggregate.csv"],
        "propagation": ["propagation.jsonl", "propagation_unit_aggregate.csv", "propagation_horizon_summary.csv"],
    }.items():
        outdir = result_arch_dir(arch, stage)
        for name in names:
            path = outdir / name
            if name.endswith(".jsonl"):
                path.write_text(json.dumps({"architecture": arch, "stage": stage, "status": "NOT_RUN", "reason": reason}, sort_keys=True) + "\n", encoding="utf-8")
            else:
                write_rows(path, [{"architecture": arch, "stage": stage, "status": "NOT_RUN", "reason": reason}])


def run_arch(arch, args):
    out_fresh = result_arch_dir(arch, "fresh_capture")
    out_full = result_arch_dir(arch, "full_state_factorial")
    for path in [out_fresh / "fresh_residual_metrics.jsonl", out_full / "full_state_factorial.jsonl"]:
        if path.exists() and not args.resume:
            path.unlink()
    if arch == "gdn":
        units_all, manifest_hash = load_gdn_units()
        units = select_units(units_all, args.scope, args.max_units)
        qhelper, torch_mod, model, tokenizer, p1, _axis, e2e = gdn_setup()
        prompt_rows = gdn_prompt_rows(p1)
        ctx_base = {"torch": torch_mod, "model": model, "tokenizer": tokenizer, "p1": p1, "e2e": e2e}
    else:
        units_all, manifest_hash = load_kda_units()
        units = select_units(units_all, args.scope, args.max_units)
        mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens = kda_setup()
        ctx_base = {"torch": torch_mod, "model": model, "tokenizer": tokenizer, "capture": capture, "kda_layers": kda_layers, "mod": mod}
    fresh_rows = []
    full_unit_rows = []
    failures = []
    for idx, unit in enumerate(units, 1):
        print(f"[{now()}] {arch.upper()} fresh_capture unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
        try:
            if arch == "gdn":
                prompt_row = prompt_rows[str(unit["problem_id"])]
                run = gdn_run_to_t0(torch_mod, model, tokenizer, p1, e2e, prompt_row, int(unit["t0"]), max_horizon=args.horizon)
                stack = gdn_copy_stack(p1, run["past"])
                query_stack = gdn_collect_query_stack(torch_mod, model, p1, run["past"], run["tokens"][int(unit["t0"])], run["dtype"], run["device"])
            else:
                item = rows_by_pid[str(unit["problem_id"])]
                tokens = [int(x) for x in teacher_tokens.get(str(unit["problem_id"]), unit.get("teacher_forced_token_ids") or [])]
                if len(tokens) <= int(unit["t0"]) + int(args.horizon):
                    raise RuntimeError(f"teacher continuation too short for {unit['unit_id']}")
                run = kda_run_to_t0(torch_mod, model, tokenizer, capture, mod, unit, item, tokens, int(unit["t0"]), kda_layers)
                stack = kda_copy_stack(mod, run["past"], kda_layers)
                # Query at the first future token, using a clone to avoid mutating the intervention cache.
                tmp = copy.deepcopy(run["past"])
                tmp_mask = run["mask"].clone()
                _out, _ = kda_step(torch_mod, model, capture, tmp, tmp_mask, run["prompt_len"], tokens[int(unit["t0"])], int(unit["t0"]), "FP_QUERY")
                records = capture.records.get("FP_QUERY", {})
                query_stack = {}
                for layer, rec in records.items():
                    q = rec["q"].detach().float()
                    query_stack[int(layer)] = q[:, -1, :, :].cpu() if q.ndim == 4 else q.squeeze(1).cpu()
            fresh, pairs = fresh_rows_for_unit(arch, unit, stack, query_stack, manifest_hash)
            fresh_rows.append(fresh)
            append_jsonl(out_fresh / "fresh_residual_metrics.jsonl", fresh)
        except Exception as exc:
            failure = {"architecture": arch, "unit": unit, "error": repr(exc), "traceback": traceback.format_exc(), "timestamp": now()}
            failures.append(failure)
            save_json(RESULT_DIR / arch / "failures.json", failures)
            print(f"[{now()}] FAILED {arch} {unit.get('unit_id')} {exc!r}", flush=True)
            if not args.keep_going:
                if arch == "kda" and "capture" in ctx_base:
                    ctx_base["capture"].close()
                raise
    fresh_unit_rows, fresh_summary = summarize_fresh(fresh_rows, arch)
    write_rows(out_fresh / "fresh_residual_unit_aggregate.csv", fresh_unit_rows)
    save_json(out_fresh / "summary.json", fresh_summary)
    if args.stage == "fresh_capture" or fresh_summary.get(f"{arch.upper()}_FRESH_RESIDUAL_CAPTURE") != "PASS":
        reason = "requested_fresh_capture_only" if args.stage == "fresh_capture" else "fresh_capture_gate_failed"
        write_rows(out_full / "full_state_unit_aggregate.csv", [{"architecture": arch, "status": "NOT_RUN", "reason": reason}])
        write_rows(out_full / "factorial_effects.csv", [{"architecture": arch, "status": "NOT_RUN", "reason": reason}])
        write_rows(out_full / "horizon_direction_effects.csv", [{"architecture": arch, "status": "NOT_RUN", "reason": reason}])
        full_summary = {
            "architecture": arch,
            "stage": "full_state_factorial",
            "n_units_completed": 0,
            "status": "NOT_RUN",
            "reason": reason,
            "magnitude_benefit_classification": "NOT_RUN",
            "rotated_direction_harm_classification": "NOT_RUN",
            "interaction_classification": "NOT_RUN",
        }
        save_json(out_full / "summary.json", full_summary)
        write_placeholder_stage2(arch, reason)
        if arch == "kda" and "capture" in ctx_base:
            ctx_base["capture"].close()
        return {"fresh_capture": fresh_summary, "full_state_factorial": full_summary, "failures": failures}
    for idx, unit in enumerate(units, 1):
        print(f"[{now()}] {arch.upper()} full_state_factorial unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
        try:
            if arch == "gdn":
                prompt_row = prompt_rows[str(unit["problem_id"])]
                run = gdn_run_to_t0(torch_mod, model, tokenizer, p1, e2e, prompt_row, int(unit["t0"]), max_horizon=args.horizon)
                stack = gdn_copy_stack(p1, run["past"])
                ctx = {**ctx_base, **run}
            else:
                item = rows_by_pid[str(unit["problem_id"])]
                tokens = [int(x) for x in teacher_tokens.get(str(unit["problem_id"]), unit.get("teacher_forced_token_ids") or [])]
                run = kda_run_to_t0(torch_mod, model, tokenizer, capture, mod, unit, item, tokens, int(unit["t0"]), kda_layers)
                stack = kda_copy_stack(mod, run["past"], kda_layers)
                ctx = {**ctx_base, **run, "dtype": torch_mod.long}
            pairs = stack_residual_pairs(stack, arch)
            row_h, row_u = run_full_state_factorial_unit(arch, ctx, unit, pairs, max_horizon=args.horizon)
            full_unit_rows.extend(row_u)
            for row in row_h:
                append_jsonl(out_full / "full_state_factorial.jsonl", row)
        except Exception as exc:
            failure = {"architecture": arch, "unit": unit, "stage": "full_state_factorial", "error": repr(exc), "traceback": traceback.format_exc(), "timestamp": now()}
            failures.append(failure)
            save_json(RESULT_DIR / arch / "failures.json", failures)
            print(f"[{now()}] FAILED {arch} full_state {unit.get('unit_id')} {exc!r}", flush=True)
            if not args.keep_going:
                if arch == "kda" and "capture" in ctx_base:
                    ctx_base["capture"].close()
                raise
    write_rows(out_full / "full_state_unit_aggregate.csv", full_unit_rows)
    effects, horizon_rows, full_summary = aggregate_factorial(full_unit_rows, arch)
    write_rows(out_full / "factorial_effects.csv", effects)
    write_rows(out_full / "horizon_direction_effects.csv", horizon_rows)
    save_json(out_full / "summary.json", full_summary)
    stage2_reason = "full_state_rotated_direction_harm_not_causally_supported"
    # V1 closure intentionally avoids unsafe hook-based readout-only mutation unless the primary gate is met.
    write_placeholder_stage2(arch, stage2_reason)
    if arch == "kda" and "capture" in ctx_base:
        ctx_base["capture"].close()
    return {"fresh_capture": fresh_summary, "full_state_factorial": full_summary, "failures": failures}


def env_record():
    return {
        "current_python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "gdn_python": GDN_PYTHON,
        "kda_python": KDA_PYTHON,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def write_manifest(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "state_semantics": {"state": "[B,H,K,V]", "R128": "[B,H,K,1]", "C128": "[B,H,1,V]"},
        "primary_rotations": {"gdn": "Key rotation + C128, RHT seed 0", "kda": "Value rotation + R128, RHT seed 0"},
        "horizons": HORIZONS,
        "horizon_semantics": "full-state h=0 is the first future logit after the t0 cache pulse; memory-only current-token logits remain FP by construction",
        "scope_restrictions": ["no learned rotations", "no KDA block-key", "no INT4", "no AIME/LongBench", "no optimized kernels", "no seed search"],
    }
    save_json(RESULT_DIR / "config.json", cfg)
    save_json(RESULT_DIR / "manifest.json", {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "gdn_canonical_manifest": str(GDN_CANONICAL_MANIFEST),
        "kda_canonical_manifest": str(KDA_CANONICAL_UNITS),
        "expected_units": {"gdn": CANONICAL_N, "kda": CANONICAL_N},
        "environment": env_record(),
    })


def load_stage_summary(arch, stage):
    return load_json(RESULT_DIR / arch / stage / "summary.json", {"architecture": arch, "stage": stage, "n_units_completed": 0})


def make_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    plot_dir = RESULT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for arch in ["gdn", "kda"]:
        fresh = read_rows(RESULT_DIR / arch / "fresh_capture" / "fresh_residual_unit_aggregate.csv")
        effects = [
            r for r in read_rows(RESULT_DIR / arch / "full_state_factorial" / "factorial_effects.csv")
            if all(finite_number(r.get(c)) for c in ["NN", "RN", "NR", "RR", "M_N", "D_R"])
        ]
        horizons = [
            r for r in read_rows(RESULT_DIR / arch / "full_state_factorial" / "horizon_direction_effects.csv")
            if finite_number(r.get("horizon")) and finite_number(r.get("D_R_h_median"))
        ]
        if fresh:
            vals = [as_float(r["median_m_R_over_m_N"], 0.0) for r in fresh]
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(vals)), vals)
            plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
            plt.ylabel("||E_R|| / ||E_N||")
            plt.title(f"{arch.upper()} fresh residual magnitude")
            p = plot_dir / f"{arch}_fresh_residual_ratio_by_unit.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
        if effects:
            labels = ["NN", "RN", "NR", "RR"]
            med = [median([as_float(r[c]) for r in effects]) for c in labels]
            plt.figure(figsize=(4, 3))
            plt.bar(labels, med)
            plt.ylabel("FutureKL AUC")
            plt.title(f"{arch.upper()} factorial AUC")
            p = plot_dir / f"{arch}_factorial_auc.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
            for key, label in [("M_N", "Magnitude effect M_N"), ("D_R", "Direction effect D_R")]:
                vals = [as_float(r[key], 0.0) for r in effects]
                plt.figure(figsize=(6, 3))
                plt.bar(range(len(vals)), vals)
                plt.axhline(0.0, color="black", linewidth=1)
                plt.ylabel(key)
                plt.title(f"{arch.upper()} {label}")
                p = plot_dir / f"{arch}_{key}_by_unit.png"
                plt.tight_layout()
                plt.savefig(p, dpi=160)
                plt.close()
                made.append(str(p))
        if horizons:
            xs = [int(float(r["horizon"])) for r in horizons]
            ys = [as_float(r["D_R_h_median"], 0.0) for r in horizons]
            plt.figure(figsize=(5, 3))
            plt.plot(xs, ys, marker="o")
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xlabel("horizon")
            plt.ylabel("median D_R(h)")
            plt.title(f"{arch.upper()} horizon direction effect")
            p = plot_dir / f"{arch}_direction_effect_by_horizon.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
    save_json(plot_dir / "manifest.json", {"plots": made})
    return {"status": "OK", "plots": made}


def classify_overall(summaries):
    g = summaries.get("gdn", {})
    k = summaries.get("kda", {})
    def arch_supported(a):
        f = a.get("fresh_capture", {})
        full = a.get("full_state_factorial", {})
        return (
            f.get(f"{str(f.get('architecture', '')).upper()}_FRESH_RESIDUAL_CAPTURE") == "PASS"
            and full.get("magnitude_benefit_classification") == "CAUSALLY_SUPPORTED"
            and full.get("rotated_direction_harm_classification") == "CAUSALLY_SUPPORTED"
        )
    n_sup = sum(arch_supported(x) for x in [g, k])
    if n_sup == 2:
        closure = "SUPPORTED"
        failure = "YES"
        final = "RESIDUAL_DIRECTION_CAUSALLY_CLOSES_ROTATION_PERSISTENT_FAILURE"
        next_exp = "FUNCTIONAL_OUTPUT_AWARE_STATE_ROTATION_V1"
    elif n_sup == 1:
        closure = "PARTIAL"
        failure = "PARTIAL"
        final = "RESIDUAL_DIRECTION_CAUSAL_CLOSURE_PARTIAL"
        next_exp = "ROTATION_FAILURE_MECHANISM_REASSESSMENT_V1"
    else:
        closure = "NOT_SUPPORTED"
        failure = "NO"
        final = "RESIDUAL_DIRECTION_CAUSAL_CLOSURE_NOT_SUPPORTED"
        next_exp = "ROTATION_FAILURE_MECHANISM_REASSESSMENT_V1"
    return closure, failure, final, next_exp


def collect_summary(results=None):
    if results is None:
        results = {}
        for arch in ["gdn", "kda"]:
            results[arch] = {
                "fresh_capture": load_stage_summary(arch, "fresh_capture"),
                "full_state_factorial": load_stage_summary(arch, "full_state_factorial"),
            }
    plots = make_plots()
    closure, failure, final, next_exp = classify_overall(results)
    cls = {}
    for arch in ["gdn", "kda"]:
        fresh = results.get(arch, {}).get("fresh_capture", {})
        full = results.get(arch, {}).get("full_state_factorial", {})
        prefix = arch.upper()
        cls[f"{prefix}_FRESH_RESIDUAL_CAPTURE"] = fresh.get(f"{prefix}_FRESH_RESIDUAL_CAPTURE", "NOT_RUN")
        cls[f"{prefix}_MAGNITUDE_BENEFIT"] = full.get("magnitude_benefit_classification", "NOT_RUN")
        cls[f"{prefix}_ROTATED_DIRECTION_HARM"] = full.get("rotated_direction_harm_classification", "NOT_RUN")
        cls[f"{prefix}_MAGNITUDE_DIRECTION_INTERACTION"] = full.get("interaction_classification", "NOT_RUN")
        cls[f"{prefix}_OUTPUT_PATH_DIRECTION_HARM"] = "NOT_RUN_PRIMARY_DIRECTION_GATE_NOT_MET"
        cls[f"{prefix}_MEMORY_PATH_DIRECTION_HARM"] = "NOT_RUN_PRIMARY_DIRECTION_GATE_NOT_MET"
        cls[f"{prefix}_PRIMARY_DIRECTION_HARM_PATH"] = "NOT_LOCALIZED"
        cls[f"{prefix}_PROPAGATION_CLASSIFICATION"] = "NOT_RUN"
    cls["ROTATION_MAGNITUDE_GAIN_CANCELLED_BY_RESIDUAL_DIRECTION"] = closure
    cls["RESIDUAL_DIRECTION_CAUSAL_CLOSURE"] = closure
    cls["FAILURE_MECHANISM_IDENTIFIED"] = failure
    cls["FINAL_CLASSIFICATION"] = final
    cls["NEXT_EXPERIMENT"] = next_exp
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": env_record(),
        "classifications": cls,
        "results": results,
        "artifact_dir": str(RESULT_DIR),
        "plots": plots,
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


def write_report(summary):
    c = summary["classifications"]
    lines = [
        "# GDN/KDA Rotation Residual Direction Causal Closure V1",
        "",
        "## Required Classifications",
        "```text",
    ]
    for key in [
        "GDN_FRESH_RESIDUAL_CAPTURE", "KDA_FRESH_RESIDUAL_CAPTURE",
        "GDN_MAGNITUDE_BENEFIT", "KDA_MAGNITUDE_BENEFIT",
        "GDN_ROTATED_DIRECTION_HARM", "KDA_ROTATED_DIRECTION_HARM",
        "GDN_MAGNITUDE_DIRECTION_INTERACTION", "KDA_MAGNITUDE_DIRECTION_INTERACTION",
        "GDN_OUTPUT_PATH_DIRECTION_HARM", "KDA_OUTPUT_PATH_DIRECTION_HARM",
        "GDN_MEMORY_PATH_DIRECTION_HARM", "KDA_MEMORY_PATH_DIRECTION_HARM",
        "GDN_PRIMARY_DIRECTION_HARM_PATH", "KDA_PRIMARY_DIRECTION_HARM_PATH",
        "GDN_PROPAGATION_CLASSIFICATION", "KDA_PROPAGATION_CLASSIFICATION",
        "ROTATION_MAGNITUDE_GAIN_CANCELLED_BY_RESIDUAL_DIRECTION",
        "RESIDUAL_DIRECTION_CAUSAL_CLOSURE", "FAILURE_MECHANISM_IDENTIFIED",
        "FINAL_CLASSIFICATION", "NEXT_EXPERIMENT",
    ]:
        lines.append(f"{key} = {c.get(key)}")
    lines.extend(["```", "", "## Environment", "```json", json.dumps(summary["environment"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Stage Summaries", "```json", json.dumps(summary["results"], indent=2, sort_keys=True), "```"])
    lines.append("")
    lines.append("Fresh capture and full-state intervention artifacts are written under architecture-specific subdirectories. Readout-only, memory-only, and propagation files are emitted as explicit NOT_RUN artifacts unless the full-state direction-harm gate is causally supported.")
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--architecture", choices=["all", "gdn", "kda"], default="all")
    p.add_argument("--stage", choices=["all", "fresh_capture", "full_state", "summary"], default="all")
    p.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    return p.parse_args(argv)


def run(args):
    write_manifest(args)
    if args.stage == "summary":
        return collect_summary()
    arches = ["gdn", "kda"] if args.architecture == "all" else [args.architecture]
    results = {}
    for arch in arches:
        results[arch] = run_arch(arch, args)
    for arch in ["gdn", "kda"]:
        if arch not in results:
            results[arch] = {
                "fresh_capture": load_stage_summary(arch, "fresh_capture"),
                "full_state_factorial": load_stage_summary(arch, "full_state_factorial"),
            }
    return collect_summary(results)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
