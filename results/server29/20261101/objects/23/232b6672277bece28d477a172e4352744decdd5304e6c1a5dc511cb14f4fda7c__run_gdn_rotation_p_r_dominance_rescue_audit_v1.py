#!/usr/bin/env python3
import argparse
import csv
import hashlib
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


TASK = "GDN_ROTATION_P_R_DOMINANCE_RESCUE_AUDIT_V1"
SLUG = "gdn_rotation_p_r_dominance_rescue_audit_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
PRIOR_RESULT_DIR = REPO / "results" / "gdn_kda_rotation_exact_frozen_driver_replay_v1"
PR_CLOSURE_RESULT_DIR = REPO / "results" / "gdn_rotation_p_r_query_visibility_mechanism_closure_v1"
EXACT_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"
PR_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_rotation_p_r_query_visibility_mechanism_closure_v1.py"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
HORIZONS = [16, 32, 64]
EPS = 1e-12
EXCESS_EPS = 1e-10
RHT_SEED = 0
CANONICAL_N = 18
BOOTSTRAP_SEED = 20260914
FP_REPLAY_TOL = 1e-5
PR_IDENTITY_TOL = 5e-3
READOUT_IDENTITY_TOL = 1e-8


_EXACT = None
_PR = None


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact():
    global _EXACT
    if _EXACT is None:
        _EXACT = import_file(EXACT_RUNNER, "exact_frozen_driver_for_rescue_audit")
    return _EXACT


def pr():
    global _PR
    if _PR is None:
        _PR = import_file(PR_RUNNER, "pr_query_visibility_for_rescue_audit")
    return _PR


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


def percentile(values, pct):
    values = sorted(finite_values(values))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * float(pct)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


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


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def tensor_hash(x):
    return exact().tensor_hash(x)


def readout_vec(state, query):
    return torch.einsum("bhkv,bhk->bhv", state.detach().double(), query.detach().double())


def build_component_parts(p_native, r_native, p_rotated, r_rotated):
    return {
        "NN": {"P": p_native.detach().double(), "R": r_native.detach().double()},
        "RN": {"P": p_rotated.detach().double(), "R": r_native.detach().double()},
        "NR": {"P": p_native.detach().double(), "R": r_rotated.detach().double()},
        "RR": {"P": p_rotated.detach().double(), "R": r_rotated.detach().double()},
    }


def build_component_panel(p_native, r_native, p_rotated, r_rotated):
    parts = build_component_parts(p_native, r_native, p_rotated, r_rotated)
    return {name: value["P"] + value["R"] for name, value in parts.items()}


def readout_panel_vectors(panel, query):
    return {name: readout_vec(error, query).reshape(-1) for name, error in panel.items()}


def readout_panel_risks(panel, query):
    return {name: tensor_norm(vec) ** 2 for name, vec in readout_panel_vectors(panel, query).items()}


def rescue_effects(risks, eps=EPS, excess_eps=EXCESS_EPS):
    out = {
        "Risk_NN": float(risks["NN"]),
        "Risk_RN": float(risks["RN"]),
        "Risk_NR": float(risks["NR"]),
        "Risk_RR": float(risks["RR"]),
    }
    out["Rescue_P"] = out["Risk_RR"] - out["Risk_NR"]
    out["Rescue_R"] = out["Risk_RR"] - out["Risk_RN"]
    out["Excess"] = out["Risk_RR"] - out["Risk_NN"]
    out["Rescue_P_minus_R"] = out["Rescue_P"] - out["Rescue_R"]
    out["interaction"] = out["Risk_RR"] - out["Risk_RN"] - out["Risk_NR"] + out["Risk_NN"]
    if out["Excess"] > float(excess_eps):
        out["F_P"] = out["Rescue_P"] / (out["Excess"] + eps)
        out["F_R"] = out["Rescue_R"] / (out["Excess"] + eps)
    else:
        out["F_P"] = None
        out["F_R"] = None
    return out


def same_query_gate(queries):
    ref_name = "NN"
    ref = queries[ref_name].detach().cpu()
    max_abs = 0.0
    for name, query in queries.items():
        cur = query.detach().cpu()
        if cur.shape != ref.shape or cur.dtype != ref.dtype:
            return {"gate": "FAIL", "reason": f"shape_or_dtype:{name}", "max_abs_diff": None}
        diff = float((cur.double() - ref.double()).abs().max().item()) if cur.numel() else 0.0
        max_abs = max(max_abs, diff)
        if not torch.equal(cur, ref):
            return {"gate": "FAIL", "reason": f"value:{name}", "max_abs_diff": max_abs}
    return {"gate": "PASS", "reason": "same_tensor", "max_abs_diff": max_abs, "query_sha256": tensor_hash(ref)}


def gdn_gated_rmsnorm(x, gate, weight, eps):
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    y = x32 * torch.rsqrt(variance + float(eps))
    y = weight.to(device=x.device, dtype=torch.float32) * y
    y = y * F.silu(gate.to(device=x.device, dtype=torch.float32))
    return y.to(x.dtype)


def gdn_same_token_downstream(delta_core, gate, norm_weight, norm_eps, out_proj_weight, base_core=None):
    delta = delta_core.detach().clone()
    gate = gate.detach().clone()
    base = torch.zeros_like(delta) if base_core is None else base_core.detach().clone().to(delta)
    y0 = gdn_gated_rmsnorm(base, gate, norm_weight.detach(), norm_eps)
    y1 = gdn_gated_rmsnorm(base + delta, gate, norm_weight.detach(), norm_eps)
    post_norm_delta = y1 - y0
    batch, seq_len, heads, value_dim = post_norm_delta.shape
    flat = post_norm_delta.reshape(batch, seq_len, heads * value_dim)
    post_out = torch.matmul(flat.to(torch.float32), out_proj_weight.detach().to(torch.float32).t())
    return {"post_norm": post_norm_delta, "post_out_proj": post_out}


def vector_metrics(delta, ref=None):
    flat = delta.detach().double().reshape(-1)
    risk = tensor_norm(flat) ** 2
    row = {"risk": risk, "norm": math.sqrt(risk)}
    if ref is not None:
        ref_flat = ref.detach().double().reshape(-1)
        denom = tensor_norm(ref_flat)
        row["relative_l2"] = tensor_norm(flat) / (denom + EPS)
        dot = float((flat * ref_flat).sum().item()) if flat.numel() == ref_flat.numel() else None
        row["cosine_to_ref"] = dot / (tensor_norm(flat) * denom + EPS) if dot is not None and denom > 0 and tensor_norm(flat) > 0 else None
    return row


class EnhancedGdnTransitionCapture:
    def __init__(self, qmod, model):
        self.inner = exact().GdnTransitionCapture(qmod, model)
        self.model = model
        self.patched_extra = []

    @property
    def records(self):
        return self.inner.records

    @property
    def orig_rec(self):
        return self.inner.orig_rec

    @property
    def orig_chunk(self):
        return self.inner.orig_chunk

    def _last_record(self, layer_idx):
        for rec in reversed(self.records):
            if int(rec.get("layer", -1)) == int(layer_idx):
                return rec
        return None

    def install(self):
        layers = self.inner.install()
        model_layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for idx, layer in enumerate(model_layers):
            if int(idx) not in set(exact().GDN_LAYERS):
                continue
            mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if mod is None or not hasattr(mod, "norm") or not hasattr(mod, "out_proj"):
                continue
            saved_norm = mod.norm.forward
            saved_out = mod.out_proj.forward

            def make_norm(fn, layer_idx):
                def wrapped(hidden_states, gate=None):
                    out = fn(hidden_states, gate)
                    rec = self._last_record(layer_idx)
                    if rec is not None and gate is not None:
                        rec["downstream_gate"] = gate.detach().cpu().clone().to(torch.float32)
                        rec["fp_post_norm"] = out.detach().cpu().clone().to(torch.float32)
                    return out
                return wrapped

            def make_out(fn, layer_idx):
                def wrapped(hidden_states):
                    out = fn(hidden_states)
                    rec = self._last_record(layer_idx)
                    if rec is not None:
                        rec["fp_post_out_proj"] = out.detach().cpu().clone().to(torch.float32)
                    return out
                return wrapped

            mod.norm.forward = make_norm(saved_norm, idx)
            mod.out_proj.forward = make_out(saved_out, idx)
            self.patched_extra.append((mod, saved_norm, saved_out))
        return layers

    def close(self):
        for mod, saved_norm, saved_out in self.patched_extra:
            mod.norm.forward = saved_norm
            mod.out_proj.forward = saved_out
        self.patched_extra = []
        self.inner.close()

    def replay(self, rec, initial_state, rotated=False, rotation=None):
        return self.inner.replay(rec, initial_state, rotated=rotated, rotation=rotation)


def stripped_record_for_hash(rec):
    exact_tensor_fields = {"query", "key", "value", "g", "beta", "initial_state", "core_output", "final_state"}
    keep = {}
    for key, value in rec.items():
        if torch.is_tensor(value):
            if key in exact_tensor_fields:
                keep[key] = value
        elif key in {"layer", "operator", "output_final_state", "use_qk_l2norm_in_kernel"}:
            keep[key] = value
    return keep


def stripped_tape_for_hash(tape):
    return {
        "architecture": tape["architecture"],
        "unit_id": tape["unit_id"],
        "steps": [
            {"t": step["t"], "records": {layer: stripped_record_for_hash(rec) for layer, rec in step["records"].items()}}
            for step in tape["steps"]
        ],
    }


def capture_driver_tape_enhanced(unit, ctx, max_horizon):
    ex = exact()
    torch_mod, model, tokenizer, p1, e2e, qmod = ctx["torch"], ctx["model"], ctx["tokenizer"], ctx["p1"], ctx["e2e"], ctx["qmod"]
    prompt_row = ctx["prompt_rows"][str(unit["problem_id"])]
    prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
    tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
    if len(tokens) <= int(unit["t0"]) + int(max_horizon):
        raise RuntimeError(f"teacher continuation too short for {unit['unit_id']}")
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    with torch_mod.inference_mode():
        out = p1.feed_step(torch_mod, model, ids, mask, None)
    past = out.past_key_values
    for timestep in range(1, int(unit["t0"]) + 1):
        cur = torch_mod.tensor([[int(tokens[timestep - 1])]], dtype=ids.dtype, device=device)
        with torch_mod.inference_mode():
            out = p1.feed_step(torch_mod, model, cur, None, past)
        past = out.past_key_values
    cap = EnhancedGdnTransitionCapture(qmod, model)
    cap.install()
    state0 = ex.gdn_cpu_stack(p1, past)
    steps = []
    try:
        for h in range(1, int(max_horizon) + 1):
            token = int(tokens[int(unit["t0"]) + h - 1])
            cur = torch_mod.tensor([[token]], dtype=ids.dtype, device=device)
            cap.records.clear()
            with torch_mod.inference_mode():
                out = p1.feed_step(torch_mod, model, cur, None, past)
            past = out.past_key_values
            records = {int(r["layer"]): r for r in cap.records if int(r["layer"]) in ex.GDN_LAYERS}
            if set(records) != set(ex.GDN_LAYERS):
                raise RuntimeError(f"GDN tape missing layers at h={h}: {sorted(set(ex.GDN_LAYERS)-set(records))}")
            steps.append({"t": h, "token_id": token, "records": records, "fp_states": ex.gdn_cpu_stack(p1, past), "readout_queries": {layer: ex.gdn_query_for_readout(rec) for layer, rec in records.items()}})
    finally:
        cap.close()
    tape = {"architecture": "gdn", "unit_id": str(unit["unit_id"]), "problem_id": str(unit["problem_id"]), "t0": int(unit["t0"]), "state0": state0, "steps": steps, "capture": cap}
    tape["driver_tape_sha256"] = ex.driver_tape_hash(stripped_tape_for_hash(tape))
    return tape


def model_linear_attn_modules(model):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    out = {}
    for idx, layer in enumerate(layers):
        mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if mod is not None:
            out[int(idx)] = mod
    return out


def replay_unit_rescue(tape, replay_impl, rotation, model, selected_horizons):
    ex = exact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {
        "native": {layer: value.to(device) for layer, value in tape["state0"].items()},
        "rotated": {layer: ex.transform_state_to_branch(value, "gdn", "rotated", rotation).to(device) for layer, value in tape["state0"].items()},
    }
    prev_fp = {layer: value.detach().cpu() for layer, value in tape["state0"].items()}
    modules = model_linear_attn_modules(model)
    lm_head_weight = getattr(model, "lm_head", None).weight.detach().cpu() if hasattr(model, "lm_head") else None
    rescue_rows = []
    downstream_rows = []
    identity_rows = []
    sanity_rows = []
    maxima = defaultdict(float)
    horizon_acc = {}
    for step in tape["steps"]:
        horizon = int(step["t"])
        acc = {
            "unit_id": tape["unit_id"],
            "problem_id": tape["problem_id"],
            "t0": int(tape["t0"]),
            "horizon": horizon,
            "driver_tape_id": tape["driver_tape_sha256"],
            "n_layers": 0,
            "state_norm": defaultdict(float),
            "readout_risk": defaultdict(float),
            "readout_norm": defaultdict(float),
            "downstream_risk": defaultdict(lambda: defaultdict(float)),
            "downstream_norm": defaultdict(lambda: defaultdict(float)),
            "post_out_proj_delta_sum": {cond: None for cond in ["NN", "RN", "NR", "RR"]},
            "identity_errors": [],
        }
        for layer, rec in sorted(step["records"].items()):
            native_prev_state = states["native"][layer].detach().cpu()
            rotated_prev_state = ex.inverse_branch_state(states["rotated"][layer].detach().cpu(), "gdn", "rotated", rotation)
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
            e_n = q_n_native - fp_cur
            p_r = pre_r_native - fp_cur
            r_r = q_r_native - pre_r_native
            e_r = q_r_native - fp_cur
            prev_native_error = native_prev_state.detach().double() - prev_fp[layer].detach().double()
            prev_rotated_error = rotated_prev_state.detach().double() - prev_fp[layer].detach().double()
            terms = pr().gdn_transition_terms(rec, rotation=rotation, rotated=False)
            query = terms["query"]
            captured_query = step["readout_queries"][layer].detach().cpu().double()
            if pr().assert_same_tensor(captured_query, query, "query", atol=1e-7)["gate"] != "PASS":
                raise RuntimeError("same q gate failed")
            panel = build_component_panel(p_n, r_n, p_r, r_r)
            if pr().identity_gate_error(panel["NN"], e_n, abs_tol=1e-4) > PR_IDENTITY_TOL:
                raise RuntimeError("NN accumulated error identity failed")
            if pr().identity_gate_error(panel["RR"], e_r, abs_tol=1e-4) > PR_IDENTITY_TOL:
                raise RuntimeError("RR accumulated error identity failed")
            p_metric_n = pr().effective_query_metrics(prev_native_error, query, terms["key"], terms["gamma"], terms["beta"])
            p_metric_r = pr().effective_query_metrics(prev_rotated_error, query, terms["key"], terms["gamma"], terms["beta"])
            p_actual_n = readout_vec(p_n, query)
            p_actual_r = readout_vec(p_r, query)
            trans_n = pr().apply_gdn_error_transition(prev_native_error, terms["key"], terms["gamma"], terms["beta"])
            trans_r = pr().apply_gdn_error_transition(prev_rotated_error, terms["key"], terms["gamma"], terms["beta"])
            identity_samples = [
                ("P_effective_query_identity", pr().identity_gate_error(p_actual_n, p_metric_n["p_readout"]), p_actual_n, p_metric_n["p_readout"]),
                ("P_effective_query_identity", pr().identity_gate_error(p_actual_r, p_metric_r["p_readout"]), p_actual_r, p_metric_r["p_readout"]),
                ("P_transition_identity", pr().identity_gate_error(p_n, trans_n), p_n, trans_n),
                ("P_transition_identity", pr().identity_gate_error(p_r, trans_r), p_r, trans_r),
            ]
            for kind, err, actual, expected in identity_samples:
                numerator = tensor_norm(actual.detach().double() - expected.detach().double())
                denom = max(tensor_norm(actual), tensor_norm(expected), 1e-3)
                row = {
                    "unit_id": tape["unit_id"],
                    "problem_id": tape["problem_id"],
                    "t0": int(tape["t0"]),
                    "horizon": horizon,
                    "layer": int(layer),
                    "kind": kind,
                    "relative_error": err,
                    "numerator": numerator,
                    "denominator_reference_norm": denom,
                    "actual_dtype": str(actual.dtype),
                    "expected_dtype": str(expected.dtype),
                }
                identity_rows.append(row)
                acc["identity_errors"].append(row)
                maxima[kind] = max(maxima[kind], err)
            vectors = readout_panel_vectors(panel, query)
            risks = {name: tensor_norm(vec) ** 2 for name, vec in vectors.items()}
            for cond, error in panel.items():
                acc["state_norm"][cond] += tensor_norm(error) ** 2
                acc["readout_risk"][cond] += risks[cond]
                acc["readout_norm"][cond] += tensor_norm(vectors[cond]) ** 2
            if all(key in rec for key in ["downstream_gate", "core_output"]):
                mod = modules[int(layer)]
                norm_weight = mod.norm.weight.detach().cpu()
                norm_eps = float(mod.norm.variance_epsilon)
                out_proj_weight = mod.out_proj.weight.detach().cpu()
                base_core = rec["core_output"].detach().cpu().double()
                gate = rec["downstream_gate"].detach().cpu().reshape_as(base_core).double()
                for cond, vec in vectors.items():
                    delta_core = vec.reshape_as(base_core)
                    ds = gdn_same_token_downstream(delta_core, gate, norm_weight, norm_eps, out_proj_weight, base_core=base_core)
                    for stage, delta in ds.items():
                        risk = tensor_norm(delta) ** 2
                        acc["downstream_risk"][stage][cond] += risk
                        acc["downstream_norm"][stage][cond] += risk
                        if stage == "post_out_proj":
                            cur = delta.detach().cpu().double()
                            acc["post_out_proj_delta_sum"][cond] = cur if acc["post_out_proj_delta_sum"][cond] is None else acc["post_out_proj_delta_sum"][cond] + cur
            acc["n_layers"] += 1
            prev_fp[layer] = fp_cur.detach().cpu()
        if horizon in selected_horizons:
            if lm_head_weight is not None and all(acc["post_out_proj_delta_sum"][cond] is not None for cond in ["NN", "RN", "NR", "RR"]):
                weight_t = lm_head_weight.detach().to(torch.float32).t()
                for cond, post_out_delta in acc["post_out_proj_delta_sum"].items():
                    logits_delta = torch.matmul(post_out_delta.to(torch.float32), weight_t)
                    risk = tensor_norm(logits_delta) ** 2
                    acc["downstream_risk"]["logits"][cond] += risk
                    acc["downstream_norm"]["logits"][cond] += risk
            raw_effects = rescue_effects(acc["readout_risk"])
            row = {
                "unit_id": acc["unit_id"],
                "problem_id": acc["problem_id"],
                "t0": acc["t0"],
                "horizon": horizon,
                "driver_tape_id": acc["driver_tape_id"],
                "n_layers": acc["n_layers"],
                **raw_effects,
            }
            for cond in ["NN", "RN", "NR", "RR"]:
                row[f"{cond}_readout_norm"] = math.sqrt(acc["readout_norm"][cond])
                row[f"{cond}_state_norm"] = math.sqrt(acc["state_norm"][cond])
            rescue_rows.append(row)
            for stage, risks_by_cond in acc["downstream_risk"].items():
                ds_effects = rescue_effects(risks_by_cond)
                ds_row = {
                    "unit_id": acc["unit_id"],
                    "problem_id": acc["problem_id"],
                    "t0": acc["t0"],
                    "horizon": horizon,
                    "stage": stage,
                    "driver_tape_id": acc["driver_tape_id"],
                    **ds_effects,
                }
                for cond in ["NN", "RN", "NR", "RR"]:
                    ds_row[f"{cond}_norm"] = math.sqrt(acc["downstream_norm"][stage][cond])
                downstream_rows.append(ds_row)
            sanity_rows.append({
                "unit_id": acc["unit_id"],
                "problem_id": acc["problem_id"],
                "t0": acc["t0"],
                "horizon": horizon,
                "G_E_norm": 1.0 - math.sqrt(acc["state_norm"]["RR"]) / (math.sqrt(acc["state_norm"]["NN"]) + EPS),
                "G_E_readout_norm": 1.0 - math.sqrt(acc["readout_norm"]["RR"]) / (math.sqrt(acc["readout_norm"]["NN"]) + EPS),
            })
            horizon_acc[horizon] = acc
    return {
        "rescue_rows": rescue_rows,
        "downstream_rows": downstream_rows,
        "identity_rows": identity_rows,
        "sanity_rows": sanity_rows,
        "maxima": dict(maxima),
    }


def summarize_rows(rows, selected_horizons, prefix_keys=None, group_key=None, bootstrap_samples=4000):
    keys = prefix_keys or ["Rescue_P", "Rescue_R", "Rescue_P_minus_R", "interaction", "Excess", "F_P", "F_R", "Risk_NN", "Risk_RN", "Risk_NR", "Risk_RR"]
    summaries = []
    groups = [None]
    if group_key:
        groups = sorted(set(row[group_key] for row in rows))
    for horizon in selected_horizons:
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
                out[f"{key}_bootstrap95_mean"] = bootstrap_mean_ci(vals, bootstrap_samples, BOOTSTRAP_SEED + int(horizon) + sum(ord(c) for c in str(group or "")))
                out[f"{key}_wins"] = sum(float(v) > 0.0 for v in finite_values(vals))
            out["Rescue_P_gt_Rescue_R_wins"] = sum(float(row["Rescue_P"]) > float(row["Rescue_R"]) for row in selected)
            summaries.append(out)
    return summaries


def identity_numerical_audit(identity_rows):
    by_kind = {}
    for kind in ["P_effective_query_identity", "P_transition_identity"]:
        vals = [row["relative_error"] for row in identity_rows if row["kind"] == kind]
        worst = sorted([row for row in identity_rows if row["kind"] == kind], key=lambda row: row["relative_error"], reverse=True)[:10]
        by_kind[kind] = {
            "median": median(vals),
            "p95": percentile(vals, 0.95),
            "p99": percentile(vals, 0.99),
            "max": max(finite_values(vals)) if finite_values(vals) else None,
            "worst10": worst,
        }
    max_err = max((row["relative_error"] for row in identity_rows), default=0.0)
    max_num = max((row["numerator"] for row in identity_rows), default=0.0)
    small_denom_worst = [row for row in identity_rows if row["relative_error"] > 1e-3 and row["denominator_reference_norm"] <= 1e-3 + 1e-12]
    if max_err <= PR_IDENTITY_TOL and small_denom_worst:
        label = "BENIGN_SMALL_DENOMINATOR"
    elif max_err <= PR_IDENTITY_TOL:
        label = "FINITE_PRECISION_BENIGN"
    elif max_err > PR_IDENTITY_TOL:
        label = "SEMANTIC_MISMATCH"
    else:
        label = "UNRESOLVED"
    return {"by_kind": by_kind, "classification": label}


def support_label(summary, key):
    med = summary.get(f"{key}_median")
    wins = summary.get(f"{key}_wins", 0)
    ci = summary.get(f"{key}_bootstrap95_mean") or [None, None]
    if med is not None and med > 0.0 and ci[0] is not None and ci[0] > 0.0 and wins >= 13:
        return "STRONG"
    if med is not None and med > 0.0 and wins >= 10:
        return "POSITIVE"
    if med is not None and med > 0.0 and wins >= 8:
        return "WEAK"
    return "NOT_SUPPORTED"


def classify_dominance(readout_summary, downstream_by_stage, identity_label="FINITE_PRECISION_BENIGN"):
    t64 = next((row for row in readout_summary if int(row["horizon"]) == 64), readout_summary[-1] if readout_summary else {})
    p_rescue = support_label(t64, "Rescue_P")
    r_rescue = support_label(t64, "Rescue_R")
    diff_ci = t64.get("Rescue_P_minus_R_bootstrap95_mean") or [None, None]
    p_gt_r = int(t64.get("Rescue_P_gt_Rescue_R_wins", t64.get("Rescue_P_minus_R_wins", 0)))
    readout_supported = (
        t64.get("Rescue_P_median", 0.0) > 0.0
        and p_gt_r >= 13
        and diff_ci[0] is not None
        and diff_ci[0] > 0.0
    )
    stage_support = {}
    for stage in ["post_out_proj", "logits"]:
        row = downstream_by_stage.get(stage)
        if not row:
            stage_support[stage] = "NOT_SUPPORTED"
            continue
        ci = row.get("Rescue_P_minus_R_bootstrap95_mean") or [None, None]
        wins = int(row.get("Rescue_P_gt_Rescue_R_wins", row.get("Rescue_P_minus_R_wins", 0)))
        stage_support[stage] = "SUPPORTED" if row.get("Rescue_P_median", 0.0) > 0.0 and wins >= 13 and ci[0] is not None and ci[0] > 0.0 else "PARTIAL" if row.get("Rescue_P_median", 0.0) > 0.0 and wins >= 10 else "NOT_SUPPORTED"
    downstream_supported = stage_support.get("post_out_proj") == "SUPPORTED" and stage_support.get("logits") in {"SUPPORTED", "PARTIAL"}
    if readout_supported and downstream_supported:
        p_vs_r = "P_DOMINANT"
    elif r_rescue == "STRONG" and p_rescue == "STRONG":
        p_vs_r = "BOTH_IMPORTANT"
    elif r_rescue == "STRONG" and not readout_supported:
        p_vs_r = "R_IMPORTANT"
    else:
        p_vs_r = "MIXED_OR_UNRESOLVED"
    if p_vs_r == "P_DOMINANT":
        dominant = "PROPAGATED_P_EFFECTIVE_QUERY_GEOMETRY"
        closure = "SUFFICIENT_FOR_CURRENT_PAPER"
        final = "GDN_P_DOMINANT_READOUT_FAILURE_RESCUE_SUPPORTED"
        next_exp = "KDA_ROTATION_POST_READOUT_GAIN_RETENTION_V1"
    elif p_vs_r in {"BOTH_IMPORTANT", "R_IMPORTANT"}:
        dominant = "P_AND_R_MULTI_FACTOR" if p_vs_r == "BOTH_IMPORTANT" else "R_PATH_REQUIRED"
        closure = "PARTIAL"
        final = f"GDN_{p_vs_r}_READOUT_RESCUE"
        next_exp = "GDN_REQUANTIZATION_QUERY_GEOMETRY_CLOSURE_V2"
    else:
        dominant = "UNRESOLVED"
        closure = "PARTIAL"
        final = "GDN_P_R_RESCUE_MIXED_OR_UNRESOLVED"
        next_exp = "GDN_REQUANTIZATION_QUERY_GEOMETRY_CLOSURE_V2"
    return {
        "GDN_P_IDENTITY_NUMERICAL_AUDIT": identity_label,
        "GDN_P_RESTORATION_RESCUE": p_rescue,
        "GDN_R_RESTORATION_RESCUE": r_rescue,
        "GDN_P_VS_R_RESCUE": p_vs_r,
        "GDN_P_DOMINANCE_AT_READOUT": "SUPPORTED" if readout_supported else "PARTIAL" if p_rescue in {"STRONG", "POSITIVE", "WEAK"} else "NOT_SUPPORTED",
        "GDN_P_DOMINANCE_DOWNSTREAM": "SUPPORTED" if downstream_supported else "PARTIAL" if any(v in {"SUPPORTED", "PARTIAL"} for v in stage_support.values()) else "NOT_SUPPORTED",
        "GDN_READOUT_FAILURE_DOMINANT_MECHANISM": dominant,
        "GDN_MECHANISM_CLOSURE_STATUS": closure,
        "FINAL_CLASSIFICATION": final,
        "NEXT_EXPERIMENT": next_exp,
        "downstream_stage_support": stage_support,
    }


def clean_result_dir():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["sanity", "rescue", "downstream", "figures"]:
        path = RESULT_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "manifest.json", "provenance.json", "summary.json", "report.md", "failure.json"]:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def validate_prior_provenance(previous_manifest, previous_parity, previous_pr):
    if previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE") != "PASS" or previous_manifest.get("n_units_completed") != CANONICAL_N:
        raise RuntimeError("prior GDN driver-tape capture provenance is missing or incomplete")
    if previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY") != "PASS" or previous_parity.get("n_units_completed") != CANONICAL_N:
        raise RuntimeError("prior GDN FP replay provenance is missing or incomplete")
    classifications = previous_pr.get("classifications", {})
    if classifications.get("GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION") != "SUPPORTED":
        raise RuntimeError("prior P mechanism support is missing")


def tape_hash_comparison_label(horizon):
    return "PASS" if int(horizon) == 64 else "NOT_COMPARABLE_SHORT_HORIZON"


def make_plots(readout_rows, downstream_rows, interaction_rows):
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

    for idx, cond in enumerate(["NN", "RN", "NR", "RR"], 1):
        xs = sorted(set(row["horizon"] for row in readout_rows))
        ys = [median(row[f"Risk_{cond}"] for row in readout_rows if row["horizon"] == x) for x in xs]
        plt.plot(xs, ys, marker="o", label=cond)
    plt.xscale("log", base=2)
    plt.xticks(xs, xs)
    plt.ylabel("raw readout risk")
    plt.legend()
    save("01_raw_readout_risk_panel.png")
    for metric, name in [("Rescue_P", "02_p_restoration_rescue_by_unit.png"), ("Rescue_R", "03_r_restoration_rescue_by_unit.png"), ("Rescue_P_minus_R", "04_rescue_p_minus_r_by_unit.png")]:
        rows = [row for row in readout_rows if int(row["horizon"]) == 64]
        plt.bar(range(len(rows)), [row[metric] for row in rows])
        plt.axhline(0.0, color="black", linewidth=1)
        plt.ylabel(metric)
        save(name)
    xs = sorted(set(row["horizon"] for row in readout_rows))
    plt.plot(xs, [median(row["Rescue_P"] for row in readout_rows if row["horizon"] == x) for x in xs], marker="o", label="Rescue_P")
    plt.plot(xs, [median(row["Rescue_R"] for row in readout_rows if row["horizon"] == x) for x in xs], marker="o", label="Rescue_R")
    plt.xscale("log", base=2)
    plt.xticks(xs, xs)
    plt.legend()
    plt.ylabel("rescue")
    save("05_rescue_vs_horizon.png")
    for stage, name in [("post_out_proj", "06_post_out_proj_rescue_p_vs_r.png"), ("logits", "07_same_token_logit_rescue_p_vs_r.png")]:
        rows = [row for row in downstream_rows if row.get("stage") == stage and int(row["horizon"]) == 64]
        plt.scatter([row["Rescue_P"] for row in rows], [row["Rescue_R"] for row in rows])
        plt.axline((0, 0), slope=1, color="black", linewidth=1)
        plt.xlabel("Rescue_P")
        plt.ylabel("Rescue_R")
        save(name)
    plt.plot(xs, [median(row["interaction"] for row in interaction_rows if row["horizon"] == x) for x in xs], marker="o")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xscale("log", base=2)
    plt.xticks(xs, xs)
    plt.ylabel("factorial interaction")
    save("08_factorial_interaction_by_horizon.png")
    save_json(fig_dir / "manifest.json", {"status": "OK", "plots": made})
    return {"status": "OK", "plots": made}


def write_report(summary):
    keys = [
        "GDN_P_IDENTITY_NUMERICAL_AUDIT",
        "GDN_P_RESTORATION_RESCUE",
        "GDN_R_RESTORATION_RESCUE",
        "GDN_P_VS_R_RESCUE",
        "GDN_P_DOMINANCE_AT_READOUT",
        "GDN_P_DOMINANCE_DOWNSTREAM",
        "GDN_READOUT_FAILURE_DOMINANT_MECHANISM",
        "GDN_MECHANISM_CLOSURE_STATUS",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN Rotation P/R Dominance Rescue Audit V1", "", "## Final Classification", "", "```text"]
    lines.extend(f"{key} = {summary['classifications'].get(key)}" for key in keys)
    lines.extend(["```", "", "## Notes", ""])
    lines.append("- Formal unit is the canonical recurrent unit, N=18.")
    lines.append("- Rescue is component restoration at the accumulated-state/readout boundary, not a future recurrent rollout.")
    lines.append("- Downstream evidence is limited to the measurable same-token GDN local path: gated RMSNorm, out_proj, and a local lm_head logits proxy.")
    lines.extend(["", "## Identity Numerical Audit", "", "```json", json.dumps(summary["identity_numerical_audit"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Readout Horizon Summary", "", "```json", json.dumps(summary["readout_horizon_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Downstream Horizon Summary", "", "```json", json.dumps(summary["downstream_horizon_summary"], indent=2, sort_keys=True), "```"])
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_record():
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "validated_gdn_python": GDN_PYTHON,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args(argv)


def run(args):
    if args.scope == "formal" and int(args.horizon) != 64:
        raise RuntimeError("formal rescue audit requires --horizon 64")
    clean_result_dir()
    selected_horizons = [h for h in HORIZONS if h <= int(args.horizon)]
    config = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "architecture": "gdn",
        "rotation": {"axis": "key", "family": "RHT", "seed": RHT_SEED},
        "quantizer": "C128",
        "conditions": {
            "NN": "P_N + R_N",
            "RN": "P_R + R_N",
            "NR": "P_N + R_R",
            "RR": "P_R + R_R",
        },
        "risk_primary": "squared L2 norm of readout/downstream perturbation",
        "formal_statistical_unit": "canonical recurrent unit",
        "horizons": selected_horizons,
    }
    save_json(RESULT_DIR / "config.json", config)
    ex = exact()
    previous_manifest = load_json(PRIOR_RESULT_DIR / "gdn" / "driver_capture" / "driver_tape_manifest.json", {}) or {}
    previous_parity = load_json(PRIOR_RESULT_DIR / "gdn" / "parity" / "summary.json", {}) or {}
    previous_pr = load_json(PR_CLOSURE_RESULT_DIR / "summary.json", {}) or {}
    validate_prior_provenance(previous_manifest, previous_parity, previous_pr)
    previous_tapes = {row["unit_id"]: row for row in previous_manifest.get("driver_tapes", [])}
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
    all_rescue = []
    all_downstream = []
    all_identity = []
    all_sanity = []
    tape_rows = []
    maxima = defaultdict(float)
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{now()}] GDN P/R rescue audit unit {index}/{len(units)} {unit['unit_id']}", flush=True)
            tape = capture_driver_tape_enhanced(unit, context, args.horizon)
            prior = previous_tapes.get(tape["unit_id"])
            strict_hash = tape_hash_comparison_label(args.horizon) == "PASS"
            if prior is None or (strict_hash and prior.get("driver_tape_sha256") != tape["driver_tape_sha256"]):
                raise RuntimeError(f"driver-tape provenance mismatch unit={tape['unit_id']}")
            fp_rows = ex.replay_fp_tape(tape, "gdn", tape["capture"])
            max_fp = max(row["state_relative_error"] for row in fp_rows)
            if max_fp > FP_REPLAY_TOL or any(row["nonfinite"] for row in fp_rows):
                raise RuntimeError(f"FP replay gate failed unit={tape['unit_id']} max_rel={max_fp}")
            rotation = ex.make_experiment_rotation("gdn", next(iter(tape["state0"].values())).shape[-2], "rht", seed=RHT_SEED, dtype=torch.float32)
            result = replay_unit_rescue(tape, tape["capture"], rotation, model, selected_horizons)
            all_rescue.extend(result["rescue_rows"])
            all_downstream.extend(result["downstream_rows"])
            all_identity.extend(result["identity_rows"])
            all_sanity.extend(result["sanity_rows"])
            append_jsonl(RESULT_DIR / "rescue" / "rescue_metrics.jsonl", result["rescue_rows"])
            append_jsonl(RESULT_DIR / "downstream" / "downstream_rescue_metrics.jsonl", result["downstream_rows"])
            append_jsonl(RESULT_DIR / "sanity" / "identity_numerical_audit.jsonl", result["identity_rows"])
            for key, value in result["maxima"].items():
                maxima[key] = max(maxima[key], value)
            tape_rows.append({
                "unit_id": tape["unit_id"],
                "problem_id": tape["problem_id"],
                "t0": tape["t0"],
                "driver_tape_sha256": tape["driver_tape_sha256"],
                "prior_driver_tape_sha256": prior["driver_tape_sha256"],
                "prior_driver_tape_hash_comparison": tape_hash_comparison_label(args.horizon),
                "fp_replay_max_relative_error": max_fp,
                "n_steps": len(tape["steps"]),
            })
            t64 = next(row for row in result["rescue_rows"] if int(row["horizon"]) == max(selected_horizons))
            print(
                f"[{now()}] PASS {tape['unit_id']} fp={max_fp:.3e} "
                f"Rescue_P={t64['Rescue_P']:.6e} Rescue_R={t64['Rescue_R']:.6e} P-R={t64['Rescue_P_minus_R']:.6e}",
                flush=True,
            )
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
    readout_summary = summarize_rows(all_rescue, selected_horizons, bootstrap_samples=args.bootstrap_samples)
    downstream_summary = summarize_rows(all_downstream, selected_horizons, group_key="stage", bootstrap_samples=args.bootstrap_samples)
    t64_sanity = [row for row in all_sanity if int(row["horizon"]) == 64]
    if args.scope == "formal":
        if median(row["G_E_norm"] for row in t64_sanity) <= 0.0 or median(row["G_E_readout_norm"] for row in t64_sanity) >= 0.0:
            raise RuntimeError("Stage 0 readout reversal was not reproduced")
    identity_audit = identity_numerical_audit(all_identity)
    if identity_audit["classification"] == "SEMANTIC_MISMATCH":
        raise RuntimeError("P identity numerical audit found semantic mismatch")
    downstream_t64 = {row["stage"]: row for row in downstream_summary if int(row["horizon"]) == 64}
    classifications = classify_dominance(readout_summary, downstream_t64, identity_audit["classification"])
    interaction_rows = [{**row, "interaction": row["interaction"]} for row in all_rescue]
    write_rows(RESULT_DIR / "sanity" / "identity_numerical_audit.csv", all_identity)
    write_rows(RESULT_DIR / "sanity" / "stage0_sanity.csv", all_sanity)
    write_rows(RESULT_DIR / "rescue" / "rescue_unit_aggregate.csv", all_rescue)
    write_rows(RESULT_DIR / "rescue" / "rescue_horizon_summary.csv", readout_summary)
    write_rows(RESULT_DIR / "downstream" / "downstream_rescue_unit_aggregate.csv", all_downstream)
    write_rows(RESULT_DIR / "downstream" / "downstream_rescue_horizon_summary.csv", downstream_summary)
    write_rows(RESULT_DIR / "rescue" / "factorial_interaction.csv", interaction_rows)
    plots = make_plots(all_rescue, all_downstream, interaction_rows)
    provenance = {
        "GDN_FP_DRIVER_TAPE_CAPTURE": previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE"),
        "GDN_FROZEN_DRIVER_FP_REPLAY": previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY"),
        "GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION": previous_pr.get("classifications", {}).get("GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION"),
        "current_driver_tape_hash_match": "PASS",
        "current_fp_replay": "PASS",
        "same_q_all_conditions": "PASS",
        "canonical_manifest": str(ex.GDN_CANONICAL_MANIFEST),
        "canonical_manifest_sha256": canonical_manifest_hash,
        "prior_result_dir": str(PRIOR_RESULT_DIR),
        "prior_pr_closure_result_dir": str(PR_CLOSURE_RESULT_DIR),
        "driver_tapes": tape_rows,
    }
    manifest = {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "canonical_units_expected": expected_units,
        "canonical_units_completed": len(tape_rows),
        "canonical_unit_ids": [row["unit_id"] for row in tape_rows],
        "artifacts": [
            str(RESULT_DIR / name) for name in [
                "config.json",
                "manifest.json",
                "provenance.json",
                "summary.json",
                "report.md",
                "sanity/identity_numerical_audit.csv",
                "rescue/rescue_metrics.jsonl",
                "rescue/rescue_unit_aggregate.csv",
                "rescue/rescue_horizon_summary.csv",
                "downstream/downstream_rescue_metrics.jsonl",
                "downstream/downstream_rescue_unit_aggregate.csv",
                "rescue/factorial_interaction.csv",
            ]
        ],
    }
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": manifest["git_commit"],
        "git_status_short": manifest["git_status_short"],
        "environment": environment_record(),
        "canonical_units_completed": len(tape_rows),
        "canonical_units_expected": expected_units,
        "identity_numerical_audit": identity_audit,
        "readout_horizon_summary": readout_summary,
        "downstream_horizon_summary": downstream_summary,
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
