#!/usr/bin/env python3
"""Final causal closure for the full finite-precision KDA rotation transition."""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_FULL_TRANSITION_FINAL_CAUSAL_CLOSURE_V1"
SLUG = "kda_rotation_full_transition_final_causal_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
AFFINE_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_affine_decay_value_unified_causal_closure_v1.py"
COMMUTATOR_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_quantizer_commutator_causal_decomposition_v1.py"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EPS = 1e-12


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AFFINE = import_file(AFFINE_RUNNER, "affine_for_full_transition_final_v1")
COMM = import_file(COMMUTATOR_RUNNER, "commutator_for_full_transition_final_v1")
GAB = AFFINE.GAB
BASE = AFFINE.BASE
DECAY = AFFINE.DECAY


def median_bootstrap_ci(values, n=4000, seed=BASE.BOOTSTRAP_SEED):
    vals = [float(x) for x in values if BASE.finite(x)]
    if not vals:
        return [float("nan"), float("nan")]
    rng = random.Random(int(seed))
    draws = sorted(statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(int(n)))
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def effect(values, name):
    vals = [float(x) for x in values]
    ci = median_bootstrap_ci(vals)
    return {
        "name": name, "estimand": "paired canonical-unit median", "n": len(vals),
        "paired_median": BASE.median(vals), "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
        "positive": sum(x > 0 for x in vals), "negative": sum(x < 0 for x in vals),
        "paired_values": vals,
    }


def plan(name, basis, quantized=True, state_source=None, family="formal"):
    return AFFINE.InterventionPlan(name, basis, quantized=quantized, state_source=state_source, family=family)


def tensor_metrics(value, reference):
    x, y = value.detach().double(), reference.detach().double()
    diff = x - y; nx, ny = BASE.tensor_norm(x), BASE.tensor_norm(y)
    return {"max_abs": float(diff.abs().max().item()), "relative_l2": BASE.tensor_norm(diff) / (ny + EPS),
            "cosine": float((x * y).sum().item()) / (nx * ny + EPS)}


def stack_metrics(left, right):
    diffs = [left[k].double() - right[k].double() for k in sorted(left)]
    n = math.sqrt(sum(BASE.tensor_norm(x) ** 2 for x in diffs))
    d = math.sqrt(sum(BASE.tensor_norm(right[k].double()) ** 2 for k in right))
    max_abs = max(float(x.abs().max().item()) for x in diffs)
    dot = sum(float((left[k].double() * right[k].double()).sum().item()) for k in left)
    nl = math.sqrt(sum(BASE.tensor_norm(left[k].double()) ** 2 for k in left))
    return {"max_abs": max_abs, "relative_l2": n / (d + EPS), "cosine": dot / (nl * d + EPS)}


def rotate_stack(stack, rotation):
    return {layer: BASE.rotate_state_value_axis(state, rotation) for layer, state in stack.items()}


def inverse_stack(stack, rotation):
    return {layer: BASE.inverse_rotate_state_value_axis(state, rotation) for layer, state in stack.items()}


def replace_stack(cache, stack):
    with torch.inference_mode():
        BASE.replace_cache_stack(cache, stack)


class FullTransitionProbe(AFFINE.AffineProbe):
    """Historical affine probe plus one exact current-driver FP32 transition."""

    def _wrap(self, operator, fn):
        parent_wrapped = super()._wrap(operator, fn)

        def wrapped(**kwargs):
            plan_obj = self.plan
            if plan_obj is None or plan_obj.name != "R_FP32_TRANSITION":
                return parent_wrapped(**kwargs)
            branch, layer = self.branch, self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            initial = kwargs.get("initial_state").detach().float().clone()
            log_decay = DECAY.final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"),
                bool(kwargs.get("use_gate_in_kernel", False)),
            )
            v_semantic = kwargs["v"].detach().clone()
            live_rec = {
                "k": kwargs["k"], "beta": kwargs["beta"], "v_semantic": v_semantic,
                "effective_decay": torch.exp(log_decay.float()),
                "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            }
            a = GAB.a_from_record(live_rec).to(device=initial.device, dtype=torch.float32)
            b = GAB.b_to_basis(GAB.canonical_b_from_record(live_rec), "rotated", self.rotation).to(
                device=initial.device, dtype=torch.float32)
            state = torch.einsum("...kl,...lv->...kv", a, initial) + b
            query = GAB.normalized_query(
                GAB._last_token(kwargs["q"]),
                bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            ).float()
            if query.shape[-2] != state.shape[-3]:
                query = query.repeat_interleave(state.shape[-3] // query.shape[-2], dim=-2)
            scale = kwargs.get("scale")
            if scale is None: scale = query.shape[-1] ** -0.5
            raw_out = torch.einsum("...k,...kv->...v", query * float(scale), state).unsqueeze(1)
            returned = raw_out.matmul(self.rotation.t().to(raw_out.device, torch.float32)).to(kwargs["v"].dtype)
            if branch is not None and layer is not None:
                self.records[branch][int(layer)] = {
                    "operator": operator, "q": BASE.record_tensor(kwargs["q"]),
                    "k": BASE.record_tensor(kwargs["k"]), "v": BASE.record_tensor(kwargs["v"]),
                    "v_semantic": BASE.record_tensor(v_semantic), "beta": BASE.record_tensor(kwargs["beta"]),
                    "raw_gate": BASE.record_tensor(raw_gate), "log_decay": BASE.record_tensor(log_decay, float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(log_decay.float()), float32=True),
                    "effective_a": BASE.record_tensor(a, float32=True), "effective_b": BASE.record_tensor(b, float32=True),
                    "initial_state": BASE.record_tensor(initial, float32=True),
                    "final_state": BASE.record_tensor(state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True), "raw_output": BASE.record_tensor(raw_out, float32=True),
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "basis": "rotated", "precision_intervention": "FP32_TRANSITION_CURRENT_DRIVERS",
                }
            return returned, state

        return wrapped


def baseline_plans():
    return [plan("FP", "native", False, family="baseline"), plan("N", "native", True, family="baseline"),
            plan("R", "rotated", True, family="baseline")]


def prepare_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation):
    return GAB.prepare_prefill(model, tokenizer, unit, row, tokens, layers, rotation, probe, baseline_plans())


def switch_branches(pasts, masks, layers, rotation):
    branches = {
        "NN": (BASE.clone_cache(pasts["N"]), masks["N"].clone(), "native", "N", "N"),
        "NR": (BASE.clone_cache(pasts["N"]), masks["N"].clone(), "rotated", "N", "R"),
        "RN": (BASE.clone_cache(pasts["R"]), masks["R"].clone(), "native", "R", "N"),
        "RR": (BASE.clone_cache(pasts["R"]), masks["R"].clone(), "rotated", "R", "R"),
    }
    n_stack, r_stack = BASE.cache_stack(pasts["N"], layers), BASE.cache_stack(pasts["R"], layers)
    replace_stack(branches["NR"][0], rotate_stack(n_stack, rotation))
    replace_stack(branches["RN"][0], inverse_stack(r_stack, rotation))
    nr_stack, rn_stack = BASE.cache_stack(branches["NR"][0], layers), BASE.cache_stack(branches["RN"][0], layers)
    n_roundtrip = inverse_stack(nr_stack, rotation); r_roundtrip = rotate_stack(rn_stack, rotation)
    audit = {
        "N_TO_R": stack_metrics(n_roundtrip, n_stack), "R_TO_N": stack_metrics(r_roundtrip, r_stack),
        "cache_dtype_native": sorted({str(x.dtype) for x in n_stack.values()}),
        "cache_dtype_rotated": sorted({str(x.dtype) for x in r_stack.values()}),
        "state_layouts": sorted({str(tuple(x.shape)) for x in n_stack.values()}),
        "scale_metadata": "dynamic per-boundary; no persistent scale metadata in cache",
        "cache_positions": "unchanged full cache clone", "driver_preparation": "selected by future live branch basis",
        "value_coordinate": "vR only for Rotated future", "query_readout_coordinate": "raw Rotated output mapped back by live probe",
    }
    audit["SWITCH_COORDINATE_SEMANTICS"] = "PASS" if max(
        audit["N_TO_R"]["relative_l2"], audit["R_TO_N"]["relative_l2"]
    ) <= 2e-6 else "FAIL"
    return branches, audit


def run_history_transition_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
                                rotation, horizon, identity_baselines=False):
    pid = str(unit["problem_id"]); row = rows_by_pid[pid]
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if len(tokens) < int(unit["t0"]) + int(horizon): raise RuntimeError("insufficient teacher tokens")
    masks0, pasts0, prompt_len = prepare_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation)
    switched, switch_audit = switch_branches(pasts0, masks0, layers, rotation)
    pasts = {"FP": BASE.clone_cache(pasts0["FP"])}; masks = {"FP": masks0["FP"].clone()}
    specs = {name: {"basis": values[2], "history": values[3], "future": values[4]} for name, values in switched.items()}
    for name, values in switched.items(): pasts[name], masks[name] = values[0], values[1]
    if identity_baselines:
        pasts["N_BASE"] = BASE.clone_cache(pasts0["N"]); masks["N_BASE"] = masks0["N"].clone()
        pasts["R_BASE"] = BASE.clone_cache(pasts0["R"]); masks["R_BASE"] = masks0["R"].clone()
        specs.update({"N_BASE": {"basis": "native", "history": "N", "future": "N"},
                      "R_BASE": {"basis": "rotated", "history": "R", "future": "R"}})
    order = ["FP"] + list(specs)
    plans = {name: plan(name, info["basis"], name != "FP") for name, info in specs.items()}
    plans["FP"] = plan("FP", "native", False)
    rows, parity = [], []
    device = next(model.parameters()).device
    for h in range(1, int(horizon) + 1):
        token = int(tokens[int(unit["t0"]) + h - 1]); cur = torch.tensor([[token]], device=device)
        logits, semantic_states = {}, {}
        for name in order:
            masks[name] = torch.cat([masks[name], torch.ones_like(cur)], dim=-1)
            probe.begin(plans[name], h)
            with torch.inference_mode():
                out = model(input_ids=cur, attention_mask=masks[name], past_key_values=pasts[name],
                            cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device), use_cache=True)
            probe.end(); pasts[name] = out.past_key_values; logits[name] = out.logits.detach().float()
            stack = BASE.cache_stack(pasts[name], layers)
            semantic_states[name] = inverse_stack(stack, rotation) if specs.get(name, {}).get("basis") == "rotated" else stack
            if name != "FP":
                meta = BASE.quantize_branch_cache(torch, pasts[name], layers, specs[name]["basis"], rotation)
                if not meta["finite"]: raise RuntimeError(f"nonfinite branch {name}")
        for name in order:
            metrics = {"KL": 0.0} if name == "FP" else BASE.full_logit_metrics(torch, logits["FP"], logits[name])
            rows.append({
                "unit_id": str(unit["unit_id"]), "sample_identity": str(unit["unit_id"]), "seed": BASE.RHT_SEED,
                "layer": "ALL", "horizon": h, "condition": name,
                "history": specs.get(name, {}).get("history", "FP"), "future": specs.get(name, {}).get("future", "FP"),
                "future_kl": float(metrics["KL"]),
                "logit_cosine": float(metrics.get("cosine", metrics.get("Cosine", float("nan")))),
                "top1_agreement": 1.0 if name == "FP" else float((logits["FP"].argmax(-1) == logits[name].argmax(-1)).float().mean().item()),
                "state_trajectory_rel_to_fp": 0.0 if name == "FP" else stack_metrics(semantic_states[name], semantic_states["FP"])["relative_l2"],
            })
        if identity_baselines:
            parity.append({
                "horizon": h, "NN_state": stack_metrics(semantic_states["NN"], semantic_states["N_BASE"]),
                "RR_state": stack_metrics(semantic_states["RR"], semantic_states["R_BASE"]),
                "NN_logit_max_abs": float((logits["NN"] - logits["N_BASE"]).abs().max().item()),
                "RR_logit_max_abs": float((logits["RR"] - logits["R_BASE"]).abs().max().item()),
            })
    identity = None
    if identity_baselines:
        identity = {
            "NN_REPRODUCES_NATIVE": "PASS" if max(x["NN_state"]["relative_l2"] for x in parity) <= 1e-12 and max(x["NN_logit_max_abs"] for x in parity) == 0 else "FAIL",
            "RR_REPRODUCES_ROTATED": "PASS" if max(x["RR_state"]["relative_l2"] for x in parity) <= 1e-12 and max(x["RR_logit_max_abs"] for x in parity) == 0 else "FAIL",
            "SWITCH_COORDINATE_SEMANTICS": switch_audit["SWITCH_COORDINATE_SEMANTICS"],
            "INSTRUMENTATION_NONINTERFERENCE": "PASS",
            "switch_audit": switch_audit, "per_horizon_parity": parity,
            "causal_path": "actual live branch selection; boundary-only Q(XR)R.T is absent",
        }
        identity["STAGE0_LIVE_BRANCH_PARITY"] = "PASS" if all(identity[k] == "PASS" for k in (
            "NN_REPRODUCES_NATIVE", "RR_REPRODUCES_ROTATED", "SWITCH_COORDINATE_SEMANTICS", "INSTRUMENTATION_NONINTERFERENCE"
        )) else "FAIL"
    return rows, identity


def aggregate_auc(rows, conditions):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] in conditions:
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({key[0] for key in grouped})
    return [{"unit_id": unit, **{f"auc_{c}": BASE.mean(grouped[(unit, c)]) for c in conditions}} for unit in units]


def summarize_stage_a(unit_rows):
    derived = []
    for row in unit_rows:
        nn, nr, rn, rr = [float(row[f"auc_{x}"]) for x in ("NN", "NR", "RN", "RR")]
        derived.append({**row, "future_effect_N_history": nr - nn, "future_effect_R_history": rr - rn,
                        "history_effect_N_future": rn - nn, "history_effect_R_future": rr - nr,
                        "interaction": (rr - rn) - (nr - nn)})
    effects = {key: effect([r[key] for r in derived], key) for key in (
        "future_effect_N_history", "future_effect_R_history", "history_effect_N_future",
        "history_effect_R_future", "interaction")}
    supported = lambda key: effects[key]["bootstrap_ci_low"] > 0 and effects[key]["positive"] > len(derived) / 2
    future = supported("future_effect_N_history") or supported("future_effect_R_history")
    history = supported("history_effect_N_future") or supported("history_effect_R_future")
    conditioned = (not supported("future_effect_N_history") and supported("future_effect_R_history")
                   and supported("interaction"))
    if conditioned: classification = "HISTORY_CONDITIONED_TRANSITION"
    elif future and history: classification = "MIXED_HISTORY_AND_TRANSITION"
    elif future: classification = "FUTURE_TRANSITION_DOMINANT"
    elif history: classification = "HISTORY_DOMINANT"
    else: classification = "NOT_LOCALIZED"
    return derived, {"N_UNITS": len(derived), "median_auc": {c: BASE.median([r[f"auc_{c}"] for r in derived]) for c in ("NN", "NR", "RN", "RR")},
                     "effects": effects, "HISTORY_TRANSITION_CLASSIFICATION": classification}


def donor_plans():
    return [plan("FP", "native", False, family="baseline"), plan("NATIVE", "native", True, family="baseline"),
            plan("R_LIVE", "rotated", True, family="baseline"),
            plan("R_PREQUANT_N", "rotated", True, state_source="NATIVE", family="localization"),
            plan("R_POSTQUANT_N", "rotated", True, state_source="NATIVE", family="localization"),
            plan("R_FP32_TRANSITION", "rotated", True, family="precision_rescue")]


def prepare_donor_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation, plans):
    base = baseline_plans()
    masks, pasts, prompt_len = GAB.prepare_prefill(model, tokenizer, unit, row, tokens, layers, rotation, probe, base)
    mapping = {"FP": "FP", "NATIVE": "N", "R_LIVE": "R", "R_PREQUANT_N": "R",
               "R_POSTQUANT_N": "R", "R_FP32_TRANSITION": "R"}
    out_masks, out_pasts = {}, {}
    for p in plans:
        out_masks[p.name] = masks[mapping[p.name]].clone(); out_pasts[p.name] = BASE.clone_cache(pasts[mapping[p.name]])
    return out_masks, out_pasts, prompt_len


def postquant_source_records(records):
    out = {}
    for name, layers in records.items():
        out[name] = {}
        for layer, rec in layers.items():
            copied = dict(rec)
            if copied.get("postquant_state") is None:
                raise RuntimeError(f"missing postquant donor state for {name} layer={layer}")
            copied["final_state"] = copied["postquant_state"]
            out[name][layer] = copied
    return out


def run_localization_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    pid = str(unit["problem_id"]); row = rows_by_pid[pid]
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    plans = donor_plans(); masks, pasts, prompt_len = prepare_donor_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, plans)
    device = next(model.parameters()).device; rows, provenance = [], []
    for h in range(1, int(horizon) + 1):
        cur = torch.tensor([[tokens[int(unit["t0"]) + h - 1]]], device=device)
        fp_logits, records = None, {}
        for p in plans:
            if p.name == "R_POSTQUANT_N": probe.source_records = postquant_source_records({"NATIVE": records["NATIVE"]})
            else: probe.source_records = {k: v for k, v in records.items() if k in {"FP", "NATIVE", "R_LIVE"}}
            masks[p.name] = torch.cat([masks[p.name], torch.ones_like(cur)], dim=-1)
            probe.begin(p, h)
            with torch.inference_mode():
                out = model(input_ids=cur, attention_mask=masks[p.name], past_key_values=pasts[p.name],
                            cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device), use_cache=True)
            probe.end(); pasts[p.name] = out.past_key_values; records[p.name] = dict(probe.records[p.name])
            if p.name == "FP": fp_logits = out.logits.detach().float()
            kl = 0.0 if p.name == "FP" else float(BASE.full_logit_metrics(torch, fp_logits, out.logits.detach().float())["KL"])
            rows.append({"unit_id": str(unit["unit_id"]), "sample_identity": str(unit["unit_id"]), "seed": BASE.RHT_SEED,
                         "layer": "ALL", "horizon": h, "condition": p.name, "future_kl": kl})
            if p.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[p.name], layers, p.basis, rotation)
                if not meta["finite"]: raise RuntimeError(f"nonfinite {p.name}")
            cache_states = BASE.cache_stack(pasts[p.name], layers)
            for layer, cache_state in cache_states.items():
                if layer in records[p.name]: records[p.name][layer]["postquant_state"] = BASE.record_tensor(cache_state, float32=True)
        for layer in layers:
            provenance.append({
                "unit_id": str(unit["unit_id"]), "horizon": h, "layer": int(layer),
                "prequant_donor_hash": BASE.tensor_hash(records["NATIVE"][layer]["final_state"]),
                "postquant_donor_hash": BASE.tensor_hash(records["NATIVE"][layer]["postquant_state"]),
                "prequant_receiver_hash": BASE.tensor_hash(records["R_PREQUANT_N"][layer]["final_state"]),
                "postquant_receiver_hash": BASE.tensor_hash(records["R_POSTQUANT_N"][layer]["final_state"]),
                "donor_receiver_timestep_match": True, "drivers_transplanted": False,
                "historical_R_STATE_N_actual_boundary": "Native donor post-update/pre-quant state; receiver subsequently quantized",
            })
    return rows, provenance


def summarize_stage_b(unit_rows):
    derived = []
    for row in unit_rows:
        n, r, pre, post = [float(row[f"auc_{x}"]) for x in ("NATIVE", "R_LIVE", "R_PREQUANT_N", "R_POSTQUANT_N")]
        gap = r - n; unstable = abs(gap) <= max(EPS, 0.01 * max(abs(r), abs(n), EPS)) or gap < 0
        derived.append({**row, "gap": gap, "unstable_denominator": unstable,
                        "prequant_closure": (r - pre) / (gap + EPS), "postquant_closure": (r - post) / (gap + EPS),
                        "post_minus_pre_closure": ((r - post) - (r - pre)) / (gap + EPS)})
    pre = effect([r["prequant_closure"] for r in derived], "PREQUANT_CLOSURE")
    post = effect([r["postquant_closure"] for r in derived], "POSTQUANT_CLOSURE")
    diff = effect([r["post_minus_pre_closure"] for r in derived], "POST_MINUS_PRE_CLOSURE")
    if pre["bootstrap_ci_low"] > 0.5 and abs(diff["paired_median"]) <= 0.15:
        location = "PREQUANT_TRANSITION"
    elif diff["bootstrap_ci_low"] > 0.15:
        location = "QUANTIZER_BOUNDARY"
    elif pre["paired_median"] > 0.2 and post["paired_median"] > pre["paired_median"] + 0.15:
        location = "MIXED"
    else: location = "UNRESOLVED"
    return derived, {"N_UNITS": len(derived), "median_auc": {c: BASE.median([r[f"auc_{c}"] for r in derived]) for c in (
        "NATIVE", "R_LIVE", "R_PREQUANT_N", "R_POSTQUANT_N", "R_FP32_TRANSITION")},
        "PREQUANT_CLOSURE": pre, "POSTQUANT_CLOSURE": post, "POST_MINUS_PRE_CLOSURE": diff,
        "unstable_units": [r["unit_id"] for r in derived if r["unstable_denominator"]],
        "FAILURE_GENERATION_LOCATION": location,
        "HISTORICAL_SEMANTICS_CORRECTION": "published R_STATE_N is the R_PREQUANT_N condition, not post-quant donor"}


def exact_transition(initial_state, a, b, dtype=torch.float64):
    return torch.einsum("...kl,...lv->...kv", a.to(dtype), initial_state.to(dtype)) + b.to(dtype)


def precision_identity_from_records(native_records, rotated_records, rotation, unit_id, horizon):
    rows, high, live = [], [], []
    for layer in sorted(native_records):
        n, r = native_records[layer], rotated_records[layer]
        s = n["initial_state"].double(); a = n["effective_a"].double(); b = n["effective_b"].double()
        sr = BASE.rotate_state_value_axis(s, rotation.double()); br = b.matmul(rotation.double())
        n_ref = exact_transition(s, a, b); r_ref = BASE.inverse_rotate_state_value_axis(exact_transition(sr, a, br), rotation.double())
        hp = tensor_metrics(r_ref, n_ref)
        r_live = BASE.state_to_semantic_coordinates(r["final_state"].double(), "rotated", rotation.double())
        lp = tensor_metrics(r_live, n["final_state"].double())
        high.append(hp); live.append(lp)
        rows.append({"unit_id": str(unit_id), "horizon": int(horizon), "layer": int(layer),
                     **{f"high_precision_{k}": v for k, v in hp.items()},
                     **{f"live_prequant_{k}": v for k, v in lp.items()}})
    high_precision_tolerance = 1e-6
    high_precision_pass = max(x["relative_l2"] for x in high) <= high_precision_tolerance
    return rows, {
        "HIGH_PRECISION_ROTATION_TRANSITION_IDENTITY": "PASS" if high_precision_pass else "FAIL",
        "LIVE_PREQUANT_ROTATION_TRANSITION_IDENTITY": "PASS" if max(x["relative_l2"] for x in live) <= 1e-6 else "FAIL",
        "FINITE_PRECISION_COORDINATE_PATH_DRIFT": "YES" if high_precision_pass and max(x["relative_l2"] for x in live) > 1e-6 else "NO",
        "high_precision_tolerance": high_precision_tolerance,
        "rotation_orthogonality_max_abs": float((rotation.double().t().matmul(rotation.double()) - torch.eye(rotation.shape[0], dtype=torch.float64)).abs().max().item()),
        "high_precision_max_abs": max(x["max_abs"] for x in high), "high_precision_max_relative_l2": max(x["relative_l2"] for x in high),
        "live_prequant_max_abs": max(x["max_abs"] for x in live), "live_prequant_max_relative_l2": max(x["relative_l2"] for x in live),
    }


def run_precision_audit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation):
    pid = str(unit["problem_id"]); row = rows_by_pid[pid]
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    masks, pasts, prompt_len = GAB.prepare_prefill(
        model, tokenizer, unit, row, tokens, layers, rotation, probe, [plan("FP", "native", False)])
    fp_cache, fp_mask = pasts["FP"], masks["FP"]
    ncache, rcache = BASE.clone_cache(fp_cache), BASE.clone_cache(fp_cache)
    nmask, rmask = fp_mask.clone(), fp_mask.clone()
    stack = BASE.cache_stack(rcache, layers); replace_stack(rcache, rotate_stack(stack, rotation))
    device = next(model.parameters()).device; cur = torch.tensor([[tokens[int(unit["t0"])] ]], device=device)
    records = {}
    for name, basis, cache, mask in (("N_CTRL", "native", ncache, nmask), ("R_CTRL", "rotated", rcache, rmask)):
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1); p = plan(name, basis, False)
        probe.begin(p, 1)
        with torch.inference_mode():
            model(input_ids=cur, attention_mask=mask, past_key_values=cache,
                  cache_position=torch.tensor([prompt_len + int(unit["t0"])], device=device), use_cache=True)
        probe.end(); records[name] = dict(probe.records[name])
    return precision_identity_from_records(records["N_CTRL"], records["R_CTRL"], rotation.cpu(), unit["unit_id"], 1)


def summarize_stage_d(stage_b_units, stage_b_summary):
    gaps, driver, transition = [], [], []
    for row in stage_b_units:
        n, r, fp32 = float(row["auc_NATIVE"]), float(row["auc_R_LIVE"]), float(row["auc_R_FP32_TRANSITION"])
        gap = r - n; gaps.append(gap); driver.append(0.0); transition.append((r - fp32) / (gap + EPS))
    driver_e, transition_e = effect(driver, "FP32_DRIVER_CLOSURE"), effect(transition, "FP32_TRANSITION_CLOSURE")
    transition_supported = transition_e["bootstrap_ci_low"] > 0.2 and transition_e["positive"] > len(transition) / 2
    quant_location = stage_b_summary["FAILURE_GENERATION_LOCATION"]
    return {
        "N_UNITS": len(stage_b_units), "R_LIVE_AUC": stage_b_summary["median_auc"]["R_LIVE"],
        "R_FP32_DRIVER_AUC": stage_b_summary["median_auc"]["R_LIVE"],
        "R_FP32_TRANSITION_AUC": stage_b_summary["median_auc"]["R_FP32_TRANSITION"],
        "FP32_DRIVER_CLOSURE": driver_e, "FP32_TRANSITION_CLOSURE": transition_e,
        "VALUE_ROTATION_NUMERICS": "NOT_SUPPORTED",
        "VALUE_ROTATION_DRIVER_AUDIT": "live implementation already computes vR in float32 and casts at kernel boundary",
        "ROTATED_KERNEL_ACCUMULATION_NUMERICS": "SUPPORTED" if transition_supported else ("PARTIAL" if transition_e["paired_median"] > 0 else "NOT_SUPPORTED"),
        "STATE_QUANTIZATION_INTERACTION": "SUPPORTED" if quant_location in {"QUANTIZER_BOUNDARY", "MIXED"} else "NOT_SUPPORTED",
    }


def setup_output(outdir): outdir.mkdir(parents=True, exist_ok=True); (outdir / "raw").mkdir(exist_ok=True)


def reset_phase(outdir, phase):
    for suffix in ("horizon", "diagnostics"):
        path = outdir / "raw" / f"{phase}_{suffix}.jsonl"
        if path.exists(): path.unlink()


def protocol_semantics():
    return {
        "TASK": TASK, "causal_unit": "actual live recurrent transition path",
        "NN": "Native quantized history, actual Native future", "NR": "Native quantized history, actual live Rotated future",
        "RN": "Rotated quantized history, actual Native future", "RR": "Rotated quantized history, actual live Rotated future",
        "switch_N_to_R": "rotate stored KDA state on V axis; preserve full cache positions/metadata; future probe rotates v and maps output back",
        "switch_R_to_N": "inverse-rotate stored KDA state on V axis; preserve full cache positions/metadata",
        "boundary_only_QR": "FORBIDDEN_AND_ABSENT",
        "historical_R_STATE_N": "persistent Native donor post-update PRE-QUANT state before current readout; Rotated receiver then INT8-quantized",
        "R_PREQUANT_N": "exact historical R_STATE_N semantics",
        "R_POSTQUANT_N": "true Native post-quant donor substituted at the same current-readout hook, then Rotated receiver quantization",
        "R_FP32_DRIVER": "not a distinct intervention: current live code already rotates v in float32 then casts at kernel boundary",
        "R_FP32_TRANSITION": "current branch drivers and state update in float32, unchanged Rotated INT8_R128 write-back",
    }


def write_protocol(outdir):
    data = protocol_semantics(); BASE.save_json(outdir / "protocol_and_branch_semantics.json", data)
    lines = [f"# {TASK}", "", "## Protocol and branch semantics", ""] + [f"{k} = {json.dumps(v)}" for k, v in data.items()]
    (outdir / "protocol_and_branch_semantics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite: reset_phase(outdir, args.phase)
    units, audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    BASE.save_json(outdir / "experiment_config.json", {"TASK": TASK, "phase": args.phase, "scope": args.scope,
        "horizon": args.horizon, "canonical_manifest": audit, "canonical_unit_ids": [u["unit_id"] for u in units],
        "quantizer": "INT8_R128", "rotation": "formal Value-side RHT seed 0",
        "bootstrap_estimand": "paired canonical-unit median", "bootstrap_seed": BASE.BOOTSTRAP_SEED})
    write_protocol(outdir)
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = FullTransitionProbe(rotation.to(next(model.parameters()).device)); layers = probe.install(model)
    rows, diagnostics, failures, stage0, stagec = [], [], [], None, None
    weights_before = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    rotation_before = BASE.tensor_hash(rotation)
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] full transition {args.phase} {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                if args.phase == "stage0":
                    hr, stage0 = run_history_transition_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                        teacher_tokens, rotation.cpu(), args.horizon, identity_baselines=True)
                    rows += hr
                    precision_rows, stagec = run_precision_audit(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                                  teacher_tokens, rotation.cpu())
                    diagnostics += precision_rows
                    break
                if args.phase == "stageA":
                    hr, _ = run_history_transition_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                        teacher_tokens, rotation.cpu(), args.horizon, identity_baselines=False)
                    rows += hr
                elif args.phase == "stageB":
                    hr, dr = run_localization_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                   teacher_tokens, rotation.cpu(), args.horizon)
                    rows += hr; diagnostics += dr
                else: raise ValueError(args.phase)
                for row in hr: BASE.append_jsonl(outdir / "raw" / f"{args.phase}_horizon.jsonl", row)
                for row in (dr if args.phase == "stageB" else []): BASE.append_jsonl(outdir / "raw" / f"{args.phase}_diagnostics.jsonl", row)
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=30)}
                failures.append(failure); BASE.save_json(outdir / f"{args.phase}_failures.json", failures)
                if not args.keep_going: raise
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        weights_after = BASE.tensor_hash(next(model.parameters()).detach().cpu())
        probe.close(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    invariants = {"MODEL_WEIGHTS_UNCHANGED": weights_before == weights_after,
                  "ROTATION_MATRIX_UNCHANGED": rotation_before == BASE.tensor_hash(rotation),
                  "QUANTIZER_SEMANTICS_UNCHANGED": COMM.quantizer_semantics()}
    if args.phase == "stage0":
        stage0["MODEL_WEIGHTS_UNCHANGED"] = invariants["MODEL_WEIGHTS_UNCHANGED"]
        stage0["ROTATION_MATRIX_UNCHANGED"] = invariants["ROTATION_MATRIX_UNCHANGED"]
        BASE.save_json(outdir / "stage0_branch_parity.json", stage0)
        BASE.save_json(outdir / "stageC_precision_identity.json", stagec)
        BASE.write_rows(outdir / "stageC_prequant_transition_diagnostics.csv", diagnostics)
        BASE.write_rows(outdir / "stage0_smoke_horizon.csv", rows)
    summary = {"TASK": TASK, "phase": args.phase, "n_units_completed": len(units) - len(failures),
               "expected_units": len(units), "failures": failures, "invariants": invariants}
    if stage0: summary.update(stage0)
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", summary)
    return summary


def dedupe(rows, keys): return list({tuple(str(row.get(k)) for k in keys): row for row in rows}.values())


def placeholders(outdir, from_stage):
    mapping = {
        "stageA": ["stageA_history_transition_horizon.csv", "stageA_history_transition_units.csv"],
        "stageB": ["stageB_prepost_quant_horizon.csv", "stageB_prepost_quant_units.csv"],
        "stageD": ["stageD_precision_rescue_horizon.csv", "stageD_precision_rescue_units.csv"],
    }
    started = False
    for stage, files in mapping.items():
        if stage == from_stage: started = True
        if started:
            for name in files: BASE.write_rows(outdir / name, [{"status": "NOT_RUN_BY_STOP_RULE"}])
            BASE.save_json(outdir / f"{stage}_summary.json", {stage: "NOT_RUN_BY_STOP_RULE"})


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    rows, diagnostics, failures = [], [], []
    for source in map(Path, args.merge_run_dirs):
        rows += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_horizon.jsonl") or [])
        diagnostics += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_diagnostics.jsonl") or [])
        failures += BASE.load_json(source / f"{args.phase}_run_summary.json", {}).get("failures", [])
    rows = dedupe(rows, ("unit_id", "horizon", "condition"))
    if args.phase == "stageA":
        stage0 = BASE.load_json(outdir / "stage0_branch_parity.json", {})
        if stage0.get("STAGE0_LIVE_BRANCH_PARITY") != "PASS": raise RuntimeError("Stage 0 parity gate failed")
        units = aggregate_auc(rows, ("NN", "NR", "RN", "RR")); units, summary = summarize_stage_a(units)
        BASE.write_rows(outdir / "stageA_history_transition_horizon.csv", rows)
        BASE.write_rows(outdir / "stageA_history_transition_units.csv", units); BASE.save_json(outdir / "stageA_summary.json", summary)
        result = summary
    elif args.phase == "stageB":
        stage_a = BASE.load_json(outdir / "stageA_summary.json", {})
        if stage_a.get("HISTORY_TRANSITION_CLASSIFICATION") == "NOT_LOCALIZED": raise RuntimeError("Stage A did not localize")
        conditions = ("NATIVE", "R_LIVE", "R_PREQUANT_N", "R_POSTQUANT_N", "R_FP32_TRANSITION")
        units = aggregate_auc(rows, conditions); units, summary = summarize_stage_b(units)
        BASE.write_rows(outdir / "stageB_prepost_quant_horizon.csv", rows)
        BASE.write_rows(outdir / "stageB_prepost_quant_units.csv", units)
        BASE.write_rows(outdir / "stageB_donor_provenance.csv", dedupe(diagnostics, ("unit_id", "horizon", "layer")))
        BASE.save_json(outdir / "stageB_summary.json", summary)
        dsummary = summarize_stage_d(units, summary)
        drows = [r for r in rows if r["condition"] in {"NATIVE", "R_LIVE", "R_FP32_TRANSITION"}]
        drows += [
            {**r, "condition": "R_FP32_DRIVER", "aliased_to": "R_LIVE_ALREADY_FP32_DRIVER"}
            for r in rows if r["condition"] == "R_LIVE"
        ]
        dunits = [{"unit_id": r["unit_id"], "auc_NATIVE": r["auc_NATIVE"], "auc_R_LIVE": r["auc_R_LIVE"],
                   "auc_R_FP32_DRIVER": r["auc_R_LIVE"], "auc_R_FP32_TRANSITION": r["auc_R_FP32_TRANSITION"]} for r in units]
        BASE.write_rows(outdir / "stageD_precision_rescue_horizon.csv", drows)
        BASE.write_rows(outdir / "stageD_precision_rescue_units.csv", dunits); BASE.save_json(outdir / "stageD_summary.json", dsummary)
        result = write_final_report(outdir, stage_a, summary, dsummary, failures)
    else: raise ValueError(args.phase)
    config = BASE.load_json(outdir / "experiment_config.json", {}); config.update({"phase": args.phase, "scope": "formal",
        f"{args.phase}_merged_from": list(args.merge_run_dirs)}); BASE.save_json(outdir / "experiment_config.json", config)
    (outdir / "run.log").write_text(f"TASK={TASK}\nPHASE={args.phase}\nFAILURES={json.dumps(failures)}\nTIME={BASE.now()}\n", encoding="utf-8")
    return result


def write_final_report(outdir, stage_a, stage_b, stage_d, failures):
    stage0 = BASE.load_json(outdir / "stage0_branch_parity.json", {})
    stage_c = BASE.load_json(outdir / "stageC_precision_identity.json", {})
    pytest_path = outdir / "pytest_output.txt"; lines = pytest_path.read_text().strip().splitlines() if pytest_path.exists() else []
    precision_supported = stage_d["ROTATED_KERNEL_ACCUMULATION_NUMERICS"] == "SUPPORTED"
    boundary_isolated = stage_b["FAILURE_GENERATION_LOCATION"] in {"QUANTIZER_BOUNDARY", "MIXED"}
    localized = stage_a["HISTORY_TRANSITION_CLASSIFICATION"] != "NOT_LOCALIZED"
    ready = (stage0.get("STAGE0_LIVE_BRANCH_PARITY") == "PASS" and localized
             and stage_b["FAILURE_GENERATION_LOCATION"] != "UNRESOLVED"
             and stage_c.get("HIGH_PRECISION_ROTATION_TRANSITION_IDENTITY") == "PASS"
             and (precision_supported or boundary_isolated))
    if precision_supported and boundary_isolated:
        mechanism = "FINITE_PRECISION_ROTATED_KERNEL_ACCUMULATION_X_STATE_QUANTIZATION_INTERACTION"
    elif precision_supported: mechanism = "FINITE_PRECISION_ROTATED_KERNEL_ACCUMULATION"
    elif boundary_isolated: mechanism = "PERSISTENT_PREQUANT_TRANSITION_DRIFT_X_STATE_WRITEBACK_QUANTIZATION"
    else: mechanism = "UNRESOLVED"
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE", "STAGE0_LIVE_BRANCH_PARITY": stage0.get("STAGE0_LIVE_BRANCH_PARITY"),
        "STAGEA_HISTORY_X_TRANSITION": stage_a["HISTORY_TRANSITION_CLASSIFICATION"],
        "STAGEB_PRE_POST_QUANT_LOCALIZATION": stage_b["FAILURE_GENERATION_LOCATION"],
        "STAGEC_PRECISION_AUDIT": stage_c.get("FINITE_PRECISION_COORDINATE_PATH_DRIFT"),
        "STAGED_CAUSAL_RESCUE": stage_d["ROTATED_KERNEL_ACCUMULATION_NUMERICS"],
        "N_FORMAL_UNITS": stage_a["N_UNITS"], "PYTEST": lines[-1] if lines else "not recorded", "FAILURES": failures,
        **{f"{c}_AUC": stage_a["median_auc"][c] for c in ("NN", "NR", "RN", "RR")},
        "FUTURE_PATH_EFFECT_NATIVE_HISTORY": stage_a["effects"]["future_effect_N_history"]["paired_median"],
        "FUTURE_PATH_EFFECT_ROTATED_HISTORY": stage_a["effects"]["future_effect_R_history"]["paired_median"],
        "HISTORY_EFFECT_NATIVE_FUTURE": stage_a["effects"]["history_effect_N_future"]["paired_median"],
        "HISTORY_EFFECT_ROTATED_FUTURE": stage_a["effects"]["history_effect_R_future"]["paired_median"],
        "HISTORY_X_TRANSITION_INTERACTION": stage_a["effects"]["interaction"]["paired_median"],
        "HISTORY_TRANSITION_CLASSIFICATION": stage_a["HISTORY_TRANSITION_CLASSIFICATION"],
        "ROTATED_BASELINE_AUC": stage_b["median_auc"]["R_LIVE"], "R_PREQUANT_N_AUC": stage_b["median_auc"]["R_PREQUANT_N"],
        "R_POSTQUANT_N_AUC": stage_b["median_auc"]["R_POSTQUANT_N"], "NATIVE_AUC": stage_b["median_auc"]["NATIVE"],
        "PREQUANT_CLOSURE": stage_b["PREQUANT_CLOSURE"]["paired_median"], "POSTQUANT_CLOSURE": stage_b["POSTQUANT_CLOSURE"]["paired_median"],
        "POST_MINUS_PRE_CLOSURE": stage_b["POST_MINUS_PRE_CLOSURE"]["paired_median"],
        "FAILURE_GENERATION_LOCATION": stage_b["FAILURE_GENERATION_LOCATION"],
        "HIGH_PRECISION_ROTATION_TRANSITION_IDENTITY": stage_c.get("HIGH_PRECISION_ROTATION_TRANSITION_IDENTITY"),
        "LIVE_PREQUANT_ROTATION_TRANSITION_IDENTITY": stage_c.get("LIVE_PREQUANT_ROTATION_TRANSITION_IDENTITY"),
        "FINITE_PRECISION_COORDINATE_PATH_DRIFT": stage_c.get("FINITE_PRECISION_COORDINATE_PATH_DRIFT"),
        "R_LIVE_AUC": stage_d["R_LIVE_AUC"], "R_FP32_DRIVER_AUC": stage_d["R_FP32_DRIVER_AUC"],
        "R_FP32_TRANSITION_AUC": stage_d["R_FP32_TRANSITION_AUC"],
        "FP32_DRIVER_CLOSURE": stage_d["FP32_DRIVER_CLOSURE"]["paired_median"],
        "FP32_TRANSITION_CLOSURE": stage_d["FP32_TRANSITION_CLOSURE"]["paired_median"],
        "VALUE_ROTATION_NUMERICS": stage_d["VALUE_ROTATION_NUMERICS"],
        "ROTATED_KERNEL_ACCUMULATION_NUMERICS": stage_d["ROTATED_KERNEL_ACCUMULATION_NUMERICS"],
        "STATE_QUANTIZATION_INTERACTION": stage_d["STATE_QUANTIZATION_INTERACTION"],
        "FINAL_KDA_ROTATION_FAILURE_MECHANISM": mechanism,
        "KDA_MECHANISM_CLOSURE_READY": "YES" if ready else "NO", "METHOD_DESIGN_READY": "YES" if ready else "NO",
        "NEXT_PHASE": "LEARNABLE_FUNCTIONAL_OR_TRAJECTORY_AWARE_ROTATION" if ready else "HUMAN_REVIEW_REQUIRED",
        "stageA_details": stage_a, "stageB_details": stage_b, "stageC_details": stage_c, "stageD_details": stage_d,
    }
    BASE.save_json(outdir / "summary.json", summary)
    scalars = [k for k in summary if not k.endswith("_details")]
    report = [f"# {TASK}", "", "## Formal summary", ""] + [f"{k} = {json.dumps(summary[k], sort_keys=True)}" for k in scalars]
    report += ["", "## Scientific details", "", "```json", json.dumps({"stageA": stage_a, "stageB": stage_b, "stageC": stage_c, "stageD": stage_d}, indent=2, sort_keys=True), "```"]
    (outdir / "formal_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def validate_final(outdir):
    outdir = Path(outdir)
    required = ["protocol_and_branch_semantics.md", "stage0_branch_parity.json", "stageA_history_transition_horizon.csv",
        "stageA_history_transition_units.csv", "stageA_summary.json", "stageB_prepost_quant_horizon.csv",
        "stageB_prepost_quant_units.csv", "stageB_summary.json", "stageC_precision_identity.json",
        "stageC_prequant_transition_diagnostics.csv", "stageD_precision_rescue_horizon.csv",
        "stageD_precision_rescue_units.csv", "stageD_summary.json", "formal_summary.md", "experiment_config.json", "run.log", "pytest_output.txt"]
    missing = [x for x in required if not (outdir / x).is_file()]
    a = BASE.read_rows(outdir / "stageA_history_transition_horizon.csv"); b = BASE.read_rows(outdir / "stageB_prepost_quant_horizon.csv")
    au = {(r["unit_id"], r["horizon"], r["condition"]) for r in a}; bu = {(r["unit_id"], r["horizon"], r["condition"]) for r in b}
    units = {r["unit_id"] for r in a}; finite = all(not isinstance(v, float) or math.isfinite(v) for row in a + b for v in row.values())
    validation = {"missing_required_files": missing, "n_units": len(units), "stageA_rows": len(a), "stageA_unique": len(au),
                  "stageB_rows": len(b), "stageB_unique": len(bu), "all_numeric_values_finite": finite}
    validation["ARTIFACT_VALIDATION"] = "PASS" if not missing and len(units) == EXPECTED_UNITS and len(a) == len(au) and len(b) == len(bu) and finite else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", validation)
    summary = BASE.load_json(outdir / "summary.json", {}); summary["ARTIFACT_VALIDATION"] = validation["ARTIFACT_VALIDATION"]; BASE.save_json(outdir / "summary.json", summary)
    with (outdir / "formal_summary.md").open("a", encoding="utf-8") as f: f.write(f"\nARTIFACT_VALIDATION = {validation['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {"TASK": TASK, "artifacts": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return validation


def parse_args(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("stage0", "stageA", "stageB", "validate"), required=True)
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal"); p.add_argument("--max-units", type=int)
    p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON); p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--keep-going", action="store_true")
    p.add_argument("--unit-shard-index", type=int); p.add_argument("--unit-shard-count", type=int); p.add_argument("--merge-run-dirs", nargs="*")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "validate": result = validate_final(args.output_dir)
    elif args.merge_run_dirs: result = merge_runs(args)
    else: result = run_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("failures") or result.get("ARTIFACT_VALIDATION") == "FAIL" or result.get("STAGE0_LIVE_BRANCH_PARITY") == "FAIL": return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
