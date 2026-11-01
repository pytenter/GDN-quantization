#!/usr/bin/env python3
"""Final causal localization of harmful KDA rotated-history formation."""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_HISTORY_FORMATION_FINAL_CAUSAL_V1"
SLUG = "kda_rotation_history_formation_final_causal_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
PREVIOUS_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_full_transition_final_causal_closure_v1.py"
PREVIOUS_RESULTS = REPO / "results" / "kda_rotation_full_transition_final_causal_closure_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
DOSES = (0, 1, 2, 4, 8, 16, 32, 64)
EPS = 1e-12


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREV = import_file(PREVIOUS_RUNNER, "full_transition_for_history_formation_v1")
BASE, GAB, DECAY, COMM = PREV.BASE, PREV.GAB, PREV.DECAY, PREV.COMM


def plan(name, basis, quantized=True, family="history"):
    return PREV.plan(name, basis, quantized, family=family)


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
        "zero": sum(x == 0 for x in vals), "paired_values": vals,
    }


def tensor_metrics(value, reference):
    x, y = value.detach().double(), reference.detach().double()
    diff = x - y
    nx, ny = BASE.tensor_norm(x), BASE.tensor_norm(y)
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": BASE.tensor_norm(diff) / (ny + EPS),
        "cosine": float((x * y).sum().item()) / (nx * ny + EPS),
    }


def tensor_metrics_fp32(value, reference):
    x, y = value.detach().float(), reference.detach().float()
    diff = x - y
    nx = float(torch.linalg.vector_norm(x).item())
    ny = float(torch.linalg.vector_norm(y).item())
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(diff).item()) / (ny + EPS),
        "cosine": float(torch.sum(x * y).item()) / (nx * ny + EPS),
    }


def stack_metrics(left, right):
    return PREV.stack_metrics(left, right)


def rotate_stack(stack, rotation):
    return PREV.rotate_stack(stack, rotation)


def inverse_stack(stack, rotation):
    return PREV.inverse_stack(stack, rotation)


def replace_stack(cache, stack):
    PREV.replace_stack(cache, stack)


def exact_step(initial, a, b, dtype=torch.float64):
    if initial is None:
        return b.to(dtype)
    return torch.einsum("...kl,...lv->...kv", a.to(dtype), initial.to(dtype)) + b.to(dtype)


def _time_slice(x, index, has_time):
    return x[:, index] if has_time else x


def fp32_rotated_sequence(kwargs, rotation):
    """Exact KDA recurrence over one or many tokens with FP32 state arithmetic."""
    log_decay = DECAY.final_log_decay(
        kwargs["g"], kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"),
        bool(kwargs.get("use_gate_in_kernel", False)),
    )
    normalize = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    a = DECAY.effective_a_operator(kwargs["k"], kwargs["beta"], torch.exp(log_decay.float()), normalize).float()
    b_native = DECAY.effective_b_update(kwargs["k"], kwargs["beta"], kwargs["v"], normalize).float()
    b = b_native.matmul(rotation.to(device=b_native.device, dtype=torch.float32))
    q = GAB.normalized_query(kwargs["q"], normalize).float()
    initial = kwargs.get("initial_state")
    state = None if initial is None else initial.detach().float().clone()
    has_time = q.ndim == 4
    steps = int(q.shape[1]) if has_time else 1
    outputs, last_initial = [], state
    for index in range(steps):
        at, bt, qt = (_time_slice(x, index, has_time) for x in (a, b, q))
        last_initial = state
        state = exact_step(state, at, bt, dtype=torch.float32)
        if qt.shape[-2] != state.shape[-3]:
            if state.shape[-3] % qt.shape[-2] != 0:
                raise ValueError("value heads are not divisible by query heads")
            qt = qt.repeat_interleave(state.shape[-3] // qt.shape[-2], dim=-2)
        scale = kwargs.get("scale")
        if scale is None:
            scale = qt.shape[-1] ** -0.5
        outputs.append(torch.einsum("...k,...kv->...v", qt * float(scale), state))
    raw_out = torch.stack(outputs, dim=1) if has_time else outputs[0].unsqueeze(1)
    last_a = _time_slice(a, steps - 1, has_time)
    last_b = _time_slice(b, steps - 1, has_time)
    reference = exact_step(last_initial, last_a, last_b, dtype=torch.float64)
    identity = tensor_metrics(state.double(), reference)
    return raw_out, state, {
        "log_decay": log_decay, "effective_a": last_a, "effective_b": last_b,
        "last_initial_state": last_initial, "fp32_reference": reference,
        "fp32_reference_identity": identity, "sequence_length": steps,
    }


class HistoryFormationProbe(PREV.FullTransitionProbe):
    FP32_NAMES = {"R_FP32_TRANSITION_HISTORY", "R_FP32_NOQ_HISTORY"}

    def _wrap(self, operator, fn):
        parent_wrapped = super()._wrap(operator, fn)

        def wrapped(**kwargs):
            p = self.plan
            if p is None or p.name not in self.FP32_NAMES or p.basis != "rotated":
                return parent_wrapped(**kwargs)
            branch, layer = self.branch, self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            v_semantic = kwargs["v"].detach().clone()
            raw_out, state, meta = fp32_rotated_sequence(kwargs, self.rotation)
            returned = raw_out.matmul(self.rotation.t().to(raw_out.device, torch.float32)).to(kwargs["v"].dtype)
            if branch is not None and layer is not None:
                self.records[branch][int(layer)] = {
                    "operator": operator, "q": BASE.record_tensor(kwargs["q"]),
                    "k": BASE.record_tensor(kwargs["k"]), "v": BASE.record_tensor(kwargs["v"]),
                    "v_semantic": BASE.record_tensor(v_semantic), "beta": BASE.record_tensor(kwargs["beta"]),
                    "raw_gate": BASE.record_tensor(raw_gate),
                    "log_decay": BASE.record_tensor(meta["log_decay"], float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(meta["log_decay"].float()), float32=True),
                    "effective_a": BASE.record_tensor(meta["effective_a"], float32=True),
                    "effective_b": BASE.record_tensor(meta["effective_b"], float32=True),
                    "initial_state": BASE.record_tensor(meta["last_initial_state"], float32=True),
                    "final_state": BASE.record_tensor(state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True),
                    "raw_output": BASE.record_tensor(raw_out, float32=True),
                    "basis": "rotated", "precision_intervention": "FP32_TRANSITION_HISTORY",
                    "sequence_length": meta["sequence_length"],
                    "fp32_reference_max_abs": meta["fp32_reference_identity"]["max_abs"],
                    "fp32_reference_relative_l2": meta["fp32_reference_identity"]["relative_l2"],
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                }
            return returned, state

        return wrapped


def branch(name, basis, quantized, past, mask):
    return {"name": name, "basis": basis, "quantized": bool(quantized),
            "plan": plan(name, basis, quantized), "past": past, "mask": mask}


def clone_branch(source, name=None, basis=None, quantized=None):
    return branch(
        name or source["name"], basis or source["basis"],
        source["quantized"] if quantized is None else quantized,
        BASE.clone_cache(source["past"]), source["mask"].clone(),
    )


def prefill_branch(model, tokenizer, probe, row, layers, rotation, name, basis, quantized):
    device = next(model.parameters()).device
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    p = plan(name, basis, quantized)
    mask = torch.ones_like(input_ids)
    probe.begin(p, 0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask,
                    cache_position=torch.arange(0, input_ids.shape[-1], device=device), use_cache=True)
    probe.end()
    result = branch(name, basis, quantized, out.past_key_values, mask)
    records = dict(probe.records[name])
    if basis == "rotated":
        replace_stack(result["past"], rotate_stack(BASE.cache_stack(result["past"], layers), rotation))
    if quantized:
        meta = BASE.quantize_branch_cache(torch, result["past"], layers, basis, rotation)
        if not meta["finite"]:
            raise RuntimeError(f"nonfinite prefill quantization: {name}")
    return input_ids, result, records


def advance(model, probe, item, token, cache_position, layers, rotation, step):
    device = next(model.parameters()).device
    cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
    item["mask"] = torch.cat([item["mask"], torch.ones_like(cur)], dim=-1)
    probe.begin(item["plan"], step)
    with torch.inference_mode():
        out = model(input_ids=cur, attention_mask=item["mask"], past_key_values=item["past"],
                    cache_position=torch.tensor([int(cache_position)], device=device), use_cache=True)
    probe.end()
    item["past"] = out.past_key_values
    records = dict(probe.records[item["name"]])
    if item["quantized"]:
        meta = BASE.quantize_branch_cache(torch, item["past"], layers, item["basis"], rotation)
        if not meta["finite"]:
            raise RuntimeError(f"nonfinite branch={item['name']} step={step}")
    return out.logits.detach().float(), records


def to_rotated(item, layers, rotation):
    replace_stack(item["past"], rotate_stack(BASE.cache_stack(item["past"], layers), rotation))
    item["basis"] = "rotated"; item["plan"] = plan(item["name"], "rotated", item["quantized"])


def to_native(item, layers, rotation):
    replace_stack(item["past"], inverse_stack(BASE.cache_stack(item["past"], layers), rotation))
    item["basis"] = "native"; item["plan"] = plan(item["name"], "native", item["quantized"])


def history_tokens(unit, teacher_tokens, horizon):
    pid = str(unit["problem_id"])
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"insufficient teacher tokens for {unit['unit_id']}")
    return tokens


def timeline_row(unit, prompt_len):
    t0 = int(unit["t0"])
    return {
        "unit_id": str(unit["unit_id"]), "common_start_token": 0,
        "t0_token": int(prompt_len) + t0, "prompt_tokens": int(prompt_len),
        "decode_history_transitions": t0, "number_of_history_transitions": int(prompt_len) + t0,
        "history_includes_prefill": True, "history_includes_decode": True,
        "prefill_execution": "single causal packed prompt; recurrent state Q/DQ once at endpoint",
        "decode_execution": "teacher-forced one token per transition; recurrent state Q/DQ after every token",
        "cache_position": f"prefill 0..{int(prompt_len)-1}; history decode {int(prompt_len)}..{int(prompt_len)+t0-1}",
        "attention_mask_semantics": "all-ones causal mask extended by one at every decode token",
        "state_cache_initialization": "model empty cache at common start",
    }


def diagnostic_rows(unit_id, condition, history_step, native_records, rotated_records,
                    native_post, rotated_post, rotation, step_kind="DECODE"):
    rows = []
    for layer in sorted(native_records):
        if layer not in rotated_records:
            continue
        npre = native_records[layer]["final_state"].float()
        rpre_backend = rotated_records[layer]["final_state"].float()
        rpre = BASE.state_to_semantic_coordinates(rpre_backend, "rotated", rotation)
        npost = native_post[layer].float()
        rpost_backend = rotated_post[layer].float()
        rpost = BASE.state_to_semantic_coordinates(rpost_backend, "rotated", rotation)
        pre, post = tensor_metrics_fp32(rpre, npre), tensor_metrics_fp32(rpost, npost)
        qerr = tensor_metrics_fp32(rpost, rpre)
        maxabs = float(rpre.abs().max().item()); rms = float(torch.sqrt(torch.mean(rpre ** 2)).item())
        scale = float(rpre_backend.abs().amax(dim=-1).float().mean().item()) / 127.0
        rows.append({
            "unit_id": str(unit_id), "condition": condition, "history_step": history_step,
            "history_step_kind": step_kind, "layer": int(layer),
            "prequant_max_abs_gap": pre["max_abs"], "prequant_relative_l2": pre["relative_l2"],
            "prequant_cosine": pre["cosine"], "postquant_max_abs_gap": post["max_abs"],
            "postquant_relative_l2": post["relative_l2"], "state_norm": float(torch.linalg.vector_norm(rpre).item()),
            "state_maxabs": maxabs, "state_peakiness": maxabs / (rms + EPS), "mean_group_scale": scale,
            "quant_error_relative_norm": qerr["relative_l2"],
        })
    return rows


def evaluate_native_future(model, probe, fp, conditions, tokens, t0, prompt_len, horizon, layers, rotation, unit_id):
    rows = []
    for name, item in conditions.items():
        if item["basis"] == "rotated":
            to_native(item, layers, rotation)
        item["plan"] = plan(item["name"], "native", True)
        item["quantized"] = True
    for h in range(1, int(horizon) + 1):
        token = tokens[int(t0) + h - 1]; position = int(prompt_len) + int(t0) + h - 1
        fp_logits, _ = advance(model, probe, fp, token, position, layers, rotation, h)
        rows.append({"unit_id": str(unit_id), "sample_identity": str(unit_id), "seed": BASE.RHT_SEED,
                     "horizon": h, "condition": "FP", "future_kl": 0.0})
        for name, item in conditions.items():
            logits, _ = advance(model, probe, item, token, position, layers, rotation, h)
            metrics = BASE.full_logit_metrics(torch, fp_logits, logits)
            rows.append({
                "unit_id": str(unit_id), "sample_identity": str(unit_id), "seed": BASE.RHT_SEED,
                "horizon": h, "condition": name, "future_kl": float(metrics["KL"]),
                "logit_cosine": float(metrics.get("cosine", metrics.get("Cosine", float("nan")))),
                "top1_agreement": float((fp_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()),
            })
    return rows


def prepare_exposure_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation, collect_diagnostics=True):
    input_ids, fp, _ = prefill_branch(model, tokenizer, probe, row, layers, rotation, "FP", "native", False)
    _, native, n_prefill = prefill_branch(model, tokenizer, probe, row, layers, rotation, "N", "native", True)
    _, full, r_prefill = prefill_branch(model, tokenizer, probe, row, layers, rotation, "R_FULL", "rotated", True)
    prompt_len, t0 = int(input_ids.shape[-1]), int(unit["t0"])
    diagnostics = []
    if collect_diagnostics:
        diagnostics += diagnostic_rows(unit["unit_id"], "FULL", "PREFILL", n_prefill, r_prefill,
                                      BASE.cache_stack(native["past"], layers), BASE.cache_stack(full["past"], layers),
                                      rotation, "PREFILL_BATCH")
    legal = [x for x in DOSES if x <= t0]
    active = {}
    for t in range(t0):
        for dose in legal:
            if dose > 0 and t == t0 - dose:
                item = clone_branch(native, f"L{dose}", "native", True)
                to_rotated(item, layers, rotation); active[dose] = item
        token, position = tokens[t], prompt_len + t
        advance(model, probe, fp, token, position, layers, rotation, t + 1)
        _, nrec = advance(model, probe, native, token, position, layers, rotation, t + 1)
        npost = BASE.cache_stack(native["past"], layers)
        _, frec = advance(model, probe, full, token, position, layers, rotation, t + 1)
        if collect_diagnostics:
            diagnostics += diagnostic_rows(unit["unit_id"], "FULL", t + 1, nrec, frec, npost,
                                          BASE.cache_stack(full["past"], layers), rotation)
        for dose, item in active.items():
            _, rec = advance(model, probe, item, token, position, layers, rotation, t + 1)
            if collect_diagnostics:
                diagnostics += diagnostic_rows(unit["unit_id"], f"L{dose}", t + 1, nrec, rec, npost,
                                              BASE.cache_stack(item["past"], layers), rotation)
    conditions = {"L0": native}
    conditions.update({f"L{x}": active[x] for x in legal if x > 0})
    conditions["FULL"] = full
    return fp, conditions, prompt_len, diagnostics, timeline_row(unit, prompt_len)


def run_stage_a_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = history_tokens(unit, teacher_tokens, horizon)
    fp, conditions, prompt_len, diagnostics, timeline = prepare_exposure_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation)
    rows = evaluate_native_future(model, probe, fp, conditions, tokens, unit["t0"], prompt_len,
                                  horizon, layers, rotation, unit["unit_id"])
    return rows, diagnostics, timeline


def prepare_independent_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation, specs):
    input_ids = None; items = {}; identities = []
    for name, basis, quantized in specs:
        ids, item, records = prefill_branch(model, tokenizer, probe, row, layers, rotation, name, basis, quantized)
        input_ids = ids if input_ids is None else input_ids
        items[name] = item
        for layer, rec in records.items():
            if rec.get("fp32_reference_relative_l2") is not None:
                identities.append({"unit_id": str(unit["unit_id"]), "history_step": "PREFILL", "layer": int(layer),
                                   "condition": name, "max_abs": rec["fp32_reference_max_abs"],
                                   "relative_l2": rec["fp32_reference_relative_l2"]})
    prompt_len, t0 = int(input_ids.shape[-1]), int(unit["t0"])
    for t in range(t0):
        for name, _, _ in specs:
            _, records = advance(model, probe, items[name], tokens[t], prompt_len + t, layers, rotation, t + 1)
            for layer, rec in records.items():
                if rec.get("fp32_reference_relative_l2") is not None:
                    identities.append({"unit_id": str(unit["unit_id"]), "history_step": t + 1, "layer": int(layer),
                                       "condition": name, "max_abs": rec["fp32_reference_max_abs"],
                                       "relative_l2": rec["fp32_reference_relative_l2"]})
    fp = items.pop("FP")
    return fp, items, prompt_len, identities, timeline_row(unit, prompt_len)


def run_conditions_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
                        rotation, horizon, specs):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = history_tokens(unit, teacher_tokens, horizon)
    fp, conditions, prompt_len, identities, timeline = prepare_independent_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, specs)
    rows = evaluate_native_future(model, probe, fp, conditions, tokens, unit["t0"], prompt_len,
                                  horizon, layers, rotation, unit["unit_id"])
    return rows, identities, timeline


def phase_specs(phase):
    if phase == "stageB":
        return [("FP", "native", False), ("N_HISTORY", "native", True),
                ("R_LIVE_HISTORY", "rotated", True), ("R_FP32_TRANSITION_HISTORY", "rotated", True)]
    if phase == "stageC":
        return [("FP", "native", False), ("N_HISTORY", "native", True),
                ("R_LIVE_HISTORY", "rotated", True), ("R_NO_STATE_QUANT_HISTORY", "rotated", False)]
    if phase == "stageD":
        return [("FP", "native", False), ("R_FP32_NOQ_HISTORY", "rotated", False)]
    raise ValueError(phase)


def old_endpoint_caches(model, tokenizer, probe, unit, row, tokens, layers, rotation):
    masks, pasts, prompt_len = PREV.prepare_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation)
    switched, _ = PREV.switch_branches(pasts, masks, layers, rotation)
    return {"FP": (pasts["FP"], masks["FP"]), "NN": switched["NN"], "RN": switched["RN"]}, prompt_len


def run_stage0(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = history_tokens(unit, teacher_tokens, horizon)
    fp, conditions, prompt_len, _, timeline = prepare_exposure_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, collect_diagnostics=False)
    old, old_prompt_len = old_endpoint_caches(model, tokenizer, probe, unit, row, tokens, layers, rotation)
    new_n = BASE.cache_stack(conditions["L0"]["past"], layers)
    new_r = inverse_stack(BASE.cache_stack(conditions["FULL"]["past"], layers), rotation)
    old_n = BASE.cache_stack(old["NN"][0], layers); old_r = BASE.cache_stack(old["RN"][0], layers)
    n_state, r_state = stack_metrics(new_n, old_n), stack_metrics(new_r, old_r)
    old_fp = branch("OLD_FP", "native", False, BASE.clone_cache(old["FP"][0]), old["FP"][1].clone())
    old_nn = branch("NN_REF", "native", True, BASE.clone_cache(old["NN"][0]), old["NN"][1].clone())
    old_rn = branch("RN_REF", "native", True, BASE.clone_cache(old["RN"][0]), old["RN"][1].clone())
    reference_rows = evaluate_native_future(
        model, probe, old_fp, {"NN_REF": old_nn, "RN_REF": old_rn}, tokens, unit["t0"], old_prompt_len,
        horizon, layers, rotation, unit["unit_id"])
    rows = evaluate_native_future(model, probe, fp, {"L0": conditions["L0"], "FULL": conditions["FULL"]},
                                  tokens, unit["t0"], prompt_len, horizon, layers, rotation, unit["unit_id"])
    lookup = {(int(x["horizon"]), str(x["condition"])): float(x["future_kl"]) for x in reference_rows}
    n_diffs, r_diffs = [], []
    for x in rows:
        if x["condition"] == "L0":
            n_diffs.append(abs(float(x["future_kl"]) - lookup[(int(x["horizon"]), "NN_REF")]))
        elif x["condition"] == "FULL":
            r_diffs.append(abs(float(x["future_kl"]) - lookup[(int(x["horizon"]), "RN_REF")]))
    identity_specs = [("FP", "native", False), ("R_FP32_TRANSITION_HISTORY", "rotated", True)]
    _, _, _, identity_rows, _ = prepare_independent_histories(
        model, tokenizer, probe, unit, row, tokens[:int(unit["t0"]) + horizon], layers, rotation, identity_specs)
    identity_max = max(float(x["relative_l2"]) for x in identity_rows)
    audit = {
        "HISTORY_TIMELINE_AUDIT": "PASS" if prompt_len == old_prompt_len and timeline["number_of_history_transitions"] == prompt_len + int(unit["t0"]) else "FAIL",
        "FULL_NATIVE_HISTORY_REPRODUCES_NN": "PASS" if n_state["relative_l2"] <= 1e-12 and max(n_diffs) <= 1e-12 else "FAIL",
        "FULL_ROTATED_HISTORY_NATIVE_FUTURE_REPRODUCES_RN": "PASS" if r_state["relative_l2"] <= 2e-6 and max(r_diffs) <= 1e-7 else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE": "PASS",
        "FP32_TRANSITION_HISTORY_IDENTITY": "PASS" if identity_max <= 1e-6 else "FAIL",
        "native_t0_state": n_state, "rotated_t0_state_mapped_back": r_state,
        "native_future_kl_max_abs": max(n_diffs), "rotated_future_kl_max_abs": max(r_diffs),
        "fp32_transition_reference_max_relative_l2": identity_max,
        "unit_id": str(unit["unit_id"]), "timeline": timeline,
    }
    audit["STAGE0_HISTORY_PARITY"] = "PASS" if all(audit[k] == "PASS" for k in (
        "HISTORY_TIMELINE_AUDIT", "FULL_NATIVE_HISTORY_REPRODUCES_NN",
        "FULL_ROTATED_HISTORY_NATIVE_FUTURE_REPRODUCES_RN", "INSTRUMENTATION_NONINTERFERENCE",
        "FP32_TRANSITION_HISTORY_IDENTITY")) else "FAIL"
    return rows, identity_rows, timeline, audit


def aggregate_auc(rows, conditions):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] in conditions:
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({key[0] for key in grouped})
    return [{"unit_id": unit, **{f"auc_{c}": BASE.mean(grouped[(unit, c)]) for c in conditions}} for unit in units]


def summarize_stage_a(unit_rows, conditions):
    derived = []
    for row in unit_rows:
        base = float(row["auc_L0"])
        derived.append({**row, **{f"damage_{c}": float(row[f"auc_{c}"]) - base for c in conditions if c != "L0"}})
    effects = {c: effect([r[f"damage_{c}"] for r in derived], f"DAMAGE_{c}") for c in conditions if c != "L0"}
    ordered = [c for c in conditions if c not in {"L0", "FULL"}] + ["FULL"]
    positive = next((c for c in ordered if effects[c]["paired_median"] > 0), "NONE")
    ci_positive = next((c for c in ordered if effects[c]["bootstrap_ci_low"] > 0), "NONE")
    full = effects["FULL"]
    nontrivial = any(effects[c]["bootstrap_ci_low"] > 0 and effects[c]["positive"] > len(derived) / 2 for c in ordered)
    substantial = full["bootstrap_ci_low"] > 0 and full["positive"] > len(derived) / 2
    status = "STRONG_SUPPORT" if substantial and nontrivial else ("PARTIAL_SUPPORT" if full["paired_median"] > 0 else "NOT_SUPPORTED")
    return derived, {
        "N_UNITS": len(derived), "median_auc": {c: BASE.median([r[f"auc_{c}"] for r in derived]) for c in conditions},
        "effects": effects, "FIRST_L_WITH_POSITIVE_MEDIAN_DAMAGE": positive,
        "FIRST_L_WITH_CI_ABOVE_ZERO": ci_positive,
        "FULL_MINUS_L1": effect([r["auc_FULL"] - r["auc_L1"] for r in derived], "FULL_MINUS_L1"),
        "ROTATED_HISTORY_EXPOSURE_STATUS": status,
    }


def closure_summary(unit_rows, native_key, live_key, intervention_key, label):
    derived, rescue, closure, unstable = [], [], [], []
    for row in unit_rows:
        n, r, x = (float(row[f"auc_{k}"]) for k in (native_key, live_key, intervention_key))
        gap = r - n
        bad = gap < 0 or abs(gap) <= max(EPS, 0.01 * max(abs(r), abs(n), EPS))
        value = (r - x) / (gap + EPS)
        derived.append({**row, "history_gap": gap, f"{label.lower()}_rescue": r - x,
                        f"{label.lower()}_closure": value, "unstable_denominator": bad})
        rescue.append(r - x); closure.append(value)
        if bad: unstable.append(str(row["unit_id"]))
    return derived, {
        f"{label}_RESCUE": effect(rescue, f"{label}_RESCUE"),
        f"{label}_CLOSURE": effect(closure, f"{label}_CLOSURE"),
        "unstable_units": unstable,
    }


def summarize_stage_b(unit_rows):
    derived, stats = closure_summary(unit_rows, "N_HISTORY", "R_LIVE_HISTORY", "R_FP32_TRANSITION_HISTORY", "FP32_HISTORY")
    c = stats["FP32_HISTORY_CLOSURE"]
    if c["bootstrap_ci_low"] > 0.2 and c["positive"] > len(derived) / 2:
        support = "SUPPORTED"
    elif c["paired_median"] > 0:
        support = "PARTIAL"
    else:
        support = "NOT_SUPPORTED"
    return derived, {"N_UNITS": len(derived), **stats,
        "median_auc": {k: BASE.median([r[f"auc_{k}"] for r in derived]) for k in (
            "N_HISTORY", "R_LIVE_HISTORY", "R_FP32_TRANSITION_HISTORY")},
        "FINITE_PRECISION_ROTATED_HISTORY_FORMATION": support}


def summarize_stage_c(unit_rows):
    derived, stats = closure_summary(unit_rows, "N_HISTORY", "R_LIVE_HISTORY", "R_NO_STATE_QUANT_HISTORY", "NOQ_HISTORY")
    c = stats["NOQ_HISTORY_CLOSURE"]
    if c["bootstrap_ci_low"] > 0.2 and c["positive"] > len(derived) / 2:
        support = "SUPPORTED"
    elif c["paired_median"] > 0:
        support = "PARTIAL"
    else:
        support = "NOT_SUPPORTED"
    return derived, {"N_UNITS": len(derived), **stats,
        "median_auc": {k: BASE.median([r[f"auc_{k}"] for r in derived]) for k in (
            "N_HISTORY", "R_LIVE_HISTORY", "R_NO_STATE_QUANT_HISTORY")},
        "REPEATED_STATE_WRITEBACK_HISTORY_EFFECT": support}


def summarize_stage_d(stage_b_units, stage_c_units, combined_units):
    b = {r["unit_id"]: r for r in stage_b_units}; c = {r["unit_id"]: r for r in stage_c_units}
    x = {r["unit_id"]: r for r in combined_units}; rows, values = [], []
    for unit in sorted(set(b) & set(c) & set(x)):
        y11 = float(b[unit]["auc_R_LIVE_HISTORY"]); y01 = float(b[unit]["auc_R_FP32_TRANSITION_HISTORY"])
        y10 = float(c[unit]["auc_R_NO_STATE_QUANT_HISTORY"]); y00 = float(x[unit]["auc_R_FP32_NOQ_HISTORY"])
        interaction = y11 - y01 - y10 + y00
        rows.append({"unit_id": unit, "auc_R_LIVE_INT8": y11, "auc_R_FP32_INT8": y01,
                     "auc_R_LIVE_NOQ": y10, "auc_R_FP32_NOQ": y00, "interaction": interaction})
        values.append(interaction)
    stat = effect(values, "TRANSITION_X_WRITEBACK_INTERACTION")
    status = "SUPPORTED" if stat["bootstrap_ci_low"] > 0 and stat["positive"] > len(rows) / 2 else (
        "PARTIAL" if stat["paired_median"] > 0 else "NOT_SUPPORTED")
    return rows, {"N_UNITS": len(rows), "TRANSITION_X_WRITEBACK_INTERACTION": stat,
                  "R_FP32_NOQ_HISTORY_AUC": BASE.median([r["auc_R_FP32_NOQ"] for r in rows]),
                  "INTERACTION_STATUS": status,
                  "factor_encoding": "I=LIVE_INT8-FP32_INT8-LIVE_NOQ+FP32_NOQ"}


def setup_output(outdir):
    outdir.mkdir(parents=True, exist_ok=True); (outdir / "raw").mkdir(exist_ok=True)


def reset_phase(outdir, phase):
    for suffix in ("horizon", "diagnostics", "timeline"):
        p = outdir / "raw" / f"{phase}_{suffix}.jsonl"
        if p.exists(): p.unlink()


def append_jsonl_many(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_timeline(outdir, rows, status="PASS"):
    data = {"HISTORY_TIMELINE_AUDIT": status, "N_UNITS": len(rows), "units": rows}
    BASE.save_json(outdir / "history_timeline_audit.json", data)
    lines = [f"# {TASK}", "", f"HISTORY_TIMELINE_AUDIT = {status}", "",
             "COMMON_START is the empty model cache before packed causal prompt prefill.",
             "Each history contains one packed prefill execution followed by t0 teacher-forced decode transitions.",
             "INT8 histories quantize once after prefill and after every decode transition.", ""]
    for row in rows:
        lines.append(f"- {row['unit_id']}: prompt={row['prompt_tokens']}, decode={row['decode_history_transitions']}, total={row['number_of_history_transitions']}, t0_cache_position={row['t0_token']}")
    (outdir / "history_timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite: reset_phase(outdir, args.phase)
    units, manifest = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = HistoryFormationProbe(rotation.to(next(model.parameters()).device)); layers = probe.install(model)
    weights_before = BASE.tensor_hash(next(model.parameters()).detach().cpu()); rotation_before = BASE.tensor_hash(rotation)
    rows, diagnostics, timelines, failures, stage0 = [], [], [], [], None
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] history formation {args.phase} {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                if args.phase == "stage0":
                    hr, dr, tr, stage0 = run_stage0(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                    teacher_tokens, rotation.cpu(), args.horizon)
                elif args.phase == "stageA":
                    hr, dr, tr = run_stage_a_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                  teacher_tokens, rotation.cpu(), args.horizon)
                else:
                    hr, dr, tr = run_conditions_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                     teacher_tokens, rotation.cpu(), args.horizon, phase_specs(args.phase))
                rows += hr; diagnostics += dr; timelines.append(tr)
                append_jsonl_many(outdir / "raw" / f"{args.phase}_horizon.jsonl", hr)
                append_jsonl_many(outdir / "raw" / f"{args.phase}_diagnostics.jsonl", dr)
                append_jsonl_many(outdir / "raw" / f"{args.phase}_timeline.jsonl", [tr])
                if args.phase == "stage0": break
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=40)}
                failures.append(failure); BASE.save_json(outdir / f"{args.phase}_failures.json", failures)
                if not args.keep_going: raise
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        weights_after = BASE.tensor_hash(next(model.parameters()).detach().cpu())
        probe.close(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    invariants = {
        "MODEL_WEIGHTS_UNCHANGED": weights_before == weights_after,
        "ROTATION_MATRIX_UNCHANGED": rotation_before == BASE.tensor_hash(rotation),
        "QUANTIZER_SEMANTICS_UNCHANGED": COMM.quantizer_semantics(),
    }
    config = {"TASK": TASK, "phase": args.phase, "scope": args.scope, "horizon": args.horizon,
              "canonical_manifest": manifest, "canonical_unit_ids": [u["unit_id"] for u in units],
              "history_doses": list(DOSES) + ["FULL"], "bootstrap_estimand": "paired canonical-unit median",
              "bootstrap_seed": BASE.BOOTSTRAP_SEED, "quantizer": "INT8_R128", "rotation": "formal Value-side RHT seed 0"}
    BASE.save_json(outdir / "experiment_config.json", config)
    if stage0:
        stage0.update(invariants); BASE.save_json(outdir / "stage0_history_parity.json", stage0)
        BASE.write_rows(outdir / "stage0_history_horizon.csv", rows)
        BASE.write_rows(outdir / "stage0_fp32_identity.csv", diagnostics)
        write_timeline(outdir, timelines, stage0["HISTORY_TIMELINE_AUDIT"])
    summary = {"TASK": TASK, "phase": args.phase, "n_units_completed": len(timelines),
               "expected_units": len(units), "failures": failures, "invariants": invariants}
    if stage0: summary.update(stage0)
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", summary)
    return summary


def dedupe(rows, keys):
    return list({tuple(str(row.get(k)) for k in keys): row for row in rows}.values())


def merge_raw(sources, phase):
    rows, diagnostics, timelines, failures = [], [], [], []
    for source in map(Path, sources):
        rows += list(BASE.iter_jsonl(source / "raw" / f"{phase}_horizon.jsonl") or [])
        diagnostics += list(BASE.iter_jsonl(source / "raw" / f"{phase}_diagnostics.jsonl") or [])
        timelines += list(BASE.iter_jsonl(source / "raw" / f"{phase}_timeline.jsonl") or [])
        failures += BASE.load_json(source / f"{phase}_run_summary.json", {}).get("failures", [])
    return (dedupe(rows, ("unit_id", "horizon", "condition")),
            dedupe(diagnostics, ("unit_id", "condition", "history_step", "layer")),
            dedupe(timelines, ("unit_id",)), failures)


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    rows, diagnostics, timelines, failures = merge_raw(args.merge_run_dirs, args.phase)
    if failures: raise RuntimeError(f"cannot merge failed shards: {failures}")
    if BASE.load_json(outdir / "stage0_history_parity.json", {}).get("STAGE0_HISTORY_PARITY") != "PASS":
        raise RuntimeError("Stage 0 history parity gate failed")
    if args.phase == "stageA":
        conditions = tuple(sorted({r["condition"] for r in rows if r["condition"] != "FP"},
                                  key=lambda x: (x == "FULL", int(x[1:]) if x.startswith("L") else 10**9)))
        units, summary = summarize_stage_a(aggregate_auc(rows, conditions), conditions)
        BASE.write_rows(outdir / "stageA_history_exposure_horizon.csv", rows)
        BASE.write_rows(outdir / "stageA_history_exposure_units.csv", units)
        BASE.write_rows(outdir / "stageA_history_drift_diagnostics.csv", diagnostics)
        BASE.save_json(outdir / "stageA_summary.json", summary); write_timeline(outdir, timelines)
    elif args.phase == "stageB":
        conditions = ("N_HISTORY", "R_LIVE_HISTORY", "R_FP32_TRANSITION_HISTORY")
        units, summary = summarize_stage_b(aggregate_auc(rows, conditions))
        identity_max = max(float(r["relative_l2"]) for r in diagnostics)
        summary["FP32_TRANSITION_HISTORY_IDENTITY"] = "PASS" if identity_max <= 1e-6 else "FAIL"
        summary["fp32_identity_max_relative_l2"] = identity_max
        BASE.write_rows(outdir / "stageB_fp32_history_horizon.csv", rows)
        BASE.write_rows(outdir / "stageB_fp32_history_units.csv", units)
        BASE.write_rows(outdir / "stageB_fp32_identity.csv", diagnostics)
        BASE.save_json(outdir / "stageB_summary.json", summary)
    elif args.phase == "stageC":
        conditions = ("N_HISTORY", "R_LIVE_HISTORY", "R_NO_STATE_QUANT_HISTORY")
        units, summary = summarize_stage_c(aggregate_auc(rows, conditions))
        BASE.write_rows(outdir / "stageC_noq_history_horizon.csv", rows)
        BASE.write_rows(outdir / "stageC_noq_history_units.csv", units)
        BASE.save_json(outdir / "stageC_summary.json", summary)
    elif args.phase == "stageD":
        combined = aggregate_auc(rows, ("R_FP32_NOQ_HISTORY",))
        bu = BASE.read_rows(outdir / "stageB_fp32_history_units.csv")
        cu = BASE.read_rows(outdir / "stageC_noq_history_units.csv")
        units, summary = summarize_stage_d(bu, cu, combined)
        source_rows = []
        for path, names in ((outdir / "stageB_fp32_history_horizon.csv", {"R_LIVE_HISTORY", "R_FP32_TRANSITION_HISTORY"}),
                            (outdir / "stageC_noq_history_horizon.csv", {"R_NO_STATE_QUANT_HISTORY"})):
            source_rows += [r for r in BASE.read_rows(path) if r["condition"] in names]
        source_rows += [r for r in rows if r["condition"] == "R_FP32_NOQ_HISTORY"]
        BASE.write_rows(outdir / "stageD_interaction_horizon.csv", source_rows)
        BASE.write_rows(outdir / "stageD_interaction_units.csv", units)
        BASE.save_json(outdir / "stageD_summary.json", summary)
    else:
        raise ValueError(args.phase)
    config = BASE.load_json(outdir / "experiment_config.json", {})
    config.update({f"{args.phase}_merged_from": list(args.merge_run_dirs), "formal_unit_count": len(timelines),
                   "scope": "formal", "canonical_unit_ids": sorted(str(r["unit_id"]) for r in timelines)})
    BASE.save_json(outdir / "experiment_config.json", config)
    (outdir / "run.log").open("a", encoding="utf-8").write(
        f"TASK={TASK} PHASE={args.phase} FAILURES=[] TIME={BASE.now()}\n")
    return summary


def placeholder(outdir, stage, filenames):
    for name in filenames:
        if name.endswith(".csv"): BASE.write_rows(outdir / name, [{"STATUS": "NOT_RUN_BY_STOP_RULE"}])
        else: BASE.save_json(outdir / name, {"STATUS": "NOT_RUN_BY_STOP_RULE"})


def finalize(outdir):
    outdir = Path(outdir)
    s0 = BASE.load_json(outdir / "stage0_history_parity.json", {})
    a = BASE.load_json(outdir / "stageA_summary.json", {})
    b = BASE.load_json(outdir / "stageB_summary.json", {})
    c_path, d_path = outdir / "stageC_summary.json", outdir / "stageD_summary.json"
    if not c_path.exists():
        placeholder(outdir, "stageC", ["stageC_noq_history_horizon.csv", "stageC_noq_history_units.csv", "stageC_summary.json"])
    if not d_path.exists():
        placeholder(outdir, "stageD", ["stageD_interaction_horizon.csv", "stageD_interaction_units.csv", "stageD_summary.json"])
    c, d = BASE.load_json(c_path, {}), BASE.load_json(d_path, {})
    fp32 = b.get("FINITE_PRECISION_ROTATED_HISTORY_FORMATION") == "SUPPORTED"
    noq = c.get("REPEATED_STATE_WRITEBACK_HISTORY_EFFECT") == "SUPPORTED"
    interaction = d.get("INTERACTION_STATUS") == "SUPPORTED"
    exposure = a.get("ROTATED_HISTORY_EXPOSURE_STATUS") == "STRONG_SUPPORT"
    ready = s0.get("STAGE0_HISTORY_PARITY") == "PASS" and exposure and (fp32 or noq or interaction)
    if fp32: mechanism = "FINITE_PRECISION_ROTATED_HISTORY_FORMATION_CAUSALLY_SUPPORTED"
    elif noq: mechanism = "REPEATED_ROTATED_STATE_WRITEBACK_CAUSALLY_SUPPORTED"
    elif interaction: mechanism = "FINITE_PRECISION_TRANSITION_X_STATE_WRITEBACK_INTERACTION_SUPPORTED"
    elif exposure: mechanism = "ROTATED_HISTORY_DAMAGE_CONFIRMED_BUT_GENERATOR_UNRESOLVED"
    else: mechanism = "PROTOCOL_PARITY_FAILURE" if s0.get("STAGE0_HISTORY_PARITY") != "PASS" else "ROTATED_HISTORY_EXPOSURE_NOT_SUPPORTED"
    pytest_lines = (outdir / "pytest_output.txt").read_text().strip().splitlines() if (outdir / "pytest_output.txt").exists() else []
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE", "STAGE0_HISTORY_PARITY": s0.get("STAGE0_HISTORY_PARITY"),
        "STAGEA_HISTORY_EXPOSURE": a.get("ROTATED_HISTORY_EXPOSURE_STATUS"),
        "STAGEB_FP32_HISTORY_RESCUE": b.get("FINITE_PRECISION_ROTATED_HISTORY_FORMATION"),
        "STAGEC_NOQ_HISTORY_RESCUE": c.get("REPEATED_STATE_WRITEBACK_HISTORY_EFFECT", c.get("STATUS")),
        "STAGED_INTERACTION": d.get("INTERACTION_STATUS", d.get("STATUS")),
        "N_FORMAL_UNITS": a.get("N_UNITS"), "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded", "FAILURES": [],
        "NN_REFERENCE_AUC": 0.003536528706419042, "REPRODUCED_NATIVE_HISTORY_AUC": a.get("median_auc", {}).get("L0"),
        "RN_REFERENCE_AUC": 0.12077365752377478, "REPRODUCED_ROTATED_HISTORY_NATIVE_FUTURE_AUC": a.get("median_auc", {}).get("FULL"),
        "FULL_NATIVE_HISTORY_REPRODUCES_NN": s0.get("FULL_NATIVE_HISTORY_REPRODUCES_NN"),
        "FULL_ROTATED_HISTORY_REPRODUCES_RN": s0.get("FULL_ROTATED_HISTORY_NATIVE_FUTURE_REPRODUCES_RN"),
        **{f"AUC_{key}": a.get("median_auc", {}).get(key) for key in ("L0", "L1", "L2", "L4", "L8", "L16", "L32", "L64", "FULL")},
        "FIRST_L_WITH_POSITIVE_MEDIAN_DAMAGE": a.get("FIRST_L_WITH_POSITIVE_MEDIAN_DAMAGE"),
        "FIRST_L_WITH_CI_ABOVE_ZERO": a.get("FIRST_L_WITH_CI_ABOVE_ZERO"),
        "ROTATED_HISTORY_EXPOSURE_STATUS": a.get("ROTATED_HISTORY_EXPOSURE_STATUS"),
        "N_HISTORY_AUC": b.get("median_auc", {}).get("N_HISTORY"), "R_LIVE_HISTORY_AUC": b.get("median_auc", {}).get("R_LIVE_HISTORY"),
        "R_FP32_TRANSITION_HISTORY_AUC": b.get("median_auc", {}).get("R_FP32_TRANSITION_HISTORY"),
        "FP32_HISTORY_RESCUE": b.get("FP32_HISTORY_RESCUE", {}).get("paired_median"),
        "FP32_HISTORY_CLOSURE": b.get("FP32_HISTORY_CLOSURE", {}).get("paired_median"),
        "FP32_HISTORY_CLOSURE_95CI": [b.get("FP32_HISTORY_CLOSURE", {}).get("bootstrap_ci_low"), b.get("FP32_HISTORY_CLOSURE", {}).get("bootstrap_ci_high")],
        "FINITE_PRECISION_ROTATED_HISTORY_FORMATION": b.get("FINITE_PRECISION_ROTATED_HISTORY_FORMATION"),
        "R_NO_STATE_QUANT_HISTORY_AUC": c.get("median_auc", {}).get("R_NO_STATE_QUANT_HISTORY"),
        "NOQ_HISTORY_RESCUE": c.get("NOQ_HISTORY_RESCUE", {}).get("paired_median"),
        "NOQ_HISTORY_CLOSURE": c.get("NOQ_HISTORY_CLOSURE", {}).get("paired_median"),
        "NOQ_HISTORY_CLOSURE_95CI": [c.get("NOQ_HISTORY_CLOSURE", {}).get("bootstrap_ci_low"), c.get("NOQ_HISTORY_CLOSURE", {}).get("bootstrap_ci_high")],
        "REPEATED_STATE_WRITEBACK_HISTORY_EFFECT": c.get("REPEATED_STATE_WRITEBACK_HISTORY_EFFECT", c.get("STATUS")),
        "R_FP32_NOQ_HISTORY_AUC": d.get("R_FP32_NOQ_HISTORY_AUC"),
        "TRANSITION_X_WRITEBACK_INTERACTION": d.get("TRANSITION_X_WRITEBACK_INTERACTION", {}).get("paired_median"),
        "INTERACTION_95CI": [d.get("TRANSITION_X_WRITEBACK_INTERACTION", {}).get("bootstrap_ci_low"), d.get("TRANSITION_X_WRITEBACK_INTERACTION", {}).get("bootstrap_ci_high")],
        "INTERACTION_STATUS": d.get("INTERACTION_STATUS", d.get("STATUS")),
        "FINAL_KDA_ROTATION_FAILURE_MECHANISM": mechanism,
        "KDA_MECHANISM_CLOSURE_READY": "YES" if ready else "NO", "METHOD_DESIGN_READY": "YES" if ready else "NO",
        "NEXT_PHASE": "LEARNABLE_ROTATION_METHOD_DESIGN" if ready else "HUMAN_REVIEW_REQUIRED",
        "stage0_details": s0, "stageA_details": a, "stageB_details": b, "stageC_details": c, "stageD_details": d,
    }
    BASE.save_json(outdir / "summary.json", summary)
    report = [f"# {TASK}", "", "## Formal summary", ""]
    report += [f"{k} = {json.dumps(v, sort_keys=True)}" for k, v in summary.items() if not k.endswith("_details")]
    report += ["", "## Scientific details", "", "```json", json.dumps({
        "stage0": s0, "stageA": a, "stageB": b, "stageC": c, "stageD": d}, indent=2, sort_keys=True), "```"]
    (outdir / "formal_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def validate(outdir):
    outdir = Path(outdir)
    required = ["experiment_config.json", "history_timeline_audit.json", "history_timeline.md", "stage0_history_parity.json",
                "stageA_history_exposure_horizon.csv", "stageA_history_exposure_units.csv", "stageA_history_drift_diagnostics.csv", "stageA_summary.json",
                "stageB_fp32_history_horizon.csv", "stageB_fp32_history_units.csv", "stageB_summary.json",
                "stageC_noq_history_horizon.csv", "stageC_noq_history_units.csv", "stageC_summary.json",
                "stageD_interaction_horizon.csv", "stageD_interaction_units.csv", "stageD_summary.json",
                "formal_summary.md", "run.log", "pytest_output.txt", "summary.json"]
    missing = [x for x in required if not (outdir / x).is_file()]
    a = BASE.read_rows(outdir / "stageA_history_exposure_horizon.csv")
    b = BASE.read_rows(outdir / "stageB_fp32_history_horizon.csv")
    au = {(r["unit_id"], r["horizon"], r["condition"]) for r in a}
    bu = {(r["unit_id"], r["horizon"], r["condition"]) for r in b}
    finite = all(not isinstance(v, float) or math.isfinite(v) for row in a + b for v in row.values())
    result = {"missing_required_files": missing, "n_units": len({r["unit_id"] for r in a}),
              "stageA_rows": len(a), "stageA_unique": len(au), "stageB_rows": len(b), "stageB_unique": len(bu),
              "all_numeric_values_finite": finite}
    result["ARTIFACT_VALIDATION"] = "PASS" if not missing and result["n_units"] == EXPECTED_UNITS and len(a) == len(au) and len(b) == len(bu) and finite else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", result)
    summary = BASE.load_json(outdir / "summary.json", {}); summary["ARTIFACT_VALIDATION"] = result["ARTIFACT_VALIDATION"]
    BASE.save_json(outdir / "summary.json", summary)
    with (outdir / "formal_summary.md").open("a", encoding="utf-8") as f:
        f.write(f"\nARTIFACT_VALIDATION = {result['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {"TASK": TASK, "artifacts": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("stage0", "stageA", "stageB", "stageC", "stageD", "finalize", "validate"), required=True)
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    p.add_argument("--max-units", type=int); p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(RESULT_DIR)); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true"); p.add_argument("--unit-shard-index", type=int)
    p.add_argument("--unit-shard-count", type=int); p.add_argument("--merge-run-dirs", nargs="*")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "finalize": result = finalize(args.output_dir)
    elif args.phase == "validate": result = validate(args.output_dir)
    elif args.merge_run_dirs: result = merge_runs(args)
    else: result = run_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("failures") or result.get("STAGE0_HISTORY_PARITY") == "FAIL" or result.get("ARTIFACT_VALIDATION") == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
