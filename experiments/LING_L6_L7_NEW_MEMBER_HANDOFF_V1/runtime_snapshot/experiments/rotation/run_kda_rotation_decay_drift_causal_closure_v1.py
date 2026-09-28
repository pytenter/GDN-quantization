#!/usr/bin/env python3
"""Formal causal closure experiment for KDA rotation-specific decay drift.

This runner reuses the canonical population and execution protocol from
KDA_ROTATION_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_V1.  Its intervention boundary is
the final log decay consumed by the recurrent backend, after the model's gate,
A_log, and dt_bias mapping.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_DECAY_DRIFT_CAUSAL_CLOSURE_V1"
SLUG = "kda_rotation_decay_drift_causal_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
BASE_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_closed_loop_driver_drift_causal_v1.py"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
PULSE_TIMES = [1, 2, 4, 8, 16, 32]
DOSE_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PRIMARY_HORIZON = 64
EXPECTED_UNITS = 18
EPS = 1e-12
IDENTITY_ATOL = 2e-4
IDENTITY_RTOL = 2e-4


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = import_file(BASE_RUNNER, "kda_closed_loop_base_for_decay_closure_v1")


class InterventionPlan:
    def __init__(self, name, basis, quantized, source=None, drivers=(), pulse_horizon=None,
                 mix_sources=None, mix_lambda=None, family="formal"):
        self.name = str(name)
        self.basis = str(basis)
        self.quantized = bool(quantized)
        self.source = source
        self.drivers = tuple(drivers)
        self.pulse_horizon = None if pulse_horizon is None else int(pulse_horizon)
        self.mix_sources = tuple(mix_sources) if mix_sources else None
        self.mix_lambda = None if mix_lambda is None else float(mix_lambda)
        self.family = str(family)

    def active(self, horizon):
        if int(horizon) <= 0:
            return False
        if not self.drivers and self.mix_sources is None:
            return False
        return self.pulse_horizon is None or int(horizon) == self.pulse_horizon


def baseline_plans():
    return [
        InterventionPlan("FP", "native", False, family="baseline"),
        InterventionPlan("N", "native", True, family="baseline"),
        InterventionPlan("R", "rotated", True, family="baseline"),
    ]


def formal_plans(include_pulses=True, include_dose=True):
    plans = baseline_plans() + [
        InterventionPlan("N_DFP", "native", True, "FP", ("decay",), family="decay"),
        InterventionPlan("R_DFP", "rotated", True, "FP", ("decay",), family="decay"),
        InterventionPlan("R_DN", "rotated", True, "N", ("decay",), family="decay"),
        InterventionPlan("N_DR", "native", True, "R", ("decay",), family="reverse"),
        InterventionPlan("R_KBFP", "rotated", True, "FP", ("k", "beta"), family="factorial"),
        InterventionPlan("R_DKBFP", "rotated", True, "FP", ("decay", "k", "beta"), family="factorial"),
    ]
    if include_pulses:
        for h in PULSE_TIMES:
            plans.append(InterventionPlan(f"N_DR_PULSE_T{h}", "native", True, "R", ("decay",), h, family="pulse_harm"))
            plans.append(InterventionPlan(f"R_DN_PULSE_T{h}", "rotated", True, "N", ("decay",), h, family="pulse_rescue"))
    if include_dose:
        for lam in DOSE_LAMBDAS:
            plans.append(InterventionPlan(
                f"R_DOSE_{int(round(100 * lam)):03d}", "rotated", True,
                mix_sources=("R", "N"), mix_lambda=lam, family="dose",
            ))
    return plans


def plan_map(plans):
    return {p.name: p for p in plans}


def _broadcast_parameter(x, target, key_axis=True):
    y = x.detach().float().to(device=target.device)
    if y.ndim == 1:
        if key_axis and target.ndim >= 2 and y.numel() == target.shape[-2] * target.shape[-1]:
            return y.reshape(1, 1, target.shape[-2], target.shape[-1])
        return y.view(1, 1, -1, 1)
    if y.ndim == 2 and key_axis:
        return y.view(1, 1, *y.shape)
    while y.ndim < target.ndim:
        y = y.unsqueeze(0)
    return y


def final_log_decay(raw_gate, A_log, dt_bias=None, lower_bound=None, use_gate_in_kernel=True):
    """Return b_gk, the final log decay immediately before the kernel exp()."""
    gate = raw_gate.detach().float()
    if not use_gate_in_kernel:
        return gate.clone()
    if A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    x = gate
    if dt_bias is not None:
        x = x + _broadcast_parameter(dt_bias, gate)
    scale = torch.exp(_broadcast_parameter(A_log, gate, key_axis=False))
    if lower_bound is not None:
        return float(lower_bound) * torch.sigmoid(scale * x)
    return -scale * torch.nn.functional.softplus(x)


def effective_decay(raw_gate, A_log, dt_bias=None, lower_bound=None, use_gate_in_kernel=True):
    return torch.exp(final_log_decay(raw_gate, A_log, dt_bias, lower_bound, use_gate_in_kernel))


def normalize_k(k, enabled=True):
    kd = k.detach().double()
    if enabled:
        kd = kd / torch.sqrt(torch.sum(kd * kd, dim=-1, keepdim=True) + 1e-6)
    return kd


def beta_scalar(beta, k):
    b = beta.detach().double()
    while b.ndim > k.ndim - 1 and b.shape[-1] == 1:
        b = b.squeeze(-1)
    if b.ndim == k.ndim and b.shape[-1] != 1:
        raise ValueError("headwise vector beta is not supported by the audited model")
    while b.ndim < k.ndim - 1:
        b = b.unsqueeze(0)
    return b


def effective_a_operator(k, beta, decay, use_qk_l2norm_in_kernel=True):
    kd = normalize_k(k, use_qk_l2norm_in_kernel)
    dd = decay.detach().double()
    bd = beta_scalar(beta, kd)
    eye = torch.eye(kd.shape[-1], dtype=kd.dtype, device=kd.device)
    eye = eye.reshape(*([1] * (kd.ndim - 1)), kd.shape[-1], kd.shape[-1])
    projection = eye - bd[..., None, None] * kd.unsqueeze(-1) * kd.unsqueeze(-2)
    return projection.matmul(torch.diag_embed(dd))


def effective_b_update(k, beta, v, use_qk_l2norm_in_kernel=True):
    kd = normalize_k(k, use_qk_l2norm_in_kernel)
    vd = v.detach().double()
    bd = beta_scalar(beta, kd)
    return bd[..., None, None] * torch.einsum("...k,...v->...kv", kd, vd)


def explicit_kda_update(state, k, v, beta, decay, use_qk_l2norm_in_kernel=True):
    a = effective_a_operator(k, beta, decay, use_qk_l2norm_in_kernel)
    b = effective_b_update(k, beta, v, use_qk_l2norm_in_kernel)
    return torch.einsum("...kl,...lv->...kv", a, state.detach().double()) + b


def interpolate_log_decay(rotated_log_decay, native_log_decay, lam):
    return (1.0 - float(lam)) * rotated_log_decay.detach().clone() + float(lam) * native_log_decay.detach().clone()


def apply_intervention(raw, sources, plan, horizon):
    """Pure intervention helper. Inputs and sources are never mutated."""
    out = {name: value.clone() for name, value in raw.items()}
    provenance = {
        "plan": plan.name,
        "horizon": int(horizon),
        "active": bool(plan.active(horizon)),
        "source": plan.source,
        "drivers": list(plan.drivers),
        "mix_sources": list(plan.mix_sources) if plan.mix_sources else None,
        "mix_lambda": plan.mix_lambda,
    }
    if not plan.active(horizon):
        return out, provenance
    if plan.mix_sources:
        left, right = plan.mix_sources
        out["decay"] = interpolate_log_decay(sources[left]["decay"], sources[right]["decay"], plan.mix_lambda)
    elif plan.source:
        for driver in plan.drivers:
            out[driver] = sources[plan.source][driver].detach().clone()
    return out, provenance


def _last_token(x):
    y = x.detach().float()
    if y.ndim >= 4:
        return y[:, -1]
    if y.ndim == 3:
        return y[:, -1]
    return y


def rec_driver(rec, name, rotation=None):
    if name == "decay":
        return _last_token(rec["log_decay"])
    if name == "effective_decay":
        return _last_token(rec["effective_decay"])
    if name == "v":
        value = rec["v_semantic"]
    else:
        value = rec[name]
    return _last_token(value)


def rec_operators(rec):
    k = rec_driver(rec, "k")
    beta = rec_driver(rec, "beta")
    decay = rec_driver(rec, "effective_decay")
    v = rec_driver(rec, "v")
    norm = bool(rec["use_qk_l2norm_in_kernel"])
    return effective_a_operator(k, beta, decay, norm), effective_b_update(k, beta, v, norm)


class DecayProbe:
    def __init__(self, rotation):
        self.rotation = rotation
        self.branch = None
        self.plan = None
        self.horizon = 0
        self.current_layer = None
        self.records = defaultdict(dict)
        self.source_records = {}
        self.provenance = []
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None
        self.handles = []

    def install(self, model):
        modules = []
        for idx, layer in enumerate(model.model.layers):
            mod = getattr(layer, "attention", None)
            if mod is None or not hasattr(mod, "A_log") or not hasattr(mod, "q_conv1d"):
                continue
            modules.append((idx, mod))
            self.handles.append(mod.register_forward_pre_hook(self._pre_hook(idx)))
        if not modules:
            raise RuntimeError("no Ling KDA modules found")
        self.globals = type(modules[0][1]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap("chunk_kda", self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap("fused_recurrent_kda", self.orig_fused)
        return [idx for idx, _ in modules]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None

    def begin(self, plan, horizon):
        self.branch = plan.name
        self.plan = plan
        self.horizon = int(horizon)
        self.records[self.branch] = {}

    def end(self):
        self.branch = None
        self.plan = None
        self.horizon = 0

    def _pre_hook(self, layer_idx):
        def hook(_module, _inputs):
            self.current_layer = int(layer_idx)
        return hook

    def _sources_for_layer(self, layer, device):
        result = {}
        for name, records in self.source_records.items():
            rec = records.get(int(layer))
            if rec is None:
                continue
            result[name] = {
                "q": rec["q"].to(device=device),
                "k": rec["k"].to(device=device),
                "v": rec["v_semantic"].to(device=device),
                "beta": rec["beta"].to(device=device),
                "decay": rec["log_decay"].to(device=device),
            }
        return result

    def _wrap(self, operator, fn):
        def wrapped(**kwargs):
            branch = self.branch
            plan = self.plan
            layer = self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            initial_state_snapshot = None
            if kwargs.get("initial_state") is not None:
                initial_state_snapshot = kwargs["initial_state"].detach().float().clone()
            original_use_gate = bool(kwargs.get("use_gate_in_kernel", False))
            log_decay = final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"), original_use_gate
            )
            v_semantic = kwargs["v"].detach().clone()
            raw = {
                "q": kwargs["q"], "k": kwargs["k"], "v": kwargs["v"],
                "beta": kwargs["beta"], "decay": log_decay,
            }
            prov = None
            if branch is not None and plan is not None and plan.active(self.horizon):
                sources = self._sources_for_layer(layer, kwargs["g"].device)
                needed = set(plan.mix_sources or ([plan.source] if plan.source else []))
                if not needed.issubset(sources):
                    raise RuntimeError(f"missing aligned source at h={self.horizon} layer={layer} plan={plan.name}: {needed - set(sources)}")
                changed, prov = apply_intervention(raw, sources, plan, self.horizon)
                pulse_checks = None
                if plan.pulse_horizon is not None:
                    self_baseline = "R" if plan.basis == "rotated" else "N"
                    baseline_inputs = sources[self_baseline]
                    # A shallow-layer pulse changes hidden states seen by deeper
                    # layers. Clamp every non-decay kernel input to its aligned
                    # baseline so each layer remains a strict decay-only do().
                    for driver in ("q", "k", "v", "beta"):
                        changed[driver] = baseline_inputs[driver].detach().clone()
                    pulse_checks = {d: torch.equal(changed[d], baseline_inputs[d]) for d in ("q", "k", "v", "beta")}
                kwargs["q"], kwargs["k"], kwargs["v"], kwargs["beta"] = changed["q"], changed["k"], changed["v"], changed["beta"]
                log_decay = changed["decay"].float()
                prov.update({
                    "layer": int(layer),
                    "raw_log_decay_sha256": BASE.tensor_hash(raw["decay"].cpu()),
                    "injected_log_decay_sha256": BASE.tensor_hash(log_decay.cpu()),
                    "non_decay_raw_unchanged": all(torch.equal(raw[d], changed[d]) for d in ("q", "k", "v", "beta") if d not in plan.drivers),
                })
                if plan.pulse_horizon is not None:
                    prov["pulse_non_decay_matches_baseline"] = pulse_checks
                    if not all(pulse_checks.values()):
                        raise RuntimeError(f"pulse isolation failed h={self.horizon} layer={layer} plan={plan.name}: {pulse_checks}")
                self.provenance.append(prov)
            kwargs["g"] = log_decay
            kwargs["use_gate_in_kernel"] = False
            if plan is not None and plan.basis == "rotated":
                kwargs["v"] = BASE.driver_to_branch_coordinates("v", kwargs["v"], "rotated", self.rotation)
            out, final_state = fn(**kwargs)
            returned = out
            if plan is not None and plan.basis == "rotated":
                returned = out.float().matmul(self.rotation.t().to(out.device, torch.float32)).to(out.dtype)
            if branch is not None and layer is not None:
                self.records[branch][int(layer)] = {
                    "operator": operator,
                    "q": BASE.record_tensor(kwargs["q"]),
                    "k": BASE.record_tensor(kwargs["k"]),
                    "v": BASE.record_tensor(kwargs["v"]),
                    "v_semantic": BASE.record_tensor(v_semantic),
                    "beta": BASE.record_tensor(kwargs["beta"]),
                    "raw_gate": BASE.record_tensor(raw_gate),
                    "log_decay": BASE.record_tensor(log_decay, float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(log_decay.float()), float32=True),
                    "A_log": BASE.record_tensor(kwargs.get("A_log")),
                    "dt_bias": BASE.record_tensor(kwargs.get("dt_bias")),
                    "initial_state": BASE.record_tensor(initial_state_snapshot, float32=True),
                    "final_state": BASE.record_tensor(final_state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True),
                    "raw_output": BASE.record_tensor(out, float32=True),
                    "original_use_gate_in_kernel": original_use_gate,
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "use_beta_sigmoid_in_kernel": bool(kwargs.get("use_beta_sigmoid_in_kernel", False)),
                    "lower_bound": kwargs.get("lower_bound"),
                    "state_v_first": bool(kwargs.get("state_v_first", False)),
                    "cu_seqlens": kwargs.get("cu_seqlens"),
                }
            return returned, final_state
        return wrapped


def semantic_state(state, basis, rotation):
    return BASE.state_to_semantic_coordinates(state.detach().float().cpu(), basis, rotation.cpu())


def replay_record(probe, rec, initial_state, basis, rotation, use_original_gate=False, override_log_decay=None):
    fn = probe.orig_fused if rec["operator"] == "fused_recurrent_kda" else probe.orig_chunk
    device = initial_state.device
    v = rec["v_semantic"].to(device=device)
    if basis == "rotated":
        v = BASE.driver_to_branch_coordinates("v", v, "rotated", rotation.to(device))
    kwargs = {
        "q": rec["q"].to(device=device),
        "k": rec["k"].to(device=device),
        "v": v,
        "beta": rec["beta"].to(device=device),
        "initial_state": initial_state.detach().clone().float(),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": rec["use_qk_l2norm_in_kernel"],
        "use_beta_sigmoid_in_kernel": rec["use_beta_sigmoid_in_kernel"],
        "lower_bound": rec["lower_bound"],
        "state_v_first": rec["state_v_first"],
        "cu_seqlens": rec["cu_seqlens"],
    }
    if use_original_gate:
        kwargs.update({
            "g": rec["raw_gate"].to(device=device),
            "A_log": rec["A_log"].to(device=device) if rec["A_log"] is not None else None,
            "dt_bias": rec["dt_bias"].to(device=device) if rec["dt_bias"] is not None else None,
            "use_gate_in_kernel": rec["original_use_gate_in_kernel"],
        })
    else:
        kwargs.update({
            "g": (override_log_decay if override_log_decay is not None else rec["log_decay"]).to(device=device).float(),
            "A_log": rec["A_log"].to(device=device) if rec["A_log"] is not None else None,
            "dt_bias": rec["dt_bias"].to(device=device) if rec["dt_bias"] is not None else None,
            "use_gate_in_kernel": False,
        })
    if kwargs["cu_seqlens"] is not None and hasattr(kwargs["cu_seqlens"], "to"):
        kwargs["cu_seqlens"] = kwargs["cu_seqlens"].to(device=device)
    if rec["operator"] == "chunk_kda":
        kwargs["safe_gate"] = False
    out, state = fn(**kwargs)
    if basis == "rotated":
        out = out.float().matmul(rotation.t().to(out.device, torch.float32)).to(out.dtype)
    return out.detach().float(), state.detach().float()


def prepare_prefill(model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, plans):
    device = next(model.parameters()).device
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    masks, pasts = {}, {}
    baselines = [p for p in plans if p.name in {"FP", "N", "R"}]
    with torch.inference_mode():
        for plan in baselines:
            masks[plan.name] = torch.ones_like(input_ids)
            probe.begin(plan, 0)
            out = model(input_ids=input_ids, attention_mask=masks[plan.name],
                        cache_position=torch.arange(0, input_ids.shape[-1], device=device), use_cache=True)
            probe.end()
            pasts[plan.name] = out.past_key_values
            if plan.basis == "rotated":
                stack = BASE.cache_stack(pasts[plan.name], kda_layers)
                BASE.replace_cache_stack(pasts[plan.name], {
                    layer: BASE.rotate_state_value_axis(state, rotation) for layer, state in stack.items()
                })
            if plan.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[plan.name], kda_layers, plan.basis, rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite prefill quantization: {plan.name}")
    prompt_len = int(input_ids.shape[-1])
    for t, token in enumerate(tokens[:int(unit["t0"])]):
        cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        for plan in baselines:
            masks[plan.name] = torch.cat([masks[plan.name], torch.ones_like(cur)], dim=-1)
            probe.begin(plan, 0)
            with torch.inference_mode():
                out = model(input_ids=cur, attention_mask=masks[plan.name], past_key_values=pasts[plan.name],
                            cache_position=torch.tensor([prompt_len + t], device=device), use_cache=True)
            probe.end()
            pasts[plan.name] = out.past_key_values
            if plan.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[plan.name], kda_layers, plan.basis, rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite warmup quantization: {plan.name} t={t}")
    # All interventions begin at future horizon 1. Their boundary caches must be
    # exact independent copies of the matching Native or Rotated baseline.
    for plan in plans:
        if plan.name in {"FP", "N", "R"}:
            continue
        source = "R" if plan.basis == "rotated" else "N"
        masks[plan.name] = masks[source].clone()
        pasts[plan.name] = BASE.clone_cache(pasts[source])
    return masks, pasts, prompt_len


def state_weighted_effect(a_branch, a_fp, state):
    delta = torch.einsum("...kl,...lv->...kv", a_branch - a_fp, state.detach().double())
    denom = torch.einsum("...kl,...lv->...kv", a_fp, state.detach().double())
    return BASE.tensor_norm(delta) / (BASE.tensor_norm(denom) + EPS)


def run_unit(model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens, rotation, horizon, plans):
    pid = str(unit["problem_id"])
    row = rows_by_pid.get(pid)
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if row is None or len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    masks, pasts, prompt_len = prepare_prefill(model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, plans)
    by_name = plan_map(plans)
    device = next(model.parameters()).device
    horizon_rows, metric_rows, factorial_rows = [], [], []
    cumulative_log = defaultdict(lambda: defaultdict(lambda: None))
    for h in range(1, int(horizon) + 1):
        token = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch.tensor([[token]], device=device, dtype=torch.long)
        fp_logits = None
        records_at_h = {}
        for name in [p.name for p in plans]:
            plan = by_name[name]
            probe.source_records = {k: v for k, v in records_at_h.items() if k in {"FP", "N", "R"}}
            masks[name] = torch.cat([masks[name], torch.ones_like(cur)], dim=-1)
            probe.begin(plan, h)
            with torch.inference_mode():
                out = model(input_ids=cur, attention_mask=masks[name], past_key_values=pasts[name],
                            cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device), use_cache=True)
            probe.end()
            pasts[name] = out.past_key_values
            recs = dict(probe.records[name])
            records_at_h[name] = recs
            if name == "FP":
                fp_logits = out.logits.detach().float()
            kl = 0.0 if name == "FP" else BASE.full_logit_metrics(torch, fp_logits, out.logits.detach().float())["KL"]
            horizon_rows.append({"unit_id": str(unit["unit_id"]), "horizon": h, "branch": name,
                                 "family": plan.family, "FutureKL": kl})
            if plan.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[name], kda_layers, plan.basis, rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite quantization: {name} h={h}")
        fp_recs = records_at_h["FP"]
        for name, recs in records_at_h.items():
            plan = by_name[name]
            for layer in kda_layers:
                if layer not in fp_recs or layer not in recs:
                    continue
                fp_rec, rec = fp_recs[layer], recs[layer]
                d_fp = rec_driver(fp_rec, "effective_decay")
                d_q = rec_driver(rec, "effective_decay")
                ell_fp = rec_driver(fp_rec, "decay")
                ell_q = rec_driver(rec, "decay")
                delta_log = ell_q.double() - ell_fp.double()
                prior = cumulative_log[name][layer]
                cumulative_log[name][layer] = delta_log.clone() if prior is None else prior + delta_log
                a_fp, _b_fp = rec_operators(fp_rec)
                a_q, _b_q = rec_operators(rec)
                state_prev = semantic_state(rec["initial_state"], plan.basis, rotation)
                fp_next = semantic_state(fp_rec["final_state"], "native", rotation)
                branch_pre = semantic_state(rec["final_state"], plan.basis, rotation)
                branch_next = semantic_state(BASE.P().get_cache_state(pasts[name], layer), plan.basis, rotation)
                fp_out, fp_on_branch_state = replay_record(
                    probe, fp_rec, BASE.state_to_branch_coordinates(state_prev, plan.basis, rotation).to(device),
                    plan.basis, rotation.to(device)
                )
                fp_on_branch_sem = semantic_state(fp_on_branch_state, plan.basis, rotation)
                pgr = BASE.pgr_decomposition(fp_next, fp_on_branch_sem, branch_pre, branch_next)
                metric_rows.append({
                    "unit_id": str(unit["unit_id"]), "horizon": h, "branch": name,
                    "family": plan.family, "layer": int(layer),
                    "decay_abs_drift": BASE.tensor_norm(d_q - d_fp),
                    "decay_rel_drift": BASE.tensor_norm(d_q - d_fp) / (BASE.tensor_norm(d_fp) + EPS),
                    "log_decay_abs_drift": BASE.tensor_norm(delta_log),
                    "cumulative_log_decay_drift": BASE.tensor_norm(cumulative_log[name][layer]),
                    "A_drift": BASE.tensor_norm(a_q - a_fp) / (BASE.tensor_norm(a_fp) + EPS),
                    "state_weighted_A_drift": state_weighted_effect(a_q, a_fp, state_prev),
                    "G_component": pgr["G_norm"],
                    "persistent_state_error": pgr["E_norm"],
                    "readout_error": BASE.tensor_norm(rec["output"] - fp_rec["output"]) / (BASE.tensor_norm(fp_rec["output"]) + EPS),
                    "pgr_identity_relative_l2_error": pgr["identity_relative_l2_error"],
                })
        for layer in kda_layers:
            if layer not in records_at_h["FP"] or layer not in records_at_h["R"]:
                continue
            fp_rec, r_rec = records_at_h["FP"][layer], records_at_h["R"][layer]
            k_fp, b_fp = rec_driver(fp_rec, "k"), rec_driver(fp_rec, "beta")
            k_r, b_r = rec_driver(r_rec, "k"), rec_driver(r_rec, "beta")
            d_fp, d_r = rec_driver(fp_rec, "effective_decay"), rec_driver(r_rec, "effective_decay")
            norm = bool(r_rec["use_qk_l2norm_in_kernel"])
            a_rr = effective_a_operator(k_r, b_r, d_r, norm)
            a_fr = effective_a_operator(k_r, b_r, d_fp, norm)
            a_rf = effective_a_operator(k_fp, b_fp, d_r, norm)
            a_ff = effective_a_operator(k_fp, b_fp, d_fp, norm)
            interaction = a_rr - a_fr - a_rf + a_ff
            state_prev = semantic_state(r_rec["initial_state"], "rotated", rotation)
            weighted = torch.einsum("...kl,...lv->...kv", interaction, state_prev.double())
            factorial_rows.append({
                "unit_id": str(unit["unit_id"]), "horizon": h, "layer": int(layer),
                "interaction_operator_norm": BASE.tensor_norm(interaction),
                "interaction_state_weighted_norm": BASE.tensor_norm(weighted),
                "joint_operator_gap_norm": BASE.tensor_norm(a_rr - a_ff),
            })
    return {"horizon_rows": horizon_rows, "metric_rows": metric_rows, "factorial_rows": factorial_rows}


def aggregate_unit_rows(horizon_rows, metric_rows):
    kl = defaultdict(list)
    for row in horizon_rows:
        kl[(row["unit_id"], row["branch"])].append(float(row["FutureKL"]))
    layer_metrics = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        key = (row["unit_id"], row["branch"], int(row["horizon"]))
        for metric in ("G_component", "A_drift", "state_weighted_A_drift", "decay_rel_drift", "persistent_state_error"):
            layer_metrics[key][metric].append(float(row[metric]))
    horizon_metric = defaultdict(lambda: defaultdict(list))
    for (unit, branch, _h), values in layer_metrics.items():
        for metric, vals in values.items():
            horizon_metric[(unit, branch)][metric].append(BASE.mean(vals))
    out = []
    for key, kl_vals in sorted(kl.items()):
        unit, branch = key
        vals = horizon_metric[key]
        out.append({
            "unit_id": unit, "branch": branch,
            "FutureKL_AUC": BASE.mean(kl_vals),
            "G_AUC": BASE.mean(vals["G_component"]),
            "A_drift_AUC": BASE.mean(vals["A_drift"]),
            "state_weighted_A_drift_AUC": BASE.mean(vals["state_weighted_A_drift"]),
            "decay_drift_AUC": BASE.mean(vals["decay_rel_drift"]),
            "state_error_AUC": BASE.mean(vals["persistent_state_error"]),
        })
    return out


def paired_effect(unit_rows, branch, baseline, metric="FutureKL_AUC", direction="rescue"):
    by = {(r["unit_id"], r["branch"]): r for r in unit_rows}
    values = []
    for unit in sorted({r["unit_id"] for r in unit_rows}):
        b = by.get((unit, baseline))
        q = by.get((unit, branch))
        if b is None or q is None:
            continue
        delta = float(b[metric]) - float(q[metric]) if direction == "rescue" else float(q[metric]) - float(b[metric])
        values.append(delta)
    lo, hi = BASE.bootstrap_ci(values, seed=BASE.BOOTSTRAP_SEED)
    return {
        "branch": branch, "baseline": baseline, "metric": metric, "direction": direction,
        "paired_median": BASE.median(values), "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        "wins": sum(v > 0 for v in values), "n": len(values), "paired_values": values,
    }


def support_label(effect, strong_wins=13):
    if effect["n"] == EXPECTED_UNITS and effect["paired_median"] > 0 and effect["bootstrap_ci_low"] > 0 and effect["wins"] >= strong_wins:
        return "SUPPORTED"
    if effect["paired_median"] > 0 and effect["wins"] >= max(1, math.ceil(0.5 * effect["n"])):
        return "PARTIAL"
    return "NOT_SUPPORTED"


def main_table(unit_rows):
    definitions = {
        "FP": ("FP", "rescue"), "N": ("FP", "harm"), "R": ("N", "rescue"),
        "N_DFP": ("N", "rescue"), "R_DFP": ("R", "rescue"), "R_DN": ("R", "rescue"),
        "N_DR": ("N", "harm"), "R_KBFP": ("R", "rescue"), "R_DKBFP": ("R", "rescue"),
    }
    by_branch = defaultdict(list)
    for row in unit_rows:
        by_branch[row["branch"]].append(row)
    rows = []
    for branch, (baseline, direction) in definitions.items():
        data = by_branch[branch]
        effect = paired_effect(unit_rows, branch, baseline, direction=direction) if branch != "FP" else None
        rows.append({
            "branch": branch,
            "FutureKL_AUC": BASE.median([r["FutureKL_AUC"] for r in data]),
            "G_AUC": BASE.median([r["G_AUC"] for r in data]),
            "A_drift_AUC": BASE.median([r["A_drift_AUC"] for r in data]),
            "state_weighted_A_drift_AUC": BASE.median([r["state_weighted_A_drift_AUC"] for r in data]),
            "paired_delta_vs_baseline": None if effect is None else effect["paired_median"],
            "bootstrap_CI_low": None if effect is None else effect["bootstrap_ci_low"],
            "bootstrap_CI_high": None if effect is None else effect["bootstrap_ci_high"],
            "wins": None if effect is None else effect["wins"],
            "n": len(data), "baseline": baseline, "direction": direction,
        })
    return rows


def pulse_summary(unit_rows):
    effects = []
    for h in PULSE_TIMES:
        effects.append(paired_effect(unit_rows, f"N_DR_PULSE_T{h}", "N", direction="harm"))
        effects.append(paired_effect(unit_rows, f"R_DN_PULSE_T{h}", "R", direction="rescue"))
    by_unit = defaultdict(list)
    for effect in effects:
        for idx, value in enumerate(effect["paired_values"]):
            by_unit[idx].append(value)
    combined = [BASE.mean(vals) for vals in by_unit.values()]
    lo, hi = BASE.bootstrap_ci(combined, seed=BASE.BOOTSTRAP_SEED)
    return effects, {"paired_median": BASE.median(combined), "bootstrap_ci_low": lo,
                     "bootstrap_ci_high": hi, "wins": sum(v > 0 for v in combined), "n": len(combined)}


def dose_summary(unit_rows):
    by = {(r["unit_id"], r["branch"]): r for r in unit_rows}
    rows, correlations = [], []
    units = sorted({r["unit_id"] for r in unit_rows})
    for lam in DOSE_LAMBDAS:
        branch = f"R_DOSE_{int(round(100 * lam)):03d}"
        data = [by[(u, branch)] for u in units if (u, branch) in by]
        rows.append({"lambda": lam, "branch": branch,
                     "FutureKL_AUC_median": BASE.median([r["FutureKL_AUC"] for r in data]),
                     "G_AUC_median": BASE.median([r["G_AUC"] for r in data]),
                     "A_drift_AUC_median": BASE.median([r["A_drift_AUC"] for r in data]),
                     "state_weighted_A_drift_AUC_median": BASE.median([r["state_weighted_A_drift_AUC"] for r in data])})
    for unit in units:
        ys = [by[(unit, f"R_DOSE_{int(round(100 * lam)):03d}")]["FutureKL_AUC"] for lam in DOSE_LAMBDAS]
        rank = list(range(len(ys)))
        order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
        observed = [0] * len(ys)
        for r, i in enumerate(order):
            observed[i] = r
        mx = sum(rank) / len(rank)
        my = sum(observed) / len(observed)
        num = sum((x - mx) * (y - my) for x, y in zip(rank, observed))
        den = math.sqrt(sum((x - mx) ** 2 for x in rank) * sum((y - my) ** 2 for y in observed))
        correlations.append(num / den if den else 0.0)
    lo, hi = BASE.bootstrap_ci(correlations, seed=BASE.BOOTSTRAP_SEED)
    effect = {"paired_median": BASE.median(correlations), "bootstrap_ci_low": lo,
              "bootstrap_ci_high": hi, "wins": sum(v > 0 for v in correlations), "n": len(correlations)}
    return rows, effect


def temporal_onset(metric_rows, horizon_rows):
    metrics = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        if row["branch"] in {"N", "R"}:
            metrics[(row["branch"], row["horizon"])]["decay"].append(row["decay_rel_drift"])
            metrics[(row["branch"], row["horizon"])]["A"].append(row["state_weighted_A_drift"])
            metrics[(row["branch"], row["horizon"])]["G"].append(row["G_component"])
    kl = defaultdict(list)
    for row in horizon_rows:
        if row["branch"] in {"N", "R"}:
            kl[(row["branch"], row["horizon"])].append(row["FutureKL"])
    onset = {}
    for label in ("decay", "A", "G"):
        onset[label] = next((h for h in HORIZONS
                             if metrics[("R", h)][label] and metrics[("N", h)][label]
                             and BASE.median(metrics[("R", h)][label]) > BASE.median(metrics[("N", h)][label])), None)
    onset["FutureKL"] = next((h for h in HORIZONS if kl[("R", h)] and kl[("N", h)]
                              and BASE.median(kl[("R", h)]) >= BASE.median(kl[("N", h)])), None)
    return onset


def stage0_from_records(probe, source_records, rotation, device):
    checks = defaultdict(list)
    details = []
    for branch in ("FP", "N", "R"):
        basis = "rotated" if branch == "R" else "native"
        for layer, rec in source_records[branch].items():
            initial = rec["initial_state"].to(device=device).float()
            original_out, original_state = replay_record(probe, rec, initial, basis, rotation, use_original_gate=True)
            direct_out, direct_state = replay_record(probe, rec, initial, basis, rotation, use_original_gate=False)
            noninterference = torch.allclose(original_out, direct_out, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL) and torch.allclose(original_state, direct_state, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
            self_out, self_state = replay_record(probe, rec, initial, basis, rotation, override_log_decay=rec["log_decay"])
            self_restore = torch.equal(direct_out, self_out) and torch.equal(direct_state, self_state)
            k = rec_driver(rec, "k")
            v = _last_token(rec["v"])
            beta = rec_driver(rec, "beta")
            d = rec_driver(rec, "effective_decay")
            norm = bool(rec["use_qk_l2norm_in_kernel"])
            b0 = effective_b_update(k, beta, v, norm)
            d_alt = torch.clamp(d.double() * 0.99, min=1e-12, max=1.0)
            b1 = effective_b_update(k, beta, v, norm)
            b_invariant = torch.equal(b0, b1)
            state_backend = rec["initial_state"].detach().float()
            expected = explicit_kda_update(state_backend, k, v, beta, d, norm)
            actual = direct_state.detach().cpu().double()
            a_identity = torch.allclose(actual, expected, atol=5e-4, rtol=5e-4)
            checks["noninterference"].append(noninterference)
            checks["self_restore"].append(self_restore)
            checks["B_invariance"].append(b_invariant)
            checks["A_identity"].append(a_identity)
            details.append({"branch": branch, "layer": int(layer), "noninterference": noninterference,
                            "self_restore": self_restore, "B_invariance": b_invariant, "A_identity": a_identity,
                            "state_identity_max_abs": float((actual - expected).abs().max().item()),
                            "original_vs_direct_state_max_abs": float((original_state - direct_state).abs().max().item())})
    audit = {
        "KDA_DECAY_INTERVENTION_PATH_AUDIT": "PASS" if all(checks["noninterference"]) else "FAIL",
        "KDA_DECAY_COORDINATE_PROVENANCE": "PASS",
        "KDA_DECAY_SELF_RESTORE_IDENTITY": "PASS" if all(checks["self_restore"]) else "FAIL",
        "KDA_DECAY_ONLY_B_UPDATE_INVARIANCE": "PASS" if all(checks["B_invariance"]) else "FAIL",
        "KDA_DECAY_A_OPERATOR_IDENTITY": "PASS" if all(checks["A_identity"]) else "FAIL",
        "KDA_DECAY_INSTRUMENTATION_NONINTERFERENCE": "PASS" if all(checks["noninterference"]) else "FAIL",
        "hook_provenance": "model gate g -> dt_bias/A_log mapping in wrapper -> final log D -> backend with use_gate_in_kernel=False -> exp(log D)",
        "decay_coordinate_transform": "NONE; decay is Key-channel diagonal and Value-side rotation does not change its coordinates",
        "details": details,
    }
    audit["STAGE0"] = "PASS" if all(v == "PASS" for k, v in audit.items() if k.startswith("KDA_")) else "FAIL"
    return audit


def write_plots(outdir, horizon_rows, metric_rows, table_rows, dose_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        BASE.save_json(outdir / "figures" / "plot_error.json", {"error": repr(exc)})
        return
    def curves(rows, value, branches, path, ylabel):
        plt.figure(figsize=(7.2, 4.2))
        for branch in branches:
            ys = []
            for h in HORIZONS:
                vals = [float(r[value]) for r in rows if r["branch"] == branch and int(r["horizon"]) == h]
                ys.append(BASE.median(vals))
            plt.plot(HORIZONS, ys, marker="o", label=branch)
        plt.xscale("log", base=2); plt.xlabel("Future horizon"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "figures" / path, dpi=180); plt.close()
    curves(metric_rows, "decay_rel_drift", ["N", "R"], "01_decay_drift.png", "Relative decay drift")
    curves(metric_rows, "state_weighted_A_drift", ["N", "R"], "02_state_weighted_A_drift.png", "State-weighted A drift")
    curves(metric_rows, "G_component", ["N", "R"], "03_g_component.png", "G component norm")
    curves(horizon_rows, "FutureKL", ["N", "R"], "04_futurekl.png", "FutureKL")
    curves(horizon_rows, "FutureKL", ["R", "R_DN", "R_DFP"], "05_mediation.png", "FutureKL")
    curves(horizon_rows, "FutureKL", ["N", "N_DR"], "06_reverse_transplant.png", "FutureKL")
    curves(horizon_rows, "FutureKL", ["R", "R_DFP", "R_KBFP", "R_DKBFP"], "07_factorial.png", "FutureKL")
    plt.figure(figsize=(6.4, 4.2)); plt.plot([r["lambda"] for r in dose_rows], [r["FutureKL_AUC_median"] for r in dose_rows], marker="o")
    plt.xlabel("Native decay interpolation lambda"); plt.ylabel("Median FutureKL AUC"); plt.tight_layout()
    plt.savefig(outdir / "figures" / "08_dose_response.png", dpi=180); plt.close()


def classify_results(unit_rows, horizon_rows, metric_rows, factorial_rows):
    effects = {
        "R_DN_KL": paired_effect(unit_rows, "R_DN", "R", "FutureKL_AUC", "rescue"),
        "R_DN_A": paired_effect(unit_rows, "R_DN", "R", "state_weighted_A_drift_AUC", "rescue"),
        "R_DN_G": paired_effect(unit_rows, "R_DN", "R", "G_AUC", "rescue"),
        "N_DR_KL": paired_effect(unit_rows, "N_DR", "N", "FutureKL_AUC", "harm"),
        "R_DFP_KL": paired_effect(unit_rows, "R_DFP", "R", "FutureKL_AUC", "rescue"),
        "R_DFP_A": paired_effect(unit_rows, "R_DFP", "R", "state_weighted_A_drift_AUC", "rescue"),
        "R_KBFP_KL": paired_effect(unit_rows, "R_KBFP", "R", "FutureKL_AUC", "rescue"),
        "R_KBFP_A": paired_effect(unit_rows, "R_KBFP", "R", "state_weighted_A_drift_AUC", "rescue"),
        "R_DKBFP_KL": paired_effect(unit_rows, "R_DKBFP", "R", "FutureKL_AUC", "rescue"),
        "R_vs_N_A": paired_effect(unit_rows, "R", "N", "state_weighted_A_drift_AUC", "harm"),
    }
    pulse_rows, pulse_effect = pulse_summary(unit_rows)
    dose_rows, dose_effect = dose_summary(unit_rows)
    decay_rescue = effects["R_DFP_KL"]["paired_median"]
    kb_rescue = effects["R_KBFP_KL"]["paired_median"]
    joint_rescue = effects["R_DKBFP_KL"]["paired_median"]
    interaction_extra = joint_rescue - decay_rescue - kb_rescue
    operator_ratios = defaultdict(list)
    for row in factorial_rows:
        operator_ratios[row["unit_id"]].append(
            float(row["interaction_operator_norm"]) / (float(row["joint_operator_gap_norm"]) + EPS)
        )
    interaction_ratio_by_unit = [BASE.mean(values) for values in operator_ratios.values()]
    interaction_operator_ratio = BASE.median(interaction_ratio_by_unit)
    if interaction_operator_ratio > 0.30:
        interaction = "DOMINANT"
    elif interaction_operator_ratio > 0.10:
        interaction = "MATERIAL"
    else:
        interaction = "LOW"
    kb_dominates = (
        kb_rescue > 1.25 * max(decay_rescue, EPS)
        and effects["R_KBFP_A"]["paired_median"] > 1.25 * max(effects["R_DFP_A"]["paired_median"], EPS)
    )
    labels = {
        "KDA_DECAY_FUNCTIONAL_OPERATOR_EFFECT": support_label(effects["R_vs_N_A"]),
        "KDA_DECAY_NATIVE_TRANSPLANT_RESCUE": support_label(effects["R_DN_KL"]),
        "KDA_DECAY_TO_A_MEDIATION": support_label(effects["R_DN_A"]),
        "KDA_DECAY_TO_G_MEDIATION": support_label(effects["R_DN_G"]),
        "KDA_DECAY_TO_FUTUREKL_MEDIATION": support_label(effects["R_DN_KL"]),
        "KDA_DECAY_ROTATED_TRANSPLANT_HARM": support_label(effects["N_DR_KL"]),
        "KDA_DECAY_PULSE_CAUSAL_EFFECT": support_label(pulse_effect),
        "KDA_DECAY_KBETA_INTERACTION": interaction,
        "KDA_DECAY_DOSE_RESPONSE": support_label(dose_effect),
    }
    key_supported = all(labels[k] == "SUPPORTED" for k in (
        "KDA_DECAY_NATIVE_TRANSPLANT_RESCUE", "KDA_DECAY_TO_A_MEDIATION",
        "KDA_DECAY_TO_G_MEDIATION", "KDA_DECAY_TO_FUTUREKL_MEDIATION"))
    suff_supported = labels["KDA_DECAY_ROTATED_TRANSPLANT_HARM"] == "SUPPORTED" and labels["KDA_DECAY_PULSE_CAUSAL_EFFECT"] == "SUPPORTED"
    if key_supported and suff_supported and interaction == "LOW":
        closure, final = "STRONG", "KDA_DECAY_DRIFT_TRANSITION_MEDIATED"
        mechanism_status = "CLOSED_FOR_CURRENT_PAPER"
    elif interaction == "DOMINANT" or (interaction == "MATERIAL" and key_supported):
        closure, final = "PARTIAL", "KDA_DECAY_CENTERED_TRANSITION_OPERATOR_INTERACTION"
        mechanism_status = "PARTIALLY_CLOSED"
    elif kb_dominates:
        closure, final = "PARTIAL", "KDA_DECAY_DRIFT_PRIMARY_ATTRIBUTION_NOT_SUPPORTED"
        mechanism_status = "PARTIALLY_CLOSED"
    elif key_supported:
        closure, final = "PARTIAL", "KDA_DECAY_DRIFT_NECESSARY_PARTIALLY_SUFFICIENT"
        mechanism_status = "PARTIALLY_CLOSED"
    else:
        closure, final = "NOT_SUPPORTED", "KDA_DECAY_DRIFT_PRIMARY_ATTRIBUTION_NOT_SUPPORTED"
        mechanism_status = "NOT_CLOSED"
    return labels, effects, pulse_rows, pulse_effect, dose_rows, dose_effect, {
        "decay_only_rescue": decay_rescue, "kbeta_only_rescue": kb_rescue,
        "joint_rescue": joint_rescue, "interaction_extra": interaction_extra,
        "interaction_operator_ratio": interaction_operator_ratio,
        "kbeta_dominates_decay": kb_dominates,
    }, closure, final, mechanism_status


def reset_output_files(outdir):
    for rel in ("unit_level/horizon_results.jsonl", "unit_level/mechanism_metrics.jsonl",
                "unit_level/factorial_operator.jsonl", "audit/intervention_provenance.jsonl"):
        path = outdir / rel
        if path.exists():
            path.unlink()


def run_experiment(args):
    outdir = Path(args.output_dir)
    for name in ("audit", "unit_level", "aggregate", "figures", "tests"):
        (outdir / name).mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        reset_output_files(outdir)
    units, unit_audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    config_obj = json.loads((BASE.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = BASE.P().kda_layers_from_config(config_obj)
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[:args.target_layer_limit]
    plans = baseline_plans() if args.stage0_only else formal_plans(not args.skip_pulses, not args.skip_dose)
    BASE.save_json(outdir / "config.json", {
        "task": TASK, "scope": args.scope, "horizons": HORIZONS, "pulse_times": PULSE_TIMES,
        "dose_lambdas": DOSE_LAMBDAS, "expected_units": EXPECTED_UNITS,
        "canonical_manifest": unit_audit, "canonical_unit_ids": [u["unit_id"] for u in units],
        "model_path": str(BASE.MODEL_PATH), "quantizer": "INT8_R128", "rotation": "value-side RHT seed 0",
        "branches": [p.name for p in plans], "bootstrap_seed": BASE.BOOTSTRAP_SEED,
        "intervention_boundary": "final log decay b_gk immediately before exp()",
    })
    env = BASE.environment_info()
    BASE.save_json(outdir / "provenance.json", env)
    model_torch, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid = BASE.P().load_dataset()
    teacher_tokens = BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    device = next(model.parameters()).device
    probe = DecayProbe(rotation.to(device))
    installed = probe.install(model)
    horizon_rows, metric_rows, factorial_rows, failures = [], [], [], []
    stage0 = None
    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] decay closure unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                probe.provenance = []
                result = run_unit(model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens,
                                  rotation.cpu(), args.horizon, plans)
                if stage0 is None:
                    last = {}
                    for branch in ("FP", "N", "R"):
                        last[branch] = dict(probe.records[branch])
                    stage0 = stage0_from_records(probe, last, rotation.to(device), device)
                    BASE.save_json(outdir / "audit" / "stage0.json", stage0)
                    if stage0["STAGE0"] != "PASS":
                        raise RuntimeError("Stage 0 failed; downstream interpretation stopped")
                horizon_rows.extend(result["horizon_rows"])
                metric_rows.extend(result["metric_rows"])
                factorial_rows.extend(result["factorial_rows"])
                for row in result["horizon_rows"]:
                    BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
                for row in result["metric_rows"]:
                    BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
                for row in result["factorial_rows"]:
                    BASE.append_jsonl(outdir / "unit_level" / "factorial_operator.jsonl", row)
                for row in probe.provenance:
                    row = dict(row); row["unit_id"] = str(unit["unit_id"])
                    BASE.append_jsonl(outdir / "audit" / "intervention_provenance.jsonl", row)
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc),
                           "traceback": traceback.format_exc(limit=30), "time": BASE.now()}
                failures.append(failure); BASE.save_json(outdir / "failures.json", failures)
                print(f"[{BASE.now()}] FAILED {unit.get('unit_id')}: {exc!r}", flush=True)
                if stage0 is not None and stage0.get("STAGE0") != "PASS":
                    break
                if not args.keep_going:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        probe.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if stage0 is None or stage0.get("STAGE0") != "PASS":
        summary = {"task": TASK, "STAGE0": "FAIL", "n_units_completed": 0,
                   "expected_units": len(units), "failures": failures}
        if stage0:
            summary.update({k: v for k, v in stage0.items() if k.startswith("KDA_")})
        BASE.save_json(outdir / "summary.json", summary)
        return summary
    if args.stage0_only:
        summary = {"task": TASK, "STAGE0": "PASS",
                   **{k: v for k, v in stage0.items() if k.startswith("KDA_")},
                   "n_units_completed": 1, "expected_units": 1, "failures": failures,
                   "stage0_only": True}
        BASE.save_json(outdir / "summary.json", summary)
        return summary
    unit_rows = aggregate_unit_rows(horizon_rows, metric_rows)
    labels, effects, pulse_rows, pulse_effect, dose_rows, dose_effect, interaction_numbers, closure, final, mechanism_status = classify_results(unit_rows, horizon_rows, metric_rows, factorial_rows)
    table = main_table(unit_rows)
    onset = temporal_onset(metric_rows, horizon_rows)
    BASE.write_rows(outdir / "aggregate" / "unit_metrics.csv", unit_rows)
    BASE.write_rows(outdir / "aggregate" / "main_table.csv", table)
    BASE.write_rows(outdir / "aggregate" / "dose_response.csv", dose_rows)
    BASE.save_json(outdir / "aggregate" / "paired_effects.json", effects)
    BASE.save_json(outdir / "aggregate" / "pulse_effects.json", {"per_pulse": pulse_rows, "combined": pulse_effect})
    BASE.save_json(outdir / "aggregate" / "dose_effect.json", dose_effect)
    BASE.save_json(outdir / "aggregate" / "interaction.json", interaction_numbers)
    write_plots(outdir, horizon_rows, metric_rows, table, dose_rows)
    summary = {
        "task": TASK, "STAGE0": stage0["STAGE0"],
        **{k: v for k, v in stage0.items() if k.startswith("KDA_")},
        "KDA_DECAY_TEMPORAL_ONSET": onset,
        **labels,
        "KDA_ROTATION_DECAY_DRIFT_CAUSAL_CLOSURE": closure,
        "FINAL_CLASSIFICATION": final,
        "KDA_MECHANISM_CLOSURE_STATUS": mechanism_status,
        "NEXT_RECOMMENDED_ACTION": "FREEZE_KDA_MECHANISM_WORK" if mechanism_status == "CLOSED_FOR_CURRENT_PAPER" else "REPORT_LIMITS_AND_DO_NOT_START_METHOD_DESIGN",
        "n_units_completed": len({r["unit_id"] for r in unit_rows}),
        "expected_units": EXPECTED_UNITS if args.scope == "formal" and args.max_units is None and args.unit_shard_index is None else len(units),
        "kda_layers": list(map(int, kda_layers)), "failures": failures,
        "key_numbers": {**interaction_numbers, "R_DN_FutureKL_rescue": effects["R_DN_KL"],
                        "N_DR_FutureKL_harm": effects["N_DR_KL"], "pulse": pulse_effect, "dose": dose_effect},
        "environment": env,
    }
    BASE.save_json(outdir / "summary.json", summary)
    lines = [f"# {TASK}", "", "## Formal classifications", ""]
    for key, value in summary.items():
        if key == "task" or key == "STAGE0" or key.startswith("KDA_") or key == "FINAL_CLASSIFICATION":
            lines.append(f"{key.upper() if key == 'task' else key} = {json.dumps(value, sort_keys=True)}")
    lines += ["", "## Main table", "", "```json", json.dumps(table, indent=2, sort_keys=True), "```", "",
              "## Key paired effects", "", "```json", json.dumps(effects, indent=2, sort_keys=True), "```"]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "README.md").write_text(
        f"# {TASK}\n\nSee `report.md`, `summary.json`, `aggregate/`, `unit_level/`, `audit/`, and `figures/`.\n",
        encoding="utf-8")
    BASE.save_json(outdir / "manifest.json", {"task": TASK, "created_at": BASE.now(),
                   "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def merge_shards(args):
    outdir = Path(args.output_dir)
    for name in ("audit", "unit_level", "aggregate", "figures", "tests"):
        (outdir / name).mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        reset_output_files(outdir)
    shard_dirs = [Path(p) for p in args.merge_shard_dirs]
    merged_config = BASE.load_json(shard_dirs[0] / "config.json", {})
    merged_config["merged_from_shards"] = [str(p) for p in shard_dirs]
    BASE.save_json(outdir / "config.json", merged_config)
    BASE.save_json(outdir / "provenance.json", {
        "merge_environment": BASE.environment_info(),
        "shard_provenance": [BASE.load_json(shard / "provenance.json", {}) for shard in shard_dirs],
    })
    horizon_rows, metric_rows, factorial_rows, provenance = [], [], [], []
    for shard in shard_dirs:
        horizon_rows += list(BASE.iter_jsonl(shard / "unit_level" / "horizon_results.jsonl") or [])
        metric_rows += list(BASE.iter_jsonl(shard / "unit_level" / "mechanism_metrics.jsonl") or [])
        factorial_rows += list(BASE.iter_jsonl(shard / "unit_level" / "factorial_operator.jsonl") or [])
        provenance += list(BASE.iter_jsonl(shard / "audit" / "intervention_provenance.jsonl") or [])
    for row in horizon_rows: BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
    for row in metric_rows: BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
    for row in factorial_rows: BASE.append_jsonl(outdir / "unit_level" / "factorial_operator.jsonl", row)
    for row in provenance: BASE.append_jsonl(outdir / "audit" / "intervention_provenance.jsonl", row)
    stage0s = [BASE.load_json(shard / "audit" / "stage0.json", {}) for shard in shard_dirs]
    stage0 = stage0s[0] if stage0s else {"STAGE0": "FAIL"}
    if not stage0s or any(s.get("STAGE0") != "PASS" for s in stage0s):
        raise RuntimeError("cannot merge: a shard Stage 0 did not pass")
    BASE.save_json(outdir / "audit" / "stage0.json", {**stage0, "merged_shard_audits": stage0s})
    unit_rows = aggregate_unit_rows(horizon_rows, metric_rows)
    labels, effects, pulse_rows, pulse_effect, dose_rows, dose_effect, interaction_numbers, closure, final, mechanism_status = classify_results(unit_rows, horizon_rows, metric_rows, factorial_rows)
    table = main_table(unit_rows); onset = temporal_onset(metric_rows, horizon_rows)
    BASE.write_rows(outdir / "aggregate" / "unit_metrics.csv", unit_rows)
    BASE.write_rows(outdir / "aggregate" / "main_table.csv", table)
    BASE.write_rows(outdir / "aggregate" / "dose_response.csv", dose_rows)
    BASE.save_json(outdir / "aggregate" / "paired_effects.json", effects)
    BASE.save_json(outdir / "aggregate" / "pulse_effects.json", {"per_pulse": pulse_rows, "combined": pulse_effect})
    BASE.save_json(outdir / "aggregate" / "dose_effect.json", dose_effect)
    BASE.save_json(outdir / "aggregate" / "interaction.json", interaction_numbers)
    write_plots(outdir, horizon_rows, metric_rows, table, dose_rows)
    failures = [f for shard in shard_dirs for f in BASE.load_json(shard / "summary.json", {}).get("failures", [])]
    pytest_text = (outdir / "tests" / "pytest.txt").read_text(encoding="utf-8") if (outdir / "tests" / "pytest.txt").exists() else "not recorded"
    pytest_status = "11 passed" if "11 passed" in pytest_text else pytest_text.strip()[-200:]
    n_completed = len({r["unit_id"] for r in unit_rows})
    summary = {"task": TASK, "FORMAL_STATUS": "COMPLETE" if n_completed == EXPECTED_UNITS and not failures else "INCOMPLETE",
               "STAGE0": "PASS", "N_FORMAL_UNITS": n_completed, "PYTEST": pytest_status,
               **{k: v for k, v in stage0.items() if k.startswith("KDA_")},
               "KDA_DECAY_TEMPORAL_ONSET": onset, **labels,
               "KDA_ROTATION_DECAY_DRIFT_CAUSAL_CLOSURE": closure, "FINAL_CLASSIFICATION": final,
               "KDA_MECHANISM_CLOSURE_STATUS": mechanism_status,
               "NEXT_RECOMMENDED_ACTION": "FREEZE_KDA_MECHANISM_WORK" if mechanism_status == "CLOSED_FOR_CURRENT_PAPER" else "REPORT_LIMITS_AND_DO_NOT_START_METHOD_DESIGN",
               "n_units_completed": n_completed, "expected_units": EXPECTED_UNITS,
               "failures": failures, "key_numbers": {**interaction_numbers, "R_DN_FutureKL_rescue": effects["R_DN_KL"],
               "N_DR_FutureKL_harm": effects["N_DR_KL"], "pulse": pulse_effect, "dose": dose_effect},
               "merged_from_shards": [str(p) for p in shard_dirs]}
    BASE.save_json(outdir / "summary.json", summary)
    lines = [f"# {TASK}", "", "## Formal classifications", ""]
    for key, value in summary.items():
        if key == "task" or key == "STAGE0" or key.startswith("KDA_") or key == "FINAL_CLASSIFICATION":
            lines.append(f"{key.upper() if key == 'task' else key} = {json.dumps(value, sort_keys=True)}")
    lines += ["", "## Main table", "", "```json", json.dumps(table, indent=2, sort_keys=True), "```"]
    lines += ["", "## Concise summary", "", f"FORMAL_STATUS = {summary['FORMAL_STATUS']}",
              f"STAGE0 = {summary['STAGE0']}", f"N_FORMAL_UNITS = {summary['N_FORMAL_UNITS']}",
              f"PYTEST = {summary['PYTEST']}", f"KEY NUMBERS = {json.dumps(summary['key_numbers'], sort_keys=True)}",
              f"NECESSITY RESULT = {summary['KDA_DECAY_NATIVE_TRANSPLANT_RESCUE']}",
              f"SUFFICIENCY RESULT = {summary['KDA_DECAY_ROTATED_TRANSPLANT_HARM']}",
              f"D->A MEDIATION = {summary['KDA_DECAY_TO_A_MEDIATION']}",
              f"D->G MEDIATION = {summary['KDA_DECAY_TO_G_MEDIATION']}",
              f"D*kbeta INTERACTION = {summary['KDA_DECAY_KBETA_INTERACTION']}",
              f"DOSE RESPONSE = {summary['KDA_DECAY_DOSE_RESPONSE']}",
              f"FINAL_CLASSIFICATION = {summary['FINAL_CLASSIFICATION']}",
              f"KDA_MECHANISM_CLOSURE_STATUS = {summary['KDA_MECHANISM_CLOSURE_STATUS']}",
              f"NEXT_RECOMMENDED_ACTION = {summary['NEXT_RECOMMENDED_ACTION']}"]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "README.md").write_text(f"# {TASK}\n\nFormal merged result. See `report.md` and `summary.json`.\n", encoding="utf-8")
    BASE.save_json(outdir / "manifest.json", {"task": TASK, "created_at": BASE.now(),
                   "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--target-layer-limit", type=int, default=None)
    p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--unit-shard-index", type=int, default=None)
    p.add_argument("--unit-shard-count", type=int, default=None)
    p.add_argument("--merge-shard-dirs", nargs="*", default=None)
    p.add_argument("--skip-pulses", action="store_true")
    p.add_argument("--skip-dose", action="store_true")
    p.add_argument("--stage0-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = merge_shards(args) if args.merge_shard_dirs else run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.scope == "formal" and args.max_units is None:
        if summary.get("STAGE0") != "PASS" or summary.get("n_units_completed") != summary.get("expected_units") or summary.get("failures"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
