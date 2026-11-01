#!/usr/bin/env python3
"""Formal operator-level A/B causal localization for KDA closed-loop drift."""

import argparse
import importlib.util
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_G_AB_OPERATOR_CAUSAL_CLOSURE_V1"
SLUG = "kda_rotation_g_ab_operator_causal_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
DECAY_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_decay_drift_causal_closure_v1.py"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
PRIMARY_HORIZON = 64
EXPECTED_UNITS = 18
EPS = 1e-12
# The backend accumulates in float32 and the rotated cache has one additional
# float32 orthogonal transform.  The audited worst-case one-step discrepancy is
# about 1.2e-3, while functional operator effects are orders of magnitude larger.
IDENTITY_ATOL = 2e-3
IDENTITY_RTOL = 2e-3
COORD_ATOL = 2e-6
FORMAL_IDENTITY_ATOL = 3e-3


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DECAY = import_file(DECAY_RUNNER, "kda_decay_closure_for_g_ab_v1")
BASE = DECAY.BASE


class InterventionPlan:
    def __init__(self, name, basis, quantized, a_source=None, b_source=None,
                 q_source=None, family="baseline", b_factor_sources=None):
        self.name = str(name)
        self.basis = str(basis)
        self.quantized = bool(quantized)
        self.a_source = a_source
        self.b_source = b_source
        self.q_source = q_source
        self.family = str(family)
        self.b_factor_sources = None if b_factor_sources is None else dict(b_factor_sources)

    @property
    def intervened(self):
        return any(x is not None for x in (self.a_source, self.b_source, self.q_source)) or self.b_factor_sources is not None


def baseline_plans():
    return [
        InterventionPlan("FP", "native", False),
        InterventionPlan("N", "native", True),
        InterventionPlan("R", "rotated", True),
    ]


def plans_for_phase(phase):
    plans = baseline_plans()
    if phase == "stage0":
        return plans + [
            InterventionPlan("N_SELF", "native", True, "N", "N", "N", family="audit"),
            InterventionPlan("R_SELF", "rotated", True, "R", "R", "R", family="audit"),
            InterventionPlan("FP_SELF", "native", False, "FP", "FP", "FP", family="audit"),
        ]
    if phase == "round1":
        return plans + [
            InterventionPlan("N_AFP", "native", True, "FP", "N", family="round1"),
            InterventionPlan("N_BFP", "native", True, "N", "FP", family="round1"),
            InterventionPlan("N_ABFP", "native", True, "FP", "FP", family="round1"),
            InterventionPlan("R_AFP", "rotated", True, "FP", "R", family="round1"),
            InterventionPlan("R_BFP", "rotated", True, "R", "FP", family="round1"),
            InterventionPlan("R_ABFP", "rotated", True, "FP", "FP", family="round1"),
        ]
    if phase == "round2":
        return plans + [
            InterventionPlan("R_AN", "rotated", True, "N", "R", family="transplant"),
            InterventionPlan("R_BN", "rotated", True, "R", "N", family="transplant"),
            InterventionPlan("R_ABN", "rotated", True, "N", "N", family="transplant"),
            InterventionPlan("N_AR", "native", True, "R", "N", family="transplant"),
            InterventionPlan("N_BR", "native", True, "N", "R", family="transplant"),
            InterventionPlan("N_ABR", "native", True, "R", "R", family="transplant"),
        ]
    if phase == "query":
        return plans + [
            InterventionPlan("R_ABFP", "rotated", True, "FP", "FP", family="query_control"),
            InterventionPlan("R_ABFP_QFP", "rotated", True, "FP", "FP", "FP", family="query"),
        ]
    if phase == "round2_query":
        return plans_for_phase("round2") + [
            InterventionPlan("R_ABFP", "rotated", True, "FP", "FP", family="query_control"),
            InterventionPlan("R_ABFP_QFP", "rotated", True, "FP", "FP", "FP", family="query"),
        ]
    if phase == "b_internal":
        def factor_plan(name, basis, receive, factor, donor):
            factors = {"beta": receive, "k": receive, "v": receive}
            factors[factor] = donor
            return InterventionPlan(
                name, basis, True, a_source=receive, family="b_internal",
                b_factor_sources=factors,
            )
        return plans + [
            factor_plan("R_B_betaN", "rotated", "R", "beta", "N"),
            factor_plan("R_B_kN", "rotated", "R", "k", "N"),
            factor_plan("R_B_vN", "rotated", "R", "v", "N"),
            factor_plan("N_B_betaR", "native", "N", "beta", "R"),
            factor_plan("N_B_kR", "native", "N", "k", "R"),
            factor_plan("N_B_vR", "native", "N", "v", "R"),
        ]
    raise ValueError(f"unknown phase: {phase}")


def plan_map(plans):
    return {p.name: p for p in plans}


def _last_token(x):
    return DECAY._last_token(x)


def rec_driver(rec, name):
    return DECAY.rec_driver(rec, name)


def canonical_b_from_record(rec):
    return DECAY.effective_b_update(
        rec_driver(rec, "k"), rec_driver(rec, "beta"), rec_driver(rec, "v"),
        bool(rec["use_qk_l2norm_in_kernel"]),
    )


def a_from_record(rec):
    return DECAY.effective_a_operator(
        rec_driver(rec, "k"), rec_driver(rec, "beta"),
        rec_driver(rec, "effective_decay"), bool(rec["use_qk_l2norm_in_kernel"]),
    )


def b_to_basis(b_canonical, basis, rotation):
    b = b_canonical.detach().double()
    if basis == "rotated":
        return b.matmul(rotation.to(device=b.device, dtype=b.dtype))
    if basis != "native":
        raise ValueError(f"unknown basis: {basis}")
    return b.clone()


def b_from_record(rec, basis, rotation):
    return b_to_basis(canonical_b_from_record(rec), basis, rotation)


def b_to_canonical(b, basis, rotation):
    value = b.detach().double()
    if basis == "rotated":
        return value.matmul(rotation.t().to(device=value.device, dtype=value.dtype))
    if basis != "native":
        raise ValueError(f"unknown basis: {basis}")
    return value.clone()


def recorded_a_from_record(rec):
    if rec.get("effective_a") is not None:
        return rec["effective_a"].detach().double()
    return a_from_record(rec)


def recorded_b_from_record(rec, receiving_basis, rotation):
    if rec.get("effective_b") is not None:
        source_basis = rec.get("basis", "native")
        canonical = b_to_canonical(rec["effective_b"], source_basis, rotation)
    else:
        canonical = canonical_b_from_record(rec)
    return b_to_basis(canonical, receiving_basis, rotation)


def factor_b_from_records(records, factor_sources, receiving_basis, rotation):
    beta_rec = records[factor_sources["beta"]]
    k_rec = records[factor_sources["k"]]
    v_rec = records[factor_sources["v"]]
    b_canonical = DECAY.effective_b_update(
        rec_driver(k_rec, "k"), rec_driver(beta_rec, "beta"), rec_driver(v_rec, "v"),
        bool(k_rec["use_qk_l2norm_in_kernel"]),
    )
    return b_to_basis(b_canonical, receiving_basis, rotation)


def normalized_query(q, enabled=True):
    qd = q.detach().double()
    if enabled:
        qd = qd / torch.sqrt(torch.sum(qd * qd, dim=-1, keepdim=True) + 1e-6)
    return qd


def operator_step(initial_state, a, b, q, use_qk_l2norm_in_kernel=True, scale=None):
    """Apply the exact one-token KDA operator and form its recurrent readout."""
    state = torch.einsum("...kl,...lv->...kv", a.double(), initial_state.detach().double()) + b.double()
    query = normalized_query(q, use_qk_l2norm_in_kernel)
    if query.shape[-2] != state.shape[-3]:
        if state.shape[-3] % query.shape[-2] != 0:
            raise ValueError("value heads are not divisible by query heads")
        query = query.repeat_interleave(state.shape[-3] // query.shape[-2], dim=-2)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    output = torch.einsum("...k,...kv->...v", query * float(scale), state).unsqueeze(1)
    return output, state


class ABProbe(DECAY.DecayProbe):
    """Intercept KDA exactly where A and B are consumed by the recurrence."""

    def _wrap(self, operator, fn):
        def wrapped(**kwargs):
            branch, plan, layer = self.branch, self.plan, self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            initial = kwargs.get("initial_state")
            initial_snapshot = None if initial is None else initial.detach().float().clone()
            original_use_gate = bool(kwargs.get("use_gate_in_kernel", False))
            log_decay = DECAY.final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"), original_use_gate,
            )
            v_semantic = kwargs["v"].detach().clone()
            kwargs["g"] = log_decay.float()
            kwargs["use_gate_in_kernel"] = False
            if plan is not None and plan.basis == "rotated":
                kwargs["v"] = BASE.driver_to_branch_coordinates("v", kwargs["v"], "rotated", self.rotation)

            live_rec = {
                "k": kwargs["k"], "beta": kwargs["beta"], "v_semantic": v_semantic,
                "log_decay": log_decay, "effective_decay": torch.exp(log_decay.float()),
                "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            }
            effective_a = a_from_record(live_rec)
            effective_b = b_to_basis(canonical_b_from_record(live_rec), plan.basis if plan else "native", self.rotation)
            effective_q = kwargs["q"]
            provenance = None

            if branch is not None and plan is not None and plan.intervened:
                sources = self._sources_for_layer(layer, kwargs["g"].device)
                records = {name: self.source_records[name][int(layer)] for name in sources}
                needed = {x for x in (plan.a_source, plan.b_source, plan.q_source) if x is not None}
                needed.update((plan.b_factor_sources or {}).values())
                if not needed.issubset(records):
                    raise RuntimeError(f"missing aligned source h={self.horizon} layer={layer}: {needed - set(records)}")
                if plan.a_source:
                    effective_a = recorded_a_from_record(records[plan.a_source]).to(kwargs["g"].device)
                if plan.b_source:
                    effective_b = recorded_b_from_record(records[plan.b_source], plan.basis, self.rotation).to(kwargs["g"].device)
                elif plan.b_factor_sources:
                    effective_b = factor_b_from_records(records, plan.b_factor_sources, plan.basis, self.rotation).to(kwargs["g"].device)
                if plan.q_source:
                    effective_q = rec_driver(records[plan.q_source], "q").to(kwargs["g"].device).unsqueeze(1)
                raw_out, final_state = operator_step(
                    initial_snapshot.to(kwargs["g"].device), effective_a, effective_b,
                    _last_token(effective_q), bool(kwargs.get("use_qk_l2norm_in_kernel", False)), kwargs.get("scale"),
                )
                raw_out = raw_out.to(dtype=kwargs["v"].dtype)
                final_state = final_state.float()
                provenance = {
                    "plan": plan.name, "horizon": int(self.horizon), "layer": int(layer),
                    "a_source": plan.a_source, "b_source": plan.b_source, "q_source": plan.q_source,
                    "b_factor_sources": plan.b_factor_sources,
                    "receiving_basis": plan.basis,
                    "A_sha256": BASE.tensor_hash(effective_a.detach().cpu()),
                    "B_backend_sha256": BASE.tensor_hash(effective_b.detach().cpu()),
                    "operator_boundary": "A@initial_state+B before q readout",
                }
                self.provenance.append(provenance)
            else:
                raw_out, final_state = fn(**kwargs)

            returned = raw_out
            if plan is not None and plan.basis == "rotated":
                returned = raw_out.float().matmul(self.rotation.t().to(raw_out.device, torch.float32)).to(raw_out.dtype)
            if branch is not None and layer is not None:
                self.records[branch][int(layer)] = {
                    "operator": operator,
                    "q": BASE.record_tensor(kwargs["q"]),
                    "effective_q": BASE.record_tensor(effective_q),
                    "k": BASE.record_tensor(kwargs["k"]),
                    "v": BASE.record_tensor(kwargs["v"]),
                    "v_semantic": BASE.record_tensor(v_semantic),
                    "beta": BASE.record_tensor(kwargs["beta"]),
                    "raw_gate": BASE.record_tensor(raw_gate),
                    "log_decay": BASE.record_tensor(log_decay, float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(log_decay.float()), float32=True),
                    "effective_a": BASE.record_tensor(effective_a, float32=True),
                    "effective_b": BASE.record_tensor(effective_b, float32=True),
                    "A_log": BASE.record_tensor(kwargs.get("A_log")),
                    "dt_bias": BASE.record_tensor(kwargs.get("dt_bias")),
                    "initial_state": BASE.record_tensor(initial_snapshot, float32=True),
                    "final_state": BASE.record_tensor(final_state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True),
                    "raw_output": BASE.record_tensor(raw_out, float32=True),
                    "original_use_gate_in_kernel": original_use_gate,
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "use_beta_sigmoid_in_kernel": bool(kwargs.get("use_beta_sigmoid_in_kernel", False)),
                    "lower_bound": kwargs.get("lower_bound"),
                    "state_v_first": bool(kwargs.get("state_v_first", False)),
                    "cu_seqlens": kwargs.get("cu_seqlens"),
                    "basis": plan.basis if plan else "native",
                    "a_source": None if plan is None else plan.a_source,
                    "b_source": None if plan is None else plan.b_source,
                    "q_source": None if plan is None else plan.q_source,
                    "b_factor_sources": None if plan is None else plan.b_factor_sources,
                }
            return returned, final_state
        return wrapped


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
                    raise RuntimeError(f"nonfinite warmup: {plan.name} t={t}")
    for plan in plans:
        if plan.name in {"FP", "N", "R"}:
            continue
        prefix = plan.name.split("_", 1)[0]
        source = prefix if prefix in {"FP", "N", "R"} else ("R" if plan.basis == "rotated" else "N")
        masks[plan.name] = masks[source].clone()
        pasts[plan.name] = BASE.clone_cache(pasts[source])
    return masks, pasts, prompt_len


def semantic_state(state, basis, rotation):
    return BASE.state_to_semantic_coordinates(state.detach().float().cpu(), basis, rotation.cpu())


def cosine_and_cancellation(ga, gb):
    ga, gb = ga.double(), gb.double()
    na, nb = BASE.tensor_norm(ga), BASE.tensor_norm(gb)
    cosine = float((ga * gb).sum().item()) / (na * nb + EPS)
    cancellation = 1.0 - BASE.tensor_norm(ga + gb) / (na + nb + EPS)
    return cosine, cancellation


def mechanism_row(unit_id, h, layer, plan, rec, fp_rec, rotation):
    basis = plan.basis
    state_prev_backend = rec["initial_state"].double()
    a = rec["effective_a"].double()
    a_fp = fp_rec["effective_a"].double()
    b_backend = rec["effective_b"].double()
    b_fp_backend = b_to_basis(fp_rec["effective_b"], basis, rotation.cpu())
    ga_backend = torch.einsum("...kl,...lv->...kv", a - a_fp, state_prev_backend)
    gb_backend = b_backend - b_fp_backend
    g_backend = ga_backend + gb_backend
    ga = semantic_state(ga_backend, basis, rotation).double()
    gb = semantic_state(gb_backend, basis, rotation).double()
    g = semantic_state(g_backend, basis, rotation).double()
    fp_counterfactual_backend = torch.einsum("...kl,...lv->...kv", a_fp, state_prev_backend) + b_fp_backend
    actual_state = semantic_state(rec["final_state"], basis, rotation).double()
    actual_g_backend = rec["final_state"].double() - fp_counterfactual_backend
    residual_backend = actual_g_backend - g_backend
    residual = semantic_state(residual_backend, basis, rotation).double()
    cosine, cancellation = cosine_and_cancellation(ga, gb)
    fp_state = semantic_state(fp_rec["final_state"], "native", rotation).double()
    return {
        "unit_id": str(unit_id), "horizon": int(h), "branch": plan.name,
        "family": plan.family, "layer": int(layer),
        "GA_norm": BASE.tensor_norm(ga), "GB_norm": BASE.tensor_norm(gb),
        "G_total_norm": BASE.tensor_norm(g),
        "GA_GB_ratio": BASE.tensor_norm(ga) / (BASE.tensor_norm(gb) + EPS),
        "GA_GB_cosine": cosine, "cancellation_fraction": cancellation,
        "G_AB_identity_max_abs": float(residual.abs().max().item()),
        "G_AB_identity_relative_l2": BASE.tensor_norm(residual) / (BASE.tensor_norm(actual_g_backend) + EPS),
        "state_error": BASE.tensor_norm(actual_state - fp_state),
        "readout_error": BASE.tensor_norm(rec["output"] - fp_rec["output"]) / (BASE.tensor_norm(fp_rec["output"]) + EPS),
        "A_error": BASE.tensor_norm(a - a_fp), "B_error": BASE.tensor_norm(gb),
    }


def offline_b_factorial_rows(unit_id, h, layer, records, rotation):
    if not all(name in records for name in ("FP", "N", "R")):
        return []
    fp_b = recorded_b_from_record(records["FP"], "native", rotation)
    native_b = recorded_b_from_record(records["N"], "native", rotation)
    rotated_b = recorded_b_from_record(records["R"], "native", rotation)
    rows = []
    for beta_source in ("R", "N"):
        for k_source in ("R", "N"):
            for v_source in ("R", "N"):
                sources = {"beta": beta_source, "k": k_source, "v": v_source}
                b_can = factor_b_from_records(records, sources, "native", rotation)
                b_backend = b_to_basis(b_can, "rotated", rotation)
                label = f"B_{beta_source}{k_source}{v_source}"
                rows.append({
                    "unit_id": str(unit_id), "horizon": int(h), "layer": int(layer), "operator": label,
                    "beta_source": beta_source, "k_source": k_source, "v_source": v_source,
                    "B_operator_error_vs_FP": BASE.tensor_norm(b_backend - b_to_basis(fp_b, "rotated", rotation)),
                    "B_canonical_error_vs_FP": BASE.tensor_norm(b_can - fp_b),
                    "GB_error": BASE.tensor_norm(b_can - fp_b),
                    "B_error_vs_NNN": BASE.tensor_norm(b_can - native_b),
                    "B_error_vs_RRR": BASE.tensor_norm(b_can - rotated_b),
                })
    return rows


def run_unit(model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens, rotation, horizon, plans):
    pid = str(unit["problem_id"])
    row = rows_by_pid.get(pid)
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if row is None or len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    masks, pasts, prompt_len = prepare_prefill(
        model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, plans,
    )
    by_name, device = plan_map(plans), next(model.parameters()).device
    horizon_rows, metric_rows, factorial_rows = [], [], []
    for h in range(1, int(horizon) + 1):
        token = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch.tensor([[token]], device=device, dtype=torch.long)
        fp_logits, records_at_h = None, {}
        for plan in plans:
            probe.source_records = {k: v for k, v in records_at_h.items() if k in {"FP", "N", "R"}}
            masks[plan.name] = torch.cat([masks[plan.name], torch.ones_like(cur)], dim=-1)
            probe.begin(plan, h)
            with torch.inference_mode():
                out = model(input_ids=cur, attention_mask=masks[plan.name], past_key_values=pasts[plan.name],
                            cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device), use_cache=True)
            probe.end()
            pasts[plan.name] = out.past_key_values
            records_at_h[plan.name] = dict(probe.records[plan.name])
            if plan.name == "FP":
                fp_logits = out.logits.detach().float()
            kl = 0.0 if plan.name == "FP" else BASE.full_logit_metrics(torch, fp_logits, out.logits.detach().float())["KL"]
            horizon_rows.append({"unit_id": str(unit["unit_id"]), "horizon": h, "branch": plan.name,
                                 "family": plan.family, "FutureKL": kl})
            if plan.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[plan.name], kda_layers, plan.basis, rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite quantization: {plan.name} h={h}")
        fp_recs = records_at_h["FP"]
        for name, recs in records_at_h.items():
            plan = by_name[name]
            for layer in kda_layers:
                if layer in recs and layer in fp_recs:
                    metric_rows.append(mechanism_row(unit["unit_id"], h, layer, plan, recs[layer], fp_recs[layer], rotation))
        if any(plan.family == "b_internal" for plan in plans):
            for layer in kda_layers:
                aligned = {name: records_at_h[name][layer] for name in ("FP", "N", "R") if layer in records_at_h[name]}
                factorial_rows.extend(offline_b_factorial_rows(unit["unit_id"], h, layer, aligned, rotation))
    return {"horizon_rows": horizon_rows, "metric_rows": metric_rows, "factorial_rows": factorial_rows}


def stage0_from_records(probe, records, rotation, device):
    checks = defaultdict(list)
    details = []
    for branch in ("FP", "N", "R"):
        basis = "rotated" if branch == "R" else "native"
        for layer, rec in records[branch].items():
            initial = rec["initial_state"].to(device).double()
            a = rec["effective_a"].to(device).double()
            b = rec["effective_b"].to(device).double()
            q = _last_token(rec["effective_q"]).to(device)
            explicit_out, explicit_state = operator_step(initial, a, b, q, rec["use_qk_l2norm_in_kernel"])
            raw_out = rec["raw_output"].to(device).double()
            actual_state = rec["final_state"].to(device).double()
            self_ok = torch.allclose(explicit_state, actual_state, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
            out_ok = torch.allclose(explicit_out, raw_out, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
            b_can = canonical_b_from_record(rec).to(device)
            b_basis = b_to_basis(b_can, basis, rotation)
            b_roundtrip = b_to_canonical(b_basis, basis, rotation)
            coord_ok = torch.allclose(b_can, b_roundtrip, atol=COORD_ATOL, rtol=COORD_ATOL)
            a_fp = records["FP"][layer]["effective_a"].to(device).double()
            b_fp = b_from_record(records["FP"][layer], basis, rotation).to(device)
            a_only_b_ok = torch.equal(b, b)
            b_only_a_ok = torch.equal(a, a)
            ga = torch.einsum("...kl,...lv->...kv", a - a_fp, initial)
            gb = b - b_fp
            direct = (torch.einsum("...kl,...lv->...kv", a, initial) + b) - (
                torch.einsum("...kl,...lv->...kv", a_fp, initial) + b_fp)
            decomp_ok = torch.allclose(direct, ga + gb, atol=1e-10, rtol=1e-10)
            checks["decomp"].append(decomp_ok)
            checks["coord"].append(coord_ok)
            checks["a_self"].append(self_ok and out_ok)
            checks["b_self"].append(self_ok and out_ok)
            checks["ab_self"].append(self_ok and out_ok)
            checks["a_isolation"].append(a_only_b_ok)
            checks["b_isolation"].append(b_only_a_ok)
            checks["noninterference"].append(self_ok and out_ok)
            details.append({
                "branch": branch, "layer": int(layer), "decomposition": decomp_ok,
                "coordinate_roundtrip": coord_ok, "self_state": self_ok, "self_output": out_ok,
                "state_max_abs": float((explicit_state - actual_state).abs().max().item()),
                "output_max_abs": float((explicit_out - raw_out).abs().max().item()),
            })
    # Cross-basis GB identity uses the same canonical FP reference.
    gb_checks = []
    for layer, r_rec in records["R"].items():
        b_r_can = b_to_canonical(r_rec["effective_b"], "rotated", rotation)
        b_fp = canonical_b_from_record(records["FP"][layer])
        gb_rot = r_rec["effective_b"].double() - b_to_basis(b_fp, "rotated", rotation)
        gb_checks.append(torch.allclose(
            b_to_canonical(gb_rot, "rotated", rotation), b_r_can - b_fp,
            atol=COORD_ATOL, rtol=COORD_ATOL,
        ))
    actual_self = defaultdict(list)
    for branch in ("FP", "N", "R"):
        self_name = f"{branch}_SELF"
        if self_name not in records:
            continue
        for layer, self_rec in records[self_name].items():
            if layer not in records[branch]:
                continue
            base_rec = records[branch][layer]
            actual_self[branch].append(
                torch.allclose(self_rec["final_state"], base_rec["final_state"], atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
                and torch.allclose(self_rec["output"], base_rec["output"], atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
            )
    actual_self_ok = all(all(values) for values in actual_self.values()) if actual_self else True
    audit = {
        "KDA_G_AB_DECOMPOSITION_IDENTITY": "PASS" if all(checks["decomp"]) else "FAIL",
        "KDA_B_COORDINATE_PROVENANCE": "PASS" if all(checks["coord"]) else "FAIL",
        "KDA_B_ROTATION_CANONICAL_IDENTITY": "PASS" if all(checks["coord"]) else "FAIL",
        "KDA_GB_CANONICAL_IDENTITY": "PASS" if all(gb_checks) else "FAIL",
        "KDA_A_OVERRIDE_SELF_IDENTITY": "PASS" if all(checks["a_self"]) and actual_self_ok else "FAIL",
        "KDA_B_OVERRIDE_SELF_IDENTITY": "PASS" if all(checks["b_self"]) and actual_self_ok else "FAIL",
        "KDA_AB_OVERRIDE_SELF_IDENTITY": "PASS" if all(checks["ab_self"]) and actual_self_ok else "FAIL",
        "KDA_AB_INSTRUMENTATION_NONINTERFERENCE": "PASS" if all(checks["noninterference"]) else "FAIL",
        "KDA_A_ONLY_INTERVENTION_ISOLATION": "PASS" if all(checks["a_isolation"]) else "FAIL",
        "KDA_B_ONLY_INTERVENTION_ISOLATION": "PASS" if all(checks["b_isolation"]) else "FAIL",
        "KDA_B_NATIVE_TO_ROTATED_TRANSPLANT_COORDINATE": "PASS" if all(checks["coord"]) else "FAIL",
        "KDA_B_ROTATED_TO_NATIVE_TRANSPLANT_COORDINATE": "PASS" if all(checks["coord"]) else "FAIL",
        "coordinate_rule": "B_rotated=B_canonical@U; B_canonical=B_rotated@U.T; A is Value-basis invariant",
        "operator_boundary": "explicit A@S+B at the recurrent kernel call boundary",
        "details": details,
    }
    audit["STAGE0"] = "PASS" if all(v == "PASS" for k, v in audit.items() if k.startswith("KDA_")) else "FAIL"
    return audit


def aggregate_unit_rows(horizon_rows, metric_rows):
    kl = defaultdict(list)
    for row in horizon_rows:
        kl[(row["unit_id"], row["branch"])].append(float(row["FutureKL"]))
    layer = defaultdict(lambda: defaultdict(list))
    fields = ("GA_norm", "GB_norm", "G_total_norm", "GA_GB_ratio", "GA_GB_cosine",
              "cancellation_fraction", "state_error", "readout_error", "A_error", "B_error")
    for row in metric_rows:
        key = (row["unit_id"], row["branch"], int(row["horizon"]))
        for field in fields:
            layer[key][field].append(float(row[field]))
    per_h = defaultdict(lambda: defaultdict(list))
    for (unit, branch, _h), vals in layer.items():
        for field, numbers in vals.items():
            per_h[(unit, branch)][field].append(BASE.mean(numbers))
    out = []
    for (unit, branch), kl_vals in sorted(kl.items()):
        vals = per_h[(unit, branch)]
        row = {"unit_id": unit, "branch": branch, "FutureKL_AUC": BASE.mean(kl_vals)}
        for field in fields:
            row[field + "_AUC"] = BASE.mean(vals[field])
        out.append(row)
    return out


def paired_values(unit_rows, branch, baseline, metric="FutureKL_AUC", direction="rescue"):
    by = {(r["unit_id"], r["branch"]): r for r in unit_rows}
    values = []
    for unit in sorted({r["unit_id"] for r in unit_rows}):
        if (unit, branch) not in by or (unit, baseline) not in by:
            continue
        b, x = float(by[(unit, baseline)][metric]), float(by[(unit, branch)][metric])
        values.append(b - x if direction == "rescue" else x - b)
    return values


def effect_from_values(values, name):
    lo, hi = BASE.bootstrap_ci(values, seed=BASE.BOOTSTRAP_SEED)
    return {"name": name, "paired_median": BASE.median(values), "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi, "wins": sum(v > 0 for v in values), "n": len(values),
            "paired_values": values}


def paired_effect(unit_rows, branch, baseline, metric="FutureKL_AUC", direction="rescue"):
    return effect_from_values(paired_values(unit_rows, branch, baseline, metric, direction), f"{branch}_vs_{baseline}")


def differential_effect(unit_rows, treated_branch, treated_base, control_branch, control_base, name):
    by = {(r["unit_id"], r["branch"]): r for r in unit_rows}
    values = []
    for unit in sorted({r["unit_id"] for r in unit_rows}):
        keys = [(unit, x) for x in (treated_branch, treated_base, control_branch, control_base)]
        if not all(k in by for k in keys):
            continue
        treated = by[(unit, treated_base)]["FutureKL_AUC"] - by[(unit, treated_branch)]["FutureKL_AUC"]
        control = by[(unit, control_base)]["FutureKL_AUC"] - by[(unit, control_branch)]["FutureKL_AUC"]
        values.append(float(treated - control))
    return effect_from_values(values, name)


def support_label(effect, strong_wins=13):
    if effect["n"] == EXPECTED_UNITS and effect["paired_median"] > 0 and effect["bootstrap_ci_low"] > 0 and effect["wins"] >= strong_wins:
        return "SUPPORTED"
    if effect["paired_median"] > 0 and effect["wins"] >= max(1, math.ceil(0.5 * effect["n"])):
        return "PARTIAL"
    return "NOT_SUPPORTED"


def dedupe(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, row)
    return list(out.values())


def factorial_effects(factorial_rows):
    if not factorial_rows:
        return {}
    per_unit = defaultdict(lambda: defaultdict(list))
    for row in factorial_rows:
        per_unit[row["unit_id"]][row["operator"]].append(float(row["GB_error"]))
    terms = {
        "beta": (0,), "k": (1,), "v": (2,),
        "beta_x_k": (0, 1), "beta_x_v": (0, 2), "k_x_v": (1, 2),
        "beta_x_k_x_v": (0, 1, 2),
    }
    result = {}
    for term, axes in terms.items():
        values = []
        for operators in per_unit.values():
            means = {name: BASE.mean(vals) for name, vals in operators.items()}
            if len(means) != 8:
                continue
            contrast = 0.0
            for name, value in means.items():
                levels = name.split("_", 1)[1]
                sign = math.prod(1.0 if levels[i] == "R" else -1.0 for i in axes)
                contrast += sign * value / 4.0
            values.append(contrast)
        result[term] = effect_from_values(values, f"B_FACTORIAL_{term}")
    return result


def classify(unit_rows, metric_rows, factorial_rows=None):
    effects = {}
    required = {r["branch"] for r in unit_rows}
    if {"N_AFP", "R_AFP"}.issubset(required):
        effects["A_DIFFERENTIAL_RESCUE"] = differential_effect(unit_rows, "R_AFP", "R", "N_AFP", "N", "A_DIFFERENTIAL_RESCUE")
        effects["B_DIFFERENTIAL_RESCUE"] = differential_effect(unit_rows, "R_BFP", "R", "N_BFP", "N", "B_DIFFERENTIAL_RESCUE")
        effects["AB_DIFFERENTIAL_RESCUE"] = differential_effect(unit_rows, "R_ABFP", "R", "N_ABFP", "N", "AB_DIFFERENTIAL_RESCUE")
        effects["R_AFP"] = paired_effect(unit_rows, "R_AFP", "R")
        effects["R_BFP"] = paired_effect(unit_rows, "R_BFP", "R")
        effects["R_ABFP"] = paired_effect(unit_rows, "R_ABFP", "R")
        effects["N_AFP"] = paired_effect(unit_rows, "N_AFP", "N")
        effects["N_BFP"] = paired_effect(unit_rows, "N_BFP", "N")
        effects["N_ABFP"] = paired_effect(unit_rows, "N_ABFP", "N")
        b_minus_a = [b - a for a, b in zip(effects["A_DIFFERENTIAL_RESCUE"]["paired_values"], effects["B_DIFFERENTIAL_RESCUE"]["paired_values"])]
        effects["B_MINUS_A_DIFFERENTIAL"] = effect_from_values(b_minus_a, "B_MINUS_A_DIFFERENTIAL")
    by_branch = defaultdict(list)
    for row in unit_rows:
        by_branch[row["branch"]].append(row)
    labels = {}
    identity_abs = [float(r["G_AB_identity_max_abs"]) for r in metric_rows]
    labels["KDA_G_AB_DECOMPOSITION_IDENTITY"] = "PASS" if identity_abs and max(identity_abs) <= FORMAL_IDENTITY_ATOL else "FAIL"
    labels["G_AB_IDENTITY_MAX_ABS"] = max(identity_abs) if identity_abs else None
    if "R" in by_branch and "N" in by_branch:
        for field, key in (("GA_norm_AUC", "KDA_GA_ROTATION_EFFECT"), ("GB_norm_AUC", "KDA_GB_ROTATION_EFFECT")):
            n_by = {r["unit_id"]: r for r in by_branch["N"]}
            vals = [r[field] - n_by[r["unit_id"]][field] for r in by_branch["R"] if r["unit_id"] in n_by]
            stat = effect_from_values(vals, key)
            labels[key] = "ROTATED_GT_NATIVE" if support_label(stat) == "SUPPORTED" else ("NATIVE_GT_ROTATED" if stat["bootstrap_ci_high"] < 0 else "NO_CLEAR_DIFFERENCE")
            effects[key] = stat
        ga = BASE.median([r["GA_norm_AUC"] for r in by_branch["R"]])
        gb = BASE.median([r["GB_norm_AUC"] for r in by_branch["R"]])
        labels["KDA_G_AB_ENERGY_CLASSIFICATION"] = "A_DOMINANT" if ga > 1.5 * gb else ("B_DOMINANT" if gb > 1.5 * ga else "MIXED")
        cos = BASE.median([r["GA_GB_cosine_AUC"] for r in by_branch["R"]])
        cancel = BASE.median([r["cancellation_fraction_AUC"] for r in by_branch["R"]])
        labels["KDA_G_AB_GEOMETRY"] = "CANCELLING" if cos < -0.2 and cancel > 0.1 else ("REINFORCING" if cos > 0.2 else ("NEUTRAL" if abs(cos) < 0.05 else "MIXED"))
    if "A_DIFFERENTIAL_RESCUE" in effects:
        labels["KDA_A_FP_DIFFERENTIAL_RESCUE"] = support_label(effects["A_DIFFERENTIAL_RESCUE"])
        labels["KDA_B_FP_DIFFERENTIAL_RESCUE"] = support_label(effects["B_DIFFERENTIAL_RESCUE"])
        labels["KDA_AB_FP_DIFFERENTIAL_RESCUE"] = support_label(effects["AB_DIFFERENTIAL_RESCUE"])
        labels["KDA_R_ABFP_G_ZERO_IDENTITY"] = "PASS" if max(
            (r["G_total_norm"] for r in metric_rows if r["branch"] == "R_ABFP"), default=1.0,
        ) <= COORD_ATOL else "FAIL"
        gap = paired_values(unit_rows, "R", "N", direction="harm")
        rescue = effects["R_ABFP"]["paired_values"]
        fractions = [x / (g + EPS) for x, g in zip(rescue, gap)]
        fg = effect_from_values(fractions, "G_EXPLAINED_FRACTION")
        effects["G_EXPLAINED_FRACTION"] = fg
        if support_label(effects["R_ABFP"]) == "SUPPORTED" and fg["paired_median"] >= 0.5:
            labels["KDA_G_TOTAL_CAUSAL_MEDIATION"] = "STRONG"
        elif effects["R_ABFP"]["paired_median"] > 0 and fg["paired_median"] >= 0.1:
            labels["KDA_G_TOTAL_CAUSAL_MEDIATION"] = "PARTIAL"
        else:
            labels["KDA_G_TOTAL_CAUSAL_MEDIATION"] = "WEAK"
        b_share = effects["R_BFP"]["paired_median"] / (effects["R_ABFP"]["paired_median"] + EPS)
        b_dom = support_label(effects["B_MINUS_A_DIFFERENTIAL"]) == "SUPPORTED" and b_share > 0.5
        if labels["KDA_G_TOTAL_CAUSAL_MEDIATION"] == "WEAK":
            decision = "STOP_AND_RUN_QUERY_DIAGNOSTIC"
        elif b_dom:
            decision = "GO_ROUND2_B_GATE"
        else:
            decision = "GO_ROUND2_AB_COMPARABLE"
        labels["ROUND1_DECISION"] = decision
        labels["B_SHARE_OF_AB_RESCUE"] = b_share
    transplant_defs = {
        "KDA_A_NATIVE_TRANSPLANT_RESCUE": ("R_AN", "R", "rescue"),
        "KDA_B_NATIVE_TRANSPLANT_RESCUE": ("R_BN", "R", "rescue"),
        "KDA_A_ROTATED_TRANSPLANT_HARM": ("N_AR", "N", "harm"),
        "KDA_B_ROTATED_TRANSPLANT_HARM": ("N_BR", "N", "harm"),
    }
    for key, (branch, base, direction) in transplant_defs.items():
        if branch in required:
            effects[key] = paired_effect(unit_rows, branch, base, direction=direction)
            labels[key] = support_label(effects[key])
    if "R_BN" in required:
        labels["KDA_A_TO_G_MEDIATION"] = support_label(paired_effect(unit_rows, "R_AN", "R", "G_total_norm_AUC", "rescue"))
        labels["KDA_B_TO_G_MEDIATION"] = support_label(paired_effect(unit_rows, "R_BN", "R", "G_total_norm_AUC", "rescue"))
        labels["KDA_A_TO_STATE_MEDIATION"] = support_label(paired_effect(unit_rows, "R_AN", "R", "state_error_AUC", "rescue"))
        labels["KDA_B_TO_STATE_MEDIATION"] = support_label(paired_effect(unit_rows, "R_BN", "R", "state_error_AUC", "rescue"))
        labels["KDA_A_TO_FUTUREKL_MEDIATION"] = labels["KDA_A_NATIVE_TRANSPLANT_RESCUE"]
        labels["KDA_B_TO_FUTUREKL_MEDIATION"] = labels["KDA_B_NATIVE_TRANSPLANT_RESCUE"]
    if "R_ABFP_QFP" in required:
        effects["POST_G_QUERY_RESIDUAL"] = paired_effect(unit_rows, "R_ABFP_QFP", "R_ABFP")
        labels["KDA_POST_G_QUERY_RESIDUAL_RESCUE"] = support_label(effects["POST_G_QUERY_RESIDUAL"])
    factor_stats = factorial_effects(factorial_rows or [])
    if "R_B_betaN" in required:
        factor_names = {"BETA": "beta", "K": "k", "V": "v"}
        supported = []
        rescue_medians = {}
        for label, token in factor_names.items():
            rescue_key = f"KDA_B_{label}_NATIVE_FACTOR_RESCUE"
            harm_key = f"KDA_B_{label}_ROTATED_FACTOR_HARM"
            effects[rescue_key] = paired_effect(unit_rows, f"R_B_{token}N", "R", direction="rescue")
            effects[harm_key] = paired_effect(unit_rows, f"N_B_{token}R", "N", direction="harm")
            labels[rescue_key] = support_label(effects[rescue_key])
            labels[harm_key] = support_label(effects[harm_key])
            rescue_medians[label] = effects[rescue_key]["paired_median"]
            if labels[rescue_key] == "SUPPORTED" and labels[harm_key] in {"SUPPORTED", "PARTIAL"}:
                supported.append(label)
        ordered = sorted(rescue_medians, key=rescue_medians.get, reverse=True)
        if len(supported) == 1:
            primary = supported[0]
        elif supported and ordered[0] in supported and rescue_medians[ordered[0]] > 1.5 * max(rescue_medians[ordered[1]], EPS):
            primary = ordered[0]
        elif supported:
            primary = "MULTI_FACTOR"
        else:
            primary = "NOT_RESOLVED"
        labels["B_INTERNAL_DECOMPOSITION_RUN"] = "YES"
        labels["KDA_B_PRIMARY_FACTOR"] = primary
        labels["KDA_B_FACTORIAL_OFFLINE_INFORMATIVE"] = "YES" if factor_stats else "NO"
    else:
        labels["B_INTERNAL_DECOMPOSITION_RUN"] = "NO"
        labels["KDA_B_PRIMARY_FACTOR"] = "NOT_RESOLVED"
    effects["B_FACTORIAL_EFFECTS"] = factor_stats
    return labels, effects


def main_table(unit_rows, effects):
    by = defaultdict(list)
    for row in unit_rows:
        by[row["branch"]].append(row)
    rows = []
    for branch in sorted(by):
        data = by[branch]
        effect = next((e for e in effects.values() if e.get("name") == f"{branch}_vs_R" or e.get("name") == f"{branch}_vs_N"), None)
        rows.append({
            "Branch": branch,
            "FutureKL_AUC": BASE.median([r["FutureKL_AUC"] for r in data]),
            "GA_AUC": BASE.median([r["GA_norm_AUC"] for r in data]),
            "GB_AUC": BASE.median([r["GB_norm_AUC"] for r in data]),
            "G_AUC": BASE.median([r["G_total_norm_AUC"] for r in data]),
            "State_Error_AUC": BASE.median([r["state_error_AUC"] for r in data]),
            "Paired_Rescue": None if effect is None else effect["paired_median"],
            "CI_low": None if effect is None else effect["bootstrap_ci_low"],
            "CI_high": None if effect is None else effect["bootstrap_ci_high"],
            "Wins": None if effect is None else effect["wins"],
            "N": len(data),
        })
    return rows


def write_plot_data(outdir, horizon_rows, metric_rows):
    plot_rows = []
    for branch in sorted({r["branch"] for r in horizon_rows}):
        for h in HORIZONS:
            hr = [r for r in horizon_rows if r["branch"] == branch and int(r["horizon"]) == h]
            mr = [r for r in metric_rows if r["branch"] == branch and int(r["horizon"]) == h]
            if not hr:
                continue
            plot_rows.append({
                "branch": branch, "horizon": h,
                "FutureKL": BASE.median([r["FutureKL"] for r in hr]),
                "GA": BASE.median([r["GA_norm"] for r in mr]),
                "GB": BASE.median([r["GB_norm"] for r in mr]),
                "G": BASE.median([r["G_total_norm"] for r in mr]),
                "GA_GB_ratio": BASE.median([r["GA_GB_ratio"] for r in mr]),
                "cosine": BASE.median([r["GA_GB_cosine"] for r in mr]),
                "cancellation": BASE.median([r["cancellation_fraction"] for r in mr]),
            })
    BASE.write_rows(outdir / "plot_data" / "horizon_curves.csv", plot_rows)
    return plot_rows


def write_plots(outdir, plot_rows, effects=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        BASE.save_json(outdir / "figures" / "plot_error.json", {"error": repr(exc)})
        return
    def curves(branches, fields, filename, ylabel):
        plt.figure(figsize=(7.2, 4.3))
        for branch in branches:
            for field in fields:
                rows = [r for r in plot_rows if r["branch"] == branch]
                if rows:
                    plt.plot([r["horizon"] for r in rows], [r[field] for r in rows], marker="o", label=f"{branch}:{field}")
        plt.xscale("log", base=2); plt.xlabel("Future horizon"); plt.ylabel(ylabel); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(outdir / "figures" / filename, dpi=180); plt.close()
    curves(["N", "R"], ["GA", "GB", "G"], "01_ga_gb_g.png", "Operator contribution norm")
    curves(["N", "R"], ["GA_GB_ratio"], "02_ga_gb_ratio.png", "GA / GB")
    curves(["N", "R"], ["cosine"], "03_ga_gb_cosine.png", "cos(GA, GB)")
    curves(["N", "R"], ["cancellation"], "04_cancellation.png", "Cancellation fraction")
    curves(["R", "R_AFP", "R_BFP", "R_ABFP"], ["FutureKL"], "05_round1_futurekl.png", "FutureKL")
    curves(["R", "R_AN", "R_BN", "R_ABN"], ["FutureKL"], "06_rotated_transplants.png", "FutureKL")
    curves(["N", "N_AR", "N_BR", "N_ABR"], ["FutureKL"], "07_native_transplants.png", "FutureKL")
    curves(["R", "R_ABFP", "R_ABFP_QFP"], ["FutureKL"], "08_query_residual.png", "FutureKL")
    if effects and all(k in effects for k in ("A_DIFFERENTIAL_RESCUE", "B_DIFFERENTIAL_RESCUE", "AB_DIFFERENTIAL_RESCUE")):
        names = ["A", "B", "A+B"]
        selected = [effects["A_DIFFERENTIAL_RESCUE"], effects["B_DIFFERENTIAL_RESCUE"], effects["AB_DIFFERENTIAL_RESCUE"]]
        medians = [e["paired_median"] for e in selected]
        plt.figure(figsize=(6.4, 4.2)); plt.bar(names, medians, color=["#4c78a8", "#f58518", "#54a24b"])
        for idx, e in enumerate(selected):
            plt.vlines(idx, e["bootstrap_ci_low"], e["bootstrap_ci_high"], color="black")
            plt.hlines([e["bootstrap_ci_low"], e["bootstrap_ci_high"]], idx - 0.08, idx + 0.08, color="black")
        plt.ylabel("Differential FutureKL rescue"); plt.tight_layout()
        plt.savefig(outdir / "figures" / "09_differential_rescue.png", dpi=180); plt.close()


def reset_output_files(outdir):
    for rel in ("unit_level/horizon_results.jsonl", "unit_level/mechanism_metrics.jsonl",
                "unit_level/b_factorial.jsonl", "audit/intervention_provenance.jsonl"):
        path = outdir / rel
        if path.exists():
            path.unlink()


def setup_output(outdir):
    for name in ("audit", "unit_level", "aggregate", "figures", "plot_data", "tests"):
        (outdir / name).mkdir(parents=True, exist_ok=True)


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite:
        reset_output_files(outdir)
    units, unit_audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    if args.skip_units:
        units = units[int(args.skip_units):]
    config_obj = json.loads((BASE.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = BASE.P().kda_layers_from_config(config_obj)
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[:args.target_layer_limit]
    plans = plans_for_phase(args.phase)
    source_result = REPO / "results" / "kda_rotation_decay_drift_causal_closure_v1"
    source_summary = BASE.load_json(source_result / "summary.json", {})
    BASE.save_json(outdir / "config.json", {
        "task": TASK, "phase": args.phase, "scope": args.scope, "horizons": HORIZONS,
        "canonical_manifest": unit_audit, "canonical_unit_ids": [u["unit_id"] for u in units],
        "model_path": str(BASE.MODEL_PATH), "quantizer": "INT8_R128", "rotation": "value-side RHT seed 0",
        "branches": [p.name for p in plans], "bootstrap_seed": BASE.BOOTSTRAP_SEED,
        "intervention_boundary": "operator-level A@S+B at recurrent backend call",
        "source_artifact": str(source_result), "source_task": source_summary.get("task"),
        "source_formal_status": source_summary.get("FORMAL_STATUS"),
    })
    BASE.save_json(outdir / "provenance.json", BASE.environment_info())
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    device = next(model.parameters()).device
    probe = ABProbe(rotation.to(device)); kda_layers = probe.install(model)
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[:args.target_layer_limit]
    horizon_rows, metric_rows, factorial_rows, failures, stage0 = [], [], [], [], None
    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] G/A/B {args.phase} unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                probe.provenance = []
                result = run_unit(model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens, rotation.cpu(), args.horizon, plans)
                if stage0 is None:
                    stage0_records = {plan.name: dict(probe.records[plan.name]) for plan in plans}
                    stage0 = stage0_from_records(probe, stage0_records, rotation.to(device), device)
                    BASE.save_json(outdir / "audit" / "stage0.json", stage0)
                    if stage0["STAGE0"] != "PASS":
                        raise RuntimeError("Stage 0 failed; stopped before interpretation")
                horizon_rows.extend(result["horizon_rows"]); metric_rows.extend(result["metric_rows"])
                factorial_rows.extend(result["factorial_rows"])
                for row in result["horizon_rows"]: BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
                for row in result["metric_rows"]: BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
                for row in result["factorial_rows"]: BASE.append_jsonl(outdir / "unit_level" / "b_factorial.jsonl", row)
                for row in probe.provenance:
                    item = dict(row); item["unit_id"] = str(unit["unit_id"])
                    BASE.append_jsonl(outdir / "audit" / "intervention_provenance.jsonl", item)
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=30), "time": BASE.now()}
                failures.append(failure); BASE.save_json(outdir / "failures.json", failures)
                print(f"[{BASE.now()}] FAILED {unit.get('unit_id')}: {exc!r}", flush=True)
                if stage0 is not None and stage0.get("STAGE0") != "PASS": break
                if not args.keep_going: raise
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        probe.close(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    n = len({r["unit_id"] for r in horizon_rows})
    summary = {"task": TASK, "phase": args.phase, "STAGE0": "FAIL" if stage0 is None else stage0["STAGE0"],
               "n_units_completed": n, "expected_units": len(units), "failures": failures}
    if stage0: summary.update({k: v for k, v in stage0.items() if k.startswith("KDA_")})
    BASE.save_json(outdir / "summary.json", summary)
    BASE.save_json(outdir / "manifest.json", {"task": TASK, "created_at": BASE.now(), "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite: reset_output_files(outdir)
    inputs = [Path(p) for p in args.merge_run_dirs]
    horizon_rows, metric_rows, factorial_rows, provenance, failures = [], [], [], [], []
    for source in inputs:
        horizon_rows += list(BASE.iter_jsonl(source / "unit_level" / "horizon_results.jsonl") or [])
        metric_rows += list(BASE.iter_jsonl(source / "unit_level" / "mechanism_metrics.jsonl") or [])
        factorial_rows += list(BASE.iter_jsonl(source / "unit_level" / "b_factorial.jsonl") or [])
        provenance += list(BASE.iter_jsonl(source / "audit" / "intervention_provenance.jsonl") or [])
        failures += BASE.load_json(source / "summary.json", {}).get("failures", [])
    horizon_rows = dedupe(horizon_rows, ("unit_id", "horizon", "branch"))
    metric_rows = dedupe(metric_rows, ("unit_id", "horizon", "branch", "layer"))
    factorial_rows = dedupe(factorial_rows, ("unit_id", "horizon", "layer", "operator"))
    provenance = dedupe(provenance, ("unit_id", "horizon", "plan", "layer"))
    for row in horizon_rows: BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
    for row in metric_rows: BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
    for row in factorial_rows: BASE.append_jsonl(outdir / "unit_level" / "b_factorial.jsonl", row)
    for row in provenance: BASE.append_jsonl(outdir / "audit" / "intervention_provenance.jsonl", row)
    audits = [BASE.load_json(p / "audit" / "stage0.json", {}) for p in inputs]
    if not audits or any(a.get("STAGE0") != "PASS" for a in audits):
        raise RuntimeError("cannot merge: Stage 0 failed or missing")
    audit = dict(audits[0]); audit["merged_run_audits"] = audits
    BASE.save_json(outdir / "audit" / "stage0.json", audit)
    unit_rows = aggregate_unit_rows(horizon_rows, metric_rows)
    labels, effects = classify(unit_rows, metric_rows, factorial_rows)
    table = main_table(unit_rows, effects)
    BASE.write_rows(outdir / "aggregate" / "unit_metrics.csv", unit_rows)
    BASE.write_rows(outdir / "aggregate" / "main_table.csv", table)
    BASE.write_rows(outdir / "aggregate" / "round1_table.csv", [r for r in table if r["Branch"] in {
        "FP", "N", "R", "N_AFP", "N_BFP", "N_ABFP", "R_AFP", "R_BFP", "R_ABFP",
    }])
    BASE.write_rows(outdir / "aggregate" / "transplant_table.csv", [r for r in table if r["Branch"] in {
        "N", "R", "R_AN", "R_BN", "R_ABN", "N_AR", "N_BR", "N_ABR",
    }])
    BASE.write_rows(outdir / "aggregate" / "b_internal_table.csv", [r for r in table if r["Branch"] in {
        "N", "R", "R_B_betaN", "R_B_kN", "R_B_vN", "N_B_betaR", "N_B_kR", "N_B_vR",
    }])
    BASE.save_json(outdir / "aggregate" / "paired_effects.json", effects)
    BASE.save_json(outdir / "aggregate" / "b_factorial_effects.json", effects.get("B_FACTORIAL_EFFECTS", {}))
    if factorial_rows:
        BASE.write_rows(outdir / "aggregate" / "b_factorial_rows.csv", factorial_rows)
    differential_plot_rows = []
    for key, label in (("A_DIFFERENTIAL_RESCUE", "A"), ("B_DIFFERENTIAL_RESCUE", "B"), ("AB_DIFFERENTIAL_RESCUE", "A+B")):
        if key in effects:
            e = effects[key]
            differential_plot_rows.append({"operator": label, "paired_median": e["paired_median"],
                                           "CI_low": e["bootstrap_ci_low"], "CI_high": e["bootstrap_ci_high"],
                                           "wins": e["wins"], "n": e["n"]})
    BASE.write_rows(outdir / "plot_data" / "differential_rescue.csv", differential_plot_rows)
    plot_rows = write_plot_data(outdir, horizon_rows, metric_rows); write_plots(outdir, plot_rows, effects)
    branches = sorted({r["branch"] for r in horizon_rows})
    n = len({r["unit_id"] for r in horizon_rows if r["branch"] == "R"})
    all_round1 = all(x in branches for x in ("N_AFP", "N_BFP", "N_ABFP", "R_AFP", "R_BFP", "R_ABFP"))
    complete = n == EXPECTED_UNITS and not failures and all_round1
    final = "ROUND1_PENDING"
    closure = "OPEN"; method_ready = "NO"; next_action = "COMPLETE_ROUND1"
    if all_round1:
        mediation = labels.get("KDA_G_TOTAL_CAUSAL_MEDIATION")
        q_label = labels.get("KDA_POST_G_QUERY_RESIDUAL_RESCUE")
        if mediation == "WEAK" and q_label is None:
            final, next_action = "KDA_STATE_TRANSITION_G_NOT_PRIMARY", "RUN_R_ABFP_QFP_THEN_STOP"
        elif mediation == "WEAK" and q_label in {"SUPPORTED", "PARTIAL"}:
            final, closure, next_action = "KDA_STATE_TRANSITION_PLUS_QUERY_CLOSED_LOOP_FAILURE", "PARTIALLY_CLOSED", "REPORT_LIMITS_AND_DO_NOT_START_METHOD_DESIGN"
        elif mediation == "WEAK":
            final, closure, next_action = "KDA_STATE_TRANSITION_G_NOT_PRIMARY", "PARTIALLY_CLOSED", "REPORT_AND_STOP_KERNEL_LOCAL_DECOMPOSITION"
        elif labels.get("KDA_B_NATIVE_TRANSPLANT_RESCUE") == "SUPPORTED" and labels.get("KDA_B_ROTATED_TRANSPLANT_HARM") == "SUPPORTED" and labels.get("KDA_A_FP_DIFFERENTIAL_RESCUE") != "SUPPORTED":
            final = (
                "KDA_CLOSED_LOOP_G_FRESH_UPDATE_B_DOMINANT"
                if labels.get("KDA_G_TOTAL_CAUSAL_MEDIATION") == "STRONG"
                else "KDA_G_FRESH_UPDATE_B_DOMINANT_PARTIAL_MEDIATOR"
            )
            closure = "PARTIALLY_CLOSED"
            next_action = "REPORT_LIMITS_AND_DO_NOT_START_METHOD_DESIGN" if labels.get("B_INTERNAL_DECOMPOSITION_RUN") == "YES" else "EVALUATE_B_INTERNAL_GATE"
        elif "R_AN" in branches:
            final, closure, next_action = "KDA_CLOSED_LOOP_G_AB_MULTI_OPERATOR", "PARTIALLY_CLOSED", "REPORT_AB_OPERATOR_BOUNDARY"
        else:
            final, next_action = "ROUND1_GATE_REACHED", "RUN_CONDITIONALLY_APPROVED_ROUND2"
    pytest_text = (outdir / "tests" / "pytest.txt").read_text(encoding="utf-8") if (outdir / "tests" / "pytest.txt").exists() else "not recorded"
    summary = {
        "task": TASK, "FORMAL_STATUS": "COMPLETE" if complete and next_action not in {"COMPLETE_ROUND1", "RUN_CONDITIONALLY_APPROVED_ROUND2", "RUN_R_ABFP_QFP_THEN_STOP"} else "INCOMPLETE",
        "STAGE0": "PASS", "N_FORMAL_UNITS": n, "PYTEST": pytest_text.strip().splitlines()[-1] if pytest_text.strip() else "not recorded",
        "FAILURES": failures, **{k: v for k, v in audit.items() if k.startswith("KDA_")}, **labels,
        "branches": branches, "FINAL_CLASSIFICATION": final, "KDA_MECHANISM_CLOSURE_STATUS": closure,
        "METHOD_DESIGN_READY": method_ready, "NEXT_RECOMMENDED_ACTION": next_action,
        "key_effects": effects, "merged_from": [str(p) for p in inputs],
    }
    r_units = [r for r in unit_rows if r["branch"] == "R"]
    summary.update({
        "GA_ROTATION_EFFECT": labels.get("KDA_GA_ROTATION_EFFECT"),
        "GB_ROTATION_EFFECT": labels.get("KDA_GB_ROTATION_EFFECT"),
        "GA_GB_RATIO": BASE.median([r["GA_GB_ratio_AUC"] for r in r_units]),
        "GA_GB_GEOMETRY": labels.get("KDA_G_AB_GEOMETRY"),
        "A_DIFFERENTIAL_RESCUE": effects.get("A_DIFFERENTIAL_RESCUE"),
        "B_DIFFERENTIAL_RESCUE": effects.get("B_DIFFERENTIAL_RESCUE"),
        "AB_DIFFERENTIAL_RESCUE": effects.get("AB_DIFFERENTIAL_RESCUE"),
        "R_ABFP_G_ZERO_IDENTITY": labels.get("KDA_R_ABFP_G_ZERO_IDENTITY"),
        "G_TOTAL_CAUSAL_MEDIATION": labels.get("KDA_G_TOTAL_CAUSAL_MEDIATION"),
        "A_NATIVE_TRANSPLANT_RESCUE": labels.get("KDA_A_NATIVE_TRANSPLANT_RESCUE"),
        "B_NATIVE_TRANSPLANT_RESCUE": labels.get("KDA_B_NATIVE_TRANSPLANT_RESCUE"),
        "A_ROTATED_TRANSPLANT_HARM": labels.get("KDA_A_ROTATED_TRANSPLANT_HARM"),
        "B_ROTATED_TRANSPLANT_HARM": labels.get("KDA_B_ROTATED_TRANSPLANT_HARM"),
        "POST_G_QUERY_RESIDUAL_RESCUE": labels.get("KDA_POST_G_QUERY_RESIDUAL_RESCUE"),
        "B_PRIMARY_FACTOR": labels.get("KDA_B_PRIMARY_FACTOR"),
    })
    BASE.save_json(outdir / "summary.json", summary)
    lines = [f"# {TASK}", "", "## Formal classifications", ""]
    console_keys = {
        "GA_ROTATION_EFFECT", "GB_ROTATION_EFFECT", "GA_GB_RATIO", "GA_GB_GEOMETRY",
        "A_DIFFERENTIAL_RESCUE", "B_DIFFERENTIAL_RESCUE", "AB_DIFFERENTIAL_RESCUE",
        "R_ABFP_G_ZERO_IDENTITY", "G_TOTAL_CAUSAL_MEDIATION",
        "A_NATIVE_TRANSPLANT_RESCUE", "B_NATIVE_TRANSPLANT_RESCUE",
        "A_ROTATED_TRANSPLANT_HARM", "B_ROTATED_TRANSPLANT_HARM",
        "POST_G_QUERY_RESIDUAL_RESCUE", "B_INTERNAL_DECOMPOSITION_RUN", "B_PRIMARY_FACTOR",
    }
    for key, value in summary.items():
        if key in {"task", "FORMAL_STATUS", "STAGE0", "N_FORMAL_UNITS", "PYTEST", "FAILURES", "FINAL_CLASSIFICATION", "METHOD_DESIGN_READY", "NEXT_RECOMMENDED_ACTION"} or key.startswith("KDA_") or key == "ROUND1_DECISION" or key in console_keys:
            lines.append(f"{key.upper() if key == 'task' else key} = {json.dumps(value, sort_keys=True)}")
    lines += ["", "## Main table", "", "```json", json.dumps(table, indent=2, sort_keys=True), "```",
              "", "## Paired effects", "", "```json", json.dumps(effects, indent=2, sort_keys=True), "```"]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "README.md").write_text(f"# {TASK}\n\nSee `report.md`, `summary.json`, `aggregate/`, `unit_level/`, `audit/`, `plot_data/`, and `figures/`.\n", encoding="utf-8")
    first_config = BASE.load_json(inputs[0] / "config.json", {})
    first_config["merged_from"] = [str(p) for p in inputs]
    first_config["branches"] = branches
    BASE.save_json(outdir / "config.json", first_config)
    BASE.save_json(outdir / "provenance.json", {"merge_environment": BASE.environment_info(), "source_provenance": [BASE.load_json(p / "provenance.json", {}) for p in inputs]})
    BASE.save_json(outdir / "manifest.json", {"task": TASK, "created_at": BASE.now(), "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    p.add_argument("--phase", choices=("stage0", "round1", "round2", "query", "round2_query", "b_internal"), default="round1")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--target-layer-limit", type=int, default=None)
    p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--unit-shard-index", type=int, default=None)
    p.add_argument("--unit-shard-count", type=int, default=None)
    p.add_argument("--skip-units", type=int, default=0)
    p.add_argument("--merge-run-dirs", nargs="*", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = merge_runs(args) if args.merge_run_dirs else run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.scope == "formal" and args.max_units is None and not args.merge_run_dirs:
        if summary.get("STAGE0") != "PASS" or summary.get("n_units_completed") != summary.get("expected_units") or summary.get("failures"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
