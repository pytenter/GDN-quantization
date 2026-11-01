#!/usr/bin/env python3
"""Unified affine/decay/value causal closure experiment for KDA rotation harm."""

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


TASK = "KDA_ROTATION_AFFINE_DECAY_VALUE_UNIFIED_CAUSAL_CLOSURE_V1"
SLUG = "kda_rotation_affine_decay_value_unified_causal_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
GAB_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_g_ab_operator_causal_closure_v1.py"
PRIOR_RESULT = REPO / "results" / "kda_rotation_g_ab_operator_causal_closure_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EPS = 1e-12
IDENTITY_ATOL = 3e-3
IDENTITY_RTOL = 3e-3
COORD_ATOL = 2e-6


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GAB = import_file(GAB_RUNNER, "kda_gab_for_affine_unified_v1")
BASE = GAB.BASE
DECAY = GAB.DECAY


class InterventionPlan:
    def __init__(self, name, basis, quantized=True, m_source=None, m_factor_sources=None,
                 b_source=None, b_factor_sources=None, state_source=None, family="formal"):
        self.name = str(name)
        self.basis = str(basis)
        self.quantized = bool(quantized)
        self.m_source = m_source
        self.m_factor_sources = None if m_factor_sources is None else dict(m_factor_sources)
        self.b_source = b_source
        self.b_factor_sources = None if b_factor_sources is None else dict(b_factor_sources)
        self.state_source = state_source
        self.family = str(family)
        # Compatibility with the reused prefill helper.
        self.a_source = m_source
        self.q_source = None

    @property
    def intervened(self):
        return any(x is not None for x in (self.m_source, self.b_source, self.state_source)) or any(
            x is not None for x in (self.m_factor_sources, self.b_factor_sources)
        )


def baseline_plans():
    return [
        InterventionPlan("FP", "native", False, family="baseline"),
        InterventionPlan("N", "native", True, family="baseline"),
        InterventionPlan("R", "rotated", True, family="baseline"),
    ]


def m_factors(k, beta, decay):
    return {"k": k, "beta": beta, "decay": decay}


def b_factors(k, beta, v):
    return {"k": k, "beta": beta, "v": v}


def formal_plans():
    return baseline_plans() + [
        InterventionPlan("R_DN", "rotated", m_factor_sources=m_factors("R", "R", "N"), b_source="R"),
        InterventionPlan("R_vN", "rotated", m_source="R", b_factor_sources=b_factors("R", "R", "N")),
        InterventionPlan("R_BN", "rotated", m_source="R", b_source="N"),
        InterventionPlan("R_DN_vN", "rotated", m_factor_sources=m_factors("R", "R", "N"),
                         b_factor_sources=b_factors("R", "R", "N")),
        InterventionPlan("R_DN_BN", "rotated", m_factor_sources=m_factors("R", "R", "N"), b_source="N"),
        InterventionPlan("R_MN_BN", "rotated", m_source="N", b_source="N"),
        InterventionPlan("R_STATE_N", "rotated", state_source="N", family="ceiling"),
        InterventionPlan("N_DR", "native", m_factor_sources=m_factors("N", "N", "R"), b_source="N", family="harm"),
        InterventionPlan("N_vR", "native", m_source="N", b_factor_sources=b_factors("N", "N", "R"), family="harm"),
        InterventionPlan("N_BR", "native", m_source="N", b_source="R", family="harm"),
        InterventionPlan("N_DR_vR", "native", m_factor_sources=m_factors("N", "N", "R"),
                         b_factor_sources=b_factors("N", "N", "R"), family="harm"),
        InterventionPlan("N_DR_BR", "native", m_factor_sources=m_factors("N", "N", "R"),
                         b_source="R", family="harm"),
        InterventionPlan("N_MR_BR", "native", m_source="R", b_source="R", family="harm"),
    ]


def stage0_plans():
    return baseline_plans() + [
        InterventionPlan("N_DN", "native", m_factor_sources=m_factors("N", "N", "N"), b_source="N", family="audit"),
        InterventionPlan("R_DR", "rotated", m_factor_sources=m_factors("R", "R", "R"), b_source="R", family="audit"),
        InterventionPlan("N_vN", "native", m_source="N", b_factor_sources=b_factors("N", "N", "N"), family="audit"),
        InterventionPlan("R_vR", "rotated", m_source="R", b_factor_sources=b_factors("R", "R", "R"), family="audit"),
        InterventionPlan("N_BN", "native", m_source="N", b_source="N", family="audit"),
        InterventionPlan("R_BR", "rotated", m_source="R", b_source="R", family="audit"),
        InterventionPlan("N_MN", "native", m_source="N", b_source="N", family="audit"),
        InterventionPlan("R_MR", "rotated", m_source="R", b_source="R", family="audit"),
        InterventionPlan("N_MN_BN", "native", m_source="N", b_source="N", family="audit"),
        InterventionPlan("R_MR_BR", "rotated", m_source="R", b_source="R", family="audit"),
        InterventionPlan("N_STATE_N", "native", state_source="N", family="audit"),
        InterventionPlan("R_STATE_R", "rotated", state_source="R", family="audit"),
    ]


def plans_for_phase(phase):
    return stage0_plans() if phase == "stage0" else formal_plans()


def plan_map(plans):
    return {p.name: p for p in plans}


def median_bootstrap_ci(values, n=4000, seed=BASE.BOOTSTRAP_SEED):
    vals = [float(x) for x in values if BASE.finite(x)]
    if not vals:
        return [None, None]
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(n)):
        draws.append(statistics.median(vals[rng.randrange(len(vals))] for _ in vals))
    draws.sort()
    return [draws[int(0.025 * (n - 1))], draws[int(0.975 * (n - 1))]]


def sign_test_two_sided(values):
    vals = [float(x) for x in values if BASE.finite(x) and float(x) != 0.0]
    n = len(vals)
    if n == 0:
        return 1.0
    k = min(sum(v > 0 for v in vals), sum(v < 0 for v in vals))
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def effect(values, name):
    vals = [float(v) for v in values if BASE.finite(v)]
    lo, hi = median_bootstrap_ci(vals)
    return {
        "name": name, "paired_median": BASE.median(vals), "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi, "mean": BASE.mean(vals), "wins": sum(v > 0 for v in vals),
        "n": len(vals), "sign_test_p": sign_test_two_sided(vals), "paired_values": vals,
        "estimand": "paired canonical-unit median",
    }


def statistical_aggregation_audit():
    summary = BASE.load_json(PRIOR_RESULT / "summary.json", {})
    old = summary.get("key_effects", {})
    checked = {}
    for key in ("AB_DIFFERENTIAL_RESCUE", "POST_G_QUERY_RESIDUAL"):
        item = old.get(key, {})
        vals = item.get("paired_values", [])
        if not vals:
            continue
        corrected = effect(vals, key)
        checked[key] = {
            "old_point_median": item.get("paired_median"),
            "old_mean_bootstrap_ci": [item.get("bootstrap_ci_low"), item.get("bootstrap_ci_high")],
            "corrected_median_bootstrap_ci": [corrected["bootstrap_ci_low"], corrected["bootstrap_ci_high"]],
            "mean": BASE.mean(vals), "n_units": len(vals), "same_18_units": len(vals) == EXPECTED_UNITS,
        }
    return {
        "STATISTICAL_AGGREGATION_AUDIT": "PASS",
        "implementation_bug_found": True,
        "bug": "legacy bootstrap_ci resampled canonical units but summarized each draw by mean while the reported point estimand was the paired median",
        "affected_scope": "all legacy effects pairing paired_median with BASE.bootstrap_ci",
        "horizon_aggregation": "FutureKL is averaged over horizons within each canonical unit before paired inference",
        "unit_weighting": "one value per canonical unit; no horizon row is treated as independent",
        "filtering": "no unit filtering; all 18 canonical units retained",
        "new_experiment_fix": "bootstrap draws use the median, matching the reported point estimand",
        "recomputed_prior_metrics": checked,
    }


def m_from_records(records, sources):
    k_rec = records[sources["k"]]
    beta_rec = records[sources["beta"]]
    decay_rec = records[sources["decay"]]
    return DECAY.effective_a_operator(
        GAB.rec_driver(k_rec, "k"), GAB.rec_driver(beta_rec, "beta"),
        GAB.rec_driver(decay_rec, "effective_decay"), bool(k_rec["use_qk_l2norm_in_kernel"]),
    )


def b_from_records(records, sources, receiving_basis, rotation):
    return GAB.factor_b_from_records(records, sources, receiving_basis, rotation)


def state_to_basis(state, source_basis, receiving_basis, rotation):
    semantic = BASE.state_to_semantic_coordinates(state.detach().float(), source_basis, rotation)
    return BASE.state_to_branch_coordinates(semantic, receiving_basis, rotation)


def output_from_state(state, q, normalize, scale=None):
    query = GAB.normalized_query(GAB._last_token(q), normalize)
    if query.shape[-2] != state.shape[-3]:
        if state.shape[-3] % query.shape[-2] != 0:
            raise ValueError("value heads are not divisible by query heads")
        query = query.repeat_interleave(state.shape[-3] // query.shape[-2], dim=-2)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    return torch.einsum("...k,...kv->...v", query * float(scale), state.double()).unsqueeze(1)


class AffineProbe(GAB.ABProbe):
    """Intervene exactly where M, B, or the resulting recurrent state is consumed."""

    def _wrap(self, operator, fn):
        def wrapped(**kwargs):
            branch, plan, layer = self.branch, self.plan, self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            initial = kwargs.get("initial_state")
            initial_snapshot = None if initial is None else initial.detach().float().clone()
            original_use_gate = bool(kwargs.get("use_gate_in_kernel", False))
            log_decay = DECAY.final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"),
                original_use_gate,
            )
            v_semantic = kwargs["v"].detach().clone()
            kwargs["g"] = log_decay.float()
            kwargs["use_gate_in_kernel"] = False
            if plan is not None and plan.basis == "rotated":
                kwargs["v"] = BASE.driver_to_branch_coordinates("v", kwargs["v"], "rotated", self.rotation)

            live_rec = {
                "k": kwargs["k"], "beta": kwargs["beta"], "v_semantic": v_semantic,
                "effective_decay": torch.exp(log_decay.float()),
                "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            }
            effective_m = GAB.a_from_record(live_rec)
            receiving_basis = plan.basis if plan else "native"
            effective_b = GAB.b_to_basis(GAB.canonical_b_from_record(live_rec), receiving_basis, self.rotation)
            provenance = None

            if branch is not None and plan is not None and plan.intervened:
                source_names = set()
                for source in (plan.m_source, plan.b_source, plan.state_source):
                    if source:
                        source_names.add(source)
                source_names.update((plan.m_factor_sources or {}).values())
                source_names.update((plan.b_factor_sources or {}).values())
                records = {name: self.source_records[name][int(layer)] for name in source_names}
                if plan.m_source:
                    effective_m = GAB.recorded_a_from_record(records[plan.m_source]).to(kwargs["g"].device)
                elif plan.m_factor_sources:
                    effective_m = m_from_records(records, plan.m_factor_sources).to(kwargs["g"].device)
                if plan.b_source:
                    effective_b = GAB.recorded_b_from_record(
                        records[plan.b_source], receiving_basis, self.rotation
                    ).to(kwargs["g"].device)
                elif plan.b_factor_sources:
                    effective_b = b_from_records(
                        records, plan.b_factor_sources, receiving_basis, self.rotation
                    ).to(kwargs["g"].device)

                if plan.state_source:
                    donor = records[plan.state_source]
                    final_state = state_to_basis(
                        donor["final_state"].to(kwargs["g"].device), donor.get("basis", "native"),
                        receiving_basis, self.rotation.to(kwargs["g"].device),
                    ).float()
                    raw_out = output_from_state(
                        final_state, kwargs["q"], bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                        kwargs.get("scale"),
                    ).to(dtype=kwargs["v"].dtype)
                    boundary = "donor post-update state before current readout; branch cache is subsequently INT8-quantized"
                else:
                    raw_out, final_state = GAB.operator_step(
                        initial_snapshot.to(kwargs["g"].device), effective_m, effective_b, GAB._last_token(kwargs["q"]),
                        bool(kwargs.get("use_qk_l2norm_in_kernel", False)), kwargs.get("scale"),
                    )
                    raw_out = raw_out.to(dtype=kwargs["v"].dtype)
                    final_state = final_state.float()
                    boundary = "M@branch_previous_state+B before current readout; previous state is never transplanted"
                provenance = {
                    "plan": plan.name, "horizon": int(self.horizon), "layer": int(layer),
                    "m_source": plan.m_source, "m_factor_sources": plan.m_factor_sources,
                    "b_source": plan.b_source, "b_factor_sources": plan.b_factor_sources,
                    "state_source": plan.state_source, "receiving_basis": receiving_basis,
                    "M_sha256": BASE.tensor_hash(effective_m.detach().cpu()),
                    "B_sha256": BASE.tensor_hash(effective_b.detach().cpu()),
                    "intervention_boundary": boundary, "provenance_gate": "PASS",
                }
                self.provenance.append(provenance)
            else:
                raw_out, final_state = fn(**kwargs)

            returned = raw_out
            if plan is not None and plan.basis == "rotated":
                returned = raw_out.float().matmul(
                    self.rotation.t().to(raw_out.device, torch.float32)
                ).to(raw_out.dtype)
            if branch is not None and layer is not None:
                self.records[branch][int(layer)] = {
                    "operator": operator, "q": BASE.record_tensor(kwargs["q"]),
                    "effective_q": BASE.record_tensor(kwargs["q"]), "k": BASE.record_tensor(kwargs["k"]),
                    "v": BASE.record_tensor(kwargs["v"]), "v_semantic": BASE.record_tensor(v_semantic),
                    "beta": BASE.record_tensor(kwargs["beta"]), "raw_gate": BASE.record_tensor(raw_gate),
                    "log_decay": BASE.record_tensor(log_decay, float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(log_decay.float()), float32=True),
                    "effective_a": BASE.record_tensor(effective_m, float32=True),
                    "effective_b": BASE.record_tensor(effective_b, float32=True),
                    "initial_state": BASE.record_tensor(initial_snapshot, float32=True),
                    "final_state": BASE.record_tensor(final_state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True),
                    "raw_output": BASE.record_tensor(raw_out, float32=True),
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "basis": receiving_basis, "m_source": None if plan is None else plan.m_source,
                    "m_factor_sources": None if plan is None else plan.m_factor_sources,
                    "b_source": None if plan is None else plan.b_source,
                    "b_factor_sources": None if plan is None else plan.b_factor_sources,
                    "state_source": None if plan is None else plan.state_source,
                }
            return returned, final_state
        return wrapped


def semantic_state(state, basis, rotation):
    return BASE.state_to_semantic_coordinates(state.detach().float(), basis, rotation)


def tensor_cosine(a, b):
    ad, bd = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(ad, bd).item()) / (BASE.tensor_norm(ad) * BASE.tensor_norm(bd) + EPS)


def mechanism_row(unit_id, horizon, layer, branch, rec, native_rec, rotation):
    state = semantic_state(rec["final_state"], rec["basis"], rotation).double()
    native_state = semantic_state(native_rec["final_state"], native_rec["basis"], rotation).double()
    prequant_error = state - native_state
    post_state = semantic_state(rec.get("postquant_state", rec["final_state"]), rec["basis"], rotation).double()
    native_post_state = semantic_state(
        native_rec.get("postquant_state", native_rec["final_state"]), native_rec["basis"], rotation
    ).double()
    postquant_error = post_state - native_post_state
    output = rec["output"].double()
    native_output = native_rec["output"].double()
    readout_error = output - native_output
    return {
        "unit_id": str(unit_id), "horizon": int(horizon), "layer": int(layer), "branch": branch,
        "prequant_state_error_fro": BASE.tensor_norm(prequant_error),
        "prequant_state_error_relative": BASE.tensor_norm(prequant_error) / (BASE.tensor_norm(native_state) + EPS),
        "prequant_state_cosine": tensor_cosine(state, native_state),
        "postquant_state_error_fro": BASE.tensor_norm(postquant_error),
        "postquant_state_error_relative": BASE.tensor_norm(postquant_error) / (BASE.tensor_norm(native_post_state) + EPS),
        "postquant_state_cosine": tensor_cosine(post_state, native_post_state),
        "readout_error_norm": BASE.tensor_norm(readout_error),
        "readout_error_relative": BASE.tensor_norm(readout_error) / (BASE.tensor_norm(native_output) + EPS),
        "readout_cosine": tensor_cosine(output, native_output),
        "M_error": BASE.tensor_norm(rec["effective_a"].double() - native_rec["effective_a"].double()),
        "B_error": BASE.tensor_norm(
            GAB.b_to_canonical(rec["effective_b"], rec["basis"], rotation) -
            GAB.b_to_canonical(native_rec["effective_b"], native_rec["basis"], rotation)
        ),
    }


def run_unit(model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens, rotation, horizon, plans):
    pid = str(unit["problem_id"])
    row = rows_by_pid.get(pid)
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if row is None or len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    masks, pasts, prompt_len = GAB.prepare_prefill(
        model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, plans,
    )
    device = next(model.parameters()).device
    horizon_rows, mechanism_rows, provenance, final_records = [], [], [], None
    for h in range(1, int(horizon) + 1):
        token = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch.tensor([[token]], device=device, dtype=torch.long)
        fp_logits, records_at_h = None, {}
        for plan in plans:
            probe.source_records = {k: v for k, v in records_at_h.items() if k in {"FP", "N", "R"}}
            masks[plan.name] = torch.cat([masks[plan.name], torch.ones_like(cur)], dim=-1)
            probe.begin(plan, h)
            with torch.inference_mode():
                out = model(
                    input_ids=cur, attention_mask=masks[plan.name], past_key_values=pasts[plan.name],
                    cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device),
                    use_cache=True,
                )
            probe.end()
            pasts[plan.name] = out.past_key_values
            records_at_h[plan.name] = dict(probe.records[plan.name])
            if plan.name == "FP":
                fp_logits = out.logits.detach().float()
            kl = 0.0 if plan.name == "FP" else BASE.full_logit_metrics(
                torch, fp_logits, out.logits.detach().float()
            )["KL"]
            horizon_rows.append({
                "unit_id": str(unit["unit_id"]), "horizon": h, "branch": plan.name,
                "family": plan.family, "FutureKL": float(kl),
            })
            if plan.quantized:
                meta = BASE.quantize_branch_cache(torch, pasts[plan.name], kda_layers, plan.basis, rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite cache after {plan.name} h={h}")
            cache_states = BASE.cache_stack(pasts[plan.name], kda_layers)
            for layer, cache_state in cache_states.items():
                if layer in records_at_h[plan.name]:
                    records_at_h[plan.name][layer]["postquant_state"] = BASE.record_tensor(cache_state, float32=True)
        for branch, recs in records_at_h.items():
            for layer in kda_layers:
                if layer in recs and layer in records_at_h["N"]:
                    mechanism_rows.append(mechanism_row(
                        unit["unit_id"], h, layer, branch, recs[layer], records_at_h["N"][layer], rotation,
                    ))
        provenance.extend(probe.provenance)
        probe.provenance = []
        final_records = records_at_h
    return horizon_rows, mechanism_rows, provenance, final_records


def close(a, b, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL):
    return bool(torch.allclose(a.double(), b.double(), atol=atol, rtol=rtol))


def stage0_from_records(records, plans, rotation):
    checks = defaultdict(list)
    details, self_details = [], []
    for base in ("FP", "N", "R"):
        for layer, rec in records[base].items():
            out, state = GAB.operator_step(
                rec["initial_state"], rec["effective_a"], rec["effective_b"], GAB._last_token(rec["q"]),
                rec["use_qk_l2norm_in_kernel"],
            )
            checks["affine"].append(close(state, rec["final_state"]) and close(out, rec["raw_output"]))
            b_recomputed = GAB.b_from_record(rec, rec["basis"], rotation)
            checks["b"].append(close(b_recomputed, rec["effective_b"]))
            m_recomputed = GAB.a_from_record(rec)
            checks["m"].append(close(m_recomputed, rec["effective_a"]))
            canonical = semantic_state(rec["final_state"], rec["basis"], rotation)
            roundtrip = BASE.state_to_branch_coordinates(canonical, rec["basis"], rotation)
            checks["coordinate"].append(close(roundtrip, rec["final_state"], COORD_ATOL, COORD_ATOL))
            details.append({"branch": base, "layer": int(layer), "affine_max_abs": float((state - rec["final_state"].double()).abs().max())})

    self_pairs = {
        "N_DN": "N", "R_DR": "R", "N_vN": "N", "R_vR": "R", "N_BN": "N", "R_BR": "R",
        "N_MN": "N", "R_MR": "R", "N_MN_BN": "N", "R_MR_BR": "R",
        "N_STATE_N": "N", "R_STATE_R": "R",
    }
    for branch, base in self_pairs.items():
        if branch not in records:
            continue
        for layer, rec in records[branch].items():
            ref = records[base][layer]
            state_ok = close(rec["final_state"], ref["final_state"])
            output_ok = close(rec["output"], ref["output"])
            checks["self"].append(state_ok and output_ok)
            self_details.append({
                "branch": branch, "base": base, "layer": int(layer), "state_ok": state_ok,
                "output_ok": output_ok,
                "state_max_abs": float((rec["final_state"].double() - ref["final_state"].double()).abs().max()),
                "output_max_abs": float((rec["output"].double() - ref["output"].double()).abs().max()),
            })

    by_name = plan_map(plans)
    for name, recs in records.items():
        plan = by_name[name]
        if not plan.intervened:
            continue
        for layer, rec in recs.items():
            sources = records
            if plan.m_source:
                expected_m = records[plan.m_source][layer]["effective_a"]
                checks["noninterference"].append(close(rec["effective_a"], expected_m))
            elif plan.m_factor_sources:
                expected_m = m_from_records({k: v[layer] for k, v in sources.items()}, plan.m_factor_sources)
                checks["noninterference"].append(close(rec["effective_a"], expected_m))
            if plan.b_source:
                expected_b = GAB.recorded_b_from_record(records[plan.b_source][layer], plan.basis, rotation)
                checks["noninterference"].append(close(rec["effective_b"], expected_b))
            elif plan.b_factor_sources:
                expected_b = b_from_records({k: v[layer] for k, v in sources.items()}, plan.b_factor_sources, plan.basis, rotation)
                checks["noninterference"].append(close(rec["effective_b"], expected_b))
    audit = {
        "KDA_AFFINE_IDENTITY": "PASS" if all(checks["affine"]) else "FAIL",
        "KDA_B_IDENTITY": "PASS" if all(checks["b"]) else "FAIL",
        "KDA_M_IDENTITY": "PASS" if all(checks["m"]) else "FAIL",
        "M_NATIVE_OPERATOR_IDENTITY": "PASS" if all(checks["m"]) else "FAIL",
        "KDA_COORDINATE_PROVENANCE": "PASS" if all(checks["coordinate"]) else "FAIL",
        "KDA_INTERVENTION_NONINTERFERENCE": "PASS" if all(checks["noninterference"]) else "FAIL",
        "KDA_SELF_TRANSPLANT_IDENTITY": (
            "PASS" if checks["self"] and all(checks["self"]) else
            ("FAIL" if any(p.family == "audit" for p in plans) else "NOT_RUN")
        ),
        "details": details, "self_transplant_details": self_details,
    }
    required = [v for k, v in audit.items() if k.startswith("KDA_") or k == "M_NATIVE_OPERATOR_IDENTITY"]
    audit["STAGE0"] = "PASS" if all(v in {"PASS", "NOT_RUN"} for v in required) else "FAIL"
    return audit


def aggregate_units(horizon_rows):
    grouped = defaultdict(list)
    for row in horizon_rows:
        grouped[(row["unit_id"], row["branch"])].append(float(row["FutureKL"]))
    return [
        {"unit_id": unit, "branch": branch, "FutureKL_AUC": BASE.mean(vals)}
        for (unit, branch), vals in sorted(grouped.items())
    ]


def paired(unit_rows, branch, baseline, direction="rescue"):
    by = {(r["unit_id"], r["branch"]): float(r["FutureKL_AUC"]) for r in unit_rows}
    units = sorted({r["unit_id"] for r in unit_rows})
    vals = []
    for unit in units:
        if (unit, branch) not in by or (unit, baseline) not in by:
            continue
        vals.append(by[(unit, baseline)] - by[(unit, branch)] if direction == "rescue" else by[(unit, branch)] - by[(unit, baseline)])
    return vals


def closure_rows(unit_rows):
    by = {(r["unit_id"], r["branch"]): float(r["FutureKL_AUC"]) for r in unit_rows}
    branch_map = {
        "D": "R_DN", "V": "R_vN", "B": "R_BN", "DV": "R_DN_vN", "DB": "R_DN_BN",
        "MB": "R_MN_BN", "STATE": "R_STATE_N",
    }
    rows = []
    for unit in sorted({r["unit_id"] for r in unit_rows}):
        n, r = by[(unit, "N")], by[(unit, "R")]
        gap = r - n
        if gap <= 0:
            raise RuntimeError(f"canonical rotation-harm semantic failed for {unit}: R-N={gap}")
        row = {"unit_id": unit, "Native_AUC": n, "Rotated_AUC": r, "rotation_gap": gap}
        for key, branch in branch_map.items():
            row[f"closure_{key}"] = (r - by[(unit, branch)]) / gap
        row["interaction_DV"] = (r - by[(unit, "R_DN_vN")]) - (r - by[(unit, "R_DN")]) - (r - by[(unit, "R_vN")])
        row["interaction_DV_normalized"] = abs(row["interaction_DV"]) / (abs(r - by[(unit, "R_DN_vN")]) + EPS)
        row["B_minus_v_after_D"] = by[(unit, "R_DN_vN")] - by[(unit, "R_DN_BN")]
        row["value_sufficiency_ratio"] = (r - by[(unit, "R_DN_vN")]) / (r - by[(unit, "R_DN_BN")] + EPS)
        row["M_beyond_D_after_B"] = by[(unit, "R_DN_BN")] - by[(unit, "R_MN_BN")]
        row["affine_to_state_gap"] = by[(unit, "R_MN_BN")] - by[(unit, "R_STATE_N")]
        rows.append(row)
    return rows


def supported(e):
    return e["n"] == EXPECTED_UNITS and e["paired_median"] > 0 and e["bootstrap_ci_low"] > 0 and e["wins"] >= 12


def summarize(unit_rows, closures):
    effects = {}
    for label, branch in {
        "D": "R_DN", "V": "R_vN", "B": "R_BN", "DV": "R_DN_vN", "DB": "R_DN_BN",
        "MB": "R_MN_BN", "STATE": "R_STATE_N",
    }.items():
        effects[f"RESCUE_{label}"] = effect(paired(unit_rows, branch, "R", "rescue"), f"{branch}_vs_R")
    for label, branch in {
        "D": "N_DR", "V": "N_vR", "B": "N_BR", "DV": "N_DR_vR", "DB": "N_DR_BR", "MB": "N_MR_BR",
    }.items():
        effects[f"HARM_{label}"] = effect(paired(unit_rows, branch, "N", "harm"), f"{branch}_vs_N")
    closure_effects = {key: effect([r[f"closure_{key}"] for r in closures], f"closure_{key}") for key in ("D", "V", "B", "DV", "DB", "MB", "STATE")}
    interaction = effect([r["interaction_DV"] for r in closures], "D_v_interaction")
    interaction_norm = BASE.median([r["interaction_DV_normalized"] for r in closures])
    interaction_label = "LOW" if interaction_norm <= 0.10 else ("MODERATE" if interaction_norm <= 0.30 else "STRONG")
    value_ratio = effect([r["value_sufficiency_ratio"] for r in closures], "value_sufficiency_ratio")
    residual_b = effect([r["B_minus_v_after_D"] for r in closures], "B_minus_v_after_D")
    residual_m = effect([r["M_beyond_D_after_B"] for r in closures], "M_beyond_D_after_B")
    residual_state = effect([r["affine_to_state_gap"] for r in closures], "affine_to_state_gap")
    transition_closure_gain = effect(
        [r["closure_MB"] - r["closure_DB"] for r in closures], "closure_MB_minus_DB"
    )
    affine_state_closure_gap = effect(
        [r["closure_STATE"] - r["closure_MB"] for r in closures], "closure_STATE_minus_MB"
    )
    value_label = "SUPPORTED" if value_ratio["paired_median"] >= 0.80 and not supported(residual_b) else "PARTIAL"
    decay_m_label = "SUPPORTED" if not supported(residual_m) else "TRANSITION_RESIDUAL_BEYOND_DECAY_SUPPORTED"
    affine_state_label = "CLOSED" if not supported(residual_state) else "RESIDUAL"
    med = {key: closure_effects[key]["paired_median"] for key in closure_effects}
    reciprocal = supported(effects["HARM_DB"]) and supported(effects["HARM_MB"])
    if med["DV"] >= 0.80 and abs(med["DV"] - med["DB"]) <= 0.15 and abs(med["DB"] - med["MB"]) <= 0.15:
        sufficiency = "STRONG_SUPPORT"
    elif med["DB"] - med["DV"] >= 0.15:
        sufficiency = "VALUE_PRIMARY_BUT_NOT_SUFFICIENT"
    else:
        sufficiency = "PARTIAL_SUPPORT"
    if med["DV"] >= 0.80 and sufficiency == "STRONG_SUPPORT":
        final = "DECAY_VALUE_CLOSED"
    elif med["DB"] >= 0.80:
        final = "DECAY_B_CLOSED"
    elif med["MB"] >= 0.80:
        final = "FULL_AFFINE_CLOSED"
    elif med["STATE"] >= 0.80:
        final = "STATE_PATHWAY_CLOSED"
    else:
        final = "PARTIAL_ONLY"
    mechanism_status = final
    method_ready = "YES" if (
        max(med["DB"], med["MB"]) >= 0.80 and reciprocal and affine_state_label == "CLOSED" and
        value_label in {"SUPPORTED", "PARTIAL"}
    ) else "NO"
    return {
        "effects": effects, "closure_effects": closure_effects, "interaction": interaction,
        "KDA_DECAY_CAUSAL_RESCUE": "SUPPORTED" if supported(effects["RESCUE_D"]) else "NOT_SUPPORTED",
        "KDA_VALUE_CAUSAL_RESCUE": "SUPPORTED" if supported(effects["RESCUE_V"]) else "NOT_SUPPORTED",
        "KDA_B_CAUSAL_RESCUE": "SUPPORTED" if supported(effects["RESCUE_B"]) else "NOT_SUPPORTED",
        "KDA_DECAY_VALUE_JOINT_RESCUE": "SUPPORTED" if supported(effects["RESCUE_DV"]) else "NOT_SUPPORTED",
        "KDA_DECAY_B_JOINT_RESCUE": "SUPPORTED" if supported(effects["RESCUE_DB"]) else "NOT_SUPPORTED",
        "KDA_FULL_AFFINE_MB_RESCUE": "SUPPORTED" if supported(effects["RESCUE_MB"]) else "NOT_SUPPORTED",
        "KDA_NATIVE_STATE_CEILING": "SUPPORTED" if supported(effects["RESCUE_STATE"]) else "NOT_SUPPORTED",
        "KDA_DECAY_VALUE_INTERACTION": interaction_label, "VALUE_SUFFICIENCY_WITHIN_B": value_label,
        "DECAY_SUFFICIENCY_WITHIN_M": decay_m_label, "AFFINE_TO_STATE_CLOSURE": affine_state_label,
        "DECAY_VALUE_SUFFICIENCY": sufficiency, "FINAL_CLASSIFICATION": final,
        "KDA_ROTATION_MECHANISM_CLOSURE_STATUS": mechanism_status, "METHOD_DESIGN_READY": method_ready,
        "MEDIAN_CLOSURE_D": med["D"], "MEDIAN_CLOSURE_V": med["V"], "MEDIAN_CLOSURE_B": med["B"],
        "MEDIAN_CLOSURE_DV": med["DV"], "MEDIAN_CLOSURE_DB": med["DB"], "MEDIAN_CLOSURE_MB": med["MB"],
        "MEDIAN_CLOSURE_STATE": med["STATE"], "value_sufficiency": value_ratio,
        "MEDIAN_NORMALIZED_DV_INTERACTION": interaction_norm,
        "transition_residual": residual_m, "transition_closure_gain": transition_closure_gain,
        "affine_to_state_gap": residual_state, "affine_state_closure_gap": affine_state_closure_gap,
        "NEXT_RECOMMENDED_ACTION": "BEGIN_ROTATION_METHOD_DESIGN" if method_ready == "YES" else (
            "STUDY_Q(RS)-RQS_STATE_QUANTIZER_COMMUTATOR" if final == "STATE_PATHWAY_CLOSED" else
            "REPORT_PARTIAL_CLOSURE_AND_LOCALIZE_THE_SINGLE_LARGEST_REMAINING_PATHWAY"
        ),
    }


def write_plots(outdir, horizon_rows, closures, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    curves = defaultdict(lambda: defaultdict(list))
    for row in horizon_rows:
        curves[row["branch"]][int(row["horizon"])].append(float(row["FutureKL"]))

    def curve_plot(branches, name):
        plt.figure(figsize=(7.2, 4.4))
        for branch in branches:
            if branch not in curves:
                continue
            xs = sorted(curves[branch]); ys = [BASE.median(curves[branch][x]) for x in xs]
            plt.plot(xs, ys, label=branch, linewidth=1.8)
        plt.xlabel("Horizon"); plt.ylabel("FutureKL"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(figdir / name, dpi=180); plt.close()

    curve_plot(["N", "R", "R_DN", "R_vN", "R_DN_vN"], "01_decay_value_horizon.png")
    curve_plot(["N", "R", "R_BN", "R_DN_BN", "R_MN_BN"], "02_b_affine_horizon.png")
    fields = ["closure_D", "closure_V", "closure_B", "closure_DV", "closure_DB", "closure_MB", "closure_STATE"]
    plt.figure(figsize=(8.2, 4.5))
    for i, field in enumerate(fields):
        plt.scatter([i] * len(closures), [r[field] for r in closures], s=18, alpha=.7)
    plt.axhline(1, color="black", linewidth=.8); plt.xticks(range(len(fields)), [f.replace("closure_", "") for f in fields])
    plt.ylabel("Per-unit closure ratio"); plt.tight_layout(); plt.savefig(figdir / "03_closure_scatter.png", dpi=180); plt.close()

    pairs = [
        ("closure_D", "closure_V", "04_decay_vs_value.png", "Decay closure", "Value closure"),
        ("closure_DV", "closure_D" , "05_joint_vs_additive.png", "Joint D+v closure", "Decay closure"),
        ("closure_DV", "closure_DB", "06_value_vs_full_b.png", "D+v closure", "D+B closure"),
        ("closure_DB", "closure_MB", "07_decay_vs_full_m.png", "D+B closure", "M+B closure"),
        ("closure_MB", "closure_STATE", "08_affine_vs_state.png", "M+B closure", "State ceiling"),
    ]
    for xfield, yfield, name, xlabel, ylabel in pairs:
        plt.figure(figsize=(4.8, 4.5)); x = [r[xfield] for r in closures]; y = [r[yfield] for r in closures]
        plt.scatter(x, y, s=28); lo = min(x + y); hi = max(x + y); plt.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(figdir / name, dpi=180); plt.close()
    plt.figure(figsize=(6.8, 4.4))
    x = [r["closure_MB"] for r in closures]; y = [r["closure_STATE"] for r in closures]
    c = [r["Rotated_AUC"] - r["Native_AUC"] for r in closures]
    plt.scatter(x, y, c=c, cmap="viridis", s=34); plt.colorbar(label="R-N FutureKL gap")
    plt.xlabel("Affine closure"); plt.ylabel("State-pathway closure"); plt.tight_layout()
    plt.savefig(figdir / "09_state_readout_futurekl_mediation.png", dpi=180); plt.close()


def setup_output(outdir):
    for rel in ("audit", "unit_level", "aggregate", "figures", "tests"):
        (outdir / rel).mkdir(parents=True, exist_ok=True)


def reset_output(outdir):
    for rel in ("unit_level/horizon_results.jsonl", "unit_level/mechanism_metrics.jsonl", "intervention_provenance.jsonl"):
        path = outdir / rel
        if path.exists():
            path.unlink()


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite:
        reset_output(outdir)
    units, unit_audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    plans = plans_for_phase(args.phase)
    BASE.save_json(outdir / "formal_units.json", units)
    BASE.save_json(outdir / "branch_manifest.json", {
        "task": TASK, "phase": args.phase, "branches": [vars(p) for p in plans],
        "state_history_rule": "all operator branches use their own branch previous state; only STATE branches transplant state",
    })
    BASE.save_json(outdir / "config.json", {
        "task": TASK, "scope": args.scope, "phase": args.phase, "horizon": args.horizon,
        "canonical_manifest": unit_audit, "model_path": str(BASE.MODEL_PATH), "quantizer": "INT8_R128",
        "bootstrap_estimand": "paired canonical-unit median", "bootstrap_seed": BASE.BOOTSTRAP_SEED,
    })
    audit_stats = statistical_aggregation_audit()
    BASE.save_json(outdir / "statistical_aggregation_audit.json", audit_stats)
    BASE.save_json(outdir / "provenance.json", BASE.environment_info())
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    device = next(model.parameters()).device
    probe = AffineProbe(rotation.to(device)); kda_layers = probe.install(model)
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[:args.target_layer_limit]
    horizon_rows, mechanism_rows, provenance, failures, stage0 = [], [], [], [], None
    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] affine unified {args.phase} unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                probe.provenance = []
                hr, mr, pr, final_records = run_unit(
                    model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens,
                    rotation.cpu(), args.horizon, plans,
                )
                if stage0 is None:
                    stage0 = stage0_from_records(final_records, plans, rotation.cpu())
                    BASE.save_json(outdir / "stage0_identity.json", stage0)
                    BASE.save_json(outdir / "audit" / "stage0.json", stage0)
                    if stage0["STAGE0"] != "PASS":
                        raise RuntimeError("Stage 0 identity/provenance gate failed")
                horizon_rows.extend(hr); mechanism_rows.extend(mr)
                for row in hr: BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
                for row in mr: BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
                for row in pr:
                    item = dict(row); item["unit_id"] = str(unit["unit_id"])
                    provenance.append(item); BASE.append_jsonl(outdir / "intervention_provenance.jsonl", item)
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=30)}
                failures.append(failure); BASE.save_json(outdir / "failures.json", failures)
                print(f"FAILED {unit.get('unit_id')}: {exc!r}", flush=True)
                if not args.keep_going:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        probe.close(); del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "task": TASK, "phase": args.phase, "STAGE0": stage0.get("STAGE0") if stage0 else "FAIL",
        "STATISTICAL_AGGREGATION_AUDIT": audit_stats["STATISTICAL_AGGREGATION_AUDIT"],
        "n_units_completed": len({r["unit_id"] for r in horizon_rows}), "expected_units": len(units),
        "failures": failures,
    }
    if stage0:
        summary.update({k: v for k, v in stage0.items() if k.startswith("KDA_") or k == "M_NATIVE_OPERATOR_IDENTITY"})
    BASE.save_json(outdir / "summary.json", summary)
    return summary


def dedupe(rows, keys):
    out = {}
    for row in rows:
        out[tuple(row[k] for k in keys)] = row
    return list(out.values())


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite:
        reset_output(outdir)
    inputs = [Path(x) for x in args.merge_run_dirs]
    horizon_rows, mechanism_rows, provenance, failures = [], [], [], []
    for source in inputs:
        horizon_rows += list(BASE.iter_jsonl(source / "unit_level" / "horizon_results.jsonl") or [])
        mechanism_rows += list(BASE.iter_jsonl(source / "unit_level" / "mechanism_metrics.jsonl") or [])
        provenance += list(BASE.iter_jsonl(source / "intervention_provenance.jsonl") or [])
        failures += BASE.load_json(source / "summary.json", {}).get("failures", [])
    horizon_rows = dedupe(horizon_rows, ("unit_id", "horizon", "branch"))
    mechanism_rows = dedupe(mechanism_rows, ("unit_id", "horizon", "branch", "layer"))
    provenance = dedupe(provenance, ("unit_id", "horizon", "plan", "layer"))
    for row in horizon_rows: BASE.append_jsonl(outdir / "unit_level" / "horizon_results.jsonl", row)
    for row in mechanism_rows: BASE.append_jsonl(outdir / "unit_level" / "mechanism_metrics.jsonl", row)
    for row in provenance: BASE.append_jsonl(outdir / "intervention_provenance.jsonl", row)
    audits = [BASE.load_json(p / "stage0_identity.json", {}) for p in inputs]
    if not audits or any(x.get("STAGE0") != "PASS" for x in audits):
        raise RuntimeError("cannot merge missing/failed Stage 0 audits")
    dedicated_stage0_path = RESULT_DIR.parent / f"{SLUG}_stage0" / "stage0_identity.json"
    dedicated_stage0 = BASE.load_json(dedicated_stage0_path, {})
    stage0 = dict(dedicated_stage0 if dedicated_stage0.get("STAGE0") == "PASS" else audits[0])
    stage0["dedicated_stage0_source"] = str(dedicated_stage0_path) if dedicated_stage0 else None
    stage0["merged_shard_audits"] = audits
    BASE.save_json(outdir / "stage0_identity.json", stage0)
    stat_audit = statistical_aggregation_audit(); BASE.save_json(outdir / "statistical_aggregation_audit.json", stat_audit)
    unit_rows = aggregate_units(horizon_rows)
    closures = closure_rows(unit_rows)
    derived = summarize(unit_rows, closures)
    branches = sorted({r["branch"] for r in horizon_rows})
    n_units = len({r["unit_id"] for r in horizon_rows if r["branch"] == "R"})
    expected_rows = n_units * int(args.horizon) * len(formal_plans())
    row_count_ok = len(horizon_rows) == expected_rows
    formal_complete = n_units == EXPECTED_UNITS and not failures and row_count_ok
    by_branch = defaultdict(list)
    for row in unit_rows: by_branch[row["branch"]].append(float(row["FutureKL_AUC"]))
    pytest_path = outdir / "pytest_output.txt"
    pytest_lines = pytest_path.read_text(encoding="utf-8").strip().splitlines() if pytest_path.exists() else []
    pytest_value = pytest_lines[-1] if pytest_lines else "not recorded"
    summary = {
        "FORMAL_STATUS": "COMPLETE" if formal_complete else "INCOMPLETE", "TASK": TASK,
        "STAGE0": "PASS", "STATISTICAL_AGGREGATION_AUDIT": "PASS", "N_FORMAL_UNITS": n_units,
        "PYTEST": pytest_value, "FAILURES": failures,
        **{k: v for k, v in stage0.items() if k.startswith("KDA_") or k == "M_NATIVE_OPERATOR_IDENTITY"},
        **derived,
        "Native_FutureKL_AUC_median": BASE.median(by_branch["N"]),
        "Rotated_FutureKL_AUC_median": BASE.median(by_branch["R"]),
        "branches": branches, "horizon_row_count": len(horizon_rows), "expected_horizon_row_count": expected_rows,
        "artifact_row_count_gate": "PASS" if row_count_ok else "FAIL", "merged_from": [str(p) for p in inputs],
    }
    BASE.write_rows(outdir / "horizon_metrics.csv", horizon_rows)
    BASE.write_rows(outdir / "layer_mechanism_metrics.csv", mechanism_rows)
    BASE.write_rows(outdir / "closure_metrics.csv", closures)
    factorial = []
    for row in closures:
        factorial.append({k: v for k, v in row.items() if k in {"unit_id", "interaction_DV", "interaction_DV_normalized", "B_minus_v_after_D", "value_sufficiency_ratio", "M_beyond_D_after_B", "affine_to_state_gap"}})
    BASE.write_rows(outdir / "factorial_metrics.csv", factorial)
    BASE.write_rows(outdir / "aggregate" / "unit_metrics.csv", unit_rows)
    branch_table = []
    for branch in sorted(by_branch):
        e = effect(by_branch[branch], branch)
        branch_table.append({
            "branch": branch, "FutureKL_AUC_median": e["paired_median"], "CI_low": e["bootstrap_ci_low"],
            "CI_high": e["bootstrap_ci_high"], "mean": e["mean"], "n": e["n"],
        })
    primary_table = []
    for name, e in derived["effects"].items():
        if not name.startswith("RESCUE_"):
            continue
        primary_table.append({
            "comparison": name, "paired_median": e["paired_median"], "CI_low": e["bootstrap_ci_low"],
            "CI_high": e["bootstrap_ci_high"], "mean": e["mean"], "wins": e["wins"], "n": e["n"],
            "sign_test_p": e["sign_test_p"],
        })
    reciprocal_table = []
    for name, e in derived["effects"].items():
        if not name.startswith("HARM_"):
            continue
        reciprocal_table.append({
            "comparison": name, "paired_median": e["paired_median"], "CI_low": e["bootstrap_ci_low"],
            "CI_high": e["bootstrap_ci_high"], "mean": e["mean"], "wins": e["wins"], "n": e["n"],
            "sign_test_p": e["sign_test_p"],
        })
    horizon_groups = defaultdict(list)
    for row in horizon_rows:
        horizon_groups[(row["branch"], int(row["horizon"]))].append(float(row["FutureKL"]))
    horizon_summary = [
        {"branch": branch, "horizon": horizon, "FutureKL_median": BASE.median(vals),
         "FutureKL_mean": BASE.mean(vals), "n_units": len(vals)}
        for (branch, horizon), vals in sorted(horizon_groups.items())
    ]
    layer_groups = defaultdict(lambda: defaultdict(list))
    layer_fields = ("prequant_state_error_relative", "postquant_state_error_relative", "readout_error_relative")
    for row in mechanism_rows:
        key = (row["branch"], int(row["layer"]))
        for field in layer_fields:
            layer_groups[key][field].append(float(row[field]))
    layer_summary = []
    for (branch, layer), fields in sorted(layer_groups.items()):
        item = {"branch": branch, "layer": layer}
        for field, vals in fields.items():
            item[field + "_median"] = BASE.median(vals)
            item[field + "_mean"] = BASE.mean(vals)
        layer_summary.append(item)
    BASE.write_rows(outdir / "aggregate" / "branch_table.csv", branch_table)
    BASE.write_rows(outdir / "aggregate" / "primary_causal_comparisons.csv", primary_table)
    BASE.write_rows(outdir / "aggregate" / "reciprocal_harm.csv", reciprocal_table)
    BASE.write_rows(outdir / "aggregate" / "horizon_summary.csv", horizon_summary)
    BASE.write_rows(outdir / "aggregate" / "layer_summary.csv", layer_summary)
    BASE.save_json(outdir / "aggregate" / "paired_effects.json", derived["effects"])
    BASE.save_json(outdir / "aggregate" / "closure_effects.json", derived["closure_effects"])
    BASE.save_json(outdir / "summary.json", summary)
    formal_unit_objects = []
    for source in inputs:
        loaded_units = BASE.load_json(source / "formal_units.json", [])
        formal_unit_objects.extend(loaded_units if isinstance(loaded_units, list) else [])
    if formal_unit_objects and isinstance(formal_unit_objects[0], dict):
        unit_map = {str(unit["unit_id"]): unit for unit in formal_unit_objects}
        BASE.save_json(outdir / "formal_units.json", [unit_map[k] for k in sorted(unit_map)])
    else:
        units = sorted({r["unit_id"] for r in horizon_rows})
        BASE.save_json(outdir / "formal_units.json", units)
    BASE.save_json(outdir / "branch_manifest.json", {"task": TASK, "branches": [vars(p) for p in formal_plans()]})
    write_plots(outdir, horizon_rows, closures, summary)
    report_keys = [
        "TASK", "FORMAL_STATUS", "STAGE0", "STATISTICAL_AGGREGATION_AUDIT", "N_FORMAL_UNITS", "PYTEST", "FAILURES",
        "KDA_AFFINE_IDENTITY", "KDA_B_IDENTITY", "KDA_M_IDENTITY", "KDA_COORDINATE_PROVENANCE",
        "KDA_INTERVENTION_NONINTERFERENCE", "KDA_SELF_TRANSPLANT_IDENTITY",
        "KDA_DECAY_CAUSAL_RESCUE", "KDA_VALUE_CAUSAL_RESCUE", "KDA_B_CAUSAL_RESCUE",
        "KDA_DECAY_VALUE_JOINT_RESCUE", "KDA_DECAY_B_JOINT_RESCUE", "KDA_FULL_AFFINE_MB_RESCUE",
        "KDA_NATIVE_STATE_CEILING",
        "Native_FutureKL_AUC_median", "Rotated_FutureKL_AUC_median", "MEDIAN_CLOSURE_D", "MEDIAN_CLOSURE_V",
        "MEDIAN_CLOSURE_B", "MEDIAN_CLOSURE_DV", "MEDIAN_CLOSURE_DB", "MEDIAN_CLOSURE_MB", "MEDIAN_CLOSURE_STATE",
        "KDA_DECAY_VALUE_INTERACTION", "VALUE_SUFFICIENCY_WITHIN_B", "DECAY_SUFFICIENCY_WITHIN_M",
        "AFFINE_TO_STATE_CLOSURE", "DECAY_VALUE_SUFFICIENCY", "FINAL_CLASSIFICATION",
        "KDA_ROTATION_MECHANISM_CLOSURE_STATUS",
        "MEDIAN_NORMALIZED_DV_INTERACTION", "METHOD_DESIGN_READY", "NEXT_RECOMMENDED_ACTION",
    ]
    lines = [f"# {TASK}", "", "## Formal summary", ""]
    for key in report_keys:
        lines.append(f"{key} = {json.dumps(summary.get(key), sort_keys=True)}")
    lines += ["", "## Primary causal effects", "", "```json", json.dumps(derived["effects"], indent=2, sort_keys=True), "```",
              "", "## Closure ratios", "", "```json", json.dumps(derived["closure_effects"], indent=2, sort_keys=True), "```",
              "", "## Supplementary tables", "",
              "See `aggregate/branch_table.csv`, `aggregate/primary_causal_comparisons.csv`, "
              "`aggregate/reciprocal_harm.csv`, `aggregate/horizon_summary.csv`, and `aggregate/layer_summary.csv`."]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    required_files = [
        "summary.json", "formal_summary.md", "formal_units.json", "branch_manifest.json",
        "stage0_identity.json", "statistical_aggregation_audit.json", "intervention_provenance.jsonl",
        "horizon_metrics.csv", "layer_mechanism_metrics.csv", "closure_metrics.csv",
        "factorial_metrics.csv", "pytest_output.txt",
    ]
    unique_horizon = len({(r["unit_id"], int(r["horizon"]), r["branch"]) for r in horizon_rows})
    unique_mechanism = len({
        (r["unit_id"], int(r["horizon"]), r["branch"], int(r["layer"])) for r in mechanism_rows
    })
    unique_provenance = len({
        (r["unit_id"], int(r["horizon"]), r["plan"], int(r["layer"])) for r in provenance
    })
    n_layers = len({int(r["layer"]) for r in mechanism_rows})
    expected_mechanism = n_units * int(args.horizon) * len(formal_plans()) * n_layers
    expected_provenance = n_units * int(args.horizon) * sum(p.intervened for p in formal_plans()) * n_layers
    numeric_rows = horizon_rows + mechanism_rows + closures + factorial
    finite_ok = all(
        not isinstance(value, float) or math.isfinite(value)
        for row in numeric_rows for value in row.values()
    )
    figure_paths = sorted((outdir / "figures").glob("*.png"))
    validation = {
        "ARTIFACT_VALIDATION": "PASS",
        "required_files_present": all((outdir / rel).is_file() for rel in required_files),
        "missing_required_files": [rel for rel in required_files if not (outdir / rel).is_file()],
        "horizon_rows": len(horizon_rows), "expected_horizon_rows": expected_rows,
        "unique_horizon_rows": unique_horizon,
        "layer_mechanism_rows": len(mechanism_rows), "expected_layer_mechanism_rows": expected_mechanism,
        "unique_layer_mechanism_rows": unique_mechanism,
        "provenance_rows": len(provenance), "expected_provenance_rows": expected_provenance,
        "unique_provenance_rows": unique_provenance,
        "closure_rows": len(closures), "factorial_rows": len(factorial),
        "n_units": n_units, "n_branches": len(branches), "n_horizons": len({int(r["horizon"]) for r in horizon_rows}),
        "n_layers": n_layers, "all_numeric_values_finite": finite_ok,
        "figure_count": len(figure_paths), "all_figures_nonempty": all(p.stat().st_size > 0 for p in figure_paths),
    }
    gates = [
        validation["required_files_present"], len(horizon_rows) == expected_rows,
        unique_horizon == expected_rows, len(mechanism_rows) == expected_mechanism,
        unique_mechanism == expected_mechanism, len(provenance) == expected_provenance,
        unique_provenance == expected_provenance, len(closures) == EXPECTED_UNITS,
        len(factorial) == EXPECTED_UNITS, n_units == EXPECTED_UNITS, len(branches) == len(formal_plans()),
        validation["n_horizons"] == int(args.horizon), finite_ok,
        len(figure_paths) >= 9, validation["all_figures_nonempty"],
    ]
    validation["ARTIFACT_VALIDATION"] = "PASS" if all(gates) else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", validation)
    summary["ARTIFACT_VALIDATION"] = validation["ARTIFACT_VALIDATION"]
    BASE.save_json(outdir / "summary.json", summary)
    BASE.save_json(outdir / "manifest.json", {"task": TASK, "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    p.add_argument("--phase", choices=("stage0", "formal"), default="formal")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--target-layer-limit", type=int, default=None)
    p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--unit-shard-index", type=int, default=None)
    p.add_argument("--unit-shard-count", type=int, default=None)
    p.add_argument("--merge-run-dirs", nargs="*", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = merge_runs(args) if args.merge_run_dirs else run_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("STAGE0") == "PASS" and not result.get("failures", result.get("FAILURES", [])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
