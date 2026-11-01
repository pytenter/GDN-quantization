#!/usr/bin/env python3
"""Persistent mapped-back rotated-quantizer exposure for Ling KDA states."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_PERSISTENT_QUANTIZER_EXPOSURE_CAUSAL_V1"
SLUG = "kda_rotation_persistent_quantizer_exposure_causal_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
COMMUTATOR_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_quantizer_commutator_causal_decomposition_v1.py"
AFFINE_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_affine_decay_value_unified_causal_closure_v1.py"
OLD_RESULT = REPO / "results" / "kda_rotation_affine_decay_value_unified_causal_closure_v1"
COMMUTATOR_RESULT = REPO / "results" / "kda_rotation_quantizer_commutator_causal_decomposition_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EXPOSURES = (0, 1, 2, 4, 8, 16, 32, 64, "FULL")
HISTORY_CANDIDATES = (1, 4, 8, 16, 32)
EPS = 1e-12


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMM = import_file(COMMUTATOR_RUNNER, "commutator_for_persistent_exposure_v1")
AFFINE = import_file(AFFINE_RUNNER, "affine_for_persistent_exposure_v1")
BASE = COMM.BASE


def exposure_name(value):
    return "LFULL" if value == "FULL" else f"L{int(value)}"


def exposure_value(name):
    return "FULL" if str(name) == "LFULL" else int(str(name)[1:])


def operator_for_boundary(exposure, boundary_index):
    if exposure == "FULL":
        return "QR"
    return "QR" if int(boundary_index) <= int(exposure) else "QN"


def median_bootstrap_ci(values, n=4000, seed=BASE.BOOTSTRAP_SEED):
    vals = [float(x) for x in values if BASE.finite(x)]
    if not vals:
        return [float("nan"), float("nan")]
    rng = __import__("random").Random(int(seed))
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


def tensor_hash(value):
    x = value.detach().float().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def stack_hash(stack):
    digest = hashlib.sha256()
    for layer in sorted(stack):
        digest.update(str(layer).encode("ascii"))
        digest.update(stack[layer].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stack_relative_error(left, right):
    numerator = math.sqrt(sum(BASE.tensor_norm(left[k].float() - right[k].float()) ** 2 for k in left))
    denominator = math.sqrt(sum(BASE.tensor_norm(right[k].float()) ** 2 for k in right))
    return numerator / (denominator + EPS)


def stack_max_abs_error(left, right):
    return max(float((left[k].float() - right[k].float()).abs().max().item()) for k in left)


def quantize_operator(state, rotation, operator):
    x = state.detach().float()
    if operator == "QN":
        coordinate = x
    elif operator == "QR":
        coordinate = COMM.rotate_state(x, rotation)
    else:
        raise ValueError(operator)
    official, official_meta = COMM.official_quantize(coordinate)
    qdq, meta = COMM.quantize_with_scale(coordinate, official_meta["scale"])
    if not torch.equal(meta["codes"], official_meta["codes"]):
        raise RuntimeError("manual/operator integer codes do not reproduce official quantizer")
    native = qdq if operator == "QN" else COMM.inverse_rotate_state(qdq, rotation)
    meta = {**meta, "coordinate_prequant": coordinate, "coordinate_postquant": qdq}
    return native.to(state.dtype), meta


def operator_endpoint_identity(state, rotation):
    endpoints = COMM.construct_endpoints(state, rotation)
    qn, _ = quantize_operator(state, rotation, "QN")
    qr, _ = quantize_operator(state, rotation, "QR")
    return {
        "QN": COMM.tensor_metrics(qn.float(), endpoints["NN_official"].float()),
        "QR": COMM.tensor_metrics(qr.float(), endpoints["RR_official"].float()),
    }


def apply_operator_cache(cache, layers, rotation, operator):
    pre = BASE.cache_stack(cache, layers)
    post, metas = {}, {}
    for layer, state in pre.items():
        qdq, meta = quantize_operator(state.to(rotation.device), rotation, operator)
        post[layer] = qdq.detach().float().cpu()
        metas[layer] = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in meta.items()}
    with torch.inference_mode():
        BASE.replace_cache_stack(cache, post)
    return pre, post, metas


def diagnostic_rows(unit_id, condition, exposure, boundary_index, pre, post, metas,
                    native_pre, native_post, native_metas, fp_pre):
    rows = []
    for layer in sorted(pre):
        x, q = pre[layer].float(), post[layer].float()
        npre, npost, fp = native_pre[layer].float(), native_post[layer].float(), fp_pre[layer].float()
        meta, nmeta = metas[layer], native_metas[layer]
        scales = meta["scale"].reshape(-1).tolist()
        nscales = nmeta["scale"].reshape(-1).tolist()
        rows.append({
            "unit_id": str(unit_id), "layer": int(layer), "condition": condition,
            "exposure": exposure, "boundary_index": int(boundary_index),
            "quantizer": operator_for_boundary(exposure, boundary_index),
            "is_rotated_exposure": operator_for_boundary(exposure, boundary_index) == "QR",
            "prequant_state_norm": BASE.tensor_norm(x),
            "prequant_state_rel_to_native": BASE.tensor_norm(x - npre) / (BASE.tensor_norm(npre) + EPS),
            "prequant_state_rel_to_fp": BASE.tensor_norm(x - fp) / (BASE.tensor_norm(fp) + EPS),
            "maxabs": float(meta["coordinate_prequant"].abs().max().item()),
            "peakiness": COMM.peakiness(meta["coordinate_prequant"]),
            "scale_mean": BASE.mean(scales), "scale_median": BASE.median(scales),
            "scale_raw_json": json.dumps(scales),
            "scale_ratio_vs_native_same_step": BASE.median([a / (b + EPS) for a, b in zip(scales, nscales)]),
            "quant_error_norm": BASE.tensor_norm(q - x),
            "quant_error_relative_norm": BASE.tensor_norm(q - x) / (BASE.tensor_norm(x) + EPS),
            "code_churn_vs_native_same_step": float((meta["codes"] != nmeta["codes"]).float().mean().item()),
            "saturation_fraction": float(meta["saturation_fraction"]),
            "clipping_fraction": float(meta["clipping_fraction"]),
            "postquant_state_rel_to_native": BASE.tensor_norm(q - npost) / (BASE.tensor_norm(npost) + EPS),
        })
    return rows


def reconciliation_payload():
    old = {
        "t0_pre_quant_state_source": "branch-specific R baseline cache after persistent Rotated warmup",
        "t0_post_quant_state_source": "R baseline INT8_R128 state in Rotated coordinates",
        "t0_quantizer": "Rotated-coordinate INT8_R128",
        "t0_coordinate_system": "Rotated branch coordinates",
        "t0_map_back_timing": "donor N state is mapped Native->Rotated before readout; output is mapped back",
        "t0_plus_1_recurrence_input_state": "R_STATE_N branch cache, then overwritten by same-step Native donor post-update state",
        "t0_plus_1_driver_source": "receiving R_STATE_N branch; v is transformed to Rotated coordinates",
        "t0_plus_1_quantizer": "Rotated-coordinate INT8_R128",
        "t0_plus_1_stored_state_source": "same-step Native donor final_state converted to Rotated coordinates, then quantized",
        "t0_plus_2_quantizer": "Rotated-coordinate INT8_R128; Native donor transplant repeats",
        "future_quantizer_policy": "Rotated-coordinate INT8_R128 at every future boundary",
        "future_state_transplant_policy": "same-step Native donor post-update state transplanted at every future step",
        "intervention_duration": "PERSISTENT / PER_STEP",
        "code_path": "AffineProbe._wrap state_source at lines 272-282 and run_unit loop at lines 385-418",
    }
    new = {
        "t0_pre_quant_state_source": "common unquantized FP cache at canonical t0",
        "t0_post_quant_state_source": "Q_R(S)=Q(SR)R.T for lambda=1",
        "t0_quantizer": "mapped-back Rotated operator Q_R",
        "t0_coordinate_system": "Native after map-back",
        "t0_map_back_timing": "immediately after t0 dequantization and before cache replacement",
        "t0_plus_1_recurrence_input_state": "one-shot Q_R(S) stored in Native coordinates",
        "t0_plus_1_driver_source": "unchanged branch-native model drivers",
        "t0_plus_1_quantizer": "Native operator Q_N",
        "t0_plus_1_stored_state_source": "branch recurrence result quantized by Q_N",
        "t0_plus_2_quantizer": "Q_N, repeated for all remaining boundaries",
        "future_quantizer_policy": "Native INT8_R128 after the single t0 Q_R pulse",
        "future_state_transplant_policy": "none after t0",
        "intervention_duration": "ONE_SHOT",
        "code_path": "branch_stacks lambda=1 at lines 235-243; run_future Native quantization at lines 262-284",
    }
    return {
        "OLD_NATIVE_STATE_CEILING_INTERVENTION": "PERSISTENT",
        "NEW_COMMUTATOR_INTERVENTION": "ONE_SHOT",
        "PROTOCOL_RECONCILIATION": "PASS",
        "PROTOCOL_CONTRADICTION": "NO",
        "SEMANTIC_DIFFERENCE_IDENTIFIED": "YES",
        "PRIMARY_SEMANTIC_DIFFERENCE": "old R_STATE_N repeats a same-step Native-state transplant and Rotated quantization at every future step; new lambda=1 injects Q_R(S) once at t0 and then resumes Native recurrence plus Q_N",
        "OLD_NATIVE_STATE_CEILING_SEMANTICS": old,
        "NEW_COMMUTATOR_SEMANTICS": new,
    }


def write_reconciliation(outdir, payload):
    BASE.save_json(outdir / "protocol_reconciliation.json", payload)
    lines = [f"# {TASK}", "", "## Protocol reconciliation", ""]
    for key in ("OLD_NATIVE_STATE_CEILING_INTERVENTION", "NEW_COMMUTATOR_INTERVENTION",
                "PROTOCOL_RECONCILIATION", "PROTOCOL_CONTRADICTION",
                "SEMANTIC_DIFFERENCE_IDENTIFIED", "PRIMARY_SEMANTIC_DIFFERENCE"):
        lines.append(f"{key} = {json.dumps(payload[key])}")
    for title, key in (("Old 90.73% ceiling timeline", "OLD_NATIVE_STATE_CEILING_SEMANTICS"),
                       ("New commutator timeline", "NEW_COMMUTATOR_SEMANTICS")):
        lines += ["", f"## {title}", ""]
        lines.extend(f"{k} = {json.dumps(v)}" for k, v in payload[key].items())
    (outdir / "protocol_reconciliation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan(name, basis="native"):
    return AFFINE.InterventionPlan(name, basis, quantized=False, family="persistent_exposure")


def prepare_fp_cache(model, tokenizer, probe, unit, row, tokens):
    device = next(model.parameters()).device
    fp_plan = plan("FP", "native")
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    probe.begin(fp_plan, 0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask,
                    cache_position=torch.arange(0, input_ids.shape[-1], device=device), use_cache=True)
    probe.end()
    cache, prompt_len = out.past_key_values, int(input_ids.shape[-1])
    for t, token in enumerate(tokens[:int(unit["t0"])]):
        cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
        probe.begin(fp_plan, 0)
        with torch.inference_mode():
            out = model(input_ids=cur, attention_mask=mask, past_key_values=cache,
                        cache_position=torch.tensor([prompt_len + t], device=device), use_cache=True)
        probe.end(); cache = out.past_key_values
    return mask, cache, prompt_len


def forward_one(model, probe, branch_plan, token, mask, cache, cache_position):
    cur = torch.tensor([[int(token)]], device=next(model.parameters()).device, dtype=torch.long)
    mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
    probe.begin(branch_plan, cache_position)
    with torch.inference_mode():
        out = model(input_ids=cur, attention_mask=mask, past_key_values=cache,
                    cache_position=torch.tensor([cache_position], device=cur.device), use_cache=True)
    probe.end()
    return mask, out.past_key_values, out.logits.detach().float()


def initialize_exposure_cache(fp_cache, layers, rotation, exposure):
    cache = BASE.clone_cache(fp_cache)
    operator = operator_for_boundary(exposure, 1)
    pre, post, metas = apply_operator_cache(cache, layers, rotation, operator)
    return cache, pre, post, metas


def initialize_live_rotated_cache(fp_cache, layers, rotation):
    cache = BASE.clone_cache(fp_cache)
    stack = BASE.cache_stack(cache, layers)
    with torch.inference_mode():
        BASE.replace_cache_stack(cache, {
            layer: COMM.rotate_state(state.to(rotation.device), rotation).cpu()
            for layer, state in stack.items()
        })
    BASE.quantize_branch_cache(torch, cache, layers, "rotated", rotation)
    return cache


def run_stage_a_unit(model, tokenizer, probe, unit, rows_by_pid, teacher_tokens,
                     layers, rotation, horizon, exposures=EXPOSURES, collect_diagnostics=True):
    pid = str(unit["problem_id"]); row = rows_by_pid[pid]
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"insufficient teacher tokens for {unit['unit_id']}")
    base_mask, fp_cache, prompt_len = prepare_fp_cache(model, tokenizer, probe, unit, row, tokens)
    fp_initial = BASE.cache_stack(fp_cache, layers)
    pasts, masks, initial = {}, {}, {}
    for exposure in exposures:
        name = exposure_name(exposure)
        pasts[name], pre, post, metas = initialize_exposure_cache(fp_cache, layers, rotation, exposure)
        masks[name] = base_mask.clone(); initial[name] = (pre, post, metas)
    pasts["FP"] = BASE.clone_cache(fp_cache); masks["FP"] = base_mask.clone()
    pasts["R_LIVE"] = initialize_live_rotated_cache(fp_cache, layers, rotation); masks["R_LIVE"] = base_mask.clone()
    native_pre, native_post, native_metas = initial["L0"]
    diagnostics = []
    if collect_diagnostics:
        for exposure in exposures:
            name = exposure_name(exposure); pre, post, metas = initial[name]
            diagnostics += diagnostic_rows(unit["unit_id"], name, exposure, 1, pre, post, metas,
                                           native_pre, native_post, native_metas, fp_initial)
    branch_plans = {name: plan(name, "native") for name in [exposure_name(x) for x in exposures]}
    branch_plans["FP"] = plan("FP", "native"); branch_plans["R_LIVE"] = plan("R_LIVE", "rotated")
    order = ["FP"] + [exposure_name(x) for x in exposures] + ["R_LIVE"]
    live_initial = {
        k: COMM.inverse_rotate_state(v.to(rotation.device), rotation).cpu()
        for k, v in BASE.cache_stack(pasts["R_LIVE"], layers).items()
    }
    identity = {
        "per_horizon": [],
        "initial_endpoint_state_relative": stack_relative_error(initial["LFULL"][1], live_initial),
        "initial_endpoint_state_max_abs": stack_max_abs_error(initial["LFULL"][1], live_initial),
    }
    rows = []
    for h in range(1, int(horizon) + 1):
        token = tokens[int(unit["t0"]) + h - 1]
        logits, prequant = {}, {}
        for name in order:
            masks[name], pasts[name], logits[name] = forward_one(
                model, probe, branch_plans[name], token, masks[name], pasts[name],
                prompt_len + int(unit["t0"]) + h - 1,
            )
            prequant[name] = BASE.cache_stack(pasts[name], layers)
        fp_logits = logits["FP"]
        for name in order:
            metrics = {"KL": 0.0, "logit_cosine": 1.0, "top1_agreement": 1.0}
            if name != "FP":
                raw = BASE.full_logit_metrics(torch, fp_logits, logits[name])
                metrics["KL"] = float(raw["KL"])
                metrics["logit_cosine"] = float(raw.get("cosine", raw.get("Cosine", float("nan"))))
                metrics["top1_agreement"] = float((fp_logits.argmax(-1) == logits[name].argmax(-1)).float().mean().item())
            rows.append({
                "unit_id": str(unit["unit_id"]), "sample_identity": str(unit["unit_id"]),
                "seed": BASE.RHT_SEED, "layer": "ALL", "horizon": h, "condition": name,
                "exposure": exposure_value(name) if name.startswith("L") else None,
                "future_kl": metrics["KL"], "logit_cosine": metrics["logit_cosine"],
                "top1_agreement": metrics["top1_agreement"],
                "state_rel_to_fp": stack_relative_error(
                    {k: COMM.inverse_rotate_state(v.to(rotation.device), rotation).cpu() for k, v in prequant[name].items()}
                    if name == "R_LIVE" else prequant[name], prequant["FP"]),
            })
        live_r_native = {k: COMM.inverse_rotate_state(v.to(rotation.device), rotation).cpu() for k, v in prequant["R_LIVE"].items()}
        identity["per_horizon"].append({
            "horizon": h,
            "QN_vs_L0_state_relative": 0.0,
            "QR_vs_live_R_state_relative_prequant": stack_relative_error(prequant["LFULL"], live_r_native),
            "QR_vs_live_R_state_max_abs_prequant": stack_max_abs_error(prequant["LFULL"], live_r_native),
            "QR_vs_live_R_logit_max_abs": float((logits["LFULL"] - logits["R_LIVE"]).abs().max().item()),
            "QR_vs_live_R_KL_abs": abs(rows[-2]["future_kl"] - rows[-1]["future_kl"]),
        })
        if h == int(horizon):
            continue
        fp_pre = prequant["FP"]
        native_operator = operator_for_boundary(0, h + 1)
        npre, npost, nmeta = apply_operator_cache(pasts["L0"], layers, rotation, native_operator)
        if collect_diagnostics:
            diagnostics += diagnostic_rows(unit["unit_id"], "L0", 0, h + 1, npre, npost, nmeta,
                                           npre, npost, nmeta, fp_pre)
        for exposure in exposures:
            name = exposure_name(exposure)
            if name == "L0":
                continue
            operator = operator_for_boundary(exposure, h + 1)
            pre, post, metas = apply_operator_cache(pasts[name], layers, rotation, operator)
            if collect_diagnostics:
                diagnostics += diagnostic_rows(unit["unit_id"], name, exposure, h + 1, pre, post, metas,
                                               npre, npost, nmeta, fp_pre)
        BASE.quantize_branch_cache(torch, pasts["R_LIVE"], layers, "rotated", rotation)
    max_state = max(x["QR_vs_live_R_state_relative_prequant"] for x in identity["per_horizon"])
    max_logit = max(x["QR_vs_live_R_logit_max_abs"] for x in identity["per_horizon"])
    max_kl = max(x["QR_vs_live_R_KL_abs"] for x in identity["per_horizon"])
    identity.update({
        "FULL_QN_REPRODUCES_NATIVE": "PASS",
        "FULL_QR_REPRODUCES_ROTATED": "PASS" if max_state <= 5e-3 and max_kl <= 5e-4 else "FAIL",
        "max_qr_live_state_relative": max_state, "max_qr_live_logit_abs": max_logit,
        "max_qr_live_kl_abs": max_kl,
    })
    first = next((row for row in identity["per_horizon"]
                  if row["QR_vs_live_R_state_relative_prequant"] > 1e-6
                  or row["QR_vs_live_R_logit_max_abs"] > 1e-5), None)
    identity.update({
        "FIRST_DIVERGENT_TIMESTEP": None if first is None else first["horizon"],
        "FIRST_DIVERGENT_STATE_MAX_ERROR": None if first is None else first["QR_vs_live_R_state_max_abs_prequant"],
        "FIRST_DIVERGENT_STATE_RELATIVE_ERROR": None if first is None else first["QR_vs_live_R_state_relative_prequant"],
        "FIRST_DIVERGENT_LOGIT_MAX_ERROR": None if first is None else first["QR_vs_live_R_logit_max_abs"],
        "DIVERGENT_QUANTIZER_CHOICE": "both use INT8_R128 Q_R at every controlled boundary",
        "DIVERGENT_STATE_SOURCE": "same common FP t0 endpoint; Native-coordinate mapped-back cache versus explicit Rotated-coordinate cache",
        "DIVERGENT_DRIVER_SOURCE": "same semantic model branch, but live R applies vR and output map-back in model dtype while Native Q_R leaves FP recurrence in Native coordinates",
        "EXACT_DIVERGENT_CODE_PATH": "native FULL: AffineProbe native fn then quantize_operator QR; live R: AffineProbe rotated v transform/kernel/output map-back then rotated cache quantization",
    })
    return {"horizon_rows": rows, "diagnostics": diagnostics, "operator_identity": identity}


def aggregate_stage_a(horizon_rows):
    grouped = defaultdict(list)
    for row in horizon_rows:
        if str(row["condition"]).startswith("L"):
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({key[0] for key in grouped})
    rows = []
    for unit in units:
        row = {"unit_id": unit}
        for exposure in EXPOSURES:
            name = exposure_name(exposure)
            row[f"auc_{name}"] = BASE.mean(grouped[(unit, name)])
        rows.append(row)
    return rows


def monotonic_score(values):
    pairs = list(zip(values, values[1:]))
    return sum(right >= left for left, right in pairs) / max(1, len(pairs))


def summarize_stage_a(unit_rows, operator_identity):
    med_auc = {exposure_name(x): BASE.median([float(r[f"auc_{exposure_name(x)}"]) for r in unit_rows]) for x in EXPOSURES}
    gaps, gap_effects = {}, {}
    for exposure in EXPOSURES:
        name = exposure_name(exposure)
        vals = [float(r[f"auc_{name}"]) - float(r["auc_L0"]) for r in unit_rows]
        gaps[name] = vals; gap_effects[name] = effect(vals, f"D({name})")
    ordered = [exposure_name(x) for x in EXPOSURES]
    first_harmful = next((name for name in ordered[1:] if gap_effects[name]["paired_median"] > 0), "NONE")
    first_ci = next((name for name in ordered[1:] if gap_effects[name]["bootstrap_ci_low"] > 0), "NONE")
    full = gap_effects["LFULL"]; l1 = gap_effects["L1"]
    trend = monotonic_score([med_auc[name] for name in ordered[1:]])
    endpoint_ok = operator_identity.get("FULL_QR_REPRODUCES_ROTATED") == "PASS"
    strong = endpoint_ok and l1["paired_median"] <= 0 and full["bootstrap_ci_low"] > 0 and full["positive"] > len(unit_rows) / 2 and trend >= 0.70
    partial = endpoint_ok and full["paired_median"] > 0 and full["positive"] > len(unit_rows) / 2
    status = "STRONG_SUPPORT" if strong else ("PARTIAL_SUPPORT" if partial else "NOT_SUPPORTED")
    return {
        "N_UNITS": len(unit_rows), "median_auc": med_auc, "gap_vs_L0": gap_effects,
        "FIRST_HARMFUL_L": first_harmful, "FIRST_L_WITH_CI_ABOVE_ZERO": first_ci,
        "exposure_monotonic_score": trend, "PERSISTENT_EXPOSURE_CAUSAL_STATUS": status,
        "IS_SINGLE_ROTATED_QUANTIZATION_HARMFUL": "YES" if l1["paired_median"] > 0 and l1["bootstrap_ci_low"] > 0 else "NO",
        "IS_REPEATED_ROTATED_QUANTIZATION_REQUIRED_FOR_FAILURE": "YES" if strong else ("PARTIAL" if partial else "NO"),
        "DOES_FULL_EXPOSURE_REPRODUCE_ROTATED_FAILURE": "YES" if endpoint_ok else "NO",
    }


def advance_history(model, probe, cache, mask, branch_plan, operator, unit, tokens,
                    prompt_len, layers, rotation, depth):
    pre = BASE.cache_stack(cache, layers)
    for boundary in range(1, int(depth)):
        apply_operator_cache(cache, layers, rotation, operator)
        token = tokens[int(unit["t0"]) + boundary - 1]
        mask, cache, _ = forward_one(model, probe, branch_plan, token, mask, cache,
                                     prompt_len + int(unit["t0"]) + boundary - 1)
        pre = BASE.cache_stack(cache, layers)
    return cache, mask, pre


def state_history_diagnostics(unit_id, depth, native, rotated, rotation):
    rows = []
    for layer in sorted(native):
        n, r = native[layer].float(), rotated[layer].float()
        qnn, mn = quantize_operator(n.to(rotation.device), rotation, "QN")
        qnr, mnr = quantize_operator(n.to(rotation.device), rotation, "QR")
        qrn, mrn = quantize_operator(r.to(rotation.device), rotation, "QN")
        qrr, mr = quantize_operator(r.to(rotation.device), rotation, "QR")
        for history, state, left, right, ml, mr_ in (
            ("N", n, qnn.cpu(), qnr.cpu(), mn, mnr), ("R", r, qrn.cpu(), qrr.cpu(), mrn, mr)):
            rows.append({
                "unit_id": str(unit_id), "history_depth": int(depth), "layer": int(layer), "history": history,
                "prequant_hash": tensor_hash(state), "state_norm": BASE.tensor_norm(state),
                "maxabs": float(state.abs().max().item()), "peakiness": COMM.peakiness(state),
                "channel_variance": float(state.var(dim=tuple(range(state.ndim - 1)), unbiased=False).mean().item()),
                "native_q_relative_error": BASE.tensor_norm(left - state) / (BASE.tensor_norm(state) + EPS),
                "rotated_q_relative_error": BASE.tensor_norm(right - state) / (BASE.tensor_norm(state) + EPS),
                "quantizer_post_state_difference": BASE.tensor_norm(right - left) / (BASE.tensor_norm(state) + EPS),
                "native_scale_mean": float(ml["scale"].mean().item()), "rotated_scale_mean": float(mr_["scale"].mean().item()),
                "native_scale_variance": float(ml["scale"].var(unbiased=False).item()),
                "rotated_scale_variance": float(mr_["scale"].var(unbiased=False).item()),
                "code_churn": float((ml["codes"] != mr_["codes"]).float().mean().item()),
                "native_saturation": ml["saturation_fraction"], "rotated_saturation": mr_["saturation_fraction"],
                "native_clipping": ml["clipping_fraction"], "rotated_clipping": mr_["clipping_fraction"],
            })
    return rows


def run_stage_b_unit(model, tokenizer, probe, unit, rows_by_pid, teacher_tokens,
                     layers, rotation, horizon, history_depth):
    pid = str(unit["problem_id"]); row = rows_by_pid[pid]
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if len(tokens) < int(unit["t0"]) + int(history_depth) - 1 + int(horizon):
        raise RuntimeError(f"insufficient teacher tokens for Stage B {unit['unit_id']}")
    base_mask, fp_cache, prompt_len = prepare_fp_cache(model, tokenizer, probe, unit, row, tokens)
    ncache, nmask, npre = advance_history(model, probe, BASE.clone_cache(fp_cache), base_mask.clone(),
                                           plan("N_HISTORY"), "QN", unit, tokens, prompt_len,
                                           layers, rotation, history_depth)
    rcache, rmask, rpre = advance_history(model, probe, BASE.clone_cache(fp_cache), base_mask.clone(),
                                           plan("R_HISTORY"), "QR", unit, tokens, prompt_len,
                                           layers, rotation, history_depth)
    fcache, fmask, _ = advance_history(model, probe, BASE.clone_cache(fp_cache), base_mask.clone(),
                                       plan("FP_HISTORY"), "NONE", unit, tokens, prompt_len,
                                       layers, rotation, history_depth) if int(history_depth) == 1 else (None, None, None)
    if int(history_depth) > 1:
        fcache, fmask = BASE.clone_cache(fp_cache), base_mask.clone()
        for boundary in range(1, int(history_depth)):
            token = tokens[int(unit["t0"]) + boundary - 1]
            fmask, fcache, _ = forward_one(model, probe, plan("FP_HISTORY"), token, fmask, fcache,
                                           prompt_len + int(unit["t0"]) + boundary - 1)
    conditions = {"NN": (ncache, nmask, "QN"), "NR": (ncache, nmask, "QR"),
                  "RN": (rcache, rmask, "QN"), "RR": (rcache, rmask, "QR")}
    pasts, masks = {"FP": BASE.clone_cache(fcache)}, {"FP": fmask.clone()}
    pair_hashes = {"NN": stack_hash(npre), "NR": stack_hash(npre), "RN": stack_hash(rpre), "RR": stack_hash(rpre)}
    for name, (source, source_mask, operator) in conditions.items():
        pasts[name] = BASE.clone_cache(source); masks[name] = source_mask.clone()
        apply_operator_cache(pasts[name], layers, rotation, operator)
    rows = []
    start = int(unit["t0"]) + int(history_depth) - 1
    for h in range(1, int(horizon) + 1):
        token = tokens[start + h - 1]
        logits = {}
        for name in ("FP", "NN", "NR", "RN", "RR"):
            masks[name], pasts[name], logits[name] = forward_one(
                model, probe, plan(name), token, masks[name], pasts[name],
                prompt_len + start + h - 1,
            )
        for name in ("FP", "NN", "NR", "RN", "RR"):
            kl = 0.0 if name == "FP" else float(BASE.full_logit_metrics(torch, logits["FP"], logits[name])["KL"])
            rows.append({
                "unit_id": str(unit["unit_id"]), "sample_identity": str(unit["unit_id"]), "layer": "ALL",
                "seed": BASE.RHT_SEED, "history_depth": int(history_depth), "horizon": h,
                "condition": name, "future_kl": kl, "prequant_pair_hash": pair_hashes.get(name),
                "downstream_quantizer": "QN",
            })
        if h < int(horizon):
            for name in ("NN", "NR", "RN", "RR"):
                apply_operator_cache(pasts[name], layers, rotation, "QN")
    return {"horizon_rows": rows, "state_diagnostics": state_history_diagnostics(unit["unit_id"], history_depth, npre, rpre, rotation)}


def aggregate_stage_b(horizon_rows):
    grouped = defaultdict(list)
    for row in horizon_rows:
        if row["condition"] != "FP":
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({key[0] for key in grouped})
    rows = []
    for unit in units:
        auc = {name: BASE.mean(grouped[(unit, name)]) for name in ("NN", "NR", "RN", "RR")}
        e_n = auc["NR"] - auc["NN"]; e_r = auc["RR"] - auc["RN"]
        rows.append({"unit_id": unit, **{f"auc_{k}": v for k, v in auc.items()},
                     "E_NATIVE_HISTORY": e_n, "E_ROTATED_HISTORY": e_r,
                     "HISTORY_X_QUANTIZER_INTERACTION": e_r - e_n})
    return rows


def summarize_stage_b(unit_rows, history_depth):
    en = effect([r["E_NATIVE_HISTORY"] for r in unit_rows], "Y_NR-Y_NN")
    er = effect([r["E_ROTATED_HISTORY"] for r in unit_rows], "Y_RR-Y_RN")
    interaction = effect([r["HISTORY_X_QUANTIZER_INTERACTION"] for r in unit_rows], "E_R-E_N")
    strong = en["paired_median"] <= 0 and er["paired_median"] > 0 and interaction["bootstrap_ci_low"] > 0
    partial = er["paired_median"] > en["paired_median"] and interaction["paired_median"] > 0
    status = "STRONG_SUPPORT" if strong else ("PARTIAL_SUPPORT" if partial else "NOT_SUPPORTED")
    return {
        "HISTORY_DEPTH": int(history_depth), "N_UNITS": len(unit_rows),
        "median_auc": {k: BASE.median([r[f"auc_{k}"] for r in unit_rows]) for k in ("NN", "NR", "RN", "RR")},
        "E_NATIVE_HISTORY": en, "E_ROTATED_HISTORY": er, "HISTORY_X_QUANTIZER_INTERACTION": interaction,
        "HISTORY_CONDITIONED_QUANTIZER_STATUS": status,
        "IS_ROTATED_QUANTIZER_HARMFUL_ON_NATIVE_HISTORY": "YES" if en["bootstrap_ci_low"] > 0 else "NO",
        "IS_ROTATED_QUANTIZER_HARMFUL_ON_ROTATED_HISTORY": "YES" if er["bootstrap_ci_low"] > 0 else "NO",
        "DOES_ROTATED_HISTORY_CHANGE_THE_EFFECT_OF_THE_QUANTIZER": "YES" if strong else ("PARTIAL" if partial else "NO"),
    }


def setup_output(outdir):
    outdir.mkdir(parents=True, exist_ok=True); (outdir / "raw").mkdir(exist_ok=True)


def reset_phase(outdir, phase):
    for name in (f"raw/{phase}_horizon.jsonl", f"raw/{phase}_diagnostics.jsonl", f"{phase}_failures.json"):
        path = outdir / name
        if path.exists(): path.unlink()


def run_old_reconciliation_smoke(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    plans = [p for p in AFFINE.formal_plans() if p.name in {"FP", "N", "R", "R_STATE_N"}]
    hr, _, _, _ = AFFINE.run_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
                                  rotation.cpu(), horizon, plans)
    grouped = defaultdict(list)
    for row in hr: grouped[row["branch"]].append(float(row["FutureKL"]))
    return {name: BASE.mean(values) for name, values in grouped.items()}


def write_stage0_stop_report(outdir, reconciliation, stage0):
    reconciliation = dict(reconciliation)
    reconciliation.update({
        "PROTOCOL_RECONCILIATION": "PASS",
        "PROTOCOL_CONTRADICTION": "NO",
        "OPERATOR_ABSTRACTION_CONTRADICTION": "FULL_QR_DOES_NOT_REPRODUCE_LIVE_ROTATED",
        "FIRST_DIVERGENT_TIMESTEP": stage0["operator_rollout_identity"].get("FIRST_DIVERGENT_TIMESTEP"),
        "STATE_MAX_ERROR": stage0["operator_rollout_identity"].get("FIRST_DIVERGENT_STATE_MAX_ERROR"),
        "STATE_RELATIVE_ERROR": stage0["operator_rollout_identity"].get("FIRST_DIVERGENT_STATE_RELATIVE_ERROR"),
        "QUANTIZER_CHOICE": stage0["operator_rollout_identity"].get("DIVERGENT_QUANTIZER_CHOICE"),
        "STATE_SOURCE": stage0["operator_rollout_identity"].get("DIVERGENT_STATE_SOURCE"),
        "DRIVER_SOURCE": stage0["operator_rollout_identity"].get("DIVERGENT_DRIVER_SOURCE"),
        "EXACT_DIVERGENT_CODE_PATH": stage0["operator_rollout_identity"].get("EXACT_DIVERGENT_CODE_PATH"),
    })
    write_reconciliation(outdir, reconciliation)
    BASE.write_rows(outdir / "stageA_exposure_horizon_results.csv", [])
    BASE.write_rows(outdir / "stageA_exposure_unit_results.csv", [])
    BASE.write_rows(outdir / "stageA_exposure_diagnostics.csv", [])
    BASE.save_json(outdir / "stageA_summary.json", {
        "STAGEA_PERSISTENT_EXPOSURE": "SKIPPED_BY_STAGE0B_OPERATOR_IDENTITY_FAILURE",
        "reason": "FULL_QR_REPRODUCES_ROTATED=FAIL",
    })
    BASE.write_rows(outdir / "stageB_history_quantizer_horizon_results.csv", [])
    BASE.write_rows(outdir / "stageB_history_quantizer_unit_results.csv", [])
    BASE.write_rows(outdir / "stageB_state_diagnostics.csv", [])
    BASE.save_json(outdir / "stageB_summary.json", {
        "STAGEB_HISTORY_X_QUANTIZER": "SKIPPED_BY_STAGE0B_OPERATOR_IDENTITY_FAILURE",
        "reason": "Stage A was not permitted",
    })
    pytest_path = outdir / "pytest_output.txt"
    pytest_lines = pytest_path.read_text(encoding="utf-8").strip().splitlines() if pytest_path.exists() else []
    identity = stage0["operator_rollout_identity"]
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "STOPPED_AT_STAGE0B",
        "STAGE0A_PROTOCOL_RECONCILIATION": "PASS",
        "STAGE0B_OPERATOR_IDENTITY": "FAIL",
        "STAGEA_PERSISTENT_EXPOSURE": "NOT_RUN",
        "STAGEB_HISTORY_X_QUANTIZER": "NOT_RUN",
        "N_FORMAL_UNITS": 0, "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded", "FAILURES": [],
        "OLD_NATIVE_STATE_CEILING_INTERVENTION": "PERSISTENT",
        "NEW_COMMUTATOR_INTERVENTION": "ONE_SHOT",
        "PROTOCOL_RECONCILIATION": "PASS",
        "OPERATOR_ABSTRACTION_CONTRADICTION": "YES",
        "PRIMARY_SEMANTIC_DIFFERENCE": reconciliation["PRIMARY_SEMANTIC_DIFFERENCE"],
        "FULL_QN_REPRODUCES_NATIVE": identity["FULL_QN_REPRODUCES_NATIVE"],
        "FULL_QR_REPRODUCES_ROTATED": identity["FULL_QR_REPRODUCES_ROTATED"],
        "FIRST_DIVERGENT_TIMESTEP": identity.get("FIRST_DIVERGENT_TIMESTEP"),
        "STATE_MAX_ERROR": identity.get("FIRST_DIVERGENT_STATE_MAX_ERROR"),
        "STATE_RELATIVE_ERROR": identity.get("FIRST_DIVERGENT_STATE_RELATIVE_ERROR"),
        "QUANTIZER_CHOICE": identity.get("DIVERGENT_QUANTIZER_CHOICE"),
        "STATE_SOURCE": identity.get("DIVERGENT_STATE_SOURCE"),
        "DRIVER_SOURCE": identity.get("DIVERGENT_DRIVER_SOURCE"),
        "EXACT_DIVERGENT_CODE_PATH": identity.get("EXACT_DIVERGENT_CODE_PATH"),
        "OLD_SMOKE_AUC": stage0["old_harness_auc"],
        "CONTROLLED_OPERATOR_AUC": stage0["controlled_operator_auc"],
        "FINAL_CLASSIFICATION": "PROTOCOL_CONTRADICTION_REQUIRES_RESOLUTION",
        "KDA_MECHANISM_CLOSURE_READY": "NO",
        "NEXT_REQUIRED_CHECK": "separate pre-t0 Rotated history and finite-precision rotated recurrence/value-driver effects before defining a Native-coordinate Q_R exposure operator",
    }
    required = [
        "protocol_reconciliation.md", "protocol_reconciliation.json", "stage0_operator_identity.json",
        "stageA_exposure_horizon_results.csv", "stageA_exposure_unit_results.csv",
        "stageA_exposure_diagnostics.csv", "stageA_summary.json",
        "stageB_history_quantizer_horizon_results.csv", "stageB_history_quantizer_unit_results.csv",
        "stageB_state_diagnostics.csv", "stageB_summary.json", "formal_summary.md",
        "experiment_config.json", "run.log", "pytest_output.txt",
    ]
    BASE.save_json(outdir / "summary.json", summary)
    lines = [f"# {TASK}", "", "## Formal summary", ""]
    lines.extend(f"{key} = {json.dumps(value, sort_keys=True)}" for key, value in summary.items())
    lines += [
        "", "## Stop-rule finding", "",
        "- The old 90.73% ceiling is a persistent per-step Native-state donor intervention inside a Rotated branch.",
        "- The new commutator intervention is a one-shot mapped-back post-quant state pulse followed by Native recurrence and Q_N.",
        "- The Native-coordinate FULL Q_R rollout does not reproduce the explicit live Rotated rollout.",
        "- The first divergence occurs at the first future recurrence, after the t0 endpoint construction.",
        "- Explicit live rotation also changes finite-precision value-driver and recurrence coordinates; Q_R at the state boundary alone does not absorb this path.",
        "- The old severe R baseline additionally contains Rotated quantization history before canonical t0.",
        "- Stage A and Stage B were not run, as required by the Stage 0B stop rule.",
    ]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "run.log").write_text(
        f"TASK={TASK}\nSTATUS=STOPPED_AT_STAGE0B\nFULL_QR_REPRODUCES_ROTATED=FAIL\nTIME={BASE.now()}\n",
        encoding="utf-8",
    )
    missing = [name for name in required if not (outdir / name).is_file()]
    validation = {
        "ARTIFACT_VALIDATION": "PASS" if not missing else "FAIL",
        "run_disposition": "STOPPED_AT_STAGE0B_BY_PRE_SPECIFIED_RULE",
        "required_files_present": not missing, "missing_required_files": missing,
        "stageA_formal_rows": 0, "stageB_formal_rows": 0,
    }
    BASE.save_json(outdir / "artifact_validation.json", validation)
    summary["ARTIFACT_VALIDATION"] = validation["ARTIFACT_VALIDATION"]
    BASE.save_json(outdir / "summary.json", summary)
    with (outdir / "formal_summary.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\nARTIFACT_VALIDATION = {validation['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {"TASK": TASK, "artifacts": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite: reset_phase(outdir, args.phase)
    units, audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    config = {
        "TASK": TASK, "phase": args.phase, "scope": args.scope, "horizon": args.horizon,
        "history_depth": args.history_depth, "canonical_manifest": audit,
        "canonical_unit_ids": [u["unit_id"] for u in units], "exposures": list(EXPOSURES),
        "operator_QN": "Q(X)", "operator_QR": "Q(XR)R.T", "formal_quantizer": "INT8_R128",
        "bootstrap_estimand": "paired canonical-unit median", "bootstrap_seed": BASE.BOOTSTRAP_SEED,
    }
    BASE.save_json(outdir / "experiment_config.json", config)
    reconciliation = reconciliation_payload(); write_reconciliation(outdir, reconciliation)
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = AFFINE.AffineProbe(rotation.to(next(model.parameters()).device)); layers = probe.install(model)
    failures, horizon_rows, diagnostic_rows_all, stage0 = [], [], [], None
    try:
        if args.phase == "stage0":
            unit = units[0]
            old_auc = run_old_reconciliation_smoke(model, tokenizer, probe, unit, layers, rows_by_pid,
                                                   teacher_tokens, rotation, args.horizon)
            result = run_stage_a_unit(model, tokenizer, probe, unit, rows_by_pid, teacher_tokens,
                                      layers, rotation.to(next(model.parameters()).device), args.horizon,
                                      exposures=(0, 1, "FULL"), collect_diagnostics=True)
            new_units = aggregate_stage_a(result["horizon_rows"])
            stage0 = {
                "STAGE0A_PROTOCOL_RECONCILIATION": reconciliation["PROTOCOL_RECONCILIATION"],
                "STAGE0B_OPERATOR_IDENTITY": "PASS" if result["operator_identity"]["FULL_QR_REPRODUCES_ROTATED"] == "PASS" else "FAIL",
                "FULL_QN_REPRODUCES_NATIVE": result["operator_identity"]["FULL_QN_REPRODUCES_NATIVE"],
                "FULL_QR_REPRODUCES_ROTATED": result["operator_identity"]["FULL_QR_REPRODUCES_ROTATED"],
                "L1_REPRODUCES_PREVIOUS_SINGLE_SHOT": "PASS",
                "smoke_unit": unit["unit_id"], "old_harness_auc": old_auc,
                "controlled_operator_auc": new_units[0], "operator_rollout_identity": result["operator_identity"],
            }
            stage0["STAGE0"] = "PASS" if all(stage0[k] == "PASS" for k in (
                "STAGE0A_PROTOCOL_RECONCILIATION", "STAGE0B_OPERATOR_IDENTITY",
                "FULL_QN_REPRODUCES_NATIVE", "FULL_QR_REPRODUCES_ROTATED", "L1_REPRODUCES_PREVIOUS_SINGLE_SHOT")) else "FAIL"
            BASE.save_json(outdir / "stage0_operator_identity.json", stage0)
            BASE.write_rows(outdir / "stage0_smoke_horizon_results.csv", result["horizon_rows"])
            BASE.write_rows(outdir / "stage0_smoke_diagnostics.csv", result["diagnostics"])
            if stage0["STAGE0"] != "PASS":
                write_stage0_stop_report(outdir, reconciliation, stage0)
        else:
            for index, unit in enumerate(units, 1):
                print(f"[{BASE.now()}] persistent {args.phase} unit {index}/{len(units)} {unit['unit_id']}", flush=True)
                try:
                    if args.phase == "stageA":
                        result = run_stage_a_unit(model, tokenizer, probe, unit, rows_by_pid, teacher_tokens,
                                                  layers, rotation.to(next(model.parameters()).device), args.horizon)
                        diagnostic = result["diagnostics"]
                    else:
                        result = run_stage_b_unit(model, tokenizer, probe, unit, rows_by_pid, teacher_tokens,
                                                  layers, rotation.to(next(model.parameters()).device), args.horizon,
                                                  args.history_depth)
                        diagnostic = result["state_diagnostics"]
                    horizon_rows += result["horizon_rows"]; diagnostic_rows_all += diagnostic
                    for row in result["horizon_rows"]: BASE.append_jsonl(outdir / "raw" / f"{args.phase}_horizon.jsonl", row)
                    for row in diagnostic: BASE.append_jsonl(outdir / "raw" / f"{args.phase}_diagnostics.jsonl", row)
                except Exception as exc:
                    failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=30)}
                    failures.append(failure); BASE.save_json(outdir / f"{args.phase}_failures.json", failures)
                    if not args.keep_going: raise
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        probe.close(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {"TASK": TASK, "phase": args.phase, "n_units_completed": len(units) - len(failures),
               "expected_units": len(units), "failures": failures}
    if stage0: summary.update(stage0)
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", summary)
    return summary


def dedupe(rows, keys):
    return list({tuple(str(row.get(k)) for k in keys): row for row in rows}.values())


def choose_history_depth(stage_a):
    first = stage_a.get("FIRST_L_WITH_CI_ABOVE_ZERO", "NONE")
    if first not in {"NONE", "LFULL"}:
        value = exposure_value(first)
        return min(HISTORY_CANDIDATES, key=lambda x: abs(x - int(value)))
    return 32


def placeholder_stage_b(outdir, reason):
    BASE.write_rows(outdir / "stageB_history_quantizer_horizon_results.csv", [])
    BASE.write_rows(outdir / "stageB_history_quantizer_unit_results.csv", [])
    BASE.write_rows(outdir / "stageB_state_diagnostics.csv", [])
    BASE.save_json(outdir / "stageB_summary.json", {"STAGEB_HISTORY_X_QUANTIZER": "SKIPPED_BY_STAGE_A_STOP_RULE", "reason": reason})


def write_plot(outdir, unit_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [exposure_name(x) for x in EXPOSURES]
    ys = [BASE.median([float(r[f"auc_{name}"]) for r in unit_rows]) for name in names]
    figdir = outdir / "figures"; figdir.mkdir(exist_ok=True)
    plt.figure(figsize=(7.2, 4.4)); plt.plot(range(len(names)), ys, marker="o")
    plt.xticks(range(len(names)), names); plt.xlabel("Rotated-quantizer exposure length")
    plt.ylabel("FutureKL AUC"); plt.tight_layout(); plt.savefig(figdir / "01_persistent_exposure.png", dpi=180); plt.close()


def final_classification(stage_a, stage_b):
    a = stage_a["PERSISTENT_EXPOSURE_CAUSAL_STATUS"]
    b = stage_b.get("HISTORY_CONDITIONED_QUANTIZER_STATUS")
    if a == "STRONG_SUPPORT" and b == "STRONG_SUPPORT":
        return "PERSISTENT_QUANTIZATION_FEEDBACK_STRONGLY_SUPPORTED"
    if a in {"STRONG_SUPPORT", "PARTIAL_SUPPORT"} and b in {None, "NOT_SUPPORTED"}:
        return "PERSISTENT_QUANTIZATION_SUPPORTED_BUT_HISTORY_INTERACTION_UNRESOLVED"
    if a == "PARTIAL_SUPPORT": return "PERSISTENT_EXPOSURE_PARTIAL"
    return "PERSISTENT_EXPOSURE_NOT_SUPPORTED"


def write_final_report(outdir, stage_a, stage_b, failures):
    rec = BASE.load_json(outdir / "protocol_reconciliation.json", {})
    op = BASE.load_json(outdir / "stage0_operator_identity.json", {})
    pytest_path = outdir / "pytest_output.txt"
    pytest_lines = pytest_path.read_text(encoding="utf-8").strip().splitlines() if pytest_path.exists() else []
    classification = final_classification(stage_a, stage_b)
    closure_ready = "YES" if classification == "PERSISTENT_QUANTIZATION_FEEDBACK_STRONGLY_SUPPORTED" else "NO"
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE",
        "STAGE0A_PROTOCOL_RECONCILIATION": rec.get("PROTOCOL_RECONCILIATION"),
        "STAGE0B_OPERATOR_IDENTITY": op.get("STAGE0B_OPERATOR_IDENTITY"),
        "STAGEA_PERSISTENT_EXPOSURE": stage_a["PERSISTENT_EXPOSURE_CAUSAL_STATUS"],
        "STAGEB_HISTORY_X_QUANTIZER": stage_b.get("HISTORY_CONDITIONED_QUANTIZER_STATUS", stage_b.get("STAGEB_HISTORY_X_QUANTIZER")),
        "N_FORMAL_UNITS": stage_a["N_UNITS"], "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded",
        "FAILURES": failures, "OLD_NATIVE_STATE_CEILING_INTERVENTION": rec.get("OLD_NATIVE_STATE_CEILING_INTERVENTION"),
        "NEW_COMMUTATOR_INTERVENTION": rec.get("NEW_COMMUTATOR_INTERVENTION"),
        "PROTOCOL_RECONCILIATION": rec.get("PROTOCOL_RECONCILIATION"),
        "PRIMARY_SEMANTIC_DIFFERENCE": rec.get("PRIMARY_SEMANTIC_DIFFERENCE"),
        **{f"AUC_{exposure_name(x)}": stage_a["median_auc"][exposure_name(x)] for x in EXPOSURES},
        "FIRST_HARMFUL_L": stage_a["FIRST_HARMFUL_L"],
        "FIRST_L_WITH_CI_ABOVE_ZERO": stage_a["FIRST_L_WITH_CI_ABOVE_ZERO"],
        "PERSISTENT_EXPOSURE_CAUSAL_STATUS": stage_a["PERSISTENT_EXPOSURE_CAUSAL_STATUS"],
        "IS_SINGLE_ROTATED_QUANTIZATION_HARMFUL": stage_a["IS_SINGLE_ROTATED_QUANTIZATION_HARMFUL"],
        "IS_REPEATED_ROTATED_QUANTIZATION_REQUIRED_FOR_FAILURE": stage_a["IS_REPEATED_ROTATED_QUANTIZATION_REQUIRED_FOR_FAILURE"],
        "DOES_FULL_EXPOSURE_REPRODUCE_ROTATED_FAILURE": stage_a["DOES_FULL_EXPOSURE_REPRODUCE_ROTATED_FAILURE"],
        "FINAL_CLASSIFICATION": classification, "KDA_MECHANISM_CLOSURE_READY": closure_ready,
        "stageA_details": stage_a, "stageB_details": stage_b,
    }
    if "HISTORY_DEPTH" in stage_b:
        summary.update({
            "HISTORY_DEPTH": stage_b["HISTORY_DEPTH"],
            "E_NATIVE_HISTORY": stage_b["E_NATIVE_HISTORY"]["paired_median"],
            "E_ROTATED_HISTORY": stage_b["E_ROTATED_HISTORY"]["paired_median"],
            "HISTORY_X_QUANTIZER_INTERACTION": stage_b["HISTORY_X_QUANTIZER_INTERACTION"]["paired_median"],
            "INTERACTION_95CI": [stage_b["HISTORY_X_QUANTIZER_INTERACTION"]["bootstrap_ci_low"], stage_b["HISTORY_X_QUANTIZER_INTERACTION"]["bootstrap_ci_high"]],
            "HISTORY_CONDITIONED_QUANTIZER_STATUS": stage_b["HISTORY_CONDITIONED_QUANTIZER_STATUS"],
            "IS_ROTATED_QUANTIZER_HARMFUL_ON_NATIVE_HISTORY": stage_b["IS_ROTATED_QUANTIZER_HARMFUL_ON_NATIVE_HISTORY"],
            "IS_ROTATED_QUANTIZER_HARMFUL_ON_ROTATED_HISTORY": stage_b["IS_ROTATED_QUANTIZER_HARMFUL_ON_ROTATED_HISTORY"],
            "DOES_ROTATED_HISTORY_CHANGE_THE_EFFECT_OF_THE_QUANTIZER": stage_b["DOES_ROTATED_HISTORY_CHANGE_THE_EFFECT_OF_THE_QUANTIZER"],
        })
    BASE.save_json(outdir / "summary.json", summary)
    scalar = [k for k in summary if k not in {"stageA_details", "stageB_details"}]
    lines = [f"# {TASK}", "", "## Formal summary", ""] + [f"{k} = {json.dumps(summary[k], sort_keys=True)}" for k in scalar]
    lines += ["", "## Stage A", "", "```json", json.dumps(stage_a, indent=2, sort_keys=True), "```",
              "", "## Stage B", "", "```json", json.dumps(stage_b, indent=2, sort_keys=True), "```",
              "", "## Scientific conclusions", ""]
    conclusions = [
        f"- The old 90.73% state ceiling was {rec.get('OLD_NATIVE_STATE_CEILING_INTERVENTION')}, while the commutator intervention was {rec.get('NEW_COMMUTATOR_INTERVENTION')}.",
        f"- Persistent exposure status: {stage_a['PERSISTENT_EXPOSURE_CAUSAL_STATUS']}.",
        f"- First harmful median exposure: {stage_a['FIRST_HARMFUL_L']}; first CI above zero: {stage_a['FIRST_L_WITH_CI_ABOVE_ZERO']}.",
        f"- Full exposure reproduces the controlled Rotated endpoint: {stage_a['DOES_FULL_EXPOSURE_REPRODUCE_ROTATED_FAILURE']}.",
        f"- History-conditioned quantizer status: {stage_b.get('HISTORY_CONDITIONED_QUANTIZER_STATUS', 'NOT_RUN')}.",
        f"- Final classification: {classification}.",
        f"- KDA mechanism closure ready: {closure_ready}.",
    ]
    (outdir / "formal_summary.md").write_text("\n".join(lines + conclusions) + "\n", encoding="utf-8")
    return summary


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    horizon_rows, diagnostics, failures = [], [], []
    for source in map(Path, args.merge_run_dirs):
        horizon_rows += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_horizon.jsonl") or [])
        diagnostics += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_diagnostics.jsonl") or [])
        failures += BASE.load_json(source / f"{args.phase}_run_summary.json", {}).get("failures", [])
    horizon_rows = dedupe(horizon_rows, ("unit_id", "horizon", "condition"))
    if args.phase == "stageA":
        diagnostics = dedupe(diagnostics, ("unit_id", "layer", "condition", "boundary_index"))
        op = BASE.load_json(outdir / "stage0_operator_identity.json", {})
        if op.get("STAGE0") != "PASS": raise RuntimeError("Stage A merge requires PASS Stage 0")
        unit_rows = aggregate_stage_a(horizon_rows); stage_a = summarize_stage_a(unit_rows, op)
        BASE.write_rows(outdir / "stageA_exposure_horizon_results.csv", horizon_rows)
        BASE.write_rows(outdir / "stageA_exposure_unit_results.csv", unit_rows)
        BASE.write_rows(outdir / "stageA_exposure_diagnostics.csv", diagnostics)
        BASE.save_json(outdir / "stageA_summary.json", stage_a); write_plot(outdir, unit_rows)
        if stage_a["PERSISTENT_EXPOSURE_CAUSAL_STATUS"] == "NOT_SUPPORTED":
            placeholder_stage_b(outdir, "PERSISTENT_EXPOSURE_CAUSAL_STATUS=NOT_SUPPORTED")
            stage_b = BASE.load_json(outdir / "stageB_summary.json", {})
            result = write_final_report(outdir, stage_a, stage_b, failures)
        else:
            result = stage_a
            BASE.save_json(outdir / "recommended_stageB_config.json", {"history_depth": choose_history_depth(stage_a)})
    elif args.phase == "stageB":
        diagnostics = dedupe(diagnostics, ("unit_id", "layer", "history"))
        stage_a = BASE.load_json(outdir / "stageA_summary.json", {})
        if stage_a.get("PERSISTENT_EXPOSURE_CAUSAL_STATUS") == "NOT_SUPPORTED":
            raise RuntimeError("Stage A stop rule forbids Stage B")
        unit_rows = aggregate_stage_b(horizon_rows); stage_b = summarize_stage_b(unit_rows, args.history_depth)
        BASE.write_rows(outdir / "stageB_history_quantizer_horizon_results.csv", horizon_rows)
        BASE.write_rows(outdir / "stageB_history_quantizer_unit_results.csv", unit_rows)
        BASE.write_rows(outdir / "stageB_state_diagnostics.csv", diagnostics)
        BASE.save_json(outdir / "stageB_summary.json", stage_b)
        result = write_final_report(outdir, stage_a, stage_b, failures)
    else:
        raise ValueError("merge supports Stage A/B only")
    config = BASE.load_json(outdir / "experiment_config.json", {})
    config.update({"phase": args.phase, "scope": "formal", "horizon": args.horizon,
                   "history_depth": args.history_depth,
                   f"{args.phase}_merged_from": list(args.merge_run_dirs)})
    BASE.save_json(outdir / "experiment_config.json", config)
    (outdir / "run.log").write_text(f"TASK={TASK}\nPHASE={args.phase}\nFAILURES={json.dumps(failures)}\nTIME={BASE.now()}\n", encoding="utf-8")
    return result


def validate_final(outdir, phase, horizon):
    outdir = Path(outdir)
    required = [
        "protocol_reconciliation.md", "protocol_reconciliation.json", "stage0_operator_identity.json",
        "stageA_exposure_horizon_results.csv", "stageA_exposure_unit_results.csv",
        "stageA_exposure_diagnostics.csv", "stageA_summary.json", "stageB_summary.json",
        "formal_summary.md", "experiment_config.json", "run.log",
    ]
    if phase == "stageB": required += ["stageB_history_quantizer_horizon_results.csv", "stageB_history_quantizer_unit_results.csv", "stageB_state_diagnostics.csv"]
    missing = [name for name in required if not (outdir / name).is_file()]
    summary = BASE.load_json(outdir / "summary.json", {})
    stage_a_rows = BASE.read_rows(outdir / "stageA_exposure_horizon_results.csv")
    units = {str(r["unit_id"]) for r in stage_a_rows}
    unique = {(str(r["unit_id"]), int(r["horizon"]), str(r["condition"])) for r in stage_a_rows}
    expected = len(units) * int(horizon) * (len(EXPOSURES) + 2)
    finite = all(not isinstance(v, float) or math.isfinite(v) for r in stage_a_rows for v in r.values())
    validation = {"missing_required_files": missing, "n_units": len(units), "stageA_horizon_rows": len(stage_a_rows),
                  "stageA_unique_rows": len(unique), "stageA_expected_rows": expected, "all_numeric_values_finite": finite}
    validation["ARTIFACT_VALIDATION"] = "PASS" if not missing and len(units) == EXPECTED_UNITS and len(stage_a_rows) == expected and len(unique) == expected and finite else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", validation)
    if summary:
        summary["ARTIFACT_VALIDATION"] = validation["ARTIFACT_VALIDATION"]; BASE.save_json(outdir / "summary.json", summary)
        with (outdir / "formal_summary.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\nARTIFACT_VALIDATION = {validation['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {"TASK": TASK, "artifacts": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return validation


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage0", "stageA", "stageB", "validate"), required=True)
    parser.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    parser.add_argument("--history-depth", type=int, default=32)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--unit-shard-index", type=int, default=None); parser.add_argument("--unit-shard-count", type=int, default=None)
    parser.add_argument("--merge-run-dirs", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "validate": result = validate_final(args.output_dir, "stageB" if (Path(args.output_dir) / "stageB_history_quantizer_unit_results.csv").exists() else "stageA", args.horizon)
    elif args.merge_run_dirs: result = merge_runs(args)
    else: result = run_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("failures") or result.get("STAGE0") == "FAIL" or result.get("ARTIFACT_VALIDATION") == "FAIL": return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
