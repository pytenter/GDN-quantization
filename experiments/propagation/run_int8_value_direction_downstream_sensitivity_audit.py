#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
REPO = ROOT / "GDN-quantization"
PREV_TRANSFER_RUN = ROOT / "runs" / "gdn_int8_value_geometry_transfer_validity_and_retest_v1"
RUN_DIR = ROOT / "runs" / "gdn_int8_value_direction_downstream_sensitivity_audit_v1"
TASK = "GDN_INT8_VALUE_DIRECTION_DOWNSTREAM_SENSITIVITY_AUDIT_V1"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_geometry_transfer_validity_and_retest as vt

PRIMARY_R = vf.PRIMARY_R
PRIMARY_C = vf.PRIMARY_C
EPS = 1e-12
HORIZON = 128
TOPK_TOKENS = 3
NORM_FLOOR = 1e-10


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows):
    path = RUN_DIR / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def read_rows():
    return vf.read_prev_stagea_rows()


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def config():
    prev = load_json(PREV_TRANSFER_RUN / "final_summary.json")
    pilot = load_json(PREV_TRANSFER_RUN / "c2_retest_pilot_summary.json")
    cfg = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "previous_transfer_run": str(PREV_TRANSFER_RUN),
        "previous_transfer_summary": {
            "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": prev["VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL"],
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": prev["VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER"],
        },
        "pilot_units": pilot["TRANSFER_CONSTRUCTION_USED"] and load_json(PREV_TRANSFER_RUN / "config.json")["same_3_pilot_units"],
        "hook": {
            "location": "Qwen3.5 linear_attn recurrent core output immediately after recurrent-state readout and before gated RMSNorm and out_proj",
            "tensor_shape": "[B, T=1, H=32, V=128] per GDN layer",
            "batch_dimension": 0,
            "token_dimension": 1,
            "head_dimension": 2,
            "value_dimension": 3,
            "layer_scope": "all canonical GDN layers",
            "head_scope": "all heads, preserving per-head Value dimension",
            "relative_to_projection": "before head concatenation/out_proj",
        },
        "token_selection": "top-3 future token offsets by pooled original readout energy ||z_R||^2 + ||z_C||^2; no KL/sensitivity used",
        "epsilon": "sqrt((||z_R||^2 + ||z_C||^2)/2) from original R/C only; reused for transfer directions",
        "primary_metric": "teacher-forced mean KL over future horizon after one-shot core-output perturbation, symmetric average over +/- epsilon",
        "horizon": HORIZON,
        "norm_floor": NORM_FLOOR,
    }
    save_json("config.json", cfg)
    return cfg


def install_core_perturb_hook(torch, model, state):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    orig_rule = qmod.torch_recurrent_gated_delta_rule
    handles = []
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers

    def wrapped_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        out, last_state = orig_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, **kwargs)
        layer = state.get("current_layer")
        if layer is not None and state.get("current_t") == state.get("target_t") and int(layer) in state.get("deltas", {}):
            out = out + state["deltas"][int(layer)].to(device=out.device, dtype=out.dtype)
        return out, last_state

    qmod.torch_recurrent_gated_delta_rule = wrapped_rule

    for idx, layer in enumerate(layers):
        if not hasattr(layer, "linear_attn"):
            continue
        mod = layer.linear_attn

        def make_pre(i):
            def pre(_module, _inputs):
                state["current_layer"] = i
            return pre

        def post(_module, _inputs, _output):
            state["current_layer"] = None

        handles.append(mod.register_forward_pre_hook(make_pre(idx)))
        handles.append(mod.register_forward_hook(post))

    def close():
        qmod.torch_recurrent_gated_delta_rule = orig_rule
        for h in handles:
            h.remove()

    return close


def batch_logits_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    return float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())


def eval_direction_token(torch, model, tokenizer, e2e, pm, t0, token_offset, deltas_by_condition):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    labels = ["FP"] + list(deltas_by_condition.keys())
    batch = len(labels)
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device).expand(batch, -1).contiguous()
    mask = enc.get("attention_mask")
    mask = mask.to(device).expand(batch, -1).contiguous() if mask is not None else None
    past = None
    target_t = int(t0) + int(token_offset)
    last_t = min(target_t + HORIZON - 1, len(cont))
    state = {"current_t": None, "target_t": target_t, "deltas": {}}
    close = install_core_perturb_hook(torch, model, state)
    kls = {label: [] for label in labels[1:]}
    try:
        with torch.inference_mode():
            for t in range(last_t + 1):
                state["current_t"] = t
                if t == target_t:
                    for layer in frozen.GDN_LAYERS:
                        pieces = [torch.zeros_like(next(iter(deltas_by_condition.values()))[layer])]
                        for label in labels[1:]:
                            pieces.append(deltas_by_condition[label][layer])
                        state["deltas"][int(layer)] = torch.cat(pieces, dim=0)
                else:
                    state["deltas"] = {}
                out = p1.feed_step(torch, model, ids, mask, past)
                past = out.past_key_values
                if t >= target_t:
                    for i, label in enumerate(labels[1:], start=1):
                        kls[label].append(batch_logits_metrics(torch, out.logits[0:1], out.logits[i:i + 1]))
                if t < len(cont):
                    nxt = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device).expand(batch, 1).contiguous()
                    ids = nxt
                    mask = None
    finally:
        close()
    return {label: mean(vals) for label, vals in kls.items()}


def build_base_and_transfer_inj(torch, model, tokenizer, e2e, row, pmap):
    pm = pmap[row["prompt_id"]]
    _fp, _ids, _cont, collector, inj, _meta, _fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
    collector["close"]()
    r_to_cv, c_to_rv, _audit = vt.svd_value_transfer(torch, inj)
    transfer = {PRIMARY_R: r_to_cv, PRIMARY_C: c_to_rv}
    return pm, inj, transfer


def scale_direction(torch, tensors, scale):
    return {layer: (t * scale).detach().float() for layer, t in tensors.items()}


def combine_direction_norm(torch, tensors):
    return math.sqrt(sum(float(torch.sum(t.detach().double() * t.detach().double()).item()) for t in tensors.values()))


def compute_direction_trajectory(torch, model, tokenizer, e2e, row, pmap, include_transfer):
    pm, inj, transfer_inj = build_base_and_transfer_inj(torch, model, tokenizer, e2e, row, pmap)
    fp_past, ids, cont, collector, _base_inj, _meta, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
    try:
        cond_names = ["R", "C"] + (["R_TO_CV", "C_TO_RV"] if include_transfer else [])
        cond_inj = {
            "R": inj[PRIMARY_R],
            "C": inj[PRIMARY_C],
            "R_TO_CV": transfer_inj[PRIMARY_R],
            "C_TO_RV": transfer_inj[PRIMARY_C],
        }
        cond_states = {
            name: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[name][layer] for layer in frozen.GDN_LAYERS}
            for name in cond_names
        }
        per_offset = []
        valid_h = min(HORIZON, len(cont) - int(row["t0"]))
        past = fp_past
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                z = {name: {} for name in cond_names}
                for layer in frozen.GDN_LAYERS:
                    rec = records[layer]
                    out = vf.fg.corrected_replay_layer_conditions(
                        torch, model, layer, rec, fp_states[layer],
                        [cond_states[name][layer] for name in cond_names],
                    )
                    for ci, name in enumerate(cond_names):
                        cond_states[name][layer] = out["next_state"][ci:ci + 1]
                        z[name][layer] = out["core_readout_error"][ci:ci + 1].detach().float()
                    fp_states[layer] = rec["final_state"].detach().float()
                norms = {name: combine_direction_norm(torch, z[name]) for name in cond_names}
                eps = math.sqrt((norms["R"] ** 2 + norms["C"] ** 2) / 2.0)
                dirs = {}
                for name in cond_names:
                    dirs[name] = scale_direction(torch, z[name], 1.0 / (norms[name] + EPS))
                per_offset.append({
                    "offset": off,
                    "z_norms": norms,
                    "epsilon": eps,
                    "selection_score": norms["R"] ** 2 + norms["C"] ** 2,
                    "directions": dirs,
                })
        selected = []
        for item in sorted(per_offset, key=lambda x: x["selection_score"], reverse=True):
            if item["z_norms"]["R"] > NORM_FLOOR and item["z_norms"]["C"] > NORM_FLOOR:
                selected.append(item)
            if len(selected) == TOPK_TOKENS:
                break
        return pm, selected
    finally:
        collector["close"]()


def stage_d0():
    cfg = config()
    prev_sum = load_json(PREV_TRANSFER_RUN / "final_summary.json")
    import torch
    torch.manual_seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    row = vf.read_prev_stagea_rows()[0]
    pmap = prompt_map()
    pm, selected = compute_direction_trajectory(torch, model, tokenizer, e2e, row, pmap, include_transfer=False)
    item = selected[0]
    zero = {layer: torch.zeros_like(t) for layer, t in item["directions"]["R"].items()}
    empty = eval_direction_token(torch, model, tokenizer, e2e, pm, row["t0"], item["offset"], {"ZERO": zero})
    identity = {
        "STAGE_D0": "PASS" if prev_sum["VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER"] == "FORMAL_SUPPORTED" and abs(empty["ZERO"]) <= 1e-12 else "FAIL",
        "clean_KL_zero_perturb": empty["ZERO"],
        "hook_semantics_file": "stageD0_hook_semantics.json",
        "direction_norm_smoke": combine_direction_norm(torch, item["directions"]["R"]),
        "epsilon_smoke": item["epsilon"],
        "selected_offset_smoke": item["offset"],
    }
    save_json("stageD0_hook_semantics.json", cfg["hook"])
    save_json("stageD0_identity.json", identity)
    return identity


def run_units(units, include_transfer, prefix):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = {r["unit"]: r for r in vf.read_prev_stagea_rows()}
    pmap = prompt_map()
    per_run = []
    per_token = []
    per_unit = []
    selected_rows = []
    secondary = []
    for unit in units:
        row = rows[unit]
        print(f"[{now()}] {prefix} {unit}", flush=True)
        pm, selected = compute_direction_trajectory(torch, model, tokenizer, e2e, row, pmap, include_transfer=include_transfer)
        token_results = []
        for item in selected:
            offset = item["offset"]
            eps = item["epsilon"]
            selected_rows.append({
                "unit_id": unit,
                "prompt_id": row["prompt_id"],
                "t0": row["t0"],
                "token_offset": offset,
                "probe_token": int(row["t0"]) + int(offset),
                "selection_score": item["selection_score"],
                "z_norm_original_R": item["z_norms"]["R"],
                "z_norm_original_C": item["z_norms"]["C"],
                "epsilon": eps,
                "direction_norm_R": combine_direction_norm(torch, item["directions"]["R"]),
                "direction_norm_C": combine_direction_norm(torch, item["directions"]["C"]),
            })
            labels = {}
            for cond in ["R", "C"] + (["R_TO_CV", "C_TO_RV"] if include_transfer else []):
                labels[f"{cond}_PLUS"] = scale_direction(torch, item["directions"][cond], eps)
                labels[f"{cond}_MINUS"] = scale_direction(torch, item["directions"][cond], -eps)
            kls = eval_direction_token(torch, model, tokenizer, e2e, pm, row["t0"], offset, labels)
            def d(cond):
                return 0.5 * (kls[f"{cond}_PLUS"] + kls[f"{cond}_MINUS"])
            dr, dc = d("R"), d("C")
            tok = {
                "unit_id": unit,
                "token_offset": offset,
                "epsilon": eps,
                "D_R": dr,
                "D_C": dc,
                "D_R_over_D_C": ratio(dr, dc),
                "D_R_minus_D_C": dr - dc,
                "KL_R_plus": kls["R_PLUS"],
                "KL_R_minus": kls["R_MINUS"],
                "KL_C_plus": kls["C_PLUS"],
                "KL_C_minus": kls["C_MINUS"],
                "sign_asymmetry_R": abs(kls["R_PLUS"] - kls["R_MINUS"]) / (dr + EPS),
                "sign_asymmetry_C": abs(kls["C_PLUS"] - kls["C_MINUS"]) / (dc + EPS),
                "R_DIRECTION_MORE_SENSITIVE": dr > dc,
            }
            if include_transfer:
                drc, crr = d("R_TO_CV"), d("C_TO_RV")
                tok.update({
                    "D_R_to_CV": drc,
                    "D_C_to_RV": crr,
                    "R_sensitivity_rescue": dr - drc,
                    "C_sensitivity_gain": crr - dc,
                    "BIDIRECTIONAL_SENSITIVITY_TRANSFER": (dr - drc) > 0 and (crr - dc) > 0,
                    "KL_R_to_CV_plus": kls["R_TO_CV_PLUS"],
                    "KL_R_to_CV_minus": kls["R_TO_CV_MINUS"],
                    "KL_C_to_RV_plus": kls["C_TO_RV_PLUS"],
                    "KL_C_to_RV_minus": kls["C_TO_RV_MINUS"],
                })
            per_token.append(tok)
            token_results.append(tok)
            for cond_name, label_name in [("R_DIRECTION", "R"), ("C_DIRECTION", "C"), ("R_TO_CV_DIRECTION", "R_TO_CV"), ("C_TO_RV_DIRECTION", "C_TO_RV")]:
                if label_name not in item["directions"]:
                    continue
                for sign in ["PLUS", "MINUS"]:
                    per_run.append({
                        "unit_id": unit,
                        "prompt_id": row["prompt_id"],
                        "layer": "all_gdn_layers",
                        "head": "all_heads",
                        "t0": row["t0"],
                        "probe_token": int(row["t0"]) + int(offset),
                        "token_offset": offset,
                        "condition": cond_name,
                        "sign": sign,
                        "z_norm_original_R": item["z_norms"]["R"],
                        "z_norm_original_C": item["z_norms"]["C"],
                        "epsilon": eps,
                        "direction_norm": combine_direction_norm(torch, item["directions"][label_name]),
                        "KL_metric": kls[f"{label_name}_{sign}"],
                        "run_valid": True,
                    })
        unit_row = {
            "unit_id": unit,
            "selected_tokens": ",".join(str(x["token_offset"]) for x in token_results),
            "median_D_R": med([x["D_R"] for x in token_results]),
            "median_D_C": med([x["D_C"] for x in token_results]),
            "median_directional_ratio": med([x["D_R_over_D_C"] for x in token_results]),
            "UNIT_R_DIRECTION_MORE_SENSITIVE": med([x["D_R_minus_D_C"] for x in token_results]) > 0,
        }
        if include_transfer:
            unit_row.update({
                "median_D_R_to_CV": med([x["D_R_to_CV"] for x in token_results]),
                "median_D_C_to_RV": med([x["D_C_to_RV"] for x in token_results]),
                "R_SENSITIVITY_RESCUE": med([x["R_sensitivity_rescue"] for x in token_results]),
                "C_SENSITIVITY_GAIN": med([x["C_sensitivity_gain"] for x in token_results]),
                "BIDIRECTIONAL_SENSITIVITY_TRANSFER": med([x["R_sensitivity_rescue"] for x in token_results]) > 0 and med([x["C_sensitivity_gain"] for x in token_results]) > 0,
            })
        per_unit.append(unit_row)
        secondary.append({
            "unit_id": unit,
            "Oq_R_over_C": ratio(row["O_R"], row["O_C"]),
            "S_R_over_C": unit_row["median_directional_ratio"],
            "F_R_over_C": ratio(row["O_R"] * unit_row["median_D_R"], row["O_C"] * unit_row["median_D_C"]),
            "KL_R_over_C": ratio(row["KL_R"], row["KL_C"]),
        })
    write_csv(f"stage{prefix}_per_run.csv", per_run)
    write_csv(f"stage{prefix}_per_token.csv", per_token)
    write_csv(f"stage{prefix}_per_unit.csv", per_unit)
    write_csv("selected_tokens_per_unit.csv" if prefix == "D1_pilot" else f"selected_tokens_{prefix}.csv", selected_rows)
    if prefix == "D2_formal":
        write_csv("secondary_observability_sensitivity_analysis.csv", secondary)
    return per_unit, per_token


def summarize_d1(per_unit):
    cnt = sum(u["UNIT_R_DIRECTION_MORE_SENSITIVE"] for u in per_unit)
    summary = {
        "DIRECTIONAL_SENSITIVITY_PILOT": "PASS" if cnt >= 2 and len(per_unit) == 3 else "NOT_SUPPORTED",
        "R_DIRECTION_MORE_SENSITIVE_UNITS": cnt,
        "valid_units": len(per_unit),
        "median_directional_sensitivity_R_over_C": med([u["median_directional_ratio"] for u in per_unit]),
    }
    save_json("stageD1_summary.json", summary)
    return summary


def summarize_d2(per_unit):
    cnt = sum(u["UNIT_R_DIRECTION_MORE_SENSITIVE"] for u in per_unit)
    if cnt >= 8:
        sig = "FORMAL_SUPPORTED"
    elif cnt >= 6:
        sig = "PARTIAL_OR_INCONCLUSIVE"
    else:
        sig = "NOT_SUPPORTED"
    summary = {
        "D2_FORMAL_RUN": "YES",
        "valid_units": len(per_unit),
        "R_DIRECTION_MORE_SENSITIVE_UNITS": cnt,
        "median_directional_sensitivity_R_over_C": med([u["median_directional_ratio"] for u in per_unit]),
        "DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL": sig,
    }
    save_json("stageD2_summary.json", summary)
    return summary


def summarize_d3(per_unit):
    r_cnt = sum(u["R_SENSITIVITY_RESCUE"] > 0 for u in per_unit)
    c_cnt = sum(u["C_SENSITIVITY_GAIN"] > 0 for u in per_unit)
    b_cnt = sum(u["BIDIRECTIONAL_SENSITIVITY_TRANSFER"] for u in per_unit)
    if r_cnt >= 8 and c_cnt >= 8 and b_cnt >= 8:
        sig = "FORMAL_SUPPORTED"
    elif r_cnt >= 8 or c_cnt >= 8:
        sig = "ASYMMETRIC"
    else:
        sig = "NOT_SUPPORTED"
    summary = {
        "D3_RUN": "YES",
        "R_TO_CV_SENSITIVITY_RESCUE_UNITS": r_cnt,
        "C_TO_RV_SENSITIVITY_GAIN_UNITS": c_cnt,
        "BIDIRECTIONAL_SENSITIVITY_TRANSFER_UNITS": b_cnt,
        "TRANSFER_SENSITIVITY_MEDIATION_SIGNAL": sig,
    }
    save_json("stageD3_summary.json", summary)
    return summary


def final_report(d0, d1, d2, d3):
    obs = "INSUFFICIENT" if d2.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL") == "FORMAL_SUPPORTED" else "INCONCLUSIVE"
    if d2.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL") == "FORMAL_SUPPORTED" and d3.get("TRANSFER_SENSITIVITY_MEDIATION_SIGNAL") == "FORMAL_SUPPORTED":
        cls = "VALUE_GEOMETRY_HARMFULNESS_EXPLAINED_BY_DOWNSTREAM_DIRECTIONAL_SENSITIVITY"
        closure = "STRONG_YES_CANDIDATE"
    elif d2.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL") == "FORMAL_SUPPORTED":
        cls = "DOWNSTREAM_DIRECTIONAL_SENSITIVITY_SUPPORTED_TRANSFER_LINK_INCOMPLETE"
        closure = "YES_CANDIDATE"
    else:
        cls = "VALUE_GEOMETRY_TRANSFER_SUPPORTED_BUT_LOCAL_DIRECTIONAL_SENSITIVITY_NOT_SUPPORTED"
        closure = "YES_CANDIDATE"
    final = {
        "STAGE_D0": d0.get("STAGE_D0"),
        "DIRECTIONAL_SENSITIVITY_PILOT": d1.get("DIRECTIONAL_SENSITIVITY_PILOT", "NOT_RUN"),
        "D2_FORMAL_RUN": d2.get("D2_FORMAL_RUN", "NO"),
        "R_DIRECTION_MORE_SENSITIVE_UNITS": d2.get("R_DIRECTION_MORE_SENSITIVE_UNITS", d1.get("R_DIRECTION_MORE_SENSITIVE_UNITS", 0)),
        "DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL": d2.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL", "NOT_RUN"),
        "D3_RUN": d3.get("D3_RUN", "NO"),
        "R_TO_CV_SENSITIVITY_RESCUE_UNITS": d3.get("R_TO_CV_SENSITIVITY_RESCUE_UNITS", 0),
        "C_TO_RV_SENSITIVITY_GAIN_UNITS": d3.get("C_TO_RV_SENSITIVITY_GAIN_UNITS", 0),
        "BIDIRECTIONAL_SENSITIVITY_TRANSFER_UNITS": d3.get("BIDIRECTIONAL_SENSITIVITY_TRANSFER_UNITS", 0),
        "TRANSFER_SENSITIVITY_MEDIATION_SIGNAL": d3.get("TRANSFER_SENSITIVITY_MEDIATION_SIGNAL", "NOT_RUN"),
        "OBSERVABILITY_ONLY_EXPLANATION": obs,
        "FINAL_MECHANISM_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": closure,
        "METHOD_DESIGN_READY": "NO",
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
        "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "FORMAL_SUPPORTED",
        "output_dir": str(RUN_DIR),
        "git_commit_start": git_commit(),
    }
    save_json("final_summary.json", final)
    lines = [
        "# GDN INT8 Value Direction Downstream Sensitivity Audit V1",
        "## 1. Executive Summary\n\n" + json.dumps(final, indent=2, ensure_ascii=False),
        "## 2. Current Formal Mechanism Evidence\n\nValue-side functional direction and specific value-geometry transfer remain FORMAL_SUPPORTED.",
        "## 3. Scientific Question\n\nDoes matched-amplitude normalized R readout direction cause larger downstream KL than C, and does validated value-geometry transfer move that sensitivity?",
        "## 4. Hook / Tensor Semantics\n\n" + json.dumps(load_json(RUN_DIR / "stageD0_hook_semantics.json"), indent=2, ensure_ascii=False),
        "## 5. Stage D0 Identity\n\n" + json.dumps(d0, indent=2, ensure_ascii=False),
        "## 6. Readout-Direction Construction\n\n`z_t(E)=E_t^T q_t`; directions are normalized before applying the shared epsilon.",
        "## 7. Token Selection\n\nTop-3 offsets by pooled original readout energy only.",
        "## 8. Matched-Amplitude Perturbation Protocol\n\nEach selected token uses `epsilon=sqrt((||z_R||^2+||z_C||^2)/2)` and symmetric +/- injection.",
        "## 9. Stage D1 Pilot\n\n" + json.dumps(d1, indent=2, ensure_ascii=False),
        "## 10. Pilot Gate\n\n" + d1.get("DIRECTIONAL_SENSITIVITY_PILOT", "NOT_RUN"),
        "## 11. Stage D2 Formal Original R vs C\n\n" + json.dumps(d2, indent=2, ensure_ascii=False),
        "## 12. Downstream Directional Sensitivity Result\n\n" + d2.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL", "NOT_RUN"),
        "## 13. Stage D3 Value-Geometry Transfer Sensitivity\n\n" + json.dumps(d3, indent=2, ensure_ascii=False),
        "## 14. R -> C_V Sensitivity Rescue\n\n" + str(final["R_TO_CV_SENSITIVITY_RESCUE_UNITS"]),
        "## 15. C -> R_V Sensitivity Gain\n\n" + str(final["C_TO_RV_SENSITIVITY_GAIN_UNITS"]),
        "## 16. Behavioral Transfer vs Sensitivity Transfer\n\nD3 tests whether the already validated behavioral transfer is mirrored by matched-amplitude directional sensitivity.",
        "## 17. Observability x Direction Sensitivity\n\nSee `secondary_observability_sensitivity_analysis.csv`.",
        "## 18. Positive Results\n\nReported at unit level only.",
        "## 19. Negative / Corrective Results\n\nNo random direction search, Jacobian, Hessian, or method design was run.",
        "## 20. Mechanism Interpretation\n\n" + cls,
        "## 21. What Is Still Not Proven\n\nNo generalizable downstream-sensitive Value subspace has been established.",
        "## 22. Mechanism Readiness\n\nMETHOD_DESIGN_READY = NO.",
        "## 23. Recommended Next Experiment\n\nIf continuing: `GDN_INT8_VALUE_SENSITIVE_SUBSPACE_GENERALIZATION_AUDIT_V1`. Not run here.",
    ]
    (RUN_DIR / "final_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return final


def event_audit():
    events = []
    for name in ["stageD0_identity.json", "stageD1_summary.json", "stageD2_summary.json", "stageD3_summary.json", "final_summary.json"]:
        p = RUN_DIR / name
        if p.exists():
            events.append({"event": name.replace(".json", ""), "payload": load_json(p)})
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")


def run_all():
    d0 = stage_d0()
    if d0["STAGE_D0"] != "PASS":
        final = final_report(d0, {}, {}, {"D3_RUN": "NO"})
        event_audit()
        return final
    cfg = load_json(RUN_DIR / "config.json")
    pilot_units = cfg["pilot_units"]
    d1_units, _ = run_units(pilot_units, include_transfer=False, prefix="D1_pilot")
    d1 = summarize_d1(d1_units)
    if d1["DIRECTIONAL_SENSITIVITY_PILOT"] != "PASS":
        final = final_report(d0, d1, {"D2_FORMAL_RUN": "NO"}, {"D3_RUN": "NO"})
        event_audit()
        return final
    all_units = [r["unit"] for r in vf.read_prev_stagea_rows()]
    d2_units, _ = run_units(all_units, include_transfer=False, prefix="D2_formal")
    d2 = summarize_d2(d2_units)
    if d2["DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL"] != "FORMAL_SUPPORTED":
        final = final_report(d0, d1, d2, {"D3_RUN": "NO"})
        event_audit()
        return final
    d3_units, _ = run_units(all_units, include_transfer=True, prefix="D3_transfer")
    d3 = summarize_d3(d3_units)
    final = final_report(d0, d1, d2, d3)
    event_audit()
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "d0"], default="all")
    args = ap.parse_args()
    if args.stage == "d0":
        print(json.dumps(stage_d0(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run_all(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
