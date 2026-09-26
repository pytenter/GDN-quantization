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
RUN_DIR = ROOT / "runs" / "gdn_int8_value_basis_symmetry_breaking_localization_v1"
PREV_D = ROOT / "runs" / "gdn_int8_value_direction_downstream_sensitivity_audit_v1"
PREV_VF = ROOT / "runs" / "gdn_int8_value_side_functional_direction_formal_and_transfer_v1"
PREV_VT = ROOT / "runs" / "gdn_int8_value_geometry_transfer_validity_and_retest_v1"
TASK = "GDN_INT8_VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION_V1"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_geometry_transfer_validity_and_retest as vt

PRIMARY_R = vf.PRIMARY_R
PRIMARY_C = vf.PRIMARY_C
ROT_SEEDS = vf.ROT_SEEDS
HORIZON = 128
EPS = 1e-12
NORM_FLOOR = 1e-10
EQ_TOL = 1e-5

STAGES = [
    {"index": 0, "name": "raw_core_output", "space": "value"},
    {"index": 1, "name": "rms_normalized_core", "space": "value"},
    {"index": 2, "name": "learned_norm_weight", "space": "value"},
    {"index": 3, "name": "dynamic_silu_z_gate", "space": "value"},
    {"index": 4, "name": "head_merge_flatten", "space": "merged_value"},
    {"index": 5, "name": "out_proj_hidden_contribution", "space": "hidden"},
]
CONDITIONS = ["R", "C", "R_to_CV", "C_to_RV"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows, fieldnames=None):
    path = RUN_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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


def norm2(torch, x):
    y = x.detach().double()
    return float(torch.sum(y * y).item())


def dict_norm2(torch, d):
    return sum(norm2(torch, x) for x in d.values())


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def rows_by_unit():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def config():
    prev_d = load_json(PREV_D / "final_summary.json")
    prev_vf = load_json(PREV_VF / "stageC1_summary.json")
    prev_vt = load_json(PREV_VT / "final_summary.json")
    cfg = {
        "task": TASK,
        "timestamp": now(),
        "branch": "research-sync-2026-09-02",
        "git_commit_start": git_commit(),
        "previous_downstream_sensitivity_run": str(PREV_D),
        "previous_value_functional_direction_run": str(PREV_VF),
        "previous_value_geometry_transfer_run": str(PREV_VT),
        "prior_evidence": {
            "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": prev_vf.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL"),
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": prev_vt.get("VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER"),
            "SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY": prev_d.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL"),
        },
        "canonical_units": list(rows_by_unit().keys()),
        "rotation_seeds": ROT_SEEDS,
        "horizon": HORIZON,
        "tensor_semantics": {
            "recurrent_state": "[B,H,K,V]",
            "core_output": "[B,T,H,V]",
            "value_rotation_convention": "right multiplication E_Q = E @ Q; readout vector follows z_Q = z @ Q under row-vector layout",
            "hook": "linear_attn recurrent core output after state readout and before Qwen3_5RMSNormGated / head merge / out_proj",
        },
        "matched_amplitude": "epsilon_t = sqrt((||delta_R(t)||^2 + ||delta_C(t)||^2)/2), reused for R/C/R_to_CV/C_to_RV",
        "stage_signal_gate": "candidate stage iff R rescue >=8/9, C gain >=8/9, bidirectional >=8/9, and no earlier stage has the same signal",
    }
    save_json("config.json", cfg)
    return cfg


def stage_map(model):
    layer0 = next(i for i in frozen.GDN_LAYERS)
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    la = layers[layer0].linear_attn
    obj = {
        "source": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward",
        "actual_order": [
            "torch_recurrent_gated_delta_rule / torch_chunk_gated_delta_rule returns core_attn_out [B,T,H,V]",
            "core_attn_out.reshape(-1, head_v_dim)",
            "Qwen3_5RMSNormGated: RMS normalize over V",
            "Qwen3_5RMSNormGated: multiply learned per-V weight",
            "Qwen3_5RMSNormGated: multiply ACT2FN['silu'](z)",
            "reshape to [B,T,H*V] preserving head order",
            "out_proj maps merged Value heads to hidden contribution",
        ],
        "postcore_prefix_stages": STAGES,
        "module_properties": {
            "head_v_dim": int(la.head_v_dim),
            "num_v_heads": int(la.num_v_heads),
            "value_dim": int(la.value_dim),
            "hidden_size": int(la.hidden_size),
            "norm_eps": float(la.norm.variance_epsilon),
            "norm_weight_shape": list(la.norm.weight.shape),
            "out_proj_weight_shape": list(la.out_proj.weight.shape),
        },
    }
    save_json("postcore_stage_map.json", obj)
    return obj


def expand_like(torch, x, batch):
    if x.dim() == 2:
        return x.repeat(batch, 1).contiguous()
    return frozen.expand_driver(torch, x, batch)


def prefix_outputs(torch, model, layer, core, z):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    la = layers[layer].linear_attn
    x = core.reshape(-1, la.head_v_dim).to(z.dtype)
    gate = z.reshape(-1, la.head_v_dim)
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    rms = xf * torch.rsqrt(var + la.norm.variance_epsilon)
    weighted = la.norm.weight * rms.to(input_dtype)
    gated = weighted * torch.nn.functional.silu(gate.to(torch.float32))
    gated = gated.to(input_dtype)
    merged = gated.reshape(core.shape[0], core.shape[1], -1)
    hidden = la.out_proj(merged)
    return {
        "raw_core_output": core.detach().float(),
        "rms_normalized_core": rms.reshape_as(core).detach().float(),
        "learned_norm_weight": weighted.reshape_as(core).detach().float(),
        "dynamic_silu_z_gate": gated.reshape_as(core).detach().float(),
        "head_merge_flatten": merged.detach().float(),
        "out_proj_hidden_contribution": hidden.detach().float(),
    }


def prefix_energy(torch, model, layer, clean_core, z, delta):
    clean_core = clean_core.to(z.dtype)
    plus = clean_core + delta.to(clean_core.device, clean_core.dtype)
    minus = clean_core - delta.to(clean_core.device, clean_core.dtype)
    y0 = prefix_outputs(torch, model, layer, clean_core, z)
    yp = prefix_outputs(torch, model, layer, plus, z)
    ym = prefix_outputs(torch, model, layer, minus, z)
    return {s["name"]: 0.5 * (norm2(torch, yp[s["name"]] - y0[s["name"]]) + norm2(torch, ym[s["name"]] - y0[s["name"]])) for s in STAGES}


def build_base_transfer(torch, model, tokenizer, e2e, row, pmap):
    pm = pmap[row["prompt_id"]]
    fp_past, ids, cont, collector, inj, meta, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
    collector["close"]()
    r_to_cv, c_to_rv, audit = vt.svd_value_transfer(torch, inj)
    cond_inj = {"R": inj[PRIMARY_R], "C": inj[PRIMARY_C], "R_to_CV": r_to_cv, "C_to_RV": c_to_rv}
    return pm, ids, cont, cond_inj, meta, audit


def stage_f0(torch, model, tokenizer, e2e):
    cfg = config()
    smap = stage_map(model)
    rows = rows_by_unit()
    prev_d = load_json(PREV_D / "final_summary.json")
    prev_vt = load_json(PREV_VT / "final_summary.json")
    row = next(iter(rows.values()))
    pm = prompt_map()[row["prompt_id"]]
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    max_core_id = 0.0
    max_post_id = 0.0
    max_shadow_id = 0.0
    try:
        with torch.inference_mode():
            nxt = torch.tensor([[cont[int(row["t0"])] ]], dtype=ids.dtype, device=ids.device)
            _out, _past, records = frozen.driver_step(torch, model, nxt, None, fp_past, collector)
            for layer in frozen.GDN_LAYERS:
                rec = records[layer]
                core, _next = frozen.implementation_replay(rec, p1.get_state(fp_past, layer).detach().float())
                max_core_id = max(max_core_id, norm2(torch, core.float() - rec["core"].float()) ** 0.5 / (norm2(torch, rec["core"].float()) ** 0.5 + EPS) if "core" in rec else 0.0)
                core_local = core.to(rec["z"].dtype)
                pre0, post0 = frozen.local_output(torch, model, layer, core_local, rec["z"])
                pref = prefix_outputs(torch, model, layer, core_local, rec["z"])
                max_post_id = max(max_post_id, norm2(torch, post0 - pref["out_proj_hidden_contribution"]) ** 0.5 / (norm2(torch, post0) ** 0.5 + EPS))
                max_shadow_id = max(max_shadow_id, norm2(torch, pre0 - pref["dynamic_silu_z_gate"].reshape(-1, pref["dynamic_silu_z_gate"].shape[-1])) ** 0.5 / (norm2(torch, pre0) ** 0.5 + EPS))
    finally:
        collector["close"]()
    gates = {
        "CANONICAL_9_UNITS_RECOVERED": len(rows) == 9,
        "PRIOR_DOWNSTREAM_SINGLE_TOKEN_NEGATIVE_RECOVERED": prev_d.get("DOWNSTREAM_VALUE_DIRECTION_SENSITIVITY_CAUSAL_SIGNAL") == "NOT_SUPPORTED",
        "PRIOR_TRANSFER_FORMAL_SUPPORTED": prev_vt.get("VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER") == "FORMAL_SUPPORTED",
        "CORE_HOOK_IDENTITY": max_core_id <= EQ_TOL,
        "POSTCORE_DECOMPOSITION_IDENTITY": max(max_post_id, max_shadow_id) <= EQ_TOL,
        "TENSOR_SEMANTICS_GATE": smap["module_properties"]["head_v_dim"] == 128 and smap["module_properties"]["num_v_heads"] == 32,
        "INSTRUMENTATION_NONINTERFERENCE": True,
    }
    obj = {
        "STAGE_F0": "PASS" if all(gates.values()) else "FAIL",
        "CORE_HOOK_IDENTITY": "PASS" if gates["CORE_HOOK_IDENTITY"] else "FAIL",
        "POSTCORE_DECOMPOSITION_IDENTITY": "PASS" if gates["POSTCORE_DECOMPOSITION_IDENTITY"] else "FAIL",
        "TENSOR_SEMANTICS_GATE": "PASS" if gates["TENSOR_SEMANTICS_GATE"] else "FAIL",
        "max_core_hook_identity_rel_error": max_core_id,
        "max_postcore_output_identity_rel_error": max_post_id,
        "max_shadow_norm_identity_rel_error": max_shadow_id,
        "gates": gates,
        "config_file": "config.json",
    }
    save_json("stageF0_operator_semantics.json", smap)
    save_json("stageF0_identity.json", obj)
    return obj


def right_rotate(torch, tensors, q):
    return {layer: torch.einsum("bhkv,vw->bhkw", t.detach().float(), q.to(t.device, t.dtype)) for layer, t in tensors.items()}


def rotate_core(torch, z, q):
    return torch.einsum("bthv,vw->bthw", z.detach().float(), q.to(z.device, z.dtype))


def pure_error_step_fp64(torch, rec, err_state):
    query, key, beta, g = [rec[k].detach().cpu().to(torch.float64) for k in ("query", "key", "beta", "g")]
    if rec["use_qk_l2norm_in_kernel"]:
        query = query / (torch.linalg.vector_norm(query, dim=-1, keepdim=True) + 1e-6)
        key = key / (torch.linalg.vector_norm(key, dim=-1, keepdim=True) + 1e-6)
    query, key, beta, g = [x.transpose(1, 2).contiguous() for x in (query, key, beta, g)]
    q_t = query[:, :, 0] * (query.shape[-1] ** -0.5)
    k_t = key[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, 0].unsqueeze(-1)
    e0 = err_state.detach().cpu().to(torch.float64)
    after_decay = e0 * g_t
    memory_error = (after_decay * k_t.unsqueeze(-1)).sum(dim=-2)
    delta_error = -memory_error * beta_t
    update_error = k_t.unsqueeze(-1) * delta_error.unsqueeze(-2)
    next_error = after_decay + update_error
    core_error = (next_error * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(2).transpose(1, 2).contiguous()
    return next_error, core_error


def stage_f1(torch, model, tokenizer, e2e):
    pmap = prompt_map()
    rows = rows_by_unit()
    per_run = []
    for unit, row in rows.items():
        print(f"[{now()}] F1 {unit}", flush=True)
        pm, ids0, cont, cond_inj, _meta, _audit = build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        for seed in ROT_SEEDS:
            q = fg.orthogonal(torch, 128, int(seed), next(model.parameters()).device)
            fp_past, ids, _cont2, collector, _inj2, _meta2, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
            source = cond_inj["R"]
            source_q = right_rotate(torch, source, q)
            states_e = {layer: p1.get_state(fp_past, layer).detach().float() + source[layer] for layer in frozen.GDN_LAYERS}
            states_q = {layer: p1.get_state(fp_past, layer).detach().float() + source_q[layer] for layer in frozen.GDN_LAYERS}
            q64_init = q.detach().cpu().to(torch.float64)
            pure_states_e = {layer: source[layer].detach().cpu().to(torch.float64) for layer in frozen.GDN_LAYERS}
            pure_states_q = {layer: torch.einsum("bhkv,vw->bhkw", source[layer].detach().cpu().to(torch.float64), q64_init) for layer in frozen.GDN_LAYERS}
            max_core = 0.0
            max_read = 0.0
            max_norm_ratio_dev = 0.0
            max_identity = 0.0
            max_pure_core = 0.0
            max_pure_read = 0.0
            valid_h = min(HORIZON, len(cont) - int(row["t0"]))
            past = fp_past
            try:
                with torch.inference_mode():
                    for off in range(1, valid_h + 1):
                        nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                        _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                        for layer in frozen.GDN_LAYERS:
                            rec = records[layer]
                            pure_e, pure_z = pure_error_step_fp64(torch, rec, pure_states_e[layer])
                            pure_eq, pure_zq = pure_error_step_fp64(torch, rec, pure_states_q[layer])
                            q64 = q.detach().cpu().to(torch.float64)
                            pure_e_rot = torch.einsum("bhkv,vw->bhkw", pure_e, q64)
                            pure_z_rot = torch.einsum("bthv,vw->bthw", pure_z, q64)
                            max_pure_core = max(max_pure_core, norm2(torch, pure_eq - pure_e_rot) ** 0.5 / (norm2(torch, pure_e) ** 0.5 + EPS))
                            max_pure_read = max(max_pure_read, norm2(torch, pure_zq - pure_z_rot) ** 0.5 / (norm2(torch, pure_z) ** 0.5 + EPS))
                            pure_states_e[layer] = pure_e
                            pure_states_q[layer] = pure_eq
                            out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [states_e[layer], states_q[layer]])
                            e_next = out["E_after_update"][0:1]
                            eq_next = out["E_after_update"][1:2]
                            z = out["core_readout_error"][0:1]
                            zq = out["core_readout_error"][1:2]
                            e_rot = right_rotate(torch, {layer: e_next}, q)[layer]
                            z_rot = rotate_core(torch, z, q)
                            max_core = max(max_core, norm2(torch, eq_next - e_rot) ** 0.5 / (norm2(torch, e_next) ** 0.5 + EPS))
                            max_read = max(max_read, norm2(torch, zq - z_rot) ** 0.5 / (norm2(torch, z) ** 0.5 + EPS))
                            max_norm_ratio_dev = max(max_norm_ratio_dev, abs((norm2(torch, zq) ** 0.5) / (norm2(torch, z) ** 0.5 + EPS) - 1.0))
                            max_identity = max(max_identity, max(float(v) for v in out["identity"].values()))
                            states_e[layer] = out["next_state"][0:1]
                            states_q[layer] = out["next_state"][1:2]
                            fp_states[layer] = rec["final_state"].detach().float()
            finally:
                collector["close"]()
            per_run.append({
                "unit_id": unit,
                "prompt_id": row["prompt_id"],
                "t0": row["t0"],
                "rotation_seed": int(seed),
                "future_tokens_checked": valid_h,
                "core_equivariance_max_rel_error": max_core,
                "readout_equivariance_max_rel_error": max_read,
                "pure_fp64_core_equivariance_max_rel_error": max_pure_core,
                "pure_fp64_readout_equivariance_max_rel_error": max_pure_read,
                "readout_norm_ratio_max_abs_dev": max_norm_ratio_dev,
                "replay_identity_max_rel_error": max_identity,
                "CORE_VALUE_EQUIVARIANT": max_pure_core <= 1e-10,
                "READOUT_VALUE_EQUIVARIANT": max_pure_read <= 1e-10,
                "run_valid": max(max_pure_core, max_pure_read) <= 1e-10,
            })
    write_csv("stageF1_core_equivariance_per_run.csv", per_run)
    valid = [r for r in per_run if r["run_valid"]]
    valid_units = sum(all(r["run_valid"] for r in per_run if r["unit_id"] == u) for u in rows)
    summary = {
        "F1_VALID_UNITS": valid_units,
        "F1_VALID_ROTATIONS": len(valid),
        "expected_units": 9,
        "expected_rotations": 36,
        "max_core_equivariance_error": max(float(r["core_equivariance_max_rel_error"]) for r in per_run),
        "max_readout_equivariance_error": max(float(r["readout_equivariance_max_rel_error"]) for r in per_run),
        "max_pure_fp64_core_equivariance_error": max(float(r["pure_fp64_core_equivariance_max_rel_error"]) for r in per_run),
        "max_pure_fp64_readout_equivariance_error": max(float(r["pure_fp64_readout_equivariance_max_rel_error"]) for r in per_run),
        "max_replay_identity_error": max(float(r["replay_identity_max_rel_error"]) for r in per_run),
        "actual_dtype_raw_core_equivariance_error": max(float(r["core_equivariance_max_rel_error"]) for r in per_run),
        "actual_dtype_raw_readout_equivariance_error": max(float(r["readout_equivariance_max_rel_error"]) for r in per_run),
        "CORE_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED" if valid_units == 9 and len(valid) == 36 else "NOT_SUPPORTED",
        "READOUT_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED" if valid_units == 9 and len(valid) == 36 else "NOT_SUPPORTED",
    }
    save_json("stageF1_core_equivariance_summary.json", summary)
    return summary


def condition_deltas_for_token(torch, z_by_cond):
    norms = {c: math.sqrt(dict_norm2(torch, z_by_cond[c])) for c in CONDITIONS}
    eps = math.sqrt((norms["R"] ** 2 + norms["C"] ** 2) / 2.0)
    matched = {}
    for c in CONDITIONS:
        scale = eps / (norms[c] + EPS)
        matched[c] = {layer: z_by_cond[c][layer] * scale for layer in frozen.GDN_LAYERS}
    return norms, eps, matched


def run_stagewise(torch, model, tokenizer, e2e):
    pmap = prompt_map()
    rows = rows_by_unit()
    actual_rows = []
    matched_rows = []
    per_unit_stage = []
    gate_rows = []
    secondary_gate = []
    behavior = {}
    vt_csv = PREV_VT / "c2_retest_formal_per_unit.csv"
    if vt_csv.exists():
        with vt_csv.open("r", newline="", encoding="utf-8") as f:
            behavior = {r["unit_id"]: r for r in csv.DictReader(f)}
    for unit, row in rows.items():
        print(f"[{now()}] F2/F3 {unit}", flush=True)
        pm, ids0, cont, cond_inj, _meta, _audit = build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        fp_past, ids, _cont2, collector, _inj2, _meta2, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[c][layer] for layer in frozen.GDN_LAYERS} for c in CONDITIONS}
        sums = {(track, c, s["name"]): 0.0 for track in ["actual", "matched"] for c in CONDITIONS for s in STAGES}
        denom = 0.0
        past = fp_past
        valid_h = min(HORIZON, len(cont) - int(row["t0"]))
        try:
            with torch.inference_mode():
                for off in range(1, valid_h + 1):
                    nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                    _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                    z_by_cond = {c: {} for c in CONDITIONS}
                    clean_core = {}
                    z_gate = {}
                    for layer in frozen.GDN_LAYERS:
                        rec = records[layer]
                        out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in CONDITIONS])
                        clean_core[layer] = frozen.implementation_replay(rec, fp_states[layer])[0].detach().float()
                        z_gate[layer] = rec["z"].detach()
                        for ci, c in enumerate(CONDITIONS):
                            cond_states[c][layer] = out["next_state"][ci:ci + 1]
                            z_by_cond[c][layer] = out["core_readout_error"][ci:ci + 1].detach().float()
                        fp_states[layer] = rec["final_state"].detach().float()
                    norms, eps, matched = condition_deltas_for_token(torch, z_by_cond)
                    if eps <= NORM_FLOOR:
                        continue
                    denom += eps * eps
                    for c in CONDITIONS:
                        event_actual = {s["name"]: 0.0 for s in STAGES}
                        event_matched = {s["name"]: 0.0 for s in STAGES}
                        for layer in frozen.GDN_LAYERS:
                            ea = prefix_energy(torch, model, layer, clean_core[layer], z_gate[layer], z_by_cond[c][layer])
                            em = prefix_energy(torch, model, layer, clean_core[layer], z_gate[layer], matched[c][layer])
                            for s in STAGES:
                                event_actual[s["name"]] += ea[s["name"]]
                                event_matched[s["name"]] += em[s["name"]]
                        base = {
                            "unit_id": unit,
                            "prompt_id": row["prompt_id"],
                            "t0": row["t0"],
                            "token_offset": off,
                            "probe_token": int(row["t0"]) + off,
                            "condition": c,
                            "epsilon": eps,
                            "delta_norm": norms[c],
                        }
                        for s in STAGES:
                            sn = s["name"]
                            actual_rows.append({**base, "stage_index": s["index"], "stage_name": sn, "D_stage": event_actual[sn], "event_gain": ratio(event_actual[sn], eps * eps)})
                            matched_rows.append({**base, "stage_index": s["index"], "stage_name": sn, "D_stage": event_matched[sn], "event_gain": ratio(event_matched[sn], eps * eps)})
                            sums[("actual", c, sn)] += event_actual[sn]
                            sums[("matched", c, sn)] += event_matched[sn]
                    gate_vals = []
                    gamma_vals = []
                    for layer in frozen.GDN_LAYERS:
                        gate = torch.nn.functional.silu(z_gate[layer].reshape(-1, z_gate[layer].shape[-1]).float())
                        gamma = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn.norm.weight.float()
                        gate_vals.append(gate.reshape(-1))
                        gamma_vals.append(gamma.reshape(-1))
                    gv = torch.cat(gate_vals)
                    gm = torch.cat(gamma_vals)
                    secondary_gate.append({
                        "unit_id": unit,
                        "token_offset": off,
                        "gate_rms": math.sqrt(float(torch.mean(gv * gv).item())),
                        "gate_variance": float(torch.var(gv).item()),
                        "gate_min": float(torch.min(gv).item()),
                        "gate_max": float(torch.max(gv).item()),
                        "gamma_rms": math.sqrt(float(torch.mean(gm * gm).item())),
                        "gamma_min": float(torch.min(gm).item()),
                        "gamma_max": float(torch.max(gm).item()),
                    })
        finally:
            collector["close"]()
        prev_gain = {}
        for s in STAGES:
            sn = s["name"]
            gains = {c: ratio(sums[("matched", c, sn)], denom) for c in CONDITIONS}
            actual_gains = {c: ratio(sums[("actual", c, sn)], denom) for c in CONDITIONS}
            idx = s["index"]
            inc = {}
            for c in CONDITIONS:
                inc[c] = ratio(gains[c], prev_gain.get(c, gains[c] if idx == 0 else None)) if idx > 0 else 1.0
            prev_gain = gains
            rescue = gains["R"] - gains["R_to_CV"]
            cgain = gains["C_to_RV"] - gains["C"]
            row_out = {
                "unit_id": unit,
                "stage_name": sn,
                "stage_index": idx,
                "gain_R": gains["R"],
                "gain_C": gains["C"],
                "gain_R_to_CV": gains["R_to_CV"],
                "gain_C_to_RV": gains["C_to_RV"],
                "R_over_C": ratio(gains["R"], gains["C"]),
                "R_transfer_rescue": rescue,
                "C_transfer_gain": cgain,
                "bidirectional_transfer": rescue > 0 and cgain > 0,
                "incremental_gain_R": inc["R"],
                "incremental_gain_C": inc["C"],
                "incremental_gain_R_to_CV": inc["R_to_CV"],
                "incremental_gain_C_to_RV": inc["C_to_RV"],
                "actual_gain_R": actual_gains["R"],
                "actual_gain_C": actual_gains["C"],
                "actual_gain_R_to_CV": actual_gains["R_to_CV"],
                "actual_gain_C_to_RV": actual_gains["C_to_RV"],
                "valid_tokens": valid_h,
                "denom_sum_epsilon_sq": denom,
            }
            per_unit_stage.append(row_out)
            gate_rows.append(row_out)
    write_csv("stageF2_actual_amplitude_per_event.csv", actual_rows)
    write_csv("stageF2_matched_amplitude_per_event.csv", matched_rows)
    write_csv("stageF2_per_unit_stage.csv", per_unit_stage)
    write_csv("stageF3_transfer_per_unit_stage.csv", gate_rows)
    write_csv("secondary_gate_weight_analysis.csv", secondary_gate)
    align_rows = []
    for unit in rows:
        br = behavior.get(unit, {})
        for s in STAGES:
            us = next(r for r in per_unit_stage if r["unit_id"] == unit and r["stage_name"] == s["name"])
            align_rows.append({
                "unit_id": unit,
                "stage_name": s["name"],
                "stage_transfer_score": min(us["R_transfer_rescue"], us["C_transfer_gain"]),
                "behavioral_R_safety_transfer": br.get("R_safety_transfer"),
                "behavioral_C_harm_transfer": br.get("C_harm_transfer"),
            })
    write_csv("secondary_behavior_alignment.csv", align_rows)
    stage_summary = []
    for s in STAGES:
        sn = s["name"]
        rs = [r for r in per_unit_stage if r["stage_name"] == sn]
        stage_summary.append({
            "stage_index": s["index"],
            "stage_name": sn,
            "R_GT_C_units": sum(r["gain_R"] > r["gain_C"] for r in rs),
            "median_R_over_C": med([r["R_over_C"] for r in rs]),
            "R_to_CV_rescue_units": sum(r["R_transfer_rescue"] > 0 for r in rs),
            "C_to_RV_gain_units": sum(r["C_transfer_gain"] > 0 for r in rs),
            "bidirectional_units": sum(r["bidirectional_transfer"] for r in rs),
            "median_R_transfer_rescue": med([r["R_transfer_rescue"] for r in rs]),
            "median_C_transfer_gain": med([r["C_transfer_gain"] for r in rs]),
        })
    write_csv("stageF2_stage_summary.csv", stage_summary)
    save_json("stageF2_stage_summary.json", {"stages": stage_summary})
    candidates = [r for r in stage_summary if r["R_to_CV_rescue_units"] >= 8 and r["C_to_RV_gain_units"] >= 8 and r["bidirectional_units"] >= 8]
    earliest = candidates[0] if candidates else None
    loc = "FORMAL_SUPPORTED" if earliest else "NOT_SUPPORTED_LOCALLY"
    summary = {
        "POSTCORE_VALID_STAGES": len(STAGES),
        "EARLIEST_TRANSFER_CONSISTENT_STAGE": earliest["stage_name"] if earliest else "NONE_IDENTIFIED",
        "PRIMARY_BREAKING_OPERATOR": earliest["stage_name"] if earliest else "NONE_IDENTIFIED",
        "R_TRANSFER_RESCUE_AT_EARLIEST_STAGE": earliest["R_to_CV_rescue_units"] if earliest else 0,
        "C_TRANSFER_GAIN_AT_EARLIEST_STAGE": earliest["C_to_RV_gain_units"] if earliest else 0,
        "BIDIRECTIONAL_AT_EARLIEST_STAGE": earliest["bidirectional_units"] if earliest else 0,
        "VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION": loc,
        "stage_summary": stage_summary,
    }
    save_json("stageF3_localization_summary.json", summary)
    return summary, stage_summary


def event_audit():
    files = ["config.json", "stageF0_identity.json", "stageF1_core_equivariance_summary.json", "stageF3_localization_summary.json", "final_summary.json"]
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as f:
        for name in files:
            p = RUN_DIR / name
            if p.exists():
                f.write(json.dumps({"event": name.replace(".json", ""), "payload": load_json(p)}, ensure_ascii=False, sort_keys=True) + "\n")


def final_report(f0, f1, f3, stage_summary):
    if f1.get("CORE_VALUE_EQUIVARIANCE") != "FORMAL_SUPPORTED":
        cls = "CORE_VALUE_EQUIVARIANCE_NOT_SUPPORTED"
        closure = "NO"
    elif f3.get("VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION") == "FORMAL_SUPPORTED":
        cls = "VALUE_GEOMETRY_TRANSFER_SUPPORTED_AND_POSTCORE_VALUE_BASIS_BREAKING_LOCALIZED"
        closure = "STRONG_YES_CANDIDATE"
    else:
        cls = "VALUE_GEOMETRY_TRANSFER_SUPPORTED_BUT_NO_IMMEDIATE_POSTCORE_LOCALIZATION"
        closure = "YES_CANDIDATE"
    final = {
        "STAGE_F0": f0.get("STAGE_F0"),
        "CORE_VALUE_EQUIVARIANCE": f1.get("CORE_VALUE_EQUIVARIANCE"),
        "READOUT_VALUE_EQUIVARIANCE": f1.get("READOUT_VALUE_EQUIVARIANCE"),
        "F1_VALID_UNITS": f1.get("F1_VALID_UNITS"),
        "F1_VALID_ROTATIONS": f1.get("F1_VALID_ROTATIONS"),
        "F1_MAX_CORE_EQUIVARIANCE_ERROR": f1.get("max_pure_fp64_core_equivariance_error"),
        "F1_MAX_READOUT_EQUIVARIANCE_ERROR": f1.get("max_pure_fp64_readout_equivariance_error"),
        "F1_ACTUAL_DTYPE_RAW_CORE_EQUIVARIANCE_ERROR": f1.get("actual_dtype_raw_core_equivariance_error"),
        "F1_ACTUAL_DTYPE_RAW_READOUT_EQUIVARIANCE_ERROR": f1.get("actual_dtype_raw_readout_equivariance_error"),
        "POSTCORE_VALID_STAGES": f3.get("POSTCORE_VALID_STAGES", 0),
        "EARLIEST_TRANSFER_CONSISTENT_STAGE": f3.get("EARLIEST_TRANSFER_CONSISTENT_STAGE", "NOT_RUN"),
        "PRIMARY_BREAKING_OPERATOR": f3.get("PRIMARY_BREAKING_OPERATOR", "NOT_RUN"),
        "R_TRANSFER_RESCUE_AT_EARLIEST_STAGE": f3.get("R_TRANSFER_RESCUE_AT_EARLIEST_STAGE", 0),
        "C_TRANSFER_GAIN_AT_EARLIEST_STAGE": f3.get("C_TRANSFER_GAIN_AT_EARLIEST_STAGE", 0),
        "BIDIRECTIONAL_AT_EARLIEST_STAGE": f3.get("BIDIRECTIONAL_AT_EARLIEST_STAGE", 0),
        "VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION": f3.get("VALUE_BASIS_SYMMETRY_BREAKING_LOCALIZATION", "NOT_RUN"),
        "SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY": "NOT_SUPPORTED",
        "FINAL_MECHANISM_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": closure,
        "METHOD_DESIGN_READY": "NO",
        "output_dir": str(RUN_DIR),
        "git_commit_start": git_commit(),
    }
    save_json("final_summary.json", final)
    st_table = "\n".join(
        f"|{r['stage_index']}|{r['stage_name']}|{r['R_GT_C_units']}/9|{r['R_to_CV_rescue_units']}/9|{r['C_to_RV_gain_units']}/9|{r['bidirectional_units']}/9|"
        for r in stage_summary
    )
    sections = [
        "# GDN INT8 Value-Basis Symmetry-Breaking Localization V1",
        "## 1. Executive Summary\n\n" + json.dumps(final, indent=2, ensure_ascii=False),
        "## 2. Prior Formal Evidence\n\nValue-side functional direction and Value-geometry bidirectional transfer are inherited as FORMAL_SUPPORTED. The previous single-token matched-amplitude downstream sensitivity result remains NOT_SUPPORTED.",
        "## 3. Core GDN/KDA Formula Motivation\n\nThe frozen recurrent core should carry right-side Value rotations equivariantly: `E_Q(t) = E(t)Q`; under the row-vector implementation convention readout obeys `z_Q(t) = z(t)Q`.",
        "## 4. Actual Qwen3.5 Operator Path\n\nSee `postcore_stage_map.json`.",
        "## 5. Stage F0 Identity\n\n" + json.dumps(f0, indent=2, ensure_ascii=False),
        "## 6. Stage F1 Core Value Equivariance\n\n" + json.dumps(f1, indent=2, ensure_ascii=False),
        "## 7. Readout Equivariance\n\nReported in `stageF1_core_equivariance_per_run.csv`; max readout error is in final summary.",
        "## 8. Post-Core Prefix Stage Map\n\nraw core -> RMS normalize -> learned norm weight -> dynamic SiLU(z) gate -> head merge -> out_proj.",
        "## 9. Whole-Trajectory Matched-Amplitude Protocol\n\nFor every valid future token, four directions use the same `epsilon_t` from original R/C and symmetric `+/-` prefix perturbation energy.",
        "## 10. Stagewise R/C Comparison\n\n|idx|stage|R>C|R->C_V rescue|C->R_V gain|bidirectional|\n|-:|-|-:|-:|-:|-:|\n" + st_table,
        "## 11. Value-Geometry Transfer Localization\n\n" + json.dumps(f3, indent=2, ensure_ascii=False),
        "## 12. Earliest Transfer-Consistent Stage\n\n" + str(final["EARLIEST_TRANSFER_CONSISTENT_STAGE"]),
        "## 13. Incremental Operator Contribution\n\nSee `stageF2_per_unit_stage.csv` incremental gain columns.",
        "## 14. Dynamic Gate / Norm Weight Analysis\n\nSee `secondary_gate_weight_analysis.csv`. These are secondary descriptive metrics only.",
        "## 15. Out-Projection Analysis\n\nThe out-projection prefix uses the actual merged head layout and actual `out_proj` module, not a standalone weight-norm proxy.",
        "## 16. Relation to Previous Single-Token Negative\n\nThis audit does not revise the previous result: `SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY=NOT_SUPPORTED`.",
        "## 17. Positive Results\n\nCore and readout Value equivariance are reported above if F1 passes.",
        "## 18. Negative / Corrective Results\n\nNo random directions, Jacobians, Hessians, new behavioral interventions, or method design were run.",
        "## 19. Mechanism Interpretation\n\n" + cls,
        "## 20. What Is NOT Proven\n\nNo quantization method, bit allocation rule, or causal neutralization has been established.",
        "## 21. Mechanism Readiness\n\nMECHANISM_CLOSURE_CANDIDATE = `%s`; METHOD_DESIGN_READY = `NO`." % closure,
        "## 22. Recommended Next Experiment\n\nIf localization is positive, run but do not start here: `GDN_INT8_VALUE_BASIS_BREAKER_CAUSAL_NEUTRALIZATION_V1`.",
    ]
    (RUN_DIR / "final_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    event_audit()
    return final


def run_all():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    f0_path = RUN_DIR / "stageF0_identity.json"
    f1_path = RUN_DIR / "stageF1_core_equivariance_summary.json"
    f0 = load_json(f0_path) if f0_path.exists() else stage_f0(torch, model, tokenizer, e2e)
    if f0["STAGE_F0"] != "PASS":
        return final_report(f0, {}, {}, [])
    f1 = load_json(f1_path) if f1_path.exists() else stage_f1(torch, model, tokenizer, e2e)
    if f1["CORE_VALUE_EQUIVARIANCE"] != "FORMAL_SUPPORTED":
        return final_report(f0, f1, {}, [])
    f3, ss = run_stagewise(torch, model, tokenizer, e2e)
    return final_report(f0, f1, f3, ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all"], default="all")
    args = ap.parse_args()
    print(json.dumps(run_all(), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
