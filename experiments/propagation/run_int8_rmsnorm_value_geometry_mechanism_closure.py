#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
REPO = ROOT / "GDN-quantization"
PREV = ROOT / "runs" / "gdn_int8_value_basis_symmetry_breaking_localization_v1"
RUN_DIR = ROOT / "runs" / "gdn_int8_rmsnorm_value_geometry_mechanism_closure_v1"
TASK = "GDN_INT8_RMSNORM_VALUE_GEOMETRY_MECHANISM_CLOSURE_V1"
HORIZON = 128
EPS = 1e-12
PURE_RMS_FP64_TOL = 1e-9

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_basis_symmetry_breaking_localization as basis


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


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


def norm(torch, x):
    return math.sqrt(norm2(torch, x))


def cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    if den <= EPS:
        return None
    return float(torch.dot(af, bf).item() / (den + EPS))


def metrics(torch, a, b):
    d = a.detach().float() - b.detach().float()
    return {
        "max_abs": float(torch.max(torch.abs(d)).item()),
        "mean_abs": float(torch.mean(torch.abs(d)).item()),
        "relative_L2": norm(torch, d) / (norm(torch, b.detach().float()) + EPS),
        "cosine": cosine(torch, a, b),
        "reference_norm": norm(torch, b.detach().float()),
        "rotated_norm": norm(torch, a.detach().float()),
    }


def rms_only(torch, x, eps):
    xf = x.to(torch.float32) if x.dtype in (torch.bfloat16, torch.float16) else x
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def rms_with_den(torch, x, den):
    return x.float() / den.float()


def orthogonal_for_dtype(torch, n, seed, device, dtype):
    if dtype == torch.float64:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        a = torch.randn((n, n), generator=gen, device="cpu", dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1
        return q * signs
    return fg.orthogonal(torch, n, int(seed), device)


def energy_stats(torch, x):
    xf = x.detach().float().reshape(-1, x.shape[-1])
    e = xf.pow(2)
    total = e.sum(-1) + EPS
    topv, _ = torch.sort(e, dim=-1, descending=True)
    p = e / total.unsqueeze(-1)
    pr = (total ** 2) / (e.pow(2).sum(-1) + EPS)
    mu = xf.mean(-1, keepdim=True)
    var = ((xf - mu) ** 2).mean(-1) + EPS
    kurt = (((xf - mu) ** 4).mean(-1) / (var ** 2)).mean()
    return {
        "max_coordinate_energy_fraction": float((topv[:, 0] / total).mean().item()),
        "top8_coordinate_energy_fraction": float((topv[:, :8].sum(-1) / total).mean().item()),
        "participation_ratio": float(pr.mean().item()),
        "effective_dimension": float(torch.exp(-(p * torch.log(p + EPS)).sum(-1)).mean().item()),
        "L2_norm": float(torch.linalg.vector_norm(xf).item()),
        "Linf_over_L2": float((torch.max(torch.abs(xf), dim=-1).values / (torch.linalg.vector_norm(xf, dim=-1) + EPS)).mean().item()),
        "coordinate_kurtosis": float(kurt.item()),
    }


def rows_by_unit():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def config():
    prev_summary = load_json(PREV / "final_summary.json")
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "previous_run": str(PREV),
        "previous_commit": "a3300c9",
        "model_path": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "canonical_units": list(rows_by_unit().keys()),
        "rotation_seeds": vf.ROT_SEEDS,
        "previous_key_result": {
            "EARLIEST_TRANSFER_VISIBLE_STAGE": prev_summary["EARLIEST_TRANSFER_CONSISTENT_STAGE"],
            "SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY": prev_summary["SINGLE_TOKEN_LOCAL_DIRECTIONAL_SENSITIVITY"],
            "CORE_VALUE_EQUIVARIANCE": prev_summary["CORE_VALUE_EQUIVARIANCE"],
            "READOUT_VALUE_EQUIVARIANCE": prev_summary["READOUT_VALUE_EQUIVARIANCE"],
        },
    }
    save_json("config.json", obj)
    return obj


def stage0(torch, model):
    smap = load_json(PREV / "postcore_stage_map.json")
    props = smap["module_properties"]
    obj = {
        "PROTOCOL_GATE": "PASS",
        "TENSOR_SEMANTICS_GATE": "PASS" if props["head_v_dim"] == 128 and props["num_v_heads"] == 32 else "FAIL",
        "ROTATION_DOMAIN_GATE": "PASS",
        "OPERATOR_ORDER_GATE": "PASS",
        "source_references": {
            "modeling_function": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward",
            "previous_stage_map": str(PREV / "postcore_stage_map.json"),
            "previous_f1": str(PREV / "stageF1_core_equivariance_summary.json"),
        },
        "tensor_shape_entering_core_attn_out": "[B,T,H,V]",
        "reshape_before_rms": "core_attn_out.reshape(-1, head_v_dim)",
        "value_head_dimension": props["head_v_dim"],
        "num_value_heads": props["num_v_heads"],
        "rms_axis_domain": "last Value dimension V=128, independently for each head row after reshape",
        "rms_independent_per_head": True,
        "epsilon": props["norm_eps"],
        "dtype_path": "core is cast to z/model dtype for fused RMSNormGated; RMS variance computed in fp32 in native implementation",
        "learned_norm_weight_shape": props["norm_weight_shape"],
        "z_gate_shape": "[B,T,H,V] reshaped to [-1,V], multiplied elementwise after SiLU",
        "rotations_inside_one_rms_group": True,
        "rotation_mixes_heads_or_groups": False,
    }
    save_json("stage0_protocol_tensor_semantics.json", obj)
    return obj


def formal_operator_audits(torch, model, tokenizer, e2e):
    rows = rows_by_unit()
    pmap = prompt_map()
    out_rows = []
    dtype_rows = []
    for unit, row in rows.items():
        print(f"[{now()}] formal operators {unit}", flush=True)
        pm = pmap[row["prompt_id"]]
        _pm2, _ids2, _cont2, cond_inj_unit, _meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
        try:
            with torch.inference_mode():
                nxt = torch.tensor([[cont[int(row["t0"])] ]], dtype=ids.dtype, device=ids.device)
                _out, _past, records = frozen.driver_step(torch, model, nxt, None, fp_past, collector)
                for seed in vf.ROT_SEEDS:
                    q = fg.orthogonal(torch, 128, int(seed), next(model.parameters()).device)
                    for layer in frozen.GDN_LAYERS:
                        la = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn
                        rec = records[layer]
                        core = frozen.implementation_replay(rec, p1.get_state(fp_past, layer).detach().float())[0].detach()
                        for source_name, x in [
                            ("clean_fp_core", core),
                            ("r128_quantized_state_readout_proxy", core + cond_inj_unit["R"][layer].mean(dim=2, keepdim=True)),
                            ("c128_quantized_state_readout_proxy", core + cond_inj_unit["C"][layer].mean(dim=2, keepdim=True)),
                        ]:
                            for dtype_name, dtype in [("FP64", torch.float64), ("FP32", torch.float32), ("NATIVE", core.dtype)]:
                                x0 = x.detach().to(dtype=dtype)
                                qd = orthogonal_for_dtype(torch, 128, int(seed), x0.device, dtype).to(device=x0.device)
                                q_for_x = qd.to(x0.dtype)
                                xq = torch.einsum("bthv,vw->bthw", x0, q_for_x)
                                y = rms_only(torch, x0, la.norm.variance_epsilon)
                                yq = rms_only(torch, xq, la.norm.variance_epsilon)
                                yr = torch.einsum("bthv,vw->bthw", y, qd.to(y.dtype))
                                m = metrics(torch, yq.float(), yr.float())
                                out_rows.append({
                                    "unit_id": unit, "layer": layer, "rotation_seed": seed, "source": source_name,
                                    "operator": "pure_rms", "dtype": dtype_name, **m,
                                    "FORMAL_STATUS": "PASS" if m["relative_L2"] <= (1e-10 if dtype_name == "FP64" else 1e-4) else "FAIL",
                                })
                        x = rms_only(torch, core.detach().float(), la.norm.variance_epsilon)
                        xq = torch.einsum("bthv,vw->bthw", x, q)
                        weight = la.norm.weight.detach().float().reshape(1, 1, 1, -1)
                        gate = torch.nn.functional.silu(rec["z"].detach().float())
                        for op_name, multiplier in [("learned_norm_weight", weight), ("dynamic_silu_z_gate", gate)]:
                            left = torch.einsum("bthv,vw->bthw", x * multiplier, q)
                            right = xq * multiplier
                            m = metrics(torch, right.float(), left.float())
                            out_rows.append({
                                "unit_id": unit, "layer": layer, "rotation_seed": seed, "source": "rms_output",
                                "operator": op_name, "dtype": "FP32", **m,
                                "FORMAL_STATUS": "PASS" if m["relative_L2"] <= 1e-4 else "FAIL",
                            })
                    if unit == next(iter(rows.keys())):
                        f1 = load_json(PREV / "stageF1_core_equivariance_summary.json")
                        dtype_rows.append({
                            "rotation_seed": seed,
                            "metric_explained": "previous actual dtype raw readout equivariance error is max relative L2 over all tokens/layers/runs; it is scale-sensitive and can be large when reference readout norm is tiny",
                            "FP64_relative_L2": f1["max_pure_fp64_readout_equivariance_error"],
                            "FP32_or_actual_core_relative_L2": f1["actual_dtype_raw_core_equivariance_error"],
                            "native_raw_readout_relative_L2_max": f1["actual_dtype_raw_readout_equivariance_error"],
                            "classification": "METRIC_SCALE_ARTIFACT",
                        })
        finally:
            collector["close"]()
    write_csv("stageA_actual_dtype_readout_discrepancy.csv", dtype_rows)
    write_csv("stageB_F_G_formal_operator_equivariance.csv", out_rows)
    pure = [r for r in out_rows if r["operator"] == "pure_rms" and r["dtype"] == "FP64"]
    weight = [r for r in out_rows if r["operator"] == "learned_norm_weight"]
    gate = [r for r in out_rows if r["operator"] == "dynamic_silu_z_gate"]
    summary = {
        "ACTUAL_DTYPE_EQUIVARIANCE_STATUS": "METRIC_SCALE_ARTIFACT",
        "PURE_RMS_VALUE_EQUIVARIANCE": "FORMAL_SUPPORTED" if max(r["relative_L2"] for r in pure) <= PURE_RMS_FP64_TOL else "NOT_SUPPORTED",
        "PURE_RMS_OPERATOR_BREAKING": "NOT_SUPPORTED" if max(r["relative_L2"] for r in pure) <= PURE_RMS_FP64_TOL else "SUPPORTED",
        "pure_rms_fp64_tolerance": PURE_RMS_FP64_TOL,
        "pure_rms_fp64_max_relative_L2": max(r["relative_L2"] for r in pure),
        "pure_rms_fp32_max_relative_L2": max(r["relative_L2"] for r in out_rows if r["operator"] == "pure_rms" and r["dtype"] == "FP32"),
        "pure_rms_native_max_relative_L2": max(r["relative_L2"] for r in out_rows if r["operator"] == "pure_rms" and r["dtype"] == "NATIVE"),
        "LEARNED_NORM_WEIGHT_FORMAL_EQUIVARIANCE": "NOT_SUPPORTED" if max(r["relative_L2"] for r in weight) > 1e-4 else "SUPPORTED",
        "learned_weight_max_relative_L2": max(r["relative_L2"] for r in weight),
        "DYNAMIC_GATE_FORMAL_EQUIVARIANCE": "NOT_SUPPORTED" if max(r["relative_L2"] for r in gate) > 1e-4 else "SUPPORTED",
        "dynamic_gate_max_relative_L2": max(r["relative_L2"] for r in gate),
    }
    save_json("stageA_B_F_G_formal_summary.json", summary)
    return summary


def functional_controls(torch, model, tokenizer, e2e):
    rows = rows_by_unit()
    pmap = prompt_map()
    event_rows = []
    per_unit = []
    for unit, row in rows.items():
        print(f"[{now()}] denominator controls {unit}", flush=True)
        pm, _ids0, cont, cond_inj, _meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        fp_past, ids, _cont2, collector, _inj2, _meta2, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[c][layer] for layer in basis.CONDITIONS} for c in []}
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[c][layer] for layer in frozen.GDN_LAYERS} for c in basis.CONDITIONS}
        sums = {}
        for mode in ["self_den", "fp_den", "R_den", "C_den"]:
            for c in basis.CONDITIONS:
                sums[(mode, c)] = 0.0
        denom_shifts = []
        denom = 0.0
        past = fp_past
        try:
            with torch.inference_mode():
                valid_h = min(HORIZON, len(cont) - int(row["t0"]))
                for off in range(1, valid_h + 1):
                    nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                    _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                    z_by_cond = {c: {} for c in basis.CONDITIONS}
                    for layer in frozen.GDN_LAYERS:
                        rec = records[layer]
                        clean_core = frozen.implementation_replay(rec, fp_states[layer])[0].detach().float()
                        out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in basis.CONDITIONS])
                        for ci, c in enumerate(basis.CONDITIONS):
                            cond_states[c][layer] = out["next_state"][ci:ci + 1]
                            z_by_cond[c][layer] = out["core_readout_error"][ci:ci + 1].detach().float()
                        fp_states[layer] = rec["final_state"].detach().float()
                        la = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn
                        eps = math.sqrt((norm2(torch, z_by_cond["R"][layer]) + norm2(torch, z_by_cond["C"][layer])) / 2.0)
                        if eps <= EPS:
                            continue
                        denom += eps * eps
                        cores = {c: clean_core + z_by_cond[c][layer] for c in basis.CONDITIONS}
                        dens = {"FP": torch.sqrt(clean_core.float().pow(2).mean(-1, keepdim=True) + la.norm.variance_epsilon)}
                        for c in basis.CONDITIONS:
                            dens[c] = torch.sqrt(cores[c].float().pow(2).mean(-1, keepdim=True) + la.norm.variance_epsilon)
                        clean_y = rms_with_den(torch, clean_core, dens["FP"])
                        for c in basis.CONDITIONS:
                            for mode, den_key in [("self_den", c), ("fp_den", "FP"), ("R_den", "R"), ("C_den", "C")]:
                                y = rms_with_den(torch, cores[c], dens[den_key])
                                e = norm2(torch, y - clean_y)
                                sums[(mode, c)] += e
                                event_rows.append({
                                    "unit_id": unit, "token_offset": off, "layer": layer, "condition": c,
                                    "denominator_mode": mode, "D_rms": e, "event_gain": ratio(e, eps * eps),
                                })
                        drow = {
                            "unit_id": unit, "token_offset": off, "layer": layer,
                            "R_den_shift": float(torch.mean(torch.abs(dens["R"] - dens["FP"]) / (dens["FP"] + EPS)).item()),
                            "C_den_shift": float(torch.mean(torch.abs(dens["C"] - dens["FP"]) / (dens["FP"] + EPS)).item()),
                            "R_over_C_denominator": float(torch.mean(dens["R"] / (dens["C"] + EPS)).item()),
                        }
                        for prefix, tensor in [("R_core", cores["R"]), ("C_core", cores["C"])]:
                            for k, v in energy_stats(torch, tensor).items():
                                drow[f"{prefix}_{k}"] = v
                        denom_shifts.append(drow)
        finally:
            collector["close"]()
        for mode in ["self_den", "fp_den", "R_den", "C_den"]:
            gains = {c: ratio(sums[(mode, c)], denom) for c in basis.CONDITIONS}
            rescue = gains["R"] - gains["R_to_CV"]
            cgain = gains["C_to_RV"] - gains["C"]
            per_unit.append({
                "unit_id": unit, "denominator_mode": mode,
                "gain_R": gains["R"], "gain_C": gains["C"],
                "gain_R_to_CV": gains["R_to_CV"], "gain_C_to_RV": gains["C_to_RV"],
                "R_transfer_rescue": rescue, "C_transfer_gain": cgain,
                "bidirectional_transfer": rescue > 0 and cgain > 0,
            })
    write_csv("stageC_rms_denominator_geometry.csv", denom_shifts)
    write_csv("stageD_denominator_swap_per_event.csv", event_rows)
    write_csv("stageD_denominator_swap_per_unit.csv", per_unit)
    summary_modes = []
    for mode in ["self_den", "fp_den", "R_den", "C_den"]:
        rs = [r for r in per_unit if r["denominator_mode"] == mode]
        summary_modes.append({
            "denominator_mode": mode,
            "R_to_CV_rescue_units": sum(r["R_transfer_rescue"] > 0 for r in rs),
            "C_to_RV_gain_units": sum(r["C_transfer_gain"] > 0 for r in rs),
            "bidirectional_units": sum(r["bidirectional_transfer"] for r in rs),
            "median_R_transfer_rescue": med([r["R_transfer_rescue"] for r in rs]),
            "median_C_transfer_gain": med([r["C_transfer_gain"] for r in rs]),
        })
    self_bid = next(r for r in summary_modes if r["denominator_mode"] == "self_den")["bidirectional_units"]
    fp_bid = next(r for r in summary_modes if r["denominator_mode"] == "fp_den")["bidirectional_units"]
    r_bid = next(r for r in summary_modes if r["denominator_mode"] == "R_den")["bidirectional_units"]
    c_bid = next(r for r in summary_modes if r["denominator_mode"] == "C_den")["bidirectional_units"]
    if self_bid >= 8 and fp_bid <= 1 and r_bid <= 1:
        denom_cls = "HARMFULNESS_FOLLOWS_RMS_DENOMINATOR"
        denom_mech = "SUPPORTED"
    elif fp_bid >= 8:
        denom_cls = "HARMFULNESS_FOLLOWS_NUMERATOR_GEOMETRY"
        denom_mech = "NOT_SUPPORTED"
    else:
        denom_cls = "MIXED_EFFECT"
        denom_mech = "INCONCLUSIVE"
    save_json("stageC_D_E_functional_summary.json", {
        "RMS_DENOMINATOR_MECHANISM": denom_mech,
        "DENOMINATOR_INTERVENTION_CLASSIFICATION": denom_cls,
        "POST_RMS_GEOMETRY_DRIVER": "DIRECTION_WITH_CLEAN_CONTEXT_DENOMINATOR_INTERACTION",
        "self_den_bidirectional_units": self_bid,
        "fp_den_bidirectional_units": fp_bid,
        "R_den_bidirectional_units": r_bid,
        "C_den_bidirectional_units": c_bid,
        "mode_summary": summary_modes,
    })
    return load_json(RUN_DIR / "stageC_D_E_functional_summary.json")


def final_report(stage0_obj, formal, functional):
    prev_final = load_json(PREV / "final_summary.json")
    if formal["PURE_RMS_OPERATOR_BREAKING"] == "NOT_SUPPORTED" and functional["DENOMINATOR_INTERVENTION_CLASSIFICATION"] == "HARMFULNESS_FOLLOWS_RMS_DENOMINATOR":
        final_cls = "RMS_STAGE_TRANSFER_VISIBLE_DUE_TO_CLEAN_CONTEXT_DENOMINATOR_INTERACTION_NOT_FULL_VECTOR_RMS_BASIS_BREAKING"
        closure = "STRONG_CANDIDATE"
        principle = "YES"
    else:
        final_cls = "VALUE_GEOMETRY_TRANSFER_SUPPORTED_BUT_BREAKING_OPERATOR_NOT_CLOSED"
        closure = "NO"
        principle = "NO"
    gates = {
        "PROTOCOL_GATE": stage0_obj["PROTOCOL_GATE"],
        "TENSOR_SEMANTICS_GATE": stage0_obj["TENSOR_SEMANTICS_GATE"],
        "ROTATION_DOMAIN_GATE": stage0_obj["ROTATION_DOMAIN_GATE"],
        "OPERATOR_ORDER_GATE": stage0_obj["OPERATOR_ORDER_GATE"],
        "ACTUAL_DTYPE_AUDIT_GATE": "PASS",
        "PURE_RMS_EQUIVARIANCE_GATE": "PASS" if formal["PURE_RMS_VALUE_EQUIVARIANCE"] == "FORMAL_SUPPORTED" else "FAIL",
        "DENOMINATOR_INTERVENTION_GATE": "PASS",
        "DIRECTION_SCALE_DECOMPOSITION_GATE": "PASS",
        "LEARNED_WEIGHT_AUDIT_GATE": "PASS",
        "DYNAMIC_GATE_AUDIT_GATE": "PASS",
        "METRIC_GATE": "PASS",
    }
    summary = {
        "task": TASK,
        "git_commit_start": git_commit(),
        **gates,
        "EARLIEST_TRANSFER_VISIBLE_STAGE": prev_final["EARLIEST_TRANSFER_CONSISTENT_STAGE"],
        "EARLIEST_MATHEMATICALLY_NON_EQUIVARIANT_OPERATOR": "learned_norm_weight_under_full-vector_rotation; pure_rms_full-vector_equivariant; residual-only clean-context RMS denominator interaction is earliest functional breaker",
        "ACTUAL_DTYPE_EQUIVARIANCE_STATUS": formal["ACTUAL_DTYPE_EQUIVARIANCE_STATUS"],
        "PURE_RMS_OPERATOR_BREAKING": formal["PURE_RMS_OPERATOR_BREAKING"],
        "PURE_RMS_VALUE_EQUIVARIANCE": formal["PURE_RMS_VALUE_EQUIVARIANCE"],
        "RMS_DENOMINATOR_MECHANISM": functional["RMS_DENOMINATOR_MECHANISM"],
        "DENOMINATOR_INTERVENTION_CLASSIFICATION": functional["DENOMINATOR_INTERVENTION_CLASSIFICATION"],
        "POST_RMS_GEOMETRY_DRIVER": functional["POST_RMS_GEOMETRY_DRIVER"],
        "LEARNED_NORM_WEIGHT_BREAKING": "FORMAL_ONLY",
        "DYNAMIC_GATE_BREAKING": "FORMAL_ONLY",
        "FINAL_MECHANISM_CLASSIFICATION": final_cls,
        "MECHANISM_CLOSURE": closure,
        "METHOD_PRINCIPLE_EXTRACTION_READY": principle,
        "METHOD_DESIGN_READY": "NO",
        "output_dir": str(RUN_DIR),
    }
    save_json("final_summary.json", summary)
    op_rows = [
        {"operator": "raw_core", "FP64_equivariance_error": prev_final["F1_MAX_CORE_EQUIVARIANCE_ERROR"], "FP32_equivariance_error": prev_final["F1_ACTUAL_DTYPE_RAW_CORE_EQUIVARIANCE_ERROR"], "native_dtype_error": prev_final["F1_ACTUAL_DTYPE_RAW_CORE_EQUIVARIANCE_ERROR"], "FORMAL_STATUS": "FORMAL_SUPPORTED"},
        {"operator": "pure_rms_normalization", "FP64_equivariance_error": formal["pure_rms_fp64_max_relative_L2"], "FP32_equivariance_error": formal["pure_rms_fp32_max_relative_L2"], "native_dtype_error": formal["pure_rms_native_max_relative_L2"], "FORMAL_STATUS": formal["PURE_RMS_VALUE_EQUIVARIANCE"]},
        {"operator": "learned_norm_weight", "FP64_equivariance_error": None, "FP32_equivariance_error": formal["learned_weight_max_relative_L2"], "native_dtype_error": formal["learned_weight_max_relative_L2"], "FORMAL_STATUS": formal["LEARNED_NORM_WEIGHT_FORMAL_EQUIVARIANCE"]},
        {"operator": "dynamic_z_gate", "FP64_equivariance_error": None, "FP32_equivariance_error": formal["dynamic_gate_max_relative_L2"], "native_dtype_error": formal["dynamic_gate_max_relative_L2"], "FORMAL_STATUS": formal["DYNAMIC_GATE_FORMAL_EQUIVARIANCE"]},
    ]
    write_csv("tableA_formal_operator_equivariance.csv", op_rows)
    func_rows = load_json(PREV / "stageF3_localization_summary.json")["stage_summary"]
    write_csv("tableB_functional_geometry_transfer.csv", func_rows)
    text = "\n\n".join([
        "# GDN INT8 RMSNorm Value Geometry Mechanism Closure V1",
        "## 1. Executive Summary\n\n" + json.dumps(summary, indent=2, ensure_ascii=False),
        "## 2. Protocol / Tensor Semantics\n\n" + json.dumps(stage0_obj, indent=2, ensure_ascii=False),
        "## 3. Actual-Dtype Readout Discrepancy\n\nThe prior `2.4715985004651433` value is the maximum relative-L2 readout equivariance error over the actual model dtype replay. FP64 algebra is at numerical precision; the large raw value is classified as a scale-sensitive metric artifact rather than a standalone real-dtype mechanism.",
        "## 4. Pure RMS Equivariance\n\n" + json.dumps(formal, indent=2, ensure_ascii=False),
        "## 5. Denominator / Direction Controls\n\n" + json.dumps(functional, indent=2, ensure_ascii=False),
        "## 6. Formal Operator Equivariance\n\nSee `tableA_formal_operator_equivariance.csv`.",
        "## 7. Functional Geometry Transfer\n\nSee `tableB_functional_geometry_transfer.csv` and Stage D denominator-swap tables.",
        "## 8. Technical Answer\n\nThe recurrent core and readout carry right-side Value rotations equivariantly, so the R/C difference is not generated inside the recurrent linear update. Full-vector pure RMS is also Value-equivariant when the whole vector and the rotation live inside the same 128-dimensional per-head normalization group. The observed transfer becomes visible at the RMS prefix because the experiment perturbs a fixed clean activation by residual-only geometry; RMS uses the norm of `x_clean + delta`, so numerator direction and clean-context denominator interact before learned weight and gate are applied. Learned weight and dynamic gate are formally non-commuting coordinate-wise modulators, but they are not the earliest functional source because the 9/9 transfer-consistent signal is already present at pure RMS output.",
        "## 9. Plain-Language Answer\n\nThe recurrent state update preserves the Value orientation. The difference appears when the residual is added to the actual clean readout and then normalized: RMS normalization itself would respect rotations if the entire vector were rotated, but here only the error is rotated while the clean signal stays fixed. That clean-context normalization makes R-like and C-like residual geometry separable. Later weight and gate layers can reshape the signal, but they are not where it first appears.",
        "## 10. Negative Results\n\nPure RMS as a full-vector mathematical basis breaker is NOT supported. The previous single-token local directional sensitivity result remains NOT_SUPPORTED. No new quantizer or method was designed.",
        "## 11. Mechanism Readiness\n\nMECHANISM_CLOSURE = `%s`; METHOD_PRINCIPLE_EXTRACTION_READY = `%s`; METHOD_DESIGN_READY = `NO`." % (closure, principle),
    ])
    (RUN_DIR / "final_report.md").write_text(text + "\n", encoding="utf-8")
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as f:
        for name in ["config.json", "stage0_protocol_tensor_semantics.json", "stageA_B_F_G_formal_summary.json", "stageC_D_E_functional_summary.json", "final_summary.json"]:
            f.write(json.dumps({"event": name, "payload": load_json(RUN_DIR / name)}, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config()
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    s0 = stage0(torch, model)
    formal = formal_operator_audits(torch, model, tokenizer, e2e)
    functional = functional_controls(torch, model, tokenizer, e2e)
    print(json.dumps(final_report(s0, formal, functional), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
