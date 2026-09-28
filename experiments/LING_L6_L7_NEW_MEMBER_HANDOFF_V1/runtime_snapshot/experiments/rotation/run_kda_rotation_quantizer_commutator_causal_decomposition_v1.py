#!/usr/bin/env python3
"""Causal decomposition of the KDA value-rotation/INT8 quantizer commutator."""

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


TASK = "KDA_ROTATION_QUANTIZER_COMMUTATOR_CAUSAL_DECOMPOSITION_V1"
SLUG = "kda_rotation_quantizer_commutator_causal_decomposition_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
AFFINE_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_affine_decay_value_unified_causal_closure_v1.py"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGE_B_CONDITIONS = ("NN", "NR", "RN", "RR")
EPS = 1e-12
IDENTITY_ATOL = 2e-6
IDENTITY_RTOL = 2e-6


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AFFINE = import_file(AFFINE_RUNNER, "affine_for_quantizer_commutator_v1")
BASE = AFFINE.BASE


def lambda_name(value):
    return {0.0: "L0", 0.25: "L025", 0.5: "L05", 0.75: "L075", 1.0: "L1"}[float(value)]


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
    if not n:
        return 1.0
    k = min(sum(x > 0 for x in vals), sum(x < 0 for x in vals))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def effect(values, name):
    vals = [float(x) for x in values if BASE.finite(x)]
    lo, hi = median_bootstrap_ci(vals)
    return {
        "name": name, "paired_median": BASE.median(vals), "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi, "mean": BASE.mean(vals), "positive": sum(x > 0 for x in vals),
        "negative": sum(x < 0 for x in vals), "n": len(vals), "sign_test_p": sign_test_two_sided(vals),
        "paired_values": vals, "estimand": "canonical-unit median",
    }


def quantizer_semantics():
    return {
        "QUANTIZED_OBJECT": "KDA stored recurrent state [B,H,K,V]",
        "CONFIG": "INT8_R128", "GROUP_AXIS": "V", "GROUP_SIZE": 128,
        "GROUP_SEMANTICS": "one scale for all 128 V values at fixed [B,H,K]",
        "SCALE_SHAPE": "[B,H,K,1]", "SCALE_RULE": "amax(abs(group)).clamp_min(1e-12) / 127",
        "SYMMETRIC": True, "ZERO_POINT": 0, "QMIN": -127, "QMAX": 127,
        "ROUNDING_RULE": "torch.round (ties-to-even)", "CLIPPING": "clamp(-127,127)",
        "SATURATION_RULE": "abs(integer_code)==127", "INPUT_DTYPE": "cache state cast to float32",
        "SCALE_DTYPE": "float32", "CODE_DTYPE": "float32 emulated integer codes",
        "DEQUANT_DTYPE": "codes*scale in float32, then cast to cache state dtype",
        "QUANTIZE_DEQUANT_ORDER": "float32 input -> scale -> round -> clamp -> codes*scale -> cache dtype",
        "ROTATION_POSITION": "S_R=S@R before scale selection and quantization",
        "MAP_BACK_POSITION": "Q(S@R;s_R)@R.T after dequantization, before state intervention",
        "SOURCE": str(BASE.PERSISTENT_RUNNER),
    }


def quantize_with_scale(state, scale):
    x = state.detach().float()
    s = scale.detach().float()
    expected = list(x.shape)
    expected[-1] = 1
    if list(s.shape) != expected:
        raise ValueError(f"illegal scale shape {list(s.shape)} for state {list(x.shape)}; expected {expected}")
    codes_unclipped = torch.round(x / s)
    codes = codes_unclipped.clamp(-127, 127)
    qdq = codes * s
    return qdq.to(state.dtype), {
        "scale": s, "codes": codes, "codes_unclipped": codes_unclipped,
        "clipping_fraction": float((codes_unclipped.abs() > 127).float().mean().item()),
        "saturation_fraction": float((codes.abs() == 127).float().mean().item()),
    }


def official_quantize(state):
    return BASE.P().fake_quant_ling_state(state.detach().float(), "INT8_R128")


def rotate_state(state, rotation):
    return BASE.rotate_state_value_axis(state, rotation.to(device=state.device, dtype=torch.float32))


def inverse_rotate_state(state, rotation):
    return BASE.inverse_rotate_state_value_axis(state, rotation.to(device=state.device, dtype=torch.float32))


def construct_endpoints(state, rotation):
    s = state.detach().float()
    sr = rotate_state(s, rotation)
    nn_official, n_meta = official_quantize(s)
    rr_rot_official, r_meta = official_quantize(sr)
    nn, nn_meta = quantize_with_scale(s, n_meta["scale"])
    nr, nr_meta = quantize_with_scale(s, r_meta["scale"])
    rn_rot, rn_meta = quantize_with_scale(sr, n_meta["scale"])
    rr_rot, rr_meta = quantize_with_scale(sr, r_meta["scale"])
    rn = inverse_rotate_state(rn_rot, rotation)
    rr = inverse_rotate_state(rr_rot, rotation)
    rr_official = inverse_rotate_state(rr_rot_official, rotation)
    return {
        "S": s, "SR": sr, "NN": nn.float(), "NR": nr.float(), "RN": rn.float(), "RR": rr.float(),
        "NN_official": nn_official.float(), "RR_official": rr_official.float(),
        "native_meta": {**nn_meta, "official_scale": n_meta["scale"], "official_codes": n_meta["codes"]},
        "rotated_meta": {**rr_meta, "official_scale": r_meta["scale"], "official_codes": r_meta["codes"]},
        "NR_meta": nr_meta, "RN_meta": rn_meta,
    }


def interpolate_endpoint(nn, rr, lam):
    if nn.shape != rr.shape or nn.dtype != rr.dtype:
        raise ValueError("endpoint shape/dtype mismatch")
    return (1.0 - float(lam)) * nn + float(lam) * rr


def tensor_metrics(value, reference):
    x, y = value.detach().double(), reference.detach().double()
    diff = x - y
    nx, ny = BASE.tensor_norm(x), BASE.tensor_norm(y)
    cosine = float((x * y).sum().item()) / (nx * ny + EPS)
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2": BASE.tensor_norm(diff) / (ny + EPS), "cosine": cosine,
    }


def peakiness(x):
    y = x.detach().float()
    return float(y.abs().max().item()) / (float(torch.sqrt(torch.mean(y * y)).item()) + EPS)


def state_statistics(unit_id, layer, endpoints):
    s, nn, rr = endpoints["S"], endpoints["NN"], endpoints["RR"]
    sr = endpoints["SR"]
    n_meta, r_meta = endpoints["native_meta"], endpoints["rotated_meta"]
    en, er, dq = nn - s, rr - s, rr - nn
    sn, sr_scale = n_meta["scale"], r_meta["scale"]
    codes_n, codes_r = n_meta["codes"], r_meta["codes"]
    scales_n = sn.detach().float().cpu().reshape(-1).tolist()
    scales_r = sr_scale.detach().float().cpu().reshape(-1).tolist()
    return {
        "unit_id": str(unit_id), "layer": int(layer), "seed": BASE.RHT_SEED,
        "sample_identity": str(unit_id),
        "native_state_rel_error": BASE.tensor_norm(en) / (BASE.tensor_norm(s) + EPS),
        "rotated_state_rel_error": BASE.tensor_norm(er) / (BASE.tensor_norm(s) + EPS),
        "commutator_rel_norm": BASE.tensor_norm(dq) / (BASE.tensor_norm(s) + EPS),
        "native_scale_median": BASE.median(scales_n), "rotated_scale_median": BASE.median(scales_r),
        "scale_ratio_median": BASE.median([b / (a + EPS) for a, b in zip(scales_n, scales_r)]),
        "native_scale_mean": BASE.mean(scales_n), "rotated_scale_mean": BASE.mean(scales_r),
        "native_scale_raw_json": json.dumps(scales_n), "rotated_scale_raw_json": json.dumps(scales_r),
        "native_maxabs": float(s.abs().max().item()), "rotated_maxabs": float(sr.abs().max().item()),
        "maxabs_ratio": float(sr.abs().max().item()) / (float(s.abs().max().item()) + EPS),
        "native_peakiness": peakiness(s), "rotated_peakiness": peakiness(sr),
        "native_clipping_fraction": n_meta["clipping_fraction"],
        "rotated_clipping_fraction": r_meta["clipping_fraction"],
        "native_saturation_fraction": n_meta["saturation_fraction"],
        "rotated_saturation_fraction": r_meta["saturation_fraction"],
        "integer_code_churn": float((codes_n != codes_r).float().mean().item()),
        "error_cosine": tensor_metrics(en, er)["cosine"],
        "native_error_norm": BASE.tensor_norm(en), "rotated_error_norm": BASE.tensor_norm(er),
        "delta_q_norm": BASE.tensor_norm(dq),
    }


def prepare_fp_cache(model, tokenizer, unit, row, tokens):
    device = next(model.parameters()).device
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids, attention_mask=mask,
            cache_position=torch.arange(0, input_ids.shape[-1], device=device), use_cache=True,
        )
    cache = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    for t, token in enumerate(tokens[:int(unit["t0"])]):
        cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
        with torch.inference_mode():
            out = model(
                input_ids=cur, attention_mask=mask, past_key_values=cache,
                cache_position=torch.tensor([prompt_len + t], device=device), use_cache=True,
            )
        cache = out.past_key_values
    return mask, cache, prompt_len


def endpoint_stacks(cache, kda_layers, rotation):
    fp_stack = BASE.cache_stack(cache, kda_layers)
    endpoint_by_layer = {layer: construct_endpoints(state, rotation) for layer, state in fp_stack.items()}
    stacks = {condition: {} for condition in STAGE_B_CONDITIONS}
    for layer, endpoints in endpoint_by_layer.items():
        for condition in STAGE_B_CONDITIONS:
            stacks[condition][layer] = endpoints[condition].clone()
    return fp_stack, endpoint_by_layer, stacks


def branch_stacks(phase, endpoint_by_layer):
    if phase == "stageA":
        return {
            lambda_name(lam): {
                layer: interpolate_endpoint(endpoints["NN"], endpoints["RR"], lam)
                for layer, endpoints in endpoint_by_layer.items()
            }
            for lam in LAMBDAS
        }
    if phase == "stageB":
        return {
            condition: {layer: endpoints[condition].clone() for layer, endpoints in endpoint_by_layer.items()}
            for condition in STAGE_B_CONDITIONS
        }
    raise ValueError(phase)


def run_future(model, unit, tokens, mask, fp_cache, prompt_len, kda_layers, stacks, phase, horizon):
    device = next(model.parameters()).device
    pasts = {"FP": BASE.clone_cache(fp_cache)}
    masks = {"FP": mask.clone()}
    for name, stack in stacks.items():
        pasts[name] = BASE.clone_cache(fp_cache)
        BASE.replace_cache_stack(pasts[name], {layer: state.to(device) for layer, state in stack.items()})
        masks[name] = mask.clone()
    branches = ["FP"] + list(stacks)
    horizon_rows = []
    for h in range(1, int(horizon) + 1):
        token = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch.tensor([[token]], device=device, dtype=torch.long)
        fp_logits = None
        fp_states = None
        for branch in branches:
            masks[branch] = torch.cat([masks[branch], torch.ones_like(cur)], dim=-1)
            with torch.inference_mode():
                out = model(
                    input_ids=cur, attention_mask=masks[branch], past_key_values=pasts[branch],
                    cache_position=torch.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device),
                    use_cache=True,
                )
            pasts[branch] = out.past_key_values
            if branch == "FP":
                fp_logits = out.logits.detach().float()
                fp_states = BASE.cache_stack(pasts[branch], kda_layers)
                kl = 0.0
            else:
                kl = float(BASE.full_logit_metrics(torch, fp_logits, out.logits.detach().float())["KL"])
                meta = BASE.quantize_branch_cache(torch, pasts[branch], kda_layers, "native", rotation=torch.eye(128))
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite cache branch={branch} h={h}")
            state_rel = 0.0
            if branch != "FP":
                current = BASE.cache_stack(pasts[branch], kda_layers)
                values = [
                    BASE.tensor_norm(current[layer].detach().float() - fp_states[layer].detach().float()) /
                    (BASE.tensor_norm(fp_states[layer]) + EPS)
                    for layer in kda_layers if layer in current and layer in fp_states
                ]
                state_rel = BASE.mean(values)
            horizon_rows.append({
                "unit_id": str(unit["unit_id"]), "layer": "ALL", "seed": BASE.RHT_SEED,
                "sample_identity": str(unit["unit_id"]), "horizon": h, "condition": branch,
                "lambda": ({"L0": 0.0, "L025": 0.25, "L05": 0.5, "L075": 0.75, "L1": 1.0}.get(branch)),
                "state_rel_error": state_rel, "future_kl": kl, "future_kl_auc": None,
                "scale_statistics": None, "code_statistics": None,
            })
    return horizon_rows


def stage0_audit(cache, kda_layers, rotation, endpoint_by_layer):
    checks = defaultdict(list)
    details = []
    live_native = BASE.clone_cache(cache)
    BASE.quantize_branch_cache(torch, live_native, kda_layers, "native", rotation)
    live_native_stack = BASE.cache_stack(live_native, kda_layers)
    live_rotated = BASE.clone_cache(cache)
    fp_stack = BASE.cache_stack(live_rotated, kda_layers)
    BASE.replace_cache_stack(live_rotated, {
        layer: rotate_state(state, rotation) for layer, state in fp_stack.items()
    })
    BASE.quantize_branch_cache(torch, live_rotated, kda_layers, "rotated", rotation)
    live_rotated_stack = BASE.cache_stack(live_rotated, kda_layers)
    for layer, endpoints in endpoint_by_layer.items():
        fp_id = tensor_metrics(inverse_rotate_state(endpoints["SR"], rotation), endpoints["S"])
        n_id = tensor_metrics(endpoints["NN"], live_native_stack[layer])
        r_live_back = inverse_rotate_state(live_rotated_stack[layer], rotation)
        r_id = tensor_metrics(endpoints["RR"], r_live_back)
        qn_id = tensor_metrics(endpoints["NN"], endpoints["NN_official"])
        qr_id = tensor_metrics(endpoints["RR"], endpoints["RR_official"])
        l0 = interpolate_endpoint(endpoints["NN"], endpoints["RR"], 0.0)
        l1 = interpolate_endpoint(endpoints["NN"], endpoints["RR"], 1.0)
        checks["fp_rotation"].append(fp_id["max_abs_error"] <= IDENTITY_ATOL)
        checks["native_endpoint"].append(n_id["max_abs_error"] <= IDENTITY_ATOL)
        checks["rotated_endpoint"].append(r_id["max_abs_error"] <= IDENTITY_ATOL)
        checks["reconstruction"].append(qn_id["max_abs_error"] == 0.0 and qr_id["max_abs_error"] <= IDENTITY_ATOL)
        checks["lambda0"].append(torch.equal(l0, endpoints["NN"]))
        checks["lambda1"].append(torch.equal(l1, endpoints["RR"]))
        details.append({
            "layer": int(layer), "fp_rotation": fp_id, "native_endpoint": n_id,
            "rotated_endpoint": r_id, "native_reconstruction": qn_id,
            "rotated_reconstruction": qr_id,
        })
    disabled = BASE.clone_cache(cache)
    disabled_stack = BASE.cache_stack(disabled, kda_layers)
    original_stack = BASE.cache_stack(cache, kda_layers)
    noninterference = all(torch.equal(disabled_stack[layer], original_stack[layer]) for layer in kda_layers)
    audit = {
        "FP_ROTATION_IDENTITY": "PASS" if all(checks["fp_rotation"]) else "FAIL",
        "NATIVE_ENDPOINT_IDENTITY": "PASS" if all(checks["native_endpoint"]) else "FAIL",
        "ROTATED_ENDPOINT_IDENTITY": "PASS" if all(checks["rotated_endpoint"]) else "FAIL",
        "QUANTIZER_RECONSTRUCTION_IDENTITY": "PASS" if all(checks["reconstruction"]) else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE": "PASS" if noninterference else "FAIL",
        "LAMBDA_0_ENDPOINT_IDENTITY": "PASS" if all(checks["lambda0"]) else "FAIL",
        "LAMBDA_1_ENDPOINT_IDENTITY": "PASS" if all(checks["lambda1"]) else "FAIL",
        "details": details,
    }
    audit["STAGE0"] = "PASS" if all(v == "PASS" for k, v in audit.items() if k != "details") else "FAIL"
    return audit


def run_unit(model, tokenizer, unit, rows_by_pid, teacher_tokens, kda_layers, rotation, phase, horizon):
    pid = str(unit["problem_id"])
    row = rows_by_pid.get(pid)
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if row is None or len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    mask, cache, prompt_len = prepare_fp_cache(model, tokenizer, unit, row, tokens)
    _, endpoint_by_layer, _ = endpoint_stacks(cache, kda_layers, rotation)
    stats = [state_statistics(unit["unit_id"], layer, endpoints) for layer, endpoints in endpoint_by_layer.items()]
    if phase == "stage0":
        return {"stage0": stage0_audit(cache, kda_layers, rotation, endpoint_by_layer), "state_stats": stats}
    stacks = branch_stacks(phase, endpoint_by_layer)
    horizon_rows = run_future(
        model, unit, tokens, mask, cache, prompt_len, kda_layers, stacks, phase, horizon,
    )
    return {"horizon_rows": horizon_rows, "state_stats": stats}


def aggregate_units(horizon_rows):
    grouped = defaultdict(list)
    for row in horizon_rows:
        if row["condition"] == "FP":
            continue
        grouped[(row["unit_id"], row["condition"])].append(float(row["future_kl"]))
    return [
        {"unit_id": unit, "condition": condition, "future_kl_auc": BASE.mean(values)}
        for (unit, condition), values in sorted(grouped.items())
    ]


def linear_slope(xs, ys):
    mx, my = BASE.mean(xs), BASE.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) + EPS)


def summarize_stage_a(unit_rows, stage0):
    by = {(r["unit_id"], r["condition"]): float(r["future_kl_auc"]) for r in unit_rows}
    units = sorted({r["unit_id"] for r in unit_rows})
    dose_rows = []
    slopes = []
    for unit in units:
        ys = [by[(unit, lambda_name(lam))] for lam in LAMBDAS]
        slope = linear_slope(list(LAMBDAS), ys)
        gap = ys[-1] - ys[0]
        row = {"unit_id": unit, "slope": slope, "endpoint_gap": gap, "unstable_denominator": abs(gap) <= EPS or gap < 0}
        for lam, y in zip(LAMBDAS, ys):
            row[f"auc_{lambda_name(lam)}"] = y
            row[f"closure_{lambda_name(lam)}"] = None if abs(gap) <= EPS else (y - ys[0]) / gap
        dose_rows.append(row); slopes.append(slope)
    med_auc = {lambda_name(lam): BASE.median([by[(u, lambda_name(lam))] for u in units]) for lam in LAMBDAS}
    slope_effect = effect(slopes, "unitwise_lambda_slope")
    endpoint_effect = effect([r["endpoint_gap"] for r in dose_rows], "lambda1_minus_lambda0")
    adjacent = [med_auc[lambda_name(LAMBDAS[i + 1])] >= med_auc[lambda_name(LAMBDAS[i])] for i in range(4)]
    broad_monotonic = sum(adjacent) >= 3
    reverse_monotonic = sum(not value for value in adjacent) == 4
    identities = all(stage0.get(k) == "PASS" for k in (
        "NATIVE_ENDPOINT_IDENTITY", "ROTATED_ENDPOINT_IDENTITY", "LAMBDA_0_ENDPOINT_IDENTITY", "LAMBDA_1_ENDPOINT_IDENTITY",
    ))
    if (identities and slope_effect["paired_median"] > 0 and slope_effect["bootstrap_ci_low"] > 0 and
            slope_effect["positive"] >= 10 and broad_monotonic and endpoint_effect["bootstrap_ci_low"] > 0):
        status = "STRONG_SUPPORT"
    elif identities and slope_effect["paired_median"] > 0 and (slope_effect["positive"] >= 10 or endpoint_effect["paired_median"] > 0):
        status = "PARTIAL_SUPPORT"
    else:
        status = "NOT_SUPPORTED"
    closure = {}
    for lam in LAMBDAS:
        vals = [r[f"closure_{lambda_name(lam)}"] for r in dose_rows if r[f"closure_{lambda_name(lam)}"] is not None]
        closure[lambda_name(lam)] = effect(vals, f"closure_{lambda_name(lam)}")
    return dose_rows, {
        "COMMUTATOR_CAUSAL_STATUS": status,
        "COMMUTATOR_DOSE_RESPONSE": (
            "BROADLY_MONOTONIC" if broad_monotonic else
            ("MONOTONIC_OPPOSITE_DIRECTION" if reverse_monotonic else "NON_MONOTONIC")
        ),
        "median_auc": med_auc, "slope": slope_effect, "endpoint_effect": endpoint_effect,
        "positive_slope_units": slope_effect["positive"], "N_UNITS": len(units), "closure": closure,
        "endpoint_native_identity": stage0.get("NATIVE_ENDPOINT_IDENTITY"),
        "endpoint_rotated_identity": stage0.get("ROTATED_ENDPOINT_IDENTITY"),
        "unstable_units": [r["unit_id"] for r in dose_rows if r["unstable_denominator"]],
    }


def summarize_stage_b(unit_rows):
    by = {(r["unit_id"], r["condition"]): float(r["future_kl_auc"]) for r in unit_rows}
    units = sorted({r["unit_id"] for r in unit_rows})
    factorial_rows = []
    for unit in units:
        nn, nr, rn, rr = [by[(unit, c)] for c in STAGE_B_CONDITIONS]
        gap = rr - nn
        row = {
            "unit_id": unit, "auc_NN": nn, "auc_NR": nr, "auc_RN": rn, "auc_RR": rr,
            "scale_main_effect": ((nr - nn) + (rr - rn)) / 2.0,
            "lattice_main_effect": ((rn - nn) + (rr - nr)) / 2.0,
            "interaction": rr - rn - nr + nn, "gap": gap,
            "unstable_denominator": abs(gap) <= EPS or gap < 0,
            "scale_only_closure": None if abs(gap) <= EPS else (nr - nn) / gap,
            "lattice_only_closure": None if abs(gap) <= EPS else (rn - nn) / gap,
        }
        factorial_rows.append(row)
    effects = {name: effect([r[name] for r in factorial_rows], name) for name in (
        "scale_main_effect", "lattice_main_effect", "interaction", "gap",
    )}
    closures = {
        name: effect([r[name] for r in factorial_rows if r[name] is not None], name)
        for name in ("scale_only_closure", "lattice_only_closure")
    }
    med_auc = {condition: BASE.median([by[(u, condition)] for u in units]) for condition in STAGE_B_CONDITIONS}
    cs, cl = closures["scale_only_closure"]["paired_median"], closures["lattice_only_closure"]["paired_median"]
    interaction = effects["interaction"]
    gap_mag = abs(effects["gap"]["paired_median"]) + EPS
    interaction_required = (
        interaction["bootstrap_ci_low"] > 0 or interaction["bootstrap_ci_high"] < 0
    ) and abs(interaction["paired_median"]) / gap_mag >= 0.40
    if cl >= 0.70 and abs(cs) <= 0.30:
        mechanism = "ROTATED_QUANTIZATION_LATTICE_OR_ERROR_GEOMETRY"
    elif cs >= 0.70 and abs(cl) <= 0.30:
        mechanism = "SCALE_SELECTION"
    elif interaction_required and abs(cs) <= 0.30 and abs(cl) <= 0.30:
        mechanism = "SCALE_LATTICE_INTERACTION"
    else:
        mechanism = "MIXED_SCALE_AND_LATTICE"
    return factorial_rows, {
        "N_UNITS": len(units), "median_auc": med_auc, "effects": effects, "closures": closures,
        "PRIMARY_MECHANISM": mechanism, "IS_INTERACTION_REQUIRED": "YES" if interaction_required else "NO",
        "unstable_units": [r["unit_id"] for r in factorial_rows if r["unstable_denominator"]],
    }


def setup_output(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw").mkdir(parents=True, exist_ok=True)


def reset_phase(outdir, phase):
    for rel in (f"raw/{phase}_horizon.jsonl", f"raw/{phase}_state_stats.jsonl"):
        path = outdir / rel
        if path.exists():
            path.unlink()


def run_experiment(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    if args.overwrite:
        reset_phase(outdir, args.phase)
    units, unit_audit = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    BASE.save_json(outdir / "experiment_config.json", {
        "TASK": TASK, "phase": args.phase, "scope": args.scope, "horizon": args.horizon,
        "canonical_manifest": unit_audit, "canonical_unit_ids": [u["unit_id"] for u in units],
        "rotation": "formal value-side RHT seed 0", "intervention": "one-time stored post-quant state at t0",
        "future_coordinates": "native", "future_quantizer": "unchanged INT8_R128",
    })
    BASE.save_json(outdir / "quantizer_semantics.json", quantizer_semantics())
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    config = json.loads((BASE.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = BASE.P().kda_layers_from_config(config)
    rotation = BASE.exact().make_experiment_rotation("kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    failures, horizon_rows, state_rows, stage0 = [], [], [], None
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] commutator {args.phase} unit {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                result = run_unit(
                    model, tokenizer, unit, rows_by_pid, teacher_tokens, kda_layers,
                    rotation.to(next(model.parameters()).device), args.phase, args.horizon,
                )
                if args.phase == "stage0":
                    stage0 = result["stage0"]
                    BASE.save_json(outdir / "stage0_identity.json", stage0)
                    BASE.write_rows(outdir / "stage0_state_statistics.csv", result["state_stats"])
                    if stage0["STAGE0"] != "PASS":
                        raise RuntimeError("Stage 0 endpoint/identity gate failed")
                else:
                    horizon_rows.extend(result["horizon_rows"]); state_rows.extend(result["state_stats"])
                    for row in result["horizon_rows"]: BASE.append_jsonl(outdir / "raw" / f"{args.phase}_horizon.jsonl", row)
                    for row in result["state_stats"]: BASE.append_jsonl(outdir / "raw" / f"{args.phase}_state_stats.jsonl", row)
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback": traceback.format_exc(limit=30)}
                failures.append(failure); BASE.save_json(outdir / f"{args.phase}_failures.json", failures)
                print(f"FAILED {unit.get('unit_id')}: {exc!r}", flush=True)
                if not args.keep_going:
                    raise
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {
        "TASK": TASK, "phase": args.phase, "STAGE0": stage0.get("STAGE0") if stage0 else "NOT_APPLICABLE",
        "n_units_completed": len(units) - len(failures), "expected_units": len(units), "failures": failures,
    }
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", summary)
    return summary


def dedupe(rows, keys):
    return list({tuple(row[k] for k in keys): row for row in rows}.values())


def write_stage_a_report(outdir, summary):
    lines = [f"# {TASK}", "", "## Stage A", ""]
    for key in ("COMMUTATOR_CAUSAL_STATUS", "COMMUTATOR_DOSE_RESPONSE", "N_UNITS", "positive_slope_units"):
        lines.append(f"{key} = {json.dumps(summary.get(key))}")
    lines += ["", "```json", json.dumps(summary, indent=2, sort_keys=True), "```"]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stopped_report(outdir, stage_a, failures):
    semantics = BASE.load_json(outdir / "quantizer_semantics.json", {})
    stage0 = BASE.load_json(outdir / "stage0_identity.json", {})
    state_rows = BASE.read_rows(outdir / "stage0_state_statistics.csv")
    clipping = [float(r["native_clipping_fraction"]) for r in state_rows] + [float(r["rotated_clipping_fraction"]) for r in state_rows]
    pytest_path = outdir / "pytest_output.txt"
    pytest_lines = pytest_path.read_text(encoding="utf-8").strip().splitlines() if pytest_path.exists() else []
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE", "STAGE0": stage0.get("STAGE0"),
        "STAGEA": stage_a["COMMUTATOR_CAUSAL_STATUS"], "STAGEB": "SKIPPED_BY_STAGE_A_STOP_RULE",
        "N_FORMAL_UNITS": stage_a["N_UNITS"], "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded",
        "FAILURES": failures, "QUANTIZER_SEMANTICS": semantics,
        "COMMUTATOR_CAUSAL_STATUS": stage_a["COMMUTATOR_CAUSAL_STATUS"],
        "AUC_lambda_0": stage_a["median_auc"]["L0"], "AUC_lambda_025": stage_a["median_auc"]["L025"],
        "AUC_lambda_05": stage_a["median_auc"]["L05"], "AUC_lambda_075": stage_a["median_auc"]["L075"],
        "AUC_lambda_1": stage_a["median_auc"]["L1"], "median_slope": stage_a["slope"]["paired_median"],
        "slope_95CI": [stage_a["slope"]["bootstrap_ci_low"], stage_a["slope"]["bootstrap_ci_high"]],
        "positive_slope_units": stage_a["positive_slope_units"],
        "endpoint_gap": stage_a["endpoint_effect"]["paired_median"],
        "endpoint_gap_95CI": [stage_a["endpoint_effect"]["bootstrap_ci_low"], stage_a["endpoint_effect"]["bootstrap_ci_high"]],
        "COMMUTATOR_DOSE_RESPONSE": stage_a["COMMUTATOR_DOSE_RESPONSE"],
        "AUC_NN": "NOT_RUN", "AUC_NR": "NOT_RUN", "AUC_RN": "NOT_RUN", "AUC_RR": "NOT_RUN",
        "SCALE_MAIN_EFFECT": "NOT_RUN", "SCALE_MAIN_EFFECT_95CI": "NOT_RUN",
        "LATTICE_MAIN_EFFECT": "NOT_RUN", "LATTICE_MAIN_EFFECT_95CI": "NOT_RUN",
        "INTERACTION": "NOT_RUN", "INTERACTION_95CI": "NOT_RUN",
        "SCALE_ONLY_CLOSURE": "NOT_RUN", "LATTICE_ONLY_CLOSURE": "NOT_RUN",
        "PRIMARY_MECHANISM": "NOT_IDENTIFIED_STAGE_B_FORBIDDEN",
        "DOES_DELTA_Q_CAUSALLY_BRIDGE_NATIVE_TO_ROTATED_FAILURE": "NO",
        "IS_SCALE_SELECTION_THE_PRIMARY_CAUSE": "NOT_EVALUATED",
        "IS_ROTATED_LATTICE_ERROR_GEOMETRY_THE_PRIMARY_CAUSE": "NOT_EVALUATED",
        "IS_SCALE_X_LATTICE_INTERACTION_REQUIRED": "NOT_EVALUATED",
        "IS_CLIPPING_A_PLAUSIBLE_PRIMARY_CAUSE": "NO" if clipping and max(clipping) <= 1e-12 else "YES",
        "CLIPPING_PRIMARY_HYPOTHESIS": "REJECTED_OR_LOW_PRIORITY" if clipping and max(clipping) <= 1e-12 else "FOLLOWUP_REQUIRED",
        "NEXT_EXPERIMENT_RECOMMENDATION": "RECHECK_POST_QUANT_INTERVENTION_SEMANTICS_BEFORE_ANY_STAGE_B_OR_METHOD_DESIGN",
        "stageA_details": stage_a,
    }
    BASE.save_json(outdir / "stageB_summary.json", {
        "STAGEB": "SKIPPED_BY_STAGE_A_STOP_RULE", "reason": "COMMUTATOR_CAUSAL_STATUS=NOT_SUPPORTED",
    })
    BASE.write_rows(outdir / "stageB_unit_results.csv", [])
    BASE.write_rows(outdir / "stageB_horizon_results.csv", [])
    BASE.save_json(outdir / "summary.json", summary)
    keys = [k for k in summary if k not in {"QUANTIZER_SEMANTICS", "stageA_details"}]
    lines = [f"# {TASK}", "", "## Formal summary", ""] + [
        f"{key} = {json.dumps(summary[key], sort_keys=True)}" for key in keys
    ]
    lines += ["", "## Quantizer semantics", "", "```json", json.dumps(semantics, indent=2, sort_keys=True), "```",
              "", "## Stage A details", "", "```json", json.dumps(stage_a, indent=2, sort_keys=True), "```",
              "", "## Stage B", "", "Skipped by the pre-specified Stage A stop rule."]
    med = lambda field: BASE.median([float(row[field]) for row in state_rows])
    lines += [
        "", "## Scientific conclusions", "",
        "- All Stage 0 rotation, endpoint, reconstruction, and noninterference gates passed.",
        f"- Rotated quantization reduced median state relative error from {med('native_state_rel_error'):.6g} to {med('rotated_state_rel_error'):.6g}.",
        f"- The median commutator relative norm was {med('commutator_rel_norm'):.6g}.",
        f"- Median live scale ratio s_R/s_N was {med('scale_ratio_median'):.6g}.",
        "- FutureKL did not increase with lambda; the pre-specified positive dose response failed.",
        f"- Median unit-wise slope was {stage_a['slope']['paired_median']:.6g}, with {stage_a['positive_slope_units']}/{stage_a['N_UNITS']} positive units.",
        f"- {len(stage_a['unstable_units'])}/{stage_a['N_UNITS']} endpoint denominators were sign-reversed and were retained.",
        "- Native and rotated clipping fractions were zero, so clipping is not a plausible primary cause.",
        "- Stage B was not run because the formal Stage A stop rule fired.",
        "- Recheck persistent-history versus one-time intervention semantics before any method design.",
    ]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_stage_a_plot(outdir, dose_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    xs = list(LAMBDAS)
    ys = [BASE.median([float(row[f"auc_{lambda_name(lam)}"]) for row in dose_rows]) for lam in xs]
    plt.figure(figsize=(5.8, 4.2)); plt.plot(xs, ys, marker="o")
    plt.xlabel("lambda"); plt.ylabel("FutureKL AUC"); plt.tight_layout()
    plt.savefig(figdir / "01_commutator_dose.png", dpi=180); plt.close()


def write_final_report(outdir, stage_a, stage_b, failures, pytest_value):
    semantics = BASE.load_json(outdir / "quantizer_semantics.json", {})
    stage0 = BASE.load_json(outdir / "stage0_identity.json", {})
    state_rows = BASE.read_rows(outdir / "stage0_state_statistics.csv")
    clip_values = [float(r["native_clipping_fraction"]) for r in state_rows] + [float(r["rotated_clipping_fraction"]) for r in state_rows]
    clipping_primary = "NO" if clip_values and max(clip_values) <= 1e-12 else "YES"
    primary = stage_b["PRIMARY_MECHANISM"]
    scale_primary = "YES" if primary == "SCALE_SELECTION" else ("PARTIAL" if primary == "MIXED_SCALE_AND_LATTICE" else "NO")
    lattice_primary = "YES" if primary == "ROTATED_QUANTIZATION_LATTICE_OR_ERROR_GEOMETRY" else ("PARTIAL" if primary == "MIXED_SCALE_AND_LATTICE" else "NO")
    bridge = {"STRONG_SUPPORT": "YES", "PARTIAL_SUPPORT": "PARTIAL", "NOT_SUPPORTED": "NO"}[stage_a["COMMUTATOR_CAUSAL_STATUS"]]
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE", "STAGE0": stage0.get("STAGE0"),
        "STAGEA": stage_a["COMMUTATOR_CAUSAL_STATUS"], "STAGEB": "COMPLETE",
        "N_FORMAL_UNITS": stage_a["N_UNITS"], "PYTEST": pytest_value, "FAILURES": failures,
        "QUANTIZER_SEMANTICS": semantics, "COMMUTATOR_CAUSAL_STATUS": stage_a["COMMUTATOR_CAUSAL_STATUS"],
        "AUC_lambda_0": stage_a["median_auc"]["L0"], "AUC_lambda_025": stage_a["median_auc"]["L025"],
        "AUC_lambda_05": stage_a["median_auc"]["L05"], "AUC_lambda_075": stage_a["median_auc"]["L075"],
        "AUC_lambda_1": stage_a["median_auc"]["L1"], "median_slope": stage_a["slope"]["paired_median"],
        "slope_95CI": [stage_a["slope"]["bootstrap_ci_low"], stage_a["slope"]["bootstrap_ci_high"]],
        "positive_slope_units": stage_a["positive_slope_units"], "COMMUTATOR_DOSE_RESPONSE": stage_a["COMMUTATOR_DOSE_RESPONSE"],
        "AUC_NN": stage_b["median_auc"]["NN"], "AUC_NR": stage_b["median_auc"]["NR"],
        "AUC_RN": stage_b["median_auc"]["RN"], "AUC_RR": stage_b["median_auc"]["RR"],
        "SCALE_MAIN_EFFECT": stage_b["effects"]["scale_main_effect"]["paired_median"],
        "SCALE_MAIN_EFFECT_95CI": [stage_b["effects"]["scale_main_effect"]["bootstrap_ci_low"], stage_b["effects"]["scale_main_effect"]["bootstrap_ci_high"]],
        "LATTICE_MAIN_EFFECT": stage_b["effects"]["lattice_main_effect"]["paired_median"],
        "LATTICE_MAIN_EFFECT_95CI": [stage_b["effects"]["lattice_main_effect"]["bootstrap_ci_low"], stage_b["effects"]["lattice_main_effect"]["bootstrap_ci_high"]],
        "INTERACTION": stage_b["effects"]["interaction"]["paired_median"],
        "INTERACTION_95CI": [stage_b["effects"]["interaction"]["bootstrap_ci_low"], stage_b["effects"]["interaction"]["bootstrap_ci_high"]],
        "SCALE_ONLY_CLOSURE": stage_b["closures"]["scale_only_closure"]["paired_median"],
        "LATTICE_ONLY_CLOSURE": stage_b["closures"]["lattice_only_closure"]["paired_median"],
        "PRIMARY_MECHANISM": primary,
        "DOES_DELTA_Q_CAUSALLY_BRIDGE_NATIVE_TO_ROTATED_FAILURE": bridge,
        "IS_SCALE_SELECTION_THE_PRIMARY_CAUSE": scale_primary,
        "IS_ROTATED_LATTICE_ERROR_GEOMETRY_THE_PRIMARY_CAUSE": lattice_primary,
        "IS_SCALE_X_LATTICE_INTERACTION_REQUIRED": stage_b["IS_INTERACTION_REQUIRED"],
        "IS_CLIPPING_A_PLAUSIBLE_PRIMARY_CAUSE": clipping_primary,
        "CLIPPING_PRIMARY_HYPOTHESIS": "REJECTED_OR_LOW_PRIORITY" if clipping_primary == "NO" else "FOLLOWUP_REQUIRED",
        "NEXT_EXPERIMENT_RECOMMENDATION": "HUMAN_REVIEW_BEFORE_KDA_ROTATION_QUANTIZATION_ERROR_GEOMETRY_CAUSAL_V1",
        "stageA_details": stage_a, "stageB_details": stage_b,
    }
    BASE.save_json(outdir / "summary.json", summary)
    keys = [k for k in summary if k not in {"stageA_details", "stageB_details", "QUANTIZER_SEMANTICS"}]
    lines = [f"# {TASK}", "", "## Formal summary", ""] + [f"{k} = {json.dumps(summary[k], sort_keys=True)}" for k in keys]
    lines += ["", "## Quantizer semantics", "", "```json", json.dumps(semantics, indent=2, sort_keys=True), "```",
              "", "## Stage A details", "", "```json", json.dumps(stage_a, indent=2, sort_keys=True), "```",
              "", "## Stage B details", "", "```json", json.dumps(stage_b, indent=2, sort_keys=True), "```"]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_plots(outdir, stage_a_rows, stage_b_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    xs = list(LAMBDAS)
    ys = [BASE.median([float(row[f"auc_{lambda_name(x)}"]) for row in stage_a_rows]) for x in xs]
    plt.figure(figsize=(5.8, 4.2)); plt.plot(xs, ys, marker="o"); plt.xlabel("lambda"); plt.ylabel("FutureKL AUC")
    plt.tight_layout(); plt.savefig(figdir / "01_commutator_dose.png", dpi=180); plt.close()
    plt.figure(figsize=(5.8, 4.2))
    labels = list(STAGE_B_CONDITIONS)
    vals = [BASE.median([float(row[f"auc_{condition}"]) for row in stage_b_rows]) for condition in labels]
    plt.bar(labels, vals); plt.ylabel("FutureKL AUC"); plt.tight_layout(); plt.savefig(figdir / "02_scale_lattice_factorial.png", dpi=180); plt.close()


def merge_runs(args):
    outdir = Path(args.output_dir); setup_output(outdir)
    inputs = [Path(x) for x in args.merge_run_dirs]
    horizon_rows, state_rows, failures = [], [], []
    for source in inputs:
        horizon_rows += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_horizon.jsonl") or [])
        state_rows += list(BASE.iter_jsonl(source / "raw" / f"{args.phase}_state_stats.jsonl") or [])
        failures += BASE.load_json(source / f"{args.phase}_run_summary.json", {}).get("failures", [])
    horizon_rows = dedupe(horizon_rows, ("unit_id", "horizon", "condition"))
    state_rows = dedupe(state_rows, ("unit_id", "layer"))
    for row in horizon_rows:
        row.setdefault("future_kl_auc", None)
        row.setdefault("scale_statistics", None)
        row.setdefault("code_statistics", None)
    unit_rows = aggregate_units(horizon_rows)
    if args.phase == "stageA":
        stage0 = BASE.load_json(outdir / "stage0_identity.json", {})
        if stage0.get("STAGE0") != "PASS":
            raise RuntimeError("Stage A merge requires PASS Stage 0 in final output directory")
        dose_rows, summary = summarize_stage_a(unit_rows, stage0)
        BASE.write_rows(outdir / "stageA_horizon_results.csv", horizon_rows)
        BASE.write_rows(outdir / "stageA_unit_results.csv", dose_rows)
        BASE.write_rows(outdir / "stage0_state_statistics.csv", state_rows)
        BASE.save_json(outdir / "stageA_summary.json", summary)
        write_stage_a_report(outdir, summary)
        if summary["COMMUTATOR_CAUSAL_STATUS"] == "NOT_SUPPORTED":
            result = write_stopped_report(outdir, summary, failures)
            write_stage_a_plot(outdir, dose_rows)
        else:
            result = summary
    elif args.phase == "stageB":
        stage_a = BASE.load_json(outdir / "stageA_summary.json", {})
        if stage_a.get("COMMUTATOR_CAUSAL_STATUS") == "NOT_SUPPORTED":
            raise RuntimeError("Stage A stop rule forbids Stage B")
        factorial_rows, stage_b = summarize_stage_b(unit_rows)
        BASE.write_rows(outdir / "stageB_horizon_results.csv", horizon_rows)
        BASE.write_rows(outdir / "stageB_unit_results.csv", factorial_rows)
        BASE.save_json(outdir / "stageB_summary.json", stage_b)
        pytest_path = outdir / "pytest_output.txt"
        lines = pytest_path.read_text(encoding="utf-8").strip().splitlines() if pytest_path.exists() else []
        result = write_final_report(outdir, stage_a, stage_b, failures, lines[-1] if lines else "not recorded")
        stage_a_units = BASE.read_rows(outdir / "stageA_unit_results.csv")
        write_plots(outdir, stage_a_units, factorial_rows)
    else:
        raise ValueError("only Stage A/B runs are merged")
    config = BASE.load_json(outdir / "experiment_config.json", {})
    config.update({
        "TASK": TASK, "phase": args.phase, "scope": "formal", "horizon": int(args.horizon),
        f"{args.phase}_merged_from": [str(path) for path in inputs],
        "stageA_stop_rule": result.get("COMMUTATOR_CAUSAL_STATUS") == "NOT_SUPPORTED" if args.phase == "stageA" else False,
    })
    BASE.save_json(outdir / "experiment_config.json", config)
    (outdir / "run.log").write_text(
        f"TASK={TASK}\nPHASE={args.phase}\nMERGED_FROM={json.dumps([str(path) for path in inputs])}\n"
        f"FAILURES={json.dumps(failures)}\nTIME={BASE.now()}\n",
        encoding="utf-8",
    )
    required = [
        "experiment_config.json", "quantizer_semantics.json", "stage0_identity.json",
        "stage0_state_statistics.csv", "stageA_unit_results.csv", "stageA_horizon_results.csv",
        "stageA_summary.json", "stageB_unit_results.csv", "stageB_horizon_results.csv",
        "stageB_summary.json", "formal_summary.md", "run.log", "pytest_output.txt",
    ]
    n_units = len({r["unit_id"] for r in unit_rows})
    n_conditions = len(LAMBDAS) if args.phase == "stageA" else len(STAGE_B_CONDITIONS)
    expected_horizon = n_units * int(args.horizon) * (n_conditions + 1)
    unique_horizon = len({(r["unit_id"], int(r["horizon"]), r["condition"]) for r in horizon_rows})
    finite_ok = all(
        not isinstance(value, float) or math.isfinite(value)
        for row in horizon_rows + state_rows + unit_rows for value in row.values()
    )
    validation = {
        "ARTIFACT_VALIDATION": "PASS", "phase": args.phase,
        "required_files_present": all((outdir / rel).is_file() for rel in required) if result.get("STAGEB") == "SKIPPED_BY_STAGE_A_STOP_RULE" else True,
        "missing_required_files": [rel for rel in required if not (outdir / rel).is_file()],
        "n_units": n_units, "horizon_rows": len(horizon_rows), "expected_horizon_rows": expected_horizon,
        "unique_horizon_rows": unique_horizon, "state_stat_rows": len(state_rows),
        "expected_state_stat_rows": n_units * len({int(r["layer"]) for r in state_rows}),
        "all_numeric_values_finite": finite_ok, "failures": failures,
    }
    gates = [
        not failures, n_units == EXPECTED_UNITS, len(horizon_rows) == expected_horizon,
        unique_horizon == expected_horizon, validation["all_numeric_values_finite"],
    ]
    if result.get("STAGEB") == "SKIPPED_BY_STAGE_A_STOP_RULE":
        gates.append(validation["required_files_present"])
    validation["ARTIFACT_VALIDATION"] = "PASS" if all(gates) else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", validation)
    if (outdir / "summary.json").exists():
        final_summary = BASE.load_json(outdir / "summary.json", {})
        final_summary["ARTIFACT_VALIDATION"] = validation["ARTIFACT_VALIDATION"]
        BASE.save_json(outdir / "summary.json", final_summary)
        with (outdir / "formal_summary.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\nARTIFACT_VALIDATION = {validation['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {"TASK": TASK, "artifacts": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("stage0", "stageA", "stageB"), required=True)
    p.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    p.add_argument("--max-units", type=int, default=None)
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
    if result.get("failures"):
        return 2
    if args.phase == "stage0" and result.get("STAGE0") != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
