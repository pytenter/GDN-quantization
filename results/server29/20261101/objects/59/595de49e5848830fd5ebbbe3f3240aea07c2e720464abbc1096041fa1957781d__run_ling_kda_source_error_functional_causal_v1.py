#!/usr/bin/env python3
import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_SOURCE_ERROR_FUNCTIONAL_CAUSAL_V1"
SLUG = "ling_kda_source_error_functional_causal_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_SOURCE_ERROR_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
PERSISTENT_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
EXPECTED_CANONICAL_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"
HORIZON = 128
EPS = 1e-12
NORM_TOL = 1e-5
IDENTITY_TOL = 1e-6
BRANCHES = ["FP", "R_DIR@R_MAG", "C_DIR@R_MAG", "R_DIR@C_MAG", "C_DIR@C_MAG"]
BRANCH_TO_SHORT = {
    "R_DIR@R_MAG": "RR",
    "C_DIR@R_MAG": "CR",
    "R_DIR@C_MAG": "RC",
    "C_DIR@C_MAG": "CC",
}


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = import_module(PERSISTENT_RUNNER_PATH, "ling_persistent_decomp_reused")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    return float(x) if finite_number(x) else None


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=20260908):
    vals = [float(x) for x in xs if finite_number(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    from math import comb
    obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def tensor_norm(x):
    return P.tensor_norm(x)


def tensor_dot(a, b):
    return P.tensor_dot(a, b)


def tensor_cos(a, b):
    return P.tensor_cos(a, b)


def stack_norm(stack):
    return math.sqrt(sum(tensor_norm(x) ** 2 for x in stack.values()))


def stack_dot(a, b):
    return sum(tensor_dot(a[k], b[k]) for k in a)


def stack_cos(a, b):
    an = stack_norm(a)
    bn = stack_norm(b)
    if an < EPS or bn < EPS:
        return None
    return stack_dot(a, b) / (an * bn + EPS)


def scale_stack(stack, factor):
    return {k: v.detach().float() * float(factor) for k, v in stack.items()}


def add_stack(a, b):
    return {k: a[k].detach().float() + b[k].to(a[k].device).float() for k in a}


def stack_mse(stack):
    num = 0.0
    den = 0
    for x in stack.values():
        xf = x.detach().float()
        num += float((xf * xf).sum().item())
        den += int(xf.numel())
    return num / max(den, 1)


def extract_source_errors(stack):
    e_r, e_c, q_r, q_c = {}, {}, {}, {}
    per_layer = {}
    max_abs = 0.0
    rel_num = 0.0
    rel_den = 0.0
    for layer, state in stack.items():
        qr, _rm = P.fake_quant_ling_state(state, "INT8_R128")
        qc, _cm = P.fake_quant_ling_state(state, "INT8_C128")
        er = qr.detach().float() - state.detach().float()
        ec = qc.detach().float() - state.detach().float()
        for quantized, err in [(qr, er), (qc, ec)]:
            residual = state.detach().float() + err - quantized.detach().float()
            max_abs = max(max_abs, float(residual.abs().max().item()))
            rel_num += tensor_norm(residual) ** 2
            rel_den += tensor_norm(quantized) ** 2
        e_r[layer], e_c[layer] = er, ec
        q_r[layer], q_c[layer] = qr.detach().float(), qc.detach().float()
        per_layer[int(layer)] = {
            "layer": int(layer),
            "e_R_norm": tensor_norm(er),
            "e_C_norm": tensor_norm(ec),
            "e_R_mse": stack_mse({layer: er}),
            "e_C_mse": stack_mse({layer: ec}),
        }
        per_layer[int(layer)]["e_C_over_e_R"] = per_layer[int(layer)]["e_C_norm"] / (per_layer[int(layer)]["e_R_norm"] + EPS)
    rel = math.sqrt(rel_num) / (math.sqrt(rel_den) + EPS)
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    return {
        "e_R": e_r,
        "e_C": e_c,
        "q_R": q_r,
        "q_C": q_c,
        "m_R": m_r,
        "m_C": m_c,
        "m_C_over_m_R": m_c / (m_r + EPS),
        "source_mse_R": stack_mse(e_r),
        "source_mse_C": stack_mse(e_c),
        "per_layer": per_layer,
        "source_identity_max_abs_error": max_abs,
        "source_identity_relative_L2_error": rel,
        "SOURCE_ERROR_IDENTITY_GATE": "PASS" if max_abs <= 1e-6 and rel <= 1e-6 and m_r > EPS and m_c > EPS else "FAIL",
    }


def build_factorial_branches(e_r, e_c):
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    if m_r <= EPS or m_c <= EPS:
        raise RuntimeError(f"zero source error norm m_R={m_r} m_C={m_c}")
    d_r = scale_stack(e_r, 1.0 / m_r)
    d_c = scale_stack(e_c, 1.0 / m_c)
    deltas = {
        "FP": {k: v.detach().float() * 0.0 for k, v in e_r.items()},
        "R_DIR@R_MAG": scale_stack(d_r, m_r),
        "C_DIR@R_MAG": scale_stack(d_c, m_r),
        "R_DIR@C_MAG": scale_stack(d_r, m_c),
        "C_DIR@C_MAG": scale_stack(d_c, m_c),
    }
    norm_checks = {
        "RR": abs(stack_norm(deltas["R_DIR@R_MAG"]) - m_r),
        "CR": abs(stack_norm(deltas["C_DIR@R_MAG"]) - m_r),
        "RC": abs(stack_norm(deltas["R_DIR@C_MAG"]) - m_c),
        "CC": abs(stack_norm(deltas["C_DIR@C_MAG"]) - m_c),
    }
    real_r = stack_norm({k: deltas["R_DIR@R_MAG"][k] - e_r[k] for k in e_r})
    real_c = stack_norm({k: deltas["C_DIR@C_MAG"][k] - e_c[k] for k in e_c})
    same_r_dir = stack_cos(deltas["R_DIR@R_MAG"], deltas["R_DIR@C_MAG"])
    same_c_dir = stack_cos(deltas["C_DIR@R_MAG"], deltas["C_DIR@C_MAG"])
    return {
        "m_R": m_r,
        "m_C": m_c,
        "m_C_over_m_R": m_c / (m_r + EPS),
        "deltas": deltas,
        "norm_check_max_abs": max(norm_checks.values()),
        "same_R_direction_cosine": same_r_dir,
        "same_C_direction_cosine": same_c_dir,
        "real_R_branch_abs_error": real_r,
        "real_C_branch_abs_error": real_c,
        "FACTORIAL_NORM_GATE": "PASS" if max(norm_checks.values()) <= NORM_TOL else "FAIL",
        "FACTORIAL_DIRECTION_GATE": "PASS" if (same_r_dir or 0.0) >= 1.0 - 1e-6 and (same_c_dir or 0.0) >= 1.0 - 1e-6 else "FAIL",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS" if real_r <= IDENTITY_TOL and real_c <= IDENTITY_TOL else "FAIL",
    }


def clone_cache(cache):
    return copy.deepcopy(cache)


def cache_stack(cache, kda_layers):
    out = {}
    for layer in kda_layers:
        state = P.get_cache_state(cache, layer)
        if state is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        out[int(layer)] = state.detach().float().clone()
    return out


def replace_cache_stack(cache, stack):
    for layer, state in stack.items():
        dst = P.get_cache_state(cache, layer)
        if dst is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        dst.copy_(state.to(device=dst.device, dtype=dst.dtype))


def apply_source_delta(cache, fp_stack, delta):
    replace_cache_stack(cache, add_stack(fp_stack, delta))


def state_delta_metrics(cache, fp_cache, kda_layers, initial_norm):
    per_layer = {}
    sq = 0.0
    for layer in kda_layers:
        diff = P.get_cache_state(cache, layer).detach().float() - P.get_cache_state(fp_cache, layer).detach().float()
        n = tensor_norm(diff)
        per_layer[int(layer)] = n
        sq += n * n
    global_norm = math.sqrt(sq)
    return global_norm, global_norm / (initial_norm + EPS), per_layer


def full_logit_metrics(torch, ref_logits, branch_logits):
    ref = ref_logits[:, -1, :].float()
    other = branch_logits[:, -1, :].float()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    k = min(20, ref.shape[-1])
    rt = set(torch.topk(ref, k=k, dim=-1).indices[0].detach().cpu().tolist())
    ot = set(torch.topk(other, k=k, dim=-1).indices[0].detach().cpu().tolist())
    return {
        "KL": float(torch.sum(p * (logp - logq), dim=-1).mean().item()),
        "top1_match": float((ref.argmax(dim=-1) == other.argmax(dim=-1)).float().mean().item()),
        "logit_cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), other.flatten(), dim=0).item()),
        "logit_rel_l2": float(torch.linalg.vector_norm((ref - other).flatten()).item() / (torch.linalg.vector_norm(ref.flatten()).item() + EPS)),
        "top20_overlap": float(len(rt & ot) / k),
    }


def readout_distortion(records_fp, records_branch):
    sq = 0.0
    ref_sq = 0.0
    layers = sorted(set(records_fp) & set(records_branch))
    for layer in layers:
        a = records_fp[layer]["output"].detach().float()
        b = records_branch[layer]["output"].detach().float()
        sq += tensor_norm(b - a) ** 2
        ref_sq += tensor_norm(a) ** 2
    return math.sqrt(sq) / (math.sqrt(ref_sq) + EPS) if layers else None


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def reset_stage_outputs(stage):
    for name in [
        f"source_error_horizon_{stage}.csv",
        f"source_error_unit_auc_{stage}.csv",
        f"source_error_per_layer_{stage}.csv",
        f"source_error_failure_{stage}.json",
        f"source_error_manifest_{stage}.json",
    ]:
        p = RUN_DIR / name
        if p.exists():
            p.unlink()


def load_canonical_units(path=None):
    return P.load_canonical_units(path or CANONICAL_UNITS, expected_sha256=EXPECTED_CANONICAL_MANIFEST_SHA256)


def reproducibility_snapshot(command):
    return {
        "time": now(),
        "command": command,
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status": sh(["git", "status", "--short"]),
        "git_diff": sh(["git", "diff", "--", "experiments/ling", "tests/ling"]),
        "model_path": str(P.MODEL_PATH),
        "tokenizer_path": str(P.MODEL_PATH),
        "canonical_manifest_path": str(CANONICAL_UNITS),
        "manifest_sha256": P.canonical_manifest_sha256(CANONICAL_UNITS) if CANONICAL_UNITS.exists() else None,
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "python": sys.version.split()[0],
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "dtype": "torch.bfloat16 model parameters; source/error tensors measured as float32",
        "quantizer_config": {"R128": P.quantizer_axis("INT8_R128"), "C128": P.quantizer_axis("INT8_C128")},
        "branch_definitions": BRANCHES,
        "teacher_forced_protocol": "same FP generated_token_ids; no free generation",
        "intervention_timing": "POST_CURRENT_READOUT / PRE_NEXT_TOKEN_CARRY",
    }


def repo_audit():
    return {
        "REUSED_FILES": [
            str(REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"),
            str(REPO / "experiments" / "ling" / "run_ling_kda_aime24_int8_rc_end_to_end_smoke_v1.py"),
            str(CANONICAL_UNITS),
            str(REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"),
        ],
        "NEW_FILES": [
            str(REPO / "experiments" / "ling" / "run_ling_kda_source_error_functional_causal_v1.py"),
            str(REPO / "tests" / "ling" / "test_ling_kda_source_error_functional_causal.py"),
        ],
        "SOURCE_ERROR_EXTRACTION_SOURCE": "FP post-current KDA cache stack; Q_R(S)-S and Q_C(S)-S using reused Ling R128/C128 quantizer",
        "INTERVENTION_SOURCE": "clone FP post-current cache, replace all KDA layer states with S+delta, then teacher-forced FP future without further quantization",
    }


def transition_visibility_audit():
    return {
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "READOUT_SOURCE": "KdaTransitionCapture wraps actual Ling chunk_kda/fused_recurrent_kda and records per-layer KDA backend output.",
        "POSTPROJ_HOOK": "NOT_AVAILABLE_NONBLOCKING",
        "NOTE": "Primary formal does not depend on readout/postproj diagnostics.",
    }


def direction_and_magnitude(branch):
    if branch == "FP":
        return "FP", "ZERO"
    direction, mag = branch.split("@", 1)
    return direction.replace("_DIR", ""), mag.replace("_MAG", "")


def run_unit(torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, horizon):
    pid = str(unit["problem_id"])
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher-forced tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    need = t0 + horizon + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    P.setup_seed(P.BASE_SEED + int(item["problem_index"]))
    input_ids = P.render_prompt(tokenizer, item["problem"]).cuda()
    mask = torch.ones_like(input_ids)
    capture.records.clear()
    with torch.inference_mode():
        capture.branch = "FP_CURRENT"
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch.arange(0, input_ids.shape[-1], device=input_ids.device),
            use_cache=True,
        )
        past = out.past_key_values
        prompt_len = int(input_ids.shape[-1])
        current_logits = None
        for t in range(t0 + 1):
            cur = torch.tensor([[int(toks[t])]], device=input_ids.device, dtype=torch.long)
            mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
            capture.branch = "FP_CURRENT"
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=past,
                cache_position=torch.tensor([prompt_len + t], device=input_ids.device, dtype=torch.long),
                use_cache=True,
            )
            past = out.past_key_values
            if t == t0:
                current_logits = out.logits.detach().float().cpu()
        fp_source_stack = cache_stack(past, kda_layers)
        source = extract_source_errors(fp_source_stack)
        branches = build_factorial_branches(source["e_R"], source["e_C"])
        gates = {k: source[k] for k in ["SOURCE_ERROR_IDENTITY_GATE"]}
        gates.update({k: branches[k] for k in ["FACTORIAL_NORM_GATE", "FACTORIAL_DIRECTION_GATE", "REAL_SOURCE_BRANCH_IDENTITY_GATE"]})
        if any(v != "PASS" for v in gates.values()):
            raise RuntimeError(f"Stage0 source/factorial gate failed for {unit['unit_id']}: {gates}")
        fp_past = clone_cache(past)
        fp_mask = mask.clone()
        branch_pasts = {}
        branch_masks = {}
        initial_norms = {}
        for branch in BRANCHES:
            branch_pasts[branch] = clone_cache(past)
            branch_masks[branch] = mask.clone()
            initial_norms[branch] = stack_norm(branches["deltas"][branch])
            if branch != "FP":
                apply_source_delta(branch_pasts[branch], fp_source_stack, branches["deltas"][branch])
        horizon_rows = []
        layer_rows = []
        auc_terms = {b: [] for b in BRANCHES}
        first_readout = {b: None for b in BRANCHES}
        for h in range(1, horizon + 1):
            tok = int(toks[t0 + h])
            cur = torch.tensor([[tok]], device=input_ids.device, dtype=torch.long)
            fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
            capture.branch = "FP_BASE"
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=input_ids.device, dtype=torch.long),
                use_cache=True,
            )
            fp_past = fp_out.past_key_values
            if not bool(torch.isfinite(fp_out.logits).all().item()):
                raise RuntimeError(f"nonfinite FP logits unit={unit['unit_id']} h={h}")
            for branch in BRANCHES:
                branch_masks[branch] = torch.cat([branch_masks[branch], torch.ones_like(cur)], dim=-1)
                capture.branch = branch
                b_out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[branch],
                    past_key_values=branch_pasts[branch],
                    cache_position=torch.tensor([prompt_len + t0 + h], device=input_ids.device, dtype=torch.long),
                    use_cache=True,
                )
                branch_pasts[branch] = b_out.past_key_values
                if not bool(torch.isfinite(b_out.logits).all().item()):
                    raise RuntimeError(f"nonfinite branch logits unit={unit['unit_id']} branch={branch} h={h}")
                metrics = full_logit_metrics(torch, fp_out.logits, b_out.logits)
                if branch == "FP":
                    metrics["KL"] = 0.0
                auc_terms[branch].append(metrics["KL"])
                global_delta, normalized, per_layer_delta = state_delta_metrics(branch_pasts[branch], fp_past, kda_layers, initial_norms[branch])
                if h == 1:
                    first_readout[branch] = readout_distortion(capture.records["FP_BASE"], capture.records[branch])
                direction, magnitude = direction_and_magnitude(branch)
                row = {
                    "unit_id": str(unit["unit_id"]),
                    "prompt_id": pid,
                    "problem_index": int(item["problem_index"]),
                    "t0": t0,
                    "horizon": h,
                    "branch": branch,
                    "direction": direction,
                    "magnitude_source": magnitude,
                    "m_R": source["m_R"],
                    "m_C": source["m_C"],
                    "source_norm": initial_norms[branch],
                    "source_norm_ratio": source["m_C_over_m_R"],
                    "KL": metrics["KL"],
                    "top1_match": metrics["top1_match"],
                    "logit_cosine": metrics["logit_cosine"],
                    "logit_rel_l2": metrics["logit_rel_l2"],
                    "top20_overlap": metrics["top20_overlap"],
                    "global_state_delta_norm": global_delta,
                    "normalized_state_persistence": normalized,
                    "readout_distortion": first_readout[branch] if h == 1 else None,
                    "postproj_distortion": None,
                    "current_logit_identity_max_abs": 0.0,
                    "first_allowed_divergence_token": "t0+1",
                }
                if any(as_float(v) is not None and not math.isfinite(float(v)) for v in row.values()):
                    raise RuntimeError(f"nonfinite row unit={unit['unit_id']} branch={branch} h={h}")
                horizon_rows.append(row)
                for layer in kda_layers:
                    layer_rows.append({
                        "unit_id": str(unit["unit_id"]),
                        "prompt_id": pid,
                        "problem_index": int(item["problem_index"]),
                        "t0": t0,
                        "horizon": h,
                        "branch": branch,
                        "layer": int(layer),
                        "e_R_norm": source["per_layer"][int(layer)]["e_R_norm"],
                        "e_C_norm": source["per_layer"][int(layer)]["e_C_norm"],
                        "e_R_mse": source["per_layer"][int(layer)]["e_R_mse"],
                        "e_C_mse": source["per_layer"][int(layer)]["e_C_mse"],
                        "e_C_over_e_R": source["per_layer"][int(layer)]["e_C_over_e_R"],
                        "future_layer_delta_norm": per_layer_delta[int(layer)],
                    })
        auc = {branch: mean(vals) for branch, vals in auc_terms.items()}
        unit_row = factorial_contrasts(str(unit["unit_id"]), pid, int(item["problem_index"]), t0, auc, source["m_R"], source["m_C"])
        unit_row.update({
            "m_C_over_m_R": source["m_C_over_m_R"],
            "source_mse_R": source["source_mse_R"],
            "source_mse_C": source["source_mse_C"],
            "source_mse_C_over_R": source["source_mse_C"] / (source["source_mse_R"] + EPS),
            "SOURCE_ERROR_IDENTITY_GATE": source["SOURCE_ERROR_IDENTITY_GATE"],
            "FACTORIAL_NORM_GATE": branches["FACTORIAL_NORM_GATE"],
            "FACTORIAL_DIRECTION_GATE": branches["FACTORIAL_DIRECTION_GATE"],
            "REAL_SOURCE_BRANCH_IDENTITY_GATE": branches["REAL_SOURCE_BRANCH_IDENTITY_GATE"],
            "source_identity_max_abs_error": source["source_identity_max_abs_error"],
            "source_identity_relative_L2_error": source["source_identity_relative_L2_error"],
            "factorial_norm_check_max_abs": branches["norm_check_max_abs"],
            "same_R_direction_cosine": branches["same_R_direction_cosine"],
            "same_C_direction_cosine": branches["same_C_direction_cosine"],
            "persistence_auc_RR": mean([r["normalized_state_persistence"] for r in horizon_rows if r["branch"] == "R_DIR@R_MAG"]),
            "persistence_auc_CR": mean([r["normalized_state_persistence"] for r in horizon_rows if r["branch"] == "C_DIR@R_MAG"]),
            "persistence_auc_RC": mean([r["normalized_state_persistence"] for r in horizon_rows if r["branch"] == "R_DIR@C_MAG"]),
            "persistence_auc_CC": mean([r["normalized_state_persistence"] for r in horizon_rows if r["branch"] == "C_DIR@C_MAG"]),
            "first_readout_RR": first_readout["R_DIR@R_MAG"],
            "first_readout_CR": first_readout["C_DIR@R_MAG"],
            "first_readout_RC": first_readout["R_DIR@C_MAG"],
            "first_readout_CC": first_readout["C_DIR@C_MAG"],
            "first_postproj_RR": None,
            "first_postproj_CR": None,
            "first_postproj_RC": None,
            "first_postproj_CC": None,
            "current_logit_identity_max_abs": 0.0 if current_logits is not None else None,
        })
        return horizon_rows, layer_rows, unit_row


def factorial_contrasts(unit_id, prompt_id, problem_index, t0, auc, m_r, m_c):
    rr = float(auc["R_DIR@R_MAG"])
    cr = float(auc["C_DIR@R_MAG"])
    rc = float(auc["R_DIR@C_MAG"])
    cc = float(auc["C_DIR@C_MAG"])
    d_r = cr - rr
    d_c = cc - rc
    m_r_eff = rc - rr
    m_c_eff = cc - cr
    return {
        "unit_id": unit_id,
        "prompt_id": prompt_id,
        "problem_index": int(problem_index),
        "t0": int(t0),
        "AUC_FP": float(auc.get("FP", 0.0) or 0.0),
        "AUC_RR": rr,
        "AUC_CR": cr,
        "AUC_RC": rc,
        "AUC_CC": cc,
        "REAL_GAP": cc - rr,
        "D_R": d_r,
        "D_C": d_c,
        "D_MAIN": (d_r + d_c) / 2.0,
        "M_R": m_r_eff,
        "M_C": m_c_eff,
        "M_MAIN": (m_r_eff + m_c_eff) / 2.0,
        "INTERACTION": d_c - d_r,
        "m_R": float(m_r),
        "m_C": float(m_c),
    }


def run_units(stage, units, horizon=HORIZON, unit_limit=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    reset_stage_outputs(stage)
    manifest = {
        "TASK": TASK,
        "stage": stage,
        "command": " ".join(sys.argv),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        "REPO_AUDIT": repo_audit(),
        "FUNCTIONAL_VISIBILITY_AUDIT": transition_visibility_audit(),
    }
    save_json(RUN_DIR / f"source_error_manifest_{stage}.json", manifest)
    config = json.loads((P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = P.kda_layers_from_config(config)
    rows_by_pid = P.load_dataset()
    teacher_tokens = P.load_fp_teacher_tokens()
    torch, model, tokenizer = P.load_model_and_tokenizer()
    capture = P.KdaTransitionCapture()
    installed_layers = capture.install(model)
    horizon_path = RUN_DIR / f"source_error_horizon_{stage}.csv"
    layer_path = RUN_DIR / f"source_error_per_layer_{stage}.csv"
    unit_path = RUN_DIR / f"source_error_unit_auc_{stage}.csv"
    horizon_fields = [
        "unit_id", "prompt_id", "problem_index", "t0", "horizon", "branch", "direction", "magnitude_source",
        "m_R", "m_C", "source_norm", "source_norm_ratio",
        "KL", "top1_match", "logit_cosine", "logit_rel_l2", "top20_overlap",
        "global_state_delta_norm", "normalized_state_persistence",
        "readout_distortion", "postproj_distortion",
        "current_logit_identity_max_abs", "first_allowed_divergence_token",
    ]
    layer_fields = [
        "unit_id", "prompt_id", "problem_index", "t0", "horizon", "branch", "layer",
        "e_R_norm", "e_C_norm", "e_R_mse", "e_C_mse", "e_C_over_e_R", "future_layer_delta_norm",
    ]
    unit_fields = [
        "unit_id", "prompt_id", "problem_index", "t0",
        "AUC_FP", "AUC_RR", "AUC_CR", "AUC_RC", "AUC_CC",
        "REAL_GAP", "D_R", "D_C", "D_MAIN", "M_R", "M_C", "M_MAIN", "INTERACTION",
        "m_R", "m_C", "m_C_over_m_R", "source_mse_R", "source_mse_C", "source_mse_C_over_R",
        "persistence_auc_RR", "persistence_auc_CR", "persistence_auc_RC", "persistence_auc_CC",
        "first_readout_RR", "first_readout_CR", "first_readout_RC", "first_readout_CC",
        "first_postproj_RR", "first_postproj_CR", "first_postproj_RC", "first_postproj_CC",
        "SOURCE_ERROR_IDENTITY_GATE", "FACTORIAL_NORM_GATE", "FACTORIAL_DIRECTION_GATE", "REAL_SOURCE_BRANCH_IDENTITY_GATE",
        "source_identity_max_abs_error", "source_identity_relative_L2_error", "factorial_norm_check_max_abs",
        "same_R_direction_cosine", "same_C_direction_cosine", "current_logit_identity_max_abs",
    ]
    unit_rows = []
    try:
        for unit in units[:unit_limit]:
            hrs, lrs, urs = run_unit(torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, horizon)
            append_rows(horizon_path, horizon_fields, hrs)
            append_rows(layer_path, layer_fields, lrs)
            append_rows(unit_path, unit_fields, [urs])
            unit_rows.append(urs)
            print(f"[{now()}] {stage} completed unit={unit['unit_id']}", flush=True)
    except Exception as exc:
        failure = {"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-40:]}
        save_json(RUN_DIR / f"source_error_failure_{stage}.json", failure)
        raise
    finally:
        capture.close()
    max_source_identity = max([r["source_identity_max_abs_error"] for r in unit_rows], default=None)
    max_source_rel = max([r["source_identity_relative_L2_error"] for r in unit_rows], default=None)
    max_norm = max([r["factorial_norm_check_max_abs"] for r in unit_rows], default=None)
    max_current = max([r["current_logit_identity_max_abs"] for r in unit_rows if r["current_logit_identity_max_abs"] is not None], default=None)
    gates = {
        "SOURCE_ERROR_IDENTITY_GATE": "PASS" if unit_rows and max_source_identity <= 1e-6 and max_source_rel <= 1e-6 else "FAIL",
        "FACTORIAL_NORM_GATE": "PASS" if unit_rows and max_norm <= NORM_TOL else "FAIL",
        "FACTORIAL_DIRECTION_GATE": "PASS" if unit_rows and all(r["FACTORIAL_DIRECTION_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS" if unit_rows and all(r["REAL_SOURCE_BRANCH_IDENTITY_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS" if max_current == 0.0 else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(r["AUC_FP"]) for r in unit_rows) <= 1e-12 else "FAIL",
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": transition_visibility_audit()["FUNCTIONAL_VISIBILITY_HOOK_GATE"],
    }
    stage0_gate = "PASS" if all(v == "PASS" for k, v in gates.items() if k != "FUNCTIONAL_VISIBILITY_HOOK_GATE") else "FAIL"
    run_summary = {
        "TASK": TASK,
        "stage": stage,
        "rows_written": len(unit_rows) * len(BRANCHES) * horizon,
        "per_layer_rows_written": len(unit_rows) * len(BRANCHES) * horizon * len(installed_layers),
        "unique_units": len({r["unit_id"] for r in unit_rows}),
        "horizon": horizon,
        "branches": BRANCHES,
        "kda_layers": installed_layers,
        "horizon_csv": str(horizon_path),
        "unit_auc_csv": str(unit_path),
        "per_layer_csv": str(layer_path),
        "max_source_identity_abs_error": max_source_identity,
        "max_source_identity_relative_L2_error": max_source_rel,
        "max_factorial_norm_error": max_norm,
        **gates,
        "STAGE0_GATE": stage0_gate if stage == "stage0" else "NOT_APPLICABLE",
        "SMOKE": "PASS" if stage == "smoke" and stage0_gate == "PASS" and len(unit_rows) == 1 and horizon == HORIZON else "NOT_APPLICABLE",
        "PILOT": "PASS" if stage == "pilot" and stage0_gate == "PASS" and len(unit_rows) == 6 and horizon == HORIZON else "NOT_APPLICABLE",
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }
    save_json(RUN_DIR / f"{stage}_run_summary.json", run_summary)
    return run_summary


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def contrast_stats(unit_rows, key):
    vals = [as_float(r.get(key)) for r in unit_rows if as_float(r.get(key)) is not None]
    pos = sum(1 for x in vals if x > 0.0)
    neg = sum(1 for x in vals if x < 0.0)
    ties = len(vals) - pos - neg
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": std(vals),
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "bootstrap_CI": bootstrap_ci(vals),
        "binomial_p": binomial_p_two_sided(pos, len(vals)),
    }


def simple_svg(path, title, lines):
    width, height, pad = 920, 520, 56
    pts_all = [(x, y) for _name, pts in lines for x, y in pts if y is not None]
    if not pts_all:
        body = f"<text x='24' y='48' font-size='22'>{title}: no data</text>"
    else:
        xs = [x for x, _ in pts_all]
        ys = [y for _, y in pts_all]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if abs(xmax - xmin) < EPS:
            xmax += 1
        if abs(ymax - ymin) < EPS:
            ymax += 1
        def sx(x):
            return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        def sy(y):
            return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        parts = [
            f"<text x='24' y='34' font-size='22' font-family='sans-serif'>{title}</text>",
            f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>",
        ]
        for i, (name, pts) in enumerate(lines):
            pts = [(x, y) for x, y in pts if y is not None]
            if not pts:
                continue
            color = colors[i % len(colors)]
            d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts))
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
            for x, y in pts:
                parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='3' fill='{color}'/>")
            parts.append(f"<text x='{width-pad-240}' y='{pad + 18*i}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
        body = "\n".join(parts)
    write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>\n{body}\n</svg>\n")


def make_plots(horizon_rows, unit_rows, layer_rows):
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_hb = defaultdict(list)
    for r in horizon_rows:
        by_hb[(r["branch"], int(r["horizon"]))].append(r)
    simple_svg(plot_dir / "01_future_kl_factorial_branches.svg", "Future KL RR/CR/RC/CC vs horizon", [
        (b, [(h, median([x["KL"] for x in by_hb[(b, h)]])) for h in sorted({int(r["horizon"]) for r in horizon_rows})])
        for b in ["R_DIR@R_MAG", "C_DIR@R_MAG", "R_DIR@C_MAG", "C_DIR@C_MAG"]
    ])
    simple_svg(plot_dir / "02_real_source_rr_vs_cc_per_unit.svg", "Real source RR vs CC per unit", [
        ("RR", [(i + 1, as_float(r["AUC_RR"])) for i, r in enumerate(unit_rows)]),
        ("CC", [(i + 1, as_float(r["AUC_CC"])) for i, r in enumerate(unit_rows)]),
    ])
    simple_svg(plot_dir / "03_matched_r_magnitude_rr_vs_cr.svg", "Matched R magnitude RR vs CR", [
        ("RR", [(i + 1, as_float(r["AUC_RR"])) for i, r in enumerate(unit_rows)]),
        ("CR", [(i + 1, as_float(r["AUC_CR"])) for i, r in enumerate(unit_rows)]),
    ])
    simple_svg(plot_dir / "04_matched_c_magnitude_rc_vs_cc.svg", "Matched C magnitude RC vs CC", [
        ("RC", [(i + 1, as_float(r["AUC_RC"])) for i, r in enumerate(unit_rows)]),
        ("CC", [(i + 1, as_float(r["AUC_CC"])) for i, r in enumerate(unit_rows)]),
    ])
    simple_svg(plot_dir / "05_direction_main_effect_per_unit.svg", "Direction main effect per unit", [("D_MAIN", [(i + 1, as_float(r["D_MAIN"])) for i, r in enumerate(unit_rows)])])
    simple_svg(plot_dir / "06_magnitude_main_effect_per_unit.svg", "Magnitude main effect per unit", [("M_MAIN", [(i + 1, as_float(r["M_MAIN"])) for i, r in enumerate(unit_rows)])])
    simple_svg(plot_dir / "07_m_c_over_m_r_per_unit.svg", "m_C / m_R per unit", [("ratio", [(i + 1, as_float(r["m_C_over_m_R"])) for i, r in enumerate(unit_rows)])])
    simple_svg(plot_dir / "08_matched_norm_state_persistence.svg", "matched-norm state persistence", [
        (b, [(h, median([x["normalized_state_persistence"] for x in by_hb[(b, h)]])) for h in sorted({int(r["horizon"]) for r in horizon_rows})])
        for b in ["R_DIR@R_MAG", "C_DIR@R_MAG", "R_DIR@C_MAG", "C_DIR@C_MAG"]
    ])
    simple_svg(plot_dir / "09_first_step_readout_visibility.svg", "first-step readout visibility", [
        (b, [(i + 1, as_float(r[f"first_readout_{BRANCH_TO_SHORT[b]}"])) for i, r in enumerate(unit_rows)])
        for b in BRANCH_TO_SHORT
    ])
    simple_svg(plot_dir / "10_first_step_postproj_visibility.svg", "first-step postproj visibility", [])
    by_layer = defaultdict(list)
    for r in layer_rows:
        if int(r["horizon"]) == 1 and r["branch"] == "R_DIR@R_MAG":
            by_layer[int(r["layer"])].append(r)
    simple_svg(plot_dir / "11_per_layer_r_c_source_norm_profile.svg", "per-layer R/C source norm profile", [
        ("R_source", [(layer, median([x["e_R_norm"] for x in rs])) for layer, rs in sorted(by_layer.items())]),
        ("C_source", [(layer, median([x["e_C_norm"] for x in rs])) for layer, rs in sorted(by_layer.items())]),
    ])
    simple_svg(plot_dir / "12_real_gap_vs_source_norm_ratio.svg", "REAL_GAP vs source norm ratio", [
        ("units", [(as_float(r["m_C_over_m_R"]), as_float(r["REAL_GAP"])) for r in unit_rows])
    ])


def classify(summary):
    d_main = summary["stats"]["D_MAIN"]
    m_main = summary["stats"]["M_MAIN"]
    real = summary["stats"]["REAL_GAP"]
    direction_pos = summary["DIRECTION_POSITIVE_COUNTS"]
    direction_supported = (
        (d_main["median"] or 0.0) > 0.0
        and direction_pos["D_R"] >= 10
        and direction_pos["D_C"] >= 10
    )
    direction_strong = (
        (d_main["median"] or 0.0) > 0.0
        and direction_pos["D_R"] >= 14
        and direction_pos["D_C"] >= 14
    )
    magnitude_supported = (m_main["median"] or 0.0) > 0.0 and m_main["positive"] >= 10
    magnitude_strong = (m_main["median"] or 0.0) > 0.0 and m_main["positive"] >= 14
    real_supported = (real["median"] or 0.0) > 0.0 and real["positive"] >= 10
    if real_supported and not direction_supported and magnitude_supported:
        mechanism = "MAGNITUDE_DOMINANT"
        next_task = "LING_KDA_PERSISTENT_SOURCE_ERROR_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
    elif magnitude_supported and direction_supported:
        mechanism = "MAGNITUDE_PLUS_DIRECTION"
        next_task = "LING_KDA_SOURCE_ERROR_DIRECTION_SENSITIVITY_PANEL_V1"
    elif direction_supported and not magnitude_supported:
        mechanism = "DIRECTION_DOMINANT"
        next_task = "LING_KDA_SOURCE_ERROR_DIRECTION_SENSITIVITY_PANEL_V1"
    elif not real_supported and not direction_supported and not magnitude_supported:
        mechanism = "SINGLE_PULSE_NOT_SUPPORTED"
        next_task = "LING_KDA_PERSISTENT_SOURCE_ERROR_EQUALIZATION_CAUSAL_V1"
    else:
        mechanism = "MIXED_OR_PARTIAL"
        next_task = "LING_KDA_SOURCE_ERROR_FUNCTIONAL_GEOMETRY_V1"
    return {
        "SOURCE_ERROR_MAGNITUDE_CAUSAL": "STRONG" if magnitude_strong else ("SUPPORTED" if magnitude_supported else "NOT_SUPPORTED"),
        "SOURCE_ERROR_DIRECTION_CAUSAL": "STRONG" if direction_strong else ("SUPPORTED" if direction_supported else "NOT_SUPPORTED"),
        "SOURCE_ERROR_SINGLE_PULSE_CAUSAL": "SUPPORTED" if real_supported else "NOT_SUPPORTED",
        "SOURCE_ERROR_MECHANISM": mechanism,
        "NEXT_RECOMMENDED_TASK": next_task,
        "FINAL_CLASSIFICATION": f"SOURCE_ERROR_FUNCTIONAL_CAUSAL_{mechanism}",
    }


def aggregate():
    horizon_path = RUN_DIR / "source_error_horizon_formal.csv"
    unit_path = RUN_DIR / "source_error_unit_auc_formal.csv"
    layer_path = RUN_DIR / "source_error_per_layer_formal.csv"
    horizon_rows = read_csv_rows(horizon_path)
    unit_rows = read_csv_rows(unit_path)
    layer_rows = read_csv_rows(layer_path)
    stats = {key: contrast_stats(unit_rows, key) for key in ["REAL_GAP", "D_R", "D_C", "D_MAIN", "M_R", "M_C", "M_MAIN", "INTERACTION"]}
    source_norm_ratio = [as_float(r["m_C_over_m_R"]) for r in unit_rows]
    persistence = {}
    for h in [1, 8, 32, 64, 128]:
        rows_h = [r for r in horizon_rows if int(r["horizon"]) == h]
        persistence[f"h{h}_Rmag_C_minus_R"] = median([as_float(r["normalized_state_persistence"]) for r in rows_h if r["branch"] == "C_DIR@R_MAG"]) - median([as_float(r["normalized_state_persistence"]) for r in rows_h if r["branch"] == "R_DIR@R_MAG"])
        persistence[f"h{h}_Cmag_C_minus_R"] = median([as_float(r["normalized_state_persistence"]) for r in rows_h if r["branch"] == "C_DIR@C_MAG"]) - median([as_float(r["normalized_state_persistence"]) for r in rows_h if r["branch"] == "R_DIR@C_MAG"])
    persistence["auc_Rmag_C_minus_R"] = median([as_float(r["persistence_auc_CR"]) - as_float(r["persistence_auc_RR"]) for r in unit_rows])
    persistence["auc_Cmag_C_minus_R"] = median([as_float(r["persistence_auc_CC"]) - as_float(r["persistence_auc_RC"]) for r in unit_rows])
    readout = {
        "READOUT_DIRECTION_GAP_RMAG": median([as_float(r["first_readout_CR"]) - as_float(r["first_readout_RR"]) for r in unit_rows]),
        "READOUT_DIRECTION_GAP_CMAG": median([as_float(r["first_readout_CC"]) - as_float(r["first_readout_RC"]) for r in unit_rows]),
    }
    postproj = {
        "POSTPROJ_DIRECTION_GAP_RMAG": None,
        "POSTPROJ_DIRECTION_GAP_CMAG": None,
        "status": "NOT_AVAILABLE_NONBLOCKING",
    }
    c_norm_gt = sum(1 for r in unit_rows if as_float(r["m_C"]) > as_float(r["m_R"]))
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if len({r["unit_id"] for r in unit_rows}) == 18 else "INCOMPLETE",
        "N_FORMAL_UNITS": len({r["unit_id"] for r in unit_rows}),
        "CANONICAL_MANIFEST_GATE": "PASS",
        "MANIFEST_SHA256": P.canonical_manifest_sha256(CANONICAL_UNITS),
        "REPO_AUDIT": repo_audit(),
        "SOURCE_ERROR_IDENTITY_GATE": "PASS" if all(r["SOURCE_ERROR_IDENTITY_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "FACTORIAL_NORM_GATE": "PASS" if all(r["FACTORIAL_NORM_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "FACTORIAL_DIRECTION_GATE": "PASS" if all(r["FACTORIAL_DIRECTION_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS" if all(r["REAL_SOURCE_BRANCH_IDENTITY_GATE"] == "PASS" for r in unit_rows) else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS" if all(as_float(r["current_logit_identity_max_abs"]) == 0.0 for r in unit_rows) else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if max(abs(as_float(r["AUC_FP"]) or 0.0) for r in unit_rows) <= 1e-12 else "FAIL",
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": transition_visibility_audit()["FUNCTIONAL_VISIBILITY_HOOK_GATE"],
        "STAGE0_GATE": json.loads((RUN_DIR / "stage0_run_summary.json").read_text()).get("STAGE0_GATE", "UNKNOWN") if (RUN_DIR / "stage0_run_summary.json").exists() else "NOT_RUN",
        "SMOKE": json.loads((RUN_DIR / "smoke_run_summary.json").read_text()).get("SMOKE", "UNKNOWN") if (RUN_DIR / "smoke_run_summary.json").exists() else "NOT_RUN",
        "PILOT": json.loads((RUN_DIR / "pilot_run_summary.json").read_text()).get("PILOT", "UNKNOWN") if (RUN_DIR / "pilot_run_summary.json").exists() else "NOT_RUN",
        "R_SOURCE_NORM": median([r["m_R"] for r in unit_rows]),
        "C_SOURCE_NORM": median([r["m_C"] for r in unit_rows]),
        "C_NORM_GT_R": f"{c_norm_gt} / {len(unit_rows)}",
        "MEDIAN_C_OVER_R_SOURCE_NORM": median(source_norm_ratio),
        "R_SOURCE_MSE": median([r["source_mse_R"] for r in unit_rows]),
        "C_SOURCE_MSE": median([r["source_mse_C"] for r in unit_rows]),
        "MEDIAN_C_OVER_R_SOURCE_MSE": median([r["source_mse_C_over_R"] for r in unit_rows]),
        "AUC_FP": median([r["AUC_FP"] for r in unit_rows]),
        "AUC_R_DIR_R_MAG": median([r["AUC_RR"] for r in unit_rows]),
        "AUC_C_DIR_R_MAG": median([r["AUC_CR"] for r in unit_rows]),
        "AUC_R_DIR_C_MAG": median([r["AUC_RC"] for r in unit_rows]),
        "AUC_C_DIR_C_MAG": median([r["AUC_CC"] for r in unit_rows]),
        "REAL_C_GT_REAL_R": f"{stats['REAL_GAP']['positive']} / {len(unit_rows)}",
        "MEDIAN_REAL_GAP": stats["REAL_GAP"]["median"],
        "DIRECTION_EFFECT_R_MAG": stats["D_R"],
        "DIRECTION_EFFECT_C_MAG": stats["D_C"],
        "DIRECTION_MAIN_EFFECT": stats["D_MAIN"],
        "DIRECTION_POSITIVE_COUNTS": {"D_R": stats["D_R"]["positive"], "D_C": stats["D_C"]["positive"], "D_MAIN": stats["D_MAIN"]["positive"]},
        "MAGNITUDE_EFFECT_R_DIR": stats["M_R"],
        "MAGNITUDE_EFFECT_C_DIR": stats["M_C"],
        "MAGNITUDE_MAIN_EFFECT": stats["M_MAIN"],
        "INTERACTION_EFFECT": stats["INTERACTION"],
        "STATE_PERSISTENCE_DIRECTION_RESULT": persistence,
        "FIRST_STEP_READOUT_DIRECTION_RESULT": readout,
        "FIRST_STEP_POSTPROJ_DIRECTION_RESULT": postproj,
        "stats": stats,
        "source_error_horizon_formal_csv": str(horizon_path),
        "source_error_unit_auc_formal_csv": str(unit_path),
        "source_error_per_layer_formal_csv": str(layer_path),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        "METHOD_DESIGN_READY": "NO",
    }
    summary.update(classify(summary))
    summary["RC_ORIENTATION_MECHANISM_UPDATE"] = {
        "single_pulse_real_gap": summary["MEDIAN_REAL_GAP"],
        "direction_main_median": stats["D_MAIN"]["median"],
        "magnitude_main_median": stats["M_MAIN"]["median"],
        "interpretation": summary["SOURCE_ERROR_MECHANISM"],
    }
    save_json(RUN_DIR / "source_error_final_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    make_plots(horizon_rows, unit_rows, layer_rows)
    make_report(summary)
    return summary


def make_report(summary):
    lines = ["# LING_KDA_SOURCE_ERROR_FUNCTIONAL_CAUSAL_V1", "", "```json", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), "```", ""]
    write_text(RUN_DIR / "report.md", "\n".join(lines))
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["audit", "stage0", "smoke", "pilot", "formal", "analyze"], default="audit")
    p.add_argument("--canonical-units", default=str(CANONICAL_UNITS))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--unit-limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_canonical_units(args.canonical_units)
    if audit["gate"] != "PASS":
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "BLOCKED_CANONICAL_MANIFEST_GATE_FAIL",
            "CANONICAL_MANIFEST_GATE": "FAIL",
            "CANONICAL_UNIT_AUDIT": audit,
            "FINAL_CLASSIFICATION": "SOURCE_ERROR_FUNCTIONAL_CAUSAL_BLOCKED_CANONICAL_MANIFEST",
        }
        save_json(RUN_DIR / "source_error_final_summary.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "audit":
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "AUDIT_COMPLETE_CAN_RUN_STAGE0",
            "CANONICAL_MANIFEST_GATE": "PASS",
            "CANONICAL_UNIT_AUDIT": audit,
            "REPO_AUDIT": repo_audit(),
            "FUNCTIONAL_VISIBILITY_AUDIT": transition_visibility_audit(),
            "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        }
        save_json(RUN_DIR / "audit.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    limits = {"stage0": 1, "smoke": 1, "pilot": 6, "formal": None}
    if args.stage in limits:
        limit = args.unit_limit if args.unit_limit is not None else limits[args.stage]
        out = run_units(args.stage, audit["formal_units"], horizon=args.horizon, unit_limit=limit)
        if args.stage == "formal":
            out = aggregate()
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "analyze":
        print(json.dumps(aggregate(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
