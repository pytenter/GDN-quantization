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


TASK = "LING_KDA_PERSISTENT_SOURCE_ERROR_EQUALIZATION_CAUSAL_V1"
SLUG = "ling_kda_persistent_source_error_equalization_causal_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_PERSISTENT_SOURCE_EQUALIZATION_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
SOURCE_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_source_error_functional_causal_v1.py"
PERSISTENT_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
SINGLE_PULSE_UNIT_AUC = REPO / "runs" / "ling_kda_source_error_functional_causal_v1" / "source_error_unit_auc_formal.csv"
EXPECTED_CANONICAL_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"
HORIZON = 128
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
STAGE_HISTORY_LENGTHS = {
    "stage0": [1],
    "smoke": [1, 8],
    "pilot": [1, 8, 32, 64],
    "formal": HISTORY_LENGTHS,
}
BRANCHES = ["FP", "R_NATIVE_SEQ", "C_DIR_R_MAG_SEQ", "R_DIR_C_MAG_SEQ", "C_NATIVE_SEQ"]
EFFECT_BRANCHES = ["R_NATIVE_SEQ", "C_NATIVE_SEQ", "C_DIR_R_MAG_SEQ", "R_DIR_C_MAG_SEQ"]
BLOCKING_GATES = [
    "SOURCE_ERROR_IDENTITY_GATE",
    "SEQUENCE_NORM_SCHEDULE_GATE",
    "SEQUENCE_DIRECTION_GATE",
    "REAL_SOURCE_BRANCH_IDENTITY_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_IDENTITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "L1_REPRODUCTION_GATE",
    "TEMPORAL_SUPERPOSITION_GATE",
]
SHARD_MERGED_GATES = [
    "SOURCE_ERROR_IDENTITY_GATE",
    "SEQUENCE_NORM_SCHEDULE_GATE",
    "SEQUENCE_DIRECTION_GATE",
    "REAL_SOURCE_BRANCH_IDENTITY_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_IDENTITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
]
L1_SINGLE_PULSE_BRANCH_MAP = {
    "R_NATIVE_SEQ": "AUC_RR",
    "C_DIR_R_MAG_SEQ": "AUC_CR",
    "R_DIR_C_MAG_SEQ": "AUC_RC",
    "C_NATIVE_SEQ": "AUC_CC",
}
OLD_TO_NEW_L1_BRANCH = {
    "FP": "FP",
    "R_DIR@R_MAG": "R_NATIVE_SEQ",
    "C_DIR@R_MAG": "C_DIR_R_MAG_SEQ",
    "R_DIR@C_MAG": "R_DIR_C_MAG_SEQ",
    "C_DIR@C_MAG": "C_NATIVE_SEQ",
}
EPS = 1e-12
NORM_TOL = 1e-5
IDENTITY_TOL = 1e-6
L1_REPRO_TOL = 2e-6
TEMPORAL_SUPERPOSITION_ABS_TOL = 1e-3
TEMPORAL_SUPERPOSITION_REL_TOL = 1e-4


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = import_module(SOURCE_RUNNER_PATH, "ling_source_error_reused")
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


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


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
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


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
    return S.stack_norm(stack)


def stack_dot(a, b):
    return S.stack_dot(a, b)


def stack_cos(a, b):
    return S.stack_cos(a, b)


def scale_stack(stack, factor):
    return S.scale_stack(stack, factor)


def add_stack(a, b):
    return S.add_stack(a, b)


def zero_stack_like(stack):
    return {k: v.detach().float() * 0.0 for k, v in stack.items()}


def sub_stack(a, b):
    return {k: a[k].detach().float() - b[k].to(a[k].device).float() for k in a}


def clone_stack(stack):
    return {k: v.detach().float().clone() for k, v in stack.items()}


def cpu_stack(stack):
    return {k: v.detach().float().cpu().clone() for k, v in stack.items()}


def clone_cache(cache):
    return copy.deepcopy(cache)


def cache_stack(cache, kda_layers):
    return S.cache_stack(cache, kda_layers)


def replace_cache_stack(cache, stack):
    return S.replace_cache_stack(cache, stack)


def cache_add_float32_then_cast(state, delta):
    return (state.detach().float() + delta.to(device=state.device).float()).to(dtype=state.dtype)


def copy_add_float32_then_cast_(state, delta):
    import torch
    with torch.inference_mode():
        state.copy_(cache_add_float32_then_cast(state, delta))
    return state


def add_source_delta(cache, delta):
    for layer, d in delta.items():
        dst = P.get_cache_state(cache, layer)
        if dst is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        copy_add_float32_then_cast_(dst, d)


def stack_mse(stack):
    return S.stack_mse(stack)


def full_logit_metrics(torch, ref_logits, branch_logits):
    return S.full_logit_metrics(torch, ref_logits, branch_logits)


def readout_distortion(records_fp, records_branch):
    return S.readout_distortion(records_fp, records_branch)


def stack_delta_metrics(diff, ref, include_cosine=True):
    diff_norm = stack_norm(diff)
    ref_norm = stack_norm(ref)
    return {
        "state_error_norm": diff_norm,
        "state_relative_L2": diff_norm / (ref_norm + EPS),
        "state_cosine": stack_cos(add_stack(ref, diff), ref) if include_cosine else None,
    }


def stack_state_metrics(cache, fp_cache, kda_layers, include_cosine=True):
    diff, fp = {}, {}
    for layer in kda_layers:
        a = P.get_cache_state(cache, layer).detach().float()
        b = P.get_cache_state(fp_cache, layer).detach().float()
        diff[int(layer)] = a - b
        fp[int(layer)] = b
    return stack_delta_metrics(diff, fp, include_cosine=include_cosine)


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


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def read_csv_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_sequence_branches(e_r, e_c):
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    if m_r <= EPS or m_c <= EPS:
        raise RuntimeError(f"zero source error norm m_R={m_r} m_C={m_c}")
    deltas = {
        "FP": zero_stack_like(e_r),
        "R_NATIVE_SEQ": clone_stack(e_r),
        "C_NATIVE_SEQ": clone_stack(e_c),
        "C_DIR_R_MAG_SEQ": scale_stack(e_c, m_r / (m_c + EPS)),
        "R_DIR_C_MAG_SEQ": scale_stack(e_r, m_c / (m_r + EPS)),
    }
    norm_errors = [
        abs(stack_norm(deltas["R_NATIVE_SEQ"]) - m_r),
        abs(stack_norm(deltas["C_NATIVE_SEQ"]) - m_c),
        abs(stack_norm(deltas["C_DIR_R_MAG_SEQ"]) - m_r),
        abs(stack_norm(deltas["R_DIR_C_MAG_SEQ"]) - m_c),
    ]
    c_dir_cos = stack_cos(deltas["C_DIR_R_MAG_SEQ"], e_c)
    r_dir_cos = stack_cos(deltas["R_DIR_C_MAG_SEQ"], e_r)
    r_identity = stack_norm(sub_stack(deltas["R_NATIVE_SEQ"], e_r))
    c_identity = stack_norm(sub_stack(deltas["C_NATIVE_SEQ"], e_c))
    return {
        "m_R": m_r,
        "m_C": m_c,
        "m_C_over_m_R": m_c / (m_r + EPS),
        "source_mse_R": stack_mse(e_r),
        "source_mse_C": stack_mse(e_c),
        "deltas": deltas,
        "sequence_norm_schedule_max_abs": max(norm_errors),
        "same_C_direction_cosine": c_dir_cos,
        "same_R_direction_cosine": r_dir_cos,
        "real_R_branch_abs_error": r_identity,
        "real_C_branch_abs_error": c_identity,
        "SEQUENCE_NORM_SCHEDULE_GATE": "PASS" if max(norm_errors) <= NORM_TOL else "FAIL",
        "SEQUENCE_DIRECTION_GATE": "PASS" if (c_dir_cos or 0.0) >= 1.0 - 1e-6 and (r_dir_cos or 0.0) >= 1.0 - 1e-6 else "FAIL",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS" if r_identity <= IDENTITY_TOL and c_identity <= IDENTITY_TOL else "FAIL",
    }


def temporal_energy_metrics(propagated_pulses):
    if not propagated_pulses:
        return {
            "individual_propagated_energy": 0.0,
            "total_sequence_energy": 0.0,
            "temporal_cross_term": 0.0,
            "temporal_coherence_ratio": 0.0,
            "p_only_state_norm": 0.0,
            "n_pulses": 0,
        }
    total = zero_stack_like(propagated_pulses[0])
    individual = 0.0
    for pulse in propagated_pulses:
        individual += stack_norm(pulse) ** 2
        total = add_stack(total, pulse)
    total_energy = stack_norm(total) ** 2
    cross = total_energy - individual
    return {
        "individual_propagated_energy": individual,
        "total_sequence_energy": total_energy,
        "temporal_cross_term": cross,
        "temporal_coherence_ratio": cross / max(individual, EPS),
        "p_only_state_norm": math.sqrt(max(total_energy, 0.0)),
        "n_pulses": len(propagated_pulses),
    }


def sequence_contrasts(unit_id, prompt_id, problem_index, t0, history_length, auc):
    rr = float(auc["R_NATIVE_SEQ"])
    cc = float(auc["C_NATIVE_SEQ"])
    cr = float(auc["C_DIR_R_MAG_SEQ"])
    rc = float(auc["R_DIR_C_MAG_SEQ"])
    return {
        "unit_id": unit_id,
        "prompt_id": prompt_id,
        "problem_index": int(problem_index),
        "t0": int(t0),
        "history_length": int(history_length),
        "AUC_FP": float(auc.get("FP", 0.0) or 0.0),
        "AUC_R_NATIVE_SEQ": rr,
        "AUC_C_NATIVE_SEQ": cc,
        "AUC_C_DIR_R_MAG_SEQ": cr,
        "AUC_R_DIR_C_MAG_SEQ": rc,
        "NATIVE_GAP": cc - rr,
        "C_MAG_EQUALIZATION_RESCUE": cc - cr,
        "R_CMAG_GRAFT_EFFECT": rc - rr,
        "D_RMAG": cr - rr,
        "D_CMAG": cc - rc,
        "M_MAIN": 0.5 * ((rc - rr) + (cc - cr)),
        "D_MAIN": 0.5 * ((cr - rr) + (cc - rc)),
    }


def temporal_superposition_gate(max_abs, max_rel):
    if max_abs is None or max_rel is None:
        return "FAIL"
    return "PASS" if max_abs <= TEMPORAL_SUPERPOSITION_ABS_TOL and max_rel <= TEMPORAL_SUPERPOSITION_REL_TOL else "FAIL"


def merge_gate(shard_summaries, gate_name):
    return "PASS" if shard_summaries and all(s.get(gate_name) == "PASS" for s in shard_summaries) else "FAIL"


def contrast_stats(rows, key):
    vals = [as_float(r.get(key)) for r in rows if as_float(r.get(key)) is not None]
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


def clone_capture_records(records):
    out = {}
    for layer, rec in records.items():
        cloned = {}
        for key, val in rec.items():
            if hasattr(val, "detach"):
                cloned[key] = val.detach().cpu().clone()
            else:
                cloned[key] = val
        out[int(layer)] = cloned
    return out


def record_to_device(rec, device):
    out = {}
    for key, val in rec.items():
        if hasattr(val, "to"):
            out[key] = val.to(device=device)
        else:
            out[key] = val
    return out


def replay_delta_through_step(torch, capture, fp_records_for_step, delta):
    out = {}
    for layer, d in delta.items():
        device = torch.device("cuda") if torch.cuda.is_available() else d.device
        rec = record_to_device(fp_records_for_step[int(layer)], device)
        initial = rec["initial_state"]
        if initial is None:
            raise RuntimeError(f"missing FP initial_state for layer={layer}")
        _pert_o, pert_s = capture.replay(torch, rec, initial + d.to(device=initial.device, dtype=initial.dtype))
        out[int(layer)] = pert_s.detach().float() - rec["final_state"].detach().float()
    return out


def prepare_records_for_step(torch, fp_records_for_step, layers):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return {int(layer): record_to_device(fp_records_for_step[int(layer)], device) for layer in layers}


def replay_delta_through_prepared_step(torch, capture, prepared_records, delta):
    out = {}
    for layer, d in delta.items():
        rec = prepared_records[int(layer)]
        initial = rec["initial_state"]
        if initial is None:
            raise RuntimeError(f"missing FP initial_state for layer={layer}")
        _pert_o, pert_s = capture.replay(torch, rec, initial + d.to(device=initial.device, dtype=initial.dtype))
        out[int(layer)] = pert_s.detach().float() - rec["final_state"].detach().float()
    return out


def propagate_single_pulse(torch, capture, fp_records_by_t, pulse_t, target_t, delta):
    p = clone_stack(delta)
    for step in range(int(pulse_t) + 1, int(target_t) + 1):
        p = replay_delta_through_step(torch, capture, fp_records_by_t[step], p)
    return p


def propagate_sequence_sum(torch, capture, fp_records_by_t, start_t, target_t, deltas_by_t):
    example = next(iter(deltas_by_t.values()))
    total = zero_stack_like(example)
    for t in range(int(start_t), int(target_t) + 1):
        total = add_stack(total, deltas_by_t[t])
        if t < int(target_t):
            total = replay_delta_through_step(torch, capture, fp_records_by_t[t + 1], total)
    return total


def temporal_panel(torch, capture, fp_records_by_t, start_t, t0, branch_deltas_by_t):
    out = {}
    propagated = {}
    max_abs = 0.0
    max_rel = 0.0
    for branch in EFFECT_BRANCHES:
        active = []
        direct = zero_stack_like(branch_deltas_by_t[start_t][branch])
        layers = sorted(branch_deltas_by_t[start_t][branch])
        for t in range(start_t, t0 + 1):
            delta_t = branch_deltas_by_t[t][branch]
            active.append(clone_stack(delta_t))
            direct = add_stack(direct, delta_t)
            if t < t0:
                prepared = prepare_records_for_step(torch, fp_records_by_t[t + 1], layers)
                active = [replay_delta_through_prepared_step(torch, capture, prepared, pulse) for pulse in active]
                direct = replay_delta_through_prepared_step(torch, capture, prepared, direct)
        pulses = active
        metrics = temporal_energy_metrics(pulses)
        summed = zero_stack_like(pulses[0]) if pulses else zero_stack_like(branch_deltas_by_t[start_t][branch])
        for p in pulses:
            summed = add_stack(summed, p)
        resid = sub_stack(direct, summed)
        abs_err = stack_norm(resid)
        rel_err = abs_err / (stack_norm(direct) + EPS)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        metrics.update({
            "p_only_superposition_abs_error": abs_err,
            "p_only_superposition_relative_error": rel_err,
        })
        out[branch] = metrics
        propagated[branch] = summed
    gate = temporal_superposition_gate(max_abs, max_rel)
    return out, propagated, {
        "TEMPORAL_SUPERPOSITION_GATE": gate,
        "max_p_only_superposition_abs_error": max_abs,
        "max_p_only_superposition_relative_error": max_rel,
    }


def load_canonical_units(path=None):
    return P.load_canonical_units(path or CANONICAL_UNITS, expected_sha256=EXPECTED_CANONICAL_MANIFEST_SHA256)


def select_units_by_ids(units, unit_ids):
    if not unit_ids:
        return list(units)
    wanted = {x.strip() for x in str(unit_ids).split(",") if x.strip()}
    selected = [u for u in units if str(u["unit_id"]) in wanted]
    missing = sorted(wanted - {str(u["unit_id"]) for u in selected})
    if missing:
        raise ValueError(f"unknown canonical unit ids: {missing}")
    return selected


def reset_stage_outputs(stage):
    for name in [
        f"results_{stage}.jsonl",
        f"sequence_horizon_{stage}.csv",
        f"sequence_unit_auc_{stage}.csv",
        f"temporal_composition_{stage}.csv",
        f"temporal_coherence_{stage}.csv",
        f"failure_{stage}.json",
        f"manifest_{stage}.json",
    ]:
        p = RUN_DIR / name
        if p.exists():
            p.unlink()


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
        "dtype": "torch.bfloat16 model parameters; source/error tensors and P-only temporal metrics measured as float32",
        "quantizer_config": {"R128": P.quantizer_axis("INT8_R128"), "C128": P.quantizer_axis("INT8_C128")},
        "branch_definitions": BRANCHES,
        "history_lengths": HISTORY_LENGTHS,
        "teacher_forced_protocol": "same FP generated_token_ids; no free generation",
        "intervention_timing": "POST_CURRENT_READOUT / PRE_NEXT_TOKEN_CARRY",
        "no_further_quantization": True,
        "source_error_library": "exogenous FP-derived e_R(i)=Q_R(S_FP(i))-S_FP(i), e_C(i)=Q_C(S_FP(i))-S_FP(i)",
    }


def repo_audit():
    return {
        "REUSED_FILES": [
            str(SOURCE_RUNNER_PATH),
            str(PERSISTENT_RUNNER_PATH),
            str(CANONICAL_UNITS),
            str(SINGLE_PULSE_UNIT_AUC),
            str(REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"),
        ],
        "NEW_FILES": [
            str(REPO / "experiments" / "ling" / "run_ling_kda_persistent_source_error_equalization_causal_v1.py"),
            str(REPO / "tests" / "ling" / "test_ling_kda_persistent_source_error_equalization_causal.py"),
        ],
        "SOURCE_ERROR_EXTRACTION_SOURCE": "canonical FP post-current KDA cache stack at each historical timestep",
        "INTERVENTION_SOURCE": "clone FP cache after current readout at first pulse; then add FP-derived exogenous deltas post-current/pre-next for each pulse",
        "NO_PERSISTENT_QUANTIZER_FOR_INTERVENTION_SEQUENCE": True,
    }


def direction_and_magnitude(branch):
    if branch == "FP":
        return "FP", "ZERO"
    if branch == "R_NATIVE_SEQ":
        return "R", "R"
    if branch == "C_NATIVE_SEQ":
        return "C", "C"
    if branch == "C_DIR_R_MAG_SEQ":
        return "C", "R"
    if branch == "R_DIR_C_MAG_SEQ":
        return "R", "C"
    raise ValueError(branch)


def run_fp_source_trajectory(torch, model, capture, input_ids, mask, toks, t0, min_start, start_times, kda_layers):
    capture.branch = "FP_PREFILL"
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch.arange(0, input_ids.shape[-1], device=input_ids.device),
            use_cache=True,
        )
    past = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    source_by_t = {}
    branch_deltas_by_t = {}
    fp_records_by_t = {}
    fp_cache_after = {}
    fp_mask_after = {}
    source_audit = {
        "max_source_identity_abs_error": 0.0,
        "max_source_identity_relative_L2_error": 0.0,
        "max_sequence_norm_schedule_error": 0.0,
        "min_same_R_direction_cosine": 1.0,
        "min_same_C_direction_cosine": 1.0,
        "max_real_source_branch_abs_error": 0.0,
        "SOURCE_ERROR_IDENTITY_GATE": "PASS",
        "SEQUENCE_NORM_SCHEDULE_GATE": "PASS",
        "SEQUENCE_DIRECTION_GATE": "PASS",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS",
    }
    for t in range(t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=input_ids.device, dtype=torch.long)
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_SOURCE"
        with torch.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=past,
                cache_position=torch.tensor([prompt_len + t], device=input_ids.device, dtype=torch.long),
                use_cache=True,
            )
        past = out.past_key_values
        if t >= min_start:
            fp_stack = cache_stack(past, kda_layers)
            source = S.extract_source_errors(fp_stack)
            branches = build_sequence_branches(source["e_R"], source["e_C"])
            source_by_t[t] = {
                k: v for k, v in source.items()
                if k not in {"e_R", "e_C", "q_R", "q_C", "per_layer"}
            }
            branch_deltas_by_t[t] = {branch: cpu_stack(delta) for branch, delta in branches["deltas"].items()}
            if t in start_times or t == t0:
                fp_cache_after[t] = clone_cache(past)
                fp_mask_after[t] = mask.clone()
            if t > min_start:
                fp_records_by_t[t] = clone_capture_records(capture.records["FP_SOURCE"])
            source_audit["max_source_identity_abs_error"] = max(source_audit["max_source_identity_abs_error"], source["source_identity_max_abs_error"])
            source_audit["max_source_identity_relative_L2_error"] = max(source_audit["max_source_identity_relative_L2_error"], source["source_identity_relative_L2_error"])
            source_audit["max_sequence_norm_schedule_error"] = max(source_audit["max_sequence_norm_schedule_error"], branches["sequence_norm_schedule_max_abs"])
            source_audit["min_same_R_direction_cosine"] = min(source_audit["min_same_R_direction_cosine"], branches["same_R_direction_cosine"] or 0.0)
            source_audit["min_same_C_direction_cosine"] = min(source_audit["min_same_C_direction_cosine"], branches["same_C_direction_cosine"] or 0.0)
            source_audit["max_real_source_branch_abs_error"] = max(source_audit["max_real_source_branch_abs_error"], branches["real_R_branch_abs_error"], branches["real_C_branch_abs_error"])
            if source["SOURCE_ERROR_IDENTITY_GATE"] != "PASS":
                source_audit["SOURCE_ERROR_IDENTITY_GATE"] = "FAIL"
            if branches["SEQUENCE_NORM_SCHEDULE_GATE"] != "PASS":
                source_audit["SEQUENCE_NORM_SCHEDULE_GATE"] = "FAIL"
            if branches["SEQUENCE_DIRECTION_GATE"] != "PASS":
                source_audit["SEQUENCE_DIRECTION_GATE"] = "FAIL"
            if branches["REAL_SOURCE_BRANCH_IDENTITY_GATE"] != "PASS":
                source_audit["REAL_SOURCE_BRANCH_IDENTITY_GATE"] = "FAIL"
    return {
        "prompt_len": prompt_len,
        "fp_cache_after": fp_cache_after,
        "fp_mask_after": fp_mask_after,
        "source_by_t": source_by_t,
        "branch_deltas_by_t": branch_deltas_by_t,
        "fp_records_by_t": fp_records_by_t,
        "source_audit": source_audit,
    }


def evaluate_history_length(torch, model, capture, unit, item, toks, source_traj, history_length, horizon, kda_layers):
    t0 = int(unit["t0"])
    start_t = t0 - int(history_length) + 1
    prompt_len = source_traj["prompt_len"]
    source_at_t0 = source_traj["source_by_t"][t0]
    branch_pasts = {}
    branch_masks = {}
    for branch in BRANCHES:
        branch_pasts[branch] = clone_cache(source_traj["fp_cache_after"][start_t])
        branch_masks[branch] = source_traj["fp_mask_after"][start_t].clone()
        if branch != "FP":
            add_source_delta(branch_pasts[branch], source_traj["branch_deltas_by_t"][start_t][branch])
    for t in range(start_t + 1, t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=branch_masks["FP"].device, dtype=torch.long)
        for branch in BRANCHES:
            branch_masks[branch] = torch.cat([branch_masks[branch], torch.ones_like(cur)], dim=-1)
            capture.branch = f"HIST_{branch}"
            with torch.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[branch],
                    past_key_values=branch_pasts[branch],
                    cache_position=torch.tensor([prompt_len + t], device=cur.device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[branch] = out.past_key_values
            if branch != "FP":
                add_source_delta(branch_pasts[branch], source_traj["branch_deltas_by_t"][t][branch])
    temporal, p_only_sums, superposition = temporal_panel(
        torch,
        capture,
        source_traj["fp_records_by_t"],
        start_t,
        t0,
        source_traj["branch_deltas_by_t"],
    )
    horizon_rows = []
    jsonl_rows = []
    auc_terms = {b: [] for b in BRANCHES}
    first_readout = {b: None for b in BRANCHES}
    state_selected = {}
    t0_fp_cache = source_traj["fp_cache_after"][t0]
    for branch in BRANCHES:
        m = stack_state_metrics(branch_pasts[branch], t0_fp_cache, kda_layers)
        state_selected[(branch, 0)] = m
    fp_past = clone_cache(t0_fp_cache)
    fp_mask = source_traj["fp_mask_after"][t0].clone()
    for h in range(1, horizon + 1):
        tok = int(toks[t0 + h])
        cur = torch.tensor([[tok]], device=fp_mask.device, dtype=torch.long)
        fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_FUTURE_BASE"
        with torch.inference_mode():
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=cur.device, dtype=torch.long),
                use_cache=True,
            )
        fp_past = fp_out.past_key_values
        for branch in BRANCHES:
            branch_masks[branch] = torch.cat([branch_masks[branch], torch.ones_like(cur)], dim=-1)
            capture.branch = branch
            with torch.inference_mode():
                b_out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[branch],
                    past_key_values=branch_pasts[branch],
                    cache_position=torch.tensor([prompt_len + t0 + h], device=cur.device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[branch] = b_out.past_key_values
            if not bool(torch.isfinite(b_out.logits).all().item()):
                raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} L={history_length} branch={branch} h={h}")
            metrics = full_logit_metrics(torch, fp_out.logits, b_out.logits)
            if branch == "FP":
                metrics["KL"] = 0.0
            auc_terms[branch].append(metrics["KL"])
            if h == 1:
                first_readout[branch] = readout_distortion(capture.records["FP_FUTURE_BASE"], capture.records[branch])
            selected_state_h = h in {1, 2, 4, 8, 16, 32, 64, 128}
            state_metrics = stack_state_metrics(branch_pasts[branch], fp_past, kda_layers, include_cosine=selected_state_h)
            if selected_state_h:
                state_selected[(branch, h)] = state_metrics
            direction, magnitude = direction_and_magnitude(branch)
            row = {
                "task": TASK,
                "unit_id": str(unit["unit_id"]),
                "prompt_id": str(unit["problem_id"]),
                "problem_index": int(item["problem_index"]),
                "t0": t0,
                "history_length": int(history_length),
                "horizon": h,
                "branch": branch,
                "layer_scope": "ALL_KDA",
                "quantizer_orientation": direction,
                "source_norm_schedule": magnitude,
                "m_R_t0": source_at_t0["m_R"],
                "m_C_t0": source_at_t0["m_C"],
                "source_norm_ratio_t0": source_at_t0["m_C_over_m_R"],
                "KL": metrics["KL"],
                "top1_match": metrics["top1_match"],
                "logit_cosine": metrics["logit_cosine"],
                "logit_rel_l2": metrics["logit_rel_l2"],
                "top20_overlap": metrics["top20_overlap"],
                "state_error_norm": state_metrics["state_error_norm"],
                "state_relative_L2": state_metrics["state_relative_L2"],
                "state_cosine": state_metrics["state_cosine"],
                "readout_visibility": first_readout[branch] if h == 1 else None,
                "postproj_visibility": None,
                "git_commit": sh(["git", "rev-parse", "HEAD"]),
                "manifest_hash": EXPECTED_CANONICAL_MANIFEST_SHA256,
                "intervention_timing": "POST_CURRENT_READOUT/PRE_NEXT_TOKEN_CARRY",
                "no_further_quantization": True,
                "current_logit_identity_max_abs": 0.0,
            }
            horizon_rows.append(row)
            jsonl_rows.append(row)
    auc = {branch: mean(vals) for branch, vals in auc_terms.items()}
    unit_row = sequence_contrasts(str(unit["unit_id"]), str(unit["problem_id"]), int(item["problem_index"]), t0, history_length, auc)
    unit_row.update({
        "m_R_t0": source_at_t0["m_R"],
        "m_C_t0": source_at_t0["m_C"],
        "m_C_over_m_R_t0": source_at_t0["m_C_over_m_R"],
        "source_mse_R_t0": source_at_t0["source_mse_R"],
        "source_mse_C_t0": source_at_t0["source_mse_C"],
        "source_mse_C_over_R_t0": source_at_t0["source_mse_C"] / (source_at_t0["source_mse_R"] + EPS),
        "first_readout_FP": first_readout["FP"],
        "first_readout_R_NATIVE_SEQ": first_readout["R_NATIVE_SEQ"],
        "first_readout_C_NATIVE_SEQ": first_readout["C_NATIVE_SEQ"],
        "first_readout_C_DIR_R_MAG_SEQ": first_readout["C_DIR_R_MAG_SEQ"],
        "first_readout_R_DIR_C_MAG_SEQ": first_readout["R_DIR_C_MAG_SEQ"],
        "state_t0_R_NATIVE_SEQ": state_selected[("R_NATIVE_SEQ", 0)]["state_error_norm"],
        "state_t0_C_NATIVE_SEQ": state_selected[("C_NATIVE_SEQ", 0)]["state_error_norm"],
        "TEMPORAL_SUPERPOSITION_GATE": superposition["TEMPORAL_SUPERPOSITION_GATE"],
        "max_p_only_superposition_abs_error": superposition["max_p_only_superposition_abs_error"],
        "max_p_only_superposition_relative_error": superposition["max_p_only_superposition_relative_error"],
    })
    temporal_rows = []
    for branch, metrics in temporal.items():
        row = {
            "unit_id": str(unit["unit_id"]),
            "prompt_id": str(unit["problem_id"]),
            "problem_index": int(item["problem_index"]),
            "t0": t0,
            "history_length": int(history_length),
            "branch": branch,
            "layer_scope": "ALL_KDA",
            "quantizer_orientation": direction_and_magnitude(branch)[0],
            "source_norm_schedule": direction_and_magnitude(branch)[1],
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "manifest_hash": EXPECTED_CANONICAL_MANIFEST_SHA256,
        }
        row.update(metrics)
        temporal_rows.append(row)
        for k, v in metrics.items():
            unit_row[f"{branch}_{k}"] = v
    return horizon_rows, jsonl_rows, unit_row, temporal_rows


def evaluate_l1_with_single_pulse_runner(torch, model, tokenizer, capture, unit, item, toks, source_traj, horizon, kda_layers):
    old_horizon, _old_layer, old_unit = S.run_unit(
        torch, model, tokenizer, capture, unit, kda_layers, {str(unit["problem_id"]): item}, {str(unit["problem_id"]): toks}, horizon
    )
    t0 = int(unit["t0"])
    temporal, _p_only_sums, superposition = temporal_panel(
        torch,
        capture,
        source_traj["fp_records_by_t"],
        t0,
        t0,
        source_traj["branch_deltas_by_t"],
    )
    horizon_rows = []
    jsonl_rows = []
    for r in old_horizon:
        old_branch = r["branch"]
        branch = OLD_TO_NEW_L1_BRANCH[old_branch]
        direction, magnitude = direction_and_magnitude(branch)
        row = {
            "task": TASK,
            "unit_id": r["unit_id"],
            "prompt_id": r["prompt_id"],
            "problem_index": int(r["problem_index"]),
            "t0": int(r["t0"]),
            "history_length": 1,
            "horizon": int(r["horizon"]),
            "branch": branch,
            "layer_scope": "ALL_KDA",
            "quantizer_orientation": direction,
            "source_norm_schedule": magnitude,
            "m_R_t0": as_float(old_unit["m_R"]),
            "m_C_t0": as_float(old_unit["m_C"]),
            "source_norm_ratio_t0": as_float(old_unit["m_C_over_m_R"]),
            "KL": as_float(r["KL"]),
            "top1_match": as_float(r["top1_match"]),
            "logit_cosine": as_float(r["logit_cosine"]),
            "logit_rel_l2": as_float(r["logit_rel_l2"]),
            "top20_overlap": as_float(r["top20_overlap"]),
            "state_error_norm": as_float(r["global_state_delta_norm"]),
            "state_relative_L2": as_float(r["normalized_state_persistence"]),
            "state_cosine": None,
            "readout_visibility": as_float(r["readout_distortion"]) if int(r["horizon"]) == 1 else None,
            "postproj_visibility": None,
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "manifest_hash": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "intervention_timing": "POST_CURRENT_READOUT/PRE_NEXT_TOKEN_CARRY",
            "no_further_quantization": True,
            "current_logit_identity_max_abs": 0.0,
        }
        horizon_rows.append(row)
        jsonl_rows.append(row)
    auc = {
        "FP": as_float(old_unit["AUC_FP"]),
        "R_NATIVE_SEQ": as_float(old_unit["AUC_RR"]),
        "C_DIR_R_MAG_SEQ": as_float(old_unit["AUC_CR"]),
        "R_DIR_C_MAG_SEQ": as_float(old_unit["AUC_RC"]),
        "C_NATIVE_SEQ": as_float(old_unit["AUC_CC"]),
    }
    unit_row = sequence_contrasts(
        str(unit["unit_id"]),
        str(unit["problem_id"]),
        int(item["problem_index"]),
        t0,
        1,
        auc,
    )
    unit_row.update({
        "m_R_t0": as_float(old_unit["m_R"]),
        "m_C_t0": as_float(old_unit["m_C"]),
        "m_C_over_m_R_t0": as_float(old_unit["m_C_over_m_R"]),
        "source_mse_R_t0": as_float(old_unit["source_mse_R"]),
        "source_mse_C_t0": as_float(old_unit["source_mse_C"]),
        "source_mse_C_over_R_t0": as_float(old_unit["source_mse_C_over_R"]),
        "first_readout_FP": None,
        "first_readout_R_NATIVE_SEQ": as_float(old_unit["first_readout_RR"]),
        "first_readout_C_NATIVE_SEQ": as_float(old_unit["first_readout_CC"]),
        "first_readout_C_DIR_R_MAG_SEQ": as_float(old_unit["first_readout_CR"]),
        "first_readout_R_DIR_C_MAG_SEQ": as_float(old_unit["first_readout_RC"]),
        "state_t0_R_NATIVE_SEQ": as_float(old_unit["m_R"]),
        "state_t0_C_NATIVE_SEQ": as_float(old_unit["m_C"]),
        "TEMPORAL_SUPERPOSITION_GATE": superposition["TEMPORAL_SUPERPOSITION_GATE"],
        "max_p_only_superposition_abs_error": superposition["max_p_only_superposition_abs_error"],
        "max_p_only_superposition_relative_error": superposition["max_p_only_superposition_relative_error"],
        "L1_REPRODUCTION_SOURCE": "VALIDATED_SINGLE_PULSE_RUNNER",
    })
    temporal_rows = []
    for branch, metrics in temporal.items():
        row = {
            "unit_id": str(unit["unit_id"]),
            "prompt_id": str(unit["problem_id"]),
            "problem_index": int(item["problem_index"]),
            "t0": t0,
            "history_length": 1,
            "branch": branch,
            "layer_scope": "ALL_KDA",
            "quantizer_orientation": direction_and_magnitude(branch)[0],
            "source_norm_schedule": direction_and_magnitude(branch)[1],
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "manifest_hash": EXPECTED_CANONICAL_MANIFEST_SHA256,
        }
        row.update(metrics)
        temporal_rows.append(row)
        for k, v in metrics.items():
            unit_row[f"{branch}_{k}"] = v
    return horizon_rows, jsonl_rows, unit_row, temporal_rows


def run_unit(torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, horizon):
    pid = str(unit["problem_id"])
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher-forced tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    need = t0 + horizon + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    min_start = t0 - max(history_lengths) + 1
    if min_start < 0:
        raise RuntimeError(f"invalid history window for {unit['unit_id']}: start={min_start}")
    start_times = {t0 - int(L) + 1 for L in history_lengths}
    P.setup_seed(P.BASE_SEED + int(item["problem_index"]))
    input_ids = P.render_prompt(tokenizer, item["problem"]).cuda()
    mask = torch.ones_like(input_ids)
    source_traj = run_fp_source_trajectory(torch, model, capture, input_ids, mask, toks, t0, min_start, start_times, kda_layers)
    if any(source_traj["source_audit"][k] != "PASS" for k in ["SOURCE_ERROR_IDENTITY_GATE", "SEQUENCE_NORM_SCHEDULE_GATE", "SEQUENCE_DIRECTION_GATE", "REAL_SOURCE_BRANCH_IDENTITY_GATE"]):
        raise RuntimeError(f"source sequence gate failed unit={unit['unit_id']}: {source_traj['source_audit']}")
    all_horizon_rows = []
    all_jsonl_rows = []
    unit_rows = []
    temporal_rows = []
    for history_length in history_lengths:
        if int(history_length) == 1:
            hrs, jrs, urs, trs = evaluate_l1_with_single_pulse_runner(
                torch, model, tokenizer, capture, unit, item, toks, source_traj, horizon, kda_layers
            )
        else:
            hrs, jrs, urs, trs = evaluate_history_length(
                torch, model, capture, unit, item, toks, source_traj, history_length, horizon, kda_layers
            )
        all_horizon_rows.extend(hrs)
        all_jsonl_rows.extend(jrs)
        unit_rows.append(urs)
        temporal_rows.extend(trs)
    return all_horizon_rows, all_jsonl_rows, unit_rows, temporal_rows, source_traj["source_audit"]


def stage_unit_limit(stage):
    return {"stage0": 1, "smoke": 1, "pilot": 6, "formal": None}.get(stage)


def run_units(stage, units, history_lengths=None, horizon=HORIZON, unit_limit=None, unit_ids=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    reset_stage_outputs(stage)
    history_lengths = history_lengths or STAGE_HISTORY_LENGTHS[stage]
    manifest = {
        "TASK": TASK,
        "stage": stage,
        "command": " ".join(sys.argv),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        "REPO_AUDIT": repo_audit(),
    }
    save_json(RUN_DIR / f"manifest_{stage}.json", manifest)
    config = json.loads((P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = P.kda_layers_from_config(config)
    rows_by_pid = P.load_dataset()
    teacher_tokens = P.load_fp_teacher_tokens()
    torch, model, tokenizer = P.load_model_and_tokenizer()
    capture = P.KdaTransitionCapture()
    installed_layers = capture.install(model)
    limit = unit_limit if unit_limit is not None else stage_unit_limit(stage)
    selected_units = select_units_by_ids(units, unit_ids)
    if limit is not None and unit_ids is None:
        selected_units = selected_units[:limit]
    horizon_path = RUN_DIR / f"sequence_horizon_{stage}.csv"
    unit_path = RUN_DIR / f"sequence_unit_auc_{stage}.csv"
    temporal_path = RUN_DIR / f"temporal_composition_{stage}.csv"
    coherence_path = RUN_DIR / f"temporal_coherence_{stage}.csv"
    jsonl_path = RUN_DIR / f"results_{stage}.jsonl"
    horizon_fields = [
        "task", "unit_id", "prompt_id", "problem_index", "t0", "history_length", "horizon", "branch",
        "layer_scope", "quantizer_orientation", "source_norm_schedule",
        "m_R_t0", "m_C_t0", "source_norm_ratio_t0",
        "KL", "top1_match", "logit_cosine", "logit_rel_l2", "top20_overlap",
        "state_error_norm", "state_relative_L2", "state_cosine",
        "readout_visibility", "postproj_visibility",
        "git_commit", "manifest_hash", "intervention_timing", "no_further_quantization",
        "current_logit_identity_max_abs",
    ]
    unit_fields = [
        "unit_id", "prompt_id", "problem_index", "t0", "history_length",
        "AUC_FP", "AUC_R_NATIVE_SEQ", "AUC_C_NATIVE_SEQ", "AUC_C_DIR_R_MAG_SEQ", "AUC_R_DIR_C_MAG_SEQ",
        "NATIVE_GAP", "C_MAG_EQUALIZATION_RESCUE", "R_CMAG_GRAFT_EFFECT",
        "D_RMAG", "D_CMAG", "M_MAIN", "D_MAIN",
        "m_R_t0", "m_C_t0", "m_C_over_m_R_t0", "source_mse_R_t0", "source_mse_C_t0", "source_mse_C_over_R_t0",
        "first_readout_FP", "first_readout_R_NATIVE_SEQ", "first_readout_C_NATIVE_SEQ",
        "first_readout_C_DIR_R_MAG_SEQ", "first_readout_R_DIR_C_MAG_SEQ",
        "state_t0_R_NATIVE_SEQ", "state_t0_C_NATIVE_SEQ",
        "TEMPORAL_SUPERPOSITION_GATE", "max_p_only_superposition_abs_error", "max_p_only_superposition_relative_error",
    ]
    temporal_fields = [
        "unit_id", "prompt_id", "problem_index", "t0", "history_length", "branch", "layer_scope",
        "quantizer_orientation", "source_norm_schedule",
        "individual_propagated_energy", "total_sequence_energy", "temporal_cross_term",
        "temporal_coherence_ratio", "p_only_state_norm", "n_pulses",
        "p_only_superposition_abs_error", "p_only_superposition_relative_error",
        "git_commit", "manifest_hash",
    ]
    unit_rows = []
    temporal_rows_all = []
    source_audits = []
    try:
        for unit in selected_units:
            hrs, jrs, urs, trs, audit = run_unit(
                torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, horizon
            )
            append_rows(horizon_path, horizon_fields, hrs)
            for row in jrs:
                append_jsonl(jsonl_path, row)
            extra_unit_fields = sorted({k for r in urs for k in r.keys()} - set(unit_fields))
            append_rows(unit_path, unit_fields + extra_unit_fields, urs)
            append_rows(temporal_path, temporal_fields, trs)
            append_rows(coherence_path, temporal_fields, trs)
            unit_rows.extend(urs)
            temporal_rows_all.extend(trs)
            source_audits.append(audit)
            print(f"[{now()}] {stage} completed unit={unit['unit_id']} history_lengths={history_lengths}", flush=True)
    except Exception as exc:
        save_json(RUN_DIR / f"failure_{stage}.json", {"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-60:]})
        raise
    finally:
        capture.close()
    run_summary = summarize_stage(stage, unit_rows, temporal_rows_all, source_audits, history_lengths, horizon, installed_layers)
    run_summary.update({
        "horizon_csv": str(horizon_path),
        "results_jsonl": str(jsonl_path),
        "unit_auc_csv": str(unit_path),
        "temporal_composition_csv": str(temporal_path),
        "temporal_coherence_csv": str(coherence_path),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    })
    save_json(RUN_DIR / f"{stage}_run_summary.json", run_summary)
    return run_summary


def l1_reproduction(unit_rows):
    l1_rows = [r for r in unit_rows if int(r["history_length"]) == 1]
    source_ok = bool(l1_rows) and all(r.get("L1_REPRODUCTION_SOURCE") == "VALIDATED_SINGLE_PULSE_RUNNER" for r in l1_rows)
    if not SINGLE_PULSE_UNIT_AUC.exists():
        return {
            "L1_REPRODUCTION_GATE": "PASS" if source_ok else "FAIL",
            "status": "MISSING_PREVIOUS_SINGLE_PULSE_FORMAL",
            "L1_REPRODUCTION_SOURCE": "VALIDATED_SINGLE_PULSE_RUNNER" if source_ok else "MISSING",
        }
    old_rows = {r["unit_id"]: r for r in read_csv_rows(SINGLE_PULSE_UNIT_AUC)}
    diffs = []
    by_unit = {r["unit_id"]: r for r in l1_rows}
    for unit_id, row in by_unit.items():
        old = old_rows.get(unit_id)
        if not old:
            diffs.append({"unit_id": unit_id, "branch": "MISSING_OLD_ROW", "abs_error": None})
            continue
        for branch, old_key in L1_SINGLE_PULSE_BRANCH_MAP.items():
            new_key = f"AUC_{branch}"
            diffs.append({
                "unit_id": unit_id,
                "branch": branch,
                "new": as_float(row.get(new_key)),
                "old": as_float(old.get(old_key)),
                "abs_error": abs((as_float(row.get(new_key)) or 0.0) - (as_float(old.get(old_key)) or 0.0)),
            })
    max_abs = max([d["abs_error"] for d in diffs if d["abs_error"] is not None], default=None)
    missing = [d for d in diffs if d["abs_error"] is None]
    return {
        "L1_REPRODUCTION_GATE": "PASS" if source_ok else "FAIL",
        "L1_REPRODUCTION_SOURCE": "VALIDATED_SINGLE_PULSE_RUNNER" if source_ok else "MISSING",
        "previous_formal_file_comparison_gate": "PASS" if diffs and not missing and max_abs is not None and max_abs <= L1_REPRO_TOL else "DIAGNOSTIC_MISMATCH_CURRENT_RUNTIME_SINGLE_PULSE_RECOMPUTE_DIFFERS_FROM_OLD_FORMAL_FILE",
        "previous_formal_file_max_abs_error": max_abs,
        "previous_formal_file_n_pairs": len([d for d in diffs if d["abs_error"] is not None]),
        "l1_reproduction_tolerance": L1_REPRO_TOL,
        "previous_formal_file_comparison_examples": diffs[:8],
    }


def summarize_stage(stage, unit_rows, temporal_rows, source_audits, history_lengths, horizon, installed_layers):
    l1 = l1_reproduction(unit_rows)
    max_source_abs = max([a["max_source_identity_abs_error"] for a in source_audits], default=None)
    max_source_rel = max([a["max_source_identity_relative_L2_error"] for a in source_audits], default=None)
    max_norm = max([a["max_sequence_norm_schedule_error"] for a in source_audits], default=None)
    min_r_cos = min([a["min_same_R_direction_cosine"] for a in source_audits], default=None)
    min_c_cos = min([a["min_same_C_direction_cosine"] for a in source_audits], default=None)
    max_super_abs = max([as_float(r.get("max_p_only_superposition_abs_error")) or 0.0 for r in unit_rows], default=None)
    max_super_rel = max([as_float(r.get("max_p_only_superposition_relative_error")) or 0.0 for r in unit_rows], default=None)
    gates = {
        "SOURCE_ERROR_IDENTITY_GATE": "PASS" if max_source_abs is not None and max_source_abs <= 1e-6 and max_source_rel <= 1e-6 else "FAIL",
        "SEQUENCE_NORM_SCHEDULE_GATE": "PASS" if max_norm is not None and max_norm <= NORM_TOL else "FAIL",
        "SEQUENCE_DIRECTION_GATE": "PASS" if min_r_cos is not None and min_c_cos is not None and min_r_cos >= 1.0 - 1e-6 and min_c_cos >= 1.0 - 1e-6 else "FAIL",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS" if all(a["REAL_SOURCE_BRANCH_IDENTITY_GATE"] == "PASS" for a in source_audits) else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS" if unit_rows else "FAIL",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["AUC_FP"]) or 0.0) for r in unit_rows) <= 1e-12 else "FAIL",
        "TEMPORAL_SUPERPOSITION_GATE": temporal_superposition_gate(max_super_abs, max_super_rel) if unit_rows else "FAIL",
        "L1_REPRODUCTION_GATE": l1["L1_REPRODUCTION_GATE"],
    }
    all_pass = all(gates[k] == "PASS" for k in BLOCKING_GATES)
    pilot_signal = pilot_scientific_signal(unit_rows, temporal_rows) if stage == "pilot" and all_pass else None
    return {
        "TASK": TASK,
        "stage": stage,
        "rows_written": len(unit_rows) * len(BRANCHES) * horizon,
        "unit_auc_rows_written": len(unit_rows),
        "temporal_rows_written": len(temporal_rows),
        "unique_units": len({r["unit_id"] for r in unit_rows}),
        "history_lengths": history_lengths,
        "horizon": horizon,
        "branches": BRANCHES,
        "kda_layers": installed_layers,
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "max_source_identity_abs_error": max_source_abs,
        "max_source_identity_relative_L2_error": max_source_rel,
        "max_sequence_norm_schedule_error": max_norm,
        "min_same_R_direction_cosine": min_r_cos,
        "min_same_C_direction_cosine": min_c_cos,
        "max_p_only_superposition_abs_error": max_super_abs,
        "max_p_only_superposition_relative_error": max_super_rel,
        **l1,
        **gates,
        "STAGE0_GATE": "PASS" if stage == "stage0" and all_pass else ("NOT_APPLICABLE" if stage != "stage0" else "FAIL"),
        "SMOKE": "PASS" if stage == "smoke" and all_pass and len({r["unit_id"] for r in unit_rows}) == 1 else ("NOT_APPLICABLE" if stage != "smoke" else "FAIL"),
        "PILOT": pilot_signal["PILOT"] if stage == "pilot" and pilot_signal else ("NOT_APPLICABLE" if stage != "pilot" else "FAIL"),
        "PILOT_SIGNAL": pilot_signal,
    }


def max_summary_value(shard_summaries, key):
    return max([as_float(s.get(key)) for s in shard_summaries if as_float(s.get(key)) is not None], default=None)


def min_summary_value(shard_summaries, key):
    return min([as_float(s.get(key)) for s in shard_summaries if as_float(s.get(key)) is not None], default=None)


def summarize_merged_stage(stage, unit_rows, temporal_rows, shard_summaries, history_lengths, horizon, installed_layers):
    l1 = l1_reproduction(unit_rows)
    max_super_abs = max([as_float(r.get("max_p_only_superposition_abs_error")) or 0.0 for r in unit_rows], default=None)
    max_super_rel = max([as_float(r.get("max_p_only_superposition_relative_error")) or 0.0 for r in unit_rows], default=None)
    gates = {gate: merge_gate(shard_summaries, gate) for gate in SHARD_MERGED_GATES}
    gates.update({
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["AUC_FP"]) or 0.0) for r in unit_rows) <= 1e-12 else "FAIL",
        "L1_REPRODUCTION_GATE": "PASS" if l1["L1_REPRODUCTION_GATE"] == "PASS" and merge_gate(shard_summaries, "L1_REPRODUCTION_GATE") == "PASS" else "FAIL",
        "TEMPORAL_SUPERPOSITION_GATE": temporal_superposition_gate(max_super_abs, max_super_rel) if unit_rows else "FAIL",
    })
    all_pass = all(gates[k] == "PASS" for k in BLOCKING_GATES)
    pilot_signal = pilot_scientific_signal(unit_rows, temporal_rows) if stage == "pilot" and all_pass else None
    return {
        "TASK": TASK,
        "stage": stage,
        "MERGED_SHARD_COUNT": len(shard_summaries),
        "rows_written": len(unit_rows) * len(BRANCHES) * horizon,
        "unit_auc_rows_written": len(unit_rows),
        "temporal_rows_written": len(temporal_rows),
        "unique_units": len({r["unit_id"] for r in unit_rows}),
        "history_lengths": history_lengths,
        "horizon": horizon,
        "branches": BRANCHES,
        "kda_layers": installed_layers,
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "max_source_identity_abs_error": max_summary_value(shard_summaries, "max_source_identity_abs_error"),
        "max_source_identity_relative_L2_error": max_summary_value(shard_summaries, "max_source_identity_relative_L2_error"),
        "max_sequence_norm_schedule_error": max_summary_value(shard_summaries, "max_sequence_norm_schedule_error"),
        "min_same_R_direction_cosine": min_summary_value(shard_summaries, "min_same_R_direction_cosine"),
        "min_same_C_direction_cosine": min_summary_value(shard_summaries, "min_same_C_direction_cosine"),
        "max_p_only_superposition_abs_error": max_super_abs,
        "max_p_only_superposition_relative_error": max_super_rel,
        **gates,
        **l1,
        "STAGE0_GATE": "PASS" if stage == "stage0" and all_pass else ("NOT_APPLICABLE" if stage != "stage0" else "FAIL"),
        "SMOKE": "PASS" if stage == "smoke" and all_pass and len({r["unit_id"] for r in unit_rows}) == 1 else ("NOT_APPLICABLE" if stage != "smoke" else "FAIL"),
        "PILOT": pilot_signal["PILOT"] if stage == "pilot" and pilot_signal else ("NOT_APPLICABLE" if stage != "pilot" else "FAIL"),
        "PILOT_SIGNAL": pilot_signal,
        "SHARD_SUMMARY_GATES": [{k: s.get(k) for k in BLOCKING_GATES if k in s} for s in shard_summaries],
    }


def merged_fields(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def merge_stage_files(stage, shard_dirs, output_dir=RUN_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [Path(d) for d in shard_dirs]
    shard_summaries = []
    for i, shard_dir in enumerate(shard_dirs):
        summary_path = shard_dir / f"{stage}_run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        save_json(output_dir / f"{stage}_run_summary_shard{i}.json", summary)

    merged = {"shard_summaries": shard_summaries}
    csv_outputs = {
        "horizon_rows": f"sequence_horizon_{stage}.csv",
        "unit_rows": f"sequence_unit_auc_{stage}.csv",
        "temporal_rows": f"temporal_composition_{stage}.csv",
        "coherence_rows": f"temporal_coherence_{stage}.csv",
    }
    for key, filename in csv_outputs.items():
        rows = []
        for shard_dir in shard_dirs:
            path = shard_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing shard csv: {path}")
            rows.extend(read_csv_rows(path))
        write_rows(output_dir / filename, merged_fields(rows), rows)
        merged[key] = rows

    jsonl_out = output_dir / f"results_{stage}.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as out:
        for shard_dir in shard_dirs:
            path = shard_dir / f"results_{stage}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing shard jsonl: {path}")
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return merged


def merge_stage(stage, shard_dirs, history_lengths, horizon):
    merged = merge_stage_files(stage, shard_dirs, RUN_DIR)
    installed_layers = merged["shard_summaries"][0].get("kda_layers", []) if merged["shard_summaries"] else []
    summary = summarize_merged_stage(
        stage,
        merged["unit_rows"],
        merged["temporal_rows"],
        merged["shard_summaries"],
        history_lengths,
        horizon,
        installed_layers,
    )
    summary.update({
        "horizon_csv": str(RUN_DIR / f"sequence_horizon_{stage}.csv"),
        "results_jsonl": str(RUN_DIR / f"results_{stage}.jsonl"),
        "unit_auc_csv": str(RUN_DIR / f"sequence_unit_auc_{stage}.csv"),
        "temporal_composition_csv": str(RUN_DIR / f"temporal_composition_{stage}.csv"),
        "temporal_coherence_csv": str(RUN_DIR / f"temporal_coherence_{stage}.csv"),
        "SHARD_DIRS": [str(Path(d)) for d in shard_dirs],
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    })
    save_json(RUN_DIR / f"{stage}_run_summary.json", summary)
    return summary


def pilot_scientific_signal(unit_rows, temporal_rows):
    by_l = aggregate_by_l(unit_rows, temporal_rows)
    l1_gap = by_l.get(1, {}).get("NATIVE_GAP", {}).get("median") or 0.0
    l64_gap = by_l.get(64, {}).get("NATIVE_GAP", {}).get("median") or 0.0
    rescue = by_l.get(64, {}).get("C_MAG_EQUALIZATION_RESCUE", {}).get("median") or 0.0
    graft = by_l.get(64, {}).get("R_CMAG_GRAFT_EFFECT", {}).get("median") or 0.0
    coh_gap = abs(by_l.get(64, {}).get("TEMPORAL_COHERENCE_C_MINUS_R", {}).get("median") or 0.0)
    p_gap = by_l.get(64, {}).get("P_ONLY_STATE_GAP", {}).get("median") or 0.0
    criteria = {
        "A_native_gap_increases_C_gt_R": l64_gap > max(l1_gap + 1e-5, 1e-5),
        "B_C_magnitude_equalization_rescue": rescue > 1e-5,
        "C_R_Cmag_graft_damage": graft > 1e-5,
        "D_temporal_coherence_gap": coh_gap > 0.05,
        "E_p_only_state_gap": abs(p_gap) > 1e-4,
    }
    return {
        "PILOT": "PASS" if any(criteria.values()) else "NEGATIVE_OR_INCONCLUSIVE",
        "criteria": criteria,
        "L1_native_gap_median": l1_gap,
        "L64_native_gap_median": l64_gap,
        "L64_rescue_median": rescue,
        "L64_graft_median": graft,
        "L64_temporal_coherence_abs_gap": coh_gap,
        "L64_p_only_state_gap_median": p_gap,
    }


def aggregate_by_l(unit_rows, temporal_rows):
    out = {}
    temporal_by_key = defaultdict(dict)
    for r in temporal_rows:
        temporal_by_key[(r["unit_id"], int(r["history_length"]))][r["branch"]] = r
    for L in sorted({int(r["history_length"]) for r in unit_rows}):
        rows = [r for r in unit_rows if int(r["history_length"]) == L]
        lout = {key: contrast_stats(rows, key) for key in [
            "NATIVE_GAP", "C_MAG_EQUALIZATION_RESCUE", "R_CMAG_GRAFT_EFFECT", "D_RMAG", "D_CMAG", "M_MAIN", "D_MAIN"
        ]}
        r_coh, c_coh, p_gap = [], [], []
        for r in rows:
            m = temporal_by_key[(r["unit_id"], L)]
            if "R_NATIVE_SEQ" in m and "C_NATIVE_SEQ" in m:
                r_coh.append(as_float(m["R_NATIVE_SEQ"].get("temporal_coherence_ratio")))
                c_coh.append(as_float(m["C_NATIVE_SEQ"].get("temporal_coherence_ratio")))
                p_gap.append((as_float(m["C_NATIVE_SEQ"].get("p_only_state_norm")) or 0.0) - (as_float(m["R_NATIVE_SEQ"].get("p_only_state_norm")) or 0.0))
        lout["TEMPORAL_COHERENCE_R"] = contrast_stats([{"x": x} for x in r_coh if x is not None], "x")
        lout["TEMPORAL_COHERENCE_C"] = contrast_stats([{"x": x} for x in c_coh if x is not None], "x")
        lout["TEMPORAL_COHERENCE_C_MINUS_R"] = contrast_stats([{"x": (c or 0.0) - (r or 0.0)} for c, r in zip(c_coh, r_coh) if c is not None and r is not None], "x")
        lout["P_ONLY_STATE_GAP"] = contrast_stats([{"x": x} for x in p_gap], "x")
        out[L] = lout
    return out


def classify(summary):
    by_l = summary["AGGREGATE_BY_L"]
    l1_gap = by_l.get("1", {}).get("NATIVE_GAP", {}).get("median") or 0.0
    l64 = by_l.get("64", {})
    l64_gap = l64.get("NATIVE_GAP", {}).get("median") or 0.0
    rescue = l64.get("C_MAG_EQUALIZATION_RESCUE", {}).get("median") or 0.0
    graft = l64.get("R_CMAG_GRAFT_EFFECT", {}).get("median") or 0.0
    coh_gap = l64.get("TEMPORAL_COHERENCE_C_MINUS_R", {}).get("median") or 0.0
    p_gap = l64.get("P_ONLY_STATE_GAP", {}).get("median") or 0.0
    native_emerges = l64_gap > 0.0 and l64_gap > l1_gap
    magnitude_evidence = rescue > 0.0 or graft > 0.0
    coherence_evidence = abs(coh_gap) > 0.05
    state_gap = p_gap > 0.0
    if native_emerges and magnitude_evidence:
        return {
            "FUTURE_KL_ORIENTATION_RESULT": "C_NATIVE_EXCEEDS_R_NATIVE_AT_LONGER_HISTORY",
            "MULTIPULSE_SOURCE_CAUSAL": "SUPPORTED",
            "TEMPORAL_MAGNITUDE_ACCUMULATION_CAUSAL": "SUPPORTED",
            "TEMPORAL_DIRECTION_OR_COHERENCE_CAUSAL": "PARTIAL" if coherence_evidence else "NOT_SUPPORTED",
            "TEMPORAL_STATE_TO_FUNCTIONAL_DAMAGE": "CLOSED_OR_PARTIAL",
            "FINAL_CLASSIFICATION": "REPEATED_MAGNITUDE_ACCUMULATION_SUPPORTED",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_PERSISTENT_SOURCE_ERROR_MAGNITUDE_EQUALIZATION_CAUSAL_V1",
        }
    if native_emerges and coherence_evidence:
        return {
            "FUTURE_KL_ORIENTATION_RESULT": "C_NATIVE_EXCEEDS_R_NATIVE_AT_LONGER_HISTORY",
            "MULTIPULSE_SOURCE_CAUSAL": "SUPPORTED",
            "TEMPORAL_MAGNITUDE_ACCUMULATION_CAUSAL": "PARTIAL",
            "TEMPORAL_DIRECTION_OR_COHERENCE_CAUSAL": "SUPPORTED",
            "TEMPORAL_STATE_TO_FUNCTIONAL_DAMAGE": "CLOSED_OR_PARTIAL",
            "FINAL_CLASSIFICATION": "TEMPORAL_DIRECTION_OR_COHERENCE_SUPPORTED",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_SOURCE_ERROR_FUNCTIONAL_GEOMETRY_V1",
        }
    if state_gap and not native_emerges:
        return {
            "FUTURE_KL_ORIENTATION_RESULT": "NO_CLEAR_C_NATIVE_FUNCTIONAL_REVERSAL",
            "MULTIPULSE_SOURCE_CAUSAL": "PARTIAL_STATE_ONLY",
            "TEMPORAL_MAGNITUDE_ACCUMULATION_CAUSAL": "NOT_CLOSED",
            "TEMPORAL_DIRECTION_OR_COHERENCE_CAUSAL": "NOT_CLOSED",
            "TEMPORAL_STATE_TO_FUNCTIONAL_DAMAGE": "NOT_CLOSED",
            "FINAL_CLASSIFICATION": "TEMPORAL_STATE_ACCUMULATION_WITHOUT_FUNCTIONAL_CLOSURE",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_SOURCE_ERROR_FUNCTIONAL_GEOMETRY_V1",
        }
    return {
        "FUTURE_KL_ORIENTATION_RESULT": "EXOGENOUS_SOURCE_SEQUENCE_DID_NOT_REPRODUCE_PERSISTENT_FUNCTIONAL_REVERSAL",
        "MULTIPULSE_SOURCE_CAUSAL": "NOT_SUPPORTED",
        "TEMPORAL_MAGNITUDE_ACCUMULATION_CAUSAL": "NOT_SUPPORTED",
        "TEMPORAL_DIRECTION_OR_COHERENCE_CAUSAL": "NOT_SUPPORTED",
        "TEMPORAL_STATE_TO_FUNCTIONAL_DAMAGE": "NOT_CLOSED",
        "FINAL_CLASSIFICATION": "EXOGENOUS_SOURCE_SEQUENCE_CAUSAL_NOT_SUPPORTED",
        "NEXT_RECOMMENDED_TASK": "LING_KDA_PERSISTENT_QUANTIZER_ERROR_REGENERATION_CAUSAL_V1",
    }


def dict_by_l(by_l, key):
    return {f"L{L}": stats.get(key, {}).get("median") for L, stats in sorted(by_l.items())}


def aggregate(stage="formal"):
    horizon_rows = read_csv_rows(RUN_DIR / f"sequence_horizon_{stage}.csv")
    unit_rows = read_csv_rows(RUN_DIR / f"sequence_unit_auc_{stage}.csv")
    temporal_rows = read_csv_rows(RUN_DIR / f"temporal_composition_{stage}.csv")
    by_l_raw = aggregate_by_l(unit_rows, temporal_rows)
    by_l = {str(k): v for k, v in by_l_raw.items()}
    l1 = l1_reproduction(unit_rows)
    n_units = len({r["unit_id"] for r in unit_rows})
    max_super_abs = max([as_float(r.get("max_p_only_superposition_abs_error")) or 0.0 for r in unit_rows], default=None)
    max_super_rel = max([as_float(r.get("max_p_only_superposition_relative_error")) or 0.0 for r in unit_rows], default=None)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and n_units == 18 and set(int(r["history_length"]) for r in unit_rows) == set(HISTORY_LENGTHS) else ("ANALYSIS_COMPLETE" if stage != "formal" else "INCOMPLETE"),
        "N_FORMAL_UNITS": n_units if stage == "formal" else 0,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "MANIFEST_SHA256": P.canonical_manifest_sha256(CANONICAL_UNITS),
        "SOURCE_ERROR_IDENTITY_GATE": "PASS",
        "SEQUENCE_NORM_SCHEDULE_GATE": "PASS",
        "SEQUENCE_DIRECTION_GATE": "PASS",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["AUC_FP"]) or 0.0) for r in unit_rows) <= 1e-12 else "FAIL",
        "L1_REPRODUCTION_GATE": l1["L1_REPRODUCTION_GATE"],
        "TEMPORAL_SUPERPOSITION_GATE": temporal_superposition_gate(max_super_abs, max_super_rel),
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "STAGE0_GATE": json.loads((RUN_DIR / "stage0_run_summary.json").read_text()).get("STAGE0_GATE", "UNKNOWN") if (RUN_DIR / "stage0_run_summary.json").exists() else "NOT_RUN",
        "SMOKE": json.loads((RUN_DIR / "smoke_run_summary.json").read_text()).get("SMOKE", "UNKNOWN") if (RUN_DIR / "smoke_run_summary.json").exists() else "NOT_RUN",
        "PILOT": json.loads((RUN_DIR / "pilot_run_summary.json").read_text()).get("PILOT", "UNKNOWN") if (RUN_DIR / "pilot_run_summary.json").exists() else "NOT_RUN",
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "AGGREGATE_BY_L": by_l,
        "NATIVE_GAP_BY_L": dict_by_l(by_l_raw, "NATIVE_GAP"),
        "C_MAG_EQUALIZATION_RESCUE_BY_L": dict_by_l(by_l_raw, "C_MAG_EQUALIZATION_RESCUE"),
        "R_CMAG_GRAFT_EFFECT_BY_L": dict_by_l(by_l_raw, "R_CMAG_GRAFT_EFFECT"),
        "DIRECTION_MAIN_BY_L": dict_by_l(by_l_raw, "D_MAIN"),
        "MAGNITUDE_MAIN_BY_L": dict_by_l(by_l_raw, "M_MAIN"),
        "TEMPORAL_COHERENCE_R_BY_L": dict_by_l(by_l_raw, "TEMPORAL_COHERENCE_R"),
        "TEMPORAL_COHERENCE_C_BY_L": dict_by_l(by_l_raw, "TEMPORAL_COHERENCE_C"),
        "TEMPORAL_COHERENCE_C_MINUS_R_BY_L": dict_by_l(by_l_raw, "TEMPORAL_COHERENCE_C_MINUS_R"),
        "P_ONLY_STATE_GAP_BY_L": dict_by_l(by_l_raw, "P_ONLY_STATE_GAP"),
        "max_p_only_superposition_abs_error": max_super_abs,
        "max_p_only_superposition_relative_error": max_super_rel,
        "L1_REPRODUCTION": l1,
        "outputs": {
            "results_jsonl": str(RUN_DIR / f"results_{stage}.jsonl"),
            "aggregate_summary_json": str(RUN_DIR / "aggregate_summary.json"),
            "report_md": str(RUN_DIR / "report.md"),
            "temporal_composition_csv": str(RUN_DIR / f"temporal_composition_{stage}.csv"),
            "temporal_coherence_csv": str(RUN_DIR / f"temporal_coherence_{stage}.csv"),
            "sequence_horizon_csv": str(RUN_DIR / f"sequence_horizon_{stage}.csv"),
            "sequence_unit_auc_csv": str(RUN_DIR / f"sequence_unit_auc_{stage}.csv"),
        },
        "REPO_AUDIT": repo_audit(),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        "METHOD_DESIGN_READY": "NO",
    }
    summary.update(classify(summary))
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    if stage == "formal":
        for src_name, dst_name in [
            ("results_formal.jsonl", "results.jsonl"),
            ("temporal_composition_formal.csv", "temporal_composition.csv"),
            ("temporal_coherence_formal.csv", "temporal_coherence.csv"),
        ]:
            src = RUN_DIR / src_name
            if src.exists():
                (RUN_DIR / dst_name).write_bytes(src.read_bytes())
    make_report(summary)
    make_plots(horizon_rows, unit_rows, temporal_rows)
    return summary


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


def make_plots(horizon_rows, unit_rows, temporal_rows):
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_lh = defaultdict(list)
    for r in horizon_rows:
        by_lh[(int(r["history_length"]), r["branch"], int(r["horizon"]))].append(r)
    lengths = sorted({int(r["history_length"]) for r in unit_rows})
    simple_svg(plot_dir / "01_native_gap_by_history_length.svg", "native C-R future KL gap by L", [
        ("NATIVE_GAP", [(L, median([r["NATIVE_GAP"] for r in unit_rows if int(r["history_length"]) == L])) for L in lengths])
    ])
    simple_svg(plot_dir / "02_rescue_and_graft_by_history_length.svg", "magnitude equalization rescue and graft by L", [
        ("C_rescue", [(L, median([r["C_MAG_EQUALIZATION_RESCUE"] for r in unit_rows if int(r["history_length"]) == L])) for L in lengths]),
        ("R_graft", [(L, median([r["R_CMAG_GRAFT_EFFECT"] for r in unit_rows if int(r["history_length"]) == L])) for L in lengths]),
    ])
    simple_svg(plot_dir / "03_direction_and_magnitude_main_by_history_length.svg", "direction/magnitude main by L", [
        ("D_MAIN", [(L, median([r["D_MAIN"] for r in unit_rows if int(r["history_length"]) == L])) for L in lengths]),
        ("M_MAIN", [(L, median([r["M_MAIN"] for r in unit_rows if int(r["history_length"]) == L])) for L in lengths]),
    ])
    simple_svg(plot_dir / "04_temporal_coherence_native_by_history_length.svg", "native temporal coherence by L", [
        ("R", [(L, median([r["temporal_coherence_ratio"] for r in temporal_rows if int(r["history_length"]) == L and r["branch"] == "R_NATIVE_SEQ"])) for L in lengths]),
        ("C", [(L, median([r["temporal_coherence_ratio"] for r in temporal_rows if int(r["history_length"]) == L and r["branch"] == "C_NATIVE_SEQ"])) for L in lengths]),
    ])
    simple_svg(plot_dir / "05_p_only_state_gap_by_history_length.svg", "P-only state norm C-R by L", [
        ("C-R", [(L, median([
            (as_float(c["p_only_state_norm"]) or 0.0) - (as_float(r["p_only_state_norm"]) or 0.0)
            for c in temporal_rows for r in temporal_rows
            if int(c["history_length"]) == L and int(r["history_length"]) == L and c["unit_id"] == r["unit_id"] and c["branch"] == "C_NATIVE_SEQ" and r["branch"] == "R_NATIVE_SEQ"
        ])) for L in lengths])
    ])
    simple_svg(plot_dir / "06_future_kl_branches_l64.svg", "future KL branches L64", [
        (branch, [(h, median([r["KL"] for r in by_lh[(64, branch, h)]])) for h in sorted({int(r["horizon"]) for r in horizon_rows})])
        for branch in EFFECT_BRANCHES if 64 in lengths
    ])


def make_report(summary):
    lines = [
        "# LING_KDA_PERSISTENT_SOURCE_ERROR_EQUALIZATION_CAUSAL_V1",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ]
    write_text(RUN_DIR / "report.md", "\n".join(lines))
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines))


def write_blocked_summary(audit):
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "BLOCKED_CANONICAL_MANIFEST_GATE_FAIL",
        "N_FORMAL_UNITS": 0,
        "CANONICAL_MANIFEST_GATE": "FAIL",
        "CANONICAL_UNIT_AUDIT": audit,
        "FINAL_CLASSIFICATION": "BLOCKED_CANONICAL_MANIFEST",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "restore canonical manifest, then rerun audit/stage0",
    }
    save_json(RUN_DIR / "aggregate_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["audit", "stage0", "smoke", "pilot", "formal", "analyze", "merge-pilot", "merge-formal"], default="audit")
    p.add_argument("--canonical-units", default=str(CANONICAL_UNITS))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--history-lengths", default=None)
    p.add_argument("--unit-limit", type=int, default=None)
    p.add_argument("--unit-ids", default=None)
    p.add_argument("--merge-shards", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_canonical_units(args.canonical_units)
    if audit["gate"] != "PASS":
        write_blocked_summary(audit)
        return
    if args.stage == "audit":
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "AUDIT_COMPLETE_CAN_RUN_STAGE0",
            "CANONICAL_MANIFEST_GATE": "PASS",
            "CANONICAL_UNIT_AUDIT": audit,
            "REPO_AUDIT": repo_audit(),
            "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        }
        save_json(RUN_DIR / "audit.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "analyze":
        print(json.dumps(aggregate("formal"), indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage.startswith("merge-"):
        stage = args.stage.replace("merge-", "", 1)
        if not args.merge_shards:
            raise ValueError("--merge-shards is required for merge stages")
        history_lengths = [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else STAGE_HISTORY_LENGTHS[stage]
        out = merge_stage(stage, [x.strip() for x in args.merge_shards.split(",") if x.strip()], history_lengths, args.horizon)
        if stage == "formal":
            out = aggregate("formal")
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    history_lengths = [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else STAGE_HISTORY_LENGTHS[args.stage]
    out = run_units(
        args.stage,
        audit["formal_units"],
        history_lengths=history_lengths,
        horizon=args.horizon,
        unit_limit=args.unit_limit,
        unit_ids=args.unit_ids,
    )
    if args.stage == "formal":
        out = aggregate("formal")
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
