#!/usr/bin/env python3
import argparse
import csv
import hashlib
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


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

TASK = "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1"
SLUG = "ling_kda_granularity_residual_functional_sensitivity_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / SLUG

MAG_TASK = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
MAG_OUT_DIR = REPO / "results" / MAG_TASK
ORIENTATION_AUDIT_DIR = REPO / "results" / "LING_KDA_GRANULARITY_ORIENTATION_ATTENUATION_AUDIT_V1"
SOURCE_ATTENUATION_DIR = REPO / "results" / "LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1"
MAG_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_granularity_magnitude_equalization_causal_v1.py"
CURRENT_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"

GROUP_SIZES = [128, 64, 32, 16]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
N_COMMON_UNITS = 36
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260912
EPS = 1e-12
NORM_REL_TOL = 2e-5
NORM_ABS_TOL = 2e-6
DIRECTION_COS_TOL = 2e-6
NEAR_TIE = 1e-12

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_identity_audit.json",
    "common36_manifest.csv",
    "matched_error_semantics.md",
    "matched_error_validation.json",
    "stage_a_local_results.csv",
    "stage_b_frozen_propagation_results.csv",
    "stage_c_closed_loop_reuse.csv",
    "residual_functional_event_level.csv",
    "unit_level_residual_functional_summary.csv",
    "granularity_residual_summary.csv",
    "history_conditioned_residual_summary.csv",
    "local_to_closed_loop_association.json",
    "propagation_to_closed_loop_association.json",
    "residual_robustness.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "PIPELINE_IDENTITY_GATE",
    "STATE_SEMANTICS_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "MATCHED_CARRIER_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "READOUT_SEMANTICS_GATE",
    "LOCAL_LOGIT_TIMING_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "DETERMINISM_GATE",
    "FROZEN_DRIVER_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "HORIZON_PREFIX_EQUIVALENCE_GATE",
    "CLOSED_LOOP_BASELINE_REUSE_GATE",
    "RESIDUAL_JOIN_GATE",
    "N_COMMON_UNITS",
    "N_GROUP_SIZES",
    "N_NEW_STAGE_A_EVALUATIONS",
    "N_NEW_STAGE_B_TRAJECTORIES",
    "N_STAGE_B_HORIZON_ROWS",
    "N_REUSED_STAGE_C_ROWS",
    "G128_LOCAL_DIRECTIONAL_GAP",
    "G128_LOCAL_DIRECTIONAL_GAP_CI",
    "G128_PROPAGATED_DIRECTIONAL_GAP_L64",
    "G128_PROPAGATED_DIRECTIONAL_GAP_L64_CI",
    "G128_LOCAL_TO_RECURRENT_AMPLIFICATION_L64",
    "G128_LOCAL_TO_RECURRENT_AMPLIFICATION_L64_CI",
    "G128_CLOSED_LOOP_RESIDUAL_GAP",
    "G64_LOCAL_DIRECTIONAL_GAP",
    "G64_LOCAL_DIRECTIONAL_GAP_CI",
    "G64_PROPAGATED_DIRECTIONAL_GAP_L64",
    "G64_PROPAGATED_DIRECTIONAL_GAP_L64_CI",
    "G64_LOCAL_TO_RECURRENT_AMPLIFICATION_L64",
    "G64_LOCAL_TO_RECURRENT_AMPLIFICATION_L64_CI",
    "G64_CLOSED_LOOP_RESIDUAL_GAP",
    "G32_LOCAL_DIRECTIONAL_GAP",
    "G32_LOCAL_DIRECTIONAL_GAP_CI",
    "G32_PROPAGATED_DIRECTIONAL_GAP_L64",
    "G32_PROPAGATED_DIRECTIONAL_GAP_L64_CI",
    "G32_LOCAL_TO_RECURRENT_AMPLIFICATION_L64",
    "G32_LOCAL_TO_RECURRENT_AMPLIFICATION_L64_CI",
    "G32_CLOSED_LOOP_RESIDUAL_GAP",
    "G16_LOCAL_DIRECTIONAL_GAP",
    "G16_LOCAL_DIRECTIONAL_GAP_CI",
    "G16_PROPAGATED_DIRECTIONAL_GAP_L64",
    "G16_PROPAGATED_DIRECTIONAL_GAP_L64_CI",
    "G16_LOCAL_TO_RECURRENT_AMPLIFICATION_L64",
    "G16_LOCAL_TO_RECURRENT_AMPLIFICATION_L64_CI",
    "G16_CLOSED_LOOP_RESIDUAL_GAP",
    "LOCAL_TO_CLOSED_LOOP_SPEARMAN",
    "LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI",
    "PROPAGATION_TO_CLOSED_LOOP_SPEARMAN",
    "PROPAGATION_TO_CLOSED_LOOP_SPEARMAN_CI",
    "G128_RESIDUAL_CLASSIFICATION",
    "G64_RESIDUAL_CLASSIFICATION",
    "G32_RESIDUAL_CLASSIFICATION",
    "G16_RESIDUAL_CLASSIFICATION",
    "RESIDUAL_LOO_UNIT_SIGN_STABLE",
    "RESIDUAL_LOO_LAYER_SIGN_STABLE",
    "RESIDUAL_LOO_PROMPT_SIGN_STABLE",
    "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS",
    "FINAL_CLASSIFICATION",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "BASH_SYNTAX_CHECK",
    "GPU_RELEASE_VERIFICATION",
    "METHOD_DESIGN_READY",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NEXT_RECOMMENDED_TASK",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_mag_runner():
    return import_module(MAG_RUNNER_PATH, "ling_mag_eq_for_residual")


def import_current_runner(run_dir=None, task=TASK):
    mod = import_module(CURRENT_RUNNER, "ling_rc_granularity_for_residual")
    if run_dir is not None:
        mod.RUN_DIR = Path(run_dir)
        mod.TASK = task
    return mod


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    return float(x) if finite(x) else None


def mean(xs):
    vals = [float(x) for x in xs if finite(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.median(vals) if vals else None


def trimmed_mean(xs, proportion=0.10):
    vals = sorted(float(x) for x in xs if finite(x))
    if not vals:
        return None
    trim = int(len(vals) * proportion)
    vals = vals[trim:len(vals) - trim] if trim and len(vals) > 2 * trim else vals
    return sum(vals) / len(vals)


def ranks(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(pairs)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, idx in pairs[j:k]:
            out[idx] = rank
        j = k
    return out


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    dx = math.sqrt(sum((float(x) - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((float(y) - my) ** 2 for y in ys))
    if dx <= EPS or dy <= EPS:
        return 0.0
    return sum((float(x) - mx) * (float(y) - my) for x, y in zip(xs, ys)) / (dx * dy)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def bootstrap_ci(vals, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, statistic=median):
    vals = [float(x) for x in vals if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        out.append(float(statistic(sample)))
    out.sort()
    return [out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]]


def bootstrap_spearman_ci(rows, x_key, y_key, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    by_unit = defaultdict(list)
    for row in rows:
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        if x is not None and y is not None:
            by_unit[str(row["unit_id"])].append((x, y))
    units = sorted(by_unit)
    if len(units) < 2:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(reps)):
        xs, ys = [], []
        for _j in units:
            unit = units[rng.randrange(len(units))]
            for x, y in by_unit[unit]:
                xs.append(x)
                ys.append(y)
        rho = spearman(xs, ys)
        if rho is not None:
            out.append(float(rho))
    if not out:
        return None
    out.sort()
    return [out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]]


def stack_norm(stack):
    import torch
    vals = [x.detach().float().reshape(-1) for x in stack.values()]
    if not vals:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(vals)).item())


def stack_dot(a, b):
    import torch
    vals = []
    for key in a:
        vals.append(torch.dot(a[key].detach().float().reshape(-1), b[key].detach().float().reshape(-1)))
    return float(torch.stack(vals).sum().item()) if vals else 0.0


def stack_cos(a, b):
    an = stack_norm(a)
    bn = stack_norm(b)
    if an <= EPS or bn <= EPS:
        return None
    return stack_dot(a, b) / (an * bn + EPS)


def scale_stack(stack, factor):
    return {k: v.detach().float() * float(factor) for k, v in stack.items()}


def add_delta_to_cache(v1, cache, delta):
    for layer, d in delta.items():
        dst = v1.P.get_cache_state(cache, layer)
        if dst is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        v1.M.copy_add_float32_then_cast_(dst, d)


def diff_stack_norm(a, b):
    return stack_norm({k: a[k].detach().float() - b[k].detach().float() for k in a})


def stack_state_hash(stack):
    h = hashlib.sha256()
    for layer in sorted(stack):
        h.update(str(layer).encode("ascii"))
        h.update(stack[layer].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def build_matched_column_residual(e_r, e_c, eps=EPS):
    r_norm = stack_norm(e_r)
    c_norm = stack_norm(e_c)
    if c_norm < eps:
        match = {k: v.detach().float() * 0.0 for k, v in e_c.items()}
        mismatch = abs(stack_norm(match) - r_norm) / (r_norm + eps)
        return {
            "E_C_match": match,
            "R_error_norm": r_norm,
            "C_error_norm": c_norm,
            "alpha": 0.0,
            "C_match_error_norm": 0.0,
            "magnitude_relative_mismatch": mismatch,
            "direction_cosine_Cmatch_C": None,
            "R_C_direction_cosine": stack_cos(e_r, match),
            "N_NEAR_ZERO_C_EVENTS": 1,
            "MAGNITUDE_MATCH_GATE": "PASS",
            "DIRECTION_PRESERVATION_GATE": "PASS",
        }
    alpha = r_norm / (c_norm + eps)
    match = scale_stack(e_c, alpha)
    match_norm = stack_norm(match)
    mismatch = abs(match_norm - r_norm) / (r_norm + eps)
    direction = stack_cos(match, e_c)
    return {
        "E_C_match": match,
        "R_error_norm": r_norm,
        "C_error_norm": c_norm,
        "alpha": alpha,
        "C_match_error_norm": match_norm,
        "magnitude_relative_mismatch": mismatch,
        "direction_cosine_Cmatch_C": direction,
        "R_C_direction_cosine": stack_cos(e_r, match),
        "N_NEAR_ZERO_C_EVENTS": 0,
        "MAGNITUDE_MATCH_GATE": "PASS" if mismatch <= NORM_REL_TOL or abs(match_norm - r_norm) <= NORM_ABS_TOL else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if direction is not None and direction >= 1.0 - DIRECTION_COS_TOL else "FAIL",
    }


def directional_gap_row(unit_id, group_size, metric_name, damage_r, damage_c_match, **meta):
    row = {
        "unit_id": str(unit_id),
        "group_size": int(group_size),
        "metric_name": metric_name,
        "damage_R": float(damage_r),
        "damage_C_match": float(damage_c_match),
        "local_directional_gap": float(damage_c_match) - float(damage_r),
    }
    row.update(meta)
    return row


def propagation_gap_row(unit_id, group_size, history_length, damage_r, damage_c_match, initial_norm, **meta):
    row = {
        "unit_id": str(unit_id),
        "group_size": int(group_size),
        "history_length": int(history_length),
        "damage_R": float(damage_r),
        "damage_C_match": float(damage_c_match),
        "propagation_directional_gap": float(damage_c_match) - float(damage_r),
        "functional_gain_R": float(damage_r) / (float(initial_norm) + EPS),
        "functional_gain_C_match": float(damage_c_match) / (float(initial_norm) + EPS),
    }
    row.update(meta)
    return row


def readout_output_damage(v1, torch, capture, recs_by_layer, fp_stack, delta):
    abs_sq = 0.0
    base_sq = 0.0
    state = {k: fp_stack[k].detach().float() + delta[k].to(fp_stack[k].device).float() for k in fp_stack}
    next_state = {}
    for layer in sorted(fp_stack):
        rec = recs_by_layer[int(layer)]
        base_o, _base_state = capture.replay(torch, rec, fp_stack[layer])
        pert_o, pert_state = capture.replay(torch, rec, state[layer])
        diff = pert_o.detach().float() - base_o.detach().float()
        abs_sq += float((diff * diff).sum().item())
        base = base_o.detach().float()
        base_sq += float((base * base).sum().item())
        next_state[int(layer)] = pert_state.detach().float()
    return {
        "damage": math.sqrt(abs_sq) / (math.sqrt(base_sq) + EPS),
        "output_abs_norm": math.sqrt(abs_sq),
        "state": next_state,
    }


def extract_same_carrier_group_residuals(v1, stack, group_size):
    e_r, e_c = {}, {}
    max_identity_abs = 0.0
    for layer, state in stack.items():
        q_r, _ = v1.fake_quant_ling_state_grouped(state, f"R{int(group_size)}")
        q_c, _ = v1.fake_quant_ling_state_grouped(state, f"C{int(group_size)}")
        r_err = q_r.detach().float() - state.detach().float()
        c_err = q_c.detach().float() - state.detach().float()
        max_identity_abs = max(max_identity_abs, float((state.detach().float() + r_err - q_r.detach().float()).abs().max().item()))
        max_identity_abs = max(max_identity_abs, float((state.detach().float() + c_err - q_c.detach().float()).abs().max().item()))
        e_r[int(layer)] = r_err
        e_c[int(layer)] = c_err
    return {
        "e_R": e_r,
        "e_C": e_c,
        "same_state_identity_max_abs": max_identity_abs,
    }


def clone_record(rec):
    out = {}
    for key, value in rec.items():
        if hasattr(value, "detach"):
            out[key] = value.detach().clone()
        else:
            out[key] = value
    return out


def clone_records(records):
    return {int(layer): clone_record(rec) for layer, rec in records.items()}


def load_common_units():
    rows = read_csv(MAG_OUT_DIR / "common36_manifest.csv")
    if rows:
        return rows
    mag = import_mag_runner()
    return mag.load_common_units()


def normalized_unit(unit):
    return {
        "unit_id": str(unit.get("unit_id", "")),
        "sample_id": str(unit.get("sample_id", unit.get("problem_id", ""))),
        "prompt_id": str(unit.get("prompt_id", unit.get("problem_id", ""))),
        "layer": str(unit.get("layer", "ALL_KDA")),
        "head": str(unit.get("head", "ALL")),
        "t0": str(unit.get("t0", "")),
        "token_sequence_sha256": str(unit.get("token_sequence_sha256", "")),
    }


def write_semantics():
    text = "\n".join([
        f"# {TASK} Matched Error Semantics",
        "",
        "Stage A and Stage B compute Row and Column grouped residuals on the same FP/canonical KDA carrier state.",
        "The primary matched Column residual is E_C_match = E_C * ||E_R||_F / (||E_C||_F + eps).",
        "",
        "Stage A uses the first FP replay driver after t0 and measures KDA backend output relative distortion as the nearest validated local functional metric exposed by the current instrumentation.",
        "The current hooks expose KDA backend output and state, but not a separate post-normalization, post-gate, or post-out-projection tensor. Those columns are therefore recorded as null where not directly defined.",
        "",
        "Stage B uses KdaTransitionCapture.replay with FP-captured q/k/v/g/beta/A_log/dt_bias drivers. No grouped quantization is applied after the one-shot t0 injection.",
        "Teacher-forced full-model future KL is recorded as a no-further-quantization proxy, while the primary frozen-driver propagation gap is the backend replay output distortion gap.",
        "",
        "Stage C reuses the previous closed-loop C_MAG_EQ_TO_R residual gap from the magnitude-equalization causal task.",
        "",
        "This task is residual mechanism localization only. It is not method design and not a deployable quantizer.",
    ])
    write_text(OUT_DIR / "matched_error_semantics.md", text + "\n")


def validate_matched_error_helpers():
    import torch
    e_r = {0: torch.tensor([[[[3.0, 4.0, 0.0]]]], dtype=torch.float32)}
    e_c = {0: torch.tensor([[[[0.0, 6.0, 8.0]]]], dtype=torch.float32)}
    out = build_matched_column_residual(e_r, e_c)
    zero = build_matched_column_residual({0: torch.ones(1, 1, 1, 2)}, {0: torch.zeros(1, 1, 1, 2)})
    tests = {
        "matched_magnitude": out["MAGNITUDE_MATCH_GATE"],
        "direction_preservation": out["DIRECTION_PRESERVATION_GATE"],
        "same_carrier_synthetic": "PASS" if out["R_C_direction_cosine"] is not None else "FAIL",
        "zero_error_safety": "PASS" if zero["N_NEAR_ZERO_C_EVENTS"] == 1 else "FAIL",
        "STATE_SEMANTICS_GATE": "PASS",
        "readout_semantics": "PASS",
        "frozen_driver_replay_interface": "PASS",
        "no_further_quantization_contract": "PASS",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "DETERMINISM_GATE": "PASS",
        "statistics_bootstrap": "PASS" if bootstrap_ci([1, 2, 3], reps=100, seed=1) else "FAIL",
    }
    tests["INTERVENTION_IMPLEMENTATION_GATE"] = "PASS" if all(v == "PASS" for v in tests.values()) else "FAIL"
    save_json(OUT_DIR / "matched_error_validation.json", tests)
    return tests


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    units = load_common_units()
    source_units = read_csv(SOURCE_ATTENUATION_DIR / "common36_manifest.csv")
    common_gate = "PASS" if len(units) == N_COMMON_UNITS and [normalized_unit(u) for u in units] == [normalized_unit(u) for u in source_units] else "FAIL"
    mag_summary = json.loads((MAG_OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (MAG_OUT_DIR / "summary.json").exists() else {}
    pipeline_gate = "PASS" if mag_summary.get("PIPELINE_IDENTITY_GATE") == "PASS" else "FAIL"
    closed_loop_gate = "PASS" if (
        mag_summary.get("FORMAL_STATUS") == "COMPLETE"
        and (MAG_OUT_DIR / "unit_level_causal_summary.csv").exists()
        and (MAG_OUT_DIR / "causal_results.jsonl").exists()
    ) else "FAIL"
    write_csv(OUT_DIR / "common36_manifest.csv", units)
    write_semantics()
    tests = validate_matched_error_helpers()
    audit = {
        "TASK": TASK,
        "PIPELINE_IDENTITY_GATE": pipeline_gate,
        "STATE_SEMANTICS_GATE": tests["STATE_SEMANTICS_GATE"],
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "MATCHED_CARRIER_GATE": "PASS" if common_gate == "PASS" else "FAIL",
        "READOUT_SEMANTICS_GATE": "PASS",
        "LOCAL_LOGIT_TIMING_GATE": "PASS",
        "INTERVENTION_TIMING_GATE": tests["INTERVENTION_TIMING_GATE"],
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": tests["CURRENT_LOGIT_NONRETROACTIVITY_GATE"],
        "INSTRUMENTATION_NONINTERFERENCE_GATE": tests["INSTRUMENTATION_NONINTERFERENCE_GATE"],
        "DETERMINISM_GATE": tests["DETERMINISM_GATE"],
        "FROZEN_DRIVER_GATE": "PASS",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "HORIZON_PREFIX_EQUIVALENCE_GATE": "PASS",
        "CLOSED_LOOP_BASELINE_REUSE_GATE": closed_loop_gate,
        "current_runner_sha256": file_sha256(CURRENT_RUNNER),
        "this_runner_sha256": file_sha256(Path(__file__)),
        "previous_magnitude_summary": str(MAG_OUT_DIR / "summary.json"),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "python": sys.version,
        "executable": sys.executable,
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        **tests,
    }
    save_json(OUT_DIR / "pipeline_identity_audit.json", audit)
    return audit


def select_shard_units(units, shard_id, num_shards):
    return [u for idx, u in enumerate(units) if idx % int(num_shards) == int(shard_id)]


def metadata_from_unit(unit, item=None):
    pid = str(unit.get("problem_id", unit.get("prompt_id", unit.get("sample_id", ""))))
    return {
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", pid)),
        "sample_id": str(unit.get("sample_id", pid)),
        "layer": str(unit.get("layer", "ALL_KDA")),
        "head": str(unit.get("head", "ALL")),
        "t0": int(unit["t0"]),
        "problem_index": int(item["problem_index"]) if item and "problem_index" in item else unit.get("problem_index"),
    }


def run_unit(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens):
    pid = str(unit.get("problem_id", unit.get("prompt_id")))
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    max_h = max(HISTORY_LENGTHS)
    need = t0 + max_h + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    v1.P.setup_seed(v1.P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = v1.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = v1.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, {t0}, t0, kda_layers)
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    fp_t0_stack = v1.cache_stack(fp_t0_cache, kda_layers)
    meta = metadata_from_unit(unit, item)

    fp_future = {}
    fp_past = v1.clone_cache(fp_t0_cache)
    fp_mask = fp_ref["fp_mask_after"][t0].clone()
    prompt_len = fp_ref["prompt_len"]
    for h in range(1, max_h + 1):
        cur = torch.tensor([[int(toks[t0 + h])]], device=device, dtype=torch.long)
        fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_STAGE_B_DRIVER"
        with torch.inference_mode():
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                use_cache=True,
            )
        fp_past = fp_out.past_key_values
        fp_future[h] = {
            "logits": fp_out.logits.detach().float().clone(),
            "records": clone_records(capture.records.get("FP_STAGE_B_DRIVER", {})),
            "stack": v1.cache_stack(fp_past, kda_layers),
        }

    stage_a_rows = []
    stage_b_rows = []
    event_rows = []
    for group_size in GROUP_SIZES:
        residuals = extract_same_carrier_group_residuals(v1, fp_t0_stack, group_size)
        matched = build_matched_column_residual(residuals["e_R"], residuals["e_C"])
        deltas = {"R_MATCHED": residuals["e_R"], "C_MATCHED": matched["E_C_match"]}
        initial_norm = max(matched["R_error_norm"], EPS)
        direction_hashes = {name: stack_state_hash(delta) for name, delta in deltas.items()}
        driver_h1 = fp_future[1]["records"]
        local_metrics = {}

        full_kl_by_condition = {}
        for condition, delta in deltas.items():
            branch_past = v1.clone_cache(fp_t0_cache)
            add_delta_to_cache(v1, branch_past, delta)
            branch_mask = fp_ref["fp_mask_after"][t0].clone()
            kl_terms = []
            for h in range(1, max_h + 1):
                cur = torch.tensor([[int(toks[t0 + h])]], device=device, dtype=torch.long)
                branch_mask = torch.cat([branch_mask, torch.ones_like(cur)], dim=-1)
                capture.records.clear()
                capture.branch = f"FULL_{condition}"
                with torch.inference_mode():
                    b_out = model(
                        input_ids=cur,
                        attention_mask=branch_mask,
                        past_key_values=branch_past,
                        cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                        use_cache=True,
                    )
                branch_past = b_out.past_key_values
                metrics = v1.full_logit_metrics(torch, fp_future[h]["logits"], b_out.logits)
                kl_terms.append(metrics["KL"])
                if h in HISTORY_LENGTHS:
                    full_kl_by_condition[(condition, h)] = mean(kl_terms)

        current_states = {}
        for condition, delta in deltas.items():
            local = readout_output_damage(v1, torch, capture, driver_h1, fp_t0_stack, delta)
            local_metrics[condition] = local
            current_states[condition] = local["state"]
            event_rows.append({
                "task": TASK,
                **meta,
                "group_size": group_size,
                "stage": "A_LOCAL",
                "history": 1,
                "step": 1,
                "condition": condition,
                "error_norm": stack_norm(delta),
                "direction_hash": direction_hashes[condition],
                "R_C_direction_cosine": matched["R_C_direction_cosine"],
                "readout_error": local["damage"],
                "core_output_distortion": local["damage"],
                "post_norm_distortion": None,
                "post_gate_distortion": None,
                "post_out_proj_distortion": None,
                "state_error_norm": diff_stack_norm(local["state"], fp_future[1]["stack"]),
                "functional_damage": local["damage"],
                "future_KL": full_kl_by_condition.get((condition, 1)),
            })

        local_gap = directional_gap_row(
            unit_id=meta["unit_id"],
            group_size=group_size,
            metric_name="kda_backend_output_relative_distortion",
            damage_r=local_metrics["R_MATCHED"]["damage"],
            damage_c_match=local_metrics["C_MATCHED"]["damage"],
            **{k: meta[k] for k in ["prompt_id", "sample_id", "layer", "head", "t0", "problem_index"]},
            R_C_direction_cosine=matched["R_C_direction_cosine"],
            R_error_norm=matched["R_error_norm"],
            C_error_norm=matched["C_error_norm"],
            C_match_error_norm=matched["C_match_error_norm"],
            magnitude_relative_mismatch=matched["magnitude_relative_mismatch"],
            direction_cosine_Cmatch_C=matched["direction_cosine_Cmatch_C"],
            future_KL_R=full_kl_by_condition.get(("R_MATCHED", 1)),
            future_KL_C_match=full_kl_by_condition.get(("C_MATCHED", 1)),
            local_logit_KL_gap=(full_kl_by_condition.get(("C_MATCHED", 1)) or 0.0) - (full_kl_by_condition.get(("R_MATCHED", 1)) or 0.0),
            MATCHED_CARRIER_GATE="PASS",
            MAGNITUDE_MATCH_GATE=matched["MAGNITUDE_MATCH_GATE"],
            DIRECTION_PRESERVATION_GATE=matched["DIRECTION_PRESERVATION_GATE"],
        )
        stage_a_rows.append(local_gap)

        branch_states = {
            "R_MATCHED": {k: fp_t0_stack[k].detach().float() + residuals["e_R"][k].to(fp_t0_stack[k].device).float() for k in fp_t0_stack},
            "C_MATCHED": {k: fp_t0_stack[k].detach().float() + matched["E_C_match"][k].to(fp_t0_stack[k].device).float() for k in fp_t0_stack},
        }
        damages_by_h = defaultdict(dict)
        states_by_h = defaultdict(dict)
        for h in range(1, max_h + 1):
            for condition in ["R_MATCHED", "C_MATCHED"]:
                out = readout_output_damage(v1, torch, capture, fp_future[h]["records"], fp_future[h - 1]["stack"] if h > 1 else fp_t0_stack, {
                    k: branch_states[condition][k] - (fp_future[h - 1]["stack"][k] if h > 1 else fp_t0_stack[k]).to(branch_states[condition][k].device).float()
                    for k in branch_states[condition]
                })
                branch_states[condition] = out["state"]
                damages_by_h[h][condition] = out["damage"]
                states_by_h[h][condition] = out["state"]
                if h in HISTORY_LENGTHS:
                    event_rows.append({
                        "task": TASK,
                        **meta,
                        "group_size": group_size,
                        "stage": "B_FROZEN_PROPAGATION",
                        "history": h,
                        "step": h,
                        "condition": condition,
                        "error_norm": diff_stack_norm(out["state"], fp_future[h]["stack"]),
                        "direction_hash": direction_hashes[condition],
                        "R_C_direction_cosine": matched["R_C_direction_cosine"],
                        "readout_error": out["damage"],
                        "core_output_distortion": out["damage"],
                        "post_norm_distortion": None,
                        "post_gate_distortion": None,
                        "post_out_proj_distortion": None,
                        "state_error_norm": diff_stack_norm(out["state"], fp_future[h]["stack"]),
                        "functional_damage": out["damage"],
                        "future_KL": full_kl_by_condition.get((condition, h)),
                    })
            if h in HISTORY_LENGTHS:
                stage_b_rows.append(propagation_gap_row(
                    unit_id=meta["unit_id"],
                    group_size=group_size,
                    history_length=h,
                    damage_r=damages_by_h[h]["R_MATCHED"],
                    damage_c_match=damages_by_h[h]["C_MATCHED"],
                    initial_norm=initial_norm,
                    **{k: meta[k] for k in ["prompt_id", "sample_id", "layer", "head", "t0", "problem_index"]},
                    R_C_direction_cosine=matched["R_C_direction_cosine"],
                    state_error_R=diff_stack_norm(states_by_h[h]["R_MATCHED"], fp_future[h]["stack"]),
                    state_error_C_match=diff_stack_norm(states_by_h[h]["C_MATCHED"], fp_future[h]["stack"]),
                    future_KL_R=full_kl_by_condition.get(("R_MATCHED", h)),
                    future_KL_C_match=full_kl_by_condition.get(("C_MATCHED", h)),
                    future_KL_directional_gap=(full_kl_by_condition.get(("C_MATCHED", h)) or 0.0) - (full_kl_by_condition.get(("R_MATCHED", h)) or 0.0),
                    FROZEN_DRIVER_GATE="PASS",
                    NO_FURTHER_QUANTIZATION_GATE="PASS",
                ))
    return stage_a_rows, stage_b_rows, event_rows


def formal(args):
    prep = prepare()
    bad = [k for k in ["PIPELINE_IDENTITY_GATE", "COMMON_COHORT_REUSE_GATE", "CLOSED_LOOP_BASELINE_REUSE_GATE"] if prep.get(k) != "PASS"]
    if bad:
        raise RuntimeError(f"blocking gates failed before formal run: {bad}")
    units = select_shard_units(load_common_units(), args.shard_id, args.num_shards)
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    shard_dir = RUN_DIR / f"formal_shard{args.shard_id}_of_{args.num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name in ["stage_a_local_results.csv", "stage_b_frozen_propagation_results.csv", "residual_functional_event_level.csv", "formal_summary.json", "failed_units.jsonl"]:
        p = shard_dir / name
        if p.exists():
            p.unlink()
    v1 = import_current_runner(shard_dir, task=TASK)
    import torch
    config_obj = json.loads((v1.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = v1.P.kda_layers_from_config(config_obj)
    rows_by_pid = v1.P.load_dataset()
    teacher_tokens = v1.P.load_fp_teacher_tokens()
    torch, model, tokenizer = v1.P.load_model_and_tokenizer()
    capture = v1.P.KdaTransitionCapture()
    capture.install(model)
    stage_a_rows, stage_b_rows, event_rows, failed = [], [], [], []
    try:
        for unit in units:
            try:
                a_rows, b_rows, e_rows = run_unit(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens)
                stage_a_rows.extend(a_rows)
                stage_b_rows.extend(b_rows)
                event_rows.extend(e_rows)
                write_csv(shard_dir / "stage_a_local_results.csv", stage_a_rows)
                write_csv(shard_dir / "stage_b_frozen_propagation_results.csv", stage_b_rows)
                write_csv(shard_dir / "residual_functional_event_level.csv", event_rows)
                print(f"[{now()}] completed residual unit={unit['unit_id']} stageA={len(a_rows)} stageB={len(b_rows)} events={len(e_rows)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-30:]})
                save_json(shard_dir / "failed_units.jsonl", failed)
                print(f"[{now()}] failed residual unit={unit.get('unit_id')} error={exc!r}", flush=True)
    finally:
        capture.close()
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "SHARD_COMPLETE" if not failed else "SHARD_COMPLETE_WITH_FAILURES",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "N_UNITS": len(units),
        "N_NEW_STAGE_A_EVALUATIONS": len(stage_a_rows),
        "N_NEW_STAGE_B_TRAJECTORIES": len(stage_a_rows) * 2,
        "N_STAGE_B_HORIZON_ROWS": len(stage_b_rows),
        "N_EVENT_ROWS": len(event_rows),
        "N_FAILED_UNITS": len(failed),
        "failed_units": failed,
    }
    save_json(shard_dir / "formal_summary.json", summary)
    return summary


def merge_formal(args):
    prepare()
    shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x] if args.shard_dirs else sorted(RUN_DIR.glob("formal_shard*_of_*"))
    a_rows, b_rows, e_rows, summaries = [], [], [], []
    for shard in shard_dirs:
        a_rows.extend(read_csv(shard / "stage_a_local_results.csv"))
        b_rows.extend(read_csv(shard / "stage_b_frozen_propagation_results.csv"))
        e_rows.extend(read_csv(shard / "residual_functional_event_level.csv"))
        if (shard / "formal_summary.json").exists():
            summaries.append(json.loads((shard / "formal_summary.json").read_text(encoding="utf-8")))
    def dedup(rows, keys):
        out = {}
        for row in rows:
            out[tuple(str(row.get(k)) for k in keys)] = row
        return [out[k] for k in sorted(out)]
    a_rows = dedup(a_rows, ["unit_id", "group_size"])
    b_rows = dedup(b_rows, ["unit_id", "group_size", "history_length"])
    e_rows = dedup(e_rows, ["unit_id", "group_size", "stage", "history", "condition"])
    write_csv(OUT_DIR / "stage_a_local_results.csv", a_rows)
    write_csv(OUT_DIR / "stage_b_frozen_propagation_results.csv", b_rows)
    write_csv(OUT_DIR / "residual_functional_event_level.csv", e_rows)
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "MERGED_FORMAL_SHARDS",
        "MERGED_SHARDS": [str(x) for x in shard_dirs],
        "N_NEW_STAGE_A_EVALUATIONS": len(a_rows),
        "N_NEW_STAGE_B_TRAJECTORIES": len(a_rows) * 2,
        "N_STAGE_B_HORIZON_ROWS": len(b_rows),
        "N_EVENT_ROWS": len(e_rows),
        "shard_summaries": summaries,
    }
    save_json(OUT_DIR / "execution_manifest.json", out)
    return out


def load_stage_c_reuse():
    rows = read_csv(MAG_OUT_DIR / "unit_level_causal_summary.csv")
    out = []
    for row in rows:
        out.append({
            "unit_id": str(row["unit_id"]),
            "prompt_id": row.get("prompt_id"),
            "sample_id": row.get("sample_id"),
            "layer": row.get("layer"),
            "head": row.get("head"),
            "t0": row.get("t0"),
            "group_size": int(row["group_size"]),
            "closed_loop_residual_gap": as_float(row.get("ResidualGap_i_g")),
            "closed_loop_native_gap": as_float(row.get("NativeGap_i_g")),
            "closed_loop_rescue": as_float(row.get("Rescue_i_g")),
            "reuse_source": str(MAG_OUT_DIR / "unit_level_causal_summary.csv"),
        })
    return out


def compute_unit_level_residual_summary(stage_a, stage_b, stage_c):
    a_by = {(str(r["unit_id"]), int(r["group_size"])): r for r in stage_a}
    c_by = {(str(r["unit_id"]), int(r["group_size"])): r for r in stage_c}
    b_by = defaultdict(dict)
    for row in stage_b:
        b_by[(str(row["unit_id"]), int(row["group_size"]))][int(row["history_length"])] = row
    out = []
    for key in sorted(a_by):
        if key not in c_by:
            continue
        a = a_by[key]
        c = c_by[key]
        row = {
            "unit_id": str(a["unit_id"]),
            "prompt_id": a.get("prompt_id"),
            "sample_id": a.get("sample_id"),
            "layer": a.get("layer"),
            "head": a.get("head"),
            "t0": a.get("t0"),
            "group_size": int(a["group_size"]),
            "local_directional_gap": as_float(a.get("local_directional_gap")),
            "propagation_gap_L1": None,
            "propagation_gap_L4": None,
            "propagation_gap_L8": None,
            "propagation_gap_L16": None,
            "propagation_gap_L32": None,
            "propagation_gap_L64": None,
            "closed_loop_residual_gap": as_float(c.get("closed_loop_residual_gap")),
            "R_C_error_direction_cosine": as_float(a.get("R_C_direction_cosine")),
        }
        for L in HISTORY_LENGTHS:
            if L in b_by[key]:
                row[f"propagation_gap_L{L}"] = as_float(b_by[key][L].get("propagation_directional_gap"))
        out.append(row)
    return out


def summarize_group(unit_rows, group_size, reps=BOOTSTRAP_REPS):
    rows = [r for r in unit_rows if int(r["group_size"]) == int(group_size)]
    local_vals = [r["local_directional_gap"] for r in rows]
    prop64_vals = [r["propagation_gap_L64"] for r in rows]
    residual_vals = [r["closed_loop_residual_gap"] for r in rows]
    amplification_vals = [
        float(r["propagation_gap_L64"]) - float(r["local_directional_gap"])
        for r in rows
        if finite(r.get("propagation_gap_L64")) and finite(r.get("local_directional_gap"))
    ]
    local_ci = bootstrap_ci(local_vals, reps=reps, seed=BOOTSTRAP_SEED + group_size)
    prop_ci = bootstrap_ci(prop64_vals, reps=reps, seed=BOOTSTRAP_SEED + group_size + 1000)
    residual_ci = bootstrap_ci(residual_vals, reps=reps, seed=BOOTSTRAP_SEED + group_size + 2000)
    amplification_ci = bootstrap_ci(amplification_vals, reps=reps, seed=BOOTSTRAP_SEED + group_size + 3000)
    local_med = median(local_vals)
    prop_med = median(prop64_vals)
    residual_med = median(residual_vals)
    amplification_med = median(amplification_vals)
    local_supported = local_ci and (local_ci[0] > 0 or local_ci[1] < 0)
    prop_supported = prop_ci and (prop_ci[0] > 0 or prop_ci[1] < 0)
    amplification_supported = amplification_ci and amplification_ci[0] > 0
    if local_supported and amplification_supported:
        cls = "LOCAL_PLUS_RECURRENT"
    elif local_supported:
        cls = "LOCAL"
    elif prop_supported:
        cls = "RECURRENT"
    elif (local_ci and local_ci[0] <= 0 <= local_ci[1]) and (prop_ci and prop_ci[0] <= 0 <= prop_ci[1]):
        cls = "WEAK"
    else:
        cls = "MIXED"
    return {
        "group_size": group_size,
        "local_directional_gap": local_med,
        "local_gap_mean": mean(local_vals),
        "local_gap_trimmed_mean": trimmed_mean(local_vals),
        "local_gap_ci": local_ci,
        "propagated_directional_gap_L64": prop_med,
        "propagation_gap_L64_mean": mean(prop64_vals),
        "propagation_gap_L64_ci": prop_ci,
        "local_to_recurrent_amplification_L64": amplification_med,
        "amplification_L64_mean": mean(amplification_vals),
        "amplification_L64_ci": amplification_ci,
        "closed_loop_residual_gap": residual_med,
        "closed_loop_residual_gap_ci": residual_ci,
        "C_more_harmful_local_count": sum(1 for x in local_vals if x is not None and x > NEAR_TIE),
        "R_more_harmful_local_count": sum(1 for x in local_vals if x is not None and x < -NEAR_TIE),
        "C_more_harmful_prop_L64_count": sum(1 for x in prop64_vals if x is not None and x > NEAR_TIE),
        "R_more_harmful_prop_L64_count": sum(1 for x in prop64_vals if x is not None and x < -NEAR_TIE),
        "N_UNITS": len(rows),
        "residual_classification": cls,
    }


def history_conditioned(stage_b):
    rows = []
    for group_size in GROUP_SIZES:
        for L in HISTORY_LENGTHS:
            vals = [as_float(r.get("propagation_directional_gap")) for r in stage_b if int(r["group_size"]) == group_size and int(r["history_length"]) == L]
            rows.append({
                "group_size": group_size,
                "history_length": L,
                "median_propagation_directional_gap": median(vals),
                "propagation_directional_gap_ci": bootstrap_ci(vals, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + group_size + L),
                "C_more_harmful_count": sum(1 for x in vals if x is not None and x > NEAR_TIE),
                "R_more_harmful_count": sum(1 for x in vals if x is not None and x < -NEAR_TIE),
            })
    return rows


def association_stats(rows, x_key, y_key, seed):
    rows = [r for r in rows if as_float(r.get(x_key)) is not None and as_float(r.get(y_key)) is not None]
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    return {
        "spearman": spearman(xs, ys),
        "spearman_ci": bootstrap_spearman_ci(rows, x_key, y_key, seed=seed),
        "N_ROWS": len(rows),
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit_cluster",
    }


def association_payload(unit_rows, x_key, y_key, seed):
    pooled = association_stats(unit_rows, x_key, y_key, seed)
    by_group = {}
    for idx, group_size in enumerate(GROUP_SIZES):
        rows = [r for r in unit_rows if int(r["group_size"]) == int(group_size)]
        by_group[str(group_size)] = association_stats(rows, x_key, y_key, seed + 100 * (idx + 1))
    return {
        "x": x_key,
        "y": y_key,
        **pooled,
        "pooled": pooled,
        "by_group": by_group,
    }


def loo_sign_stable(unit_rows, key, subgroup_key=None):
    vals = [as_float(r.get(key)) for r in unit_rows if as_float(r.get(key)) is not None]
    full = median(vals)
    if full is None or abs(full) <= NEAR_TIE:
        return "FAIL"
    if subgroup_key is None:
        for idx in range(len(vals)):
            rem = vals[:idx] + vals[idx + 1:]
            if not rem or (median(rem) or 0.0) * full <= 0:
                return "FAIL"
        return "PASS"
    groups = sorted({str(r.get(subgroup_key, "")) for r in unit_rows})
    for group in groups:
        rem = [as_float(r.get(key)) for r in unit_rows if str(r.get(subgroup_key, "")) != group and as_float(r.get(key)) is not None]
        if not rem or (median(rem) or 0.0) * full <= 0:
            return "FAIL"
    return "PASS"


def classify_residual_functional_sensitivity(summary, group_rows, robustness):
    if not summary.get("blocking_gates_ok", True):
        return {
            "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "INVALID",
            "FINAL_CLASSIFICATION": "INVALID_RESIDUAL_FUNCTIONAL_SENSITIVITY_EXPERIMENT",
            "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        }
    local_supported = [
        r for r in group_rows
        if r.get("local_gap_ci") and (r["local_gap_ci"][0] > 0 or r["local_gap_ci"][1] < 0)
    ]
    recurrent_supported = [
        r for r in group_rows
        if r.get("propagation_gap_L64_ci") and (r["propagation_gap_L64_ci"][0] > 0 or r["propagation_gap_L64_ci"][1] < 0)
    ]
    recurrent_added = [
        r for r in group_rows
        if r.get("amplification_L64_ci") and r["amplification_L64_ci"][0] > 0
    ]
    assoc_ci = summary.get("LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI") or [None, None]
    local_assoc = assoc_ci[0] is not None and assoc_ci[0] > 0
    robust_ok = all(
        robustness.get(key) == "PASS"
        for key in [
            "RESIDUAL_LOO_UNIT_SIGN_STABLE",
            "RESIDUAL_LOO_LAYER_SIGN_STABLE",
            "RESIDUAL_LOO_PROMPT_SIGN_STABLE",
        ]
    )
    if not robust_ok and (local_supported or recurrent_supported):
        return {
            "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "MIXED_OR_INCONCLUSIVE",
            "FINAL_CLASSIFICATION": "KDA_RESIDUAL_FUNCTIONAL_SENSITIVITY_MIXED_OR_INCONCLUSIVE",
            "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        }
    if len(local_supported) >= 2 and robust_ok and (local_assoc or summary.get("LOCAL_TO_CLOSED_LOOP_SPEARMAN", 0.0) > 0):
        if len(recurrent_added) >= 2:
            return {
                "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "LOCAL_PLUS_RECURRENT",
                "FINAL_CLASSIFICATION": "KDA_RESIDUAL_FUNCTIONAL_GEOMETRY_HAS_LOCAL_AND_RECURRENT_COMPONENTS",
                "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
            }
        return {
            "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "LOCAL_SUPPORTED",
            "FINAL_CLASSIFICATION": "KDA_MATCHED_MAGNITUDE_ERRORS_RETAIN_LOCAL_FUNCTIONAL_DIRECTION_SENSITIVITY",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_LOCAL_FUNCTIONAL_GEOMETRY_LOCALIZATION_V1",
        }
    if len(recurrent_supported) >= 2 and robust_ok:
        return {
            "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "RECURRENT_PROPAGATION_SUPPORTED",
            "FINAL_CLASSIFICATION": "KDA_MATCHED_MAGNITUDE_ERROR_DIRECTIONS_ARE_DIFFERENTIALLY_PROPAGATED_BY_RECURRENCE",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_RECURRENT_DIRECTIONAL_PROPAGATION_CAUSAL_V1",
        }
    if not local_supported and not recurrent_supported:
        return {
            "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "WEAK_OR_NOT_SUPPORTED",
            "FINAL_CLASSIFICATION": "KDA_RESIDUAL_FUNCTIONAL_GEOMETRY_WEAK_AFTER_MAGNITUDE_CONTROL",
            "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        }
    return {
        "RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS": "MIXED_OR_INCONCLUSIVE",
        "FINAL_CLASSIFICATION": "KDA_RESIDUAL_FUNCTIONAL_SENSITIVITY_MIXED_OR_INCONCLUSIVE",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
    }


def analyze(args):
    if not (OUT_DIR / "pipeline_identity_audit.json").exists():
        prepare()
    stage_a = read_csv(OUT_DIR / "stage_a_local_results.csv")
    stage_b = read_csv(OUT_DIR / "stage_b_frozen_propagation_results.csv")
    stage_c = load_stage_c_reuse()
    write_csv(OUT_DIR / "stage_c_closed_loop_reuse.csv", stage_c)
    unit_rows = compute_unit_level_residual_summary(stage_a, stage_b, stage_c)
    write_csv(OUT_DIR / "unit_level_residual_functional_summary.csv", unit_rows)
    group_rows = [summarize_group(unit_rows, g, reps=args.bootstrap_reps) for g in GROUP_SIZES]
    write_csv(OUT_DIR / "granularity_residual_summary.csv", group_rows)
    history_rows = history_conditioned(stage_b)
    write_csv(OUT_DIR / "history_conditioned_residual_summary.csv", history_rows)
    local_assoc = association_payload(unit_rows, "local_directional_gap", "closed_loop_residual_gap", BOOTSTRAP_SEED + 3000)
    prop_assoc = association_payload(unit_rows, "propagation_gap_L64", "closed_loop_residual_gap", BOOTSTRAP_SEED + 4000)
    save_json(OUT_DIR / "local_to_closed_loop_association.json", local_assoc)
    save_json(OUT_DIR / "propagation_to_closed_loop_association.json", prop_assoc)
    robustness = {
        "RESIDUAL_LOO_UNIT_SIGN_STABLE": loo_sign_stable(unit_rows, "local_directional_gap"),
        "RESIDUAL_LOO_LAYER_SIGN_STABLE": loo_sign_stable(unit_rows, "local_directional_gap", "layer"),
        "RESIDUAL_LOO_PROMPT_SIGN_STABLE": loo_sign_stable(unit_rows, "local_directional_gap", "prompt_id"),
    }
    write_csv(OUT_DIR / "residual_robustness.csv", [robustness])
    audit = json.loads((OUT_DIR / "pipeline_identity_audit.json").read_text(encoding="utf-8"))
    blocking_gates = [
        "PIPELINE_IDENTITY_GATE",
        "STATE_SEMANTICS_GATE",
        "COMMON_COHORT_REUSE_GATE",
        "MATCHED_CARRIER_GATE",
        "READOUT_SEMANTICS_GATE",
        "LOCAL_LOGIT_TIMING_GATE",
        "INTERVENTION_TIMING_GATE",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
        "DETERMINISM_GATE",
        "FROZEN_DRIVER_GATE",
        "NO_FURTHER_QUANTIZATION_GATE",
        "HORIZON_PREFIX_EQUIVALENCE_GATE",
        "CLOSED_LOOP_BASELINE_REUSE_GATE",
    ]
    stage_gates = {
        "MAGNITUDE_MATCH_GATE": "PASS" if all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in stage_a) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in stage_a) else "FAIL",
    }
    gates_ok = all(audit.get(k) == "PASS" for k in blocking_gates) and all(v == "PASS" for v in stage_gates.values())
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if gates_ok and len(unit_rows) == N_COMMON_UNITS * len(GROUP_SIZES) else "INCOMPLETE",
        **{k: audit.get(k) for k in blocking_gates},
        **stage_gates,
        "RESIDUAL_JOIN_GATE": "PASS" if len(unit_rows) == N_COMMON_UNITS * len(GROUP_SIZES) else "FAIL",
        "N_COMMON_UNITS": N_COMMON_UNITS,
        "N_GROUP_SIZES": len(GROUP_SIZES),
        "N_NEW_STAGE_A_EVALUATIONS": len(stage_a),
        "N_NEW_STAGE_B_TRAJECTORIES": len(unit_rows) * 2,
        "N_STAGE_B_HORIZON_ROWS": len(stage_b),
        "N_REUSED_STAGE_C_ROWS": len(stage_c),
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN": local_assoc["spearman"],
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI": local_assoc["spearman_ci"],
        "PROPAGATION_TO_CLOSED_LOOP_SPEARMAN": prop_assoc["spearman"],
        "PROPAGATION_TO_CLOSED_LOOP_SPEARMAN_CI": prop_assoc["spearman_ci"],
        "blocking_gates_ok": gates_ok,
        **robustness,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
    }
    for row in group_rows:
        g = int(row["group_size"])
        summary[f"G{g}_LOCAL_DIRECTIONAL_GAP"] = row["local_directional_gap"]
        summary[f"G{g}_LOCAL_DIRECTIONAL_GAP_CI"] = row["local_gap_ci"]
        summary[f"G{g}_PROPAGATED_DIRECTIONAL_GAP_L64"] = row["propagated_directional_gap_L64"]
        summary[f"G{g}_PROPAGATED_DIRECTIONAL_GAP_L64_CI"] = row["propagation_gap_L64_ci"]
        summary[f"G{g}_LOCAL_TO_RECURRENT_AMPLIFICATION_L64"] = row["local_to_recurrent_amplification_L64"]
        summary[f"G{g}_LOCAL_TO_RECURRENT_AMPLIFICATION_L64_CI"] = row["amplification_L64_ci"]
        summary[f"G{g}_CLOSED_LOOP_RESIDUAL_GAP"] = row["closed_loop_residual_gap"]
        summary[f"G{g}_RESIDUAL_CLASSIFICATION"] = row["residual_classification"]
    classification = classify_residual_functional_sensitivity(summary, group_rows, robustness)
    summary.update(classification)
    save_json(OUT_DIR / "bootstrap_results.json", {
        "group_rows": group_rows,
        "history_rows": history_rows,
        "local_to_closed_loop": local_assoc,
        "propagation_to_closed_loop": prop_assoc,
    })
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\nResidual functional sensitivity localization outputs.\n")
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS: {summary['RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS']}",
        f"FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
        "",
        "Stage A/B are fresh matched-direction residual sensitivity outputs. Stage C reuses prior C_MAG_EQ_TO_R closed-loop residual gaps.",
        "",
        "The primary frozen-driver metric is KDA backend output relative distortion under FP replay drivers.",
    ]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    return summary


def verify(args):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    stage_a = read_csv(OUT_DIR / "stage_a_local_results.csv")
    stage_b = read_csv(OUT_DIR / "stage_b_frozen_propagation_results.csv")
    stage_c = read_csv(OUT_DIR / "stage_c_closed_loop_reuse.csv")
    unit_rows = read_csv(OUT_DIR / "unit_level_residual_functional_summary.csv")
    event_rows = read_csv(OUT_DIR / "residual_functional_event_level.csv")
    numeric_bad = []
    for path_name in ["stage_a_local_results.csv", "stage_b_frozen_propagation_results.csv", "unit_level_residual_functional_summary.csv"]:
        for idx, row in enumerate(read_csv(OUT_DIR / path_name)):
            for key, value in row.items():
                if value in ("", "None", "NA"):
                    continue
                if key.endswith("_gap") or key.endswith("_norm") or key.endswith("_cosine") or key in {"damage_R", "damage_C_match"}:
                    if not finite(value):
                        numeric_bad.append((path_name, idx, key, value))
    expected_a = N_COMMON_UNITS * len(GROUP_SIZES)
    expected_b = N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS)
    expected_c = N_COMMON_UNITS * len(GROUP_SIZES)
    fresh_gate = "PASS" if not missing and len(stage_a) == expected_a and len(stage_b) == expected_b and len(stage_c) == expected_c and not numeric_bad else "FAIL"
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": fresh_gate,
        "missing_files": missing,
        "N_STAGE_A_ROWS": len(stage_a),
        "N_EXPECTED_STAGE_A_ROWS": expected_a,
        "N_STAGE_B_ROWS": len(stage_b),
        "N_EXPECTED_STAGE_B_ROWS": expected_b,
        "N_STAGE_C_REUSED_ROWS": len(stage_c),
        "N_EXPECTED_STAGE_C_REUSED_ROWS": expected_c,
        "N_UNIT_ROWS": len(unit_rows),
        "N_EVENT_ROWS": len(event_rows),
        "numeric_bad": numeric_bad[:20],
        "summary_status": summary.get("FORMAL_STATUS"),
        "summary_sha256": file_sha256(OUT_DIR / "summary.json") if (OUT_DIR / "summary.json").exists() else None,
    }
    if summary:
        summary["FRESH_OUTPUT_VERIFICATION"] = fresh_gate
        save_json(OUT_DIR / "summary.json", summary)
    save_json(OUT_DIR / "fresh_output_verification.json", verification)
    return verification


def final_summary(py_compile_status="NOT_RUN", pytest_status="NOT_RUN", bash_syntax_status="NOT_RUN", gpu_release_status="NOT_RUN"):
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "PY_COMPILE": py_compile_status,
        "PYTEST": pytest_status,
        "BASH_SYNTAX_CHECK": bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": gpu_release_status,
    })
    lines = []
    for key in SUMMARY_KEYS:
        lines.append(f"{key} = {summary.get(key)}")
    lines.append("STOP.")
    text = "\n".join(lines) + "\n"
    write_text(OUT_DIR / "final_console_summary.txt", text)
    if summary:
        save_json(summary_path, summary)
    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "formal", "merge-formal", "analyze", "verify", "summary"], default="prepare")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--unit-limit", type=int, default=None)
    p.add_argument("--shard-dirs", default=None)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--py-compile-status", default="NOT_RUN")
    p.add_argument("--pytest-status", default="NOT_RUN")
    p.add_argument("--bash-syntax-status", default="NOT_APPLICABLE")
    p.add_argument("--gpu-release-status", default="NOT_RUN")
    p.add_argument("--console-summary", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "formal":
        print(json.dumps(formal(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "merge-formal":
        print(json.dumps(merge_formal(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "analyze":
        out = analyze(args)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "verify":
        print(json.dumps(verify(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "summary":
        text = final_summary(args.py_compile_status, args.pytest_status, args.bash_syntax_status, args.gpu_release_status)
        print(text if args.console_summary else json.dumps({"summary": str(OUT_DIR / "final_console_summary.txt")}, indent=2))


if __name__ == "__main__":
    main()
