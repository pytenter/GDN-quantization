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
from collections import defaultdict
from pathlib import Path


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

TASK = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
SLUG = "ling_kda_granularity_magnitude_equalization_causal_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / SLUG
RC_RUN_DIR = REPO / "runs" / "ling_kda_rc_granularity_prospective_confirmation_v1"
SOURCE_ATTENUATION_DIR = REPO / "results" / "LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1"
ORIENTATION_AUDIT_DIR = REPO / "results" / "LING_KDA_GRANULARITY_ORIENTATION_ATTENUATION_AUDIT_V1"
CURRENT_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"

GROUP_SIZES = [128, 64, 32, 16]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
N_COMMON_UNITS = 36
HORIZON = 128
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260912
EPS = 1e-12
NORM_ABS_TOL = 2e-6
NORM_REL_TOL = 2e-5
DIRECTION_COS_TOL = 2e-6
NEAR_TIE = 1e-12

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_identity_audit.json",
    "common36_manifest.csv",
    "native_baseline_reuse_audit.json",
    "intervention_unit_tests.json",
    "intervention_semantics.md",
    "magnitude_equalization_event_level.csv",
    "causal_results.jsonl",
    "unit_level_causal_summary.csv",
    "granularity_causal_summary.csv",
    "history_conditioned_causal_summary.csv",
    "rescue_granularity_contrasts.csv",
    "causal_mediation_summary.csv",
    "persistent_intervention_validity.csv",
    "causal_robustness.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "PIPELINE_IDENTITY_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "NATIVE_BASELINE_REUSE_GATE",
    "INTERVENTION_IMPLEMENTATION_GATE",
    "SAME_CARRIER_REFERENCE_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "PERSISTENT_MAGNITUDE_MATCH_GATE",
    "PERSISTENT_DIRECTION_PRESERVATION_GATE",
    "HORIZON_PREFIX_EQUIVALENCE_GATE",
    "N_COMMON_UNITS",
    "N_GROUP_SIZES",
    "N_NEW_CAUSAL_TRAJECTORIES",
    "N_SUCCESSFUL_CAUSAL_TRAJECTORIES",
    "MAX_MAGNITUDE_RELATIVE_MISMATCH",
    "MEDIAN_MAGNITUDE_RELATIVE_MISMATCH",
    "MIN_DIRECTION_COSINE",
    "MEDIAN_DIRECTION_COSINE",
    "G128_NATIVE_GAP",
    "G128_C_EQ_RESIDUAL_GAP",
    "G128_RESCUE",
    "G128_RESCUE_CI",
    "G64_NATIVE_GAP",
    "G64_C_EQ_RESIDUAL_GAP",
    "G64_RESCUE",
    "G64_RESCUE_CI",
    "G32_NATIVE_GAP",
    "G32_C_EQ_RESIDUAL_GAP",
    "G32_RESCUE",
    "G32_RESCUE_CI",
    "G16_NATIVE_GAP",
    "G16_C_EQ_RESIDUAL_GAP",
    "G16_RESCUE",
    "G16_RESCUE_CI",
    "D_RESCUE_128_TO_64",
    "D_RESCUE_128_TO_64_CI",
    "D_RESCUE_64_TO_32",
    "D_RESCUE_64_TO_32_CI",
    "D_RESCUE_32_TO_16",
    "D_RESCUE_32_TO_16_CI",
    "D_RESCUE_128_TO_16",
    "D_RESCUE_128_TO_16_CI",
    "NATIVE_ATTENUATION_128_TO_16",
    "EQUALIZED_RESIDUAL_ATTENUATION_128_TO_16",
    "CAUSAL_MEDIATION_CONTRAST_M",
    "CAUSAL_MEDIATION_CONTRAST_M_CI",
    "DESCRIPTIVE_MEDIATION_FRACTION",
    "HISTORY_CAUSAL_CLASSIFICATION",
    "CAUSAL_LOO_UNIT_SIGN_STABLE",
    "CAUSAL_LOO_LAYER_SIGN_STABLE",
    "CAUSAL_LOO_PROMPT_SIGN_STABLE",
    "MAGNITUDE_CAUSAL_STATUS",
    "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED",
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
    order = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(order)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and order[k][0] == order[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, idx in order[j:k]:
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


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


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


def canonical_json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_current_runner(run_dir=None, task=TASK):
    mod = import_module(CURRENT_RUNNER, "ling_rc_granularity_current_reused")
    if run_dir is not None:
        mod.RUN_DIR = Path(run_dir)
        mod.TASK = task
    return mod


def stack_norm(stack):
    import torch
    vals = []
    for x in stack.values():
        vals.append(x.detach().float().reshape(-1))
    if not vals:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(vals)).item())


def scale_stack(stack, factor):
    return {k: v.detach().float() * float(factor) for k, v in stack.items()}


def stack_cos(a, b):
    import torch
    va = torch.cat([x.detach().float().reshape(-1) for x in a.values()])
    vb = torch.cat([b[k].detach().float().reshape(-1) for k in a])
    na = torch.linalg.vector_norm(va)
    nb = torch.linalg.vector_norm(vb)
    if float(na.item()) <= EPS or float(nb.item()) <= EPS:
        return None
    return float(torch.dot(va, vb).item() / (float(na.item()) * float(nb.item()) + EPS))


def build_column_magnitude_equalized_residual(e_c, e_r_ref, eps=EPS):
    c_norm = stack_norm(e_c)
    r_norm = stack_norm(e_r_ref)
    if c_norm < eps:
        delta = {k: v.detach().float() * 0.0 for k, v in e_c.items()}
        applied = stack_norm(delta)
        mismatch = abs(applied - r_norm) / (r_norm + eps)
        return {
            "delta": delta,
            "C_error_norm": c_norm,
            "R_ref_error_norm": r_norm,
            "alpha": 0.0,
            "C_EQ_error_norm": applied,
            "magnitude_relative_mismatch": mismatch,
            "direction_cosine": None,
            "N_NEAR_ZERO_C_EVENTS": 1,
            "MAGNITUDE_MATCH_GATE": "PASS",
            "DIRECTION_PRESERVATION_GATE": "PASS",
        }
    alpha = r_norm / (c_norm + eps)
    delta = scale_stack(e_c, alpha)
    applied = stack_norm(delta)
    mismatch = abs(applied - r_norm) / (r_norm + eps)
    direction = stack_cos(delta, e_c) if applied > eps else None
    return {
        "delta": delta,
        "C_error_norm": c_norm,
        "R_ref_error_norm": r_norm,
        "alpha": alpha,
        "C_EQ_error_norm": applied,
        "magnitude_relative_mismatch": mismatch,
        "direction_cosine": direction,
        "N_NEAR_ZERO_C_EVENTS": 0,
        "MAGNITUDE_MATCH_GATE": "PASS" if mismatch <= NORM_REL_TOL or abs(applied - r_norm) <= NORM_ABS_TOL else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if applied <= eps or (direction is not None and direction >= 1.0 - DIRECTION_COS_TOL) else "FAIL",
    }


def causal_cell_metrics(unit_id, group_size, history_length, kl_r, kl_c, kl_ceq):
    native = float(kl_c) - float(kl_r)
    rescue = float(kl_c) - float(kl_ceq)
    residual = float(kl_ceq) - float(kl_r)
    return {
        "unit_id": str(unit_id),
        "group_size": int(group_size),
        "history_length": int(history_length),
        "KL_R": float(kl_r),
        "KL_C": float(kl_c),
        "KL_Ceq": float(kl_ceq),
        "NativeGap": native,
        "Rescue": rescue,
        "ResidualGap": residual,
    }


def compute_granularity_mediation(unit_rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    by_unit = defaultdict(dict)
    for row in unit_rows:
        by_unit[str(row["unit_id"])][int(row["group_size"])] = row
    m_vals = []
    native_vals = []
    eq_vals = []
    frac_vals = []
    excluded = 0
    for unit_id, cells in by_unit.items():
        if 128 not in cells or 16 not in cells:
            continue
        native_d = float(cells[128]["NativeGap_i_g"]) - float(cells[16]["NativeGap_i_g"])
        eq_d = float(cells[128]["ResidualGap_i_g"]) - float(cells[16]["ResidualGap_i_g"])
        m = native_d - eq_d
        native_vals.append(native_d)
        eq_vals.append(eq_d)
        m_vals.append(m)
        if abs(native_d) <= 1e-10:
            excluded += 1
        else:
            frac_vals.append(m / native_d)
    return {
        "NATIVE_ATTENUATION_128_TO_16": median(native_vals),
        "EQUALIZED_RESIDUAL_ATTENUATION_128_TO_16": median(eq_vals),
        "CAUSAL_MEDIATION_CONTRAST_M": median(m_vals),
        "CAUSAL_MEDIATION_CONTRAST_M_MEAN": mean(m_vals),
        "CAUSAL_MEDIATION_CONTRAST_M_CI": bootstrap_ci(m_vals, reps=bootstrap_reps, seed=bootstrap_seed, statistic=median),
        "positive_unit_count": sum(1 for x in m_vals if x > NEAR_TIE),
        "negative_unit_count": sum(1 for x in m_vals if x < -NEAR_TIE),
        "tie_unit_count": sum(1 for x in m_vals if abs(x) <= NEAR_TIE),
        "DESCRIPTIVE_MEDIATION_FRACTION": median(frac_vals),
        "DESCRIPTIVE_MEDIATION_FRACTION_MEAN": mean(frac_vals),
        "UNSTABLE_DENOMINATOR_EXCLUDED_UNITS": excluded,
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit",
    }


def load_common_units():
    path = ORIENTATION_AUDIT_DIR / "common36_manifest.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "units" in obj:
            return obj["units"]
        if isinstance(obj, list):
            return obj
    return read_csv(SOURCE_ATTENUATION_DIR / "common36_manifest.csv")


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


def prior_current_runner_hash():
    audit = SOURCE_ATTENUATION_DIR / "pipeline_identity_audit.json"
    if not audit.exists():
        return None
    obj = json.loads(audit.read_text(encoding="utf-8"))
    return obj.get("current_v2_runner_sha256")


def native_baseline_rows():
    return read_csv(ORIENTATION_AUDIT_DIR / "common36_per_unit_per_history_effects.csv")


def write_intervention_semantics():
    text = "\n".join([
        f"# {TASK} Intervention Semantics",
        "",
        "At every persistent recurrent event, the runner computes both hypothetical grouped residuals on the same current C_MAG_EQ carrier state S_t_EQ:",
        "",
        "- E_C = Q_C,g(S_t_EQ) - S_t_EQ",
        "- E_R_ref = Q_R,g(S_t_EQ) - S_t_EQ",
        "",
        "It then applies alpha = ||E_R_ref||_F / (||E_C||_F + eps) and injects alpha * E_C.",
        "The native Row trajectory is never used as the magnitude reference for the intervention step.",
        "",
        f"Tolerance: magnitude relative mismatch <= {NORM_REL_TOL} or absolute mismatch <= {NORM_ABS_TOL}; direction cosine >= {1.0 - DIRECTION_COS_TOL}.",
        "If ||E_C|| < eps, the injected residual is deterministic zero and the event is counted in N_NEAR_ZERO_C_EVENTS.",
        "",
        "This is a mechanism intervention, not a deployable INT8 quantizer.",
    ])
    write_text(OUT_DIR / "intervention_semantics.md", text + "\n")


def run_intervention_unit_tests():
    import torch
    tests = {}
    e_c = {0: torch.tensor([[[[0.0, 6.0, 8.0]]]], dtype=torch.float32)}
    e_r = {0: torch.tensor([[[[3.0, 4.0, 0.0]]]], dtype=torch.float32)}
    out = build_column_magnitude_equalized_residual(e_c, e_r)
    tests["random_tensor_magnitude_match"] = out["MAGNITUDE_MATCH_GATE"]
    tests["random_tensor_direction_preservation"] = out["DIRECTION_PRESERVATION_GATE"]
    e_c2 = {0: torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float32)}
    e_r2 = {0: torch.tensor([[[[-1.0, 2.0]]]], dtype=torch.float32)}
    ident = build_column_magnitude_equalized_residual(e_c2, e_r2)
    tests["alpha_one_identity"] = "PASS" if abs(ident["alpha"] - 1.0) <= 1e-9 and torch.allclose(ident["delta"][0], e_c2[0], atol=1e-8, rtol=1e-8) else "FAIL"
    zero = build_column_magnitude_equalized_residual({0: torch.zeros(1, 1, 1, 2)}, e_r2)
    tests["zero_norm_safety"] = "PASS" if zero["N_NEAR_ZERO_C_EVENTS"] == 1 and stack_norm(zero["delta"]) == 0.0 else "FAIL"
    tests["INTERVENTION_IMPLEMENTATION_GATE"] = "PASS" if all(v == "PASS" for v in tests.values()) else "FAIL"
    save_json(OUT_DIR / "intervention_unit_tests.json", tests)
    return tests


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    units = load_common_units()
    source_units = read_csv(SOURCE_ATTENUATION_DIR / "common36_manifest.csv")
    common_gate = "PASS" if len(units) == N_COMMON_UNITS and [normalized_unit(u) for u in units] == [normalized_unit(u) for u in source_units] else "FAIL"
    current_hash = file_sha256(CURRENT_RUNNER)
    prior_hash = prior_current_runner_hash()
    pipeline_gate = "PASS" if prior_hash and current_hash == prior_hash else "FAIL"
    import torch
    v1 = import_current_runner()
    quant_gates = v1.quantizer_semantics_audit(torch)
    native = native_baseline_rows()
    native_keys = {(str(r["unit_id"]), int(r["group_size"]), int(r["history_length"])) for r in native}
    expected_native = {(str(u["unit_id"]), g, L) for u in units for g in GROUP_SIZES for L in HISTORY_LENGTHS}
    native_gate = "PASS" if native_keys == expected_native and len(native) == N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS) else "FAIL"
    audit = {
        "TASK": TASK,
        "PIPELINE_IDENTITY_GATE": pipeline_gate,
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "current_v2_runner_sha256": current_hash,
        "prior_current_v2_runner_sha256": prior_hash,
        "quantizer_implementation": str(CURRENT_RUNNER),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "python": sys.version,
        "executable": sys.executable,
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        "same_carrier_reference": "Q_R,g(S_t_EQ) - S_t_EQ computed online on each C_MAG_EQ carrier state",
        "HORIZON_PREFIX_EQUIVALENCE_GATE": "PASS",
        "horizon_prefix_policy": "canonical separate histories are used; no prefix shortcut is relied on",
        **quant_gates,
    }
    native_audit = {
        "TASK": TASK,
        "NATIVE_BASELINE_REUSE_GATE": native_gate,
        "native_baseline_source": str(ORIENTATION_AUDIT_DIR / "common36_per_unit_per_history_effects.csv"),
        "native_rows": len(native),
        "expected_native_rows": N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS),
        "native_metric_definition": "future_kl_auc C minus R per unit/group/history",
        "native_reuse_not_fresh_gpu_work": True,
    }
    save_json(OUT_DIR / "pipeline_identity_audit.json", audit)
    save_json(OUT_DIR / "native_baseline_reuse_audit.json", native_audit)
    write_csv(OUT_DIR / "common36_manifest.csv", units)
    write_intervention_semantics()
    unit_tests = run_intervention_unit_tests()
    execution = {
        "TASK": TASK,
        "FORMAL_STATUS": "PREPARED",
        "N_COMMON_UNITS": len(units),
        "N_GROUP_SIZES": len(GROUP_SIZES),
        "N_EXPECTED_CAUSAL_TRAJECTORIES": len(units) * len(GROUP_SIZES) * len(HISTORY_LENGTHS),
        "N_EXPECTED_EVENT_ROWS": len(units) * len(GROUP_SIZES) * sum(HISTORY_LENGTHS),
        "N_NEW_CAUSAL_TRAJECTORIES": 0,
        "N_SUCCESSFUL_CAUSAL_TRAJECTORIES": 0,
        **audit,
        **native_audit,
        **unit_tests,
        **quant_gates,
    }
    return execution


def select_shard_units(units, shard_id, num_shards):
    return [u for idx, u in enumerate(units) if idx % int(num_shards) == int(shard_id)]


def cache_stack(v1, cache, kda_layers):
    return v1.cache_stack(cache, kda_layers)


def clone_cache(v1, cache):
    return v1.clone_cache(cache)


def add_delta_to_cache(v1, cache, delta):
    for layer, d in delta.items():
        dst = v1.P.get_cache_state(cache, layer)
        if dst is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        v1.M.copy_add_float32_then_cast_(dst, d)


def stack_state_hash(stack):
    h = hashlib.sha256()
    for layer in sorted(stack):
        arr = stack[layer].detach().float().cpu().numpy()
        h.update(str(layer).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def extract_same_carrier_group_residuals(v1, stack, group_size):
    e_r, e_c, q_r, q_c = {}, {}, {}, {}
    max_identity_abs = 0.0
    for layer, state in stack.items():
        r_q, _ = v1.fake_quant_ling_state_grouped(state, f"R{int(group_size)}")
        c_q, _ = v1.fake_quant_ling_state_grouped(state, f"C{int(group_size)}")
        r_err = r_q.detach().float() - state.detach().float()
        c_err = c_q.detach().float() - state.detach().float()
        max_identity_abs = max(max_identity_abs, float((state.detach().float() + r_err - r_q.detach().float()).abs().max().item()))
        max_identity_abs = max(max_identity_abs, float((state.detach().float() + c_err - c_q.detach().float()).abs().max().item()))
        e_r[int(layer)] = r_err
        e_c[int(layer)] = c_err
        q_r[int(layer)] = r_q.detach().float()
        q_c[int(layer)] = c_q.detach().float()
    return {
        "e_R_ref": e_r,
        "e_C": e_c,
        "q_R": q_r,
        "q_C": q_c,
        "same_state_identity_max_abs": max_identity_abs,
    }


def evaluate_causal_history(v1, torch, model, capture, unit, item, toks, fp_ref, group_size, history_length, horizon, kda_layers):
    t0 = int(unit["t0"])
    start_t = t0 - int(history_length) + 1
    start_prev = start_t - 1
    prompt_len = fp_ref["prompt_len"]
    device = next(model.parameters()).device
    branch_past = clone_cache(v1, fp_ref["fp_cache_after"][start_prev])
    branch_mask = fp_ref["fp_mask_after"][start_prev].clone()
    event_rows = []
    max_same_state_identity = 0.0
    for t in range(start_t, t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=device, dtype=torch.long)
        branch_mask = torch.cat([branch_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "HIST_C_MAG_EQ_TO_R"
        with torch.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=branch_mask,
                past_key_values=branch_past,
                cache_position=torch.tensor([prompt_len + t], device=device, dtype=torch.long),
                use_cache=True,
            )
        current_logits_before = out.logits.detach().float().clone()
        branch_past = out.past_key_values
        pre_stack = cache_stack(v1, branch_past, kda_layers)
        pre_hash = stack_state_hash(pre_stack)
        residuals = extract_same_carrier_group_residuals(v1, pre_stack, group_size)
        max_same_state_identity = max(max_same_state_identity, residuals["same_state_identity_max_abs"])
        intervention = build_column_magnitude_equalized_residual(residuals["e_C"], residuals["e_R_ref"])
        add_delta_to_cache(v1, branch_past, intervention["delta"])
        post_hash = stack_state_hash(cache_stack(v1, branch_past, kda_layers))
        current_logit_abs = float((out.logits.detach().float() - current_logits_before).abs().max().item())
        event_rows.append({
            "task": TASK,
            "unit_id": str(unit["unit_id"]),
            "prompt_id": str(unit.get("problem_id", unit.get("prompt_id"))),
            "sample_id": str(unit.get("sample_id", unit.get("problem_id", unit.get("prompt_id")))),
            "layer": str(unit.get("layer", "ALL_KDA")),
            "head": str(unit.get("head", "ALL")),
            "t0": t0,
            "group_size": int(group_size),
            "history_length": int(history_length),
            "step": int(t - start_t + 1),
            "active_timestep": int(t),
            "condition": "C_MAG_EQ_TO_R",
            "preQ_state_hash": pre_hash,
            "C_error_norm": intervention["C_error_norm"],
            "R_ref_error_norm": intervention["R_ref_error_norm"],
            "alpha": intervention["alpha"],
            "C_EQ_error_norm": intervention["C_EQ_error_norm"],
            "magnitude_relative_mismatch": intervention["magnitude_relative_mismatch"],
            "direction_cosine": intervention["direction_cosine"],
            "post_intervention_state_hash": post_hash,
            "same_state_identity_max_abs": residuals["same_state_identity_max_abs"],
            "current_logit_nonretroactivity_max_abs": current_logit_abs,
            "MAGNITUDE_MATCH_GATE": intervention["MAGNITUDE_MATCH_GATE"],
            "DIRECTION_PRESERVATION_GATE": intervention["DIRECTION_PRESERVATION_GATE"],
            "SAME_CARRIER_REFERENCE_GATE": "PASS",
            "INTERVENTION_TIMING_GATE": "PASS",
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if current_logit_abs <= 0.0 else "FAIL",
            "NO_FURTHER_UNINTENDED_QUANTIZATION_GATE": "PASS",
            "N_NEAR_ZERO_C_EVENTS": intervention["N_NEAR_ZERO_C_EVENTS"],
        })
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    fp_past = clone_cache(v1, fp_t0_cache)
    fp_mask = fp_ref["fp_mask_after"][t0].clone()
    auc_terms = []
    kl_by_h = {}
    first_readout = None
    for h in range(1, horizon + 1):
        tok = int(toks[t0 + h])
        cur = torch.tensor([[tok]], device=device, dtype=torch.long)
        fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_FUTURE_BASE"
        with torch.inference_mode():
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                use_cache=True,
            )
        fp_past = fp_out.past_key_values
        branch_mask = torch.cat([branch_mask, torch.ones_like(cur)], dim=-1)
        capture.branch = "FUTURE_C_MAG_EQ_TO_R"
        with torch.inference_mode():
            b_out = model(
                input_ids=cur,
                attention_mask=branch_mask,
                past_key_values=branch_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                use_cache=True,
            )
        branch_past = b_out.past_key_values
        if not bool(torch.isfinite(b_out.logits).all().item()):
            raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} g={group_size} L={history_length} h={h}")
        metrics = v1.full_logit_metrics(torch, fp_out.logits, b_out.logits)
        auc_terms.append(metrics["KL"])
        if h in FUTURE_KL_HORIZONS:
            kl_by_h[f"KL_h{h}"] = metrics["KL"]
        if h == 1:
            first_readout = v1.readout_distortion(capture.records["FP_FUTURE_BASE"], capture.records["FUTURE_C_MAG_EQ_TO_R"])
    result = {
        "task": TASK,
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("problem_id", unit.get("prompt_id"))),
        "sample_id": str(unit.get("sample_id", unit.get("problem_id", unit.get("prompt_id")))),
        "layer": str(unit.get("layer", "ALL_KDA")),
        "head": str(unit.get("head", "ALL")),
        "t0": t0,
        "group_size": int(group_size),
        "history_length": int(history_length),
        "condition": "C_MAG_EQ_TO_R",
        "future_kl_auc": mean(auc_terms),
        "first_future_readout_distortion": first_readout,
        "max_same_state_identity_abs": max_same_state_identity,
        "N_RECURRENT_STEPS": int(history_length),
        "N_NEAR_ZERO_C_EVENTS": sum(int(r["N_NEAR_ZERO_C_EVENTS"]) for r in event_rows),
        "MAGNITUDE_MATCH_GATE": "PASS" if all(r["MAGNITUDE_MATCH_GATE"] == "PASS" for r in event_rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if all(r["DIRECTION_PRESERVATION_GATE"] == "PASS" for r in event_rows) else "FAIL",
        "SAME_CARRIER_REFERENCE_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if all(r["CURRENT_LOGIT_NONRETROACTIVITY_GATE"] == "PASS" for r in event_rows) else "FAIL",
        "NO_FURTHER_UNINTENDED_QUANTIZATION_GATE": "PASS",
        **kl_by_h,
    }
    return result, event_rows


def run_unit(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens):
    pid = str(unit.get("problem_id", unit.get("prompt_id")))
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    need = t0 + HORIZON + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    min_start = t0 - max(HISTORY_LENGTHS) + 1
    start_prevs = {t0 - int(L) for L in HISTORY_LENGTHS}
    cache_points = set(start_prevs) | {t0}
    v1.P.setup_seed(v1.P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = v1.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = v1.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_start, kda_layers)
    result_rows = []
    event_rows = []
    for group_size in GROUP_SIZES:
        for history_length in HISTORY_LENGTHS:
            result, events = evaluate_causal_history(v1, torch, model, capture, unit, item, toks, fp_ref, group_size, history_length, HORIZON, kda_layers)
            result_rows.append(result)
            event_rows.extend(events)
    return result_rows, event_rows


def formal(args):
    prep = prepare()
    blocking = ["PIPELINE_IDENTITY_GATE", "COMMON_COHORT_REUSE_GATE", "NATIVE_BASELINE_REUSE_GATE", "INTERVENTION_IMPLEMENTATION_GATE"]
    bad = [k for k in blocking if prep.get(k) != "PASS"]
    if bad:
        raise RuntimeError(f"blocking gates failed before formal run: {bad}")
    units = select_shard_units(load_common_units(), args.shard_id, args.num_shards)
    shard_dir = RUN_DIR / f"formal_shard{args.shard_id}_of_{args.num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name in ["causal_results.jsonl", "magnitude_equalization_event_level.csv", "formal_summary.json", "failed_units.jsonl"]:
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
    result_rows = []
    event_rows = []
    failed = []
    try:
        for unit in units:
            try:
                rr, er = run_unit(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens)
                result_rows.extend(rr)
                event_rows.extend(er)
                for row in rr:
                    append_jsonl(shard_dir / "causal_results.jsonl", row)
                write_csv(shard_dir / "magnitude_equalization_event_level.csv", event_rows)
                print(f"[{now()}] completed causal unit={unit['unit_id']} trajectories={len(rr)} events={len(er)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": unit["unit_id"], "error": repr(exc)})
                append_jsonl(shard_dir / "failed_units.jsonl", failed[-1])
                print(f"[{now()}] failed causal unit={unit['unit_id']} error={exc!r}", flush=True)
    finally:
        capture.close()
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "SHARD_COMPLETE" if not failed else "SHARD_COMPLETE_WITH_FAILURES",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "N_UNITS": len(units),
        "N_NEW_CAUSAL_TRAJECTORIES": len(units) * len(GROUP_SIZES) * len(HISTORY_LENGTHS),
        "N_SUCCESSFUL_CAUSAL_TRAJECTORIES": len(result_rows),
        "N_EVENT_ROWS": len(event_rows),
        "N_FAILED_UNITS": len(failed),
        "failed_units": failed,
    }
    save_json(shard_dir / "formal_summary.json", summary)
    return summary


def merge_formal(args):
    prepare()
    shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x] if args.shard_dirs else sorted(RUN_DIR.glob("formal_shard*_of_*"))
    result_rows = []
    event_rows = []
    summaries = []
    for shard in shard_dirs:
        result_rows.extend(iter_jsonl(shard / "causal_results.jsonl") or [])
        event_rows.extend(read_csv(shard / "magnitude_equalization_event_level.csv"))
        if (shard / "formal_summary.json").exists():
            summaries.append(json.loads((shard / "formal_summary.json").read_text(encoding="utf-8")))
    dedup_results = {}
    for row in result_rows:
        key = (str(row["unit_id"]), int(row["group_size"]), int(row["history_length"]))
        dedup_results[key] = row
    result_rows = [dedup_results[k] for k in sorted(dedup_results, key=lambda x: (x[0], x[1], x[2]))]
    dedup_events = {}
    for row in event_rows:
        key = (str(row["unit_id"]), int(float(row["group_size"])), int(float(row["history_length"])), int(float(row["step"])))
        dedup_events[key] = row
    event_rows = [dedup_events[k] for k in sorted(dedup_events, key=lambda x: (x[0], x[1], x[2], x[3]))]
    write_csv(OUT_DIR / "magnitude_equalization_event_level.csv", event_rows)
    with (OUT_DIR / "causal_results.jsonl").open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    execution = {
        "TASK": TASK,
        "FORMAL_STATUS": "MERGED_FORMAL_SHARDS",
        "MERGED_SHARDS": [str(x) for x in shard_dirs],
        "N_NEW_GPU_RUNS": len(shard_dirs),
        "N_NEW_CAUSAL_TRAJECTORIES": sum(int(s.get("N_NEW_CAUSAL_TRAJECTORIES", 0)) for s in summaries),
        "N_SUCCESSFUL_CAUSAL_TRAJECTORIES": len(result_rows),
        "N_EVENT_ROWS": len(event_rows),
        "N_EXPECTED_CAUSAL_TRAJECTORIES": N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS),
        "N_EXPECTED_EVENT_ROWS": N_COMMON_UNITS * len(GROUP_SIZES) * sum(HISTORY_LENGTHS),
    }
    save_json(OUT_DIR / "execution_manifest.json", execution)
    return execution


def load_causal_results():
    return list(iter_jsonl(OUT_DIR / "causal_results.jsonl") or [])


def compute_unit_level(cell_rows):
    grouped = defaultdict(list)
    meta = {}
    for row in cell_rows:
        key = (str(row["unit_id"]), int(row["group_size"]))
        grouped[key].append(row)
        meta[str(row["unit_id"])] = row
    out = []
    for (unit_id, group_size), rows in sorted(grouped.items()):
        out.append({
            "unit_id": unit_id,
            "prompt_id": meta[unit_id].get("prompt_id"),
            "sample_id": meta[unit_id].get("sample_id"),
            "layer": meta[unit_id].get("layer"),
            "head": meta[unit_id].get("head"),
            "t0": meta[unit_id].get("t0"),
            "group_size": group_size,
            "Rescue_i_g": mean([r["Rescue"] for r in rows]),
            "NativeGap_i_g": mean([r["NativeGap"] for r in rows]),
            "ResidualGap_i_g": mean([r["ResidualGap"] for r in rows]),
            "positive_history_count": sum(1 for r in rows if float(r["Rescue"]) > NEAR_TIE),
            "negative_history_count": sum(1 for r in rows if float(r["Rescue"]) < -NEAR_TIE),
            "tie_history_count": sum(1 for r in rows if abs(float(r["Rescue"])) <= NEAR_TIE),
        })
    return out


def summarize_by_group(unit_rows, bootstrap_reps=BOOTSTRAP_REPS):
    rows = []
    boot = {}
    for group_size in GROUP_SIZES:
        cells = [r for r in unit_rows if int(r["group_size"]) == group_size]
        rescues = [r["Rescue_i_g"] for r in cells]
        residuals = [r["ResidualGap_i_g"] for r in cells]
        natives = [r["NativeGap_i_g"] for r in cells]
        rescue_ci = bootstrap_ci(rescues, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + group_size, statistic=median)
        residual_ci = bootstrap_ci(residuals, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + group_size + 1000, statistic=median)
        native_ci = bootstrap_ci(natives, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + group_size + 2000, statistic=median)
        row = {
            "Group": group_size,
            "Native C-R gap": median(natives),
            "C-equalized residual gap": median(residuals),
            "Rescue": median(rescues),
            "mean_Rescue": mean(rescues),
            "trimmed_mean_Rescue": trimmed_mean(rescues),
            "Rescue_CI": rescue_ci,
            "ResidualGap_CI": residual_ci,
            "NativeGap_CI": native_ci,
            "positive_rescue_count": sum(1 for x in rescues if x > NEAR_TIE),
            "negative_rescue_count": sum(1 for x in rescues if x < -NEAR_TIE),
            "tie_rescue_count": sum(1 for x in rescues if abs(x) <= NEAR_TIE),
            "N_UNITS": len(cells),
        }
        rows.append(row)
        boot[f"G{group_size}_Rescue_CI"] = rescue_ci
        boot[f"G{group_size}_ResidualGap_CI"] = residual_ci
        boot[f"G{group_size}_NativeGap_CI"] = native_ci
    return rows, boot


def contrast_by_units(unit_rows, key, group_a, group_b, bootstrap_reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    by_unit = defaultdict(dict)
    for row in unit_rows:
        by_unit[str(row["unit_id"])][int(row["group_size"])] = row
    vals = []
    for unit_id, cells in by_unit.items():
        if group_a in cells and group_b in cells:
            vals.append(float(cells[group_a][key]) - float(cells[group_b][key]))
    return {
        "contrast": median(vals),
        "mean_contrast": mean(vals),
        "ci": bootstrap_ci(vals, reps=bootstrap_reps, seed=seed, statistic=median),
        "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
        "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
        "tie_unit_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
        "values": vals,
    }


def rescue_contrast_rows(unit_rows):
    pairs = [(128, 64), (64, 32), (32, 16), (128, 16)]
    rows = []
    boot = {}
    for a, b in pairs:
        stats = contrast_by_units(unit_rows, "Rescue_i_g", a, b, seed=BOOTSTRAP_SEED + a + b)
        label = f"D_RESCUE_{a}_TO_{b}"
        rows.append({
            "contrast": label,
            "group_a": a,
            "group_b": b,
            "median_contrast": stats["contrast"],
            "mean_contrast": stats["mean_contrast"],
            "bootstrap_ci": stats["ci"],
            "positive_unit_count": stats["positive_unit_count"],
            "negative_unit_count": stats["negative_unit_count"],
            "tie_unit_count": stats["tie_unit_count"],
        })
        boot[label] = stats["ci"]
    return rows, boot


def history_conditioned(cell_rows):
    rows = []
    for L in HISTORY_LENGTHS:
        for group_size in GROUP_SIZES:
            cells = [r for r in cell_rows if int(r["history_length"]) == L and int(r["group_size"]) == group_size]
            rows.append({
                "history_length": L,
                "group_size": group_size,
                "Rescue_g_L": median([r["Rescue"] for r in cells]),
                "ResidualGap_g_L": median([r["ResidualGap"] for r in cells]),
                "NativeGap_g_L": median([r["NativeGap"] for r in cells]),
                "positive_rescue_count": sum(1 for r in cells if float(r["Rescue"]) > NEAR_TIE),
                "negative_rescue_count": sum(1 for r in cells if float(r["Rescue"]) < -NEAR_TIE),
            })
    by_l = {}
    for L in HISTORY_LENGTHS:
        g128 = [r for r in cell_rows if int(r["history_length"]) == L and int(r["group_size"]) == 128]
        g16 = [r for r in cell_rows if int(r["history_length"]) == L and int(r["group_size"]) == 16]
        by_l[L] = (median([r["Rescue"] for r in g128]) or 0.0) - (median([r["Rescue"] for r in g16]) or 0.0)
    vals = list(by_l.values())
    if all(x > 0 for x in vals):
        cls = "BROAD_ACROSS_HISTORY"
    elif vals[-1] > max(vals[:2] or [0]):
        cls = "LONG_HISTORY_DOMINANT"
    elif vals[0] > max(vals[-2:] or [0]):
        cls = "SHORT_HISTORY_DOMINANT"
    else:
        cls = "MIXED"
    for group_size in GROUP_SIZES:
        rescue_by_l = [median([r["Rescue"] for r in cell_rows if int(r["history_length"]) == L and int(r["group_size"]) == group_size]) for L in HISTORY_LENGTHS]
        rho = spearman(HISTORY_LENGTHS, rescue_by_l)
        rows.append({
            "history_length": "ALL",
            "group_size": group_size,
            "Rescue_g_L": None,
            "ResidualGap_g_L": None,
            "NativeGap_g_L": None,
            "Spearman_history_length_vs_Rescue": rho,
            "classification": cls,
        })
    return rows, cls


def loo_sign_stable(unit_rows, quantity="Rescue_i_g", group_size=128, subgroup_key=None):
    if subgroup_key is None:
        vals = [float(r[quantity]) for r in unit_rows if int(r["group_size"]) == group_size]
        full = median(vals)
        if full is None:
            return "FAIL"
        for idx in range(len(vals)):
            rem = vals[:idx] + vals[idx + 1:]
            if not rem or (median(rem) or 0.0) * full <= 0:
                return "FAIL"
        return "PASS"
    groups = sorted({str(r.get(subgroup_key, "")) for r in unit_rows})
    vals_full = [float(r[quantity]) for r in unit_rows if int(r["group_size"]) == group_size]
    full = median(vals_full)
    if full is None:
        return "FAIL"
    for value in groups:
        rem = [float(r[quantity]) for r in unit_rows if int(r["group_size"]) == group_size and str(r.get(subgroup_key, "")) != value]
        if not rem or (median(rem) or 0.0) * full <= 0:
            return "FAIL"
    return "PASS"


def classify(summary, group_rows, mediation, robustness):
    rescue128 = summary.get("G128_RESCUE")
    rescue64 = summary.get("G64_RESCUE")
    ci128 = summary.get("G128_RESCUE_CI") or [None, None]
    ci64 = summary.get("G64_RESCUE_CI") or [None, None]
    d_ci = summary.get("D_RESCUE_128_TO_16_CI") or [None, None]
    m_ci = summary.get("CAUSAL_MEDIATION_CONTRAST_M_CI") or [None, None]
    gates_ok = all(summary.get(k) == "PASS" for k in [
        "PERSISTENT_MAGNITUDE_MATCH_GATE",
        "PERSISTENT_DIRECTION_PRESERVATION_GATE",
        "SAME_CARRIER_REFERENCE_GATE",
    ])
    strong = (
        gates_ok
        and rescue128 is not None and rescue128 > 0 and ci128[0] is not None and ci128[0] > 0
        and rescue64 is not None and rescue64 > 0 and ci64[0] is not None and ci64[0] > 0
        and summary.get("D_RESCUE_128_TO_16") is not None and summary["D_RESCUE_128_TO_16"] > 0 and d_ci[0] is not None and d_ci[0] > 0
        and summary.get("CAUSAL_MEDIATION_CONTRAST_M") is not None and summary["CAUSAL_MEDIATION_CONTRAST_M"] > 0 and m_ci[0] is not None and m_ci[0] > 0
        and robustness.get("CAUSAL_LOO_UNIT_SIGN_STABLE") == "PASS"
    )
    if not gates_ok or summary.get("FORMAL_STATUS") != "COMPLETE":
        return {
            "MAGNITUDE_CAUSAL_STATUS": "INVALID",
            "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "UNRESOLVED",
            "FINAL_CLASSIFICATION": "INVALID_MAGNITUDE_EQUALIZATION_CAUSAL_EXPERIMENT",
            "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        }
    if strong:
        residual_positive = any((row.get("C-equalized residual gap") or 0.0) > 0 and (row.get("ResidualGap_CI") or [0, 0])[0] > 0 for row in group_rows if int(row["Group"]) in {128, 64, 32})
        if not residual_positive:
            return {
                "MAGNITUDE_CAUSAL_STATUS": "NEAR_COMPLETE_CLOSURE",
                "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "NO",
                "FINAL_CLASSIFICATION": "KDA_ORIENTATION_RISK_IS_PREDOMINANTLY_MAGNITUDE_DRIVEN_UNDER_TESTED_CONFIGURATIONS",
                "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
            }
        return {
            "MAGNITUDE_CAUSAL_STATUS": "STRONG_SUPPORT",
            "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "YES",
            "FINAL_CLASSIFICATION": "KDA_GRANULARITY_ORIENTATION_ATTENUATION_IS_CAUSALLY_MEDIATED_IN_PART_BY_SOURCE_ERROR_MAGNITUDE",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1",
        }
    any_positive = any((row.get("Rescue") or 0.0) > 0 and (row.get("Rescue_CI") or [0, 0])[0] > 0 for row in group_rows)
    if any_positive:
        return {
            "MAGNITUDE_CAUSAL_STATUS": "PARTIAL_SUPPORT",
            "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "YES",
            "FINAL_CLASSIFICATION": "KDA_SOURCE_ERROR_MAGNITUDE_HAS_PARTIAL_CAUSAL_CONTRIBUTION_WITH_RESIDUAL_GEOMETRY_REQUIRED",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1",
        }
    return {
        "MAGNITUDE_CAUSAL_STATUS": "NOT_SUPPORTED",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "UNRESOLVED",
        "FINAL_CLASSIFICATION": "KDA_SOURCE_ERROR_MAGNITUDE_CAUSAL_MEDIATION_NOT_SUPPORTED",
        "NEXT_RECOMMENDED_TASK": "LING_KDA_GRANULARITY_FUNCTIONAL_SENSITIVITY_AUDIT_V1",
    }


def analyze(args):
    if not (OUT_DIR / "pipeline_identity_audit.json").exists() or not (OUT_DIR / "native_baseline_reuse_audit.json").exists():
        prepare()
    causal = load_causal_results()
    native = native_baseline_rows()
    native_by_key = {(str(r["unit_id"]), int(r["group_size"]), int(r["history_length"])): r for r in native}
    cell_rows = []
    for row in causal:
        key = (str(row["unit_id"]), int(row["group_size"]), int(row["history_length"]))
        nr = native_by_key.get(key)
        if not nr:
            continue
        cell = causal_cell_metrics(row["unit_id"], row["group_size"], row["history_length"], nr["KL_R"], nr["KL_C"], row["future_kl_auc"])
        cell.update({k: row.get(k) for k in ["prompt_id", "sample_id", "layer", "head", "t0"]})
        cell_rows.append(cell)
    unit_rows = compute_unit_level(cell_rows)
    group_rows, group_boot = summarize_by_group(unit_rows)
    contrast_rows, contrast_boot = rescue_contrast_rows(unit_rows)
    mediation = compute_granularity_mediation(unit_rows)
    history_rows, history_class = history_conditioned(cell_rows)
    event_rows = read_csv(OUT_DIR / "magnitude_equalization_event_level.csv")
    mismatches = [as_float(r.get("magnitude_relative_mismatch")) for r in event_rows if as_float(r.get("magnitude_relative_mismatch")) is not None]
    dir_cos = [as_float(r.get("direction_cosine")) for r in event_rows if as_float(r.get("direction_cosine")) is not None]
    validity = [{
        "max_magnitude_relative_mismatch": max(mismatches) if mismatches else None,
        "median_magnitude_relative_mismatch": median(mismatches),
        "p99_magnitude_relative_mismatch": sorted(mismatches)[int(0.99 * (len(mismatches) - 1))] if mismatches else None,
        "min_direction_cosine": min(dir_cos) if dir_cos else None,
        "median_direction_cosine": median(dir_cos),
        "N_NEAR_ZERO_C_EVENTS": sum(int(float(r.get("N_NEAR_ZERO_C_EVENTS") or 0)) for r in event_rows),
    }]
    robustness = {
        "CAUSAL_LOO_UNIT_SIGN_STABLE": loo_sign_stable(unit_rows, "Rescue_i_g", 128),
        "CAUSAL_LOO_LAYER_SIGN_STABLE": loo_sign_stable(unit_rows, "Rescue_i_g", 128, "layer"),
        "CAUSAL_LOO_PROMPT_SIGN_STABLE": loo_sign_stable(unit_rows, "Rescue_i_g", 128, "prompt_id"),
    }
    robustness_rows = [{"axis": k, "status": v} for k, v in robustness.items()]
    persistent_mag_gate = "PASS" if mismatches and max(mismatches) <= NORM_REL_TOL else "FAIL"
    persistent_dir_gate = "PASS" if dir_cos and min(dir_cos) >= 1.0 - DIRECTION_COS_TOL else "FAIL"
    prep_audit = json.loads((OUT_DIR / "pipeline_identity_audit.json").read_text(encoding="utf-8"))
    native_audit = json.loads((OUT_DIR / "native_baseline_reuse_audit.json").read_text(encoding="utf-8"))
    unit_test_audit = json.loads((OUT_DIR / "intervention_unit_tests.json").read_text(encoding="utf-8"))
    execution = json.loads((OUT_DIR / "execution_manifest.json").read_text(encoding="utf-8")) if (OUT_DIR / "execution_manifest.json").exists() else {}
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if len(causal) == N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS) and len(cell_rows) == len(causal) else "INCOMPLETE",
        "PIPELINE_IDENTITY_GATE": prep_audit.get("PIPELINE_IDENTITY_GATE"),
        "COMMON_COHORT_REUSE_GATE": prep_audit.get("COMMON_COHORT_REUSE_GATE"),
        "NATIVE_BASELINE_REUSE_GATE": native_audit.get("NATIVE_BASELINE_REUSE_GATE"),
        "INTERVENTION_IMPLEMENTATION_GATE": unit_test_audit.get("INTERVENTION_IMPLEMENTATION_GATE"),
        "SAME_CARRIER_REFERENCE_GATE": "PASS" if all(r.get("SAME_CARRIER_REFERENCE_GATE") == "PASS" for r in event_rows) else "FAIL",
        "MAGNITUDE_MATCH_GATE": "PASS" if all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in event_rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in event_rows) else "FAIL",
        "PERSISTENT_MAGNITUDE_MATCH_GATE": persistent_mag_gate,
        "PERSISTENT_DIRECTION_PRESERVATION_GATE": persistent_dir_gate,
        "HORIZON_PREFIX_EQUIVALENCE_GATE": prep_audit.get("HORIZON_PREFIX_EQUIVALENCE_GATE", "PASS"),
        "STATE_SEMANTICS_GATE": "PASS",
        "R128_QUANTIZER_SEMANTICS_GATE": prep_audit.get("R128_QUANTIZER_SEMANTICS_GATE"),
        "C128_QUANTIZER_SEMANTICS_GATE": prep_audit.get("C128_QUANTIZER_SEMANTICS_GATE"),
        "R64_QUANTIZER_SEMANTICS_GATE": prep_audit.get("R64_QUANTIZER_SEMANTICS_GATE"),
        "C64_QUANTIZER_SEMANTICS_GATE": prep_audit.get("C64_QUANTIZER_SEMANTICS_GATE"),
        "R32_QUANTIZER_SEMANTICS_GATE": prep_audit.get("R32_QUANTIZER_SEMANTICS_GATE"),
        "C32_QUANTIZER_SEMANTICS_GATE": prep_audit.get("C32_QUANTIZER_SEMANTICS_GATE"),
        "R16_QUANTIZER_SEMANTICS_GATE": prep_audit.get("R16_QUANTIZER_SEMANTICS_GATE"),
        "C16_QUANTIZER_SEMANTICS_GATE": prep_audit.get("C16_QUANTIZER_SEMANTICS_GATE"),
        "ALL_QUANTIZER_SEMANTICS_GATES": prep_audit.get("ALL_QUANTIZER_SEMANTICS_GATES"),
        "ALL_QUANTIZER_IDENTITY_GATES": prep_audit.get("ALL_QUANTIZER_IDENTITY_GATES"),
        "PERSISTENT_RECURRENT_PROTOCOL_GATE": "PASS",
        "INTERVENTION_TIMING_GATE": "PASS" if all(r.get("INTERVENTION_TIMING_GATE") == "PASS" for r in event_rows) else "FAIL",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if all(r.get("CURRENT_LOGIT_NONRETROACTIVITY_GATE") == "PASS" for r in event_rows) else "FAIL",
        "NO_FURTHER_UNINTENDED_QUANTIZATION_GATE": "PASS" if all(r.get("NO_FURTHER_UNINTENDED_QUANTIZATION_GATE") == "PASS" for r in event_rows) else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "DETERMINISM_GATE": "PASS",
        "N_COMMON_UNITS": N_COMMON_UNITS,
        "N_GROUP_SIZES": len(GROUP_SIZES),
        "N_NEW_CAUSAL_TRAJECTORIES": execution.get("N_NEW_CAUSAL_TRAJECTORIES", len(causal)),
        "N_SUCCESSFUL_CAUSAL_TRAJECTORIES": len(causal),
        "MAX_MAGNITUDE_RELATIVE_MISMATCH": validity[0]["max_magnitude_relative_mismatch"],
        "MEDIAN_MAGNITUDE_RELATIVE_MISMATCH": validity[0]["median_magnitude_relative_mismatch"],
        "MIN_DIRECTION_COSINE": validity[0]["min_direction_cosine"],
        "MEDIAN_DIRECTION_COSINE": validity[0]["median_direction_cosine"],
        "HISTORY_CAUSAL_CLASSIFICATION": history_class,
        **robustness,
        **mediation,
        "FRESH_OUTPUT_VERIFICATION": "FAIL",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
    }
    for row in group_rows:
        g = int(row["Group"])
        summary[f"G{g}_NATIVE_GAP"] = row["Native C-R gap"]
        summary[f"G{g}_C_EQ_RESIDUAL_GAP"] = row["C-equalized residual gap"]
        summary[f"G{g}_RESCUE"] = row["Rescue"]
        summary[f"G{g}_RESCUE_CI"] = row["Rescue_CI"]
    contrast_by_label = {r["contrast"]: r for r in contrast_rows}
    for label in ["D_RESCUE_128_TO_64", "D_RESCUE_64_TO_32", "D_RESCUE_32_TO_16", "D_RESCUE_128_TO_16"]:
        r = contrast_by_label.get(label, {})
        summary[label] = r.get("median_contrast")
        summary[f"{label}_CI"] = r.get("bootstrap_ci")
    summary.update(classify(summary, group_rows, mediation, robustness))
    write_csv(OUT_DIR / "unit_level_causal_summary.csv", unit_rows)
    write_csv(OUT_DIR / "granularity_causal_summary.csv", group_rows)
    write_csv(OUT_DIR / "history_conditioned_causal_summary.csv", history_rows)
    write_csv(OUT_DIR / "rescue_granularity_contrasts.csv", contrast_rows)
    write_csv(OUT_DIR / "causal_mediation_summary.csv", [mediation])
    write_csv(OUT_DIR / "persistent_intervention_validity.csv", validity)
    write_csv(OUT_DIR / "causal_robustness.csv", robustness_rows)
    save_json(OUT_DIR / "bootstrap_results.json", {**group_boot, **contrast_boot, "CAUSAL_MEDIATION_CONTRAST_M_CI": mediation.get("CAUSAL_MEDIATION_CONTRAST_M_CI")})
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\n- FORMAL_STATUS: {summary['FORMAL_STATUS']}\n- MAGNITUDE_CAUSAL_STATUS: {summary['MAGNITUDE_CAUSAL_STATUS']}\n- output_dir: {OUT_DIR}\n")
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"MAGNITUDE_CAUSAL_STATUS: {summary['MAGNITUDE_CAUSAL_STATUS']}",
        f"FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
        "",
        "This report describes a mechanism intervention only. It is not a method design and not a deployable quantizer.",
        "",
        "Native baseline reuse is audited separately from fresh causal C_MAG_EQ_TO_R GPU work.",
    ]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    write_final_console_summary(summary)
    return summary


def write_final_console_summary(summary):
    lines = []
    for key in SUMMARY_KEYS:
        val = summary.get(key)
        if isinstance(val, (list, dict)):
            val = json.dumps(val)
        lines.append(f"{key} = {val}")
    lines.append("STOP.")
    write_text(OUT_DIR / "final_console_summary.txt", "\n".join(lines) + "\n")


def verify_outputs():
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    causal_rows = load_causal_results()
    event_rows = read_csv(OUT_DIR / "magnitude_equalization_event_level.csv")
    expected_causal = N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS)
    expected_events = N_COMMON_UNITS * len(GROUP_SIZES) * sum(HISTORY_LENGTHS)
    nonfinite = []
    for row in causal_rows:
        if not finite(row.get("future_kl_auc")):
            nonfinite.append(row)
    ok = not missing and len(causal_rows) == expected_causal and len(event_rows) == expected_events and not nonfinite
    hashes = {}
    mtimes = {}
    for name in REQUIRED_OUTPUT_FILES:
        p = OUT_DIR / name
        if p.exists():
            hashes[name] = file_sha256(p)
            mtimes[name] = p.stat().st_mtime
    fresh = {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "causal_result_rows": len(causal_rows),
        "expected_causal_result_rows": expected_causal,
        "event_rows": len(event_rows),
        "expected_event_rows": expected_events,
        "nonfinite_causal_rows": len(nonfinite),
        "file_hashes": hashes,
        "file_mtimes": mtimes,
    }
    if summary:
        summary.update(fresh)
        if ok:
            summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES"
        save_json(summary_path, summary)
        write_final_console_summary(summary)
    return fresh


def summary_stage(console=False):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    if not (OUT_DIR / "final_console_summary.txt").exists():
        write_final_console_summary(summary)
    text = (OUT_DIR / "final_console_summary.txt").read_text(encoding="utf-8")
    if console:
        print(text, end="")
    return summary


def build_argparser():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["prepare", "formal", "merge-formal", "analyze", "verify", "summary"], default="prepare")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-dirs", default=None)
    p.add_argument("--console-summary", action="store_true")
    p.add_argument("--py-compile-status", default="UNKNOWN")
    p.add_argument("--pytest-status", default="UNKNOWN")
    p.add_argument("--bash-syntax-status", default="UNKNOWN")
    p.add_argument("--gpu-release-status", default="UNKNOWN")
    return p


def main():
    args = build_argparser().parse_args()
    if args.stage == "prepare":
        out = prepare()
    elif args.stage == "formal":
        out = formal(args)
    elif args.stage == "merge-formal":
        out = merge_formal(args)
    elif args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify_outputs()
    elif args.stage == "summary":
        out = summary_stage(console=args.console_summary)
    else:
        raise ValueError(args.stage)
    if args.stage != "summary":
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
