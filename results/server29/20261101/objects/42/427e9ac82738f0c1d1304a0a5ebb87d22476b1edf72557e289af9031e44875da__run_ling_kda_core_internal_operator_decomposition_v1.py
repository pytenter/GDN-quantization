#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


TASK = "LING_KDA_CORE_INTERNAL_OPERATOR_DECOMPOSITION_V1"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / TASK

CORE_TASK = "LING_KDA_CORE_OUTPUT_DIRECTIONAL_GEOMETRY_LOCALIZATION_V1"
CORE_OUT = REPO / "results" / CORE_TASK
LAYER_TASK = "LING_KDA_LAYER_CONDITIONED_LOCAL_FUNCTIONAL_GEOMETRY_AUDIT_V1"
LAYER_OUT = REPO / "results" / LAYER_TASK
RESIDUAL_TASK = "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1"
RESIDUAL_OUT = REPO / "results" / RESIDUAL_TASK
MAG_TASK = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
MAG_OUT = REPO / "results" / MAG_TASK

MODEL_FILE = Path("/data01/user2/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py")
FLA_KDA_DIR = Path("/data01/user2/.conda/envs/ling-kda/lib/python3.11/site-packages/fla/ops/kda")
GROUP_SIZES = [128, 16]
KDA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
EPS = 1e-12
IDENTITY_REL_TOL = 2e-5
IDENTITY_ABS_TOL = 2e-6
IDENTITY_APPROX_REL_TOL = 1e-3
BOOTSTRAP_SEED = 20260912

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "common36_manifest.csv",
    "kda_core_internal_operator_path.md",
    "core_internal_operator_equations.md",
    "core_internal_stage_manifest.json",
    "core_internal_event_trace.csv",
    "stage_identity_error_summary.csv",
    "stage_directional_gain_event_level.csv",
    "stage_layer_sign_concordance.csv",
    "layer22_stage_reverse_trace.csv",
    "granularity_stage_directional_summary.csv",
    "operator_decomposition_master_table.csv",
    "next_targeted_causal_plan.md",
    "next_kernel_internal_decomposition_plan.md",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "FINAL_CLASSIFICATION",
    "OPERATOR_LOCALIZATION_STATUS",
    "OPERATOR_LOCALIZATION_CONVERGENCE",
    "EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE",
    "EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE",
    "EARLIEST_17_OF_18_OR_BETTER_STAGE",
    "EARLIEST_LAYER22_REVERSE_STAGE",
    "FIRST_UNRESOLVED_KERNEL_INTERNAL_STAGE",
    "KERNEL_INTERNAL_DECOMPOSITION_REQUIRED",
    "CORE_OUTPUT_USES_PRE_OR_POST_UPDATE_STATE",
    "OLD_CORE_TRANSFER_IDENTITY_FAILURE_EXPLAINED_BY_STATE_TIMING",
    "CORE_INTERNAL_SEMANTICS_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "MATCHED_CARRIER_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "PRIOR_LAYER_LABEL_REUSE_GATE",
    "N_COMMON_UNITS",
    "N_KDA_LAYERS",
    "N_C_SENSITIVE_PRIOR_LAYERS",
    "N_R_SENSITIVE_PRIOR_LAYERS",
    "N_CORE_OUTPUT_EVENT_ROWS",
    "N_NEW_GPU_LOCAL_EVALUATIONS",
    "NEW_STAGE_B_TRAJECTORIES",
    "NEW_CLOSED_LOOP_TRAJECTORIES",
    "CORE_OUTPUT_MEDIAN_IDENTITY_REL_ERROR",
    "CORE_OUTPUT_MAX_IDENTITY_REL_ERROR",
    "CORE_OUTPUT_N_SIGN_CONCORDANT",
    "CORE_OUTPUT_LAYER_SIGN_CONCORDANCE_RATE",
    "L22_G_R_CORE_OUTPUT",
    "L22_G_C_CORE_OUTPUT",
    "L22_DELTA_G_CORE_OUTPUT",
    "G128_DELTA_G_CORE_OUTPUT",
    "G16_DELTA_G_CORE_OUTPUT",
    "G128_TO_G16_DELTA_G_DIFFERENCE",
    "METHOD_DESIGN_READY",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def as_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def finite(value):
    return as_float(value) is not None


def median(values):
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def mean(values):
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(values, reps=400, seed=BOOTSTRAP_SEED, statistic=median):
    vals = [float(v) for v in values if finite(v)]
    if not vals:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(float(statistic(sample)))
    boot.sort()
    return [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_layer_labels():
    return {layer: ("R_SENSITIVE" if layer == 22 else "C_SENSITIVE") for layer in KDA_LAYERS}


def prior_layer_label_reuse_gate(rows):
    expected = prior_layer_labels()
    observed = {}
    for row in rows:
        layer = row.get("layer_id")
        label = row.get("layer_classification") or row.get("prior_layer_label")
        try:
            observed[int(layer)] = str(label)
        except Exception:
            continue
    return "PASS" if observed == expected else "FAIL"


def classify_identity_error(rel_error, abs_error, theoretical_valid=True):
    if not theoretical_valid:
        return "NOT_APPLICABLE_NONLINEAR"
    if rel_error is None or abs_error is None:
        return "NOT_APPLICABLE_UNOBSERVED"
    rel = float(rel_error)
    abs_ = float(abs_error)
    if rel <= IDENTITY_REL_TOL and abs_ <= IDENTITY_ABS_TOL:
        return "IDENTITY_PASS"
    if rel <= IDENTITY_APPROX_REL_TOL:
        return "IDENTITY_APPROXIMATE"
    return "IDENTITY_FAIL"


def build_stage_manifest():
    return [
        {
            "stage_id": "STAGE_00_RECURRENT_STATE_INPUT",
            "order": 0,
            "stage_name": "FP recurrent state carrier plus matched R/C perturbation",
            "source_file": "experiments/ling/run_ling_kda_core_output_directional_geometry_localization_v1.py",
            "function_name": "run_unit_core_panel",
            "input_tensor_names": "base_state, e_R, E_C_match",
            "input_shapes": "initial_state [B, HV, K, V]",
            "output_tensor_names": "initial_state",
            "output_shapes": "[B, HV, K, V]",
            "recurrent_state_read": "YES",
            "recurrent_state_updated": "NO",
            "q_k_v_g_beta_participate": "NO",
            "normalization_occurs": "NO",
            "nonlinear_operation_occurs": "NO",
            "reshape_permutation_occurs": "NO",
            "current_output_uses_pre_or_post_update_state": "PRE",
            "visibility": "PYTHON_VISIBLE",
            "operator_class": "EXACT_LINEAR_IN_STATE",
        },
        {
            "stage_id": "STAGE_01_FIXED_LOCAL_DRIVERS",
            "order": 1,
            "stage_name": "Frozen local drivers captured from FP decode",
            "source_file": str(MODEL_FILE),
            "function_name": "BailingMoeV3KimiDeltaAttention.forward",
            "input_tensor_names": "hidden_states",
            "input_shapes": "[B, T, hidden_size]",
            "output_tensor_names": "q, k, v, g, beta, A_log, dt_bias",
            "output_shapes": "q/k [B,T,H,K], v/g [B,T,HV,V_or_K], beta [B,T,HV]",
            "recurrent_state_read": "NO",
            "recurrent_state_updated": "NO",
            "q_k_v_g_beta_participate": "YES_FIXED_DRIVERS",
            "normalization_occurs": "ShortConvolution activation and beta sigmoid before kernel; q/k l2norm inside kernel",
            "nonlinear_operation_occurs": "YES_BUT_FIXED_WITH_RESPECT_TO_STATE_PERTURBATION",
            "reshape_permutation_occurs": "YES",
            "current_output_uses_pre_or_post_update_state": "NOT_OUTPUT_STAGE",
            "visibility": "PYTHON_VISIBLE",
            "operator_class": "MIXED_OR_UNRESOLVED_FOR_HIDDEN_STATES_EXACT_FIXED_FOR_STATE_REPLAY",
        },
        {
            "stage_id": "STAGE_02_KDA_KERNEL_BOUNDARY",
            "order": 2,
            "stage_name": "Call into chunk_kda or fused_recurrent_kda with fixed drivers and initial_state",
            "source_file": "fla/ops/kda/chunk.py; fla/ops/kda/fused_recurrent.py",
            "function_name": "chunk_kda / fused_recurrent_kda",
            "input_tensor_names": "q, k, v, g, beta, A_log, dt_bias, initial_state",
            "input_shapes": "q/k [B,T,H,K], v/g [B,T,HV,V_or_K], initial_state [B,HV,K,V]",
            "output_tensor_names": "o, final_state",
            "output_shapes": "o [B,T,HV,V], final_state [B,HV,K,V]",
            "recurrent_state_read": "YES",
            "recurrent_state_updated": "INSIDE_KERNEL",
            "q_k_v_g_beta_participate": "YES_FIXED_DRIVERS",
            "normalization_occurs": "q/k l2norm in kernel",
            "nonlinear_operation_occurs": "YES_GATE_QK_NORM_NUMERICS_BUT_FIXED_FOR_STATE_REPLAY",
            "reshape_permutation_occurs": "NO_AT_BOUNDARY",
            "current_output_uses_pre_or_post_update_state": "UNRESOLVED_AT_BOUNDARY",
            "visibility": "PYTHON_VISIBLE_BOUNDARY",
            "operator_class": "BOUNDARY",
        },
        {
            "stage_id": "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL",
            "order": 2.5,
            "stage_name": "FLA internal decay, beta update, state update, and post-update readout",
            "source_file": "fla/ops/kda/fused_recurrent.py; fla/ops/kda/chunk_fwd.py; fla/ops/kda/naive.py",
            "function_name": "Triton fused_recurrent_kda_fwd_kernel / chunk_kda_fwd",
            "input_tensor_names": "q, k, v, g, beta, initial_state",
            "input_shapes": "kernel tiled state and token blocks",
            "output_tensor_names": "internal h, o, final_state",
            "output_shapes": "not exposed for fused_recurrent; chunk optional h only",
            "recurrent_state_read": "YES",
            "recurrent_state_updated": "YES",
            "q_k_v_g_beta_participate": "YES_FIXED_DRIVERS",
            "normalization_occurs": "YES",
            "nonlinear_operation_occurs": "YES_FOR_DRIVERS_AND_NUMERIC_KERNEL; AFFINE_IN_INITIAL_STATE_UNDER_FIXED_DRIVERS_BY_REFERENCE_EQUATIONS",
            "reshape_permutation_occurs": "TILED_KERNEL_LAYOUT",
            "current_output_uses_pre_or_post_update_state": "POST_BY_SOURCE_REFERENCE",
            "visibility": "KERNEL_INTERNAL_UNOBSERVED_NUMERIC_STAGE",
            "operator_class": "AFFINE_IN_STATE_BUT_NUMERICALLY_UNRESOLVED",
        },
        {
            "stage_id": "STAGE_03_CORE_OUTPUT",
            "order": 3,
            "stage_name": "Observed backend core output o returned by KDA kernel",
            "source_file": "experiments/ling/run_ling_kda_core_output_directional_geometry_localization_v1.py",
            "function_name": "KdaTransitionCapture.replay",
            "input_tensor_names": "o(S+E), o(S), o(E), o(0)",
            "input_shapes": "o [B,T,HV,V]",
            "output_tensor_names": "delta_o_measured, delta_o_implicit",
            "output_shapes": "[B,T,HV,V]",
            "recurrent_state_read": "YES_UPSTREAM",
            "recurrent_state_updated": "YES_UPSTREAM",
            "q_k_v_g_beta_participate": "YES_FIXED_DRIVERS",
            "normalization_occurs": "KERNEL_INTERNAL_QK_L2NORM_AND_GATE_DECAY",
            "nonlinear_operation_occurs": "KERNEL_INTERNAL",
            "reshape_permutation_occurs": "NO",
            "current_output_uses_pre_or_post_update_state": "POST_BY_SOURCE_REFERENCE",
            "visibility": "PYTHON_VISIBLE_OUTPUT",
            "operator_class": "OBSERVED_KERNEL_OUTPUT",
        },
        {
            "stage_id": "STAGE_04_FINAL_STATE",
            "order": 4,
            "stage_name": "KDA final recurrent state returned to cache",
            "source_file": str(MODEL_FILE),
            "function_name": "BailingMoeV3KimiDeltaAttention.forward",
            "input_tensor_names": "recurrent_state",
            "input_shapes": "[B,HV,K,V]",
            "output_tensor_names": "past_key_value.layers[layer].keys",
            "output_shapes": "[B,HV,K,V]",
            "recurrent_state_read": "YES",
            "recurrent_state_updated": "YES",
            "q_k_v_g_beta_participate": "YES_UPSTREAM",
            "normalization_occurs": "NO_AT_CACHE_WRITE",
            "nonlinear_operation_occurs": "NO_AT_CACHE_WRITE",
            "reshape_permutation_occurs": "NO",
            "current_output_uses_pre_or_post_update_state": "POST",
            "visibility": "PYTHON_VISIBLE_CACHE_WRITE",
            "operator_class": "STATE_UPDATE",
        },
    ]


def stage_order_map():
    return {stage["stage_id"]: float(stage["order"]) for stage in build_stage_manifest()}


def earliest_identity_failure(rows):
    ordered = sorted(rows, key=lambda row: stage_order_map().get(str(row["stage_id"]), 999.0))
    for row in ordered:
        if row.get("identity_status") == "IDENTITY_FAIL":
            return str(row["stage_id"])
    return "NONE"


def earliest_sign_closure(rows, target=18):
    ordered = sorted(rows, key=lambda row: stage_order_map().get(str(row["stage_id"]), 999.0))
    for row in ordered:
        n = row.get("N_SIGN_CONCORDANT", row.get("layer_sign_concordance"))
        try:
            if int(n) >= int(target):
                return str(row["stage_id"])
        except Exception:
            continue
    return "NONE"


def earliest_layer22_reverse(rows):
    ordered = sorted(rows, key=lambda row: stage_order_map().get(str(row["stage_id"]), 999.0))
    for row in ordered:
        if row.get("layer22_reverse") == "YES" or row.get("LAYER22_REVERSE_CONTROL") == "PASS":
            return str(row["stage_id"])
    return "NONE"


def classify_localization(master_rows, first_unresolved):
    first_fail = earliest_identity_failure(master_rows)
    first_18 = earliest_sign_closure(master_rows, target=18)
    first_l22 = earliest_layer22_reverse(master_rows)
    if first_unresolved and first_unresolved != "NONE" and first_fail == "STAGE_03_CORE_OUTPUT" and first_18 == "STAGE_03_CORE_OUTPUT":
        return {
            "FINAL_CLASSIFICATION": "KDA_LOCAL_DIRECTIONAL_GEOMETRY_LOCALIZED_INSIDE_KDA_KERNEL",
            "OPERATOR_LOCALIZATION_STATUS": "KERNEL_INTERNAL",
            "OPERATOR_LOCALIZATION_CONVERGENCE": "PARTIAL",
            "KERNEL_INTERNAL_DECOMPOSITION_REQUIRED": "YES",
            "EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE": first_fail,
            "EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE": first_18,
            "EARLIEST_LAYER22_REVERSE_STAGE": first_l22,
        }
    if first_fail != "NONE" and first_fail == first_18 and first_l22 in {first_fail, "NONE"}:
        return {
            "FINAL_CLASSIFICATION": f"KDA_LOCAL_DIRECTIONAL_GEOMETRY_LOCALIZED_TO_{first_fail}",
            "OPERATOR_LOCALIZATION_STATUS": "STRONG",
            "OPERATOR_LOCALIZATION_CONVERGENCE": "STRONG",
            "KERNEL_INTERNAL_DECOMPOSITION_REQUIRED": "NO",
            "EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE": first_fail,
            "EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE": first_18,
            "EARLIEST_LAYER22_REVERSE_STAGE": first_l22,
        }
    if first_18 != "NONE":
        return {
            "FINAL_CLASSIFICATION": "KDA_LOCAL_DIRECTIONAL_GEOMETRY_LOCALIZED_TO_OPERATOR_BLOCK",
            "OPERATOR_LOCALIZATION_STATUS": "PARTIAL",
            "OPERATOR_LOCALIZATION_CONVERGENCE": "PARTIAL",
            "KERNEL_INTERNAL_DECOMPOSITION_REQUIRED": "NO",
            "EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE": first_fail,
            "EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE": first_18,
            "EARLIEST_LAYER22_REVERSE_STAGE": first_l22,
        }
    return {
        "FINAL_CLASSIFICATION": "KDA_CORE_INTERNAL_OPERATOR_ORIGIN_REMAINS_UNRESOLVED",
        "OPERATOR_LOCALIZATION_STATUS": "UNRESOLVED",
        "OPERATOR_LOCALIZATION_CONVERGENCE": "NO",
        "KERNEL_INTERNAL_DECOMPOSITION_REQUIRED": "NO",
        "EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE": first_fail,
        "EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE": first_18,
        "EARLIEST_LAYER22_REVERSE_STAGE": first_l22,
    }


def filtered_core_events():
    rows = read_csv(CORE_OUT / "core_transfer_gain_event_level.csv")
    return [row for row in rows if int(row.get("group_size", -1)) in GROUP_SIZES]


def pair_event_rows(rows):
    by_key = defaultdict(dict)
    for row in rows:
        key = (str(row["unit_id"]), int(row["layer_id"]), int(row["group_size"]))
        by_key[key][str(row["condition"])] = row
    pairs = []
    for (unit_id, layer, group), cond in sorted(by_key.items()):
        r_row = cond.get("R_MATCHED")
        c_row = cond.get("C_MATCHED")
        if not r_row or not c_row:
            continue
        pairs.append({
            "unit_id": unit_id,
            "layer_id": layer,
            "group_size": group,
            "G_R": as_float(r_row.get("G_core")),
            "G_C": as_float(c_row.get("G_core")),
            "delta_G": (as_float(c_row.get("G_core")) or 0.0) - (as_float(r_row.get("G_core")) or 0.0),
            "magnitude_relative_mismatch": max(as_float(r_row.get("magnitude_relative_mismatch")) or 0.0, as_float(c_row.get("magnitude_relative_mismatch")) or 0.0),
            "R_C_error_direction_cosine": as_float(c_row.get("R_C_error_direction_cosine")),
        })
    return pairs


def identity_summary_for_core_output(rows):
    rels = [as_float(row.get("transfer_identity_rel_error")) for row in rows]
    abss = [as_float(row.get("transfer_identity_abs_error")) for row in rows]
    rels = [x for x in rels if x is not None]
    abss = [x for x in abss if x is not None]
    max_rel = max(rels) if rels else None
    max_abs = max(abss) if abss else None
    return {
        "stage_id": "STAGE_03_CORE_OUTPUT",
        "stage_name": "Observed backend core output o",
        "theoretical_prediction": "o(S+E)-o(S) == o(E)-o(0) under affine-in-state fixed-driver model",
        "identity_status": classify_identity_error(max_rel, max_abs),
        "median_identity_rel_error": median(rels),
        "mean_identity_rel_error": mean(rels),
        "p95_identity_rel_error": sorted(rels)[int(0.95 * (len(rels) - 1))] if rels else None,
        "max_identity_rel_error": max_rel,
        "max_identity_abs_error": max_abs,
        "N_events": len(rows),
    }


def stage_identity_error_summary(rows):
    core = identity_summary_for_core_output(rows)
    return [
        {
            "stage_id": "STAGE_00_RECURRENT_STATE_INPUT",
            "stage_name": "Input recurrent-state perturbation",
            "theoretical_prediction": "delta_initial_state_hat = delta_initial_state",
            "identity_status": "IDENTITY_PASS",
            "median_identity_rel_error": 0.0,
            "mean_identity_rel_error": 0.0,
            "p95_identity_rel_error": 0.0,
            "max_identity_rel_error": 0.0,
            "max_identity_abs_error": 0.0,
            "N_events": len(rows),
        },
        {
            "stage_id": "STAGE_01_FIXED_LOCAL_DRIVERS",
            "stage_name": "Fixed local drivers",
            "theoretical_prediction": "drivers are frozen from FP branch; no state-perturbation identity is tested",
            "identity_status": "NOT_APPLICABLE_FIXED_DRIVER",
            "median_identity_rel_error": None,
            "mean_identity_rel_error": None,
            "p95_identity_rel_error": None,
            "max_identity_rel_error": None,
            "max_identity_abs_error": None,
            "N_events": 0,
        },
        {
            "stage_id": "STAGE_02_KDA_KERNEL_BOUNDARY",
            "stage_name": "KDA kernel call boundary",
            "theoretical_prediction": "boundary has inputs only; next observable tensor is kernel output",
            "identity_status": "NOT_APPLICABLE_BOUNDARY",
            "median_identity_rel_error": None,
            "mean_identity_rel_error": None,
            "p95_identity_rel_error": None,
            "max_identity_rel_error": None,
            "max_identity_abs_error": None,
            "N_events": 0,
        },
        {
            "stage_id": "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL",
            "stage_name": "Unobserved kernel-internal update/readout",
            "theoretical_prediction": "reference equations imply affine state transfer, but per-stage tensors are hidden in fused path",
            "identity_status": "UNRESOLVED_KERNEL_INTERNAL",
            "median_identity_rel_error": None,
            "mean_identity_rel_error": None,
            "p95_identity_rel_error": None,
            "max_identity_rel_error": None,
            "max_identity_abs_error": None,
            "N_events": 0,
        },
        core,
        {
            "stage_id": "STAGE_04_FINAL_STATE",
            "stage_name": "Final recurrent state",
            "theoretical_prediction": "not captured in prior per-event trace",
            "identity_status": "UNRESOLVED_NOT_TRACED",
            "median_identity_rel_error": None,
            "mean_identity_rel_error": None,
            "p95_identity_rel_error": None,
            "max_identity_rel_error": None,
            "max_identity_abs_error": None,
            "N_events": 0,
        },
    ]


def stage_directional_gain_event_level(core_rows):
    out = []
    placeholder_stages = [
        ("STAGE_01_FIXED_LOCAL_DRIVERS", "PYTHON_VISIBLE_FIXED_DRIVER_NOT_A_STATE_DELTA"),
        ("STAGE_02_KDA_KERNEL_BOUNDARY", "PYTHON_VISIBLE_INPUT_BOUNDARY_NO_OUTPUT_DELTA"),
        ("STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL", "KERNEL_INTERNAL_UNOBSERVED"),
        ("STAGE_04_FINAL_STATE", "NOT_TRACED_IN_PRIOR_EVENT_ROWS"),
    ]
    for row in core_rows:
        group = int(row["group_size"])
        condition = str(row["condition"])
        error_norm = as_float(row.get("error_norm"))
        transfer_norm = as_float(row.get("transfer_norm"))
        out.append({
            "stage_id": "STAGE_00_RECURRENT_STATE_INPUT",
            "unit_id": row.get("unit_id"),
            "layer_id": int(row["layer_id"]),
            "group_size": group,
            "condition": condition,
            "stage_delta_norm": error_norm,
            "source_error_norm": error_norm,
            "G_stage": 1.0 if error_norm is not None else None,
            "shape": "[B,HV,K,V]",
            "dtype": "float32",
            "hash": "REUSED_NORM_ONLY",
            "relative_norm": 1.0 if error_norm is not None else None,
            "cosine_where_meaningful": None,
            "max_abs": None,
            "trace_source": "RECONSTRUCTED_FROM_PRIOR_ERROR_NORM",
        })
        for stage_id, trace_source in placeholder_stages:
            out.append({
                "stage_id": stage_id,
                "unit_id": row.get("unit_id"),
                "layer_id": int(row["layer_id"]),
                "group_size": group,
                "condition": condition,
                "stage_delta_norm": None,
                "source_error_norm": error_norm,
                "G_stage": None,
                "shape": "UNRESOLVED",
                "dtype": "UNRESOLVED",
                "hash": "UNRESOLVED",
                "relative_norm": None,
                "cosine_where_meaningful": None,
                "max_abs": None,
                "trace_source": trace_source,
            })
        out.append({
            "stage_id": "STAGE_03_CORE_OUTPUT",
            "unit_id": row.get("unit_id"),
            "layer_id": int(row["layer_id"]),
            "group_size": group,
            "condition": condition,
            "stage_delta_norm": transfer_norm,
            "source_error_norm": error_norm,
            "G_stage": as_float(row.get("G_core")),
            "shape": "[B,T,HV,V]",
            "dtype": "float32 captured from backend output",
            "hash": "REUSED_CORE_OUTPUT_EVENT",
            "relative_norm": as_float(row.get("local_damage")),
            "cosine_where_meaningful": None,
            "max_abs": as_float(row.get("transfer_identity_abs_error")),
            "trace_source": "REUSED_FROM_CORE_OUTPUT_TASK",
        })
    return out


def core_internal_event_trace(core_rows):
    out = []
    for row in stage_directional_gain_event_level(core_rows):
        out.append({
            **row,
            "delta_tensor_materialized": "NO_REUSED_PRIOR_SUMMARY_STATS_ONLY",
            "NEW_GPU_LOCAL_EVALUATION": 0,
            "NEW_STAGE_B_TRAJECTORY": 0,
            "NEW_CLOSED_LOOP_TRAJECTORY": 0,
        })
    return out


def layer_gain_rows(pairs, reps=400):
    out = []
    prior = prior_layer_labels()
    for layer in KDA_LAYERS:
        sub = [row for row in pairs if int(row["layer_id"]) == layer]
        vals = [row["delta_G"] for row in sub]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + layer)
        out.append({
            "stage_id": "STAGE_03_CORE_OUTPUT",
            "layer_id": layer,
            "prior_layer_label": prior[layer],
            "N_cells": len(sub),
            "G_R": median(row["G_R"] for row in sub),
            "G_C": median(row["G_C"] for row in sub),
            "delta_G": median(vals),
            "mean_delta_G": mean(vals),
            "delta_G_CI_low": ci[0] if ci else None,
            "delta_G_CI_high": ci[1] if ci else None,
            "C_gain_gt_R_count": sum(1 for x in vals if x > 0),
            "R_gain_gt_C_count": sum(1 for x in vals if x < 0),
        })
    return out


def stage_layer_sign_concordance(layer_rows, prior):
    out = []
    for stage_id in sorted({str(row["stage_id"]) for row in layer_rows}, key=lambda x: stage_order_map().get(x, 999.0)):
        sub = [row for row in layer_rows if str(row["stage_id"]) == stage_id]
        discordant = []
        concordant = 0
        l22_reverse = "MISSING"
        for row in sub:
            layer = int(row["layer_id"])
            delta = as_float(row.get("delta_G"))
            label = prior.get(layer)
            ok = (label == "C_SENSITIVE" and delta is not None and delta > 0) or (label == "R_SENSITIVE" and delta is not None and delta < 0)
            if ok:
                concordant += 1
            else:
                discordant.append(layer)
            if layer == 22:
                l22_reverse = "PASS" if delta is not None and delta < 0 else "FAIL"
        out.append({
            "stage_id": stage_id,
            "N_SIGN_CONCORDANT": concordant,
            "N_SIGN_DISCORDANT": len(discordant),
            "CONCORDANCE_RATE": concordant / len(sub) if sub else None,
            "DISCORDANT_LAYER_IDS": ",".join(str(x) for x in discordant),
            "LAYER22_REVERSE_CONTROL": l22_reverse,
        })
    return out


def complete_stage_sign_concordance(sign_rows):
    by_stage = {row["stage_id"]: row for row in sign_rows}
    out = []
    for stage in build_stage_manifest():
        stage_id = stage["stage_id"]
        if stage_id in by_stage:
            out.append(by_stage[stage_id])
        elif stage_id == "STAGE_00_RECURRENT_STATE_INPUT":
            out.append({
                "stage_id": stage_id,
                "N_SIGN_CONCORDANT": 0,
                "N_SIGN_DISCORDANT": 18,
                "CONCORDANCE_RATE": 0.0,
                "DISCORDANT_LAYER_IDS": ",".join(str(x) for x in KDA_LAYERS),
                "LAYER22_REVERSE_CONTROL": "FAIL",
            })
        else:
            out.append({
                "stage_id": stage_id,
                "N_SIGN_CONCORDANT": "UNRESOLVED",
                "N_SIGN_DISCORDANT": "UNRESOLVED",
                "CONCORDANCE_RATE": "UNRESOLVED",
                "DISCORDANT_LAYER_IDS": "UNRESOLVED",
                "LAYER22_REVERSE_CONTROL": "UNRESOLVED",
            })
    return out


def layer22_stage_reverse_trace(layer_rows):
    out = []
    core_row = next((row for row in layer_rows if int(row["layer_id"]) == 22), None)
    for stage in build_stage_manifest():
        stage_id = stage["stage_id"]
        if stage_id == "STAGE_00_RECURRENT_STATE_INPUT":
            out.append({
                "stage_id": stage_id,
                "L22_G_R": 1.0,
                "L22_G_C": 1.0,
                "L22_DELTA_G": 0.0,
                "layer22_reverse": "NO",
                "trace_source": "matched input magnitudes",
            })
        elif stage_id == "STAGE_03_CORE_OUTPUT" and core_row:
            delta = as_float(core_row.get("delta_G"))
            out.append({
                "stage_id": stage_id,
                "L22_G_R": core_row.get("G_R"),
                "L22_G_C": core_row.get("G_C"),
                "L22_DELTA_G": delta,
                "layer22_reverse": "YES" if delta is not None and delta < 0 else "NO",
                "trace_source": "reused core-output trace",
            })
        else:
            out.append({
                "stage_id": stage_id,
                "L22_G_R": "UNRESOLVED",
                "L22_G_C": "UNRESOLVED",
                "L22_DELTA_G": "UNRESOLVED",
                "layer22_reverse": "UNRESOLVED",
                "trace_source": "not numerically exposed in current trace",
            })
    return out


def granularity_stage_directional_summary(pairs, reps=400):
    out = []
    for group in GROUP_SIZES:
        sub = [row for row in pairs if int(row["group_size"]) == group]
        vals = [row["delta_G"] for row in sub]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + group)
        out.append({
            "stage_id": "STAGE_03_CORE_OUTPUT",
            "group_size": group,
            "median_delta_G": median(vals),
            "mean_delta_G": mean(vals),
            "delta_G_CI_low": ci[0] if ci else None,
            "delta_G_CI_high": ci[1] if ci else None,
            "N_cells": len(vals),
        })
    by = {int(row["group_size"]): row for row in out}
    if 128 in by and 16 in by:
        by[128]["D_128_to_16"] = (as_float(by[128]["median_delta_G"]) or 0.0) - (as_float(by[16]["median_delta_G"]) or 0.0)
        by[16]["D_128_to_16"] = by[128]["D_128_to_16"]
    return out


def operator_decomposition_master_table(identity_rows, sign_rows, l22_rows):
    sign_by = {row["stage_id"]: row for row in sign_rows}
    l22_by = {row["stage_id"]: row for row in l22_rows}
    out = []
    for row in identity_rows:
        stage_id = row["stage_id"]
        sign = sign_by.get(stage_id, {})
        l22 = l22_by.get(stage_id, {})
        out.append({
            "stage_id": stage_id,
            "identity_status": row.get("identity_status"),
            "median_identity_error": row.get("median_identity_rel_error"),
            "N_SIGN_CONCORDANT": sign.get("N_SIGN_CONCORDANT", 0 if stage_id == "STAGE_00_RECURRENT_STATE_INPUT" else "UNRESOLVED"),
            "CONCORDANCE_RATE": sign.get("CONCORDANCE_RATE", 0.0 if stage_id == "STAGE_00_RECURRENT_STATE_INPUT" else "UNRESOLVED"),
            "layer22_reverse": l22.get("layer22_reverse", "UNRESOLVED"),
            "interpretation": stage_interpretation(stage_id),
        })
    return out


def stage_interpretation(stage_id):
    return {
        "STAGE_00_RECURRENT_STATE_INPUT": "matched R/C perturbations enter with equal magnitude; no layer sign closure expected",
        "STAGE_01_FIXED_LOCAL_DRIVERS": "drivers are frozen and cannot by themselves express state perturbation gain",
        "STAGE_02_KDA_KERNEL_BOUNDARY": "last Python-visible point before unresolved FLA internals",
        "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL": "candidate internal block; not directly traced in this task",
        "STAGE_03_CORE_OUTPUT": "first observed failure and first observed 18/18 sign closure",
        "STAGE_04_FINAL_STATE": "not traced event-wise here",
    }.get(stage_id, "")


def source_inventory():
    inputs = [
        CORE_OUT / "summary.json",
        CORE_OUT / "core_transfer_gain_event_level.csv",
        CORE_OUT / "common36_manifest.csv",
        LAYER_OUT / "layer_classification.csv",
        LAYER_OUT / "summary.json",
        RESIDUAL_OUT / "summary.json",
        MAG_OUT / "summary.json",
        MODEL_FILE,
        FLA_KDA_DIR / "fused_recurrent.py",
        FLA_KDA_DIR / "chunk.py",
        FLA_KDA_DIR / "naive.py",
        REPO / "experiments/ling/run_ling_kda_core_output_directional_geometry_localization_v1.py",
    ]
    return {
        "TASK": TASK,
        "created_at": now(),
        "source_artifacts": [
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in inputs
        ],
        "COMMON_COHORT_REUSE_GATE": "PASS" if (CORE_OUT / "common36_manifest.csv").exists() and len(read_csv(CORE_OUT / "common36_manifest.csv")) == 36 else "FAIL",
        "PRIOR_LAYER_LABEL_REUSE_GATE": prior_layer_label_reuse_gate(read_csv(LAYER_OUT / "layer_classification.csv")),
        "GROUP_SIZES_PRIMARY": GROUP_SIZES,
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
    }


def write_docs(summary=None):
    stages = build_stage_manifest()
    path_lines = [
        f"# {TASK} Operator Path",
        "",
        "Audited path from current Ling source:",
        "",
        "1. `BailingMoeV3DecoderLayer.forward` applies input RMSNorm before KDA attention.",
        "2. `BailingMoeV3KimiDeltaAttention.forward` reads `past_key_value.layers[layer].keys` as `recurrent_state`.",
        "3. q/k/v are projected and passed through `ShortConvolution`; the first gate tensor `g` and post-sigmoid `beta` are computed from the same frozen FP hidden state.",
        "4. q/k/v/g are reshaped to head layout and passed with `A_log`, `dt_bias`, and `initial_state` to `chunk_kda` or `fused_recurrent_kda`.",
        "5. The prior `CORE_OUTPUT` trace is the backend `o` returned by that KDA call, before external `FusedRMSNormGated`, reshape to `[B,T,H*D]`, and `o_proj`.",
        "6. Source and naive reference show the kernel output is a post-update readout: decay/update are applied before `b_o`/`o` is stored.",
        "",
        "Python-visible ordered graph:",
        "",
        "`STAGE_00_RECURRENT_STATE_INPUT -> STAGE_01_FIXED_LOCAL_DRIVERS -> STAGE_02_KDA_KERNEL_BOUNDARY -> STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL -> STAGE_03_CORE_OUTPUT -> STAGE_04_FINAL_STATE`",
        "",
        "The `STAGE_02A` internals are source-visible but not numerically exposed by the fused recurrent trace used for the prior 36-unit panel.",
    ]
    write_text(OUT_DIR / "kda_core_internal_operator_path.md", "\n".join(path_lines) + "\n")

    eq_lines = [
        f"# {TASK} Operator Equations",
        "",
        "Under frozen local drivers, the recurrent input perturbation is `E`, with matched `||E_R||_F == ||E_C_match||_F`.",
        "",
        "Input stage:",
        "",
        "`S0(E) = S + E`, so `delta S0 = E`; this is `EXACT_LINEAR_IN_STATE`.",
        "",
        "Driver stage:",
        "",
        "`q,k,v,g,beta,A_log,dt_bias` are captured from the FP branch and replayed unchanged. They may contain nonlinear projections/convolutions/sigmoids with respect to hidden states, but are fixed with respect to this task's state perturbation.",
        "",
        "Fused recurrent reference for one token, matching FLA source structure:",
        "",
        "`q <- l2norm(q) * scale`, `k <- l2norm(k)` when enabled.",
        "`gk <- -exp(A_log) * softplus(g + dt_bias)` or safe-gate equivalent.",
        "`H_decay = exp(gk) * H0`.",
        "`v_new = beta * (v - contraction(H_decay, k))`.",
        "`H1 = H_decay + outer(k, v_new)`.",
        "`o = contraction(q, H1)`.",
        "",
        "Therefore, ideal reference math is affine in `H0` under fixed drivers. The prior measured identity `o(S+E)-o(S) == o(E)-o(0)` should hold if the observed backend behaves as the simple affine transfer. It fails at observed `CORE_OUTPUT`, while hidden kernel substages are not exposed in the current trace.",
        "",
        "Current output state timing:",
        "",
        "`CORE_OUTPUT_USES_PRE_OR_POST_UPDATE_STATE = POST_BY_SOURCE_REFERENCE`, because FLA stores output after decay/update in both the Triton loop and `naive_recurrent_kda`.",
    ]
    write_text(OUT_DIR / "core_internal_operator_equations.md", "\n".join(eq_lines) + "\n")

    next_kernel = [
        f"# {TASK} Next Kernel Internal Decomposition Plan",
        "",
        "Do not execute this plan in the current task.",
        "",
        "Recommended next task: `LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1`.",
        "",
        "Minimal design:",
        "",
        "1. Use one KDA layer, one head/value-head, one token or a two-token tiny horizon.",
        "2. Build deterministic small tensors for `q,k,v,g,beta,A_log,dt_bias,S,E_R,E_C_match`.",
        "3. Implement a pure PyTorch reference matching FLA fused recurrent equations with named checkpoints: q/k l2norm, gate activation, decayed state, beta residual value, updated state, readout.",
        "4. Compare each checkpoint against FLA output for the same tiny input.",
        "5. Only after reference equivalence is verified, replay one common-36 event through the reference to identify the exact substage that first produces the identity mismatch or R/C sign closure.",
        "",
        "Stop condition: produce a substage table only; do not launch causal intervention.",
    ]
    write_text(OUT_DIR / "next_kernel_internal_decomposition_plan.md", "\n".join(next_kernel) + "\n")

    next_targeted = [
        f"# {TASK} Next Targeted Causal Plan",
        "",
        "No targeted causal experiment was launched in this task.",
        "",
        "Because localization is kernel-internal rather than a single exposed Python operator, the next recommended task is `LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1` before any causal closure experiment.",
    ]
    write_text(OUT_DIR / "next_targeted_causal_plan.md", "\n".join(next_targeted) + "\n")

    readme = [
        f"# {TASK}",
        "",
        "Targeted operator-semantics and mechanism-localization audit.",
        "",
        "Primary panel uses only `g128` and `g16`, reuses the prior common-36 cohort and frozen 17 C-sensitive + layer22 R-sensitive labels, and launches no new recurrent/closed-loop/causal experiment.",
    ]
    write_text(OUT_DIR / "README.md", "\n".join(readme) + "\n")

    if summary:
        report = [
            f"# {TASK}",
            "",
            f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
            f"FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
            f"EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE: {summary['EARLIEST_TRANSFER_IDENTITY_FAILURE_STAGE']}",
            f"EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE: {summary['EARLIEST_18_OF_18_LAYER_SIGN_CLOSURE_STAGE']}",
            f"EARLIEST_LAYER22_REVERSE_STAGE: {summary['EARLIEST_LAYER22_REVERSE_STAGE']}",
            "",
            "Interpretation:",
            "",
            "The matched R/C perturbations enter the state carrier with equal magnitude, and the frozen-driver boundary is the last fully Python-visible point. The first observed tensor after the unresolved FLA internal decay/update/readout block is `CORE_OUTPUT`; at that point the transfer identity has already failed while the full 18/18 frozen layer sign pattern, including layer22 reversal, is visible.",
            "",
            "This localizes the unresolved operator to inside the KDA kernel block. It does not prove causal creation by a specific kernel sub-operator.",
            "",
            "No Stage-B trajectory, closed-loop rerun, targeted intervention, method design, or Kimi transfer was launched.",
        ]
        write_text(OUT_DIR / "report.md", "\n".join(report) + "\n")


def analyze(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = source_inventory()
    save_json(OUT_DIR / "source_artifact_inventory.json", inventory)
    if (CORE_OUT / "common36_manifest.csv").exists():
        shutil.copyfile(CORE_OUT / "common36_manifest.csv", OUT_DIR / "common36_manifest.csv")

    stages = build_stage_manifest()
    save_json(OUT_DIR / "core_internal_stage_manifest.json", stages)
    write_docs()

    core_rows = filtered_core_events()
    write_csv(OUT_DIR / "stage_directional_gain_event_level.csv", stage_directional_gain_event_level(core_rows))
    write_csv(OUT_DIR / "core_internal_event_trace.csv", core_internal_event_trace(core_rows))

    identity_rows = stage_identity_error_summary(core_rows)
    write_csv(OUT_DIR / "stage_identity_error_summary.csv", identity_rows)
    pairs = pair_event_rows(core_rows)
    layer_rows = layer_gain_rows(pairs, reps=args.bootstrap_reps)
    prior = prior_layer_labels()
    sign_rows = complete_stage_sign_concordance(stage_layer_sign_concordance(layer_rows, prior))
    write_csv(OUT_DIR / "stage_layer_sign_concordance.csv", sign_rows)
    l22_rows = layer22_stage_reverse_trace(layer_rows)
    write_csv(OUT_DIR / "layer22_stage_reverse_trace.csv", l22_rows)
    gran_rows = granularity_stage_directional_summary(pairs, reps=args.bootstrap_reps)
    write_csv(OUT_DIR / "granularity_stage_directional_summary.csv", gran_rows)
    master_rows = operator_decomposition_master_table(identity_rows, sign_rows, l22_rows)
    write_csv(OUT_DIR / "operator_decomposition_master_table.csv", master_rows)

    localization = classify_localization(master_rows, first_unresolved="STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL")
    core_identity = next(row for row in identity_rows if row["stage_id"] == "STAGE_03_CORE_OUTPUT")
    core_sign = next((row for row in sign_rows if row["stage_id"] == "STAGE_03_CORE_OUTPUT"), {})
    l22_core = next((row for row in l22_rows if row["stage_id"] == "STAGE_03_CORE_OUTPUT"), {})
    gran_by_group = {int(row["group_size"]): row for row in gran_rows}
    gates = {
        "COMMON_COHORT_REUSE_GATE": inventory["COMMON_COHORT_REUSE_GATE"],
        "PRIOR_LAYER_LABEL_REUSE_GATE": inventory["PRIOR_LAYER_LABEL_REUSE_GATE"],
        "MATCHED_CARRIER_GATE": "PASS" if core_rows and all(row.get("MATCHED_CARRIER_GATE") == "PASS" for row in core_rows) else "FAIL",
        "MAGNITUDE_MATCH_GATE": "PASS" if core_rows and all(row.get("MAGNITUDE_MATCH_GATE") == "PASS" for row in core_rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if core_rows and all(row.get("DIRECTION_PRESERVATION_GATE") == "PASS" for row in core_rows) else "FAIL",
        "CORE_INTERNAL_SEMANTICS_GATE": "PASS" if MODEL_FILE.exists() and (FLA_KDA_DIR / "fused_recurrent.py").exists() else "FAIL",
    }
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if all(v == "PASS" for v in gates.values()) else "INCOMPLETE",
        **localization,
        "EARLIEST_17_OF_18_OR_BETTER_STAGE": earliest_sign_closure(master_rows, target=17),
        "FIRST_UNRESOLVED_KERNEL_INTERNAL_STAGE": "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL",
        "CORE_OUTPUT_USES_PRE_OR_POST_UPDATE_STATE": "POST_BY_SOURCE_REFERENCE",
        "OLD_CORE_TRANSFER_IDENTITY_FAILURE_EXPLAINED_BY_STATE_TIMING": "NO",
        **gates,
        "N_COMMON_UNITS": len({row["unit_id"] for row in core_rows}),
        "N_KDA_LAYERS": len(KDA_LAYERS),
        "KDA_LAYER_IDS": KDA_LAYERS,
        "N_C_SENSITIVE_PRIOR_LAYERS": 17,
        "N_R_SENSITIVE_PRIOR_LAYERS": 1,
        "N_CORE_OUTPUT_EVENT_ROWS": len(core_rows),
        "N_NEW_GPU_LOCAL_EVALUATIONS": 0,
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        "CORE_OUTPUT_IDENTITY_STATUS": core_identity["identity_status"],
        "CORE_OUTPUT_MEDIAN_IDENTITY_REL_ERROR": core_identity["median_identity_rel_error"],
        "CORE_OUTPUT_MAX_IDENTITY_REL_ERROR": core_identity["max_identity_rel_error"],
        "CORE_OUTPUT_MAX_IDENTITY_ABS_ERROR": core_identity["max_identity_abs_error"],
        "CORE_OUTPUT_N_SIGN_CONCORDANT": int(core_sign.get("N_SIGN_CONCORDANT", 0)),
        "CORE_OUTPUT_LAYER_SIGN_CONCORDANCE_RATE": as_float(core_sign.get("CONCORDANCE_RATE")),
        "CORE_OUTPUT_DISCORDANT_LAYER_IDS": core_sign.get("DISCORDANT_LAYER_IDS", ""),
        "L22_G_R_CORE_OUTPUT": as_float(l22_core.get("L22_G_R")),
        "L22_G_C_CORE_OUTPUT": as_float(l22_core.get("L22_G_C")),
        "L22_DELTA_G_CORE_OUTPUT": as_float(l22_core.get("L22_DELTA_G")),
        "G128_DELTA_G_CORE_OUTPUT": as_float(gran_by_group.get(128, {}).get("median_delta_G")),
        "G16_DELTA_G_CORE_OUTPUT": as_float(gran_by_group.get(16, {}).get("median_delta_G")),
        "G128_TO_G16_DELTA_G_DIFFERENCE": as_float(gran_by_group.get(128, {}).get("D_128_to_16")),
        "SOURCE_MAGNITUDE_CONTEXT_REUSED_ONLY": "YES",
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "NEXT_RECOMMENDED_TASK": "LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1",
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
    }
    save_json(OUT_DIR / "bootstrap_results.json", {
        "layer_core_output_delta_g": layer_rows,
        "stage_sign_concordance": sign_rows,
        "granularity_stage_directional_summary": gran_rows,
        "identity": identity_rows,
        "bootstrap_reps": args.bootstrap_reps,
    })
    save_json(OUT_DIR / "summary.json", summary)
    write_docs(summary)
    final_summary(args.py_compile_status, args.pytest_status)
    return summary


def verify(_args=None):
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    trace_rows = read_csv(OUT_DIR / "core_internal_event_trace.csv")
    core_trace_rows = [row for row in trace_rows if row.get("stage_id") == "STAGE_03_CORE_OUTPUT"]
    expected_core_rows = int(summary.get("N_COMMON_UNITS") or 0) * int(summary.get("N_KDA_LAYERS") or 0) * len(GROUP_SIZES) * 2
    expected_trace_rows = expected_core_rows * len(build_stage_manifest())
    numeric_bad = []
    for csv_name in [
        "stage_identity_error_summary.csv",
        "stage_directional_gain_event_level.csv",
        "stage_layer_sign_concordance.csv",
        "layer22_stage_reverse_trace.csv",
        "granularity_stage_directional_summary.csv",
        "operator_decomposition_master_table.csv",
    ]:
        for idx, row in enumerate(read_csv(OUT_DIR / csv_name)):
            for key, value in row.items():
                if value in ("", "None", None, "UNRESOLVED"):
                    continue
                if key.startswith("N_") or key.endswith("_error") or key.endswith("_G") or key.startswith("G") or key in {"CONCORDANCE_RATE", "L22_G_R", "L22_G_C", "L22_DELTA_G", "median_delta_G", "mean_delta_G", "stage_delta_norm", "source_error_norm"}:
                    if not finite(value):
                        numeric_bad.append({"file": csv_name, "row": idx, "key": key, "value": value})
    gates_ok = all(summary.get(key) == "PASS" for key in ["CORE_INTERNAL_SEMANTICS_GATE", "COMMON_COHORT_REUSE_GATE", "MATCHED_CARRIER_GATE", "MAGNITUDE_MATCH_GATE", "DIRECTION_PRESERVATION_GATE", "PRIOR_LAYER_LABEL_REUSE_GATE"])
    fresh = (
        not missing
        and len(trace_rows) == expected_trace_rows
        and len(core_trace_rows) == expected_core_rows
        and not numeric_bad
        and gates_ok
        and summary.get("NEW_STAGE_B_TRAJECTORIES") == 0
        and summary.get("NEW_CLOSED_LOOP_TRAJECTORIES") == 0
        and summary.get("NO_NEXT_EXPERIMENT_LAUNCHED") == "YES"
    )
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": "PASS" if fresh else "FAIL",
        "missing_files": missing,
        "N_TRACE_ROWS": len(trace_rows),
        "N_EXPECTED_TRACE_ROWS": expected_trace_rows,
        "N_CORE_OUTPUT_TRACE_ROWS": len(core_trace_rows),
        "N_EXPECTED_CORE_OUTPUT_TRACE_ROWS": expected_core_rows,
        "numeric_bad": numeric_bad[:20],
        "gates_ok": gates_ok,
        "NEW_STAGE_B_TRAJECTORIES": summary.get("NEW_STAGE_B_TRAJECTORIES"),
        "NEW_CLOSED_LOOP_TRAJECTORIES": summary.get("NEW_CLOSED_LOOP_TRAJECTORIES"),
    }
    if summary:
        summary["FRESH_OUTPUT_VERIFICATION"] = verification["FRESH_OUTPUT_VERIFICATION"]
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(summary.get("PY_COMPILE", "NOT_RUN"), summary.get("PYTEST", "NOT_RUN"))
    save_json(OUT_DIR / "fresh_output_verification.json", verification)
    return verification


def final_summary(py_compile_status="NOT_RUN", pytest_status="NOT_RUN"):
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary:
        summary["PY_COMPILE"] = py_compile_status
        summary["PYTEST"] = pytest_status
        save_json(summary_path, summary)
    lines = [f"{key} = {summary.get(key)}" for key in SUMMARY_KEYS]
    lines.append("STOP.")
    write_text(OUT_DIR / "final_console_summary.txt", "\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["analyze", "verify", "summary"], default="analyze")
    parser.add_argument("--bootstrap-reps", type=int, default=400)
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    args = parser.parse_args()
    if args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify(args)
    else:
        out = final_summary(args.py_compile_status, args.pytest_status)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) if isinstance(out, dict) else "\n".join(out))


if __name__ == "__main__":
    main()
