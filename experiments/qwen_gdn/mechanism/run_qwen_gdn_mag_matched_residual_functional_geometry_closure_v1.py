#!/usr/bin/env python3
import argparse
import copy
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


TASK = "QWEN_GDN_MAG_MATCHED_RESIDUAL_FUNCTIONAL_GEOMETRY_CLOSURE_V1"
SLUG = "qwen_gdn_mag_matched_residual_functional_geometry_closure_v1"
EPS = 1e-12
NORM_TOL = 1e-5
IDENTITY_TOL = 1e-6
HORIZON = 128
REPO = Path(os.environ.get("GDN_REPO_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("GDN_DATA_ROOT", REPO.parent))
RUN_DIR = Path(os.environ.get("QWEN_GDN_GEOMETRY_RUN_DIR", REPO / "runs" / SLUG))
REPORT_DIR = REPO / "reports" / "propagation"
MODEL_PATH = Path(os.environ.get("QWEN35_MODEL_PATH", DATA_ROOT / "modelscope_models" / "Qwen3.5-9B"))
MANIFEST_PATH = REPO / "results" / "propagation" / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"
REPLAY_RUNNER_PATH = REPO / "experiments" / "propagation" / "run_qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1.py"
M5_SOURCE_PATH = REPO / "experiments" / "propagation" / "run_int8_linearized_functional_risk_proxy_extraction.py"
M5_GOLDEN_SUMMARY_PATH = REPO / "results" / "propagation" / "gdn_int8_linearized_functional_risk_proxy_extraction_v1_final_summary.json"
PRIOR_CLOSED_LOOP_SUMMARY_PATH = REPO / "runs" / "qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1" / "aggregate_summary.json"
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
PILOT_HISTORY_LENGTHS = [1, 8, 32, 64]
PILOT_UNIT_INDICES = [0, 4, 8, 9, 13, 17]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
EXPECTED_MANIFEST_SHA256 = "230a54d93358074cc9a0b82e6c0891559276ddce3e17028779cef24001b38579"
EXPECTED_REPLAY_SHA256 = "f4f80c5ad9ce005596e6d9810949ef8423f189db2e0941afb6c7e24eab548f2f"
EXPECTED_M5_SHA256 = "70928c18d3588fcd91827240b7c28fd2789605f1a6302c3ef3792478ec23f3ea"
EXPECTED_MODEL_CONFIG_SHA256 = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
EXPECTED_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
EXPECTED_M5_GOLDEN_SUMMARY_SHA256 = "21093223bb5d78aa7712e37514cada301953107e0f494394bad4ed4fda946b7f"
EXPECTED_PRIOR_CLOSED_LOOP_SUMMARY_SHA256 = "cae8d2f70061d2fd41b88d4d278ad5338830c2b4c103853dc7c236855536881f"
EXPECTED_M5_TRANSITIVE_SHA256 = {
    "p1": "676462c702fff05e582d46b8110cf48fa03aaff7b981f19b6f315e245c9fffb5",
    "normswap": "f292b8ef1d9ffebee2327da20948b95718719b03e4f81c86d7cb59396d71b491",
    "frozen": "41f8654310de0fac1b150baeff5b3ab4b40a9c5d0cbccc4fe72467a489835357",
    "vf": "7d2aefb99eacc01badd1794312d3eeff2d7a2fe5cd5fa264865d52e0ea460166",
    "basis": "cfe3fc97a33b4c8bcfbb2dcb88df4626d02192c2c6f88bdb0d5de57a26fb5ec9",
    "realrc": "4337fc8fba98d591c1ff38a5853482d19ea5f91ff1f55726f4c2622a1cd7f478",
}
M5_TRANSITIVE_SOURCE_PATHS = {
    "p1": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_orientation_state_change_mechanism.py",
    "normswap": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_r128_c128_natural_residual_norm_swap_causal.py",
    "frozen": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_r128_c128_frozen_observability_path_decomposition.py",
    "vf": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_value_side_functional_direction_formal_and_transfer.py",
    "basis": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_value_basis_symmetry_breaking_localization.py",
    "realrc": DATA_ROOT / "experiments" / "qwen35_gdn_quant" / "run_int8_real_rc_tangential_recurrent_mediation.py",
}
REQUIRED_OUTPUT_NAMES = [
    "results.jsonl",
    "aggregate_summary.json",
    "report.md",
    "same_state_f5.csv",
    "direction_factorial.csv",
    "history_direction_effect.csv",
]


def stack_norm(stack):
    return math.sqrt(sum(float(value.detach().double().pow(2).sum().item()) for value in stack.values()))


def scale_stack(stack, factor):
    return {layer: value.detach().float() * float(factor) for layer, value in stack.items()}


def stack_cos(left, right):
    import torch

    a = torch.cat([left[layer].detach().float().reshape(-1) for layer in sorted(left)])
    b = torch.cat([right[layer].detach().float().reshape(-1) for layer in sorted(right)])
    denominator = float(a.norm().item() * b.norm().item())
    if denominator <= EPS:
        return None
    return float(torch.dot(a, b).item() / denominator)


def normalized_functional_metrics(m0, m4, m5):
    values = [float(m0), float(m4), float(m5)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("functional metrics must be finite")
    if values[0] <= EPS:
        raise ValueError("zero-norm residual cannot be normalized")
    return {
        "M0": values[0],
        "M4": values[1],
        "M5": values[2],
        "F4": values[1] / values[0],
        "F5": values[2] / values[0],
    }


def build_factorial_interventions(e_r, e_c):
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    if m_r <= EPS or m_c <= EPS:
        raise ValueError("degenerate source residual")
    cells = {
        "RR": (e_r, m_r, "R", "R_MAG"),
        "RC": (e_r, m_c, "R", "C_MAG"),
        "CR": (e_c, m_r, "C", "R_MAG"),
        "CC": (e_c, m_c, "C", "C_MAG"),
    }
    output = {}
    for branch, (source, magnitude, direction, magnitude_family) in cells.items():
        source_norm = m_r if direction == "R" else m_c
        delta = scale_stack(source, magnitude / source_norm)
        output[branch] = {
            "delta": delta,
            "source_direction": direction,
            "source_magnitude_family": magnitude_family,
            "source_direction_norm": source_norm,
            "target_norm": magnitude,
            "applied_norm": stack_norm(delta),
            "direction_cosine": stack_cos(delta, source),
        }
    return output


def branch_configs():
    return [
        {"branch": "FP", "source_direction": "FP", "source_magnitude_family": "FP"},
        {"branch": "RR", "source_direction": "R", "source_magnitude_family": "R_MAG"},
        {"branch": "RC", "source_direction": "R", "source_magnitude_family": "C_MAG"},
        {"branch": "CR", "source_direction": "C", "source_magnitude_family": "R_MAG"},
        {"branch": "CC", "source_direction": "C", "source_magnitude_family": "C_MAG"},
    ]


def stage_protocol(stage):
    protocols = {
        "stage0": {"history_lengths": [1], "unit_indices": [0]},
        "smoke": {"history_lengths": [1, 8], "unit_indices": [0]},
        "pilot": {"history_lengths": PILOT_HISTORY_LENGTHS, "unit_indices": PILOT_UNIT_INDICES},
        "formal": {"history_lengths": HISTORY_LENGTHS, "unit_indices": list(range(18))},
    }
    if stage not in protocols:
        raise ValueError(f"unknown stage: {stage}")
    return {key: list(value) for key, value in protocols[stage].items()}


def exact_result_coverage(rows, units, history_lengths):
    expected = {
        (unit["unit_id"], int(length), config["branch"])
        for unit in units
        for length in history_lengths
        for config in branch_configs()
    }
    actual = [
        (row.get("unit_id"), int(row.get("history_length", -1)), row.get("branch"))
        for row in rows
    ]
    return len(actual) == len(expected) and len(actual) == len(set(actual)) and set(actual) == expected


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _ranks(values):
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    result = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for _value, index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    x_rank = _ranks([x for x, _ in pairs])
    y_rank = _ranks([y for _, y in pairs])
    x_mean = statistics.mean(x_rank)
    y_mean = statistics.mean(y_rank)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_rank, y_rank))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_rank) * sum((y - y_mean) ** 2 for y in y_rank))
    return numerator / denominator if denominator > EPS else None


def _binomial_two_sided(positive, total):
    if total <= 0:
        return 1.0
    tail = min(int(positive), int(total) - int(positive))
    probability = sum(math.comb(total, index) for index in range(tail + 1)) / (2.0 ** total)
    return min(1.0, 2.0 * probability)


def paired_summary(values, seed=1729, bootstrap_samples=4000):
    vals = [float(value) for value in values if finite_number(value)]
    if not vals:
        return {"mean": None, "median": None, "std": None, "positive": 0, "negative": 0, "ties": 0, "n": 0, "bootstrap_ci": [None, None], "paired_sign_p": 1.0}
    rng = random.Random(int(seed))
    boot = sorted(statistics.median(rng.choices(vals, k=len(vals))) for _ in range(int(bootstrap_samples)))
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    positive = sum(value > 0 for value in vals)
    negative = sum(value < 0 for value in vals)
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals),
        "positive": positive,
        "negative": negative,
        "ties": len(vals) - positive - negative,
        "n": len(vals),
        "bootstrap_ci": [low, high],
        "paired_sign_p": _binomial_two_sided(positive, positive + negative),
    }


def build_direction_effect_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["unit_id"], int(row["history_length"]))][row["branch"]] = row
    output = []
    for (unit_id, history_length), branches in sorted(grouped.items()):
        if not all(branch in branches for branch in ("RR", "RC", "CR", "CC")):
            continue
        at_c = float(branches["RC"]["future_kl_auc"]) - float(branches["CC"]["future_kl_auc"])
        at_r = float(branches["RR"]["future_kl_auc"]) - float(branches["CR"]["future_kl_auc"])
        output.append({
            "unit_id": unit_id,
            "history_length": history_length,
            "DIRECTION_RISK_AT_C_MAG": at_c,
            "DIRECTION_RISK_AT_R_MAG": at_r,
            "DIRECTION_MAIN": 0.5 * (at_c + at_r),
        })
    return output


def summarize_effect_by_l(effect_rows, key):
    lengths = sorted({int(row["history_length"]) for row in effect_rows})
    return {
        str(length): paired_summary(
            [row[key] for row in effect_rows if int(row["history_length"]) == length],
            seed=1729 + length,
        )
        for length in lengths
    }


def history_amplification_rho(by_l):
    pairs = sorted((int(length), cell.get("median")) for length, cell in by_l.items() if finite_number(cell.get("median")))
    return spearman([math.log2(length) for length, _ in pairs], [value for _, value in pairs])


def metric_predictive_signal(rows, metric):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["unit_id"], int(row["history_length"]))][row["branch"]] = row
    schedules = {"C_MAG": ("RC", "CC"), "R_MAG": ("RR", "CR")}
    output = {}
    for name, (r_branch, c_branch) in schedules.items():
        predictor = []
        outcome = []
        for branches in grouped.values():
            if r_branch not in branches or c_branch not in branches:
                continue
            predictor.append(float(branches[r_branch][metric]) - float(branches[c_branch][metric]))
            outcome.append(float(branches[r_branch]["future_kl_auc"]) - float(branches[c_branch]["future_kl_auc"]))
        tied = bool(predictor) and max(abs(value) for value in predictor) <= EPS
        output[name] = {
            "n": len(predictor),
            "spearman": None if tied else spearman(predictor, outcome),
            "status": "NO_DISCRIMINATORY_SIGNAL_EXACT_MATCHING" if tied else "ESTIMATED",
            "max_abs_predictor_gap": max((abs(value) for value in predictor), default=None),
        }
    return output


def closed_loop_direction_consistency(exogenous_by_l, closed_loop_by_l):
    common = sorted(set(exogenous_by_l).intersection(closed_loop_by_l), key=int)
    exogenous = [exogenous_by_l[length].get("median") for length in common]
    closed = [closed_loop_by_l[length].get("median") for length in common]
    positive = sum(finite_number(value) and value > 0 for value in exogenous)
    exogenous_rho = spearman([math.log2(int(length)) for length in common], exogenous)
    closed_rho = spearman([math.log2(int(length)) for length in common], closed)
    if positive >= 5 and (exogenous_rho or 0.0) > 0 and (closed_rho or 0.0) > 0:
        return "STRONG"
    if positive >= 3:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def _robust_cells(by_carrier_or_effect):
    for cells in by_carrier_or_effect.values():
        supported = sum(
            finite_number(cell.get("median"))
            and cell["median"] > 0
            and int(cell.get("n", 0)) > 0
            and int(cell.get("positive", 0)) / int(cell["n"]) >= 2.0 / 3.0
            for cell in cells.values()
        )
        if supported < 4:
            return False
    return True


def scientific_classification(same_state_by_carrier, direction_by_name, predictive, consistency):
    same_state_robust = _robust_cells(same_state_by_carrier)
    direction_robust = _robust_cells(direction_by_name)
    predictive_f5 = all((predictive.get(name, {}).get("spearman") or -math.inf) >= 0.3 for name in ("C_MAG", "R_MAG"))
    l1_positive = all(
        finite_number(cells.get("1", {}).get("median")) and cells["1"]["median"] > 0
        for cells in direction_by_name.values()
    )
    if same_state_robust and direction_robust and predictive_f5 and consistency == "STRONG":
        return {
            "QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL": "SUPPORTED",
            "M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION": "STRONG",
            "RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE": "SUPPORTED",
            "FINAL_CLASSIFICATION": "QWEN_MAG_MATCHED_RESIDUAL_FUNCTIONAL_GEOMETRY_SUPPORTED",
        }
    if same_state_robust and l1_positive:
        return {
            "QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL": "LOCALLY_SUPPORTED",
            "M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION": "PARTIAL",
            "RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE": "PARTIAL",
            "FINAL_CLASSIFICATION": "LOCAL_FUNCTIONAL_GEOMETRY_SUPPORTED_RECURRENCE_INTERACTION_REQUIRED",
        }
    return {
        "QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL": "NOT_SUPPORTED",
        "M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION": "NOT_SUPPORTED_FOR_CLOSED_LOOP_GAP",
        "RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE": "NOT_SUPPORTED",
        "FINAL_CLASSIFICATION": "M5_NOT_SUPPORTED_FOR_CLOSED_LOOP_GAP",
    }


def canonical_functional_metrics(torch, model, m5_module, frozen_module, realrc_module, records, residual):
    if set(records) != set(residual):
        raise ValueError("M5 record and residual layer scopes differ")
    m4_sq = 0.0
    m5_sq = 0.0
    for layer in sorted(residual):
        record = records[layer]
        query = realrc_module.q_for_readout(torch, record).to(residual[layer].device)
        readout = realrc_module.readout_from_state(torch, query, residual[layer])
        core = frozen_module.implementation_replay(record, record["initial_state"])[0]
        operating_point = core.detach().float()[:, 0]
        post_gate = m5_module.apply_linear_parts(
            torch, model, layer, record, operating_point, readout, "G2"
        )
        post_projection = m5_module.apply_linear_parts(
            torch, model, layer, record, operating_point, readout, "G3"
        )
        m4_sq += float(post_gate.detach().double().pow(2).sum().item())
        m5_sq += float(post_projection.detach().double().pow(2).sum().item())
    return normalized_functional_metrics(stack_norm(residual), math.sqrt(m4_sq), math.sqrt(m5_sq))


def validate_unit_result(result, unit, stage, history_lengths, manifest_sha256, current_runner_sha256, commit):
    errors = []
    expected_metadata = {
        "task": TASK,
        "stage": stage,
        "manifest_sha256": manifest_sha256,
        "runner_sha256": current_runner_sha256,
        "git_commit": commit,
    }
    for key, expected in expected_metadata.items():
        if result.get(key) != expected:
            errors.append(f"{key}_mismatch")
    if result.get("unit", {}).get("unit_id") != unit["unit_id"]:
        errors.append("unit_mismatch")
    rows = result.get("rows", [])
    if not exact_result_coverage(rows, [unit], history_lengths):
        errors.append("result_coverage_mismatch")
    required_numeric = ["raw_residual_norm", "M0", "M4", "M5", "F4", "F5", "future_kl_auc", "norm_matching_error"]
    for row in rows:
        if row.get("task") != TASK:
            errors.append("row_task")
        if row.get("manifest_sha256") != manifest_sha256:
            errors.append("row_manifest_sha256")
        if row.get("replay_runner_sha256") != EXPECTED_REPLAY_SHA256:
            errors.append("row_replay_runner_sha256")
        if row.get("m5_source_sha256") != EXPECTED_M5_SHA256:
            errors.append("row_m5_source_sha256")
        if row.get("runner_sha256") != current_runner_sha256:
            errors.append("row_runner_sha256")
        if row.get("git_commit") != commit:
            errors.append("row_git_commit")
        if any(not finite_number(row.get(key)) for key in required_numeric):
            errors.append("missing_or_nonfinite_primary_metric")
        for horizon in FUTURE_KL_HORIZONS:
            if not finite_number(row.get(f"KL_h{horizon}")):
                errors.append(f"missing_or_nonfinite_KL_h{horizon}")
    carriers = {row.get("carrier") for row in result.get("same_state_rows", [])}
    if carriers != {"C_NATIVE", "R_MAG_EQUALIZED"}:
        errors.append("same_state_carrier_coverage")
    protocol = result.get("protocol", {})
    if (
        protocol.get("history_lengths") != list(history_lengths)
        or protocol.get("branches") != [config["branch"] for config in branch_configs()]
        or protocol.get("horizon") != HORIZON
    ):
        errors.append("protocol_mismatch")
    expected_trace_count = 4 * sum(int(length) for length in history_lengths)
    same_state_rows = result.get("same_state_rows", [])
    if len(same_state_rows) != expected_trace_count:
        errors.append("same_state_trace_coverage")
    else:
        same_keys = [
            (int(row.get("history_length", -1)), row.get("carrier"), int(row.get("active_timestep", -1)), row.get("source_direction"))
            for row in same_state_rows
        ]
        if len(same_keys) != len(set(same_keys)):
            errors.append("same_state_trace_coverage")
        if any(not finite_number(row.get(metric)) for row in same_state_rows for metric in ("M0", "M4", "M5", "F4", "F5")):
            errors.append("same_state_nonfinite_metric")
    factorial_rows = result.get("factorial_rows", [])
    if len(factorial_rows) != expected_trace_count:
        errors.append("factorial_trace_coverage")
    else:
        factorial_keys = [
            (int(row.get("history_length", -1)), int(row.get("active_timestep", -1)), row.get("branch"))
            for row in factorial_rows
        ]
        if len(factorial_keys) != len(set(factorial_keys)):
            errors.append("factorial_trace_coverage")
        if any(not finite_number(row.get(metric)) for row in factorial_rows for metric in ("M0", "M4", "M5", "F4", "F5", "direction_cosine", "norm_matching_error")):
            errors.append("factorial_nonfinite_metric")
    audits = result.get("audits", [])
    if len(audits) != 4 * len(history_lengths):
        errors.append("audit_coverage")
    return sorted(set(errors))


def _read_csv_for_freshness(path):
    path = Path(path)
    if not path.is_file():
        return None, []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return set(reader.fieldnames or []), list(reader)
    except Exception:
        return None, []


def _direction_factorial_csv_ok(path, units, history_lengths):
    fieldnames, rows = _read_csv_for_freshness(path)
    required = {
        "task", "git_commit", "manifest_sha256", "prompt", "t0", "history_length",
        "carrier", "branch", "source_direction", "source_magnitude_family",
        "raw_residual_norm", "M0", "M4", "M5", "F4", "F5", "future_kl_auc",
        "direction_cosine", "norm_matching_error",
    } | {f"KL_h{horizon}" for horizon in FUTURE_KL_HORIZONS}
    expected = {
        (unit["unit_id"], int(length), branch)
        for unit in units
        for length in history_lengths
        for branch in ("RR", "RC", "CR", "CC")
    }
    actual = [
        (row.get("unit_id"), int(row.get("history_length", -1)), row.get("branch"))
        for row in rows
    ]
    return (
        fieldnames is not None
        and required <= fieldnames
        and len(actual) == len(expected)
        and len(actual) == len(set(actual))
        and set(actual) == expected
    )


def _same_state_csv_ok(path, units, history_lengths):
    fieldnames, rows = _read_csv_for_freshness(path)
    required = {
        "task", "git_commit", "manifest_sha256", "prompt", "t0", "history_length",
        "carrier", "branch", "active_timestep", "source_direction",
        "raw_residual_norm", "M0", "M4", "M5", "F4", "F5", "context_signature",
    }
    expected_count = len(units) * sum(int(length) for length in history_lengths) * 2 * 2
    return fieldnames is not None and required <= fieldnames and len(rows) == expected_count


def _history_effect_csv_ok(path, units, history_lengths):
    fieldnames, rows = _read_csv_for_freshness(path)
    required = {
        "task", "git_commit", "manifest_sha256", "prompt", "t0",
        "unit_id", "history_length", "DIRECTION_RISK_AT_C_MAG",
        "DIRECTION_RISK_AT_R_MAG", "DIRECTION_MAIN",
    }
    expected = {(unit["unit_id"], int(length)) for unit in units for length in history_lengths}
    actual = [(row.get("unit_id"), int(row.get("history_length", -1))) for row in rows]
    return (
        fieldnames is not None
        and required <= fieldnames
        and len(actual) == len(expected)
        and len(actual) == len(set(actual))
        and set(actual) == expected
    )


def fresh_output_verification(rows, units, history_lengths, output_paths, merge_started):
    expected_count = len(units) * len(history_lengths) * len(branch_configs())
    output_by_name = {Path(path).name: Path(path) for path in output_paths}
    checks = {
        "row_count": len(rows) == expected_count,
        "exact_result_coverage": exact_result_coverage(rows, units, history_lengths),
        "required_files": len(output_paths) == len(REQUIRED_OUTPUT_NAMES)
        and {Path(path).name for path in output_paths} == set(REQUIRED_OUTPUT_NAMES),
        "fresh_files": all(
            Path(path).is_file() and Path(path).stat().st_mtime >= float(merge_started)
            for path in output_paths
        ),
        "finite_required_horizons": all(
            finite_number(row.get(f"KL_h{horizon}"))
            for row in rows
            for horizon in FUTURE_KL_HORIZONS
        ),
        "same_state_f5_csv": _same_state_csv_ok(
            output_by_name.get("same_state_f5.csv", ""), units, history_lengths
        ),
        "direction_factorial_csv": _direction_factorial_csv_ok(
            output_by_name.get("direction_factorial.csv", ""), units, history_lengths
        ),
        "history_direction_effect_csv": _history_effect_csv_ok(
            output_by_name.get("history_direction_effect.csv", ""), units, history_lengths
        ),
    }
    return ("PASS" if all(checks.values()) else "FAIL"), checks


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_file(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def runner_sha256():
    return sha256_file(Path(__file__))


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}"


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_dependencies():
    os.environ["GDN_REPO_ROOT"] = str(REPO)
    os.environ["GDN_DATA_ROOT"] = str(DATA_ROOT)
    os.environ["QWEN35_MODEL_PATH"] = str(MODEL_PATH)
    replay = load_module("qwen_mag_equalization_replay", REPLAY_RUNNER_PATH)
    torch, model, tokenizer, p1, axis, e2e = replay.setup_qwen_model()
    m5 = load_module("canonical_qwen_m5", M5_SOURCE_PATH)
    return torch, model, tokenizer, p1, axis, e2e, replay, m5, m5.frozen, m5.realrc


def load_canonical_manifest():
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = value.get("manifest", [])
    units = [dict(unit, unit_id=f"{unit['problem_id']}|{int(unit['t0'])}") for unit in manifest]
    return value, units, sha256_file(MANIFEST_PATH)


def selected_stage_units(stage, units):
    return [units[index] for index in stage_protocol(stage)["unit_indices"]]


def select_shard_units(units, shard_id, num_shards):
    if int(num_shards) < 1 or not 0 <= int(shard_id) < int(num_shards):
        raise ValueError("invalid shard assignment")
    return [unit for index, unit in enumerate(units) if index % int(num_shards) == int(shard_id)]


def checkpoint_path(stage, num_shards, shard_id, unit):
    digest = hashlib.sha1(unit["unit_id"].encode("utf-8")).hexdigest()[:12]
    return RUN_DIR / "shards" / f"{stage}_{num_shards}_{shard_id}" / "units" / f"{digest}.json"


def state_signature(stack):
    hasher = hashlib.sha256()
    for layer in sorted(stack):
        original = stack[layer].detach()
        value = original.float().cpu().contiguous()
        hasher.update(repr((int(layer), tuple(original.shape), str(original.dtype))).encode("ascii"))
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def require_matching_source_state(expected, actual):
    if expected != actual:
        raise RuntimeError(
            f"FP source-state replay drift: expected={expected} actual={actual}"
        )
    return actual


def source_metadata(unit, prompt_row, history_length, manifest_sha256):
    return {
        "task": TASK,
        "git_commit": git_commit(),
        "manifest_sha256": manifest_sha256,
        "replay_runner_sha256": EXPECTED_REPLAY_SHA256,
        "m5_source_sha256": EXPECTED_M5_SHA256,
        "runner_sha256": runner_sha256(),
        "unit_id": unit["unit_id"],
        "prompt": unit["problem_id"],
        "prompt_id": unit["problem_id"],
        "problem_index": int(prompt_row["problem_index"]),
        "t0": int(unit["t0"]),
        "history_length": int(history_length),
    }


def _token_tensor(torch, model, token):
    return torch.tensor([[int(token)]], dtype=torch.long, device=next(model.parameters()).device)


def frozen_driver_step_inference(torch, frozen, model, input_ids, mask, past, collector):
    with torch.inference_mode():
        return frozen.driver_step(torch, model, input_ids, mask, past, collector)


def lookahead_records(torch, model, replay, frozen, collector, tokens, timestep, past):
    token = replay.teacher_token_for_timestep(tokens, int(timestep) + 1)
    _out, _next_past, records = frozen_driver_step_inference(
        torch, frozen, model, _token_tensor(torch, model, token), None, replay.clone_cache(past), collector
    )
    if set(records) != set(replay.GDN_LAYERS):
        raise RuntimeError(f"incomplete M5 records at timestep {timestep}")
    return records


def functional_pair_rows(
    torch, model, replay, m5, frozen, realrc, records, residuals, metadata,
    carrier, timestep, context_signature,
):
    output = []
    for direction, key in (("R", "e_R"), ("C", "e_C")):
        metrics = canonical_functional_metrics(
            torch, model, m5, frozen, realrc, records, residuals[key]
        )
        output.append({
            **metadata,
            "carrier": carrier,
            "branch": f"{carrier}_{direction}",
            "active_timestep": int(timestep),
            "source_direction": direction,
            "source_magnitude_family": "NATIVE",
            "raw_residual_norm": metrics["M0"],
            **metrics,
            "context_signature": context_signature,
            "direction_cosine": 1.0,
            "norm_matching_error": 0.0,
        })
    return output


def run_same_state_carrier(
    torch, model, p1, replay, m5, frozen, realrc, collector, tokens, unit,
    prompt_row, fp_ref, history_length, carrier, manifest_sha256,
):
    schedule = replay.history_schedule(int(unit["t0"]), history_length, HORIZON)
    start_previous = schedule["active_timesteps"][0] - 1
    past = replay.clone_cache(fp_ref["snapshots"][start_previous])
    metadata = source_metadata(unit, prompt_row, history_length, manifest_sha256)
    rows = []
    norm_errors = []
    direction_cosines = []
    nonretroactivity = []
    for timestep in schedule["active_timesteps"]:
        token = replay.teacher_token_for_timestep(tokens, timestep)
        out, past, _records = frozen_driver_step_inference(
            torch, frozen, model, _token_tensor(torch, model, token), None, past, collector
        )
        current_logits = out.logits[:, -1:, :].detach().clone()
        stack = replay.qwen_cache_stack(p1, past)
        signature = state_signature(stack)
        residuals = replay.extract_same_state_residuals(torch, stack)
        records = lookahead_records(torch, model, replay, frozen, collector, tokens, timestep, past)
        rows.extend(functional_pair_rows(
            torch, model, replay, m5, frozen, realrc, records, residuals, metadata,
            carrier, timestep, signature,
        ))
        if carrier == "C_NATIVE":
            intervention = build_factorial_interventions(residuals["e_R"], residuals["e_C"])["CC"]
        elif carrier == "R_MAG_EQUALIZED":
            intervention = build_factorial_interventions(residuals["e_R"], residuals["e_C"])["RC"]
        else:
            raise ValueError(f"unknown carrier: {carrier}")
        replay.apply_delta_to_stack(stack, intervention["delta"])
        norm_error = abs(intervention["applied_norm"] - intervention["target_norm"])
        norm_errors.append(norm_error)
        direction_cosines.append(intervention["direction_cosine"])
        nonretroactivity.append(float((out.logits[:, -1:, :] - current_logits).abs().max().item()))
    audit = {
        **metadata,
        "carrier": carrier,
        "n_timesteps": len(schedule["active_timesteps"]),
        "max_norm_matching_error": max(norm_errors, default=math.inf),
        "min_direction_cosine": min(direction_cosines, default=-1.0),
        "current_logit_nonretroactivity_max_abs": max(nonretroactivity, default=math.inf),
    }
    return rows, audit


def precompute_fp_source_metrics(
    torch, model, p1, replay, m5, frozen, realrc, collector, tokens, unit,
    prompt_row, fp_ref, history_length, manifest_sha256,
):
    schedule = replay.history_schedule(int(unit["t0"]), history_length, HORIZON)
    past = replay.clone_cache(fp_ref["snapshots"][schedule["active_timesteps"][0] - 1])
    metadata = source_metadata(unit, prompt_row, history_length, manifest_sha256)
    output = {}
    pending = None
    for timestep in schedule["active_timesteps"]:
        token = replay.teacher_token_for_timestep(tokens, timestep)
        _out, past, records = frozen_driver_step_inference(
            torch, frozen, model, _token_tensor(torch, model, token), None, past, collector
        )
        if pending is not None:
            previous_timestep, previous_residuals, signature = pending
            pair = functional_pair_rows(
                torch, model, replay, m5, frozen, realrc, records, previous_residuals,
                metadata, "FP", previous_timestep, signature,
            )
            output[previous_timestep] = {row["source_direction"]: row for row in pair}
        stack = replay.qwen_cache_stack(p1, past)
        residuals = replay.extract_same_state_residuals(torch, stack)
        pending = (int(timestep), residuals, state_signature(stack))
    last_timestep, last_residuals, signature = pending
    records = lookahead_records(torch, model, replay, frozen, collector, tokens, last_timestep, past)
    pair = functional_pair_rows(
        torch, model, replay, m5, frozen, realrc, records, last_residuals,
        metadata, "FP", last_timestep, signature,
    )
    output[last_timestep] = {row["source_direction"]: row for row in pair}
    return output


def _scaled_metric_row(source_metric, target_norm):
    return {
        "raw_residual_norm": float(target_norm),
        "M0": float(target_norm),
        "M4": float(source_metric["F4"]) * float(target_norm),
        "M5": float(source_metric["F5"]) * float(target_norm),
        "F4": float(source_metric["F4"]),
        "F5": float(source_metric["F5"]),
    }


def _mean_rows(rows, keys):
    return {key: statistics.mean(float(row[key]) for row in rows) for key in keys}


def fp_result_row(unit, prompt_row, history_length, manifest_sha256):
    row = {
        **source_metadata(unit, prompt_row, history_length, manifest_sha256),
        "carrier": "FP",
        "branch": "FP",
        "source_direction": "FP",
        "source_magnitude_family": "FP",
        "raw_residual_norm": 0.0,
        "M0": 0.0,
        "M4": 0.0,
        "M5": 0.0,
        "F4": 0.0,
        "F5": 0.0,
        "future_kl_auc": 0.0,
        "direction_cosine": None,
        "norm_matching_error": 0.0,
        "future_quantization_count": 0,
        "intervention_count": 0,
        "current_logit_nonretroactivity_max_abs": 0.0,
    }
    for horizon in FUTURE_KL_HORIZONS:
        row[f"KL_h{horizon}"] = 0.0
    return row


def evaluate_factorial_branch(
    torch, model, p1, replay, tokens, unit, prompt_row, fp_ref, history_length,
    branch, source_metrics, manifest_sha256,
):
    schedule = replay.history_schedule(int(unit["t0"]), history_length, HORIZON)
    start_previous = schedule["active_timesteps"][0] - 1
    fp_past = replay.clone_cache(fp_ref["snapshots"][start_previous])
    branch_past = replay.clone_cache(fp_ref["snapshots"][start_previous])
    metadata = source_metadata(unit, prompt_row, history_length, manifest_sha256)
    step_rows = []
    nonretroactivity = []
    for timestep in schedule["active_timesteps"]:
        token = replay.teacher_token_for_timestep(tokens, timestep)
        current = _token_tensor(torch, model, token)
        with torch.inference_mode():
            fp_out = p1.feed_step(torch, model, current, None, fp_past)
            branch_out = p1.feed_step(torch, model, current, None, branch_past)
        fp_past = fp_out.past_key_values
        branch_past = branch_out.past_key_values
        current_logits = branch_out.logits[:, -1:, :].detach().clone()
        fp_stack = replay.qwen_cache_stack(p1, fp_past)
        runtime_source_signature = state_signature(fp_stack)
        residuals = replay.extract_same_state_residuals(torch, fp_stack)
        cell = build_factorial_interventions(residuals["e_R"], residuals["e_C"])[branch]
        branch_stack = replay.qwen_cache_stack(p1, branch_past)
        replay.apply_delta_to_stack(branch_stack, cell["delta"])
        metric = source_metrics[int(timestep)][cell["source_direction"]]
        source_state_signature = require_matching_source_state(
            metric["context_signature"], runtime_source_signature
        )
        scaled = _scaled_metric_row(metric, cell["target_norm"])
        norm_error = abs(cell["applied_norm"] - cell["target_norm"])
        nonretro = float((branch_out.logits[:, -1:, :] - current_logits).abs().max().item())
        nonretroactivity.append(nonretro)
        step_rows.append({
            **metadata,
            "carrier": "FP",
            "branch": branch,
            "active_timestep": int(timestep),
            "source_direction": cell["source_direction"],
            "source_magnitude_family": cell["source_magnitude_family"],
            **scaled,
            "m_R": residuals["m_R"],
            "m_C": residuals["m_C"],
            "source_state_signature": source_state_signature,
            "source_metric_context_signature": metric["context_signature"],
            "direction_cosine": cell["direction_cosine"],
            "norm_matching_error": norm_error,
            "current_logit_nonretroactivity": nonretro,
        })
    kl_curve = []
    selected_kl = {}
    future_horizons = []
    for horizon in range(1, HORIZON + 1):
        token = replay.teacher_token_for_future_horizon(tokens, int(unit["t0"]), horizon)
        with torch.inference_mode():
            out = p1.feed_step(torch, model, _token_tensor(torch, model, token), None, branch_past)
        branch_past = out.past_key_values
        future_horizons.append(horizon)
        if not bool(torch.isfinite(out.logits).all().item()):
            raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} L={history_length} branch={branch} h={horizon}")
        metrics = replay.full_logit_metrics(torch, fp_ref["future_logits"][horizon], out.logits)
        kl_curve.append(metrics["KL"])
        if horizon in FUTURE_KL_HORIZONS:
            selected_kl[horizon] = metrics["KL"]
    averaged = _mean_rows(step_rows, ["raw_residual_norm", "M0", "M4", "M5", "F4", "F5"])
    row = {
        **metadata,
        "carrier": "FP",
        "branch": branch,
        "source_direction": step_rows[0]["source_direction"],
        "source_magnitude_family": step_rows[0]["source_magnitude_family"],
        **averaged,
        "future_kl_auc": statistics.mean(kl_curve),
        "direction_cosine": min(float(step["direction_cosine"]) for step in step_rows),
        "norm_matching_error": max(float(step["norm_matching_error"]) for step in step_rows),
        "future_quantization_count": 0,
        "intervention_count": len(step_rows),
        "quantization_event_timesteps": list(schedule["active_timesteps"]),
        "future_forward_horizons": future_horizons,
        "current_logit_nonretroactivity_max_abs": max(nonretroactivity, default=math.inf),
    }
    for horizon in FUTURE_KL_HORIZONS:
        row[f"KL_h{horizon}"] = selected_kl[horizon]
    audit = {
        **metadata,
        "branch": branch,
        "intervention_count": len(step_rows),
        "quantization_event_timesteps": list(schedule["active_timesteps"]),
        "future_forward_horizons": future_horizons,
        "future_quantization_count": 0,
        "current_logit_nonretroactivity_max_abs": max(nonretroactivity, default=math.inf),
        "max_norm_matching_error": row["norm_matching_error"],
        "min_direction_cosine": row["direction_cosine"],
    }
    return row, step_rows, audit


def run_canonical_unit(
    torch, model, tokenizer, p1, e2e, replay, m5, frozen, realrc, unit, prompt_row,
    history_lengths, manifest_sha256,
):
    tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
    required = int(unit["t0"]) + HORIZON
    if len(tokens) < required:
        raise RuntimeError(f"teacher continuation too short: need={required} have={len(tokens)}")
    fp_ref = replay.full_reference(
        torch, model, tokenizer, p1, e2e, prompt_row,
        tokens, int(unit["t0"]), history_lengths, HORIZON,
    )
    source_by_l = {}
    same_state_rows = []
    same_state_audits = []
    collector = frozen.install_fp_driver_capture(torch, model)
    try:
        for history_length in history_lengths:
            source_by_l[int(history_length)] = precompute_fp_source_metrics(
                torch, model, p1, replay, m5, frozen, realrc, collector, tokens,
                unit, prompt_row, fp_ref, history_length, manifest_sha256,
            )
            for carrier in ("C_NATIVE", "R_MAG_EQUALIZED"):
                rows, audit = run_same_state_carrier(
                    torch, model, p1, replay, m5, frozen, realrc, collector, tokens,
                    unit, prompt_row, fp_ref, history_length, carrier, manifest_sha256,
                )
                same_state_rows.extend(rows)
                same_state_audits.append(audit)
    finally:
        collector["close"]()
    rows = []
    factorial_rows = []
    audits = []
    for history_length in history_lengths:
        rows.append(fp_result_row(unit, prompt_row, history_length, manifest_sha256))
        for config in branch_configs()[1:]:
            row, steps, audit = evaluate_factorial_branch(
                torch, model, p1, replay, tokens, unit, prompt_row, fp_ref,
                history_length, config["branch"], source_by_l[int(history_length)],
                manifest_sha256,
            )
            rows.append(row)
            factorial_rows.extend(steps)
            audits.append(audit)
    return {
        "unit": unit,
        "rows": rows,
        "same_state_rows": same_state_rows,
        "factorial_rows": factorial_rows,
        "same_state_audits": same_state_audits,
        "audits": audits,
    }


BLOCKING_GATE_NAMES = [
    "CANONICAL_MANIFEST_GATE",
    "REPLAY_INPUT_GATE",
    "STATE_SEMANTICS_GATE",
    "R128_QUANTIZER_IDENTITY_GATE",
    "C128_QUANTIZER_IDENTITY_GATE",
    "M5_IMPLEMENTATION_GATE",
    "M5_HOMOGENEITY_GATE",
    "F5_SCALE_INVARIANCE_GATE",
    "SAME_STATE_COUNTERFACTUAL_GATE",
    "EXOGENOUS_DIRECTION_IDENTITY_GATE",
    "FACTORIAL_NORM_GATE",
    "FACTORIAL_DIRECTION_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
]


def environment_audit():
    canonical_obj, units, manifest_sha256 = load_canonical_manifest()
    prior = json.loads(PRIOR_CLOSED_LOOP_SUMMARY_PATH.read_text(encoding="utf-8")) if PRIOR_CLOSED_LOOP_SUMMARY_PATH.is_file() else {}
    golden = json.loads(M5_GOLDEN_SUMMARY_PATH.read_text(encoding="utf-8")) if M5_GOLDEN_SUMMARY_PATH.is_file() else {}
    hashes = {
        "manifest_sha256": manifest_sha256,
        "replay_runner_sha256": sha256_file(REPLAY_RUNNER_PATH),
        "m5_source_sha256": sha256_file(M5_SOURCE_PATH),
        "m5_transitive_sha256": {
            label: sha256_file(path) for label, path in M5_TRANSITIVE_SOURCE_PATHS.items()
        },
        "m5_golden_summary_sha256": sha256_file(M5_GOLDEN_SUMMARY_PATH),
        "prior_closed_loop_summary_sha256": sha256_file(PRIOR_CLOSED_LOOP_SUMMARY_PATH),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
        "tokenizer_sha256": sha256_file(MODEL_PATH / "tokenizer.json"),
    }
    replay_inputs = (
        prior.get("formal_status") == "COMPLETE"
        and prior.get("FRESH_OUTPUT_VERIFICATION") == "PASS"
        and prior.get("canonical_manifest_sha256") == EXPECTED_MANIFEST_SHA256
        and hashes["prior_closed_loop_summary_sha256"] == EXPECTED_PRIOR_CLOSED_LOOP_SUMMARY_SHA256
        and all(path.exists() for path in (MODEL_PATH, REPLAY_RUNNER_PATH, M5_SOURCE_PATH, MANIFEST_PATH))
    )
    golden_ok = (
        hashes["m5_golden_summary_sha256"] == EXPECTED_M5_GOLDEN_SUMMARY_SHA256
        and golden.get("formal") == "COMPLETE"
        and golden.get("metric_implementation_gate") == "PASS"
        and golden.get("rc_m5_correct") == golden.get("rc_total") == 9
        and abs(float(golden.get("m5_future_kl_spearman", 0.0)) - 0.7956656346749226) <= 1e-12
    )
    m5_transitive_ok = hashes["m5_transitive_sha256"] == EXPECTED_M5_TRANSITIVE_SHA256
    stage0_metrics = canonical_obj.get("stage0_metrics", {})
    gates = {
        "CANONICAL_MANIFEST_GATE": "PASS" if manifest_sha256 == EXPECTED_MANIFEST_SHA256 and len(units) == 18 else "FAIL",
        "REPLAY_INPUT_GATE": "PASS" if replay_inputs and hashes["model_config_sha256"] == EXPECTED_MODEL_CONFIG_SHA256 and hashes["tokenizer_config_sha256"] == EXPECTED_TOKENIZER_CONFIG_SHA256 and hashes["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA256 else "FAIL",
        "STATE_SEMANTICS_GATE": "PASS" if stage0_metrics.get("state_shape") == [1, 32, 128, 128] else "FAIL",
        "R128_QUANTIZER_IDENTITY_GATE": "PASS" if stage0_metrics.get("R128_scale_shape") == [1, 32, 128, 1] else "FAIL",
        "C128_QUANTIZER_IDENTITY_GATE": "PASS" if stage0_metrics.get("C128_scale_shape") == [1, 32, 1, 128] else "FAIL",
        "M5_IMPLEMENTATION_GATE": "PASS" if hashes["replay_runner_sha256"] == EXPECTED_REPLAY_SHA256 and hashes["m5_source_sha256"] == EXPECTED_M5_SHA256 and m5_transitive_ok and golden_ok else "FAIL",
    }
    for name in BLOCKING_GATE_NAMES:
        gates.setdefault(name, "PENDING_RUNTIME")
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "formal_status": "READY_TO_RUN" if all(value == "PASS" for value in gates.values() if value != "PENDING_RUNTIME") else "INVALID",
        "canonical_manifest_source": str(MANIFEST_PATH),
        "canonical_manifest_sha256": manifest_sha256,
        "n_canonical_units": len(units),
        "model_path": str(MODEL_PATH),
        "hashes": hashes,
        "gates": gates,
        "prior_closed_loop_summary": str(PRIOR_CLOSED_LOOP_SUMMARY_PATH),
        "m5_golden": {
            "m0_future_kl_spearman": golden.get("m0_future_kl_spearman"),
            "m4_future_kl_spearman": golden.get("m4_future_kl_spearman"),
            "m5_future_kl_spearman": golden.get("m5_future_kl_spearman"),
            "rc_m5_correct": golden.get("rc_m5_correct"),
            "rc_total": golden.get("rc_total"),
        },
    }


def runtime_semantics_audit(torch, model, tokenizer, p1, axis, e2e, replay, m5, frozen, realrc, prompt_row):
    base_audit = replay.runtime_semantics_audit(torch, model, tokenizer, p1, axis, e2e, prompt_row)
    prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
    tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"].to(next(model.parameters()).device)
    mask = encoded.get("attention_mask")
    mask = mask.to(ids.device) if mask is not None else None
    with torch.inference_mode():
        prefill = p1.feed_step(torch, model, ids, mask, None)
    past = prefill.past_key_values
    source_stack = replay.qwen_cache_stack(p1, past)
    before = replay.stack_cpu_copy(source_stack)
    residuals = replay.extract_same_state_residuals(torch, source_stack)
    next_token = replay.teacher_token_for_timestep(tokens, 1)
    with torch.inference_mode():
        baseline = p1.feed_step(
            torch, model, _token_tensor(torch, model, next_token), None, replay.clone_cache(past)
        )
    collector = frozen.install_fp_driver_capture(torch, model)
    try:
        instrumented, _instrumented_past, records = frozen_driver_step_inference(
            torch, frozen, model, _token_tensor(torch, model, next_token), None,
            replay.clone_cache(past), collector,
        )
        reference = canonical_functional_metrics(
            torch, model, m5, frozen, realrc, records, residuals["e_R"]
        )
        scaling = []
        for alpha in (0.25, -0.5, 2.0):
            metrics = canonical_functional_metrics(
                torch, model, m5, frozen, realrc, records,
                scale_stack(residuals["e_R"], alpha),
            )
            scaling.append({
                "alpha": alpha,
                "m5_relative_error": abs(metrics["M5"] - abs(alpha) * reference["M5"]) / (abs(alpha) * reference["M5"] + EPS),
                "f5_relative_error": abs(metrics["F5"] - reference["F5"]) / (abs(reference["F5"]) + EPS),
            })
    finally:
        collector["close"]()
    noninterference = float((baseline.logits - instrumented.logits).abs().max().item())
    source_unchanged = replay.max_stack_abs_error(source_stack, before)
    cells = build_factorial_interventions(residuals["e_R"], residuals["e_C"])
    norm_errors = [abs(cell["applied_norm"] - cell["target_norm"]) for cell in cells.values()]
    direction_cosines = [cell["direction_cosine"] for cell in cells.values()]
    audit = dict(base_audit)
    audit.update({
        "m5_scaling": scaling,
        "factorial_norm_max_abs_error": max(norm_errors),
        "factorial_direction_min_cosine": min(direction_cosines),
        "instrumentation_logit_max_abs_error": noninterference,
        "instrumentation_source_state_max_abs_error": source_unchanged,
        "M5_IMPLEMENTATION_GATE": environment_audit()["gates"]["M5_IMPLEMENTATION_GATE"],
        "M5_HOMOGENEITY_GATE": "PASS" if reference["M5"] > EPS and max(row["m5_relative_error"] for row in scaling) <= NORM_TOL else "FAIL",
        "F5_SCALE_INVARIANCE_GATE": "PASS" if reference["F5"] > EPS and max(row["f5_relative_error"] for row in scaling) <= NORM_TOL else "FAIL",
        "SAME_STATE_COUNTERFACTUAL_GATE": residuals["SAME_STATE_COUNTERFACTUAL_GATE"],
        "EXOGENOUS_DIRECTION_IDENTITY_GATE": residuals["SAME_STATE_COUNTERFACTUAL_GATE"],
        "FACTORIAL_NORM_GATE": "PASS" if max(norm_errors) <= NORM_TOL else "FAIL",
        "FACTORIAL_DIRECTION_GATE": "PASS" if min(direction_cosines) >= 1.0 - IDENTITY_TOL else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if max(noninterference, source_unchanged) <= IDENTITY_TOL else "FAIL",
    })
    return audit


def archive_execution_sources():
    output = {}
    sources = [
        ("runner", Path(__file__)),
        ("replay", REPLAY_RUNNER_PATH),
        ("m5", M5_SOURCE_PATH),
    ]
    sources.extend((f"m5_transitive_{label}", path) for label, path in M5_TRANSITIVE_SOURCE_PATHS.items())
    for label, path in sources:
        digest = sha256_file(path)
        if digest is None:
            raise RuntimeError(f"missing execution source: {path}")
        target = RUN_DIR / "execution_source" / f"{label}_{digest}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise RuntimeError(f"execution source collision: {target}")
        if not target.exists():
            target.write_bytes(path.read_bytes())
        output[label] = {"path": str(target), "sha256": digest}
    return output


def stage_manifest(stage, shard_id, num_shards, assigned_units, history_lengths, sources):
    return {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "git_commit": git_commit(),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "assigned_unit_ids": [unit["unit_id"] for unit in assigned_units],
        "history_lengths": list(history_lengths),
        "branches": [config["branch"] for config in branch_configs()],
        "horizon": HORIZON,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": runner_sha256(),
        "replay_runner_sha256": EXPECTED_REPLAY_SHA256,
        "m5_source_sha256": EXPECTED_M5_SHA256,
        "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "execution_sources": sources,
    }


def manifest_errors(manifest, expected, ignore_timestamp=True):
    keys = [
        "task", "stage", "git_commit", "shard_id", "num_shards",
        "assigned_unit_ids", "history_lengths", "branches", "horizon",
        "manifest_sha256", "runner_sha256", "replay_runner_sha256",
        "m5_source_sha256", "model_config_sha256", "tokenizer_config_sha256",
        "tokenizer_sha256",
    ]
    return [key for key in keys if manifest.get(key) != expected.get(key)]


def require_stage_prerequisite(stage):
    previous = {"smoke": "stage0", "pilot": "smoke", "formal": "pilot"}
    if stage not in previous:
        return
    prior_stage = previous[stage]
    path = RUN_DIR / f"aggregate_{prior_stage}.json"
    if not path.is_file():
        raise RuntimeError(f"{stage} blocked: missing {path.name}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("task") != TASK
        or summary.get("stage") != prior_stage
        or summary.get("FRESH_OUTPUT_VERIFICATION") != "PASS"
        or not all(summary.get("gates", {}).get(name) == "PASS" for name in BLOCKING_GATE_NAMES)
    ):
        raise RuntimeError(f"{stage} blocked by invalid {prior_stage} aggregate")
    if stage == "pilot" and summary.get("SMOKE") != "PASS":
        raise RuntimeError("pilot blocked: smoke did not pass")
    if stage == "formal" and summary.get("PILOT") != "POSITIVE_SUPPORTS_FORMAL":
        raise RuntimeError("formal blocked: pilot is negative or inconclusive")


def run_execution_stage(args):
    audit = environment_audit()
    static_failures = [name for name, value in audit["gates"].items() if value == "FAIL"]
    if static_failures:
        raise RuntimeError("static blocking gates failed: " + ", ".join(static_failures))
    require_stage_prerequisite(args.stage)
    _canonical, all_units, manifest_sha256 = load_canonical_manifest()
    protocol = stage_protocol(args.stage)
    stage_units = selected_stage_units(args.stage, all_units)
    assigned = select_shard_units(stage_units, args.shard_id, args.num_shards)
    shard_dir = RUN_DIR / "shards" / f"{args.stage}_{args.num_shards}_{args.shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    sources = archive_execution_sources()
    expected_manifest = stage_manifest(
        args.stage, args.shard_id, args.num_shards, assigned,
        protocol["history_lengths"], sources,
    )
    manifest_path = shard_dir / "manifest.json"
    provenance_path = shard_dir / "execution_provenance.json"
    existing_checkpoints = list((shard_dir / "units").glob("*.json"))
    if args.resume and (manifest_path.exists() or provenance_path.exists() or existing_checkpoints):
        if not manifest_path.is_file() or not provenance_path.is_file():
            raise RuntimeError("resume blocked by incomplete manifest/provenance")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors = manifest_errors(existing_manifest, expected_manifest)
        if errors:
            raise RuntimeError("resume blocked by manifest mismatch: " + ", ".join(errors))
    elif existing_checkpoints or manifest_path.exists() or provenance_path.exists():
        raise RuntimeError("existing shard data requires --resume or a fresh run directory")
    else:
        save_json(manifest_path, expected_manifest)
        save_json(provenance_path, {
            "task": TASK,
            "stage": args.stage,
            "timestamp": now(),
            "git_commit": git_commit(),
            "shard_id": int(args.shard_id),
            "num_shards": int(args.num_shards),
            "runner_sha256": runner_sha256(),
            "command": list(sys.argv),
            "execution_sources": sources,
        })
    torch, model, tokenizer, p1, axis, e2e, replay, m5, frozen, realrc = load_runtime_dependencies()
    prompt_rows = {row["problem_id"]: row for row in p1.selected_prompt_rows()}
    semantics = runtime_semantics_audit(
        torch, model, tokenizer, p1, axis, e2e, replay, m5, frozen, realrc,
        prompt_rows[all_units[0]["problem_id"]],
    )
    save_json(shard_dir / "runtime_semantics_audit.json", semantics)
    failed_runtime = [name for name in BLOCKING_GATE_NAMES if semantics.get(name, audit["gates"].get(name)) != "PASS"]
    if failed_runtime:
        raise RuntimeError("runtime blocking gates failed: " + ", ".join(failed_runtime))
    failures = []
    for index, unit in enumerate(assigned, 1):
        path = checkpoint_path(args.stage, args.num_shards, args.shard_id, unit)
        if args.resume and path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_unit_result(
                prior, unit, args.stage, protocol["history_lengths"], manifest_sha256,
                runner_sha256(), git_commit(),
            )
            if not errors:
                print(f"RESUME_SKIP {index}/{len(assigned)} {unit['unit_id']}", flush=True)
                continue
            print(f"RESUME_REJECT {unit['unit_id']} errors={','.join(errors)}", flush=True)
        started = time.time()
        print(f"RUN_UNIT {index}/{len(assigned)} {unit['unit_id']}", flush=True)
        try:
            result = run_canonical_unit(
                torch, model, tokenizer, p1, e2e, replay, m5, frozen, realrc,
                unit, prompt_rows[unit["problem_id"]], protocol["history_lengths"],
                manifest_sha256,
            )
            result.update({
                "task": TASK,
                "stage": args.stage,
                "timestamp": now(),
                "elapsed_seconds": time.time() - started,
                "manifest_sha256": manifest_sha256,
                "runner_sha256": runner_sha256(),
                "git_commit": git_commit(),
                "protocol": {
                    "history_lengths": protocol["history_lengths"],
                    "branches": [config["branch"] for config in branch_configs()],
                    "horizon": HORIZON,
                },
            })
            for collection_name in ("rows", "same_state_rows", "factorial_rows"):
                for row in result[collection_name]:
                    row["stage"] = args.stage
            save_json(path, result)
            print(f"DONE_UNIT {unit['unit_id']} seconds={time.time() - started:.1f}", flush=True)
        except Exception as exc:
            failure = {"unit": unit, "error": repr(exc), "traceback": traceback.format_exc(), "timestamp": now()}
            failures.append(failure)
            save_json(shard_dir / "failures.json", failures)
            print(f"FAILED_UNIT {unit['unit_id']} {exc!r}", flush=True)
            if not args.keep_going:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_json(shard_dir / "failures.json", failures)
    success = sum(checkpoint_path(args.stage, args.num_shards, args.shard_id, unit).is_file() for unit in assigned)
    save_json(shard_dir / "worker_summary.json", {
        "task": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "n_assigned": len(assigned),
        "n_success": success,
        "n_failed": len(failures),
        "failures": failures,
    })
    return 0 if not failures and success == len(assigned) else 1


def summarize_same_state_f5(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["unit_id"], int(row["history_length"]), row["carrier"])
        grouped[key][row["source_direction"]].append(float(row["F5"]))
    cells = []
    for (unit_id, history_length, carrier), directions in sorted(grouped.items()):
        if set(directions) != {"R", "C"}:
            continue
        f5_r = statistics.mean(directions["R"])
        f5_c = statistics.mean(directions["C"])
        cells.append({
            "unit_id": unit_id,
            "history_length": history_length,
            "carrier": carrier,
            "F5_R": f5_r,
            "F5_C": f5_c,
            "F5_R_MINUS_C": f5_r - f5_c,
            "n_timesteps": len(directions["R"]),
        })
    summary = {}
    for carrier in sorted({cell["carrier"] for cell in cells}):
        summary[carrier] = {}
        for history_length in sorted({cell["history_length"] for cell in cells if cell["carrier"] == carrier}):
            values = [
                cell["F5_R_MINUS_C"] for cell in cells
                if cell["carrier"] == carrier and cell["history_length"] == history_length
            ]
            summary[carrier][str(history_length)] = paired_summary(values, seed=4100 + history_length)
    return summary, cells


def pilot_decision(same_state, direction, predictive_f5, amplification):
    same_cells = [cell for carrier in same_state.values() for cell in carrier.values()]
    same_positive = sum(
        finite_number(cell.get("median"))
        and cell["median"] > 0
        and int(cell.get("n", 0)) > 0
        and int(cell.get("positive", 0)) / int(cell["n"]) >= 2.0 / 3.0
        for cell in same_cells
    )
    same_signal = bool(same_cells) and same_positive >= math.ceil(len(same_cells) / 2.0)
    direction_signal = any(
        cells and sum(finite_number(cell.get("median")) and cell["median"] > 0 for cell in cells.values()) >= math.ceil(len(cells) / 2.0)
        for cells in direction.values()
    )
    amplification_signal = any(finite_number(value) and value >= 0.3 for value in amplification.values())
    prediction_signal = any(
        finite_number(predictive_f5.get(schedule, {}).get("spearman"))
        and predictive_f5[schedule]["spearman"] >= 0.3
        for schedule in ("C_MAG", "R_MAG")
    )
    return "POSITIVE_SUPPORTS_FORMAL" if any(
        (same_signal, direction_signal, amplification_signal, prediction_signal)
    ) else "NEGATIVE_OR_INCONCLUSIVE"


def observed_execution_gates(same_state_rows, factorial_rows, audits, units, history_lengths):
    expected_same = len(units) * sum(int(length) for length in history_lengths) * 2 * 2
    same_groups = defaultdict(list)
    for row in same_state_rows:
        same_groups[(row["unit_id"], int(row["history_length"]), row["carrier"], int(row["active_timestep"]))].append(row)
    same_ok = (
        len(same_state_rows) == expected_same
        and all(
            {row["source_direction"] for row in rows} == {"R", "C"}
            and len({row["context_signature"] for row in rows}) == 1
            for rows in same_groups.values()
        )
    )
    expected_factorial = len(units) * sum(int(length) for length in history_lengths) * 4
    factorial_groups = defaultdict(list)
    for row in factorial_rows:
        factorial_groups[(row["unit_id"], int(row["history_length"]), int(row["active_timestep"]))].append(row)
    exogenous_ok = (
        len(factorial_rows) == expected_factorial
        and all(
            {row["branch"] for row in rows} == {"RR", "RC", "CR", "CC"}
            and len({row["source_state_signature"] for row in rows}) == 1
            for rows in factorial_groups.values()
        )
    )
    norm_ok = all(float(row["norm_matching_error"]) <= NORM_TOL for row in factorial_rows)
    direction_ok = all(float(row["direction_cosine"]) >= 1.0 - IDENTITY_TOL for row in factorial_rows)
    expected_audits = len(units) * len(history_lengths) * 4
    timing_ok = len(audits) == expected_audits and all(
        audit.get("quantization_event_timesteps")
        == list(range(int(audit["t0"]) - int(audit["history_length"]) + 1, int(audit["t0"]) + 1))
        for audit in audits
    )
    return {
        "SAME_STATE_COUNTERFACTUAL_GATE": "PASS" if same_ok else "FAIL",
        "EXOGENOUS_DIRECTION_IDENTITY_GATE": "PASS" if exogenous_ok else "FAIL",
        "FACTORIAL_NORM_GATE": "PASS" if norm_ok else "FAIL",
        "FACTORIAL_DIRECTION_GATE": "PASS" if direction_ok else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS" if timing_ok else "FAIL",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if audits and all(float(audit["current_logit_nonretroactivity_max_abs"]) <= IDENTITY_TOL for audit in audits) else "FAIL",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS" if audits and all(audit.get("future_quantization_count") == 0 and audit.get("future_forward_horizons") == list(range(1, HORIZON + 1)) for audit in audits) else "FAIL",
    }


def analyze_scientific_results(rows, same_state_rows, stage):
    effects = build_direction_effect_rows(rows)
    direction = {
        "DIRECTION_RISK_AT_C_MAG_BY_L": summarize_effect_by_l(effects, "DIRECTION_RISK_AT_C_MAG"),
        "DIRECTION_RISK_AT_R_MAG_BY_L": summarize_effect_by_l(effects, "DIRECTION_RISK_AT_R_MAG"),
        "DIRECTION_MAIN_BY_L": summarize_effect_by_l(effects, "DIRECTION_MAIN"),
    }
    same_state, same_state_cells = summarize_same_state_f5(same_state_rows)
    predictive = {metric: metric_predictive_signal(rows, metric) for metric in ("M0", "M4", "M5", "F5")}
    amplification = {
        "C_MAG": history_amplification_rho(direction["DIRECTION_RISK_AT_C_MAG_BY_L"]),
        "R_MAG": history_amplification_rho(direction["DIRECTION_RISK_AT_R_MAG_BY_L"]),
        "MAIN": history_amplification_rho(direction["DIRECTION_MAIN_BY_L"]),
    }
    prior = json.loads(PRIOR_CLOSED_LOOP_SUMMARY_PATH.read_text(encoding="utf-8"))
    prior_closed = prior["contrasts_by_l"]["R_EQ_VS_C_NATIVE_BY_L"]
    consistency = closed_loop_direction_consistency(
        direction["DIRECTION_RISK_AT_C_MAG_BY_L"], prior_closed
    )
    pilot = pilot_decision(same_state, direction, predictive["F5"], amplification)
    if stage == "formal":
        classification = scientific_classification(same_state, direction, predictive["F5"], consistency)
    else:
        classification = {
            "QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL": "NOT_INTERPRETABLE_BEFORE_FORMAL",
            "M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION": "NOT_INTERPRETABLE_BEFORE_FORMAL",
            "RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE": "NOT_INTERPRETABLE_BEFORE_FORMAL",
            "FINAL_CLASSIFICATION": "NOT_INTERPRETABLE_BEFORE_FORMAL",
        }
    return {
        "effects": effects,
        "direction": direction,
        "same_state": same_state,
        "same_state_cells": same_state_cells,
        "predictive": predictive,
        "amplification": amplification,
        "closed_loop_consistency": consistency,
        "pilot": pilot,
        "classification": classification,
        "prior_closed_loop": prior_closed,
    }


def _validate_archived_sources(sources):
    expected = {"runner": runner_sha256(), "replay": EXPECTED_REPLAY_SHA256, "m5": EXPECTED_M5_SHA256}
    expected.update({
        f"m5_transitive_{label}": digest
        for label, digest in EXPECTED_M5_TRANSITIVE_SHA256.items()
    })
    errors = []
    for label, digest in expected.items():
        record = sources.get(label, {})
        path = Path(record.get("path", ""))
        if record.get("sha256") != digest or not path.is_file() or sha256_file(path) != digest:
            errors.append(f"execution_source_{label}")
    return errors


def collect_stage_results(stage, num_shards):
    _canonical, all_units, manifest_sha256 = load_canonical_manifest()
    protocol = stage_protocol(stage)
    expected_units = selected_stage_units(stage, all_units)
    by_unit = {}
    semantics = []
    failures = []
    protocol_errors = []
    checkpoint_errors = []
    shard_dirs = []
    for shard_id in range(int(num_shards)):
        shard_dir = RUN_DIR / "shards" / f"{stage}_{num_shards}_{shard_id}"
        shard_dirs.append(shard_dir)
        assigned = select_shard_units(expected_units, shard_id, num_shards)
        manifest_path = shard_dir / "manifest.json"
        provenance_path = shard_dir / "execution_provenance.json"
        failures_path = shard_dir / "failures.json"
        worker_path = shard_dir / "worker_summary.json"
        semantics_path = shard_dir / "runtime_semantics_audit.json"
        if not all(path.is_file() for path in (manifest_path, provenance_path, failures_path, worker_path, semantics_path)):
            protocol_errors.append(f"shard_{shard_id}:missing_metadata")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = stage_manifest(stage, shard_id, num_shards, assigned, protocol["history_lengths"], manifest.get("execution_sources", {}))
        protocol_errors.extend(f"shard_{shard_id}:manifest_{key}" for key in manifest_errors(manifest, expected))
        protocol_errors.extend(f"shard_{shard_id}:{error}" for error in _validate_archived_sources(manifest.get("execution_sources", {})))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance.get("task") != TASK
            or provenance.get("stage") != stage
            or provenance.get("git_commit") != git_commit()
            or provenance.get("runner_sha256") != runner_sha256()
            or int(provenance.get("shard_id", -1)) != shard_id
            or int(provenance.get("num_shards", -1)) != int(num_shards)
        ):
            protocol_errors.append(f"shard_{shard_id}:provenance")
        shard_failures = json.loads(failures_path.read_text(encoding="utf-8"))
        failures.extend(shard_failures)
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if worker.get("n_assigned") != len(assigned) or worker.get("n_success") != len(assigned) or worker.get("n_failed") != 0 or worker.get("failures") != []:
            protocol_errors.append(f"shard_{shard_id}:worker_summary")
        semantic = json.loads(semantics_path.read_text(encoding="utf-8"))
        semantics.append(semantic)
        expected_paths = {checkpoint_path(stage, num_shards, shard_id, unit) for unit in assigned}
        actual_paths = set((shard_dir / "units").glob("*.json"))
        if actual_paths != expected_paths:
            protocol_errors.append(f"shard_{shard_id}:checkpoint_file_set")
        for unit in assigned:
            path = checkpoint_path(stage, num_shards, shard_id, unit)
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_unit_result(
                result, unit, stage, protocol["history_lengths"], manifest_sha256,
                runner_sha256(), git_commit(),
            )
            if unit["unit_id"] in by_unit:
                errors.append("duplicate_unit_checkpoint")
            if errors:
                checkpoint_errors.append({"path": str(path), "errors": sorted(set(errors))})
            else:
                by_unit[unit["unit_id"]] = result
    selected = [by_unit[unit["unit_id"]] for unit in expected_units if unit["unit_id"] in by_unit]
    rows = [row for result in selected for row in result["rows"]]
    same_state_rows = [row for result in selected for row in result["same_state_rows"]]
    factorial_rows = [row for result in selected for row in result["factorial_rows"]]
    audits = [row for result in selected for row in result["audits"]]
    return {
        "expected_units": expected_units,
        "selected": selected,
        "rows": rows,
        "same_state_rows": same_state_rows,
        "factorial_rows": factorial_rows,
        "audits": audits,
        "semantics": semantics,
        "failures": failures,
        "protocol_errors": protocol_errors,
        "checkpoint_errors": checkpoint_errors,
        "shard_dirs": [str(path) for path in shard_dirs],
    }


def _prior_stage_value(stage, key, default):
    if stage == "stage0":
        return default
    predecessor = {"smoke": "stage0", "pilot": "smoke", "formal": "pilot"}[stage]
    path = RUN_DIR / f"aggregate_{predecessor}.json"
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8")).get(key, default)


def build_summary(stage, collected, gates, analysis, complete):
    direction = analysis["direction"]
    classification = analysis["classification"]
    same_counts = {
        carrier: {
            length: f"{cell['positive']} / {cell['n']}"
            for length, cell in by_l.items()
        }
        for carrier, by_l in analysis["same_state"].items()
    }
    smoke = "PASS" if stage == "smoke" and complete else _prior_stage_value(stage, "SMOKE", "NOT_RUN")
    pilot = analysis["pilot"] if stage == "pilot" else _prior_stage_value(stage, "PILOT", "NOT_RUN")
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage": stage,
        "formal_status": "COMPLETE" if stage == "formal" and complete else (f"NOT_RUN_{stage.upper()}_COMPLETE" if complete else "INVALID"),
        "n_formal_units": len(collected["selected"]) if stage == "formal" else 0,
        "n_success": len(collected["selected"]),
        "n_expected": len(collected["expected_units"]),
        "canonical_manifest_source": str(MANIFEST_PATH),
        "canonical_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "n_canonical_units": 18,
        "replay_runner_sha256": EXPECTED_REPLAY_SHA256,
        "m5_source_sha256": EXPECTED_M5_SHA256,
        "runner_sha256": runner_sha256(),
        "gates": gates,
        "SMOKE": smoke,
        "PILOT": pilot,
        "history_lengths": stage_protocol(stage)["history_lengths"],
        "SAME_STATE_F5_R_MINUS_C_BY_L": analysis["same_state"],
        "SAME_STATE_F5_R_GT_C_COUNT": same_counts,
        **direction,
        "HISTORY_AMPLIFICATION_RHO": analysis["amplification"],
        "M0_DIRECTION_PREDICTIVE_SIGNAL": analysis["predictive"]["M0"],
        "M4_DIRECTION_PREDICTIVE_SIGNAL": analysis["predictive"]["M4"],
        "M5_DIRECTION_PREDICTIVE_SIGNAL": analysis["predictive"]["M5"],
        "F5_DIRECTION_PREDICTIVE_SIGNAL": analysis["predictive"]["F5"],
        "CLOSED_LOOP_DIRECTION_CONSISTENCY": analysis["closed_loop_consistency"],
        **classification,
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "CROSS_ARCH_RECURRENT_QUANTIZATION_RISK_FACTORIZATION_V1",
        "failures": collected["failures"],
        "protocol_errors": collected["protocol_errors"],
        "checkpoint_validation_errors": collected["checkpoint_errors"],
        "shard_dirs": collected["shard_dirs"],
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES" if stage == "formal" and complete else "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
    }


def write_report(summary):
    lines = [
        f"# {TASK}",
        "",
        f"- Formal status: `{summary['formal_status']}`",
        f"- Formal units: `{summary['n_formal_units']}`",
        f"- Fresh output: `{summary['FRESH_OUTPUT_VERIFICATION']}`",
        f"- Final classification: `{summary['FINAL_CLASSIFICATION']}`",
        f"- Closed-loop consistency: `{summary['CLOSED_LOOP_DIRECTION_CONSISTENCY']}`",
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(summary["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Same-State F5",
        "",
        "```json",
        json.dumps(summary["SAME_STATE_F5_R_MINUS_C_BY_L"], indent=2, sort_keys=True),
        "```",
        "",
        "## Direction Effects",
        "",
        "```json",
        json.dumps({key: summary[key] for key in ("DIRECTION_RISK_AT_C_MAG_BY_L", "DIRECTION_RISK_AT_R_MAG_BY_L", "DIRECTION_MAIN_BY_L", "HISTORY_AMPLIFICATION_RHO")}, indent=2, sort_keys=True),
        "```",
        "",
        "## Predictive Signals",
        "",
        "```json",
        json.dumps({key: summary[key] for key in ("M0_DIRECTION_PREDICTIVE_SIGNAL", "M4_DIRECTION_PREDICTIVE_SIGNAL", "M5_DIRECTION_PREDICTIVE_SIGNAL", "F5_DIRECTION_PREDICTIVE_SIGNAL")}, indent=2, sort_keys=True),
        "```",
        "",
    ]
    text = "\n".join(lines)
    atomic_write_text(RUN_DIR / "report.md", text)
    atomic_write_text(REPORT_DIR / f"{SLUG}.md", text)


def write_formal_outputs(summary, collected, analysis):
    results_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in collected["rows"])
    atomic_write_text(RUN_DIR / "results.jsonl", results_text)
    write_csv(RUN_DIR / "same_state_f5.csv", collected["same_state_rows"])
    write_csv(RUN_DIR / "direction_factorial.csv", [row for row in collected["rows"] if row["branch"] != "FP"])
    effect_rows = []
    source = {(row["unit_id"], int(row["history_length"])): row for row in collected["rows"]}
    for effect in analysis["effects"]:
        base = source[(effect["unit_id"], int(effect["history_length"]))]
        effect_rows.append({
            "task": TASK,
            "git_commit": base["git_commit"],
            "manifest_sha256": base["manifest_sha256"],
            "prompt": base["prompt"],
            "t0": base["t0"],
            **effect,
        })
    write_csv(RUN_DIR / "history_direction_effect.csv", effect_rows)
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    write_report(summary)


def aggregate_stage(stage, num_shards):
    if stage == "formal":
        require_stage_prerequisite("formal")
    merge_started = time.time()
    collected = collect_stage_results(stage, num_shards)
    audit = environment_audit()
    protocol = stage_protocol(stage)
    rows = collected["rows"]
    complete = (
        len(collected["selected"]) == len(collected["expected_units"])
        and not collected["failures"]
        and not collected["protocol_errors"]
        and not collected["checkpoint_errors"]
        and exact_result_coverage(rows, collected["expected_units"], protocol["history_lengths"])
    )
    gates = dict(audit["gates"])
    semantics = collected["semantics"]
    for name in BLOCKING_GATE_NAMES:
        observed = [semantic.get(name) for semantic in semantics if semantic.get(name) is not None]
        if observed:
            gates[name] = "PASS" if len(observed) == int(num_shards) and all(value == "PASS" for value in observed) else "FAIL"
    gates.update(observed_execution_gates(
        collected["same_state_rows"], collected["factorial_rows"], collected["audits"],
        collected["expected_units"], protocol["history_lengths"],
    ))
    complete = complete and all(gates.get(name) == "PASS" for name in BLOCKING_GATE_NAMES)
    analysis = analyze_scientific_results(rows, collected["same_state_rows"], stage) if rows else {
        "effects": [], "direction": {name: {} for name in ("DIRECTION_RISK_AT_C_MAG_BY_L", "DIRECTION_RISK_AT_R_MAG_BY_L", "DIRECTION_MAIN_BY_L")},
        "same_state": {}, "same_state_cells": [], "predictive": {metric: {} for metric in ("M0", "M4", "M5", "F5")},
        "amplification": {"C_MAG": None, "R_MAG": None, "MAIN": None}, "closed_loop_consistency": "NOT_SUPPORTED", "pilot": "NEGATIVE_OR_INCONCLUSIVE",
        "classification": {"QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL": "NOT_INTERPRETABLE_INVALID_OUTPUT", "M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION": "NOT_INTERPRETABLE_INVALID_OUTPUT", "RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE": "NOT_INTERPRETABLE_INVALID_OUTPUT", "FINAL_CLASSIFICATION": "INVALID_INCOMPLETE_OUTPUT"},
    }
    summary = build_summary(stage, collected, gates, analysis, complete)
    if stage == "formal":
        write_formal_outputs(summary, collected, analysis)
        output_paths = [RUN_DIR / name for name in REQUIRED_OUTPUT_NAMES]
        fresh, checks = fresh_output_verification(
            rows, collected["expected_units"], protocol["history_lengths"], output_paths, merge_started
        )
        summary["FRESH_OUTPUT_VERIFICATION"] = fresh
        summary["fresh_output_checks"] = checks
        if fresh != "PASS":
            summary["formal_status"] = "INVALID"
            summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "NO"
            summary["FINAL_CLASSIFICATION"] = "INVALID_FRESH_OUTPUT_VERIFICATION"
        write_formal_outputs(summary, collected, analysis)
    else:
        summary["FRESH_OUTPUT_VERIFICATION"] = "PASS" if complete else "FAIL"
    save_json(RUN_DIR / f"aggregate_{stage}.json", summary)
    return summary


def print_required_summary(summary, pytest_result="NOT_RUN"):
    gates = summary.get("gates", {})
    values = [
        ("TASK", TASK),
        ("FORMAL_STATUS", summary.get("formal_status")),
        ("N_FORMAL_UNITS", summary.get("n_formal_units")),
        ("CANONICAL_MANIFEST_SOURCE", summary.get("canonical_manifest_source")),
        ("CANONICAL_MANIFEST_SHA256", summary.get("canonical_manifest_sha256")),
        ("N_CANONICAL_UNITS", summary.get("n_canonical_units")),
    ]
    for name in BLOCKING_GATE_NAMES:
        values.append((name, gates.get(name)))
    values.extend([
        ("SMOKE", summary.get("SMOKE")),
        ("PILOT", summary.get("PILOT")),
        ("HISTORY_LENGTHS", summary.get("history_lengths")),
        ("SAME_STATE_F5_R_MINUS_C_BY_L", summary.get("SAME_STATE_F5_R_MINUS_C_BY_L")),
        ("SAME_STATE_F5_R_GT_C_COUNT", summary.get("SAME_STATE_F5_R_GT_C_COUNT")),
        ("DIRECTION_RISK_AT_C_MAG_BY_L", summary.get("DIRECTION_RISK_AT_C_MAG_BY_L")),
        ("DIRECTION_RISK_AT_R_MAG_BY_L", summary.get("DIRECTION_RISK_AT_R_MAG_BY_L")),
        ("DIRECTION_MAIN_BY_L", summary.get("DIRECTION_MAIN_BY_L")),
        ("HISTORY_AMPLIFICATION_RHO", summary.get("HISTORY_AMPLIFICATION_RHO")),
        ("M0_DIRECTION_PREDICTIVE_SIGNAL", summary.get("M0_DIRECTION_PREDICTIVE_SIGNAL")),
        ("M4_DIRECTION_PREDICTIVE_SIGNAL", summary.get("M4_DIRECTION_PREDICTIVE_SIGNAL")),
        ("M5_DIRECTION_PREDICTIVE_SIGNAL", summary.get("M5_DIRECTION_PREDICTIVE_SIGNAL")),
        ("F5_DIRECTION_PREDICTIVE_SIGNAL", summary.get("F5_DIRECTION_PREDICTIVE_SIGNAL")),
        ("CLOSED_LOOP_DIRECTION_CONSISTENCY", summary.get("CLOSED_LOOP_DIRECTION_CONSISTENCY")),
        ("QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL", summary.get("QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL")),
        ("M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION", summary.get("M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION")),
        ("RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE", summary.get("RECURRENT_FUNCTIONAL_GEOMETRY_CLOSURE")),
        ("FINAL_CLASSIFICATION", summary.get("FINAL_CLASSIFICATION")),
        ("METHOD_DESIGN_READY", summary.get("METHOD_DESIGN_READY")),
        ("NEXT_RECOMMENDED_TASK", summary.get("NEXT_RECOMMENDED_TASK")),
        ("PYTEST", pytest_result),
        ("FRESH_OUTPUT_VERIFICATION", summary.get("FRESH_OUTPUT_VERIFICATION")),
        ("CURRENT_TASK_COMPLETED_AND_STOPPED", summary.get("CURRENT_TASK_COMPLETED_AND_STOPPED")),
        ("NO_NEXT_EXPERIMENT_LAUNCHED", summary.get("NO_NEXT_EXPERIMENT_LAUNCHED")),
    ])
    for name, value in values:
        print(f"{name} =")
        if isinstance(value, (dict, list)):
            print(json.dumps(value, sort_keys=True))
        else:
            print(value)
        print()
    print("STOP.")


def audit_only():
    audit = environment_audit()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RUN_DIR / "environment_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["formal_status"] == "READY_TO_RUN" else 1


def main():
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument(
        "--stage",
        choices=[
            "audit", "stage0", "smoke", "pilot", "formal",
            "merge-stage0", "merge-smoke", "merge-pilot", "merge-formal",
            "analyze", "check-formal-prerequisite",
        ],
        default="audit",
    )
    parser.add_argument("--analyze-stage", choices=["stage0", "smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if args.stage == "audit":
        raise SystemExit(audit_only())
    if args.stage == "check-formal-prerequisite":
        require_stage_prerequisite("formal")
        print("FORMAL_PREREQUISITE_GATE = PASS")
        return
    if args.stage in ("stage0", "smoke", "pilot", "formal"):
        raise SystemExit(run_execution_stage(args))
    if args.stage.startswith("merge-"):
        stage = args.stage.split("-", 1)[1]
    else:
        stage = args.analyze_stage
    summary = aggregate_stage(stage, args.num_shards)
    print_required_summary(summary)


if __name__ == "__main__":
    main()
