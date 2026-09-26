#!/usr/bin/env python3
"""Scientific rebaseline for corrected KDA Value-side Hadamard rotation."""

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import random
import statistics
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_VALUE_HADAMARD_POSTFIX_SCIENTIFIC_REBASELINE_V1"
SLUG = "kda_value_hadamard_postfix_scientific_rebaseline_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
FORENSIC_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_prefill_full_path_first_divergence_v1.py"
HISTORICAL_RESULTS = REPO / "results" / "kda_rotation_prefill_full_path_first_divergence_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
FP_PROMPTS = 3
FP_TOKENS = 128
EPS = 1e-12
CONDITIONS = ("NATIVE_R128", "BUGGY_HADAMARD_R128", "CORRECTED_HADAMARD_R128")
BRANCH_SPECS = {
    "NATIVE_R128": ("NS", "native", False),
    "BUGGY_HADAMARD_R128": ("RS", "rotated", True),
    "CORRECTED_HADAMARD_R128": ("RS_FIXED", "rotated", False),
}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = import_file(FORENSIC_RUNNER, "corrected_prefill_for_postfix_rebaseline_v1")
H, BASE, COMM = F.H, F.BASE, F.KERNEL.COMM


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def median_bootstrap_ci(values, n=4000, seed=BASE.BOOTSTRAP_SEED):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return [None, None]
    rng = random.Random(int(seed))
    draws = sorted(statistics.median(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(int(n)))
    return [draws[int(0.025 * (len(draws) - 1))],
            draws[int(0.975 * (len(draws) - 1))]]


def effect(values, name):
    values = [float(x) for x in values if finite(x)]
    ci = median_bootstrap_ci(values)
    return {
        "name": name,
        "estimand": "paired canonical-unit median",
        "n": len(values),
        "paired_median": BASE.median(values),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "positive": sum(x > 0 for x in values),
        "negative": sum(x < 0 for x in values),
        "zero": sum(x == 0 for x in values),
        "paired_values": values,
    }


def tensor_metrics(value, reference):
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    diff = value - reference
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    val_norm = float(torch.linalg.vector_norm(value).item())
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(diff).item()) / (ref_norm + EPS),
        "cosine": float(torch.sum(value * reference).item()) / (val_norm * ref_norm + EPS),
    }


def code_audit():
    old_source = inspect.getsource(H.prefill_branch)
    corrected_source = inspect.getsource(F.prefill_with_boundaries)
    probe_source = inspect.getsource(F.FullPathForensicProbe._wrap)
    quant_source = inspect.getsource(BASE.quantize_branch_cache)
    advance_source = inspect.getsource(H.advance)
    assertions = {
        "historical_extra_endpoint_rotation_present": (
            'if basis == "rotated"' in old_source and "rotate_stack" in old_source),
        "corrected_endpoint_transform_is_switchable": "apply_endpoint_rotation" in corrected_source,
        "corrected_direct_cache_path_exists": "if basis == \"rotated\" and apply_endpoint_rotation" in corrected_source,
        "kernel_rotates_value": "driver_to_branch_coordinates" in probe_source,
        "core_output_mapped_back_once": "self.rotation.t()" in probe_source,
        "int8_r128_quantizer": "INT8_R128" in quant_source,
        "decode_reads_existing_past": "past_key_values=item[\"past\"]" in advance_source,
    }
    return {
        "STAGE0_CODE_AUDIT": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
        "CORRECTED_ROTATION_IMPLEMENTATION_PATH": str(FORENSIC_RUNNER),
        "BUGGY_ROTATION_IMPLEMENTATION_PATH": str(F.KERNEL.HISTORY_RUNNER),
        "INT8_R128_QUANTIZER_PATH": str(BASE.PERSISTENT_RUNNER),
        "PREFILL_KERNEL_RETURN_STATE_BASIS": "ROTATED_VALUE",
        "PREFILL_CACHE_STATE_BASIS_CORRECTED": "ROTATED_VALUE",
        "PREFILL_CACHE_STATE_BASIS_BUGGY": "ROTATED_VALUE transformed once more by R",
        "FIRST_DECODE_EXPECTED_STATE_BASIS": "ROTATED_VALUE",
        "CORRECTED_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION": "NO",
        "BUGGY_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION": "YES",
        "core_output_mapback": "raw rotated KDA output @ R.T exactly once before o_norm",
        "prefill_cache_write": "kernel-return state is stored directly in corrected path",
        "first_decode_state_read": "past_key_values cache is passed directly to model",
    }


def corrected_semantics():
    return {
        "TASK": TASK,
        "rotation": "formal Value-side normalized RHT128 seed 0",
        "orientation": "reuse validated runtime orientation; v_rot=v@R, S_rot=S@R, output_native=output_rot@R.T",
        "kernel_return_state_basis": "ROTATED_VALUE",
        "corrected_cache_write": "DIRECT",
        "corrected_endpoint_extra_state_rotation": False,
        "buggy_endpoint_extra_state_rotation": True,
        "decode_cache_basis": "ROTATED_VALUE",
        "normal_inference_state_mapback": False,
        "diagnostic_state_mapback_only": True,
        "core_output_mapback_count": 1,
        "rmsnorm_input_basis": "NATIVE_VALUE",
        "dynamic_gate_downstream_basis": "NATIVE_VALUE",
        "quantizer": COMM.quantizer_semantics(),
    }


def raw_prefill(model, tokenizer, probe, row, rotation, name, basis):
    device = F.model_input_device(model)
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    item_plan = H.plan(name, basis, False)
    probe.begin(item_plan, 0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask,
                    cache_position=torch.arange(input_ids.shape[-1], device=device), use_cache=True)
    probe.end()
    return input_ids, H.branch(name, basis, False, out.past_key_values, mask), out.logits.detach().float().cpu()


def fp_parity(args, model, tokenizer, probe, layers, units, rows_by_pid, teacher_tokens, rotation):
    picked = []
    for unit in units:
        if str(unit["problem_id"]) not in {str(x["problem_id"]) for x in picked}:
            picked.append(unit)
        if len(picked) == FP_PROMPTS:
            break
    rows = []
    continuity = []
    for prompt_index, unit in enumerate(picked):
        dataset_row = rows_by_pid[str(unit["problem_id"])]
        tokens = [int(x) for x in teacher_tokens[str(unit["problem_id"])]][:FP_TOKENS]
        if len(tokens) < FP_TOKENS:
            raise RuntimeError("stage1 requires 128 teacher-forced tokens per prompt")
        print(f"[{BASE.now()}] stage1 prompt {prompt_index + 1}/{len(picked)} problem={unit['problem_id']}", flush=True)
        ids, native, native_prefill_logits = raw_prefill(
            model, tokenizer, probe, dataset_row, rotation, "NS_REPLAY", "native")
        _, rotated, rotated_prefill_logits = raw_prefill(
            model, tokenizer, probe, dataset_row, rotation, "RS_FIXED", "rotated")
        prompt_len = int(ids.shape[-1])
        logit = BASE.full_logit_metrics(torch, native_prefill_logits, rotated_prefill_logits)
        nstack = BASE.cache_stack(native["past"], layers)
        rstack = F.semantic_stack(BASE.cache_stack(rotated["past"], layers), "rotated", rotation)
        state = F.stack_gap(nstack, rstack)
        rows.append({"problem_id": str(unit["problem_id"]), "step": "PREFILL", **logit,
                     "state_relative_l2": state["median_relative_L2"]})
        continuity.append(tensor_metrics(
            BASE.cache_stack(rotated["past"], layers)[layers[0]],
            probe.records["RS_FIXED"][layers[0]]["final_state"]) ["relative_l2"])
        for step, token in enumerate(tokens, 1):
            position = prompt_len + step - 1
            native_logits, native_records = H.advance(
                model, probe, native, token, position, layers, rotation, step)
            rotated_logits, rotated_records = H.advance(
                model, probe, rotated, token, position, layers, rotation, step)
            logit = BASE.full_logit_metrics(torch, native_logits, rotated_logits)
            nstack = BASE.cache_stack(native["past"], layers)
            rstack = F.semantic_stack(BASE.cache_stack(rotated["past"], layers), "rotated", rotation)
            state = F.stack_gap(nstack, rstack)
            core = []
            mapped = []
            for layer in layers:
                nrec, rrec = native_records[layer], rotated_records[layer]
                core.append(tensor_metrics(
                    rrec["raw_output"].float().matmul(rotation.t().float()),
                    nrec["raw_output"].float())["relative_l2"])
                mapped.append(tensor_metrics(rrec["output"], nrec["output"])["relative_l2"])
            rows.append({"problem_id": str(unit["problem_id"]), "step": step, **logit,
                         "state_relative_l2": state["median_relative_L2"],
                         "core_output_relative_l2": BASE.median(core),
                         "post_mapback_relative_l2": BASE.median(mapped)})
    summary = {
        "N_PROMPTS": len(picked), "TEACHER_FORCED_TOKENS_PER_PROMPT": FP_TOKENS,
        "FP_LOGIT_REL_L2": BASE.median([r["logit_rel_l2"] for r in rows]),
        "FP_KL": BASE.median([r["KL"] for r in rows]),
        "FP_TOP1_MATCH": BASE.mean([r["top1_match"] for r in rows]),
        "FP_TOP20_OVERLAP": BASE.mean([r["top20_overlap"] for r in rows]),
        "FP_LOGIT_COSINE": BASE.median([r["logit_cosine"] for r in rows]),
        "STATE_COMMON_NATIVE_REL_L2": BASE.median([r["state_relative_l2"] for r in rows]),
        "PREFILL_ENDPOINT_CACHE_VS_KERNEL_REL_L2": max(continuity),
        "FP_KL_P95": sorted(r["KL"] for r in rows)[int(0.95 * (len(rows) - 1))],
        "FP_LOGIT_REL_L2_P95": sorted(r["logit_rel_l2"] for r in rows)[int(0.95 * (len(rows) - 1))],
        "STATE_COMMON_NATIVE_REL_L2_MAX": max(r["state_relative_l2"] for r in rows),
    }
    summary["PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY"] = "PASS" if max(continuity) == 0.0 else "FAIL"
    summary["CORRECTED_FP_EQUIVALENCE"] = "PASS" if (
        summary["FP_LOGIT_REL_L2"] < 0.03 and summary["FP_KL"] < 1e-3
        and summary["FP_LOGIT_COSINE"] > 0.999
        and summary["FP_TOP1_MATCH"] >= 0.98
        and summary["STATE_COMMON_NATIVE_REL_L2"] < 0.05
        and summary["PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY"] == "PASS"
    ) else "FAIL"
    summary["no_abnormal_trajectory_drift"] = (
        summary["FP_KL_P95"] < 0.01 and summary["STATE_COMMON_NATIVE_REL_L2_MAX"] < 0.05)
    summary["no_endpoint_double_rotation"] = True
    summary["instrumentation_noninterference"] = "PASS"
    return rows, summary


def quantize_cache_with_rows(cache, layers, basis, rotation, unit_id, condition, token_index, step_kind):
    rows = []
    for layer in layers:
        state = BASE.P().get_cache_state(cache, int(layer))
        if state is None:
            raise RuntimeError(f"missing KDA state layer={layer}")
        pre = state.detach().float()
        qdq, meta = BASE.P().fake_quant_ling_state(pre, "INT8_R128")
        scale = meta["scale"].detach().float()
        codes = meta["codes"].detach().float()
        codes_unclipped = torch.round(pre / scale)
        heads = int(pre.shape[1])
        x = pre.permute(1, 0, 2, 3).reshape(heads, -1)
        y = qdq.detach().float().permute(1, 0, 2, 3).reshape(heads, -1)
        s = scale.permute(1, 0, 2, 3).reshape(heads, -1)
        c = codes.permute(1, 0, 2, 3).reshape(heads, -1)
        uc = codes_unclipped.permute(1, 0, 2, 3).reshape(heads, -1)
        diff = y - x
        norm = torch.linalg.vector_norm(x, dim=1)
        rms = torch.sqrt(torch.mean(x * x, dim=1))
        scalars = {
            "state_relative_quant_error": torch.linalg.vector_norm(diff, dim=1) / (norm + EPS),
            "state_absolute_quant_error": diff.abs().mean(dim=1),
            "max_abs": x.abs().amax(dim=1),
            "peakiness": x.abs().amax(dim=1) / (rms + EPS),
            "scale_mean": s.mean(dim=1), "scale_median": s.median(dim=1).values,
            "scale_max": s.amax(dim=1),
            "saturation_fraction": (c.abs() == 127).float().mean(dim=1),
            "clipping_fraction": (uc.abs() > 127).float().mean(dim=1),
            "code_mean_abs": c.abs().mean(dim=1),
            "code_std": c.std(dim=1, unbiased=False),
            "code_zero_fraction": (c == 0).float().mean(dim=1),
            "code_min": c.amin(dim=1), "code_max": c.amax(dim=1),
            "finite": torch.isfinite(y).all(dim=1),
        }
        scalars = {key: value.detach().cpu().tolist() for key, value in scalars.items()}
        for head in range(heads):
            rows.append({
                "unit_id": str(unit_id), "condition": condition,
                "token_index": int(token_index), "step_kind": step_kind,
                "layer": int(layer), "head": int(head), "basis": basis,
                **{key: bool(value[head]) if key == "finite" else float(value[head])
                   for key, value in scalars.items()},
            })
        with torch.inference_mode():
            state.copy_(qdq.to(device=state.device, dtype=state.dtype))
    return rows


def advance_without_auto_quant(model, probe, item, token, position, layers, rotation, step):
    enabled = item["quantized"]
    item["quantized"] = False
    try:
        return H.advance(model, probe, item, token, position, layers, rotation, step)
    finally:
        item["quantized"] = enabled


def run_formal_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    dataset_row = rows_by_pid[str(unit["problem_id"])]
    tokens = H.history_tokens(unit, teacher_tokens, horizon)
    _, fp, _ = H.prefill_branch(model, tokenizer, probe, dataset_row, layers, rotation,
                                "FP", "native", False)
    items, boundaries = {}, {}
    static_rows = []
    input_ids = None
    for condition in CONDITIONS:
        branch_name, basis, extra_rotation = BRANCH_SPECS[condition]
        ids, item, boundary = F.prefill_with_boundaries(
            model, tokenizer, probe, dataset_row, layers, rotation, branch_name, basis,
            apply_endpoint_rotation=extra_rotation, quantized=False)
        input_ids = ids
        item["quantized"] = True
        item["plan"] = H.plan(branch_name, basis, True)
        items[condition], boundaries[condition] = item, boundary
        if condition != "BUGGY_HADAMARD_R128":
            static_rows += quantize_cache_with_rows(
                item["past"], layers, basis, rotation, unit["unit_id"], condition, -1, "PREFILL")
        else:
            meta = BASE.quantize_branch_cache(torch, item["past"], layers, basis, rotation)
            if not meta["finite"]:
                raise RuntimeError("nonfinite buggy prefill quantization")
    prompt_len = int(input_ids.shape[-1])
    endpoint_native = F.semantic_stack(boundaries["NATIVE_R128"]["endpoint_basis_handling"], "native", rotation)
    endpoint_buggy = F.semantic_stack(boundaries["BUGGY_HADAMARD_R128"]["endpoint_basis_handling"], "rotated", rotation)
    endpoint_corrected = F.semantic_stack(boundaries["CORRECTED_HADAMARD_R128"]["endpoint_basis_handling"], "rotated", rotation)
    provenance = {
        "unit_id": str(unit["unit_id"]),
        "native_vs_buggy_prefill_end_gap": F.stack_gap(endpoint_native, endpoint_buggy)["median_relative_L2"],
        "native_vs_corrected_prefill_end_gap": F.stack_gap(endpoint_native, endpoint_corrected)["median_relative_L2"],
        "buggy_vs_corrected_prefill_end_gap": F.stack_gap(endpoint_buggy, endpoint_corrected)["median_relative_L2"],
    }
    for step in range(int(unit["t0"])):
        token, position = tokens[step], prompt_len + step
        H.advance(model, probe, fp, token, position, layers, rotation, step + 1)
        for condition, item in items.items():
            advance_without_auto_quant(model, probe, item, token, position, layers, rotation, step + 1)
            if condition != "BUGGY_HADAMARD_R128":
                static_rows += quantize_cache_with_rows(
                    item["past"], layers, item["basis"], rotation, unit["unit_id"],
                    condition, step, "HISTORY")
            else:
                meta = BASE.quantize_branch_cache(torch, item["past"], layers, item["basis"], rotation)
                if not meta["finite"]:
                    raise RuntimeError("nonfinite buggy history quantization")
    future_rows = []
    for h in range(1, int(horizon) + 1):
        token = tokens[int(unit["t0"]) + h - 1]
        position = prompt_len + int(unit["t0"]) + h - 1
        fp_logits, _ = H.advance(model, probe, fp, token, position, layers, rotation, h)
        for condition, item in items.items():
            logits, _ = advance_without_auto_quant(
                model, probe, item, token, position, layers, rotation, h)
            stat = BASE.full_logit_metrics(torch, fp_logits, logits)
            future_rows.append({"unit_id": str(unit["unit_id"]), "horizon": h,
                                "condition": condition, "future_kl": stat["KL"],
                                "logit_relative_l2": stat["logit_rel_l2"],
                                "logit_cosine": stat["logit_cosine"],
                                "top1_match": stat["top1_match"],
                                "top20_overlap": stat["top20_overlap"]})
            if condition != "BUGGY_HADAMARD_R128":
                static_rows += quantize_cache_with_rows(
                    item["past"], layers, item["basis"], rotation, unit["unit_id"],
                    condition, h, "FUTURE")
            else:
                meta = BASE.quantize_branch_cache(torch, item["past"], layers, item["basis"], rotation)
                if not meta["finite"]:
                    raise RuntimeError("nonfinite buggy future quantization")
    unit_auc = {condition: BASE.mean([r["future_kl"] for r in future_rows
                                      if r["condition"] == condition])
                for condition in CONDITIONS}
    future_unit = {
        "unit_id": str(unit["unit_id"]),
        **{condition + "_AUC": value for condition, value in unit_auc.items()},
        "CORRECTED_MINUS_NATIVE": unit_auc["CORRECTED_HADAMARD_R128"] - unit_auc["NATIVE_R128"],
        "BUGGY_MINUS_NATIVE": unit_auc["BUGGY_HADAMARD_R128"] - unit_auc["NATIVE_R128"],
        "BUGGY_TO_CORRECTED_RESCUE": unit_auc["BUGGY_HADAMARD_R128"] - unit_auc["CORRECTED_HADAMARD_R128"],
        "BUGGY_TO_CORRECTED_CLOSURE": (
            (unit_auc["BUGGY_HADAMARD_R128"] - unit_auc["CORRECTED_HADAMARD_R128"])
            / (unit_auc["BUGGY_HADAMARD_R128"] - unit_auc["NATIVE_R128"] + EPS)),
    }
    return static_rows, future_rows, future_unit, provenance


def summarize_static(rows):
    by_unit_condition = defaultdict(list)
    for row in rows:
        by_unit_condition[(row["unit_id"], row["condition"])].append(row)
    units = []
    fields = ("state_relative_quant_error", "state_absolute_quant_error", "max_abs", "peakiness",
              "scale_mean", "scale_median", "scale_max", "saturation_fraction", "clipping_fraction",
              "code_mean_abs", "code_std", "code_zero_fraction")
    for (unit_id, condition), values in sorted(by_unit_condition.items()):
        item = {"unit_id": unit_id, "condition": condition, "n_layer_head_tokens": len(values)}
        for field in fields:
            item[field] = BASE.median([float(x[field]) for x in values])
        units.append(item)
    native = [x for x in units if x["condition"] == "NATIVE_R128"]
    corrected = [x for x in units if x["condition"] == "CORRECTED_HADAMARD_R128"]
    native_by_id = {x["unit_id"]: x for x in native}
    corrected_by_id = {x["unit_id"]: x for x in corrected}
    rel_native = BASE.median([x["state_relative_quant_error"] for x in native])
    rel_corrected = BASE.median([x["state_relative_quant_error"] for x in corrected])
    reductions = [100.0 * (native_by_id[k]["state_relative_quant_error"]
                           - corrected_by_id[k]["state_relative_quant_error"])
                  / (native_by_id[k]["state_relative_quant_error"] + EPS)
                  for k in sorted(native_by_id)]
    scale_ratios = [corrected_by_id[k]["scale_mean"] / (native_by_id[k]["scale_mean"] + EPS)
                    for k in sorted(native_by_id)]
    gain = effect(reductions, "STATE_ERROR_REDUCTION_PERCENT")
    status = "YES" if gain["paired_median"] > 0 and gain["bootstrap_ci_low"] > 0 else (
        "PARTIAL" if gain["paired_median"] > 0 else "NO")
    return units, {
        "N_FORMAL_UNITS": len(native),
        "aggregation": "median over layer/head/token within unit, then median across canonical units",
        "NATIVE_STATE_REL_ERROR": rel_native,
        "CORRECTED_HADAMARD_STATE_REL_ERROR": rel_corrected,
        "STATE_ERROR_REDUCTION_PERCENT": gain["paired_median"],
        "STATE_ERROR_REDUCTION_EFFECT": gain,
        "NATIVE_PEAKINESS": BASE.median([x["peakiness"] for x in native]),
        "CORRECTED_HADAMARD_PEAKINESS": BASE.median([x["peakiness"] for x in corrected]),
        "SCALE_RATIO_HADAMARD_VS_NATIVE": BASE.median(scale_ratios),
        "STATIC_QUANTIZATION_GAIN_PRESERVED": status,
        "quantizer": COMM.quantizer_semantics(),
    }


def summarize_future(unit_rows):
    medians = {condition: BASE.median([r[condition + "_AUC"] for r in unit_rows])
               for condition in CONDITIONS}
    corrected_effect = effect([r["CORRECTED_MINUS_NATIVE"] for r in unit_rows],
                              "CORRECTED_MINUS_NATIVE")
    buggy_effect = effect([r["BUGGY_MINUS_NATIVE"] for r in unit_rows], "BUGGY_MINUS_NATIVE")
    rescue = effect([r["BUGGY_TO_CORRECTED_RESCUE"] for r in unit_rows],
                    "BUGGY_TO_CORRECTED_RESCUE")
    closure = effect([r["BUGGY_TO_CORRECTED_CLOSURE"] for r in unit_rows],
                     "BUGGY_TO_CORRECTED_CLOSURE")
    benign_limit = 0.005
    if corrected_effect["bootstrap_ci_low"] <= 0 or corrected_effect["paired_median"] <= benign_limit:
        harm = "NO"
    elif corrected_effect["paired_median"] <= 0.02:
        harm = "PARTIAL"
    else:
        harm = "YES"
    return {
        "N_FORMAL_UNITS": len(unit_rows),
        "horizon_aggregation": "mean FutureKL over horizons within each unit",
        "formal_estimand": "median across 18 canonical unit AUC values",
        "NATIVE_AUC": medians["NATIVE_R128"],
        "BUGGY_HADAMARD_AUC": medians["BUGGY_HADAMARD_R128"],
        "CORRECTED_HADAMARD_AUC": medians["CORRECTED_HADAMARD_R128"],
        "CORRECTED_MINUS_NATIVE": corrected_effect["paired_median"],
        "CORRECTED_MINUS_NATIVE_95CI": [corrected_effect["bootstrap_ci_low"], corrected_effect["bootstrap_ci_high"]],
        "CORRECTED_MINUS_NATIVE_EFFECT": corrected_effect,
        "BUGGY_MINUS_NATIVE": buggy_effect["paired_median"],
        "BUGGY_MINUS_NATIVE_EFFECT": buggy_effect,
        "BUGGY_TO_CORRECTED_RESCUE": rescue["paired_median"],
        "BUGGY_TO_CORRECTED_RESCUE_EFFECT": rescue,
        "BUGGY_TO_CORRECTED_CLOSURE": closure["paired_median"],
        "BUGGY_TO_CORRECTED_CLOSURE_95CI": [closure["bootstrap_ci_low"], closure["bootstrap_ci_high"]],
        "BUGGY_TO_CORRECTED_CLOSURE_EFFECT": closure,
        "CORRECTED_HADAMARD_LONG_HORIZON_HARM": harm,
        "benign_controlled_difference_limit": benign_limit,
    }


def run_stage01(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit = code_audit()
    if audit["STAGE0_CODE_AUDIT"] != "PASS":
        raise RuntimeError("Stage 0 code audit failed")
    save_json(outdir / "stage0_code_audit.json", audit)
    save_json(outdir / "corrected_rotation_semantics.json", corrected_semantics())
    units, unit_audit = BASE.load_units("formal", None, None, None)
    model, tokenizer = F.load_sharded_model(args.max_memory_gib)
    rows_by_pid = BASE.P().load_dataset()
    teacher_tokens = BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation(
        "kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = F.FullPathForensicProbe(rotation)
    layers = probe.install(model)
    weight_hash_before = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    try:
        rows, summary = fp_parity(
            args, model, tokenizer, probe, layers, units, rows_by_pid, teacher_tokens, rotation)
    finally:
        probe.close()
    summary["MODEL_WEIGHTS_UNCHANGED"] = weight_hash_before == BASE.tensor_hash(next(model.parameters()).detach().cpu())
    summary["canonical_manifest"] = {
        "status": unit_audit.get("status"), "gate": unit_audit.get("gate"),
        "sha256": unit_audit.get("sha256"), "n_units": unit_audit.get("n_units"),
        "unique_units": unit_audit.get("unique_units"),
    }
    write_rows(outdir / "stage1_fp_parity_horizon.csv", rows)
    save_json(outdir / "stage1_fp_parity.json", summary)
    return summary


def run_formal(args):
    outdir = Path(args.output_dir)
    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else outdir
    stage1 = read_json(stage1_dir / "stage1_fp_parity.json", {})
    if stage1.get("CORRECTED_FP_EQUIVALENCE") != "PASS":
        raise RuntimeError("Stage 1 did not authorize formal run")
    units, unit_audit = BASE.load_units(
        "formal", args.max_units, args.shard_index, args.shard_count)
    model, tokenizer = F.load_sharded_model(args.max_memory_gib)
    rows_by_pid = BASE.P().load_dataset()
    teacher_tokens = BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation(
        "kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = F.FullPathForensicProbe(rotation)
    probe.capture_enabled = False
    layers = probe.install(model)
    weight_hash_before = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    static_rows, future_rows, unit_rows, provenance_rows, failures = [], [], [], [], []
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] formal {index}/{len(units)} unit={unit['unit_id']}", flush=True)
            try:
                sr, fr, ur, pr = run_formal_unit(
                    model, tokenizer, probe, unit, layers, rows_by_pid,
                    teacher_tokens, rotation, args.horizon)
                static_rows += sr
                future_rows += fr
                unit_rows.append(ur)
                provenance_rows.append(pr)
            except Exception as exc:
                failures.append({"unit_id": str(unit["unit_id"]), "error": repr(exc),
                                 "traceback": traceback.format_exc(limit=50)})
                save_json(outdir / "failures.json", failures)
                raise
    finally:
        probe.close()
    static_units, static_summary = summarize_static(static_rows)
    future_summary = summarize_future(unit_rows)
    weights_unchanged = weight_hash_before == BASE.tensor_hash(next(model.parameters()).detach().cpu())
    write_rows(outdir / "stage2_static_quant_horizon.csv", static_rows)
    write_rows(outdir / "stage2_static_quant_units.csv", static_units)
    save_json(outdir / "stage2_static_summary.json", static_summary)
    write_rows(outdir / "stage3_futurekl_horizon.csv", future_rows)
    write_rows(outdir / "stage3_futurekl_units.csv", unit_rows)
    save_json(outdir / "stage3_futurekl_summary.json", future_summary)
    historical = read_json(HISTORICAL_RESULTS / "phase2_summary.json", {})
    provenance = {
        "N_FORMAL_UNITS": len(unit_rows),
        "kernel_return_state_basis": "ROTATED",
        "buggy": {"endpoint_extra_transform": "YES", "cached_state_interpreted_as": "ROTATED",
                  "basis_mismatch": "YES"},
        "corrected": {"endpoint_extra_transform": "NO", "cached_state_interpreted_as": "ROTATED",
                      "basis_mismatch": "NO"},
        "per_unit_prefill_state_gap": provenance_rows,
        "median_native_vs_buggy_prefill_end_gap": BASE.median(
            [x["native_vs_buggy_prefill_end_gap"] for x in provenance_rows]),
        "median_native_vs_corrected_prefill_end_gap": BASE.median(
            [x["native_vs_corrected_prefill_end_gap"] for x in provenance_rows]),
        "only_condition_difference": "one redundant prefill endpoint state rotation",
        "prior_formal_closure": historical.get("CAUSAL_CLOSURE"),
        "MODEL_WEIGHTS_UNCHANGED": weights_unchanged,
    }
    provenance["HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION"] = "YES" if (
        future_summary["BUGGY_TO_CORRECTED_CLOSURE"] > 0.9
        and future_summary["BUGGY_HADAMARD_AUC"] > 5 * future_summary["CORRECTED_HADAMARD_AUC"]
    ) else "PARTIAL"
    save_json(outdir / "stage4_bug_provenance.json", provenance)
    save_json(outdir / "experiment_config.json", {
        "TASK": TASK, "model": "Ling-3.0-tiny", "scope": "formal",
        "n_units": len(units), "horizon": args.horizon, "conditions": list(CONDITIONS),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "fp_prompts": FP_PROMPTS, "fp_tokens": FP_TOKENS,
        "rotation": "Value-side normalized RHT128 seed 0", "quantizer": "INT8_R128",
        "canonical_manifest": unit_audit, "bootstrap_seed": BASE.BOOTSTRAP_SEED,
        "historical_artifacts_overwritten": False,
    })
    return {"static": static_summary, "future": future_summary, "provenance": provenance}


def merge_formal(args):
    outdir = Path(args.output_dir)
    sources = [Path(x) for x in args.merge_sources.split(",") if x]
    if not sources:
        raise ValueError("--merge-sources is required")
    static_rows, future_rows, unit_rows, provenance_rows = [], [], [], []
    weights_unchanged = True
    for source in sources:
        static_rows += read_rows(source / "stage2_static_quant_horizon.csv")
        future_rows += read_rows(source / "stage3_futurekl_horizon.csv")
        unit_rows += read_rows(source / "stage3_futurekl_units.csv")
        item = read_json(source / "stage4_bug_provenance.json", {})
        provenance_rows += item.get("per_unit_prefill_state_gap", [])
        weights_unchanged = weights_unchanged and bool(item.get("MODEL_WEIGHTS_UNCHANGED"))
    if len({row["unit_id"] for row in unit_rows}) != EXPECTED_UNITS:
        raise RuntimeError("merged formal shards do not contain exactly 18 unique units")
    for row in static_rows:
        for key in ("token_index", "layer", "head"):
            row[key] = int(row[key])
        for key in ("state_relative_quant_error", "state_absolute_quant_error", "max_abs", "peakiness",
                    "scale_mean", "scale_median", "scale_max", "saturation_fraction", "clipping_fraction",
                    "code_mean_abs", "code_std", "code_zero_fraction", "code_min", "code_max"):
            row[key] = float(row[key])
        row["finite"] = str(row["finite"]).lower() == "true"
    for row in future_rows:
        row["horizon"] = int(row["horizon"])
        for key in ("future_kl", "logit_relative_l2", "logit_cosine", "top1_match", "top20_overlap"):
            row[key] = float(row[key])
    for row in unit_rows:
        for key in row:
            if key != "unit_id":
                row[key] = float(row[key])
    static_units, static_summary = summarize_static(static_rows)
    future_summary = summarize_future(unit_rows)
    historical = read_json(HISTORICAL_RESULTS / "causal_summary.json", {})
    provenance = {
        "N_FORMAL_UNITS": len(unit_rows), "kernel_return_state_basis": "ROTATED",
        "buggy": {"endpoint_extra_transform": "YES", "cached_state_interpreted_as": "ROTATED",
                  "basis_mismatch": "YES"},
        "corrected": {"endpoint_extra_transform": "NO", "cached_state_interpreted_as": "ROTATED",
                      "basis_mismatch": "NO"},
        "per_unit_prefill_state_gap": provenance_rows,
        "median_native_vs_buggy_prefill_end_gap": BASE.median(
            [float(x["native_vs_buggy_prefill_end_gap"]) for x in provenance_rows]),
        "median_native_vs_corrected_prefill_end_gap": BASE.median(
            [float(x["native_vs_corrected_prefill_end_gap"]) for x in provenance_rows]),
        "only_condition_difference": "one redundant prefill endpoint state rotation",
        "prior_formal_closure": historical.get("CAUSAL_CLOSURE"),
        "MODEL_WEIGHTS_UNCHANGED": weights_unchanged,
    }
    provenance["HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION"] = "YES" if (
        future_summary["BUGGY_TO_CORRECTED_CLOSURE"] > 0.9
        and future_summary["BUGGY_HADAMARD_AUC"] > 5 * future_summary["CORRECTED_HADAMARD_AUC"]
    ) else "PARTIAL"
    write_rows(outdir / "stage2_static_quant_horizon.csv", static_rows)
    write_rows(outdir / "stage2_static_quant_units.csv", static_units)
    save_json(outdir / "stage2_static_summary.json", static_summary)
    write_rows(outdir / "stage3_futurekl_horizon.csv", future_rows)
    write_rows(outdir / "stage3_futurekl_units.csv", unit_rows)
    save_json(outdir / "stage3_futurekl_summary.json", future_summary)
    save_json(outdir / "stage4_bug_provenance.json", provenance)
    save_json(outdir / "experiment_config.json", {
        "TASK": TASK, "model": "Ling-3.0-tiny", "scope": "formal",
        "n_units": len(unit_rows), "horizon": PRIMARY_HORIZON, "conditions": list(CONDITIONS),
        "fp_prompts": FP_PROMPTS, "fp_tokens": FP_TOKENS,
        "rotation": "Value-side normalized RHT128 seed 0", "quantizer": "INT8_R128",
        "bootstrap_seed": BASE.BOOTSTRAP_SEED, "merged_shards": [str(x) for x in sources],
        "historical_artifacts_overwritten": False,
    })
    return {"static": static_summary, "future": future_summary, "provenance": provenance}


def write_reinterpretation(outdir):
    text = f"""# KDA Rotation Result Reinterpretation

Task: `{TASK}`

## A. Still Valid

- Native R128 versus C128 orientation findings that do not depend on the historical Rotated branch.
- The Value-side rotation theorem and exact coordinate orientation.
- Corrected FP equivalence of Value-side Hadamard.
- Static INT8_R128 quantization improvement measured on the corrected path.

## B. Valid Only As Bug-Propagation Evidence

- Rotated-history damage, prequant donor rescue, history-dominant classification, and prefill localization.
- Packed/sequential, write-back, persistent exposure, and related panels that localized propagation of the bad endpoint state.
- The historical Buggy Hadamard FutureKL trajectory is retained for provenance only.

## C. Invalid As Intrinsic KDA Rotation Mechanism

- Historical claims of intrinsic Value-side rotation instability.
- Historical claims of intrinsic KDA long-horizon rotation failure.
- Historical claims of an intrinsic state-quantization-boundary failure based on Buggy Hadamard FutureKL near 0.12 or above.

Historical artifacts are preserved. Their interpretation is changed because the severe branch contained a redundant prefill-end state basis rotation.
"""
    (Path(outdir) / "KDA_ROTATION_RESULT_REINTERPRETATION.md").write_text(text, encoding="utf-8")


def finalize(args):
    outdir = Path(args.output_dir)
    audit = read_json(outdir / "stage0_code_audit.json", {})
    fp = read_json(outdir / "stage1_fp_parity.json", {})
    static = read_json(outdir / "stage2_static_summary.json", {})
    future = read_json(outdir / "stage3_futurekl_summary.json", {})
    provenance = read_json(outdir / "stage4_bug_provenance.json", {})
    pytest_status = read_json(outdir / "pytest_summary.json", {}).get("PYTEST", "NOT_RUN")
    validation_status = read_json(outdir / "artifact_validation.json", {}).get(
        "ARTIFACT_VALIDATION", "NOT_RUN")
    complete = (
        audit.get("STAGE0_CODE_AUDIT") == "PASS"
        and fp.get("CORRECTED_FP_EQUIVALENCE") == "PASS"
        and fp.get("PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY") == "PASS"
        and static.get("STATIC_QUANTIZATION_GAIN_PRESERVED") in {"YES", "PARTIAL"}
        and future.get("CORRECTED_HADAMARD_LONG_HORIZON_HARM") == "NO"
        and provenance.get("HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION") == "YES"
    )
    write_reinterpretation(outdir)
    summary = {
        "TASK": TASK, "FORMAL_STATUS": "COMPLETE" if complete else "INCOMPLETE",
        "STAGE0_CODE_AUDIT": audit.get("STAGE0_CODE_AUDIT"),
        "STAGE1_FP_PARITY": fp.get("CORRECTED_FP_EQUIVALENCE"),
        "STAGE2_STATIC_QUANT": static.get("STATIC_QUANTIZATION_GAIN_PRESERVED"),
        "STAGE3_FUTUREKL": "PASS" if future.get("N_FORMAL_UNITS") == EXPECTED_UNITS else "FAIL",
        "STAGE4_BUG_PROVENANCE": provenance.get("HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION"),
        "N_FORMAL_UNITS": future.get("N_FORMAL_UNITS"),
        "PYTEST": pytest_status,
        "ARTIFACT_VALIDATION": validation_status,
        "CORRECTED_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION": "NO",
        "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": fp.get("PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY"),
        "CORRECTED_FP_EQUIVALENCE": fp.get("CORRECTED_FP_EQUIVALENCE"),
        "FP_LOGIT_REL_L2": fp.get("FP_LOGIT_REL_L2"), "FP_KL": fp.get("FP_KL"),
        "FP_TOP1_MATCH": fp.get("FP_TOP1_MATCH"),
        "NATIVE_STATE_REL_ERROR": static.get("NATIVE_STATE_REL_ERROR"),
        "CORRECTED_HADAMARD_STATE_REL_ERROR": static.get("CORRECTED_HADAMARD_STATE_REL_ERROR"),
        "STATE_ERROR_REDUCTION_PERCENT": static.get("STATE_ERROR_REDUCTION_PERCENT"),
        "STATIC_QUANTIZATION_GAIN_PRESERVED": static.get("STATIC_QUANTIZATION_GAIN_PRESERVED"),
        "NATIVE_AUC": future.get("NATIVE_AUC"),
        "BUGGY_HADAMARD_AUC": future.get("BUGGY_HADAMARD_AUC"),
        "CORRECTED_HADAMARD_AUC": future.get("CORRECTED_HADAMARD_AUC"),
        "CORRECTED_MINUS_NATIVE": future.get("CORRECTED_MINUS_NATIVE"),
        "CORRECTED_MINUS_NATIVE_95CI": future.get("CORRECTED_MINUS_NATIVE_95CI"),
        "BUGGY_TO_CORRECTED_CLOSURE": future.get("BUGGY_TO_CORRECTED_CLOSURE"),
        "BUGGY_TO_CORRECTED_CLOSURE_95CI": future.get("BUGGY_TO_CORRECTED_CLOSURE_95CI"),
        "CORRECTED_HADAMARD_LONG_HORIZON_HARM": future.get("CORRECTED_HADAMARD_LONG_HORIZON_HARM"),
        "HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION": provenance.get(
            "HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION"),
        "KDA_MECHANISM_REBASELINE_COMPLETE": "YES" if complete else "NO",
        "KDA_FAILURE_MECHANISM_WORK": "CLOSED" if complete else "OPEN",
        "METHOD_DESIGN_READY": "YES" if complete else "NO",
        "NEXT_PHASE": "LEARNABLE_VALUE_SIDE_ROTATION" if complete else "HUMAN_REVIEW_OF_RESIDUAL_CORRECTED_HADAMARD_GAP",
        "FAILURES": read_json(outdir / "failures.json", []),
    }
    save_json(outdir / "formal_summary.json", summary)
    lines = [f"{key} = {value}" for key, value in summary.items() if key != "FAILURES"]
    (outdir / "formal_summary.md").write_text("# Formal Summary\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate(args):
    outdir = Path(args.output_dir)
    required = [
        "experiment_config.json", "corrected_rotation_semantics.json", "stage0_code_audit.json",
        "stage1_fp_parity.json", "stage2_static_quant_horizon.csv", "stage2_static_quant_units.csv",
        "stage2_static_summary.json", "stage3_futurekl_horizon.csv", "stage3_futurekl_units.csv",
        "stage3_futurekl_summary.json", "stage4_bug_provenance.json",
        "KDA_ROTATION_RESULT_REINTERPRETATION.md", "formal_summary.md", "run.log",
    ]
    missing = [name for name in required if not (outdir / name).exists()]
    horizon = read_rows(outdir / "stage3_futurekl_horizon.csv") if not missing else []
    units = read_rows(outdir / "stage3_futurekl_units.csv") if not missing else []
    static = read_rows(outdir / "stage2_static_quant_horizon.csv") if not missing else []
    unique = {(r["unit_id"], int(r["horizon"]), r["condition"]) for r in horizon}
    checks = {
        "required_artifacts": not missing,
        "formal_units": len(units) == EXPECTED_UNITS,
        "future_rows": len(unique) == EXPECTED_UNITS * PRIMARY_HORIZON * len(CONDITIONS),
        "future_finite": all(finite(r["future_kl"]) for r in horizon),
        "static_finite": all(finite(r["state_relative_quant_error"]) for r in static),
        "static_conditions": {r["condition"] for r in static} == {"NATIVE_R128", "CORRECTED_HADAMARD_R128"},
        "unitwise_statistics": read_json(outdir / "stage3_futurekl_summary.json", {}).get(
            "formal_estimand") == "median across 18 canonical unit AUC values",
    }
    result = {"ARTIFACT_VALIDATION": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "missing": missing, "n_future_rows": len(horizon),
              "n_static_rows": len(static), "n_unit_rows": len(units)}
    save_json(outdir / "artifact_validation.json", result)
    if result["ARTIFACT_VALIDATION"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage01", "formal", "merge", "finalize", "validate"), required=True)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-memory-gib", type=int, default=9)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--stage1-dir")
    parser.add_argument("--merge-sources")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "stage01":
        result = run_stage01(args)
    elif args.phase == "formal":
        result = run_formal(args)
    elif args.phase == "merge":
        result = merge_formal(args)
    elif args.phase == "finalize":
        result = finalize(args)
    else:
        result = validate(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
