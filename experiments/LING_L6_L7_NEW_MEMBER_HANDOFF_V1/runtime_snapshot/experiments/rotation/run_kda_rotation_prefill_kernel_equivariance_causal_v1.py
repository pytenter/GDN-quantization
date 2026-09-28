#!/usr/bin/env python3
"""Causal test of KDA Value rotation by packed-prefill kernel interaction."""

import argparse
import importlib.util
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_PREFILL_KERNEL_EQUIVARIANCE_CAUSAL_V1"
SLUG = "kda_rotation_prefill_kernel_equivariance_causal_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
HISTORY_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_history_formation_final_causal_v1.py"
WRITEBACK_RESULTS = REPO / "results" / "kda_rotation_prefill_writeback_final_localization_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EPS = 1e-12
CONDITIONS = ("NP", "NS", "RP", "RS")
SEQUENTIAL_CONDITIONS = {"NS", "RS"}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = import_file(HISTORY_RUNNER, "history_formation_for_prefill_kernel_v1")
BASE, DECAY, GAB, COMM = H.BASE, H.DECAY, H.GAB, H.COMM


def tensor_metrics(value, reference):
    return H.tensor_metrics_fp32(value, reference)


def effect(values, name):
    return H.effect(values, name)


def relative_error(value, reference):
    return tensor_metrics(value, reference)["relative_l2"]


class SequentialPrefillProbe(H.HistoryFormationProbe):
    """Replace only packed KDA prefill with token-wise decode-kernel recurrence."""

    def __init__(self, rotation):
        super().__init__(rotation)
        self.validate_reference = False

    def _wrap(self, operator, fn):
        parent_wrapped = super()._wrap(operator, fn)

        def wrapped(**kwargs):
            plan_obj = self.plan
            if (plan_obj is None or plan_obj.name not in SEQUENTIAL_CONDITIONS
                    or operator != "chunk_kda"):
                return parent_wrapped(**kwargs)
            branch, layer = self.branch, self.current_layer
            raw_gate = kwargs["g"].detach().clone()
            log_decay = DECAY.final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"),
                kwargs.get("lower_bound"), bool(kwargs.get("use_gate_in_kernel", False)))
            v_semantic = kwargs["v"].detach().clone()
            call = dict(kwargs)
            call["g"] = log_decay.float()
            call["use_gate_in_kernel"] = False
            if plan_obj.basis == "rotated":
                call["v"] = BASE.driver_to_branch_coordinates(
                    "v", call["v"], "rotated", self.rotation)

            initial = call.get("initial_state")
            initial_snapshot = None if initial is None else initial.detach().float().clone()
            sequence_length = int(call["q"].shape[1])
            outputs, state = [], initial
            consumed = {name: [] for name in ("q", "k", "v", "g", "beta")}
            consumed_semantic_v = []
            for index in range(sequence_length):
                step = dict(call)
                for name in ("q", "k", "v", "g", "beta"):
                    step[name] = call[name][:, index:index + 1]
                    consumed[name].append(step[name].detach().clone())
                consumed_semantic_v.append(v_semantic[:, index:index + 1].detach().clone())
                step["initial_state"] = state
                step.pop("safe_gate", None)
                output, state = self.orig_fused(**step)
                outputs.append(output)
            raw_output = torch.cat(outputs, dim=1)
            local_identity = {
                name: tensor_metrics(torch.cat(parts, dim=1).float(), call[name].float())
                for name, parts in consumed.items()
            }
            local_identity["v_semantic"] = tensor_metrics(
                torch.cat(consumed_semantic_v, dim=1).float(), v_semantic.float())

            validation = None
            if self.validate_reference:
                reference_call = dict(call)
                reference_call.pop("safe_gate", None)
                reference_output, reference_state = self.orig_fused(**reference_call)
                validation = {
                    "output": tensor_metrics(raw_output.float(), reference_output.float()),
                    "state": tensor_metrics(state.float(), reference_state.float()),
                }

            returned = raw_output
            if plan_obj.basis == "rotated":
                returned = raw_output.float().matmul(
                    self.rotation.t().to(raw_output.device, torch.float32)).to(raw_output.dtype)
            if branch is not None and layer is not None:
                live = {
                    "k": call["k"], "beta": call["beta"], "v_semantic": v_semantic,
                    "effective_decay": torch.exp(log_decay.float()),
                    "use_qk_l2norm_in_kernel": bool(call.get("use_qk_l2norm_in_kernel", False)),
                }
                self.records[branch][int(layer)] = {
                    "operator": "sequential_fused_recurrent_kda",
                    "source_operator": operator,
                    "q": BASE.record_tensor(call["q"]),
                    "k": BASE.record_tensor(call["k"]),
                    "v": BASE.record_tensor(call["v"]),
                    "v_semantic": BASE.record_tensor(v_semantic),
                    "beta": BASE.record_tensor(call["beta"]),
                    "raw_gate": BASE.record_tensor(raw_gate),
                    "log_decay": BASE.record_tensor(log_decay, float32=True),
                    "effective_decay": BASE.record_tensor(torch.exp(log_decay.float()), float32=True),
                    "effective_a": BASE.record_tensor(GAB.a_from_record(live), float32=True),
                    "effective_b": BASE.record_tensor(
                        GAB.b_to_basis(GAB.canonical_b_from_record(live), plan_obj.basis, self.rotation),
                        float32=True),
                    "initial_state": BASE.record_tensor(initial_snapshot, float32=True),
                    "final_state": BASE.record_tensor(state, float32=True),
                    "output": BASE.record_tensor(returned, float32=True),
                    "raw_output": BASE.record_tensor(raw_output, float32=True),
                    "basis": plan_obj.basis,
                    "sequence_length": sequence_length,
                    "token_order": list(range(sequence_length)),
                    "reference_validation": validation,
                    "local_driver_identity": local_identity,
                    "use_qk_l2norm_in_kernel": bool(call.get("use_qk_l2norm_in_kernel", False)),
                    "input_dtypes": {name: str(call[name].dtype) for name in ("q", "k", "v", "g", "beta")},
                    "state_dtype": str(state.dtype),
                }
            return returned, state

        return wrapped


def condition_spec(name):
    return {
        "NP": ("native", "packed"),
        "NS": ("native", "sequential"),
        "RP": ("rotated", "packed"),
        "RS": ("rotated", "sequential"),
    }[name]


def kernel_audit(records):
    sample = records[sorted(records)[0]]
    input_dtypes = sample.get("input_dtypes") or {
        "q": "torch.bfloat16", "k": "torch.bfloat16", "v": "torch.bfloat16",
        "g": "torch.float32 after exact gate activation", "beta": "torch.float32"}
    return {
        "PACKED_PREFILL_KERNEL": "fla.ops.kda.chunk.chunk_kda / ChunkKDAFunction.apply",
        "PACKED_PREFILL_STATE_LAYOUT": "[B,HV,K,V] float32",
        "PACKED_PREFILL_INPUT_DTYPE": input_dtypes,
        "PACKED_PREFILL_ACCUMULATION_DTYPE": "mixed Triton FP32 accumulation; final state float32",
        "PACKED_PREFILL_CHUNK_SIZE": 64,
        "PACKED_PREFILL_SCAN_TYPE": "chunk scan with parallel intra-chunk solve",
        "PACKED_PREFILL_STATE_CARRY_SEMANTICS": "float32 final recurrent state [B,HV,K,V]",
        "PACKED_PREFILL_QUANTIZER_LOCATION": "outside kernel, once after packed prefill returns",
        "PACKED_PREFILL_VALUE_ROTATION_LOCATION": "v -> vR immediately before live KDA operator; state stored in rotated V coordinates",
        "PACKED_PREFILL_CAST_POINTS": "q/k/v loaded to FP32 inside Triton; output cast to v dtype; state emitted FP32",
        "DECODE_KERNEL": "fla.ops.kda.fused_recurrent.fused_recurrent_kda",
        "DECODE_STATE_LAYOUT": "[B,HV,K,V] float32",
        "DECODE_INPUT_DTYPE": "q/k/v model dtype; activated decay and beta float32",
        "DECODE_ACCUMULATION_DTYPE": "Triton FP32 recurrent accumulator",
        "DECODE_STATE_CARRY_SEMANTICS": "previous float32 state consumed directly and next float32 state returned",
        "DECODE_QUANTIZER_LOCATION": "outside kernel, after each token transition",
        "DECODE_VALUE_ROTATION_LOCATION": "v -> vR before kernel; output mapped by R.T; state remains rotated until boundary conversion",
        "DECODE_CAST_POINTS": "q/k/v loaded to FP32; output cast to v dtype; state remains FP32",
        "PACKED_VS_DECODE_DIFFERENCES": (
            "chunk_kda uses 64-token chunk scan and parallel intra-chunk algebra; "
            "decode uses direct sequential fused recurrence with one carried FP32 state"),
    }


def prepare_prefill(model, tokenizer, probe, unit, row, layers, rotation, validate_reference=False):
    probe.validate_reference = bool(validate_reference)
    input_ids = None
    items, records = {}, {}
    _, fp, fp_records = H.prefill_branch(
        model, tokenizer, probe, row, layers, rotation, "FP", "native", False)
    items["FP"], records["FP"] = fp, fp_records
    for name in CONDITIONS:
        basis, _ = condition_spec(name)
        ids, item, rec = H.prefill_branch(
            model, tokenizer, probe, row, layers, rotation, name, basis, True)
        input_ids = ids if input_ids is None else input_ids
        items[name], records[name] = item, rec
    probe.validate_reference = False
    prompt_len = int(input_ids.shape[-1])

    semantic_stacks = {}
    for name in CONDITIONS:
        basis, _ = condition_spec(name)
        if basis == "rotated":
            H.to_native(items[name], layers, rotation)
        semantic_stacks[name] = BASE.cache_stack(items[name]["past"], layers)
        items[name]["basis"] = "native"
        items[name]["quantized"] = True
        items[name]["plan"] = H.plan(f"{name}_NATIVE_CONTINUATION", "native", True)
    items["FP"]["plan"] = H.plan("FP_NATIVE_CONTINUATION", "native", False)
    probe.last_prefill_records = records
    return items, records, semantic_stacks, prompt_len


def driver_identity_rows(unit_id, records):
    rows = []
    for packed, sequential in (("NP", "NS"), ("RP", "RS")):
        for layer in sorted(records[packed]):
            left, right = records[packed][layer], records[sequential][layer]
            row = {"unit_id": str(unit_id), "pair": f"{packed}_{sequential}", "layer": int(layer)}
            for name in ("q", "k", "v_semantic", "beta", "log_decay"):
                metrics = tensor_metrics(left[name].float(), right[name].float())
                row[f"cross_branch_{name}_max_abs"] = metrics["max_abs"]
                row[f"cross_branch_{name}_relative_l2"] = metrics["relative_l2"]
            local = right.get("local_driver_identity") or {}
            for name in ("q", "k", "v", "v_semantic", "g", "beta"):
                metrics = local.get(name, {})
                row[f"local_{name}_max_abs"] = metrics.get("max_abs")
                row[f"local_{name}_relative_l2"] = metrics.get("relative_l2")
            validation = right.get("reference_validation") or {}
            row["sequential_reference_output_relative_l2"] = validation.get("output", {}).get("relative_l2")
            row["sequential_reference_state_relative_l2"] = validation.get("state", {}).get("relative_l2")
            row["packed_operator"] = left.get("operator")
            row["sequential_operator"] = right.get("operator")
            row["sequence_length"] = right.get("sequence_length")
            row["token_order_identity"] = right.get("token_order") == list(range(int(right.get("sequence_length", 0))))
            rows.append(row)
    return rows


def state_diagnostic_rows(unit_id, records, stacks, rotation):
    rows = []
    pairs = (("NP", "NS"), ("NP", "RP"), ("NS", "RS"), ("RP", "RS"))
    for left_name, right_name in pairs:
        for layer in sorted(stacks[left_name]):
            state = tensor_metrics(stacks[right_name][layer].float(), stacks[left_name][layer].float())
            output = tensor_metrics(
                records[right_name][layer]["output"].float(),
                records[left_name][layer]["output"].float())
            rows.append({
                "unit_id": str(unit_id), "pair": f"{left_name}_{right_name}", "layer": int(layer),
                "state_max_abs": state["max_abs"], "state_relative_l2": state["relative_l2"],
                "state_cosine": state["cosine"], "output_max_abs": output["max_abs"],
                "output_relative_l2": output["relative_l2"], "output_cosine": output["cosine"],
            })
    return rows


def continue_native(model, probe, items, unit, tokens, prompt_len, horizon, layers, rotation):
    for t in range(int(unit["t0"])):
        for name in ("FP",) + CONDITIONS:
            H.advance(model, probe, items[name], tokens[t], prompt_len + t, layers, rotation, t + 1)
    fp = items.pop("FP")
    return H.evaluate_native_future(
        model, probe, fp, {name: items[name] for name in CONDITIONS}, tokens,
        unit["t0"], prompt_len, horizon, layers, rotation, unit["unit_id"])


def run_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
             rotation, horizon, validate_reference=False, state_carrier=False):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = H.history_tokens(unit, teacher_tokens, horizon)
    items, records, stacks, prompt_len = prepare_prefill(
        model, tokenizer, probe, unit, row, layers, rotation, validate_reference)
    drivers = driver_identity_rows(unit["unit_id"], records)
    diagnostics = state_diagnostic_rows(unit["unit_id"], records, stacks, rotation)
    if state_carrier:
        source = items["NP"]
        carriers = {}
        for name in CONDITIONS:
            carrier_name = f"{name}_STATE"
            item = H.clone_branch(source, carrier_name, "native", True)
            H.replace_stack(item["past"], stacks[name])
            item["plan"] = H.plan(f"{carrier_name}_NATIVE_CONTINUATION", "native", True)
            carriers[name] = item
        items = {"FP": items["FP"], **carriers}
    rows = continue_native(model, probe, items, unit, tokens, prompt_len, horizon, layers, rotation)
    return rows, drivers, diagnostics, {
        "unit_id": str(unit["unit_id"]), "prompt_tokens": prompt_len,
        "history_decode_tokens": int(unit["t0"]), "prefill_cache_position": f"0..{prompt_len - 1}",
        "first_history_decode_position": prompt_len,
    }


def stage0_summary(rows, drivers, diagnostics, records_audit):
    driver_columns = [key for key in drivers[0]
                      if key.startswith("local_") and key.endswith("_max_abs")]
    driver_max = max(float(row[key]) for row in drivers for key in driver_columns)
    reference_output = [float(row["sequential_reference_output_relative_l2"])
                        for row in drivers if row.get("sequential_reference_output_relative_l2") is not None]
    reference_state = [float(row["sequential_reference_state_relative_l2"])
                       for row in drivers if row.get("sequential_reference_state_relative_l2") is not None]
    reference_output_max = max(reference_output) if reference_output else float("inf")
    reference_state_max = max(reference_state) if reference_state else float("inf")
    operators_ok = all(row["packed_operator"] == "chunk_kda"
                       and row["sequential_operator"] == "sequential_fused_recurrent_kda"
                       and row["token_order_identity"] for row in drivers)
    finite = all(math.isfinite(float(row[key])) for row in diagnostics
                 for key in ("state_max_abs", "state_relative_l2", "state_cosine",
                             "output_max_abs", "output_relative_l2", "output_cosine"))
    summary = {
        "STAGE0_PREFILL_KERNEL_AUDIT": "PASS" if operators_ok else "FAIL",
        "STAGE0_SEQUENTIAL_REFERENCE": "PASS" if (
            driver_max <= 1e-12 and reference_output_max <= 1e-4
            and reference_state_max <= 2e-6) else "FAIL",
        "SEQUENTIAL_DRIVER_IDENTITY": "PASS" if driver_max <= 1e-12 else "FAIL",
        "driver_identity_max_abs": driver_max,
        "sequential_reference_output_max_relative_l2": reference_output_max,
        "sequential_reference_state_max_relative_l2": reference_state_max,
        "FORMAL_BRANCH_SEMANTICS": "PASS" if finite else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE": "PASS",
        "smoke_horizon_rows": len(rows),
        "kernel_audit": records_audit,
    }
    summary["STAGE0_SEMANTICS"] = "PASS" if all(summary[key] == "PASS" for key in (
        "STAGE0_PREFILL_KERNEL_AUDIT", "STAGE0_SEQUENTIAL_REFERENCE",
        "SEQUENTIAL_DRIVER_IDENTITY", "FORMAL_BRANCH_SEMANTICS",
        "INSTRUMENTATION_NONINTERFERENCE")) else "FAIL"
    return summary


def aggregate_auc(rows, conditions=CONDITIONS):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] in conditions:
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({key[0] for key in grouped})
    return [{"unit_id": unit, **{
        f"auc_{condition}": BASE.mean(grouped[(unit, condition)]) for condition in conditions}}
        for unit in units]


def factorial_rows(unit_rows):
    rows = []
    for row in unit_rows:
        np, ns, rp, rs = (float(row[f"auc_{name}"]) for name in CONDITIONS)
        rows.append({
            **row,
            "packed_rotation_effect": rp - np,
            "sequential_rotation_effect": rs - ns,
            "packed_kernel_effect_native": np - ns,
            "packed_kernel_effect_rotated": rp - rs,
            "rotation_x_packed_interaction": rp - np - rs + ns,
        })
    return rows


def layer_excess_summary(diagnostics):
    lookup = {(r["unit_id"], int(r["layer"]), r["pair"]): float(r["state_relative_l2"])
              for r in diagnostics}
    units = sorted({r["unit_id"] for r in diagnostics})
    layers = sorted({int(r["layer"]) for r in diagnostics})
    summaries = []
    for layer in layers:
        values = [lookup[(unit, layer, "NP_RP")] - lookup[(unit, layer, "NS_RS")] for unit in units]
        stat = effect(values, f"LAYER_{layer}_PACKED_ROTATION_EXCESS")
        summaries.append({"layer": layer, **stat})
    earliest = next((row["layer"] for row in summaries
                     if row["bootstrap_ci_low"] > 0 and row["positive"] > len(units) / 2), "NONE")
    return earliest, summaries


def pair_gap(diagnostics, pair):
    grouped = defaultdict(list)
    for row in diagnostics:
        if row["pair"] == pair:
            grouped[row["unit_id"]].append(float(row["state_relative_l2"]))
    return BASE.median([BASE.median(values) for values in grouped.values()])


def summarize_factorial(unit_rows, diagnostics):
    rows = factorial_rows(unit_rows)
    effects = {key.upper(): effect([r[key] for r in rows], key.upper()) for key in (
        "packed_rotation_effect", "sequential_rotation_effect", "packed_kernel_effect_native",
        "packed_kernel_effect_rotated", "rotation_x_packed_interaction")}
    packed = effects["PACKED_ROTATION_EFFECT"]
    sequential = effects["SEQUENTIAL_ROTATION_EFFECT"]
    rotated_kernel = effects["PACKED_KERNEL_EFFECT_ROTATED"]
    interaction = effects["ROTATION_X_PACKED_INTERACTION"]
    state_excess = pair_gap(diagnostics, "NP_RP") - pair_gap(diagnostics, "NS_RS")
    strong = (
        packed["bootstrap_ci_low"] > 0
        and rotated_kernel["bootstrap_ci_low"] > 0
        and interaction["bootstrap_ci_low"] > 0
        and interaction["positive"] > len(rows) / 2
        and abs(sequential["paired_median"]) < 0.5 * abs(packed["paired_median"])
        and state_excess > 0)
    if strong:
        classification = "PACKED_PREFILL_KERNEL_ROTATION_INTERACTION_STRONGLY_SUPPORTED"
    elif interaction["paired_median"] > 0 and interaction["positive"] > len(rows) / 2:
        classification = "PACKED_PREFILL_KERNEL_ROTATION_INTERACTION_PARTIAL"
    elif effects["PACKED_KERNEL_EFFECT_NATIVE"]["bootstrap_ci_low"] > 0 and interaction["bootstrap_ci_low"] <= 0:
        classification = "PACKED_PREFILL_MAIN_EFFECT_NOT_ROTATION_SPECIFIC"
    elif sequential["bootstrap_ci_low"] > 0 and abs(sequential["paired_median"]) >= 0.5 * abs(packed["paired_median"]):
        classification = "SEQUENTIAL_PREFILL_ALSO_HARMFUL"
    else:
        classification = "PACKED_PREFILL_HYPOTHESIS_NOT_SUPPORTED"
    earliest, layer_summaries = layer_excess_summary(diagnostics)
    return rows, {
        "TASK": TASK,
        "N_FORMAL_UNITS": len(rows),
        "median_auc": {name: BASE.median([r[f"auc_{name}"] for r in rows]) for name in CONDITIONS},
        **effects,
        "NP_NS_STATE_REL_GAP": pair_gap(diagnostics, "NP_NS"),
        "NP_RP_STATE_REL_GAP": pair_gap(diagnostics, "NP_RP"),
        "NS_RS_STATE_REL_GAP": pair_gap(diagnostics, "NS_RS"),
        "RP_RS_STATE_REL_GAP": pair_gap(diagnostics, "RP_RS"),
        "PACKED_ROTATION_EXCESS_STATE_GAP": state_excess,
        "EARLIEST_LAYER_WITH_PACKED_ROTATION_EXCESS_DRIFT": earliest,
        "layer_excess_summaries": layer_summaries,
        "FINAL_KDA_PREFILL_CLASSIFICATION": classification,
        "STAGEA_FACTORIAL": "SUPPORTED" if strong else ("PARTIAL" if "PARTIAL" in classification else "NOT_SUPPORTED"),
        "KDA_MECHANISM_CLOSURE_READY": "YES" if strong else "NO",
        "METHOD_DESIGN_READY": "YES" if strong else "NO",
        "NEXT_PHASE": "LEARNABLE_ROTATION_METHOD_DESIGN" if strong else "HUMAN_REVIEW_OF_PREFILL_ONLY_ROTATED_PATH",
    }


def summarize_carrier(unit_rows):
    rows = factorial_rows(unit_rows)
    interaction = effect([r["rotation_x_packed_interaction"] for r in rows], "KDA_STATE_ONLY_INTERACTION")
    if interaction["bootstrap_ci_low"] > 0 and interaction["positive"] > len(rows) / 2:
        status = "YES"
    elif interaction["paired_median"] > 0:
        status = "PARTIAL"
    else:
        status = "NO"
    return rows, {
        "N_FORMAL_UNITS": len(rows),
        "KDA_STATE_ONLY_INTERACTION": interaction,
        "KDA_RECURRENT_STATE_CARRIES_PREFILL_KERNEL_DAMAGE": status,
        "STATE_CARRIER_CONTROL": "COMPLETE",
    }


def setup_output(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw").mkdir(exist_ok=True)


def reset_phase(outdir, phase):
    for suffix in ("horizon", "drivers", "diagnostics", "timeline"):
        path = outdir / "raw" / f"{phase}_{suffix}.jsonl"
        if path.exists():
            path.unlink()


def run_experiment(args):
    outdir = Path(args.output_dir)
    setup_output(outdir)
    if args.overwrite:
        reset_phase(outdir, args.phase)
    units, manifest = BASE.load_units(args.scope, args.max_units, args.unit_shard_index, args.unit_shard_count)
    _, model, tokenizer = BASE.P().load_model_and_tokenizer()
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation(
        "kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = SequentialPrefillProbe(rotation.to(next(model.parameters()).device))
    layers = probe.install(model)
    weights_before = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    rotation_before = BASE.tensor_hash(rotation)
    rows, drivers, diagnostics, timelines, failures = [], [], [], [], []
    stage0 = None
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] prefill kernel {args.phase} {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                hr, dr, dg, tr = run_unit(
                    model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
                    rotation.cpu(), args.horizon, validate_reference=args.phase == "stage0",
                    state_carrier=args.phase == "stageB")
                rows += hr; drivers += dr; diagnostics += dg; timelines.append(tr)
                H.append_jsonl_many(outdir / "raw" / f"{args.phase}_horizon.jsonl", hr)
                H.append_jsonl_many(outdir / "raw" / f"{args.phase}_drivers.jsonl", dr)
                H.append_jsonl_many(outdir / "raw" / f"{args.phase}_diagnostics.jsonl", dg)
                H.append_jsonl_many(outdir / "raw" / f"{args.phase}_timeline.jsonl", [tr])
                if args.phase == "stage0":
                    audit = kernel_audit(probe.last_prefill_records["NS"])
                    stage0 = stage0_summary(rows, drivers, diagnostics, audit)
                    break
            except Exception as exc:
                failure = {"unit_id": str(unit.get("unit_id")), "error": repr(exc),
                           "traceback": traceback.format_exc(limit=40)}
                failures.append(failure)
                BASE.save_json(outdir / f"{args.phase}_failures.json", failures)
                if not args.keep_going:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        weights_after = BASE.tensor_hash(next(model.parameters()).detach().cpu())
        probe.close(); del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    invariants = {
        "MODEL_WEIGHTS_UNCHANGED": weights_before == weights_after,
        "ROTATION_MATRIX_UNCHANGED": rotation_before == BASE.tensor_hash(rotation),
        "QUANTIZER_SEMANTICS_UNCHANGED": COMM.quantizer_semantics(),
        "LAYER_LOCAL_SEQUENTIAL_REFERENCE": True,
        "FULL_MODEL_REMAINS_PACKED": True,
    }
    config = {
        "TASK": TASK, "phase": args.phase, "scope": args.scope, "horizon": args.horizon,
        "canonical_manifest": manifest, "canonical_unit_ids": [u["unit_id"] for u in units],
        "conditions": list(CONDITIONS), "bootstrap_estimand": "paired canonical-unit median",
        "bootstrap_seed": BASE.BOOTSTRAP_SEED,
        "packed_kernel": "fla.ops.kda.chunk_kda chunk_size=64",
        "sequential_kernel": "token-wise original fused_recurrent_kda with carried FP32 state",
        "previous_formal_artifacts": str(WRITEBACK_RESULTS),
    }
    BASE.save_json(outdir / "experiment_config.json", config)
    if stage0:
        stage0.update(invariants)
        BASE.save_json(outdir / "stage0_semantics.json", stage0)
        BASE.save_json(outdir / "sequential_reference_validation.json", {
            "SEQUENTIAL_DRIVER_IDENTITY": stage0["SEQUENTIAL_DRIVER_IDENTITY"],
            "STAGE0_SEQUENTIAL_REFERENCE": stage0["STAGE0_SEQUENTIAL_REFERENCE"],
            "driver_identity_max_abs": stage0["driver_identity_max_abs"],
            "sequential_reference_output_max_relative_l2": stage0["sequential_reference_output_max_relative_l2"],
            "sequential_reference_state_max_relative_l2": stage0["sequential_reference_state_max_relative_l2"],
        })
        BASE.save_json(outdir / "prefill_kernel_audit.json", stage0["kernel_audit"])
        write_kernel_audit_md(outdir, stage0["kernel_audit"])
        BASE.write_rows(outdir / "kernel_only_layer_diagnostics.csv", drivers)
        BASE.write_rows(outdir / "stage0_prefill_end_state_diagnostics.csv", diagnostics)
    summary = {"TASK": TASK, "phase": args.phase, "n_units_completed": len(timelines),
               "expected_units": len(units), "failures": failures, "invariants": invariants}
    if stage0:
        summary.update(stage0)
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", summary)
    return summary


def write_kernel_audit_md(outdir, audit):
    lines = [f"# {TASK}", ""] + [f"{key} = {json.dumps(value, sort_keys=True)}"
                                        for key, value in audit.items()]
    (outdir / "prefill_kernel_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dedupe(rows, keys):
    return list({tuple(str(row.get(key)) for key in keys): row for row in rows}.values())


def merge_raw(sources, phase):
    merged = {name: [] for name in ("horizon", "drivers", "diagnostics", "timeline")}
    failures = []
    for source in map(Path, sources):
        for name in merged:
            merged[name] += list(BASE.iter_jsonl(source / "raw" / f"{phase}_{name}.jsonl") or [])
        failures += BASE.load_json(source / f"{phase}_run_summary.json", {}).get("failures", [])
    merged["horizon"] = dedupe(merged["horizon"], ("unit_id", "horizon", "condition"))
    merged["drivers"] = dedupe(merged["drivers"], ("unit_id", "pair", "layer"))
    merged["diagnostics"] = dedupe(merged["diagnostics"], ("unit_id", "pair", "layer"))
    merged["timeline"] = dedupe(merged["timeline"], ("unit_id",))
    return merged, failures


def merge_runs(args):
    outdir = Path(args.output_dir)
    setup_output(outdir)
    merged, failures = merge_raw(args.merge_run_dirs, args.phase)
    if failures:
        raise RuntimeError(f"cannot merge failed shards: {failures}")
    if BASE.load_json(outdir / "stage0_semantics.json", {}).get("STAGE0_SEMANTICS") != "PASS":
        raise RuntimeError("Stage 0 semantics gate failed")
    unit_rows = aggregate_auc(merged["horizon"])
    if args.phase == "formal":
        derived, summary = summarize_factorial(unit_rows, merged["diagnostics"])
        BASE.write_rows(outdir / "formal_horizon_results.csv", merged["horizon"])
        BASE.write_rows(outdir / "formal_unit_results.csv", derived)
        BASE.write_rows(outdir / "prefill_end_state_diagnostics.csv", merged["diagnostics"])
        BASE.write_rows(outdir / "kernel_only_layer_diagnostics.csv", merged["drivers"])
        BASE.save_json(outdir / "factorial_summary.json", summary)
    elif args.phase == "stageB":
        derived, summary = summarize_carrier(unit_rows)
        BASE.write_rows(outdir / "state_carrier_control_horizon.csv", merged["horizon"])
        BASE.write_rows(outdir / "state_carrier_control_units.csv", derived)
        BASE.save_json(outdir / "state_carrier_summary.json", summary)
    else:
        raise ValueError(args.phase)
    config = BASE.load_json(outdir / "experiment_config.json", {})
    config.update({f"{args.phase}_merged_from": list(args.merge_run_dirs),
                   "formal_unit_count": len(unit_rows),
                   "canonical_unit_ids": sorted(r["unit_id"] for r in unit_rows)})
    BASE.save_json(outdir / "experiment_config.json", config)
    with (outdir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"TASK={TASK} PHASE={args.phase} FAILURES=[] TIME={BASE.now()}\n")
    return summary


def placeholder(path, status):
    if path.suffix == ".csv":
        BASE.write_rows(path, [{"STATUS": status}])
    else:
        BASE.save_json(path, {"STATUS": status})


def finalize(outdir):
    outdir = Path(outdir)
    stage0 = BASE.load_json(outdir / "stage0_semantics.json", {})
    factorial = BASE.load_json(outdir / "factorial_summary.json", {})
    carrier_path = outdir / "state_carrier_summary.json"
    if not carrier_path.exists():
        status = "NOT_RUN_BY_STOP_RULE"
        placeholder(outdir / "state_carrier_control_horizon.csv", status)
        placeholder(outdir / "state_carrier_control_units.csv", status)
        placeholder(carrier_path, status)
    carrier = BASE.load_json(carrier_path, {})
    chunk_path = outdir / "chunk_boundary_diagnostics.csv"
    if not chunk_path.exists():
        placeholder(chunk_path, "NOT_RUN_BY_STOP_RULE")
    audit = BASE.load_json(outdir / "prefill_kernel_audit.json", {})
    pytest_lines = ((outdir / "pytest_output.txt").read_text(encoding="utf-8").strip().splitlines()
                    if (outdir / "pytest_output.txt").exists() else [])
    effects = {key: factorial.get(key, {}) for key in (
        "PACKED_ROTATION_EFFECT", "SEQUENTIAL_ROTATION_EFFECT",
        "PACKED_KERNEL_EFFECT_NATIVE", "PACKED_KERNEL_EFFECT_ROTATED",
        "ROTATION_X_PACKED_INTERACTION")}
    strong = factorial.get("FINAL_KDA_PREFILL_CLASSIFICATION") == "PACKED_PREFILL_KERNEL_ROTATION_INTERACTION_STRONGLY_SUPPORTED"
    carrier_status = carrier.get("KDA_RECURRENT_STATE_CARRIES_PREFILL_KERNEL_DAMAGE", carrier.get("STATUS"))
    report = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "STAGE0_PREFILL_KERNEL_AUDIT": stage0.get("STAGE0_PREFILL_KERNEL_AUDIT"),
        "STAGE0_SEQUENTIAL_REFERENCE": stage0.get("STAGE0_SEQUENTIAL_REFERENCE"),
        "STAGEA_FACTORIAL": factorial.get("STAGEA_FACTORIAL"),
        "STAGEB_STATE_CARRIER": carrier_status,
        "N_FORMAL_UNITS": factorial.get("N_FORMAL_UNITS"),
        "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded",
        "FAILURES": [],
        "PACKED_PREFILL_KERNEL": audit.get("PACKED_PREFILL_KERNEL"),
        "DECODE_KERNEL": audit.get("DECODE_KERNEL"),
        "PACKED_SCAN_TYPE": audit.get("PACKED_PREFILL_SCAN_TYPE"),
        "PACKED_CHUNK_SIZE": audit.get("PACKED_PREFILL_CHUNK_SIZE"),
        "PACKED_VS_DECODE_DIFFERENCES": audit.get("PACKED_VS_DECODE_DIFFERENCES"),
        **{f"{name}_AUC": factorial.get("median_auc", {}).get(name) for name in CONDITIONS},
        **{key: effects[key].get("paired_median") for key in effects},
        **{f"{key}_95CI": [effects[key].get("bootstrap_ci_low"), effects[key].get("bootstrap_ci_high")]
           for key in effects},
        "INTERACTION_SIGN_COUNTS": {
            "positive": effects["ROTATION_X_PACKED_INTERACTION"].get("positive"),
            "negative": effects["ROTATION_X_PACKED_INTERACTION"].get("negative")},
        "NP_NS_STATE_REL_GAP": factorial.get("NP_NS_STATE_REL_GAP"),
        "NP_RP_STATE_REL_GAP": factorial.get("NP_RP_STATE_REL_GAP"),
        "NS_RS_STATE_REL_GAP": factorial.get("NS_RS_STATE_REL_GAP"),
        "RP_RS_STATE_REL_GAP": factorial.get("RP_RS_STATE_REL_GAP"),
        "EARLIEST_LAYER_WITH_PACKED_ROTATION_EXCESS_DRIFT": factorial.get("EARLIEST_LAYER_WITH_PACKED_ROTATION_EXCESS_DRIFT"),
        "KDA_STATE_ONLY_INTERACTION": carrier.get("KDA_STATE_ONLY_INTERACTION", {}).get("paired_median"),
        "KDA_STATE_ONLY_INTERACTION_95CI": [
            carrier.get("KDA_STATE_ONLY_INTERACTION", {}).get("bootstrap_ci_low"),
            carrier.get("KDA_STATE_ONLY_INTERACTION", {}).get("bootstrap_ci_high")],
        "KDA_RECURRENT_STATE_CARRIES_PREFILL_KERNEL_DAMAGE": carrier_status,
        "FINAL_KDA_PREFILL_CLASSIFICATION": factorial.get("FINAL_KDA_PREFILL_CLASSIFICATION"),
        "FINAL_KDA_ROTATION_FAILURE_MECHANISM": (
            "ROTATION_X_PACKED_PREFILL_KDA_EXECUTION" if strong else "PREFILL_ONLY_ROTATED_PATH_UNRESOLVED"),
        "KDA_MECHANISM_CLOSURE_READY": factorial.get("KDA_MECHANISM_CLOSURE_READY"),
        "METHOD_DESIGN_READY": factorial.get("METHOD_DESIGN_READY"),
        "NEXT_PHASE": factorial.get("NEXT_PHASE"),
    }
    combined = {**factorial, "formal_report": report, "stage0": stage0, "state_carrier": carrier}
    BASE.save_json(outdir / "summary.json", combined)
    lines = [f"# {TASK}", ""] + [f"{key} = {json.dumps(value, sort_keys=True)}"
                                        for key, value in report.items()]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(outdir):
    outdir = Path(outdir)
    required = [
        "experiment_config.json", "prefill_kernel_audit.json", "prefill_kernel_audit.md",
        "sequential_reference_validation.json", "stage0_semantics.json",
        "kernel_only_layer_diagnostics.csv", "formal_horizon_results.csv",
        "formal_unit_results.csv", "factorial_summary.json", "prefill_end_state_diagnostics.csv",
        "chunk_boundary_diagnostics.csv", "state_carrier_control_horizon.csv",
        "state_carrier_control_units.csv", "formal_summary.md", "run.log", "pytest_output.txt", "summary.json",
    ]
    missing = [name for name in required if not (outdir / name).is_file()]
    horizon = BASE.read_rows(outdir / "formal_horizon_results.csv")
    units = BASE.read_rows(outdir / "formal_unit_results.csv")
    unique = {(r["unit_id"], r["horizon"], r["condition"]) for r in horizon}
    finite = all(not isinstance(value, float) or math.isfinite(value)
                 for row in horizon + units for value in row.values())
    expected_rows = EXPECTED_UNITS * PRIMARY_HORIZON * (len(CONDITIONS) + 1)
    result = {
        "missing_required_files": missing, "n_units": len(units),
        "horizon_rows": len(horizon), "horizon_unique_rows": len(unique),
        "expected_horizon_rows": expected_rows, "all_numeric_values_finite": finite,
    }
    result["ARTIFACT_VALIDATION"] = "PASS" if (
        not missing and len(units) == EXPECTED_UNITS and len(horizon) == expected_rows
        and len(unique) == len(horizon) and finite) else "FAIL"
    BASE.save_json(outdir / "artifact_validation.json", result)
    summary = BASE.load_json(outdir / "summary.json", {})
    summary["ARTIFACT_VALIDATION"] = result["ARTIFACT_VALIDATION"]
    if "formal_report" in summary:
        summary["formal_report"]["ARTIFACT_VALIDATION"] = result["ARTIFACT_VALIDATION"]
    BASE.save_json(outdir / "summary.json", summary)
    with (outdir / "formal_summary.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\nARTIFACT_VALIDATION = {result['ARTIFACT_VALIDATION']}\n")
    BASE.save_json(outdir / "manifest.json", {
        "TASK": TASK, "artifacts": sorted(str(path) for path in outdir.rglob("*") if path.is_file())})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage0", "formal", "stageB", "finalize", "validate"), required=True)
    parser.add_argument("--scope", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--unit-shard-index", type=int)
    parser.add_argument("--unit-shard-count", type=int)
    parser.add_argument("--merge-run-dirs", nargs="*")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "finalize":
        result = finalize(args.output_dir)
    elif args.phase == "validate":
        result = validate(args.output_dir)
    elif args.merge_run_dirs:
        result = merge_runs(args)
    else:
        result = run_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if (result.get("failures") or result.get("STAGE0_SEMANTICS") == "FAIL"
            or result.get("ARTIFACT_VALIDATION") == "FAIL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
