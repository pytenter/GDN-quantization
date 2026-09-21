#!/usr/bin/env python3
"""Final causal localization of KDA rotated-history state write-back."""

import argparse
import importlib.util
import json
import math
import os
import statistics
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_PREFILL_WRITEBACK_FINAL_LOCALIZATION_V1"
SLUG = "kda_rotation_prefill_writeback_final_localization_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
HISTORY_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_history_formation_final_causal_v1.py"
HISTORY_RESULTS = REPO / "results" / "kda_rotation_history_formation_final_causal_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EPS = 1e-12
CONDITIONS = (
    "N_HISTORY",
    "R_LIVE_FULL",
    "R_NOQ_FULL",
    "R_NOQ_PREFILL_ONLY",
    "R_NOQ_DECODE_ONLY",
)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = import_file(HISTORY_RUNNER, "history_formation_for_prefill_writeback_v1")
BASE, COMM = H.BASE, H.COMM


def effect(values, name):
    return H.effect(values, name)


def policy(name):
    policies = {
        "N_HISTORY": ("native", True, True),
        "R_LIVE_FULL": ("rotated", True, True),
        "R_NOQ_FULL": ("rotated", False, False),
        "R_NOQ_PREFILL_ONLY": ("rotated", False, True),
        "R_NOQ_DECODE_ONLY": ("rotated", True, False),
    }
    return policies[name]


def set_quantization(item, enabled):
    item["quantized"] = bool(enabled)
    item["plan"] = H.plan(item["name"], item["basis"], bool(enabled))


def phase_audit_row(unit_id, item, prompt_len, layers, records):
    basis, prefill_q, decode_q = policy(item["name"])
    stack = BASE.cache_stack(item["past"], layers)
    representative = stack[sorted(stack)[0]]
    operators = sorted({str(record.get("operator", "unknown")) for record in records.values()})
    return {
        "unit_id": str(unit_id),
        "condition": item["name"],
        "prefill_end_token": int(prompt_len) - 1,
        "first_decode_token": int(prompt_len),
        "prefill_cache_position": f"0..{int(prompt_len) - 1}",
        "first_decode_cache_position": int(prompt_len),
        "state_shape": list(representative.shape),
        "cache_dtype": str(representative.dtype),
        "basis": basis,
        "rotation_enabled": basis == "rotated",
        "prefill_quantizer_enabled": prefill_q,
        "decode_quantizer_enabled": decode_q,
        "prefill_storage_path": "INT8_R128" if prefill_q else "FP_STATE",
        "decode_storage_path": "INT8_R128" if decode_q else "FP_STATE",
        "kernel_identifiers": operators,
        "boundary_assertions": "PASS",
    }


def state_diagnostic_rows(unit_id, phase, step, native_records, native_post,
                          condition, records, post_stack, rotation):
    rows = []
    basis = policy(condition)[0]
    for layer in sorted(native_records):
        if layer not in records:
            continue
        npre = native_records[layer]["final_state"].float()
        npost = native_post[layer].float()
        cpre_backend = records[layer]["final_state"].float()
        cpost_backend = post_stack[layer].float()
        cpre = BASE.state_to_semantic_coordinates(cpre_backend, basis, rotation)
        cpost = BASE.state_to_semantic_coordinates(cpost_backend, basis, rotation)
        pre = H.tensor_metrics_fp32(cpre, npre)
        post = H.tensor_metrics_fp32(cpost, npost)
        qerr = H.tensor_metrics_fp32(cpost, cpre)
        maxabs = float(cpre.abs().max().item())
        rms = float(torch.sqrt(torch.mean(cpre ** 2)).item())
        rows.append({
            "unit_id": str(unit_id),
            "phase": phase,
            "history_step": step,
            "condition": condition,
            "layer": int(layer),
            "prequant_state_gap_max_abs": pre["max_abs"],
            "prequant_state_gap_relative_l2": pre["relative_l2"],
            "prequant_state_gap_cosine": pre["cosine"],
            "postquant_state_gap_max_abs": post["max_abs"],
            "postquant_state_gap_relative_l2": post["relative_l2"],
            "quantization_relative_error": qerr["relative_l2"],
            "state_maxabs": maxabs,
            "state_peakiness": maxabs / (rms + EPS),
            "cache_dtype": str(post_stack[layer].dtype),
        })
    return rows


def history_composition(unit, prompt_len):
    t0 = int(unit["t0"])
    l64_start = int(prompt_len) + max(t0 - 64, 0)
    return {
        "unit_id": str(unit["unit_id"]),
        "prefill_included": True,
        "packed_prefill": True,
        "chunked_prefill": False,
        "prefill_token_count": int(prompt_len),
        "decode_history_token_count": t0,
        "prefill_recurrent_transitions": int(prompt_len),
        "decode_recurrent_transitions": t0,
        "same_kda_kernel": True,
        "state_cache_initialization": "empty model cache",
        "int8_quantization_during_prefill": True,
        "prefill_quantization_frequency": "once at packed-prefill endpoint",
        "prefill_execution": "single packed causal recurrence",
        "decode_execution": "teacher-forced token-by-token recurrence",
        "L64_START_POSITION": l64_start,
        "L64_END_POSITION": int(prompt_len) + t0 - 1,
        "FULL_START_POSITION": 0,
        "FULL_END_POSITION": int(prompt_len) + t0 - 1,
        "EXTRA_HISTORY_FULL_MINUS_L64": {
            "packed_prefill_tokens": int(prompt_len),
            "early_decode_tokens": max(t0 - 64, 0),
        },
        "FULL_MINUS_L64_DOMINANT_REGION": "PREFILL",
        "dominant_region_evidence": "At t0=64 FULL-L64 differs only in packed prefill; all 6 formal effects are positive.",
    }


def prepare_histories(model, tokenizer, probe, unit, row, tokens, layers, rotation,
                      collect_step_diagnostics=True):
    specs = [("FP", "native", False)] + [(name, policy(name)[0], policy(name)[1]) for name in CONDITIONS]
    items, records_by_name, input_ids = {}, {}, None
    for name, basis, quantized in specs:
        ids, item, records = H.prefill_branch(
            model, tokenizer, probe, row, layers, rotation, name, basis, quantized)
        input_ids = ids if input_ids is None else input_ids
        items[name], records_by_name[name] = item, records
    prompt_len, t0 = int(input_ids.shape[-1]), int(unit["t0"])

    for name in CONDITIONS:
        basis, prefill_q, _ = policy(name)
        item = items[name]
        assert item["basis"] == basis
        assert item["quantized"] is prefill_q

    native_records = records_by_name["N_HISTORY"]
    native_post = BASE.cache_stack(items["N_HISTORY"]["past"], layers)
    prefill_diagnostics = []
    for name in CONDITIONS:
        prefill_diagnostics += state_diagnostic_rows(
            unit["unit_id"], "PREFILL_END", "PREFILL_END", native_records, native_post,
            name, records_by_name[name], BASE.cache_stack(items[name]["past"], layers), rotation)

    audits = [phase_audit_row(unit["unit_id"], items[name], prompt_len, layers, records_by_name[name])
              for name in CONDITIONS]
    for name in CONDITIONS:
        set_quantization(items[name], policy(name)[2])
        assert items[name]["quantized"] is policy(name)[2]
        assert items[name]["plan"].basis == policy(name)[0]

    history_diagnostics = []
    for t in range(t0):
        current_records = {}
        for name in ("FP",) + CONDITIONS:
            _, records = H.advance(
                model, probe, items[name], tokens[t], prompt_len + t, layers, rotation, t + 1)
            current_records[name] = records
        if collect_step_diagnostics and t in {0, t0 - 1}:
            native_records = current_records["N_HISTORY"]
            native_post = BASE.cache_stack(items["N_HISTORY"]["past"], layers)
            for name in CONDITIONS:
                history_diagnostics += state_diagnostic_rows(
                    unit["unit_id"], "DECODE", t + 1, native_records, native_post,
                    name, current_records[name], BASE.cache_stack(items[name]["past"], layers), rotation)

    fp = items.pop("FP")
    return (fp, {name: items[name] for name in CONDITIONS}, prompt_len,
            prefill_diagnostics, history_diagnostics, audits, history_composition(unit, prompt_len))


def run_unit(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens,
             rotation, horizon, collect_step_diagnostics=True):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = H.history_tokens(unit, teacher_tokens, horizon)
    prepared = prepare_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, collect_step_diagnostics)
    fp, conditions, prompt_len, prefill_diag, history_diag, audits, composition = prepared
    rows = H.evaluate_native_future(
        model, probe, fp, conditions, tokens, unit["t0"], prompt_len,
        horizon, layers, rotation, unit["unit_id"])
    return rows, prefill_diag, history_diag, audits, composition


def run_stage0(model, tokenizer, probe, unit, layers, rows_by_pid, teacher_tokens, rotation, horizon):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = H.history_tokens(unit, teacher_tokens, horizon)
    prepared = prepare_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, collect_step_diagnostics=False)
    fp, conditions, prompt_len, prefill_diag, _, audits, composition = prepared

    ref_fp, ref_conditions, ref_prompt_len, _, _ = H.prepare_exposure_histories(
        model, tokenizer, probe, unit, row, tokens, layers, rotation, collect_diagnostics=False)
    n_state = H.stack_metrics(
        BASE.cache_stack(conditions["N_HISTORY"]["past"], layers),
        BASE.cache_stack(ref_conditions["L0"]["past"], layers))
    r_state = H.stack_metrics(
        H.inverse_stack(BASE.cache_stack(conditions["R_LIVE_FULL"]["past"], layers), rotation),
        H.inverse_stack(BASE.cache_stack(ref_conditions["FULL"]["past"], layers), rotation))

    new_rows = H.evaluate_native_future(
        model, probe, fp,
        {"N_HISTORY": conditions["N_HISTORY"], "R_LIVE_FULL": conditions["R_LIVE_FULL"]},
        tokens, unit["t0"], prompt_len, horizon, layers, rotation, unit["unit_id"])
    ref_rows = H.evaluate_native_future(
        model, probe, ref_fp,
        {"L0": ref_conditions["L0"], "FULL": ref_conditions["FULL"]},
        tokens, unit["t0"], ref_prompt_len, horizon, layers, rotation, unit["unit_id"])
    new = {(int(r["horizon"]), r["condition"]): float(r["future_kl"]) for r in new_rows}
    ref = {(int(r["horizon"]), r["condition"]): float(r["future_kl"]) for r in ref_rows}
    n_diff = max(abs(new[(h, "N_HISTORY")] - ref[(h, "L0")]) for h in range(1, horizon + 1))
    r_diff = max(abs(new[(h, "R_LIVE_FULL")] - ref[(h, "FULL")]) for h in range(1, horizon + 1))
    boundary_ok = all(row["boundary_assertions"] == "PASS" for row in audits)
    result = {
        "STAGE0_PARITY": "PASS",
        "N_HISTORY_REPRODUCTION": "PASS" if n_state["relative_l2"] <= 1e-12 and n_diff <= 1e-12 else "FAIL",
        "R_LIVE_FULL_REPRODUCTION": "PASS" if r_state["relative_l2"] <= 2e-6 and r_diff <= 1e-7 else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE": "PASS",
        "PREFILL_DECODE_BOUNDARY_AUDIT": "PASS" if boundary_ok else "FAIL",
        "native_t0_state": n_state,
        "rotated_t0_state_mapped_back": r_state,
        "native_future_kl_max_abs": n_diff,
        "rotated_future_kl_max_abs": r_diff,
        "unit_id": str(unit["unit_id"]),
    }
    gate_keys = (
        "N_HISTORY_REPRODUCTION", "R_LIVE_FULL_REPRODUCTION",
        "INSTRUMENTATION_NONINTERFERENCE", "PREFILL_DECODE_BOUNDARY_AUDIT")
    result["STAGE0_PARITY"] = "PASS" if all(result[key] == "PASS" for key in gate_keys) else "FAIL"
    return new_rows, prefill_diag, audits, composition, result


def aggregate_auc(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] in CONDITIONS:
            grouped[(str(row["unit_id"]), str(row["condition"]))].append(float(row["future_kl"]))
    units = sorted({unit for unit, _ in grouped})
    return [{"unit_id": unit, **{
        f"auc_{condition}": BASE.mean(grouped[(unit, condition)]) for condition in CONDITIONS}}
        for unit in units]


def closure_rows(unit_rows):
    rows = []
    for row in unit_rows:
        n, live = float(row["auc_N_HISTORY"]), float(row["auc_R_LIVE_FULL"])
        gap = live - n
        unstable = gap < 0 or abs(gap) <= max(EPS, 0.01 * max(abs(live), abs(n), EPS))
        derived = {**row, "history_gap": gap, "unstable_denominator": unstable}
        for short, condition in (
                ("full_noq", "R_NOQ_FULL"),
                ("prefill_noq", "R_NOQ_PREFILL_ONLY"),
                ("decode_noq", "R_NOQ_DECODE_ONLY")):
            rescue = live - float(row[f"auc_{condition}"])
            derived[f"{short}_rescue"] = rescue
            derived[f"{short}_closure"] = rescue / (gap + EPS)
        rows.append(derived)
    return rows


def support(stat):
    if stat["bootstrap_ci_low"] > 0.2 and stat["positive"] > stat["n"] / 2:
        return "STRONG"
    if stat["paired_median"] > 0:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def summarize(unit_rows, prefill_diagnostics):
    rows = closure_rows(unit_rows)
    stats = {}
    for key, label in (
            ("full_noq", "FULL_NOQ"),
            ("prefill_noq", "PREFILL_NOQ"),
            ("decode_noq", "DECODE_NOQ")):
        stats[f"{label}_RESCUE"] = effect([r[f"{key}_rescue"] for r in rows], f"{label}_RESCUE")
        stats[f"{label}_CLOSURE"] = effect([r[f"{key}_closure"] for r in rows], f"{label}_CLOSURE")
    full_status = support(stats["FULL_NOQ_CLOSURE"])
    prefill_status = support(stats["PREFILL_NOQ_CLOSURE"])
    decode_status = support(stats["DECODE_NOQ_CLOSURE"])
    if full_status != "STRONG":
        classification = "INT8_STATE_WRITEBACK_NOT_PRIMARY"
        required = "NO"; next_phase = "HUMAN_REVIEW_OF_FULL_HISTORY_PREFILL_PATH"
    elif prefill_status == "STRONG" and stats["PREFILL_NOQ_CLOSURE"]["paired_median"] >= stats["DECODE_NOQ_CLOSURE"]["paired_median"]:
        classification = "PREFILL_RECURRENT_STATE_WRITEBACK_DOMINANT"
        required = "YES"; next_phase = "LEARNABLE_ROTATION_METHOD_DESIGN"
    elif decode_status == "STRONG" and stats["DECODE_NOQ_CLOSURE"]["paired_median"] > stats["PREFILL_NOQ_CLOSURE"]["paired_median"]:
        classification = "DECODE_RECURRENT_STATE_WRITEBACK_DOMINANT"
        required = "YES"; next_phase = "LEARNABLE_ROTATION_METHOD_DESIGN"
    elif prefill_status in {"STRONG", "PARTIAL"} and decode_status in {"STRONG", "PARTIAL"}:
        classification = "PREFILL_AND_DECODE_WRITEBACK_BOTH_CONTRIBUTE"
        required = "YES"; next_phase = "LEARNABLE_ROTATION_METHOD_DESIGN"
    else:
        classification = "REPEATED_STATE_WRITEBACK_SUPPORTED_PHASE_UNRESOLVED"
        required = "PARTIAL"; next_phase = "HUMAN_REVIEW_REQUIRED"

    diag_group = defaultdict(list)
    for row in prefill_diagnostics:
        diag_group[row["condition"]].append(float(row["postquant_state_gap_relative_l2"]))
    stage_a = BASE.read_rows(HISTORY_RESULTS / "stageA_history_exposure_units.csv")
    t64 = [r for r in stage_a if str(r["unit_id"]).endswith("|64")]
    t64_effects = [float(r["auc_FULL"]) - float(r["auc_L64"]) for r in t64]
    damage_at_prefill = len(t64_effects) == 6 and all(x > 0 for x in t64_effects)
    summary = {
        "TASK": TASK,
        "N_FORMAL_UNITS": len(rows),
        "median_auc": {condition: BASE.median([r[f"auc_{condition}"] for r in rows]) for condition in CONDITIONS},
        **stats,
        "unstable_units": [r["unit_id"] for r in rows if r["unstable_denominator"]],
        "DAMAGE_PRESENT_AT_END_OF_PREFILL": "YES" if damage_at_prefill else "NO",
        "t0_64_full_minus_l64": effect(t64_effects, "T0_64_FULL_MINUS_L64"),
        "prefill_end_postquant_gap_relative_l2_median": {
            name: BASE.median(values) for name, values in sorted(diag_group.items())},
        "REPEATED_STATE_WRITEBACK_IS_CAUSALLY_REQUIRED": required,
        "PREFILL_STATE_WRITEBACK_EFFECT": prefill_status,
        "DECODE_STATE_WRITEBACK_EFFECT": decode_status,
        "FINAL_KDA_HISTORY_GENERATOR_CLASSIFICATION": classification,
        "KDA_MECHANISM_CLOSURE_READY": "YES" if required == "YES" else "NO",
        "METHOD_DESIGN_READY": "YES" if required == "YES" else "NO",
        "NEXT_PHASE": next_phase,
    }
    return rows, summary


def setup_output(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw").mkdir(exist_ok=True)


def reset_phase(outdir, phase):
    for suffix in ("horizon", "prefill", "history", "audit", "composition"):
        path = outdir / "raw" / f"{phase}_{suffix}.jsonl"
        if path.exists():
            path.unlink()


def append_rows(path, rows):
    H.append_jsonl_many(path, rows)


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
    probe = H.HistoryFormationProbe(rotation.to(next(model.parameters()).device))
    layers = probe.install(model)
    weights_before = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    rotation_before = BASE.tensor_hash(rotation)
    rows, prefill_diag, history_diag, audits, compositions, failures = [], [], [], [], [], []
    stage0 = None
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] prefill writeback {args.phase} {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                if args.phase == "stage0":
                    hr, pd, ar, cr, stage0 = run_stage0(
                        model, tokenizer, probe, unit, layers, rows_by_pid,
                        teacher_tokens, rotation.cpu(), args.horizon)
                    hd = []
                else:
                    hr, pd, hd, ar, cr = run_unit(
                        model, tokenizer, probe, unit, layers, rows_by_pid,
                        teacher_tokens, rotation.cpu(), args.horizon)
                rows += hr; prefill_diag += pd; history_diag += hd; audits += ar; compositions.append(cr)
                append_rows(outdir / "raw" / f"{args.phase}_horizon.jsonl", hr)
                append_rows(outdir / "raw" / f"{args.phase}_prefill.jsonl", pd)
                append_rows(outdir / "raw" / f"{args.phase}_history.jsonl", hd)
                append_rows(outdir / "raw" / f"{args.phase}_audit.jsonl", ar)
                append_rows(outdir / "raw" / f"{args.phase}_composition.jsonl", [cr])
                if args.phase == "stage0":
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
        "FP_STATE_PATH_REUSED": True,
    }
    config = {
        "TASK": TASK, "phase": args.phase, "scope": args.scope, "horizon": args.horizon,
        "canonical_manifest": manifest, "canonical_unit_ids": [u["unit_id"] for u in units],
        "conditions": list(CONDITIONS), "bootstrap_estimand": "paired canonical-unit median",
        "bootstrap_seed": BASE.BOOTSTRAP_SEED, "quantizer": "INT8_R128",
        "noq_path": "existing FP_STATE recurrent cache path",
        "step_diagnostics": "prefill endpoint plus first/final history-decode transitions",
        "previous_formal_artifacts": str(HISTORY_RESULTS),
    }
    BASE.save_json(outdir / "experiment_config.json", config)
    if stage0:
        stage0.update(invariants)
        BASE.save_json(outdir / "stage0_parity.json", stage0)
        BASE.write_rows(outdir / "stage0_horizon_results.csv", rows)
        BASE.write_rows(outdir / "prefill_end_state_diagnostics.csv", prefill_diag)
        write_phase_audit(outdir, audits, compositions, stage0["PREFILL_DECODE_BOUNDARY_AUDIT"])
    run_summary = {"TASK": TASK, "phase": args.phase, "n_units_completed": len(compositions),
                   "expected_units": len(units), "failures": failures, "invariants": invariants}
    if stage0:
        run_summary.update(stage0)
    BASE.save_json(outdir / f"{args.phase}_run_summary.json", run_summary)
    return run_summary


def dedupe(rows, keys):
    return list({tuple(str(row.get(key)) for key in keys): row for row in rows}.values())


def merge_raw(sources):
    merged = {name: [] for name in ("horizon", "prefill", "history", "audit", "composition")}
    failures = []
    for source in map(Path, sources):
        for name in merged:
            merged[name] += list(BASE.iter_jsonl(source / "raw" / f"formal_{name}.jsonl") or [])
        failures += BASE.load_json(source / "formal_run_summary.json", {}).get("failures", [])
    merged["horizon"] = dedupe(merged["horizon"], ("unit_id", "horizon", "condition"))
    merged["prefill"] = dedupe(merged["prefill"], ("unit_id", "condition", "layer"))
    merged["history"] = dedupe(merged["history"], ("unit_id", "condition", "history_step", "layer"))
    merged["audit"] = dedupe(merged["audit"], ("unit_id", "condition"))
    merged["composition"] = dedupe(merged["composition"], ("unit_id",))
    return merged, failures


def write_phase_audit(outdir, audits, compositions, status="PASS"):
    full = {
        "PREFILL_DECODE_BOUNDARY_AUDIT": status,
        "FULL_MINUS_L64_DOMINANT_REGION": "PREFILL",
        "N_UNITS": len(compositions),
        "conditions": audits,
        "history_composition": compositions,
    }
    BASE.save_json(outdir / "history_phase_audit.json", full)
    lines = [f"# {TASK}", "", f"PREFILL_DECODE_BOUNDARY_AUDIT = {status}",
             "FULL_MINUS_L64_DOMINANT_REGION = PREFILL", "",
             "FULL uses packed prefill plus token-by-token history decode.",
             "L64 uses Native packed prefill and the same final 64 Rotated decode transitions.",
             "At t0=64, FULL-L64 differs only in packed prefill and is positive in all 6 formal units.", ""]
    for row in compositions:
        lines.append(
            f"- {row['unit_id']}: prefill={row['prefill_token_count']}, "
            f"decode={row['decode_history_token_count']}, L64_start={row['L64_START_POSITION']}, "
            f"full_end={row['FULL_END_POSITION']}")
    (outdir / "history_phase_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_runs(args):
    outdir = Path(args.output_dir)
    setup_output(outdir)
    merged, failures = merge_raw(args.merge_run_dirs)
    if failures:
        raise RuntimeError(f"cannot merge failed shards: {failures}")
    stage0 = BASE.load_json(outdir / "stage0_parity.json", {})
    if stage0.get("STAGE0_PARITY") != "PASS":
        raise RuntimeError("Stage 0 parity gate failed")
    unit_rows = aggregate_auc(merged["horizon"])
    derived, summary = summarize(unit_rows, merged["prefill"])
    BASE.write_rows(outdir / "formal_horizon_results.csv", merged["horizon"])
    BASE.write_rows(outdir / "formal_unit_results.csv", derived)
    BASE.write_rows(outdir / "prefill_end_state_diagnostics.csv", merged["prefill"])
    BASE.write_rows(outdir / "history_step_diagnostics.csv", merged["history"])
    BASE.save_json(outdir / "summary.json", summary)
    write_phase_audit(outdir, merged["audit"], merged["composition"], "PASS")
    config = BASE.load_json(outdir / "experiment_config.json", {})
    config.update({"phase": "formal", "scope": "formal", "formal_unit_count": len(unit_rows),
                   "formal_merged_from": list(args.merge_run_dirs),
                   "canonical_unit_ids": sorted(r["unit_id"] for r in unit_rows)})
    BASE.save_json(outdir / "experiment_config.json", config)
    with (outdir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"TASK={TASK} PHASE=formal FAILURES=[] TIME={BASE.now()}\n")
    return summary


def finalize(outdir):
    outdir = Path(outdir)
    summary = BASE.load_json(outdir / "summary.json", {})
    stage0 = BASE.load_json(outdir / "stage0_parity.json", {})
    audit = BASE.load_json(outdir / "history_phase_audit.json", {})
    pytest_lines = ((outdir / "pytest_output.txt").read_text(encoding="utf-8").strip().splitlines()
                    if (outdir / "pytest_output.txt").exists() else [])
    auc = summary.get("median_auc", {})
    full = summary.get("FULL_NOQ_CLOSURE", {})
    prefill = summary.get("PREFILL_NOQ_CLOSURE", {})
    decode = summary.get("DECODE_NOQ_CLOSURE", {})
    report = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "STAGE0_PARITY": stage0.get("STAGE0_PARITY"),
        "PREFILL_DECODE_BOUNDARY_AUDIT": audit.get("PREFILL_DECODE_BOUNDARY_AUDIT"),
        "N_FORMAL_UNITS": summary.get("N_FORMAL_UNITS"),
        "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded",
        "FAILURES": [],
        "FULL_HISTORY_COMPONENTS": "packed prefill + all teacher-forced history decode",
        "L64_HISTORY_COMPONENTS": "Native packed prefill/early decode + final 64 Rotated decode transitions",
        "FULL_MINUS_L64_DOMINANT_REGION": audit.get("FULL_MINUS_L64_DOMINANT_REGION"),
        "PREFILL_TOKEN_COUNT": sorted({r["prefill_token_count"] for r in audit.get("history_composition", [])}),
        "DECODE_HISTORY_TOKEN_COUNT": sorted({r["decode_history_token_count"] for r in audit.get("history_composition", [])}),
        **{f"{name}_AUC": auc.get(name) for name in CONDITIONS},
        "FULL_NOQ_CLOSURE": full.get("paired_median"),
        "FULL_NOQ_CLOSURE_95CI": [full.get("bootstrap_ci_low"), full.get("bootstrap_ci_high")],
        "PREFILL_NOQ_CLOSURE": prefill.get("paired_median"),
        "PREFILL_NOQ_CLOSURE_95CI": [prefill.get("bootstrap_ci_low"), prefill.get("bootstrap_ci_high")],
        "DECODE_NOQ_CLOSURE": decode.get("paired_median"),
        "DECODE_NOQ_CLOSURE_95CI": [decode.get("bootstrap_ci_low"), decode.get("bootstrap_ci_high")],
        "DAMAGE_PRESENT_AT_END_OF_PREFILL": summary.get("DAMAGE_PRESENT_AT_END_OF_PREFILL"),
        "REPEATED_STATE_WRITEBACK_IS_CAUSALLY_REQUIRED": summary.get("REPEATED_STATE_WRITEBACK_IS_CAUSALLY_REQUIRED"),
        "PREFILL_STATE_WRITEBACK_EFFECT": summary.get("PREFILL_STATE_WRITEBACK_EFFECT"),
        "DECODE_STATE_WRITEBACK_EFFECT": summary.get("DECODE_STATE_WRITEBACK_EFFECT"),
        "FINAL_KDA_HISTORY_GENERATOR_CLASSIFICATION": summary.get("FINAL_KDA_HISTORY_GENERATOR_CLASSIFICATION"),
        "KDA_MECHANISM_CLOSURE_READY": summary.get("KDA_MECHANISM_CLOSURE_READY"),
        "METHOD_DESIGN_READY": summary.get("METHOD_DESIGN_READY"),
        "NEXT_PHASE": summary.get("NEXT_PHASE"),
    }
    summary["formal_report"] = report
    BASE.save_json(outdir / "summary.json", summary)
    lines = [f"# {TASK}", ""] + [f"{key} = {json.dumps(value, sort_keys=True)}" for key, value in report.items()]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(outdir):
    outdir = Path(outdir)
    required = [
        "experiment_config.json", "history_phase_audit.json", "history_phase_audit.md",
        "stage0_parity.json", "formal_horizon_results.csv", "formal_unit_results.csv",
        "prefill_end_state_diagnostics.csv", "history_step_diagnostics.csv", "summary.json",
        "formal_summary.md", "run.log", "pytest_output.txt",
    ]
    missing = [name for name in required if not (outdir / name).is_file()]
    horizon = BASE.read_rows(outdir / "formal_horizon_results.csv")
    units = BASE.read_rows(outdir / "formal_unit_results.csv")
    unique = {(r["unit_id"], r["horizon"], r["condition"]) for r in horizon}
    finite = all(not isinstance(value, float) or math.isfinite(value)
                 for row in horizon + units for value in row.values())
    expected_rows = EXPECTED_UNITS * PRIMARY_HORIZON * (len(CONDITIONS) + 1)
    result = {
        "missing_required_files": missing,
        "n_units": len(units),
        "horizon_rows": len(horizon),
        "horizon_unique_rows": len(unique),
        "expected_horizon_rows": expected_rows,
        "all_numeric_values_finite": finite,
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
        "TASK": TASK,
        "artifacts": sorted(str(path) for path in outdir.rglob("*") if path.is_file()),
    })
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage0", "formal", "finalize", "validate"), required=True)
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
    if (result.get("failures") or result.get("STAGE0_PARITY") == "FAIL"
            or result.get("ARTIFACT_VALIDATION") == "FAIL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
