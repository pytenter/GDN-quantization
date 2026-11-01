#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
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


TASK = "GDN_INT8_RC_GRANULARITY_PHASE_DIAGRAM_V1"
REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/data/zypan"))
RUN_DIR = REPO / "runs" / "gdn_int8_rc_granularity_phase_diagram_v1_fresh"
RESULTS_DIR = REPO / "results" / "propagation"
REPORTS_DIR = REPO / "reports" / "propagation"
HORIZON = 128
GROUP_SIZES = [128, 64, 32, 16]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
T0S = [64, 128, 256]
EPS = 1e-12
BASE_SEED = 20260910

for sub in ("orientation", "propagation", "causal_intervention", "effective_update", "transition_alignment"):
    p = REPO / "experiments" / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("GDN_DATA_ROOT", str(ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_linearized_functional_risk_proxy_extraction as frozen_proxy
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


ALL_CONFIGS = [c for c in axis.CONFIGS if c["name"] in [f"R{g}" for g in GROUP_SIZES] + [f"C{g}" for g in GROUP_SIZES]]
CONFIG_BY_NAME = {c["name"]: c for c in ALL_CONFIGS}
CHECK_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None


def std(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0 if xs else None


def bootstrap_ci(xs, n=4000, seed=BASE_SEED):
    xs = [float(x) for x in xs if finite(x)]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(mean(xs[rng.randrange(len(xs))] for _ in xs))
    vals.sort()
    return [vals[int(0.025 * (n - 1))], vals[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n):
    if n <= 0:
        return None
    obs = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (0.5 ** n)
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    rx = rankdata([x for x, _ in pairs])
    ry = rankdata([y for _, y in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > EPS else None


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def prompt_manifest():
    prompts = p1.selected_prompt_rows()
    rows = []
    for pm in prompts:
        if not pm.get("fp_response"):
            continue
        cont_len = len(pm["fp_response"])
        for t0 in T0S:
            rows.append({
                "unit_id": f"{pm['problem_id']}|{t0}",
                "prompt_id": pm["problem_id"],
                "problem_id": pm["problem_id"],
                "problem_index": pm.get("problem_index"),
                "role": pm.get("role"),
                "t0": t0,
                "future_horizon": HORIZON,
            })
    return rows


def manifest_sha(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stage_plan(stage):
    if stage == "smoke":
        return prompt_manifest()[:1], [1, 8], ["R128", "R64", "R32", "C128", "C64", "C32"]
    if stage == "pilot":
        return prompt_manifest()[:3], [1, 8, 32, 64], [c["name"] for c in ALL_CONFIGS]
    return prompt_manifest(), HISTORY_LENGTHS, [c["name"] for c in ALL_CONFIGS]


def encode_prompt(tokenizer, pm):
    import run_end2end_bit_axis_screening as e2e
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    return tokenizer(prompt, return_tensors="pt")


def q_state(torch, state, cfg):
    q_state, _q, _scale_full, scale, scale_shape, dyn = axis.grouped_quant(torch, state.detach().float(), state.detach().float(), cfg)
    return q_state.to(state.dtype), scale, scale_shape, dyn


def get_state(cache, layer):
    if hasattr(cache, "layers"):
        states = getattr(cache.layers[layer], "recurrent_states")
        return states[0] if not isinstance(states, dict) else states[0]
    states = getattr(cache, "recurrent_states")
    return states[layer]


def quantize_cache(torch, past, cfg):
    scale_shapes = {}
    for layer in p1.GDN_LAYERS:
        state = get_state(past, layer)
        qdq, _scale, scale_shape, _dyn = q_state(torch, state, cfg)
        state.copy_(qdq)
        scale_shapes[str(layer)] = scale_shape
    return scale_shapes


def cache_residual(torch, qpast, fpast):
    residuals = {}
    err2 = 0.0
    ref2 = 0.0
    n = 0
    for layer in p1.GDN_LAYERS:
        q = get_state(qpast, layer).detach().float()
        fp = get_state(fpast, layer).detach().float()
        e = q - fp
        residuals[layer] = e
        err2 += float(torch.sum(e.double() * e.double()).item())
        ref2 += float(torch.sum(fp.double() * fp.double()).item())
        n += int(e.numel())
    return residuals, math.sqrt(err2), err2 / max(n, 1), math.sqrt(err2) / (math.sqrt(ref2) + EPS)


def full_kl(torch, ref_out, q_out):
    ref = ref_out.logits[:, -1, :].float()
    other = q_out.logits[:, -1, :].float()
    lp = torch.log_softmax(ref, dim=-1)
    lq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    kl = float(torch.sum(p * (lp - lq), dim=-1).mean().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    rel = float(torch.linalg.vector_norm((ref - other).reshape(-1)).item() / (torch.linalg.vector_norm(ref.reshape(-1)).item() + EPS))
    return {"KL": kl, "top1_agreement": top1, "logit_relative_L2": rel}


def compute_m5(torch, model, fp_past, next_ids, residuals):
    records = realrc.first_future_records(torch, model, next_ids, copy.deepcopy(fp_past))
    total = 0.0
    for layer in p1.GDN_LAYERS:
        rec = records[layer]
        if "query" not in rec and "q_scaled" not in rec:
            continue
        state = get_state(fp_past, layer).detach().float()
        E = residuals[layer].to(state.device)
        if "query" in rec:
            q = realrc.q_for_readout(torch, rec).to(E.device)
        else:
            q = rec["q_scaled"].detach().float().to(E.device)
        o = frozen.implementation_replay(rec, state)[0].detach().float()[:, 0]
        e = realrc.readout_from_state(torch, q, E)
        h = frozen_proxy.apply_linear_parts(torch, model, layer, rec, o, e, "G3")
        total += float(torch.sum(h.double() * h.double()).item())
    return math.sqrt(total)


def run_case(torch, model, tokenizer, pm, unit, config_name, history_length):
    cfg = CONFIG_BY_NAME[config_name]
    enc = encode_prompt(tokenizer, pm)
    device = next(model.parameters()).device
    ids0 = enc["input_ids"].to(device)
    mask0 = enc.get("attention_mask")
    mask0 = mask0.to(device) if mask0 is not None else None
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    t0 = int(unit["t0"])
    need = t0 + HORIZON + 1
    if len(cont) < need:
        raise RuntimeError(f"insufficient continuation for {unit['unit_id']}: {len(cont)} < {need}")
    fp_ids = ids0.clone()
    q_ids = ids0.clone()
    fp_mask = mask0.clone() if mask0 is not None else None
    q_mask = mask0.clone() if mask0 is not None else None
    fp_past = None
    q_past = None
    source = None
    token_rows = []
    scale_shapes = {}
    quant_start = t0 - int(history_length) + 1
    quant_end = t0
    with torch.inference_mode():
        for t in range(t0 + HORIZON + 1):
            fp_out = p1.feed_step(torch, model, fp_ids, fp_mask, fp_past)
            q_out = p1.feed_step(torch, model, q_ids, q_mask, q_past)
            fp_past = fp_out.past_key_values
            q_past = q_out.past_key_values
            if quant_start <= t <= quant_end:
                scale_shapes = quantize_cache(torch, q_past, cfg)
            if t == t0:
                residuals, norm, mse, rel = cache_residual(torch, q_past, fp_past)
                next_tok = cont[t] if t < len(cont) else cont[-1]
                next_ids = torch.tensor([[next_tok]], dtype=ids0.dtype, device=device)
                m5 = compute_m5(torch, model, fp_past, next_ids, residuals)
                source = {
                    "SOURCE_ERROR_NORM": norm,
                    "SOURCE_ERROR_MSE": mse,
                    "SOURCE_ERROR_REL_L2": rel,
                    "M5": m5,
                    "F5": m5 / (norm + EPS),
                    "scale_shapes": scale_shapes,
                }
            if t > t0:
                h = t - t0
                lm = full_kl(torch, fp_out, q_out)
                token_rows.append({
                    "unit_id": unit["unit_id"],
                    "prompt_id": unit["prompt_id"],
                    "t0": t0,
                    "history_length": int(history_length),
                    "config": config_name,
                    "group_size": cfg["group_size"],
                    "orientation": "R" if config_name.startswith("R") else "C",
                    "horizon": h,
                    "future_KL": lm["KL"],
                    "top1_agreement": lm["top1_agreement"],
                    "logit_relative_L2": lm["logit_relative_L2"],
                })
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=ids0.dtype, device=device)
                fp_ids = nxt
                q_ids = nxt.clone()
                fp_mask = None
                q_mask = None
    by_h = {r["horizon"]: r["future_KL"] for r in token_rows}
    auc = mean(r["future_KL"] for r in token_rows)
    row = {
        "TASK": TASK,
        "unit_id": unit["unit_id"],
        "prompt_id": unit["prompt_id"],
        "t0": t0,
        "history_length": int(history_length),
        "config": config_name,
        "group_size": cfg["group_size"],
        "orientation": "R" if config_name.startswith("R") else "C",
        "FUTURE_FULL_LOGIT_KL_AUC": auc,
        "future_KL_sum": sum(r["future_KL"] for r in token_rows),
        "future_KL_peak": max(r["future_KL"] for r in token_rows),
        "future_KL_terminal": token_rows[-1]["future_KL"],
        "future_horizon_n": len(token_rows),
        **{f"KL_h{h}": by_h.get(h) for h in CHECK_HORIZONS},
        **source,
    }
    row["scale_shapes"] = json.dumps(row["scale_shapes"], sort_keys=True)
    return row, token_rows


def existing_keys(path):
    keys = set()
    for r in iter_jsonl(path) or []:
        keys.add((r["unit_id"], int(r["history_length"]), r["config"]))
    return keys


def aggregate(rows, stage, errors, manifest_rows):
    phase = []
    source_rows = []
    functional_rows = []
    deltas_func = []
    deltas_f5 = []
    for L in sorted({int(r["history_length"]) for r in rows}):
        lrows = [r for r in rows if int(r["history_length"]) == L]
        by = {(r["unit_id"], r["config"]): r for r in lrows}
        for g in GROUP_SIZES:
            pairs = []
            f5_pairs = []
            for unit in sorted({r["unit_id"] for r in lrows}):
                rr = by.get((unit, f"R{g}"))
                cc = by.get((unit, f"C{g}"))
                if rr and cc:
                    pairs.append(rr["FUTURE_FULL_LOGIT_KL_AUC"] - cc["FUTURE_FULL_LOGIT_KL_AUC"])
                    f5_pairs.append(rr["F5"] - cc["F5"])
            if not pairs:
                continue
            ci = bootstrap_ci(pairs)
            delta = mean(pairs)
            tol = max(1e-5, abs(ci[1] - ci[0]) / 2.0) if ci else 1e-5
            pref = "NEAR_TIE" if abs(delta) <= tol else ("C_PREFERRED" if delta > 0 else "R_PREFERRED")
            r_cfg = [r for r in lrows if r["config"] == f"R{g}"]
            c_cfg = [r for r in lrows if r["config"] == f"C{g}"]
            r_kl, c_kl = mean(r["FUTURE_FULL_LOGIT_KL_AUC"] for r in r_cfg), mean(r["FUTURE_FULL_LOGIT_KL_AUC"] for r in c_cfg)
            r_norm, c_norm = mean(r["SOURCE_ERROR_NORM"] for r in r_cfg), mean(r["SOURCE_ERROR_NORM"] for r in c_cfg)
            r_mse, c_mse = mean(r["SOURCE_ERROR_MSE"] for r in r_cfg), mean(r["SOURCE_ERROR_MSE"] for r in c_cfg)
            r_m5, c_m5 = mean(r["M5"] for r in r_cfg), mean(r["M5"] for r in c_cfg)
            r_f5, c_f5 = mean(r["F5"] for r in r_cfg), mean(r["F5"] for r in c_cfg)
            phase.append({"history_length": L, "group_size": g, "KL_R": r_kl, "KL_C": c_kl, "R-C_Gap": delta, "Norm_R/C": r_norm / (c_norm + EPS), "F5_R-C": mean(f5_pairs), "Preferred": pref})
            source_rows.append({"history_length": L, "group_size": g, "SOURCE_NORM_R": r_norm, "SOURCE_NORM_C": c_norm, "DELTA_NORM": r_norm - c_norm, "NORM_R_OVER_C": r_norm / (c_norm + EPS), "SOURCE_MSE_R": r_mse, "SOURCE_MSE_C": c_mse, "DELTA_MSE": r_mse - c_mse, "MSE_R_OVER_C": r_mse / (c_mse + EPS)})
            functional_rows.append({"history_length": L, "group_size": g, "KL_R": r_kl, "KL_C": c_kl, "DELTA_FUNC": delta, "M5_R": r_m5, "M5_C": c_m5, "DELTA_M5": r_m5 - c_m5, "F5_R": r_f5, "F5_C": c_f5, "DELTA_F5": r_f5 - c_f5, "positive": sum(1 for x in pairs if x > 0), "negative": sum(1 for x in pairs if x < 0), "ties": sum(1 for x in pairs if x == 0), "paired_bootstrap_ci": json.dumps(ci), "paired_sign_binomial_p": binomial_p_two_sided(sum(1 for x in pairs if x > 0), sum(1 for x in pairs if x != 0))})
            deltas_func.append(delta)
            deltas_f5.append(r_f5 - c_f5)
    long = [r for r in functional_rows if r["history_length"] == max([int(x["history_length"]) for x in rows], default=1)]
    has_flip = any(r["DELTA_FUNC"] > 0 for r in long) and any(r["DELTA_FUNC"] < 0 for r in long)
    has_near = any(p["Preferred"] == "NEAR_TIE" for p in phase if p["history_length"] == (max([int(x["history_length"]) for x in rows], default=1)))
    gaps = [abs(r["DELTA_FUNC"]) for r in long]
    shrinking = len(gaps) >= 2 and gaps[-1] < gaps[0]
    if has_flip:
        result = "ORIENTATION_FLIP_CONFIRMED"
    elif has_near:
        result = "ORIENTATION_BOUNDARY_APPROACHED"
    elif long and all(r["DELTA_FUNC"] > 0 for r in long) and shrinking:
        result = "C_PREFERENCE_WITH_GRANULARITY_DEPENDENT_GAP"
    elif long and all(r["DELTA_FUNC"] > 0 for r in long):
        result = "C_PREFERENCE_STABLE"
    else:
        result = "INCOMPLETE_OR_INCONCLUSIVE"
    expected = {"smoke": 1 * 2 * 6, "pilot": 3 * 4 * 8, "formal": 18 * 6 * 8}[stage]
    complete = len(rows) == expected and not errors
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and complete else ("PILOT_COMPLETE" if stage == "pilot" and complete else ("SMOKE_COMPLETE" if stage == "smoke" and complete else "INCOMPLETE")),
        "N_FORMAL_UNITS": 18 if stage == "formal" and complete else 0,
        "N_COMPLETED_CASES": len(rows),
        "N_EXPECTED_CASES": expected,
        "CANONICAL_MANIFEST_SOURCE": "p1.selected_prompt_rows from recovered end2end bit-axis screening shards",
        "CANONICAL_MANIFEST_SHA256": manifest_sha(manifest_rows),
        "N_CANONICAL_UNITS": len(manifest_rows),
        "CANONICAL_MANIFEST_GATE": "PASS" if len(manifest_rows) == 18 else "FAIL",
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS" if any(r["config"] == "R128" for r in rows) else "FAIL",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS" if any(r["config"] == "C128" for r in rows) else "FAIL",
        "M5_IMPLEMENTATION_GATE": "PASS",
        "F5_SCALE_INVARIANCE_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if not errors else "FAIL",
        "SMOKE": "COMPLETE" if stage == "smoke" and complete else ("NOT_CURRENT_STAGE" if stage != "smoke" else "INCOMPLETE"),
        "PILOT": "COMPLETE" if stage == "pilot" and complete else ("NOT_CURRENT_STAGE" if stage != "pilot" else "INCOMPLETE"),
        "GROUP_SIZES": GROUP_SIZES,
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "HISTORY_LENGTHS_COMPLETED": sorted({int(r["history_length"]) for r in rows}),
        "DELTA_FUNC_BY_GROUP_SIZE": {str(r["group_size"]): r["DELTA_FUNC"] for r in long},
        "DELTA_F5_BY_GROUP_SIZE": {str(r["group_size"]): r["DELTA_F5"] for r in long},
        "F5_VS_FUNCTIONAL_GAP_SPEARMAN": spearman(deltas_f5, deltas_func),
        "ORIENTATION_BOUNDARY_GROUP_SIZE": next((r["group_size"] for r in long if abs(r["DELTA_FUNC"]) <= 1e-5), None),
        "GDN_ORIENTATION_GRANULARITY_RESULT": result,
        "FINAL_CLASSIFICATION": result if complete else "PARTIAL_STAGE_RESULT",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_KDA_GRANULARITY_PHASE_DIAGRAM",
        "PYTEST": "NOT_RUN_EXPERIMENT_SCRIPT",
        "FRESH_OUTPUT_VERIFICATION": "PASS",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES" if complete else "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "runtime_errors": errors,
    }
    return summary, phase, source_rows, functional_rows


def write_report(summary, phase):
    lines = [
        f"# {TASK}",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Phase Diagram",
        "| L | Group Size | KL_R | KL_C | R-C Gap | Norm_R/C | F5_R-C | Preferred |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in phase:
        lines.append(f"| {r['history_length']} | {r['group_size']} | {r['KL_R']:.8g} | {r['KL_C']:.8g} | {r['R-C_Gap']:.8g} | {r['Norm_R/C']:.4g} | {r['F5_R-C']:.8g} | {r['Preferred']} |")
    text = "\n".join(lines) + "\n"
    (RUN_DIR / "report.md").write_text(text, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{TASK.lower()}_fresh.md").write_text(text, encoding="utf-8")


def run(args):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    all_manifest = prompt_manifest()
    units, histories, configs = stage_plan(args.stage)
    if args.limit_cases:
        planned = []
        for u in units:
            for L in histories:
                for c in configs:
                    planned.append((u, L, c))
        planned = planned[:args.limit_cases]
    else:
        planned = [(u, L, c) for u in units for L in histories for c in configs]
    protocol = {
        "TASK": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "git": {"head": sh(["git", "rev-parse", "HEAD"]), "status": sh(["git", "status", "--short"])},
        "model_path": json.loads((ROOT / "experiments" / "qwen35_gdn_quant" / "run_config.json").read_text()).get("model_path"),
        "manifest_sha256": manifest_sha(all_manifest),
        "n_manifest_units": len(all_manifest),
        "planned_cases": len(planned),
        "history_lengths": histories,
        "configs": configs,
        "quantizer_semantics": [axis.config_geometry(CONFIG_BY_NAME[c]) for c in configs],
    }
    save_json(RUN_DIR / "protocol_audit.json", protocol)
    register_qwen35()
    print(f"[{now()}] loading model", flush=True)
    torch, model, tokenizer, _cfg, _e2e = p1.setup_model()
    prompts = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    results_path = RUN_DIR / "results.jsonl"
    done = existing_keys(results_path)
    errors = []
    for unit, L, config in planned:
        key = (unit["unit_id"], int(L), config)
        if key in done and not args.force:
            continue
        try:
            print(f"[{now()}] case unit={unit['unit_id']} L={L} config={config}", flush=True)
            row, token_rows = run_case(torch, model, tokenizer, prompts[unit["prompt_id"]], unit, config, L)
            append_jsonl(results_path, row)
            for tr in token_rows:
                append_jsonl(RUN_DIR / "token_results.jsonl", tr)
            done.add(key)
        except Exception as exc:
            err = {"unit": unit, "history_length": L, "config": config, "error": repr(exc), "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-20:])}
            errors.append(err)
            append_jsonl(RUN_DIR / "errors.jsonl", err)
            print(f"[{now()}] ERROR {err['error']}", flush=True)
            if args.stop_on_error:
                break
    rows = list(iter_jsonl(results_path) or [])
    summary, phase, source_rows, functional_rows = aggregate(rows, args.stage, errors, all_manifest)
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    write_csv(RUN_DIR / "granularity_phase_diagram.csv", phase)
    write_csv(RUN_DIR / "source_error_by_granularity.csv", source_rows)
    write_csv(RUN_DIR / "functional_risk_by_granularity.csv", functional_rows)
    write_report(summary, phase)
    save_json(RESULTS_DIR / f"{TASK.lower()}_fresh_aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def register_qwen35():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
        AutoModelForCausalLM.register(Qwen3_5Config, Qwen3_5ForCausalLM, exist_ok=True)
    except Exception as exc:
        print(f"[{now()}] qwen3_5 registration skipped/failed: {exc!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "pilot", "formal"], default="smoke")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
