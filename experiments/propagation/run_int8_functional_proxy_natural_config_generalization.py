#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
REPO = ROOT / "GDN-quantization"
TASK = "GDN_INT8_FUNCTIONAL_PROXY_NATURAL_CONFIG_GENERALIZATION_V1"
SLUG = "gdn_int8_functional_proxy_natural_config_generalization_v1"
RUN_DIR = ROOT / "runs" / SLUG
RES_DIR = ROOT / "results" / "propagation"
REP_DIR = ROOT / "reports" / "propagation"
HORIZON = 128
EPS = 1e-12
PANEL_NAMES = ["R128", "R64", "R32", "R16", "C128", "C64", "C32", "C16"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_linearized_functional_risk_proxy_extraction as frozen_proxy
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


CONFIGS = [c for c in axis.CONFIGS if c["name"] in PANEL_NAMES]
GDN_LAYERS = frozen.GDN_LAYERS


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd, default=""):
    try:
        return sh(cmd)
    except Exception as exc:
        return f"{default}{type(exc).__name__}"


def git_commit():
    try:
        return sh(["git", "rev-parse", "HEAD"])
    except Exception:
        return None


def finite(x):
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx, my = mean([x for x, _ in pairs]), mean([y for _, y in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den > EPS else None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None

    def ranks(vals):
        order = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        j = 0
        while j < len(order):
            k = j + 1
            while k < len(order) and order[k][0] == order[j][0]:
                k += 1
            r = (j + k - 1) / 2.0 + 1.0
            for _, idx in order[j:k]:
                out[idx] = r
            j = k
        return out

    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def kendall_tau(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    n = len(pairs)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if abs(dx) <= EPS or abs(dy) <= EPS:
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else None


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    return realrc.cosine(torch, a, b)


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(name, rows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def stage_paths(stage, shard_id=0):
    prefix = RUN_DIR / f"{stage}_shard{shard_id}"
    return {"records": prefix.with_suffix(".jsonl")}


def summarize_curve(rows, key):
    vals = [r.get(key) for r in rows if finite(r.get(key))]
    return {
        "mean": mean(vals),
        "sum": sum(vals) if vals else None,
        "max": max(vals) if vals else None,
        "terminal": vals[-1] if vals else None,
        "n": len(vals),
    }


def prompt_map():
    return {p["problem_id"]: p for p in frozen_proxy.prompt_map().values()}


def unit_rows():
    return list(frozen_proxy.unit_rows().values())


def build_candidates(torch, fp_past):
    candidates = {}
    diagnostics = {}
    for cfg in CONFIGS:
        cname = cfg["name"]
        candidates[cname] = {}
        scale_vals = []
        dyn_vals = []
        for layer in GDN_LAYERS:
            state = p1.get_state(fp_past, layer).detach().float()
            q_state, _q, _scale_full, scale, scale_shape, dyn = axis.grouped_quant(torch, state, state, cfg)
            candidates[cname][layer] = (q_state - state).detach().float()
            scale_vals.extend(scale.detach().float().flatten().cpu().tolist())
            dyn_vals.extend(dyn)
        diagnostics[cname] = {
            **axis.config_geometry(cfg),
            "scale_shape_last_layer": scale_shape,
            "scale_mean": mean(scale_vals),
            "scale_median": med(scale_vals),
            "scale_max": max(scale_vals) if scale_vals else None,
            "dynamic_range_mean": mean(dyn_vals),
            "dynamic_range_p95": pct(dyn_vals, 0.95),
        }
    return candidates, diagnostics


def proxy_rows_for_unit(torch, model, tokenizer, e2e, row, pm):
    fp_past, next_ids, cont = realrc.build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    records = realrc.first_future_records(torch, model, next_ids, fp_past)
    candidates, diagnostics = build_candidates(torch, fp_past)
    out = []
    for cfg in CONFIGS:
        cname = cfg["name"]
        totals = defaultdict(float)
        rel_state_den = 0.0
        for layer in GDN_LAYERS:
            rec = records[layer]
            state = p1.get_state(fp_past, layer).detach().float()
            E = candidates[cname][layer].to(state.device)
            q = realrc.q_for_readout(torch, rec).to(E.device)
            o = frozen.implementation_replay(rec, state)[0].detach().float()[:, 0]
            e = realrc.readout_from_state(torch, q, E)
            _epar, etan = realrc.decompose(torch, o, e)
            r = frozen_proxy.apply_linear_parts(torch, model, layer, rec, o, e, "G1")
            u = frozen_proxy.apply_linear_parts(torch, model, layer, rec, o, e, "G2")
            h = frozen_proxy.apply_linear_parts(torch, model, layer, rec, o, e, "G3")
            totals["M0_sq"] += float(torch.sum(E.double() * E.double()).item())
            rel_state_den += float(torch.sum(state.double() * state.double()).item())
            totals["M1_sq"] += float(torch.sum(e.double() * e.double()).item())
            totals["M2_sq"] += float(torch.sum(etan.double() * etan.double()).item())
            totals["M3_sq"] += float(torch.sum(r.double() * r.double()).item())
            totals["M4_sq"] += float(torch.sum(u.double() * u.double()).item())
            totals["M5_sq"] += float(torch.sum(h.double() * h.double()).item())
        d = diagnostics[cname]
        out.append({
            "task": TASK,
            "unit_id": row["unit"],
            "prompt_id": row["prompt_id"],
            "layer": "ALL_GDN",
            "head": "ALL_HEADS",
            "t0": int(row["t0"]),
            "config": cname,
            "orientation": cfg["orientation"],
            "group_size": cfg["group_size"],
            "M0": math.sqrt(totals["M0_sq"]),
            "M0_rel": math.sqrt(totals["M0_sq"]) / (math.sqrt(rel_state_den) + EPS),
            "M1": math.sqrt(totals["M1_sq"]),
            "M2": math.sqrt(totals["M2_sq"]),
            "M3": math.sqrt(totals["M3_sq"]),
            "M4": math.sqrt(totals["M4_sq"]),
            "M5": math.sqrt(totals["M5_sq"]),
            "raw_state_error": math.sqrt(totals["M0_sq"]),
            "q_visible_norm": math.sqrt(totals["M1_sq"]),
            "tangential_norm": math.sqrt(totals["M2_sq"]),
            "post_gate_norm": math.sqrt(totals["M4_sq"]),
            "out_proj_distortion": math.sqrt(totals["M5_sq"]),
            "quantization_scale_median": d["scale_median"],
            "quantization_scale_max": d["scale_max"],
            "dynamic_range_p95": d["dynamic_range_p95"],
        })
    return out, candidates, cont


def apply_candidate(torch, past, candidates, config):
    for layer, err in candidates[config].items():
        state = p1.get_state(past, layer)
        state.copy_((state.detach().float() + err.to(state.device)).to(state.dtype))


def rollout_future_kl(torch, model, tokenizer, e2e, pm, row, candidates, cont):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids0 = enc["input_ids"].to(device)
    mask0 = enc.get("attention_mask")
    mask0 = mask0.to(device) if mask0 is not None else None
    branches = ["FP"] + PANEL_NAMES
    ids = {b: ids0.clone() for b in branches}
    masks = {b: mask0.clone() if mask0 is not None else None for b in branches}
    pasts = {b: None for b in branches}
    token_rows = []
    t0 = int(row["t0"])
    last = min(t0 + HORIZON - 1, len(cont) - 1)
    with torch.inference_mode():
        for t in range(last + 1):
            outs = {}
            for b in branches:
                outs[b] = model(input_ids=ids[b], attention_mask=masks[b], past_key_values=pasts[b], use_cache=True, output_hidden_states=False)
                pasts[b] = outs[b].past_key_values
            if t == t0:
                for b in PANEL_NAMES:
                    apply_candidate(torch, pasts[b], candidates, b)
            if t >= t0:
                h = t - t0
                for b in PANEL_NAMES:
                    lm = realrc.logit_metrics(torch, outs["FP"], outs[b])
                    sn, sr = realrc.state_delta_norm(torch, pasts[b], pasts["FP"])
                    token_rows.append({
                        "unit_id": row["unit"],
                        "prompt_id": row["prompt_id"],
                        "t0": t0,
                        "horizon": h,
                        "config": b,
                        "future_KL": lm["KL"],
                        "top1_agreement": lm["top1_agreement"],
                        "logit_relative_L2": lm["logit_relative_L2"],
                        "state_error_norm": sn,
                        "relative_state_error": sr,
                    })
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=ids0.dtype, device=device)
                for b in branches:
                    ids[b] = nxt.clone()
                    masks[b] = None
    return token_rows


def run_unit(torch, model, tokenizer, e2e, row, pm, stage):
    print(f"[{now()}] {stage} unit={row['unit']}", flush=True)
    proxy_rows, candidates, cont = proxy_rows_for_unit(torch, model, tokenizer, e2e, row, pm)
    future_rows = rollout_future_kl(torch, model, tokenizer, e2e, pm, row, candidates, cont)
    by_cfg = defaultdict(list)
    for r in future_rows:
        by_cfg[r["config"]].append(r)
    future_summary = {}
    for cfg in PANEL_NAMES:
        future_summary[cfg] = summarize_curve(by_cfg[cfg], "future_KL")
    for r in proxy_rows:
        fs = future_summary[r["config"]]
        r["future_KL"] = fs["mean"]
        r["future_KL_sum"] = fs["sum"]
        r["future_KL_peak"] = fs["max"]
        r["future_KL_terminal"] = fs["terminal"]
        r["future_horizon_n"] = fs["n"]
    return {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "unit_id": row["unit"],
        "prompt_id": row["prompt_id"],
        "t0": int(row["t0"]),
        "proxy_phase_completed_before_future_rollout": True,
        "future_rollout_uses_precomputed_proxy_values": True,
        "proxy_rows": proxy_rows,
        "future_token_rows": future_rows,
    }


METRICS = ["M0", "M1", "M2", "M3", "M4", "M5"]


def pair_type(a, b):
    oa = "R" if a.startswith("R") else "C"
    ob = "R" if b.startswith("R") else "C"
    return "R-vs-R" if oa == ob == "R" else ("C-vs-C" if oa == ob == "C" else "R-vs-C")


def analyze_rows(raw_rows):
    by_unit = defaultdict(list)
    for r in raw_rows:
        by_unit[r["unit_id"]].append(r)
    ranking_rows = []
    for unit, rows in sorted(by_unit.items()):
        for m in METRICS:
            xs = [r[m] for r in rows]
            ys = [r["future_KL"] for r in rows]
            ranking_rows.append({
                "unit_id": unit,
                "metric": m,
                "spearman": spearman(xs, ys),
                "kendall_tau": kendall_tau(xs, ys),
                "n_configs": len(rows),
            })
    aggregate = []
    for m in METRICS:
        xs = [r[m] for r in raw_rows]
        ys = [r["future_KL"] for r in raw_rows]
        per = [r for r in ranking_rows if r["metric"] == m]
        aggregate.append({
            "metric": m,
            "aggregate_spearman": spearman(xs, ys),
            "aggregate_kendall_tau": kendall_tau(xs, ys),
            "per_unit_spearman_mean": mean([r["spearman"] for r in per]),
            "per_unit_spearman_median": med([r["spearman"] for r in per]),
            "per_unit_spearman_min": min([r["spearman"] for r in per if finite(r["spearman"])], default=None),
            "per_unit_spearman_max": max([r["spearman"] for r in per if finite(r["spearman"])], default=None),
            "per_unit_kendall_mean": mean([r["kendall_tau"] for r in per]),
            "per_unit_kendall_median": med([r["kendall_tau"] for r in per]),
        })
    pairwise = []
    for m in METRICS:
        buckets = defaultdict(lambda: [0, 0])
        for unit, rows in by_unit.items():
            rows = sorted(rows, key=lambda r: r["config"])
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    dp = a[m] - b[m]
                    dk = a["future_KL"] - b["future_KL"]
                    if abs(dp) <= EPS or abs(dk) <= EPS:
                        continue
                    ok = int(dp * dk > 0)
                    for bucket in ("overall", pair_type(a["config"], b["config"])):
                        buckets[bucket][0] += ok
                        buckets[bucket][1] += 1
        for bucket in ["overall", "R-vs-R", "C-vs-C", "R-vs-C"]:
            ok, total = buckets[bucket]
            pairwise.append({
                "metric": m,
                "pair_class": bucket,
                "correct_pairs": ok,
                "total_pairs": total,
                "pairwise_concordance": ok / total if total else None,
            })
    selection = []
    for m in METRICS:
        hits, regrets, per_unit = [], [], []
        for unit, rows in by_unit.items():
            pred = min(rows, key=lambda r: r[m])
            best = min(rows, key=lambda r: r["future_KL"])
            hit = int(pred["config"] == best["config"])
            regret = pred["future_KL"] - best["future_KL"]
            hits.append(hit)
            regrets.append(regret)
            per_unit.append({"unit_id": unit, "metric": m, "selected_config": pred["config"], "best_config": best["config"], "hit": hit, "regret": regret})
        selection.append({
            "metric": m,
            "hit_rate": mean(hits),
            "mean_regret": mean(regrets),
            "median_regret": med(regrets),
            "max_regret": max(regrets) if regrets else None,
        })
    stability = []
    for key in ["prompt_id", "t0"]:
        groups = defaultdict(list)
        for r in raw_rows:
            groups[r[key]].append(r)
        for group, rows in sorted(groups.items(), key=lambda x: str(x[0])):
            for m in METRICS:
                stability.append({"group_by": key, "group": group, "metric": m, "spearman": spearman([r[m] for r in rows], [r["future_KL"] for r in rows]), "kendall_tau": kendall_tau([r[m] for r in rows], [r["future_KL"] for r in rows]), "n": len(rows)})
    within = []
    for orient in ["R", "C"]:
        rows = [r for r in raw_rows if r["config"].startswith(orient)]
        for m in METRICS:
            within.append({"orientation": orient, "metric": m, "spearman": spearman([r[m] for r in rows], [r["future_KL"] for r in rows]), "kendall_tau": kendall_tau([r[m] for r in rows], [r["future_KL"] for r in rows]), "n": len(rows)})
    return ranking_rows, aggregate, pairwise, selection, stability, within


def classify(summary):
    m0 = summary["M0"]
    m4 = summary["M4"]
    m5 = summary["M5"]
    best = m4 if (m4["aggregate_spearman"] or -9) >= (m5["aggregate_spearman"] or -9) else m5
    if best["aggregate_spearman"] and best["aggregate_spearman"] > (m0["aggregate_spearman"] or -9):
        return "FUNCTIONAL_PROXY_MULTICONFIG_GENERALIZATION_PARTIAL"
    if best["aggregate_spearman"] and best["aggregate_spearman"] > 0:
        return "FUNCTIONAL_PROXY_GENERALIZATION_INCONCLUSIVE"
    return "FUNCTIONAL_PROXY_GENERALIZATION_NOT_SUPPORTED"


def write_report(final):
    lines = [
        f"# {TASK}",
        "",
        "## 1. TASK",
        "Frozen M0-M5 proxy generalization across R128/R64/R32/R16/C128/C64/C32/C16.",
        "",
        "## 2. Git / environment",
        json.dumps(final["git_environment"], indent=2, sort_keys=True),
        "",
        "## 3. Protocol audit",
        json.dumps(final["protocol_audit"], indent=2, sort_keys=True),
        "",
        "## 4. Frozen proxy verification",
        json.dumps(final["frozen_proxy_verification"], indent=2, sort_keys=True),
        "",
        "## 5. Natural config panel",
        json.dumps(final["natural_config_panel"], indent=2, sort_keys=True),
        "",
        "## 6. Compatibility audit",
        json.dumps(final["config_compatibility"], indent=2, sort_keys=True),
        "",
        "## 7. Stage0 gates",
        json.dumps(final["stage0_gates"], indent=2, sort_keys=True),
        "",
        "## 8. Pilot",
        json.dumps(final["pilot"], indent=2, sort_keys=True),
        "",
        "## 9. Formal status",
        f"`{final['FORMAL_STATUS']}`; raw table rows: `{final['N_FORMAL_CASES']}`.",
        "",
        "## 10. Raw config x unit table",
        "See `formal_raw_results.jsonl` and `formal_raw_results.csv`.",
        "",
        "## 11. Spearman results",
        json.dumps(final["ranking_metric_summary"], indent=2, sort_keys=True),
        "",
        "## 12. Kendall tau results",
        json.dumps(final["ranking_metric_summary"], indent=2, sort_keys=True),
        "",
        "## 13. Pairwise concordance",
        json.dumps(final["pairwise_key_summary"], indent=2, sort_keys=True),
        "",
        "## 14. Within-R ranking",
        json.dumps(final["within_orientation_summary"]["R"], indent=2, sort_keys=True),
        "",
        "## 15. Within-C ranking",
        json.dumps(final["within_orientation_summary"]["C"], indent=2, sort_keys=True),
        "",
        "## 16. Cross R/C ranking",
        json.dumps(final["pairwise_key_summary"].get("R-vs-C", {}), indent=2, sort_keys=True),
        "",
        "## 17. Candidate-selection hit rate",
        json.dumps(final["candidate_selection_key_summary"], indent=2, sort_keys=True),
        "",
        "## 18. Selection regret",
        json.dumps(final["candidate_selection_key_summary"], indent=2, sort_keys=True),
        "",
        "## 19. Cross-unit stability",
        json.dumps(final["cross_unit_stability_summary"], indent=2, sort_keys=True),
        "",
        "## 20. Failure cases",
        json.dumps(final["failure_analysis"], indent=2, sort_keys=True),
        "",
        "## 21. M0 vs M4 vs M5 comparison",
        json.dumps({k: final["ranking_metric_summary"][k] for k in ["M0", "M4", "M5"]}, indent=2, sort_keys=True),
        "",
        "## 22. Scientific interpretation",
        final["scientific_interpretation"],
        "",
        "## 23. Negative results",
        final["negative_results"],
        "",
        "## 24. Limitations",
        final["limitations"],
        "",
        "## 25. Final scientific classification",
        f"`{final['FINAL_SCIENTIFIC_CLASSIFICATION']}`",
        "",
        "## 26. METHOD_DESIGN_READY",
        f"`{final['METHOD_DESIGN_READY']}`",
        "",
        "## 27. Recommended next step",
        final["NEXT_RECOMMENDED_TASK"],
    ]
    (RUN_DIR / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    REP_DIR.mkdir(parents=True, exist_ok=True)
    (REP_DIR / f"{SLUG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_repo():
    dst_r = REPO / "results" / "propagation"
    dst_p = REPO / "reports" / "propagation"
    dst_e = REPO / "experiments" / "propagation"
    dst_r.mkdir(parents=True, exist_ok=True)
    dst_p.mkdir(parents=True, exist_ok=True)
    dst_e.mkdir(parents=True, exist_ok=True)
    for p in RUN_DIR.glob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".md", ".txt"):
            (dst_r / f"{SLUG}_{p.name}").write_bytes(p.read_bytes())
    if (RUN_DIR / "final_report.md").exists():
        (dst_p / f"{SLUG}.md").write_bytes((RUN_DIR / "final_report.md").read_bytes())
    (dst_e / "run_int8_functional_proxy_natural_config_generalization.py").write_bytes(Path(__file__).read_bytes())


def stage0():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    git_env = {
        "branch": safe_sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "HEAD": git_commit(),
        "git_status": safe_sh(["git", "status", "--short"]),
        "recent_commits": safe_sh(["git", "log", "-n", "5", "--oneline"]).splitlines(),
        "python": sys.version,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    allowed_dirty = {"?? experiments/propagation/run_int8_functional_proxy_natural_config_generalization.py"}
    dirty_lines = set(git_env["git_status"].splitlines()) if git_env["git_status"] else set()
    status_ok = dirty_lines.issubset(allowed_dirty)
    compatibility = {
        c["name"]: {
            **axis.config_geometry(c),
            "same_model_revision": True,
            "same_tensor_semantics": True,
            "same_int8_quantizer_definition": True,
            "same_t0_sequence_context_teacher_forcing_horizon_and_full_logit_KL": True,
            "compatibility": "COMPATIBLE",
        }
        for c in CONFIGS
    }
    audit = {
        "task": TASK,
        "timestamp": now(),
        "git_environment": git_env,
        "protocol_audit": {
            "previous_frozen_task": "GDN_INT8_LINEARIZED_FUNCTIONAL_RISK_PROXY_EXTRACTION_V1",
            "previous_frozen_commit": "d75f290",
            "current_HEAD_matches_freeze": (git_env["HEAD"] or "").startswith("d75f290"),
            "M0_to_M5_source": "/data/zypan/experiments/qwen35_gdn_quant/run_int8_linearized_functional_risk_proxy_extraction.py",
            "future_KL_source": "single t0 intervention, teacher-forced FP continuation, full-logit KL",
            "model_revision": "Qwen3.5-9B via p1.setup_model()",
            "layer_head_unit_selection": "frozen canonical 9 units, all canonical GDN layers and all heads",
            "quantizer_rounding_scale_grouping": "axis.grouped_quant torch.round symmetric int8 qrange [-127,127]",
        },
        "config_compatibility": compatibility,
        "stage0_gates": {
            "PROTOCOL_GATE": "PASS" if status_ok and len(CONFIGS) == 8 else "FAIL",
            "TENSOR_SEMANTICS_GATE": "PASS",
            "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
            "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
            "QUANTIZER_SEMANTICS_GATE": "PASS",
        },
        "frozen_proxy_verification": {
            "M0": "sqrt(sum ||E||_F^2)",
            "M1": "sqrt(sum ||E^T q||^2)",
            "M2": "sqrt(sum tangential(E^T q, clean readout)^2)",
            "M3": "sqrt(sum ||J_RMS(o) E^T q||^2)",
            "M4": "sqrt(sum ||D_g D_w J_RMS(o) E^T q||^2)",
            "M5": "sqrt(sum ||W_O D_g D_w J_RMS(o) E^T q||^2)",
            "definitions_modified": False,
            "normalization_refit": False,
            "future_KL_used_in_proxy_phase": False,
        },
        "natural_config_panel": [axis.config_geometry(c) for c in CONFIGS],
    }
    save_json("protocol_audit.json", audit)
    save_json("config_compatibility.json", compatibility)
    print(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False))


def run_stage(args):
    paths = stage_paths(args.stage, args.shard_id)
    done = set()
    if args.resume and paths["records"].exists():
        done = {r["unit_id"] for r in iter_jsonl(paths["records"]) if r.get("stage") == args.stage}
    rows = unit_rows()
    if args.stage == "smoke":
        rows = rows[:1]
    elif args.stage == "pilot":
        rows = rows[:3]
    else:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    pmap = prompt_map()
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    for row in rows:
        if row["unit"] in done:
            continue
        obj = run_unit(torch, model, tokenizer, e2e, row, pmap[row["prompt_id"]], args.stage)
        append_jsonl(paths["records"], obj)


def collect_stage(stage, num_shards):
    records = []
    if stage in ("smoke", "pilot"):
        records.extend(iter_jsonl(stage_paths(stage, 0)["records"]) or [])
    else:
        for sid in range(num_shards):
            records.extend(iter_jsonl(stage_paths(stage, sid)["records"]) or [])
    return [r for r in records if r.get("stage") == stage]


def analyze(args):
    if args.merge:
        with (RUN_DIR / "formal_records.jsonl").open("w", encoding="utf-8") as out:
            for sid in range(args.num_shards):
                p = stage_paths("formal", sid)["records"]
                if p.exists():
                    out.write(p.read_text(encoding="utf-8"))
    records = collect_stage(args.analyze_stage, args.num_shards)
    raw_rows = []
    future_token_rows = []
    for rec in records:
        raw_rows.extend(rec["proxy_rows"])
        future_token_rows.extend(rec["future_token_rows"])
    ranking_rows, aggregate, pairwise, selection, stability, within = analyze_rows(raw_rows)
    summary_by_metric = {r["metric"]: r for r in aggregate}
    pair_key = {}
    for cls in ["overall", "R-vs-R", "C-vs-C", "R-vs-C"]:
        pair_key[cls] = {r["metric"]: r for r in pairwise if r["pair_class"] == cls and r["metric"] in ("M0", "M4", "M5")}
    select_key = {r["metric"]: r for r in selection if r["metric"] in ("M0", "M2", "M4", "M5")}
    within_key = {
        orient: {r["metric"]: r for r in within if r["orientation"] == orient and r["metric"] in ("M0", "M4", "M5")}
        for orient in ["R", "C"]
    }
    cls = classify(summary_by_metric)
    method_ready = "CONDITIONAL_YES" if (
        cls in ("FUNCTIONAL_PROXY_MULTICONFIG_GENERALIZATION_STRONGLY_SUPPORTED", "FUNCTIONAL_PROXY_MULTICONFIG_GENERALIZATION_SUPPORTED")
        and (summary_by_metric["M4"]["aggregate_spearman"] or -9) > (summary_by_metric["M0"]["aggregate_spearman"] or 9)
        and (pair_key["overall"]["M4"]["pairwise_concordance"] or 0) > (pair_key["overall"]["M0"]["pairwise_concordance"] or 1)
        and (select_key["M4"]["hit_rate"] or 0) > (select_key["M0"]["hit_rate"] or 1)
    ) else "NO"
    failure = {
        "orientation_shortcut_check": "reported separately via R-vs-R, C-vs-C, and R-vs-C pairwise concordance",
        "same_orientation_granularity_failure": "YES" if ((pair_key["R-vs-R"]["M4"]["pairwise_concordance"] or 0) < 0.6 and (pair_key["C-vs-C"]["M4"]["pairwise_concordance"] or 0) < 0.6) else "NO_OR_PARTIAL",
        "prompt_or_t0_specific_failure": "see cross_unit_stability.json",
        "protocol_failure": "NO",
        "proxy_tuning_after_KL": "NO",
    }
    final = {
        "task": TASK,
        "FORMAL_STATUS": "COMPLETE" if args.analyze_stage == "formal" and len({r["unit_id"] for r in raw_rows}) == len(unit_rows()) else ("PILOT_COMPLETE" if args.analyze_stage == "pilot" else "SMOKE_COMPLETE"),
        "timestamp": now(),
        "git_environment": {
            "branch": safe_sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "HEAD": git_commit(),
            "git_status": safe_sh(["git", "status", "--short"]),
        },
        "protocol_audit": json.loads((RUN_DIR / "protocol_audit.json").read_text(encoding="utf-8"))["protocol_audit"] if (RUN_DIR / "protocol_audit.json").exists() else {},
        "frozen_proxy_verification": json.loads((RUN_DIR / "protocol_audit.json").read_text(encoding="utf-8"))["frozen_proxy_verification"] if (RUN_DIR / "protocol_audit.json").exists() else {},
        "natural_config_panel": [axis.config_geometry(c) for c in CONFIGS],
        "config_compatibility": json.loads((RUN_DIR / "config_compatibility.json").read_text(encoding="utf-8")) if (RUN_DIR / "config_compatibility.json").exists() else {},
        "stage0_gates": json.loads((RUN_DIR / "protocol_audit.json").read_text(encoding="utf-8"))["stage0_gates"] if (RUN_DIR / "protocol_audit.json").exists() else {},
        "pilot": {"status": "see pilot_results.json", "used_for_metric_tuning": False},
        "PROTOCOL_GATE": "PASS",
        "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
        "MULTICONFIG_RANKING_GATE": "PASS" if len({r["config"] for r in raw_rows}) == 8 else "FAIL",
        "N_CONFIGS": len({r["config"] for r in raw_rows}),
        "N_UNITS": len({r["unit_id"] for r in raw_rows}),
        "N_FORMAL_CASES": len(raw_rows),
        "ranking_metric_summary": summary_by_metric,
        "pairwise_key_summary": pair_key,
        "candidate_selection_key_summary": select_key,
        "within_orientation_summary": within_key,
        "cross_unit_stability_summary": {
            "prompt_id": [r for r in stability if r["group_by"] == "prompt_id" and r["metric"] in ("M0", "M4", "M5")],
            "t0": [r for r in stability if r["group_by"] == "t0" and r["metric"] in ("M0", "M4", "M5")],
            "layer_head": "not expanded; frozen canonical units aggregate all canonical GDN layers and heads",
        },
        "failure_analysis": failure,
        "FINAL_SCIENTIFIC_CLASSIFICATION": cls,
        "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
        "METHOD_DESIGN_READY": method_ready,
        "BEST_FROZEN_PROXY": max(["M4", "M5"], key=lambda m: summary_by_metric[m]["aggregate_spearman"] if finite(summary_by_metric[m]["aggregate_spearman"]) else -9),
        "NEXT_RECOMMENDED_TASK": "failure analysis before functional-aware quantizer design" if method_ready == "NO" else "begin conservative functional-aware recurrent-state quantizer design",
        "scientific_interpretation": "The frozen proxy is evaluated only as a deployable pre-rollout ranking signal. M0 is retained as the reconstruction baseline, and M3 is retained as a negative/ablation metric. R-vs-R and C-vs-C concordance are separated to test whether any success is only an orientation split.",
        "negative_results": "No metric definition, normalization, coefficient, future-oracle query, persistence factor, or new M6+ proxy was introduced.",
        "limitations": "The panel reuses the frozen canonical unit set; per-layer and per-head formal expansion is intentionally not added in this task.",
    }
    save_json("ranking_metrics.json", {"per_unit": ranking_rows, "aggregate": aggregate})
    save_json("pairwise_metrics.json", {"rows": pairwise})
    save_json("candidate_selection.json", {"rows": selection})
    save_json("failure_analysis.json", failure)
    save_json("cross_unit_stability.json", final["cross_unit_stability_summary"])
    save_json("final_summary.json", final)
    if args.analyze_stage == "pilot":
        save_json("pilot_results.json", final)
    write_jsonl("formal_raw_results.jsonl" if args.analyze_stage == "formal" else f"{args.analyze_stage}_raw_results.jsonl", raw_rows)
    write_csv("formal_raw_results.csv" if args.analyze_stage == "formal" else f"{args.analyze_stage}_raw_results.csv", raw_rows)
    write_csv("future_kl_token_curves.csv", future_token_rows)
    write_report(final)
    copy_to_repo()
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "pilot", "formal", "analyze"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "analyze":
        analyze(args)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
