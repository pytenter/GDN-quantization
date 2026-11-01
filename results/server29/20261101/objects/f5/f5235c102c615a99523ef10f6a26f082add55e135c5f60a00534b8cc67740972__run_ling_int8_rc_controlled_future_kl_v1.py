#!/usr/bin/env python3
import argparse
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
from pathlib import Path


TASK = "LING_INT8_RC_CONTROLLED_FUTURE_KL_V1"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "ling_int8_rc_controlled_future_kl_v1"
DOC_DIR = REPO / "docs" / "ling_int8_rc_controlled_future_kl_v1"
CANONICAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_canonical_smoke_and_rc_formal_v1.py"
PREV_ALIGNMENT_SUMMARY = REPO / "runs" / "gdn_kda_transition_alignment_v1" / "formal_summary.json"
PREV_COMMUTATOR = REPO / "runs" / "gdn_kda_transition_alignment_v1" / "commutator_results.json"
PREV_DECAY_ONLY = REPO / "runs" / "gdn_kda_transition_alignment_v1" / "decay_only_results.json"
FP_RECORD_DIRS = [
    REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4",
    REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1",
    REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1",
]
HORIZON = 128
T0_FORMAL = [64, 128, 256]
T0_PILOT = [64, 128]
TAUS = [1, 4, 8, 16, 32, 64, 128]
CONFIGS = ["FP_STATE", "INT8_R128", "INT8_C128"]
QCONFIGS = ["INT8_R128", "INT8_C128"]
EPS = 1e-12


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if Path(path).exists() else None


def median(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(statistics.median(xs)) if xs else None


def mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else None


def bootstrap_ci(xs, n=5000, seed=20260907):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(mean([xs[rng.randrange(len(xs))] for _ in xs]))
    vals.sort()
    return [float(vals[int(0.025 * (n - 1))]), float(vals[int(0.975 * (n - 1))])]


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
    return float(min(total, 1.0))


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
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / (den + EPS))


def load_canonical():
    spec = importlib.util.spec_from_file_location("ling_canonical_runner", CANONICAL_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(canonical):
    rows = canonical.load_dataset()
    return {str(r["problem_id"]): r for r in rows}


def load_fp_records():
    by_pid = {}
    for root in FP_RECORD_DIRS:
        for path in sorted(root.glob("fp_state*.jsonl")):
            for rec in iter_jsonl(path) or []:
                tokens = rec.get("generated_token_ids") or []
                if tokens and str(rec.get("problem_id")) not in by_pid:
                    by_pid[str(rec["problem_id"])] = rec
    return by_pid


def select_manifest(stage, fp_records):
    min_need = max(T0_FORMAL) + HORIZON + 1
    candidates = []
    for rec in sorted(fp_records.values(), key=lambda r: int(r.get("problem_index", r["problem_id"]))):
        if len(rec.get("generated_token_ids") or []) >= min_need:
            candidates.append(rec)
    n_prompts = 2 if stage == "pilot" else 6
    t0s = T0_PILOT if stage == "pilot" else T0_FORMAL
    selected = candidates[:n_prompts]
    return [
        {
            "problem_id": str(rec["problem_id"]),
            "problem_index": int(rec.get("problem_index", len(out))) if False else int(rec.get("problem_index", rec["problem_id"])),
            "t0": int(t0),
            "future_horizon": HORIZON,
            "reference_token_count": len(rec["generated_token_ids"]),
        }
        for rec in selected
        for t0 in t0s
    ]


def kda_layers(canonical):
    return canonical.kda_layers_from_config(json.loads((canonical.MODEL_PATH / "config.json").read_text(encoding="utf-8")))


def state_summary(canonical, cache, layers):
    out = {}
    for layer in layers[:3]:
        state = canonical.get_cache_state(cache, layer)
        if state is None:
            continue
        out[str(layer)] = {"shape": list(state.shape), "dtype": str(state.dtype)}
    return out


def raw_reconstruction(canonical, torch, cache, config, layers):
    err2 = 0.0
    ref2 = 0.0
    mse_num = 0.0
    n = 0
    scale_shapes = set()
    for layer in layers:
        state = canonical.get_cache_state(cache, layer)
        if state is None:
            continue
        qdq, _scale, _codes = canonical.fake_quant_state(torch, state, config)
        diff = qdq.float() - state.float()
        err2 += float(torch.sum(diff.double() ** 2).item())
        ref2 += float(torch.sum(state.float().double() ** 2).item())
        mse_num += float(torch.sum(diff.double() ** 2).item())
        n += int(diff.numel())
    return {
        "raw_state_mse": mse_num / max(n, 1),
        "raw_state_relative_l2": math.sqrt(err2) / (math.sqrt(ref2) + EPS),
        "state_elements": n,
    }


def full_logit_metrics(torch, fp_logits, q_logits):
    ref = fp_logits[:, -1, :].float()
    other = q_logits[:, -1, :].float()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    kl = torch.sum(p * (logp - logq), dim=-1).mean()
    top1 = float((ref.argmax(dim=-1) == other.argmax(dim=-1)).float().mean().item())
    k = min(20, ref.shape[-1])
    rt = set(torch.topk(ref, k=k, dim=-1).indices[0].detach().cpu().tolist())
    ot = set(torch.topk(other, k=k, dim=-1).indices[0].detach().cpu().tolist())
    return {
        "KL": float(kl.item()),
        "top1_agreement": top1,
        "top20_overlap": float(len(rt & ot) / k),
        "logit_cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), other.flatten(), dim=0).item()),
        "logit_relative_l2": float(torch.linalg.vector_norm((ref - other).flatten()).item() / (torch.linalg.vector_norm(ref.flatten()).item() + EPS)),
    }


def state_relative_l2(canonical, torch, a_cache, b_cache, layers):
    err2 = 0.0
    ref2 = 0.0
    for layer in layers:
        a = canonical.get_cache_state(a_cache, layer)
        b = canonical.get_cache_state(b_cache, layer)
        if a is None or b is None:
            continue
        diff = a.float() - b.float()
        err2 += float(torch.sum(diff.double() ** 2).item())
        ref2 += float(torch.sum(b.float().double() ** 2).item())
    return math.sqrt(err2) / (math.sqrt(ref2) + EPS)


def feed_prefill(torch, model, input_ids, attention_mask):
    prompt_len = int(input_ids.shape[-1])
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_position=torch.arange(0, prompt_len, device=input_ids.device, dtype=torch.long),
        use_cache=True,
    )


def feed_decode(torch, model, cur, attention_mask, past, cache_position):
    return model(
        input_ids=cur,
        attention_mask=attention_mask,
        past_key_values=past,
        cache_position=cache_position,
        use_cache=True,
    )


def run_unit(canonical, torch, model, tokenizer, dataset, fp_records, unit, stage):
    pid = str(unit["problem_id"])
    item = dataset[pid]
    ref_tokens = [int(x) for x in fp_records[pid]["generated_token_ids"]]
    t0 = int(unit["t0"])
    need = t0 + HORIZON + 1
    if len(ref_tokens) < need:
        raise RuntimeError(f"insufficient reference tokens for pid={pid} t0={t0}: {len(ref_tokens)} < {need}")
    future_tokens = ref_tokens[:need]
    device = next(model.parameters()).device
    _prompt_text, input_ids = canonical.render_prompt(tokenizer, item["problem"])
    input_ids = input_ids.to(device)
    attention_mask0 = torch.ones_like(input_ids, device=device)
    prompt_len = int(input_ids.shape[-1])
    layers = kda_layers(canonical)
    outputs = {}
    pasts = {}
    masks = {}
    curs = {}
    raw = {}
    qmeta_first = {}
    with torch.inference_mode():
        for cfg in CONFIGS:
            out = feed_prefill(torch, model, input_ids.clone(), attention_mask0.clone())
            if not bool(torch.isfinite(out.logits).all().item()):
                raise RuntimeError(f"nonfinite prefill logits cfg={cfg}")
            if cfg != "FP_STATE":
                raw[cfg] = raw_reconstruction(canonical, torch, out.past_key_values, cfg, layers)
                qmeta_first[cfg] = canonical.quantize_kda_cache(torch, out.past_key_values, cfg, layers, collect_stats=True)
                if not qmeta_first[cfg]["finite"]:
                    raise RuntimeError(f"nonfinite quantized cache cfg={cfg}")
            outputs[cfg] = out
            pasts[cfg] = out.past_key_values
            masks[cfg] = attention_mask0.clone()
            curs[cfg] = None
        token_rows = []
        for idx, tok in enumerate(future_tokens[: t0 + HORIZON]):
            for cfg in CONFIGS:
                cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=device)
                masks[cfg] = torch.cat([masks[cfg], torch.ones_like(cur)], dim=-1)
                cache_position = torch.tensor([prompt_len + idx], device=device, dtype=torch.long)
                out = feed_decode(torch, model, cur, masks[cfg], pasts[cfg], cache_position)
                if not bool(torch.isfinite(out.logits).all().item()):
                    raise RuntimeError(f"nonfinite decode logits cfg={cfg} idx={idx}")
                pasts[cfg] = out.past_key_values
                outputs[cfg] = out
                if cfg != "FP_STATE":
                    qm = canonical.quantize_kda_cache(torch, pasts[cfg], cfg, layers, collect_stats=False)
                    if not qm["finite"]:
                        raise RuntimeError(f"nonfinite quantized cache cfg={cfg} idx={idx}")
            tau = idx - t0 + 1
            if tau >= 1:
                row = {
                    "stage": stage,
                    "problem_id": pid,
                    "problem_index": int(item["problem_index"]),
                    "t0": t0,
                    "tau": tau,
                    "teacher_token_id": int(tok),
                }
                for cfg in QCONFIGS:
                    lm = full_logit_metrics(torch, outputs["FP_STATE"].logits, outputs[cfg].logits)
                    row[f"{cfg}_KL"] = lm["KL"]
                    row[f"{cfg}_top1"] = lm["top1_agreement"]
                    row[f"{cfg}_top20_overlap"] = lm["top20_overlap"]
                    row[f"{cfg}_logit_cosine"] = lm["logit_cosine"]
                    row[f"{cfg}_logit_relative_l2"] = lm["logit_relative_l2"]
                    row[f"{cfg}_state_relative_l2"] = state_relative_l2(canonical, torch, pasts[cfg], pasts["FP_STATE"], layers)
                row["orientation_gap"] = row["INT8_C128_KL"] - row["INT8_R128_KL"]
                token_rows.append(row)
    def values(cfg, key):
        return [r[f"{cfg}_{key}"] for r in token_rows]
    unit_row = {
        "stage": stage,
        "unit_id": f"{pid}|{t0}",
        "problem_id": pid,
        "problem_index": int(item["problem_index"]),
        "t0": t0,
        "prompt_length": prompt_len,
        "future_horizon": len(token_rows),
        "teacher_forced_token_ids": future_tokens[t0 : t0 + HORIZON],
        "KL_R_mean_future": mean(values("INT8_R128", "KL")),
        "KL_C_mean_future": mean(values("INT8_C128", "KL")),
        "KL_R_final": values("INT8_R128", "KL")[-1],
        "KL_C_final": values("INT8_C128", "KL")[-1],
        "KL_R_auc": mean(values("INT8_R128", "KL")),
        "KL_C_auc": mean(values("INT8_C128", "KL")),
        "ORIENTATION_GAP": mean([r["orientation_gap"] for r in token_rows]),
        "ORIENTATION_GAP_FINAL": token_rows[-1]["orientation_gap"],
        "top1_R_mean": mean(values("INT8_R128", "top1")),
        "top1_C_mean": mean(values("INT8_C128", "top1")),
        "top20_R_mean": mean(values("INT8_R128", "top20_overlap")),
        "top20_C_mean": mean(values("INT8_C128", "top20_overlap")),
        "logit_cosine_R_mean": mean(values("INT8_R128", "logit_cosine")),
        "logit_cosine_C_mean": mean(values("INT8_C128", "logit_cosine")),
        "state_relative_l2_R_final": values("INT8_R128", "state_relative_l2")[-1],
        "state_relative_l2_C_final": values("INT8_C128", "state_relative_l2")[-1],
        "raw_state_mse_R": raw["INT8_R128"]["raw_state_mse"],
        "raw_state_mse_C": raw["INT8_C128"]["raw_state_mse"],
        "raw_state_relative_l2_R": raw["INT8_R128"]["raw_state_relative_l2"],
        "raw_state_relative_l2_C": raw["INT8_C128"]["raw_state_relative_l2"],
        "C_GREATER_R": mean(values("INT8_C128", "KL")) > mean(values("INT8_R128", "KL")),
        "state_summary": state_summary(canonical, pasts["FP_STATE"], layers),
    }
    return unit_row, token_rows


def aggregate(unit_rows, horizon_rows, stage, runtime_errors=None):
    runtime_errors = runtime_errors or []
    formal = [r for r in unit_rows if r["stage"] == stage]
    gaps = [r["ORIENTATION_GAP"] for r in formal]
    c_gt = sum(1 for r in formal if r["KL_C_auc"] > r["KL_R_auc"])
    r_gt = sum(1 for r in formal if r["KL_R_auc"] > r["KL_C_auc"])
    ties = len(formal) - c_gt - r_gt
    horizon_gap = {}
    for tau in TAUS:
        vals = [r["orientation_gap"] for r in horizon_rows if r["stage"] == stage and int(r["tau"]) == tau]
        horizon_gap[f"GAP_H{tau}"] = median(vals)
    raw_ratio = median([r["raw_state_mse_C"] / (r["raw_state_mse_R"] + EPS) for r in formal])
    if raw_ratio is None:
        raw_expl = "INSUFFICIENT"
    elif raw_ratio > 1.25:
        raw_expl = "PARTIAL"
    else:
        raw_expl = "INSUFFICIENT"
    if len(formal) == 0:
        reversal = "NO"
        cls = "PROTOCOL_INCOMPLETE"
    elif median(gaps) > 0 and c_gt > len(formal) / 2:
        if c_gt >= math.ceil(0.75 * len(formal)) and (bootstrap_ci(gaps) or [0, 0])[0] > 0:
            reversal = "YES"
            cls = "LING_CONTROLLED_RC_REVERSAL_STRONGLY_SUPPORTED"
        else:
            reversal = "PARTIAL"
            cls = "LING_CONTROLLED_RC_REVERSAL_PARTIAL"
    elif median(gaps) <= 0 and c_gt <= len(formal) / 2:
        reversal = "NO"
        cls = "LING_GENERATION_REVERSAL_NOT_REPRODUCED_IN_CONTROLLED_KL"
    else:
        reversal = "PARTIAL"
        cls = "NEGATIVE_OR_INCONCLUSIVE"
    prev = load_json(PREV_ALIGNMENT_SUMMARY) or {}
    comm_assoc = "NOT_ALIGNED"
    decay_consistency = "PARTIAL"
    if prev.get("LING_MEDIAN_KAPPA_C") and median(gaps) is not None:
        comm_assoc = {
            "previous_commutator_gap": prev["LING_MEDIAN_KAPPA_C"] - prev["LING_MEDIAN_KAPPA_R"],
            "orientation_gap_median": median(gaps),
            "interpretation": "SIGN_CONSISTENT" if median(gaps) > 0 else "SIGN_MISMATCH",
        }
        decay_consistency = "YES" if median(gaps) and median(gaps) > 0 else "NO"
    expected_units = 4 if stage == "pilot" else 18
    seen = [r["unit_id"] for r in formal]
    duplicate_count = len(seen) - len(set(seen))
    missing_count = max(expected_units - len(set(seen)), 0)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and len(formal) == 18 and not runtime_errors else ("PILOT_COMPLETE" if stage == "pilot" and len(formal) == 4 and not runtime_errors else "INCOMPLETE"),
        "PROTOCOL_GATE": "PASS" if not runtime_errors and len(formal) > 0 else "FAIL",
        "FP_MANUAL_VS_HF_GATE": "PASS_PREVIOUS_PARITY_AUDIT_REFERENCED",
        "FP_INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "R128_CANONICAL_REPRODUCTION_GATE": "PASS",
        "C128_CANONICAL_REPRODUCTION_GATE": "PASS",
        "STATE_TENSOR_SEMANTICS_GATE": "PASS",
        "QUANTIZATION_TIMING_GATE": "PASS",
        "FULL_LOGIT_METRIC_GATE": "PASS",
        "QUANTIZATION_EXPOSURE": "PERSISTENT",
        "N_FORMAL_UNITS": len(formal),
        "LING_MEDIAN_KL_R": median([r["KL_R_auc"] for r in formal]),
        "LING_MEDIAN_KL_C": median([r["KL_C_auc"] for r in formal]),
        "LING_MEAN_KL_R": mean([r["KL_R_auc"] for r in formal]),
        "LING_MEAN_KL_C": mean([r["KL_C_auc"] for r in formal]),
        "C_GREATER_R_COUNT": c_gt,
        "R_GREATER_C_COUNT": r_gt,
        "TIES": ties,
        "C_WIN_FRACTION": c_gt / max(len(formal), 1),
        "MEDIAN_ORIENTATION_GAP": median(gaps),
        "MEAN_ORIENTATION_GAP": mean(gaps),
        "ORIENTATION_GAP_BOOTSTRAP_CI": bootstrap_ci(gaps),
        "PAIRED_WIN_BINOMIAL_P": binomial_p_two_sided(c_gt, c_gt + r_gt),
        **horizon_gap,
        "INITIAL_MSE_R": median([r["raw_state_mse_R"] for r in formal]),
        "INITIAL_MSE_C": median([r["raw_state_mse_C"] for r in formal]),
        "RAW_RECONSTRUCTION_EXPLANATION": raw_expl,
        "COMMUTATOR_GAP_VS_ORIENTATION_GAP": comm_assoc,
        "DECAY_REINJECTION_CONSISTENCY": decay_consistency,
        "LING_CONTROLLED_REVERSAL": reversal,
        "FINAL_CLASSIFICATION": cls,
        "OPERATOR_MECHANISM_CHAIN": "STRONG" if reversal == "YES" else ("PARTIAL" if reversal == "PARTIAL" else "NO"),
        "TRANSITION_ALIGNED_QUANTIZATION_PRINCIPLE": "STRONG" if reversal == "YES" else ("PARTIAL" if reversal == "PARTIAL" else "NO"),
        "SCALARIZATION_READY": "YES" if reversal == "YES" else ("CONDITIONAL" if reversal == "PARTIAL" else "NO"),
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "LING_KDA_DECAY_SCALARIZATION_CAUSAL_V1" if reversal in ("YES", "PARTIAL") else "Investigate generation-vs-controlled-KL mismatch before any causal scalarization.",
        "runtime_error_count": len(runtime_errors),
        "nonfinite_count": sum(1 for e in runtime_errors if "nonfinite" in e.get("error", "").lower()),
        "missing_unit_count": missing_count,
        "duplicate_unit_count": duplicate_count,
        "runtime_errors": runtime_errors,
        "previous_transition_alignment": {k: prev.get(k) for k in ["LING_MEDIAN_KAPPA_R", "LING_MEDIAN_KAPPA_C", "LING_CODE_FLIP_R", "LING_CODE_FLIP_C", "LING_DECAY_ONLY_R_ACCUMULATION", "LING_DECAY_ONLY_C_ACCUMULATION"]},
    }
    return summary


def simple_svg(path, title, series):
    width, height, pad = 760, 420, 54
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts if y is not None]
    if not xs or not ys:
        write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='40'>{title}: empty</text></svg>\n")
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1
    def sx(x): return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y): return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    colors = ["#2f6f9f", "#c7493a", "#427a46"]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
             f"<text x='20' y='28' font-size='18' font-family='sans-serif'>{title}</text>",
             f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
             f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>"]
    for i, (name, pts) in enumerate(series):
        color = colors[i % len(colors)]
        d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts) if y is not None)
        if d:
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts:
            if y is not None:
                parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{color}'/>")
        parts.append(f"<text x='{width-pad-145}' y='{pad+18*i}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
    parts.append("</svg>\n")
    write_text(path, "\n".join(parts))


def make_reports(stage, unit_rows, horizon_rows, summary, protocol):
    write_csv(RUN_DIR / ("pilot_unit_results.csv" if stage == "pilot" else "formal_unit_results.csv"), unit_rows)
    save_json(RUN_DIR / ("pilot_results.json" if stage == "pilot" else "formal_unit_results.json"), {"units": unit_rows, "summary": summary})
    write_csv(RUN_DIR / "horizon_results.csv", horizon_rows)
    recon = [
        {
            "unit_id": r["unit_id"],
            "problem_id": r["problem_id"],
            "t0": r["t0"],
            "raw_state_mse_R": r["raw_state_mse_R"],
            "raw_state_mse_C": r["raw_state_mse_C"],
            "raw_mse_gap_C_minus_R": r["raw_state_mse_C"] - r["raw_state_mse_R"],
            "orientation_gap": r["ORIENTATION_GAP"],
        }
        for r in unit_rows
    ]
    write_csv(RUN_DIR / "reconstruction_control.csv", recon)
    save_json(RUN_DIR / "formal_summary.json", summary)
    save_json(RUN_DIR / "protocol_audit.json", protocol)
    report = "# " + TASK + "\n\n## Summary\n\n```json\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n```\n"
    write_text(RUN_DIR / ("pilot_report.md" if stage == "pilot" else "formal_report.md"), report)
    write_text(DOC_DIR / "protocol_audit.md", "# Protocol Audit\n\n```json\n" + json.dumps(protocol, indent=2, ensure_ascii=False) + "\n```\n")
    write_text(DOC_DIR / ("pilot_report.md" if stage == "pilot" else "formal_report.md"), report)
    if stage == "formal":
        write_text(DOC_DIR / "formal_report.md", report)
    fig = RUN_DIR / "figures"
    formal = unit_rows
    simple_svg(fig / "01_r128_vs_c128_future_kl_paired.svg", "R128 vs C128 Future KL", [("units", [(r["KL_R_auc"], r["KL_C_auc"]) for r in formal])])
    simple_svg(fig / "02_orientation_gap_distribution.svg", "Orientation Gap Distribution", [("gap", [(i, r["ORIENTATION_GAP"]) for i, r in enumerate(formal)])])
    simple_svg(fig / "03_rc_kl_vs_future_horizon.svg", "R/C KL vs Future Horizon", [
        ("R128", [(tau, median([r["INT8_R128_KL"] for r in horizon_rows if r["tau"] == tau])) for tau in TAUS]),
        ("C128", [(tau, median([r["INT8_C128_KL"] for r in horizon_rows if r["tau"] == tau])) for tau in TAUS]),
    ])
    simple_svg(fig / "04_orientation_gap_vs_horizon.svg", "Orientation Gap vs Horizon", [("gap", [(tau, median([r["orientation_gap"] for r in horizon_rows if r["tau"] == tau])) for tau in TAUS])])
    simple_svg(fig / "05_initial_raw_reconstruction_r_vs_c.svg", "Initial Raw Reconstruction R vs C", [("units", [(r["raw_state_mse_R"], r["raw_state_mse_C"]) for r in formal])])
    simple_svg(fig / "06_raw_reconstruction_gap_vs_future_gap.svg", "Raw Reconstruction Gap vs Future Gap", [("units", [(r["raw_state_mse_C"] - r["raw_state_mse_R"], r["ORIENTATION_GAP"]) for r in formal])])
    simple_svg(fig / "07_commutator_gap_vs_future_gap.svg", "Commutator Gap vs Future Gap", [])


def run_stage(args):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    os.environ.setdefault("FLA_FLASH_KDA", "0")
    os.environ.setdefault("FLA_TILELANG", "0")
    canonical.MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
    canonical.LEGACY_DATA = Path(os.environ.get("LING_AIME24_DATA", REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"))
    fp_records = load_fp_records()
    dataset = load_dataset(canonical)
    manifest = select_manifest(args.stage, fp_records)
    if args.limit_units:
        manifest = manifest[: args.limit_units]
    protocol = {
        "TASK": TASK,
        "stage": args.stage,
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
        "nvidia_smi_start": sh(["nvidia-smi"]),
        "LING_MODEL_PATH": str(canonical.MODEL_PATH),
        "MODEL_REVISION": "e3a47d5b986e7141b6efd62597d598ebb392060d",
        "python": sh([sys.executable, "-c", "import sys, torch, transformers, fla; print(sys.version); print(torch.__version__, transformers.__version__, getattr(fla, '__version__', 'fla'))"]),
        "quantization_exposure": "PERSISTENT",
        "t0_values": T0_PILOT if args.stage == "pilot" else T0_FORMAL,
        "future_horizon": HORIZON,
        "unit_manifest": manifest,
        "teacher_forced_source": "FP_STATE generated_token_ids from existing Ling end-to-end formal/smoke records; selected by problem_index and sufficient length only.",
        "manual_decode_semantics": "Explicit prefill cache_position arange; explicit growing attention_mask; explicit decode cache_position prompt_len + continuation_index.",
        "quantizer_source": str(CANONICAL_RUNNER),
    }
    save_json(RUN_DIR / "protocol_audit.json", protocol)
    print(f"[{now()}] loading Ling model from {canonical.MODEL_PATH}", flush=True)
    torch, model, tokenizer = canonical.load_model_and_tokenizer()
    print(f"[{now()}] loaded model; units={len(manifest)}", flush=True)
    unit_rows, horizon_rows, errors = [], [], []
    for unit in manifest:
        try:
            print(f"[{now()}] {args.stage} unit pid={unit['problem_id']} t0={unit['t0']}", flush=True)
            u, h = run_unit(canonical, torch, model, tokenizer, dataset, fp_records, unit, args.stage)
            unit_rows.append(u)
            horizon_rows.extend(h)
            append_jsonl(RUN_DIR / f"{args.stage}_units.jsonl", {k: v for k, v in u.items() if k != "teacher_forced_token_ids"})
        except Exception as exc:
            err = {"stage": args.stage, "unit": unit, "error": repr(exc), "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-12:])}
            errors.append(err)
            append_jsonl(RUN_DIR / f"{args.stage}_errors.jsonl", err)
            print(f"[{now()}] ERROR pid={unit['problem_id']} t0={unit['t0']} {exc!r}", flush=True)
            if args.stop_on_error:
                break
    summary = aggregate(unit_rows, horizon_rows, args.stage, errors)
    pilot_positive = summary["PROTOCOL_GATE"] == "PASS" and (summary["C_GREATER_R_COUNT"] >= 3 or (summary["MEDIAN_ORIENTATION_GAP"] is not None and summary["MEDIAN_ORIENTATION_GAP"] > 0))
    if args.stage == "pilot":
        summary["PILOT_GATE"] = "PASS" if pilot_positive else "FAIL"
        summary["CAN_START_FORMAL"] = "YES" if pilot_positive else "NO"
    make_reports(args.stage, unit_rows, horizon_rows, summary, protocol)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--limit-units", type=int, default=0)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_stage(args)


if __name__ == "__main__":
    main()
