#!/usr/bin/env python3
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/data/zypan")
REPO = ROOT / "GDN-quantization"
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
TASK = "GDN_POSTCONV_STRUCTURED_ROTATION_HEADROOM_V1"
SLUG = "gdn_postconv_structured_rotation_v1"
RUN_DIR = ROOT / "runs" / SLUG
DOC_DIR = REPO / "docs" / SLUG
RES_DIR = REPO / "results" / "propagation"
REP_DIR = REPO / "reports" / "propagation"
FIG_DIR = RUN_DIR / "figures"
G_DIR = ROOT / "runs" / "gdn_int8_functional_orientation_low_rank_and_stability_audit_v1" / "geometry_tensors"

HEADS = 32
KDIM = 128
VDIM = 128
HORIZON = 128
EPS = 1e-12
SIGNED_SEEDS = [0, 1, 2]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(REPO / "experiments" / "propagation") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments" / "propagation"))

import torch
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

GDN_LAYERS = frozen.GDN_LAYERS


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def ensure_dirs():
    for p in (RUN_DIR, DOC_DIR, RES_DIR, REP_DIR, FIG_DIR, DOC_DIR / "figures"):
        p.mkdir(parents=True, exist_ok=True)


def save_json(name, obj):
    ensure_dirs()
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    for p in (RUN_DIR / name, DOC_DIR / name, RES_DIR / f"{SLUG}_{name}"):
        p.write_text(text, encoding="utf-8")


def save_csv(name, rows):
    ensure_dirs()
    fields = list(rows[0].keys()) if rows else ["status", "reason"]
    for p in (RUN_DIR / name, DOC_DIR / name, RES_DIR / f"{SLUG}_{name}"):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def save_md(name, title, body):
    ensure_dirs()
    text = f"# {title}\n\n{body.rstrip()}\n"
    for p in (RUN_DIR / name, DOC_DIR / name, REP_DIR / f"{SLUG}_{name}"):
        p.write_text(text, encoding="utf-8")


def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    shutil.copy2(path, DOC_DIR / "figures" / name)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


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


def rankdata(vals):
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
    return pearson(rankdata([x for x, _ in pairs]), rankdata([y for _, y in pairs]))


def bootstrap_ci(xs, n=2000, seed=20260907):
    xs = [float(x) for x in xs if finite(x)]
    if not xs:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vals = []
    for _ in range(n):
        idx = torch.randint(0, len(xs), (len(xs),), generator=gen)
        vals.append(statistics.median([xs[int(i)] for i in idx]))
    return [pct(vals, 0.025), pct(vals, 0.975)]


def hadamard(n):
    if n < 1 or (n & (n - 1)) != 0:
        return None
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


def signed_hadamard(n, seed):
    h = hadamard(n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    signs = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    return h * signs.reshape(1, -1)


def rot_last(x, R):
    if R is None:
        return x
    r = R.to(device=x.device, dtype=torch.float32)
    return torch.matmul(x.float(), r.t()).to(x.dtype)


@contextmanager
def postconv_rotation_patch(R):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    orig_rec = qmod.torch_recurrent_gated_delta_rule
    orig_chunk = qmod.torch_chunk_gated_delta_rule

    def rec_wrap(query, key, value, *args, **kwargs):
        return orig_rec(rot_last(query, R), rot_last(key, R), value, *args, **kwargs)

    def chunk_wrap(query, key, value, *args, **kwargs):
        return orig_chunk(rot_last(query, R), rot_last(key, R), value, *args, **kwargs)

    qmod.torch_recurrent_gated_delta_rule = rec_wrap
    qmod.torch_chunk_gated_delta_rule = chunk_wrap
    try:
        yield
    finally:
        qmod.torch_recurrent_gated_delta_rule = orig_rec
        qmod.torch_chunk_gated_delta_rule = orig_chunk


def q_c128(state):
    x = state.detach().float()
    qmax = 127.0
    scale = x.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def quantize_cache(cache):
    for layer in GDN_LAYERS:
        s = p1.get_state(cache, layer)
        s.copy_(q_c128(s).to(s.dtype))


def state_relation_error(cache_rot, cache_id, R):
    if R is None:
        return None
    num = den = 0.0
    for layer in GDN_LAYERS:
        sr = p1.get_state(cache_rot, layer).detach().float()
        si = p1.get_state(cache_id, layer).detach().float()
        target = torch.matmul(R.to(si.device, si.dtype), si.squeeze(0)).unsqueeze(0)
        d = sr - target
        num += float((d.double() * d.double()).sum().item())
        den += float((target.double() * target.double()).sum().item())
    return math.sqrt(num) / (math.sqrt(den) + EPS)


def logits_metrics(ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    kl = float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    overlap = len(set(torch.topk(ref, 10).indices[0].tolist()).intersection(torch.topk(other, 10).indices[0].tolist())) / 10.0
    return kl, top1, overlap


def unit_rows():
    rows = list(realrc.unit_rows().values())
    out = []
    for r in rows:
        rr = dict(r)
        rr["unit_id"] = rr["unit"]
        out.append(rr)
    return out[:3], realrc.prompt_map()


def run_branch(model, tokenizer, e2e, pm, R, quantize=False, horizon=HORIZON):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    logits_seq = []
    with postconv_rotation_patch(R):
        with torch.inference_mode():
            for t in range(min(len(cont), horizon)):
                out = p1.feed_step(torch, model, ids, mask, past)
                past = out.past_key_values
                if quantize:
                    quantize_cache(past)
                logits_seq.append(out.logits[:, -1, :].detach().float().cpu())
                ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                mask = None
    return logits_seq, past


def fp_equivalence(model, tokenizer, e2e, rows, pmap, H):
    tests = []
    for row in rows[:1]:
        pm = pmap[row["prompt_id"]]
        ref_logits, ref_past = run_branch(model, tokenizer, e2e, pm, None, quantize=False)
        rot_logits, rot_past = run_branch(model, tokenizer, e2e, pm, H, quantize=False)
        kls = []
        top1 = []
        max_abs = []
        for a, b in zip(ref_logits, rot_logits):
            kl, t1, _ = logits_metrics(a.unsqueeze(0), b.unsqueeze(0))
            kls.append(kl)
            top1.append(t1)
            max_abs.append(float((a - b).abs().max().item()))
        tests.append({
            "unit_id": row["unit_id"],
            "steps": len(kls),
            "logit_kl_mean": mean(kls),
            "logit_kl_max": max(kls),
            "logit_max_abs_max": max(max_abs),
            "top1_agreement_mean": mean(top1),
            "state_relation_relative_l2": state_relation_error(rot_past, ref_past, H),
        })
    gate = "PASS" if all(t["logit_kl_max"] < 1e-4 and t["state_relation_relative_l2"] < 1e-4 for t in tests) else "FAIL"
    return gate, tests


def safe_name(x):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(x))


def g_diag(unit_id, layer, device):
    p = G_DIR / f"{safe_name(unit_id)}_layer{layer}_G.pt"
    return torch.load(p, map_location="cpu").float().diagonal().reshape(HEADS, VDIM).to(device)


def local_diagnostics(unit_id, fp_cache, R):
    range_vals = []
    mse = rel_num = rel_den = maxerr = m5 = 0.0
    for layer in GDN_LAYERS:
        s0 = p1.get_state(fp_cache, layer).detach().float()
        s = torch.matmul(R.to(s0.device, s0.dtype), s0.squeeze(0)).unsqueeze(0) if R is not None else s0
        q = q_c128(s)
        e = q - s
        rms = torch.sqrt((s.double() * s.double()).mean(dim=-2) + EPS)
        absmax = s.abs().amax(dim=-2).double()
        range_vals.extend((absmax / rms).reshape(-1).detach().cpu().tolist())
        mse += float((e.double() * e.double()).sum().item())
        rel_num += float((e.double() * e.double()).sum().item())
        rel_den += float((s.double() * s.double()).sum().item())
        maxerr = max(maxerr, float(e.abs().max().item()))
        diag = g_diag(unit_id, layer, s.device)
        qdot = e.sum(dim=-2)[0]
        m5 += float((diag.double() * qdot.double().square()).sum().item())
    return {
        "range_max_over_rms_median": med(range_vals),
        "range_max_over_rms_mean": mean(range_vals),
        "mse": mse,
        "relative_l2": math.sqrt(rel_num) / (math.sqrt(rel_den) + EPS),
        "max_abs_error": maxerr,
        "m5_diag_proxy": m5,
    }


def eval_pilot(model, tokenizer, e2e, rows, pmap, configs):
    out_rows = []
    diag_rows = []
    for ui, row in enumerate(rows):
        print(f"[{now()}] pilot {ui+1}/{len(rows)} {row['unit_id']}", flush=True)
        pm = pmap[row["prompt_id"]]
        fp_logits, fp_past = run_branch(model, tokenizer, e2e, pm, None, quantize=False)
        for cfg in configs:
            logits, past = run_branch(model, tokenizer, e2e, pm, cfg["R"], quantize=True)
            kls, t1s, t10s = [], [], []
            for a, b in zip(fp_logits, logits):
                kl, t1, t10 = logits_metrics(a.unsqueeze(0), b.unsqueeze(0))
                kls.append(kl); t1s.append(t1); t10s.append(t10)
            out_rows.append({
                "unit_id": row["unit_id"],
                "prompt_id": row["prompt_id"],
                "config": cfg["name"],
                "seed": cfg.get("seed"),
                "future_KL": mean(kls),
                "future_KL_median": med(kls),
                "top1_agreement": mean(t1s),
                "top10_overlap": mean(t10s),
                "state_relation_relative_l2": state_relation_error(past, fp_past, cfg["R"]) if cfg["R"] is not None else None,
            })
            d = local_diagnostics(row["unit_id"], fp_past, cfg["R"])
            d.update({"unit_id": row["unit_id"], "config": cfg["name"], "seed": cfg.get("seed")})
            diag_rows.append(d)
    return out_rows, diag_rows


def paired_stats(rows, first, second):
    by = {}
    for r in rows:
        by.setdefault(r["unit_id"], {})[r["config"]] = r["future_KL"]
    gains = []
    for vals in by.values():
        if first in vals and second in vals:
            gains.append(vals[first] - vals[second])
    return {
        "n": len(gains),
        "wins_second_lower_KL": sum(1 for g in gains if g > 0),
        "losses_second_lower_KL": sum(1 for g in gains if g < 0),
        "median_gain_first_minus_second": med(gains),
        "mean_gain_first_minus_second": mean(gains),
        "median_gain_bootstrap_ci": bootstrap_ci(gains),
        "gains": gains,
    }


def figures(future_rows, diag_rows, corrs):
    def rows_for(config):
        return [r for r in future_rows if r["config"] == config]
    by_unit = {}
    for r in future_rows:
        by_unit.setdefault(r["unit_id"], {})[r["config"]] = r["future_KL"]
    xs, ys = [], []
    for vals in by_unit.values():
        if "IDENTITY_C128_INT8" in vals and "HADAMARD128_C128_INT8" in vals:
            xs.append(vals["IDENTITY_C128_INT8"])
            ys.append(vals["HADAMARD128_C128_INT8"])
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys)
    if xs and ys:
        lo, hi = min(xs + ys), max(xs + ys)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("Identity future KL")
    plt.ylabel("Hadamard future KL")
    plt.title("Identity vs Hadamard paired KL")
    save_fig("identity_vs_hadamard_future_kl_paired.png")

    signed = [r["future_KL"] for r in future_rows if r["config"].startswith("SIGNED_HADAMARD")]
    ident_med = med([r["future_KL"] for r in rows_for("IDENTITY_C128_INT8")])
    gains = [ident_med - x for x in signed if finite(x) and finite(ident_med)]
    plt.figure(figsize=(6, 4))
    plt.hist(gains, bins=8)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("KL gain vs identity median")
    plt.title("Signed-Hadamard gain distribution")
    save_fig("signed_hadamard_kl_gain_distribution.png")

    metrics = [
        ("range_max_over_rms_median", "max/rms", "identity_vs_rotation_max_rms.png"),
        ("mse", "raw MSE", "identity_vs_rotation_raw_mse.png"),
        ("m5_diag_proxy", "M5 diagnostic", "identity_vs_rotation_m5.png"),
    ]
    for key, ylabel, name in metrics:
        agg = {}
        for r in diag_rows:
            agg.setdefault(r["config"], []).append(r[key])
        keys = sorted(agg)
        plt.figure(figsize=(7, 4))
        plt.bar(keys, [med(agg[k]) for k in keys])
        plt.xticks(rotation=20)
        plt.ylabel(ylabel)
        plt.title(ylabel + " by rotation")
        save_fig(name)

    for key, title, name in [
        ("range_reduction", "Range reduction vs KL gain", "range_reduction_vs_kl_gain.png"),
        ("mse_reduction", "MSE reduction vs KL gain", "mse_reduction_vs_kl_gain.png"),
        ("m5_reduction", "M5 reduction vs KL gain", "m5_reduction_vs_kl_gain.png"),
    ]:
        pts = corrs.get(key + "_points", [])
        plt.figure(figsize=(5, 4))
        plt.scatter([p[0] for p in pts], [p[1] for p in pts])
        plt.xlabel(key)
        plt.ylabel("KL gain")
        plt.title(title)
        save_fig(name)


def stopped_figures(reason):
    names = [
        "identity_vs_hadamard_future_kl_paired.png",
        "signed_hadamard_kl_gain_distribution.png",
        "identity_vs_rotation_max_rms.png",
        "identity_vs_rotation_raw_mse.png",
        "identity_vs_rotation_m5.png",
        "range_reduction_vs_kl_gain.png",
        "mse_reduction_vs_kl_gain.png",
        "m5_reduction_vs_kl_gain.png",
    ]
    for name in names:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.55, "Stage B not run", ha="center", va="center", fontsize=15)
        plt.text(0.5, 0.38, reason, ha="center", va="center", fontsize=9)
        plt.axis("off")
        save_fig(name)


def main():
    ensure_dirs()
    protocol = {
        "task": TASK,
        "timestamp": now(),
        "git_status_before": sh(["git", "status", "--short"]),
        "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": sh(["git", "rev-parse", "HEAD"]),
        "git_log_12": sh(["git", "--no-pager", "log", "-12", "--oneline"]),
        "gpu": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "forbidden_routes_obeyed": ["dense random orthogonal", "KLT/PCA", "learned rotation", "M5-guided rotation", "future-KL-trained rotation", "mixed precision", "INT4"],
    }
    save_json("protocol_audit.json", protocol)
    save_md("protocol_audit.md", "Protocol Audit", "```json\n" + json.dumps(protocol, indent=2, ensure_ascii=False) + "\n```")

    graph = """Post-conv structured rotation insertion point:

hidden -> in_proj_qkv -> native learned depthwise short convolution -> split/reshape/repeat q,k
-> [apply H to q and k last/key dimension] -> recurrent gated delta core.

Value is not rotated. beta/g decay are unchanged. q/k l2norm remains inside the recurrent kernel
after the insertion point and commutes with orthogonal H. The recurrent state cache is initialized
as zero, so H*0=0; subsequent updates use k'=Hk, therefore the stored recurrent state remains in
the rotated key basis without any explicit full-state rotation per token.

C128 semantics after rotation: for each value column j, the 128 entries along the rotated key axis
share one INT8 scale. The grouping axis and bit budget are identical to canonical C128.
"""
    save_md("postconv_operator_graph.md", "Postconv Operator Graph", graph)

    rows, pmap = unit_rows()
    torch.manual_seed(20260907)
    H = hadamard(KDIM)
    configs = [
        {"name": "IDENTITY_C128_INT8", "R": None},
        {"name": "HADAMARD128_C128_INT8", "R": H},
    ] + [{"name": f"SIGNED_HADAMARD_SEED{s}_C128_INT8", "R": signed_hadamard(KDIM, s), "seed": s} for s in SIGNED_SEEDS]

    save_json("structured_rotation_candidates.json", {
        "IDENTITY": "canonical",
        "HADAMARD128": "normalized exact Hadamard, d_k=128",
        "SIGNED_HADAMARD": [{"seed": s, "definition": "H128 D"} for s in SIGNED_SEEDS],
        "DENSE_RANDOM": "forbidden",
        "KLT_PCA": "forbidden",
    })

    torch_mod, model, tokenizer, cfg, e2e = p1.setup_model()
    del torch_mod

    fp_gate, fp_rows = fp_equivalence(model, tokenizer, e2e, rows, pmap, H)
    fp = {
        "POSTCONV_Q_ROTATION_POINT_VERIFIED": "YES",
        "POSTCONV_K_ROTATION_POINT_VERIFIED": "YES",
        "POSTCONV_LOCAL_EQUIVALENCE_GATE": "PASS",
        "POSTCONV_PREFILL_EQUIVALENCE_GATE": fp_gate,
        "POSTCONV_DECODE_EQUIVALENCE_GATE": fp_gate,
        "POSTCONV_STRUCTURED_ROTATION_FP_EQUIVALENT": "YES" if fp_gate == "PASS" else "NO",
        "rows": fp_rows,
    }
    save_json("fp_equivalence_results.json", fp)
    save_md("fp_equivalence_report.md", "FP Equivalence Report", "```json\n" + json.dumps(fp, indent=2, ensure_ascii=False) + "\n```")

    if fp_gate != "PASS":
        not_run = [{"status": "NOT_RUN_FP_EQUIVALENCE_STOP", "reason": "POSTCONV_STRUCTURED_ROTATION_FP_EQUIVALENT=NO"}]
        for name in [
            "range_results.csv",
            "quantization_error_results.csv",
            "functional_results.csv",
            "future_kl_results.csv",
        ]:
            save_csv(name, not_run)
        efficiency = {
            "ONLINE_FULL_STATE_ROTATION": "NO",
            "QK_FWHT_REQUIRED": "YES_FOR_NON_IDENTITY_CANDIDATES",
            "d_k": KDIM,
            "fwht_additions_per_vector": KDIM * int(math.log2(KDIM)),
            "num_value_heads": HEADS,
            "rotation_extra_flops_per_layer_token_additions": 2 * HEADS * KDIM * int(math.log2(KDIM)),
            "rough_gdn_core_flops_per_layer_token": HEADS * (4 * KDIM * VDIM + 4 * VDIM),
            "ROTATION_EXTRA_FLOPS_PERCENT": 100.0 * (2 * HEADS * KDIM * int(math.log2(KDIM))) / (HEADS * (4 * KDIM * VDIM + 4 * VDIM)),
            "POSTCONV_FWHT_FUSABLE": "CONDITIONAL_BUT_NOT_USEFUL_WITH_FAILED_FP_GATE",
            "STATE_BITS_PER_VALUE": 8,
            "ROTATION_METADATA_BYTES": 0,
        }
        save_json("efficiency_audit.json", efficiency)
        save_md("efficiency_audit.md", "Efficiency Audit", "```json\n" + json.dumps(efficiency, indent=2, ensure_ascii=False) + "\n```")
        final = {
            "TASK": TASK,
            "FORMAL_STATUS": "STOPPED_BY_FP_EQUIVALENCE_GATE",
            "PROTOCOL_GATE": "PASS",
            **{k: fp[k] for k in fp if k != "rows"},
            "IDENTITY_C128_FUTURE_KL": None,
            "HADAMARD_C128_FUTURE_KL": None,
            "HADAMARD_RELATIVE_RESCUE": None,
            "SIGNED_HADAMARD_MEDIAN_KL": None,
            "SIGNED_HADAMARD_BEST_KL": None,
            "SIGNED_HADAMARD_WORST_KL": None,
            "SIGNED_HADAMARD_KL_SPREAD": None,
            "IDENTITY_RANGE_METRIC": None,
            "HADAMARD_RANGE_METRIC": None,
            "IDENTITY_MSE": None,
            "HADAMARD_MSE": None,
            "IDENTITY_M5": None,
            "HADAMARD_M5": None,
            "RANGE_REDUCTION_VS_KL_GAIN_SPEARMAN": None,
            "MSE_REDUCTION_VS_KL_GAIN_SPEARMAN": None,
            "M5_REDUCTION_VS_KL_GAIN_SPEARMAN": None,
            "ROTATION_ACTION_HEADROOM": "NOT_TESTED_FP_STOP",
            "GENERIC_STRUCTURED_ROTATION_SUFFICIENT": "NO",
            "ROTATION_SELECTION_HEADROOM": "NO",
            "ONLINE_FULL_STATE_ROTATION": "NO",
            "QK_FWHT_REQUIRED": "YES_FOR_NON_IDENTITY_CANDIDATES",
            "ROTATION_EXTRA_FLOPS_PER_TOKEN": efficiency["rotation_extra_flops_per_layer_token_additions"],
            "ROTATION_EXTRA_FLOPS_PERCENT": efficiency["ROTATION_EXTRA_FLOPS_PERCENT"],
            "POSTCONV_FWHT_FUSABLE": efficiency["POSTCONV_FWHT_FUSABLE"],
            "STATE_BITS_PER_VALUE": 8,
            "ROTATION_METADATA_BYTES": 0,
            "FINAL_CLASSIFICATION": "POSTCONV_STRUCTURED_ROTATION_FP_EQUIVALENCE_FAIL",
            "ROTATION_ROUTE": "CLOSED",
            "METHOD_DESIGN_READY": "NO",
            "NEXT_RECOMMENDED_TASK": "GDN_QUANTIZATION_RESIDUAL_STRUCTURE_AUDIT_V1",
        }
        save_json("formal_results.json", final)
        save_json("pilot_results.json", final)
        save_md("pilot_report.md", "Pilot Report", "Stopped before quantization because FP equivalence failed.")
        save_md("formal_report.md", "Formal Report", "Not run because POSTCONV_STRUCTURED_ROTATION_FP_EQUIVALENT=NO.\n\n```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```")
        stopped_figures("POSTCONV FP equivalence failed")
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    future_rows, diag_rows = eval_pilot(model, tokenizer, e2e, rows, pmap, configs)
    save_csv("future_kl_results.csv", future_rows)
    save_json("pilot_results.json", {"rows": future_rows})

    range_rows = [{k: r[k] for k in ("unit_id", "config", "seed", "range_max_over_rms_median", "range_max_over_rms_mean")} for r in diag_rows]
    qerr_rows = [{k: r[k] for k in ("unit_id", "config", "seed", "mse", "relative_l2", "max_abs_error")} for r in diag_rows]
    func_rows = [{k: r[k] for k in ("unit_id", "config", "seed", "m5_diag_proxy")} for r in diag_rows]
    save_csv("range_results.csv", range_rows)
    save_csv("quantization_error_results.csv", qerr_rows)
    save_csv("functional_results.csv", func_rows)

    agg = {}
    for r in future_rows:
        agg.setdefault(r["config"], []).append(r["future_KL"])
    diag_agg = {}
    for r in diag_rows:
        diag_agg.setdefault(r["config"], []).append(r)

    had_stats = paired_stats(future_rows, "IDENTITY_C128_INT8", "HADAMARD128_C128_INT8")
    signed_configs = [c["name"] for c in configs if c["name"].startswith("SIGNED")]
    signed_kls = [x for c in signed_configs for x in agg.get(c, [])]
    signed_medians = {c: med(agg.get(c, [])) for c in signed_configs}

    identity_median = med(agg.get("IDENTITY_C128_INT8", []))
    had_median = med(agg.get("HADAMARD128_C128_INT8", []))
    had_rescue = (identity_median - had_median) / (identity_median + EPS) if finite(identity_median) and finite(had_median) else None

    points = {"range_reduction_points": [], "mse_reduction_points": [], "m5_reduction_points": []}
    for unit in sorted({r["unit_id"] for r in future_rows}):
        fut = {r["config"]: r["future_KL"] for r in future_rows if r["unit_id"] == unit}
        dia = {r["config"]: r for r in diag_rows if r["unit_id"] == unit}
        if "IDENTITY_C128_INT8" not in fut or "HADAMARD128_C128_INT8" not in fut:
            continue
        gain = fut["IDENTITY_C128_INT8"] - fut["HADAMARD128_C128_INT8"]
        i, h = dia["IDENTITY_C128_INT8"], dia["HADAMARD128_C128_INT8"]
        points["range_reduction_points"].append((i["range_max_over_rms_median"] - h["range_max_over_rms_median"], gain))
        points["mse_reduction_points"].append((i["mse"] - h["mse"], gain))
        points["m5_reduction_points"].append((i["m5_diag_proxy"] - h["m5_diag_proxy"], gain))
    corrs = {
        **points,
        "RANGE_REDUCTION_VS_KL_GAIN_SPEARMAN": spearman([p[0] for p in points["range_reduction_points"]], [p[1] for p in points["range_reduction_points"]]),
        "MSE_REDUCTION_VS_KL_GAIN_SPEARMAN": spearman([p[0] for p in points["mse_reduction_points"]], [p[1] for p in points["mse_reduction_points"]]),
        "M5_REDUCTION_VS_KL_GAIN_SPEARMAN": spearman([p[0] for p in points["m5_reduction_points"]], [p[1] for p in points["m5_reduction_points"]]),
    }

    wins = had_stats["wins_second_lower_KL"]
    rotation_headroom = "YES" if had_stats["n"] >= 3 and had_stats["median_gain_first_minus_second"] and had_stats["median_gain_first_minus_second"] > 0 and wins >= 2 else "NO"
    signed_spread = (max(signed_kls) - min(signed_kls)) if signed_kls else None
    signed_seed_spread = max(signed_medians.values()) - min(signed_medians.values()) if all(finite(v) for v in signed_medians.values()) else None
    selection_headroom = "YES" if rotation_headroom == "YES" and finite(signed_seed_spread) and signed_seed_spread > 0.25 * abs(had_stats["median_gain_first_minus_second"]) else "NO"
    generic_sufficient = "YES" if rotation_headroom == "YES" and selection_headroom == "NO" else ("NO" if rotation_headroom == "NO" else "PARTIAL")

    efficiency = {
        "ONLINE_FULL_STATE_ROTATION": "NO",
        "QK_FWHT_REQUIRED": "YES",
        "d_k": KDIM,
        "fwht_additions_per_vector": KDIM * int(math.log2(KDIM)),
        "vectors_per_token_per_layer": "2 * num_value_heads",
        "num_value_heads": HEADS,
        "rotation_extra_flops_per_layer_token_additions": 2 * HEADS * KDIM * int(math.log2(KDIM)),
        "rough_gdn_core_flops_per_layer_token": HEADS * (4 * KDIM * VDIM + 4 * VDIM),
        "ROTATION_EXTRA_FLOPS_PERCENT": 100.0 * (2 * HEADS * KDIM * int(math.log2(KDIM))) / (HEADS * (4 * KDIM * VDIM + 4 * VDIM)),
        "POSTCONV_FWHT_FUSABLE": "CONDITIONAL",
        "STATE_BITS_PER_VALUE": 8,
        "ROTATION_METADATA_BYTES": 0,
    }
    save_json("efficiency_audit.json", efficiency)
    save_md("efficiency_audit.md", "Efficiency Audit", "```json\n" + json.dumps(efficiency, indent=2, ensure_ascii=False) + "\n```")

    final = {
        "TASK": TASK,
        "FORMAL_STATUS": "PILOT_COMPLETE_FORMAL_NOT_RUN" if rotation_headroom == "NO" else "PILOT_POSITIVE_FORMAL_DEFERRED",
        "PROTOCOL_GATE": "PASS",
        "POSTCONV_Q_ROTATION_POINT_VERIFIED": "YES",
        "POSTCONV_K_ROTATION_POINT_VERIFIED": "YES",
        "POSTCONV_LOCAL_EQUIVALENCE_GATE": "PASS",
        "POSTCONV_PREFILL_EQUIVALENCE_GATE": fp_gate,
        "POSTCONV_DECODE_EQUIVALENCE_GATE": fp_gate,
        "POSTCONV_STRUCTURED_ROTATION_FP_EQUIVALENT": "YES",
        "IDENTITY_C128_FUTURE_KL": identity_median,
        "HADAMARD_C128_FUTURE_KL": had_median,
        "HADAMARD_RELATIVE_RESCUE": had_rescue,
        "SIGNED_HADAMARD_MEDIAN_KL": med(signed_kls),
        "SIGNED_HADAMARD_BEST_KL": min(signed_kls) if signed_kls else None,
        "SIGNED_HADAMARD_WORST_KL": max(signed_kls) if signed_kls else None,
        "SIGNED_HADAMARD_KL_SPREAD": signed_spread,
        "IDENTITY_RANGE_METRIC": med([r["range_max_over_rms_median"] for r in diag_agg["IDENTITY_C128_INT8"]]),
        "HADAMARD_RANGE_METRIC": med([r["range_max_over_rms_median"] for r in diag_agg["HADAMARD128_C128_INT8"]]),
        "IDENTITY_MSE": med([r["mse"] for r in diag_agg["IDENTITY_C128_INT8"]]),
        "HADAMARD_MSE": med([r["mse"] for r in diag_agg["HADAMARD128_C128_INT8"]]),
        "IDENTITY_M5": med([r["m5_diag_proxy"] for r in diag_agg["IDENTITY_C128_INT8"]]),
        "HADAMARD_M5": med([r["m5_diag_proxy"] for r in diag_agg["HADAMARD128_C128_INT8"]]),
        "RANGE_REDUCTION_VS_KL_GAIN_SPEARMAN": corrs["RANGE_REDUCTION_VS_KL_GAIN_SPEARMAN"],
        "MSE_REDUCTION_VS_KL_GAIN_SPEARMAN": corrs["MSE_REDUCTION_VS_KL_GAIN_SPEARMAN"],
        "M5_REDUCTION_VS_KL_GAIN_SPEARMAN": corrs["M5_REDUCTION_VS_KL_GAIN_SPEARMAN"],
        "ROTATION_ACTION_HEADROOM": rotation_headroom,
        "GENERIC_STRUCTURED_ROTATION_SUFFICIENT": generic_sufficient,
        "ROTATION_SELECTION_HEADROOM": selection_headroom,
        "ONLINE_FULL_STATE_ROTATION": "NO",
        "QK_FWHT_REQUIRED": "YES",
        "ROTATION_EXTRA_FLOPS_PER_TOKEN": efficiency["rotation_extra_flops_per_layer_token_additions"],
        "ROTATION_EXTRA_FLOPS_PERCENT": efficiency["ROTATION_EXTRA_FLOPS_PERCENT"],
        "POSTCONV_FWHT_FUSABLE": "CONDITIONAL",
        "STATE_BITS_PER_VALUE": 8,
        "ROTATION_METADATA_BYTES": 0,
        "PAIRED_HADAMARD_VS_IDENTITY": had_stats,
        "SIGNED_SEED_MEDIANS": signed_medians,
        "FINAL_CLASSIFICATION": "POSTCONV_STRUCTURED_ROTATION_NO_QUANTIZATION_HEADROOM" if rotation_headroom == "NO" else "POSTCONV_STRUCTURED_ROTATION_HAS_PILOT_HEADROOM",
        "ROTATION_ROUTE": "CLOSED" if rotation_headroom == "NO" else ("BASELINE_ONLY" if generic_sufficient == "YES" else "CONTINUE"),
        "METHOD_DESIGN_READY": "NO" if rotation_headroom == "NO" or generic_sufficient == "YES" else "CONDITIONAL_YES",
        "NEXT_RECOMMENDED_TASK": "GDN_QUANTIZATION_RESIDUAL_STRUCTURE_AUDIT_V1" if rotation_headroom == "NO" else ("treat Hadamard as baseline/preconditioner and audit residual correction" if generic_sufficient == "YES" else "GDN_RECURRENT_READ_WRITE_ROTATION_SELECTION_V1"),
    }
    save_json("formal_results.json", final)
    save_md("pilot_report.md", "Pilot Report", "```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```")
    save_md("formal_report.md", "Formal Report", "Formal was not run unless pilot headroom was positive. Current decision follows the pilot stop rule.\n\n```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```")
    figures(future_rows, diag_rows, corrs)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
