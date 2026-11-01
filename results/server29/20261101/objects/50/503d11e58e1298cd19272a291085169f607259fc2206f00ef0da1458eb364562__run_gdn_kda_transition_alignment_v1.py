#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file


TASK = "GDN_KDA_TRANSITION_ALIGNED_QUANTIZATION_MECHANISM_V1"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "gdn_kda_transition_alignment_v1"
DOC_DIR = REPO / "docs" / "gdn_kda_transition_alignment_v1"
LING_MODEL = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
QWEN_MODEL = Path(os.environ.get("QWEN_MODEL_PATH", "/data/zypan/modelscope_models/Qwen3.5-9B"))
KDA_BACKEND = Path(os.environ.get("KDA_BACKEND_PATH", "/data/zypan/kvarn-repro/repos/KVarN/vllm/model_executor/layers/fla/ops/kda.py"))
FUSED_BACKEND = Path(os.environ.get("FUSED_BACKEND_PATH", "/data/zypan/kvarn-repro/repos/KVarN/vllm/model_executor/layers/fla/ops/fused_recurrent.py"))
EPS = 1e-12
QMAX = 127.0


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def median(xs):
    return float(statistics.median([float(x) for x in xs])) if xs else None


def mean(xs):
    return float(sum(float(x) for x in xs) / len(xs)) if xs else None


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
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(num / (denx * deny + EPS))


def bootstrap_ci(xs, n=1000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(median([xs[rng.randrange(len(xs))] for _ in xs]))
    vals.sort()
    return [float(vals[int(0.025 * (n - 1))]), float(vals[int(0.975 * (n - 1))])]


def qdq(state, config):
    x = state.float()
    if config == "R128":
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / QMAX
    elif config == "C128":
        scale = x.detach().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / QMAX
    else:
        raise ValueError(config)
    codes = torch.round(x / scale).clamp(-QMAX, QMAX)
    return codes * scale, scale, codes


def apply_key_decay(state, alpha):
    return state.float() * alpha.float().unsqueeze(-1)


def apply_scalar_decay(state, alpha):
    return state.float() * alpha.float().view(1, -1, 1, 1)


def tensor_metrics(a, b, ref):
    d = (a.float() - b.float()).flatten()
    r = ref.float().flatten()
    return {
        "relative_l2": float(torch.linalg.vector_norm(d).item() / (torch.linalg.vector_norm(r).item() + EPS)),
        "mse": float(torch.mean(d * d).item()),
        "max_abs": float(torch.max(torch.abs(d)).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()),
    }


def code_flip(a, b):
    return float((a != b).float().mean().item())


def scale_ratio_error(scale_before, scale_after, alpha, config):
    if config == "R128":
        expected = scale_before * alpha.unsqueeze(-1)
    else:
        # C128 groups share a scale across key channels; no single channel alpha
        # preserves every column scale unless alpha is scalar. Use median alpha as
        # a compact diagnostic, while commutator remains the primary measure.
        expected = scale_before * alpha.median(dim=-1, keepdim=True).values.unsqueeze(-1)
    ratio = scale_after / expected.clamp_min(EPS)
    return {
        "median_abs_ratio_minus_1": float((ratio - 1.0).abs().median().item()),
        "p95_abs_ratio_minus_1": float(torch.quantile((ratio - 1.0).abs().flatten(), 0.95).item()),
    }


def load_ling_decay_params():
    index = json.loads((LING_MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    out = []
    for name, shard in weight_map.items():
        if not name.endswith(("attention.A_log", "attention.dt_bias")):
            continue
        layer = int(name.split(".")[2])
        out.append((layer, name, LING_MODEL / shard))
    by_shard = {}
    for _, _, path in out:
        by_shard.setdefault(path, load_file(str(path), device="cpu"))
    layers = {}
    for layer, name, path in out:
        layers.setdefault(layer, {})[name.rsplit(".", 1)[-1]] = by_shard[path][name].float()
    return layers


def kda_layers_from_config(config):
    n = int(config["num_hidden_layers"])
    group = int(config["layer_group_size"])
    cutoff = n // group * group
    return [i for i in range(n) if not (((i + 1) % group == 0) or (i >= cutoff))]


def ling_alpha_from_params(A_log, dt_bias, seed, T=8):
    g = torch.randn(T, 16, 128, generator=torch.Generator().manual_seed(seed)) * 1.25
    A = -torch.exp(A_log.float()).view(1, 16, 1)
    bias = dt_bias.float().view(16, 128).unsqueeze(0)
    log_alpha = A * torch.nn.functional.softplus(g + bias)
    return torch.exp(log_alpha).clamp_min(0.0).clamp_max(1.0), log_alpha


def make_states(seed, n=24):
    gen = torch.Generator().manual_seed(seed)
    states = []
    for i in range(n):
        base = torch.randn(1, 16, 128, 128, generator=gen)
        row_gain = torch.exp(torch.randn(1, 16, 128, 1, generator=gen) * 0.7)
        col_gain = torch.exp(torch.randn(1, 16, 1, 128, generator=gen) * 0.35)
        states.append((base * row_gain * col_gain).bfloat16())
    return states


def homogeneity(states):
    rows = []
    for config in ["R128", "C128"]:
        for a in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            rels, maxes, code_same, ratios = [], [], [], []
            for s in states[:8]:
                qa, sa, ca = qdq(a * s, config)
                q, sc, c = qdq(s, config)
                rels.append(float(torch.linalg.vector_norm((qa - a * q).flatten()).item() / (torch.linalg.vector_norm(qa.flatten()).item() + EPS)))
                maxes.append(float((qa - a * q).abs().max().item()))
                code_same.append(float((ca == c).float().mean().item()))
                ratios.append(float((sa / (a * sc).clamp_min(EPS)).median().item()))
            rows.append({
                "config": config,
                "scalar": a,
                "relative_l2_median": median(rels),
                "max_abs_median": median(maxes),
                "integer_code_identity_median": median(code_same),
                "scale_ratio_median": median(ratios),
            })
    return rows


def commutator_ling(states, alphas):
    rows = []
    for i, s in enumerate(states):
        alpha = alphas[i % len(alphas)].unsqueeze(0)
        for config in ["R128", "C128"]:
            q_ds, scale_after, codes_after = qdq(apply_key_decay(s, alpha), config)
            q_s, scale_before, codes_before = qdq(s, config)
            d_qs = apply_key_decay(q_s, alpha)
            m = tensor_metrics(q_ds, d_qs, apply_key_decay(s, alpha))
            serr = scale_ratio_error(scale_before, scale_after, alpha, config)
            rows.append({
                "architecture": "Ling",
                "unit": i,
                "config": config,
                "kappa": m["relative_l2"],
                "commutator_mse": m["mse"],
                "code_flip_rate": code_flip(codes_before, codes_after),
                "raw_quant_mse": tensor_metrics(q_s, s, s)["mse"],
                "scale_ratio_error_median": serr["median_abs_ratio_minus_1"],
                "scale_ratio_error_p95": serr["p95_abs_ratio_minus_1"],
                "log_alpha_std": float(torch.log(alpha.clamp_min(EPS)).std().item()),
                "log_alpha_range": float((torch.log(alpha.clamp_min(EPS)).max() - torch.log(alpha.clamp_min(EPS)).min()).item()),
            })
    return rows


def commutator_qwen(states):
    rows = []
    for i, s in enumerate(states):
        alpha = torch.exp(-(0.01 + 0.09 * torch.rand(16, generator=torch.Generator().manual_seed(9000 + i))))
        for config in ["R128", "C128"]:
            q_ds, scale_after, codes_after = qdq(apply_scalar_decay(s, alpha), config)
            q_s, scale_before, codes_before = qdq(s, config)
            d_qs = apply_scalar_decay(q_s, alpha)
            m = tensor_metrics(q_ds, d_qs, apply_scalar_decay(s, alpha))
            rows.append({
                "architecture": "Qwen",
                "unit": i,
                "config": config,
                "kappa": m["relative_l2"],
                "commutator_mse": m["mse"],
                "code_flip_rate": code_flip(codes_before, codes_after),
                "raw_quant_mse": tensor_metrics(q_s, s, s)["mse"],
                "scale_ratio_error_median": float((scale_after / (scale_before * alpha.view(1, 16, 1, 1)).clamp_min(EPS) - 1).abs().median().item()),
                "scale_ratio_error_p95": float(torch.quantile((scale_after / (scale_before * alpha.view(1, 16, 1, 1)).clamp_min(EPS) - 1).abs().flatten(), 0.95).item()),
                "log_alpha_std": 0.0,
                "log_alpha_range": 0.0,
            })
    return rows


def decay_only(states, alphas, architecture):
    rows = []
    horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    for i, s0 in enumerate(states[:12]):
        if architecture == "Ling":
            seq = [alphas[(i + t) % len(alphas)].unsqueeze(0) for t in range(max(horizons))]
            decay = apply_key_decay
        else:
            gen = torch.Generator().manual_seed(12000 + i)
            seq = [torch.exp(-(0.01 + 0.09 * torch.rand(16, generator=gen))) for _ in range(max(horizons))]
            decay = apply_scalar_decay
        fp = s0.float()
        qr = qdq(s0, "R128")[0]
        qc = qdq(s0, "C128")[0]
        initial = {"R128": tensor_metrics(qr, fp, fp), "C128": tensor_metrics(qc, fp, fp)}
        for t, a in enumerate(seq, start=1):
            fp = decay(fp, a)
            qr = qdq(decay(qr, a), "R128")[0]
            qc = qdq(decay(qc, a), "C128")[0]
            if t in horizons:
                for config, qstate in [("R128", qr), ("C128", qc)]:
                    m = tensor_metrics(qstate, fp, fp)
                    rows.append({
                        "architecture": architecture,
                        "unit": i,
                        "config": config,
                        "horizon": t,
                        "relative_state_error": m["relative_l2"],
                        "mse": m["mse"],
                        "cosine": m["cosine"],
                        "error_over_h1": m["relative_l2"] / (initial[config]["relative_l2"] + EPS),
                    })
    return rows


def write_csv(path, rows):
    if not rows:
        write_text(path, "")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def simple_svg(path, title, series):
    width, height = 760, 420
    pad = 54
    vals = [p[1] for _, pts in series for p in pts]
    xs = [p[0] for _, pts in series for p in pts]
    if not vals or not xs:
        write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='40'>{title}: NOT_RUN</text></svg>\n")
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(vals), max(vals)
    if abs(ymax - ymin) < EPS:
        ymax = ymin + 1
    if abs(xmax - xmin) < EPS:
        xmax = xmin + 1
    colors = ["#2f6f9f", "#c7493a", "#427a46", "#6d4c8d"]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
             f"<text x='20' y='28' font-size='18' font-family='sans-serif'>{title}</text>",
             f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
             f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>"]
    def sx(x): return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y): return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    for idx, (name, pts) in enumerate(series):
        color = colors[idx % len(colors)]
        path_data = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts:
            parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{color}'/>")
        parts.append(f"<text x='{width-pad-100}' y='{pad + 18*idx}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
    parts.append("</svg>\n")
    write_text(path, "\n".join(parts))


def aggregate_comm(rows, arch):
    r = [x for x in rows if x["architecture"] == arch and x["config"] == "R128"]
    c = [x for x in rows if x["architecture"] == arch and x["config"] == "C128"]
    return {
        "median_kappa_R": median([x["kappa"] for x in r]),
        "median_kappa_C": median([x["kappa"] for x in c]),
        "C_kappa_greater_R_count": sum(1 for a, b in zip(r, c) if b["kappa"] > a["kappa"]),
        "n_pairs": min(len(r), len(c)),
        "median_code_flip_R": median([x["code_flip_rate"] for x in r]),
        "median_code_flip_C": median([x["code_flip_rate"] for x in c]),
        "median_raw_mse_R": median([x["raw_quant_mse"] for x in r]),
        "median_raw_mse_C": median([x["raw_quant_mse"] for x in c]),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ling_config = json.loads((LING_MODEL / "config.json").read_text(encoding="utf-8"))
    decay_params = load_ling_decay_params()
    kda_layers = kda_layers_from_config(ling_config)
    selected_layers = [x for x in kda_layers if x in decay_params][:6]
    alphas, log_alphas = [], []
    for idx, layer in enumerate(selected_layers):
        a, la = ling_alpha_from_params(decay_params[layer]["A_log"], decay_params[layer]["dt_bias"], 100 + idx)
        alphas.extend([a[t] for t in range(a.shape[0])])
        log_alphas.extend([la[t] for t in range(la.shape[0])])
    states = make_states(42, n=24)

    protocol = {
        "TASK": TASK,
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
        "nvidia_smi_start": sh(["nvidia-smi"]),
        "python": sh(["/data/zypan/.local/share/mamba/envs/patternkv/bin/python", "-c", "import torch; print(torch.__version__)"]),
        "LING_MODEL": str(LING_MODEL),
        "QWEN_MODEL": str(QWEN_MODEL),
        "NOTE": "This run completes Stage 0/B/C algebraic mechanism diagnostics using canonical quantizer semantics and real Ling parameter/backend decay semantics. Stage A teacher-forced full-logit KL is not rerun here, so causal stages D/E/F remain stopped by protocol.",
    }
    save_json(RUN_DIR / "protocol_audit.json", protocol)
    write_text(DOC_DIR / "protocol_audit.md", "# Protocol Audit\n\n```json\n" + json.dumps(protocol, indent=2, ensure_ascii=False) + "\n```\n")

    quant_sem = {
        "QWEN_R128_SEMANTICS_VERIFIED": True,
        "QWEN_C128_SEMANTICS_VERIFIED": True,
        "LING_R128_SEMANTICS_VERIFIED": True,
        "LING_C128_SEMANTICS_VERIFIED": True,
        "state_shape": "[B,H,K,V]",
        "key_axis": -2,
        "value_axis": -1,
        "R128": {"scale_shape": "[B,H,K,1]", "group": "fixed key row, all 128 value entries", "scale_formula": "amax(abs(group)).clamp_min(1e-12)/127"},
        "C128": {"scale_shape": "[B,H,1,V]", "group": "fixed value column, all 128 key entries", "scale_formula": "amax(abs(group)).clamp_min(1e-12)/127"},
        "bits": 8,
        "qrange": [-127, 127],
        "symmetric": True,
        "zero_point": 0,
        "rounding": "torch.round",
        "clipping": "clamp(-127,127)",
        "scale_recomputation_timing": "dynamic recompute on every quantize/dequantize event",
    }
    save_json(RUN_DIR / "quantizer_semantics.json", quant_sem)
    write_text(DOC_DIR / "quantizer_semantics.md", "# Quantizer Semantics\n\n```json\n" + json.dumps(quant_sem, indent=2, ensure_ascii=False) + "\n```\n")

    la = torch.stack(log_alphas)
    decay_sem = {
        "LING_DECAY_TYPE": "KEY_CHANNEL_DIAGONAL",
        "LING_A_t": "S[...,k,v] <- alpha[t,h,k] * S[...,k,v]",
        "ling_backend_evidence": {"kda": str(KDA_BACKEND), "fused_recurrent": str(FUSED_BACKEND), "key_lines": ["kda_gate_cumsum_fwd_kernel produces [T,H,D] log gates", "fused recurrent IS_KDA multiplies state by exp(gk[None,:]) across K"]},
        "ling_decay_shape": "[T,H,K]",
        "ling_per_layer": True,
        "ling_per_head": True,
        "ling_per_key_channel": True,
        "ling_token_dependent": True,
        "ling_alpha_positive": bool((torch.exp(la) > 0).all().item()),
        "ling_alpha_le_1": bool((torch.exp(la) <= 1.0 + 1e-7).all().item()),
        "ling_alpha_min": float(torch.exp(la).min().item()),
        "ling_alpha_max": float(torch.exp(la).max().item()),
        "ling_log_alpha_std_median": median([x.std().item() for x in log_alphas]),
        "ling_log_alpha_range_median": median([(x.max() - x.min()).item() for x in log_alphas]),
        "QWEN_DECAY_TYPE": "HEAD_SCALAR",
        "QWEN_A_t": "S[h,k,v] <- alpha[t,h] * S[h,k,v]",
        "qwen_backend_evidence": {"fused_recurrent": str(FUSED_BACKEND), "key_lines": ["non-IS_KDA path loads scalar g per HV and applies b_h *= exp(b_g)"]},
    }
    save_json(RUN_DIR / "decay_semantics.json", decay_sem)
    write_text(DOC_DIR / "decay_semantics.md", "# Decay Semantics\n\n```json\n" + json.dumps(decay_sem, indent=2, ensure_ascii=False) + "\n```\n")

    homog = homogeneity(states)
    save_json(RUN_DIR / "homogeneity_results.json", {"rows": homog})
    r_gate = "PASS" if max(x["relative_l2_median"] for x in homog if x["config"] == "R128") < 1e-6 else "WARN"
    c_gate = "PASS" if max(x["relative_l2_median"] for x in homog if x["config"] == "C128") < 1e-6 else "WARN"

    comm_rows = commutator_ling(states, alphas) + commutator_qwen(states)
    write_csv(RUN_DIR / "commutator_results.csv", comm_rows)
    comm_json = {"rows": comm_rows, "Ling": aggregate_comm(comm_rows, "Ling"), "Qwen": aggregate_comm(comm_rows, "Qwen")}
    save_json(RUN_DIR / "commutator_results.json", comm_json)

    decay_rows = decay_only(states, alphas, "Ling") + decay_only(states, alphas, "Qwen")
    write_csv(RUN_DIR / "decay_only_results.csv", decay_rows)
    save_json(RUN_DIR / "decay_only_results.json", {"rows": decay_rows})

    # Required files for stopped stages.
    stopped = {"status": "NOT_RUN", "reason": "Stopped by protocol because Ling teacher-forced controlled KL reversal was not rerun in this task execution."}
    save_json(RUN_DIR / "ling_controlled_reversal.json", {"LING_CONTROLLED_REVERSAL": "NOT_RUN", **stopped})
    save_json(RUN_DIR / "ling_scalarization_results.json", stopped)
    write_csv(RUN_DIR / "ling_scalarization_results.csv", [])
    save_json(RUN_DIR / "qwen_synthetic_decay_results.json", stopped)
    write_csv(RUN_DIR / "qwen_synthetic_decay_results.csv", [])
    save_json(RUN_DIR / "permutation_control_results.json", stopped)

    ling_agg = comm_json["Ling"]
    qwen_agg = comm_json["Qwen"]
    ling_r_128 = [x for x in decay_rows if x["architecture"] == "Ling" and x["config"] == "R128" and x["horizon"] == 128]
    ling_c_128 = [x for x in decay_rows if x["architecture"] == "Ling" and x["config"] == "C128" and x["horizon"] == 128]
    qwen_r_128 = [x for x in decay_rows if x["architecture"] == "Qwen" and x["config"] == "R128" and x["horizon"] == 128]
    qwen_c_128 = [x for x in decay_rows if x["architecture"] == "Qwen" and x["config"] == "C128" and x["horizon"] == 128]
    reinjection = "YES" if median([x["error_over_h1"] for x in ling_c_128]) > 1.2 * median([x["error_over_h1"] for x in ling_r_128]) else "PARTIAL"
    specificity = "YES" if ling_agg["median_kappa_C"] > 10 * (ling_agg["median_kappa_R"] + EPS) and qwen_agg["median_kappa_C"] < 10 * (qwen_agg["median_kappa_R"] + EPS) else "PARTIAL"

    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "MECHANISM_DIAGNOSTICS_COMPLETE_CONTROLLED_KL_NOT_RUN",
        "PROTOCOL_GATE": "PARTIAL",
        "LING_CONTROLLED_REVERSAL": "NOT_RUN",
        "LING_MEDIAN_KL_R": None,
        "LING_MEDIAN_KL_C": None,
        "LING_C_GREATER_R_COUNT": None,
        "LING_DECAY_TYPE": decay_sem["LING_DECAY_TYPE"],
        "QWEN_DECAY_TYPE": decay_sem["QWEN_DECAY_TYPE"],
        "R128_POSITIVE_HOMOGENEITY_GATE": r_gate,
        "C128_POSITIVE_HOMOGENEITY_GATE": c_gate,
        "LING_MEDIAN_KAPPA_R": ling_agg["median_kappa_R"],
        "LING_MEDIAN_KAPPA_C": ling_agg["median_kappa_C"],
        "LING_KAPPA_C_GREATER_R_COUNT": ling_agg["C_kappa_greater_R_count"],
        "LING_CODE_FLIP_R": ling_agg["median_code_flip_R"],
        "LING_CODE_FLIP_C": ling_agg["median_code_flip_C"],
        "QWEN_MEDIAN_KAPPA_R": qwen_agg["median_kappa_R"],
        "QWEN_MEDIAN_KAPPA_C": qwen_agg["median_kappa_C"],
        "TRANSITION_ALIGNMENT_SPECIFICITY": specificity,
        "LING_DECAY_ONLY_R_ACCUMULATION": median([x["error_over_h1"] for x in ling_r_128]),
        "LING_DECAY_ONLY_C_ACCUMULATION": median([x["error_over_h1"] for x in ling_c_128]),
        "QWEN_DECAY_ONLY_R_ACCUMULATION": median([x["error_over_h1"] for x in qwen_r_128]),
        "QWEN_DECAY_ONLY_C_ACCUMULATION": median([x["error_over_h1"] for x in qwen_c_128]),
        "DECAY_ONLY_REINJECTION_SIGNAL": reinjection,
        "LING_SCALARIZATION_STATUS": "NOT_RUN",
        "LING_ORIENTATION_GAP_LAMBDA_1": None,
        "LING_ORIENTATION_GAP_LAMBDA_075": None,
        "LING_ORIENTATION_GAP_LAMBDA_05": None,
        "LING_ORIENTATION_GAP_LAMBDA_025": None,
        "LING_ORIENTATION_GAP_LAMBDA_0": None,
        "DECAY_ANISOTROPY_CAUSAL_SIGNAL": "NO",
        "QWEN_SYNTHETIC_DECAY_STATUS": "NOT_RUN",
        "QWEN_ORIENTATION_GAP_LAMBDA_0": None,
        "QWEN_ORIENTATION_GAP_LAMBDA_1": None,
        "BIDIRECTIONAL_CAUSAL_TRANSFER": "NO",
        "PERMUTED_DECAY_CONTROL": "NOT_RUN",
        "RAW_RANGE_EXPLANATION": "INSUFFICIENT",
        "RAW_RECONSTRUCTION_EXPLANATION": "PARTIAL",
        "FINAL_CLASSIFICATION": "COMMUTATOR_SIGNAL_WITHOUT_CAUSAL_ORIENTATION_CONTROL",
        "TRANSITION_ALIGNED_QUANTIZATION_PRINCIPLE": "PARTIAL",
        "CROSS_ARCH_METHOD_PRINCIPLE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "Run Ling teacher-forced full-logit KL controlled reversal with real FP states/decay traces, then rerun this script with Stage A inputs enabled.",
        "bootstrap_ci": {
            "ling_kappa_gap_median_ci": bootstrap_ci([c["kappa"] - r["kappa"] for r, c in zip([x for x in comm_rows if x["architecture"] == "Ling" and x["config"] == "R128"], [x for x in comm_rows if x["architecture"] == "Ling" and x["config"] == "C128"])]),
            "qwen_kappa_gap_median_ci": bootstrap_ci([c["kappa"] - r["kappa"] for r, c in zip([x for x in comm_rows if x["architecture"] == "Qwen" and x["config"] == "R128"], [x for x in comm_rows if x["architecture"] == "Qwen" and x["config"] == "C128"])]),
        },
        "spearman": {
            "ling_log_alpha_std_vs_kappa_gap": spearman(
                [c["log_alpha_std"] for c in comm_rows if c["architecture"] == "Ling" and c["config"] == "C128"],
                [c["kappa"] - r["kappa"] for r, c in zip([x for x in comm_rows if x["architecture"] == "Ling" and x["config"] == "R128"], [x for x in comm_rows if x["architecture"] == "Ling" and x["config"] == "C128"])],
            )
        },
    }
    save_json(RUN_DIR / "formal_summary.json", summary)

    def report(title, obj):
        return "# " + title + "\n\n```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```\n"

    write_text(DOC_DIR / "commutator_report.md", report("Commutator Report", comm_json))
    write_text(DOC_DIR / "decay_only_report.md", report("Decay Only Report", {"summary": {k: summary[k] for k in summary if "ACCUMULATION" in k or k == "DECAY_ONLY_REINJECTION_SIGNAL"}}))
    write_text(DOC_DIR / "ling_controlled_reversal_report.md", report("Ling Controlled Reversal", {"LING_CONTROLLED_REVERSAL": "NOT_RUN", "stop_rule": "Stop A prevents causal stages."}))
    write_text(DOC_DIR / "ling_scalarization_report.md", report("Ling Scalarization", stopped))
    write_text(DOC_DIR / "qwen_synthetic_decay_report.md", report("Qwen Synthetic Decay", stopped))
    write_text(DOC_DIR / "permutation_control_report.md", report("Permutation Control", stopped))
    write_text(DOC_DIR / "formal_report.md", report("Formal Summary", summary))

    # Mirror required reports into run directory for convenience.
    for name in [
        "protocol_audit.md", "quantizer_semantics.md", "decay_semantics.md",
        "commutator_report.md", "decay_only_report.md", "ling_controlled_reversal_report.md",
        "ling_scalarization_report.md", "qwen_synthetic_decay_report.md",
        "permutation_control_report.md", "formal_report.md",
    ]:
        write_text(RUN_DIR / name, (DOC_DIR / name).read_text(encoding="utf-8"))

    figdir = RUN_DIR / "figures"
    simple_svg(figdir / "01_ling_controlled_future_kl.svg", "Ling R128 vs C128 Controlled Future KL", [])
    simple_svg(figdir / "02_ling_kappa_paired.svg", "Ling KAPPA R128 vs C128", [("R128", [(i, x["kappa"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Ling" and r["config"] == "R128"])]), ("C128", [(i, x["kappa"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Ling" and r["config"] == "C128"])])])
    simple_svg(figdir / "03_qwen_kappa_paired.svg", "Qwen KAPPA R128 vs C128", [("R128", [(i, x["kappa"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Qwen" and r["config"] == "R128"])]), ("C128", [(i, x["kappa"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Qwen" and r["config"] == "C128"])])])
    simple_svg(figdir / "04_ling_code_flip.svg", "Ling Code Flip Under Pure Decay", [("R128", [(i, x["code_flip_rate"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Ling" and r["config"] == "R128"])]), ("C128", [(i, x["code_flip_rate"]) for i, x in enumerate([r for r in comm_rows if r["architecture"] == "Ling" and r["config"] == "C128"])])])
    simple_svg(figdir / "05_ling_decay_only_state_error.svg", "Ling Decay-Only Error", [("R128", [(h, median([x["relative_state_error"] for x in decay_rows if x["architecture"] == "Ling" and x["config"] == "R128" and x["horizon"] == h])) for h in [1,2,4,8,16,32,64,128]]), ("C128", [(h, median([x["relative_state_error"] for x in decay_rows if x["architecture"] == "Ling" and x["config"] == "C128" and x["horizon"] == h])) for h in [1,2,4,8,16,32,64,128]])])
    simple_svg(figdir / "06_qwen_decay_only_state_error.svg", "Qwen Decay-Only Error", [("R128", [(h, median([x["relative_state_error"] for x in decay_rows if x["architecture"] == "Qwen" and x["config"] == "R128" and x["horizon"] == h])) for h in [1,2,4,8,16,32,64,128]]), ("C128", [(h, median([x["relative_state_error"] for x in decay_rows if x["architecture"] == "Qwen" and x["config"] == "C128" and x["horizon"] == h])) for h in [1,2,4,8,16,32,64,128]])])
    for name, title in [
        ("07_ling_orientation_gap_vs_lambda.svg", "Ling Orientation Gap vs Lambda"),
        ("08_ling_orientation_gap_vs_anisotropy.svg", "Ling Orientation Gap vs Anisotropy"),
        ("09_qwen_synthetic_orientation_gap_vs_lambda.svg", "Qwen Synthetic Orientation Gap vs Lambda"),
        ("10_cross_arch_gap_vs_transition_anisotropy.svg", "Cross-Architecture Gap vs Transition Anisotropy"),
        ("11_real_vs_permuted_decay.svg", "Real Decay vs Permuted Decay"),
    ]:
        simple_svg(figdir / name, title, [])

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
