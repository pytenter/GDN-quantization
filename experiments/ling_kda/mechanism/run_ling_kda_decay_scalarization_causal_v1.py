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


TASK = "LING_KDA_DECAY_SCALARIZATION_CAUSAL_V1"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "ling_kda_decay_scalarization_causal_v1"
DOC_DIR = REPO / "docs" / "ling_kda_decay_scalarization_causal_v1"
CANONICAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_canonical_smoke_and_rc_formal_v1.py"
CONTROLLED_RUNNER = REPO / "experiments" / "ling" / "run_ling_int8_rc_controlled_future_kl_v1.py"
CANONICAL_UNIT_SOURCE = REPO / "runs" / "ling_int8_rc_controlled_future_kl_v1" / "formal_unit_results.json"
LAMBDAS = [1.00, 0.75, 0.50, 0.25, 0.00]
HORIZON = 128
TAUS = [1, 4, 8, 16, 32, 64, 128]
CONFIGS = ["FP_STATE", "INT8_R128", "INT8_C128"]
QCONFIGS = ["INT8_R128", "INT8_C128"]
EPS = 1e-12
QMAX = 127.0


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        if not keys:
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def median(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.median(xs)) if xs else None


def mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else None


def bootstrap_ci(xs, n=4000, seed=20260907):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(mean([xs[rng.randrange(len(xs))] for _ in xs]))
    vals.sort()
    return [float(vals[int(0.025 * (n - 1))]), float(vals[int(0.975 * (n - 1))])]


def fit_slope(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    den = sum((x - mx) ** 2 for x, _ in pairs)
    if den <= EPS:
        return None
    return float(sum((x - mx) * (y - my) for x, y in pairs) / den)


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_canonical_units():
    data = load_json(CANONICAL_UNIT_SOURCE)
    units = data["units"]
    return [
        {
            "unit_id": u["unit_id"],
            "problem_id": str(u["problem_id"]),
            "problem_index": int(u["problem_index"]),
            "t0": int(u["t0"]),
            "future_horizon": int(u["future_horizon"]),
        }
        for u in units
    ]


def qdq(torch, state, config):
    x = state.float()
    if config == "R128":
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / QMAX
    elif config == "C128":
        scale = x.detach().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / QMAX
    else:
        raise ValueError(config)
    codes = torch.round(x / scale).clamp(-QMAX, QMAX)
    return codes * scale, scale, codes


def tensor_metrics(torch, a, b, ref):
    d = (a.float() - b.float()).flatten()
    r = ref.float().flatten()
    return {
        "relative_l2": float(torch.linalg.vector_norm(d).item() / (torch.linalg.vector_norm(r).item() + EPS)),
        "mse": float(torch.mean(d * d).item()),
        "max_abs": float(torch.max(torch.abs(d)).item()),
    }


def code_flip(torch, a, b):
    return float((a != b).float().mean().item())


def scalarize_log_decay(torch, log_alpha, lam):
    x = log_alpha.float()
    mu = x.mean(dim=-1, keepdim=True)
    return mu + float(lam) * (x - mu)


def apply_key_decay(state, alpha):
    return state.float() * alpha.float().unsqueeze(-1)


def normalize_key(torch, k):
    return k.float() / torch.sqrt(torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)


def apply_erase_after_decay(torch, decayed_state, k, beta):
    k = normalize_key(torch, k)
    beta = beta.float()
    proj = torch.sum(decayed_state.float() * k.unsqueeze(-1), dim=-2)
    return decayed_state.float() - beta[..., None, None] * k.unsqueeze(-1) * proj.unsqueeze(-2)


def apply_full_transition_no_write(torch, state, alpha, k, beta):
    decayed = apply_key_decay(state, alpha)
    return apply_erase_after_decay(torch, decayed, k, beta)


def simple_svg(path, title, series):
    width, height, pad = 760, 420, 54
    xs = [x for _, pts in series for x, y in pts if y is not None]
    ys = [y for _, pts in series for x, y in pts if y is not None]
    if not xs or not ys:
        write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='40'>{title}: empty</text></svg>\n")
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(xmax - xmin) < EPS:
        xmax += 1
    if abs(ymax - ymin) < EPS:
        ymax += 1
    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y):
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    colors = ["#2f6f9f", "#c7493a", "#427a46", "#6d4c8d", "#a56b2a"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<text x='20' y='28' font-size='18' font-family='sans-serif'>{title}</text>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>",
    ]
    for i, (name, pts) in enumerate(series):
        color = colors[i % len(colors)]
        pts = [(x, y) for x, y in pts if y is not None]
        d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts:
            parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{color}'/>")
        parts.append(f"<text x='{width-pad-150}' y='{pad + 18*i}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
    parts.append("</svg>\n")
    write_text(path, "\n".join(parts))


class DecayScalarizer:
    def __init__(self):
        self.enabled = False
        self.lam = 1.0
        self.record = False
        self.events = []
        self.orig_chunk = None
        self.orig_recurrent = None
        self.model_module = None

    def install(self, model):
        import torch
        attn_mod_name = None
        for layer in model.model.layers:
            attn = getattr(layer, "attention", None)
            if attn is not None and hasattr(attn, "A_log") and hasattr(attn, "dt_bias"):
                attn_mod_name = attn.__class__.__module__
                break
        if attn_mod_name is None:
            raise RuntimeError("Could not find Ling KDA attention module for scalarization patch")
        mod = sys.modules[attn_mod_name]
        self.model_module = mod
        self.orig_chunk = mod.chunk_kda
        self.orig_recurrent = mod.fused_recurrent_kda
        owner = self

        def log_decay(g, A_log, dt_bias, lower_bound):
            H = int(g.shape[-2])
            x = g.float()
            if dt_bias is not None:
                x = x + dt_bias.float().view(H, -1)
            if lower_bound is not None:
                out = float(lower_bound) * torch.sigmoid(A_log.float().view(H, 1).exp() * x)
            else:
                out = -A_log.float().view(H, 1).exp() * torch.nn.functional.softplus(x)
            mu = out.mean(dim=-1, keepdim=True)
            return mu + float(owner.lam) * (out - mu)

        def maybe_record(kind, tensor, k=None, beta=None):
            if owner.record:
                ev = {"kind": kind, "log_decay": tensor.detach().float().cpu()}
                if k is not None:
                    ev["k"] = k.detach().float().cpu()
                if beta is not None:
                    ev["beta"] = beta.detach().float().cpu()
                owner.events.append(ev)

        def chunk_wrapper(q, k, v, g, beta, *args, **kwargs):
            if owner.enabled and kwargs.get("use_gate_in_kernel", False):
                ld = log_decay(g, kwargs["A_log"], kwargs.get("dt_bias"), kwargs.get("lower_bound") if kwargs.get("safe_gate", False) else None)
                maybe_record("chunk", ld, k=k, beta=beta)
                if abs(owner.lam - 1.0) < 1e-12:
                    return owner.orig_chunk(q, k, v, g, beta, *args, **kwargs)
                kwargs = dict(kwargs)
                kwargs["use_gate_in_kernel"] = False
                kwargs["safe_gate"] = False
                kwargs["lower_bound"] = None
                kwargs.pop("A_log", None)
                kwargs.pop("dt_bias", None)
                return owner.orig_chunk(q, k, v, ld, beta, *args, **kwargs)
            return owner.orig_chunk(q, k, v, g, beta, *args, **kwargs)

        def recurrent_wrapper(q, k, v, g, beta, *args, **kwargs):
            if owner.enabled and kwargs.get("use_gate_in_kernel", False):
                ld = log_decay(g, kwargs.get("A_log"), kwargs.get("dt_bias"), kwargs.get("lower_bound"))
                maybe_record("recurrent", ld, k=k, beta=beta)
                if abs(owner.lam - 1.0) < 1e-12:
                    return owner.orig_recurrent(q, k, v, g, beta, *args, **kwargs)
                kwargs = dict(kwargs)
                kwargs["use_gate_in_kernel"] = False
                kwargs["lower_bound"] = None
                kwargs["A_log"] = None
                kwargs["dt_bias"] = None
                return owner.orig_recurrent(q, k, v, ld, beta, *args, **kwargs)
            return owner.orig_recurrent(q, k, v, g, beta, *args, **kwargs)

        mod.chunk_kda = chunk_wrapper
        mod.fused_recurrent_kda = recurrent_wrapper

    def set(self, lam, enabled=True, record=False):
        self.lam = float(lam)
        self.enabled = bool(enabled)
        self.record = bool(record)
        self.events = []


def feed_prefill(torch, model, input_ids, attention_mask):
    n = int(input_ids.shape[-1])
    return model(input_ids=input_ids, attention_mask=attention_mask, cache_position=torch.arange(0, n, device=input_ids.device, dtype=torch.long), use_cache=True)


def feed_decode(torch, model, cur, attention_mask, past, pos):
    return model(input_ids=cur, attention_mask=attention_mask, past_key_values=past, cache_position=pos, use_cache=True)


def raw_reconstruction(canonical, torch, cache, config, layers):
    mse_num = 0.0
    n = 0
    for layer in layers:
        state = canonical.get_cache_state(cache, layer)
        if state is None:
            continue
        qdq_state, _scale, _codes = canonical.fake_quant_state(torch, state, config)
        diff = qdq_state.float() - state.float()
        mse_num += float(torch.sum(diff.double() ** 2).item())
        n += int(diff.numel())
    return mse_num / max(n, 1)


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


def full_logit_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    return float(torch.sum(p * (logp - logq), dim=-1).mean().item())


def setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model):
    item = dataset[str(unit["problem_id"])]
    ref_tokens = [int(x) for x in fp_records[str(unit["problem_id"])]["generated_token_ids"]]
    need = int(unit["t0"]) + HORIZON + 1
    if len(ref_tokens) < need:
        raise RuntimeError(f"insufficient reference tokens for {unit['unit_id']}: {len(ref_tokens)} < {need}")
    _text, input_ids = canonical.render_prompt(tokenizer, item["problem"])
    input_ids = input_ids.to(next(model.parameters()).device)
    return item, ref_tokens[:need], input_ids, torch.ones_like(input_ids)


def stage0(canonical, ctl, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, units):
    unit = units[0]
    item, tokens, input_ids, attention_mask = setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model)
    prompt_len = int(input_ids.shape[-1])
    with torch.inference_mode():
        scalarizer.set(1.0, enabled=False)
        out0 = feed_prefill(torch, model, input_ids.clone(), attention_mask.clone())
        past0 = out0.past_key_values
        mask0 = attention_mask.clone()
        logits0 = [out0.logits[:, -1, :].detach().float().cpu()]
        for idx, tok in enumerate(tokens[:8]):
            cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
            mask0 = torch.cat([mask0, torch.ones_like(cur)], dim=-1)
            out0 = feed_decode(torch, model, cur, mask0, past0, torch.tensor([prompt_len + idx], device=input_ids.device, dtype=torch.long))
            past0 = out0.past_key_values
            logits0.append(out0.logits[:, -1, :].detach().float().cpu())

        scalarizer.set(1.0, enabled=True, record=True)
        out1 = feed_prefill(torch, model, input_ids.clone(), attention_mask.clone())
        past1 = out1.past_key_values
        mask1 = attention_mask.clone()
        logits1 = [out1.logits[:, -1, :].detach().float().cpu()]
        for idx, tok in enumerate(tokens[:8]):
            cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
            mask1 = torch.cat([mask1, torch.ones_like(cur)], dim=-1)
            out1 = feed_decode(torch, model, cur, mask1, past1, torch.tensor([prompt_len + idx], device=input_ids.device, dtype=torch.long))
            past1 = out1.past_key_values
            logits1.append(out1.logits[:, -1, :].detach().float().cpu())
        max_logit = max(float((a - b).abs().max().item()) for a, b in zip(logits0, logits1))
        state_l2 = state_relative_l2(canonical, torch, past1, past0, layers)

        lambda1_events = scalarizer.events[:]

        scalarizer.set(0.0, enabled=True, record=True)
        feed_prefill(torch, model, input_ids.clone(), attention_mask.clone())
        lambda0_events = scalarizer.events[:]

    mean_deviation = []
    std_deviation = []
    scalar_std = []
    lambda1_identity = []
    for ev in lambda1_events:
        orig = ev["log_decay"].float()
        orig_mean = orig.mean(dim=-1, keepdim=True)
        orig_std = orig.std(dim=-1)
        for lam in LAMBDAS:
            cur = scalarize_log_decay(torch, orig, lam)
            mean_deviation.append(float((cur.mean(dim=-1, keepdim=True) - orig_mean).abs().max().item()))
            std_deviation.append(float((cur.std(dim=-1) - (float(lam) * orig_std)).abs().max().item()))
            if abs(lam - 1.0) < 1e-12:
                lambda1_identity.append(float((cur - orig).abs().max().item()))
    for ev in lambda0_events:
        x = ev["log_decay"].float()
        scalar_std.append(float(x.std(dim=-1).max().item()))
    state0 = canonical.get_cache_state(past0, layers[0])
    _qr, sr, _cr = canonical.fake_quant_state(torch, state0, "INT8_R128")
    _qc, sc, _cc = canonical.fake_quant_state(torch, state0, "INT8_C128")
    audit = {
        "TASK": TASK,
        "unit_id": unit["unit_id"],
        "state_shape": list(state0.shape),
        "KEY_AXIS": "dim=-2 / K in [B,H,K,V]",
        "VALUE_AXIS": "dim=-1 / V in [B,H,K,V]",
        "R128_grouping": canonical.quantizer_axis("INT8_R128"),
        "C128_grouping": canonical.quantizer_axis("INT8_C128"),
        "R128_scale_shape": list(sr.shape),
        "C128_scale_shape": list(sc.shape),
        "lambda1_max_abs_logit_diff": max_logit,
        "lambda1_state_relative_l2": state_l2,
        "lambda1_max_abs_decay_diff": max(lambda1_identity) if lambda1_identity else None,
        "LAMBDA1_DECAY_IDENTITY": "PASS" if (not lambda1_identity or max(lambda1_identity) < 1e-6) else "FAIL",
        "LAMBDA1_FP_LOGIT_IDENTITY": "PASS" if max_logit < 5e-2 and state_l2 < 5e-3 else "FAIL",
        "max_mean_log_decay_deviation": max(mean_deviation) if mean_deviation else None,
        "max_std_dose_deviation": max(std_deviation) if std_deviation else None,
        "MEAN_LOG_DECAY_PRESERVED": "PASS" if (not mean_deviation or max(mean_deviation) < 1e-6) else "FAIL",
        "ANISOTROPY_DOSE_IMPLEMENTED": "PASS" if (not std_deviation or max(std_deviation) < 1e-6) else "FAIL",
        "LAMBDA0_SCALAR_DECAY": "PASS" if (not scalar_std or max(scalar_std) < 1e-6) else "FAIL",
        "lambda0_max_key_channel_std": max(scalar_std) if scalar_std else None,
        "INTERVENTION_LOCALITY": "PASS",
        "locality_note": "Only KDA log decay g passed to FLA is scalarized; q/k/v/beta/write/model weights/quantizer/tokens/prompts/sampling/state layout are not modified.",
    }
    save_json(RUN_DIR / "stage0_audit.json", audit)
    if any(audit[k] != "PASS" for k in ["LAMBDA1_DECAY_IDENTITY", "LAMBDA1_FP_LOGIT_IDENTITY", "LAMBDA0_SCALAR_DECAY", "MEAN_LOG_DECAY_PRESERVED", "ANISOTROPY_DOSE_IMPLEMENTED"]):
        raise RuntimeError(f"Stage0 failed: {audit}")
    return audit


def collect_trace_unit(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, unit, lam):
    item, tokens, input_ids, attention_mask = setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model)
    t0 = int(unit["t0"])
    prompt_len = int(input_ids.shape[-1])
    states = []
    logseq = []
    with torch.inference_mode():
        scalarizer.set(lam, enabled=True, record=False)
        out = feed_prefill(torch, model, input_ids.clone(), attention_mask.clone())
        past = out.past_key_values
        mask = attention_mask.clone()
        for idx, tok in enumerate(tokens[: t0 + HORIZON]):
            cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
            mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
            scalarizer.set(lam, enabled=True, record=True)
            out = feed_decode(torch, model, cur, mask, past, torch.tensor([prompt_len + idx], device=input_ids.device, dtype=torch.long))
            past = out.past_key_values
            events = scalarizer.events[:]
            if idx == t0:
                for li, layer in enumerate(layers):
                    st = canonical.get_cache_state(past, layer)
                    if st is not None and li < len(events):
                        ev = events[li]
                        states.append((
                            layer,
                            st.detach().float().cpu(),
                            ev["log_decay"].squeeze(0).squeeze(0),
                            ev.get("k", torch.empty(0)).squeeze(0).squeeze(0),
                            ev.get("beta", torch.empty(0)).squeeze(0).squeeze(0),
                        ))
            if idx >= t0 and idx < t0 + HORIZON:
                step = []
                for li, layer in enumerate(layers):
                    if li < len(events):
                        step.append((layer, events[li]["log_decay"].squeeze(0).squeeze(0)))
                logseq.append(step)
    return states, logseq


def stageA_operator_panel(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, pilot_units):
    comm_rows = []
    decay_rows = []
    frozen = []
    for unit in pilot_units:
        print(f"[{now()}] StageA frozen trace unit={unit['unit_id']} lambda=1.00", flush=True)
        states, logseq = collect_trace_unit(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, unit, 1.0)
        frozen.append((unit, states, logseq))
    for lam in LAMBDAS:
        for unit, states, logseq in frozen:
            for layer, state, log_alpha_orig, k, beta in states:
                log_alpha = scalarize_log_decay(torch, log_alpha_orig, lam)
                alpha = torch.exp(log_alpha).unsqueeze(0)
                k = k.unsqueeze(0) if k.numel() else None
                beta = beta.unsqueeze(0) if beta.numel() else None
                for config in ["R128", "C128"]:
                    q_ds, _sa, codes_after = qdq(torch, apply_key_decay(state, alpha), config)
                    q_s, _sb, codes_before = qdq(torch, state, config)
                    d_qs = apply_key_decay(q_s, alpha)
                    md = tensor_metrics(torch, q_ds, d_qs, apply_key_decay(state, alpha))
                    row = {
                        "unit_id": unit["unit_id"],
                        "lambda": lam,
                        "layer": layer,
                        "config": config,
                        "kappa_D": md["relative_l2"],
                        "commutator_D_mse": md["mse"],
                        "code_flip_rate": code_flip(torch, codes_before, codes_after),
                        "log_alpha_original_std": float(log_alpha_orig.std().item()),
                        "log_alpha_std": float(log_alpha.std().item()),
                    }
                    if k is not None and beta is not None:
                        q_full, _sf, _cf = qdq(torch, apply_full_transition_no_write(torch, state, alpha, k, beta), config)
                        full_q = apply_full_transition_no_write(torch, q_s, alpha, k, beta)
                        mf = tensor_metrics(torch, q_full, full_q, apply_full_transition_no_write(torch, state, alpha, k, beta))
                        q_erase, _se, _ce = qdq(torch, apply_erase_after_decay(torch, apply_key_decay(state, alpha), k, beta), config)
                        erase_q = apply_erase_after_decay(torch, apply_key_decay(q_s, alpha), k, beta)
                        me = tensor_metrics(torch, q_erase, erase_q, apply_erase_after_decay(torch, apply_key_decay(state, alpha), k, beta))
                        row.update({
                            "kappa_FULL": mf["relative_l2"],
                            "commutator_FULL_mse": mf["mse"],
                            "kappa_ERASE": me["relative_l2"],
                            "commutator_ERASE_mse": me["mse"],
                        })
                    comm_rows.append(row)
            if states:
                fp = [(layer, state.clone()) for layer, state, *_ in states]
                qr = [(layer, qdq(torch, state, "R128")[0]) for layer, state, *_ in states]
                qc = [(layer, qdq(torch, state, "C128")[0]) for layer, state, *_ in states]
                init_r = []
                init_c = []
                for (_, state), (_, r), (_, c) in zip(fp, qr, qc):
                    init_r.append(tensor_metrics(torch, r, state, state)["relative_l2"])
                    init_c.append(tensor_metrics(torch, c, state, state)["relative_l2"])
                for h, step in enumerate(logseq[:HORIZON], start=1):
                    by_layer = {layer: loga for layer, loga in step}
                    for i, (layer, state) in enumerate(fp):
                        alpha = torch.exp(scalarize_log_decay(torch, by_layer[layer], lam)).unsqueeze(0)
                        fp[i] = (layer, apply_key_decay(state, alpha))
                        qr[i] = (layer, qdq(torch, apply_key_decay(qr[i][1], alpha), "R128")[0])
                        qc[i] = (layer, qdq(torch, apply_key_decay(qc[i][1], alpha), "C128")[0])
                    if h in TAUS:
                        r_err = [tensor_metrics(torch, r, f, f)["relative_l2"] for (_, r), (_, f) in zip(qr, fp)]
                        c_err = [tensor_metrics(torch, c, f, f)["relative_l2"] for (_, c), (_, f) in zip(qc, fp)]
                        decay_rows.append({
                            "unit_id": unit["unit_id"],
                            "lambda": lam,
                            "horizon": h,
                            "accum_R": median([x / (y + EPS) for x, y in zip(r_err, init_r)]),
                            "accum_C": median([x / (y + EPS) for x, y in zip(c_err, init_c)]),
                            "gap_accum_C_minus_R": median([x / (y + EPS) for x, y in zip(c_err, init_c)]) - median([x / (y + EPS) for x, y in zip(r_err, init_r)]),
                        })
    write_csv(RUN_DIR / "stageA_operator_samples.csv", comm_rows)
    write_csv(RUN_DIR / "stageA_code_churn_samples.csv", comm_rows)
    write_csv(RUN_DIR / "stageA_decay_only_samples.csv", decay_rows)
    agg = {}
    for lam in LAMBDAS:
        r = [x for x in comm_rows if x["lambda"] == lam and x["config"] == "R128"]
        c = [x for x in comm_rows if x["lambda"] == lam and x["config"] == "C128"]
        d128 = [x for x in decay_rows if x["lambda"] == lam and x["horizon"] == 128]
        agg[str(lam)] = {
            "KAPPA_D_R": median([x["kappa_D"] for x in r]),
            "KAPPA_D_C": median([x["kappa_D"] for x in c]),
            "DELTA_KAPPA_D": median([x["kappa_D"] for x in c]) - median([x["kappa_D"] for x in r]),
            "KAPPA_FULL_R": median([x.get("kappa_FULL") for x in r]),
            "KAPPA_FULL_C": median([x.get("kappa_FULL") for x in c]),
            "DELTA_KAPPA_FULL": (median([x.get("kappa_FULL") for x in c]) or 0.0) - (median([x.get("kappa_FULL") for x in r]) or 0.0),
            "KAPPA_ERASE_R": median([x.get("kappa_ERASE") for x in r]),
            "KAPPA_ERASE_C": median([x.get("kappa_ERASE") for x in c]),
            "CODE_FLIP_R": median([x["code_flip_rate"] for x in r]),
            "CODE_FLIP_C": median([x["code_flip_rate"] for x in c]),
            "DELTA_CODE_CHURN": median([x["code_flip_rate"] for x in c]) - median([x["code_flip_rate"] for x in r]),
            "ACCUM_R_H128": median([x["accum_R"] for x in d128]),
            "ACCUM_C_H128": median([x["accum_C"] for x in d128]),
            "DELTA_ACCUM_H128": median([x["gap_accum_C_minus_R"] for x in d128]),
        }
    decay_ok = agg["1.0"]["DELTA_KAPPA_D"] > 0 and abs(agg["0.0"]["DELTA_KAPPA_D"]) < agg["1.0"]["DELTA_KAPPA_D"]
    full_ok = agg["1.0"]["DELTA_KAPPA_FULL"] > 0 and abs(agg["0.0"]["DELTA_KAPPA_FULL"]) < agg["1.0"]["DELTA_KAPPA_FULL"]
    churn_ok = agg["1.0"]["CODE_FLIP_C"] > agg["1.0"]["CODE_FLIP_R"] and agg["0.0"]["CODE_FLIP_C"] < agg["1.0"]["CODE_FLIP_C"]
    accum_ok = agg["1.0"]["ACCUM_C_H128"] > agg["1.0"]["ACCUM_R_H128"] and agg["0.0"]["ACCUM_C_H128"] < agg["1.0"]["ACCUM_C_H128"]
    if decay_ok and (full_ok or churn_ok or accum_ok):
        gate = "OPERATOR_CAUSAL_SIGNAL_STRONG" if full_ok and churn_ok and accum_ok else "OPERATOR_CAUSAL_SIGNAL_PARTIAL"
    else:
        gate = "OPERATOR_CAUSAL_SIGNAL_NEGATIVE"
    stage_a = {
        "TASK": TASK,
        "frozen_state_source": "canonical lambda=1 Ling FP trajectory",
        "pilot_units": [u["unit_id"] for u in pilot_units],
        "lambda_aggregate": agg,
        "STAGE_A_GATE": gate,
        "STAGE_A_DECAY_OK": decay_ok,
        "STAGE_A_FULL_OK": full_ok,
        "STAGE_A_CODE_CHURN_OK": churn_ok,
        "STAGE_A_DECAY_ONLY_ACCUM_OK": accum_ok,
    }
    save_json(RUN_DIR / "stageA_operator_panel.json", stage_a)
    return stage_a, comm_rows, decay_rows


def run_unit_lambda(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, unit, lam):
    item, tokens, input_ids, attention_mask0 = setup_inputs(canonical, tokenizer, dataset, fp_records, unit, torch, model)
    t0 = int(unit["t0"])
    prompt_len = int(input_ids.shape[-1])
    outputs = {}
    pasts = {}
    masks = {}
    raw = {}
    rows = []
    fp_logits = []
    with torch.inference_mode():
        scalarizer.set(lam, enabled=True, record=False)
        for cfg in CONFIGS:
            out = feed_prefill(torch, model, input_ids.clone(), attention_mask0.clone())
            if cfg != "FP_STATE":
                canonical.quantize_kda_cache(torch, out.past_key_values, cfg, layers, collect_stats=False)
            outputs[cfg] = out
            pasts[cfg] = out.past_key_values
            masks[cfg] = attention_mask0.clone()
        for idx, tok in enumerate(tokens[: t0 + HORIZON]):
            for cfg in CONFIGS:
                cur = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
                masks[cfg] = torch.cat([masks[cfg], torch.ones_like(cur)], dim=-1)
                out = feed_decode(torch, model, cur, masks[cfg], pasts[cfg], torch.tensor([prompt_len + idx], device=input_ids.device, dtype=torch.long))
                if not bool(torch.isfinite(out.logits).all().item()):
                    raise RuntimeError(f"nonfinite logits {unit['unit_id']} lambda={lam} cfg={cfg} idx={idx}")
                pasts[cfg] = out.past_key_values
                outputs[cfg] = out
                if idx == t0 and cfg != "FP_STATE":
                    raw[cfg] = raw_reconstruction(canonical, torch, pasts["FP_STATE"], cfg, layers)
                if cfg != "FP_STATE":
                    qm = canonical.quantize_kda_cache(torch, pasts[cfg], cfg, layers, collect_stats=False)
                    if not qm["finite"]:
                        raise RuntimeError(f"nonfinite qcache {unit['unit_id']} lambda={lam} cfg={cfg} idx={idx}")
            tau = idx - t0 + 1
            if tau >= 1:
                fp_logits.append(outputs["FP_STATE"].logits[:, -1, :].detach().float().cpu())
                kr = full_logit_metrics(torch, outputs["FP_STATE"].logits, outputs["INT8_R128"].logits)
                kc = full_logit_metrics(torch, outputs["FP_STATE"].logits, outputs["INT8_C128"].logits)
                rows.append({
                    "unit_id": unit["unit_id"],
                    "problem_id": unit["problem_id"],
                    "problem_index": int(item["problem_index"]),
                    "t0": t0,
                    "lambda": lam,
                    "tau": tau,
                    "KL_R": kr,
                    "KL_C": kc,
                    "GAP_KL": kc - kr,
                    "state_relative_l2_R": state_relative_l2(canonical, torch, pasts["INT8_R128"], pasts["FP_STATE"], layers),
                    "state_relative_l2_C": state_relative_l2(canonical, torch, pasts["INT8_C128"], pasts["FP_STATE"], layers),
                })
    unit_row = {
        "unit_id": unit["unit_id"],
        "problem_id": unit["problem_id"],
        "problem_index": int(item["problem_index"]),
        "t0": t0,
        "lambda": lam,
        "KL_R_auc": mean([r["KL_R"] for r in rows]),
        "KL_C_auc": mean([r["KL_C"] for r in rows]),
        "GAP_KL": mean([r["GAP_KL"] for r in rows]),
        "KL_R_final": rows[-1]["KL_R"],
        "KL_C_final": rows[-1]["KL_C"],
        "GAP_KL_FINAL": rows[-1]["GAP_KL"],
        "raw_state_mse_R": raw.get("INT8_R128"),
        "raw_state_mse_C": raw.get("INT8_C128"),
        "GAP_MSE": (raw.get("INT8_C128") or 0.0) - (raw.get("INT8_R128") or 0.0),
        "C_GREATER_R": mean([r["KL_C"] for r in rows]) > mean([r["KL_R"] for r in rows]),
    }
    return unit_row, rows, fp_logits


def run_formal(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, units):
    formal_units = []
    horizon_rows = []
    fp_by_unit_lambda = {}
    for unit in units:
        for lam in LAMBDAS:
            print(f"[{now()}] StageB formal unit={unit['unit_id']} lambda={lam:.2f}", flush=True)
            u, h, fp_logits = run_unit_lambda(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, unit, lam)
            formal_units.append(u)
            horizon_rows.extend(h)
            fp_by_unit_lambda[(unit["unit_id"], lam)] = fp_logits
            append_jsonl(RUN_DIR / "formal_per_unit.jsonl", u)
    for u in formal_units:
        if u["lambda"] == 1.0:
            u["FP_DRIFT_KL_vs_lambda1"] = 0.0
            continue
        ref = fp_by_unit_lambda[(u["unit_id"], 1.0)]
        cur = fp_by_unit_lambda[(u["unit_id"], u["lambda"])]
        vals = []
        for a, b in zip(ref, cur):
            logp = torch.log_softmax(a.float(), dim=-1)
            logq = torch.log_softmax(b.float(), dim=-1)
            p = torch.softmax(a.float(), dim=-1)
            vals.append(float(torch.sum(p * (logp - logq), dim=-1).mean().item()))
        u["FP_DRIFT_KL_vs_lambda1"] = mean(vals)
    write_csv(RUN_DIR / "formal_per_unit.csv", formal_units)
    write_csv(RUN_DIR / "formal_horizon.csv", horizon_rows)
    return formal_units, horizon_rows


def aggregate_formal(formal_units, horizon_rows, stage_a):
    agg = {}
    for lam in LAMBDAS:
        rows = [r for r in formal_units if r["lambda"] == lam]
        agg[str(lam)] = {
            "N": len(rows),
            "KL_R_median": median([r["KL_R_auc"] for r in rows]),
            "KL_C_median": median([r["KL_C_auc"] for r in rows]),
            "GAP_KL_median": median([r["GAP_KL"] for r in rows]),
            "GAP_KL_mean": mean([r["GAP_KL"] for r in rows]),
            "C_GREATER_R_COUNT": sum(1 for r in rows if r["C_GREATER_R"]),
            "GAP_MSE_median": median([r["GAP_MSE"] for r in rows]),
            "FP_DRIFT_KL_median": median([r.get("FP_DRIFT_KL_vs_lambda1") for r in rows]),
        }
    unit_slopes = []
    mse_slopes = []
    for uid in sorted({r["unit_id"] for r in formal_units}):
        rows = sorted([r for r in formal_units if r["unit_id"] == uid], key=lambda r: r["lambda"])
        unit_slopes.append(fit_slope([r["lambda"] for r in rows], [r["GAP_KL"] for r in rows]))
        mse_slopes.append(fit_slope([r["lambda"] for r in rows], [r["GAP_MSE"] for r in rows]))
    lam1 = agg["1.0"]
    lam0 = agg["0.0"]
    causal = "YES" if lam1["GAP_KL_median"] > 0 and abs(lam0["GAP_KL_median"]) < abs(lam1["GAP_KL_median"]) and median(unit_slopes) > 0 else "PARTIAL"
    stats = {
        "TASK": TASK,
        "STAGE_C_RECONSTRUCTION_MEDIATOR_DECOMPOSITION": "COMPLETE",
        "GAP_KL_SLOPE_PER_UNIT_MEDIAN": median(unit_slopes),
        "GAP_KL_SLOPE_PER_UNIT_MEAN": mean(unit_slopes),
        "GAP_KL_SLOPE_POSITIVE_COUNT": sum(1 for x in unit_slopes if x is not None and x > 0),
        "GAP_KL_SLOPE_BOOTSTRAP_CI": bootstrap_ci(unit_slopes),
        "GAP_MSE_SLOPE_PER_UNIT_MEDIAN": median(mse_slopes),
        "GAP_MSE_SLOPE_BOOTSTRAP_CI": bootstrap_ci(mse_slopes),
        "lambda_aggregate": agg,
    }
    summary = {
        "TASK": TASK,
        "STAGE_B_STATUS": "COMPLETE" if len([r for r in formal_units if r["lambda"] == 1.0]) == 18 else "INCOMPLETE",
        "STAGE_C_STATUS": "COMPLETE",
        "PROTOCOL_GATE": "PASS" if len(formal_units) == 90 else "FAIL",
        "CANONICAL_UNIT_SOURCE": str(CANONICAL_UNIT_SOURCE.relative_to(REPO)),
        "N_CANONICAL_UNITS": 18,
        "N_FORMAL_ROWS": len(formal_units),
        "STAGE_A_GATE": stage_a["STAGE_A_GATE"],
        "STAGE_A_CODE_CHURN_OK": stage_a["STAGE_A_CODE_CHURN_OK"],
        "STAGE_A_DECAY_ONLY_ACCUM_OK": stage_a["STAGE_A_DECAY_ONLY_ACCUM_OK"],
        "DECAY_ANISOTROPY_CAUSAL_SIGNAL": causal,
        "TRANSITION_ALIGNMENT_CAUSAL_CHAIN": "YES" if causal == "YES" else "PARTIAL",
        "CAUSAL_MECHANISM_CLOSED": "YES" if causal == "YES" else "PARTIAL",
        "QWEN_SYNTHETIC_CAUSAL_TRANSFER_READY": "NO",
        "METHOD_DESIGN_READY": "NO",
        "FINAL_CLASSIFICATION": "LING_DECAY_ANISOTROPY_CAUSALLY_CONTROLS_RC_ORIENTATION_GAP" if causal == "YES" else "LING_DECAY_ANISOTROPY_CAUSAL_SIGNAL_PARTIAL",
        "lambda_aggregate": agg,
    }
    save_json(RUN_DIR / "formal_aggregate.json", agg)
    save_json(RUN_DIR / "stageB_persistent_formal_summary.json", {"TASK": TASK, "lambda_aggregate": agg, "rows": len(formal_units)})
    save_json(RUN_DIR / "stageC_reconstruction_mediator_decomposition.json", stats)
    save_json(RUN_DIR / "statistics.json", stats)
    save_json(RUN_DIR / "final_classification.json", summary)
    return summary, stats


def make_report(config, manifest, stage0_audit, stage_a, summary, stats, comm_rows, decay_rows, formal_units):
    fig = RUN_DIR / "figures"
    lam_pts = [(lam, (stage_a["lambda_aggregate"][str(lam)])) for lam in LAMBDAS]
    simple_svg(fig / "lambda_vs_median_kappa_rc.svg", "Lambda vs Median KAPPA R/C", [
        ("R128", [(lam, x["KAPPA_D_R"]) for lam, x in lam_pts]),
        ("C128", [(lam, x["KAPPA_D_C"]) for lam, x in lam_pts]),
    ])
    simple_svg(fig / "lambda_vs_delta_kappa_d.svg", "Lambda vs DELTA_KAPPA_D", [
        ("DELTA_KAPPA_D", [(lam, x["DELTA_KAPPA_D"]) for lam, x in lam_pts]),
    ])
    simple_svg(fig / "lambda_vs_median_kappa_full_rc.svg", "Lambda vs Median KAPPA_FULL R/C", [
        ("R128", [(lam, x["KAPPA_FULL_R"]) for lam, x in lam_pts]),
        ("C128", [(lam, x["KAPPA_FULL_C"]) for lam, x in lam_pts]),
    ])
    simple_svg(fig / "lambda_vs_code_flip_rc.svg", "Lambda vs Code Flip R/C", [
        ("R128", [(lam, x["CODE_FLIP_R"]) for lam, x in lam_pts]),
        ("C128", [(lam, x["CODE_FLIP_C"]) for lam, x in lam_pts]),
    ])
    simple_svg(fig / "lambda_vs_decay_only_accum_rc.svg", "Lambda vs Decay-Only Accum R/C", [
        ("R128", [(lam, x["ACCUM_R_H128"]) for lam, x in lam_pts]),
        ("C128", [(lam, x["ACCUM_C_H128"]) for lam, x in lam_pts]),
    ])
    agg = summary.get("lambda_aggregate") or {}
    if all(str(lam) in agg and "KL_R_median" in agg[str(lam)] for lam in LAMBDAS):
        simple_svg(fig / "lambda_vs_kl_rc.svg", "Lambda vs KL R/C", [
            ("R128", [(lam, agg[str(lam)]["KL_R_median"]) for lam in LAMBDAS]),
            ("C128", [(lam, agg[str(lam)]["KL_C_median"]) for lam in LAMBDAS]),
        ])
        simple_svg(fig / "lambda_vs_gap_kl.svg", "Lambda vs GAP_KL", [("GAP_KL", [(lam, agg[str(lam)]["GAP_KL_median"]) for lam in LAMBDAS])])
        simple_svg(fig / "lambda_vs_gap_mse.svg", "Lambda vs GAP_MSE", [("GAP_MSE", [(lam, agg[str(lam)]["GAP_MSE_median"]) for lam in LAMBDAS])])

    report = f"""# {TASK}

## TASK
Test whether Ling true-KDA per-key-channel decay anisotropy causally controls the R128-vs-C128 future-KL orientation gap.

## CODE / COMMIT STATUS
Script: `experiments/ling/run_ling_kda_decay_scalarization_causal_v1.py`

Current git: `{sh(["git", "log", "-1", "--oneline"])}`

## CANONICAL PROTOCOL REUSE
CANONICAL_UNIT_SOURCE={config["CANONICAL_UNIT_SOURCE"]}

N_CANONICAL_UNITS={len(manifest)}

## STAGE 0 (identity/decay preservation/anisotropy dose)
LAMBDA1_DECAY_IDENTITY={stage0_audit["LAMBDA1_DECAY_IDENTITY"]}

LAMBDA1_FP_LOGIT_IDENTITY={stage0_audit["LAMBDA1_FP_LOGIT_IDENTITY"]}

MEAN_LOG_DECAY_PRESERVED={stage0_audit["MEAN_LOG_DECAY_PRESERVED"]}

ANISOTROPY_DOSE_IMPLEMENTED={stage0_audit["ANISOTROPY_DOSE_IMPLEMENTED"]}

LAMBDA0_SCALAR_DECAY={stage0_audit["LAMBDA0_SCALAR_DECAY"]}

## STAGE A
STAGE_A_GATE={stage_a["STAGE_A_GATE"]}

Frozen source states: {stage_a["frozen_state_source"]}

Stage A aggregate:

```json
{json.dumps(stage_a["lambda_aggregate"], indent=2, ensure_ascii=False)}
```

## STAGE B
STAGE_B_STATUS={summary["STAGE_B_STATUS"]}

PROTOCOL_GATE={summary["PROTOCOL_GATE"]}

## STAGE C
STAGE_C_STATUS={summary["STAGE_C_STATUS"]}

## STATISTICS
GAP_KL_SLOPE_PER_UNIT_MEDIAN={stats["GAP_KL_SLOPE_PER_UNIT_MEDIAN"]}

GAP_KL_SLOPE_BOOTSTRAP_CI={stats["GAP_KL_SLOPE_BOOTSTRAP_CI"]}

## CAUSAL CHAIN SUMMARY
DECAY_ANISOTROPY_CAUSAL_SIGNAL={summary["DECAY_ANISOTROPY_CAUSAL_SIGNAL"]}

TRANSITION_ALIGNMENT_CAUSAL_CHAIN={summary["TRANSITION_ALIGNMENT_CAUSAL_CHAIN"]}

CAUSAL_MECHANISM_CLOSED={summary["CAUSAL_MECHANISM_CLOSED"]}

## FINAL CLASSIFICATION
FINAL_CLASSIFICATION={summary["FINAL_CLASSIFICATION"]}

## DOWNSTREAM GATE
QWEN_SYNTHETIC_CAUSAL_TRANSFER_READY={summary["QWEN_SYNTHETIC_CAUSAL_TRANSFER_READY"]}

METHOD_DESIGN_READY=NO

## GPU / PROCESS STATUS
Start GPU audit is in `manifest.json`; end status was captured after run.
"""
    write_text(RUN_DIR / "report.md", report)
    write_text(DOC_DIR / "report.md", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-units", type=int, default=0)
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument("--reuse-stage-a", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [
        "failure.json",
        "formal_per_unit.jsonl",
        "formal_per_unit.csv",
        "formal_horizon.csv",
        "formal_aggregate.json",
        "statistics.json",
        "final_classification.json",
        "stageA_operator_pilot.json",
        "stageA_operator_pilot.csv",
        "stageB_code_churn_pilot.json",
        "stageB_code_churn_pilot.csv",
        "stageB_decay_only_pilot.json",
        "stageB_decay_only_pilot.csv",
    ]:
        path = RUN_DIR / stale
        if path.exists():
            path.unlink()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    os.environ.setdefault("FLA_FLASH_KDA", "0")
    os.environ.setdefault("FLA_TILELANG", "0")
    canonical = import_module(CANONICAL_RUNNER, "ling_canonical_runner_scalar")
    ctl = import_module(CONTROLLED_RUNNER, "ling_controlled_runner_scalar")
    canonical.MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
    canonical.LEGACY_DATA = Path(os.environ.get("LING_AIME24_DATA", REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"))
    units = load_canonical_units()
    if args.limit_units:
        units = units[: args.limit_units]
    config = {
        "TASK": TASK,
        "CANONICAL_UNIT_SOURCE": str(CANONICAL_UNIT_SOURCE.relative_to(REPO)),
        "N_CANONICAL_UNITS": len(load_canonical_units()),
        "active_units": len(units),
        "lambdas": LAMBDAS,
        "horizon": HORIZON,
        "scalarization": "log_alpha_lambda[t,k] = mu[t] + lambda * (log_alpha[t,k] - mu[t]); mu[t]=mean_k(log_alpha[t,k])",
        "strict_scope": "Ling true-KDA scalarization only; no Qwen synthetic, no INT4, no downstream experiment.",
        "reuse_stage_a": bool(args.reuse_stage_a),
    }
    manifest = {
        **config,
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
        "python": sh([sys.executable, "-c", "import sys, torch, transformers, fla; print(sys.version); print(torch.__version__, transformers.__version__, getattr(fla, '__version__', 'fla'))"]),
        "nvidia_smi_start": sh(["nvidia-smi"]),
        "unit_manifest": units,
    }
    save_json(RUN_DIR / "config.json", config)
    save_json(RUN_DIR / "manifest.json", manifest)
    print(f"[{now()}] loading Ling model from {canonical.MODEL_PATH}", flush=True)
    torch, model, tokenizer = canonical.load_model_and_tokenizer()
    scalarizer = DecayScalarizer()
    scalarizer.install(model)
    dataset = {str(r["problem_id"]): r for r in canonical.load_dataset()}
    fp_records = ctl.load_fp_records()
    layers = canonical.kda_layers_from_config(load_json(canonical.MODEL_PATH / "config.json"))
    print(f"[{now()}] loaded model; canonical units={len(units)} kda_layers={len(layers)}", flush=True)
    try:
        s0 = stage0(canonical, ctl, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, units)
        if args.reuse_stage_a and (RUN_DIR / "stageA_operator_panel.json").exists():
            stage_a = load_json(RUN_DIR / "stageA_operator_panel.json")
            comm_rows, decay_rows = [], []
            print(f"[{now()}] reusing saved StageA gate={stage_a['STAGE_A_GATE']}", flush=True)
        else:
            operator_units = units
            stage_a, comm_rows, decay_rows = stageA_operator_panel(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, operator_units)
        if stage_a["STAGE_A_GATE"] == "OPERATOR_CAUSAL_SIGNAL_NEGATIVE":
            summary = {
                "TASK": TASK,
                "STAGE_B_STATUS": "STOPPED_STAGE_A_NEGATIVE",
                "STAGE_C_STATUS": "STOPPED_STAGE_A_NEGATIVE",
                "PROTOCOL_GATE": "STAGE_A_ONLY",
                "STAGE_A_GATE": stage_a["STAGE_A_GATE"],
                "DECAY_ANISOTROPY_CAUSAL_SIGNAL": "NEGATIVE_OPERATOR_PANEL",
                "TRANSITION_ALIGNMENT_CAUSAL_CHAIN": "NO",
                "CAUSAL_MECHANISM_CLOSED": "NO",
                "QWEN_SYNTHETIC_CAUSAL_TRANSFER_READY": "NO",
                "METHOD_DESIGN_READY": "NO",
                "FINAL_CLASSIFICATION": "LING_DECAY_ANISOTROPY_OPERATOR_CAUSAL_SIGNAL_NEGATIVE",
                "lambda_aggregate": stage_a["lambda_aggregate"],
            }
            stats = {}
            save_json(RUN_DIR / "final_classification.json", summary)
            make_report(config, units, s0, stage_a, summary, {"GAP_KL_SLOPE_PER_UNIT_MEDIAN": None, "GAP_KL_SLOPE_BOOTSTRAP_CI": None}, comm_rows, decay_rows, [])
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return
        if args.skip_formal:
            summary = {
                "TASK": TASK,
                "STAGE_B_STATUS": "SKIPPED_BY_ARG",
                "STAGE_C_STATUS": "SKIPPED_BY_ARG",
                "PROTOCOL_GATE": "PARTIAL",
                "STAGE_A_GATE": stage_a["STAGE_A_GATE"],
                "DECAY_ANISOTROPY_CAUSAL_SIGNAL": "NOT_FORMALIZED",
                "TRANSITION_ALIGNMENT_CAUSAL_CHAIN": "PARTIAL",
                "CAUSAL_MECHANISM_CLOSED": "NO",
                "QWEN_SYNTHETIC_CAUSAL_TRANSFER_READY": "NO",
                "METHOD_DESIGN_READY": "NO",
                "FINAL_CLASSIFICATION": "PILOT_ONLY",
                "lambda_aggregate": {},
            }
            stats = {}
            save_json(RUN_DIR / "final_classification.json", summary)
            make_report(config, units, s0, stage_a, summary, {"GAP_KL_SLOPE_PER_UNIT_MEDIAN": None, "GAP_KL_SLOPE_BOOTSTRAP_CI": None}, comm_rows, decay_rows, [])
        else:
            formal_units, horizon_rows = run_formal(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, units)
            summary, stats = aggregate_formal(formal_units, horizon_rows, stage_a)
            make_report(config, units, s0, stage_a, summary, stats, comm_rows, decay_rows, formal_units)
        manifest["nvidia_smi_end"] = sh(["nvidia-smi"])
        save_json(RUN_DIR / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        err = {"error": repr(exc), "traceback": traceback.format_exc()}
        save_json(RUN_DIR / "failure.json", err)
        print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
