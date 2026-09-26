#!/usr/bin/env python3
"""Qwen teacher-forced replay mechanism capture.

Three aligned batch rows are evaluated in one forward stream:
  0: FP_STATE
  1: INT8_C128
  2: INT8_C128 + key-side H128

The original FP generated token sequence is the only continuation.  No sampling,
scoring, training, or parameter modification is performed.  Full recurrent states
are reduced online to per-layer/per-token mechanism scalars to bound disk use.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


TASK = "QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
DATA_ROOT = Path("/data/zypan")
REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
MODEL_PATH = DATA_ROOT / "modelscope_models/Qwen3.5-9B"
TRANSFORMERS_SRC = DATA_ROOT / "transformers-qwen35/src"
STATE_QUANT_SRC = DATA_ROOT / "experiments/qwen35_gdn_quant"
AIME_ROOT = DATA_ROOT / "worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards"
ALIGNMENT = REPO / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics"
OUT = REPO / "experiments" / TASK
GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
CONDITIONS = ("INT8_C128", "INT8_C128_KEY_HADAMARD")
HORIZONS = (1, 4, 8, 16, 32)
ALPHA = 0.9
EPS = 1e-12
DECISION_EPS = 1e-6


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stat(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def hadamard(torch, n=128, device=None):
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def recover_hadamard(torch, state, h):
    return torch.matmul(h, state)


def qdq_c128(torch, state):
    scale = state.detach().float().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(state.detach().float() / scale).clamp(-127, 127)
    return codes * scale


def rms_jvp(torch, o, e, eps):
    of, ef = o.float(), e.float()
    r = torch.sqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    inner = (of * ef).mean(-1, keepdim=True)
    return ef / r - of * inner / (r ** 3)


class BatchedMechanismPatch:
    """Batch-specific FP/native/Hadamard recurrent semantics with online capture."""

    def __init__(self, torch, model):
        self.torch = torch
        self.model = model
        self.h = hadamard(torch, 128, next(model.parameters()).device)
        self.capture_enabled = False
        self.quantize_enabled = False
        self.current_layer = None
        self.pending = {}
        self.chunk_metrics = {}
        self.saved = []
        self.handles = []
        self.calls = 0
        self.layers_seen = set()
        self.prefill_max_abs_logit_diff = None

    def begin_chunk(self):
        self.pending.clear()
        self.chunk_metrics.clear()

    def _condition_errors(self, state):
        fp = state[0].float()
        native = state[1].float()
        had = recover_hadamard(self.torch, state[2].float(), self.h)
        return native - fp, had - fp, fp

    def _run(self, query, key, value, g, beta, initial_state, output_final_state, use_norm):
        torch = self.torch
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

        if query.shape[0] != 3:
            raise RuntimeError(f"capture requires batch=3, got {query.shape[0]}")
        initial_dtype = query.dtype
        if use_norm:
            query = qmod.l2norm(query, dim=-1, eps=1e-6)
            key = qmod.l2norm(key, dim=-1, eps=1e-6)
        query = query.clone()
        key = key.clone()
        query[2] = query[2].float() @ self.h
        key[2] = key[2].float() @ self.h
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
        ]
        bsz, heads, length, kdim = key.shape
        vdim = value.shape[-1]
        query = query * (query.shape[-1] ** -0.5)
        state = (
            torch.zeros(bsz, heads, kdim, vdim, dtype=torch.float32, device=value.device)
            if initial_state is None
            else initial_state.float().clone()
        )
        core = torch.empty(bsz, heads, length, vdim, dtype=torch.float32, device=value.device)
        m0 = {c: [] for c in CONDITIONS}
        m0_rel = {c: [] for c in CONDITIONS}
        e_lag1 = {c: [] for c in CONDITIONS}

        for i in range(length):
            q_t = query[:, :, i]
            if self.capture_enabled:
                en, eh, fp_prev = self._condition_errors(state)
                qfp = q_t[0]
                e_lag1["INT8_C128"].append((qfp.unsqueeze(-1) * en).sum(dim=-2))
                e_lag1["INT8_C128_KEY_HADAMARD"].append((qfp.unsqueeze(-1) * eh).sum(dim=-2))

            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            state = state * g_t
            memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - memory) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

            if self.quantize_enabled:
                state[1] = qdq_c128(torch, state[1])
                state[2] = qdq_c128(torch, state[2])
            if self.capture_enabled:
                en, eh, fp_now = self._condition_errors(state)
                fp_norm = torch.linalg.vector_norm(fp_now).clamp_min(EPS)
                for cond, err in (("INT8_C128", en), ("INT8_C128_KEY_HADAMARD", eh)):
                    n = torch.linalg.vector_norm(err)
                    m0[cond].append(n)
                    m0_rel[cond].append(n / fp_norm)

        if self.capture_enabled:
            layer = int(self.current_layer)
            self.pending[layer] = {
                "core_fp": core[0].transpose(0, 1).contiguous(),  # [T,H,V]
                "e_lag1": {c: torch.stack(e_lag1[c], 0) for c in CONDITIONS},
                "m0": {c: torch.stack(m0[c]) for c in CONDITIONS},
                "m0_rel": {c: torch.stack(m0_rel[c]) for c in CONDITIONS},
            }
            self.layers_seen.add(layer)
        self.calls += 1
        final_state = state if output_final_state else None
        return core.transpose(1, 2).contiguous().to(initial_dtype), final_state

    def _recurrent(self, query, key, value, g, beta, initial_state, output_final_state,
                   use_qk_l2norm_in_kernel=False, **kwargs):
        if not self.capture_enabled and not self.quantize_enabled:
            import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
            if use_qk_l2norm_in_kernel:
                query = qmod.l2norm(query, dim=-1, eps=1e-6)
                key = qmod.l2norm(key, dim=-1, eps=1e-6)
            query, key = query.clone(), key.clone()
            query[2] = query[2].float() @ self.h
            key[2] = key[2].float() @ self.h
            return self.original_recurrent(
                query, key, value, g, beta, initial_state, output_final_state,
                use_qk_l2norm_in_kernel=False, **kwargs,
            )
        return self._run(query, key, value, g, beta, initial_state, output_final_state,
                         bool(use_qk_l2norm_in_kernel))

    def _chunk(self, query, key, value, g, beta, chunk_size=64, initial_state=None,
               output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if not self.capture_enabled and not self.quantize_enabled:
            import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
            if use_qk_l2norm_in_kernel:
                query = qmod.l2norm(query, dim=-1, eps=1e-6)
                key = qmod.l2norm(key, dim=-1, eps=1e-6)
            query, key = query.clone(), key.clone()
            query[2] = query[2].float() @ self.h
            key[2] = key[2].float() @ self.h
            return self.original_chunk(
                query, key, value, g, beta, chunk_size=chunk_size,
                initial_state=initial_state, output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False, **kwargs,
            )
        return self._run(query, key, value, g, beta, initial_state, output_final_state,
                         bool(use_qk_l2norm_in_kernel))

    def _norm_pre(self, layer, module, inputs):
        if not self.capture_enabled:
            return
        torch = self.torch
        rec = self.pending.get(layer)
        if rec is None:
            raise RuntimeError(f"missing recurrent capture before norm at layer {layer}")
        o = rec["core_fp"]
        length, heads, vdim = o.shape
        z = inputs[1].detach().float().reshape(3, length, heads, vdim)[0]
        weight = module.weight.detach().float().reshape(1, 1, vdim)
        la = self._layers()[layer].linear_attn
        W = la.out_proj.weight.detach().float()
        metrics = {}
        for cond in CONDITIONS:
            e = rec["e_lag1"][cond]
            j = rms_jvp(torch, o, e, module.variance_epsilon)
            u = j * weight * torch.nn.functional.silu(z)
            projected = u.reshape(length, -1) @ W.T
            m5 = torch.linalg.vector_norm(projected, dim=-1)
            metrics[cond] = {
                "m0": rec["m0"][cond].detach().float().cpu().numpy(),
                "m0_rel": rec["m0_rel"][cond].detach().float().cpu().numpy(),
                "m5": m5.detach().float().cpu().numpy(),
            }
        self.chunk_metrics[layer] = metrics
        del self.pending[layer]

    def _layers(self):
        return self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

        self.saved = [
            (qmod, "torch_recurrent_gated_delta_rule", qmod.torch_recurrent_gated_delta_rule),
            (qmod, "torch_chunk_gated_delta_rule", qmod.torch_chunk_gated_delta_rule),
        ]
        self.original_recurrent = qmod.torch_recurrent_gated_delta_rule
        self.original_chunk = qmod.torch_chunk_gated_delta_rule
        qmod.torch_recurrent_gated_delta_rule = self._recurrent
        qmod.torch_chunk_gated_delta_rule = self._chunk
        for idx, layer in enumerate(self._layers()):
            mod = getattr(layer, "linear_attn", None)
            if mod is None:
                continue

            def make_layer_pre(i):
                def hook(_module, _inputs):
                    self.current_layer = i
                return hook

            def make_norm_pre(i):
                def hook(module, inputs):
                    self._norm_pre(i, module, inputs)
                return hook

            self.handles.append(mod.register_forward_pre_hook(make_layer_pre(idx)))
            self.handles.append(mod.norm.register_forward_pre_hook(make_norm_pre(idx)))
        return self

    def __exit__(self, *_):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for h in self.handles:
            h.remove()


def load_rows():
    rows = []
    for p in sorted(AIME_ROOT.glob("worker*/fp_state.jsonl")):
        rows.extend(load_jsonl(p))
    rows.sort(key=lambda r: (int(r["canonical_index"]), int(r["seed"])))
    labels = defaultdict(dict)
    for r in load_jsonl(ALIGNMENT / "sample_metrics.jsonl"):
        labels[(r["problem_id"], int(r["seed"]))][r["condition"]] = bool(r["v4_correct"])
    for r in rows:
        d = labels[(r["problem_id"], int(r["seed"]))]
        fp, native, had = d["fp_state"], d["int8_c128"], d["int8_c128_key_h"]
        if fp and native:
            group = "A_BOTH_CORRECT"
        elif fp and not native and had:
            group = "B_HADAMARD_RESCUED"
        elif fp and not native and not had:
            group = "C_HADAMARD_FAILED"
        else:
            group = "D_FP_WRONG"
        r["analysis_group"] = group
        r["label_fp_correct"] = fp
        r["label_int8_correct"] = native
        r["label_hadamard_correct"] = had
    return rows


def load_model(torch):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for p in (str(STATE_QUANT_SRC), str(TRANSFORMERS_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    return model, tok


def persistence(values):
    x = np.asarray(values, dtype=np.float64)
    acc = np.zeros_like(x)
    available = np.zeros(len(x), dtype=bool)
    for k in HORIZONS:
        if k < len(x):
            acc[:-k] += (ALPHA ** k) * x[k:] / (x[:-k] + EPS)
            available[:-k] = True
    out = np.full_like(x, np.nan)
    out[available] = acc[available]
    return out


def decision_metrics(torch, logits):
    fp = logits[0].detach().float()
    top = torch.topk(fp, 2, dim=-1)
    ids = top.indices[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]
    result = {}
    for batch, cond in ((1, "INT8_C128"), (2, "INT8_C128_KEY_HADAMARD")):
        qlog = logits[batch].detach().float()
        shifted = qlog.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - top.values[:, 0]
        d = shifted.abs() / margin.clamp_min(DECISION_EPS)
        result[cond] = {
            "margin": margin.cpu().numpy(),
            "delta_z": shifted.cpu().numpy(),
            "decision_sensitivity": d.cpu().numpy(),
            "top1_changed": (qlog.argmax(-1) != ids).cpu().numpy().astype(np.int8),
        }
    return result


TOKEN_FIELDS = [
    "problem_id", "seed", "group", "condition", "timestep", "token_id",
    "m0_global_l2", "m0_layer_mean", "m0_layer_median", "m0_layer_p95", "m0_layer_max",
    "m0_rel_layer_mean", "m0_rel_layer_p95", "m0_rel_layer_max",
    "m5_global_l2", "m5_layer_mean", "m5_layer_median", "m5_layer_p95", "m5_layer_max",
    "persistence", "fp_margin", "delta_z_fp_winner", "decision_sensitivity", "top1_changed",
]


def summarize_trajectory(row, condition, m0, m0rel, m5, pvals, decision):
    layers = []
    for j, layer in enumerate(GDN_LAYERS):
        layers.append({
            "layer": layer,
            "m0": stat(m0[:, j]),
            "m0_relative": stat(m0rel[:, j]),
            "m5": stat(m5[:, j]),
        })
    d = decision["decision_sensitivity"]
    return {
        "problem_id": row["problem_id"],
        "seed": int(row["seed"]),
        "canonical_index": int(row["canonical_index"]),
        "group": row["analysis_group"],
        "condition": condition,
        "tokens": int(m0.shape[0]),
        "m0": stat(m0),
        "m0_relative": stat(m0rel),
        "m0_global_l2": stat(np.sqrt(np.square(m0).sum(axis=1))),
        "m5": stat(m5),
        "m5_global_l2": stat(np.sqrt(np.square(m5).sum(axis=1))),
        "persistence": stat(pvals),
        "decision_sensitivity": {
            **stat(d),
            "count_gt1": int((d > 1.0).sum()),
            "fraction_gt1": float((d > 1.0).mean()),
            "top1_changed_count": int(decision["top1_changed"].sum()),
            "margin_le_epsilon_count": int((decision["margin"] <= DECISION_EPS).sum()),
        },
        "per_layer": layers,
    }


def capture_one(torch, model, tokenizer, patch, row, chunk_size, max_tokens, token_writer,
                offload_cache=False):
    device = next(model.parameters()).device
    prompt = tokenizer.apply_chat_template(
        row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device).repeat(3, 1)
    attention = enc.get("attention_mask")
    attention = attention.to(device).repeat(3, 1) if attention is not None else None
    tokens = list(map(int, row["generated_token_ids"]))
    if max_tokens:
        tokens = tokens[:max_tokens]

    patch.capture_enabled = False
    patch.quantize_enabled = False
    cache = None
    if offload_cache:
        from transformers import DynamicCache
        # Only full-attention KV tensors are offloaded. Linear-attention/GDN
        # recurrent state remains resident and is captured without approximation.
        cache = DynamicCache(
            config=model.config, offloading=True, offload_only_non_sliding=True
        )
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill = model(
            input_ids=prompt_ids, attention_mask=attention,
            past_key_values=cache, use_cache=True,
        )
    past = prefill.past_key_values
    last = prefill.logits[:, -1].detach().float()
    patch.prefill_max_abs_logit_diff = {
        "native_vs_fp": float((last[1] - last[0]).abs().max().cpu()),
        "hadamard_vs_fp": float((last[2] - last[0]).abs().max().cpu()),
    }
    del prefill, last, prompt_ids, attention

    store = {c: {"m0": [], "m0_rel": [], "m5": [], "decision": defaultdict(list)} for c in CONDITIONS}
    patch.capture_enabled = True
    patch.quantize_enabled = True
    started = time.time()
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
        patch.begin_chunk()
        with torch.inference_mode():
            output = model(input_ids=ids, past_key_values=past, use_cache=True)
        past = output.past_key_values
        missing = sorted(set(GDN_LAYERS) - set(patch.chunk_metrics))
        if missing:
            raise RuntimeError(f"missing chunk metrics for layers {missing}")
        decisions = decision_metrics(torch, output.logits)
        for cond in CONDITIONS:
            store[cond]["m0"].append(np.stack([patch.chunk_metrics[l][cond]["m0"] for l in GDN_LAYERS], axis=1))
            store[cond]["m0_rel"].append(np.stack([patch.chunk_metrics[l][cond]["m0_rel"] for l in GDN_LAYERS], axis=1))
            store[cond]["m5"].append(np.stack([patch.chunk_metrics[l][cond]["m5"] for l in GDN_LAYERS], axis=1))
            for key, arr in decisions[cond].items():
                store[cond]["decision"][key].append(arr)
        del output, ids
        if (start // chunk_size) % 64 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(json.dumps({
                "event": "progress", "problem_id": row["problem_id"], "seed": row["seed"],
                "tokens": min(start + len(chunk), len(tokens)), "total": len(tokens),
                "tokens_per_s": (start + len(chunk)) / elapsed,
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            }), flush=True)

    summaries = []
    for cond in CONDITIONS:
        m0 = np.concatenate(store[cond]["m0"], axis=0)
        m0rel = np.concatenate(store[cond]["m0_rel"], axis=0)
        m5 = np.concatenate(store[cond]["m5"], axis=0)
        dec = {k: np.concatenate(v) for k, v in store[cond]["decision"].items()}
        m0global = np.sqrt(np.square(m0).sum(axis=1))
        pvals = persistence(m0global)
        summaries.append(summarize_trajectory(row, cond, m0, m0rel, m5, pvals, dec))
        for t in range(len(tokens)):
            m0t, m0rt, m5t = m0[t], m0rel[t], m5[t]
            token_writer.writerow({
                "problem_id": row["problem_id"], "seed": int(row["seed"]),
                "group": row["analysis_group"], "condition": cond,
                "timestep": t, "token_id": tokens[t],
                "m0_global_l2": float(m0global[t]),
                "m0_layer_mean": float(m0t.mean()), "m0_layer_median": float(np.median(m0t)),
                "m0_layer_p95": float(np.quantile(m0t, 0.95)), "m0_layer_max": float(m0t.max()),
                "m0_rel_layer_mean": float(m0rt.mean()), "m0_rel_layer_p95": float(np.quantile(m0rt, 0.95)),
                "m0_rel_layer_max": float(m0rt.max()),
                "m5_global_l2": float(np.sqrt(np.square(m5t).sum())),
                "m5_layer_mean": float(m5t.mean()), "m5_layer_median": float(np.median(m5t)),
                "m5_layer_p95": float(np.quantile(m5t, 0.95)), "m5_layer_max": float(m5t.max()),
                "persistence": float(pvals[t]), "fp_margin": float(dec["margin"][t]),
                "delta_z_fp_winner": float(dec["delta_z"][t]),
                "decision_sensitivity": float(dec["decision_sensitivity"][t]),
                "top1_changed": int(dec["top1_changed"][t]),
            })
    del past, store
    torch.cuda.empty_cache()
    return summaries


def command_capture(args):
    import torch

    rows = load_rows()
    rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
    if args.max_units:
        rows = rows[: args.max_units]
    shard = Path(args.output_dir) / f"shard_{args.shard_index:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    token_path = shard / "token_metrics.csv.gz"
    summary_path = shard / "trajectory_summary.jsonl"
    status_path = shard / "status.json"
    completed = 0
    if args.resume and status_path.exists():
        old_status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = int(old_status.get("completed_units", 0))
        if completed < 0 or completed > len(rows):
            raise RuntimeError(f"invalid resume count {completed} for {len(rows)} rows")
    elif any(p.exists() for p in (token_path, summary_path, status_path)):
        raise FileExistsError(f"shard already contains output; use --resume: {shard}")
    jdump(status_path, {
        "status": "STARTING", "shard": args.shard_index, "units": len(rows),
        "completed_units": completed, "resume": bool(args.resume),
        "offload_cache": bool(args.offload_cache),
    })
    model, tokenizer = load_model(torch)
    start_time = time.time()
    try:
        token_mode = "at" if completed else "wt"
        summary_mode = "a" if completed else "w"
        with gzip.open(token_path, token_mode, newline="", encoding="utf-8") as tf, summary_path.open(summary_mode, encoding="utf-8") as sf:
            writer = csv.DictWriter(tf, fieldnames=TOKEN_FIELDS)
            if not completed:
                writer.writeheader()
            with BatchedMechanismPatch(torch, model) as patch:
                for row in rows[completed:]:
                    print(json.dumps({"event": "trajectory_start", "problem_id": row["problem_id"], "seed": row["seed"], "tokens": min(len(row["generated_token_ids"]), args.max_tokens or 10**12)}), flush=True)
                    summaries = capture_one(
                        torch, model, tokenizer, patch, row, args.chunk_size,
                        args.max_tokens, writer, offload_cache=args.offload_cache,
                    )
                    for s in summaries:
                        s["prefill_max_abs_logit_diff"] = patch.prefill_max_abs_logit_diff
                        sf.write(json.dumps(s, sort_keys=True) + "\n")
                    sf.flush(); tf.flush()
                    completed += 1
                    jdump(status_path, {
                        "status": "RUNNING", "shard": args.shard_index, "completed_units": completed,
                        "total_units": len(rows), "last_problem_id": row["problem_id"], "last_seed": row["seed"],
                        "elapsed_seconds": time.time() - start_time,
                    })
        jdump(status_path, {
            "status": "COMPLETE", "shard": args.shard_index, "completed_units": completed,
            "total_units": len(rows), "elapsed_seconds": time.time() - start_time,
            "token_csv_gz": str(token_path), "trajectory_summary": str(summary_path),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        })
    except Exception as exc:
        jdump(status_path, {
            "status": "FAILED", "shard": args.shard_index, "completed_units": completed,
            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.time() - start_time,
        })
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--max-units", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--offload-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    command_capture(args)


if __name__ == "__main__":
    main()
