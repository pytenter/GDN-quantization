#!/usr/bin/env python3
"""Deterministic, teacher-forced-only functional replay for Qwen/GDN.

No outcome labels are loaded. No token is sampled. Full logits are reduced online
to exact KL, delta-NLL, and FP-token margin degradation and are never serialized.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
CAPTURE_DIR = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
CAPTURE_SCRIPT = CAPTURE_DIR / "run_capture.py"
AIME_ROOT = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
CONDITIONS = ((1, "Identity"), (2, "Hadamard"))
HORIZONS = (1, 4, 8, 16, 32, 64)
ANCHOR_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
LOCAL_CHUNK = 64
PREFIX_CHUNK = 64
EPS = 1e-12
DISK_SAFETY_STOP_GB = 20
FP_PARITY_SAMPLE_INDICES = (0, 20, 40, 59)
FP_PARITY_REL_L2_P95_MAX = 0.03
FP_PARITY_TOP1_AGREEMENT_MIN = 0.995
FP_PARITY_KL_P95_MAX = 0.05


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = import_file(CAPTURE_SCRIPT, "qwen_prior_capture_for_replay")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_source_rows():
    rows = []
    for path in sorted(AIME_ROOT.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
    rows.sort(key=lambda r: (int(r["canonical_index"]), int(r["seed"])))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 FP source trajectories, found {len(rows)}")
    return rows


def get_state(cache, layer_idx):
    if hasattr(cache, "recurrent_states"):
        state = cache.recurrent_states[layer_idx]
        return state[0] if isinstance(state, (dict, tuple, list)) else state
    state = cache.layers[layer_idx].recurrent_states
    return state[0] if isinstance(state, (dict, tuple, list)) else state


def quantize_condition_rows(cache):
    with torch.inference_mode():
        for layer in C.GDN_LAYERS:
            state = get_state(cache, layer)
            state[1].copy_(C.qdq_c128(torch, state[1]).to(state.dtype))
            state[2].copy_(C.qdq_c128(torch, state[2]).to(state.dtype))


def new_cache(model):
    from transformers import DynamicCache
    return DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)


def prompt_batch(tokenizer, row, device):
    prompt = tokenizer.apply_chat_template(
        row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device).repeat(3, 1)
    attention = enc.get("attention_mask")
    attention = attention.to(device).repeat(3, 1) if attention is not None else None
    return ids, attention


def reduce_logits(logits, target_ids):
    """Return exact full-vocabulary metrics for both quantized rows."""
    fp = logits[0].detach().float()
    logp = torch.log_softmax(fp, dim=-1)
    p = logp.exp()
    top = torch.topk(fp, 2, dim=-1)
    fp_id = top.indices[:, 0]
    fp_margin = top.values[:, 0] - top.values[:, 1]
    target = torch.as_tensor(target_ids, dtype=torch.long, device=fp.device)
    fp_target_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    records = {}
    for batch, candidate in CONDITIONS:
        qlogits = logits[batch].detach().float()
        logq = torch.log_softmax(qlogits, dim=-1)
        kl = (p * (logp - logq)).sum(-1)
        q_target_logp = logq.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        delta_nll = fp_target_logp - q_target_logp
        q_selected = qlogits.gather(-1, fp_id.unsqueeze(-1)).squeeze(-1)
        qtop = torch.topk(qlogits, 2, dim=-1)
        competitor = torch.where(qtop.indices[:, 0] == fp_id, qtop.values[:, 1], qtop.values[:, 0])
        q_margin = q_selected - competitor
        delta_margin = fp_margin - q_margin
        delta_z = q_selected - top.values[:, 0]
        decision = delta_z.abs() / fp_margin.clamp_min(1e-6)
        records[candidate] = {
            "kl": kl.cpu().numpy(),
            "delta_nll": delta_nll.cpu().numpy(),
            "delta_margin": delta_margin.cpu().numpy(),
            "fp_margin": fp_margin.cpu().numpy(),
            "q_margin_on_fp_token": q_margin.cpu().numpy(),
            "delta_z_fp_winner": delta_z.cpu().numpy(),
            "decision_sensitivity": decision.cpu().numpy(),
            "top1_changed": (qlogits.argmax(-1) != fp_id).to(torch.int8).cpu().numpy(),
        }
        del qlogits, logq, kl, q_target_logp
    del fp, logp, p, top, fp_id, fp_margin, target, fp_target_logp
    return records


def prefill(model, tokenizer, patch, row, device):
    patch.capture_enabled = False
    patch.quantize_enabled = False
    ids, attention = prompt_batch(tokenizer, row, device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=attention, past_key_values=new_cache(model), use_cache=True)
    cache = output.past_key_values
    last = output.logits[:, -1].detach().float()
    parity = {
        "native_vs_fp_max_abs": float((last[1] - last[0]).abs().max().cpu()),
        "hadamard_vs_fp_max_abs": float((last[2] - last[0]).abs().max().cpu()),
    }
    del output, last, ids, attention
    return cache, parity


def feed_prefix(model, patch, cache, tokens, device):
    patch.capture_enabled = False
    patch.quantize_enabled = False
    for start in range(0, len(tokens), PREFIX_CHUNK):
        ids = torch.tensor(tokens[start:start + PREFIX_CHUNK], dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
        with torch.inference_mode():
            output = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        del output, ids
    return cache


def local_replay(model, tokenizer, patch, row, device, max_tokens=0):
    tokens = list(map(int, row["generated_token_ids"]))
    if max_tokens:
        tokens = tokens[:max_tokens]
    cache, prefill_parity = prefill(model, tokenizer, patch, row, device)
    patch.capture_enabled = True
    patch.quantize_enabled = True
    pieces = []
    started = time.time()
    for start in range(0, max(0, len(tokens) - 1), LOCAL_CHUNK):
        input_tokens = tokens[start:min(start + LOCAL_CHUNK, len(tokens) - 1)]
        target_tokens = tokens[start + 1:start + 1 + len(input_tokens)]
        ids = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
        patch.begin_chunk()
        with torch.inference_mode():
            output = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        metrics = reduce_logits(output.logits, target_tokens)
        for _, candidate in CONDITIONS:
            data = metrics[candidate]
            n = len(input_tokens)
            pieces.append(pd.DataFrame({
                "problem_id": row["problem_id"], "seed": int(row["seed"]), "canonical_index": int(row["canonical_index"]),
                "candidate": candidate, "input_timestep": np.arange(start, start + n, dtype=np.int32),
                "input_token_id": np.asarray(input_tokens, dtype=np.int32), "target_timestep": np.arange(start + 1, start + 1 + n, dtype=np.int32),
                "target_token_id": np.asarray(target_tokens, dtype=np.int32), **data,
            }))
        del output, ids, metrics
        if start % 4096 == 0:
            print(json.dumps({"event": "local_progress", "problem_id": row["problem_id"], "seed": row["seed"], "tokens": start + len(input_tokens), "total": len(tokens), "tokens_per_s": (start + len(input_tokens)) / max(time.time() - started, 1e-6)}), flush=True)
    patch.quantize_enabled = False
    patch.capture_enabled = False
    patch.begin_chunk()
    del cache
    gc.collect(); torch.cuda.empty_cache()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return frame, prefill_parity


def frozen_anchors(token_count):
    maximum = token_count - max(HORIZONS) - 1
    if maximum < 0:
        raise RuntimeError(f"trajectory too short for H=64: {token_count}")
    anchors = []
    for fraction in ANCHOR_FRACTIONS:
        value = int(math.floor(fraction * maximum))
        if value not in anchors:
            anchors.append(value)
    return anchors


def persistent_replay(model, tokenizer, patch, row, device, anchors=None, max_tokens=0):
    tokens = list(map(int, row["generated_token_ids"]))
    if max_tokens:
        tokens = tokens[:max_tokens]
    anchors = frozen_anchors(len(tokens)) if anchors is None else list(anchors)
    rows = []
    for ordinal, anchor in enumerate(anchors):
        cache, _ = prefill(model, tokenizer, patch, row, device)
        cache = feed_prefix(model, patch, cache, tokens[:anchor], device)
        quantize_condition_rows(cache)
        patch.quantize_enabled = True
        for step in range(1, max(HORIZONS) + 1):
            input_token = tokens[anchor + step - 1]
            target_token = tokens[anchor + step]
            ids = torch.tensor([[input_token]], dtype=torch.long, device=device).repeat(3, 1)
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if step in HORIZONS:
                metrics = reduce_logits(output.logits, [target_token])
                for _, candidate in CONDITIONS:
                    data = metrics[candidate]
                    rows.append({
                        "problem_id": row["problem_id"], "seed": int(row["seed"]), "canonical_index": int(row["canonical_index"]),
                        "candidate": candidate, "anchor_ordinal": ordinal, "anchor_fraction": ANCHOR_FRACTIONS[ordinal] if len(anchors) == len(ANCHOR_FRACTIONS) else None,
                        "anchor_timestep": anchor, "horizon": step, "input_timestep": anchor + step - 1,
                        "input_token_id": input_token, "target_timestep": anchor + step, "target_token_id": target_token,
                        **{k: float(v[0]) if k != "top1_changed" else int(v[0]) for k, v in data.items()},
                    })
                del metrics
            del output, ids
        patch.quantize_enabled = False
        del cache
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"event": "anchor_complete", "problem_id": row["problem_id"], "seed": row["seed"], "anchor": anchor, "ordinal": ordinal}), flush=True)
    return pd.DataFrame(rows)


def parity_mode(model, tokenizer, patch, rows, device):
    row = rows[0]
    local, prefill_parity = local_replay(model, tokenizer, patch, row, device, max_tokens=129)
    prior = pd.read_csv(CAPTURE_DIR / "token_level_metrics.csv", nrows=25000)
    prior = prior[(prior["problem_id"] == row["problem_id"]) & (prior["seed"] == int(row["seed"])) & (prior["timestep"] < 128)].copy()
    cond = {"Identity": "INT8_C128", "Hadamard": "INT8_C128_KEY_HADAMARD"}
    comparisons = {}
    passed = True
    for candidate, old_cond in cond.items():
        a = local[local["candidate"] == candidate].sort_values("input_timestep")
        b = prior[prior["condition"] == old_cond].sort_values("timestep")
        if len(a) != 128 or len(b) != 128:
            raise RuntimeError(f"parity row count mismatch {candidate}: {len(a)} {len(b)}")
        margin_err = float(np.max(np.abs(a["fp_margin"].to_numpy() - b["fp_margin"].to_numpy())))
        delta_err = float(np.max(np.abs(a["delta_z_fp_winner"].to_numpy() - b["delta_z_fp_winner"].to_numpy())))
        top1_equal = bool(np.array_equal(a["top1_changed"].to_numpy(), b["top1_changed"].to_numpy()))
        token_equal = bool(np.array_equal(a["input_token_id"].to_numpy(), b["token_id"].to_numpy()))
        ok = margin_err <= 1e-5 and delta_err <= 1e-4 and top1_equal and token_equal
        passed &= ok
        comparisons[candidate] = {"fp_margin_max_abs_error": margin_err, "delta_z_max_abs_error": delta_err, "top1_changed_exact": top1_equal, "token_ids_exact": token_equal, "status": "PASS" if ok else "FAIL"}
    first = persistent_replay(model, tokenizer, patch, row, device, anchors=[0], max_tokens=129)
    second = persistent_replay(model, tokenizer, patch, row, device, anchors=[0], max_tokens=129)
    numeric = ["kl", "delta_nll", "delta_margin", "fp_margin", "q_margin_on_fp_token", "delta_z_fp_winner", "decision_sensitivity"]
    repeat_error = max(float(np.max(np.abs(first[c].to_numpy() - second[c].to_numpy()))) for c in numeric)
    repeat_exact_top1 = bool(np.array_equal(first["top1_changed"].to_numpy(), second["top1_changed"].to_numpy()))
    repeat_ok = repeat_error <= 1e-7 and repeat_exact_top1
    passed &= repeat_ok
    result = {
        "task": "QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1", "mode": "deterministic_teacher_forced_replay_parity",
        "sample": {"problem_id": row["problem_id"], "seed": row["seed"], "tokens_checked": 128},
        "prefill_parity": prefill_parity, "prior_capture_comparison": comparisons,
        "repeat_anchor_determinism": {"max_abs_error": repeat_error, "top1_changed_exact": repeat_exact_top1, "status": "PASS" if repeat_ok else "FAIL"},
        "REPLAY_PARITY_GATE": "PASS" if passed else "FAIL", "new_generation": False, "labels_loaded": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(OUT / "replay_parity/parity_results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("REPLAY_PARITY_GATE_FAIL")


def fp_rotation_parity_mode(model, tokenizer, patch, rows, device):
    prior = json.loads((OUT / "replay_parity/parity_results.json").read_text(encoding="utf-8"))
    if prior["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("FP output-space parity requires passing replay parity")
    observations = []
    cases = [(index, 0) for index in FP_PARITY_SAMPLE_INDICES]
    cases.append((0, frozen_anchors(len(rows[0]["generated_token_ids"]))[0]))
    for index, anchor in cases:
        row = rows[index]
        tokens = list(map(int, row["generated_token_ids"]))
        cache, prefill_parity = prefill(model, tokenizer, patch, row, device)
        if anchor:
            cache = feed_prefix(model, patch, cache, tokens[:anchor], device)
        patch.capture_enabled = True
        patch.quantize_enabled = False
        for horizon in range(1, max(HORIZONS) + 1):
            patch.begin_chunk()
            ids = torch.tensor([[tokens[anchor + horizon - 1]]], dtype=torch.long, device=device).repeat(3, 1)
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
                logits = output.logits[:, -1].float()
                fp, had = logits[0], logits[2]
                logp = torch.log_softmax(fp, dim=-1)
                logq = torch.log_softmax(had, dim=-1)
                record = {
                    "canonical_index": index, "problem_id": row["problem_id"], "seed": int(row["seed"]),
                    "anchor_timestep": anchor, "horizon": horizon,
                    "logit_max_abs_error": float((had - fp).abs().max().cpu()),
                    "logit_relative_l2": float((torch.linalg.vector_norm(had - fp) / torch.linalg.vector_norm(fp).clamp_min(EPS)).cpu()),
                    "full_vocab_kl": float((logp.exp() * (logp - logq)).sum().cpu()),
                    "top1_agreement": bool(fp.argmax() == had.argmax()),
                    "prefill_hadamard_max_abs": prefill_parity["hadamard_vs_fp_max_abs"],
                }
            cache = output.past_key_values
            observations.append(record)
            del output, ids, logits
        patch.capture_enabled = False
        del cache
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"event": "fp_parity_case_complete", "canonical_index": index, "anchor": anchor}), flush=True)
    rel = np.asarray([r["logit_relative_l2"] for r in observations], dtype=np.float64)
    kl = np.asarray([r["full_vocab_kl"] for r in observations], dtype=np.float64)
    top1 = np.asarray([r["top1_agreement"] for r in observations], dtype=np.float64)
    finite = bool(np.isfinite(rel).all() and np.isfinite(kl).all())
    summary = {
        "observations": len(observations), "logit_max_abs_error": float(max(r["logit_max_abs_error"] for r in observations)),
        "logit_relative_l2_p50": float(np.median(rel)), "logit_relative_l2_p95": float(np.quantile(rel, .95)),
        "full_vocab_kl_p50": float(np.median(kl)), "full_vocab_kl_p95": float(np.quantile(kl, .95)),
        "top1_agreement": float(top1.mean()), "all_finite": finite,
    }
    passed = finite and summary["logit_relative_l2_p95"] <= FP_PARITY_REL_L2_P95_MAX and summary["top1_agreement"] >= FP_PARITY_TOP1_AGREEMENT_MIN and summary["full_vocab_kl_p95"] <= FP_PARITY_KL_P95_MAX
    result = {
        "FP_OUTPUT_SPACE_PARITY": "PASS" if passed else "FAIL", "summary": summary,
        "tolerances": {"relative_l2_p95_max": FP_PARITY_REL_L2_P95_MAX, "top1_agreement_min": FP_PARITY_TOP1_AGREEMENT_MIN, "kl_p95_max": FP_PARITY_KL_P95_MAX},
        "measurement_horizons": list(HORIZONS), "case_indices": list(FP_PARITY_SAMPLE_INDICES),
        "extra_nonzero_anchor": {"canonical_index": 0, "fraction": ANCHOR_FRACTIONS[0]},
        "observations_by_case_and_horizon": [r for r in observations if r["horizon"] in HORIZONS],
        "new_generation": False, "labels_loaded": False, "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(OUT / "replay_parity/fp_rotation_parity.json", result)
    print(json.dumps(result["summary"]), flush=True)
    print("FP_OUTPUT_SPACE_PARITY = " + result["FP_OUTPUT_SPACE_PARITY"], flush=True)
    if not passed:
        raise RuntimeError("FP_OUTPUT_SPACE_PARITY_FAIL")


def shard_mode(model, tokenizer, patch, rows, device, shard_index, shard_count):
    parity = json.loads((OUT / "replay_parity/parity_results.json").read_text(encoding="utf-8"))
    if parity["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("formal replay requires passing parity gate")
    fp_parity = json.loads((OUT / "replay_parity/fp_rotation_parity.json").read_text(encoding="utf-8"))
    if fp_parity["FP_OUTPUT_SPACE_PARITY"] != "PASS":
        raise RuntimeError("formal replay requires passing FP output-space parity")
    assigned = [(i, r) for i, r in enumerate(rows) if i % shard_count == shard_index]
    shard_dir = OUT / "replay" / f"shard_{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_dir / "status.json"
    completed = 0
    started = time.time()
    for global_index, row in assigned:
        free_gb = shutil.disk_usage(OUT).free / (1024 ** 3)
        if free_gb < DISK_SAFETY_STOP_GB:
            raise RuntimeError(f"disk safety stop: {free_gb:.2f} GiB free")
        stem = f"{global_index:03d}_{row['problem_id']}_seed{int(row['seed'])}"
        local_path = shard_dir / f"{stem}_local.parquet"
        persistent_path = shard_dir / f"{stem}_persistent.parquet"
        meta_path = shard_dir / f"{stem}_meta.json"
        if local_path.exists() and persistent_path.exists() and meta_path.exists():
            completed += 1
            continue
        if local_path.exists() or persistent_path.exists() or meta_path.exists():
            raise RuntimeError(f"partial unit exists and will not be overwritten: {stem}")
        print(json.dumps({"event": "trajectory_start", "global_index": global_index, "problem_id": row["problem_id"], "seed": row["seed"], "tokens": len(row["generated_token_ids"]), "anchors": frozen_anchors(len(row["generated_token_ids"]))}), flush=True)
        local, prefill_parity = local_replay(model, tokenizer, patch, row, device)
        persistent = persistent_replay(model, tokenizer, patch, row, device)
        local_tmp = local_path.with_suffix(".parquet.tmp")
        persistent_tmp = persistent_path.with_suffix(".parquet.tmp")
        local.to_parquet(local_tmp, engine="pyarrow", compression="zstd", index=False)
        persistent.to_parquet(persistent_tmp, engine="pyarrow", compression="zstd", index=False)
        os.replace(local_tmp, local_path); os.replace(persistent_tmp, persistent_path)
        save_json(meta_path, {
            "global_index": global_index, "problem_id": row["problem_id"], "seed": int(row["seed"]),
            "canonical_index": int(row["canonical_index"]), "generated_tokens": len(row["generated_token_ids"]),
            "anchors": frozen_anchors(len(row["generated_token_ids"])), "local_rows": len(local), "persistent_rows": len(persistent),
            "prefill_parity": prefill_parity, "new_generation": False, "labels_loaded": False,
        })
        completed += 1
        save_json(status_path, {"status": "RUNNING", "shard": shard_index, "completed": completed, "assigned": len(assigned), "last_unit": stem, "elapsed_seconds": time.time() - started})
        del local, persistent
        gc.collect(); torch.cuda.empty_cache()
    save_json(status_path, {"status": "COMPLETE", "shard": shard_index, "completed": completed, "assigned": len(assigned), "elapsed_seconds": time.time() - started, "labels_loaded": False, "new_generation": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parity", "fp_parity", "shard"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = load_source_rows()
    model, tokenizer = C.load_model(torch)
    device = next(model.parameters()).device
    with C.BatchedMechanismPatch(torch, model) as patch:
        if args.mode == "parity":
            parity_mode(model, tokenizer, patch, rows, device)
        elif args.mode == "fp_parity":
            fp_rotation_parity_mode(model, tokenizer, patch, rows, device)
        else:
            shard_mode(model, tokenizer, patch, rows, device, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
