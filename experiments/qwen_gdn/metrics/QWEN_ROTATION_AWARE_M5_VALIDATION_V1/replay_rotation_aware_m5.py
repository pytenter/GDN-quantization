#!/usr/bin/env python3
"""Observation-only M5 primitive capture under deterministic teacher forcing."""

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
import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"
PRIOR = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
PRIOR_SCRIPT = PRIOR / "run_capture.py"
AIME_ROOT = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
CONDITIONS = ("INT8_C128", "INT8_C128_KEY_HADAMARD")
FP_AUDIT_INDICES = (0, 20, 40, 59)
EPS = 1e-12
FIELDS = [
    "problem_id", "seed", "canonical_index", "condition", "timestep", "token_id", "layer",
    "m0", "m0_relative", "readout_error_norm", "rms_denominator_min", "rms_denominator_mean",
    "rms_jvp_norm", "gate_rms", "weighted_jvp_norm", "m5_canonical", "fp_signal_norm",
    "fp_signal_denominator_lt_1e8", "fp_signal_denominator_lt_1e6", "fp_signal_denominator_lt_1e4",
    "m5_over_m0", "m5_over_fp_signal", "readout_over_m0", "jvp_over_readout", "m5_over_jvp",
    "head_readout_max", "head_readout_argmax", "head_readout_concentration",
    "head_jvp_max", "head_jvp_argmax", "head_jvp_concentration",
    "head_projected_max", "head_projected_argmax", "head_projected_concentration",
    "fp_margin", "delta_z_fp_winner", "decision_sensitivity", "top1_changed",
]


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = import_file(PRIOR_SCRIPT, "qwen_frozen_capture_for_m5_components")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_rows():
    rows = []
    for path in sorted(AIME_ROOT.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda r: (int(r["canonical_index"]), int(r["seed"])))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 trajectories, got {len(rows)}")
    return rows


def concentration(torch, values):
    total = values.sum(-1).clamp_min(EPS)
    maximum, argmax = values.max(-1)
    return maximum, argmax, maximum / total


class PrimitivePatch(C.BatchedMechanismPatch):
    """Same recurrent semantics as the frozen prior patch; only the norm hook is richer."""

    def _norm_pre(self, layer, module, inputs):
        if not self.capture_enabled:
            return
        torch = self.torch
        rec = self.pending.get(layer)
        if rec is None:
            raise RuntimeError(f"missing recurrent capture before norm at layer {layer}")
        o = rec["core_fp"].float()  # [T,H,V]
        length, heads, vdim = o.shape
        z = inputs[1].detach().float().reshape(3, length, heads, vdim)[0]
        weight = module.weight.detach().float().reshape(1, 1, vdim)
        gate = torch.nn.functional.silu(z)
        la = self._layers()[layer].linear_attn
        W = la.out_proj.weight.detach().float()
        r = torch.sqrt(o.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
        fp_u = (o / r) * weight * gate
        fp_projected = fp_u.reshape(length, -1) @ W.T
        fp_signal = torch.linalg.vector_norm(fp_projected, dim=-1)
        W_heads = W.reshape(W.shape[0], heads, vdim)
        metrics = {}
        for cond in CONDITIONS:
            e = rec["e_lag1"][cond].float()
            j = C.rms_jvp(torch, o, e, module.variance_epsilon)
            u = j * weight * gate
            projected = u.reshape(length, -1) @ W.T
            m5 = torch.linalg.vector_norm(projected, dim=-1)
            m0 = rec["m0"][cond].float()
            m0rel = rec["m0_rel"][cond].float()
            readout = torch.linalg.vector_norm(e.reshape(length, -1), dim=-1)
            jnorm = torch.linalg.vector_norm(j.reshape(length, -1), dim=-1)
            unorm = torch.linalg.vector_norm(u.reshape(length, -1), dim=-1)
            gate_rms = torch.sqrt(gate.pow(2).mean(dim=(-1, -2)))
            readout_h = torch.linalg.vector_norm(e, dim=-1)
            j_h = torch.linalg.vector_norm(j, dim=-1)
            projected_h = []
            for head in range(heads):
                ph = u[:, head] @ W_heads[:, head].T
                projected_h.append(torch.linalg.vector_norm(ph, dim=-1))
                del ph
            projected_h = torch.stack(projected_h, dim=-1)
            rh_max, rh_arg, rh_conc = concentration(torch, readout_h)
            jh_max, jh_arg, jh_conc = concentration(torch, j_h)
            ph_max, ph_arg, ph_conc = concentration(torch, projected_h)
            values = {
                "m0": m0, "m0_relative": m0rel, "readout_error_norm": readout,
                "rms_denominator_min": r.squeeze(-1).min(-1).values,
                "rms_denominator_mean": r.squeeze(-1).mean(-1),
                "rms_jvp_norm": jnorm, "gate_rms": gate_rms, "weighted_jvp_norm": unorm,
                "m5_canonical": m5, "fp_signal_norm": fp_signal,
                "fp_signal_denominator_lt_1e8": (fp_signal < 1e-8).to(torch.int8),
                "fp_signal_denominator_lt_1e6": (fp_signal < 1e-6).to(torch.int8),
                "fp_signal_denominator_lt_1e4": (fp_signal < 1e-4).to(torch.int8),
                "m5_over_m0": m5 / m0.clamp_min(EPS),
                "m5_over_fp_signal": m5 / fp_signal.clamp_min(EPS),
                "readout_over_m0": readout / m0.clamp_min(EPS),
                "jvp_over_readout": jnorm / readout.clamp_min(EPS),
                "m5_over_jvp": m5 / jnorm.clamp_min(EPS),
                "head_readout_max": rh_max, "head_readout_argmax": rh_arg.to(torch.int16), "head_readout_concentration": rh_conc,
                "head_jvp_max": jh_max, "head_jvp_argmax": jh_arg.to(torch.int16), "head_jvp_concentration": jh_conc,
                "head_projected_max": ph_max, "head_projected_argmax": ph_arg.to(torch.int16), "head_projected_concentration": ph_conc,
            }
            metrics[cond] = {name: value.detach().cpu().numpy() for name, value in values.items()}
            del e, j, u, projected, projected_h, values
        self.chunk_metrics[int(layer)] = metrics
        self.layers_seen.add(int(layer))
        del self.pending[layer], fp_u, fp_projected, fp_signal, W_heads


def capture_trajectory(model, tokenizer, patch, row, output_path, quantize, max_tokens=0, chunk_size=64):
    device = next(model.parameters()).device
    prompt = tokenizer.apply_chat_template(row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device).repeat(3, 1)
    attention = enc.get("attention_mask")
    attention = attention.to(device).repeat(3, 1) if attention is not None else None
    tokens = list(map(int, row["generated_token_ids"]))
    if max_tokens:
        tokens = tokens[:max_tokens]
    patch.capture_enabled = False
    patch.quantize_enabled = False
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)
    with torch.inference_mode():
        prefill = model(input_ids=prompt_ids, attention_mask=attention, past_key_values=cache, use_cache=True)
    cache = prefill.past_key_values
    last = prefill.logits[:, -1].detach().float()
    prefill_diff = {"native_vs_fp": float((last[1] - last[0]).abs().max().cpu()), "hadamard_vs_fp": float((last[2] - last[0]).abs().max().cpu())}
    del prefill, last, prompt_ids, attention
    patch.capture_enabled = True
    patch.quantize_enabled = bool(quantize)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    rows_written = 0
    started = time.time()
    try:
        for start in range(0, len(tokens), chunk_size):
            chunk = tokens[start:start + chunk_size]
            ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
            patch.begin_chunk()
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            missing = sorted(set(C.GDN_LAYERS) - set(patch.chunk_metrics))
            if missing:
                raise RuntimeError(f"missing primitive metrics for layers {missing}")
            decisions = C.decision_metrics(torch, output.logits)
            frames = []
            for layer in C.GDN_LAYERS:
                for cond in CONDITIONS:
                    data = patch.chunk_metrics[layer][cond]
                    dec = decisions[cond]
                    n = len(chunk)
                    frame = pd.DataFrame({
                        "problem_id": row["problem_id"], "seed": int(row["seed"]), "canonical_index": int(row["canonical_index"]),
                        "condition": cond, "timestep": np.arange(start, start + n, dtype=np.int32),
                        "token_id": np.asarray(chunk, dtype=np.int32), "layer": int(layer),
                        **data,
                        "fp_margin": dec["margin"], "delta_z_fp_winner": dec["delta_z"],
                        "decision_sensitivity": dec["decision_sensitivity"], "top1_changed": dec["top1_changed"],
                    })
                    frames.append(frame[FIELDS])
            frame = pd.concat(frames, ignore_index=True)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd", compression_level=7)
            writer.write_table(table)
            rows_written += len(frame)
            del output, ids, decisions, frames, frame, table
            if start % 4096 == 0:
                print(json.dumps({"event": "component_progress", "problem_id": row["problem_id"], "seed": row["seed"], "quantize": quantize, "tokens": start + len(chunk), "total": len(tokens), "tokens_per_s": (start + len(chunk)) / max(time.time() - started, 1e-6)}), flush=True)
        if writer is not None:
            writer.close(); writer = None
        os.replace(tmp, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        raise
    finally:
        patch.capture_enabled = False
        patch.quantize_enabled = False
        del cache
        gc.collect(); torch.cuda.empty_cache()
    return {"tokens": len(tokens), "rows": rows_written, "prefill_max_abs_logit_diff": prefill_diff, "quantization": bool(quantize), "elapsed_seconds": time.time() - started}


def parity(model, tokenizer, patch, rows):
    row = rows[0]
    path = OUT / "source_audit/replay_parity_capture.parquet"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    meta = capture_trajectory(model, tokenizer, patch, row, path, quantize=True, max_tokens=128)
    current = pd.read_parquet(path)
    prior = pd.read_csv(PRIOR / "token_level_metrics.csv", nrows=25000)
    prior = prior[(prior["problem_id"] == row["problem_id"]) & (prior["seed"] == int(row["seed"])) & (prior["timestep"] < 128)]
    checks = {}
    passed = True
    for cond in CONDITIONS:
        cur = current[current["condition"] == cond]
        agg = cur.groupby("timestep").agg(
            token_id=("token_id", "first"), m0_layer_p95=("m0", lambda x: np.quantile(x, .95)),
            m5_layer_p95=("m5_canonical", lambda x: np.quantile(x, .95)), fp_margin=("fp_margin", "first"),
            delta_z_fp_winner=("delta_z_fp_winner", "first"), top1_changed=("top1_changed", "first"),
        ).reset_index().sort_values("timestep")
        old = prior[prior["condition"] == cond].sort_values("timestep")
        errors = {
            "m0_layer_p95_max_abs": float(np.max(np.abs(agg["m0_layer_p95"].to_numpy() - old["m0_layer_p95"].to_numpy()))),
            "m5_layer_p95_max_abs": float(np.max(np.abs(agg["m5_layer_p95"].to_numpy() - old["m5_layer_p95"].to_numpy()))),
            "fp_margin_max_abs": float(np.max(np.abs(agg["fp_margin"].to_numpy() - old["fp_margin"].to_numpy()))),
            "delta_z_max_abs": float(np.max(np.abs(agg["delta_z_fp_winner"].to_numpy() - old["delta_z_fp_winner"].to_numpy()))),
            "token_ids_exact": bool(np.array_equal(agg["token_id"].to_numpy(), old["token_id"].to_numpy())),
            "top1_changed_exact": bool(np.array_equal(agg["top1_changed"].to_numpy(), old["top1_changed"].to_numpy())),
        }
        ok = errors["m0_layer_p95_max_abs"] <= 1e-5 and errors["m5_layer_p95_max_abs"] <= 1e-4 and errors["fp_margin_max_abs"] <= 1e-5 and errors["delta_z_max_abs"] <= 1e-4 and errors["token_ids_exact"] and errors["top1_changed_exact"]
        errors["status"] = "PASS" if ok else "FAIL"
        checks[cond] = errors
        passed &= ok
    result = {"REPLAY_PARITY_GATE": "PASS" if passed else "FAIL", "sample": {"problem_id": row["problem_id"], "seed": row["seed"], "tokens": 128}, "checks": checks, "capture_meta": meta, "labels_loaded": False, "new_generation": False, "checked_at": datetime.now(timezone.utc).isoformat()}
    save_json(OUT / "source_audit/replay_parity.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("REPLAY_PARITY_GATE_FAIL")


def fp_audit(model, tokenizer, patch, rows, audit_slot):
    parity_result = json.loads((OUT / "source_audit/replay_parity.json").read_text(encoding="utf-8"))
    if parity_result["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("FP audit requires replay parity PASS")
    source_index = FP_AUDIT_INDICES[audit_slot]
    row = rows[source_index]
    directory = OUT / "coordinate_audit/fp_equivalent_replay"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{source_index:03d}_{row['problem_id']}_seed{int(row['seed'])}"
    path = directory / f"{stem}.parquet"
    meta_path = directory / f"{stem}.json"
    if path.exists() and meta_path.exists():
        print(json.dumps({"event": "fp_audit_skip_complete", "source_index": source_index}), flush=True)
        return
    if path.exists() or meta_path.exists():
        raise RuntimeError(f"partial FP audit unit exists: {stem}")
    meta = capture_trajectory(model, tokenizer, patch, row, path, quantize=False, max_tokens=512)
    meta.update({"source_index": source_index, "problem_id": row["problem_id"], "seed": int(row["seed"]), "labels_loaded": False})
    save_json(meta_path, meta)


def formal_shard(model, tokenizer, patch, rows, shard_index, shard_count):
    parity_result = json.loads((OUT / "source_audit/replay_parity.json").read_text(encoding="utf-8"))
    if parity_result["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("formal replay requires parity PASS")
    audit_files = list((OUT / "coordinate_audit/fp_equivalent_replay").glob("*.json"))
    if len(audit_files) != len(FP_AUDIT_INDICES):
        raise RuntimeError("formal replay requires all preregistered FP-equivalent audit units")
    assigned = [(i, row) for i, row in enumerate(rows) if i % shard_count == shard_index]
    directory = OUT / "component_replay" / f"shard_{shard_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    completed = 0
    started = time.time()
    for source_index, row in assigned:
        free = shutil.disk_usage("/data").free
        if free < 20 * (1 << 30):
            save_json(status_path, {"status": "DISK_SAFETY_BLOCK", "free_bytes": free, "completed": completed, "assigned": len(assigned)})
            raise RuntimeError("DISK_SAFETY_BLOCK")
        stem = f"{source_index:03d}_{row['problem_id']}_seed{int(row['seed'])}"
        path = directory / f"{stem}.parquet"
        meta_path = directory / f"{stem}.json"
        if path.exists() and meta_path.exists():
            completed += 1
            continue
        if path.exists() or meta_path.exists():
            raise RuntimeError(f"partial component unit exists: {stem}")
        print(json.dumps({"event": "trajectory_start", "source_index": source_index, "problem_id": row["problem_id"], "seed": row["seed"], "tokens": len(row["generated_token_ids"])}), flush=True)
        meta = capture_trajectory(model, tokenizer, patch, row, path, quantize=True)
        meta.update({"source_index": source_index, "problem_id": row["problem_id"], "seed": int(row["seed"]), "labels_loaded": False})
        save_json(meta_path, meta)
        completed += 1
        save_json(status_path, {"status": "RUNNING", "shard": shard_index, "completed": completed, "assigned": len(assigned), "last_unit": stem, "elapsed_seconds": time.time() - started, "free_bytes": shutil.disk_usage('/data').free})
    save_json(status_path, {"status": "COMPLETE", "shard": shard_index, "completed": completed, "assigned": len(assigned), "elapsed_seconds": time.time() - started, "labels_loaded": False, "new_generation": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parity", "fp-audit", "formal"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = source_rows()
    model, tokenizer = C.load_model(torch)
    with PrimitivePatch(torch, model) as patch:
        if args.mode == "parity":
            parity(model, tokenizer, patch, rows)
        elif args.mode == "fp-audit":
            fp_audit(model, tokenizer, patch, rows, args.shard_index)
        else:
            formal_shard(model, tokenizer, patch, rows, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
