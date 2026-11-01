#!/usr/bin/env python3
"""Observation-only, deterministic teacher-forced trajectory stability replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1"
PRIOR = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
M5 = ROOT / "experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"
CONDITIONS = ((1, "Identity"), (2, "Hadamard"))
CHUNK = 64
EPS = 1e-12


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load_file(PRIOR / "replay_persistent_functional_v3.py", "prior_persistent_replay")
C = P.C


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class StabilityPatch(C.BatchedMechanismPatch):
    """Reuse frozen recurrent semantics; only observe norm input and saved state errors."""

    def _norm_pre(self, layer, module, inputs):
        if not self.capture_enabled:
            return
        rec = self.pending.get(layer)
        if rec is None:
            raise RuntimeError(f"missing recurrent capture for layer {layer}")
        raw = inputs[0].detach().float()
        length = len(rec["m0_rel"]["INT8_C128"])
        readout = raw.reshape(3, length, -1)
        fp = readout[0]
        fp_norm = torch.linalg.vector_norm(fp, dim=-1)
        result = {}
        for batch, label, prior_label in (
            (1, "Identity", "INT8_C128"),
            (2, "Hadamard", "INT8_C128_KEY_HADAMARD"),
        ):
            numerator = torch.linalg.vector_norm(readout[batch] - fp, dim=-1)
            result[label] = {
                "D_state": rec["m0_rel"][prior_label].detach().float().cpu().numpy(),
                "state_error_norm": rec["m0"][prior_label].detach().float().cpu().numpy(),
                "D_readout": (numerator / fp_norm.clamp_min(EPS)).cpu().numpy(),
                "readout_error_norm": numerator.cpu().numpy(),
                "fp_readout_norm": fp_norm.cpu().numpy(),
            }
        self.chunk_metrics[int(layer)] = result
        self.layers_seen.add(int(layer))
        del self.pending[layer]


def logits_metrics(logits):
    fp = logits[0].detach().float()
    logp = torch.log_softmax(fp, -1)
    p = logp.exp()
    fp_entropy = -(p * logp).sum(-1)
    fptop = torch.topk(fp, 2, -1)
    fp_id = fptop.indices[:, 0]
    fp_margin = fptop.values[:, 0] - fptop.values[:, 1]
    result = {}
    for batch, label in CONDITIONS:
        q = logits[batch].detach().float()
        logq = torch.log_softmax(q, -1)
        qprob = logq.exp()
        qtop = torch.topk(q, 2, -1)
        q_selected = q.gather(-1, fp_id.unsqueeze(-1)).squeeze(-1)
        competitor = torch.where(qtop.indices[:, 0] == fp_id, qtop.values[:, 1], qtop.values[:, 0])
        result[label] = {
            "D_logit_KL": (p * (logp - logq)).sum(-1).cpu().numpy(),
            "top1_agreement": (q.argmax(-1) == fp_id).to(torch.int8).cpu().numpy(),
            "fp_margin": fp_margin.cpu().numpy(),
            "q_margin_on_fp_top1": (q_selected - competitor).cpu().numpy(),
            "q_top1_margin": (qtop.values[:, 0] - qtop.values[:, 1]).cpu().numpy(),
            "fp_entropy": fp_entropy.cpu().numpy(),
            "q_entropy": -(qprob * logq).sum(-1).cpu().numpy(),
        }
    return result


def capture(model, tokenizer, patch, row, limit=None, quantize=True, output_stem=None):
    tokens = list(map(int, row["generated_token_ids"]))
    if limit is not None:
        tokens = tokens[:limit]
    if len(tokens) < 2:
        raise RuntimeError("trajectory requires at least two tokens")
    device = next(model.parameters()).device
    cache, prefill_parity = P.prefill(model, tokenizer, patch, row, device)
    patch.capture_enabled = True
    patch.quantize_enabled = bool(quantize)
    layer_writer = logit_writer = None
    all_layer_frames, all_logit_frames = [], []
    started = time.time()
    try:
        for start in range(0, len(tokens) - 1, CHUNK):
            chunk = tokens[start:min(start + CHUNK, len(tokens) - 1)]
            ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
            patch.begin_chunk()
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            missing = set(C.GDN_LAYERS) - set(patch.chunk_metrics)
            if missing:
                raise RuntimeError(f"missing GDN readout layers: {sorted(missing)}")
            zm = logits_metrics(output.logits)
            base = {
                "problem_id": row["problem_id"], "seed": int(row["seed"]),
                "canonical_index": int(row["canonical_index"]),
                "token_position": np.arange(start, start + len(chunk), dtype=np.int32),
                "input_token_id": np.asarray(chunk, dtype=np.int32),
                "target_token_id": np.asarray(tokens[start + 1:start + 1 + len(chunk)], dtype=np.int32),
            }
            layer_frames, logit_frames = [], []
            for _, label in CONDITIONS:
                logit_frames.append(pd.DataFrame({**base, "rotation": label, **zm[label]}))
                for layer in C.GDN_LAYERS:
                    layer_frames.append(pd.DataFrame({**base, "rotation": label, "layer": int(layer), **patch.chunk_metrics[layer][label]}))
            layer_frame = pd.concat(layer_frames, ignore_index=True)
            logit_frame = pd.concat(logit_frames, ignore_index=True)
            if output_stem is None:
                all_layer_frames.append(layer_frame)
                all_logit_frames.append(logit_frame)
            else:
                layer_table = pa.Table.from_pandas(layer_frame, preserve_index=False)
                logit_table = pa.Table.from_pandas(logit_frame, preserve_index=False)
                if layer_writer is None:
                    layer_writer = pq.ParquetWriter(str(output_stem) + "_layers.parquet.tmp", layer_table.schema, compression="zstd")
                    logit_writer = pq.ParquetWriter(str(output_stem) + "_logits.parquet.tmp", logit_table.schema, compression="zstd")
                layer_writer.write_table(layer_table)
                logit_writer.write_table(logit_table)
            del output, ids, zm, layer_frame, logit_frame
            if start % 4096 == 0:
                print(json.dumps({"event": "stability_progress", "problem_id": row["problem_id"], "seed": row["seed"], "tokens": start + len(chunk), "total": len(tokens) - 1, "tokens_per_s": (start + len(chunk)) / max(time.time() - started, 1e-6)}), flush=True)
        if layer_writer is not None:
            layer_writer.close(); logit_writer.close()
            layer_writer = logit_writer = None
            os.replace(str(output_stem) + "_layers.parquet.tmp", str(output_stem) + "_layers.parquet")
            os.replace(str(output_stem) + "_logits.parquet.tmp", str(output_stem) + "_logits.parquet")
    finally:
        if layer_writer is not None:
            layer_writer.close()
        if logit_writer is not None:
            logit_writer.close()
        patch.capture_enabled = False
        patch.quantize_enabled = False
        patch.begin_chunk()
        del cache
        gc.collect(); torch.cuda.empty_cache()
    if output_stem is None:
        return pd.concat(all_layer_frames, ignore_index=True), pd.concat(all_logit_frames, ignore_index=True), prefill_parity
    return {"layers": str(output_stem) + "_layers.parquet", "logits": str(output_stem) + "_logits.parquet", "prefill_parity": prefill_parity}


def parity(model, tokenizer, patch, rows):
    out = OUT / "replay_parity"
    out.mkdir(parents=True, exist_ok=True)
    row = rows[0]
    a, za, pref_a = capture(model, tokenizer, patch, row, limit=128)
    b, zb, pref_b = capture(model, tokenizer, patch, row, limit=128)
    fp_equiv, _, _ = capture(model, tokenizer, patch, row, limit=128, quantize=False)
    m5_path = next((M5 / "component_replay").glob("shard_*/000_*parquet"))
    old_m5 = pd.read_parquet(m5_path)
    old_m5 = old_m5[(old_m5.timestep < 127) & (old_m5.layer.isin(C.GDN_LAYERS))]
    old_m5 = old_m5.rename(columns={"condition": "rotation", "timestep": "token_position"})
    old_m5["rotation"] = old_m5["rotation"].map({"INT8_C128": "Identity", "INT8_C128_KEY_HADAMARD": "Hadamard"})
    merged = a.merge(old_m5[["rotation", "token_position", "layer", "m0_relative"]], on=["rotation", "token_position", "layer"], validate="one_to_one")
    if len(merged) != len(a):
        raise RuntimeError("state reference parity pairing incomplete")
    state_max = float(np.max(np.abs(merged.D_state.to_numpy() - merged.m0_relative.to_numpy())))
    old_path = next((PRIOR / "replay").glob("shard_*/000_*_local.parquet"))
    old = pd.read_parquet(old_path)
    joined = za.merge(old[["candidate", "input_timestep", "kl", "top1_changed"]].rename(columns={"candidate": "rotation", "input_timestep": "token_position"}), on=["rotation", "token_position"], validate="one_to_one")
    joined = joined[joined.token_position < 127]
    if len(joined) != len(za):
        raise RuntimeError("logit reference parity pairing incomplete")
    kl_max = float(np.max(np.abs(joined.D_logit_KL.to_numpy() - joined.kl.to_numpy())))
    top1_exact = bool(np.array_equal((1 - joined.top1_agreement.to_numpy()).astype(int), joined.top1_changed.to_numpy().astype(int)))
    repeat_max = max(float(np.max(np.abs(a[col].to_numpy() - b[col].to_numpy()))) for col in ("D_state", "D_readout"))
    repeat_max = max(repeat_max, float(np.max(np.abs(za.D_logit_KL.to_numpy() - zb.D_logit_KL.to_numpy()))), float(np.max(np.abs(za.q_entropy.to_numpy() - zb.q_entropy.to_numpy()))))
    fp_readout_p95 = float(np.quantile(fp_equiv.D_readout.to_numpy(), 0.95))
    token_exact = bool(np.array_equal(za.input_token_id.to_numpy(), zb.input_token_id.to_numpy()) and np.array_equal(za.target_token_id.to_numpy(), zb.target_token_id.to_numpy()))
    source_hash = hashlib.sha256(json.dumps(list(map(int, row["generated_token_ids"])), separators=(",", ":")).encode()).hexdigest()
    inventory = json.loads((PRIOR / "anchor_protocol.json").read_text(encoding="utf-8"))["trajectory_anchors"]
    inventory_match = any(x["problem_id"] == row["problem_id"] and int(x["seed"]) == int(row["seed"]) and x["token_ids_sha256"] == source_hash for x in inventory)
    prior_manifest = json.loads((PRIOR / "artifact_manifest.json").read_text(encoding="utf-8"))
    qsrc = inspect.getsource(C.qdq_c128).encode()
    result = {
        "sample": {"problem_id": row["problem_id"], "seed": row["seed"], "positions": 127},
        "checks": {"state_reference_max_abs": state_max, "kl_reference_max_abs": kl_max, "top1_exact": top1_exact, "repeat_max_abs": repeat_max, "fp_equivalent_readout_relative_p95": fp_readout_p95, "token_exact": token_exact, "source_trajectory_hash_match": inventory_match, "prefill_parity_a": pref_a, "prefill_parity_b": pref_b},
        "provenance": {"prior_artifact_audit": prior_manifest["ARTIFACT_AUDIT"], "prior_replay_parity": json.loads((PRIOR / "replay_parity/parity_results.json").read_text())["REPLAY_PARITY_GATE"], "checkpoint_combined_sha256": prior_manifest["checkpoint_combined_sha256"], "quantizer_function_sha256": hashlib.sha256(qsrc).hexdigest(), "quantizer_source_sha256": digest(C.__file__), "rotation_h128_sha256": hashlib.sha256(patch.h.detach().cpu().numpy().tobytes()).hexdigest(), "rotation_orthogonality_max_abs": float((patch.h.T @ patch.h - torch.eye(128, device=patch.h.device)).abs().max().cpu()), "source_token_ids_sha256": source_hash},
        "new_generation": False,
    }
    result["REPLAY_PARITY_GATE"] = "PASS" if all((state_max <= 1e-5, kl_max <= 1e-5, top1_exact, repeat_max <= 1e-5, fp_readout_p95 <= 0.03, token_exact, inventory_match, result["provenance"]["prior_artifact_audit"] == "PASS", result["provenance"]["prior_replay_parity"] == "PASS", result["provenance"]["rotation_orthogonality_max_abs"] <= 1e-5)) else "FAIL"
    save_json(out / "parity_results.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("REPLAY_PARITY_GATE_FAIL")


def shard(model, tokenizer, patch, rows, index, count):
    gate = json.loads((OUT / "replay_parity/parity_results.json").read_text(encoding="utf-8"))
    if gate["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("formal stability replay requires parity PASS")
    assigned = [(i, row) for i, row in enumerate(rows) if i % count == index]
    directory = OUT / "trajectory_metrics" / f"shard_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    completed = 0
    started = time.time()
    for global_index, row in assigned:
        free = shutil.disk_usage(OUT).free / 1024**3
        if free < 20:
            raise RuntimeError(f"disk safety stop at {free:.2f} GiB")
        stem = directory / f"{global_index:03d}_{row['problem_id']}_seed{int(row['seed'])}"
        layer_path = Path(str(stem) + "_layers.parquet")
        logit_path = Path(str(stem) + "_logits.parquet")
        meta_path = Path(str(stem) + "_meta.json")
        if layer_path.exists() and logit_path.exists() and meta_path.exists():
            completed += 1
            continue
        if layer_path.exists() or logit_path.exists() or meta_path.exists():
            raise RuntimeError(f"partial unit exists; no overwrite: {stem}")
        print(json.dumps({"event": "trajectory_start", "index": global_index, "problem_id": row["problem_id"], "seed": row["seed"], "tokens": len(row["generated_token_ids"])}), flush=True)
        output = capture(model, tokenizer, patch, row, output_stem=stem)
        token_hash = hashlib.sha256(json.dumps(list(map(int, row["generated_token_ids"])), separators=(",", ":")).encode()).hexdigest()
        save_json(meta_path, {"global_index": global_index, "problem_id": row["problem_id"], "seed": int(row["seed"]), "generated_tokens": len(row["generated_token_ids"]), "token_ids_sha256": token_hash, "prefill_parity": output["prefill_parity"], "layer_file_sha256": digest(layer_path), "logit_file_sha256": digest(logit_path), "new_generation": False, "labels_loaded": False})
        completed += 1
        save_json(status, {"status": "RUNNING", "assigned": len(assigned), "completed": completed, "shard": index, "elapsed_seconds": time.time() - started})
        gc.collect(); torch.cuda.empty_cache()
    save_json(status, {"status": "COMPLETE", "assigned": len(assigned), "completed": completed, "shard": index, "elapsed_seconds": time.time() - started, "new_generation": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("parity", "shard"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = P.load_source_rows()
    model, tokenizer = C.load_model(torch)
    with StabilityPatch(torch, model) as patch:
        if args.mode == "parity":
            parity(model, tokenizer, patch, rows)
        else:
            shard(model, tokenizer, patch, rows, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
