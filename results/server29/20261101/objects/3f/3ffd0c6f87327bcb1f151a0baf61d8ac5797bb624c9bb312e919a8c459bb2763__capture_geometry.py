#!/usr/bin/env python3
"""Phase-1 scalar geometry observation over frozen teacher-forced Qwen tokens.

The recurrent recurrence, QDQ, and basis transforms are inherited unchanged from
the previously parity-validated capture. No tensor state or logits are stored.
"""
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
OUT = ROOT / "experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1"
STABILITY = ROOT / "experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1"
PERSISTENT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
EPS = 1e-12
CHUNK = 64
SAFETY_GIB = 20


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load_file(STABILITY / "replay_stability_v2.py", "frozen_stability_replay_for_geometry")
P = S.P
C = S.C


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_sha(row):
    tokens = list(map(int, row["generated_token_ids"]))
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable_cos(dot, norm_a, norm_b):
    valid = (norm_a > EPS) & (norm_b > EPS)
    raw = dot / (norm_a * norm_b).clamp_min(EPS)
    nan = torch.full_like(raw, float("nan"))
    return torch.where(valid, raw.clamp(-1.0, 1.0), nan)


class GeometryPatch(S.StabilityPatch):
    """Observe post-QDQ full-layer state errors without changing base _run."""

    def begin_chunk(self):
        super().begin_chunk()
        self.geometry = {}
        self.geometry_calls = {}

    def _condition_errors(self, state):
        en, eh, fp = super()._condition_errors(state)
        if not self.capture_enabled:
            return en, eh, fp
        layer = int(self.current_layer)
        call = self.geometry_calls.get(layer, 0)
        self.geometry_calls[layer] = call + 1
        # Frozen _run calls this exactly twice per token: before recurrent
        # update, then after the existing per-token QDQ. Retain only post-QDQ.
        if call % 2:
            ni = torch.linalg.vector_norm(en)
            nh = torch.linalg.vector_norm(eh)
            nf = torch.linalg.vector_norm(fp)
            entry = torch.stack((
                ni, nh, nf, ni / nf.clamp_min(EPS), nh / nf.clamp_min(EPS),
                stable_cos(torch.sum(en * fp), ni, nf),
                stable_cos(torch.sum(eh * fp), nh, nf),
                stable_cos(torch.sum(en * eh), ni, nh),
            )).detach()
            self.geometry.setdefault(layer, []).append(entry)
        return en, eh, fp

    def _norm_pre(self, layer, module, inputs):
        if not self.capture_enabled:
            return
        rec = self.pending.get(layer)
        if rec is None:
            raise RuntimeError(f"missing recurrent capture at layer {layer}")
        n = len(rec["m0"]["INT8_C128"])
        if self.geometry_calls.get(layer) != 2 * n or len(self.geometry.get(layer, ())) != n:
            raise RuntimeError(f"unexpected pre/post state capture cadence at layer {layer}")
        g = torch.stack(self.geometry[layer], dim=0)
        reference_i = rec["m0"]["INT8_C128"].float()
        reference_h = rec["m0"]["INT8_C128_KEY_HADAMARD"].float()
        if (g[:, 0] - reference_i).abs().max().item() > 1e-5 or (g[:, 1] - reference_h).abs().max().item() > 1e-5:
            raise RuntimeError(f"post-QDQ geometry does not match frozen M0 at layer {layer}")

        readout = inputs[0].detach().float().reshape(3, n, -1)
        rfp = readout[0]
        nr = torch.linalg.vector_norm(rfp, dim=-1)
        extra = {}
        for batch, label in ((1, "identity"), (2, "hadamard")):
            er = readout[batch] - rfp
            ne = torch.linalg.vector_norm(er, dim=-1)
            extra[f"readout_error_norm_{label}"] = ne.cpu().numpy()
            extra[f"cos_readout_error_fp_{label}"] = stable_cos((er * rfp).sum(-1), ne, nr).cpu().numpy()
        names = (
            "error_norm_identity", "error_norm_hadamard", "fp_state_norm",
            "relative_error_identity", "relative_error_hadamard",
            "cos_error_fp_identity", "cos_error_fp_hadamard", "cos_error_identity_hadamard",
        )
        values = {name: g[:, index].cpu().numpy() for index, name in enumerate(names)}
        self.chunk_metrics[int(layer)] = {**values, **extra}
        self.layers_seen.add(int(layer))
        del self.pending[layer]
        del self.geometry[layer]


def capture(model, tokenizer, patch, row, limit=None, stem=None):
    tokens = list(map(int, row["generated_token_ids"]))
    if limit is not None:
        tokens = tokens[:limit]
    if len(tokens) < 2:
        raise RuntimeError("trajectory is too short")
    device = next(model.parameters()).device
    cache, prefill_parity = P.prefill(model, tokenizer, patch, row, device)
    patch.capture_enabled = True
    patch.quantize_enabled = True
    writer = None
    frames = []
    started = time.time()
    try:
        for start in range(0, len(tokens) - 1, CHUNK):
            chunk = tokens[start:min(start + CHUNK, len(tokens) - 1)]
            ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0).repeat(3, 1)
            patch.begin_chunk()
            with torch.inference_mode():
                output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if set(patch.chunk_metrics) != set(C.GDN_LAYERS):
                raise RuntimeError("missing or extra GDN layer geometry")
            base = {
                "problem_id": row["problem_id"], "seed": int(row["seed"]),
                "canonical_index": int(row["canonical_index"]),
                "token_position": np.arange(start, start + len(chunk), dtype=np.int32),
                "input_token_id": np.asarray(chunk, dtype=np.int32),
                "target_token_id": np.asarray(tokens[start + 1:start + len(chunk) + 1], dtype=np.int32),
            }
            pieces = [pd.DataFrame({**base, "layer": int(layer), **patch.chunk_metrics[layer]}) for layer in C.GDN_LAYERS]
            frame = pd.concat(pieces, ignore_index=True)
            if stem is None:
                frames.append(frame)
            else:
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(stem) + ".parquet.tmp", table.schema, compression="zstd")
                writer.write_table(table)
            del output, ids, frame, pieces
            if start % 4096 == 0:
                print(json.dumps({"event": "geometry_progress", "problem_id": row["problem_id"],
                                  "seed": row["seed"], "tokens": start + len(chunk),
                                  "total": len(tokens) - 1,
                                  "tokens_per_s": (start + len(chunk)) / max(time.time() - started, 1e-6)}), flush=True)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(str(stem) + ".parquet.tmp", str(stem) + ".parquet")
    finally:
        if writer is not None:
            writer.close()
        patch.capture_enabled = False
        patch.quantize_enabled = False
        patch.begin_chunk()
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    return (pd.concat(frames, ignore_index=True) if stem is None else None), prefill_parity


def audit(rows):
    prereg = OUT / "preregistration.json"
    if not prereg.exists():
        raise RuntimeError("preregistration missing")
    prior = json.loads((STABILITY / "results.json").read_text(encoding="utf-8"))
    stable_manifest = json.loads((STABILITY / "run_manifest.json").read_text(encoding="utf-8"))
    prior_gate = json.loads((STABILITY / "replay_parity/parity_results.json").read_text(encoding="utf-8"))
    if prior["row_audit"]["trajectories"] != 60 or not prior["row_audit"]["token_pairing_exact"]:
        raise RuntimeError("prior paired trajectory audit failed")
    if not prior["row_audit"]["all_unit_hashes_valid"] or prior_gate["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("prior unit hashes or replay parity failed")
    if sha256(STABILITY / "results.json") != stable_manifest["outputs"]["results.json"]["sha256"]:
        raise RuntimeError("prior final result hash mismatch")
    if len(rows) != 60:
        raise RuntimeError("source trajectory count changed")
    anchor = json.loads((PERSISTENT / "anchor_protocol.json").read_text(encoding="utf-8"))["trajectory_anchors"]
    lookup = {(x["problem_id"], int(x["seed"])): x["token_ids_sha256"] for x in anchor}
    checked = 0
    positions = 0
    for index, row in enumerate(rows):
        key = (row["problem_id"], int(row["seed"]))
        if key not in lookup or token_sha(row) != lookup[key]:
            raise RuntimeError(f"source token hash mismatch {key}")
        stem = STABILITY / "trajectory_metrics" / f"shard_{index % 4:02d}" / f"{index:03d}_{key[0]}_seed{key[1]}"
        meta_path = Path(str(stem) + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["token_ids_sha256"] != lookup[key]:
            raise RuntimeError(f"prior replay token hash mismatch {key}")
        for kind in ("layers", "logits"):
            path = Path(str(stem) + f"_{kind}.parquet")
            if sha256(path) != meta[f"{kind[:-1] if kind == 'layers' else 'logit'}_file_sha256"]:
                raise RuntimeError(f"prior replay parquet hash mismatch {path}")
        checked += 1
        positions += len(row["generated_token_ids"]) - 1
    result = {
        "task": "QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1",
        "phase": "PHASE_1",
        "source_trajectories": checked,
        "token_positions_per_condition": positions,
        "expected_paired_layer_rows": positions * len(C.GDN_LAYERS),
        "gdn_layers": list(map(int, C.GDN_LAYERS)),
        "layer_count": len(C.GDN_LAYERS),
        "token_hashes_verified": checked,
        "prior_unit_files_hash_verified": checked * 2,
        "same_teacher_forced_history": True,
        "native_basis_recovery": "C.recover_hadamard in frozen BatchedMechanismPatch._condition_errors",
        "frozen_checkpoint_sha256": json.loads((STABILITY / "artifact_manifest.json").read_text(encoding="utf-8"))["checkpoint_combined_sha256"],
        "prior_results_sha256": sha256(STABILITY / "results.json"),
        "prior_replay_script_sha256": sha256(STABILITY / "replay_stability_v2.py"),
        "capture_script_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(prereg),
        "hash_verification": "PASS",
        "pairing_status": "PASS",
        "new_generation": False,
    }
    save_json(OUT / "analysis/replay_audit.json", result)
    print(json.dumps({"event": "geometry_audit_complete", "trajectories": checked,
                      "token_positions_per_condition": positions, "layer_count": len(C.GDN_LAYERS)}), flush=True)


def parity(model, tokenizer, patch, rows):
    audit_info = json.loads((OUT / "analysis/replay_audit.json").read_text(encoding="utf-8"))
    if audit_info["hash_verification"] != "PASS":
        raise RuntimeError("artifact audit not passed")
    row = rows[0]
    a, pre_a = capture(model, tokenizer, patch, row, limit=128)
    b, pre_b = capture(model, tokenizer, patch, row, limit=128)
    if len(a) != 127 * 24 or len(b) != len(a):
        raise RuntimeError("parity layer-time grid incomplete")
    fields = ("error_norm_identity", "error_norm_hadamard", "fp_state_norm",
              "relative_error_identity", "relative_error_hadamard",
              "cos_error_fp_identity", "cos_error_fp_hadamard",
              "cos_error_identity_hadamard", "readout_error_norm_identity",
              "readout_error_norm_hadamard", "cos_readout_error_fp_identity",
              "cos_readout_error_fp_hadamard")
    repeat_max = max(float(np.nanmax(np.abs(a[x].to_numpy() - b[x].to_numpy()))) for x in fields)
    first = STABILITY / "trajectory_metrics/shard_00/000_aime26_01_seed1_layers.parquet"
    old = pd.read_parquet(first)
    old = old[old.token_position < 127]
    old_i = old[old.rotation == "Identity"].sort_values(["token_position", "layer"]).reset_index(drop=True)
    old_h = old[old.rotation == "Hadamard"].sort_values(["token_position", "layer"]).reset_index(drop=True)
    now = a.sort_values(["token_position", "layer"]).reset_index(drop=True)
    if not (np.array_equal(now.token_position.to_numpy(), old_i.token_position.to_numpy())
            and np.array_equal(now.layer.to_numpy(), old_i.layer.to_numpy())
            and np.array_equal(now.input_token_id.to_numpy(), old_i.input_token_id.to_numpy())
            and np.array_equal(now.target_token_id.to_numpy(), old_i.target_token_id.to_numpy())):
        raise RuntimeError("geometry/prior replay token or layer pairing differs")
    checks = {
        "repeat_max_abs": repeat_max,
        "identity_norm_vs_prior_max_abs": float(np.max(np.abs(now.error_norm_identity - old_i.state_error_norm))),
        "hadamard_norm_vs_prior_max_abs": float(np.max(np.abs(now.error_norm_hadamard - old_h.state_error_norm))),
        "identity_relative_vs_prior_max_abs": float(np.max(np.abs(now.relative_error_identity - old_i.D_state))),
        "hadamard_relative_vs_prior_max_abs": float(np.max(np.abs(now.relative_error_hadamard - old_h.D_state))),
        "token_and_layer_pairing_exact": True,
        "source_token_hash_match": token_sha(row) == json.loads((PERSISTENT / "anchor_protocol.json").read_text(encoding="utf-8"))["trajectory_anchors"][0]["token_ids_sha256"],
        "prefill_parity_a": pre_a,
        "prefill_parity_b": pre_b,
        "rotation_orthogonality_max_abs": float((patch.h.T @ patch.h - torch.eye(128, device=patch.h.device)).abs().max().cpu()),
    }
    core = ("error_norm_identity", "error_norm_hadamard", "fp_state_norm",
            "relative_error_identity", "relative_error_hadamard")
    checks["nonfinite_core"] = int(sum((~np.isfinite(now[k].to_numpy())).sum() for k in core))
    checks["cosine_out_of_range"] = int(sum((np.abs(now[k].to_numpy()[np.isfinite(now[k].to_numpy())]) > 1.000001).sum()
                                             for k in fields if k.startswith("cos_")))
    result = {
        "sample": {"problem_id": row["problem_id"], "seed": int(row["seed"]), "tokens": 127, "layers": 24},
        "checks": checks,
        "REPLAY_PARITY_GATE": "PASS" if all((repeat_max <= 1e-5,
            checks["identity_norm_vs_prior_max_abs"] <= 1e-5,
            checks["hadamard_norm_vs_prior_max_abs"] <= 1e-5,
            checks["identity_relative_vs_prior_max_abs"] <= 1e-5,
            checks["hadamard_relative_vs_prior_max_abs"] <= 1e-5,
            checks["source_token_hash_match"], checks["nonfinite_core"] == 0,
            checks["cosine_out_of_range"] == 0,
            checks["rotation_orthogonality_max_abs"] <= 1e-5)) else "FAIL",
        "new_generation": False,
    }
    save_json(OUT / "analysis/replay_parity.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("geometry parity gate failed")


def shard(model, tokenizer, patch, rows, index, count):
    gate = json.loads((OUT / "analysis/replay_parity.json").read_text(encoding="utf-8"))
    if gate["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("formal replay requires parity PASS")
    assigned = [(i, row) for i, row in enumerate(rows) if i % count == index]
    directory = OUT / "analysis" / f"shard_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    completed = 0
    started = time.time()
    for global_index, row in assigned:
        free = shutil.disk_usage(OUT).free / 1024**3
        if free < SAFETY_GIB:
            raise RuntimeError(f"disk safety stop at {free:.2f} GiB")
        stem = directory / f"{global_index:03d}_{row['problem_id']}_seed{int(row['seed'])}"
        data_path = Path(str(stem) + ".parquet")
        meta_path = Path(str(stem) + "_meta.json")
        if data_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if sha256(data_path) != meta["parquet_sha256"] or token_sha(row) != meta["token_ids_sha256"]:
                raise RuntimeError(f"existing completed unit failed hash check: {stem}")
            completed += 1
            continue
        if data_path.exists() or meta_path.exists() or Path(str(stem) + ".parquet.tmp").exists():
            raise RuntimeError(f"partial unit exists, will not overwrite: {stem}")
        print(json.dumps({"event": "geometry_start", "index": global_index,
                          "problem_id": row["problem_id"], "seed": int(row["seed"]),
                          "tokens": len(row["generated_token_ids"])}), flush=True)
        _, prefill_check = capture(model, tokenizer, patch, row, stem=stem)
        meta = {"global_index": global_index, "problem_id": row["problem_id"],
                "seed": int(row["seed"]), "generated_tokens": len(row["generated_token_ids"]),
                "rows": (len(row["generated_token_ids"]) - 1) * len(C.GDN_LAYERS),
                "token_ids_sha256": token_sha(row), "parquet_sha256": sha256(data_path),
                "prefill_parity": prefill_check, "labels_loaded": False,
                "new_generation": False}
        save_json(meta_path, meta)
        completed += 1
        save_json(status, {"status": "RUNNING", "shard": index, "assigned": len(assigned),
                           "completed": completed, "elapsed_seconds": time.time() - started})
        gc.collect()
        torch.cuda.empty_cache()
    save_json(status, {"status": "COMPLETE", "shard": index, "assigned": len(assigned),
                       "completed": completed, "elapsed_seconds": time.time() - started,
                       "new_generation": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "parity", "shard"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = P.load_source_rows()
    if args.mode == "audit":
        audit(rows)
        return
    model, tokenizer = C.load_model(torch)
    with GeometryPatch(torch, model) as patch:
        if args.mode == "parity":
            parity(model, tokenizer, patch, rows)
        else:
            shard(model, tokenizer, patch, rows, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
