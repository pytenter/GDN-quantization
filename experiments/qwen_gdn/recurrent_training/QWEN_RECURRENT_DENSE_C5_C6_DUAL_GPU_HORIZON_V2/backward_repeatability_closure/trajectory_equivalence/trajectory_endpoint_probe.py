#!/usr/bin/env python3
"""Fixed non-AIME teacher-forced probe of one eight-update endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT.parents[1]
CLOSURE = ROOT.parent
ANALYSIS = ROOT / "analysis"
RAW = ROOT / "raw"
CONFIGS = ROOT / "configs"


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = import_file(V2_DIR / "run_dual_gpu_v2.py", "qwen_v2_endpoint_probe_read_only")
V1 = V2.V1_IMPL
LAYERS = V1.GDN_LAYERS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def new_json(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def new_npz(path: Path, tensors: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **tensors)


def run(mode: str, run_id: int) -> None:
    stem = f"{mode}_{run_id:02d}"
    trajectory_path = ANALYSIS / f"{stem}.json"
    trajectory = json.loads(trajectory_path.read_text())
    protocol_path = CONFIGS / "frozen_trajectory_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if trajectory["protocol_sha256"] != sha(protocol_path):
        raise RuntimeError("protocol mismatch")
    metadata_path = ANALYSIS / f"{stem}_probe.json"
    raw_path = RAW / f"{stem}_probe.npz"
    error_path = ANALYSIS / f"{stem}_probe.error.json"
    if metadata_path.exists() or raw_path.exists() or error_path.exists():
        raise RuntimeError(f"refusing duplicate probe {stem}")
    random.seed(V1.TRAIN_SEED)
    torch.manual_seed(V1.TRAIN_SEED)
    if mode == "single":
        model, tokenizer, bank, device = V1.load_model_bank()
        patch_factory = lambda: V1.RecurrentExperimentPatch(model, bank, V1.hadamard(device))
    else:
        model, tokenizer, bank, layer_devices = V2.load_dual_model_bank(protocol["dual_gpu_contiguous_split_block"])
        device = model.get_input_embeddings().weight.device
        patch_factory = lambda: V2.DualRecurrentPatch(model, bank)
    endpoint_path = RAW / trajectory["endpoint_theta_artifact"]
    if sha(endpoint_path) != trajectory["endpoint_theta_sha256"]:
        raise RuntimeError("endpoint theta hash mismatch")
    with np.load(endpoint_path, allow_pickle=False) as arrays, torch.no_grad():
        for layer in LAYERS:
            target = bank.layer(layer).theta
            source = torch.from_numpy(arrays[str(layer)]).to(device=target.device, dtype=target.dtype)
            target.copy_(source)
    reference = json.loads((CLOSURE / "configs/frozen_backward_reference.json").read_text())
    row = V1.corpus_rows("TRAIN")[0]
    ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:192]
    positions = V1.sampled_positions(row, len(ids))
    if ids != reference["input_token_ids"] or positions != reference["target_positions"]:
        raise RuntimeError("fixed probe tokens mismatch")
    trace_set = set(reference["trace_token_indices"])
    position_set = set(positions)
    raw = {}
    hashes = {}
    losses = []
    current_token = -1
    cache = None
    original_qdq = V1.CAYLEY.qwen_c128_ste
    try:
        with torch.no_grad(), patch_factory() as patch:
            targets = V1.collect_teacher_targets(model, patch, ids, positions)
            teacher_hashes = {}
            for pos in sorted(targets):
                for part in ("prev", "out"):
                    for layer, value in sorted(targets[pos][part].items()):
                        teacher_hashes[f"{pos}:{part}:{layer}"] = tensor_hash(value)

            def traced_qdq(value):
                quant = original_qdq(value)
                if current_token in trace_set:
                    layer = int(patch.current_layer)
                    for name, tensor in (("state", value), ("scale", quant.scale),
                                         ("qcodes", quant.codes), ("post_qdq", quant.dequant)):
                        key = f"{name}_{current_token}_{layer}"
                        raw[key] = tensor.detach().contiguous().cpu().numpy().copy()
                        hashes[key] = tensor_hash(tensor)
                return quant

            def logits_hook(_module, _args, output):
                if current_token in position_set:
                    key = f"logits_{current_token}"
                    raw[key] = output.detach().float().contiguous().cpu().numpy().copy()
                    hashes[key] = tensor_hash(output.detach().float())

            handle = model.lm_head.register_forward_hook(logits_hook)
            V1.CAYLEY.qwen_c128_ste = traced_qdq
            try:
                for index, token_id in enumerate(ids):
                    current_token = index
                    capture = index in position_set
                    if capture:
                        if mode == "single":
                            V1.install_teacher_target(patch, targets[index], device)
                        else:
                            V2.install_dual_teacher_target(patch, targets[index], layer_devices)
                    cache = V1.forward_student(model, patch, token_id, cache, capture)
                    if capture:
                        state, functional, c5, c6 = patch.captured_losses()
                        losses.append({"token_index": index,
                                       "state": float(state.detach().cpu()),
                                       "functional": float(functional.detach().cpu()),
                                       "c5": float(c5.detach().cpu()),
                                       "c6": float(c6.detach().cpu())})
                        patch.clear_capture()
                    if (index+1) % 32 == 0:
                        V1.detach_cache(cache)
            finally:
                handle.remove()
                V1.CAYLEY.qwen_c128_ste = original_qdq
        if len(losses) != 8:
            raise RuntimeError("probe loss count mismatch")
        expected_keys = 10 * len(LAYERS) * 4 + 8
        if len(raw) != expected_keys:
            raise RuntimeError(f"probe tensor count {len(raw)} != {expected_keys}")
        new_npz(raw_path, raw)
        new_json(metadata_path, {
            "stem": stem, "mode": mode,
            "trajectory_sha256": sha(trajectory_path),
            "endpoint_theta_sha256": sha(endpoint_path),
            "protocol_sha256": sha(protocol_path),
            "input_ids_sha256": json_hash(ids), "teacher_ids_sha256": json_hash(ids),
            "target_positions_sha256": json_hash(positions),
            "teacher_target_hashes": teacher_hashes,
            "loss_components": losses,
            "tensor_hashes": hashes,
            "raw_artifact": raw_path.name, "raw_artifact_sha256": sha(raw_path),
        })
        print(json.dumps({"stem": stem, "PROBE": "COMPLETE", "tensors": len(raw)}, indent=2), flush=True)
    except Exception as exc:
        if not error_path.exists():
            new_json(error_path, {"exception": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()})
        raise
    finally:
        V1.CAYLEY.qwen_c128_ste = original_qdq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "dual"), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    run(args.mode, args.run_id)


if __name__ == "__main__":
    main()
