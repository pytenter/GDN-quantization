#!/usr/bin/env python3
"""Streaming 64-document persistent-headroom evaluation for Qwen/GDN.

This program performs inference only.  It writes one compact JSONL row per
document and never serializes recurrent states, logits, or other tensor traces.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "hadamard_init_dense_orthogonal_oracle_v1"
EXPECTED = {
    "Dense_State": "a3597986bf9fd57e8af8dfcd13cbb8b443e1e641459444510a41ee556955074c",
    "Dense_Functional": "758a5ab138d23fa5dae5666cd51039550ec662b565d9b4f4a3dcbb50ca61cc4c",
}
HORIZONS = (1, 4, 8, 16, 32, 64, 128)
METHODS = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")
GIB = 1 << 30


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = import_file(OLD / "evaluate_qwen_dense_oracle.py", "persistent_qwen_old_eval")
R = E.R


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer) -> dict:
    root = Path(str(getattr(tokenizer, "name_or_path", "")))
    files = {}
    if root.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
            path = root / name
            if path.is_file():
                files[name] = sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name_or_path": str(root),
        "files": files,
        "fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def disk_gate(output_dir: Path) -> dict:
    free = shutil.disk_usage("/data").free
    added = tree_bytes(output_dir)
    status = {
        "data_free_bytes": free,
        "experiment_output_bytes": added,
        "minimum_free_bytes": 8 * GIB,
        "maximum_temporary_bytes": 4 * GIB,
        "DISK_SAFETY_BLOCK": free < 8 * GIB or added > 4 * GIB,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "disk_safety_latest.json", status)
    if status["DISK_SAFETY_BLOCK"]:
        raise RuntimeError("DISK_SAFETY_BLOCK")
    return status


def reference_logits(model, ids: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    values = {}
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = output.past_key_values
        del output
        for step in range(1, 129):
            output = model(input_ids=torch.tensor([[ids[127 + step]]], device=device), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if step in HORIZONS:
                values[step] = output.logits[:, -1].detach().float().cpu()
            del output
    del cache
    return values


def condition_logits(model, patch, ids, device, method, mats, h, reference):
    rows = []
    patch.enabled = E.use_rotation(method)
    patch.mats = mats
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = output.past_key_values
        E.quantize_cache(cache)
        del output
        for step in range(1, 129):
            output = model(input_ids=torch.tensor([[ids[127 + step]]], device=device), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            E.quantize_cache(cache)
            if step in HORIZONS:
                rows.append({"horizon": step, **E.logit_metrics(output.logits[:, -1].detach().float().cpu(), reference[step])})
            del output
    patch.enabled = False
    del cache
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["document_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", default="/data/zypan")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")

    checkpoints = {"Dense_State": Path(args.state_checkpoint), "Dense_Functional": Path(args.functional_checkpoint)}
    observed = {name: sha256(path) for name, path in checkpoints.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"CHECKPOINT_PROVENANCE_MISMATCH: {observed}")

    output_dir = Path(args.output_dir)
    shard_file = output_dir / f"shard_{args.shard_index:02d}.jsonl"
    status_file = output_dir / f"shard_{args.shard_index:02d}_status.json"
    disk_gate(output_dir)
    model, tokenizer, _config, _e2e = R.load_model(args)
    device = model.get_input_embeddings().weight.device
    token_meta = tokenizer_fingerprint(tokenizer)
    state_bank, _ = E.load_bank(checkpoints["Dense_State"], device)
    functional_bank, _ = E.load_bank(checkpoints["Dense_Functional"], device)
    banks = {"Dense_State": state_bank, "Dense_Functional": functional_bank}
    h = R.hadamard(device)
    mats = {method: E.matrices(method, banks, device) for method in METHODS}
    patch = E.RuntimePatch(model, mats["Hadamard"], h)
    patch.install()
    documents = [row for row in R.load_corpus(Path(args.corpus), "HELDOUT") if int(row["panel_index"]) % args.shard_count == args.shard_index]
    done = completed_ids(shard_file)
    peak = tree_bytes(output_dir)
    started = time.time()
    try:
        for ordinal, document in enumerate(documents, 1):
            if document["document_id"] in done:
                continue
            token_ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
            if len(token_ids) < 256:
                raise RuntimeError(f"document too short: {document['document_id']} {len(token_ids)}")
            reference_logits_value, reference_states = E.reference_trajectory(model, token_ids, device, 128)
            method_rows = {}
            for method in METHODS:
                patch.mats = mats[method]
                rows = E.condition_trajectory(
                    model,
                    patch,
                    token_ids,
                    device,
                    method,
                    mats[method],
                    h,
                    reference_logits_value,
                    reference_states,
                    128,
                )
                method_rows[method] = {"auc": E.auc(rows), "horizons": rows}
            row = {
                "task": "PERSISTENT_HEADROOM_CONFIRMATION_V1",
                "model": "Qwen3.5-9B/GDN",
                "document_id": document["document_id"],
                "panel_index": document["panel_index"],
                "original_16": document["original_16"],
                "raw_text_sha256": document["raw_text_sha256"],
                "token_count": len(token_ids),
                "tokenizer_fingerprint_sha256": token_meta["fingerprint_sha256"],
                "methods": method_rows,
                "paired_auc": {
                    "Hadamard_minus_Dense_State": method_rows["Hadamard"]["auc"] - method_rows["Dense_State"]["auc"],
                    "Hadamard_minus_Dense_Functional": method_rows["Hadamard"]["auc"] - method_rows["Dense_Functional"]["auc"],
                },
                "nonfinite_total": sum(metric["nonfinite"] for value in method_rows.values() for metric in value["horizons"]),
            }
            append_jsonl(shard_file, row)
            done.add(document["document_id"])
            del reference_logits_value, reference_states, method_rows, row, token_ids
            gc.collect()
            torch.cuda.empty_cache()
            peak = max(peak, tree_bytes(output_dir))
            save_json(status_file, {
                "status": "RUNNING",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed": len(done),
                "assigned": len(documents),
                "last_document": document["document_id"],
                "checkpoint_sha256": observed,
                "tokenizer": token_meta,
                "peak_output_bytes": peak,
                "elapsed_seconds": time.time() - started,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "training_performed": False,
                "model_weights_changed": False,
                "AIME26_used": False,
            })
            if len(done) % 8 == 0:
                disk_gate(output_dir)
            print(f"SHARD {args.shard_index} {len(done)}/{len(documents)} {document['document_id']}", flush=True)
    finally:
        patch.close()
    save_json(status_file, {
        "status": "COMPLETE",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "completed": len(done),
        "assigned": len(documents),
        "checkpoint_sha256": observed,
        "tokenizer": token_meta,
        "peak_output_bytes": peak,
        "elapsed_seconds": time.time() - started,
        "training_performed": False,
        "model_weights_changed": False,
        "AIME26_used": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
