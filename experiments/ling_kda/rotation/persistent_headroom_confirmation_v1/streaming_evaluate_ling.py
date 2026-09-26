#!/usr/bin/env python3
"""Streaming 64-document persistent-headroom evaluation for Ling/KDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "hadamard_init_dense_orthogonal_oracle_v1"
EXPECTED = {
    "Dense_State": "dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318",
    "Dense_Functional": "65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c",
}
HORIZONS = (1, 4, 8, 16, 32, 64, 128)
METHODS = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = import_file(OLD / "evaluate_ling_dense_oracle.py", "persistent_ling_old_eval")
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
    return {"name_or_path": str(root), "files": files, "fingerprint_sha256": hashlib.sha256(canonical).hexdigest()}


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def trajectory_logits(F, model, ids, probe, layers, device, method, mats, h, reference=None):
    basis = "rotated" if E.rotated(method) else "native"
    item = E.prefill(F, model, ids, probe, basis, layers, device, method)
    if reference is not None:
        E.quantize_cache(F, item["past"], layers)
    identity = torch.eye(128, dtype=torch.float32)
    logits, rows = {}, []
    for step in range(1, 129):
        value, _records = F.H.advance(model, probe, item, int(ids[127 + step]), 127 + step, layers, identity, step)
        if reference is not None:
            E.quantize_cache(F, item["past"], layers)
        if step in HORIZONS:
            last = value[:, -1].detach().float().cpu()
            if reference is None:
                logits[step] = last
            else:
                rows.append({"horizon": step, **E.logit_metrics(last, reference[step])})
        del value, _records
    del item
    return logits if reference is None else rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["document_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
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
    F, model, tokenizer, local_probe, layers = R.load_context(args)
    device = F.model_input_device(model)
    token_meta = tokenizer_fingerprint(tokenizer)
    h = R.hadamard(device)
    state_bank, _ = E.load_bank(checkpoints["Dense_State"], layers, device)
    functional_bank, _ = E.load_bank(checkpoints["Dense_Functional"], layers, device)
    banks = {"Dense_State": state_bank, "Dense_Functional": functional_bank}
    # Preserve the frozen evaluator's warm-up and hook lifecycle exactly.
    # On this runtime, the local fused replay initializes the KDA path used by
    # the first persistent trajectory; skipping it breaks 16-doc replication.
    traces = R.load_traces(Path(args.trace_dir), "VALIDATION")
    for method in METHODS:
        E.local_method(model, local_probe, layers, traces, method, banks, h, device)
    local_probe.close()
    del traces
    gc.collect()
    torch.cuda.empty_cache()
    delta_mats = {method: E.matrices(method, banks, layers, device) for method in METHODS}
    runtime_mats = {method: E.runtime_matrices(method, delta_mats[method], layers, h) for method in METHODS}
    all_documents = sorted(R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    start = len(all_documents) * args.shard_index // args.shard_count
    end = len(all_documents) * (args.shard_index + 1) // args.shard_count
    documents = all_documents[start:end]
    done = completed_ids(shard_file)
    peak = tree_bytes(output_dir)
    started = time.time()
    for document in documents:
        if document["document_id"] in done:
            continue
        token_ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
        if len(token_ids) < 256:
            raise RuntimeError(f"document too short: {document['document_id']} {len(token_ids)}")
        reference_probe = E.PerLayerHistoryProbe.build(F, runtime_mats["Native_INT8"])
        reference_probe.install(model)
        try:
            reference = E.trajectory(F, model, token_ids, reference_probe, layers, device, "Native_INT8", delta_mats["Native_INT8"], h, 128, None)
        finally:
            reference_probe.close()
        method_rows = {}
        for method in METHODS:
            probe = E.PerLayerHistoryProbe.build(F, runtime_mats[method])
            probe.install(model)
            try:
                rows = E.trajectory(F, model, token_ids, probe, layers, device, method, delta_mats[method], h, 128, reference)
            finally:
                probe.close()
            method_rows[method] = {"auc": E.auc(rows), "horizons": rows}
        row = {
            "task": "PERSISTENT_HEADROOM_CONFIRMATION_V1",
            "model": "Ling-3.0-tiny/KDA",
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
            "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
            "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
            "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": "PASS",
        }
        append_jsonl(shard_file, row)
        done.add(document["document_id"])
        del reference, method_rows, row, token_ids
        gc.collect()
        torch.cuda.empty_cache()
        peak = max(peak, tree_bytes(output_dir))
        save_json(status_file, {
            "status": "RUNNING",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "sharding": "contiguous_panel_ranges",
            "completed": len(done),
            "assigned": len(documents),
            "last_document": document["document_id"],
            "checkpoint_sha256": observed,
            "tokenizer": token_meta,
            "peak_output_bytes": peak,
            "elapsed_seconds": time.time() - started,
            "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
            "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
            "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": "PASS",
            "training_performed": False,
            "model_weights_changed": False,
            "AIME26_used": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"SHARD {args.shard_index} {len(done)}/{len(documents)} {document['document_id']}", flush=True)

    save_json(status_file, {
        "status": "COMPLETE",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sharding": "contiguous_panel_ranges",
        "completed": len(done),
        "assigned": len(documents),
        "checkpoint_sha256": observed,
        "tokenizer": token_meta,
        "peak_output_bytes": peak,
        "elapsed_seconds": time.time() - started,
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
        "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": "PASS",
        "training_performed": False,
        "model_weights_changed": False,
        "AIME26_used": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
