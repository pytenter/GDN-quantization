#!/usr/bin/env python3
"""Compact Ling/KDA runtime-repeatability audit.

The audit reuses the frozen persistent-evaluation path and stores only scalar
metrics plus exact byte hashes.  It never writes model tensors to disk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROTATION = HERE.parents[1] / "rotation"
PERSISTENT = ROTATION / "persistent_headroom_confirmation_v1" / "streaming_evaluate_ling.py"
METHODS = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")
HORIZONS = (1, 4, 8, 16, 32, 64, 128)
SELECTED_INDICES = (0, 7, 15)
EXPECTED = {
    "Dense_State": "dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318",
    "Dense_Functional": "65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c",
}


def apply_diagnostic_gated_rmsnorm_config() -> bool:
    if os.environ.get("LING_GATED_RMSNORM_FIXED_CONFIG") != "BT16_W8":
        return False
    import triton
    import fla.modules.fused_norm_gate as module
    current = module.layer_norm_gated_fwd_kernel
    while current is not None:
        if hasattr(current, "configs"):
            current.configs = [triton.Config({"BT": 16}, num_warps=8, num_stages=3)]
            if isinstance(getattr(current, "cache", None), dict):
                current.cache.clear()
        current = getattr(current, "fn", None)
    return True


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = import_file(PERSISTENT, "ling_persistent_for_repeatability_v1")
E, R = P.E, P.R


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().contiguous().cpu()
    return cpu.view(torch.uint8).numpy().tobytes()


def tensor_signature(value: torch.Tensor) -> dict:
    cpu = value.detach().contiguous().cpu()
    numeric = cpu.float()
    return {
        "sha256": hashlib.sha256(tensor_bytes(cpu)).hexdigest(),
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "max_abs": float(numeric.abs().max()) if numeric.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(numeric)) if numeric.numel() else 0.0,
        "mean": float(numeric.mean()) if numeric.numel() else 0.0,
        "nonfinite": int((~torch.isfinite(numeric)).sum()),
    }


def cache_signature(F, cache, layers) -> dict:
    stack = F.BASE.cache_stack(cache, layers)
    digest = hashlib.sha256()
    layer_rows = []
    total_nonfinite = 0
    norm_sq = 0.0
    max_abs = 0.0
    dtypes = set()
    for layer in sorted(layers):
        value = stack[layer].detach().contiguous().cpu()
        signature = tensor_signature(value)
        digest.update(str(layer).encode("ascii"))
        digest.update(tensor_bytes(value))
        total_nonfinite += signature["nonfinite"]
        norm_sq += signature["l2"] ** 2
        max_abs = max(max_abs, signature["max_abs"])
        dtypes.add(signature["dtype"])
        layer_rows.append({"layer": int(layer), **signature})
    return {
        "sha256": digest.hexdigest(),
        "layers": len(layer_rows),
        "dtypes": sorted(dtypes),
        "max_abs": max_abs,
        "l2": norm_sq ** 0.5,
        "nonfinite": total_nonfinite,
        "layer_hashes": [{"layer": row["layer"], "sha256": row["sha256"]} for row in layer_rows],
        "layer_summaries": [
            {key: row[key] for key in ("layer", "dtype", "max_abs", "l2", "mean", "nonfinite")}
            for row in layer_rows
        ],
    }


def runtime_flags() -> dict:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "fla_cache_mode": os.environ.get("FLA_CACHE_MODE", "<unset: disabled>") ,
        "fla_config_dir": os.environ.get("FLA_CONFIG_DIR"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_compile_used_by_audit": False,
        "cuda_graph_used_by_audit": False,
        "radix_cache_used_by_audit": False,
        "mamba_radix_cache_used_by_audit": False,
        "sampling": False,
        "teacher_forcing": True,
    }


def seed_all(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {"python_random": seed, "numpy": seed, "torch": seed, "torch_cuda": seed}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_context(args):
    checkpoints = {
        "Dense_State": Path(args.state_checkpoint),
        "Dense_Functional": Path(args.functional_checkpoint),
    }
    observed = {name: sha256_file(path) for name, path in checkpoints.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"CHECKPOINT_PROVENANCE_FAIL: {observed}")
    F, model, tokenizer, local_probe, layers = R.load_context(args)
    device = F.model_input_device(model)
    h = R.hadamard(device)
    state_bank, _ = E.load_bank(checkpoints["Dense_State"], layers, device)
    functional_bank, _ = E.load_bank(checkpoints["Dense_Functional"], layers, device)
    banks = {"Dense_State": state_bank, "Dense_Functional": functional_bank}
    traces = R.load_traces(Path(args.trace_dir), "VALIDATION")
    for method in METHODS:
        E.local_method(model, local_probe, layers, traces, method, banks, h, device)
    local_probe.close()
    del traces
    gc.collect()
    torch.cuda.empty_cache()
    delta_mats = {method: E.matrices(method, banks, layers, device) for method in METHODS}
    runtime_mats = {
        method: E.runtime_matrices(method, delta_mats[method], layers, h)
        for method in METHODS
    }
    return F, model, tokenizer, layers, device, h, delta_mats, runtime_mats, observed


def reference_run(F, model, token_ids, layers, device, h, delta_mats, runtime_mats):
    probe = E.PerLayerHistoryProbe.build(F, runtime_mats["Native_INT8"])
    probe.install(model)
    try:
        result = E.trajectory(
            F, model, token_ids, probe, layers, device, "Native_INT8",
            delta_mats["Native_INT8"], h, 128, None,
        )
    finally:
        probe.close()
    logits, states = result
    return result, {
        "logit_hashes": {str(step): tensor_signature(logits[step])["sha256"] for step in HORIZONS},
        "state_hashes": {
            str(step): hashlib.sha256(
                b"".join(tensor_bytes(states[step][layer]) for layer in sorted(layers))
            ).hexdigest()
            for step in HORIZONS
        },
    }


def condition_run(F, model, token_ids, layers, device, method, mats, runtime_mats, h, references):
    probe = E.PerLayerHistoryProbe.build(F, runtime_mats[method])
    probe.install(model)
    basis = "rotated" if E.rotated(method) else "native"
    checkpoints = {}
    rows = []
    identity = torch.eye(128, dtype=torch.float32)
    try:
        item = E.prefill(F, model, token_ids, probe, basis, layers, device, method)
        checkpoints["prefill_cache_pre_quant"] = cache_signature(F, item["past"], layers)
        E.quantize_cache(F, item["past"], layers)
        checkpoints["prefill_cache_post_quant"] = cache_signature(F, item["past"], layers)
        checkpoints["first_decode_input"] = checkpoints["prefill_cache_post_quant"]
        for step in range(1, 129):
            value, _records = F.H.advance(
                model, probe, item, int(token_ids[127 + step]), 127 + step,
                layers, identity, step,
            )
            if step == 1:
                checkpoints["first_decode_writeback_pre_quant"] = cache_signature(F, item["past"], layers)
            E.quantize_cache(F, item["past"], layers)
            if step == 1:
                checkpoints["first_decode_writeback_post_quant"] = cache_signature(F, item["past"], layers)
            if step in HORIZONS:
                last = value[:, -1].detach().float().cpu()
                metric = E.logit_metrics(last, references[0][step])
                recovered = E.recovered_snapshot(F, item["past"], layers, method, mats, h)
                metric["state_relative_l2"] = E.state_rel(recovered, references[1][step], layers)
                metric["logits_sha256"] = tensor_signature(last)["sha256"]
                metric["cache_sha256"] = cache_signature(F, item["past"], layers)["sha256"]
                rows.append({"horizon": step, **metric})
            del value, _records
        del item
    finally:
        probe.close()
    auc = E.auc(rows)
    return {
        "auc": auc,
        "horizons": rows,
        "checkpoints": checkpoints,
        "nonfinite": sum(int(row["nonfinite"]) for row in rows),
    }


def checkpoint_hash_map(run: dict) -> dict:
    output = {name: value["sha256"] for name, value in run["checkpoints"].items()}
    for row in run["horizons"]:
        output[f"h{row['horizon']}_logits"] = row["logits_sha256"]
        output[f"h{row['horizon']}_cache"] = row["cache_sha256"]
    return output


def summarize_runs(rows: list[dict]) -> dict:
    output = {}
    for document_id in sorted({row["document_id"] for row in rows}):
        output[document_id] = {}
        for method in METHODS:
            group = [row for row in rows if row["document_id"] == document_id and row["condition"] == method]
            values = [row["auc"] for row in group]
            maps = [row["checkpoint_hashes"] for row in group]
            common_keys = sorted(set.intersection(*(set(item) for item in maps))) if maps else []
            output[document_id][method] = {
                "repeats": len(group),
                "auc_values": values,
                "auc_range": max(values) - min(values),
                "auc_std": statistics.pstdev(values),
                "max_pairwise_difference": max(values) - min(values),
                "repeat_1_vs_2_bitwise_by_checkpoint": {
                    key: maps[0][key] == maps[1][key] for key in common_keys
                } if len(maps) >= 2 else {},
                "repeat_1_vs_2_all_key_checkpoints_bitwise": (
                    all(maps[0][key] == maps[1][key] for key in common_keys)
                    if len(maps) >= 2 else None
                ),
            }
    all_zero = all(
        item["auc_range"] == 0.0 and item["repeat_1_vs_2_all_key_checkpoints_bitwise"]
        for document in output.values() for item in document.values()
    )
    return {"per_doc_condition": output, "SAME_PROCESS_REPEATABILITY": "PASS" if all_zero else "FAIL"}


def manifest(tokenizer, documents: list[dict], corpus: Path) -> dict:
    source_manifest = corpus.parent / "PERSISTENT_HEADROOM_64DOC_MANIFEST_BASE.json"
    items = []
    for document in documents:
        token_ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
        items.append({
            "panel_index": int(document["panel_index"]),
            "document_id": document["document_id"],
            "raw_text_sha256": document["raw_text_sha256"],
            "token_ids_sha256": token_hash(token_ids),
            "token_length": len(token_ids),
        })
    value = {
        "task": "LING_RUNTIME_REPEATABILITY_CLOSURE_V1",
        "selection_rule": "manifest indices [0, 7, 15]",
        "documents": items,
        "tokenizer": P.tokenizer_fingerprint(tokenizer),
        "source_manifest_sha256": sha256_file(source_manifest),
        "corpus_sha256": sha256_file(corpus),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("same-process", "single"), default="same-process")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--doc-index", type=int, default=0)
    parser.add_argument("--condition", choices=METHODS, default="Hadamard")
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--seed-mode", choices=("untouched", "fixed"), default="untouched")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    args = parser.parse_args()

    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    seed_record = None
    if args.seed_mode == "fixed":
        seed_record = seed_all(args.seed)
    gated_rmsnorm_fixed = apply_diagnostic_gated_rmsnorm_config()
    started = time.time()
    F, model, tokenizer, layers, device, h, delta_mats, runtime_mats, observed = load_context(args)
    documents = sorted(R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    selected = [documents[index] for index in SELECTED_INDICES]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_value = manifest(tokenizer, selected, Path(args.corpus))
    write_json(out / "LING_REPEATABILITY_3DOC_MANIFEST.json", manifest_value)
    environment = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime_flags": runtime_flags(),
        "seed_mode": args.seed_mode,
        "seeds": seed_record,
        "checkpoint_sha256": observed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
        "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": "PASS",
        "LING_GATED_RMSNORM_FIXED_CONFIG": os.environ.get("LING_GATED_RMSNORM_FIXED_CONFIG"),
        "diagnostic_gated_rmsnorm_fixed_config_applied": gated_rmsnorm_fixed,
    }
    write_json(out / f"runtime_environment_{args.mode}_{args.seed_mode}.json", environment)

    rows = []
    if args.mode == "same-process":
        work = [(doc, repeat, method) for doc in selected for repeat in range(args.repeats) for method in METHODS]
    else:
        if args.doc_index not in SELECTED_INDICES:
            raise ValueError("single-mode doc-index must be one of 0, 7, 15")
        work = [(documents[args.doc_index], args.repeat_id, args.condition)]

    active_doc = None
    active_repeat = None
    references = None
    reference_hashes = None
    for sequence, (document, repeat, method) in enumerate(work, 1):
        if document["document_id"] != active_doc or repeat != active_repeat:
            token_ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
            references, reference_hashes = reference_run(
                F, model, token_ids, layers, device, h, delta_mats, runtime_mats,
            )
            active_doc, active_repeat = document["document_id"], repeat
        result = condition_run(
            F, model, token_ids, layers, device, method, delta_mats[method],
            runtime_mats, h, references,
        )
        row = {
            "document_id": document["document_id"],
            "panel_index": int(document["panel_index"]),
            "token_ids_sha256": token_hash(token_ids),
            "condition": method,
            "repeat": int(repeat),
            "auc": result["auc"],
            "horizons": result["horizons"],
            "state_summary": {
                name: {key: value[key] for key in (
                    "dtypes", "max_abs", "l2", "nonfinite", "layer_hashes", "layer_summaries",
                )}
                for name, value in result["checkpoints"].items()
            },
            "checkpoint_hashes": checkpoint_hash_map(result),
            "reference_hashes": reference_hashes,
            "nonfinite": result["nonfinite"],
            "process_id": os.getpid(),
            "seed_mode": args.seed_mode,
        }
        rows.append(row)
        if args.mode == "same-process":
            append_jsonl(out / "same_process_runs.jsonl", row)
        print(
            f"{args.mode} {sequence}/{len(work)} {document['document_id']} repeat={repeat} "
            f"condition={method} auc={result['auc']:.12g}", flush=True,
        )
        del result
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode == "same-process":
        summary = summarize_runs(rows)
        summary.update({
            "task": "LING_RUNTIME_REPEATABILITY_CLOSURE_V1",
            "manifest_sha256": manifest_value["manifest_sha256"],
            "elapsed_seconds": time.time() - started,
            "runtime_environment": environment,
            "durable_tensor_bytes": 0,
        })
        write_json(out / "same_process_summary.json", summary)
        print(json.dumps({"SAME_PROCESS_REPEATABILITY": summary["SAME_PROCESS_REPEATABILITY"]}, indent=2))
    else:
        target = out / "fresh_process_runs" / (
            f"doc{args.doc_index:02d}_{args.condition}_{args.seed_mode}_r{args.repeat_id:02d}.json"
        )
        write_json(target, rows[0])
        print(json.dumps({"output": str(target), "auc": rows[0]["auc"]}, indent=2))


if __name__ == "__main__":
    main()
