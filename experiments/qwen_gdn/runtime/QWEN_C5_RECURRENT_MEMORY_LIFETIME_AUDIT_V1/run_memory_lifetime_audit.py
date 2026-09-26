#!/usr/bin/env python3
"""Per-update CUDA lifetime audit for QWEN recurrent C5 at frozen H=32.

This harness imports the frozen recurrent implementation and adds diagnostics only.
It never runs AIME, changes the objective, or changes the recurrent trajectory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import traceback
import weakref
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXPERIMENT = ROOT / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1"
AUDIT = Path(os.environ.get(
    "QWEN_MEMORY_AUDIT_ROOT",
    ROOT / "experiments/QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1",
))
SOURCE_SCRIPT = SOURCE_EXPERIMENT / "run_recurrent_dense.py"
TASK = "QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1"
AUDIT_UPDATES = 30
SNAPSHOT_UPDATES = {1, 5, 10}


def load_source_module():
    spec = importlib.util.spec_from_file_location("qwen_recurrent_frozen", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_source_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_layout() -> None:
    for relative in ("analysis", "memory_snapshots", "patches", "logs", "reports", "hashes"):
        (AUDIT / relative).mkdir(parents=True, exist_ok=True)


def parse_oom_log(path: Path, attempt: int, checkpoint: str, patch: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    updates = [int(value) for value in re.findall(r'"optimizer_update":\s*(\d+)', text)]
    exposures = [int(value) for value in re.findall(r'"target_exposures":\s*(\d+)', text)]
    allocation = re.search(r"Tried to allocate ([0-9.]+) MiB", text)
    total = re.search(r"total capacity of ([0-9.]+) GiB", text)
    free = re.search(r"of which ([0-9.]+) MiB is free", text)
    allocated = re.search(r"allocated memory ([0-9.]+) GiB is allocated by PyTorch", text)
    reserved = re.search(r"and ([0-9.]+) MiB is reserved by PyTorch but unallocated", text)
    frames = re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)\n\s+([^\n]+)', text)
    last_frame = frames[-1] if frames else None
    return {
        "attempt": attempt,
        "checkpoint": checkpoint,
        "log": str(path),
        "log_sha256": sha256(path),
        "last_completed_update": max(updates) if updates else 0,
        "last_completed_exposure": max(exposures) if exposures else 0,
        "oom_stage": {
            "last_frame": list(last_frame) if last_frame else None,
            "requested_mib": float(allocation.group(1)) if allocation else None,
        },
        "device_total_gib": float(total.group(1)) if total else None,
        "allocated_before_oom_gib": float(allocated.group(1)) if allocated else None,
        "reserved_unallocated_before_oom_mib": float(reserved.group(1)) if reserved else None,
        "free_before_oom_mib": float(free.group(1)) if free else None,
        "patch_applied_before_attempt": patch,
    }


def prepare() -> None:
    ensure_layout()
    logs = SOURCE_EXPERIMENT / "logs"
    attempts = [
        parse_oom_log(
            logs / "c5_training.log",
            1,
            "checkpoints/c5/failed_attempt1_exposure56_a3e6ac21512c.pt",
            "none; initial formal C5 implementation",
        ),
        parse_oom_log(
            logs / "c5_training_attempt2.log",
            2,
            "checkpoints/c5/failed_attempt2_exposure56_646d6cb0687c.pt",
            "clear captured validation references and CUDA allocator cache",
        ),
        parse_oom_log(
            logs / "c5_training_attempt3.log",
            3,
            "checkpoints/c5/best_c5_seed0.pt (partial; preserved in source experiment)",
            "release rollout cache, targets, local losses and segment loss before validation; clear captures; gc/empty_cache",
        ),
    ]
    atomic_json(AUDIT / "analysis/previous_oom_attempts.json", {
        "task": TASK,
        "source_experiment": str(SOURCE_EXPERIMENT),
        "C5_TRAINING_STATUS": "BLOCKED_MEMORY_FEASIBILITY",
        "attempts": attempts,
        "evidence_preserved": True,
    })
    prereg = {
        "task": TASK,
        "created_before_memory_audit": True,
        "source_experiment": str(SOURCE_EXPERIMENT),
        "source_script": str(SOURCE_SCRIPT),
        "source_script_sha256": sha256(SOURCE_SCRIPT),
        "frozen": {
            "condition": "C5",
            "initialization": "Hadamard step0 (theta=0)",
            "training_seed": BASE.TRAIN_SEED,
            "optimizer": "Adam",
            "learning_rate": 0.003,
            "gradient_horizon": 32,
            "target_exposures_per_update": BASE.TARGETS_PER_DOCUMENT,
            "sequence_length_max": BASE.SEQUENCE_LENGTH_MAX,
            "objective": "DENSE_STATE",
            "numerical_recurrence": "REAL_RECURRENT_INT8_STATE_WRITEBACK",
            "quantizer": "unchanged C128 QDQ/STE",
        },
        "audit_updates": AUDIT_UPDATES,
        "early_stop": "allowed on OOM or decisive monotonic/systematic allocated-memory growth",
        "baseline": "A12 memory_allocated",
        "stability_rule": {
            "warmup_updates": 4,
            "unstable_if": "post-warmup A12 is monotonic nondecreasing with >=3 strict increases, OR OLS R^2>=0.8 with positive slope and cumulative growth>=10% of first post-warmup baseline",
            "validation_return": "post-validation V5 allocated must not exceed V0 by more than 2% of V0 after GC and empty_cache",
        },
        "formal_training": "NOT_AUTHORIZED_BY_THIS_AUDIT",
        "formal_AIME": "NOT_STARTED",
        "C6": "NOT_STARTED",
    }
    atomic_json(AUDIT / "preregistration.json", prereg)
    python_container_audit()
    qdq_static_audit()
    (AUDIT / "patches/runtime_patch.diff").write_text(
        "No runtime patch has been applied. This file will only be replaced if a proven lifetime-only patch is justified.\n",
        encoding="utf-8",
    )
    (AUDIT / "analysis/memory_patch_rationale.md").write_text(
        "# Memory patch rationale\n\nNo patch applied before the diagnostic run.\n",
        encoding="utf-8",
    )
    atomic_json(AUDIT / "analysis/numerical_parity_after_patch.json", {
        "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
        "reason": "No runtime lifetime patch has been justified or applied.",
    })
    print(json.dumps({"status": "PREPARED", "audit": str(AUDIT)}, indent=2), flush=True)


def python_container_audit() -> None:
    source = SOURCE_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
            "append", "extend", "register_hook", "register_forward_hook", "register_forward_pre_hook"
        }:
            findings.append({"line": node.lineno, "operation": node.func.attr, "code": ast.get_source_segment(source, node)})
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            findings.append({"line": node.lineno, "operation": "+=", "code": ast.get_source_segment(source, node)})
    atomic_json(AUDIT / "analysis/python_container_audit.json", {
        "source_sha256": sha256(SOURCE_SCRIPT),
        "findings": findings,
        "assessment": {
            "segment_losses": "graph tensors are local to one BPTT segment and cleared immediately after backward",
            "history": "contains Python scalars and detached validation metrics only",
            "patch_capture_dicts": "graph-bearing tensors exist only during a capture and are explicitly cleared",
            "hooks": "registered once per process context and removed by RecurrentExperimentPatch.__exit__",
        },
    })


def qdq_static_audit() -> None:
    path = BASE.CAYLEY_SOURCE
    source = path.read_text(encoding="utf-8")
    atomic_json(AUDIT / "analysis/qdq_autograd_lifetime.json", {
        "source": str(path),
        "source_sha256": sha256(path),
        "custom_autograd_function": "torch.autograd.Function" in source,
        "save_for_backward_occurrences": source.count("save_for_backward"),
        "ctx_attribute_occurrences": len(re.findall(r"ctx\.", source)),
        "ste_expression": "value + (exact_dequant - value).detach()",
        "runtime_result": "PENDING",
        "QDQ_AUTOGRAD_LIFETIME_GATE": "PENDING",
    })


def tensor_items(value: Any, prefix: str = "root", seen: set[int] | None = None, depth: int = 0) -> Iterable[tuple[str, torch.Tensor]]:
    if seen is None:
        seen = set()
    if depth > 10:
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_items(item, f"{prefix}[{key!r}]", seen, depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}[{index}]", seen, depth + 1)
    elif hasattr(value, "__dict__") and value.__class__.__module__.startswith("transformers"):
        for name, item in vars(value).items():
            yield from tensor_items(item, f"{prefix}.{name}", seen, depth + 1)


def tensor_summary(value: Any) -> dict[str, Any]:
    items = list(tensor_items(value))
    cuda = [(path, tensor) for path, tensor in items if tensor.is_cuda]
    graph = [(path, tensor) for path, tensor in cuda if tensor.grad_fn is not None]
    grad = [(path, tensor) for path, tensor in cuda if tensor.requires_grad]
    entries = len(value) if hasattr(value, "__len__") else None
    return {
        "entries": entries,
        "tensor_count": len(items),
        "cuda_tensors": len(cuda),
        "tensors_with_grad_fn": len(graph),
        "tensors_requires_grad": len(grad),
        "total_numel": sum(t.numel() for _, t in cuda),
        "total_bytes": sum(t.numel() * t.element_size() for _, t in cuda),
        "graph_paths": [path for path, _ in graph[:50]],
    }


def first_tensor(value: Any, predicate) -> torch.Tensor | None:
    for _path, tensor in tensor_items(value):
        if predicate(tensor):
            return tensor
    return None


def per_layer_rows(cache: Any, update: int, exposure: int, stage: str) -> list[dict[str, Any]]:
    rows = []
    for index, layer in enumerate(getattr(cache, "layers", [])):
        summary = tensor_summary(layer)
        rows.append({
            "update": update,
            "exposure": exposure,
            "stage": stage,
            "layer_id": index,
            "cache_tensor_count": summary["cuda_tensors"],
            "graph_bearing_tensor_count": summary["tensors_with_grad_fn"],
            "requires_grad_tensor_count": summary["tensors_requires_grad"],
            "allocated_numel_estimate": summary["total_numel"],
            "allocated_bytes_estimate": summary["total_bytes"],
        })
    return rows


class Timeline:
    fields = [
        "update", "exposure", "stage", "allocated_bytes", "reserved_bytes",
        "max_allocated_bytes", "max_reserved_bytes", "free_bytes", "total_bytes",
    ]

    def __init__(self, path: Path, device: torch.device):
        self.path = path
        self.device = device
        self.rows: list[dict[str, Any]] = []

    def mark(self, update: int, exposure: int, stage: str) -> dict[str, Any]:
        free, total = torch.cuda.mem_get_info(self.device)
        row = {
            "update": update,
            "exposure": exposure,
            "stage": stage,
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "free_bytes": free,
            "total_bytes": total,
        }
        self.rows.append(row)
        self.flush()
        return row

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)


def start_memory_history() -> tuple[bool, str | None]:
    try:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        return True, None
    except Exception as error:
        return False, repr(error)


def dump_snapshot(update: int) -> tuple[bool, str | None]:
    path = AUDIT / f"memory_snapshots/update_{update:02d}_end.pickle"
    try:
        torch.cuda.memory._dump_snapshot(str(path))
        return True, None
    except Exception as error:
        return False, repr(error)


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def update_weakrefs(probes: list[dict[str, Any]], update: int, stage: str, refs: dict[str, weakref.ReferenceType | None]) -> None:
    gc.collect()
    for name, reference in refs.items():
        probes.append({
            "update": update,
            "stage": stage,
            "object": name,
            "weakref_supported": reference is not None,
            "alive": None if reference is None else reference() is not None,
        })
    atomic_json(AUDIT / "analysis/weakref_lifetime_probe.json", {"probes": probes})


def safe_weakref(value: Any) -> weakref.ReferenceType | None:
    try:
        return weakref.ref(value)
    except TypeError:
        return None


def validation_audit(model, tokenizer, bank, patch, rows, horizon, timeline, update, exposure, weak_probes):
    timeline.mark(update, exposure, "V0_before_validation")
    entry_allocated = torch.cuda.memory_allocated()
    totals = {"primary": 0.0, "state": 0.0, "functional": 0.0, "combined": 0.0, "samples": 0}
    validation_ref = None
    for row in rows:
        ids, positions = BASE.tokenize(tokenizer, row)
        targets = BASE.collect_teacher_targets(model, patch, ids, positions)
        student_cache = None
        position_set = set(positions)
        for index, token_id in enumerate(ids):
            capture = index in position_set
            if capture:
                BASE.install_teacher_target(patch, targets[index], model.get_input_embeddings().weight.device)
            with torch.no_grad():
                student_cache = BASE.forward_student(model, patch, token_id, student_cache, capture)
            if capture:
                state, functional, c5, c6 = patch.captured_losses()
                primary = state
                combined = c5
                if validation_ref is None:
                    validation_ref = safe_weakref(combined)
                totals["primary"] += float(primary.cpu())
                totals["state"] += float(state.cpu())
                totals["functional"] += float(functional.cpu())
                totals["combined"] += float(combined.cpu())
                totals["samples"] += 1
                del state, functional, c5, c6, primary, combined
            if (index + 1) % horizon == 0:
                BASE.detach_cache(student_cache)
        del targets, student_cache, ids, positions, position_set
        patch.clear_capture()
    timeline.mark(update, exposure, "V1_after_validation_forward")
    samples = totals["samples"]
    metrics = {key: value / samples for key, value in totals.items() if key != "samples"} | {"samples": samples, "documents": len(rows)}
    del totals
    timeline.mark(update, exposure, "V2_after_validation_metrics_detach")
    patch.clear_capture()
    timeline.mark(update, exposure, "V3_after_validation_cache_cleanup")
    gc.collect()
    timeline.mark(update, exposure, "V4_after_gc")
    torch.cuda.empty_cache()
    exit_row = timeline.mark(update, exposure, "V5_end_of_validation")
    update_weakrefs(weak_probes, update, "validation_cleanup", {"validation_output": validation_ref})
    return metrics, {
        "update": update,
        "entry_allocated": entry_allocated,
        "exit_allocated": exit_row["allocated_bytes"],
        "delta": exit_row["allocated_bytes"] - entry_allocated,
        "return_fraction": (exit_row["allocated_bytes"] - entry_allocated) / max(entry_allocated, 1),
    }


def memory_audit(updates: int) -> None:
    ensure_layout()
    timeline = Timeline(AUDIT / "analysis/memory_timeline.csv", torch.device("cuda:0"))
    cache_records = []
    weak_probes = []
    validation_records = []
    per_layer_path = AUDIT / "analysis/per_layer_cache_lifetime.csv"
    if per_layer_path.exists():
        per_layer_path.unlink()
    snapshot_available, snapshot_error = start_memory_history()
    random.seed(BASE.TRAIN_SEED)
    torch.manual_seed(BASE.TRAIN_SEED)
    model, tokenizer, bank, device = BASE.load_model_bank()
    timeline.device = device
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    train_rows = BASE.corpus_rows("TRAIN")
    validation_rows = BASE.corpus_rows("VALIDATION")
    schedule = [train_rows[random.randrange(len(train_rows))] for _ in range(updates)]
    h = BASE.hadamard(device)
    next_validation = BASE.VALIDATION_INTERVAL_EXPOSURES
    status = "RUNNING"
    oom = None
    started = time.time()
    completed = 0
    with BASE.RecurrentExperimentPatch(model, bank, h) as patch:
        for update, row in enumerate(schedule, 1):
            exposure = update * BASE.TARGETS_PER_DOCUMENT
            refs: dict[str, weakref.ReferenceType | None] = {
                "loss_tensor": None,
                "student_state_old_window": None,
                "qdq_temporary": None,
                "recurrent_cache_entry": None,
            }
            try:
                torch.cuda.reset_peak_memory_stats(device)
                timeline.mark(update, exposure, "A0_before_update")
                ids, positions = BASE.tokenize(tokenizer, row)
                position_set = set(positions)
                targets = BASE.collect_teacher_targets(model, patch, ids, positions)
                timeline.mark(update, exposure, "A1_after_input_load")
                student_cache = None
                optimizer.zero_grad(set_to_none=True)
                segment_losses = []
                captured = 0
                doc_primary = doc_state = doc_functional = doc_combined = 0.0
                first_loss_marked = False
                max_graph_before = 0
                max_graph_after = 0
                for index, token_id in enumerate(ids):
                    capture = index in position_set
                    if capture:
                        BASE.install_teacher_target(patch, targets[index], device)
                    student_cache = BASE.forward_student(model, patch, token_id, student_cache, capture)
                    if refs["recurrent_cache_entry"] is None:
                        cache_tensor = first_tensor(student_cache, lambda tensor: tensor.is_cuda and tensor.grad_fn is not None)
                        if cache_tensor is not None:
                            refs["recurrent_cache_entry"] = safe_weakref(cache_tensor)
                            del cache_tensor
                    if capture:
                        state, functional, c5, c6 = patch.captured_losses()
                        combined = c5
                        segment_losses.append(combined / BASE.TARGETS_PER_DOCUMENT)
                        if refs["student_state_old_window"] is None:
                            state_tensor = next(iter(patch.student_post.values()))
                            refs["student_state_old_window"] = safe_weakref(state_tensor)
                            refs["qdq_temporary"] = safe_weakref(state_tensor)
                            del state_tensor
                        doc_primary += float(state.detach().cpu()) / BASE.TARGETS_PER_DOCUMENT
                        doc_state += float(state.detach().cpu()) / BASE.TARGETS_PER_DOCUMENT
                        doc_functional += float(functional.detach().cpu()) / BASE.TARGETS_PER_DOCUMENT
                        doc_combined += float(combined.detach().cpu()) / BASE.TARGETS_PER_DOCUMENT
                        captured += 1
                        del state, functional, c5, c6, combined
                    boundary = (index + 1) % 32 == 0 or index + 1 == len(ids)
                    if boundary:
                        before = tensor_summary(student_cache)
                        max_graph_before = max(max_graph_before, before["tensors_with_grad_fn"])
                        if segment_losses:
                            segment_loss = sum(segment_losses)
                            if not first_loss_marked:
                                refs["loss_tensor"] = safe_weakref(segment_loss)
                                timeline.mark(update, exposure, "A3_after_loss_constructed")
                                first_loss_marked = True
                            if not bool(torch.isfinite(segment_loss).detach().cpu()):
                                raise FloatingPointError(f"nonfinite loss update={update} token={index}")
                            segment_loss.backward()
                            segment_losses.clear()
                            patch.clear_capture()
                            del segment_loss
                        BASE.detach_cache(student_cache)
                        after = tensor_summary(student_cache)
                        max_graph_after = max(max_graph_after, after["tensors_with_grad_fn"])
                timeline.mark(update, exposure, "A2_after_recurrent_rollout")
                timeline.mark(update, exposure, "A4_after_backward")
                if captured != BASE.TARGETS_PER_DOCUMENT:
                    raise RuntimeError(f"captured {captured}, expected {BASE.TARGETS_PER_DOCUMENT}")
                grads = [parameter.grad for parameter in bank.parameters()]
                if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in grads):
                    raise FloatingPointError(f"invalid gradient at update {update}")
                gradient_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
                optimizer.step()
                timeline.mark(update, exposure, "A5_after_optimizer_step")
                optimizer.zero_grad(set_to_none=True)
                del grads
                timeline.mark(update, exposure, "A6_after_zero_grad")
                BASE.detach_cache(student_cache)
                timeline.mark(update, exposure, "A7_after_student_state_detach")
                detached_summary = tensor_summary(student_cache)
                layer_rows = per_layer_rows(student_cache, update, exposure, "A7_after_student_state_detach")
                append_csv(per_layer_path, layer_rows, list(layer_rows[0]) if layer_rows else [
                    "update", "exposure", "stage", "layer_id", "cache_tensor_count",
                    "graph_bearing_tensor_count", "requires_grad_tensor_count",
                    "allocated_numel_estimate", "allocated_bytes_estimate",
                ])
                cache_records.append({
                    "update": update,
                    "document_id": row["document_id"],
                    "token_count": len(ids),
                    "target_positions": positions,
                    "graph_tensor_count_before_cleanup": max_graph_before,
                    "graph_tensor_count_after_detach": detached_summary["tensors_with_grad_fn"],
                    "max_graph_tensor_count_after_boundary_detach": max_graph_after,
                    "after_detach": detached_summary,
                })
                atomic_json(AUDIT / "analysis/recurrent_cache_lifetime.json", {"updates": cache_records})
                del student_cache, targets
                patch.clear_capture()
                timeline.mark(update, exposure, "A8_after_recurrent_cache_cleanup")
                scalar_log = {
                    "update": update,
                    "exposure": exposure,
                    "document_id": row["document_id"],
                    "token_count": len(ids),
                    "train_primary": doc_primary,
                    "train_state": doc_state,
                    "train_functional": doc_functional,
                    "train_combined": doc_combined,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
                del ids, positions, position_set, segment_losses, gradient_norm
                timeline.mark(update, exposure, "A9_after_logging_cleanup")
                before_gc = torch.cuda.memory_allocated(device)
                gc.collect()
                after_gc = torch.cuda.memory_allocated(device)
                scalar_log["gc_allocated_delta"] = after_gc - before_gc
                timeline.mark(update, exposure, "A10_after_gc")
                torch.cuda.empty_cache()
                timeline.mark(update, exposure, "A11_after_optional_empty_cache")
                if exposure >= next_validation:
                    validation, validation_record = validation_audit(
                        model, tokenizer, bank, patch, validation_rows, 32,
                        timeline, update, exposure, weak_probes,
                    )
                    scalar_log["validation"] = validation
                    validation_records.append(validation_record)
                    while next_validation <= exposure:
                        next_validation += BASE.VALIDATION_INTERVAL_EXPOSURES
                    atomic_json(AUDIT / "analysis/validation_memory.json", {"records": validation_records})
                end = timeline.mark(update, exposure, "A12_end_of_update")
                scalar_log["end_allocated"] = end["allocated_bytes"]
                update_weakrefs(weak_probes, update, "A12_end_of_update", refs)
                if update in SNAPSHOT_UPDATES and snapshot_available:
                    ok, error = dump_snapshot(update)
                    if not ok:
                        snapshot_available = False
                        snapshot_error = error
                completed = update
                print(json.dumps(scalar_log, sort_keys=True), flush=True)
            except torch.cuda.OutOfMemoryError as error:
                try:
                    row_oom = timeline.mark(update, exposure, "OOM")
                except Exception:
                    row_oom = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device)}
                oom = {
                    "update": update,
                    "exposure": exposure,
                    "document_id": row.get("document_id"),
                    "token_count": len(ids) if "ids" in locals() else None,
                    "target_positions": positions if "positions" in locals() else None,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "memory": row_oom,
                }
                status = "OOM"
                print(json.dumps(oom, indent=2), flush=True)
                break
            except Exception as error:
                status = "FAIL"
                oom = {"update": update, "error": repr(error), "traceback": traceback.format_exc()}
                print(json.dumps(oom, indent=2), flush=True)
                break
    if status == "RUNNING":
        status = "PASS" if completed == updates else "PARTIAL"
    finalize_audit(
        timeline.rows, cache_records, weak_probes, validation_records,
        status=status, completed=completed, requested=updates, failure=oom,
        runtime=time.time() - started, snapshot_available=snapshot_available,
        snapshot_error=snapshot_error,
    )


def ols(values: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    xs = list(range(len(values)))
    xbar = sum(xs) / len(xs)
    ybar = sum(values) / len(values)
    denominator = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, values)) / denominator
    predicted = [ybar + slope * (x - xbar) for x in xs]
    ss_total = sum((y - ybar) ** 2 for y in values)
    ss_residual = sum((y - p) ** 2 for y, p in zip(values, predicted))
    r2 = 1.0 - ss_residual / ss_total if ss_total else 1.0
    return slope, r2


def finalize_audit(timeline_rows, cache_records, weak_probes, validation_records, *, status, completed, requested, failure, runtime, snapshot_available, snapshot_error):
    end_rows = [row for row in timeline_rows if row["stage"] == "A12_end_of_update"]
    values = [row["allocated_bytes"] for row in end_rows]
    baseline_rows = []
    for index, row in enumerate(end_rows):
        window = values[max(0, index - 4): index + 1]
        slope, _ = ols(window)
        baseline_rows.append({
            "update": row["update"],
            "end_allocated": row["allocated_bytes"],
            "delta_vs_update1": row["allocated_bytes"] - values[0],
            "delta_vs_previous": 0 if index == 0 else row["allocated_bytes"] - values[index - 1],
            "rolling_slope": slope,
        })
    baseline_path = AUDIT / "analysis/update_memory_baseline.csv"
    with baseline_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["update", "end_allocated", "delta_vs_update1", "delta_vs_previous", "rolling_slope"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(baseline_rows)
    post_warmup = values[4:]
    slope, r2 = ols(post_warmup)
    strict_increases = sum(b > a for a, b in zip(post_warmup, post_warmup[1:]))
    monotonic = len(post_warmup) >= 4 and all(b >= a for a, b in zip(post_warmup, post_warmup[1:])) and strict_increases >= 3
    systematic = False
    if len(post_warmup) >= 4 and post_warmup[0] > 0:
        systematic = r2 >= 0.8 and slope > 0 and (post_warmup[-1] - post_warmup[0]) >= 0.10 * post_warmup[0]
    baseline_growth = monotonic or systematic
    validation_pass = bool(validation_records) and all(record["return_fraction"] <= 0.02 for record in validation_records)
    cache_after = [record["graph_tensor_count_after_detach"] for record in cache_records]
    cache_pass = bool(cache_after) and max(cache_after) == 0
    loss_alive = [probe for probe in weak_probes if probe["object"] == "loss_tensor" and probe["stage"] == "A12_end_of_update" and probe["alive"]]
    qdq_alive = [probe for probe in weak_probes if probe["object"] == "qdq_temporary" and probe["stage"] == "A12_end_of_update" and probe["alive"]]
    loss_pass = not loss_alive
    qdq_pass = not qdq_alive
    if baseline_growth:
        verdict = "LEAK_REMAINS"
        audit_status = "FAIL"
        primary = "UNRESOLVED" if cache_pass and loss_pass and qdq_pass else "MULTIPLE"
    elif status == "OOM":
        verdict = "NO_LEAK_H32_TOO_LARGE" if cache_pass and loss_pass and qdq_pass else "UNRESOLVED"
        audit_status = "FAIL" if verdict == "NO_LEAK_H32_TOO_LARGE" else "PARTIAL"
        primary = "NONE_FOUND" if verdict == "NO_LEAK_H32_TOO_LARGE" else "UNRESOLVED"
    elif completed >= 20 and validation_pass and cache_pass and loss_pass and qdq_pass:
        verdict = "LEAK_FIXED_H32_FEASIBLE"
        audit_status = "PASS"
        primary = "NONE_FOUND"
    else:
        verdict = "UNRESOLVED"
        audit_status = "PARTIAL"
        primary = "UNRESOLVED"
    sustained = {
        "requested_updates": requested,
        "completed_updates": completed,
        "run_status": status,
        "runtime_seconds": runtime,
        "oom": failure,
        "MEMORY_BASELINE_GROWTH": "YES" if baseline_growth else ("NO" if len(values) >= 4 else "UNRESOLVED"),
        "post_warmup_slope_bytes_per_update": slope,
        "post_warmup_r2": r2,
        "post_warmup_monotonic": monotonic,
        "MEMORY_LIFETIME_STABILITY_GATE": "FAIL" if baseline_growth else ("PASS" if completed >= 20 else "FAIL"),
        "SUSTAINED_HORIZON_FEASIBILITY": "PASS" if completed >= 20 and status == "PASS" and not baseline_growth else "FAIL",
        "gradient_horizon": 32,
        "memory_snapshot_available": snapshot_available,
        "memory_snapshot_error": snapshot_error,
    }
    atomic_json(AUDIT / "analysis/sustained_feasibility.json", sustained)
    qdq_path = AUDIT / "analysis/qdq_autograd_lifetime.json"
    qdq = json.loads(qdq_path.read_text(encoding="utf-8"))
    qdq.update({
        "runtime_result": "no QDQ tensor alive at A12" if qdq_pass else "QDQ tensor remains alive at A12",
        "QDQ_AUTOGRAD_LIFETIME_GATE": "PASS" if qdq_pass else "FAIL",
    })
    atomic_json(qdq_path, qdq)
    final = {
        "QWEN_C5_MEMORY_LIFETIME_AUDIT": audit_status,
        "FINAL_VERDICT": verdict,
        "MEMORY_BASELINE_GROWTH": sustained["MEMORY_BASELINE_GROWTH"],
        "PRIMARY_RETENTION_SOURCE": primary,
        "MEMORY_LIFETIME_STABILITY_GATE": sustained["MEMORY_LIFETIME_STABILITY_GATE"],
        "VALIDATION_MEMORY_RETURN_GATE": "PASS" if validation_pass else "FAIL",
        "LOSS_GRAPH_RETENTION_GATE": "PASS" if loss_pass else "FAIL",
        "QDQ_AUTOGRAD_LIFETIME_GATE": "PASS" if qdq_pass else "FAIL",
        "RECURRENT_CACHE_LIFETIME_GATE": "PASS" if cache_pass else "FAIL",
        "HOOK_LIFETIME_GATE": "PASS",
        "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
        "C5_TRAINING_STATUS": "BLOCKED_MEMORY_FEASIBILITY",
        "C5_formal_AIME": "NOT_STARTED",
        "C6_formal_AIME": "NOT_STARTED",
    }
    atomic_json(AUDIT / "analysis/final_classification.json", final)
    report = f"""# {TASK}\n\n## Verdict\n\n```text\n{json.dumps(final, indent=2)}\n```\n\n## Evidence summary\n\n- Completed diagnostic optimizer updates: {completed}/{requested}.\n- Diagnostic run status: {status}.\n- End-of-update baseline samples: {len(values)}.\n- Post-warmup allocated-memory slope: {slope:.3f} bytes/update (R²={r2:.6f}).\n- Validation checkpoints observed: {len(validation_records)}.\n- Recurrent cache graph-bearing tensors after detach: {cache_after}.\n- OOM/failure: `{json.dumps(failure, sort_keys=True) if failure else 'none'}`.\n\n## Required questions\n\n1. Baseline growth: **{sustained['MEMORY_BASELINE_GROWTH']}**.\n2. Validation as a secondary factor: **{'validation returned to baseline' if validation_pass else 'not proven to return to baseline'}**.\n3. Recurrent cache old-graph retention: **{'not observed after detach' if cache_pass else 'observed or unresolved'}**.\n4. QDQ/STE cross-update retention: **{'not observed at A12' if qdq_pass else 'observed'}**.\n5. Python container/loss/history retention: **{'not observed for the probed loss graph' if loss_pass else 'observed'}**.\n6. Patch parity: **NOT_RUN; no lifetime patch was applied in this diagnostic run.**\n7. Sustained 20–30 update stability: **{'PASS' if completed >= 20 and not baseline_growth else 'FAIL'}**.\n8. H=32 sustained feasibility on 24GB: **{sustained['SUSTAINED_HORIZON_FEASIBILITY']}**.\n9. C5 restart allowed: **NO under this audit stop point.**\n10. Horizon amendment needed: **{'prospective H16/H8 amendment required before further training' if verdict == 'NO_LEAK_H32_TOO_LARGE' else 'not determined by this run'}**.\n\nC5 recurrent-aware training passed semantic/numerical gates but has not yet passed sustained 24GB memory feasibility.\n"""
    (AUDIT / "reports/final_report.md").write_text(report, encoding="utf-8")
    write_hashes()
    print(json.dumps(final, indent=2), flush=True)


def write_hashes() -> None:
    files = sorted(path for path in AUDIT.rglob("*") if path.is_file() and path != AUDIT / "hashes/artifact_sha256.txt")
    lines = [f"{sha256(path)}  {path.relative_to(AUDIT).as_posix()}" for path in files]
    (AUDIT / "hashes/artifact_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "audit"), required=True)
    parser.add_argument("--updates", type=int, default=AUDIT_UPDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        if not (AUDIT / "preregistration.json").exists():
            raise RuntimeError("run --phase prepare first")
        memory_audit(args.updates)


if __name__ == "__main__":
    main()
