#!/usr/bin/env python3
"""Formal, preregistered H64 recurrent C5/C6 training and frozen-R export."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
READINESS = ROOT.parent / "QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2"
RUNTIME_PATH = READINESS / "scripts/runtime.py"
FROZEN_PROBE_PATH = READINESS / "scripts/frozen_probe.py"
BRANCH = "exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1"
BASE_HEAD = "40c06322d5f9ad092fbfb908fabb29a107a0446f"
HORIZON = 64
MEMORY_CREEP_BYTES = 256 * 1024 * 1024
MIN_FREE_DISK_BYTES = 2 * 1024 * 1024 * 1024
EXPLODING_GRAD_NORM = 1.0e6


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = import_file(RUNTIME_PATH, "qwen_formal_runtime")
# frozen_probe imports the readiness helper as the top-level name `runtime`.
sys.modules["runtime"] = runtime
frozen_probe = import_file(FROZEN_PROBE_PATH, "qwen_formal_frozen_probe")
core = runtime.load_core()
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    ensure_disk_space(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"checkpoint write failure: {path}")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def ensure_disk_space(path: Path) -> None:
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"unsafe disk space: {free} bytes free at {parent}")


def cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}


def theta_manifest(bank) -> dict:
    rows = {}
    for layer in core.GDN_LAYERS:
        theta = bank.layer(layer).theta.detach().float().cpu().contiguous()
        rows[str(layer)] = {"shape": list(theta.shape), "dtype": str(theta.dtype),
                            "sha256": runtime.tensor_sha256(theta),
                            "norm": float(torch.linalg.vector_norm(theta))}
    return {"per_layer": rows, "manifest_sha256": json_sha256(rows),
            "total_norm": math.sqrt(sum(row["norm"] ** 2 for row in rows.values()))}


def model_sentinels(model) -> dict[str, str]:
    selected = {}
    for name, parameter in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            selected[name] = runtime.tensor_sha256(parameter.detach().flatten()[:1024])
    if not selected:
        raise RuntimeError("base-model sentinel selection failed")
    return selected


def memory_snapshot(device, stage: str) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def load_protocol() -> dict:
    path = ROOT / "preregistration.json"
    if not path.is_file():
        raise RuntimeError("preregistration missing")
    protocol = json.loads(path.read_text())
    if protocol["formal_protocol_status"] != "PROSPECTIVE_NOT_YET_TRAINED":
        raise RuntimeError("unexpected preregistration status")
    if protocol["training"]["common"]["horizon"] != HORIZON:
        raise RuntimeError("H != 64")
    return protocol


def verify_static_protocol(condition: str) -> tuple[dict, dict]:
    condition = condition.upper()
    if condition not in ("C5", "C6"):
        raise ValueError("condition must be C5 or C6")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("formal branch mismatch")
    if git("merge-base", "--is-ancestor", BASE_HEAD, "HEAD") != "":
        # git merge-base --is-ancestor prints nothing on success.
        raise RuntimeError("unexpected ancestry output")
    protocol = load_protocol()
    provenance = runtime.check_parent_sources()
    recorded = protocol["source_hashes"]
    paths = {
        "formal_pipeline": Path(__file__).resolve(),
        "readiness_runtime": RUNTIME_PATH,
        "readiness_fulldoc_common": READINESS / "scripts/fulldoc_common.py",
        "readiness_frozen_probe": FROZEN_PROBE_PATH,
        "canonical_c5_c6": runtime.SOURCE,
        "canonical_rotation": runtime.ROTATION,
        "canonical_modeling": runtime.SOURCE_ROOT / "transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
    }
    for name, path in paths.items():
        if file_sha256(path) != recorded[name]["sha256"]:
            raise RuntimeError(f"source/hash drift: {name}")
    if file_sha256(core.CORPUS) != protocol["training"]["common"].get(
            "corpus_sha256", "37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0"):
        raise RuntimeError("training data hash mismatch")
    rows = core.corpus_rows("TRAIN")
    schedule_rng = random.Random(core.TRAIN_SEED)
    schedule = [rows[schedule_rng.randrange(len(rows))] for _ in range(core.OPTIMIZER_UPDATES)]
    schedule_ids = [row["document_id"] for row in schedule]
    if schedule_ids != protocol["training"]["common"]["document_order"]:
        raise RuntimeError("document order mismatch")
    validation_ids = [row["document_id"] for row in core.corpus_rows("VALIDATION")]
    if validation_ids != protocol["validation"]["document_ids"]:
        raise RuntimeError("validation panel mismatch")
    if condition == "C6":
        c5_export = ROOT / "analysis/C5_export_summary.json"
        if not c5_export.is_file() or json.loads(c5_export.read_text()).get("status") != "PASS":
            raise RuntimeError("C6 blocked until C5 formal artifact is complete")
    return protocol, provenance


def fresh_bank(device) -> Any:
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS).to(device)
    if any(bool(torch.count_nonzero(parameter).detach().cpu()) for parameter in bank.parameters()):
        raise RuntimeError("initialization mismatch: theta is not zero")
    if sum(parameter.numel() for parameter in bank.parameters()) != 195072:
        raise RuntimeError("rotation parameter count mismatch")
    return bank


def verify_initialization(bank, protocol: dict) -> dict:
    observed = theta_manifest(bank)
    expected = protocol["rotation"]["initialization_manifest_sha256"]
    if observed["manifest_sha256"] != expected or observed["total_norm"] != 0.0:
        raise RuntimeError("rotation initialization hash mismatch")
    return observed


def preflight(condition: str) -> None:
    condition = condition.upper()
    output = ROOT / f"analysis/{condition}_preflight.json"
    error_output = ROOT / f"analysis/{condition}_preflight.error.json"
    if output.exists() or error_output.exists():
        raise RuntimeError(f"preflight already attempted for {condition}")
    protocol, provenance = verify_static_protocol(condition)
    random.seed(core.TRAIN_SEED)
    torch.manual_seed(core.TRAIN_SEED)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("formal training requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    started = time.time()
    try:
        model, tokenizer, device = runtime.load_model_and_tokenizer(core)
        before_model = model_sentinels(model)
        bank = fresh_bank(device)
        initial = verify_initialization(bank, protocol)
        first_row = core.corpus_rows("TRAIN")[0]
        ids, positions = core.tokenize(tokenizer, first_row)
        frozen_first = json.loads((ROOT / "configs/training_document_panel.json").read_text())["unique_panel"][0]
        token_hash = hashlib.sha256(b"".join(int(value).to_bytes(8, "little", signed=True)
                                              for value in ids)).hexdigest()
        if (first_row["document_id"] != frozen_first["document_id"] or len(ids) != 1024 or
                positions != frozen_first["capture_positions_zero_based"] or
                token_hash != frozen_first["token_ids_le_int64_sha256"]):
            raise RuntimeError("capture positions or first-document token digest mismatch")
        torch.cuda.reset_peak_memory_stats(device)
        probe = frozen_probe.run_fixed_probe(core, model, tokenizer, bank)
        after_probe = memory_snapshot(device, "after_no_update_probe")
        final_initial = theta_manifest(bank)
        if final_initial["manifest_sha256"] != initial["manifest_sha256"]:
            raise RuntimeError("no-update probe changed theta")
        if not all(probe["writeback_exact_by_layer"].values()):
            raise RuntimeError("recurrent state writeback mismatch")
        if not (-127 <= probe["C128_QDQ_code_min"] <= probe["C128_QDQ_code_max"] <= 127):
            raise RuntimeError("quantizer code range mismatch")
        after_model = model_sentinels(model)
        if before_model != after_model:
            raise RuntimeError("base model changed during preflight")
        prereg_path = ROOT / "preregistration.json"
        result = {
            "condition": condition,
            "status": "PASS",
            "formal_training_started": False,
            "H": HORIZON,
            "config_sha256": file_sha256(ROOT / f"configs/{condition}.json"),
            "preregistration_sha256": file_sha256(prereg_path),
            "preregistration_commit": git("log", "-1", "--format=%H", "--", str(prereg_path.relative_to(REPO))),
            "train_corpus_sha256": file_sha256(core.CORPUS),
            "training_manifest_sha256": provenance["training_manifest"]["actual_sha256"],
            "capture_positions_first_document": positions,
            "corrected_token_digest": token_hash,
            "rotation_initialization": initial,
            "quantizer": protocol["quantization"],
            "no_update_probe": probe,
            "memory": after_probe,
            "model_sentinels_unchanged": True,
            "source_provenance": provenance,
            "wall_clock_seconds": time.time() - started,
            "AIME_used": False,
        }
        atomic_json(output, result)
        print(json.dumps({"event": "preflight_complete", "condition": condition,
                          "status": "PASS", "wall_clock_seconds": result["wall_clock_seconds"]}), flush=True)
    except BaseException as exc:
        atomic_json(error_output, {"condition": condition, "status": "FAIL",
                                   "error_type": type(exc).__name__, "error": str(exc),
                                   "traceback": traceback.format_exc(), "formal_training_started": False})
        raise


def train_document(model, tokenizer, bank, patch, optimizer, device, row: dict,
                   condition: str, update: int) -> dict:
    objective = "DENSE_STATE" if condition == "C5" else "DENSE_FUNCTIONAL"
    ids, positions = core.tokenize(tokenizer, row)
    if len(ids) != 1024 or len(positions) != 8:
        raise RuntimeError("document length/capture count mismatch")
    position_set = set(positions)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    qdq_before = patch.qdq_audit["calls"]
    targets = core.collect_teacher_targets(model, patch, ids, positions)
    student_cache = None
    optimizer.zero_grad(set_to_none=True)
    aggregate = {"primary": 0.0, "state": 0.0, "functional": 0.0, "combined": 0.0}
    capture_losses = []
    segment_losses = []
    captured = 0
    for index, token_id in enumerate(ids):
        capture = index in position_set
        if capture:
            core.install_teacher_target(patch, targets[index], device)
        student_cache = core.forward_student(model, patch, token_id, student_cache, capture)
        if capture:
            state, functional, c5_loss, c6_loss = patch.captured_losses()
            primary = state if objective == "DENSE_STATE" else functional
            combined = c5_loss if objective == "DENSE_STATE" else c6_loss
            values = {"position_zero_based": index,
                      "state": float(state.detach().cpu()),
                      "functional": float(functional.detach().cpu()),
                      "primary": float(primary.detach().cpu()),
                      "combined": float(combined.detach().cpu())}
            if not all(math.isfinite(value) for key, value in values.items() if key != "position_zero_based"):
                raise FloatingPointError(f"NaN/Inf capture loss at update {update} token {index}")
            capture_losses.append(values)
            for key in aggregate:
                aggregate[key] += values[key] / core.TARGETS_PER_DOCUMENT
            segment_losses.append(combined / core.TARGETS_PER_DOCUMENT)
            captured += 1
            del state, functional, c5_loss, c6_loss, primary, combined
        boundary = (index + 1) % HORIZON == 0 or index + 1 == len(ids)
        if boundary:
            if segment_losses:
                segment_loss = sum(segment_losses)
                if not bool(torch.isfinite(segment_loss).detach().cpu()):
                    raise FloatingPointError(f"NaN/Inf segment loss update={update} token={index}")
                segment_loss.backward()
                segment_losses.clear()
                patch.clear_capture()
                del segment_loss
            core.detach_cache(student_cache)
    if captured != core.TARGETS_PER_DOCUMENT:
        raise RuntimeError(f"captured {captured}, expected {core.TARGETS_PER_DOCUMENT}")
    qdq_calls = patch.qdq_audit["calls"] - qdq_before
    if qdq_calls != len(core.GDN_LAYERS) * len(ids):
        raise RuntimeError(f"quantizer semantics mismatch: {qdq_calls} QDQ calls")
    parameters = list(bank.parameters())
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError(f"NaN/Inf/missing gradient at update {update}")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    raw_grad_norm = float(grad_norm.detach().cpu())
    if not math.isfinite(raw_grad_norm) or raw_grad_norm > EXPLODING_GRAD_NORM:
        raise FloatingPointError(f"catastrophic exploding gradient norm {raw_grad_norm}")
    optimizer.step()
    del targets, student_cache
    patch.clear_capture()
    gc.collect()
    torch.cuda.empty_cache()
    ortho = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
    if ortho["status"] != "PASS":
        raise RuntimeError(f"rotation orthogonality catastrophic failure at update {update}")
    theta = theta_manifest(bank)
    memory = memory_snapshot(device, "after_document_cleanup")
    return {
        "update_index": update,
        "document_id": row["document_id"],
        "document_tokens": len(ids),
        "capture_positions_zero_based": positions,
        "capture_losses": capture_losses,
        "aggregate_loss": aggregate,
        "gradient_norm_before_clip": raw_grad_norm,
        "gradient_clip_global_norm": 1.0,
        "theta_norm": theta["total_norm"],
        "theta_manifest_sha256": theta["manifest_sha256"],
        "rotation_orthogonality": {key: ortho[key] for key in
                                   ("status", "threshold", "max_abs_rt_r_minus_i", "all_finite", "all_proper")},
        "learning_rate": optimizer.param_groups[0]["lr"],
        "CUDA_allocated_bytes": memory["allocated_bytes"],
        "CUDA_reserved_bytes": memory["reserved_bytes"],
        "peak_CUDA_allocated_bytes": memory["peak_allocated_bytes"],
        "peak_CUDA_reserved_bytes": memory["peak_reserved_bytes"],
        "wall_clock_seconds": time.time() - started,
        "NaN_Inf_flags": False,
        "QDQ_calls": qdq_calls,
        "QDQ_code_min_observed": patch.qdq_audit["code_min"],
        "QDQ_code_max_observed": patch.qdq_audit["code_max"],
        "optimizer_steps_this_document": 1,
        "H": HORIZON,
    }


def checkpoint_payload(condition: str, update: int, exposure: int, validation: dict,
                       bank, protocol: dict, provenance: dict) -> dict:
    return {
        "experiment": ROOT.name,
        "condition": condition,
        "objective": "DENSE_STATE" if condition == "C5" else "DENSE_FUNCTIONAL",
        "optimized_loss": "state" if condition == "C5" else "functional + 0.1 * state",
        "training_mode": "REAL_RECURRENT_INT8_STATE_WRITEBACK",
        "seed": core.TRAIN_SEED,
        "lr": 0.003 if condition == "C5" else 0.001,
        "target_exposures": exposure,
        "optimizer_updates": update,
        "gradient_horizon": HORIZON,
        "validation": validation,
        "selection_metric": "validation.primary",
        "bank": cpu_state_dict(bank),
        "layer_ids": tuple(core.GDN_LAYERS),
        "source_git_commit": git("rev-parse", "HEAD"),
        "preregistration_sha256": file_sha256(ROOT / "preregistration.json"),
        "canonical_source_sha256": provenance["canonical_c5_c6_source"]["actual_sha256"],
        "rotation_source_sha256": provenance["rotation_source"]["actual_sha256"],
        "document_order_sha256": protocol["training"]["common"]["document_order_sha256"],
        "AIME_used": False,
    }


def formal_train(condition: str) -> None:
    condition = condition.upper()
    summary_path = ROOT / f"analysis/{condition}_training_summary.json"
    failure_path = ROOT / f"analysis/{condition}_formal_failure.json"
    if summary_path.exists() or failure_path.exists():
        raise RuntimeError(f"formal {condition} run already attempted; no silent retry")
    preflight_path = ROOT / f"analysis/{condition}_preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text()).get("status") != "PASS":
        raise RuntimeError(f"{condition} preflight is not PASS")
    protocol, provenance = verify_static_protocol(condition)
    random.seed(core.TRAIN_SEED)
    torch.manual_seed(core.TRAIN_SEED)
    lr = 0.003 if condition == "C5" else 0.001
    objective = "DENSE_STATE" if condition == "C5" else "DENSE_FUNCTIONAL"
    run_started = time.time()
    last_committed_update = 0
    try:
        model, tokenizer, device = runtime.load_model_and_tokenizer(core)
        model_before = model_sentinels(model)
        bank = fresh_bank(device)
        initial = verify_initialization(bank, protocol)
        optimizer = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=0.0)
        train_rows = core.corpus_rows("TRAIN")
        validation_rows = core.corpus_rows("VALIDATION")
        schedule_rng = random.Random(core.TRAIN_SEED)
        schedule = [train_rows[schedule_rng.randrange(len(train_rows))]
                    for _ in range(core.OPTIMIZER_UPDATES)]
        if [row["document_id"] for row in schedule] != protocol["training"]["common"]["document_order"]:
            raise RuntimeError("document order mismatch before formal run")
        validation_updates = set(protocol["validation"]["actual_validation_updates"])
        history = []
        validation_candidates = []
        best_primary = float("inf")
        selected = None
        cleanup_allocations = []
        peak_allocated = 0
        peak_reserved = 0
        log_path = ROOT / f"logs/{condition}_updates.jsonl"
        recovery_path = ROOT / f"checkpoints/{condition}/recovery_latest.pt"
        if log_path.exists() or recovery_path.exists():
            raise RuntimeError("preexisting training log/recovery checkpoint; refusing retry")
        h = core.hadamard(device)
        print(json.dumps({"event": "formal_training_start", "condition": condition,
                          "updates": core.OPTIMIZER_UPDATES, "H": HORIZON,
                          "initialization_manifest_sha256": initial["manifest_sha256"]}), flush=True)
        with core.RecurrentExperimentPatch(model, bank, h) as patch:
            for update, row in enumerate(schedule, 1):
                record = train_document(model, tokenizer, bank, patch, optimizer, device,
                                        row, condition, update)
                last_committed_update = update
                exposure = update * core.TARGETS_PER_DOCUMENT
                record["target_exposures"] = exposure
                cleanup_allocations.append(record["CUDA_allocated_bytes"])
                peak_allocated = max(peak_allocated, record["peak_CUDA_allocated_bytes"])
                peak_reserved = max(peak_reserved, record["peak_CUDA_reserved_bytes"])
                if len(cleanup_allocations) >= 3 and max(cleanup_allocations) - min(cleanup_allocations) > MEMORY_CREEP_BYTES:
                    raise RuntimeError("post-cleanup allocated-memory span exceeds frozen 256 MiB stop rule")

                atomic_torch(recovery_path, {
                    "phase": "post_step_pre_validation" if update in validation_updates else "ready_next_update",
                    "condition": condition, "last_committed_optimizer_step": update,
                    "next_update": update + 1, "bank": cpu_state_dict(bank),
                    "optimizer": optimizer.state_dict(), "schedule_document_id": row["document_id"],
                    "document_order_sha256": protocol["training"]["common"]["document_order_sha256"],
                    "preregistration_sha256": file_sha256(ROOT / "preregistration.json"),
                    "retry_count": 0,
                })
                record["recovery_checkpoint_sha256_post_step"] = file_sha256(recovery_path)

                if update in validation_updates:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    validation_started = time.time()
                    with torch.no_grad():
                        validation = core.evaluate_validation(model, tokenizer, bank, patch,
                                                              validation_rows, HORIZON, objective)
                    validation_memory = memory_snapshot(device, "after_validation")
                    if not all(math.isfinite(float(value)) for key, value in validation.items()
                               if key not in ("samples", "documents")):
                        raise FloatingPointError(f"NaN/Inf validation at update {update}")
                    ortho = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
                    if ortho["status"] != "PASS":
                        raise RuntimeError(f"invalid rotation after validation update {update}")
                    candidate_path = ROOT / f"checkpoints/{condition}/candidate_update_{update:03d}_exposure_{exposure:04d}.pt"
                    payload = checkpoint_payload(condition, update, exposure, validation,
                                                 bank, protocol, provenance)
                    atomic_torch(candidate_path, payload)
                    candidate = {
                        "update": update, "target_exposures": exposure,
                        "validation": validation,
                        "primary": validation["primary"],
                        "checkpoint_path": str(candidate_path.relative_to(ROOT)),
                        "checkpoint_sha256": file_sha256(candidate_path),
                        "validation_wall_clock_seconds": time.time() - validation_started,
                        "validation_peak_allocated_bytes": validation_memory["peak_allocated_bytes"],
                        "validation_peak_reserved_bytes": validation_memory["peak_reserved_bytes"],
                    }
                    validation_candidates.append(candidate)
                    if validation["primary"] < best_primary:
                        best_primary = validation["primary"]
                        selected = candidate.copy()
                    record["validation"] = candidate
                    atomic_torch(recovery_path, {
                        "phase": "ready_next_update", "condition": condition,
                        "last_committed_optimizer_step": update, "next_update": update + 1,
                        "bank": cpu_state_dict(bank), "optimizer": optimizer.state_dict(),
                        "schedule_document_id": row["document_id"],
                        "document_order_sha256": protocol["training"]["common"]["document_order_sha256"],
                        "preregistration_sha256": file_sha256(ROOT / "preregistration.json"),
                        "validation_candidate": candidate, "retry_count": 0,
                    })
                    record["recovery_checkpoint_sha256_post_validation"] = file_sha256(recovery_path)
                history.append(record)
                append_jsonl(log_path, record)
                print(json.dumps({"event": "update_complete", "condition": condition,
                                  "update": update, "document_id": row["document_id"],
                                  "aggregate_loss": record["aggregate_loss"],
                                  "gradient_norm": record["gradient_norm_before_clip"],
                                  "theta_norm": record["theta_norm"],
                                  "peak_reserved_bytes": record["peak_CUDA_reserved_bytes"],
                                  "validation_primary": record.get("validation", {}).get("primary")}), flush=True)
        if selected is None or len(validation_candidates) != len(validation_updates):
            raise RuntimeError("checkpoint-selection ambiguity or missing validation candidates")
        if len(history) != core.OPTIMIZER_UPDATES:
            raise RuntimeError("formal update count mismatch")
        model_after = model_sentinels(model)
        if model_before != model_after or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("base model changed during formal training")
        memory_span = max(cleanup_allocations) - min(cleanup_allocations)
        training_summary = {
            "experiment": ROOT.name, "condition": condition,
            f"FORMAL_{condition}_TRAINING": "COMPLETE",
            "status": "PASS", "objective": objective,
            "optimized_loss": "state" if condition == "C5" else "functional + 0.1 * state",
            "total_updates": len(history), "total_target_exposures": len(history) * 8,
            "unique_training_documents": len(set(item["document_id"] for item in history)),
            "H": HORIZON, "seed": core.TRAIN_SEED, "optimizer": "Adam", "lr": lr,
            "weight_decay": 0.0, "gradient_clip_global_norm": 1.0,
            "initialization": initial,
            "selected_checkpoint": selected,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "post_cleanup_allocated_span_bytes": memory_span,
            "memory_creep": "NO" if memory_span <= MEMORY_CREEP_BYTES else "YES",
            "NaN_Inf": False, "retry_count": 0,
            "runtime_seconds": time.time() - run_started,
            "model_sentinels_unchanged": True,
            "AIME_EVALUATION": "NOT_STARTED",
            "history": history,
        }
        validation_summary = {
            "condition": condition, "status": "PASS",
            "primary_metric": protocol["validation"][f"{condition}_primary_metric"],
            "candidate_count": len(validation_candidates),
            "candidates": validation_candidates,
            "selected_update": selected["update"],
            "selected_primary": selected["primary"],
            "AIME_used": False,
        }
        selection_summary = {
            "condition": condition, "status": "PASS",
            "selection_rule": protocol["validation"]["selection_rule"],
            "tie_breaking_rule": protocol["validation"]["tie_breaking_rule"],
            "all_candidates": validation_candidates,
            "selected_checkpoint": selected["checkpoint_path"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "selected_update": selected["update"],
            "selected_target_exposures": selected["target_exposures"],
            "selected_validation_primary": selected["primary"],
            "AIME_used": False,
        }
        atomic_json(summary_path, training_summary)
        atomic_json(ROOT / f"analysis/{condition}_validation_summary.json", validation_summary)
        atomic_json(ROOT / f"analysis/{condition}_checkpoint_selection.json", selection_summary)
        print(json.dumps({"event": "formal_training_complete", "condition": condition,
                          "selected_checkpoint": selected["checkpoint_path"],
                          "selected_primary": selected["primary"],
                          "runtime_seconds": training_summary["runtime_seconds"]}), flush=True)
    except BaseException as exc:
        atomic_json(failure_path, {
            "condition": condition, "status": "FAILED", "last_committed_optimizer_step": last_committed_update,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "retry_count": 0, "AIME_EVALUATION": "NOT_STARTED",
        })
        raise


def differing_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [path for key in sorted(set(left) | set(right))
                for path in differing_paths(left.get(key), right.get(key), f"{prefix}.{key}")]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [path for index, (a, b) in enumerate(zip(left, right))
                for path in differing_paths(a, b, f"{prefix}[{index}]")]
    return [] if left == right else [prefix]


def export_condition(condition: str) -> None:
    condition = condition.upper()
    output = ROOT / f"analysis/{condition}_export_summary.json"
    error_output = ROOT / f"analysis/{condition}_export.error.json"
    if output.exists() or error_output.exists():
        raise RuntimeError(f"{condition} export already attempted")
    protocol, provenance = verify_static_protocol(condition)
    summary = json.loads((ROOT / f"analysis/{condition}_training_summary.json").read_text())
    selection = json.loads((ROOT / f"analysis/{condition}_checkpoint_selection.json").read_text())
    if summary.get("status") != "PASS" or selection.get("status") != "PASS":
        raise RuntimeError(f"{condition} training/selection not complete")
    selected_path = ROOT / selection["selected_checkpoint"]
    if file_sha256(selected_path) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected theta checkpoint hash mismatch")
    try:
        artifact_dir = ROOT / f"artifacts/{condition}"
        deployment_dir = ROOT / f"deployment/{condition}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        deployment_dir.mkdir(parents=True, exist_ok=True)
        theta_artifact = artifact_dir / "theta_checkpoint.pt"
        if theta_artifact.exists():
            raise RuntimeError("theta artifact already exists")
        shutil.copy2(selected_path, theta_artifact)
        theta_sha = file_sha256(theta_artifact)
        if theta_sha != selection["selected_checkpoint_sha256"]:
            raise RuntimeError("theta artifact copy hash mismatch")
        payload = torch.load(theta_artifact, map_location="cpu", weights_only=False)
        bank = fresh_bank(torch.device("cuda:0"))
        bank.load_state_dict(payload["bank"], strict=True)
        matrices = {}
        per_layer = {}
        identity = torch.eye(128, dtype=torch.float32)
        for layer in core.GDN_LAYERS:
            matrix = bank.layer(layer).matrix().detach().float().cpu().contiguous()
            residual = float((matrix.T @ matrix - identity).abs().max())
            sign, log_abs_det = torch.linalg.slogdet(matrix)
            if (matrix.shape != (128, 128) or not bool(torch.isfinite(matrix).all()) or
                    residual > core.ORTHOGONALITY_THRESHOLD or float(sign) <= 0):
                raise RuntimeError(f"invalid rotation matrix at layer {layer}")
            digest = runtime.tensor_sha256(matrix)
            matrices[str(layer)] = matrix
            per_layer[str(layer)] = {
                "layer_id": layer, "shape": [128, 128], "dtype": "torch.float32",
                "sha256": digest, "orthogonality_max_abs_rt_r_minus_i": residual,
                "determinant_sign": float(sign), "determinant_log_abs": float(log_abs_det),
            }
        frozen_path = artifact_dir / "frozen_rotation_fp32.pt"
        frozen_payload = {
            "format_version": 1, "experiment": ROOT.name, "condition": condition,
            "layer_ids": tuple(core.GDN_LAYERS), "matrices": matrices,
            "matrix_dtype": "torch.float32", "matrix_shape": [128, 128],
            "matrix_meaning": "Canonical Cayley Delta_l materialized on the training GPU and copied as contiguous FP32 CPU bytes; not precomposed H@Delta.",
            "orientation": protocol["rotation"]["runtime_orientation"],
            "theta_checkpoint_sha256": theta_sha,
            "rotation_source_sha256": provenance["rotation_source"]["actual_sha256"],
            "canonical_source_sha256": provenance["canonical_c5_c6_source"]["actual_sha256"],
            "source_git_commit": git("rev-parse", "HEAD"),
            "export_host": {"hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        }
        atomic_torch(frozen_path, frozen_payload)
        frozen_sha = file_sha256(frozen_path)
        rotation_manifest = {
            "condition": condition, "status": "EXPORTED",
            "theta_checkpoint_path": str(theta_artifact.relative_to(ROOT)),
            "theta_checkpoint_sha256": theta_sha,
            "frozen_rotation_path": str(frozen_path.relative_to(ROOT)),
            "frozen_rotation_sha256": frozen_sha,
            "layer_count": len(matrices), "layer_ids": list(core.GDN_LAYERS),
            "dtype": "torch.float32", "shape_per_layer": [128, 128],
            "serialization_format": "torch.save pickle/zip payload",
            "matrix_meaning": frozen_payload["matrix_meaning"],
            "runtime_orientation": frozen_payload["orientation"],
            "per_layer": per_layer,
        }
        atomic_json(artifact_dir / "rotation_manifest.json", rotation_manifest)
        atomic_json(artifact_dir / "per_layer_hashes.json",
                    {layer: row["sha256"] for layer, row in per_layer.items()})

        frozen_bank = runtime.FrozenMatrixRotationBank(core.GDN_LAYERS, matrices).to("cuda:0")
        exact_layers = {}
        for layer in core.GDN_LAYERS:
            theta_hash = runtime.tensor_sha256(bank.layer(layer).matrix().detach().cpu().contiguous())
            frozen_hash = runtime.tensor_sha256(frozen_bank.layer(layer).matrix().detach().cpu().contiguous())
            exact_layers[str(layer)] = theta_hash == frozen_hash == per_layer[str(layer)]["sha256"]
        if not all(exact_layers.values()):
            raise RuntimeError("frozen/theta per-layer matrix hash mismatch")
        probe_input = torch.arange(256, dtype=torch.float32, device="cuda:0").reshape(2, 128) / 257.0
        h = core.hadamard(torch.device("cuda:0"))
        operator_theta = probe_input @ h @ bank.layer(core.GDN_LAYERS[0]).matrix()
        operator_frozen = probe_input @ h @ frozen_bank.layer(core.GDN_LAYERS[0]).matrix()
        operator_exact = runtime.tensor_sha256(operator_theta) == runtime.tensor_sha256(operator_frozen)
        if not operator_exact:
            raise RuntimeError("rotation operator check failed")
        model, tokenizer, _ = runtime.load_model_and_tokenizer(core)
        theta_probe = frozen_probe.run_fixed_probe(core, model, tokenizer, bank)
        frozen_result = frozen_probe.run_fixed_probe(core, model, tokenizer, frozen_bank)
        mismatches = differing_paths(theta_probe, frozen_result)
        if mismatches:
            raise RuntimeError(f"frozen functional probe mismatch: {mismatches[:8]}")
        sanity = {
            "condition": condition, "status": "PASS",
            "all_24_matrix_hashes_exact": all(exact_layers.values()),
            "per_layer_exact": exact_layers,
            "rotation_operator_exact": operator_exact,
            "C128_quantization_path_check": "PASS",
            "recurrent_state_writeback_check": "PASS" if all(theta_probe["writeback_exact_by_layer"].values()) else "FAIL",
            "theta_probe": theta_probe, "frozen_probe": frozen_result,
            "exact_probe_mismatch_paths": mismatches,
            "AIME_used": False,
        }
        if sanity["recurrent_state_writeback_check"] != "PASS":
            raise RuntimeError("recurrent state writeback sanity failed")
        atomic_json(ROOT / f"analysis/{condition}_frozen_functional_sanity.json", sanity)

        deployment_frozen = deployment_dir / "frozen_rotation_fp32.pt"
        shutil.copy2(frozen_path, deployment_frozen)
        if file_sha256(deployment_frozen) != frozen_sha:
            raise RuntimeError("deployment artifact copy hash mismatch")
        deployment_manifest = {
            "condition": condition, "status": "READY",
            "loading_rule": "load frozen rotation directly; do not reconstruct R from theta on inference host",
            "frozen_rotation_file": "frozen_rotation_fp32.pt",
            "file_sha256": frozen_sha,
            "per_layer_sha256": {layer: row["sha256"] for layer, row in per_layer.items()},
            "expected_layer_ids": list(core.GDN_LAYERS),
            "expected_shape_per_layer": [128, 128], "expected_dtype": "torch.float32",
            "source_checkpoint_sha256": theta_sha,
            "source_git_commit": git("rev-parse", "HEAD"),
            "quantizer_mode": "symmetric INT8 C128; key/K axis -2; round-to-even; clamp [-127,127]",
            "rotation_side": "canonical Qwen/GDN Key-side two-stage H128 then stored Delta",
            "INT8_grouping": "C128 over K dimension",
            "functional_probe_instructions": "Load payload matrices directly into FrozenMatrixRotationBank; verify file and 24 tensor hashes; run the fixed non-AIME 32-token operator/QDQ/writeback probe.",
            "AIME_EVALUATION": "NOT_STARTED",
        }
        atomic_json(deployment_dir / "deployment_manifest.json", deployment_manifest)
        atomic_text(deployment_dir / "LOADING_INSTRUCTIONS.md", f"""# {condition} frozen rotation deployment

Verify `frozen_rotation_fp32.pt` SHA256 is `{frozen_sha}` and verify all 24 tensor hashes in `deployment_manifest.json`. Load the stored FP32 matrices directly. Do not reconstruct matrices from theta. Apply each stored Delta after canonical H128 on row-vector q/k; recurrent recovery applies Delta and then H128 on the key axis. Keep symmetric INT8-C128 grouping on K (`dim=-2`), round-to-even, and clamp to [-127,127]. Run the fixed non-AIME functional probe before downstream evaluation. AIME remains NOT_STARTED in this task.
""")
        result = {
            "condition": condition, "status": "PASS",
            f"{condition}_DEPLOYMENT_ARTIFACT_READY": "YES",
            "theta_checkpoint_sha256": theta_sha,
            "frozen_rotation_sha256": frozen_sha,
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_validation_primary": selection["selected_validation_primary"],
            "layer_count": 24, "functional_sanity": "PASS",
            "deployment_manifest": str((deployment_dir / "deployment_manifest.json").relative_to(ROOT)),
            "AIME_EVALUATION": "NOT_STARTED",
        }
        atomic_json(output, result)
        print(json.dumps({"event": "export_complete", **result}), flush=True)
    except BaseException as exc:
        atomic_json(error_output, {"condition": condition, "status": "FAIL",
                                   "error_type": type(exc).__name__, "error": str(exc),
                                   "traceback": traceback.format_exc(),
                                   "AIME_EVALUATION": "NOT_STARTED"})
        raise


def build_final_reports() -> None:
    records = {}
    for condition in ("C5", "C6"):
        train = json.loads((ROOT / f"analysis/{condition}_training_summary.json").read_text())
        validation = json.loads((ROOT / f"analysis/{condition}_validation_summary.json").read_text())
        selection = json.loads((ROOT / f"analysis/{condition}_checkpoint_selection.json").read_text())
        export = json.loads((ROOT / f"analysis/{condition}_export_summary.json").read_text())
        if any(item.get("status") != "PASS" for item in (train, validation, selection, export)):
            raise RuntimeError(f"{condition} final gate is not PASS")
        records[condition] = {"train": train, "validation": validation,
                              "selection": selection, "export": export}
    environment = json.loads((ROOT / "configs/environment.json").read_text())
    lines = [
        f"# Final report: {ROOT.name}", "",
        "## Environment", "",
        f"Training used one `{environment['gpu_model']}` with {environment['vram_gib']:.3f} GiB visible VRAM. No distributed framework was used.", "",
        "## Protocol", "",
        "The prospective protocol was frozen before training. Both conditions used H64, 1024-token documents, eight canonical captures per document, real recurrent INT8-C128 writeback, H64 graph detachment, and exactly one optimizer step per document. C5 and C6 independently initialized from theta=0 / Hadamard.", "",
        "## Results", "",
        "| Condition | Status | Updates | Exposures | Validation primary | Selected checkpoint | Peak reserved GiB | Memory creep | NaN/Inf | Retry | Deployment |",
        "|---|---|---:|---:|---:|---|---:|---|---|---:|---|",
    ]
    for condition in ("C5", "C6"):
        row = records[condition]
        train, select, export = row["train"], row["selection"], row["export"]
        lines.append(
            f"| {condition} | COMPLETE | {train['total_updates']} | {train['total_target_exposures']} | "
            f"{select['selected_validation_primary']:.12g} | `{select['selected_checkpoint']}` | "
            f"{train['peak_reserved_bytes'] / 2**30:.3f} | {train['memory_creep']} | NO | "
            f"{train['retry_count']} | {export[f'{condition}_DEPLOYMENT_ARTIFACT_READY']} |"
        )
    lines.extend(["", "## Comparison", "",
                  "Only training, held-out local/recurrent validation, and frozen-loader diagnostics are compared here. No downstream benchmark was run and no AIME conclusion is made.", ""])
    c5 = records["C5"]["selection"]["selected_validation_primary"]
    c6 = records["C6"]["selection"]["selected_validation_primary"]
    lines.append(f"C5 state-primary minimum: `{c5:.12g}`. C6 functional-primary minimum: `{c6:.12g}`. These are different preregistered metrics and are reported without treating their raw magnitudes as a direct superiority test.")
    lines.extend(["", "## Deployment", "",
                  "`C5_DEPLOYMENT_ARTIFACT_READY=YES`", "",
                  "`C6_DEPLOYMENT_ARTIFACT_READY=YES`", "",
                  "Both deployment packages require directly loading the frozen FP32 matrices; inference-host reconstruction from theta is non-canonical.", "",
                  "## Downstream", "", "`AIME_EVALUATION=NOT_STARTED`", ""])
    atomic_text(ROOT / "reports/FINAL_REPORT.md", "\n".join(lines))
    result_lines = [f"# Formal training result: {ROOT.name}", "",
                    "FORMAL_PROTOCOL_FROZEN = YES",
                    "C5_FORMAL_TRAINING = COMPLETE",
                    "C6_FORMAL_TRAINING = COMPLETE",
                    f"C5_SELECTED_CHECKPOINT = {records['C5']['selection']['selected_checkpoint']}",
                    f"C6_SELECTED_CHECKPOINT = {records['C6']['selection']['selected_checkpoint']}",
                    f"C5_VALIDATION_PRIMARY = {c5:.12g}",
                    f"C6_VALIDATION_PRIMARY = {c6:.12g}",
                    f"C5_THETA_SHA256 = {records['C5']['export']['theta_checkpoint_sha256']}",
                    f"C6_THETA_SHA256 = {records['C6']['export']['theta_checkpoint_sha256']}",
                    f"C5_FROZEN_ROTATION_SHA256 = {records['C5']['export']['frozen_rotation_sha256']}",
                    f"C6_FROZEN_ROTATION_SHA256 = {records['C6']['export']['frozen_rotation_sha256']}",
                    "C5_DEPLOYMENT_READY = YES", "C6_DEPLOYMENT_READY = YES",
                    "AIME_EVALUATION = NOT_STARTED", ""]
    atomic_text(ROOT / "reports/FORMAL_TRAINING_RESULT.md", "\n".join(result_lines))

    hash_dir = ROOT / "hashes"
    hash_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = hash_dir / "artifact_sha256.txt"
    provenance_path = hash_dir / "inventory_provenance.json"
    excluded = {inventory_path.resolve(), provenance_path.resolve()}

    def write_inventory() -> list[Path]:
        files = sorted(path for path in ROOT.rglob("*") if path.is_file() and
                       path.resolve() not in excluded and ".tmp." not in path.name)
        content = "".join(f"{file_sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files)
        atomic_text(inventory_path, content)
        return files

    files = write_inventory()
    artifact_count = len(files)
    # Add the stable count to the result report, then refresh the inventory.
    with (ROOT / "reports/FORMAL_TRAINING_RESULT.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"ARTIFACT_COUNT = {artifact_count}\nHASH_VERIFICATION = PASS\n")
    files = write_inventory()
    for line in inventory_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"artifact hash verification failed: {relative}")
    inventory_sha = file_sha256(inventory_path)
    inventory_provenance = {
        "status": "PASS", "artifact_count": len(files),
        "inventory_path": str(inventory_path.relative_to(ROOT)),
        "inventory_sha256": inventory_sha,
        "verified_entries": len(files),
        "intentional_self_reference_exclusions": [
            str(inventory_path.relative_to(ROOT)), str(provenance_path.relative_to(ROOT))
        ],
        "git_branch": git("branch", "--show-current"),
        "git_head_before_final_report_commit": git("rev-parse", "HEAD"),
        "AIME_EVALUATION": "NOT_STARTED",
    }
    atomic_json(provenance_path, inventory_provenance)
    print(json.dumps({"event": "final_reports_complete", "artifact_count": len(files),
                      "inventory_sha256": inventory_sha, "HASH_VERIFICATION": "PASS",
                      "AIME_EVALUATION": "NOT_STARTED"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train", "export"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--condition", choices=("C5", "C6"), required=True)
    subparsers.add_parser("final-report")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.condition)
    elif args.command == "train":
        formal_train(args.condition)
    elif args.command == "export":
        export_condition(args.condition)
    else:
        build_final_reports()


if __name__ == "__main__":
    main()
