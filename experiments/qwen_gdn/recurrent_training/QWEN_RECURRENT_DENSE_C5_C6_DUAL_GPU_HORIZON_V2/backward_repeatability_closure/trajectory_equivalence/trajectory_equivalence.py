#!/usr/bin/env python3
"""Append-only eight-update C5 H32 trajectory diagnostic; never formal training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CLOSURE = ROOT.parent
V2_DIR = CLOSURE.parent
REPO = V2_DIR.parents[1]
CONFIGS = ROOT / "configs"
ANALYSIS = ROOT / "analysis"
RAW = ROOT / "raw"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
HASHES = ROOT / "hashes"
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = import_file(V2_DIR / "run_dual_gpu_v2.py", "qwen_v2_for_trajectory_read_only")
V1 = V2.V1_IMPL


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def new_json(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def new_npz(path: Path, tensors: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **tensors)


def git_head(path: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def tokenizer_only():
    base = V1.BASE
    if str(base.TRANSFORMERS_SRC) not in sys.path:
        sys.path.insert(0, str(base.TRANSFORMERS_SRC))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(base.MODEL_PATH), trust_remote_code=True, local_files_only=True)


def schedule_rows() -> list[dict]:
    rows = V1.corpus_rows("TRAIN")
    rng = random.Random(V1.TRAIN_SEED)
    return [rows[rng.randrange(len(rows))] for _ in range(8)]


def prepare() -> None:
    if (CONFIGS / "frozen_trajectory_protocol.json").exists():
        raise RuntimeError("trajectory already preregistered")
    for path in (CONFIGS, ANALYSIS, RAW, LOGS, REPORTS, HASHES):
        path.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_only()
    schedule = []
    for index, row in enumerate(schedule_rows(), 1):
        ids, positions = V1.tokenize(tokenizer, row)
        if len(ids) != 1024 or len(positions) != 8:
            raise RuntimeError(f"unexpected original-training schedule at update {index}")
        schedule.append({
            "update": index, "document_id": row["document_id"],
            "input_token_ids": ids, "input_token_ids_sha256": sha_json(ids),
            "teacher_token_ids_sha256": sha_json(ids),
            "target_positions": positions, "target_positions_sha256": sha_json(positions),
            "trace_token_indices": sorted(set((31, 32, *positions))),
            "token_count": len(ids),
        })
    baseline = json.loads((CLOSURE / "analysis/closure_verdict.json").read_text())
    if baseline["SINGLE_GPU_BACKWARD_BITWISE_REPEATABILITY"] != "FAIL":
        raise RuntimeError("prior closure assumption changed")
    runtime = json.loads((CLOSURE / "configs/runtime_source_revisions.json").read_text())
    protocol = {
        "task": "QWEN_DUAL_GPU_TRAJECTORY_EQUIVALENCE_CLOSURE_V1",
        "parent": "QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2",
        "prior_backward_closure_verdict_sha256": sha_file(CLOSURE / "analysis/closure_verdict.json"),
        "prior_numerical_protocol_sha256": sha_file(CLOSURE / "protocol_amendments/backward_numerical_equivalence_v1.json"),
        "model_path": str(V1.BASE.MODEL_PATH),
        "model_config_sha256": runtime["model_config_sha256"],
        "transformers_qwen35_git_commit": runtime["transformers_qwen35_git_commit"],
        "qwen_experiment_git_commit": git_head(REPO),
        "v1_training_source_sha256": sha_file(V2.V1_SCRIPT),
        "v2_topology_source_sha256": sha_file(V2_DIR / "run_dual_gpu_v2.py"),
        "qdq_source_sha256": sha_file(V1.CAYLEY_SOURCE),
        "diagnostic_script_sha256_at_preregistration": sha_file(Path(__file__)),
        "condition": "C5_DENSE_STATE",
        "rotation_initialization": "Hadamard_step0_theta0",
        "gradient_horizon": 32,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "seed": V1.TRAIN_SEED,
        "single_gpu_repeats": 10,
        "dual_gpu_repeats": 5,
        "single_gpu_physical_device": 0,
        "dual_gpu_visible_devices": [0, 1],
        "dual_gpu_contiguous_split_block": 16,
        "optimizer_updates": 8,
        "schedule": schedule,
        "schedule_source": "exact_first_eight_rows_of_original_C5_seed0_train_schedule_full_1024_tokens",
        "validation": "disabled_identically_for_all_diagnostic_runs_no_optimizer_effect",
        "comparison_pairs": {"single_internal": 45, "single_dual": 50, "dual_internal_supplemental": 10},
        "epsilon": 1e-12,
        "envelope_aggregation": "strict_max_of_all_45_single_pairs_per_update_per_layer_and_global_no_multiplier",
        "loss_metrics": ["absolute_difference", "relative_difference"],
        "gradient_metrics": ["relative_l2", "max_abs", "cosine_error", "gradient_norm_absolute_difference"],
        "rotation_metrics": ["relative_l2", "max_abs", "cosine_error", "orthogonality_error_absolute_difference"],
        "endpoint_global_rotation_metrics": ["relative_l2", "max_abs", "cosine_error"],
        "strict_gate": "every_single_dual_pair_each_update_each_layer_and_metric_le_frozen_single_max",
        "hashes": "provenance_only_not_equivalence_gate",
        "forward_exact_interpretation": {
            "each_run_each_update_exact": ["input_ids", "teacher_token_ids", "teacher_target_tensors", "target_positions", "layer_mapping", "recurrent_writeback", "BPTT_boundary"],
            "common_theta0_update1_exact_across_topologies": ["240_sampled_pre_QDQ", "scale", "qcodes", "post_QDQ", "consumed_student_state", "loss_components"],
            "updates_2_to_8": "student_state_QDQ_and_loss_may_drift_after_valid_optimizer_updates_and_are_measured_by_trajectory_and_endpoint_probe_not_bitwise_compared",
        },
        "functional_probe": {
            "fixed_non_AIME_input": "first_192_tokens_of_prior_frozen_backward_reference",
            "single_envelope": "strict_max_of_45_single_endpoint_pairs_for_numeric_tensors_and_qcode_disagreement_count",
            "cross_gate": "all_50_single_dual_pairs_within_frozen_single_endpoint_envelope",
            "numeric_metrics": ["relative_l2", "max_abs"],
            "discrete_metric": "qcodes_mismatch_count",
            "freeze_before_dual_probe_analysis": True,
        },
        "no_AIME_or_formal_training": True,
        "stop_after_final_report_even_if_PASS": True,
    }
    new_json(CONFIGS / "frozen_trajectory_protocol.json", protocol)
    new_json(ANALYSIS / "frozen_trajectory_protocol.json", protocol)
    if sha_file(CONFIGS / "frozen_trajectory_protocol.json") != sha_file(ANALYSIS / "frozen_trajectory_protocol.json"):
        raise RuntimeError("frozen protocol copies mismatch")
    print(json.dumps({"PREREGISTERED": True, "schedule_documents": [row["document_id"] for row in schedule],
                      "protocol_sha256": sha_file(CONFIGS / "frozen_trajectory_protocol.json")}, indent=2))


def per_layer_vector_stats(value: torch.Tensor) -> dict:
    cpu = value.detach().float().cpu()
    return {"sha256": tensor_hash(value), "shape": list(value.shape), "dtype": str(value.dtype),
            "frobenius_norm": float(torch.linalg.vector_norm(cpu)),
            "max_abs": float(cpu.abs().max()),
            "finite": bool(torch.isfinite(cpu).all())}


def rotation_stats(rotation: np.ndarray, initial: np.ndarray) -> dict:
    r = rotation.astype(np.float64)
    h = initial.astype(np.float64)
    delta = r - h
    cosine = float(np.dot(r.ravel(), h.ravel()) / (np.linalg.norm(r) * np.linalg.norm(h)))
    eye = np.eye(r.shape[0], dtype=np.float64)
    return {
        "rotation_frobenius_norm": float(np.linalg.norm(r)),
        "rotation_max_abs": float(np.max(np.abs(r))),
        "delta_R_frobenius_norm": float(np.linalg.norm(delta)),
        "delta_R_max_abs": float(np.max(np.abs(delta))),
        "cosine_similarity_to_initial_H": cosine,
        "orthogonality_error_max_abs": float(np.max(np.abs(r.T @ r - eye))),
        "rotation_sha256": hashlib.sha256(rotation.tobytes()).hexdigest(),
    }


def run(mode: str, run_id: int) -> None:
    if mode not in ("single", "dual") or run_id < 1:
        raise ValueError("invalid run")
    stem = f"{mode}_{run_id:02d}"
    complete_path = ANALYSIS / f"{stem}.json"
    error_path = ANALYSIS / f"{stem}.error.json"
    progress_path = LOGS / f"{stem}_updates.jsonl"
    if complete_path.exists() or error_path.exists() or progress_path.exists():
        raise RuntimeError(f"refusing to overwrite trajectory {stem}")
    protocol_path = CONFIGS / "frozen_trajectory_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if sha_file(protocol_path) != sha_file(ANALYSIS / "frozen_trajectory_protocol.json"):
        raise RuntimeError("frozen protocol mismatch")
    random.seed(V1.TRAIN_SEED)
    torch.manual_seed(V1.TRAIN_SEED)
    if mode == "single":
        model, tokenizer, bank, device = V1.load_model_bank()
        patch_factory = lambda: V1.RecurrentExperimentPatch(model, bank, V1.hadamard(device))
    else:
        model, tokenizer, bank, layer_devices = V2.load_dual_model_bank(protocol["dual_gpu_contiguous_split_block"])
        device = model.get_input_embeddings().weight.device
        patch_factory = lambda: V2.DualRecurrentPatch(model, bank)
    if any(bool((bank.layer(layer).theta != 0).any()) for layer in LAYERS):
        raise RuntimeError("theta0 nonzero")
    initial_rotation = {str(layer): bank.layer(layer).matrix().detach().cpu().numpy().copy() for layer in LAYERS}
    optimizer = torch.optim.Adam(bank.parameters(), lr=protocol["learning_rate"], weight_decay=0.0)
    selected_rows = schedule_rows()
    if [row["document_id"] for row in selected_rows] != [row["document_id"] for row in protocol["schedule"]]:
        raise RuntimeError("schedule document mismatch")
    results = []
    original_qdq = V1.CAYLEY.qwen_c128_ste
    try:
        with patch_factory() as patch:
            for update, (row, frozen) in enumerate(zip(selected_rows, protocol["schedule"]), 1):
                ids, positions = V1.tokenize(tokenizer, row)
                if ids != frozen["input_token_ids"] or positions != frozen["target_positions"]:
                    raise RuntimeError(f"frozen tokens/positions mismatch update {update}")
                trace_set = set(frozen["trace_token_indices"])
                position_set = set(positions)
                current_token = -1
                trace = {}
                per_target_losses = []
                boundary = []
                segments = []
                cache = None
                for gpu in range(2 if mode == "dual" else 1):
                    torch.cuda.reset_peak_memory_stats(gpu)
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
                        key = f"{current_token}:{layer}"
                        trace.setdefault(key, {}).update({
                            "pre_qdq_state_sha256": tensor_hash(value),
                            "scale_sha256": tensor_hash(quant.scale),
                            "qcodes_sha256": tensor_hash(quant.codes),
                            "post_qdq_state_sha256": tensor_hash(quant.dequant),
                        })
                    return quant

                V1.CAYLEY.qwen_c128_ste = traced_qdq
                try:
                    optimizer.zero_grad(set_to_none=True)
                    for index, token_id in enumerate(ids):
                        current_token = index
                        capture = index in position_set
                        if capture:
                            if mode == "single":
                                V1.install_teacher_target(patch, targets[index], device)
                            else:
                                V2.install_dual_teacher_target(patch, targets[index], layer_devices)
                        if cache is not None and index in trace_set:
                            for layer in LAYERS:
                                trace.setdefault(f"{index}:{layer}", {})["consumed_prev_state_sha256"] = tensor_hash(V1.BASE.get_state(cache, layer))
                        cache = V1.forward_student(model, patch, token_id, cache, capture)
                        if capture:
                            state, functional, c5, c6 = patch.captured_losses()
                            segments.append(c5 / V1.TARGETS_PER_DOCUMENT)
                            per_target_losses.append({"token_index": index,
                                "state": float(state.detach().cpu()), "functional": float(functional.detach().cpu()),
                                "c5": float(c5.detach().cpu()), "c6": float(c6.detach().cpu())})
                            del state, functional, c5, c6
                        if (index + 1) % 32 == 0 or index + 1 == len(ids):
                            if segments:
                                segment_loss = sum(segments)
                                if not bool(torch.isfinite(segment_loss).detach().cpu()):
                                    raise FloatingPointError(f"nonfinite loss at update {update} token {index}")
                                segment_loss.backward()
                                segments.clear()
                                patch.clear_capture()
                                del segment_loss
                            if index == 31:
                                before = {str(layer): tensor_hash(V1.BASE.get_state(cache, layer)) for layer in LAYERS}
                            V1.detach_cache(cache)
                            if index == 31:
                                after = {str(layer): tensor_hash(V1.BASE.get_state(cache, layer)) for layer in LAYERS}
                                boundary.append({"token_index": index, "value_preserved": before == after})
                    if len(per_target_losses) != 8:
                        raise RuntimeError(f"expected eight targets update {update}")
                    gradient_arrays = {}
                    gradient_stats = {}
                    for layer in LAYERS:
                        grad = bank.layer(layer).theta.grad
                        if grad is None or not bool(torch.isfinite(grad).all()):
                            raise FloatingPointError(f"invalid gradient update {update} layer {layer}")
                        gradient_arrays[str(layer)] = grad.detach().cpu().numpy().copy()
                        gradient_stats[str(layer)] = per_layer_vector_stats(grad)
                    gradient_norm = float(np.sqrt(sum(np.sum(x.astype(np.float64) ** 2) for x in gradient_arrays.values())))
                    grad_path = RAW / f"{stem}_update{update:02d}_gradients.npz"
                    new_npz(grad_path, gradient_arrays)
                    torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
                    optimizer.step()
                    rotation_arrays = {str(layer): bank.layer(layer).matrix().detach().cpu().numpy().copy() for layer in LAYERS}
                    theta_arrays = {str(layer): bank.layer(layer).theta.detach().cpu().numpy().copy() for layer in LAYERS}
                    rotation_stats_by_layer = {str(layer): rotation_stats(rotation_arrays[str(layer)], initial_rotation[str(layer)]) for layer in LAYERS}
                    rotation_path = RAW / f"{stem}_update{update:02d}_rotations.npz"
                    new_npz(rotation_path, rotation_arrays)
                    forward_path = RAW / f"{stem}_update{update:02d}_forward.json"
                    writeback = all(trace[f"31:{layer}"]["post_qdq_state_sha256"] ==
                                    trace[f"32:{layer}"]["consumed_prev_state_sha256"] for layer in LAYERS)
                    boundary_pass = writeback and all(item["value_preserved"] for item in boundary)
                    if not writeback or not boundary_pass:
                        raise RuntimeError(f"recurrent writeback/BPTT failure update {update}")
                    new_json(forward_path, {
                        "update": update, "document_id": row["document_id"],
                        "input_token_ids_sha256": sha_json(ids), "teacher_token_ids_sha256": sha_json(ids),
                        "target_positions_sha256": sha_json(positions), "teacher_target_tensor_hashes": teacher_hashes,
                        "trace": trace, "loss_components": per_target_losses,
                        "REAL_RECURRENT_WRITEBACK_GATE": "PASS", "BPTT_BOUNDARY_GATE": "PASS",
                    })
                    result = {
                        "update": update, "document_id": row["document_id"],
                        "loss": sum(x["c5"] for x in per_target_losses) / 8,
                        "state_loss": sum(x["state"] for x in per_target_losses) / 8,
                        "functional_loss": sum(x["functional"] for x in per_target_losses) / 8,
                        "loss_components": per_target_losses,
                        "gradient_norm": gradient_norm,
                        "gradient_per_layer": gradient_stats,
                        "rotation_per_layer": rotation_stats_by_layer,
                        "gradient_artifact": grad_path.name, "gradient_sha256": sha_file(grad_path),
                        "rotation_artifact": rotation_path.name, "rotation_sha256": sha_file(rotation_path),
                        "forward_artifact": forward_path.name, "forward_sha256": sha_file(forward_path),
                        "peak_allocated_bytes": {str(gpu): torch.cuda.max_memory_allocated(gpu)
                                                 for gpu in range(2 if mode == "dual" else 1)},
                    }
                    results.append(result)
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"update": update, "loss": result["loss"],
                            "gradient_norm": gradient_norm, "peak_allocated_bytes": result["peak_allocated_bytes"]}) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    print(json.dumps({"stem": stem, "update": update, "loss": result["loss"],
                                      "gradient_norm": gradient_norm}), flush=True)
                finally:
                    V1.CAYLEY.qwen_c128_ste = original_qdq
                del targets, cache
                patch.clear_capture()
                gc.collect()
                torch.cuda.empty_cache()
        endpoint_theta_path = RAW / f"{stem}_endpoint_theta.npz"
        new_npz(endpoint_theta_path, theta_arrays)
        new_json(complete_path, {
            "stem": stem, "mode": mode, "run_id": run_id,
            "protocol_sha256": sha_file(protocol_path),
            "initial_theta_all_zero": True, "optimizer_fresh": True,
            "updates": results,
            "endpoint_theta_artifact": endpoint_theta_path.name,
            "endpoint_theta_sha256": sha_file(endpoint_theta_path),
            "ALL_REAL_RECURRENT_WRITEBACK_GATES": "PASS",
            "ALL_BPTT_BOUNDARY_GATES": "PASS",
        })
        print(json.dumps({"stem": stem, "COMPLETE_UPDATES": len(results),
                          "endpoint_theta_sha256": sha_file(endpoint_theta_path)}), flush=True)
    except Exception as exc:
        if not error_path.exists():
            new_json(error_path, {"stem": stem, "exception": type(exc).__name__,
                                  "message": str(exc), "traceback": traceback.format_exc(),
                                  "completed_updates": len(results),
                                  "protocol_sha256": sha_file(protocol_path)})
        raise
    finally:
        V1.CAYLEY.qwen_c128_ste = original_qdq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--mode", choices=("single", "dual"))
    parser.add_argument("--run-id", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        if args.mode is None or args.run_id is None:
            parser.error("run requires --mode and --run-id")
        run(args.mode, args.run_id)


if __name__ == "__main__":
    main()
