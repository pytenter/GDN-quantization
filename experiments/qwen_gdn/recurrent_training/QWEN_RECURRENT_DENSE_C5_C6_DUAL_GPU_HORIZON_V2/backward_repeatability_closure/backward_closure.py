#!/usr/bin/env python3
"""Independent, non-AIME H32 backward-repeatability diagnostic for Qwen V2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT.parent
REPO = V2_DIR.parents[1]
ANALYSIS = ROOT / "analysis"
CONFIGS = ROOT / "configs"
RAW = ROOT / "raw"


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = import_file(V2_DIR / "run_dual_gpu_v2.py", "qwen_v2_closure_read_only")
V1 = V2.V1_IMPL


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_new(path: Path, data) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def command_output(*cmd: str) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare() -> None:
    if (ROOT / "preregistration.json").exists():
        raise RuntimeError("closure already preregistered")
    for path in (ANALYSIS, CONFIGS, RAW, ROOT / "protocol_amendments", ROOT / "logs", ROOT / "reports", ROOT / "hashes"):
        path.mkdir(parents=True, exist_ok=True)
    previous = json.loads((V2_DIR / "analysis/one_step_c5_single_repeat1.json").read_text())
    ids = previous["input_token_ids"]
    targets = previous["target_positions"]
    reference = {
        "condition": "C5",
        "rotation_init": "Hadamard_step0_theta0",
        "gradient_horizon": 32,
        "topology_reference": "single_GPU0",
        "source_document_id": previous["source_document_id"],
        "input_token_ids": ids,
        "input_token_ids_sha256": hash_json(ids),
        "teacher_token_ids_sha256": hash_json(ids),
        "target_positions": targets,
        "target_positions_sha256": hash_json(targets),
        "sequence_length": len(ids),
        "objective": "DENSE_STATE",
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "optimizer_state": "fresh_identical",
        "seed": V1.TRAIN_SEED,
        "trace_token_indices": previous["trace_token_indices"],
        "qdq_implementation_sha256": sha_file(V1.CAYLEY_SOURCE),
        "v1_training_source_sha256": sha_file(V2.V1_SCRIPT),
        "v2_topology_source_sha256": sha_file(V2_DIR / "run_dual_gpu_v2.py"),
        "original_c5_config_sha256": sha_file(V2_DIR / "configs/original_c5_config.json"),
        "model_revision": command_output("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "runtime_commit": command_output("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "source_one_step_sha256": sha_file(V2_DIR / "analysis/one_step_c5_single_repeat1.json"),
    }
    write_new(CONFIGS / "frozen_backward_reference.json", reference)
    gpu = []
    for index in range(min(4, torch.cuda.device_count())):
        props = torch.cuda.get_device_properties(index)
        gpu.append({"index": index, "name": props.name, "total_memory": props.total_memory,
                    "compute_capability": [props.major, props.minor]})
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": version("triton"),
        "fla": version("flash-linear-attention"),
        "transformers": version("transformers"),
        "gpu": gpu,
        "nvidia_smi": command_output("nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_status_short": command_output("git", "-C", str(REPO), "status", "--short"),
    }
    write_new(CONFIGS / "environment_manifest.json", environment)
    write_new(ROOT / "preregistration.json", {
        "task": "QWEN_BACKWARD_REPEATABILITY_AND_TOPOLOGY_CLOSURE_V1",
        "parent_task": V2.TASK,
        "reference_case_sha256": sha_file(CONFIGS / "frozen_backward_reference.json"),
        "environment_manifest_sha256": sha_file(CONFIGS / "environment_manifest.json"),
        "stage_a": "5_fresh_process_single_GPU_forward_backward_only",
        "stage_b": "source_audit_and_deterministic_diagnostic",
        "exact_path": "5_fresh_process_single_then_dual_exact_parity_if_runtime_preserves_forward",
        "numerical_path": "preregister_max_of_45_single_pairs_before_new_dual_magnitude_comparison",
        "epsilon": 1e-12,
        "no_AIME_or_formal_training": True,
    })
    print(json.dumps({"PREREGISTRATION": "PASS", "reference_case_sha256": sha_file(CONFIGS / "frozen_backward_reference.json")}, indent=2))


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def apply_deterministic_diagnostic() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic diagnostic requires CUBLAS_WORKSPACE_CONFIG=:4096:8 at process start")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def gradient_stats(value: torch.Tensor) -> dict:
    cpu = value.detach().float().cpu()
    finite = torch.isfinite(cpu)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value),
        "norm": float(torch.linalg.vector_norm(cpu)),
        "max_abs": float(cpu.abs().max()), "mean": float(cpu.mean()),
        "std": float(cpu.std(unbiased=False)),
        "finite_count": int(finite.sum()), "nonfinite_count": int((~finite).sum()),
    }


def repeat(mode: str, variant: str, run_id: int, step: bool) -> None:
    stem = f"{mode}_{variant}_{run_id:02d}"
    result_file = ANALYSIS / f"{stem}.json"
    gradient_file = RAW / f"{stem}_gradients.npz"
    update_file = RAW / f"{stem}_update.npz"
    if result_file.exists() or gradient_file.exists() or update_file.exists():
        raise RuntimeError(f"refusing to overwrite run {stem}")
    reference = json.loads((CONFIGS / "frozen_backward_reference.json").read_text())
    if variant == "deterministic":
        apply_deterministic_diagnostic()
    random.seed(V1.TRAIN_SEED)
    torch.manual_seed(V1.TRAIN_SEED)
    if mode == "single":
        model, tokenizer, bank, device = V1.load_model_bank()
        patch_factory = lambda: V1.RecurrentExperimentPatch(model, bank, V1.hadamard(device))
    elif mode == "dual":
        model, tokenizer, bank, layer_devices = V2.load_dual_model_bank(16)
        device = model.get_input_embeddings().weight.device
        patch_factory = lambda: V2.DualRecurrentPatch(model, bank)
    else:
        raise ValueError(mode)
    row = V1.corpus_rows("TRAIN")[0]
    ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:192]
    positions = V1.sampled_positions(row, len(ids))
    if row["document_id"] != reference["source_document_id"] or ids != reference["input_token_ids"] or positions != reference["target_positions"]:
        raise RuntimeError("frozen input/teacher token mismatch")
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    trace_set = set(reference["trace_token_indices"])
    position_set = set(positions)
    records = {}
    losses = []
    boundary = []
    cache = None
    segments = []
    current_token = -1
    original_qdq = V1.CAYLEY.qwen_c128_ste
    with patch_factory() as patch:
        targets = V1.collect_teacher_targets(model, patch, ids, positions)
        teacher_hashes = {}
        for pos in sorted(targets):
            for part in ("prev", "out"):
                for layer_id, value in sorted(targets[pos][part].items()):
                    teacher_hashes[f"{pos}:{part}:{layer_id}"] = tensor_hash(value)

        def traced_qdq(value):
            quant = original_qdq(value)
            if current_token in trace_set:
                layer_id = int(patch.current_layer)
                key = f"{current_token}:{layer_id}"
                records.setdefault(key, {}).update({
                    "token_index": current_token, "layer_id": layer_id,
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
                    for layer_id in V1.GDN_LAYERS:
                        records.setdefault(f"{index}:{layer_id}", {})["consumed_prev_state_sha256"] = tensor_hash(V1.BASE.get_state(cache, layer_id))
                cache = V1.forward_student(model, patch, token_id, cache, capture)
                if capture:
                    state, functional, c5, c6 = patch.captured_losses()
                    segments.append(c5 / V1.TARGETS_PER_DOCUMENT)
                    losses.append({"token_index": index, "state": float(state.detach().cpu()),
                                   "functional": float(functional.detach().cpu()),
                                   "c5": float(c5.detach().cpu()), "c6": float(c6.detach().cpu())})
                    del state, functional, c5, c6
                if (index + 1) % 32 == 0 or index + 1 == len(ids):
                    if segments:
                        segment_loss = sum(segments)
                        segment_loss.backward()
                        segments.clear()
                        patch.clear_capture()
                        del segment_loss
                    if index == 31:
                        before = {str(layer): tensor_hash(V1.BASE.get_state(cache, layer)) for layer in V1.GDN_LAYERS}
                    V1.detach_cache(cache)
                    if index == 31:
                        after = {str(layer): tensor_hash(V1.BASE.get_state(cache, layer)) for layer in V1.GDN_LAYERS}
                        boundary.append({"token_index": index, "value_preserved": before == after})
            if len(losses) != 8:
                raise RuntimeError(f"expected 8 targets, found {len(losses)}")
            gradient_arrays = {}
            gradient_summary = {}
            for layer_id in V1.GDN_LAYERS:
                gradient = bank.layer(layer_id).theta.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"invalid gradient layer {layer_id}")
                gradient_arrays[str(layer_id)] = gradient.detach().cpu().numpy().copy()
                gradient_summary[str(layer_id)] = gradient_stats(gradient)
            np.savez_compressed(gradient_file, **gradient_arrays)
            global_gradient_norm = float(np.sqrt(sum(
                np.sum(value.astype(np.float64) ** 2) for value in gradient_arrays.values()
            )))
            update_hashes = None
            if step:
                torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
                optimizer.step()
                delta_arrays = {str(layer): bank.layer(layer).theta.detach().cpu().numpy().copy() for layer in V1.GDN_LAYERS}
                np.savez_compressed(update_file, **delta_arrays)
                update_hashes = {str(layer): V2.tensor_hash(bank.layer(layer).theta) for layer in V1.GDN_LAYERS}
        finally:
            V1.CAYLEY.qwen_c128_ste = original_qdq
    for index in trace_set:
        for layer in V1.GDN_LAYERS:
            if f"{index}:{layer}" not in records:
                raise RuntimeError(f"missing forward trace {index}:{layer}")
    writeback = all(records[f"31:{layer}"]["post_qdq_state_sha256"] ==
                    records[f"32:{layer}"]["consumed_prev_state_sha256"] for layer in V1.GDN_LAYERS)
    result = {
        "stem": stem, "mode": mode, "variant": variant, "run_id": run_id, "optimizer_step_performed": step,
        "reference_case_sha256": sha_file(CONFIGS / "frozen_backward_reference.json"),
        "input_token_ids_sha256": hash_json(ids), "teacher_token_ids_sha256": hash_json(ids),
        "target_positions_sha256": hash_json(positions), "teacher_target_tensor_hashes": teacher_hashes,
        "records": records, "loss_components": losses,
        "total_training_loss": sum(value["c5"] for value in losses) / 8,
        "global_gradient_norm": global_gradient_norm,
        "gradient_summary": gradient_summary,
        "gradient_artifact": gradient_file.name, "gradient_artifact_sha256": sha_file(gradient_file),
        "update_artifact": update_file.name if step else None,
        "update_artifact_sha256": sha_file(update_file) if step else None,
        "update_theta_sha256": update_hashes,
        "REAL_RECURRENT_WRITEBACK_GATE": "PASS" if writeback else "FAIL",
        "BPTT_BOUNDARY_GATE": "PASS" if writeback and all(value["value_preserved"] for value in boundary) else "FAIL",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    write_new(result_file, result)
    print(json.dumps({"stem": stem, "loss": result["total_training_loss"],
                      "gradient_norm": global_gradient_norm, "step": step,
                      "writeback": result["REAL_RECURRENT_WRITEBACK_GATE"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "repeat"), required=True)
    parser.add_argument("--mode", choices=("single", "dual"))
    parser.add_argument("--variant", choices=("baseline", "deterministic"))
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--step", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if args.mode is None or args.variant is None or args.run_id is None:
        parser.error("repeat requires --mode, --variant, --run-id")
    try:
        repeat(args.mode, args.variant, args.run_id, args.step)
    except Exception as exc:
        error_path = ANALYSIS / f"{args.mode}_{args.variant}_{args.run_id:02d}.error.json"
        if not error_path.exists():
            write_new(error_path, {"exception_type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc(), "deterministic": args.variant == "deterministic"})
        raise


if __name__ == "__main__":
    main()
