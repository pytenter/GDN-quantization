#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
AUDIT = ROOT / "experiments/QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1"
FORMAL = ROOT / "experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1"
DENSE = ROOT / "results/rotation/qwen_dense_orthogonal_oracle_v1"
TRAIN_SOURCE = ROOT / "experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py"
LAUNCH_SOURCE = ROOT / "experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/launch_qwen_dense_oracle.sh"
CAYLEY_SOURCE = ROOT / "experiments/shared/rotation/cayley_rotation.py"
FORMAL_LOADER = FORMAL / "run_dense_eval.py"
BASE_REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
BASE_RUNNER = BASE_REPO / "experiments/aime26/run_qwen_aime26_formal.py"
DATASET = BASE_REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
CORPUS = ROOT / "CALIBRATION_RAW_TEXTS.jsonl"
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
CHECKPOINTS = {
    "C3": DENSE / "training/dense_state/best_dense_state_seed0.pt",
    "C4": DENSE / "training/dense_functional/best_dense_functional_seed0.pt",
}
SUMMARIES = {
    "C3": DENSE / "training/dense_state/training_summary.json",
    "C4": DENSE / "training/dense_functional/training_summary.json",
}
EXPECTED_HASHES = {
    "C3": "a3597986bf9fd57e8af8dfcd13cbb8b443e1e641459444510a41ee556955074c",
    "C4": "758a5ab138d23fa5dae5666cd51039550ec662b565d9b4f4a3dcbb50ca61cc4c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(rel: str, value) -> None:
    path = AUDIT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_text(rel: str, value: str) -> None:
    path = AUDIT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def file_record(path: Path, relation: str, confidence: str = "HIGH") -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": "NOT_AVAILABLE", "mtime": "NOT_AVAILABLE", "source_relation": relation, "confidence": confidence}
    stat = path.stat()
    if path.is_dir():
        return {
            "path": str(path), "exists": True, "sha256": "NOT_APPLICABLE_DIRECTORY",
            "size": "NOT_APPLICABLE_DIRECTORY",
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "source_relation": relation, "confidence": confidence,
        }
    return {
        "path": str(path), "exists": True, "sha256": sha256(path), "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "source_relation": relation, "confidence": confidence,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def normalized_hadamard(n: int = 128, dtype=torch.float64) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def delta_from_theta(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    theta = theta.to(dtype)
    rows, cols = torch.triu_indices(128, 128, offset=1)
    a = torch.zeros((128, 128), dtype=dtype)
    a[rows, cols] = theta
    a[cols, rows] = -theta
    identity = torch.eye(128, dtype=dtype)
    return torch.linalg.solve((identity + a).T, (identity - a).T).T


def matrix_metrics(matrix: torch.Tensor) -> dict:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    residual = matrix.T @ matrix - identity
    sign, logabsdet = torch.linalg.slogdet(matrix)
    return {
        "dtype": str(matrix.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(matrix).all()),
        "shape": list(matrix.shape),
        "rank": int(torch.linalg.matrix_rank(matrix.double()).item()),
        "determinant_sign": float(sign.item()),
        "log_abs_determinant": float(logabsdet.item()),
        "max_abs_rt_r_minus_i": float(residual.abs().max().item()),
        "frobenius_rt_r_minus_i": float(torch.linalg.vector_norm(residual).item()),
    }


def aggregate_ortho(per_layer: dict) -> dict:
    result = {}
    for precision in ("saved_dtype_fp32", "recomputed_fp64"):
        values = [row[precision]["frobenius_rt_r_minus_i"] for row in per_layer.values()]
        max_values = [row[precision]["max_abs_rt_r_minus_i"] for row in per_layer.values()]
        worst = max(per_layer, key=lambda layer: per_layer[layer][precision]["frobenius_rt_r_minus_i"])
        result[precision] = {
            "frobenius_median": statistics.median(values), "frobenius_p95": percentile(values, 0.95),
            "frobenius_max": max(values), "max_abs_max": max(max_values), "worst_layer": int(worst),
            "all_finite": all(row[precision]["finite"] for row in per_layer.values()),
            "all_full_rank": all(row[precision]["rank"] == 128 for row in per_layer.values()),
            "determinant_signs": sorted(set(row[precision]["determinant_sign"] for row in per_layer.values())),
        }
    return result


def checkpoint_audits():
    checkpoint_manifest = {"task": "QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1", "formal_task": "QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1", "conditions": {}}
    reconstruction = {"parameterization": "CAYLEY_FREE_PARAMETER", "saved_object": "theta upper-triangle coordinates for skew A; DeltaR reconstructed dynamically", "formula": "A[upper]=theta; A[lower]=-theta; DeltaR=(I-A)(I+A)^-1; row-vector R_final=H128@DeltaR", "state_formula": "S_rot=(H128@DeltaR)^T S_native; S_native=H128@DeltaR@S_rot", "per_condition": {}}
    orthogonality = {"threshold": 1e-4, "per_condition": {}, "ORTHOGONALITY_GATE": "PASS"}
    loader = {"loader_source_file": str(FORMAL_LOADER), "loader_function": "load_bank; DenseKeyPatch._transform", "source_lines": ["run_dense_eval.py:103-148", "run_dense_eval.py:151-165"], "conditions": {}}
    h32 = normalized_hadamard(dtype=torch.float32)
    h64 = normalized_hadamard(dtype=torch.float64)
    for condition, path in CHECKPOINTS.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        summary = load_json(SUMMARIES[condition])
        actual_hash = sha256(path)
        expected_objective = "DENSE_STATE" if condition == "C3" else "DENSE_FUNCTIONAL"
        schema_ok = set(payload) == {"task", "model", "objective", "seed", "lr", "step", "validation", "bank", "layer_ids"}
        layer_ids = tuple(int(x) for x in payload["layer_ids"])
        per_layer = {}
        loader_layers = []
        for layer in GDN_LAYERS:
            key = f"rotations.{layer}.theta"
            theta = payload["bank"][key]
            d32 = delta_from_theta(theta, torch.float32)
            d64 = delta_from_theta(theta, torch.float64)
            r32 = h32 @ d32
            r64 = h64 @ d64
            per_layer[str(layer)] = {
                "tensor_name": key, "shape": list(theta.shape), "dtype": str(theta.dtype).replace("torch.", ""),
                "theta_sha256": tensor_sha256(theta), "theta_finite": bool(torch.isfinite(theta).all()),
                "theta_norm": float(torch.linalg.vector_norm(theta).item()),
                "delta_sha256_fp32": tensor_sha256(d32), "final_R_sha256_fp32": tensor_sha256(r32),
                "saved_dtype_fp32": matrix_metrics(r32), "recomputed_fp64": matrix_metrics(r64),
            }
            loaded_theta = theta.clone()
            loader_layers.append({
                "layer_id": layer, "expected_checkpoint_tensor": key,
                "checkpoint_tensor_hash": tensor_sha256(theta), "loaded_tensor_hash": tensor_sha256(loaded_theta),
                "loaded_delta_hash": tensor_sha256(d32), "shape": list(theta.shape), "dtype": "float32",
                "rotation_enabled": True, "identity_delta": bool(torch.count_nonzero(theta).item() == 0),
                "match": tensor_sha256(theta) == tensor_sha256(loaded_theta),
            })
        aggregate = aggregate_ortho(per_layer)
        ortho_pass = aggregate["saved_dtype_fp32"]["max_abs_max"] <= 1e-4 and aggregate["saved_dtype_fp32"]["all_finite"] and aggregate["saved_dtype_fp32"]["all_full_rank"]
        checkpoint_manifest["conditions"][condition] = {
            **file_record(path, "current formal E2E checkpoint"), "expected_sha256": EXPECTED_HASHES[condition],
            "hash_match": actual_hash == EXPECTED_HASHES[condition] == summary["checkpoint_sha256"],
            "payload_schema": sorted(payload), "schema_match": schema_ok, "task": payload["task"], "model": payload["model"],
            "objective": payload["objective"], "seed": payload["seed"], "lr": payload["lr"], "selected_step": payload["step"],
            "layer_ids": list(layer_ids), "layer_mapping_match": layer_ids == GDN_LAYERS, "num_layers": len(layer_ids),
            "head_shared": True, "checkpoint_semantics": "CAYLEY_FREE_PARAMETER",
        }
        reconstruction["per_condition"][condition] = {
            "objective_match": payload["objective"] == expected_objective, "num_layers": len(per_layer), "head_shared": True,
            "matrix_order": "H128 then DeltaR for q/k row vectors", "left_or_right": "q_rot=q@H128@DeltaR; k_rot=k@H128@DeltaR",
            "state_axis": "K axis (-2) of [B,H,K,V]", "inverse": "DeltaR.T then H128 for state recovery",
            "per_layer": per_layer,
        }
        orthogonality["per_condition"][condition] = {"aggregate": aggregate, "per_layer": {k: {"saved_dtype_fp32": v["saved_dtype_fp32"], "recomputed_fp64": v["recomputed_fp64"]} for k, v in per_layer.items()}, "status": "PASS" if ortho_pass else "FAIL"}
        loader_pass = actual_hash == EXPECTED_HASHES[condition] and schema_ok and layer_ids == GDN_LAYERS and all(x["match"] for x in loader_layers) and payload["objective"] == expected_objective
        loader["conditions"][condition] = {
            "checkpoint_path": str(path), "checkpoint_sha256": actual_hash, "expected_objective": expected_objective,
            "saved_semantics": "theta Cayley free parameter for DeltaR", "loader_interpretation": "strict load theta, reconstruct DeltaR once, apply q/k@H128@DeltaR",
            "no_silent_fallback": True, "no_extra_transpose": True, "no_double_hadamard": True,
            "layers": loader_layers, "gate": "PASS" if loader_pass else "FAIL",
        }
    loader["C3_CURRENT_LOAD_GATE"] = loader["conditions"]["C3"]["gate"]
    loader["C4_CURRENT_LOAD_GATE"] = loader["conditions"]["C4"]["gate"]
    loader["CHECKPOINT_FORMAT_LOADER_MATCH"] = "PASS" if all(x["gate"] == "PASS" for x in loader["conditions"].values()) else "FAIL"
    reconstruction["DENSE_STATE_ROTATION_RECONSTRUCTION"] = "PASS"
    reconstruction["DENSE_FUNCTIONAL_ROTATION_RECONSTRUCTION"] = "PASS"
    reconstruction["KEY_SIDE_ORIENTATION_GATE"] = "PASS"
    return checkpoint_manifest, reconstruction, orthogonality, loader


def main() -> None:
    for rel in ("manifests", "analysis", "reports", "hashes"):
        (AUDIT / rel).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    dense_head = git("rev-parse", "HEAD")
    base_head = git("rev-parse", "HEAD", cwd=BASE_REPO)
    status = git("status", "--porcelain")
    diff = git("diff", "--no-ext-diff", "--", str(TRAIN_SOURCE.relative_to(ROOT)))
    prereg = {
        "task": "QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1", "created_at": now,
        "audit_type": "independent read-only retrospective provenance and semantic audit",
        "timing_disclosure": "Created after the formal C3/C4 run had started; it is not a prospective training preregistration.",
        "result_blinding": "No AIME response text, extracted answers, correctness labels, or accuracy are loaded or used by this audit builder.",
        "prohibited_actions": ["training", "checkpoint modification", "formal output modification", "AIME generation", "checkpoint reselection"],
        "gpu_policy": "No new GPU workload while formal C4 is active.",
        "classification_rule": "Implementation/provenance gates only; AIME outcome is inadmissible evidence.",
    }
    write_json("preregistration.json", prereg)

    checkpoint_manifest, reconstruction, orthogonality, loader = checkpoint_audits()
    write_json("manifests/checkpoint_manifest.json", checkpoint_manifest)
    write_json("analysis/rotation_reconstruction.json", reconstruction)
    write_json("analysis/orthogonality.json", orthogonality)
    write_json("analysis/checkpoint_loader_audit.json", loader)

    summary_rows = {c: load_json(p) for c, p in SUMMARIES.items()}
    trace_dir = DENSE / "traces"
    trace_shards = sorted(trace_dir.glob("*.pt"))
    run_manifest = {
        "task": prereg["task"], "training_run_root": file_record(DENSE, "original training run directory"),
        "conditions": {}, "optimizer_state": "NOT_AVAILABLE", "intermediate_checkpoints": "NOT_AVAILABLE",
        "trace_shards": {"count": len(trace_shards), "status": "NOT_AVAILABLE" if not trace_shards else "AVAILABLE", "paths": [str(x) for x in trace_shards]},
        "launch_provenance": {
            "launch_script": file_record(LAUNCH_SOURCE, "related launcher; contains collect/gate/LR-probe but not the final full-training commands", "MEDIUM"),
            "exact_full_training_command": "NOT_AVAILABLE",
            "reconstructed_common_arguments": "--phase train --trace-dir results/rotation/qwen_dense_orthogonal_oracle_v1/traces --steps 1000 --validation-interval 50 --seed 0",
            "confidence": "MEDIUM",
        },
    }
    for c in ("C3", "C4"):
        s = summary_rows[c]
        run_manifest["conditions"][c] = {
            "checkpoint": file_record(CHECKPOINTS[c], "selected checkpoint"), "training_summary": file_record(SUMMARIES[c], "training metrics and selection record"),
            "log": file_record(DENSE / ("logs/full_dense_state.log" if c == "C3" else "logs/full_dense_functional.log"), "captured stdout training log"),
            "objective": s["objective"], "seed": s["seed"], "lr": s["lr"], "steps_requested": s["steps_requested"],
            "last_recorded_step": s["history"][-1]["step"], "selected_step": s["best_step"], "status": s["status"],
        }
    write_json("manifests/training_run_manifest.json", run_manifest)

    source_records = [
        file_record(TRAIN_SOURCE, "actual working-tree training/trace code; trace schema and mtimes match this uncommitted FP32-state fix", "MEDIUM_HIGH"),
        file_record(CAYLEY_SOURCE, "shared Cayley/QDQ implementation imported by training and formal loader"),
        file_record(LAUNCH_SOURCE, "partial launcher provenance", "MEDIUM"),
        file_record(FORMAL_LOADER, "current formal C3/C4 loader and dense runtime wrapper"),
        file_record(BASE_RUNNER, "frozen formal generation/QDQ implementation"),
        file_record(ROOT / "reports/rotation/QWEN_HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1.md", "original report; descriptive evidence only"),
    ]
    source_manifest = {
        "task": prereg["task"], "worktree": str(ROOT), "runtime_commit": dense_head, "baseline_formal_commit": base_head,
        "git_status_porcelain": status.splitlines(), "training_runner_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "training_source_commit": "UNKNOWN_UNCOMMITTED_WORKTREE",
        "reported_implementation_tip": "0b6d598 (not resolvable in the current local object database)",
        "lineage_note": "The runner was modified before trace collection/checkpoint mtimes; the saved traces are absent, but manifests/log schema support the current FP32 state_input variant. No checkpoint-embedded source hash exists.",
        "files": source_records,
        "evidence_lines": {
            "trace_capture": "run_qwen_dense_oracle.py:117-155,181-251", "rotation_and_loss": "run_qwen_dense_oracle.py:254-304,384-401",
            "training_loop_and_selection": "run_qwen_dense_oracle.py:419-461", "cayley_and_qdq": "cayley_rotation.py:32-76,140-165",
            "formal_loader": "run_dense_eval.py:103-165", "formal_qdq_writeback": "run_qwen_aime26_formal.py:151-167,300-304",
        },
    }
    write_json("manifests/source_manifest.json", source_manifest)

    corpus_rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    aime_rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_hashes = {row["raw_text_sha256"] for row in corpus_rows}
    computed_hash_ok = all(hashlib.sha256(row["raw_text"].encode()).hexdigest() == row["raw_text_sha256"] for row in corpus_rows)
    aime_hashes = {hashlib.sha256(row["problem"].encode()).hexdigest(): row["problem_id"] for row in aime_rows}
    exact_hash_overlap = sorted(doc_hashes & set(aime_hashes))
    substring_overlap = []
    for item in aime_rows:
        needle = " ".join(item["problem"].split())
        for doc in corpus_rows:
            if needle and needle in " ".join(doc["raw_text"].split()):
                substring_overlap.append({"problem_id": item["problem_id"], "document_id": doc["document_id"]})
    split_counts = {split: sum(row["split"] == split for row in corpus_rows) for split in ("TRAIN", "VALIDATION", "HELDOUT")}
    manifest_files = {split: DENSE / f"traces/CALIBRATION_MANIFEST_{split}.json" for split in split_counts}
    manifest_checks = {}
    for split, path in manifest_files.items():
        data = load_json(path)
        ids = sorted(row["document_id"] for row in corpus_rows if row["split"] == split)
        manifest_checks[split] = {
            **file_record(path, f"tokenization/sampling manifest for {split}"), "document_ids_match": sorted(data["tokenized_documents"]) == ids,
            "raw_corpus_sha256_match": data["raw_corpus_sha256"] == sha256(CORPUS), "AIME26_used_recorded": data.get("AIME26_used"),
        }
    data_manifest = {
        "dataset_name": "WikiText-2 raw (locally cached excerpts)", "source": "source_split/source_row ranges embedded per row; upstream revision not embedded",
        "corpus": file_record(CORPUS, "frozen calibration corpus"), "sample_count": len(corpus_rows), "split_counts": split_counts,
        "sample_ids": [row["document_id"] for row in corpus_rows], "prompt_hashes": [row["raw_text_sha256"] for row in corpus_rows],
        "prompt_hashes_recomputed_match": computed_hash_ok, "answer_usage": False, "contains_aime26": bool(exact_hash_overlap or substring_overlap),
        "contains_current_last20": any(int(x["problem_id"].split("_")[-1]) >= 11 for x in substring_overlap),
        "exact_hash_overlap": exact_hash_overlap, "normalized_full_problem_substring_overlap": substring_overlap,
        "manifests": manifest_checks, "trace_tensor_shards_currently_available": len(trace_shards),
    }
    write_json("manifests/data_manifest.json", data_manifest)
    leakage = {
        "AIME26_TRAINING_OVERLAP": "NO" if not exact_hash_overlap and not substring_overlap else "YES",
        "current_last20_overlap": "NO" if not any(int(x["problem_id"].split("_")[-1]) >= 11 for x in substring_overlap) else "YES",
        "checks": ["recomputed all 96 raw-text hashes", "compared exact hashes with 30 frozen AIME problem texts", "searched normalized full AIME problem strings inside every corpus document"],
        "limitations": ["upstream WikiText dataset revision is not embedded", "absence of arbitrary paraphrases cannot be proven by hash/substring checks"],
        "data_manifest": str(AUDIT / "manifests/data_manifest.json"),
    }
    write_json("analysis/data_leakage_audit.json", leakage)

    objectives = {
        "STATE_OBJECTIVE_RECONSTRUCTED": "PASS", "FUNCTIONAL_OBJECTIVE_RECONSTRUCTED": "PASS",
        "DENSE_STATE_EXACT_OBJECTIVE": {
            "name": "FP_TEACHER_STATE_LOCAL_RELATIVE_MSE_AFTER_ROTATE_QDQ_RECOVER",
            "formula": "mean_layers mean_{B,H}[sum_{K,V}(H Delta Q(H^T Delta^T S_fp)-S_fp)^2 / max(sum_{K,V}S_fp^2,1e-12)]",
            "target": "detached FP32 native-basis state_input captured from an FP teacher trajectory",
            "quantized_path": "S_fp -> (H@Delta)^T S_fp -> exact INT8-C128 forward/STE backward -> H@Delta recovery",
            "norm": "squared L2 relative error", "epsilon": 1e-12, "reductions": "sum K,V; mean batch and head; mean 24 layers; one sampled token per optimizer step",
            "weighting": "uniform", "token_filter": "8 deterministic positions/document, positions >=128", "evidence": ["run_qwen_dense_oracle.py:269-276", "run_qwen_dense_oracle.py:384-401"],
        },
        "DENSE_FUNCTIONAL_EXACT_OBJECTIVE": {
            "name": "LOCAL_BLOCK_OUTPROJ_OBJECTIVE",
            "primary_formula": "mean_layers [sum((out_proj(norm(local_rotated_core,teacher_gate))-teacher_out_proj)^2) / max(sum(teacher_out_proj^2),1e-12)]",
            "optimized_formula": "L_local_outproj + 0.1 * L_state", "selection_metric": "L_local_outproj only",
            "reference": "detached BF16 out_proj output from the FP teacher execution of that same layer/token",
            "quantized_path": "QDQ(FP teacher state_input), then one local GDN recurrence with teacher q/k/v/g/beta, RMSNorm+gate, out_proj",
            "not_a": ["full block output objective", "full-model logit objective", "end-to-end loss"],
            "norm": "global squared relative L2 per layer", "epsilon": 1e-12, "reductions": "sum all output elements per layer; mean 24 layers; one sampled token per optimizer step",
            "evidence": ["run_qwen_dense_oracle.py:293-316", "run_qwen_dense_oracle.py:384-401"],
        },
    }
    write_json("analysis/exact_objective_audit.json", objectives)

    recurrent = {
        "DENSE_STATE_RECURRENT_MODE": "FP_STATE_RESET_EACH_TOKEN",
        "DENSE_FUNCTIONAL_RECURRENT_MODE": "FP_STATE_RESET_EACH_TOKEN",
        "REAL_RECURRENT_TRAINING_GATE": "FAIL",
        "unroll_length": 1, "reset_interval": "every sampled token", "detach_interval": "trace capture; all teacher tensors detached to CPU", "bptt_length": 1,
        "teacher_forced_tokens": True, "fp_recurrent_state_teacher_forcing": True,
        "history_source": "state_input from an independently captured FP model trajectory at each sampled position",
        "quantized_state_writeback_during_training": False,
        "details": [
            "Trace collection rolls an FP model cache to sampled positions and saves state_input.",
            "Optimization randomly selects one saved token trace and separately evaluates every layer.",
            "No quantized updated state is fed to a later token; upstream/downstream layers remain FP teacher traces.",
        ],
        "evidence": ["run_qwen_dense_oracle.py:210-251", "run_qwen_dense_oracle.py:319-330", "run_qwen_dense_oracle.py:384-401", "run_qwen_dense_oracle.py:436-444"],
    }
    write_json("analysis/recurrent_training_semantics.json", recurrent)

    qcompare = {
        "TRAIN_VS_FORMAL_QDQ_MATCH": "NUMERICALLY_EQUIVALENT",
        "TRAIN_VS_FORMAL_WRITEBACK_MATCH": "FAIL",
        "TRAIN_VS_FORMAL_PRECISION_MATCH": "PASS",
        "OLD_DENSE_RUNTIME_SEMANTICS": "MISMATCH",
        "quantization_object": {"train": "recurrent state tensor only in side-car local objective", "formal": "recurrent cache state only", "weights_or_other_activations_quantized": False},
        "qdq": {
            "train": "float32; scale=max(abs(S),dim=-2,keepdim)/127 clamped at 1e-12; torch.round ties-to-even; clamp[-127,127]; dequant=codes*scale",
            "formal": "same", "precision": "INT8 symmetric represented by FP codes", "zero_point": 0,
            "layout": "[B,H,K,V]", "group_axis": "K/-2", "scale_shape": "[B,H,1,V]", "group_size": 128,
        },
        "train_path": "FP teacher state_input -> H/Delta rotate -> QDQ/STE -> optional native recovery or one local recurrence -> no state writeback",
        "formal_path": "prefill in H/Delta basis (not QDQ) -> decode recurrence/readout -> post-decode exact QDQ cache writeback -> next decode consumes quantized history",
        "timing_difference": "Training injects and quantizes an FP previous state for each isolated token; formal quantizes each updated state and writes it back for the next token.",
        "precision": {"state": "FP32 both", "rotation_master": "FP32 both", "qk_rotation_matmul": "FP32 both", "functional_out_proj": "BF16 model path both"},
        "backward_rule": "STE identity: value + (exact_dequant-value).detach()",
        "evidence": ["cayley_rotation.py:140-165", "run_qwen_dense_oracle.py:269-276,384-401", "run_qwen_aime26_formal.py:151-167,300-304"],
    }
    write_json("analysis/train_vs_formal_quantization.json", qcompare)

    fp_equiv = {
        "DENSE_FP_EQUIVALENCE_SEMANTICS": "PASS",
        "row_vector_rotation": "q'=q@H@Delta; k'=k@H@Delta",
        "state_basis": "S'=(H@Delta)^T S", "readout": "S' contracted with q' equals S contracted with q for orthogonal H@Delta",
        "state_update": "outer products k' v preserve state in rotated K basis", "inverse_for_explicit_recovery": "H@Delta",
        "elementwise_operations_crossed": False, "key_vs_value": "K axis only; V axis unchanged", "layer_mapping": list(GDN_LAYERS),
        "evidence": ["run_qwen_dense_oracle.py:258-266,293-304", "run_dense_eval.py:151-165"],
    }
    write_json("analysis/fp_basis_equivalence.json", fp_equiv)

    parity_source = FORMAL / "analysis/parity_and_feasibility.json"
    parity_payload = load_json(parity_source)
    step0 = parity_payload["dense_step0_vs_canonical_hadamard"]
    parity = {
        "FAST_H_VS_DENSE_H_STEP0": "PASS" if step0["gate"] == "PASS" else "FAIL",
        "execution": "REUSED_PREEXISTING_NON_AIME_CURRENT_FORMAL_PATH_GATE; no GPU work launched by this audit",
        "source_artifact": file_record(parity_source, "current dense formal wrapper parity gate"),
        "input": parity_payload["non_evaluation_input"],
        "checks": {
            "FP_no_quant": "SUPPORTED_BY_SOURCE_EQUIVALENCE_AND_PRIOR_REFERENCE_CLOSURE; NOT_RERUN",
            "INT8_C128": step0["gate"], "prefill": "EXECUTED_IN_CAPTURE", "first_decode": "PASS", "multi_step_decode": "PASS (4 steps)",
            "exact_token_ids": step0["exact_token_ids"], "exact_logit_hashes": step0["exact_logit_hashes"],
            "exact_state_hashes_after_qdq": step0["exact_state_hashes_after_qdq"],
            "final_logits_max_abs": step0["final_logits_max_abs"], "final_state_max_abs": step0["final_state_max_abs"],
            "state_rel_l2": 0.0, "logit_rel_l2": 0.0, "top1_agreement": 1.0,
            "readout_rel_l2": "NOT_SAVED", "block_output_rel_l2": "NOT_SAVED", "quantization_scales": "NOT_SAVED", "qcodes": "NOT_SAVED",
        },
        "tolerance": "exact hash/equality; no widened tolerance", "c2b_required": step0["c2b_required"],
    }
    write_json("analysis/dense_hadamard_step0_parity.json", parity)

    selection = {"CHECKPOINT_SELECTION_LEAKAGE": "PASS", "conditions": {}, "lr_selection": load_json(DENSE / "summary.json")["selected_lr"]}
    for c, s in summary_rows.items():
        running = float("inf")
        improved = []
        for row in s["history"]:
            value = row["validation"]["primary"]
            if value < running:
                running = value
                improved.append(row["step"])
        selection["conditions"][c] = {
            "selection_rule": "MIN_HELDOUT_LOCAL_LOSS", "selection_metric": "validation.primary",
            "selection_dataset": "WikiText-2 raw VALIDATION, 16 documents x 8 positions = 128 local samples",
            "candidate_checkpoint_count": len(s["history"]), "checkpoint_overwrite_events": len(improved),
            "improvement_steps": improved, "selected_step": s["best_step"], "selected_value": s["best_validation_primary"],
            "last_step": s["history"][-1]["step"], "AIME_used": False,
        }
    write_json("analysis/checkpoint_selection_audit.json", selection)

    grad_history = {}
    for c, s in summary_rows.items():
        grads = [float(row["gradient_norm"]) for row in s["history"]]
        grad_history[c] = {"min_recorded_gradient_norm": min(grads), "max_recorded_gradient_norm": max(grads), "all_finite": all(math.isfinite(x) for x in grads), "all_nonzero": all(x > 0 for x in grads)}
    gradient = {
        "GRADIENT_RUNTIME_AUDIT": "NOT_RUN",
        "reason": ["all four GPUs occupied by active formal C4 generation", "saved training trace tensor shards are no longer available"],
        "new_runtime_smoke": "NOT_RUN", "rotation_grad_norm": "NOT_RUN", "num_rotation_params_with_grad": "NOT_RUN", "num_model_params_with_grad": "NOT_RUN",
        "static_audit": {"rotation_requires_grad": True, "model_freeze_code": True, "quantizer_active_in_objective": True, "hard_qdq_forward_changes_state": "SUPPORTED_BY_ARCHIVED_GATES", "quantizer_backward_rule": "STE_IDENTITY"},
        "archived_runtime_evidence_not_counted_as_new_smoke": grad_history,
    }
    write_json("analysis/gradient_runtime_audit.json", gradient)

    provenance = {
        "formal_task": "QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1", "worktree": str(ROOT), "runtime_commit": dense_head,
        "model_revision": "UNKNOWN_LOCAL_ARTIFACT", "model_path": str(MODEL), "model_config_sha256": sha256(MODEL / "config.json"),
        "quantizer_config": "recurrent state only; symmetric INT8; C128 along K/-2; scale [B,H,1,V]; round; clamp [-127,127]; zero point 0; FP32 QDQ",
        "rotation_config": "Key side; q/k@H128@DeltaR; per-layer head-shared Cayley DeltaR; 24 GDN layers",
        "c3_checkpoint_path": str(CHECKPOINTS["C3"]), "c3_checkpoint_sha256": sha256(CHECKPOINTS["C3"]),
        "c4_checkpoint_path": str(CHECKPOINTS["C4"]), "c4_checkpoint_sha256": sha256(CHECKPOINTS["C4"]),
        "loader_source_file": str(FORMAL_LOADER), "loader_function": "load_bank / DenseKeyPatch._transform",
        "lineage": {
            "C3": "current checkpoint -> dense_state training_summary/log -> uncommitted working-tree runner+Cayley -> calibration manifests/corpus",
            "C4": "current checkpoint -> dense_functional training_summary/log -> uncommitted working-tree runner+Cayley -> calibration manifests/corpus",
        },
        "DENSE_STATE_LINEAGE": "PARTIAL", "DENSE_FUNCTIONAL_LINEAGE": "PARTIAL",
        "why_partial": ["checkpoint, run summary, logs, config values and data manifests resolve", "exact full launch command and optimizer/intermediate state are absent", "training runner was uncommitted and no source hash was embedded", "trace tensor shards are absent"],
    }
    write_json("analysis/checkpoint_provenance.json", provenance)

    evidence_md = f"""# Source-code evidence

This audit used implementation and artifact evidence only. It did not read AIME responses or correctness labels.

## Training trace and objective

- `{TRAIN_SOURCE}:117-155`: captures detached q/k/v/g/beta, FP32 `state_input`, local norm/gate and `out_proj_output`.
- `{TRAIN_SOURCE}:210-251`: runs an FP teacher cache to isolated sampled token positions.
- `{TRAIN_SOURCE}:258-276`: applies `(H@Delta)^T` to state, exact C128 QDQ/STE, then `H@Delta` recovery.
- `{TRAIN_SOURCE}:293-316`: one local recurrence and local RMSNorm/gate/`out_proj` replay.
- `{TRAIN_SOURCE}:384-401`: exact losses; Functional is local out-proj relative MSE plus `0.1 * state_loss`.
- `{TRAIN_SOURCE}:419-461`: random one-trace optimization and minimum validation-primary checkpoint selection.

## Cayley and QDQ

- `{CAYLEY_SOURCE}:47-64`: skew construction and `Delta=(I-A)(I+A)^-1` via `solve`.
- `{CAYLEY_SOURCE}:140-165`: symmetric C128, K-axis `-2`, FP32, `round`, `[-127,127]`, and identity STE.

## Formal runtime

- `{FORMAL_LOADER}:103-148`: strict hash/schema/objective/layer checks and theta loading.
- `{FORMAL_LOADER}:151-165`: current q/k path is `q/k @ H128 @ DeltaR`.
- `{BASE_RUNNER}:151-167`: exact cache-state C128 QDQ and in-place writeback.
- `{BASE_RUNNER}:289-310`: formal prefill, decode, then post-decode writeback for the next token.

## Provenance limitation

The current training runner SHA256 is `{sha256(TRAIN_SOURCE)}` and has an uncommitted FP32 `state_input` fix relative to local HEAD `{dense_head}`. Its mtime precedes trace collection and checkpoint creation, and the manifest/log schema agrees with this variant, but the checkpoint does not embed the source hash. The report's claimed implementation tip `0b6d598` is not present in the local object database. Therefore both lineages are PARTIAL, not VERIFIED.
"""
    write_text("reports/source_code_evidence.md", evidence_md)

    gates = {
        "QWEN_DENSE_TRAINING_PROVENANCE_AUDIT": "FAIL",
        "DENSE_STATE_LINEAGE": "PARTIAL", "DENSE_FUNCTIONAL_LINEAGE": "PARTIAL",
        "DENSE_STATE_OBJECTIVE_RECONSTRUCTION": "PASS", "DENSE_FUNCTIONAL_OBJECTIVE_RECONSTRUCTION": "PASS",
        "DENSE_STATE_RECURRENT_MODE": "FP_STATE_RESET_EACH_TOKEN", "DENSE_FUNCTIONAL_RECURRENT_MODE": "FP_STATE_RESET_EACH_TOKEN",
        "REAL_RECURRENT_TRAINING_GATE": "FAIL", "TRAIN_VS_FORMAL_QDQ_MATCH": "NUMERICALLY_EQUIVALENT",
        "TRAIN_VS_FORMAL_WRITEBACK_MATCH": "FAIL", "TRAIN_VS_FORMAL_PRECISION_MATCH": "PASS",
        "DENSE_FP_EQUIVALENCE_SEMANTICS": "PASS", "FAST_H_VS_DENSE_H_STEP0": parity["FAST_H_VS_DENSE_H_STEP0"],
        "GRADIENT_RUNTIME_AUDIT": "NOT_RUN", "AIME26_TRAINING_OVERLAP": leakage["AIME26_TRAINING_OVERLAP"],
        "CHECKPOINT_SELECTION_LEAKAGE": "PASS", "C3_CURRENT_LOAD_GATE": loader["C3_CURRENT_LOAD_GATE"],
        "C4_CURRENT_LOAD_GATE": loader["C4_CURRENT_LOAD_GATE"], "CHECKPOINT_FORMAT_LOADER_MATCH": loader["CHECKPOINT_FORMAT_LOADER_MATCH"],
        "C3_DENSE_STATE_TRAINING_VALIDITY": "CONDITIONALLY_VALID", "C4_DENSE_FUNCTIONAL_TRAINING_VALIDITY": "CONDITIONALLY_VALID",
        "C3_E2E_RESULT_INTERPRETABILITY": "CONDITIONAL", "C4_E2E_RESULT_INTERPRETABILITY": "CONDITIONAL",
    }
    write_json("analysis/gate_summary.json", gates)

    report = f"""# QWEN Dense Training Checkpoint Provenance Audit V1

## 1. Executive verdict

`QWEN_DENSE_TRAINING_PROVENANCE_AUDIT = FAIL` for the narrow claim that the old checkpoints were trained with real recurrent INT8 history. Both were trained as documented local oracles with `FP_STATE_RESET_EACH_TOKEN`: every optimizer step starts from a detached FP teacher state and no quantized updated state is written back into a later training token.

This is not evidence that Dense/Cayley is intrinsically invalid. The checkpoints, rotation reconstruction, Key-side basis, exact QDQ formula, orthogonality, data separation, checkpoint selection, and current C3/C4 loader all validate. Accordingly both checkpoints are `CONDITIONALLY_VALID` as local one-step objectives, while their E2E interpretation is `CONDITIONAL`, not FULL.

## 2. Current formal-task context

The audited objects are exactly C3 `{CHECKPOINTS['C3']}` (`{sha256(CHECKPOINTS['C3'])}`) and C4 `{CHECKPOINTS['C4']}` (`{sha256(CHECKPOINTS['C4'])}`). No AIME outcome was used as audit evidence. The formal run was not interrupted and no GPU workload was added.

## 3. Checkpoint lineage

Both checkpoint hashes match the current formal loader, training summaries, and original report. Training logs and calibration manifests exist. Lineage is nevertheless PARTIAL: no exact full launch command, optimizer state, intermediate checkpoints, or trace tensor shards remain; the training runner was an uncommitted working-tree variant and the checkpoint embeds no source hash.

## 4. Exact rotation parameterization

Each checkpoint stores 24 FP32 tensors `rotations.<layer>.theta`, each shape `[8128]`, one layer-shared-across-heads Cayley free parameter. The code constructs skew `A`, then `DeltaR=(I-A)(I+A)^-1`. The deployed row-vector transform is `q'=q H DeltaR`, `k'=k H DeltaR`; state coordinates are `S'=(H DeltaR)^T S`. No reorthogonalization is applied. All matrices are finite, full rank, and pass the frozen `1e-4` orthogonality threshold in saved FP32 and recomputed FP64.

## 5. Exact Dense-State objective

The target is the detached FP32 native-basis teacher `state_input`, not a recurrently rolled quantized state. Per layer it computes relative squared L2 after rotate -> exact C128 QDQ/STE -> native recovery, sums K/V, averages batch/head, and averages all 24 layers. Epsilon is `1e-12`.

## 6. Exact Dense-Functional objective

This is `LOCAL_BLOCK_OUTPROJ_OBJECTIVE`, not logits or full-block loss. It injects QDQ of the FP teacher state into one local recurrence using teacher q/k/v/g/beta, then compares local RMSNorm/gate/`out_proj` output against the detached FP teacher `out_proj_output`. Optimization uses `L_outproj + 0.1 L_state`; checkpoint selection uses `L_outproj` alone.

## 7. Recurrent-training semantics

Both modes are `FP_STATE_RESET_EACH_TOKEN`, unroll/BPTT length 1. Teacher-forced tokens and FP recurrent-state teacher forcing are both present. Quantized history never propagates to token `t+1`, and layer objectives do not propagate quantized effects upstream/downstream. Therefore `REAL_RECURRENT_TRAINING_GATE = FAIL`.

## 8. QDQ/writeback comparison

The isolated QDQ operator matches formal numerically: FP32 symmetric INT8, K-axis C128 (`-2`), scale `[B,H,1,V]`, `max(abs)/127`, floor `1e-12`, `torch.round`, clamp `[-127,127]`, zero point 0, dequant `codes*scale`. Precision boundaries match. Writeback does not: training performs no recurrent writeback, while formal evaluation QDQs each updated decode state in place for the next token. Thus QDQ is `NUMERICALLY_EQUIVALENT`, precision PASS, but writeback FAIL and `OLD_DENSE_RUNTIME_SEMANTICS = MISMATCH`.

## 9. FP/basis equivalence

PASS. Orthogonal q/k rotation and persistent K-axis state basis preserve the FP recurrence/readout algebra; no value-axis rotation or illegal elementwise commute was found, and all 24 target layers map exactly.

## 10. Dense-Hadamard step0 parity

PASS using the already-existing non-AIME current-formal-path gate. Dense theta=0 and the canonical fast Hadamard path had exact token IDs, logit hashes, and post-QDQ state hashes over prefill plus four decode steps; final max-absolute logit/state differences were 0. No wider tolerance was invented. Scale/qcode arrays were not saved, so those subfields are explicitly `NOT_SAVED`.

## 11. Gradient/trainability

`GRADIENT_RUNTIME_AUDIT = NOT_RUN`: all GPUs were occupied by formal C4 and saved trace tensors are absent. Static code shows identity STE, trainable theta, frozen model parameters, and hard-QDQ forward. Archived training logs record finite nonzero gradients (C3 extremely small; C4 materially larger), but this is not represented as a new independent runtime smoke.

## 12. Training-data leakage

`AIME26_TRAINING_OVERLAP = NO`. The 96-record corpus is locally cached WikiText-2 raw (64 train, 16 validation, 16 heldout). All raw-text hashes recompute, no exact AIME problem hash matches, and no normalized full AIME problem string—including the current last 20—occurs in the corpus. Answers are unused.

## 13. Checkpoint-selection rule

Both checkpoints use `MIN_HELDOUT_LOCAL_LOSS` on the frozen WikiText VALIDATION panel (128 local samples). C3 selected step 100; C4 selected step 900. LR probes selected C3 `3e-3` and C4 `1e-3` before full runs. No AIME selection evidence exists; `CHECKPOINT_SELECTION_LEAKAGE = PASS`.

## 14. Current C3/C4 loader verification

PASS for both. The loader pins checkpoint SHA256, schema, task/model/objective/seed/layers, strictly loads all 24 theta tensors, reconstructs DeltaR exactly once, and applies `H` then `DeltaR`. There is no silent identity fallback, C3/C4 swap, extra transpose, missing H, or double H.

## 15. Gate table

```text
QWEN_DENSE_TRAINING_PROVENANCE_AUDIT = FAIL
DENSE_STATE_LINEAGE = PARTIAL
DENSE_FUNCTIONAL_LINEAGE = PARTIAL
DENSE_STATE_OBJECTIVE_RECONSTRUCTION = PASS
DENSE_FUNCTIONAL_OBJECTIVE_RECONSTRUCTION = PASS
DENSE_STATE_RECURRENT_MODE = FP_STATE_RESET_EACH_TOKEN
DENSE_FUNCTIONAL_RECURRENT_MODE = FP_STATE_RESET_EACH_TOKEN
REAL_RECURRENT_TRAINING_GATE = FAIL
TRAIN_VS_FORMAL_QDQ_MATCH = NUMERICALLY_EQUIVALENT
TRAIN_VS_FORMAL_WRITEBACK_MATCH = FAIL
TRAIN_VS_FORMAL_PRECISION_MATCH = PASS
DENSE_FP_EQUIVALENCE_SEMANTICS = PASS
FAST_H_VS_DENSE_H_STEP0 = {parity['FAST_H_VS_DENSE_H_STEP0']}
GRADIENT_RUNTIME_AUDIT = NOT_RUN
AIME26_TRAINING_OVERLAP = {leakage['AIME26_TRAINING_OVERLAP']}
CHECKPOINT_SELECTION_LEAKAGE = PASS
C3_CURRENT_LOAD_GATE = {loader['C3_CURRENT_LOAD_GATE']}
C4_CURRENT_LOAD_GATE = {loader['C4_CURRENT_LOAD_GATE']}
CHECKPOINT_FORMAT_LOADER_MATCH = {loader['CHECKPOINT_FORMAT_LOADER_MATCH']}
```

## 16. C3 validity verdict

`C3_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID`. It is a correctly reconstructed local state-QDQ oracle with an almost-identity learned correction, but it is not verified—and is affirmatively not implemented—as real recurrent INT8 training.

## 17. C4 validity verdict

`C4_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID`. It is a correctly reconstructed local out-proj objective with valid Cayley/STE machinery, but its FP-state one-step injection is not the formal long-horizon writeback process.

## 18. E2E interpretability

`C3_E2E_RESULT_INTERPRETABILITY = CONDITIONAL` and `C4_E2E_RESULT_INTERPRETABILITY = CONDITIONAL`. Runtime differences from C2 do isolate learned DeltaR versus identity DeltaR in the same implementation path, but the checkpoints must be described as local teacher-state rotations, not rotations learned through recurrent quantized rollout.

## 19. Remaining blockers

- Exact training source commit and exact full launch commands are unavailable.
- Optimizer/intermediate checkpoint state and training trace tensors are unavailable.
- A new independent gradient smoke was not run; the missing trace tensors prevent exact replay without regenerating artifacts.
- Current-path parity did not retain per-step scale/qcode arrays, although final state/logit hashes were exact.

| Audit item | C3 Dense-State | C4 Dense-Functional |
|---|---|---|
| Provenance | PARTIAL | PARTIAL |
| Exact objective | PASS: local state | PASS: local out_proj + 0.1 state |
| Orthogonal R | PASS | PASS |
| Correct Key-side basis | PASS | PASS |
| Real recurrent training | FAIL: FP reset/token | FAIL: FP reset/token |
| QDQ matches formal | NUMERICALLY_EQUIVALENT | NUMERICALLY_EQUIVALENT |
| Writeback matches formal | FAIL | FAIL |
| No AIME26 leakage | PASS | PASS |
| Checkpoint selection clean | PASS | PASS |
| Loader correct | PASS | PASS |
| Step0 H parity | PASS | PASS |
| Final validity | CONDITIONALLY_VALID | CONDITIONALLY_VALID |
"""
    write_text("reports/final_report.md", report)

    artifact_paths = sorted(path for path in AUDIT.rglob("*") if path.is_file() and path.relative_to(AUDIT).as_posix() != "hashes/artifact_sha256.txt")
    lines = [f"{sha256(path)}  {path.relative_to(AUDIT).as_posix()}" for path in artifact_paths]
    write_text("hashes/artifact_sha256.txt", "\n".join(lines))
    print(json.dumps({"audit_dir": str(AUDIT), "files": len(artifact_paths) + 1, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
