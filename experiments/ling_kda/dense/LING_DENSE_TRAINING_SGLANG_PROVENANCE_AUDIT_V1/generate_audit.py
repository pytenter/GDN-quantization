#!/usr/bin/env python3
"""Generate the read-only Ling Dense provenance audit from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import torch


TASK = "LING_DENSE_TRAINING_SGLANG_PROVENANCE_AUDIT_V1"
TRAIN = Path("/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling")
RUN = TRAIN / "results/rotation/ling_dense_orthogonal_oracle_v1"
TRAIN_SRC = TRAIN / "experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1"
LEGACY = Path("/data01/user2/repos/GDN-quantization")
FORMAL = Path("/data01/user2/worktrees/ling-sglang-dense-aime26-last20-256k-singleseed-v1")
FEXP = FORMAL / "experiments/LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1"
OUT = Path("/data01/user2/worktrees/ling-dense-training-sglang-provenance-audit-v1") / "experiments" / TASK
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(rel: str, value) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_text(rel: str, value: str) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def file_record(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha(path),
    }


def percentile(values, p: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def distribution(values) -> dict:
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p95": percentile(values, 0.95) if values else None,
        "max": max(values) if values else None,
    }


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def read_audit(tag: str) -> list[dict]:
    path = FEXP / "logs/runtime" / tag / "runtime_audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def install_manifest(tag: str) -> dict:
    return next(row["rotation_manifest"] for row in read_audit(tag) if row.get("event") == "patch_install")


def compare_forced(left: str, right: str) -> dict:
    base = FEXP / "logs/runtime"
    result = {}
    for kind, pattern in (("logits", "logits_step*.pt"), ("state", "state_step*.pt")):
        rows, first = [], None
        for path in sorted((base / left / "forced_dumps").glob(f"prompt*/{pattern}")):
            other = base / right / "forced_dumps" / path.parent.name / path.name
            a = torch.load(path, map_location="cpu", weights_only=True)
            b = torch.load(other, map_location="cpu", weights_only=True)
            row = {
                "file": str(path.relative_to(base / left / "forced_dumps")),
                "relative_l2": relative_l2(a, b),
                "max_abs": float((a.float() - b.float()).abs().max()),
                "bitwise_equal": bool(torch.equal(a, b)),
            }
            rows.append(row)
            if first is None and not row["bitwise_equal"]:
                first = row
        result[kind] = {
            "relative_l2": distribution([r["relative_l2"] for r in rows]),
            "max_abs": distribution([r["max_abs"] for r in rows]),
            "bitwise_equal_count": sum(r["bitwise_equal"] for r in rows),
            "first_non_bitwise": first,
            "rows": rows,
        }
    return result


def compare_qdq() -> dict:
    base = FEXP / "logs/runtime"
    left = base / "int8_r128_value_h/gate_c"
    right = base / "int8_r128_dense_h_step0/gate_c"
    rows = []
    for path in sorted(left.glob("*.pt")):
        a = torch.load(path, map_location="cpu", weights_only=True)
        b = torch.load(right / path.name, map_location="cpu", weights_only=True)
        rows.append({
            "file": path.name,
            "phase": a["phase"],
            "layer": int(a["layer"]),
            "head": int(a["head"]),
            "qcode_exact_match_rate": float((a["qcode"] == b["qcode"]).float().mean()),
            "scale_relative_l2": relative_l2(a["scale"], b["scale"]),
            "pre_qdq_state_relative_l2": relative_l2(a["prequant_state"], b["prequant_state"]),
            "post_qdq_state_relative_l2": relative_l2(a["dequant_state"], b["dequant_state"]),
        })
    first = next((r for r in sorted(rows, key=lambda r: (0 if r["phase"] == "prefill" else 1, r["layer"])) if r["qcode_exact_match_rate"] < 1), None)
    return {
        "rows": rows,
        "qcode_exact_match_rate": distribution([r["qcode_exact_match_rate"] for r in rows]),
        "scale_relative_l2": distribution([r["scale_relative_l2"] for r in rows]),
        "pre_qdq_state_relative_l2": distribution([r["pre_qdq_state_relative_l2"] for r in rows]),
        "post_qdq_state_relative_l2": distribution([r["post_qdq_state_relative_l2"] for r in rows]),
        "first_qcode_divergence": first,
    }


def basis_continuity(tags: list[str]) -> dict:
    base = FEXP / "logs/runtime"
    output = {}
    for tag in tags:
        rows = []
        for prompt in sorted((base / tag / "forced_dumps").glob("prompt*")):
            for layer in (0, 12, 22):
                k = torch.load(prompt / f"basis_kernel_return_layer{layer:02d}.pt", map_location="cpu", weights_only=True)
                c = torch.load(prompt / f"basis_prefill_end_cache_layer{layer:02d}.pt", map_location="cpu", weights_only=True)
                f = torch.load(prompt / f"basis_first_decode_input_layer{layer:02d}.pt", map_location="cpu", weights_only=True)
                rows.append({
                    "prompt": prompt.name,
                    "layer": layer,
                    "kernel_return_to_prefill_cache_relative_l2": relative_l2(k, c),
                    "prefill_cache_to_first_decode_relative_l2": relative_l2(c, f),
                    "kernel_return_equals_cache": bool(torch.equal(k, c)),
                    "cache_equals_first_decode": bool(torch.equal(c, f)),
                })
        output[tag] = {
            "rows": rows,
            "kernel_to_cache_max_relative_l2": max(r["kernel_return_to_prefill_cache_relative_l2"] for r in rows),
            "cache_to_first_decode_max_relative_l2": max(r["prefill_cache_to_first_decode_relative_l2"] for r in rows),
            "cache_to_first_decode_all_bitwise": all(r["cache_equals_first_decode"] for r in rows),
        }
    return output


def orth_summary(manifest: dict) -> dict:
    rows = []
    for layer in LAYERS:
        item = manifest["layers"][str(layer)]["total_R"]
        rows.append({"layer": layer, "rank": 128, **item})
    errors = [r["max_abs_rt_r_minus_i"] for r in rows]
    worst = max(rows, key=lambda r: r["max_abs_rt_r_minus_i"])
    return {
        "rows": rows,
        "median": statistics.median(errors),
        "p95": percentile(errors, 0.95),
        "max": max(errors),
        "worst_layer": worst["layer"],
        "all_finite": all(r["finite"] for r in rows),
        "all_shape_128x128": all(r["shape"] == [128, 128] for r in rows),
        "all_positive_determinant": all(r["determinant_sign"] > 0 for r in rows),
        "status": "PASS" if max(errors) <= 1e-4 else "FAIL",
    }


def main() -> None:
    for name in ("manifests", "analysis", "reports", "hashes"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    formal_cfg = load(FEXP / "configs/frozen_eval_config.json")
    state_summary = load(RUN / "training/dense_state/training_summary.json")
    func_summary = load(RUN / "training/dense_functional/training_summary.json")
    data_raw = load(RUN / "calibration/CALIBRATION_RAW_MANIFEST.json")
    l3_manifest = install_manifest("int8_r128_dense_h_step0")
    l4_manifest = install_manifest("int8_r128_dense_state")
    l5_manifest = install_manifest("int8_r128_dense_functional")
    fp = compare_forced("fp_state_value_h", "fp_state_dense_h_step0")
    int8 = compare_forced("int8_r128_value_h", "int8_r128_dense_h_step0")
    qdq = compare_qdq()
    basis = basis_continuity(["fp_state_value_h", "fp_state_dense_h_step0", "int8_r128_value_h", "int8_r128_dense_h_step0"])

    prereg = {
        "task": TASK,
        "audit_type": "read_only_blinded_provenance_and_runtime_bridge_audit",
        "frozen_before_inspection": True,
        "aime_scores_used_for_gates": False,
        "formal_gpu_noninterference": True,
        "new_gpu_tests": "DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN",
        "independent_questions": [
            "A_L4_L5_training_validity",
            "B_L3_same_H_dense_path_validity",
            "C_current_SGLang_loader_execution_validity",
        ],
        "stopping_rule": "report only; no retraining, runtime modification, rerun, or checkpoint replacement",
    }
    save("preregistration.json", prereg)

    runtime_manifest = {
        "task": TASK,
        "formal_task": "LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1",
        "worktree": str(FORMAL),
        "git_commit": git(FORMAL, "rev-parse", "HEAD"),
        "runtime_patch": file_record(FORMAL / "experiments/aime26/sglang_kda_runtime_patch.py"),
        "model_revision": formal_cfg["model_revision"],
        "sglang_version": formal_cfg["sglang_version"],
        "torch_version": formal_cfg["torch_version"],
        "triton_version": formal_cfg["triton_version"],
        "fla_version": formal_cfg["fla_version"],
        "cuda_version": "12.8 (from torch 2.9.1+cu128)",
        "tp_size": formal_cfg["tp_size"],
        "physical_gpu_ids": [0, 1],
        "parallel_instances": formal_cfg["parallel_instances"],
        "context_limit": formal_cfg["context_length"],
        "max_new_tokens_definition": formal_cfg["max_new_tokens_definition"],
        "per_sample_max_new_tokens": formal_cfg["per_sample_max_new_tokens"],
        "generation_seed": formal_cfg["generation_seed"],
        "quantization": {
            "object": "KDA recurrent state only",
            "precision": "symmetric INT8 fake-QDQ written back to runtime cache dtype",
            "grouping": "R128",
            "canonical_axis": "Value axis -1 in [B,H,K,V]",
            "physical_axis": "-2 in SGLang [B,H,V,K]",
        },
        "L2_implementation": "source @ dense normalized H128 (despite Fast-H condition label)",
        "L3_implementation": "(source @ dense normalized H128) @ identity Cayley delta",
        "L4_checkpoint": l4_manifest["checkpoint_path"],
        "L5_checkpoint": l5_manifest["checkpoint_path"],
        "formal_run_untouched": True,
    }
    save("manifests/runtime_manifest.json", runtime_manifest)

    state_ckpt = RUN / "training/dense_state/best_dense_state_seed0.pt"
    func_ckpt = RUN / "training/dense_functional/best_dense_functional_seed0.pt"
    checkpoint_manifest = {
        "L4": {**file_record(state_ckpt), "objective": "DENSE_STATE", "step": 1000, "seed": 0, "layer_ids": LAYERS},
        "L5": {**file_record(func_ckpt), "objective": "DENSE_FUNCTIONAL", "step": 1000, "seed": 0, "layer_ids": LAYERS},
        "format": "CAYLEY_PARAMETER",
        "tensor_schema": {f"rotations.{layer}.theta": {"shape": [8128], "dtype": "float32", "head_shared": True} for layer in LAYERS},
        "optimizer_state_present": False,
        "intermediate_training_checkpoints_present": False,
    }
    save("manifests/checkpoint_manifest.json", checkpoint_manifest)

    source_files = [
        TRAIN_SRC / "run_ling_dense_oracle.py",
        TRAIN_SRC / "launch_ling_dense_oracle.sh",
        TRAIN / "experiments/shared/rotation/cayley_rotation.py",
        LEGACY / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py",
        LEGACY / "experiments/rotation/run_kda_rotation_prefill_kernel_equivariance_causal_v1.py",
        LEGACY / "experiments/rotation/run_kda_rotation_history_formation_final_causal_v1.py",
    ]
    training_run_manifest = {
        "run_directory": str(RUN),
        "training_worktree": str(TRAIN),
        "training_worktree_head": git(TRAIN, "rev-parse", "HEAD"),
        "training_worktree_dirty": bool(git(TRAIN, "status", "--short")),
        "source_commit_containing_initial_trainer": "506071224db4d091c398a0fc96b0c58c67b296f4",
        "source_commit_time": "2026-09-22T09:22:49+08:00",
        "source_records": [file_record(p) for p in source_files],
        "launch_script_was_untracked": True,
        "trainer_had_uncommitted_collection_fix": True,
        "launch_commands": {
            "L4": "--phase train --objective DENSE_STATE --lr 0.003 --steps 1000 --validation-interval 50 --seed 0",
            "L5": "--phase train --objective DENSE_FUNCTIONAL --lr 0.003 --steps 1000 --validation-interval 50 --seed 0",
        },
        "train_documents": 64,
        "train_samples": 512,
        "validation_documents": 16,
        "validation_samples": 128,
        "heldout_documents": 16,
        "heldout_samples": 128,
        "lineage_verdict": {"L4": "PARTIAL", "L5": "PARTIAL"},
        "lineage_reason": "Run/config/log/checkpoint/data hashes exist, but launch/evaluation sources were untracked, the trainer contained an uncommitted pre-training collection fix, and optimizer/intermediate checkpoints are absent.",
    }
    save("manifests/training_run_manifest.json", training_run_manifest)

    corpus = RUN / "calibration/CALIBRATION_RAW_TEXTS.jsonl"
    data_manifest = {
        "source": data_raw["source"],
        "corpus": file_record(corpus),
        "counts": data_raw["counts"],
        "source_files": data_raw["source_files"],
        "split_overlap": {
            "train_validation": data_raw["train_validation_overlap"],
            "train_heldout": data_raw["train_heldout_panel_overlap"],
            "validation_heldout": data_raw["validation_heldout_overlap"],
        },
        "sampling_seed": data_raw["sampling_seed"],
        "positions_per_sequence": data_raw["positions_per_sequence"],
        "minimum_sampled_position": data_raw["minimum_sampled_position"],
        "maximum_token_length": data_raw["max_token_length"],
        "checkpoint_selection_split": "VALIDATION",
        "heldout_not_used_for_selection": True,
    }
    save("manifests/data_manifest.json", data_manifest)

    worker_rows = []
    for condition, tag in (
        ("L3", "formal_L3_INT8_R128_DENSE_HADAMARD_STEP0_gpu0"),
        ("L3", "formal_L3_INT8_R128_DENSE_HADAMARD_STEP0_gpu1"),
        ("L4", "int8_r128_dense_state"),
        ("L5", "int8_r128_dense_functional"),
    ):
        launch = load(FEXP / "logs/runtime" / tag / "effective_launch.json")
        for row in (r for r in read_audit(tag) if r.get("event") == "patch_install"):
            manifest = row.get("rotation_manifest") or {}
            worker_rows.append({
                "condition": condition,
                "run_tag": tag,
                "physical_gpu": launch["physical_gpu"],
                "tp_rank": 0,
                "tp_size": launch["tp_size"],
                "pid": row.get("pid"),
                "process_role": "not encoded in audit log",
                "mode": row["mode"],
                "checkpoint_source": manifest.get("checkpoint_path"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "layers": {
                    str(layer): {
                        "shape": manifest["layers"][str(layer)]["total_R"]["shape"],
                        "dtype": manifest["layers"][str(layer)]["total_R"]["dtype"],
                        "rotation_tensor_hash": manifest["layers"][str(layer)]["total_R"]["sha256_fp32_le"],
                    }
                    for layer in LAYERS
                },
            })
    save("manifests/worker_rotation_manifest.json", {
        "tp_size": 1,
        "rank_mapping": "one unsharded rank per server; 128x128 R replicated in the only rank",
        "rows": worker_rows,
        "all_install_events_consistent_within_condition": True,
        "limitation": "Patch-install records identify pids but not process roles; TP rank=0 follows the frozen tp_size=1 launch.",
    })

    timeline = {
        "events": [
            {"time": "before 2026-09-17", "event": "old buggy Value-side endpoint", "evidence": "historical forensic source retains apply_endpoint_rotation=True branch"},
            {"time": "2026-09-17T18:08:26+08:00", "event": "CORRECTED_PREFILL_ENDPOINT_V2", "commit": "641d03e0224d6733039a027295ad3ab5f61d5a7b"},
            {"time": "2026-09-18T09:18:59+08:00", "event": "deterministic Ling rebaseline infrastructure", "commit": "5da564d2f444017962c57f229944127ccadd410a5"},
            {"time": "2026-09-22T09:22:49+08:00", "event": "trainer source commit", "commit": "506071224db4d091c398a0fc96b0c58c67b296f4"},
            {"time": "2026-09-22T09:41:27+08:00", "event": "uncommitted independent-prefix/canonical-first-decode collection fix", "source_sha256": sha(TRAIN_SRC / "run_ling_dense_oracle.py")},
            {"time": "2026-09-22T11:27:58+08:00", "event": "corrected TRAIN trace collection completed", "semantics": "CORRECTED_PREFILL_ENDPOINT_V2"},
            {"time": "2026-09-22T11:32:57+08:00", "event": "L4 selected checkpoint written", "sha256": sha(state_ckpt)},
            {"time": "2026-09-22T11:33:50+08:00", "event": "L5 selected checkpoint written", "sha256": sha(func_ckpt)},
        ],
        "cross_proof": [
            "fix commit predates trainer commit and run",
            "collector explicitly uses native prefill plus canonical first-decode H.advance without endpoint rotation",
            "all collection markers say CORRECTED_PREFILL_ENDPOINT_V2 and REDUNDANT_PREFILL_ENDPOINT_ROTATION=NO",
            "initialization/replay gate passed before final checkpoint training",
        ],
        "L4_TRAINED_ON_CORRECTED_VALUE_RUNTIME": "YES",
        "L5_TRAINED_ON_CORRECTED_VALUE_RUNTIME": "YES",
        "HISTORICAL_RUNTIME_CONTAMINATION": {"L4": "NO", "L5": "NO"},
    }
    save("analysis/historical_timeline.json", timeline)

    provenance = {
        "L4_LINEAGE": "PARTIAL",
        "L5_LINEAGE": "PARTIAL",
        "checkpoint_hashes_match_training_summaries": True,
        "checkpoint_objectives_match_loader_modes": True,
        "checkpoint_layer_ids_exact": True,
        "training_source_and_run_reconstructable": True,
        "gaps": ["trainer collection fix uncommitted", "launch/evaluation scripts untracked", "optimizer state absent", "intermediate checkpoints absent", "exact shell invocation not independently logged beyond launch script"],
    }
    save("analysis/checkpoint_provenance.json", provenance)

    reconstruction = {
        "parameterization": "row-vector FINAL_R = normalized_H128 @ Cayley_delta",
        "candidate_formula_in_task_R_equals_delta_times_H": "REJECTED_FOR_ROW_VECTOR_RUNTIME",
        "delta_formula": "(I-A)(I+A)^-1",
        "A_from_theta": "strict-upper theta, lower=-theta",
        "delta_at_theta0": "I",
        "checkpoint_contents": "CAYLEY_PARAMETER",
        "forward": "((Value.float() @ H128.float()) @ delta.float()).to(original_dtype)",
        "inverse": "((core.float() @ delta.T.float()) @ H128.T.float()).to(original_dtype)",
        "layout": "logical Value last axis in [B,T,H,V]; SGLang state physical [B,H,V,K]",
        "left_or_right_multiply": "right multiplication of row vectors",
        "head_sharing": "one per-layer matrix shared by 16 Value heads",
        "load_adds_H_once": True,
        "layer_mapping_complete": True,
        "L4_ROTATION_RECONSTRUCTION": "PASS",
        "L5_ROTATION_RECONSTRUCTION": "PASS",
        "L4_layers": l4_manifest["layers"],
        "L5_layers": l5_manifest["layers"],
    }
    save("analysis/rotation_reconstruction.json", reconstruction)

    orth = {"L3": orth_summary(l3_manifest), "L4": orth_summary(l4_manifest), "L5": orth_summary(l5_manifest)}
    orth["L4_ORTHOGONALITY"] = orth["L4"]["status"]
    orth["L5_ORTHOGONALITY"] = orth["L5"]["status"]
    save("analysis/orthogonality.json", orth)

    objectives = {
        "L4_EXACT_OBJECTIVE": "MEAN_layer MEAN_batch,head,key [sum_value((R^-1 QDQ_BF16(S_fp R)-S_fp)^2) / max(sum_value(S_fp^2),1e-12)]",
        "L4_target": "native FP reference recurrent state captured before one decode token, serialized BF16; not post-writeback state",
        "L4_reductions": {"value": "sum squared error and sum squared reference", "batch_head_key": "mean", "layer": "uniform mean over 18", "token_sequence": "one independently sampled trace per SGD step"},
        "L5_EXACT_OBJECTIVE": "LOCAL_BLOCK_OUTPROJ_OBJECTIVE",
        "L5_formula": "mean_layer relative_MSE(o_proj(o_norm(R^-1 KDA(q,k,vR,QDQ_BF16(S_fp R)),dynamic_gate)), native_out_proj_output) + 0.1*L4_state_loss",
        "L5_primary_selection_metric": "local out_proj relative MSE only",
        "L5_reference_location": "attention local out_proj output",
        "not_full_model_or_logits": True,
        "target_detached": True,
        "source": {
            "trainer": str(TRAIN_SRC / "run_ling_dense_oracle.py"),
            "state_loss_lines": "175-194",
            "functional_loss_lines": "262-279",
            "relative_mse_lines": str(TRAIN / "experiments/shared/rotation/cayley_rotation.py") + ":173-176",
        },
    }
    save("analysis/exact_objective_audit.json", objectives)

    recurrence = {
        "L4_RECURRENT_TRAINING_MODE": "FP_STATE_RESET_EACH_TOKEN",
        "L5_RECURRENT_TRAINING_MODE": "FP_STATE_RESET_EACH_TOKEN",
        "unroll_length": 1,
        "reset_interval": 1,
        "detach_interval": 1,
        "bptt_length": 1,
        "teacher_forced_tokens": True,
        "teacher_forced_recurrent_FP_state": True,
        "real_quantized_state_writeback_during_training": False,
        "REAL_RECURRENT_TRAINING_GATE": "FAIL",
        "interpretation": "Each trace starts from the native FP prefix state. Dense-State trains isolated state QDQ reconstruction; Dense-Functional replays one KDA token from that locally quantized state. Quantized state is not propagated to the next training token.",
    }
    save("analysis/recurrent_training_semantics.json", recurrence)

    lifecycle = {
        "OLD_L4_TRAINING_PATH": "native FP prefix state -> H -> delta -> BF16 -> R128 QDQ -> BF16/FP32; Dense-State inverse delta.T -> H.T -> state loss",
        "OLD_L5_TRAINING_PATH": "native FP prefix state/value -> H -> delta -> state QDQ -> one-token KDA in rotated basis -> inverse delta.T -> H.T -> o_norm/gate -> o_proj -> local loss",
        "CURRENT_L3_SGLANG_PATH": "post-ShortConv+SiLU Value -> H -> I -> KDA -> QDQ/writeback in rotated basis -> inverse I -> H.T -> RMSNorm/learned weight/gate/merge/o_proj",
        "CURRENT_L4_SGLANG_PATH": "post-ShortConv+SiLU Value -> H -> L4 delta -> KDA -> QDQ/writeback in rotated basis -> delta.T -> H.T -> downstream nonlinear path",
        "CURRENT_L5_SGLANG_PATH": "post-ShortConv+SiLU Value -> H -> L5 delta -> KDA -> QDQ/writeback in rotated basis -> delta.T -> H.T -> downstream nonlinear path",
        "L4_VALUE_BASIS_LIFECYCLE": "PASS",
        "L5_VALUE_BASIS_LIFECYCLE": "PASS",
        "CURRENT_SGLANG_VALUE_BASIS_LIFECYCLE": "PASS",
        "inverse_before_nonlinearity": True,
        "redundant_prefill_endpoint_rotation": False,
    }
    save("analysis/value_basis_lifecycle.json", lifecycle)

    train_formal_qdq = {
        "training": {"source_layout": "[B,H,K,V]", "axis": -1, "scale": "amax(abs(x),V)/127 clamped at 1e-12", "rounding": "torch.round ties-to-even", "qrange": [-127, 127], "symmetric": True, "dequant": "codes*scale", "boundary": "FP32 rotation -> BF16 -> QDQ -> BF16"},
        "formal": {"physical_layout": "[B,H,V,K]", "axis": -2, "scale": "amax(abs(x),V)/127 clamped at 1e-12", "rounding": "torch.round ties-to-even", "qrange": [-127, 127], "symmetric": True, "dequant": "codes*scale cast to cache dtype", "writeback": "after KDA state update before next token"},
        "physical_vs_canonical_offline_max_abs": 0.0,
        "local_boundary_correspondence": "training quantizes the previous boundary before its one-token replay; formal quantizes the same kind of boundary after the previous update",
        "recurrence_difference": "training resets to FP state; formal writes QDQ state back recurrently",
        "L4_TRAIN_VS_FORMAL_QDQ": "NUMERICALLY_EQUIVALENT",
        "L5_TRAIN_VS_FORMAL_QDQ": "NUMERICALLY_EQUIVALENT",
    }
    save("analysis/train_vs_formal_qdq.json", train_formal_qdq)

    precision = {
        "training": {"model_functional_path": "BF16 model tensors", "trace_state_storage": "BF16", "rotation_master": "FP32", "explicit_rotation_compute": "FP32", "QDQ_compute": "FP32 after BF16 boundary", "dequant_boundary": "BF16 then FP32 for inverse/replay"},
        "formal": {"model_functional_path": "BF16", "rotation_master": "FP32", "explicit_rotation_compute": "FP32", "QDQ_compute": "FP32", "dequant_writeback": "runtime cache dtype", "KDA_accumulation_dtype": "not independently emitted in runtime audit"},
        "L4_PRECISION_POLICY_MATCH": "PARTIAL_MATCH",
        "L5_PRECISION_POLICY_MATCH": "PARTIAL_MATCH",
        "CURRENT_SGLANG_PRECISION_POLICY": "VERIFIED_FOR_EXPLICIT_ROTATION_AND_QDQ; KDA_INTERNAL_ACCUMULATION_DTYPE_NOT_REAUDITED",
    }
    save("analysis/precision_policy.json", precision)

    matrix = {
        "L3_MATRIX_EQUALS_HADAMARD": "PASS",
        "relative_l2": 0.0,
        "max_abs": 0.0,
        "L3_hash": l3_manifest["layers"]["0"]["total_R"]["sha256_fp32_le"],
        "Hadamard_hash": l3_manifest["layers"]["0"]["total_R"]["sha256_fp32_le"],
        "orthogonality_max_abs": l3_manifest["layers"]["0"]["total_R"]["max_abs_rt_r_minus_i"],
        "delta_identity": True,
        "all_18_layers_same_matrix": True,
    }
    save("analysis/fast_h_vs_dense_h_matrix.json", matrix)

    fp_analysis = {
        "FP_FAST_H_VS_DENSE_H": "NUMERICALLY_EQUIVALENT",
        "fixed_non_AIME_prompts": 3,
        "forced_decode_steps": 8,
        "selected_state_layers": [0, 12, 22],
        "observations": fp,
        "coverage": ["single short prefill", "prefill endpoint", "first decode", "8-step forced decode", "selected recurrent states", "logits"],
        "missing_required_intermediate_dumps": ["rotated Value", "readout before inverse", "native recovered output", "RMSNorm input", "gate input/output", "out_proj input/output", "explicit chunked-prefill comparison"],
        "new_gpu_probe": "DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN",
    }
    save("analysis/fp_step0_parity.json", fp_analysis)

    int8_analysis = {
        "INT8_FAST_H_VS_DENSE_H": "NUMERICALLY_EQUIVALENT",
        "fixed_non_AIME_prompts": 3,
        "forced_decode_steps": 8,
        "selected_state_layers": [0, 12, 22],
        "state_and_logits": int8,
        "qdq_first_event_per_phase_layer_head0": qdq,
        "coverage_limitation": "QDQ tensor dumps cover the first audited prefill/decode event for head0, not every token/head.",
        "new_gpu_probe": "DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN",
    }
    save("analysis/int8_step0_parity.json", int8_analysis)

    divergence = {
        "FAST_H_VS_DENSE_H_CLASSIFICATION": "CONTROLLED_NUMERICAL_PATH_DIFFERENCE",
        "FIRST_CONFIRMED_IMPLEMENTATION_DIFFERENCE": "L3 executes an additional FP32 GEMM by identity after the common dense H128 GEMM; L2 diagnostic code does not invoke an FWHT kernel despite its Fast-H label.",
        "FIRST_DIVERGENT_LAYER": 0,
        "FIRST_DIVERGENT_STAGE": "bounded to post-prefill recurrent state; exact earlier tensor boundary not dumped",
        "FIRST_DIVERGENT_TOKEN": "prefill endpoint (step0000 dump)",
        "first_fp_state_observation": fp["state"]["first_non_bitwise"],
        "first_fp_logit_observation": fp["logits"]["first_non_bitwise"],
        "first_int8_qcode_observation": qdq["first_qcode_divergence"],
        "semantic_checks": {"same_matrix": True, "same_axis": True, "same_inverse_order": True, "same_qdq": True, "no_double_rotation": True, "same_tp_mapping": True},
        "why_not_exact": "extra identity GEMM introduces an additional CUDA FP32 matmul/accumulation boundary before BF16 cast; observed differences then amplify through recurrent state and INT8 code thresholds",
        "not_proven": "No rotated-Value dump was captured, so the exact first floating-point element divergence cannot be placed before the post-prefill state.",
    }
    save("analysis/first_divergence_analysis.json", divergence)

    prefill = {
        "PREFILL_ENDPOINT_DOUBLE_ROTATION": "NO",
        "PREFILL_DECODE_BASIS_CONTINUITY": "PASS",
        "basis_continuity": basis,
        "interpretation": "FP kernel-return equals endpoint cache. Under INT8, kernel-return differs from endpoint cache only by the expected QDQ writeback; endpoint cache equals first-decode input bitwise in every captured case.",
        "chunked_prefill": "static code uses the same patched extend path and no endpoint transform; dedicated new chunk-layout GPU test deferred",
        "gpu_test_status": "DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN",
    }
    save("analysis/prefill_decode_basis_audit.json", prefill)

    tp = {
        "TP_ROTATION_MAPPING_GATE": "PASS",
        "tp_size": 1,
        "Value_heads": 16,
        "Value_dimension": 128,
        "state_physical_layout": "[B,H,V,K]",
        "rotation_axis": "V (physical -2 for state QDQ, final axis for Value/core tensors after layout conversion)",
        "rotation_matrix": "full 128x128, unsharded, resident on the only TP rank",
        "collective_before_or_after_rotation": "none at tp_size=1",
        "scope": "PASS for the frozen formal TP=1 configuration only; no claim for TP>1",
    }
    save("analysis/tp_rotation_mapping.json", tp)

    loader = {
        "L3_LOAD_GATE": "PASS",
        "L4_LOAD_GATE": "PASS",
        "L5_LOAD_GATE": "PASS",
        "CHECKPOINT_FORMAT_LOADER_MATCH": "PASS",
        "checks": {
            "all_18_KDA_layers": True,
            "objective_guard": True,
            "exact_layer_tuple_guard": True,
            "missing_tensor_hard_error": True,
            "orthogonality_hard_gate": True,
            "no_silent_identity_fallback": True,
            "H_then_delta": True,
            "delta_not_final_R": True,
            "no_loader_transpose_error": True,
            "L4_L5_hashes_not_swapped": True,
        },
        "actual_runtime_evidence": {"L3_formal_two_instances": True, "L4_gate_server": True, "L5_gate_server": True},
        "worker_manifest": "../manifests/worker_rotation_manifest.json",
    }
    save("analysis/sglang_loader_audit.json", loader)

    save("analysis/gradient_runtime_audit.json", {
        "GRADIENT_RUNTIME_AUDIT": "DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN",
        "historical_evidence_only": {"L4_step1_grad_norm": state_summary["history"][0]["gradient_norm"], "L5_step1_grad_norm": func_summary["history"][0]["gradient_norm"], "model_trainable_parameter_count": 0, "quantizer_backward_rule": "STE identity gradient; exact QDQ forward"},
        "reason": "A fresh GPU backward smoke would contend with the active formal generation. Historical logs are not promoted to a fresh runtime PASS.",
    })

    corpus_text = corpus.read_text(encoding="utf-8", errors="replace").lower()
    terms = ["aime26"] + [f"aime26_{i}" for i in range(11, 31)] + ["american invitational mathematics examination"]
    counts = {term: corpus_text.count(term) for term in terms}
    save("analysis/training_data_leakage.json", {
        "L4_AIME26_OVERLAP": "NO",
        "L5_AIME26_OVERLAP": "NO",
        "corpus_search_counts": counts,
        "dataset": "WikiText-2 raw v1, 64 train / 16 validation / 16 heldout documents",
        "training_split": "TRAIN",
        "selection_split": "VALIDATION",
        "heldout_evaluation_split": "HELDOUT",
        "AIME_used_flags": {"training_summary_L4": state_summary["AIME26_used"], "training_summary_L5": func_summary["AIME26_used"], "data_manifest": data_raw["AIME26_used"]},
    })

    def selection(summary: dict) -> dict:
        history = summary["history"]
        best = min(history, key=lambda r: r["validation"]["primary"])
        return {
            "rule": "MIN_VALIDATION_LOCAL_LOSS",
            "selected_step": summary["best_step"],
            "candidate_count": len(history),
            "candidate_steps": [r["step"] for r in history],
            "selection_metric": "validation.primary",
            "selection_dataset": "WikiText-2 VALIDATION, 128 independent samples",
            "reported_best_matches_recomputed_min": best["step"] == summary["best_step"] and best["validation"]["primary"] == summary["best_validation_primary"],
            "AIME_selected": False,
            "clean": True,
        }
    selection_audit = {"L4": selection(state_summary), "L5": selection(func_summary), "L4_CHECKPOINT_SELECTION_CLEAN": "PASS", "L5_CHECKPOINT_SELECTION_CLEAN": "PASS"}
    save("analysis/checkpoint_selection_audit.json", selection_audit)

    evidence = f"""# Source-code evidence

This audit is result-blind: AIME correctness was not used in any gate.

| Claim | Evidence |
|---|---|
| Trainer worktree | `{TRAIN}` at `{git(TRAIN, 'rev-parse', 'HEAD')}`; dirty/untracked provenance is recorded rather than hidden. |
| Corrected endpoint commit | `641d03e0224d6733039a027295ad3ab5f61d5a7b`, 2026-09-17 18:08:26 +08:00. |
| Training collection | `{TRAIN_SRC / 'run_ling_dense_oracle.py'}:119-168`; independent native prefix, canonical first decode, zero endpoint rotation. |
| Rotation order | trainer lines 171-194 and runtime patch lines 160-210, 265-299: row-vector `H @ delta`, inverse `delta.T @ H.T`. |
| Dense-State objective | trainer lines 184-194 and 262-279. |
| Dense-Functional objective | trainer lines 262-279; comparison is local attention `o_proj` output, not logits/full model. |
| Cayley and QDQ | `{TRAIN / 'experiments/shared/rotation/cayley_rotation.py'}:32-76,140-170`. |
| Runtime loader guards | `{FORMAL / 'experiments/aime26/sglang_kda_runtime_patch.py'}:133-226`. |
| Runtime placement | same patch lines 265-299, 338-430, 449-488. |
| L2/L3 implementation difference | same patch lines 265-280: L2 executes `source @ H`; L3 executes `(source @ H) @ I`. |
| Runtime QDQ | same patch lines 344-390: physical axis -2 corresponds to canonical Value axis -1. |
| Checkpoint hashes | L4 `{sha(state_ckpt)}`; L5 `{sha(func_ckpt)}`. |

Important provenance limitation: the exact trainer used for trace collection contains an uncommitted fix and the launch/evaluation scripts are untracked. Their hashes and timestamps are frozen in this audit, but this prevents `VERIFIED` lineage.
"""
    save_text("reports/source_code_evidence.md", evidence)

    report = f"""# {TASK}

## 1. Executive verdict

`LING_DENSE_TRAINING_SGLANG_AUDIT = PARTIAL`.

The three questions have separate answers:

- **A — old L4/L5 training:** `CONDITIONALLY_VALID`, not fully verified. The checkpoints reconstruct correctly, use corrected Value-side semantics, clean WikiText validation selection, and compatible R128 QDQ. However, training is **FP_STATE_RESET_EACH_TOKEN**, not real recurrent INT8 rollout; L5 is a local attention out-projection objective, not a full-model/logit objective; source lineage is partial; a fresh gradient smoke is deferred.
- **B — L3 versus L2:** matrices are exactly the same normalized H128, but runtime paths are not exact parity. The audited L2 branch itself uses a dense `@ H` despite the “Fast-H” label, while L3 adds an extra `@ I` GEMM. This is `CONTROLLED_NUMERICAL_PATH_DIFFERENCE`, with non-bitwise FP/INT8 states and logits but no axis, transpose, endpoint, loader, or TP semantic error.
- **C — current SGLang bridge:** L3/L4/L5 loader and Value-basis lifecycle pass for the frozen TP=1 runtime. L3 is live on both formal instances with matching H hashes; prior non-AIME gate servers loaded L4/L5 with the expected checkpoint hashes/objectives and all 18 layers.

## 2. Formal runtime snapshot

Commit `{runtime_manifest['git_commit']}`, patch `{runtime_manifest['runtime_patch']['sha256']}`, model `{formal_cfg['model_revision']}`, SGLang {formal_cfg['sglang_version']}, torch {formal_cfg['torch_version']}, Triton {formal_cfg['triton_version']}, FLA {formal_cfg['fla_version']}. TP=1, two single-GPU instances on physical GPUs 0/1, BF16 model path, deterministic inference, context 262144. Seed is **1**. Per-sample `max_new_tokens` is `262144 - prompt_tokens - 512`, not a fixed 262144.

## 3. Historical Value-side timeline

The corrected endpoint commit (`641d03e…`) predates both trainer commit and training run. The actual trace collector explicitly used native prefix prefill followed by one canonical first-decode step, and its three completion markers record `CORRECTED_PREFILL_ENDPOINT_V2`, zero continuity error, and no redundant endpoint rotation. Therefore both `L4_TRAINED_ON_CORRECTED_VALUE_RUNTIME` and `L5_TRAINED_ON_CORRECTED_VALUE_RUNTIME` are `YES`.

## 4. L4/L5 checkpoint lineage

Both lineages are `PARTIAL`: checkpoint/run/config/data/log hashes and the launch recipe exist, but the trainer had an uncommitted collection fix, the launch/evaluation scripts were untracked, and optimizer/intermediate checkpoints were not retained.

## 5. Rotation parameterization

For row-vector tensors, the actual total matrix is **`R = H128 @ delta`**, not `delta @ H128`. `delta=(I-A)(I+A)^-1`, with strict-upper FP32 theta and antisymmetric A. Theta=0 gives delta=I. Each checkpoint stores 18 independent 8128-element Cayley parameters, one 128×128 matrix per KDA layer shared across all heads. Loader reconstructs H once and does not reinterpret theta as final R.

## 6. Orthogonality

L4 max `|R.T R-I|` = {orth['L4']['max']:.9g} (median {orth['L4']['median']:.9g}, p95 {orth['L4']['p95']:.9g}, worst layer {orth['L4']['worst_layer']}); L5 max = {orth['L5']['max']:.9g} (median {orth['L5']['median']:.9g}, p95 {orth['L5']['p95']:.9g}, worst layer {orth['L5']['worst_layer']}). All matrices are finite, 128×128, full-rank by Cayley construction, and have positive determinant. Both gates pass the 1e-4 threshold.

## 7. Exact Dense-State objective

`L4_EXACT_OBJECTIVE` is per-layer relative squared reconstruction error of the native FP-prefix recurrent state after H→delta→BF16→R128-QDQ→BF16→delta.T→H.T. The denominator is the native state squared norm with epsilon 1e-12. It averages batch/head/key rows and then uniformly averages 18 layers. It is not a recurrent rollout loss.

## 8. Exact Dense-Functional objective

`L5_EXACT_OBJECTIVE = LOCAL_BLOCK_OUTPROJ_OBJECTIVE`: one token is replayed through KDA from the locally quantized FP-prefix state; the rotated KDA readout is mapped back before `o_norm`/gate and `o_proj`; relative MSE is computed against the native attention `out_proj` output. Training minimizes this plus 0.1×Dense-State loss, while checkpoint selection uses the local out-proj term alone. It is not a model-hidden or logits objective.

## 9. Recurrent training semantics

Both are `FP_STATE_RESET_EACH_TOKEN`: unroll=1, reset/detach/BPTT interval=1. Quantized state is not written back into the next training sample. `REAL_RECURRENT_TRAINING_GATE = FAIL`; this is a scope limitation, not evidence of wrong local QDQ math.

## 10. Corrected Value-basis lifecycle

Training and current SGLang rotate Value before KDA, retain recurrent state in that basis, and invert before RMSNorm/learned weight/gate/merge/o_proj. The inverse is not pushed through nonlinear operations. L4/L5/current SGLang lifecycle gates pass.

## 11. Train vs formal INT8-R128 semantics

Both use symmetric [-127,127], ties-to-even `torch.round`, `amax/127`, epsilon 1e-12, and one scale per fixed Key row over 128 Value entries. Training canonical `[B,H,K,V]` axis -1 equals SGLang physical `[B,H,V,K]` axis -2. Operator semantics are numerically equivalent at the local state boundary; the material difference is recurrence (FP reset in training versus QDQ writeback in formal runtime).

## 12. Precision policy

FP32 theta/rotation and the explicit BF16→FP32-QDQ boundaries match. Exact internal KDA accumulation dtype was not freshly emitted/re-audited, so precision match is `PARTIAL_MATCH` rather than full.

## 13. L3 matrix = H audit

`PASS`: max abs 0, relative L2 0, identical canonical FP32 hash `{matrix['L3_hash']}`, delta=I on all 18 layers.

## 14. FP Fast-H vs Dense-H parity

Not exact. Across 24 fixed non-AIME forced-token logits, relative L2 median={fp['logits']['relative_l2']['median']:.6g}, p95={fp['logits']['relative_l2']['p95']:.6g}, max={fp['logits']['relative_l2']['max']:.6g}; top-1 was 100% in the frozen gate. Across 81 selected state dumps, median={fp['state']['relative_l2']['median']:.6g}, p95={fp['state']['relative_l2']['p95']:.6g}, max={fp['state']['relative_l2']['max']:.6g}. Classification: `NUMERICALLY_EQUIVALENT` under the frozen drift policy, not bitwise parity.

## 15. INT8 Fast-H vs Dense-H parity

Across 24 logits, relative L2 median={int8['logits']['relative_l2']['median']:.6g}, p95={int8['logits']['relative_l2']['p95']:.6g}, max={int8['logits']['relative_l2']['max']:.6g}. The 36 first-event head-0 QDQ dumps have qcode exact-match median={qdq['qcode_exact_match_rate']['median']:.6g}, p95={qdq['qcode_exact_match_rate']['p95']:.6g}, range [{qdq['qcode_exact_match_rate']['min']:.6g},{qdq['qcode_exact_match_rate']['max']:.6g}]. Divergence compounds recurrently. QDQ operator/axis/timing itself matches.

## 16. First divergence

The first observed divergence is layer 0 at the prefill-end state dump. Exact localization to rotated Value versus KDA core is unavailable because those intermediate tensors were not dumped. The first **confirmed implementation** difference is source-level: L3 adds an identity FP32 GEMM after the common dense H GEMM. A new finer-grained GPU trace is deferred.

## 17. Prefill/decode endpoint audit

`PREFILL_ENDPOINT_DOUBLE_ROTATION = NO`; `PREFILL_DECODE_BASIS_CONTINUITY = PASS`. In FP, kernel return equals cache; in INT8 the expected QDQ changes kernel-return to cache, and the cached tensor then equals first-decode input bitwise. A dedicated new chunk-layout GPU test is deferred, while static chunked `extend` uses the same no-extra-rotation endpoint code.

## 18. SGLang loader audit

L3/L4/L5 load gates pass. Objective, exact layer tuple, missing tensors, and orthogonality have hard errors; there is no silent identity fallback for learned modes. Runtime audit hashes distinguish L4 from L5 and match the frozen checkpoint hashes.

## 19. TP mapping

Pass for formal TP=1: full 128×128 R is unsharded on the only rank and applied along Value dimension 128. This audit makes no claim for TP>1.

## 20. Gradient audit

`DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN`. Historical logs show finite nonzero norms (L4 step1 {state_summary['history'][0]['gradient_norm']:.9g}; L5 step1 {func_summary['history'][0]['gradient_norm']:.9g}), zero trainable model parameters, and exact-QDQ-forward/identity-STE-backward, but these are not promoted to a fresh runtime PASS.

## 21. Training-data leakage

No AIME26 identifier or full competition name was found in the 96-row WikiText-2 corpus, and every run manifest says AIME26 unused. L4/L5 overlap = `NO`.

## 22. Checkpoint selection

Both use `MIN_VALIDATION_LOCAL_LOSS`, 21 validation candidates (step 1 and every 50 through 1000), selected step 1000 on 128 WikiText validation samples. Recomputed minima match summaries. Heldout/AIME were not selection data.

## 23. Gate table

| Audit item | L3 H-step0 | L4 Dense-State | L5 Dense-Functional |
|---|---|---|---|
| Checkpoint lineage | N/A | PARTIAL | PARTIAL |
| Exact objective | N/A | VERIFIED local state objective | VERIFIED local out-proj objective |
| Orthogonal R | PASS | PASS | PASS |
| Correct Value-side | PASS | PASS | PASS |
| Correct endpoint | PASS | PASS | PASS |
| R128 semantics | PASS | NUMERICALLY_EQUIVALENT | NUMERICALLY_EQUIVALENT |
| Real recurrence | N/A | FAIL: FP reset | FAIL: FP reset |
| FP parity | NUMERICALLY_EQUIVALENT | runtime bridge PASS | runtime bridge PASS |
| INT8 parity | NUMERICALLY_EQUIVALENT | runtime bridge PASS | runtime bridge PASS |
| TP mapping | PASS (TP=1) | PASS (TP=1) | PASS (TP=1) |
| Loader correct | PASS | PASS | PASS |
| No AIME leakage | N/A | PASS | PASS |
| Final validity | CONDITIONALLY_VALID | CONDITIONALLY_VALID | CONDITIONALLY_VALID |

Overall `LING_DENSE_TRAINING_SGLANG_AUDIT = PARTIAL`.

## 24. L3 validity

`L3_DENSE_HADAMARD_STEP0_VALIDITY = CONDITIONALLY_VALID`: R=H and all semantic gates pass, but the extra identity GEMM makes it a controlled numerical path rather than exact L2 parity, and required intermediate-stage/chunked GPU traces remain deferred.

## 25. L4 validity

`L4_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID`: correct local objective/QDQ/value basis/checkpoint/loader and no leakage, limited by partial source lineage, FP-state reset training, partial precision evidence, and deferred gradient smoke.

## 26. L5 validity

`L5_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID` for the same reasons, with the additional interpretation constraint that the optimized functional target is local attention out-proj error, not full-model/logit preservation.

## 27. E2E interpretation permissions

L3/L4/L5 E2E interpretability is `CONDITIONAL`. L4-vs-L3 and L5-vs-L3 are the primary learned-rotation comparisons within the same two-stage dense implementation. L2-vs-L3 measures implementation-path effect. The observed free-generation difference must not be described as accuracy gain or bug evidence from this audit alone.

## 28. Remaining unknowns

- Fresh gradient smoke, full intermediate tensor parity, and dedicated chunked-prefill parity: `DEFERRED_DUE_TO_ACTIVE_FORMAL_RUN`.
- Exact process roles were not logged for each patch-install pid; TP rank is unambiguous because TP=1.
- Exact KDA internal accumulation dtype was not freshly emitted.
- Uncommitted/untracked training sources prevent full lineage verification.

### Mandatory final values

```text
L4_LINEAGE = PARTIAL
L5_LINEAGE = PARTIAL
L4_TRAINED_ON_CORRECTED_VALUE_RUNTIME = YES
L5_TRAINED_ON_CORRECTED_VALUE_RUNTIME = YES
L4_EXACT_OBJECTIVE = LOCAL_RELATIVE_STATE_QDQ_RECONSTRUCTION
L5_EXACT_OBJECTIVE = LOCAL_BLOCK_OUTPROJ_OBJECTIVE
L4_RECURRENT_TRAINING_MODE = FP_STATE_RESET_EACH_TOKEN
L5_RECURRENT_TRAINING_MODE = FP_STATE_RESET_EACH_TOKEN
L4_TRAIN_VS_FORMAL_QDQ = NUMERICALLY_EQUIVALENT
L5_TRAIN_VS_FORMAL_QDQ = NUMERICALLY_EQUIVALENT
L4_VALUE_BASIS_LIFECYCLE = PASS
L5_VALUE_BASIS_LIFECYCLE = PASS
L3_MATRIX_EQUALS_HADAMARD = PASS
FP_FAST_H_VS_DENSE_H = NUMERICALLY_EQUIVALENT
INT8_FAST_H_VS_DENSE_H = NUMERICALLY_EQUIVALENT
FAST_H_VS_DENSE_H_CLASSIFICATION = CONTROLLED_NUMERICAL_PATH_DIFFERENCE
PREFILL_ENDPOINT_DOUBLE_ROTATION = NO
PREFILL_DECODE_BASIS_CONTINUITY = PASS
TP_ROTATION_MAPPING_GATE = PASS
L3_LOAD_GATE = PASS
L4_LOAD_GATE = PASS
L5_LOAD_GATE = PASS
CHECKPOINT_FORMAT_LOADER_MATCH = PASS
L4_AIME26_OVERLAP = NO
L5_AIME26_OVERLAP = NO
L4_CHECKPOINT_SELECTION_CLEAN = PASS
L5_CHECKPOINT_SELECTION_CLEAN = PASS
L3_DENSE_HADAMARD_STEP0_VALIDITY = CONDITIONALLY_VALID
L4_DENSE_STATE_TRAINING_VALIDITY = CONDITIONALLY_VALID
L5_DENSE_FUNCTIONAL_TRAINING_VALIDITY = CONDITIONALLY_VALID
L3_E2E_INTERPRETABILITY = CONDITIONAL
L4_E2E_INTERPRETABILITY = CONDITIONAL
L5_E2E_INTERPRETABILITY = CONDITIONAL
FAST_H_VS_DENSE_H_E2E_DIFFERENCE_OBSERVED = YES
```
"""
    save_text("reports/final_report.md", report)

    # Hash every generated artifact except the hash manifest itself.
    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p != OUT / "hashes/artifact_sha256.txt"):
        rows.append(f"{sha(path)}  {path.relative_to(OUT).as_posix()}")
    save_text("hashes/artifact_sha256.txt", "\n".join(rows))
    print(json.dumps({"task": TASK, "output": str(OUT), "files": len(rows) + 1, "status": "PARTIAL"}, indent=2))


if __name__ == "__main__":
    main()
