#!/usr/bin/env python3
"""Create the prospective formal protocol before any C5/C6 optimizer update."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
READINESS = ROOT.parent / "QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2"
RUNTIME_PATH = READINESS / "scripts/runtime.py"
CANONICAL_SHA = "48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e"
ROTATION_SHA = "62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067"
BASE_HEAD = "40c06322d5f9ad092fbfb908fabb29a107a0446f"
BRANCH = "exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1"


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    if ROOT.exists() and any((ROOT / name).exists() for name in ("preregistration.json", "analysis")):
        raise RuntimeError("formal protocol/output already exists; refusing overwrite")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("formal branch mismatch")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", BASE_HEAD, "HEAD"], cwd=REPO)
    head_before_preregistration = git("rev-parse", "HEAD")
    dirty_before = git("status", "--porcelain=v1", "-uall")
    if dirty_before:
        raise RuntimeError(f"formal worktree is not clean before preregistration:\n{dirty_before}")

    runtime = load_file(RUNTIME_PATH, "qwen_formal_readiness_runtime")
    core = runtime.load_core()
    import torch
    import transformers
    import triton
    from transformers import AutoTokenizer

    provenance = runtime.check_parent_sources()
    if provenance["canonical_c5_c6_source"]["actual_sha256"] != CANONICAL_SHA:
        raise RuntimeError("canonical C5/C6 source hash drift")
    if provenance["rotation_source"]["actual_sha256"] != ROTATION_SHA:
        raise RuntimeError("rotation source hash drift")

    parent_manifest = json.loads((READINESS / "configs/parent_manifest.json").read_text())
    full_document = json.loads((READINESS / "configs/full_document_protocol.json").read_text())
    erratum = json.loads((READINESS / "configs/token_digest_erratum.json").read_text())
    corpus_sha = file_sha256(core.CORPUS)
    if corpus_sha != "37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0":
        raise RuntimeError("frozen corpus hash drift")

    tokenizer = AutoTokenizer.from_pretrained(
        str(runtime.MODEL), trust_remote_code=True, local_files_only=True
    )
    train_rows = core.corpus_rows("TRAIN")
    validation_rows = core.corpus_rows("VALIDATION")
    if len(train_rows) != 64 or len(validation_rows) != 16:
        raise RuntimeError("canonical train/validation panel size drift")
    schedule_rng = random.Random(core.TRAIN_SEED)
    schedule = [train_rows[schedule_rng.randrange(len(train_rows))] for _ in range(core.OPTIMIZER_UPDATES)]
    schedule_ids = [row["document_id"] for row in schedule]
    if schedule_ids[:3] != full_document["canonical_train_schedule_first_three_document_ids"]:
        raise RuntimeError("canonical document schedule drift")

    def panel(rows: list[dict]) -> list[dict]:
        output = []
        for row in rows:
            ids, positions = core.tokenize(tokenizer, row)
            token_bytes = b"".join(int(value).to_bytes(8, "little", signed=True) for value in ids)
            output.append({
                "document_id": row["document_id"],
                "raw_text_sha256": row["raw_text_sha256"],
                "token_count": len(ids),
                "token_ids_le_int64_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "capture_positions_zero_based": positions,
            })
        return output

    train_panel = panel(train_rows)
    validation_panel = panel(validation_rows)
    first = train_panel[0]
    if (first["document_id"] != full_document["fixed_first_document"] or
            first["token_count"] != full_document["first_document_token_count"] or
            first["capture_positions_zero_based"] != full_document["first_document_capture_positions_zero_based"] or
            first["token_ids_le_int64_sha256"] != erratum["corrected_sha256"]):
        raise RuntimeError("first-document identity/token/capture/erratum gate failed")
    if (full_document["first_document_token_sha256"] != erratum["frozen_protocol_invalid_sha256"] or
            len(full_document["first_document_token_sha256"]) != 65 or
            full_document["first_document_token_sha256"][:8] + full_document["first_document_token_sha256"][9:] != erratum["corrected_sha256"]):
        raise RuntimeError("historical token-digest erratum provenance drift")

    # Express the canonical validation while-loop exactly and transparently.
    validation_updates = []
    next_validation = core.VALIDATION_INTERVAL_EXPOSURES
    for update in range(1, core.OPTIMIZER_UPDATES + 1):
        exposure = update * core.TARGETS_PER_DOCUMENT
        if exposure >= next_validation or exposure == core.TARGET_EXPOSURES:
            validation_updates.append(update)
            while next_validation <= exposure:
                next_validation += core.VALIDATION_INTERVAL_EXPOSURES

    experiment_files = {
        "formal_pipeline": ROOT / "scripts/formal_pipeline.py",
        "freeze_protocol": Path(__file__).resolve(),
        "readiness_runtime": RUNTIME_PATH,
        "readiness_fulldoc_common": READINESS / "scripts/fulldoc_common.py",
        "readiness_frozen_probe": READINESS / "scripts/frozen_probe.py",
        "canonical_c5_c6": runtime.SOURCE,
        "canonical_rotation": runtime.ROTATION,
        "canonical_modeling": runtime.SOURCE_ROOT / "transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
    }
    source_hashes = {name: {"path": str(path), "sha256": file_sha256(path)}
                     for name, path in experiment_files.items()}

    snapshot_dir = ROOT / "configs/source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "canonical_run_recurrent_dense.py": runtime.SOURCE,
        "canonical_cayley_rotation.py": runtime.ROTATION,
        "readiness_runtime.py": RUNTIME_PATH,
        "readiness_fulldoc_common.py": READINESS / "scripts/fulldoc_common.py",
        "token_digest_erratum.json": READINESS / "configs/token_digest_erratum.json",
    }
    for target_name, source in snapshots.items():
        shutil.copy2(source, snapshot_dir / target_name)

    model_files = [runtime.MODEL / "config.json", runtime.MODEL / "model.safetensors.index.json"]
    model_files.extend(sorted(runtime.MODEL.glob("*.safetensors")))
    model_inventory = [{"path": str(path), "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path)} for path in model_files]
    model_total = sum(item["size_bytes"] for item in model_inventory)

    objective_audit_path = (REPO / "experiments/qwen_gdn/dense/"
                            "QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1/"
                            "analysis/exact_objective_audit.json")
    selection_audit_path = (REPO / "experiments/qwen_gdn/dense/"
                            "QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1/"
                            "analysis/checkpoint_selection_audit.json")
    objective_audit = json.loads(objective_audit_path.read_text())
    selection_audit = json.loads(selection_audit_path.read_text())

    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_model": torch.cuda.get_device_name(0),
        "visible_cuda_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "driver": torch.cuda.get_device_properties(0).name and subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0],
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_path": transformers.__file__,
        "triton": triton.__version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "git_branch": BRANCH,
        "git_head_before_preregistration": head_before_preregistration,
        "git_dirty_status_before_preregistration": dirty_before,
        "deterministic_flags": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    model = {
        "name": "Qwen3.5-9B/GDN",
        "path": str(runtime.MODEL),
        "inventory": model_inventory,
        "total_inventory_bytes": model_total,
        "config_sha256": provenance["model_config"]["actual_sha256"],
        "index_sha256": provenance["model_index"]["actual_sha256"],
        "gdn_layer_count": len(core.GDN_LAYERS),
        "gdn_layer_ids": list(core.GDN_LAYERS),
        "recurrent_state_layout": "[B,H,K,V]",
        "rotation_key_dimension": 128,
    }
    quantization = {
        "scope": "only recurrent state is quantized",
        "mode": "symmetric INT8 C128",
        "implementation": "source.float(); scale=amax(abs(source),dim=-2,keepdim=True).clamp_min(1e-12)/127; codes=torch.round(source/scale).clamp(-127,127); dequant=codes*scale; STE identity gradient",
        "rounding": "torch.round ties-to-even",
        "clipping": "[-127,127]",
        "group_axis": "K / -2 for [B,H,K,V]",
        "state_storage_dtype": "torch.float32 dequantized recurrent state",
        "accumulation_dtype": "torch.float32 for rotation, quantization, and loss reductions",
        "source_sha256": ROTATION_SHA,
    }
    rotation = {
        "side": "canonical Qwen/GDN Key-side; identical row-vector transform applied to q and k",
        "parameterization": "A[upper]=theta; A[lower]=-theta; Delta=(I-A)(I+A)^-1",
        "runtime_orientation": "q/k.float() @ H128 @ Delta; recovery applies Delta then H128 on key axis",
        "initialization": "theta=0, Delta=I, final rotation=H128",
        "orthogonality_mechanism": "Cayley dense proper rotation",
        "trainable_parameters": 195072,
        "per_layer_degrees_of_freedom": 8128,
        "head_shared": True,
        "source_sha256": ROTATION_SHA,
    }
    initial_bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS)
    initial_rows = {
        str(layer): {
            "shape": list(initial_bank.layer(layer).theta.shape),
            "dtype": str(initial_bank.layer(layer).theta.dtype),
            "sha256": runtime.tensor_sha256(initial_bank.layer(layer).theta),
            "norm": float(torch.linalg.vector_norm(initial_bank.layer(layer).theta)),
        }
        for layer in core.GDN_LAYERS
    }
    rotation["initial_theta_per_layer"] = initial_rows
    rotation["initialization_manifest_sha256"] = hashlib.sha256(
        json.dumps(initial_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    common_training = {
        "horizon": 64,
        "tokens_per_document": 1024,
        "captures_per_document": 8,
        "capture_rule": "canonical sampled_positions: Random(int(raw_text_sha256[:16],16) XOR 20260921).sample(range(128,token_count),8), sorted",
        "target_exposures": core.TARGET_EXPOSURES,
        "optimizer_updates": core.OPTIMIZER_UPDATES,
        "training_documents": len(train_rows),
        "document_sampling": "Python random.Random(seed=0), with replacement, 125 draws from canonical TRAIN rows in file order",
        "document_order_sha256": text_sha256(schedule_ids),
        "document_order": schedule_ids,
        "corpus_sha256": corpus_sha,
        "optimizer_step_semantics": "one optimizer.step after the complete document; theta gradients accumulate across H64 segments",
        "graph_detach_semantics": "backward at each H64 boundary only when that segment contains capture loss; recurrent cache detached after every H64 segment; numerical INT8 state remains continuous",
        "recurrent_writeback": "real student post-QDQ INT8-C128 dequantized state",
        "gradient_clip_global_norm": 1.0,
        "weight_decay": 0.0,
        "random_seeds": {"python_random": 0, "torch_manual_seed": 0, "schedule_seed": 0},
        "distributed_training": False,
    }
    validation = {
        "dataset": "fixed WikiText-2 raw VALIDATION panel",
        "document_ids": [row["document_id"] for row in validation_rows],
        "documents": len(validation_rows),
        "captures_per_document": 8,
        "samples": len(validation_rows) * 8,
        "nominal_interval_target_exposures": core.VALIDATION_INTERVAL_EXPOSURES,
        "actual_validation_updates": validation_updates,
        "actual_validation_exposures": [update * 8 for update in validation_updates],
        "C5_primary_metric": "mean recurrent state relative-MSE over 16x8 held-out captures",
        "C6_primary_metric": "mean local out-proj relative-MSE over 16x8 held-out captures; excludes the 0.1 state auxiliary term",
        "selection_rule": "minimum validation.primary among all 20 validation candidates",
        "tie_breaking_rule": "strict-less update; exact ties retain the earliest candidate",
        "AIME_used": False,
    }
    stop_rules = [
        "canonical C5/C6 definition ambiguity", "source/hash drift", "missing or changed training data",
        "document order or capture position mismatch", "H != 64", "quantizer or recurrent writeback mismatch",
        "initialization mismatch or C6 inheritance from C5", "NaN or Inf", "catastrophic exploding gradients",
        "rotation orthogonality > 1e-4, nonfinite matrix, or non-positive determinant",
        "unexpected recurrent-state corruption", "CUDA OOM", "post-cleanup allocated-memory span > 256 MiB",
        "less than 2 GiB free on the artifact filesystem", "checkpoint write/hash failure",
        "GPU reset or unrecoverable process failure", "checkpoint-selection ambiguity",
    ]
    exclusions = [
        "AIME evaluation is not part of this task.",
        "AIME is not used for checkpoint selection.",
        "H128 is not retried.",
        "No post-hoc objective modification.",
        "No new loss term.",
        "No hyperparameter tuning based on downstream results.",
        "No AIME26, AIME24, MATH-500, or free-generation downstream benchmark.",
    ]

    canonical_audit = {
        "status": "PASS_UNIQUE",
        "why_this_version": "Readiness V2 parent_manifest points to this exact external canonical source and SHA; the same hash is reverified locally and readiness H64 used it.",
        "readiness_head": BASE_HEAD,
        "readiness_baseline_parent": parent_manifest["parent_sha"],
        "canonical_source": source_hashes["canonical_c5_c6"],
        "rotation_source": source_hashes["canonical_rotation"],
        "objective_audit": {"path": str(objective_audit_path), "sha256": file_sha256(objective_audit_path)},
        "selection_audit": {"path": str(selection_audit_path), "sha256": file_sha256(selection_audit_path)},
        "C5": objective_audit["DENSE_STATE_EXACT_OBJECTIVE"],
        "C6": objective_audit["DENSE_FUNCTIONAL_EXACT_OBJECTIVE"],
        "checkpoint_selection": selection_audit["conditions"],
        "canonical_definition_ambiguity": False,
    }
    protocol = {
        "experiment": ROOT.name,
        "formal_protocol_status": "PROSPECTIVE_NOT_YET_TRAINED",
        "environment": environment,
        "model": model,
        "quantization": quantization,
        "rotation": rotation,
        "training": {
            "common": common_training,
            "C5": {"objective": "DENSE_STATE exact objective", "optimized_loss": "state relative-MSE", "optimizer": "Adam", "lr": 0.003},
            "C6": {"objective": "DENSE_FUNCTIONAL exact objective", "optimized_loss": "local out-proj relative-MSE + 0.1 * state relative-MSE", "optimizer": "Adam", "lr": 0.001},
            "independence": "C5 and C6 each start from a fresh theta=0 bank and fresh Adam state; C6 never reads C5 theta or optimizer state",
        },
        "validation": validation,
        "stop_rules": stop_rules,
        "retry_rule": "Only a transient non-scientific infrastructure retry from an atomically hashed recovery checkpoint after the last committed optimizer step; sample/order/seed must remain identical and every retry is logged. Otherwise STOP.",
        "explicit_exclusions": exclusions,
        "historical_erratum": {
            "source": str(READINESS / "configs/token_digest_erratum.json"),
            "source_sha256": file_sha256(READINESS / "configs/token_digest_erratum.json"),
            "frozen_invalid_digest": erratum["frozen_protocol_invalid_sha256"],
            "corrected_digest": erratum["corrected_sha256"],
            "history_preserved": True,
        },
        "source_hashes": source_hashes,
    }

    atomic_json(ROOT / "configs/environment.json", environment)
    atomic_json(ROOT / "configs/model_provenance.json", model)
    atomic_json(ROOT / "configs/quantization.json", quantization)
    atomic_json(ROOT / "configs/rotation.json", rotation)
    atomic_json(ROOT / "configs/formal_training_protocol.json", protocol)
    atomic_json(ROOT / "configs/canonical_definition_audit.json", canonical_audit)
    atomic_json(ROOT / "configs/source_provenance.json", source_hashes)
    atomic_json(ROOT / "configs/C5.json", {**common_training, "condition": "C5", "objective": protocol["training"]["C5"], "validation": validation})
    atomic_json(ROOT / "configs/C6.json", {**common_training, "condition": "C6", "objective": protocol["training"]["C6"], "validation": validation})
    atomic_json(ROOT / "configs/training_document_panel.json", {"unique_panel": train_panel, "schedule": schedule_ids, "schedule_sha256": text_sha256(schedule_ids)})
    atomic_json(ROOT / "configs/validation_panel.json", {"panel": validation_panel, "document_ids": validation["document_ids"]})
    atomic_json(ROOT / "configs/token_digest_erratum_reference.json", erratum)
    atomic_json(ROOT / "preregistration.json", protocol)

    report = f"""# Prospective preregistration: {ROOT.name}

Status: **PROTOCOL FROZEN; C5/C6 FORMAL TRAINING NOT STARTED**

## Canonical-definition audit

- Unique canonical C5/C6 source: `{runtime.SOURCE}`
- Canonical source SHA256: `{CANONICAL_SHA}`
- Rotation/quantizer source SHA256: `{ROTATION_SHA}`
- Selection rule: minimum fixed held-out local primary loss; exact ties retain the earliest candidate.
- C5 optimized objective: recurrent state relative-MSE.
- C6 optimized objective: local out-proj relative-MSE + 0.1 x state relative-MSE; selection primary is local out-proj relative-MSE alone.

## Frozen matched protocol

- One 48GB vGPU; no distributed training.
- H=64; 1024 tokens/document; 8 canonical deterministic captures/document.
- 1000 target exposures = 125 document-level Adam updates.
- C5 LR 0.003; C6 LR 0.001; weight decay 0; global-norm clip 1.0.
- Real recurrent INT8-C128 writeback; graph detached only at H64 boundaries; one optimizer step per document.
- C5 and C6 use the identical frozen document order and validation panel, and independently start at theta=0 / Hadamard.
- Validation uses 16 documents x 8 captures at the 20 frozen boundaries listed in `configs/formal_training_protocol.json`.

## Provenance and erratum

Readiness V2 at `{BASE_HEAD}` selected H64. Its baseline parent is `{parent_manifest['parent_sha']}`. The historical invalid 65-character first-document digest remains preserved; the explicit one-character erratum is referenced, and the corrected canonical digest is `{erratum['corrected_sha256']}`.

## Stop and exclusion policy

All stop rules are frozen in `preregistration.json`. AIME is not part of this task, is not used for checkpoint selection, and remains NOT_STARTED. H128 is not retried. No objective, loss, LR, update count, precision, batch, capture count, or architecture may be changed post hoc.
"""
    atomic_text(ROOT / "reports/PREREGISTRATION.md", report)
    print(json.dumps({
        "status": "PROTOCOL_FILES_CREATED",
        "experiment": str(ROOT),
        "canonical_definition": "PASS_UNIQUE",
        "schedule_sha256": text_sha256(schedule_ids),
        "validation_candidate_count": len(validation_updates),
        "model_inventory_files": len(model_inventory),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
