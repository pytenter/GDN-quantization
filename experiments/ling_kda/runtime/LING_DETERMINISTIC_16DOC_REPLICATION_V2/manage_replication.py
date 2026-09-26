#!/usr/bin/env python3
"""Prepare, validate, aggregate, and report the deterministic Ling 16-doc run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASK = "LING_DETERMINISTIC_16DOC_REPLICATION_V2"
METHODS = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")
HORIZONS = (1, 4, 8, 16, 32, 64, 128)
REPEATS = 5
SEED = 20260922
EXPECTED_HOST = "nlpg-SYS-4029GP-TRT"
EXPECTED_SYSTEM_GPU_COUNT = 8
SELECTED_GPU_INDICES = (2, 3)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_PANEL_SHA256 = "80133307bb1fecba49ddc8929368a532e083e9c8314a9b9d3b4b05adc41ceb40"
EXPECTED_CHECKPOINTS = {
    "Dense_State": "dad5086c51057dd3b6965e99cb0f42774718061e18d0e1e55e77a94d65ce8318",
    "Dense_Functional": "65f21f18d1c4080cf8e689ef37389fc3c95c7f34afab71dcde3518d164e5297c",
}
EXPECTED_DEPENDENCIES = {
    "experiments/ling_kda/runtime/repeatability_closure_v1/run_ling_repeatability.py":
        "dc9ed574b9b3e29fd904cbb528d67d323775d1a3326ff6075403643c8f011e39",
    "experiments/ling_kda/rotation/persistent_headroom_confirmation_v1/streaming_evaluate_ling.py":
        "5a66aa8426c8b4100700ea8b4e2db1e8b74af338612f97c5674da266b76d1eca",
    "experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/evaluate_ling_dense_oracle.py":
        "68d3c7d36e3cadc685c57f12f264c6f73cdbb24999b6f1a9692ad84f40b4f4e9",
    "experiments/shared/rotation/cayley_rotation.py":
        "62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067",
    "experiments/shared/rotation/orthogonal_matrix_generator.py":
        "dc198c1c45830e440d0e053c81964f4ad69f968fee09cce8769e36274c10fbd5",
}
PANEL = ROOT / "results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl"
PANEL_BASE = PANEL.parent / "PERSISTENT_HEADROOM_64DOC_MANIFEST_BASE.json"
FIXED_CONFIG_DIR = ROOT / "experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/fixed_fla_configs"
CLOSURE_SUMMARY = ROOT / "results/runtime/ling_fresh_process_first_divergence_closure_v2/closure_matrix_full_fixed/closure_matrix_summary.json"
CLOSURE_RUNS = CLOSURE_SUMMARY.parent / "fresh_process_runs"
LEGACY_SOURCE = Path("/data/zypan/repos/GDN-quantization/experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py")
MODEL = Path("/data/zypan/models/Ling-3.0-tiny")
CHECKPOINTS = {
    "Dense_State": ROOT / "results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt",
    "Dense_Functional": ROOT / "results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_inventory() -> list[dict]:
    raw = command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in raw.splitlines():
        index, name, uuid, memory, driver = [part.strip() for part in line.split(",")]
        rows.append({
            "index": int(index), "name": name, "uuid": uuid,
            "memory_mib": int(memory), "driver_version": driver,
        })
    return rows


def gpu_gate(gpus: list[dict]) -> list[dict]:
    if platform.node() != EXPECTED_HOST:
        raise RuntimeError(f"HOST_GATE_FAIL: {platform.node()} != {EXPECTED_HOST}")
    if len(gpus) != EXPECTED_SYSTEM_GPU_COUNT:
        raise RuntimeError(
            f"GPU_COUNT_GATE_FAIL: {len(gpus)} != {EXPECTED_SYSTEM_GPU_COUNT}"
        )
    selected = [row for row in gpus if row["index"] in SELECTED_GPU_INDICES]
    if [row["index"] for row in selected] != list(SELECTED_GPU_INDICES):
        raise RuntimeError(f"SELECTED_GPU_INDEX_GATE_FAIL: {selected}")
    if any(row["name"] != EXPECTED_GPU_NAME for row in selected):
        raise RuntimeError(f"GPU_MODEL_GATE_FAIL: {gpus}")
    expected_visible = ",".join(map(str, SELECTED_GPU_INDICES))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES_GATE_FAIL: {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return selected


def prepare() -> None:
    import torch
    import transformers
    import triton
    from transformers import AutoTokenizer

    system_gpus = gpu_inventory()
    gpus = gpu_gate(system_gpus)
    all_compute_processes = command(
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits", check=False,
    )
    selected_uuids = {row["uuid"] for row in gpus}
    compute_processes = "\n".join(
        line for line in all_compute_processes.splitlines()
        if line.split(",", 1)[0].strip() in selected_uuids
    )
    if compute_processes.strip():
        raise RuntimeError(f"GPU_BUSY_GATE_FAIL:\n{compute_processes}")

    if sha256_file(PANEL) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("PANEL_HASH_MISMATCH")
    dependency_hashes = {path: sha256_file(ROOT / path) for path in EXPECTED_DEPENDENCIES}
    if dependency_hashes != EXPECTED_DEPENDENCIES:
        raise RuntimeError(f"AUDITED_DEPENDENCY_HASH_MISMATCH: {dependency_hashes}")
    dependency_hashes[str(LEGACY_SOURCE)] = sha256_file(LEGACY_SOURCE)
    dependency_hashes[str(HERE / "run_one.py")] = sha256_file(HERE / "run_one.py")
    dependency_hashes[str(HERE / "manage_replication.py")] = sha256_file(HERE / "manage_replication.py")
    dependency_hashes[str(HERE / "launch.sh")] = sha256_file(HERE / "launch.sh")

    checkpoint_hashes = {name: sha256_file(path) for name, path in CHECKPOINTS.items()}
    if checkpoint_hashes != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"CHECKPOINT_HASH_MISMATCH: {checkpoint_hashes}")

    closure = load_json(CLOSURE_SUMMARY)
    required_closure = (
        closure.get("LING_REPEATABILITY_VERDICT") == "REPEATABILITY_CLOSED"
        and closure.get("16_doc_authorized") is True
        and closure.get("64_doc_authorized") is False
        and closure.get("all_key_checkpoints_bitwise") is True
        and closure.get("AUC_NOISE_P95_AFTER") == 0.0
        and closure.get("AUC_NOISE_MAX_AFTER") == 0.0
    )
    if not required_closure:
        raise RuntimeError(f"CLOSURE_GATE_FAIL: {closure}")

    kernel_files = []
    for path in sorted(FIXED_CONFIG_DIR.glob("*.json")):
        kernel_files.append({
            "name": path.name,
            "sha256": sha256_file(path),
            "config": load_json(path),
        })
    if len(kernel_files) != 6:
        raise RuntimeError(f"FIXED_KERNEL_COUNT_FAIL: {len(kernel_files)}")
    kernel_freeze = {
        "fla_cache_mode": "default",
        "fla_config_dir": str(FIXED_CONFIG_DIR),
        "files": kernel_files,
        "fused_rmsnorm_gated_override": {
            "selector": "LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8",
            "kwargs": {"BT": 16}, "num_warps": 8, "num_stages": 3,
        },
        "autotune_search_enabled": False,
    }
    kernel_hash = canonical_hash(kernel_freeze)
    generation = {
        "protocol": "historical_persistent_future_kl_teacher_forcing",
        "sampling": False, "teacher_forcing": True,
        "input_token_cap": 1024, "prefill_tokens": 128,
        "decode_trajectory_length": 128, "horizons": list(HORIZONS),
        "auc": "trapezoidal Future-KL over horizons, divided by 127",
        "warmup": "four-method local replay in audited runner",
    }
    generation_hash = canonical_hash(generation)

    rows = [json.loads(line) for line in PANEL.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["panel_index"]))
    selected = rows[:16]
    if [row["panel_index"] for row in selected] != list(range(16)) or not all(row["original_16"] for row in selected):
        raise RuntimeError("ORIGINAL_16_IDENTITY_GATE_FAIL")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    documents = []
    for row in selected:
        token_ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
        documents.append({
            key: row[key] for key in (
                "panel_index", "document_id", "original_16", "raw_text_sha256",
                "source", "source_split", "source_row_start", "source_row_end",
            )
        } | {
            "document_order": int(row["panel_index"]),
            "tokenized_length": len(token_ids),
            "token_ids_sha256": canonical_hash(token_ids),
        })
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = MODEL / name
        if path.is_file():
            tokenizer_files[name] = sha256_file(path)
    dataset_manifest = {
        "task": TASK, "created_at": now(), "document_count": 16,
        "selection": "original ordered panel indices 0..15; no shuffle or filtering",
        "source_corpus": str(PANEL), "source_corpus_sha256": sha256_file(PANEL),
        "source_manifest": str(PANEL_BASE), "source_manifest_sha256": sha256_file(PANEL_BASE),
        "tokenizer_path": str(MODEL), "tokenizer_file_sha256": tokenizer_files,
        "documents": documents,
    }
    dataset_manifest["dataset_manifest_sha256"] = canonical_hash(dataset_manifest)
    write_json(HERE / "dataset_manifest.json", dataset_manifest)

    status = command("git", "-C", str(ROOT), "status", "--short", "--branch")
    actual_commit = command("git", "-C", str(ROOT), "rev-parse", "HEAD")
    actual_branch = command("git", "-C", str(ROOT), "branch", "--show-current")
    expected_probe = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", "d757a35^{commit}"],
        capture_output=True,
    )
    disk = command("df", "-B1", "--output=size,used,avail,pcent,target", str(ROOT)).splitlines()[-1].split()
    env_keys = (
        "CUDA_VISIBLE_DEVICES", "FLA_CACHE_MODE", "FLA_CONFIG_DIR",
        "LING_GATED_RMSNORM_FIXED_CONFIG", "TRITON_CACHE_DIR",
        "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING",
        "PYTORCH_CUDA_ALLOC_CONF", "LING_MODEL_PATH",
    )
    model_hashes = {}
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        path = MODEL / name
        if path.is_file():
            model_hashes[name] = sha256_file(path)
    runtime_payload = {
        "task": TASK,
        "captured_at": now(),
        "repository": {
            "root": str(ROOT), "commit": actual_commit, "branch": actual_branch,
            "status": status, "dirty": len(status.splitlines()) > 1,
            "recorded_closure_branch": "exp/ling-fresh-process-first-divergence-closure-v2",
            "recorded_closure_commit": "d757a35",
            "recorded_closure_commit_present": expected_probe.returncode == 0,
            "provenance_resolution": (
                "recorded closure commit absent from server object database; execution is pinned by exact audited dependency, checkpoint, panel, closure-artifact, and kernel hashes without altering the historical worktree"
            ),
        },
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "triton": triton.__version__, "fla_core": package_version("fla-core"),
            "transformers": transformers.__version__, "sglang": package_version("sglang"),
            "variables": {key: os.environ.get(key) for key in env_keys},
        },
        "hardware": {
            "hostname": platform.node(),
            "system_gpus": system_gpus,
            "system_gpu_count": len(system_gpus),
            "selected_gpu_indices": list(SELECTED_GPU_INDICES),
            "gpus": gpus,
            "gpu_count": len(gpus),
            "canonical_ling_environment": {
                "hostname": "lthpc1",
                "gpu_model": "NVIDIA GeForce RTX 4090",
                "gpu_count": 2,
                "driver_version": "535.113.01",
                "python": "3.11.16",
                "torch": "2.7.1+cu128",
                "cuda_runtime": "12.8",
                "triton": "3.3.1",
                "transformers": "4.57.6",
                "fla_core": "0.5.2",
            },
        },
        "storage": {"bytes_total": int(disk[0]), "bytes_used": int(disk[1]), "bytes_available": int(disk[2]), "use_percent": disk[3], "mount": disk[4]},
        "runtime": {
            "kernel_freeze": kernel_freeze, "kernel_configuration_sha256": kernel_hash,
            "generation_configuration": generation, "generation_configuration_sha256": generation_hash,
            "torch_compile": False, "cuda_graph": False, "sampling": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "scientific_invariants": {
            "model": "Ling-3.0-tiny/KDA", "model_path": str(MODEL),
            "model_file_sha256": model_hashes,
            "quantizer": "INT8_R128", "group_axis": "R / tensor dim -1",
            "group_size": 128, "rounding": "ties-to-even via torch.round",
            "clamp": [-127, 127], "scale_shape": "[B,H,K,1]",
            "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
            "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
        },
        "closure": {
            "summary_path": str(CLOSURE_SUMMARY), "summary_sha256": sha256_file(CLOSURE_SUMMARY),
            "summary": closure,
        },
        "dependencies_sha256": dependency_hashes,
        "checkpoints_sha256": checkpoint_hashes,
        "panel_sha256": sha256_file(PANEL),
        "gpu_compute_processes_before_launch": compute_processes.splitlines(),
    }
    runtime_hash = canonical_hash(runtime_payload)
    runtime_manifest = runtime_payload | {"runtime_manifest_sha256": runtime_hash}
    write_json(HERE / "runtime_manifest.json", runtime_manifest)
    write_json(HERE / "launch_metadata.json", {
        "runtime_manifest_sha256": runtime_hash,
        "kernel_configuration_sha256": kernel_hash,
        "generation_configuration_sha256": generation_hash,
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
    })
    (HERE / "runtime_manifest.md").write_text(
        f"""# {TASK} runtime manifest

- Host: `{platform.node()}`
- System GPUs: `8 x NVIDIA GeForce RTX 3090`
- Selected physical GPUs: `2,3` only (`CUDA_VISIBLE_DEVICES=2,3`)
- Actual repository: `{actual_branch}` at `{actual_commit}` (dirty working tree preserved)
- Recorded closure commit `d757a35` present: `{expected_probe.returncode == 0}`
- Runtime manifest hash: `{runtime_hash}`
- Kernel configuration hash: `{kernel_hash}`
- Generation configuration hash: `{generation_hash}`
- Panel hash: `{sha256_file(PANEL)}`
- Closure verdict: `{closure['LING_REPEATABILITY_VERDICT']}`
- AUC noise after closure: p95 `{closure['AUC_NOISE_P95_AFTER']}`, max `{closure['AUC_NOISE_MAX_AFTER']}`

The recorded closure commit is absent from the server object database. The historical
working tree was not reset or modified. This replication is instead pinned to exact
audited code, kernel, panel, checkpoint, model-index, and closure-artifact hashes.
""",
        encoding="utf-8",
    )
    (HERE / "orchestration_diff.md").write_text(
        """# Orchestration-only extension

No historical evaluator file is edited. `run_one.py` imports the audited
`run_ling_repeatability.py`, changes only `SELECTED_INDICES` from `(0, 7, 15)`
to `tuple(range(16))`, and invokes its existing `main()`. Model loading, warmup,
rotation, INT8_R128 quantization, prefill, recurrent decode, Future-KL, AUC, and
checkpoint hashing therefore remain implemented by the audited code. Smoke parity
against yesterday's doc0 Hadamard and Dense-State fixed-config artifacts is mandatory.
""",
        encoding="utf-8",
    )
    print(json.dumps(load_json(HERE / "launch_metadata.json"), indent=2))


def comparable_payload(row: dict) -> dict:
    return {key: row[key] for key in (
        "document_id", "panel_index", "token_ids_sha256", "condition", "auc",
        "horizons", "checkpoint_hashes", "reference_hashes", "nonfinite", "seed_mode",
    )}


def check_smoke() -> None:
    run_dir = HERE / "smoke/gpu3/fresh_process_runs"
    paths = sorted(run_dir.glob("*.json"))
    rows = [load_json(path) for path in paths]
    required = {(method, repeat) for method in METHODS for repeat in (0, 1)}
    observed = {(row["condition"], int(row["repeat"])) for row in rows}
    checks = {
        "run_count": len(rows) == 8,
        "required_runs": observed == required,
        "all_finite": all(int(row["nonfinite"]) == 0 for row in rows),
        "all_doc0": all(int(row["panel_index"]) == 0 for row in rows),
    }
    by_key = {(row["condition"], int(row["repeat"])): row for row in rows}
    checks["fresh_process_pid_distinct"] = all(
        by_key[(method, 0)]["process_id"] != by_key[(method, 1)]["process_id"]
        for method in METHODS
    )
    checks["fresh_process_auc_identical"] = all(
        by_key[(method, 0)]["auc"] == by_key[(method, 1)]["auc"]
        for method in METHODS
    )
    checks["fresh_process_checkpoint_hashes_identical"] = all(
        by_key[(method, 0)]["checkpoint_hashes"]
        == by_key[(method, 1)]["checkpoint_hashes"]
        for method in METHODS
    )
    checks["fresh_process_reference_hashes_identical"] = all(
        by_key[(method, 0)]["reference_hashes"]
        == by_key[(method, 1)]["reference_hashes"]
        for method in METHODS
    )
    checks["fresh_process_trajectory_metrics_identical"] = all(
        by_key[(method, 0)]["horizons"] == by_key[(method, 1)]["horizons"]
        and by_key[(method, 0)]["token_ids_sha256"]
        == by_key[(method, 1)]["token_ids_sha256"]
        for method in METHODS
    )
    parity = {}
    for method in ("Hadamard", "Dense_State"):
        current = by_key[(method, 0)]
        historical = load_json(CLOSURE_RUNS / f"doc00_{method}_fixed_r00.json")
        parity[method] = comparable_payload(current) == comparable_payload(historical)
    environment = load_json(HERE / "smoke/gpu3/runtime_environment_single_fixed.json")
    checks["fixed_kernel_path_active"] = (
        environment.get("diagnostic_gated_rmsnorm_fixed_config_applied") is True
        and environment.get("LING_GATED_RMSNORM_FIXED_CONFIG") == "BT16_W8"
        and environment.get("gpu_name") == EXPECTED_GPU_NAME
        and environment.get("runtime_flags", {}).get("fla_cache_mode") == "default"
        and environment.get("runtime_flags", {}).get("fla_config_dir") == str(FIXED_CONFIG_DIR)
    )
    report = {
        "task": TASK, "stage": "SMOKE", "created_at": now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "historical_parity_by_condition": parity,
        "fresh_process_auc_by_condition": {
            method: [by_key[(method, 0)]["auc"], by_key[(method, 1)]["auc"]]
            for method in METHODS
        },
        "process_ids_by_condition": {
            method: [
                by_key[(method, 0)]["process_id"],
                by_key[(method, 1)]["process_id"],
            ]
            for method in METHODS
        },
    }
    write_json(HERE / "smoke_report.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("SMOKE_GATE_FAIL")


def bootstrap_median_ci(deltas: list[float], resamples: int = 10000) -> list[float]:
    rng = random.Random(20260921)
    values = []
    for _ in range(resamples):
        values.append(statistics.median(deltas[rng.randrange(len(deltas))] for _ in deltas))
    values.sort()
    return [values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]]


def trajectory_hash(row: dict) -> str:
    return canonical_hash({
        "token_ids_sha256": row["token_ids_sha256"],
        "checkpoint_hashes": row["checkpoint_hashes"],
        "reference_hashes": row["reference_hashes"],
        "horizons": row["horizons"],
    })


def finalize() -> None:
    launch_meta = load_json(HERE / "launch_metadata.json")
    runtime = load_json(HERE / "runtime_manifest.json")
    dataset = load_json(HERE / "dataset_manifest.json")
    token_counts = {row["document_id"]: row["tokenized_length"] for row in dataset["documents"]}
    gpu_by_index = {row["index"]: row for row in runtime["hardware"]["gpus"]}
    source_paths = sorted((HERE / "full").glob("gpu*/fresh_process_runs/*.json"))
    raw_rows = [load_json(path) for path in source_paths]
    normalized = []
    for path, row in zip(source_paths, raw_rows):
        physical_gpu = int(path.parents[1].name.removeprefix("gpu"))
        run_id = f"doc{int(row['panel_index']):02d}_{row['condition']}_r{int(row['repeat']):02d}_gpu{physical_gpu}"
        normalized.append({
            "run_id": run_id,
            "process_id": int(row["process_id"]),
            "condition": row["condition"],
            "document_id": row["document_id"],
            "panel_index": int(row["panel_index"]),
            "process_repeat": int(row["repeat"]),
            "random_seed": SEED,
            "seed_mode": row["seed_mode"],
            "physical_gpu_index": physical_gpu,
            "gpu_uuid": gpu_by_index[physical_gpu]["uuid"],
            "runtime_manifest_sha256": launch_meta["runtime_manifest_sha256"],
            "kernel_configuration_sha256": launch_meta["kernel_configuration_sha256"],
            "generation_configuration_sha256": launch_meta["generation_configuration_sha256"],
            "dataset_manifest_sha256": launch_meta["dataset_manifest_sha256"],
            "future_kl_auc": float(row["auc"]),
            "trajectory_length": 128,
            "token_count": int(token_counts[row["document_id"]]),
            "termination_reason": "fixed_teacher_forced_horizon_128",
            "repetition_metrics": None,
            "repetition_metrics_reason": "not applicable to teacher-forced persistent Future-KL protocol",
            "top_k_statistics": row["horizons"],
            "nonfinite": int(row["nonfinite"]),
            "token_ids_sha256": row["token_ids_sha256"],
            "trajectory_statistics_sha256": trajectory_hash(row),
            "checkpoint_hashes": row["checkpoint_hashes"],
            "reference_hashes": row["reference_hashes"],
            "source_result_sha256": canonical_hash(row),
        })

    expected = {(doc, method, repeat) for doc in range(16) for method in METHODS for repeat in range(REPEATS)}
    observed = {(row["panel_index"], row["condition"], row["process_repeat"]) for row in normalized}
    duplicate_free = len(observed) == len(normalized)
    groups = defaultdict(list)
    for row in normalized:
        groups[(row["panel_index"], row["condition"])].append(row)
    per_group = {}
    ranges = []
    for key, values in sorted(groups.items()):
        aucs = [row["future_kl_auc"] for row in values]
        trajectories = {row["trajectory_statistics_sha256"] for row in values}
        pids = [row["process_id"] for row in values]
        auc_range = max(aucs) - min(aucs)
        ranges.append(auc_range)
        per_group[f"doc{key[0]:02d}|{key[1]}"] = {
            "repeats": len(values), "process_ids": pids, "auc_values": aucs,
            "auc_range": auc_range, "auc_std": statistics.pstdev(aucs),
            "trajectory_statistics_bitwise_identical": len(trajectories) == 1,
            "all_finite": all(row["nonfinite"] == 0 for row in values),
        }
    noise = {
        "definition": "distribution of per-document-condition max(AUC)-min(AUC) across five fresh processes",
        "p50": percentile(ranges, 0.50), "p95": percentile(ranges, 0.95),
        "max": max(ranges, default=0.0),
    }
    repeatability_pass = (
        len(normalized) == 16 * len(METHODS) * REPEATS
        and observed == expected and duplicate_free
        and all(item["repeats"] == REPEATS for item in per_group.values())
        and all(item["auc_range"] == 0.0 for item in per_group.values())
        and all(item["trajectory_statistics_bitwise_identical"] for item in per_group.values())
        and all(item["all_finite"] for item in per_group.values())
    )
    repeatability = {
        "task": TASK, "status": "PASS" if repeatability_pass else "FAIL",
        "expected_runs": 320, "observed_runs": len(normalized),
        "fresh_processes_per_document_condition": REPEATS,
        "auc_runtime_noise": noise, "per_document_condition": per_group,
    }
    write_json(HERE / "fresh_process_repeatability.json", repeatability)

    canonical = {}
    for (doc, method), values in groups.items():
        canonical[(doc, method)] = sorted(values, key=lambda row: row["process_repeat"])[0]["future_kl_auc"]
    condition_stats = {}
    for method in METHODS:
        values = [canonical[(doc, method)] for doc in range(16)]
        condition_stats[method] = {
            "mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.pstdev(values), "min": min(values), "max": max(values),
            "per_document": values,
        }

    comparison_specs = (
        ("Hadamard_vs_Native", "Native_INT8", "Hadamard"),
        ("Dense_State_vs_Hadamard", "Hadamard", "Dense_State"),
        ("Dense_Functional_vs_Hadamard", "Hadamard", "Dense_Functional"),
        ("Dense_State_vs_Dense_Functional", "Dense_Functional", "Dense_State"),
    )
    comparisons = {}
    for name, baseline, candidate in comparison_specs:
        deltas = [canonical[(doc, baseline)] - canonical[(doc, candidate)] for doc in range(16)]
        wins = sum(delta > 0 for delta in deltas)
        ties = sum(delta == 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        comparisons[name] = {
            "baseline": baseline, "candidate": candidate,
            "delta_definition": "baseline Future-KL AUC minus candidate Future-KL AUC; positive favors candidate",
            "paired_mean_delta": statistics.mean(deltas),
            "paired_median_delta": statistics.median(deltas),
            "bootstrap_median_delta_ci95": bootstrap_median_ci(deltas),
            "bootstrap_resamples": 10000, "bootstrap_seed": 20260921,
            "candidate_win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
            "per_document_delta": deltas,
        }
    write_json(HERE / "paired_comparisons.json", comparisons)

    dense_state = comparisons["Dense_State_vs_Hadamard"]
    dense_functional = comparisons["Dense_Functional_vs_Hadamard"]
    state_credible = dense_state["bootstrap_median_delta_ci95"][0] > 0
    functional_credible = dense_functional["bootstrap_median_delta_ci95"][0] > 0
    all_finite = all(row["nonfinite"] == 0 for row in normalized)
    gate_checks = {
        "documents_16_of_16": {row["panel_index"] for row in normalized} == set(range(16)),
        "four_conditions_complete": {row["condition"] for row in normalized} == set(METHODS),
        "runs_320_of_320": len(normalized) == 320 and observed == expected,
        "no_nan_or_nonfinite": all_finite,
        "rotation_semantics_unchanged": runtime["scientific_invariants"]["KDA_ROTATION_SEMANTICS_VERSION"] == "CORRECTED_PREFILL_ENDPOINT_V2" and runtime["scientific_invariants"]["REDUNDANT_PREFILL_ENDPOINT_ROTATION"] == "NO",
        "quantizer_semantics_unchanged": runtime["scientific_invariants"]["quantizer"] == "INT8_R128",
        "fixed_config_runtime": runtime["runtime"]["kernel_freeze"]["autotune_search_enabled"] is False,
        "fresh_process_repeatability": repeatability_pass,
        "auc_noise_zero": noise["p95"] == 0.0 and noise["max"] == 0.0,
        "dataset_identity": dataset["source_corpus_sha256"] == EXPECTED_PANEL_SHA256 and dataset["document_count"] == 16,
        "artifacts_and_hashes_saved": True,
    }
    gate = "PASS" if all(gate_checks.values()) else "FAIL"
    results = {
        "task": TASK, "created_at": now(), "LING_DETERMINISTIC_16DOC_GATE": gate,
        "gate_checks": gate_checks, "condition_statistics": condition_stats,
        "auc_runtime_noise": noise, "records": normalized,
    }
    write_json(HERE / "results.json", results)
    csv_fields = (
        "run_id", "process_id", "condition", "document_id", "panel_index", "process_repeat",
        "random_seed", "physical_gpu_index", "gpu_uuid", "future_kl_auc", "trajectory_length",
        "token_count", "termination_reason", "nonfinite", "trajectory_statistics_sha256",
        "runtime_manifest_sha256", "kernel_configuration_sha256", "generation_configuration_sha256",
    )
    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in normalized:
            writer.writerow({key: row[key] for key in csv_fields})

    def fmt(value: float) -> str:
        return f"{value:.12g}"

    report_lines = [
        f"# {TASK}", "",
        f"1. `LING_DETERMINISTIC_16DOC_GATE = {gate}`", "",
        f"2. Fresh-process repeatability: **{repeatability['status']}**", "",
        f"3. AUC runtime noise: p50 = `{fmt(noise['p50'])}`, p95 = `{fmt(noise['p95'])}`, max = `{fmt(noise['max'])}`", "",
    ]
    labels = (("Native", "Native_INT8"), ("Hadamard", "Hadamard"), ("Dense-State", "Dense_State"), ("Dense-Functional", "Dense_Functional"))
    for number, (label, method) in enumerate(labels, 4):
        stats = condition_stats[method]
        report_lines.append(
            f"{number}. {label} AUC: mean `{fmt(stats['mean'])}`, median `{fmt(stats['median'])}`, "
            f"std `{fmt(stats['std'])}`, min `{fmt(stats['min'])}`, max `{fmt(stats['max'])}`"
        )
        report_lines.append("")
    for number, key in (
        (8, "Hadamard_vs_Native"),
        (9, "Dense_State_vs_Hadamard"),
        (10, "Dense_Functional_vs_Hadamard"),
    ):
        item = comparisons[key]
        wtl = item["candidate_win_tie_loss"]
        report_lines.extend([
            f"{number}. {key.replace('_', ' ')}:", "",
            f"   - paired mean delta: `{fmt(item['paired_mean_delta'])}`",
            f"   - paired median delta: `{fmt(item['paired_median_delta'])}`",
            f"   - bootstrap 95% CI (median delta): `[{fmt(item['bootstrap_median_delta_ci95'][0])}, {fmt(item['bootstrap_median_delta_ci95'][1])}]`",
            f"   - win/tie/loss: `{wtl['win']}/{wtl['tie']}/{wtl['loss']}`", "",
        ])
    either = state_credible or functional_credible
    canonical = runtime["hardware"]["canonical_ling_environment"]
    current = runtime["environment"]
    selected_gpus = runtime["hardware"]["gpus"]
    report_lines.extend([
        f"11. Credible Dense persistent headroom over Hadamard: **{'YES' if either else 'NO'}**. Dense-State: {'YES' if state_credible else 'NO'}; Dense-Functional: {'YES' if functional_credible else 'NO'}.", "",
        f"12. Previous large local Dense gains reflected persistently: **{'YES, directionally and with a positive paired bootstrap interval' if either else 'NO; the deterministic persistent panel does not establish translation of the large local gains'}**.", "",
        f"13. Fixed-config path ready on this RTX3090 environment: **{'YES' if gate == 'PASS' else 'NO'}**. Exact runtime dependency hashes are preserved.", "",
        f"14. 64-document confirmation scientifically justified: **{'YES' if gate == 'PASS' else 'NO'}**; whether it should be run next based on effect size: **{'YES' if gate == 'PASS' and either else 'NO'}**. It was not started by this task.", "",
        "15. RTX3090 execution environment versus canonical Ling RTX4090:", "",
        f"   - host: `{runtime['hardware']['hostname']}` vs `{canonical['hostname']}`",
        f"   - selected GPUs: `2 x {selected_gpus[0]['name']}` (physical {','.join(map(str, runtime['hardware']['selected_gpu_indices']))} of 8) vs `2 x {canonical['gpu_model']}`",
        f"   - driver: `{selected_gpus[0]['driver_version']}` vs `{canonical['driver_version']}`",
        f"   - Python/PyTorch/CUDA/Triton/Transformers/FLA: `{current['python'].split()[0]}` / `{current['torch']}` / `{current['cuda_runtime']}` / `{current['triton']}` / `{current['transformers']}` / `{current['fla_core']}`; canonical `{canonical['python']}` / `{canonical['torch']}` / `{canonical['cuda_runtime']}` / `{canonical['triton']}` / `{canonical['transformers']}` / `{canonical['fla_core']}`",
        "   - frozen panel, checkpoints, audited evaluator dependencies, generation protocol, and kernel configuration files are SHA-256 identical to V1.", "",
        "Positive paired delta means the candidate has lower Future-KL AUC. Historical pre-freeze AUC values are not used as targets or direct comparators.",
    ])
    (HERE / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (HERE / "reproducibility_report.md").write_text(
        "\n".join([
            f"# {TASK} reproducibility report", "",
            f"- Fresh processes bitwise identical at audited trajectory/checkpoint hashes: **{'YES' if repeatability_pass else 'NO'}**",
            f"- Teacher-forced trajectory statistics identical: **{'YES' if repeatability_pass else 'NO'}**",
            f"- AUC variance/range zero: **{'YES' if noise['max'] == 0.0 else 'NO'}**",
            f"- All four conditions stable: **{'YES' if repeatability_pass else 'NO'}**",
            f"- AUC noise p50/p95/max: `{fmt(noise['p50'])}` / `{fmt(noise['p95'])}` / `{fmt(noise['max'])}`",
            "",
            "Each document-condition cell was evaluated in five independent Python processes. Full tensors were not retained; compact SHA-256 evidence covers token IDs, reference horizons, condition checkpoints, logits, caches, and scalar horizon metrics.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "LING_DETERMINISTIC_16DOC_GATE": gate,
        "repeatability": repeatability["status"], "auc_noise": noise,
        "condition_statistics": {key: {k: v[k] for k in ("mean", "median")} for key, v in condition_stats.items()},
        "dense_state_credible": state_credible, "dense_functional_credible": functional_credible,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "check-smoke", "finalize"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "check-smoke":
        check_smoke()
    else:
        finalize()


if __name__ == "__main__":
    main()
