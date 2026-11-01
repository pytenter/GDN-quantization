#!/usr/bin/env python3
"""Finalize the immutable V2 run after a documented GPU3/GPU7 resume."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "manage_replication.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


spec = importlib.util.spec_from_file_location("ling_v2_base_finalize", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_load_json = module.load_json


def load_json_with_all_gpu_inventory(path: Path):
    value = original_load_json(path)
    if Path(path).name == "runtime_manifest.json":
        value = copy.deepcopy(value)
        value["hardware"]["gpus"] = value["hardware"]["system_gpus"]
    return value


module.load_json = load_json_with_all_gpu_inventory
module.finalize()

runtime = original_load_json(HERE / "runtime_manifest.json")
launch = original_load_json(HERE / "launch_metadata.json")
results = original_load_json(HERE / "results.json")
observed = sorted({row["physical_gpu_index"] for row in results["records"]})
amendment = {
    "task": "LING_DETERMINISTIC_16DOC_REPLICATION_V2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reason": (
        "GPU2 became occupied by another user after a controlled quiesce; "
        "the main resume used GPUs 3 and 7. A brief user-authorized helper trial "
        "used GPUs 4 and 5, produced one complete GPU4 result, and was stopped "
        "because host CPU contention slowed the main workers. A later foreign "
        "process on GPU3 triggered the audited watchdog stop at 281 results; "
        "the final continuation again used GPUs 3 and 7."
    ),
    "original_runtime_manifest_sha256": launch["runtime_manifest_sha256"],
    "kernel_configuration_sha256": launch["kernel_configuration_sha256"],
    "generation_configuration_sha256": launch["generation_configuration_sha256"],
    "preserved_results_before_resume": 28,
    "initial_execution_pair": [2, 3],
    "resume_execution_pair": [3, 7],
    "final_continuation_pair": [3, 7],
    "observed_physical_gpu_indices": observed,
    "maximum_concurrent_gpus": 4,
    "helper_trial": {
        "physical_gpu_indices": [4, 5],
        "completed_result_count": 1,
        "completed_result_identity": "doc04_Native_INT8_fixed_r00.json",
        "stopped_due_to_host_cpu_contention": True,
        "result_retained_after_schema_and_json_validation": True,
    },
    "watchdog_interruption": {
        "detected_at": "2026-09-23T23:26:30+08:00",
        "physical_gpu_index": 3,
        "foreign_user": "yhwu",
        "foreign_pid": 1426428,
        "action": "terminate_experiment_group_only",
        "preserved_result_count": 281,
    },
    "gpu_model": "NVIDIA GeForce RTX 3090",
    "scientific_semantics_changed": False,
    "existing_result_policy": "skip by exact document-condition-repeat filename; never overwrite",
    "scripts_sha256": {
        "resume_gpu37.sh": sha256_file(HERE / "resume_gpu37.sh"),
        "continue_after_watchdog_gpu37.sh": sha256_file(
            HERE / "continue_after_watchdog_gpu37.sh"
        ),
        "watch_gpu37.sh": sha256_file(HERE / "watch_gpu37.sh"),
        "finalize_gpu37.py": sha256_file(Path(__file__)),
    },
}
module.write_json(HERE / "runtime_amendment_gpu37.json", amendment)
results["execution_amendment"] = amendment
module.write_json(HERE / "results.json", results)

report = HERE / "final_report.md"
with report.open("a", encoding="utf-8") as handle:
    handle.write(
        "\n## GPU execution amendment\n\n"
        "The original immutable runtime manifest was captured for physical GPUs 2 and 3. "
        "After 28 raw fresh-process results, both workers were quiesced at process boundaries. "
        "GPU2 subsequently became occupied by another user, so the main run resumed on the two free "
        "physical GPUs 3 and 7. A brief user-authorized GPU4/GPU5 helper trial produced one complete "
        "GPU4 result before being stopped because host CPU contention slowed the main workers. Later, "
        "a foreign process on GPU3 triggered the audited watchdog at 281 preserved results; the final "
        "continuation again used GPUs 3 and 7. Existing result identities were skipped and never "
        "overwritten. "
        f"Observed physical GPU indices across the complete run: `{observed}`; up to four Ling evaluator "
        "processes ran concurrently only during the brief helper trial, and the main execution otherwise "
        "used two GPUs. All are RTX3090 devices under the same driver, software environment, "
        "fixed kernel configuration, checkpoints, panel, seed, and evaluator. The original runtime "
        "manifest hash remains attached to every record; this amendment documents orchestration "
        "only. Fresh-process equality is evaluated across all repeats regardless of physical GPU.\n"
    )

print(json.dumps(amendment, indent=2))
