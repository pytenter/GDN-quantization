#!/usr/bin/env python3
"""Record the required final post-termination verification."""

import json
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2")
PATH = ROOT / "provenance/V1_FINAL_PRE_TERMINATION_SNAPSHOT.json"


value = json.loads(PATH.read_text(encoding="utf-8"))
value["immediate_gpu_release_observation"] = {
    "status": "TRANSIENT_DRIVER_TABLE_RESIDUE",
    "old_worker_pids_still_listed_immediately_after_process_exit": [3571, 16958, 13717, 10627],
    "note": "No corresponding /proc process existed; the required later final verification found an empty compute-process list.",
}
value["final_verification"] = {
    "checked_utc": "2026-09-30T03:09:32Z",
    "relevant_process_search": [],
    "gpu_compute_processes": [],
    "gpu_summary": [
        "0, NVIDIA GeForce RTX 3090, 3 MiB / 24576 MiB, 0%",
        "1, NVIDIA GeForce RTX 3090, 3 MiB / 24576 MiB, 0%",
        "2, NVIDIA GeForce RTX 3090, 3 MiB / 24576 MiB, 0%",
        "3, NVIDIA GeForce RTX 3090, 3 MiB / 24576 MiB, 0%"
    ],
    "data_free_bytes": 33388761088,
    "data_free_gib": 33388761088 / (1024 ** 3),
}
value.update({
    "V1_GPU_RELEASE_GATE": "PASS",
    "V1_ALL_PROCESSES_STOPPED": True,
    "V1_PYTHON_WORKERS_REMAINING": 0,
    "V1_WRAPPERS_REMAINING": 0,
    "V1_DOWNSTREAM_PROCESS_REMAINING": 0,
    "NO_NEW_V1_DOWNSTREAM_STAGE_STARTED": True,
    "UNEXPECTED_V1_DOWNSTREAM_LAUNCH": False,
})
PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({key: value[key] for key in (
    "V1_ALL_PROCESSES_STOPPED", "V1_GPU_RELEASE_GATE", "V1_CONTROLLED_COMPLETED",
    "V1_RECURRENT_BASELINE_COMPLETED", "NO_NEW_V1_DOWNSTREAM_STAGE_STARTED")}, indent=2))
