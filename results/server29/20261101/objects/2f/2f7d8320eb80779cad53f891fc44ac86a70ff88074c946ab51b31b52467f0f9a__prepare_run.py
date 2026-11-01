"""Freeze science and engineering before the new measurements; no label join."""
import json
import platform
import subprocess
import time
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, HIST_SOURCE, sha256, atomic_json

old = ROOT.parent / "LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1"
failures = []
for p in sorted((old / "runs").glob("*.STOP.json")):
    x = json.loads(p.read_text())
    failures.append(dict(path=str(p), sha256=sha256(p), error=x["error"],
                         gpu_hours=(x["stopped_epoch"] - x["started_epoch"]) / 3600))
result = subprocess.run(["/data/zypan/envs/ling-kda/bin/python", str(ROOT / "test_proxy_math_v2.py")],
                        capture_output=True, text=True)
atomic_json(ROOT / "CPU_TESTS.json", {"exit_code": result.returncode, "stdout": result.stdout,
                                      "stderr": result.stderr, "time_epoch": time.time()})
assert result.returncode == 0
paths = [ROOT / n for n in ["frozen_config.json", "SURROGATE_PROTOCOL.md",
         "TRAIN_TIME_INFORMATION_AUDIT.md", "run_short_horizon_proxy_v2.py", "test_proxy_math_v2.py"]]
paths += [HIST_SOURCE / "run_ling_kda_query_readout_causal_closure_v1.py",
          HIST_SOURCE / "pytorch_kda_reference.py"]
assert sha256(ROOT / "frozen_config.json") == sha256(old / "frozen_config.json")
atomic_json(ROOT / "PREMEASUREMENT_FREEZE.json", {
    "frozen_epoch": time.time(), "status": "FROZEN_BEFORE_PROXY_AND_LABEL_JOIN",
    "discovery_not_confirmation": True, "science_unchanged_from_parent": True,
    "files": {str(p): sha256(p) for p in paths}, "parent_failed_probes": failures,
    "prior_failure_gpu_hours": sum(x["gpu_hours"] for x in failures),
    "engineering_changes": ["reuse exact historical K3/prepared-query functions",
         "support canonical two- or three-dimensional model logits",
         "verify next kernel receives every post-QDQ state",
         "count complete subprocess wall time as conservative GPU time"],
    "budget_margin": "full projection multiplied by 1.25 before panel launch",
    "reduced_order": ["H", "L6", "L7", "IDENTITY", "DA_fromH", "DA_fromL6", "L7R", "L8"],
    "no_outcome_based_schedule": True, "kernel": platform.release(),
    "host_devices": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True),
    "compute_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True),
})
print("FREEZE_AND_CPU_TESTS_PASS", ROOT)
