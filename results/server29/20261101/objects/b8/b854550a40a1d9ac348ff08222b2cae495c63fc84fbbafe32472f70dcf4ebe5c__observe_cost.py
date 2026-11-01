"""CPU-only observation of child exit times; never sends signals."""
import json
import time
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, atomic_json, sha256

launches = sorted((ROOT / "runs").glob("formal_*.LAUNCH.json"))
assert len(launches) == 2
pending = {json.loads(p.read_text())["pid"]: (p, json.loads(p.read_text())) for p in launches}
observed = []
while pending:
    for pid, (path, info) in list(pending.items()):
        proc = Path(f"/proc/{pid}")
        if not proc.exists() or (proc / "stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            endpoint = time.time()
            observed.append(dict(info, observed_exit_upper_epoch=endpoint,
                                  observed_complete_process_gpu_hours=(endpoint-info["started_epoch"])/3600,
                                  launch_receipt_sha256=sha256(path)))
            del pending[pid]
    if pending:
        time.sleep(1)
micro = json.loads((ROOT / "MICRO_VALIDATION.json").read_text())
atomic_json(ROOT / "OBSERVED_PROCESS_COST_LEDGER.json", {
    "method": "read-only /proc exit observation; <=1s sampling overhead per process; no signals",
    "runs": observed, "prior_failures_micro_gpu_hours_conservative": micro["consumed_gpu_hours"],
    "total_observed_gpu_hours_upper_bound": micro["consumed_gpu_hours"] + sum(x["observed_complete_process_gpu_hours"] for x in observed),
    "controller_ledger_may_overcount_earlier_finishing_gpu_wait": True})
print("PROCESS_COST_OBSERVATION_COMPLETE", flush=True)
