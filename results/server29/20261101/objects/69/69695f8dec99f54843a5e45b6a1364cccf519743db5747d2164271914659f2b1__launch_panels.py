"""One-shot outcome-blind scheduling and conservative GPU budget accounting."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json, canonical_hash

PYTHON = "/data/zypan/envs/ling-kda/bin/python"


def available(uuid):
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                    "--format=csv,noheader"], text=True)
    return not any(uuid in row for row in rows.splitlines())


def main():
    config = json.loads((ROOT / "frozen_config.json").read_text())
    freeze = json.loads((ROOT / "PREMEASUREMENT_FREEZE.json").read_text())
    for path, digest in freeze["files"].items():
        assert sha256(Path(path)) == digest, ("FROZEN_FILE_CHANGED", path)
    micro_path = ROOT / "runs/micro_v2.json"
    assert micro_path.exists() and not (ROOT / "runs/micro_v2.STOP.json").exists()
    micro = json.loads(micro_path.read_text())
    assert micro["valid"] and micro["fp_reference"]["replay_parity"]["passed"]
    assert all(x["mechanical_gates"]["passed"] for x in micro["results"])
    repeats = [x for x in micro["results"] if x["condition"] == "H"]
    assert len(repeats) == 2 and repeats[0]["windows"] == repeats[1]["windows"], "DETERMINISM_FAIL"
    prior = freeze["prior_failure_gpu_hours"]
    # Include all load/reference/audit work and an extra 10 s for micro shutdown.
    consumed = prior + (micro["total_wall_seconds"] + 10) / 3600
    longest = max(x["elapsed_seconds"] for x in micro["results"])
    scaled = longest * 1152 / 384
    reference_and_load = micro["fp_reference"]["elapsed_seconds"] * 3 + micro["model_loading_seconds"] + 30
    order = config["conditions"]
    projected = consumed + 1.25 * 2 * (reference_and_load + len(order) * scaled) / 3600
    full_projection = projected
    if projected > config["max_gpu_hours"]:
        order = freeze["reduced_order"]
        while len(order) > 3 and consumed + 1.25 * 2 * (reference_and_load + len(order) * scaled) / 3600 > 4:
            order = order[:-1]
        projected = consumed + 1.25 * 2 * (reference_and_load + len(order) * scaled) / 3600
    atomic_json(ROOT / "MICRO_VALIDATION.json", {
        "status": "PASS", "micro_sha256": sha256(micro_path), "same_window_repeat": "BITWISE_SCORE_IDENTITY",
        "fp_replay": micro["fp_reference"]["replay_parity"], "consumed_gpu_hours": consumed,
        "full_panel_conservative_projected_total_gpu_hours": full_projection,
        "selected_projected_total_gpu_hours": projected, "longest_condition_seconds": longest,
        "selected_conditions": order, "future_labels_read_for_schedule": False})
    if projected > 4:
        atomic_json(ROOT / "BUDGET_STOP.json", {"status": "INSUFFICIENT_DATA_OR_RUNTIME_BUDGET",
                    "consumed_gpu_hours": consumed, "projected": projected})
        return
    per_doc_limit = (4 - consumed) * 3600 / 2 - 30
    # Freeze selection, allocations and projection before launch, independent of outcomes.
    atomic_json(ROOT / "PANEL_SCHEDULE_FREEZE.json", {
        "conditions": order, "documents": config["documents"], "document_gpu_map": config["document_gpu_map"],
        "conservative_projected_gpu_hours": projected, "per_process_wall_limit_seconds": per_doc_limit,
        "micro_and_prior_gpu_hours": consumed, "frozen_epoch": time.time(),
        "launch_order": config["documents"], "same_protocol_parallel": True,
        "historical_futurekl_used": False})
    # Rehash the inherited canonical + sidecar inventory without reading/scoring targets.
    manifest_path = Path("/data/zypan/experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1/CANONICAL_SOURCE_MANIFEST.json")
    source = json.loads(manifest_path.read_text())
    for row in source["files"]:
        for key in ["canonical", "sidecar"]:
            item = row[key]
            assert sha256(Path(item["path"])) == item["sha256"], ("INHERITED_HASH_FAIL", item["path"])
    atomic_json(ROOT / "INHERITED_SOURCE_HASH_AUDIT.json", {"status": "PASS", "canonicals": len(source["files"]),
                "sidecars": len(source["files"]), "manifest_sha256": sha256(manifest_path),
                "targets_used": False, "old_evidence_modified": False})
    locks, children, ledger = [], [], []
    for document in config["documents"]:
        gpu = config["document_gpu_map"][document]
        uuid = config["gpu_uuids"][str(gpu)]
        lock = open(f"/tmp/ling-short-surrogate-gpu{gpu}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        while not available(uuid):
            print(json.dumps({"event": "wait_for_assigned_gpu", "uuid": uuid}), flush=True)
            time.sleep(30)
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        assert f"{gpu}, {uuid}" in rows
        tag = f"formal_{document}_gpu{gpu}"
        log = open(ROOT / "runs" / f"{tag}.log", "x")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LING_GATED_RMSNORM_FIXED_CONFIG="BT16_W8", PYTHONHASHSEED="0")
        start = time.time()
        proc = subprocess.Popen([PYTHON, str(ROOT / "run_short_horizon_proxy_v2.py"), "--document", document,
               "--gpu", str(gpu), "--conditions", ",".join(order), "--tag", tag,
               "--max-wall-seconds", str(per_doc_limit)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((proc, log, tag, start, uuid))
        atomic_json(ROOT / "runs" / f"{tag}.LAUNCH.json", {"pid": proc.pid, "started_epoch": start,
                    "gpu_uuid": uuid, "tag": tag, "runner_sha256": sha256(ROOT / "run_short_horizon_proxy_v2.py")})
        print(json.dumps({"event": "panel_started", "document": document, "uuid": uuid, "pid": proc.pid}), flush=True)
    for proc, log, tag, start, uuid in children:
        rc = proc.wait()
        elapsed = time.time() - start
        # proc.wait on the other GPU can overestimate its elapsed time; conservative, never undercounts.
        ledger.append({"tag": tag, "gpu_uuid": uuid, "pid": proc.pid, "exit_code": rc,
                       "gpu_hours": elapsed / 3600})
        log.close()
    total = consumed + sum(x["gpu_hours"] for x in ledger)
    atomic_json(ROOT / "GPU_COST_LEDGER.json", {"prior_failures_and_micro_gpu_hours": consumed,
                "formal_runs": ledger, "total_gpu_hours_conservative": total, "max_gpu_hours": 4,
                "within_budget": total <= 4, "accounting": "complete subprocess wall time, including load/FP/cleanup"})
    assert total <= 4
    if not all(x["exit_code"] == 0 for x in ledger):
        atomic_json(ROOT / "PANEL_STOP.json", {"status": "MEASUREMENT_FAIL_NO_RETRY", "ledger": ledger})
        return
    print("PANELS_COMPLETE", flush=True)
    rc = subprocess.call([PYTHON, str(ROOT / "finalize_surrogate.py")], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert rc == 0


if __name__ == "__main__":
    main()
