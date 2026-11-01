"""CPU-only, fixed new-document launcher: smoke or full16 calibration, NEVER 32K."""
import argparse
import os
import subprocess
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import RUNS, FREEZE, verify, read, checked_json, audit_b, idle_device


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "LAUNCHER_MUST_BE_CPU_ONLY"
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=(3, 6), type=int, required=True)
    ap.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = ap.parse_args()
    cfg = verify()
    doc = next(d for d, gpu in cfg["document_gpu_map"].items() if gpu == args.gpu)
    tag = f"{args.mode}_{doc}_gpu{args.gpu}"
    receipt_path = RUNS / f"{tag}.LAUNCH_AUTHORIZATION.json"
    if receipt_path.exists() or any((RUNS / (tag + s)).exists() for s in (".STARTED.json", ".json", ".STOP.json")):
        print("ALREADY_EXECUTED_OR_AUTHORIZED_NO_DUPLICATE", tag); return
    ledger = audit_b()
    assert all(x["status"] in ("ACTIVE_OR_FINALIZING", "COMPLETE_MECHANICAL_PASS")
               for x in ledger["phase_b_calibration_runs"]), "UNRESOLVED_B_CAL_STATE"
    reservation = checked_json(ROOT / "PHASE_B_SHORT_STAGE_RESERVATION.json")
    assert reservation["status"] == "PASS" and reservation["conservative_projected_total_gpu_hours"] <= 16
    if args.mode == "formal":
        gate = checked_json(ROOT / "PHASE_B_CALIBRATION_BUDGET_GATE.json")
        assert gate["status"] == "PASS" and gate["conservative_projected_total_gpu_hours"] <= 16
        wall_limit = gate["per_document_wall_cap_seconds"][doc]
    else:
        wall_limit = cfg["smoke_wall_cap_seconds"]
    idle, evidence = idle_device(args.gpu, cfg)
    if not idle:
        print("AWAITING_GPU_NATURAL_RELEASE", evidence["gpu_uuid"]); return
    session = f"ling_l7_b_cal_{args.mode}_gpu{args.gpu}"
    assert subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0
    log = RUNS / f"{tag}.log"
    assert not log.exists(), "EXISTING_LOG_NO_DUPLICATE"
    publish(receipt_path, dict(document=doc, mode=args.mode, epoch=time.time(), session=session,
        max_wall_seconds=wall_limit, runner_sha256=digest(ROOT / "run_phase_b_calibration_v2.py"),
        freeze_sha256=digest(FREEZE), device_evidence=evidence, gpu_uuid=evidence["gpu_uuid"],
        no_future_labels_or_long_run_authorization=True))
    command = (f"cd {ROOT} && exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES={args.gpu} "
        f"PYTHONHASHSEED=0 LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8 "
        f"/data/zypan/envs/ling-kda/bin/python -B -u {ROOT}/run_phase_b_calibration_v2.py "
        f"--document {doc} --gpu {args.gpu} --mode {args.mode} > {log} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    print("PHASE_B_CALIBRATION_LAUNCHED", session, evidence["gpu_uuid"])


if __name__ == "__main__":
    main()
