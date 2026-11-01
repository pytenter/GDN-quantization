"""CPU-only fixed mapping with no-duplicate, qualification, and complete cost gates."""
import argparse
import fcntl
import os
import subprocess
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from persistent_stage_common_v1 import RUNS, FREEZE, MODES, verify, complete, ledger, checked_json, idle_device


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ONLY_LAUNCHER_REQUIRED"
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, choices=(3, 6), required=True)
    ap.add_argument("--mode", choices=tuple(MODES), required=True)
    args = ap.parse_args()
    cfg, selection, freeze = verify()
    entry = next(x for x in selection["documents"] if x["gpu"] == args.gpu)
    doc = entry["document"]; tag = f"{args.mode}_{doc}_gpu{args.gpu}"
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "authorization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any((RUNS / (tag + s)).exists() for s in (".LAUNCH_AUTHORIZATION.json", ".STARTED.json", ".json", ".STOP.json", ".log")):
            print("ALREADY_AUTHORIZED_OR_EXECUTED_NO_DUPLICATE", tag); return
        if args.mode in ("micro", "formal"):
            for x in selection["documents"]: complete("qualification", x["document"], x["gpu"])
        if args.mode == "formal":
            gate_path = ROOT / "PERSISTENT_COMPLETE_COST_GATE_V1.json"
            gate = checked_json(gate_path)
            assert gate["status"] == "PASS" and gate["conservative_entire_task_gpu_hours"] <= 16
            for x in selection["documents"]:
                _, p = complete("micro", x["document"], x["gpu"])
                assert digest(p) == gate["micro_sha256"][x["document"]]
            cap = gate["document_projections"][doc]["wall_cap_seconds"]
        else: cap = MODES[args.mode]["wall"]
        account = ledger()
        assert all(x["status"] in ("COMPLETE_MECHANICAL_PASS", "ACTIVE_OR_FINALIZING") for x in account["persistent_runs"])
        # Reserve both modes' complete remaining roster, not a partial document.
        if args.mode != "formal":
            done = {x["document"] for x in account["persistent_runs"] if x["mode"] == args.mode}
            pending = sum(MODES[args.mode]["wall"] / 3600 for x in selection["documents"] if x["document"] not in done)
        else:
            done = {x["document"] for x in account["persistent_runs"] if x["mode"] == "formal"}
            pending = sum(gate["document_projections"][x["document"]]["gpu_hours"] for x in selection["documents"] if x["document"] not in done)
        reserved = sum(x["remaining_wall_reservation_hours"] for x in account["persistent_runs"])
        assert account["total_gpu_hours_upper_bound"] + pending + reserved <= 16, "COMPLETE_ROSTER_RESERVATION_FAIL"
        idle, hardware = idle_device(args.gpu, cfg)
        if not idle:
            print("AWAITING_GPU_NATURAL_RELEASE", hardware["gpu_uuid"]); return
        session = f"ling_l7_b_persistent_{args.mode}_gpu{args.gpu}"
        assert subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0
        log = RUNS / (tag + ".log")
        publish(RUNS / (tag + ".LAUNCH_AUTHORIZATION.json"), dict(epoch=time.time(), gpu=args.gpu,
            gpu_uuid=hardware["gpu_uuid"], document=doc, mode=args.mode, session=session, max_wall_seconds=cap,
            runner_sha256=digest(ROOT / "run_persistent_stage_v1.py"), freeze_sha256=digest(FREEZE),
            pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
            cost_gate_sha256=digest(gate_path) if args.mode == "formal" else None,
            entire_task_prelaunch_ledger=account, complete_roster_reservation_gpu_hours=pending+reserved,
            hardware=hardware, qualification_is_not_formal_authority=args.mode != "formal"))
        command = (f"cd {ROOT} && exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES={args.gpu} "
            f"PYTHONHASHSEED=0 LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            f"/data/zypan/envs/ling-kda/bin/python -B -u {ROOT}/run_persistent_stage_v1.py "
            f"--document {doc} --gpu {args.gpu} --mode {args.mode} > {log} 2>&1")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
        print("PERSISTENT_STAGE_LAUNCHED", args.mode, session, hardware["gpu_uuid"])


if __name__ == "__main__":
    main()
