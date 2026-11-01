"""Idle-device and identity guard; starts only a fixed Phase A tmux job.

Being busy is normal waiting, not a failed measurement or a reason to signal.
No analysis, Future-KL access, pair selection, or long-horizon launch occurs here.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from run_matched_l7 import ROOT, publish
from objective import digest

ap = argparse.ArgumentParser()
ap.add_argument("--gpu",type=int,choices=[3,6],required=True)
ap.add_argument("--mode",choices=["smoke","formal"],required=True)
args = ap.parse_args()
cfg = json.loads((ROOT/"PHASE_A_CONFIG.json").read_text())
doc = next(d for d,g in cfg["document_gpu_map"].items() if g == args.gpu)
tag = f"{args.mode}_{doc}_gpu{args.gpu}"
receipt = ROOT/"runs"/f"{tag}.LAUNCH_AUTHORIZATION.json"
if receipt.exists():
    print("ALREADY_AUTHORIZED_NO_DUPLICATE",tag); raise SystemExit(0)
if any((ROOT/"runs").glob("*.STOP.json")):
    raise RuntimeError("FAIL_CLOSED_RETAINED_STOP_NO_RETRY")
freeze = json.loads((ROOT/"PREMEASUREMENT_CODE_FREEZE.json").read_text())
for name,sha in freeze["files"].items():
    assert digest(ROOT/name) == sha,("CODE_FREEZE_CHANGED",name)
for name,sha in freeze["dependencies"].items():
    assert digest(name) == sha,("DEPENDENCY_CHANGED",name)
supplement = json.loads((ROOT/"PREMEASUREMENT_ASSET_SUPPLEMENT.json").read_text())
for name,sha in supplement["files"].items():
    assert digest(ROOT/name) == sha,("SUPPLEMENT_CHANGED",name)
weights = json.loads((ROOT/"FRESH_MODEL_WEIGHT_AUDIT.json").read_text())
assert weights["status"] == "PASS" and weights["files_rehashed"] == 32
for entry in weights["files"]:
    st = Path(entry["path"]).stat()
    assert st.st_size == entry["bytes"] and st.st_mtime_ns == entry["mtime_ns"],"WEIGHT_HASH_FRESHNESS_FAIL"
assert not Path("/var/run/reboot-required").exists(),"MAINTENANCE_GATE_FAIL"
assert shutil.disk_usage(ROOT).free >= 1024**3,"INSUFFICIENT_SHORT_STAGE_STORAGE"
lines = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"],text=True).splitlines()
parts = next([x.strip() for x in line.split(",")] for line in lines
             if int(line.split(",")[0]) == args.gpu)
assert parts[1] == cfg["gpu_uuids"][str(args.gpu)],"GPU_UUID_MISMATCH"
processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader"],text=True)
holders = subprocess.run(["fuser",f"/dev/nvidia{args.gpu}"],capture_output=True,text=True)
if parts[1] in processes or int(parts[2]) > 128 or int(parts[3]) != 0 or holders.returncode == 0:
    print("AWAITING_GPU_NATURAL_RELEASE",parts[1]); raise SystemExit(0)
assert holders.returncode == 1,"DEVICE_EXCLUSIVITY_NOT_PROVABLE"
if args.mode == "formal":
    gate = json.loads((ROOT/"PHASE_A_BUDGET_GATE.json").read_text())
    assert gate["status"] == "PASS" and gate["conservative_projected_gpu_hours"] <= 2
session = f"ling_l7_a_{args.mode}_gpu{args.gpu}"
present = subprocess.run(["tmux","has-session","-t",session],capture_output=True)
assert present.returncode != 0,"TMUX_COLLISION_NO_DUPLICATE"
publish(receipt,{"document":doc,"gpu_uuid":parts[1],"mode":args.mode,"epoch":time.time(),
    "session":session,"runner_sha256":digest(ROOT/"run_matched_l7.py"),
    "code_freeze_sha256":digest(ROOT/"PREMEASUREMENT_CODE_FREEZE.json"),
    "supplement_sha256":digest(ROOT/"PREMEASUREMENT_ASSET_SUPPLEMENT.json"),
    "device_holders":holders.stdout.strip(),"device_holder_return_code":holders.returncode,
    "no_signals_or_external_process_changes":True})
log = ROOT/"runs"/f"{tag}.log"
assert not log.exists()
# All shell arguments come from the fixed scientific document/GPU map above.
command = (f"cd {ROOT} && exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES={args.gpu} "
    f"PYTHONHASHSEED=0 LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8 "
    f"/data/zypan/envs/ling-kda/bin/python -u {ROOT}/run_matched_l7.py "
    f"--document {doc} --gpu {args.gpu} --mode {args.mode} > {log} 2>&1")
subprocess.run(["tmux","new-session","-d","-s",session,command],check=True)
print("PHASE_A_LAUNCHED",session,parts[1])
