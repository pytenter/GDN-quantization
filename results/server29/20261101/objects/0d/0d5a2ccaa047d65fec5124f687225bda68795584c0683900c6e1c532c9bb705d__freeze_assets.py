"""Append asset/launcher identities without altering the first code freeze."""
import json
import time
from objective import digest
from run_matched_l7 import ROOT, publish

files = ["verify_model_weights.py","FRESH_MODEL_WEIGHT_AUDIT.json","budget_phase_a.py",
    "launch_phase_a.py","freeze_assets.py","CPU_OBSERVER_PLUMBING_TESTS.json",
    "CPU_OBJECTIVE_PARITY_REMOTE.json","HISTORICAL_IMMUTABILITY_BASELINE.json","SOURCE_IDENTITIES.json",
    "ENGINEERING_AUDIT_APPENDIX_001.json"]
assert json.loads((ROOT/"CPU_OBSERVER_PLUMBING_TESTS.json").read_text())["status"] == "CPU_OBSERVER_PLUMBING_PASS"
assert json.loads((ROOT/"FRESH_MODEL_WEIGHT_AUDIT.json").read_text())["status"] == "PASS"
assert not list((ROOT/"runs").glob("*.json")), "GPU_EVIDENCE_ALREADY_EXISTS_BEFORE_SUPPLEMENT"
publish(ROOT/"PREMEASUREMENT_ASSET_SUPPLEMENT.json", {
    "epoch":time.time(),"status":"FROZEN_GPU_QUALIFICATION_PENDING",
    "parent_freeze_sha256":digest(ROOT/"PREMEASUREMENT_CODE_FREEZE.json"),
    "files":{n:digest(ROOT/n) for n in files},"no_gpu_outcomes_exist":not list((ROOT/"runs").glob("*.json")),
    "numerical_loss_or_scientific_threshold_changes":False})
print("ASSET_AND_LAUNCHER_FREEZE_COMPLETE; GPU_QUALIFICATION_PENDING")
