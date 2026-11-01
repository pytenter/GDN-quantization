"""CPU-only premeasurement freeze; does not allocate CUDA or read outcome values."""
from __future__ import annotations
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from objective import digest
from run_matched_l7 import publish, ROOT, PRIOR, TRAINER

PARENT = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1")
SCRIPTS = PARENT / "inherited/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1/scripts"
DIRECTION = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py")
TRANSPORT = PARENT / "experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py"
TRIAGE = Path("/data/zypan/experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1")

def command(args):
    return subprocess.check_output(args,text=True).strip()

def main():
    config = json.loads((ROOT/"PHASE_A_CONFIG.json").read_text())
    previous = json.loads((PRIOR/"frozen_config.json").read_text())
    for key in ["documents","document_gpu_map","gpu_uuids","windows","layers","conditions"]:
        assert config[key] == previous[key], key
    manifest = TRIAGE/"CANONICAL_SOURCE_MANIFEST.json"
    rows = json.loads(manifest.read_text())["files"]
    assert len(rows) == 80
    checked = []
    for row in rows:
        for kind in ["canonical","sidecar"]:
            entry = row[kind]
            assert Path(entry["path"]).stat().st_size == entry["bytes"]
            assert digest(entry["path"]) == entry["sha256"]
        checked.append({"key": row["key"], "canonical": row["canonical"], "sidecar": row["sidecar"]})
    stopped = [Path("/data/zypan/worktrees/ling-observability-gpu3-expansion-v1/hardware_expansion_gpu3_v1/prospective_parallel/EXTERNAL_HOST_REBOOT_discovery_STOP.json"),
        Path("/data/zypan/experiment_runs/LING_OBSERVABILITY_ATTEMPT05_CROSS_RUNTIME_ELIGIBILITY_AUDIT_V1/ATTEMPT05_QUALIFICATION_FAIL_AND_STOP.json")]
    assert digest(stopped[0]) == "63a652bc297bf6cf18abd2a9fc9a170fced93539ff7485c5640b9b260e3e623c"
    inherited_repro = json.loads((PRIOR/"REPRODUCIBILITY_IDENTITIES.json").read_text())
    # Identity-only extraction: do not access any historical or active KL outcomes.
    rotation_ids = inherited_repro["formal_run_identities"][0]["rotation_artifacts"]
    token_ids = {x["document"]:x["token_file"] for x in inherited_repro["formal_run_identities"]}
    for item in list(rotation_ids.values()) + list(token_ids.values()):
        assert digest(item["path"]) == item["sha256"]
    original = TRAINER.parents[1]
    source_names = [TRAINER, original/"configs/recurrent_training_config.json",
        original/"manifests/train_data_manifest.json", original/"manifests/checkpoint_manifest.json",
        original/"manifests/rotation_manifest.json", original/"checkpoints/l7/training_summary.json"]
    deps = [PRIOR/"run_short_horizon_proxy_v2.py", PRIOR/"frozen_config.json", DIRECTION, TRANSPORT,
        *[SCRIPTS/n for n in ["capture.py","replay_core.py","common.py"]], *source_names]
    hashes = {str(p):digest(p) for p in deps}
    publish(ROOT/"HISTORICAL_IMMUTABILITY_BASELINE.json", {
        "status": "READ_ONLY_REHASH_PASS", "canonical_count": 80,"sidecar_count":80,
        "manifest_sha256":digest(manifest),"files":checked,
        "attempt04_attempt05_permanent_stop": {str(p):digest(p) for p in stopped},
        "no_outcome_values_read_for_selection":True})
    host = {"epoch":time.time(),"kernel":platform.release(),"host":command(["hostname"]),
        "gpus":command(["nvidia-smi","--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu","--format=csv,noheader"]),
        "processes":command(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader"]),
        "tmux":command(["tmux","list-sessions"]),
        "disk_free_bytes":shutil.disk_usage(ROOT).free,
        "reboot_required":Path("/var/run/reboot-required").exists(),
        "active_external_experiment":"LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1",
        "external_document_ids_excluded_from_new_holdouts":["pg19-test-48693","pg19-test-51410"],
        "our_gpu_processes":[],"our_gpu_hours":0,
        "status":"AWAITING_GPU_NATURAL_RELEASE"}
    publish(ROOT/"HOST_WAIT_AUDIT_INITIAL.json",host)
    publish(ROOT/"SOURCE_IDENTITIES.json", {"source_hashes":hashes,
        "original_trainer_and_inherited_copy_equal": digest(TRAINER) == digest(ROOT/"sources/historical_trainer.py"),
        "model_identity":inherited_repro["source_audit"]["model_identity"],
        "model_weight_validation":"PRIOR_FRESH_HASH_AUDIT_IDENTIFIED; RECHECK_BEFORE_NEW_GPU_LAUNCH",
        "rotations":rotation_ids,"tokens":token_ids,
        "prior_scores_sha256":digest(PRIOR/"SHORT_HORIZON_PROXY_SCORES.csv"),
        "prior_fp_intermediate_hash_available":False,
        "matching_policy":"FRESH_SAME_REFERENCE_ALL_METRIC_RECOMPUTATION",
        "source_repositories":inherited_repro["source_audit"]["source_commits"]})
    tracked = ["objective.py","audit_l7_cpu_v2.py","run_matched_l7.py","test_observer_cpu.py",
        "prepare_l7.py","PHASE_A_CONFIG.json","EXPERIMENT_PLAN.md","L7_SOURCE_AND_CPU_PARITY_AUDIT.md"]
    publish(ROOT/"PREMEASUREMENT_CODE_FREEZE.json", {
        "status":"SOURCE_AND_SCIENCE_FROZEN_GPU_UNQUALIFIED","epoch":time.time(),
        "files":{p:digest(ROOT/p) for p in tracked},"dependencies":hashes,
        "outcomes_generated_or_used":False,
        "require_matched_gpu_smoke_on_both_UUIDs":True,
        "budget_smoke_projection_margin":1.5,
        "route_and_thresholds_frozen_in":"PHASE_A_CONFIG.json"})
    publish(ROOT/"GPU_COST_LEDGER.json", {
        "max_gpu_hours":16,"phase_a_target_cap_gpu_hours":2,
        "completed_gpu_hours":0,"active_gpu_hours":0,"gpu_runs":[],
        "cpu_audits_gpu_hours":0,"unrelated_external_experiment_hours_counted":False,
        "prospective_budget_gate":"NOT_EXECUTED","status":"AWAITING_GPU_NATURAL_RELEASE"})
    print("PREMEASUREMENT_SOURCE_FREEZE_COMPLETE; GPU_UNQUALIFIED; historical 80+80 rehashed")

if __name__ == "__main__":
    main()
