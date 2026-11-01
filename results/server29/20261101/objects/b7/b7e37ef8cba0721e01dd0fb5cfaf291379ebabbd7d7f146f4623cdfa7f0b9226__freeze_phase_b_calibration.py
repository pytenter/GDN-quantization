"""Exclusive, outcome-blind source/config freeze and short-stage reservation."""
import json
import os
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, PRIOR, publish, load_module
from phase_b_calibration_common import read, checked_json, audit_a


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    route_path = ROOT / "phase_a_analysis_20261011/PHASE_A_ROUTE.json"
    assert digest(route_path) == "284f7e417f6956266b0c677013bd117f65c600a05a194abc0cfa6791e01b6346"
    assert read(route_path)["route"] == "A"
    tests = checked_json(ROOT / "PHASE_B_CALIBRATION_CPU_TESTS.json")
    assert tests["status"] == "CPU_INPUT_ADAPTER_AND_SOURCE_CHECK_PASS_GPU_NOT_RUN"
    for name, sha in tests["source_sha256"].items():
        assert digest(ROOT / name) == sha
    cfg = read(ROOT / "PHASE_A_CONFIG.json")
    cfg["documents"] = [x["document_id"] for x in selection["selected_documents"]]
    cfg["document_gpu_map"] = {x["document_id"]: x["gpu"] for x in selection["selected_documents"]}
    cfg.update(stage="B_CALIBRATION_ONLY_NO_32K_AUTHORIZATION", route="A", smoke_wall_cap_seconds=600,
               long_horizon_outcome_access_allowed=False, document_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json"))
    cfg_path = ROOT / "PHASE_B_CALIBRATION_CONFIG.json"
    publish(cfg_path, cfg)
    a = audit_a(ROOT, time.time())
    assert len(a["gpu_runs"]) == 4 and all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in a["gpu_runs"])
    formal_caps, cost_sources = {}, {}
    for doc in cfg["documents"]:
        gpu = cfg["document_gpu_map"][doc]
        old_doc = next(d for d, g in read(ROOT / "PHASE_A_CONFIG.json")["document_gpu_map"].items() if g == gpu)
        prior = ROOT / "runs" / f"formal_{old_doc}_gpu{gpu}.json"
        result = checked_json(prior)
        assert len(result["results"]) == 16 and result["valid"]
        loading, fp = result["model_loading_seconds"], result["fp_reference"]["elapsed_seconds"]
        slow = max(x["elapsed_seconds"] for x in result["results"])
        overhead = max(0, result["total_wall_seconds"] - loading - fp - sum(x["elapsed_seconds"] for x in result["results"]))
        formal_caps[doc] = 1.5 * (loading + fp + 16 * slow + overhead) + 20
        cost_sources[doc] = dict(path=str(prior), sha256=digest(prior), max_condition_seconds=slow,
            fp_seconds=fp, loading_seconds=loading, overhead_seconds=overhead)
    projected = a["total_gpu_hours_upper_bound"] + (sum(formal_caps.values()) + 1200) / 3600
    path = ROOT / "PHASE_B_SHORT_STAGE_RESERVATION.json"
    publish(path, dict(status="PASS" if projected <= 16 else "ENGINEERING_OR_BUDGET_STOP",
        epoch=time.time(), completed_phase_a_gpu_hours=a["total_gpu_hours_upper_bound"],
        phase_a_derived_formal_caps=formal_caps, cost_sources=cost_sources,
        new_two_gpu_smoke_reservation_seconds=1200, conservative_projected_total_gpu_hours=projected,
        margin=1.5, score_or_future_label_values_used=False, long_run_authorized=False))
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    sources = ["phase_b_calibration_common.py", "run_phase_b_calibration.py", "launch_phase_b_calibration.py",
               "budget_phase_b_calibration.py", "test_phase_b_calibration_cpu.py", "audit_phase_b_calibration.py",
               "freeze_phase_b_calibration.py", "PHASE_B_CALIBRATION_CONFIG.json", "PHASE_B_SHORT_STAGE_RESERVATION.json",
               "PHASE_B_CALIBRATION_CPU_TESTS.json", "PROSPECTIVE_DOCUMENT_SELECTION.json",
               "PHASE_B_DOCUMENT_SELECTION_RULE_FREEZE.json", "PRIOR_DOCUMENT_PATH_IDENTITY_AUDIT.json",
               "PREMEASUREMENT_CODE_FREEZE.json", "PREMEASUREMENT_ASSET_SUPPLEMENT.json", "SOURCE_IDENTITIES.json",
               "audit_phase_a_ledger_snapshot.py", "audit_phase_a_hardware_and_history.py"]
    dependencies = {}
    for name in ("PREMEASUREMENT_CODE_FREEZE.json", "PREMEASUREMENT_ASSET_SUPPLEMENT.json"):
        old = read(ROOT / name)
        dependencies.update(old.get("dependencies", {}))
        dependencies.update({str(ROOT / file): sha for file, sha in old["files"].items()})
    identities = read(ROOT / "SOURCE_IDENTITIES.json")
    dependencies.update({item["path"]: item["sha256"] for item in identities["rotations"].values()})
    model = identities["model_identity"]
    for filename, key in (("config.json", "config_sha256"), ("tokenizer.json", "tokenizer_sha256"),
                          ("tokenizer_config.json", "tokenizer_config_sha256"),
                          ("generation_config.json", "generation_config_sha256"),
                          ("model.safetensors.index.json", "weight_index_sha256")):
        dependencies[str(Path(model["path"]) / filename)] = model[key]
    base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "b_freeze_verified_base")
    for filename in ("run_ling_kda_query_readout_causal_closure_v1.py", "pytorch_kda_reference.py"):
        path = base.HIST_SOURCE / filename
        dependencies[str(path)] = digest(path)
    dependencies[str(route_path)] = digest(route_path)
    for entry in selection["selected_documents"]:
        for key in ("raw_text", "token_file"):
            dependencies[entry[key]["path"]] = entry[key]["sha256"]
    for path, sha in dependencies.items():
        assert digest(path) == sha, ("FREEZE_IDENTITY_MISMATCH", path)
    freeze_path = ROOT / "PHASE_B_CALIBRATION_CODE_FREEZE.json"
    publish(freeze_path, dict(status="CPU_QUALIFIED_FROZEN_GPU_SMOKE_PENDING", epoch=time.time(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        files={name: digest(ROOT / name) for name in sources}, dependencies=dependencies,
        unchanged_loss_science_window_roster=True, long_run_authorized=False,
        gpu_outcomes_generated=False, calibration_projection=projected))
    publish(freeze_path.with_suffix(".sha256.json"), dict(path=str(freeze_path), sha256=digest(freeze_path)))
    (ROOT / "runs_b_calibration").mkdir(exist_ok=True)
    print(json.dumps(dict(status="B_CALIBRATION_SOURCE_FROZEN_NOT_GPU_PASS", sha256=digest(freeze_path),
                         projected_total_gpu_hours=projected)))


if __name__ == "__main__":
    main()
