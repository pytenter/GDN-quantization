"""Exclusive pre-GPU code/assets/input/prediction/protocol publication."""
import json
import os
import subprocess
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import verify, audit_b, checked_json, read


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cfg = verify(); cost = audit_b()
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in cost["phase_b_calibration_runs"])
    for file, status in (("PERSISTENT_ENGINE_CPU_TESTS_V1.json", "CPU_SYNTHETIC_ENGINE_TESTS_PASS_GPU_NOT_QUALIFIED"),
                         ("PERSISTENT_SOURCE_CPU_AUDIT_V3.json", "CPU_SOURCE_AND_ARITHMETIC_AUDIT_PASS_REAL_GPU_PENDING"),
                         ("PERSISTENT_STAGE_CPU_TESTS_V1.json", "CPU_PROTOCOL_AND_COST_TESTS_PASS_GPU_PENDING")):
        assert checked_json(ROOT / file)["status"] == status
    assert not (ROOT / "runs_b_persistent").exists(), "PRE_GPU_FREEZE_MUST_PRECEDE_LAUNCH"
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    assert selection["status"] == "BOTH_DOCUMENTS_DISCRIMINATIVE_PAIRS_FROZEN"
    source = checked_json(ROOT / "PERSISTENT_SOURCE_CPU_AUDIT_V3.json")
    names = ["persistent_engine_v1.py", "persistent_stage_common_v1.py", "run_persistent_stage_v1.py",
             "launch_persistent_stage_v1.py", "audit_persistent_stage_v1.py", "budget_persistent_complete_v1.py",
             "test_persistent_engine_cpu_v1.py", "test_persistent_stage_cpu_v1.py", "audit_persistent_source_cpu_v1.py",
             "audit_persistent_source_cpu_v2.py", "audit_persistent_source_cpu_v3.py", "freeze_persistent_stage_v1.py",
             "PERSISTENT_EXECUTION_PROTOCOL_V1.md"]
    files = {name:digest(ROOT/name) for name in names}
    dependencies = dict(source["source_identities"])
    for name in ("PERSISTENT_ENGINE_CPU_TESTS_V1.json", "PERSISTENT_SOURCE_CPU_AUDIT_V3.json",
                 "PERSISTENT_STAGE_CPU_TESTS_V1.json", "PERSISTENT_SOURCE_CPU_ENGINEERING_FAILURE_V1.json",
                 "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json", "PHASE_B_CALIBRATION_MATCHED_TABLES.json",
                 "SOURCE_IDENTITIES.json", "FRESH_MODEL_WEIGHT_AUDIT.json", "MODEL_WEIGHT_SHA256.txt",
                 "PHASE_B_CALIBRATION_CODE_FREEZE.json", "PHASE_B_CALIBRATION_CONFIG.json",
                 "PROSPECTIVE_DOCUMENT_SELECTION.json", "run_phase_b_calibration_v2.py",
                 "phase_b_calibration_common_v2.py", "run_matched_l7.py", "objective.py",
                 "audit_phase_a_ledger_snapshot.py"):
        dependencies[str(ROOT/name)] = digest(ROOT/name)
    for path, sha in read(ROOT / "PHASE_B_CALIBRATION_CODE_FREEZE.json")["dependencies"].items():
        assert digest(path) == sha; dependencies[path] = sha
    input_selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    for entry in input_selection["selected_documents"]:
        for key in ("token_file", "raw_text"):
            assert digest(entry[key]["path"]) == entry[key]["sha256"]
            dependencies[entry[key]["path"]] = entry[key]["sha256"]
    rotations = read(ROOT / "SOURCE_IDENTITIES.json")["rotations"]
    for entry in selection["documents"]:
        for condition in (entry["pair"]["condition_a"], entry["pair"]["condition_b"]):
            item = rotations[condition]; assert digest(item["path"]) == item["sha256"]
            dependencies[item["path"]] = item["sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in names:
        tracked = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        import hashlib
        assert hashlib.sha256(tracked).hexdigest() == files[name], ("UNCOMMITTED_PERSISTENT_CODE", name)
    value = dict(status="CPU_FROZEN_REAL_PERSISTENT_GPU_QUALIFICATION_PENDING", epoch=time.time(),
        code_commit=head, files=files, dependencies=dependencies,
        pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
        predictions_and_primary_precede_first_new_long_outcome=True,
        completed_gpu_hours_upper_bound=cost["total_gpu_hours_upper_bound"],
        qualification_endpoint=512, qualification_repeats=2, micro_endpoint=8448,
        formal_endpoint=32768, formal_primary=[8192,32768], formal_secondary=[2048,32768],
        same_stream_code_all_modes=True, persistent_every_token_qdq=True,
        quantizer_and_model_step_unchanged=True, original_loss_math_unchanged=True,
        qualification_and_micro_not_formal_launch_authority=True, max_formal_units=4,
        max_entire_task_gpu_hours=16, no_new_rotation_or_training=True,
        complete_cost_margin=1.5, remaining_attention_growth_envelope=32768/8448,
        capacity_limit_fraction=.85, scientific_outcomes_used_to_change_code_or_rules=False)
    path = ROOT / "PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json"
    publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status=value["status"], code_commit=head, sha256=digest(path), long_authorized=False)))


if __name__ == "__main__":
    main()
