"""CPU-only append-only final stop package. No GPU retry or invented KL rows."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import verify as verify_cal, read, checked_json
from audit_phase_a_hardware_and_history import main as historical_guard
from persistent_stage_common_v1 import RUNS, FREEZE, LATCH, ledger

OUT = ROOT / "final_stop_20261011"


def text(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(value)
        if not value.endswith("\n"): f.write("\n")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cfg = verify_cal()
    historical_guard()
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    freeze = checked_json(FREEZE)
    for name, sha in freeze["files"].items(): assert digest(ROOT/name) == sha
    for name, sha in freeze["dependencies"].items(): assert digest(name) == sha
    account = ledger()
    assert len(account["persistent_runs"]) == 2
    assert all(x["mode"] == "qualification" and x["status"] == "FAIL_AND_STOP_NO_RETRY"
               and not x["process_alive"] for x in account["persistent_runs"])
    assert LATCH.exists() and not list(RUNS.glob("micro_*.LAUNCH_AUTHORIZATION.json"))
    assert not list(RUNS.glob("formal_*.LAUNCH_AUTHORIZATION.json"))
    stops, hardware = {}, {}
    for row in account["persistent_runs"]:
        p = RUNS / (row["tag"] + ".STOP.json"); stop = read(p)
        assert "assert torch.are_deterministic_algorithms_enabled()" in stop["traceback"]
        assert not (RUNS / (row["tag"] + ".json")).exists()
        log = RUNS / (row["tag"] + ".log")
        assert "persistent_progress" not in log.read_text()
        assert subprocess.run(["tmux", "has-session", "-t", row["session"]], capture_output=True).returncode != 0
        stops[row["document"]] = dict(path=str(p), sha256=digest(p), content=stop,
            log_sha256=digest(log), no_canonical=True, measured_stream_tokens=0, process_alive=False)
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    for gpu in (3, 6):
        row = next([x.strip() for x in line.split(",")] for line in gpu_text.splitlines() if int(line.split(",")[0]) == gpu)
        assert row[1] == cfg["gpu_uuids"][str(gpu)] and row[2] == cfg["runtime"]["driver"]
        held = subprocess.run(["fuser", f"/dev/nvidia{gpu}"], capture_output=True, text=True)
        assert held.returncode in (0, 1)
        hardware[str(gpu)] = dict(physical_row=row, compute_apps=[x for x in apps.splitlines() if row[1] in x],
            device_holders=held.stdout.strip(), no_own_process_remains=True, no_signals=True)
    # Preserve a separate release-observation bound, never call it GPU work.
    # First read-only /proc=false ledger observation in this heartbeat's output.
    release_observed_epoch = 1791702553.8878078
    prior = account["prior_a_and_b_calibration"]["total_gpu_hours_upper_bound"]
    release_upper = prior + sum((release_observed_epoch - x["launch_epoch"])/3600 for x in account["persistent_runs"])
    assert release_upper >= account["total_gpu_hours_upper_bound"] and release_upper < 16
    account.update(status="FINAL_ENGINEERING_STOP_NO_ACTIVE_RUNS", final=True,
        receipt_to_stop_gpu_hours=account["total_gpu_hours_upper_bound"],
        first_verified_process_release_observation_epoch=release_observed_epoch,
        conservative_device_release_gpu_hours_upper_bound=release_upper,
        release_bound_is_accounting_uncertainty_not_extra_measured_gpu_work=True,
        persistent_measured_tokens=0, micro_units=0, formal_32k_units=0)
    OUT.mkdir()
    copied = ["EXPERIMENT_PLAN.md", "L7_OBJECTIVE_IDENTITY_AND_PARITY.md", "MATCHED_SHORT_HORIZON_SCORES.csv",
              "P1_L7_REDUNDANCY_ANALYSIS.md", "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json",
              "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.sha256.json"]
    for name in copied:
        data = (ROOT/name).read_bytes()
        with (OUT/name).open("xb") as f: f.write(data)
        assert digest(OUT/name) == digest(ROOT/name)
    publish(OUT/"GPU_COST_LEDGER.json", account)
    publish((OUT/"GPU_COST_LEDGER.json").with_suffix(".sha256.json"),
            dict(path=str(OUT/"GPU_COST_LEDGER.json"), sha256=digest(OUT/"GPU_COST_LEDGER.json")))
    evidence = dict(status="FINAL_READ_ONLY_AUDIT_PASS_WITH_RETAINED_PERSISTENT_RUNTIME_STOP",
        epoch=time.time(), frozen_persistent_code_and_asset_hashes_unchanged=True,
        historical_canonical_and_sidecar_files_rehashed=160, historical_attempt04_05_stops_unchanged=True,
        model_32_weight_freshness_metadata_pass=True, old_initial_zero_ledger_unchanged=True,
        old_initial_zero_ledger_sha256=digest(ROOT/"GPU_COST_LEDGER.json"),
        persistent_stops=stops, hardware=hardware, storage_free_bytes=shutil.disk_usage(ROOT).free,
        active_own_processes=[], micro_launched=False, formal_32k_launched=False,
        no_gpu_retry_or_signals_or_other_experiment_outcome_access=True,
        final_ledger_sha256=digest(OUT/"GPU_COST_LEDGER.json"), source_code_sha256=digest(__file__))
    publish(OUT/"FINAL_STOP_AUDIT.json", evidence)
    decision = dict(experiment=cfg["experiment"], decision="ENGINEERING_OR_BUDGET_STOP",
        reason="NEW_PERSISTENT_WORKER_GLOBAL_DETERMINISTIC_ALGORITHMS_GUARD_FALSE_AFTER_LOADING",
        stop_category="ENGINEERING_RUNTIME_NOT_SCIENTIFIC_HYPOTHESIS_FAILURE", budget_exhausted=False,
        task_formally_stopped=True, automatic_gpu_retry_authorized=False, phase_a="COMPLETE",
        route="A", independent_calibration_documents=2, both_pairs_frozen=True,
        persistent_gpu_qualification="FAIL_BEFORE_FIRST_STREAM_TOKEN", microbenchmark="NOT_EXECUTED",
        formal_32k_units=0, prospective_actual_winners_available=False,
        P1_L7_RANKING_REDUNDANCY="HIGH_OVERLAP_WITH_INFORMATIVE_NONIDENTICAL_RANKINGS",
        P1_INCREMENTAL_INFORMATION_OVER_L7="POSITIVE_EXPLORATORY_RESIDUAL_RANK_ASSOCIATION_PERSISTENT_UNCONFIRMED",
        new_loss_needed="NOT_ESTABLISHED", P1_training_superiority="NOT_ESTABLISHED",
        Q2_persistent_prediction="UNRESOLVED", Q3_optimization_explanation="UNRESOLVED",
        trainability_audit="DEFERRED_NO_PERSISTENT_SUPPORT", temporal_experiment="PROPOSAL_ONLY_NOT_EXECUTED",
        next_steps_proposal=["Audit applicability and setup order of global PyTorch determinism without weakening evidence gates",
            "Any GPU recovery requires new explicit authorization and new isolated freeze preserving this STOP",
            "Retain the original document/pairs/winners/metric; do not replace with outcome-selected contrasts",
            "No conclusion favoring a new P1 loss over L7 until qualified complete persistent validation"],
        gpu_cost_ledger_sha256=digest(OUT/"GPU_COST_LEDGER.json"), stop_evidence_sha256=digest(OUT/"FINAL_STOP_AUDIT.json"),
        pair_selection_sha256=digest(ROOT/"PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
        persistent_measurement_code_commit=freeze["code_commit"], phase_a_code_commit="a74255b1b173d27a3e51f16dbc4cb26b46104d08",
        calibration_code_commit="c7131f3f144846ab9d3f0558210493b37a896097")
    publish(ROOT/"NEXT_RESEARCH_DECISION.json", decision)
    publish(OUT/"NEXT_RESEARCH_DECISION.json", decision)
    correct = f"""# Persistent QDQ correctness — qualification STOP

Final status: ENGINEERING_OR_BUDGET_STOP. New persistent GPU qualification:
FAIL_BEFORE_FIRST_STREAM_TOKEN on both authorized UUIDs. This is an engineering
runtime guard failure, not a scientific test rejecting P1 or L7.

The committed worker `{freeze['code_commit']}` loads the original Capture/model,
then requires `torch.are_deterministic_algorithms_enabled()` at line63. It was
false on both GPUs. The worker did not explicitly enable that global mode before
this assertion. The inherited qualified-runtime helper applies fixed FLA
selectors; its name alone does not establish this global PyTorch flag.
This newly added assertion is stricter than a fixed seed/selector. No repeat
experiment actually ran, so this failure must not be described as measured
repeat nondeterminism or used to invalidate previously executed short parity.

Both models loaded; their measured streaming token count is zero. No new FP
reference/logit trace, real persistent QDQ counter, hybrid-cache forward inventory,
post8192 consumption, repeat equality, or 32K KL exists. CPU synthetic cache/QDQ
checks and six original-scorer arithmetic checks passed, but are not GPU PASS.
The prior short all16 calibration remains mechanically PASS on both documents.

860 STOP SHA256: `{stops['pg19-test-860']['sha256']}`.
5956 STOP SHA256: `{stops['pg19-test-5956']['sha256']}`.
Runner SHA256: `{digest(ROOT/'run_persistent_stage_v1.py')}`.
Freeze SHA256: `{digest(FREEZE)}`. The canonical STOPs and logs remain unchanged.
Both PIDs and tmux sessions exited naturally. No signal/kill/migration/retry.
The shared fail-closed latch prevents qualification, micro and formal re-entry.

Microbenchmark and complete four-unit cost/VRAM gate were NOT_EXECUTED, not FAIL
or PASS. The four formal units were NOT_EXECUTED. Budget was not exhausted.
All original science, rotations, quantization and historical files are retained.
"""
    text(ROOT/"PERSISTENT_QDQ_CORRECTNESS_REPORT.md", correct)
    text(OUT/"PERSISTENT_QDQ_CORRECTNESS_REPORT.md", correct)
    comparison = """# Prospective objective comparison — no long outcomes

Final decision: ENGINEERING_OR_BUDGET_STOP. Route A remains frozen.

| Independent document | GPU | P1 predicted winner | Original L7_TOTAL predicted winner | P1/L7 midrank gaps |
| --- | --- | --- | --- | --- |
| pg19-test-860 | 6 | H_to_L6_a50 | H_to_L7_a25 | 8 / 5 |
| pg19-test-5956 | 3 | H_to_L6_a50 | H_to_L7_a50 | 6 / 5 |

B0 and P3 also predict H_to_L6_a50 on both pairs. Pairs were selected from all16
with opposite ranks, both gaps>=3, descending minimum gap, scientific-key
tie-break, before new long outcomes. Document seed/hash order and exclusion audit
are retained. No rule relaxation, document replacement, or comparator switch.

Actual 32K winners, signed primary gaps, primary/secondary KL, AUC, p50/p99:
UNAVAILABLE because new persistent qualification stopped before token0. No
placeholder scientific rows or PERSISTENT_FUTURE_KL_RESULTS.csv are created.

Q1: P1 and original L7 have high but nonidentical natural-rotation rankings:
aggregate Spearman0.900000/0.852941 on37702/5396. Full per-window correlations,
concordance, ranks and leave-one-rotation-out are in the actual Phase A analysis.
Old-label partial associations are positive but DISCOVERY_ONLY, four anchors from
two documents, no-further-QDQ continuation and a different NVIDIA driver.
PERSISTENT_INCREMENTAL_VALUE_UNCONFIRMED. No gradient-equivalence claim.

Q2 remains unanswered. We cannot select P1_OUTPERFORMS_L7, L7_OUTPERFORMS_P1,
or a numerical MIXED outcome. Q3 remains hypothetical: STE/forward mismatch,
H128 truncation, state regularization, data shift, insufficient optimization,
or objective/reasoning mismatch are possible explanations, not findings.
No backward, optimizer.step, training, new rotation, AIME or free generation.

Phase C is only a conditional proposal: source-error-preserved versus removed
branches could both continue genuine QDQ through H32/H128 from a shared state.
It must differ from old pulse/no-further-QDQ and factorial work. Because Q2 is
unresolved, this task does not establish that temporal work should replace L7
or P1 validation. No temporal experiment has been executed or authorized here.
"""
    text(ROOT/"PROSPECTIVE_OBJECTIVE_COMPARISON.md", comparison)
    text(OUT/"PROSPECTIVE_OBJECTIVE_COMPARISON.md", comparison)
    repro = f"""# Reproducibility and final delivered stage

Final scientific decision: ENGINEERING_OR_BUDGET_STOP. Phase A and independent
Phase B calibration/selection are complete. New persistent qualification stopped;
microbenchmark and formal32K are unexecuted. This package is not a successful
full-task persistent validation. Root initial0 ledger remains frozen; use this
directory's GPU_COST_LEDGER.json as the all-stage actual receipt/STOP ledger.

Receipt-to-completion/STOP device-time sum: {account['total_gpu_hours_upper_bound']:.15f}
GPU-hours, including both devices, loading, FP, all smokes/calibration and failed
qualification startups. Waiting, other experiments and CPU are excluded. A separate
first-verified process-release upper bound is {release_upper:.15f} GPU-hours;
its safety gap is not additional measured work. Both bounds are below16.

Code commits: PhaseA a74255b1b173d27a3e51f16dbc4cb26b46104d08;
CPU analysis fb2c838a01bcbc09d2c2aef9ad4e166bb9c42c48;
calibration c7131f3f144846ab9d3f0558210493b37a896097;
pair/audit 288bdc0c6b98098d5710c10cff41980e82148bed;
persistent frozen {freeze['code_commit']}.
No measurement code changed after GPU STARTED; failed pre-GPU versions retained.

Model: Ling-3.0-tiny. Config SHA9750d847957913f665a13c0b5a6537199e33c6f3ec970d9fcb55a0e5076d4012;
tokenizer40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60;
32 weight shards rehashed before measurement, unchanged stat freshness finalPASS;
weight manifest d8a7cf059fd4b4f2fd7f7d0b1118417be02f7b39a5c428461c027f5d71faff6d.
H872d4e45d21b03ccf8c3c11421331a07684f3b11ec6e5523ce7ced9254a44412;
L6 51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964;
L7 fdf4863869066ca5d42e9e64cb44a3ab7061b17e24ac34487d8ef42073df7ce3.
Selected rotation, raw-text/token, calibration canonical+sidecar, real FP tensor
identities and GPU UUIDs are fully retained in PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json,
SOURCE_IDENTITIES.json, PHASE_B_CALIBRATION_MATCHED_TABLES.json and the code/assets
freeze. No new persistent FP identity exists. Fresh short reference identities
are not relabeled as32K FP references.

Actual runtime: {json.dumps(cfg['runtime'], sort_keys=True)}.
GPU6 GPU-de53f243-b271-521c-d875-d96611ed8574;
GPU3 GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae. No other GPU used.
Final readonly audit rehashed all80 old canonicals and80 sidecars, preserved both
Attempt04/05 STOPs, verified all frozen sources/assets and current process exit.

MATCHED_SHORT_HORIZON_SCORES.csv and the original PhaseA reports are copied
byte-identically. L7 objective report retains its earlier stage timestamp/text;
PhaseA analysis and this final chronology supersede its then-running status.
The original loss definition and matched-window deployment application are
established; historical512-token training sampling/BPTT optimization is not replayed.
Full ranking/residual/LOO tables remain in the independent experiment root.

The eleven requested filenames are indexed in DELIVERABLE_MANIFEST.json with
actual stage status and hashes. PERSISTENT_FUTURE_KL_RESULTS.csv is intentionally
NOT_PRODUCED_UNEXECUTED rather than populated with fabricated/empty long results.
All emitted result/report hashes are in RESULT_AND_DELIVERY_HASHES.json and
SHA256SUMS.txt. No old evidence or initial ledger was overwritten.
"""
    text(ROOT/"REPRODUCIBILITY_REPORT.md", repro); text(OUT/"REPRODUCIBILITY_REPORT.md", repro)
    names = ["EXPERIMENT_PLAN.md", "L7_OBJECTIVE_IDENTITY_AND_PARITY.md", "MATCHED_SHORT_HORIZON_SCORES.csv",
        "P1_L7_REDUNDANCY_ANALYSIS.md", "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json", "PERSISTENT_QDQ_CORRECTNESS_REPORT.md",
        "PERSISTENT_FUTURE_KL_RESULTS.csv", "PROSPECTIVE_OBJECTIVE_COMPARISON.md", "GPU_COST_LEDGER.json",
        "NEXT_RESEARCH_DECISION.json", "REPRODUCIBILITY_REPORT.md"]
    statuses = {"PERSISTENT_QDQ_CORRECTNESS_REPORT.md":"ACTUAL_RUNTIME_QUALIFICATION_FAILURE_REPORT",
        "PERSISTENT_FUTURE_KL_RESULTS.csv":"NOT_PRODUCED_UNEXECUTED",
        "PROSPECTIVE_OBJECTIVE_COMPARISON.md":"PREDICTIONS_COMPLETE_ACTUAL_LONG_COMPARISON_UNEXECUTED"}
    manifest = dict(decision=decision["decision"], complete_persistent_validation=False,
        deliverables=[dict(name=n, status=statuses.get(n,"PRODUCED_ACTUAL_COMPLETED_STAGE"),
            sha256=digest(OUT/n) if (OUT/n).exists() else None,
            reason="No32K scientific trajectory exists; do not fabricate rows" if n.endswith("FUTURE_KL_RESULTS.csv") else "") for n in names])
    publish(OUT/"DELIVERABLE_MANIFEST.json", manifest)
    result_paths = [p for p in OUT.iterdir() if p.is_file()]
    result_paths += [LATCH, FREEZE, ROOT/"phase_a_analysis_20261011/PHASE_A_ANALYSIS.json",
                    ROOT/"phase_a_analysis_20261011/PHASE_A_ROUTE.json", ROOT/"PHASE_B_CALIBRATION_MATCHED_TABLES.json"]
    result_paths += list(RUNS.glob("*.*"))
    hashes = {str(p):digest(p) for p in sorted(result_paths) if p.is_file() and p.name != "authorization.lock"}
    publish(OUT/"RESULT_AND_DELIVERY_HASHES.json", hashes)
    text(OUT/"SHA256SUMS.txt", "\n".join(f"{sha}  {name}" for name,sha in hashes.items()))
    for n in ("NEXT_RESEARCH_DECISION.json", "PERSISTENT_QDQ_CORRECTNESS_REPORT.md", "PROSPECTIVE_OBJECTIVE_COMPARISON.md", "REPRODUCIBILITY_REPORT.md"):
        publish((ROOT/n).with_suffix(".sha256.json"), dict(path=str(ROOT/n), sha256=digest(ROOT/n)))
    print(json.dumps(dict(status="FINAL_ENGINEERING_STOP_DELIVERED", output=str(OUT),
        decision_sha256=digest(OUT/"NEXT_RESEARCH_DECISION.json"),
        manifest_sha256=digest(OUT/"DELIVERABLE_MANIFEST.json"), actual_gpu_hours=account["total_gpu_hours_upper_bound"],
        release_upper_gpu_hours=release_upper, active_processes=[], formal_32k_units=0,
        no_fabricated_persistent_result_csv=True)))


if __name__ == "__main__":
    main()
