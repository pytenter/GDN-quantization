#!/usr/bin/env python3
"""CPU-only finalizer admission/report tests with synthetic filesystem receipts.

No model, CUDA, SSH, controller, worker, CPU score subprocess or real finisher
is launched. This verifies finalizer binding/rejection logic, not GPU evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.root.resolve()
    module = load(source / "scripts/resident_finalize_offline.py", "_resident_finisher_cpu_test")
    analysis = load(source / "src/resident_analysis.py", "_resident_analysis_cpu_test")
    checks = []
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    task = "CPU_ONLY_SYNTHETIC_TASK"

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))

    def read(path):
        return json.loads(path.read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def check(name, result):
        if not result:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, function):
        try:
            function()
        except (RuntimeError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)

    qual = read(source / "QUALIFICATION_JOB_MANIFEST.json")["units"]
    formal = read(source / "FORMAL_JOB_MANIFEST.json")["jobs"]

    def fixture(root, complete):
        root.mkdir()
        (root / "logs").mkdir()
        write(root / "SOURCE_FREEZE.json", {"files": [{"path": "synthetic_fixture_only"}]})
        write(root / "GPU_PERMIT.json", {"fixture": "qualification"})
        write(root / "FORMAL_GPU_PERMIT.json", {"fixture": "formal"})
        performance = dict(status="PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED", scope="MOCK_CPU_TEST_ONLY_NOT_ACTUAL_QUALIFICATION")
        write(root / "PERFORMANCE_ADMISSION.json", performance)
        write(root / "AUTHORIZATION_FORMAL.json", dict(fixture="all58", source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"),
              performance_admission_sha256=sha(root / "PERFORMANCE_ADMISSION.json")))
        write(root / "QUALIFICATION_JOB_MANIFEST.json", {"units": qual})
        write(root / "FORMAL_JOB_MANIFEST.json", {"jobs": formal})
        controller = dict(pid=100, starttick=1000, cmdline=["python", str(root / "scripts/resident_controller.py"), "--root", str(root), "--execute"])
        write(root / "CONTROLLER_STARTED.json", dict(task_id=task, execution_attempt=root.name, identity=controller))
        write(root / "FINALIZER_CHAIN_STARTED.json", dict(controller_process_really_returned=True, controller_returncode=0 if complete else 1))
        failure = None if complete else dict(status="FAIL_CLOSED_STOP", error="SYNTHETIC_CPU_TEST_NOT_REAL_GPU")
        if failure:
            write(root / "FIRST_FAILURE.json", failure)
        phase_receipts = {"qualification": [], "formal": []}
        interval_rows, identities = [], []
        units = [("qualification", unit, True) for unit in qual] + [("formal", job, True) for job in formal] if complete else [("qualification", qual[0], False)]
        for index, (phase, unit, legal) in enumerate(units):
            directory = root / phase / unit["id"]
            directory.mkdir(parents=True)
            script = "qualify_gpu.py" if phase == "qualification" else "resident_formal_worker.py"
            identity = dict(pid=1000 + index, starttick=2000 + index,
                            cmdline=["python", str(root / "scripts" / script), "--root", str(root), "--unit", unit["id"], "--gpu", str(unit["gpu"])])
            permit = "GPU_PERMIT.json" if phase == "qualification" else "FORMAL_GPU_PERMIT.json"
            plan = "QUALIFICATION_JOB_MANIFEST.json" if phase == "qualification" else "FORMAL_JOB_MANIFEST.json"
            dispatch = dict(task_id=task, execution_attempt=root.name, phase=phase, unit=unit["id"], unit_record=unit,
                            controller_identity=controller, gpu=unit["gpu"], gpu_uuid=unit["gpu_uuid"], source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"),
                            permit_sha256=sha(root / permit), base_gpu_permit_sha256=sha(root / "GPU_PERMIT.json"), plan_sha256=sha(root / plan), dispatch_monotonic=1.)
            if phase == "formal":
                dispatch["formal_authorization_sha256"] = sha(root / "AUTHORIZATION_FORMAL.json")
            write(root / "dispatch/qualification" / (unit["id"] + ".json") if phase == "qualification" else directory / "DISPATCH.json", dispatch)
            interval = dict(unit=unit["id"], phase=phase, gpu=unit["gpu"], gpu_uuid=unit["gpu_uuid"], identity=identity,
                            controller_identity=controller, dispatch_monotonic_lower=1., identity_monotonic_upper=2.,
                            exit_first_observed_monotonic=3., release_first_observed_monotonic=4., release="PASS", exit_returncode=0 if legal else 1,
                            source_freeze_sha256=dispatch["source_freeze_sha256"], permit_sha256=dispatch["permit_sha256"], monotonic_same_host=True)
            write(directory / "RESOURCE_INTERVAL.json", interval)
            interval_rows.append(interval); identities.append(identity)
            if legal:
                write(directory / "STARTED.json", {"identity": identity})
                write(directory / "RESULT.json", dict(status="PASS", result=dict(synthetic=True, wall_seconds=1.)))
                write(directory / "CONTROLLER_VALIDATION.json", {"status": "CPU_TEST_PLACEHOLDER_NOT_REAL_VALIDATION"})
                phase_receipts[phase].append(dict(unit=unit["id"], result_sha256=sha(directory / "RESULT.json"), resource_interval_sha256=sha(directory / "RESOURCE_INTERVAL.json"), validation_sha256=sha(directory / "CONTROLLER_VALIDATION.json")))
        session = dict(task_id=task, execution_attempt=root.name, status=module.SUCCESS if complete else "FAIL_CLOSED_STOP",
                       controller_identity=controller, own_release_all_PASS=True, all_own_processes_released=True, unresolved_launches=[],
                       resource_intervals=len(units), attempted_units=len(units), qualification_completed=len(phase_receipts["qualification"]), formal_completed=len(phase_receipts["formal"]),
                       qualification_receipts=phase_receipts["qualification"], formal_receipts=phase_receipts["formal"], first_failure=failure, reference_pair_retained=None)
        write(root / "SESSION_RESULT.json", session)
        write(root / "FORMAL_GENERATIONS_COMPLETE.json", dict(task_id=task, execution_attempt=root.name, planned=60, count=session["formal_completed"], formal_receipts=session["formal_receipts"],
              source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"), all_own_processes_released=True, status="COMPLETE_60_60" if complete else "PARTIAL_FAIL_CLOSED_STOP"))
        calls, actual, live, apps = [], {}, [], []
        control = types.SimpleNamespace(TASK=task, ROOT=root, read=read, sha=sha,
             verify_sources=lambda r: calls.append("sources") or sha(root / "SOURCE_FREEZE.json"),
             verify_model_and_external=lambda r, fresh_external: calls.append(("fresh61deps17stats", fresh_external)),
             same_process=lambda identity: identity in live, process_identity=lambda pid: actual.get(pid),
             query_gpu=lambda: dict(apps=apps),
             validate_qualification_result=lambda r, unit: calls.append(("qual", unit["id"])) or read(r / "qualification" / unit["id"] / "RESULT.json"),
             validate_formal_result=lambda r, unit: calls.append(("formal", unit["id"])) or read(r / "formal" / unit["id"] / "RESULT.json"),
             performance_audit=lambda r, units, wrappers: calls.append(("performance_actual_wrapper_count", len(wrappers))) or performance,
             atomic_new=lambda path, value: write(path, value))
        return control, session, interval_rows, identities, calls, actual, live, apps

    try:
        with tempfile.TemporaryDirectory(prefix="resident_finisher_CPU_ONLY_") as temporary:
            root = Path(temporary) / "stop"
            c, session, intervals, identities, calls, actual, live, apps = fixture(root, False)
            admitted = module.admission(c, root)
            check("real_nonzero_exit_released_STOP_not_complete", admitted["complete60"] is False and admitted["session"]["formal_completed"] == 0)
            check("honest_preformal_STOP_no_PERF_requirement", admitted["performance_admission_final_verification"]["status"] == "NOT_REQUIRED_NO_FORMAL_DISPATCH")
            check("fresh_source61deps17stat_called", calls[:2] == ["sources", ("fresh61deps17stats", True)])
            check("failed_worker_without_STARTED_allowed_only_as_failed_interval", len(admitted["actual_intervals"]) == 1 and not (root / "qualification" / qual[0]["id"] / "STARTED.json").exists())
            resource = module.resource_summary(intervals, [])
            check("STOP_physical_GPUh_UNRESOLVED_not_zero", resource["EXACT_physical_GPUh"] == "UNRESOLVED_NOT_KERNEL_BUSY_TIME" and resource["reservation_GPUh_upper"] == 3. / 3600.)
            module.scientific_report(c, root, admitted, None)
            text = (root / "FINAL_SCIENTIFIC_REPORT.md").read_text()
            check("STOP_report_actual_count_no_accuracy", "Legal formal completions: 0/60" in text and "|Condition|" not in text)
            check("STOP_report_does_not_fabricate_score_CSV", not (root / "AIME26_RESIDENT_PER_PROBLEM.csv").exists() and not (root / "FROZEN_V4_SCORES.json").exists())
            check("STOP_report_no_inherited_FP_PASS", read(root / "FP_ROTATION_NUMERICAL_DIAGNOSTICS.json")["status"] == "NOT_RUN_OR_NOT_LEGALLY_PASSED")
            def mutate_reject(name, path, key, value):
                original = read(path)
                changed = copy.deepcopy(original); changed[key] = value
                write(path, changed)
                try:
                    reject(name, lambda: module.admission(c, root))
                finally:
                    write(path, original)
            mutate_reject("wrong_task_rejected", root / "SESSION_RESULT.json", "task_id", "other")
            mutate_reject("wrong_execution_slug_rejected", root / "SESSION_RESULT.json", "execution_attempt", "other")
            mutate_reject("live_controller_rejected", root / "FINALIZER_CHAIN_STARTED.json", "controller_process_really_returned", False)
            mutate_reject("boolean_returncode_not_integer_exit", root / "FINALIZER_CHAIN_STARTED.json", "controller_returncode", True)
            mutate_reject("unreleased_session_rejected", root / "SESSION_RESULT.json", "all_own_processes_released", False)
            mutate_reject("unresolved_launch_rejected", root / "SESSION_RESULT.json", "unresolved_launches", [{"pid": 42}])
            mutate_reject("interval_count_mismatch_rejected", root / "SESSION_RESULT.json", "resource_intervals", 2)
            mutate_reject("attempt_count_underreported_rejected", root / "SESSION_RESULT.json", "attempted_units", 0)
            mutate_reject("first_failure_history_mismatch_rejected", root / "FIRST_FAILURE.json", "error", "changed")
            target = root / "qualification" / qual[0]["id"]
            mutate_reject("monotonic_order_rejected", target / "RESOURCE_INTERVAL.json", "identity_monotonic_upper", 5.)
            mutate_reject("wrong_UUID_rejected", target / "RESOURCE_INTERVAL.json", "gpu_uuid", "foreign")
            mutate_reject("wrong_dispatch_freeze_rejected", root / "dispatch/qualification" / (qual[0]["id"] + ".json"), "source_freeze_sha256", "0" * 64)
            live.append(session["controller_identity"])
            reject("exact_controller_still_alive_rejected", lambda: module.admission(c, root)); live.clear()
            live.append(identities[0])
            reject("exact_worker_still_alive_rejected", lambda: module.admission(c, root)); live.clear()
            apps.append({"pid": 999999})
            actual[999999] = dict(pid=999999, starttick=999999, cmdline=["foreign_authorized_job"])
            check("foreign_NVML_allowed", module.admission(c, root)["complete60"] is False)
            apps.clear(); actual.clear()
            apps.append({"pid": identities[0]["pid"]})
            reject("unresolved_own_NVML_PID_rejected", lambda: module.admission(c, root))
            actual[identities[0]["pid"]] = dict(pid=identities[0]["pid"], starttick=999999, cmdline=["foreign_reused_pid"])
            check("PID_reused_distinct_foreign_identity_allowed", module.admission(c, root)["complete60"] is False)
            actual[identities[0]["pid"]] = identities[0]
            reject("own_NVML_actual_identity_rejected", lambda: module.admission(c, root))
            apps.clear(); actual.clear()
            fullroot = Path(temporary) / "complete"
            full, sess, rows, _identities, _calls, _actual, _live, _apps = fixture(fullroot, True)
            passed = module.admission(full, fullroot)
            check("complete_58_60_all_118_intervals_admitted", passed["complete60"] is True and len(passed["actual_intervals"]) == 118)
            check("final_PERF_recomputed_all58_actual_wrappers", passed["performance_admission_final_verification"]["actual_qualification_wrapper_count"] == 58 and ("performance_actual_wrapper_count", 58) in _calls)
            check("all_58_qualification_60_formal_validators_called", sum(isinstance(x, tuple) and x[0] == "qual" for x in _calls) == 58 and sum(isinstance(x, tuple) and x[0] == "formal" for x in _calls) == 60)
            perf_original = read(fullroot / "PERFORMANCE_ADMISSION.json")
            write(fullroot / "PERFORMANCE_ADMISSION.json", dict(perf_original, tampered=True))
            reject("final_PERF_SHA_tamper_rejected", lambda: module.admission(full, fullroot))
            auth_original = read(fullroot / "AUTHORIZATION_FORMAL.json")
            write(fullroot / "AUTHORIZATION_FORMAL.json", dict(auth_original, performance_admission_sha256=sha(fullroot / "PERFORMANCE_ADMISSION.json")))
            # Bind the synthetic formal dispatch SHA to this deliberately
            # altered authorization so this negative reaches recomputation.
            saved_dispatches = []
            for job in formal:
                p = fullroot / "formal" / job["id"] / "DISPATCH.json"
                d = read(p); saved_dispatches.append((p, d))
                write(p, dict(d, formal_authorization_sha256=sha(fullroot / "AUTHORIZATION_FORMAL.json")))
            reject("final_PERF_recomputed_verdict_tamper_rejected", lambda: module.admission(full, fullroot))
            write(fullroot / "AUTHORIZATION_FORMAL.json", auth_original)
            write(fullroot / "PERFORMANCE_ADMISSION.json", perf_original)
            for p, d in saved_dispatches:
                write(p, d)
            original = read(fullroot / "FINALIZER_CHAIN_STARTED.json")
            write(fullroot / "FINALIZER_CHAIN_STARTED.json", dict(original, controller_returncode=1))
            reject("complete_with_nonzero_controller_exit_rejected", lambda: module.admission(full, fullroot))
            write(fullroot / "FINALIZER_CHAIN_STARTED.json", original)
            original = read(fullroot / "SESSION_RESULT.json")
            write(fullroot / "SESSION_RESULT.json", dict(original, qualification_receipts=original["qualification_receipts"][:-1], qualification_completed=57))
            reject("formal_without_all58_qualification_rejected", lambda: module.admission(full, fullroot))
            write(fullroot / "SESSION_RESULT.json", original)
            original_generation = read(fullroot / "FORMAL_GENERATIONS_COMPLETE.json")
            write(fullroot / "FORMAL_GENERATIONS_COMPLETE.json", dict(original_generation, formal_receipts=[]))
            reject("generation_legal_receipts_mismatch_rejected", lambda: module.admission(full, fullroot))
            write(fullroot / "FORMAL_GENERATIONS_COMPLETE.json", original_generation)
            zero_session = copy.deepcopy(read(fullroot / "SESSION_RESULT.json"))
            zero_session.update(status="FAIL_CLOSED_STOP", formal_receipts=[], formal_completed=0,
                                first_failure=dict(status="FAIL_CLOSED_STOP", error="CPU_SYNTHETIC_FORMAL_START_NO_LEGAL_COMPLETION"))
            write(fullroot / "SESSION_RESULT.json", zero_session)
            write(fullroot / "FIRST_FAILURE.json", zero_session["first_failure"])
            write(fullroot / "FORMAL_GENERATIONS_COMPLETE.json", dict(original_generation, count=0, formal_receipts=[], status="PARTIAL_FAIL_CLOSED_STOP"))
            write(fullroot / "FINALIZER_CHAIN_STARTED.json", dict(controller_process_really_returned=True, controller_returncode=1))
            dispatched_zero = module.admission(full, fullroot)
            check("formal_dispatched_zero_legal_still_checks_PERF", dispatched_zero["complete60"] is False and dispatched_zero["performance_admission_final_verification"]["status"].startswith("PASS_AUTHORIZATION_SHA"))
            write(fullroot / "SESSION_RESULT.json", sess)
            (fullroot / "FIRST_FAILURE.json").unlink()
            write(fullroot / "FORMAL_GENERATIONS_COMPLETE.json", original_generation)
            write(fullroot / "FINALIZER_CHAIN_STARTED.json", dict(controller_process_really_returned=True, controller_returncode=0))
            synthetic_scores = [dict(condition=job["condition"], problem_id=job["problem_id"], seed=1, correct=(index % 3 == 0),
                                     generated_token_count=100 + index, wall_seconds=10., finish_reason="eos", peak_allocated_bytes=1000) for index, job in enumerate(formal)]
            summary = analysis.summarize(synthetic_scores, history={"P0": [], "P1": []})
            module.scientific_report(full, fullroot, passed, summary)
            report = (fullroot / "FINAL_SCIENTIFIC_REPORT.md").read_text()
            check("complete_report_two_conditions_and_actual_counts", "Legal formal completions: 60/60" in report and "|C1|" in report and "|S3|" in report)
            check("complete_report_packedINT8_limitation", "not physically packed INT8 storage" in report)
            check("complete_report_no81K_stability_claim", "does not prove81K" in report)
            check("complete_report_old_failures_unchanged", "Original V1/V2/V3 failures" in report)
            check("CPU_test_did_not_import_Torch", "torch" not in sys.modules)
    finally:
        if original_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
    result = dict(status="PASS_CPU_SYNTHETIC_FINISHER_ADMISSION_AND_TERMINAL_REPORTS", count=len(checks), checks=checks,
                  finisher_sha256=sha(source / "scripts/resident_finalize_offline.py"), scope="Synthetic CPU receipts and mocked control methods; no real release/9B/GPU/score-process/archive evidence",
                  new_GPU_dispatches=0, new_AIME_samples=0, real_two_fresh_CPU_FrozenV4_passes_still_required=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], count=len(checks))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
