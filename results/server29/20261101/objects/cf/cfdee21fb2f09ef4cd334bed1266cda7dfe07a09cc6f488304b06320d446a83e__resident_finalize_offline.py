#!/usr/bin/env python3
"""Once-chained CPU-only resident finalizer after the real controller exit.

No launch/retry/science patch, no original-task writes and no forged SESSION.
The complete60 case alone receives two independent CPU FrozenV4 passes and
scientific CSV/accuracy. A real STOP keeps its actual counts and first failure.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SUCCESS = "RESIDENT_ROTATION_QUALIFIED_FORMAL_COMPLETE_60_60_AND_STOP"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(control, root, name, value):
    control.atomic_new(Path(root) / name, value)


def current_own_nvml(control, identities, root):
    observation = control.query_gpu()
    known = {identity["pid"]: identity for identity in identities}
    own = []
    for app in observation["apps"]:
        actual = control.process_identity(app["pid"])
        if actual is None and app["pid"] in known:
            own.append(dict(app=app, identity="UNRESOLVED_NVML_PID_AFTER_PROCESS_EXIT"))
        elif actual is not None and (str(root) in actual["cmdline"] or actual == known.get(app["pid"])):
            own.append(dict(app=app, identity=actual))
    require(not own, "A current own GPU process or unresolved own NVML allocation remains")
    return observation


def admission(control, root):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU finisher requires explicitly empty CVD")
    source_sha = control.verify_sources(root)
    control.verify_model_and_external(root, fresh_external=True)
    session = control.read(root / "SESSION_RESULT.json")
    started = control.read(root / "CONTROLLER_STARTED.json")
    chain = control.read(root / "FINALIZER_CHAIN_STARTED.json")
    identity = session["controller_identity"]
    require(session["task_id"] == started["task_id"] == control.TASK
            and session["execution_attempt"] == started["execution_attempt"] == root.name,
            "Actual controller/session task or execution slug differs")
    require(identity == started["identity"] and not control.same_process(identity)
            and str(root / "scripts/resident_controller.py") in identity["cmdline"]
            and str(root) in identity["cmdline"], "Exact original controller is not genuinely exited")
    require(chain["controller_process_really_returned"] is True
            and type(chain["controller_returncode"]) is int, "Shell did not observe controller wait/exit")
    require(session["all_own_processes_released"] is True and session["own_release_all_PASS"] is True
            and not session["unresolved_launches"], "Session still has unresolved process/release scope")
    if session["status"] == SUCCESS:
        require(chain["controller_returncode"] == 0, "Success SESSION cannot coexist with nonzero controller exit")
    qual = control.read(root / "QUALIFICATION_JOB_MANIFEST.json")["units"]
    formal = control.read(root / "FORMAL_JOB_MANIFEST.json")["jobs"]
    require(len(qual) == 58 and len({unit["id"] for unit in qual}) == 58
            and len(formal) == 60 and len({job["id"] for job in formal}) == 60, "Frozen58/60 roster differs")
    all_units = {unit["id"]: ("qualification", unit) for unit in qual}
    all_units.update({job["id"]: ("formal", job) for job in formal})
    intervals, identities = [], [identity]
    for phase in ("qualification", "formal"):
        directory = root / phase
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*/RESOURCE_INTERVAL.json")):
            interval = control.read(path)
            unit_id = path.parent.name
            require(unit_id in all_units and all_units[unit_id][0] == phase
                    and interval["unit"] == unit_id and interval["phase"] == phase,
                    "Unknown or mismatched actual resource interval")
            unit = all_units[unit_id][1]
            own = interval["identity"]
            script = root / "scripts" / ("qualify_gpu.py" if phase == "qualification" else "resident_formal_worker.py")
            dispatch_path = (root / "dispatch/qualification" / (unit_id + ".json") if phase == "qualification"
                             else path.parent / "DISPATCH.json")
            dispatch = control.read(dispatch_path)
            permit_path = root / ("GPU_PERMIT.json" if phase == "qualification" else "FORMAL_GPU_PERMIT.json")
            plan_path = root / ("QUALIFICATION_JOB_MANIFEST.json" if phase == "qualification" else "FORMAL_JOB_MANIFEST.json")
            require(dispatch["task_id"] == control.TASK and dispatch["execution_attempt"] == root.name
                    and dispatch["phase"] == phase and dispatch["unit"] == unit_id
                    and dispatch["unit_record"] == unit and dispatch["controller_identity"] == identity
                    and dispatch["gpu"] == unit["gpu"] and dispatch["gpu_uuid"] == unit["gpu_uuid"]
                    and dispatch["source_freeze_sha256"] == interval["source_freeze_sha256"] == source_sha
                    and dispatch["permit_sha256"] == interval["permit_sha256"] == control.sha(permit_path)
                    and dispatch["base_gpu_permit_sha256"] == control.sha(root / "GPU_PERMIT.json")
                    and dispatch["plan_sha256"] == control.sha(plan_path),
                    "Actual resource interval no longer matches its exact frozen dispatch")
            if phase == "formal":
                require(dispatch["formal_authorization_sha256"] == control.sha(root / "AUTHORIZATION_FORMAL.json"),
                        "Formal dispatch authorization differs")
            args = own["cmdline"]
            require(str(script) in args and str(root) in args
                    and "--unit" in args and args[args.index("--unit") + 1] == unit_id
                    and "--gpu" in args and args[args.index("--gpu") + 1] == str(unit["gpu"]),
                    "Resource process identity does not belong to its own exact unit")
            if (path.parent / "STARTED.json").exists():
                require(control.read(path.parent / "STARTED.json")["identity"] == own,
                        "Actual started process and released process differ")
            require(interval["gpu_uuid"] == unit["gpu_uuid"]
                    and interval["gpu"] == unit["gpu"] and interval["release"] == "PASS"
                    and interval["controller_identity"] == identity and interval["monotonic_same_host"] is True
                    and not control.same_process(own), "Actual unit identity/UUID/release is unresolved")
            require(interval["dispatch_monotonic_lower"] <= interval["identity_monotonic_upper"]
                    <= interval["exit_first_observed_monotonic"] <= interval["release_first_observed_monotonic"],
                    "Same-host monotonic resource ordering differs")
            intervals.append(interval); identities.append(own)
    require(len(intervals) == session["resource_intervals"] <= session["attempted_units"],
            "Session/actual resource interval or attempted dispatch count differs")
    current_gpu = current_own_nvml(control, identities, root)
    actual_qualification_results = {}
    for phase, key in (("qualification", "qualification_receipts"), ("formal", "formal_receipts")):
        receipts = session[key]
        require(len({row["unit"] for row in receipts}) == len(receipts), "Duplicate controller legal receipt")
        for receipt in receipts:
            require(receipt["unit"] in all_units and all_units[receipt["unit"]][0] == phase,
                    "Completed receipt outside frozen roster")
            unit = all_units[receipt["unit"]][1]
            target = root / phase / unit["id"]
            require(receipt["result_sha256"] == control.sha(target / "RESULT.json")
                    and receipt["resource_interval_sha256"] == control.sha(target / "RESOURCE_INTERVAL.json")
                    and receipt["validation_sha256"] == control.sha(target / "CONTROLLER_VALIDATION.json"),
                    "Actual completed result/interval/controller validation SHA differs")
            interval = control.read(target / "RESOURCE_INTERVAL.json")
            require(interval["exit_returncode"] == 0, "Completed legal sample has nonzero exit")
            validated = (control.validate_qualification_result if phase == "qualification" else control.validate_formal_result)(root, unit)
            if phase == "qualification":
                actual_qualification_results[unit["id"]] = validated
    require(session["qualification_completed"] == len(session["qualification_receipts"])
            and session["formal_completed"] == len(session["formal_receipts"]), "Actual legal completed counts differ")
    generations = control.read(root / "FORMAL_GENERATIONS_COMPLETE.json")
    require(generations["count"] == session["formal_completed"]
            and generations["task_id"] == control.TASK and generations["execution_attempt"] == root.name
            and generations["planned"] == 60 and generations["formal_receipts"] == session["formal_receipts"]
            and generations["source_freeze_sha256"] == source_sha
            and generations["all_own_processes_released"] is True, "Actual generation completion receipt differs")
    formal_dispatched = any((root / "formal").glob("*/DISPATCH.json"))
    performance_verification = dict(status="NOT_REQUIRED_NO_FORMAL_DISPATCH")
    if formal_dispatched or session["formal_completed"]:
        require(session["qualification_completed"] == 58,
                "Any actual formal dispatch requires all58 real qualification receipts")
        auth = control.read(root / "AUTHORIZATION_FORMAL.json")
        performance = control.read(root / "PERFORMANCE_ADMISSION.json")
        performance_sha = control.sha(root / "PERFORMANCE_ADMISSION.json")
        require(auth["source_freeze_sha256"] == source_sha
                and auth["performance_admission_sha256"] == performance_sha
                and performance["status"] == "PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED",
                "Final performance verdict no longer matches the actual formal authorization")
        require(control.performance_audit(root, qual, actual_qualification_results) == performance,
                "Final performance verdict differs when recomputed from the actual58 qualification wrappers")
        performance_verification = dict(status="PASS_AUTHORIZATION_SHA_AND_RECOMPUTED_ACTUAL_QUALIFICATION_VERDICT",
                                       sha256=performance_sha, actual_qualification_wrapper_count=len(actual_qualification_results))
    failure = root / "FIRST_FAILURE.json"
    require((not failure.exists() and session["first_failure"] is None)
            or (failure.exists() and control.read(failure) == session["first_failure"]),
            "Original first failure and actual terminal SESSION differ")
    complete = session["status"] == SUCCESS
    if complete:
        require(session["qualification_completed"] == 58 and session["formal_completed"] == 60
                and generations["status"] == "COMPLETE_60_60" and session["first_failure"] is None
                and not (root / "FIRST_FAILURE.json").exists(), "Cannot declare complete60 with a missing/failed unit")
    else:
        require(session["status"] in ("FAIL_CLOSED_STOP", "RESIDENT_ROTATION_MATH_FAIL_CLOSED_STOP", "RESOURCE_BLOCKED_NO_DISPATCH"),
                "Unknown scientific terminal; do not fabricate a recovery status")
    return dict(status="PASS_TRUE_CONTROLLER_EXIT_UNIT_RELEASE_SOURCE61_EXTERNAL17_STAT_AND_NO_OWN_NVML",
                source_freeze_sha256=source_sha, source_count=len(control.read(root / "SOURCE_FREEZE.json")["files"]),
                external_fresh_SHA_count=61, model_stat_count=17, session=session, actual_intervals=intervals,
                exact_exited_identities=identities, observed_NVML=current_gpu, complete60=complete,
                performance_admission_final_verification=performance_verification)


def cpu_score_pass(control, root, index):
    output = root / ("SCORE_PASS_" + str(index) + ".json")
    log_path = root / "logs" / ("score_pass_" + str(index) + ".log")
    require(not output.exists() and not log_path.exists(), "Independent score pass already attempted; no overwrite/retry")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-u", "-B", str(root / "scripts/resident_score_pass_cpu.py"),
               "--root", str(root), "--output", str(output)]
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        identity = control.process_identity(process.pid)
        require(identity is not None, "Fresh independent CPU scoring PID/starttick unresolved")
        save(control, root, "operations/SCORE_PASS_" + str(index) + "_STARTED.json",
             dict(identity=identity, command=command, monotonic=time.monotonic(), CVD="", no_model_load=True))
        code = process.wait()
    save(control, root, "operations/SCORE_PASS_" + str(index) + "_EXIT.json",
         dict(identity=identity, actual_wait_returncode=code, exact_identity_exited=not control.same_process(identity),
              monotonic=time.monotonic(), output_sha256=control.sha(output) if output.exists() else None))
    require(code == 0 and output.is_file() and not control.same_process(identity), "Independent original FrozenV4 CPU score pass failed")
    result = control.read(output)
    require(result["status"] == "PASS_CPU_ORIGINAL_FROZEN_V4_EXACT60_AND_TOKEN_REDECODE"
            and len(result["scores"]) == 60, "CPU scoring result is not genuine exact60/redecode PASS")
    return result, identity


def history_inputs(control, root):
    freeze = control.read(root / "HISTORY_INPUT_FREEZE.json")
    bound = freeze["history_scores"]
    relative = Path(bound["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "History input escapes new root")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and not path.is_symlink()
            and path.stat().st_size == bound["bytes"] and control.sha(path) == bound["sha256"],
            "Pre-launch frozen historical scores changed")
    frozen_sources = {row["path"]: row for row in control.read(root / "SOURCE_FREEZE.json")["files"]}
    require(str(relative) in frozen_sources and "HISTORY_INPUT_FREEZE.json" in frozen_sources,
            "History inputs must be bound before the GPU source freeze")
    sources = freeze["external_sources"]
    require(sources, "Missing actual immutable historical source provenance")
    for row in sources:
        original = Path(row["path"])
        require(original.is_absolute() and original.is_file() and not original.is_symlink()
                and original.stat().st_size == row["bytes"] and control.sha(original) == row["sha256"],
                "Historical original immutable input changed: " + str(original))
    history = control.read(path)
    require(history["status"] == "PASS_TWO_INDEPENDENT_FROZEN_V4_EXISTING_PAYLOADS"
            and isinstance(history["P0"], list) and isinstance(history["P1"], list)
            and history["provenance"], "Historical scored inputs lack independent original FrozenV4 provenance")
    return history, dict(history_file_sha256=control.sha(path), input_freeze_sha256=control.sha(root / "HISTORY_INPUT_FREEZE.json"),
                         external_fresh_SHA_count=len(sources), history_P1_partial=len(history["P1"]) < 30,
                         no_new_old_generation=True)


def inherited_engineering_probe_scope(control, root):
    """Reference a once-only engineering receipt, outside all58/60 science.

No probe is run, retried or granted qualification credit here. Cost uses only
its independently recorded same-host monotonic interval, never wall clocks.
"""
    binding_path=Path(root)/'operations/ENGINEERING_GPU_EVIDENCE_BINDING.json'
    if binding_path.exists():
        import importlib.util
        spec=importlib.util.spec_from_file_location('final_engineering_evidence',Path(root)/'operations/engineering_gpu_evidence.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        evidence=module.validate(root);records=[]
        for entry in evidence['binding']['engineering_resource_records']:
            receipt_path=module.bound(root,entry['receipt']);receipt=control.read(receipt_path)
            interval=receipt['resource_interval']
            names=('dispatch_monotonic_lower','identity_monotonic_upper','exit_first_observed_monotonic','release_first_observed_monotonic')
            values=[interval[n]for n in names]
            require(interval['monotonic_same_host']is True and all(type(v)in(int,float)and math.isfinite(v)for v in values)
                    and values==sorted(values),'Invalid engineering-only monotonic resource interval')
            records.append(dict(id=entry['id'],kind=entry['kind'],receipt=entry['receipt'],recorded_status=receipt.get('status'),
                actual_returncode=receipt.get('actual_wait_returncode',receipt.get('actual_wrapper_wait_returncode',receipt.get('actual_Popen_wait_returncode','UNRESOLVED'))),
                resource_interval=interval,reservation_GPUh_upper=(values[-1]-values[0])/3600.))
        return dict(status='RECORDED_ALL_NATIVE_EXIT_ENGINEERING_HISTORY_SEPARATE_COST_SCOPE',
            receipt=dict(path=str(binding_path.relative_to(root)),bytes=binding_path.stat().st_size,sha256=control.sha(binding_path)),
            qualification_units_contributed=0,formal_samples_contributed=0,engineering_dispatch_count=len(records),
            records=records,reservation_GPUh_upper=sum(x['reservation_GPUh_upper']for x in records),
            EXACT_physical_GPUh='UNRESOLVED_NOT_KERNEL_BUSY_TIME',original_failures_preserved=True,
            scope='All failed/successful engineering loader and native-exit diagnostics; excluded from58 qualification/60formal reservation totals')
    path=Path(root)/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json'
    result=dict(status='NO_ENGINEERING_LOAD_ONLY_PROBE_COMPLETION_RECEIPT',receipt=None,
                qualification_units_contributed=0,formal_samples_contributed=0,
                reservation_GPUh_upper='UNRESOLVED_NO_ENGINEERING_PROBE_RECEIPT',
                EXACT_physical_GPUh='UNRESOLVED_NOT_KERNEL_BUSY_TIME',
                scope='Independent loader/profiler engineering only; not a58-unit qualification PASS, formal sample, GDN trajectory or throughput benchmark')
    if not os.path.lexists(path):
        return result
    require(path.is_file() and not path.is_symlink(),'Engineering completion must be an ordinary readonly receipt')
    receipt=control.read(path)
    result.update(status='RECORDED_ENGINEERING_LOAD_ONLY_PROBE_SEPARATE_COST_SCOPE',
        receipt=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=control.sha(path)),
        recorded_status=receipt.get('status'),recorded_identity=receipt.get('identity'),
        recorded_GPU_uuid=receipt.get('gpu_uuid',receipt.get('GPU_UUID')),
        recorded_actual_returncode=receipt.get('actual_returncode',receipt.get('actual_wait_returncode')),
        recorded_release=receipt.get('release',receipt.get('own_release')),
        recorded_model_load_seconds=receipt.get('model_load_seconds'),
        recorded_peak_allocated_bytes=receipt.get('peak_allocated_bytes'),
        reservation_GPUh_upper='UNRESOLVED_NO_VALID_SAME_HOST_ENGINEERING_INTERVAL')
    interval=receipt.get('resource_interval')
    if isinstance(interval,dict):
        result['recorded_resource_interval']=interval
        names=('dispatch_monotonic_lower','identity_monotonic_upper','exit_first_observed_monotonic','release_first_observed_monotonic')
        valid=(interval.get('monotonic_same_host')is True and all(type(interval.get(name))in(int,float)
               and math.isfinite(interval[name]) and interval[name]>=0 for name in names))
        if valid and all(interval[names[index]]<=interval[names[index+1]]for index in range(3)):
            result['reservation_GPUh_upper']=(interval[names[-1]]-interval[names[0]])/3600.
            result['same_host_monotonic']=True
    return result


def engineering_probe_scope(control, root):
    inherited = inherited_engineering_probe_scope(control, root)
    evidence_path = Path(root) / 'analysis/PROFILER_ENGINEERING_EVIDENCE.json'
    if not evidence_path.exists():
        return inherited
    import importlib.util
    spec = importlib.util.spec_from_file_location('current_profiler_engineering_evidence',
        Path(root) / 'operations/profiler_engineering_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = module.validate(root, control)
    require(current == control.read(evidence_path), 'Current profiler engineering evidence changed')
    return dict(status='RECORDED_DISTINCT_INHERITED12_CURRENT15_ENGINEERING_COST_SCOPES',
        receipt=module.describe(Path(root).resolve(), evidence_path),
        inherited_engineering=inherited, current_profiler_engineering=current,
        engineering_dispatch_count=inherited['engineering_dispatch_count'] + current['engineering_dispatch_count'],
        reservation_GPUh_upper=inherited['reservation_GPUh_upper'] + current['reservation_GPUh_upper'],
        qualification_units_contributed=0, formal_samples_contributed=0,
        EXACT_physical_GPUh='UNRESOLVED_NOT_KERNEL_BUSY_TIME', original_failures_preserved=True)


def resource_summary(intervals, scores, engineering_probe=None):
    upper = sum(row["release_first_observed_monotonic"] - row["dispatch_monotonic_lower"] for row in intervals) / 3600.
    return dict(EXACT_physical_GPUh="UNRESOLVED_NOT_KERNEL_BUSY_TIME", same_host_monotonic=True,
                dispatched_interval_count=len(intervals), reservation_GPUh_upper=upper,
                legally_completed_formal_timed_count=len(scores),
                formal_live_model_timed_generation_hours=sum(row["wall_seconds"] for row in scores) / 3600.,
                engineering_load_only_probe=(engineering_probe if engineering_probe is not None else
                    dict(status='NO_ENGINEERING_PROBE_RECEIPT_SUPPLIED',qualification_units_contributed=0,formal_samples_contributed=0)),
                reservation_GPUh_upper_excludes_separate_engineering_probe=True,
                scope="Reservation upper uses dispatch-to-first-release observation; timed generation is host interval while the live model is held, excludes loading/qualification/cleanup; neither is exact physical busyGPUh")


def complete_analysis(control, root, admitted):
    first, id0 = cpu_score_pass(control, root, 0)
    second, id1 = cpu_score_pass(control, root, 1)
    require(id0 != id1 and first == second, "Two fresh original FrozenV4/redecode passes differ")
    save(control, root, "FROZEN_V4_SCORES.json", first)
    history, provenance = history_inputs(control, root)
    save(control, root, "HISTORY_INPUT_FRESH_VERIFICATION.json", provenance)
    from resident_analysis import summarize
    summary = summarize(first["scores"], history={key: history[key] for key in ("P0", "P1")})
    summary["history_protocol_provenance"] = history["provenance"]
    summary["actual_resource_cost"] = resource_summary(admitted["actual_intervals"], first["scores"], admitted.get('engineering_load_only_probe'))
    save(control, root, "AIME26_RESIDENT_ANALYSIS.json", summary)
    by = {(row["condition"], row["problem_id"]): row for row in first["scores"]}
    previous = {key: {row["problem_id"]: row for row in history[key]} for key in ("P0", "P1")}
    fields = ["problem_id", "seed", "gpu_uuid", "C1_correct", "S3_correct", "C1_answer", "S3_answer",
              "C1_tokens", "S3_tokens", "C1_finish", "S3_finish", "S3_rescue", "S3_lost", "P0_correct", "P1_correct"]
    with (root / "AIME26_RESIDENT_PER_PROBLEM.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for problem in sorted({row["problem_id"] for row in first["scores"]}):
            c, s = by[("C1", problem)], by[("S3", problem)]
            require(c["gpu_uuid"] == s["gpu_uuid"], "Paired current C1/S3 did not use the same frozen UUID")
            writer.writerow(dict(problem_id=problem, seed=1, gpu_uuid=c["gpu_uuid"],
                C1_correct=c["correct"], S3_correct=s["correct"], C1_answer=c["extracted_answer"], S3_answer=s["extracted_answer"],
                C1_tokens=c["generated_token_count"], S3_tokens=s["generated_token_count"],
                C1_finish=c["finish_reason"], S3_finish=s["finish_reason"],
                S3_rescue=s["correct"] and not c["correct"], S3_lost=c["correct"] and not s["correct"],
                P0_correct=previous["P0"].get(problem, {}).get("correct", "NA"),
                P1_correct=previous["P1"].get(problem, {}).get("correct", "NA")))
        stream.flush(); os.fsync(stream.fileno())
    return summary


def scientific_report(control, root, admitted, summary):
    session = admitted["session"]
    qual_units = control.read(root / "QUALIFICATION_JOB_MANIFEST.json")["units"]
    performance, fp = [], None
    legal_qualification_ids = {row["unit"] for row in session["qualification_receipts"]}
    for unit in qual_units:
        if unit["id"] not in legal_qualification_ids:
            continue
        target = root / "qualification" / unit["id"] / "RESULT.json"
        if not target.exists():
            continue
        wrapper = control.read(target)
        if wrapper.get("status") != "PASS":
            continue
        if unit["kind"].startswith(("perf_", "profile_")):
            performance.append(dict(unit=unit["id"], kind=unit["kind"], condition=unit["condition"],
                                    gpu_uuid=unit["gpu_uuid"], actual_result=wrapper["result"]))
        elif unit["kind"] == "fp128":
            fp = wrapper["result"]
    save(control, root, "ACTUAL_GPU_PERFORMANCE_AND_MEMORY.json", dict(rows=performance,
          TPS_source="Actual nonAIME perf windows only; profiler is not throughput and no81K freegeneration guarantee"))
    save(control, root, "FP_ROTATION_NUMERICAL_DIAGNOSTICS.json", dict(
          status="ACTUAL_GPU_FP128_DIAGNOSTICS" if fp is not None else "NOT_RUN_OR_NOT_LEGALLY_PASSED",
          result=fp, maxabs_not_original_4xH_hard_gate=True, diagonal_2e6_not_used=True))
    lines = ["# Resident UH/UHP scientific delivery", "", "Terminal: `" + session["status"] + "`.",
             "Legal formal completions: %d/60; real qualification completions: %d/58." %
             (session["formal_completed"], session["qualification_completed"]), "",
             "C1 keeps State in UH; S3 keeps State in UHP. Original BF16 Q/K normalization precedes FP32 unique16-head batched rotation and repeat2. The unchanged GDN helper applies Query scaling once; Gate/Beta/Value/readout remain original. Persistent Symmetric INT8 Key32 QDQ writes directly to resident coordinates; S3 does not permute inside QDQ. The FP32 reconstructed carrier is not physically packed INT8 storage.", ""]
    engineering=admitted.get('engineering_load_only_probe')
    if engineering and engineering.get('receipt') is not None:
        lines.extend(['An independently recorded engineering loader/profiler probe is referenced by its original receipt SHA. Its separate same-host reservation upper is '+str(engineering['reservation_GPUh_upper'])+' GPUh. It contributes zero qualification units and zero formal samples; it is not GDN/token throughput evidence.', ''])
    else:
        lines.extend(['No engineering loader-only probe completion receipt is supplied. No probe completion or cost is invented or added to the58 qualification/60 formal counts.', ''])
    if summary is not None:
        lines.extend(["Both independent original FrozenV4 CPU passes and exact sampled-token re-decodes agree.", "",
                      "|Condition|Correct/30|Accuracy|EOS|Length|Mean tokens|Median tokens|",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for name, value in summary["conditions"].items():
            lines.append("|%s|%d/30|%.2f%%|%d|%d|%.2f|%.2f|" % (name, value["correct"],
                value["accuracy"] * 100, value["EOS"], value["length_terminated"], value["length_mean"], value["length_median"]))
        pair = summary["S3_vs_C1"]
        lines.extend(["", "S3 vs C1: rescue%d, lost%d, net%d, delta%.2f percentage points on identical30 questions." %
                      (len(pair["rescue"]), len(pair["lost"]), pair["net"], pair["delta_percentage_points"]),
                      "Historical P0/P1 comparisons and I=(S3-C1)-(P1-P0) use only the same actual common problem set. The JSON records its actual N and partial history; no30-vs-partial direct subtraction."])
    else:
        lines.extend(["This is a real technical STOP/PARTIAL. No complete accuracy, scientific score CSV or fabricated ungenerated rows are produced.",
                      "The original first failure, completed legal payloads, partial generated tokens and actual resource intervals remain in the archive.",
                      "First failure: `" + json.dumps(session.get("first_failure"), ensure_ascii=False, sort_keys=True) + "`."])
    lines.extend(["", "Original V1/V2/V3 failures and human-cancelled P1 remain unchanged; no old samples were regenerated. All comparisons are exploratory:30 fixed questions, one seed, AIME already observed, and changed16-vs32 normalization batching/runtime/UUID can change finite rounding. Historical protocol differences preclude strict causal claims; the interaction is descriptive, not a mechanism proof.",
                  "", "Actual FP rotation, QDQ field/writeback/next-consumption and performance evidence are preserved in their unit receipts and the diagnostic/performance JSONs. Synthetic82293 KV/SDPA pressure does not prove81K GDN numerical stability. Physical busyGPUh is UNRESOLVED; same-host monotonic reservation observations and actual generation host intervals are reported separately.",
                  "", "Failed external /dev/shm engineering references, if SESSION marks them retained, are not deleted or called continuation State. This archive seals persistent files inside this execution root; external reference preservation is separately identified and is not falsely claimed replicated.",
                  "", "The independent29 archive requires fresh SHA256/bytes for every dynamic file, refuses unknown extras, and publishes once with renameat2 NOREPLACE. The exact original29 controls are mirrored to55. Raw tensors and model weights are not downloaded toMac.", ""])
    path = root / "FINAL_SCIENTIFIC_REPORT.md"
    with path.open("x") as stream:
        stream.write("\n".join(lines)); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU finisher requires empty CVD before imports")
    sys.path.insert(0, str(root / "src"))
    control = importlib.import_module("resident_execution_control")
    archive = importlib.import_module("resident_archive")
    require(root == control.ROOT, "Finisher accepts only its independent execution root")
    require(not any((root / name).exists() for name in
                    archive.CONTROLS | {"FINISHER_STARTED.json", "FINISHER_RESULT.json", "FINISHER_FAILURE.json"}),
            "Finisher/seal was already attempted; inspect without retry")
    identity = control.process_identity(os.getpid())
    require(identity is not None, "Actual finisher process identity unresolved")
    save(control, root, "FINISHER_STARTED.json", dict(identity=identity, monotonic=time.monotonic(),
         once_only=True, CUDA_visible_devices="", no_GPU_or_controller_restart=True))
    try:
        admitted = admission(control, root)
        admitted['engineering_load_only_probe']=engineering_probe_scope(control,root)
        save(control, root, "FINAL_CPU_ADMISSION_AND_RELEASE.json", admitted)
        save(control, root, 'ENGINEERING_LOAD_ONLY_PROBE_COST_SCOPE.json', admitted['engineering_load_only_probe'])
        legal_formal_rows = [control.read(root / "formal" / row["unit"] / "RESULT.json")["result"]
                             for row in admitted["session"]["formal_receipts"]]
        save(control, root, "ACTUAL_RESOURCE_COST_SUMMARY.json",
             resource_summary(admitted["actual_intervals"], legal_formal_rows, admitted['engineering_load_only_probe']))
        summary = complete_analysis(control, root, admitted) if admitted["complete60"] else None
        if summary is None:
            require(not (root / "AIME26_RESIDENT_PER_PROBLEM.csv").exists()
                    and not (root / "FROZEN_V4_SCORES.json").exists(),
                    "STOP/PARTIAL cannot inherit a complete scientific CSV or complete scoring output")
        scientific_report(control, root, admitted, summary)
        task = dict(task_id=control.TASK, execution_attempt=root.name, status=admitted["session"]["status"],
             planned_formal=60, actual_formal_completed=admitted["session"]["formal_completed"],
             actual_qualification_completed=admitted["session"]["qualification_completed"],
             complete_accuracy=summary["conditions"] if summary is not None else None,
             old_science_terminations_unchanged=True, physical_GPUh="UNRESOLVED", no_retry=True,
             engineering_load_only_probe=admitted['engineering_load_only_probe'],
             external_engineering_reference_scope=admitted["session"].get("reference_pair_retained"))
        save(control, root, "TASK_RESULT.json", task)
        control.verify_sources(root)
        control.verify_model_and_external(root, fresh_external=True)
        current_own_nvml(control, admitted["exact_exited_identities"], root)
        result = archive.publish(root, task_id=control.TASK, terminal=task["status"],
                    formal_completed=task["actual_formal_completed"], finisher_identity=identity)
        save(control, root, "FINISHER_RESULT.json", dict(status="PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29",
             terminal=task["status"], identity=identity, actual_formal_completed=task["actual_formal_completed"],
             archive=result, independent_local_delivery_and_post_exit_audit_still_required=True,
             monotonic=time.monotonic()))
        print(json.dumps(dict(status="PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29", terminal=task["status"],
                             actual_formal_completed=task["actual_formal_completed"], archive=result["archive_root"])), flush=True)
        return 0
    except BaseException as error:
        if not (root / "FINISHER_FAILURE.json").exists():
            save(control, root, "FINISHER_FAILURE.json", dict(status="REAL_CPU_FINISHER_FAILURE_NO_RETRY_NO_SEAL_REWRITE",
                 identity=identity, error=repr(error), traceback=traceback.format_exc(), monotonic=time.monotonic()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
