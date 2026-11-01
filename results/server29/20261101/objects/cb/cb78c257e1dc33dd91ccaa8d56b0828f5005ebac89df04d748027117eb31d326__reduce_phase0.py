#!/usr/bin/env python3
"""CPU-only reduction after the one-shot Phase0 worker/controller truly release.

Uses only stdlib. Does not import torch, run GPU code, launch scientific work,
score AIME, choose an optimization or decide a STOP/scientific direction.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import tempfile
from chrome_trace_categories import corrected_profile

TASK = "QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1"
CONDITIONS = ("FP_NATIVE", "H_C128", "H_DUAL_M_REFERENCE")
SCOPES = ("REFERENCE/model_forward", "REFERENCE/finite_state_and_logits_gate", "REFERENCE/QDQ_and_original_safety")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    require(bool(values) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values), "Missing/nonfinite statistics")
    x = sorted(values)
    return dict(count=len(x), min=x[0], max=x[-1], mean=math.fsum(x)/len(x),
                p50=statistics.median(x), p95=x[max(0, math.ceil(.95*len(x))-1)])


def relative_path(value):
    p = PurePosixPath(value)
    require(isinstance(value, str) and value and not p.is_absolute() and ".." not in p.parts and not any(ord(c)<32 for c in value), "Unsafe relative path")
    return value


def verify_file(root, entry):
    path = root / relative_path(entry["path"])
    require(path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "File SHA/bytes mismatch: "+str(path))
    return path


def release_gate(root, proof_path, live):
    phase = root/"execution/profile"
    worker = read(phase/"RESULT.json")
    controller = read(phase/"CONTROLLER_RESULT.json")
    started = read(phase/"CONTROLLER_STARTED.json")
    ledger_path = root/"GPU_RESOURCE_LEDGER.json"
    ledger = read(ledger_path)
    require(worker["status"] == controller["status"] == "PASS", "Phase0 not PASS; no scientific CSV permitted")
    require(controller["all_own_GPUs_released"] and ledger["all_own_GPUs_released"], "GPU release not established")
    require(sha(ledger_path) == controller["ledger_sha256"], "Current ledger is not the phase-completion ledger; use immutable Phase0 snapshot")
    jobs = [j for j in ledger["dispatches"] if j["mode"] == "profile"]
    require(len(jobs) == 1, "Expected one Phase0 GPU dispatch")
    job = jobs[0]
    require(job["process_exited"] and job["NVML_released"] and job["returncode"] == 0, "Worker did not truly complete/release")
    require(not (root/"FIRST_FAILURE.json").exists(), "Real failure present; do not generate successful profile tables")
    pids = [started["pid"], job["pid"]]
    if live:
        require(Path("/proc").is_dir(), "Live process verification requires the execution host")
        require(all(not Path("/proc", str(pid)).exists() for pid in pids), "Old worker/controller PID still exists")
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=30)
        nvml_pids = {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}
        require(not nvml_pids.intersection(pids), "Old tracked process still in NVML")
        verification = dict(status="PASS", method="CPU read-only /proc absence and nvidia-smi tracked-PID absence", tracked_pids=pids)
    else:
        require(proof_path is not None, "Supply --verify-live-release on55 or an independent --release-proof JSON")
        proof = read(proof_path)
        require(proof.get("status") == "PASS" and proof.get("task_id") == TASK, "Release proof not PASS for this task")
        require(proof.get("phase0_ledger_sha256") == sha(ledger_path), "Release proof must bind phase0 ledger SHA")
        require(set(proof.get("tracked_pids", [])) == set(pids), "Release proof PID roster differs")
        require(proof.get("tracked_processes_absent") is True and proof.get("tracked_NVML_pids_absent") is True, "Missing true process/NVML release proof")
        verification = dict(status="PASS", method="Independent bound read-only execution-host proof", path=str(proof_path), sha256=sha(proof_path))
    return worker, controller, ledger, verification


def CPU_operators(record):
    # Kineto key_averages can contain CUDA user_annotation rows with identical
    # names and cpu_total_us=0. Those are not CPU scopes or CUDA kernels.
    return [o for o in record["profile"]["operators"] if o["cpu_total_us"] > 0]


def profile_view(record):
    raw = record["profile"]
    profile = record["_chrome_category_corrected"]
    n = record["measured_decode_tokens"]
    activities = profile["CUDA_device_activity_costs_by_name"]
    require(bool(activities) and profile["CUDA_kernel_activity_count_excluding_memcpy_memset"] > 0, "No actual cat==kernel CUDA activities captured")
    operators = CPU_operators(record)
    phases = [o for o in operators if o["name"] in SCOPES]
    require(len(phases) == len(SCOPES) and {o["name"] for o in phases} == set(SCOPES), "Missing/duplicate original CPU instrumentation scopes")
    require(all(o["count"] == n for o in phases), "Missing per-token CPU instrumentation scopes")
    trace_owners = profile["CPU_owner_scopes_from_chrome"]
    for phase in phases:
        other = trace_owners.get(phase["name"])
        require(other and other["count"] == phase["count"] and math.isclose(other["cpu_total_us"],phase["cpu_total_us"],rel_tol=1e-4,abs_tol=1.0), "CPU owner attribution disagrees with category-correct Chrome scopes")
    runtime = [o for o in operators if any(token in o["name"].lower() for token in ("cudalaunch", "cudadevicesynchronize", "cudaeventsynchronize", "cudastreamsynchronize", "cudamemcpy", "aten::item", "aten::_local_scalar_dense"))]
    return dict(condition=record["condition"], measurement_stage="INSTRUMENTED_PROFILE", measured_tokens=n,
                decode_window_wall_seconds=record["decode_window_wall_seconds"],
                whole_replay_wall_seconds_through_profile_rows=record["whole_replay_wall_seconds"],
                CHROME_CATEGORY_CORRECTED=profile,
                RAW_DERIVED_UNCORRECTED=dict(CUDA_device_activity_count=raw["CUDA_device_activity_count"],
                    CUDA_kernel_activity_count_excluding_memcpy_memset=raw["CUDA_kernel_activity_count_excluding_memcpy_memset"],
                    CUDA_active_interval_union_us=raw["CUDA_active_interval_union_us"],
                    CUDA_activity_duration_sum_us=raw["CUDA_activity_duration_sum_us"],
                    defect="Frozen worker profile_rows treated all CUDA DeviceType events, including gpu_user_annotation, as deviceactivity/kernel. Raw original JSON preserved; these fields cannot establish truekernels orGPUactive."),
                actual_device_activity_count=profile["CUDA_device_activity_count"],
                actual_kernel_activity_count=profile["CUDA_kernel_activity_count_excluding_memcpy_memset"],
                actual_kernel_activities_per_measured_token=profile["CUDA_kernel_activity_count_excluding_memcpy_memset"]/n,
                CUDA_device_active_interval_union_us=profile["CUDA_active_interval_union_us"],
                CUDA_activity_duration_sum_us=profile["CUDA_activity_duration_sum_us"],
                CUDA_stream_window_ms=record["CUDA_stream_window_ms"],
                actual_device_activity_names=activities,
                top3_device_activity_by_summed_duration=sorted(activities, key=lambda x:x["summed_duration_us"], reverse=True)[:3],
                top3_CPU_operators_by_self_time=sorted(operators, key=lambda x:x["self_cpu_us"], reverse=True)[:3],
                original_disjoint_CPU_phase_scopes=phases,
                runtime_launch_sync_and_scalar_operators=runtime,
                nonadditivity="Chrome cat classification excludes GPU annotations. Activity sum may overlap. CPU/ATen/phase attributed device totals are nested; do not sum together or callwall. CUDAstreamelapsed includesidlegaps. WallminusactiveunionisNOTpureCPUdispatch.")


def codec_events(record):
    rows = record.get("original_DualM_event_timings_per_token_ms", [])
    if record["condition"] != "H_DUAL_M_REFERENCE":
        require(not rows, "Unexpected DualM timing for another method")
        return None
    require(len(rows) == record["measured_decode_tokens"], "Incomplete original DualM events")
    keys = ("encode_ms", "decode_ms", "writeback_ms", "safety_and_metric_audit_ms")
    require(all(all(k in row and row[k] >= 0 and math.isfinite(row[k]) for k in keys) for row in rows), "Missing/invalid original event timings")
    totals = [math.fsum(row[k] for k in keys) for row in rows]
    wall_ms = record["decode_window_wall_seconds"]*1000
    return dict(scope="Original batch24 DualM same-stream event spans, including device idle gaps between boundaries; excludes cat before event0 and some host setup. No micro-single-layer substitution.",
                per_token_ms={k:stats([r[k] for r in rows]) for k in keys},
                summed_sequential_event_span_ms=math.fsum(totals),
                summed_event_span_over_decode_wall_ratio=math.fsum(totals)/wall_ms,
                per_token_sum_event_span_ms=stats(totals),
                strict_removable_wall_fraction="UNKNOWN", strict_Amdahl_upper_ceiling="UNKNOWN",
                reason="Event spans are not exclusive CPU+GPU codec wall and can include launch gaps. Cannot subtract event-time sums from wall to establish an end-to-end removable-cost upper bound.")


def csv_bytes(rows, fields):
    require(bool(rows), "Do not create an empty scientific/performance CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def reduce(root, output, proof, live):
    worker, controller, ledger, release = release_gate(root, proof, live)
    auth = read(root/"AUTHORIZATION_PHASE0.json")
    environment = read(root/"execution/profile/ENVIRONMENT.json")
    require(sha(root/"AUTHORIZATION_PHASE0.json") == worker["authorization_sha256"] and auth["status"] == "PASS", "Phase0 authorization binding failed")
    require(environment["gpu_uuid"] == auth["gpu_uuid"] and environment["physical_gpu"] == 0 and environment["CUDA_VISIBLE_DEVICES"] == "0", "Actual GPU environment differs")
    require(environment["deterministic_algorithms"] is True and environment["TF32"] is False and environment["chunk_binding"] == "original qualified compensated", "Frozen reference runtime differs")
    require(environment["parameter_dtypes"] == ["torch.bfloat16"] and ledger["cumulative_GPU_hours"] <= 2.0, "Model dtype or cumulative budget differs")
    source_manifest = read(root/"SOURCE_MANIFEST.json")
    for entry in source_manifest["execution_sources"]:
        verify_file(root, entry)
    require(sha(root/"SOURCE_MANIFEST.json") == worker["source_manifest_sha256"], "Source manifest binding failed")
    for filename, key in [("PLAN.json", "plan_sha256"), ("REFERENCE_BINDING.json", "reference_binding_sha256")]:
        require(sha(root/filename) == worker[key], "Frozen source binding failed: "+filename)
    inputs = []
    baseline = []
    profiles = []
    seen = set()
    require(len(worker["completed_units"]) == 12, "Expected9 actual short baseline and3 separate profile records")
    for entry in worker["completed_units"]:
        path = verify_file(root, entry)
        r = read(path)
        require(r["status"] == "PASS" and r["task_id"] == TASK, "Actual record not PASS")
        require(r["condition"] in CONDITIONS and r["length"] == r["measured_decode_tokens"] == 128 and r["warmup_tokens"] == 128 and r["consumed_tokens"] == 256, "Wrong window/condition")
        require(r["sampling"] == "NOT_RUN_FIXED_TEACHER_FORCED_TOKENS" and r["AIME"] == "NOT_RUN", "Unexpected science/sampler scope")
        require(r["source_manifest_sha256"] == worker["source_manifest_sha256"] and r["reference_binding_sha256"] == worker["reference_binding_sha256"] and r["plan_sha256"] == worker["plan_sha256"], "Record source binding differs")
        key = (r["condition"], r["repeat"], r["instrumented"])
        require(key not in seen, "Duplicate timing unit")
        seen.add(key)
        require(len(r["per_token_wall_dispatch_seconds"]) == 128 and all(x>0 and math.isfinite(x) for x in r["per_token_wall_dispatch_seconds"]), "Dispatch timing incomplete")
        require(r["decode_window_wall_seconds"] > 0 and math.isclose(r["decode_tokens_per_second"], 128/r["decode_window_wall_seconds"], rel_tol=1e-10), "Window throughput inconsistent")
        require(r["native_core_calls"] == 256*24 and r["persistent_recurrent_state_tensor_bytes"] == 50331648, "Native call/carrier contract mismatch")
        if r["condition"] == "FP_NATIVE":
            require(r["QDQ_calls"] == r["layer_writebacks"] == r["next_token_consumption_checks"] == 0, "FP hasunexpected QDQ")
        else:
            require(r["QDQ_calls"] == 256 and r["layer_writebacks"] == 256*24 and r["next_token_consumption_checks"] == 255*24, "Persistent QDQ/consumption count mismatch")
        if r["instrumented"]:
            require(r["measurement_stage"] == "INSTRUMENTED_PROFILE" and r["repeat"] == 0 and len(r["per_token_CUDA_stream_ms"]) == 128, "Wrong instrumented unit")
            trace_entry = r.get("chrome_trace")
            require(isinstance(trace_entry, dict), "Actual category-bearing Chrome trace required")
            trace_path = root / relative_path(trace_entry["path"])
            r["_chrome_category_corrected"] = corrected_profile(trace_path,trace_entry["sha256"],trace_entry["bytes"],SCOPES)
            profiles.append(r)
        else:
            require(r["measurement_stage"] == "SHORT_NONINSTRUMENTED_BASELINE" and r["repeat"] in (0,1,2) and r["profile"] is None and not r["per_token_CUDA_stream_ms"], "Wrong noninstrumented unit")
            baseline.append(r)
        inputs.append(dict(path=entry["path"], sha256=entry["sha256"], bytes=entry["bytes"]))
    require(len(baseline) == 9 and len(profiles) == 3, "Measured trial roster incomplete")
    summary = []
    for condition in CONDITIONS:
        trials = sorted([r for r in baseline if r["condition"] == condition], key=lambda x:x["repeat"])
        require([r["repeat"] for r in trials] == [0,1,2], "Three repetitions required")
        record = dict(condition=condition, warmup_tokens=128, measured_decode_tokens_per_trial=128, actual_repeats=3,
                      decode_window_wall_seconds=stats([r["decode_window_wall_seconds"] for r in trials]),
                      decode_tokens_per_second=stats([r["decode_tokens_per_second"] for r in trials]),
                      weighted_tokens_per_second=384/math.fsum(r["decode_window_wall_seconds"] for r in trials),
                      pooled_host_iteration_dispatch_seconds=stats([v for r in trials for v in r["per_token_wall_dispatch_seconds"]]),
                      peak_allocated_bytes=stats([r["peak_allocated_bytes"] for r in trials]),
                      peak_reserved_bytes=stats([r["peak_reserved_bytes"] for r in trials]),
                      persistent_recurrent_state_tensor_bytes=50331648,
                      per_token_dispatch_scope="Hostiteration elapsed incl reference checks/waits, not synchronized pertoken GPU completion latency.")
        if condition == "H_DUAL_M_REFERENCE":
            record["batch24_original_codec_event_spans_by_trial"] = [codec_events(r) for r in trials]
        summary.append(record)
    views = [profile_view(r) for r in profiles]
    overhead = []
    for r in profiles:
        median = next(x for x in summary if x["condition"] == r["condition"])["decode_window_wall_seconds"]["p50"]
        overhead.append(dict(condition=r["condition"], profile_repeats=1, baseline_repeats=3,
                             instrumented_wall_seconds=r["decode_window_wall_seconds"], noninstrumented_median_wall_seconds=median,
                             observed_profile_wall_over_baseline_median=r["decode_window_wall_seconds"]/median,
                             scope="Same128-tokenwindow+warmup, one separate instrumented trial. Includes trial variability and profiler overhead; not algorithm slowdown.",
                             host_iteration_dispatch_p50_p95_seconds=stats(r["per_token_wall_dispatch_seconds"]),
                             per_token_CUDA_same_stream_interval_ms=stats(r["per_token_CUDA_stream_ms"])))
    micro_path = root/"execution/profile/CODEC_MICRO.json"
    micro = read(micro_path)
    require(micro["status"] == "PASS" and micro["fixture_shape"] == [1,32,128,128] and len(micro["rows"]) == 3, "Realstate microbenchmark incomplete")
    diagnostic = micro["stage_diagnostic"]
    require(diagnostic["status"] == "PASS_BYTE_EXACT_INSTRUMENTED_VS_ORIGINAL" and set(diagnostic["output_parity"]) == {"codes", "rows", "columns", "QDQ"} and all(x["status"] == "BYTE_EXACT" for x in diagnostic["output_parity"].values()), "Diagnostic observer parity missing")
    stages = [row for row in diagnostic["stages"] if row["cpu_total_us"] > 0]
    require(len(stages) == 8 and {x["name"] for x in stages} == {"CODEC_STAGE/"+s for s in ("INPUT_VALIDATION","ROW","INITIAL_COLUMN","INITIAL_LEVELS","LS","FINAL_LEVELS","VALIDATE","DECODE")}, "Incomplete original codecstage diagnosis")
    stage_device_total = math.fsum(max(0, x["device_total_us"]) for x in stages)
    ls = next(x for x in stages if x["name"] == "CODEC_STAGE/LS")
    ls_us = ls["device_total_us"] if ls["device_total_us"] > 0 else None
    micro_summary = dict(scope="REAL_STATE_SINGLE_LAYER_[1,32,128,128]; not batched24layerfullmodel", fixture_source=micro["fixture_source"], fixture_sha256=micro["fixture_sha256"],
                         measured_repeats=3, encode_host_seconds_including_reference_checks=stats([x["encode_host_seconds_including_reference_checks"] for x in micro["rows"]]),
                         encode_decode_finite_complete_wall_seconds=stats([x["encode_decode_finite_complete_wall_seconds"] for x in micro["rows"]]),
                         encoded_tensor_bytes=[x["encoded_tensor_bytes"] for x in micro["rows"]],
                         instrumented_original_stage_costs=stages,
                         single_layer_actual_kernel_count="UNKNOWN_NO_EXPORTED_CATEGORY_TRACE",
                         single_layer_actual_device_active_union_us="UNKNOWN_NO_EXPORTED_CATEGORY_TRACE",
                         RAW_DERIVED_UNCORRECTED=dict(profile=diagnostic["profile"],stages=diagnostic["stages"],worker_LS_device_attributed_us=diagnostic["LS_device_attributed_us"],reason="Frozen raw CUDAdeviceevents includeGPUuserannotations. Only positiveCPUowner rows usedforgroupedLS; category-correctmicrokernels/unionnotrecoverablewithoutaChromeexport."),
                         LS_grouped_device_attributed_us=ls_us,
                         LS_fraction_of_exclusive_named_stage_device_attribution=(ls_us/stage_device_total if ls_us is not None and stage_device_total>0 else None),
                         LS_fraction_denominator="Sum of eight original nonnested CODEC_STAGE owner-scope attributed device totals only; excludes nestedATen rows; diagnostic local activity ratio, NOT fullmodel or wall fraction. CPU/GPU overlap and idlegaps not inferred.",
                         instrumented_stage_diagnostic_wall_seconds=diagnostic["instrumented_wall_seconds"],
                         instrumentation_output_parity=diagnostic["output_parity"],
                         limitations=micro["limitations"]+" sys.settrace/record_function diagnostic overhead remains separate from3noninstrumentedmicrorepeats.")
    amdahl = []
    for r in profiles:
        if r["condition"] != "H_DUAL_M_REFERENCE":
            continue
        phase = next(x for x in CPU_operators(r) if x["name"] == "REFERENCE/QDQ_and_original_safety")
        fraction = phase["cpu_total_us"]/(r["decode_window_wall_seconds"]*1e6)
        amdahl.append(dict(scope="Instrumentedsame-runQDQCPUownerregion only, not noninstrumented production estimate",
                          observed_QDQ_owner_region_cpu_elapsed_us=phase["cpu_total_us"], decode_window_wall_seconds=r["decode_window_wall_seconds"],
                          observed_owner_region_fraction=fraction,
                          conditional_ideal_delete_entire_owner_region_ceiling=(1/(1-fraction) if 0<=fraction<1 else None),
                          assumptions="Conditional ceiling only if ownerregion elapsed is exclusive/nonoverlapped and all of it removable with no added dependency/cost; includesdiagnostic overhead. Does not prove actualcandidate ceiling.",
                          noninstrumented_strict_Amdahl_ceiling="UNKNOWN_NOT_EXCLUSIVE_CODEC_WALL_MEASURED",
                          same_run_original_codec_event_span=codec_events(r)))
    result = dict(task_id=TASK, status="ACTUAL_PHASE0_PROFILE_REDUCED_NO_OPTIMIZATION_DECISION", scientific_scope="TeacherforcedfixednonAIME replay, noaccuracy/new32Kscience/candidate",
                  source_manifest_sha256=sha(root/"SOURCE_MANIFEST.json"), reference_binding_sha256=sha(root/"REFERENCE_BINDING.json"), plan_sha256=sha(root/"PLAN.json"),
                  input_record_verification=inputs, release_verification=release,
                  controller_result_sha256=sha(root/"execution/profile/CONTROLLER_RESULT.json"), phase0_ledger_sha256=sha(root/"GPU_RESOURCE_LEDGER.json"),
                  measured_environment=environment, measured_environment_sha256=sha(root/"execution/profile/ENVIRONMENT.json"),
                  cumulative_GPU_hours=ledger["cumulative_GPU_hours"],
                  profiling_occupancy_cost_scope="Task GPU ledger charges dispatch through process exit and NVMLrelease, including model load, warmup, shortbaseline, profilercollection, CPU key_averages/events parsing, Chrome trace export, fixturediagnosis and teardown while worker ownsGPU. Do not report only the128-tokencollection window as totalprofilingcost.",
                  profile_host_postprocessing_individual_time="NOT_SEPARATELY_TIMED: whole_replay_wall includes warmup andprofile_rows parsing before record creation, but trace export follows record creation. Decodewindow ends before profile_rows/export. Ledger includes all of them; differences are not pure parsing overhead.",
                  short_noninstrumented_baselines=summary, separate_instrumented_profiles=views,
                  profiling_overhead_vs_short_baseline=overhead, codec_micro=micro_summary,
                  conservative_Amdahl_analysis=amdahl,
                  not_run=dict(noninstrumented512="NOT_RUN_INITIAL_PROFILE_SCOPE", noninstrumented2048="NOT_RUN_INITIAL_PROFILE_SCOPE", optimized_candidate="NOT_RUN", speedup="NOT_RUN", formal_AIME="NOT_RUN", new32Kscience="NOT_RUN"),
                  limitations=["Three actual noninstrumented trials percondition, one separate instrumented profile percondition.128measuredtokens after128warmup only.",
                               "Per-token hostdispatch p50/p95 are not complete-token GPU latency. Same-streamCUDAinterval includes idle gaps; deviceactive union is observed deviceactivitycoverage.",
                               "No sum of nested CPUphase/ATen/device kernel durations is presented as walltime. Wall minus CUDAactive union is not pureCPUdispatch time.",
                               "Micro singlelayer groupedLS fraction cannot be multiplied by24 or used as fullmodel wall fraction.",
                               "No top-operation cause, CaseABC, optimization candidate or STOP decision is automatically assigned."],
                  reducer_source_sha256=sha(Path(__file__)),category_parser_source_sha256=sha(Path(__file__).with_name("chrome_trace_categories.py")))
    kernel_csv=[]
    for r in profiles:
        for a in r["_chrome_category_corrected"]["CUDA_device_activity_costs_by_name"]:
            kernel_csv.append(dict(record_scope="FULL_MODEL_INSTRUMENTED_128", condition=r["condition"], activity_type=a["classification"], name=a["name"], count=a["count"],
                                   summed_device_duration_us=a["summed_duration_us"], per_activity_p50_us=a["duration_summary_us"]["p50"], per_activity_p95_us=a["duration_summary_us"]["p95"],
                                   CPU_total_us="", self_CPU_us="", attributed_device_total_us="", self_attributed_device_us="", warning="Actualdeviceactivities; summed durations not wall and may overlap."))
        for a in CPU_operators(r):
            kernel_csv.append(dict(record_scope="FULL_MODEL_INSTRUMENTED_128", condition=r["condition"], activity_type="CPU_OPERATOR_OR_OWNER_SCOPE", name=a["name"], count=a["count"],
                                   summed_device_duration_us="", per_activity_p50_us="", per_activity_p95_us="", CPU_total_us=a["cpu_total_us"], self_CPU_us=a["self_cpu_us"],
                                   attributed_device_total_us=a["device_total_us"], self_attributed_device_us=a["self_device_us"], warning="Nestedinclusiveowner/ATen attribution; do notsumwithchild rows orkernelactivity rows."))
    for a in stages:
        kernel_csv.append(dict(record_scope="REAL_SINGLE_LAYER_CODEC_STAGE_DIAGNOSTIC", condition="H_DUAL_M_REFERENCE", activity_type="DIAGNOSTIC_STAGE_OWNER_SCOPE", name=a["name"], count=a["count"],
                               summed_device_duration_us="", per_activity_p50_us="", per_activity_p95_us="", CPU_total_us=a["cpu_total_us"], self_CPU_us=a["self_cpu_us"],
                               attributed_device_total_us=a["device_total_us"], self_attributed_device_us=a["self_device_us"], warning="Singlelayergroupeddiagnosis+traceoverhead; not fullmodeltime orwall."))
    # Micro diagnostic has no exported Chrome trace. Keep its raw purported
    # kernel/union fields in RAW_DERIVED_UNCORRECTED only; no false kernel CSV.
    full_csv=[]
    for r in sorted(baseline, key=lambda x:(CONDITIONS.index(x["condition"]),x["repeat"])):
        host=stats(r["per_token_wall_dispatch_seconds"])
        full_csv.append(dict(condition=r["condition"],measurement_stage=r["measurement_stage"],instrumented=False,repeat=r["repeat"],warmup_tokens=128,measured_decode_tokens=128,
                             decode_window_wall_seconds=r["decode_window_wall_seconds"],decode_tokens_per_second=r["decode_tokens_per_second"],whole_replay_wall_seconds=r["whole_replay_wall_seconds"],
                             host_iteration_dispatch_p50_seconds=host["p50"],host_iteration_dispatch_p95_seconds=host["p95"],host_iteration_dispatch_min_seconds=host["min"],host_iteration_dispatch_max_seconds=host["max"],
                             peak_allocated_bytes=r["peak_allocated_bytes"],peak_reserved_bytes=r["peak_reserved_bytes"],persistent_state_bytes=r["persistent_recurrent_state_tensor_bytes"],
                             warning="Hostiterationp50p95notGPUcompletedtokenlatency; wholedecodewindowhasboundarysynchronization; no newpertokenglobalsync."))
    contents={"BASELINE_RUNTIME_PROFILE.json":(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode(),
              "PER_KERNEL_COST.csv":csv_bytes(kernel_csv,list(kernel_csv[0])),
              "FULL_MODEL_BASELINE.csv":csv_bytes(full_csv,list(full_csv[0]))}
    output.mkdir(parents=True,exist_ok=True)
    require(not any((output/name).exists() for name in contents),"Refuse overwriting prioranalysis outputs")
    # All actual data, release and identity checks precede any result or CSV write.
    for name,data in contents.items():
        fd,temp=tempfile.mkstemp(prefix=".phase0_",dir=output)
        with os.fdopen(fd,"wb") as handle:handle.write(data)
        os.rename(temp,output/name)
    print(json.dumps(dict(status=result["status"],output=str(output),noninstrumented_trials=len(baseline),separate_profiles=len(profiles),
                          outputs=[dict(path=str(output/name),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in contents.items()]),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,help="Default root/phase0_analysis; never writes frozen source")
    parser.add_argument("--release-proof",type=Path,help="Independent execution-host proof per REDUCTION_SCHEMA.json")
    parser.add_argument("--verify-live-release",action="store_true",help="On55only: CPU /proc and read-only nvidia-smi check, no CUDA context")
    args=parser.parse_args();root=args.root.resolve();output=(args.output or root/"phase0_analysis").resolve()
    reduce(root,output,args.release_proof,args.verify_live_release)


if __name__=="__main__":
    main()
