"""Pre-frozen, stdlib-only performance admission over genuine qualifier receipts.

The user supplied no FPS/slowdown/GPU-hour cutoff. Admission therefore requires
the complete controlled measurements and objective efficient execution graph,
and discloses measured cost; speed ratios are diagnostic, never post-hoc gates.
This module never dispatches work, chooses candidates or changes mathematics.
"""
from __future__ import annotations

import math
from pathlib import Path

LAYERS = tuple(i for i in range(32) if i % 4 != 3)
CONDITIONS = ("LEGACY_H", "H", "C1", "S3")
COMPONENTS = {"resident.QK", "resident.GDN", "resident.QDQ"}
LATE_INPUTS = (129, 511, 1023, 2047, 3071, 3967)
SOURCE_SHA = "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a"
CODEC_SHA = "110a3e928ffe9d81fa2db6fcfb0fecdfd39bd478dde7e41dc9d9209807c731de"


def require(ok, message):
    if not ok:
        raise RuntimeError("Preformal performance admission: " + message)


def number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def same_number(left, right):
    return number(left) and number(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def audit(root, units, results):
    """Return a documented verdict or raise before any formal authorization.

``units`` is the frozen58-unit list. ``results`` is a mapping unit-id -> actual
PASS wrapper (status/unit/result), already independently SHA/exit/release
validated by execution control. All14 performance/profile units are mandatory.
The function reads no results/answers and performs no IO or execution.
"""
    require(isinstance(results, dict), "results must map frozen unit IDs to actual wrappers")
    selected = [unit for unit in units if unit["kind"].startswith(("perf_", "profile_"))]
    expected = {(condition, kind) for condition in CONDITIONS for kind in
                ("perf_short", "perf_late", "profile_short", "profile_late")
                if condition != "LEGACY_H" or kind.startswith("perf_")}
    pairs = {(unit["condition"], unit["kind"]) for unit in selected}
    require(len(selected) == len(pairs) == 14 and pairs == expected,
            "the fixed LegacyH/H/C1/S3 short/late and resident component roster differs")
    require(len({unit["gpu_uuid"] for unit in selected}) == 1
            and {unit["gpu"] for unit in selected} == {1}
            and len({unit["fixture_sha256"] for unit in selected}) == 1
            and len({unit["fixture"] for unit in selected}) == 1,
            "controlled comparisons require the same actual UUID and nonAIME fixture")
    measured, profiles, evidence = {}, {}, []
    for unit in selected:
        wrapper = results.get(unit["id"])
        require(isinstance(wrapper, dict) and wrapper["status"] == "PASS" and wrapper["unit"] == unit,
                "a required genuine qualification wrapper is missing or mismatched")
        value = wrapper["result"]
        kind, condition = unit["kind"], unit["condition"]
        profile, late = kind.startswith("profile_"), kind.endswith("late")
        warmup, window = (3968 if late else 128), (8 if profile else 128)
        total = warmup + window
        require(unit["performance_window"] == dict(warmup=warmup, measured=window)
                and unit["tokens"] == total and unit["frozen_fixture_tokens"] == 32768
                and unit["prefix_only_no_cycle"] is True,
                "the frozen controlled measurement window or fixture prefix changed")
        require(value["condition"] == condition and value["GPU_UUID"] == unit["gpu_uuid"]
                and value["fixture_sha256"] == unit["fixture_sha256"]
                and value["warmup_tokens"] == warmup and value["measured_tokens"] == window
                and value["continuous_QDQ_input_forwards"] == total,
                "actual timing identity/window differs from its frozen unit")
        require(value["actual_original_sampler_invoked"] is True
                and value["QDQ_and_finite_checks_included"] is True
                and value["observer_fullbytes_disabled_in_measurement"] is True
                and value["legacy_H_control"] is (condition == "LEGACY_H"),
                "timed path omits original sampling/QDQ/finite checks or includes full-byte observers")
        prompt = value["initial_prompt128_window"]
        require(prompt["tokens"] == 128 and number(prompt["seconds"], positive=True)
                and same_number(prompt["TPS"], 128 / prompt["seconds"]), "actual first128 prompt timing is missing")
        require(number(value["peak_allocated_bytes"], positive=True)
                and number(value["peak_reserved_bytes"], positive=True)
                and value["peak_reserved_bytes"] >= value["peak_allocated_bytes"], "actual peak VRAM is invalid")
        graph = value["audit"]
        if condition == "LEGACY_H":
            require(graph["tokens"] == total and graph["qdq_layer_updates"] == total * 24
                    and graph["legacy_control"] is True and graph["QK_rotation_after32head_repeat"] is True
                    and graph["QDQ_after_full_model_forward"] is True and graph["H_cached"] is True,
                    "the historical H control is not its documented unchanged comparison graph")
        else:
            for field in ("qk_batched_matmul_calls", "original_gdn_helper_calls", "qdq_calls"):
                require(graph[field] == total * 24, "extra/missing resident operation: " + field)
            require(graph["unique_qk_vectors_rotated"] == total * 24 * 32
                    and graph["full_token_finite_checks"] == total and graph["processed_tokens"] == total
                    and graph["chunk_prefill_calls"] == 24 and graph["recurrent_decode_calls"] == (total - 1) * 24
                    and graph["qk_matmul_batches"] == 16 and graph["shared_head_repeat"] == 2,
                    "Q/K must rotate16 unique heads once before repeat; one full-token finite verdict")
            require(graph["state_rotation_matmul_calls"] == 0 and graph["P_inside_QDQ"] is False
                    and graph["high_precision_continuation_shadow"] is False and graph["qdq"] is True
                    and graph["finite_policy"] == "token" and graph["state_dtype"] == "FP32"
                    and graph["scale_dtype"] == "FP32" and graph["packed_INT8_persistent_storage"] is False
                    and graph["original_torch_helper_body_sha256"] == SOURCE_SHA
                    and graph["original_codec_source_sha256"] == CODEC_SHA
                    and graph["qk_normalize"] == "original incoming BF16 arithmetic once before FP32 rotation"
                    and graph["q_scale"] == "original helper 1/sqrt(128) exactly once",
                    "forbidden State transforms/duplicate PBR/FP shadow/normalization/scaling or altered helper")
            sparse = value["long_sparse_next_consumption"]
            expected_inputs = list(LATE_INPUTS) if late else []
            require(sparse["input_indices"] == expected_inputs
                    and sparse["checked_layer_inputs"] == len(expected_inputs) * 24
                    and graph["next_consumption_full_value_byte_checks"] == len(expected_inputs) * 24
                    and sparse["checks_outside_measured_and_prompt128_windows"] is True
                    and sparse["reference_is_actual_previous_encoded_fields_not_reencoding"] is True,
                    "full-byte consumption evidence is absent or contaminates timed windows")
            require({(row["input_index"], row["layer"]) for row in sparse["rows"]}
                    == {(token, layer) for token in expected_inputs for layer in LAYERS}
                    and len(sparse["rows"]) == len(expected_inputs) * 24,
                    "late warmup consumption does not cover the fixed sparse positions/layers")
        if profile:
            require(value["status"] == "PASS_SEPARATE_ACTUAL_CUDA_PROFILE"
                    and value["TPS"] is None and value["profiled_TPS_is_not_throughput"] is True,
                    "profiler overhead must not be reported as genuine throughput")
            for name in ("component_CUDA_events", "component_profiler_ranges"):
                require(set(value[name]) == COMPONENTS and all(row["calls"] == window * 24 for row in value[name].values()),
                        "actual QK/GDN/QDQ component coverage is incomplete")
            require(all(number(row["total_CUDA_ms"]) for row in value["component_CUDA_events"].values())
                    and all(number(row["CPU_total_us"]) and number(row["CUDA_total_us"]) for row in value["component_profiler_ranges"].values()),
                    "actual component duration is nonfinite or negative")
            require(type(value["actual_CUDA_kernel_count"]) is int and value["actual_CUDA_kernel_count"] > 0
                    and value["actual_CUDA_kernel_count"] == len(value["CUDA_kernel_events"])
                    and same_number(value["actual_CUDA_kernels_per_measured_token"], value["actual_CUDA_kernel_count"] / window),
                    "actual GPU kernel launch count is absent or inconsistent")
            require(value["wholeState_dense_rotations_observed"] == 0 and value["QK16head_bmm_calls"] == window * 24,
                    "profile observes a full-State transform or lacks unique16-head rotations")
            operations = value["matrix_operator_shapes"]
            calls = sum(row["calls"] for row in operations if row["operator"] == "aten::bmm"
                        and row["input_shapes"][:2] == [[16, 2, 128], [16, 128, 128]])
            forbidden = [row for row in operations if len(row["input_shapes"]) >= 2
                         and len(row["input_shapes"][0]) >= 2 and len(row["input_shapes"][1]) >= 2
                         and row["input_shapes"][0][-2:] == [128, 128] and row["input_shapes"][1][-2:] == [128, 128]]
            require(calls == window * 24 and not forbidden, "actual profiler matrix shapes contradict resident16-head execution")
            profiles[(condition, "late" if late else "short")] = dict(unit=unit["id"], component_CUDA_events=value["component_CUDA_events"],
                actual_CUDA_kernel_count=value["actual_CUDA_kernel_count"], kernels_per_token=value["actual_CUDA_kernels_per_measured_token"],
                QK16head_bmm_calls=calls, wholeState_dense_rotations_observed=0)
        else:
            require(value["status"] == "PASS_ACTUAL_UNPROFILED_FIXED_INPUT_WINDOW"
                    and number(value["seconds"], positive=True) and number(value["TPS"], positive=True)
                    and same_number(value["TPS"], window / value["seconds"])
                    and same_number(value["ms_per_token"], value["seconds"] * 1000 / window)
                    and value["not_free_generation_or81K_speed_guarantee"] is True,
                    "actual unprofiled TPS/window is invalid")
            require(same_number(value["all60_hit81920_conditional_generation_GPUh"], 60 * 81920 / value["TPS"] / 3600)
                    and same_number(value["perUUID20samples_hit81920_conditional_hours"], 20 * 81920 / value["TPS"] / 3600),
                    "the actualTPS-based cap-cost calculation differs")
            measured[(condition, "late" if late else "short")] = dict(unit=unit["id"], TPS=value["TPS"], seconds=value["seconds"],
                initial_prompt128=value["initial_prompt128_window"], peak_allocated_bytes=value["peak_allocated_bytes"],
                peak_reserved_bytes=value["peak_reserved_bytes"])
        evidence.append(unit["id"])
    comparisons, costs = {}, {}
    for window in ("short", "late"):
        comparisons[window] = {condition: dict(**measured[(condition, window)],
              TPS_ratio_to_legacyH=measured[(condition, window)]["TPS"] / measured[("LEGACY_H", window)]["TPS"],
              TPS_ratio_to_residentH=measured[(condition, window)]["TPS"] / measured[("H", window)]["TPS"])
              for condition in CONDITIONS}
        hours = {condition: 30 * 81920 / measured[(condition, window)]["TPS"] / 3600 for condition in ("C1", "S3")}
        costs[window] = dict(per_condition30_hit81920_host_generation_GPUh=hours,
             combined60_hit81920_host_generation_GPUh=sum(hours.values()),
             balanced_three_UUID_conditional_hours=sum(hours.values()) / 3,
             excludes_model_loading_qualification_archive_cleanup=True,
             conditional_nonAIME_fixed_input_scenario_not_ETA_or81K_speed_guarantee=True)
    return dict(status="PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED",
         execution_root=str(Path(root)), qualification_performance_units=evidence, actual_same_GPU_uuid=selected[0]["gpu_uuid"],
         same_nonAIME_fixture_sha256=selected[0]["fixture_sha256"], actual_comparisons=comparisons,
         component_profiles={condition + "_" + window: value for (condition, window), value in profiles.items()},
         formal60_cap_cost_scenarios=costs, no_wholeState_rotation=True, unique_QK_heads=16,
         old_F32_error_or_accuracy_not_used_as_performance_gate=True, arbitrary_numeric_slowdown_threshold=None,
         rationale="The user specified no FPS/slowdown/GPU-time cap. Accept only complete sameGPU measurements and the objective efficient resident graph: once16-head QK, original helper/QDQ once perlayer, deferred finite verdict, no fullState conversion/duplicate PBR/FP shadow/full-byte observers in timing. Report measured ratios/components/kernels/VRAM and cap costs without post-result selection or a new threshold.",
         mathematical_GPU_and_QDQ_gates_still_required_separately=True)
