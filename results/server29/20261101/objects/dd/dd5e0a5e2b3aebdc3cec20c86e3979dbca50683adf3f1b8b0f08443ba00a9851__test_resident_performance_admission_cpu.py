#!/usr/bin/env python3
"""Stdlib synthetic schema/admission fixtures; no real timing/GPU claims."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    path = root / "src/resident_performance_admission.py"
    spec = importlib.util.spec_from_file_location("_performance_cpu_schema", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    units = json.loads((root / "QUALIFICATION_JOB_MANIFEST.json").read_text())["units"]
    selected = [unit for unit in units if unit["kind"].startswith(("perf_", "profile_"))]
    results, checks = {}, []
    for unit in selected:
        condition, kind = unit["condition"], unit["kind"]
        profile, late = kind.startswith("profile_"), kind.endswith("late")
        warm, measured = unit["performance_window"]["warmup"], unit["performance_window"]["measured"]
        total = warm + measured
        if condition == "LEGACY_H":
            graph = dict(tokens=total, qdq_layer_updates=24 * total, legacy_control=True,
                         QK_rotation_after32head_repeat=True, QDQ_after_full_model_forward=True, H_cached=True)
        else:
            graph = dict(qk_batched_matmul_calls=total * 24, original_gdn_helper_calls=total * 24, qdq_calls=total * 24,
                unique_qk_vectors_rotated=total * 24 * 32, full_token_finite_checks=total, processed_tokens=total,
                chunk_prefill_calls=24, recurrent_decode_calls=(total - 1) * 24, qk_matmul_batches=16, shared_head_repeat=2,
                state_rotation_matmul_calls=0, P_inside_QDQ=False, high_precision_continuation_shadow=False, qdq=True,
                finite_policy="token", state_dtype="FP32", scale_dtype="FP32", packed_INT8_persistent_storage=False,
                original_torch_helper_body_sha256=module.SOURCE_SHA, original_codec_source_sha256=module.CODEC_SHA,
                qk_normalize="original incoming BF16 arithmetic once before FP32 rotation",
                q_scale="original helper 1/sqrt(128) exactly once", next_consumption_full_value_byte_checks=144 if late else 0)
        result = dict(condition=condition, GPU_UUID=unit["gpu_uuid"], fixture_sha256=unit["fixture_sha256"],
            warmup_tokens=warm, measured_tokens=measured, continuous_QDQ_input_forwards=total,
            actual_original_sampler_invoked=True, QDQ_and_finite_checks_included=True,
            observer_fullbytes_disabled_in_measurement=True, legacy_H_control=condition == "LEGACY_H",
            initial_prompt128_window=dict(tokens=128, seconds=2., TPS=64.), peak_allocated_bytes=1000,
            peak_reserved_bytes=2000, audit=graph)
        if condition != "LEGACY_H":
            inputs = list(module.LATE_INPUTS) if late else []
            result["long_sparse_next_consumption"] = dict(input_indices=inputs, checked_layer_inputs=len(inputs) * 24,
                checks_outside_measured_and_prompt128_windows=True, reference_is_actual_previous_encoded_fields_not_reencoding=True,
                rows=[dict(input_index=token, layer=layer) for token in inputs for layer in module.LAYERS])
        if profile:
            result.update(status="PASS_SEPARATE_ACTUAL_CUDA_PROFILE", TPS=None, profiled_TPS_is_not_throughput=True,
                component_CUDA_events={name: dict(calls=24 * measured, total_CUDA_ms=10.) for name in module.COMPONENTS},
                component_profiler_ranges={name: dict(calls=24 * measured, CPU_total_us=1000., CUDA_total_us=1000.) for name in module.COMPONENTS},
                actual_CUDA_kernel_count=800, actual_CUDA_kernels_per_measured_token=800 / measured,
                CUDA_kernel_events=[dict(name="synthetic_not_measured", duration_us=1.) for _ in range(800)],
                wholeState_dense_rotations_observed=0, QK16head_bmm_calls=24 * measured,
                matrix_operator_shapes=[dict(operator="aten::bmm", calls=24 * measured, input_shapes=[[16, 2, 128], [16, 128, 128]])])
        else:
            tps = {"LEGACY_H": 30., "H": 40., "C1": 32., "S3": 31.}[condition]
            result.update(status="PASS_ACTUAL_UNPROFILED_FIXED_INPUT_WINDOW", TPS=tps, seconds=measured / tps,
                ms_per_token=1000. / tps, all60_hit81920_conditional_generation_GPUh=60 * 81920 / tps / 3600,
                perUUID20samples_hit81920_conditional_hours=20 * 81920 / tps / 3600,
                not_free_generation_or81K_speed_guarantee=True)
        results[unit["id"]] = dict(status="PASS", unit=unit, result=result)

    def check(name, predicate):
        if not predicate:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, mutation, *, modify_units=False):
        candidate = copy.deepcopy(units if modify_units else results)
        mutation(candidate)
        try:
            module.audit(root, candidate if modify_units else units, results if modify_units else candidate)
        except (RuntimeError, KeyError):
            checks.append(name)
        else:
            raise AssertionError("Invalid performance admission accepted: " + name)

    passed = module.audit(root, units, results)
    check("complete_actual_schema_passes", passed["status"] == "PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED")
    check("all14_sameGPU_controlled_units_required", len(passed["qualification_performance_units"]) == 14)
    check("no_posthoc_TPS_threshold", passed["arbitrary_numeric_slowdown_threshold"] is None)
    check("profile_not_TPS", len(passed["component_profiles"]) == 6)
    check("measured_ratios_are_diagnostics", passed["actual_comparisons"]["late"]["C1"]["TPS_ratio_to_residentH"] == .8)
    expected = (30 * 81920 / 32. + 30 * 81920 / 31.) / 3600.
    check("conditional_C1_30_S3_30_cost_from_actualTPS", passed["formal60_cap_cost_scenarios"]["late"]["combined60_hit81920_host_generation_GPUh"] == expected)
    check("balanced_threeGPU_cost_disclosed_not_ETA", passed["formal60_cap_cost_scenarios"]["late"]["balanced_three_UUID_conditional_hours"] == expected / 3)
    perf = next(unit["id"] for unit in selected if unit["condition"] == "C1" and unit["kind"] == "perf_late")
    profile = next(unit["id"] for unit in selected if unit["condition"] == "S3" and unit["kind"] == "profile_short")
    legacy = next(unit["id"] for unit in selected if unit["condition"] == "LEGACY_H" and unit["kind"] == "perf_short")
    def value_change(key, value):
        return lambda rows: rows[perf]["result"].__setitem__(key, value)
    def graph_change(key, value):
        return lambda rows: rows[perf]["result"]["audit"].__setitem__(key, value)
    reject("missing_baseline_rejected", lambda rows: rows.pop(legacy))
    reject("missing_method_profile_rejected", lambda rows: rows.pop(profile))
    reject("template_roster_missing_unit_rejected", lambda rows: rows.pop(next(i for i, unit in enumerate(rows) if unit["id"] == profile)), modify_units=True)
    reject("different_UUID_rejected", value_change("GPU_UUID", "foreign"))
    reject("different_fixture_rejected", value_change("fixture_sha256", "0" * 64))
    reject("wrong_window_rejected", value_change("measured_tokens", 64))
    reject("sampler_omitted_rejected", value_change("actual_original_sampler_invoked", False))
    reject("QDQ_finite_omitted_rejected", value_change("QDQ_and_finite_checks_included", False))
    reject("byte_observers_in_timed_window_rejected", value_change("observer_fullbytes_disabled_in_measurement", False))
    reject("nonfinite_TPS_rejected", value_change("TPS", float("nan")))
    reject("zero_TPS_rejected", value_change("TPS", 0.))
    reject("negative_duration_rejected", value_change("seconds", -1.))
    reject("inconsistent_TPS_arithmetic_rejected", value_change("TPS", 35.))
    reject("invalid_memory_rejected", value_change("peak_allocated_bytes", 3000))
    reject("cost_mismatch_rejected", value_change("all60_hit81920_conditional_generation_GPUh", 0.))
    reject("32head_duplicate_rotation_rejected", graph_change("qk_matmul_batches", 32))
    reject("extra_QK_bmm_rejected", graph_change("qk_batched_matmul_calls", 4096 * 48))
    reject("missing_QDQ_rejected", graph_change("qdq_calls", 0))
    reject("fullState_rotation_rejected", graph_change("state_rotation_matmul_calls", 1))
    reject("duplicate_PBR_rejected", graph_change("P_inside_QDQ", True))
    reject("FP_shadow_rejected", graph_change("high_precision_continuation_shadow", True))
    reject("extra_finite_sync_policy_rejected", graph_change("finite_policy", "layer"))
    reject("duplicate_normalization_rejected", graph_change("qk_normalize", "normalize twice"))
    reject("changed_qscale_rejected", graph_change("q_scale", "no scaling"))
    reject("wrong_source_helper_rejected", graph_change("original_torch_helper_body_sha256", "0" * 64))
    reject("missing_late_consumption_rejected", lambda rows: rows[perf]["result"]["long_sparse_next_consumption"]["rows"].pop())
    reject("profile_reported_as_TPS_rejected", lambda rows: rows[profile]["result"].__setitem__("TPS", 50.))
    reject("missing_component_rejected", lambda rows: rows[profile]["result"]["component_CUDA_events"].pop("resident.QDQ"))
    reject("profile_missing_actual_kernels_rejected", lambda rows: rows[profile]["result"].__setitem__("actual_CUDA_kernel_count", 0))
    reject("profile_wrong_bmm_shapes_rejected", lambda rows: rows[profile]["result"]["matrix_operator_shapes"][0].__setitem__("input_shapes", [[32, 2, 128], [32, 128, 128]]))
    reject("profile_wholeState_shape_rejected", lambda rows: rows[profile]["result"]["matrix_operator_shapes"].append(dict(operator="aten::bmm", calls=1, input_shapes=[[16, 128, 128], [16, 128, 128]])))
    check("stdlib_no_torch", "torch" not in sys.modules)
    result = dict(status="PASS_CPU_PERFORMANCE_ADMISSION_SCHEMA_AND_OBJECTIVE_PREDICATE", count=len(checks), checks=checks,
                  performance_admission_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope="Synthetic qualifier-schema fields only, no measured TPS/profiler/9B/GPU evidence; all real58 gates and objective preformal verdict remain required", new_GPU_dispatches=0, new_AIME_samples=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], count=len(checks))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
