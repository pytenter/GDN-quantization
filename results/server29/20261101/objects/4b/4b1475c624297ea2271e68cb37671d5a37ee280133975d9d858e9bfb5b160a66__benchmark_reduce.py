"""CPU-only timing reduction of one declared H-only measurement, never score AIME."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, obj):
    with Path(path).open("x") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    auth = read(root / "AUTHORIZATION.json")
    assert auth["status"] == "PASS" and auth["allowed_mode"] == "throughput"
    assert sha(root / "PLAN.json") == auth["plan_sha256"]
    assert sha(root / "SOURCE_MANIFEST.json") == auth["SOURCE_MANIFEST_sha256"]
    for entry in read(root / "SOURCE_MANIFEST.json")["execution_sources"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
    plan = read(root / "PLAN.json")
    assert auth["task_id"] == plan["task_id"] == "QWEN_GDN_H_DUALM_AIME26_THROUGHPUT_20261017_V1"
    assert read(root / "COORDINATOR_RESULT.json")["status"] == "PASS"
    ledger = read(root / "GPU_RESOURCE_LEDGER.json")
    assert ledger["all_GPUs_released"] and len(ledger["dispatches"]) == 4
    assert not (root / "FIRST_FAILURE.json").exists()
    for job in ledger["dispatches"]:
        assert job["process_exited"] and job["NVML_released"] and job["returncode"] == 0
        p = Path("/proc") / str(job["pid"]) / "stat"
        if p.exists():
            stat = p.read_text()
            assert stat[stat.rfind(")") + 2:].split()[19] != job["identity"]["start_tick"], "Tracked process still alive"
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    tracked = {job["pid"] for job in ledger["dispatches"]}
    assert not any(int(line.split(",")[0]) in tracked for line in apps.splitlines() if line.strip())
    rows = []
    sources = []
    for gpu in range(4):
        result_path = root / "outputs" / ("worker" + str(gpu)) / "RESULT.json"
        result = read(result_path)
        assert result["status"] == "PASS" and result["method"] == "H_DUALM"
        raw_path = root / "outputs" / ("worker" + str(gpu)) / "RAW_BENCHMARK_PREFIX.json"
        assert result["raw_sha256"] == sha(raw_path) and result["raw_bytes"] == raw_path.stat().st_size
        assert result["task_id"] == plan["task_id"] and result["RUN_ID"] == plan["RUN_ID"]
        assert result["benchmark_plan_sha256"] == auth["plan_sha256"]
        assert result["benchmark_source_manifest_sha256"] == auth["SOURCE_MANIFEST_sha256"]
        assert result["authorization_sha256"] == sha(root / "AUTHORIZATION.json")
        record = read(raw_path)
        original_payload = record.pop("output_payload_sha256")
        computed = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert computed == original_payload, "Raw prefix payload changed"
        assert record["task_id"] == plan["task_id"] and record["RUN_ID"] == plan["RUN_ID"]
        assert record["original_generation_status"] == "COMPLETE" and record["status"] == "BENCHMARK_COMPLETE_PREFIX"
        assert record["benchmark_plan_sha256"] == auth["plan_sha256"]
        assert record["benchmark_source_manifest_sha256"] == auth["SOURCE_MANIFEST_sha256"]
        assert record["authorization_sha256"] == sha(root / "AUTHORIZATION.json")
        assert record["runtime_error"] is None and record["method"] == "H_DUALM"
        n = record["output_tokens"]
        assert n == len(record["generated_token_ids"]) and 0 < n <= plan["output_cap"]
        assert record["gpu"]["UUID"] == plan["gpu_uuid_map"][str(gpu)]
        assert record["input_ids_hash"] == plan["units"][gpu]["input_ids_hash"]
        assert record["seed"] == plan["units"][gpu]["seed"] and record["canonical_index"] == plan["units"][gpu]["canonical_index"]
        q = record["quantizer_audit"]
        h = record["hadamard_audit"]
        assert q["calls"] == n and q["layer_writebacks"] == 24 * n and q["next_consumption_checks"] == 24 * (n - 1)
        assert h["calls"] == 24 * n and h["phase_counts"] == {"chunk_prefill": 24, "recurrent_decode": 24 * (n - 1)}
        assert record["generation_config"]["max_new_tokens"] == plan["output_cap"]
        events = []
        log = root / ("gpu" + str(gpu) + ".log")
        for line in log.read_bytes().splitlines():
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") == "PROGRESS":
                assert event["method"] == "H_DUALM" and event["canonical_index"] == plan["units"][gpu]["canonical_index"]
                events.append(event)
        first = next((event for event in events if event["output_tokens"] == plan["warmup_tokens"]), None)
        last = events[-1] if events else None
        steady_n = last["output_tokens"] - first["output_tokens"] if first and last else 0
        steady_seconds = last["wall_seconds"] - first["wall_seconds"] if steady_n else None
        windows = [(z["output_tokens"] - a["output_tokens"]) / (z["wall_seconds"] - a["wall_seconds"])
                   for a, z in zip(events, events[1:]) if a["output_tokens"] >= plan["warmup_tokens"]]
        rows.append(dict(gpu=gpu, UUID=record["gpu"]["UUID"], problem_id=record["problem_id"], seed=record["seed"],
                         actual_output_tokens=n, finish_reason=record["finish_reason"], generation_seconds=record["latency_s"],
                         generation_tokens_per_second=n / record["latency_s"], warmup_generated_tokens=plan["warmup_tokens"],
                         measured_after_warmup_tokens=steady_n, measured_after_warmup_seconds=steady_seconds,
                         after_warmup_tokens_per_second=steady_n / steady_seconds if steady_n else None,
                         median_128token_window_rate=statistics.median(windows) if windows else None,
                         coverage="FULL_DECLARED_PREFIX" if n == plan["output_cap"] else "EARLY_EOS_NO_RETRY",
                         forward_wall_seconds=record["forward_wall_seconds"],
                         QDQ_and_consumption_audit_wall_seconds=record["qdq_and_consumption_audit_wall_seconds"],
                         peak_cuda_allocated_bytes=record["peak_allocated_bytes"], peak_cuda_reserved_bytes=record["peak_reserved_bytes"],
                         QDQ_layer_writebacks=q["layer_writebacks"], next_consumption_checks=q["next_consumption_checks"], H_calls=h["calls"],
                         raw_prefix_sha256=sha(raw_path), log_sha256=sha(log)))
        sources.extend([dict(path=str(p.relative_to(root)), sha256=sha(p), bytes=p.stat().st_size) for p in (raw_path, result_path, log)])
    all_full = all(row["coverage"] == "FULL_DECLARED_PREFIX" for row in rows)
    result = dict(status="COMPLETE_AND_STOP", task_id=plan["task_id"], RUN_ID=plan["RUN_ID"], method="H_DUALM",
                  scope="HUMAN_REQUESTED_4GPU_SHORT_AIME_THROUGHPUT_ONLY_NOT_FORMAL_ACCURACY",
                  all_GPUs_released=True, declared_generated_token_cap_per_worker=plan["output_cap"],
                  original_formal_cap=plan["original_formal_cap"], all_four_full_declared_prefixes=all_full,
                  total_sampled_tokens=sum(row["actual_output_tokens"] for row in rows), per_GPU=rows,
                  weighted_single_GPU_generation_tokens_per_second=sum(row["actual_output_tokens"] for row in rows) / math.fsum(row["generation_seconds"] for row in rows),
                  weighted_single_GPU_after_warmup_tokens_per_second=sum(row["measured_after_warmup_tokens"] for row in rows) / math.fsum(row["measured_after_warmup_seconds"] or 0 for row in rows) if any(row["measured_after_warmup_seconds"] for row in rows) else None,
                  generation_rate_scope="Original generate latency incl prefill,QDQ,finite/consumption checks,sampling and finaltext decoding, excludes model loading and final cache cleanup.",
                  after_warmup_rate_scope="Difference of original PROGRESS monotonic counters from token128 to last128boundary; includes nativeforward+QDQ+sampling/checks/logging, not pureCUDA kernel rate.",
                  model_load_and_process_cost="GPU_RESOURCE_LEDGER dispatch through actual exit/NVML release; separate from generation rate.",
                  GPU_resource_ledger=ledger, sources=sources,
                  limitations=["Only declared2048-generated-token AIME prefixes; not81K long-context speed/stability or complete-question accuracy.",
                               "Four fixed question/seed units covering two questions and two seeds, not a large statistical performance benchmark.",
                               "No STEPQuant/FP/baseline generation was executed by this new benchmark.",
                               "Both H_DualM scalesFP16 and reconstructed persistentFP32 state carrier unchanged; not physicalpackedINT8 inference.",
                               "Old paired120run remains USER_REQUESTED_STOP/incomplete, oneSTEP complete unscored; no benchmarkprefix is backfilled into it."])
    save_new(root / "THROUGHPUT_RESULTS.json", result)
    lines = ["# Hadamard + Dual-M AIME26 吞吐测试", "", "状态：COMPLETE_AND_STOP。四卡已真实退出并释放。", "",
             "本次仅测固定 AIME 输入的短生成前缀，未评分，不构成81920长度完整评测。", "",
             "| GPU | 实际生成token | 生成耗时秒 | 含prefill token/s | 预热后token/s |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        steady = "未建立" if row["after_warmup_tokens_per_second"] is None else f"{row['after_warmup_tokens_per_second']:.3f}"
        lines.append(f"| {row['gpu']} | {row['actual_output_tokens']} | {row['generation_seconds']:.3f} | {row['generation_tokens_per_second']:.3f} | {steady} |")
    lines.extend(["", f"单卡加权生成速度：{result['weighted_single_GPU_generation_tokens_per_second']:.3f} token/s。", "",
                  "预热后指标从原日志token128计时点开始，仍包含H变换、Dual-M、采样及安全检查，不能称为纯CUDA算子速度。", "",
                  "完整结果、源身份、实际GPU消耗和限制见 THROUGHPUT_RESULTS.json。原120条对照已由用户主动停止，历史结果和STOP保留。", ""])
    with (root / "BENCHMARK_REPORT.md").open("x") as handle:
        handle.write("\n".join(lines))
    files = []
    excluded_names = {"SHA256_MANIFEST.json", "SHA256SUMS.txt", "reducer.log", "ROOT.json", "HANDOFF.md"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name not in excluded_names:
            files.append(dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size))
    save_new(root / "SHA256_MANIFEST.json", dict(status="SEALED_BENCHMARK_SMALL_ASSETS", files=files,
             exclusions="Self/checksumfile, live reducer stdout, mutable ROOT/HANDOFF, Python bytecode caches"))
    with (root / "SHA256SUMS.txt").open("x") as handle:
        handle.write("\n".join(row["sha256"] + "  " + row["path"] for row in files) + "\n")
    print(json.dumps(dict(status="COMPLETE_AND_STOP", all_GPUs_released=True,
                         single_GPU_tokens_per_second=result["weighted_single_GPU_generation_tokens_per_second"],
                         result_sha256=sha(root / "THROUGHPUT_RESULTS.json")), ensure_ascii=False))


if __name__ == "__main__":
    main()
