#!/usr/bin/env python3
"""Independent CPU preparation/scoring/report for the authorized P1 cancellation.

Never writes original P1 source, SESSION, ledger, finalizer or generation files.
No GPU context/dispatch, old-finisher invocation, trajectory concatenation or
automatic retry. Two score invocations must be separate fresh CPU processes.
Publication/download remain explicitly separate reviewed CPU operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


CONTINUATION = Path("/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022")
P1 = Path("/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022")
TASK = "QWEN_GDN_OSCAR_P1_HUMAN_CANCELLED_PARTIAL_DELIVERY_20261022"
TERMINAL = "HUMAN_CANCELLED_P1_PARTIAL_15_OF_30_C1_S3_TECHNICAL_STOP_ZERO"
P1_SOURCE = "3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14"
P1_PERMIT = "bfd67ac400600efe3680ee8cfc6dafa5059442c52273f1940f3d2d6892b6b604"
JOBS_SHA = "a75e71a560456ad01d36f49d4c936be83b1947ad1699b155262a330646427362"
AUTH_SHA = "d00363479cef8ca7cdea8b288b5db4ca87bccbc0b0fd2d9bf00b21b3fd5fdf44"
SCORER_SHA = "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b"
RAW_SUFFIXES = (".bin", ".npz", ".npy", ".f32", ".pt", ".pth", ".safetensor", ".safetensors", ".ckpt")
LEGAL_UNITS = {
    "aime26_04_P1_seed1", "aime26_15_P1_seed1", "aime26_08_P1_seed1", "aime26_14_P1_seed1", "aime26_25_P1_seed1",
    "aime26_19_P1_seed1", "aime26_27_P1_seed1", "aime26_13_P1_seed1", "aime26_18_P1_seed1",
    "aime26_01_P1_seed1", "aime26_16_P1_seed1", "aime26_09_P1_seed1", "aime26_07_P1_seed1", "aime26_22_P1_seed1", "aime26_03_P1_seed1",
}
CANCELLED_UNITS = {"aime26_23_P1_seed1", "aime26_06_P1_seed1", "aime26_10_P1_seed1"}
ARCHIVE_PROPOSED = "/data/zypan/qwen_oscar_sym32_unified_v2_archive/" + TASK


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save(path, value):
    write_new(path, canonical(value))


def identity():
    raw = Path("/proc/self/stat").read_text()
    return dict(pid=os.getpid(), starttick=int(raw[raw.rfind(")") + 2:].split()[19]),
                cmdline=Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))


def original_bindings():
    require(sha(P1 / "SOURCE_FREEZE.json") == P1_SOURCE and sha(P1 / "GPU_PERMIT.json") == P1_PERMIT
            and sha(P1 / "FORMAL_JOB_MANIFEST.json") == JOBS_SHA, "Original P1 scientific source/permit/roster changed")
    require(not (P1 / "SESSION_RESULT.json").exists() and not (P1 / "FINALIZER_STARTED.json").exists(),
            "Unexpected original session/finalizer record; inspect without fabricating or overwriting")
    sys.path.insert(0, str(P1 / "src"))
    import execution_control as control
    from record_validation import validate
    require(control.verify_sources(P1) == P1_SOURCE and control.verify_external(P1) == 61,
            "Original622 source or61 dependencies changed")
    control.model_guard(P1)
    return control, validate


def own_exit_check(control, stop):
    expected = [stop["original_coordinator"], stop["original_launcher"], *stop["stopped_workers"]]
    require(all(not control.same_process(item) for item in expected), "An exact original cancelled process is still alive")
    original_scripts = {str(P1 / "scripts" / name) for name in
                        ("gpu_unit_worker.py", "gpu_controller.py", "run_offline.sh", "finalize_offline.py")}
    residual = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            current = control.process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if original_scripts.intersection(current["cmdline"]):
            residual.append(current)
    require(not residual, "An original P1 scheduler, worker or finisher still has an actual cmdline")
    own = []
    observation = control.query_gpu()
    for app in observation["apps"]:
        try:
            current = control.process_identity(app["pid"])
        except FileNotFoundError:
            continue
        if str(P1) in " ".join(current["cmdline"]):
            own.append(dict(app=app, identity=current))
    require(not own, "Own original P1 CUDA process remains; new resident foreign jobs are allowed")
    return dict(status="PASS_ORIGINAL_CANCELLED_IDENTITIES_EXITED_NO_OWN_NVML", expected=expected,
                original_root_process_scan=residual, NVML=observation, own=[], unrelated_resident_workers_allowed=True)


def stop_receipt(path, expected_sha):
    require(expected_sha and sha(path) == expected_sha, "Exact original STOP receipt bytes must be explicitly bound")
    stop = load(path)
    require(stop["status"] == "PASS_HUMAN_AUTHORIZED_P1_STOP_EXACT_IDENTITIES_EXITED_OWN_GPU_RELEASED"
            and stop["terminal"] == "USER_CANCELLED_P1_PARTIAL_DO_NOT_RESUME"
            and stop["P1_source_freeze_sha256"] == P1_SOURCE and stop["P1_permit_sha256"] == P1_PERMIT
            and stop["human_authorization_sha256"] == AUTH_SHA, "Original authorized cancellation receipt identity differs")
    require(stop["all_exact_own_processes_exited"] and stop["own_GPU_UUIDs_released"]
            and stop["no_future_original_scheduler_dispatch"] and not stop["original_SESSION_RESULT_fabricated"]
            and not stop["original_ledger_rewritten"], "Cancellation release/preservation evidence incomplete")
    require(stop["original_completed_candidate_count"] == 15 and
            {row["unit"] for row in stop["original_completed_candidates"]} == LEGAL_UNITS,
            "Actual complete15 original intervals differ from the stopped snapshot")
    stopped_units = {row["cmdline"][row["cmdline"].index("--unit") + 1] for row in stop["stopped_workers"]}
    require(stopped_units == CANCELLED_UNITS, "Cancelled in-flight unit roster differs")
    return stop


def checked_payloads(control, validate, stop):
    jobs = {row["id"]: row for row in load(P1 / "FORMAL_JOB_MANIFEST.json")["jobs"]}
    p1_jobs = [job for job in jobs.values() if job["condition"] == "P1"]
    require(len(p1_jobs) == 30 and len(jobs) == 90, "Original P1/C1/S3 formal roster count differs")
    dispatched = {job["id"] for job in p1_jobs if (P1 / "formal" / job["id"] / "DISPATCH.json").exists()}
    require(dispatched == LEGAL_UNITS | CANCELLED_UNITS, "Actual original P1 dispatch roster differs from complete15/cancelled3")
    require(not [job["id"] for job in jobs.values() if job["condition"] in ("C1", "S3")
                 and (P1 / "formal" / job["id"] / "RESULT.json").exists()],
            "Old C1/S3 technical STOP must retain zero formal payloads")
    observed = {job["id"] for job in jobs.values() if job["condition"] == "P1"
                and (P1 / "formal" / job["id"] / "RESULT.json").exists()}
    require(observed == LEGAL_UNITS, "Unexpected original complete payload set; do not guess or include cancelled samples")
    receipt_rows = {row["unit"]: row for row in stop["original_completed_candidates"]}
    records = []
    for unit_id in sorted(LEGAL_UNITS):
        job = jobs[unit_id]
        target = P1 / "formal" / unit_id
        result, interval = target / "RESULT.json", target / "RESOURCE_INTERVAL.json"
        receipt = receipt_rows[unit_id]
        require(sha(result) == receipt["result_sha256"] and sha(interval) == receipt["interval_sha256"],
                "Original result/interval changed after actual cancellation:" + unit_id)
        wrapper, resource = load(result), load(interval)
        require(resource["exit_returncode"] == 0 and resource["release"] == "PASS"
                and resource["identity"] == wrapper["identity"] and not control.same_process(resource["identity"]),
                "Only original true exit0/releasePASS payloads may be scored")
        row = validate(wrapper, job, P1)
        require(row["condition"] == "P1", "Only P1 complete payloads in this cancellation delivery")
        identities = []
        for name in ("RESULT.json", "RESOURCE_INTERVAL.json", "STARTED.json", "DISPATCH.json"):
            path = target / name
            identities.append(dict(path=str(path.relative_to(P1)), bytes=path.stat().st_size, sha256=sha(path)))
        records.append(dict(unit=unit_id, job=job, payload_sha256=wrapper["payload_sha256"],
                            files=identities, generated_token_count=row["generated_token_count"],
                            finish_reason=row["finish_reason"], gpu_uuid=row["gpu_uuid"]))
    require(len(records) == 15, "Actual legal complete payload count differs")
    return records


def original_file_catalog():
    rows = []
    for path in sorted(P1.rglob("*")):
        require(not path.is_symlink(), "Unexpected original P1 symlink; preserve and inspect")
        if path.is_dir():
            continue
        require(path.is_file(), "Unexpected nonregular original P1 asset")
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        require((before.st_size, before.st_ino, before.st_mtime_ns) == (after.st_size, after.st_ino, after.st_mtime_ns),
                "Original P1 artifact still changing after authorized stop:" + str(path))
        relative = str(path.relative_to(P1))
        rows.append(dict(path=relative, archive_path="original_P1/" + relative,
                         bytes=after.st_size, sha256=digest,
                         remote_only_raw=relative.lower().endswith(RAW_SUFFIXES)))
    require(sum(row["bytes"] for row in rows) < 20 * 1024 ** 3, "Original P1 snapshot exceeds20GiB bound")
    return rows


def check_tool(root, expected):
    path = root / "operations/p1_cancelled_delivery.py"
    require(Path(__file__).resolve() == path and expected and sha(path) == expected,
            "This independent CPU tool must have the exact explicitly reviewed SHA")
    manifest = load(root / "operations/P1_CANCELLED_DELIVERY_TOOL_MANIFEST.json")
    rows = [row for row in manifest["files"] if row["path"] == "operations/p1_cancelled_delivery.py"]
    require(len(rows) == 1 and rows[0]["sha256"] == expected and rows[0]["bytes"] == path.stat().st_size,
            "Independent cancellation delivery tool manifest mismatch")


def prepare(args, work):
    require(args.stop_receipt and args.stop_receipt_sha256, "Prepare needs exact original STOP receipt path and SHA")
    require(not work.exists(), "Preparation already attempted; preserve and inspect, no implicit retry")
    work.mkdir(parents=True, exist_ok=False)
    save(work / "PREPARE_STARTED.json", dict(identity=identity(), monotonic=time.monotonic(),
                tool_sha256=args.expected_self_sha256, task_id=TASK, original_root_writes=0, GPU_dispatches=0))
    try:
        stop = stop_receipt(args.stop_receipt, args.stop_receipt_sha256)
        control, validate = original_bindings()
        release = own_exit_check(control, stop)
        records = checked_payloads(control, validate, stop)
        originals = original_file_catalog()
        write_new(work / "evidence/USER_STOP_RECEIPT.json", Path(args.stop_receipt).read_bytes())
        write_new(work / "evidence/HUMAN_CONTINUATION_AUTHORIZATION.json",
                  (args.root / "HUMAN_CONTINUATION_AUTHORIZATION.json").read_bytes())
        require(sha(work / "evidence/HUMAN_CONTINUATION_AUTHORIZATION.json") == AUTH_SHA, "Human stop authorization changed")
        manifest = dict(status="PASS_ACTUAL15_PAYLOADS_ORIGINAL_EXIT0_RELEASE_VALIDATED_READONLY",
                        task_id=TASK, terminal=TERMINAL, original_root=str(P1),
                        original_source_freeze_sha256=P1_SOURCE, original_permit_sha256=P1_PERMIT,
                        original_jobs_sha256=JOBS_SHA, stop_receipt_original_sha256=args.stop_receipt_sha256,
                        human_authorization_sha256=AUTH_SHA, complete=15, originally_planned_P1=30,
                        C1_formal_complete=0, S3_formal_complete=0, records=records,
                        cancelled_units=sorted(CANCELLED_UNITS), never_started_P1_count=12,
                        incomplete_trajectories_score_eligible=False, concatenation_permitted=False,
                        full30_accuracy=None, old_SESSION_fabricated=False, original_root_writes=0)
        save(work / "P1_LEGAL_PAYLOAD_MANIFEST.json", manifest)
        save(work / "ORIGINAL_P1_FULL_FILE_CATALOG.json", dict(status="FROZEN_AFTER_REAL_AUTHORIZED_STOP_ALL_FILES_FRESH_SHA",
                    original_root=str(P1), files=originals, count=len(originals), bytes=sum(x["bytes"] for x in originals),
                    raw_count=sum(x["remote_only_raw"] for x in originals),
                    raw_bytes=sum(x["bytes"] for x in originals if x["remote_only_raw"]),
                    original_source_raw_files=89, original_source_raw_bytes=194895804,
                    original_execution_raw_additional_files_included=True,
                    exclusions=[], nested_basename_metadata_exclusions=False))
        save(work / "P1_CANCELLED_PROCESS_RELEASE_AUDIT.json", release)
        plan = dict(status="CPU_SCORING_READY_PUBLICATION_NOT_RUN", steps=[
            "Run this same frozen tool --phase score --pass-index0 in a fresh CPU process",
            "Run this same frozen tool --phase score --pass-index1 in a different fresh CPU process",
            "Run --phase report only after both processes actually exit and score data match",
            "Main must provide/review once-only independent CPU archive publisher: original_P1/ fullcatalog plus cancelled_delivery/ all final new CPU files",
            "Archive29 needs20GiB budget+100GiB margin, fresh everyfile SHA/bytes and renameat2 NOREPLACE; no originalfinisher/SESSION fabrication",
            "Independent Mac read-only verifier freshchecks bothhosts, original622source61external17modelstat, rawdynamic catalog, exact exitedPIDs/NVML, then downloads nonraw only and verifies localSHA",
        ], proposed_archive_root=ARCHIVE_PROPOSED, publisher_implemented_by_this_tool=False,
                    local_download_implemented_by_this_tool=False, GPU_dispatch_functions=False,
                    no_automatic_retries=True, old_P1_waiter_not_required=True)
        save(work / "CANCELLED_PARTIAL_DELIVERY_PLAN.json", plan)
        summary = dict(status="PASS_ACTUAL15_LEGAL_MANIFEST_READY_FOR_TWO_FRESH_CPU_SCORES", work_root=str(work),
                       manifest_sha256=sha(work / "P1_LEGAL_PAYLOAD_MANIFEST.json"), complete=15, planned_P1=30,
                       original_files=len(originals), original_raw_count=sum(x["remote_only_raw"] for x in originals),
                       publication="NOT_RUN", GPU_dispatches=0)
        save(work / "PREPARE_RESULT.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as error:
        save(work / "PREPARE_FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(),
             original_root_writes=0, no_auto_retry=True))
        raise


def manifest_and_stop(args, work):
    path = work / "P1_LEGAL_PAYLOAD_MANIFEST.json"
    require(args.expected_manifest_sha256 and sha(path) == args.expected_manifest_sha256,
            "CPU scoring/report must bind the independently reviewed exact legal15 manifest")
    manifest = load(path)
    require(manifest["complete"] == 15 and {x["unit"] for x in manifest["records"]} == LEGAL_UNITS,
            "Actual legal subset roster changed")
    stop = stop_receipt(work / "evidence/USER_STOP_RECEIPT.json", manifest["stop_receipt_original_sha256"])
    return manifest, stop


def score(args, work):
    require(args.pass_index in (0, 1), "Two explicitly named independent score passes only")
    prefix = "SCORE_PASS_" + str(args.pass_index)
    require(not (work / (prefix + "_STARTED.json")).exists() and not (work / (prefix + ".json")).exists(),
            "Score pass already attempted; no automatic retry/overwrite")
    save(work / (prefix + "_STARTED.json"), dict(identity=identity(), monotonic=time.monotonic(),
                tool_sha256=args.expected_self_sha256, actual_GPU_dispatches=0))
    try:
        manifest, stop = manifest_and_stop(args, work)
        control, validate = original_bindings()
        own_exit_check(control, stop)
        actual = checked_payloads(control, validate, stop)
        require(actual == manifest["records"], "Legal payload inventory changed before independent score")
        path = P1 / "protocol/aime26_scorer_v4.py"
        require(sha(path) == SCORER_SHA, "Original FrozenV4 source identity changed")
        spec = importlib.util.spec_from_file_location("original_frozen_v4_cancelled_P1", path)
        scorer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scorer
        spec.loader.exec_module(scorer)
        sys.path.insert(0, "/data/zypan/transformers-qwen35/src")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/data/zypan/modelscope_models/Qwen3.5-9B",
                                                 trust_remote_code=True, local_files_only=True)
        gold = {row["problem_id"]: row for row in [json.loads(line) for line in
                (P1 / "protocol/aime26_frozen.jsonl").read_text().splitlines() if line.strip()]}
        results = []
        for record in manifest["records"]:
            wrapper = load(P1 / "formal" / record["unit"] / "RESULT.json")
            row = validate(wrapper, record["job"], P1)
            decoded = tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True)
            require(decoded == row["response"], "Original exact token payload re-decode differs")
            g = gold[row["problem_id"]]
            extracted = scorer.extract_v4(decoded, row["finish_reason"], g["problem"])
            correct = bool(scorer.compare_with_gold(extracted, g["answer"]))
            results.append(dict(sample_id=record["unit"], problem_id=row["problem_id"], condition="P1", seed=1,
                gpu=row["gpu"], gpu_uuid=row["gpu_uuid"], correct=correct, extracted_answer=extracted.normalized_value,
                extraction=extracted.to_dict(), gold=g["answer"], generated_token_count=row["generated_token_count"],
                finish_reason=row["finish_reason"], wall_seconds=row["wall_seconds"], TPS=row["TPS"],
                payload_sha256=wrapper["payload_sha256"], exact_original_token_redecode=True))
        require(len(results) == 15, "Incorrect legal score count")
        checked_payloads(control, validate, stop)
        save(work / (prefix + ".json"), dict(status="PASS_ORIGINAL_FROZEN_V4_EXACT15_LEGAL_P1_PAYLOADS",
             task_id=TASK, terminal=TERMINAL, legal_manifest_sha256=args.expected_manifest_sha256,
             original_source_freeze_sha256=P1_SOURCE, scorer_sha256=SCORER_SHA,
             scores=results, cancelled_samples_scored=0, full30_accuracy=None, actual_GPU_dispatches=0))
        print(json.dumps(dict(status="PASS", score_pass=args.pass_index, completed=15, planned_P1=30,
                              output_sha256=sha(work / (prefix + ".json"))), sort_keys=True), flush=True)
    except BaseException as error:
        save(work / (prefix + "_FAILURE.json"), dict(error=repr(error), traceback=traceback.format_exc(),
             original_root_writes=0, no_auto_retry=True))
        raise


def report(args, work):
    require(not (work / "REPORT_STARTED.json").exists(), "Report already attempted; no overwrite")
    save(work / "REPORT_STARTED.json", dict(identity=identity(), monotonic=time.monotonic(), original_root_writes=0))
    try:
        manifest, stop = manifest_and_stop(args, work)
        control, validate = original_bindings()
        release = own_exit_check(control, stop)
        checked_payloads(control, validate, stop)
        started = [load(work / f"SCORE_PASS_{i}_STARTED.json")["identity"] for i in (0, 1)]
        require(started[0] != started[1] and all(not control.same_process(x) for x in started),
                "Two score passes must be fresh processes and truly exited")
        a, b = [load(work / f"SCORE_PASS_{i}.json") for i in (0, 1)]
        require(a == b and len(a["scores"]) == 15, "Independent originalFrozenV4 CPU score data differ")
        save(work / "FROZEN_V4_PARTIAL_SCORES.json", a)
        scores = a["scores"]
        correct = sum(x["correct"] for x in scores)
        lengths = dict(N=15, total_tokens=sum(x["generated_token_count"] for x in scores),
                       mean_tokens=statistics.mean(x["generated_token_count"] for x in scores),
                       median_tokens=statistics.median(x["generated_token_count"] for x in scores),
                       EOS=sum(x["finish_reason"] == "eos" for x in scores),
                       length=sum(x["finish_reason"] == "length" for x in scores),
                       completed_sample_wall_seconds=sum(x["wall_seconds"] for x in scores),
                       original_resource_ledger_sha256=sha(P1 / "RESOURCE_LEDGER.json"),
                       original_append_only_ledger_sha256=sha(P1 / "GPU_RESOURCE_LEDGER.jsonl"),
                       cancelled_sample_intervals_not_fabricated=True, exact_physical_GPUh="UNRESOLVED")
        save(work / "PARTIAL_LENGTH_EOS_AND_COST.json", lengths)
        with (work / "P1_PARTIAL_PER_PROBLEM_RESULTS.csv").open("x", newline="") as handle:
            fields = ["sample_id", "problem_id", "correct", "extracted_answer", "generated_token_count", "finish_reason", "gpu_uuid", "wall_seconds", "TPS"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in scores:
                writer.writerow({key: row[key] for key in fields})
        table = "\n".join(f"|{x['problem_id']}|{'正确' if x['correct'] else '错误'}|{x['generated_token_count']}|{x['finish_reason']}|" for x in scores)
        text = f"""# P1人工取消：部分科学交付

终态：`{TERMINAL}`。P1原计划30条，真实合法完成 **15/30**，其中答对 **{correct}/15**。
两份独立fresh CPU原FrozenV4评分及同一token payload重新decode完全一致。
这是已完成子集的部分结果；**完整30题准确率未定义**，未完成题没有按错误填充，也没有补跑。
C1/S3此前数值资格技术STOP保留，各0条正式生成，不能将其写为0%准确率。

|题号|结果|生成token|结束原因|
|---|---|---:|---|
{table}

三条当时运行的样本 `{', '.join(sorted(CANCELLED_UNITS))}` 由人类明确要求停止。
这些部分轨迹及TOKEN_CHUNKS原样保留，不评分、不补EOS、不缩短cap、不续跑、不拼接。
另12条P1从未派发。取消是资源优先级决定，不是按正确率、State误差或答案内容筛选。

P1原622冻结源、旧U/FP资格失败、原许可/dispatch/STARTED/资源账本及所有真实技术失败均只读保留。
本交付不创建原SESSION_RESULT或FINALIZER_STARTED，不调用原finisher。
停止操作中最初的pidfd缺失、job-control干扰和控制工具失败均保留为独立操作证据；不掩盖为科学PASS。

模型State持久载体仍是QDQ后的FP32约48MiB，不是物理packed INT8。
seed1、小N、已观察AIME和不同GPU/运行时间点的限制保留，不能由本15题子集推断PBR整体因果效果。
实际成本使用原同主机monotonic账本和真实停止观测；未完成区间不伪造exit0，EXACT_GPUh仍UNRESOLVED。

本报告只说明CPU部分评分完成。独立29逐文件SHA/bytes、原子NOREPLACE归档和Mac非raw交付验收仍须真实执行后单独给PASS。
"""
        write_new(work / "FINAL_PARTIAL_SCIENTIFIC_REPORT.md", text.encode())
        save(work / "CPU_PARTIAL_SCORE_AND_REPORT_RESULT.json", dict(status="PASS_TWO_FRESH_CPU_PARTIAL15_SCORE_REPORT_ONLY",
             task_id=TASK, terminal=TERMINAL, legal_complete=15, planned_P1=30, correct_on_completed15=correct,
             full30_accuracy=None, C1_S3_formal_counts=dict(C1=0, S3=0), incomplete_scored=0,
             score_processes=started, process_release=release, archive_verification="NOT_RUN", local_delivery="NOT_RUN"))
        print(json.dumps(dict(status="PASS_CPU_PARTIAL_REPORT_ONLY", completed=15, planned_P1=30,
                             correct_on_completed15=correct, full30_accuracy=None, archive="NOT_RUN")), flush=True)
    except BaseException as error:
        save(work / "REPORT_FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(), no_auto_retry=True))
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("plan", "prepare", "score", "report"), default="plan")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-root", type=Path)
    ap.add_argument("--expected-self-sha256")
    ap.add_argument("--stop-receipt", type=Path)
    ap.add_argument("--stop-receipt-sha256")
    ap.add_argument("--expected-manifest-sha256")
    ap.add_argument("--pass-index", type=int)
    args = ap.parse_args(argv)
    if args.phase == "plan":
        print(json.dumps(dict(task_id=TASK, phases=dict(prepare="IMPLEMENTED_NOT_RUN", score="IMPLEMENTED_NOT_RUN_TWO_FRESH_REQUIRED",
              report="IMPLEMENTED_NOT_RUN", archive="MAIN_REVIEWED_CPU_PUBLISHER_REQUIRED_NOT_RUN",
              local_delivery="INDEPENDENT_READONLY_VERIFIER_REQUIRED_NOT_RUN"),
              legal_unit_roster=sorted(LEGAL_UNITS), cancelled_unit_roster=sorted(CANCELLED_UNITS),
              original_source_raw_count=89, original_source_raw_bytes=194895804,
              proposed_archive_root=ARCHIVE_PROPOSED, original_P1_root_writes=0, GPU_launch_functions=False), indent=2))
        return 0
    args.root = args.root.resolve()
    require(args.root == CONTINUATION, "CPU cancellation tool only accepts the exact continuation operations root")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("USE_TORCH") == "0"
            and os.environ.get("USE_TF") == "0" and sys.dont_write_bytecode,
            "Run fresh CPU process with CUDA_VISIBLE_DEVICES='', USE_TORCH=0, USE_TF=0 and Python -B")
    check_tool(args.root, args.expected_self_sha256)
    require(sha(args.root / "HUMAN_CONTINUATION_AUTHORIZATION.json") == AUTH_SHA, "Original new human stop authorization changed")
    work = (args.work_root or args.root / "operations/P1_CANCELLED_CPU_DELIVERY").resolve()
    require(work.is_relative_to(args.root / "operations") and work != args.root / "operations",
            "New CPU delivery outputs must stay in independent continuation operations subroot")
    if args.phase == "prepare":
        prepare(args, work)
    elif args.phase == "score":
        score(args, work)
    else:
        report(args, work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
