#!/usr/bin/env python3
"""Coordinate non-overlapping Ling 256K Value-Hadamard helper workers.

Each worker owns one already-running SGLang endpoint.  Work is claimed with an
atomic directory creation on a shared filesystem.  Completed records may live
in several provenance directories; they are never copied, rewritten, or
overwritten by this coordinator.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


METHOD = "int8_r128_value_h"
STAGE = "formal256k"
SEED = 1


def record_path(root: Path, problem_id: str) -> Path:
    return root / f"{problem_id}_seed{SEED}.json"


def valid_record(path: Path) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    condition = row.get("condition") or row.get("configuration") or row.get("method")
    accepted = {METHOD, "LING_INT8_R128_VALUE_HADAMARD"}
    return bool(row.get("successful_sample")) and condition in accepted


def completed_source(problem_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = record_path(root, problem_id)
        if valid_record(candidate):
            return candidate
    return None


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def try_claim(claim_root: Path, problem_id: str, worker_id: str, gpu: int) -> Path | None:
    claim = claim_root / problem_id
    try:
        claim.mkdir(parents=False)
    except FileExistsError:
        return None
    write_json_atomic(
        claim / "owner.json",
        {
            "problem_id": problem_id,
            "worker_id": worker_id,
            "physical_gpu": gpu,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_unix_time": time.time(),
        },
    )
    return claim


def release_claim(claim: Path) -> None:
    owner = claim / "owner.json"
    try:
        owner.unlink()
    except FileNotFoundError:
        pass
    try:
        claim.rmdir()
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--source-record-dir", action="append", default=[])
    parser.add_argument("--first-problem", type=int, default=1)
    parser.add_argument("--last-problem", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-failures-per-problem", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    records = output / "records" / STAGE / METHOD
    roots = [records] + [Path(item).resolve() for item in args.source_record_dir]
    claims = output / "coordination" / "claims"
    logs = output / "coordination" / "worker_logs" / args.worker_id
    claims.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}
    ids = [f"aime26_{index:02d}" for index in range(args.first_problem, args.last_problem + 1)]

    write_json_atomic(
        output / "coordination" / f"worker_{args.worker_id}.json",
        {
            "worker_id": args.worker_id,
            "physical_gpu": args.physical_gpu,
            "base_url": args.base_url,
            "source_record_dirs": [str(path) for path in roots],
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "status": "RUNNING",
            "started_unix_time": time.time(),
        },
    )

    while True:
        incomplete = [problem_id for problem_id in ids if completed_source(problem_id, roots) is None]
        if not incomplete:
            break
        acquired = False
        for problem_id in incomplete:
            if failures.get(problem_id, 0) >= args.max_failures_per_problem:
                continue
            claim = try_claim(claims, problem_id, args.worker_id, args.physical_gpu)
            if claim is None:
                continue
            acquired = True
            if completed_source(problem_id, roots) is not None:
                release_claim(claim)
                continue
            log_path = logs / f"{problem_id}.attempt{failures.get(problem_id, 0) + 1}.log"
            command = [
                args.python,
                args.runner,
                "--base-url", args.base_url,
                "--model", args.model,
                "--dataset", args.dataset,
                "--output-dir", str(output),
                "--config-json", args.config_json,
                "--method", METHOD,
                "--stage", STAGE,
                "--physical-gpu", str(args.physical_gpu),
                "--parallel-instance-id", args.worker_id,
                "--problem-ids", problem_id,
                "--resume",
            ]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("COMMAND=" + json.dumps(command) + "\n")
                handle.flush()
                return_code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
            if return_code == 0 and valid_record(record_path(records, problem_id)):
                write_json_atomic(
                    claim / "completed.json",
                    {"return_code": 0, "record": str(record_path(records, problem_id)), "finished_unix_time": time.time()},
                )
            else:
                failures[problem_id] = failures.get(problem_id, 0) + 1
                release_claim(claim)
            break
        if not acquired:
            eligible = [pid for pid in incomplete if failures.get(pid, 0) < args.max_failures_per_problem]
            if not eligible:
                break
            time.sleep(args.poll_seconds)

    remaining = [problem_id for problem_id in ids if completed_source(problem_id, roots) is None]
    status = "COMPLETE" if not remaining else "INCOMPLETE"
    write_json_atomic(
        output / "coordination" / f"worker_{args.worker_id}.json",
        {
            "worker_id": args.worker_id,
            "physical_gpu": args.physical_gpu,
            "base_url": args.base_url,
            "source_record_dirs": [str(path) for path in roots],
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "status": status,
            "remaining_problem_ids": remaining,
            "failures": failures,
            "finished_unix_time": time.time(),
        },
    )
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
