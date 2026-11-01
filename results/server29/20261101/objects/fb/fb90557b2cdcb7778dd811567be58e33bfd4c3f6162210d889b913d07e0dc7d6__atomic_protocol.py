#!/usr/bin/env python3
"""Atomic sample artifacts and fail-closed/resume rules for future Stage B."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


VERSION = "QWEN_STAGE_B_ATOMIC_ARTIFACT_V1"


def canonical_bytes(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit_sample(root, sample_id, payload, identity):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{sample_id}.json"
    checksum = root / f"{sample_id}.sha256"
    done = root / f"{sample_id}.done"
    if any(p.exists() for p in (final, checksum, done)):
        raise FileExistsError(f"refuse overwrite of committed sample {sample_id}")
    data = canonical_bytes(payload)
    digest = sha256_bytes(data)
    tmp = root / f"{sample_id}.tmp"
    checksum_tmp = root / f"{sample_id}.sha256.tmp"
    done_tmp = root / f"{sample_id}.done.tmp"
    with tmp.open("wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with checksum_tmp.open("w", encoding="utf-8") as f:
        f.write(digest + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, final)
    os.replace(checksum_tmp, checksum)
    done_payload = {
        "protocol_version": VERSION, "sample_id": sample_id,
        "artifact": final.name, "sha256": digest, "identity": identity,
    }
    with done_tmp.open("wb") as f:
        f.write(canonical_bytes(done_payload)); f.flush(); os.fsync(f.fileno())
    os.replace(done_tmp, done)
    _fsync_directory(str(root))
    return done_payload


def validate_sample(root, sample_id, expected_identity):
    root = Path(root)
    final = root / f"{sample_id}.json"
    checksum = root / f"{sample_id}.sha256"
    done = root / f"{sample_id}.done"
    if not all(p.is_file() for p in (final, checksum, done)):
        return {"valid": False, "reason": "missing committed triplet"}
    try:
        done_payload = json.loads(done.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "reason": f"invalid done JSON: {exc}"}
    actual = hashlib.sha256(final.read_bytes()).hexdigest()
    recorded = checksum.read_text(encoding="utf-8").strip()
    if actual != recorded or actual != done_payload.get("sha256"):
        return {"valid": False, "reason": "hash mismatch"}
    if done_payload.get("identity") != expected_identity:
        return {"valid": False, "reason": "identity mismatch"}
    return {"valid": True, "reason": "atomic committed artifact and identity valid", "sha256": actual}


def resume_scan(root, sample_ids, expected_identity):
    valid, pending = [], []
    evidence = {}
    for sample_id in sample_ids:
        result = validate_sample(root, sample_id, expected_identity)
        evidence[sample_id] = result
        (valid if result["valid"] else pending).append(sample_id)
    return {"valid_completed": valid, "pending": pending, "evidence": evidence}


def fail_closed_decision(worker_exit_codes, expected_samples, valid_sample_ids, oom_observed=False, free_gb=100.0):
    any_worker_failure = any(int(code) != 0 for code in worker_exit_codes)
    complete = len(set(valid_sample_ids)) == int(expected_samples)
    emergency_disk = float(free_gb) < 10.0
    stage_pass = not any_worker_failure and not oom_observed and not emergency_disk and complete
    return {
        "worker_exit_codes": list(map(int, worker_exit_codes)),
        "any_worker_nonzero": any_worker_failure,
        "oom_observed": bool(oom_observed), "emergency_disk_stop": emergency_disk,
        "expected_samples": int(expected_samples), "actual_success": len(set(valid_sample_ids)),
        "completeness_gate": "PASS" if complete else "FAIL",
        "stage_gate": "PASS" if stage_pass else "FAIL",
        "downstream_launch_authorized": False,
        "automatic_oom_retry": False,
    }
