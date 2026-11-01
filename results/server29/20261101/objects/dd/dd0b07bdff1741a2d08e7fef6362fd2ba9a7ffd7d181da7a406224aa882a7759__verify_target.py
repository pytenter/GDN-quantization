#!/usr/bin/env python3
import gzip
import hashlib
import json
import os
import random
import struct
import zipfile
from pathlib import Path


ROOT = Path("/data/zypan/research_archive/qwen_gdn/QWEN_SCIENTIFIC_RETENTION_AND_3090_MIGRATION_V1")
MANIFEST = ROOT / "transfer" / "transfer_manifest.json"
OUT_JSON = ROOT / "transfer" / "target_sha256.json"
OUT_SUMS = ROOT / "transfer" / "TARGET_SHA256SUMS"
OUT_READBACK = ROOT / "transfer" / "remote_readback.json"


def sha256_file(path: Path):
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def chunk_readback(path: Path):
    size = path.stat().st_size
    chunks = []
    with path.open("rb") as f:
        for label, offset in (("head", 0), ("middle", max(0, size // 2 - 32768)), ("tail", max(0, size - 65536))):
            f.seek(offset)
            data = f.read(min(65536, size - offset))
            chunks.append({"label": label, "offset": offset, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return chunks


def structured_readback(path: Path):
    name = path.name.lower()
    if name.endswith(".npz"):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"bad npz member: {bad}")
            return {"format": "npz", "members": len(names), "first_members": names[:5], "zip_crc_gate": "PASS"}
    if name.endswith(".parquet"):
        with path.open("rb") as f:
            if f.read(4) != b"PAR1":
                raise ValueError("missing parquet leading magic")
            f.seek(-8, os.SEEK_END)
            footer_length = struct.unpack("<I", f.read(4))[0]
            magic = f.read(4)
            if magic != b"PAR1":
                raise ValueError("missing parquet trailing magic")
            if footer_length <= 0 or footer_length > path.stat().st_size - 12:
                raise ValueError("invalid parquet footer length")
            f.seek(-8 - footer_length, os.SEEK_END)
            footer = f.read(footer_length)
            return {"format": "parquet", "footer_length": footer_length, "footer_sha256": hashlib.sha256(footer).hexdigest(), "magic_gate": "PASS"}
    if name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            line = f.readline(1024 * 1024)
            if not line:
                raise ValueError("empty gzip stream")
            return {"format": "gzip_text", "first_line_bytes": len(line), "first_line_sha256": hashlib.sha256(line).hexdigest()}
    if name.endswith(".json"):
        if path.stat().st_size <= 5 * 1024 * 1024:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            return {"format": "json", "parsed_type": type(obj).__name__, "parse_gate": "PASS"}
        with path.open("rb") as f:
            first = f.read(1)
            f.seek(max(0, path.stat().st_size - 4096), os.SEEK_SET)
            tail = f.read().rstrip()
            last = tail[-1:] if tail else b""
        if first not in (b"{", b"[") or last not in (b"}", b"]"):
            raise ValueError("large JSON boundary check failed")
        return {"format": "large_json", "boundary_gate": "PASS"}
    if name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            json.loads(f.readline())
        return {"format": "jsonl", "first_record_parse_gate": "PASS"}
    if name.endswith(".csv"):
        with path.open("rb") as f:
            line = f.readline(1024 * 1024)
        if b"," not in line:
            raise ValueError("CSV header check failed")
        return {"format": "csv", "header_bytes": len(line), "header_sha256": hashlib.sha256(line).hexdigest()}
    return {"format": "binary", "chunk_read_gate": "PASS"}


def pick_samples(files, count=20):
    chosen = []
    def add(item):
        if item not in chosen:
            chosen.append(item)
    for item in sorted(files, key=lambda x: x["size_bytes"], reverse=True)[:5]:
        add(item)
    predicates = [
        lambda p: p.endswith(".npz"),
        lambda p: p.endswith(".parquet"),
        lambda p: p.endswith(".csv.gz"),
        lambda p: p.endswith(".jsonl.gz"),
        lambda p: p.endswith(".jsonl"),
        lambda p: p.endswith(".json"),
        lambda p: "checkpoint" in p,
        lambda p: "status.json" in p,
    ]
    for predicate in predicates:
        match = next((x for x in files if predicate(x["target_path"].lower())), None)
        if match:
            add(match)
    rng = random.Random(20260927)
    pool = [x for x in files if x not in chosen]
    rng.shuffle(pool)
    for item in pool:
        if len(chosen) >= count:
            break
        add(item)
    return chosen[:count]


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["files"]
    checks = []
    for item in expected:
        path = Path(item["target_path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = None
        if exists and actual_size == item["size_bytes"]:
            actual_sha, _ = sha256_file(path)
        checks.append({
            **item,
            "exists": exists,
            "actual_size_bytes": actual_size,
            "actual_sha256": actual_sha,
            "size_match": actual_size == item["size_bytes"],
            "sha256_match": actual_sha == item["sha256"],
        })

    count_match = len(checks) == manifest["file_count"] and all(x["exists"] for x in checks)
    byte_match = sum((x["actual_size_bytes"] or 0) for x in checks) == manifest["total_bytes"] and all(x["size_match"] for x in checks)
    sha_match = all(x["sha256_match"] for x in checks)
    result = {
        "expected_file_count": manifest["file_count"],
        "actual_file_count": sum(1 for x in checks if x["exists"]),
        "expected_total_bytes": manifest["total_bytes"],
        "actual_total_bytes": sum((x["actual_size_bytes"] or 0) for x in checks),
        "FILE_COUNT_MATCH": "PASS" if count_match else "FAIL",
        "TOTAL_BYTES_MATCH": "PASS" if byte_match else "FAIL",
        "PER_FILE_SHA256_MATCH": "PASS" if sha_match else "FAIL",
        "files": checks,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT_SUMS.write_text("".join(f"{x['actual_sha256']}  {x['target_relative_path']}\n" for x in checks if x["actual_sha256"]), encoding="utf-8")

    readback = []
    readback_gate = True
    if count_match and byte_match and sha_match:
        for item in pick_samples(expected, 20):
            path = Path(item["target_path"])
            try:
                details = structured_readback(path)
                chunks = chunk_readback(path)
                status = "PASS"
            except Exception as exc:
                details = {"error": repr(exc)}
                chunks = []
                status = "FAIL"
                readback_gate = False
            readback.append({
                "target_path": str(path),
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "status": status,
                "details": details,
                "chunks": chunks,
            })
    else:
        readback_gate = False
    readback_payload = {
        "sample_count": len(readback),
        "required_sample_count": 20,
        "REMOTE_READBACK_GATE": "PASS" if readback_gate and len(readback) >= 20 else "FAIL",
        "samples": readback,
    }
    OUT_READBACK.write_text(json.dumps(readback_payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "FILE_COUNT_MATCH": result["FILE_COUNT_MATCH"],
        "TOTAL_BYTES_MATCH": result["TOTAL_BYTES_MATCH"],
        "PER_FILE_SHA256_MATCH": result["PER_FILE_SHA256_MATCH"],
        "REMOTE_READBACK_GATE": readback_payload["REMOTE_READBACK_GATE"],
        "sample_count": len(readback),
    }, indent=2))


if __name__ == "__main__":
    main()
