#!/usr/bin/env python3
"""Preserve the first manifest and replace only its live-log self-race."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HASHES = ROOT / "hashes"
LOGS = ROOT / "logs"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    original = HASHES / "artifact_sha256.txt"
    preserved = HASHES / "artifact_sha256.initial.txt"
    note = HASHES / "manifest_repair.json"
    if not original.is_file() or preserved.exists() or note.exists():
        raise RuntimeError("unexpected manifest state; refusing repair")
    verification = LOGS / "hash_verification.log"
    failures = [line for line in verification.read_text().splitlines() if line.endswith(": FAILED")]
    if failures != ["logs/finalizer.log: FAILED"]:
        raise RuntimeError(f"unexpected hash failures: {failures}")
    if len(original.read_text().splitlines()) != 500:
        raise RuntimeError("unexpected initial manifest size")
    original_sha = digest(original)
    original.rename(preserved)
    record = {
        "repair_reason": "the original manifest hashed logs/finalizer.log while its producer was still appending the completion JSON",
        "scientific_artifacts_changed": False,
        "initial_manifest_sha256": original_sha,
        "initial_manifest_preserved_as": preserved.relative_to(ROOT).as_posix(),
        "initial_verification_failures": failures,
        "new_manifest_policy": "hash all stable closure files; exclude self, pycache and the future verification output",
    }
    with note.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    exclusions = {original, LOGS / "hash_verification_final.log"}
    paths = sorted(path for path in ROOT.rglob("*")
                   if path.is_file() and path not in exclusions and
                   "__pycache__" not in path.parts and path.suffix != ".pyc")
    with original.open("x") as handle:
        for path in paths:
            handle.write(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n")
    print(json.dumps({"initial_manifest_sha256": original_sha,
                      "repaired_manifest_sha256": digest(original),
                      "artifact_files": len(paths)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
