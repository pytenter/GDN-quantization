#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or (
        args.root / "LING_RECURRENT_DENSE_L6_L7_V1" / "manifests" / "artifact_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    failed = False
    for expected in manifest["canonical_binaries"]:
        path = args.root / expected["path"]
        row = {"path": expected["path"], "exists": path.is_file()}
        if row["exists"]:
            row["size"] = path.stat().st_size
            row["sha256"] = sha256(path)
            row["size_gate"] = "PASS" if row["size"] == expected["size"] else "FAIL"
            row["sha256_gate"] = "PASS" if row["sha256"] == expected["sha256"] else "FAIL"
        else:
            row["size_gate"] = row["sha256_gate"] = "FAIL"
        failed |= row["size_gate"] != "PASS" or row["sha256_gate"] != "PASS"
        rows.append(row)
    result = {
        "task": "LING_L6_L7_NEW_MEMBER_HANDOFF_V1",
        "rows": rows,
        "HF_SMALL_ARTIFACT_READBACK_GATE": "FAIL" if failed else "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
