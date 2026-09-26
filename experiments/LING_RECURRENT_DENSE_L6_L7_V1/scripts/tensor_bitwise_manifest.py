#!/usr/bin/env python3
"""Create canonical tensor-content hashes for a forced-token INT8 parity run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


GROUPS = {
    "post_update_recurrent_state": "state_*.pt",
    "prefill_and_selected_readout": "basis_*.pt",
    "scale_qcode_post_qdq": "runtime_*.pt",
    "logits": "logits_*.pt",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_hash(digest: "hashlib._Hash", value: object, metadata: list[dict], key: str = "root") -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        row = {"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": len(raw)}
        metadata.append(row)
        digest.update(b"T\0")
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
    elif isinstance(value, dict):
        digest.update(b"D\0")
        for child_key in sorted(value):
            digest.update(str(child_key).encode("utf-8") + b"\0")
            update_hash(digest, value[child_key], metadata, f"{key}.{child_key}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"L\0")
        for index, child in enumerate(value):
            update_hash(digest, child, metadata, f"{key}[{index}]")
    else:
        digest.update(b"P\0")
        digest.update(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")


def object_manifest(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    metadata: list[dict] = []
    update_hash(digest, value, metadata)
    return {"canonical_tensor_sha256": digest.hexdigest(), "tensors": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--condition", choices=("L6", "L7"), required=True)
    parser.add_argument("--host-class", choices=("RTX4090", "RTX3090"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = args.run_dir / "client_result.json"
    client_value = json.loads(client.read_text(encoding="utf-8"))
    if client_value.get("status") != "PASS":
        raise RuntimeError("forced-token parity client did not pass")
    groups = {}
    for group, pattern in GROUPS.items():
        rows = {}
        for path in sorted(args.run_dir.rglob(pattern)):
            rows[str(path.relative_to(args.run_dir))] = object_manifest(path)
        if not rows:
            raise RuntimeError(f"no files for required parity group {group}: {pattern}")
        groups[group] = rows
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "status": "PASS",
        "condition": args.condition,
        "runtime_mode": "int8_r128_unified_final_r",
        "host_class": args.host_class,
        "rotation_sha256": file_sha256(args.rotation),
        "client_result_sha256": file_sha256(client),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "condition": args.condition, "groups": {k: len(v) for k, v in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
