#!/usr/bin/env python3
"""Stage-1 identity and CPU-only structure audit for C5/C6 deployment files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1")
COMMIT = "fe7105fbc44b49db7a5bab1bf693c25e0f9e63fa"
BRANCH = "exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1"
PREFIX = "experiments/qwen_gdn/recurrent_training/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1/deployment"
EXPECTED = {
    "C5": {
        "frozen_rotation_fp32.pt": (1581566, "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9"),
        "deployment_manifest.json": (3097, "138bd68e00375da4403c0ee4c12e888c5923b5656c39292effe08bfa15f03d78"),
        "LOADING_INSTRUCTIONS.md": (614, "42c2d871c51bb47516fc450a2f9f98b3fc07fcd19c4c068d86968d3d618bb088"),
    },
    "C6": {
        "frozen_rotation_fp32.pt": (1581566, "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c"),
        "deployment_manifest.json": (3097, "29131fc7bff3c8df3b2b41c681d45d9be9b1ff0d3be1b9212be1e34e32844203"),
        "LOADING_INSTRUCTIONS.md": (614, "65f1bcd8257943bd7342613dfa4dfcf43086cf129e60fc1d4bc436288f00b361"),
    },
}
LAYER_IDS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def audit(condition: str) -> dict:
    base = ROOT / "deployment" / condition
    files = []
    file_gate = True
    for name, (expected_size, expected_sha) in EXPECTED[condition].items():
        path = base / name
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha = file_sha256(path) if path.is_file() else None
        exact = actual_size == expected_size and actual_sha == expected_sha
        file_gate &= exact
        files.append({
            "name": name,
            "source_branch": BRANCH,
            "source_commit": COMMIT,
            "source_path": f"{PREFIX}/{condition}/{name}",
            "destination_path": str(path),
            "expected_size": expected_size,
            "actual_size": actual_size,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "exact": exact,
        })
    if not file_gate:
        raise RuntimeError(f"{condition} deployment file identity mismatch")

    manifest = json.loads((base / "deployment_manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(base / "frozen_rotation_fp32.pt", map_location="cpu", weights_only=False)
    layer_ids = tuple(int(x) for x in payload.get("layer_ids", ()))
    matrices = payload.get("matrices", {})
    per_layer = []
    structure_gate = (
        payload.get("condition") == condition
        and manifest.get("condition") == condition
        and layer_ids == LAYER_IDS
        and tuple(int(x) for x in manifest.get("expected_layer_ids", ())) == LAYER_IDS
        and set(matrices) == {str(x) for x in LAYER_IDS}
        and len(matrices) == 24
        and payload.get("matrix_dtype") == "torch.float32"
        and payload.get("matrix_shape") == [128, 128]
        and manifest.get("expected_dtype") == "torch.float32"
        and manifest.get("expected_shape_per_layer") == [128, 128]
        and payload.get("theta_checkpoint_sha256") == manifest.get("source_checkpoint_sha256")
        and manifest.get("file_sha256") == EXPECTED[condition]["frozen_rotation_fp32.pt"][1]
        and manifest.get("loading_rule") == "load frozen rotation directly; do not reconstruct R from theta on inference host"
    )
    eye = torch.eye(128, dtype=torch.float32)
    for layer in LAYER_IDS:
        value = matrices.get(str(layer))
        is_tensor = torch.is_tensor(value)
        shape = list(value.shape) if is_tensor else None
        dtype = str(value.dtype) if is_tensor else None
        finite = bool(torch.isfinite(value).all()) if is_tensor else False
        actual_hash = tensor_sha256(value) if is_tensor else None
        expected_hash = manifest.get("per_layer_sha256", {}).get(str(layer))
        hash_exact = actual_hash == expected_hash
        orth_err = float(torch.max(torch.abs(value.T @ value - eye))) if is_tensor and shape == [128, 128] else None
        exact = is_tensor and shape == [128, 128] and dtype == "torch.float32" and finite and hash_exact
        structure_gate &= exact
        per_layer.append({
            "layer_id": layer, "is_tensor": is_tensor, "shape": shape, "dtype": dtype,
            "finite": finite, "expected_sha256": expected_hash, "actual_sha256": actual_hash,
            "hash_exact": hash_exact, "max_orthogonality_error_fp32": orth_err, "exact": exact,
        })
    return {
        "condition": condition,
        "source_branch": BRANCH,
        "source_commit": COMMIT,
        "files": files,
        "FILE_HASH_GATE": "PASS" if file_gate else "FAIL",
        "payload_keys": sorted(payload),
        "format_version": payload.get("format_version"),
        "experiment": payload.get("experiment"),
        "layer_count": len(matrices),
        "expected_layer_count": 24,
        "matrix_meaning": payload.get("matrix_meaning"),
        "orientation": payload.get("orientation"),
        "theta_checkpoint_sha256": payload.get("theta_checkpoint_sha256"),
        "rotation_source_sha256": payload.get("rotation_source_sha256"),
        "canonical_source_sha256": payload.get("canonical_source_sha256"),
        "loading_rule": manifest.get("loading_rule"),
        "per_layer": per_layer,
        "FROZEN_STRUCTURE_GATE": "PASS" if structure_gate else "FAIL",
    }


def main() -> None:
    results = {c: audit(c) for c in ("C5", "C6")}
    for c, value in results.items():
        write_once(ROOT / "analysis" / f"{c}_frozen_rotation_audit.json", value)
    sync_manifest = {
        "task": "QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1",
        "source": "verified GitHub raw content from exact commit, relayed via the local Codex host because direct server TCP connection to raw.githubusercontent.com remained SYN-SENT",
        "source_branch": BRANCH,
        "source_commit": COMMIT,
        "files": [row for c in ("C5", "C6") for row in results[c]["files"]],
        "file_count": 6,
        "total_bytes": sum(row["actual_size"] for c in ("C5", "C6") for row in results[c]["files"]),
        "all_exact": all(row["exact"] for c in ("C5", "C6") for row in results[c]["files"]),
        "from_vgpu": False,
        "rotation_reconstructed_from_theta": False,
    }
    write_once(ROOT / "analysis" / "deployment_sync_manifest.json", sync_manifest)
    gates = {
        "C5_SYNC": "PASS",
        "C6_SYNC": "PASS",
        "C5_FILE_HASH_GATE": results["C5"]["FILE_HASH_GATE"],
        "C6_FILE_HASH_GATE": results["C6"]["FILE_HASH_GATE"],
        "C5_FROZEN_STRUCTURE_GATE": results["C5"]["FROZEN_STRUCTURE_GATE"],
        "C6_FROZEN_STRUCTURE_GATE": results["C6"]["FROZEN_STRUCTURE_GATE"],
    }
    write_once(ROOT / "analysis" / "stage1_gate_summary.json", gates)
    print(json.dumps(gates, indent=2))
    if not all(v == "PASS" for v in gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
