#!/usr/bin/env python3
"""Create a deterministic artifact manifest after all acceleration reports exist."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_STAGE_B_STREAMING_ACCELERATION_V1")
MANIFEST = ROOT / "hashes/SHA256SUMS"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text("".join(f"{digest(path)}  {path.relative_to(ROOT)}\n" for path in paths), encoding="utf-8")
print(f"files={len(paths)} manifest={MANIFEST} sha256={digest(MANIFEST)}")
