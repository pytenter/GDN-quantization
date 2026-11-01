#!/usr/bin/env python3
"""Hash the completed design-only artifact tree."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "hashes/SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text("".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in files), encoding="utf-8")
print(f"DESIGN_HASH_MANIFEST=PASS files={len(files)} sha256={sha256(MANIFEST)}")
