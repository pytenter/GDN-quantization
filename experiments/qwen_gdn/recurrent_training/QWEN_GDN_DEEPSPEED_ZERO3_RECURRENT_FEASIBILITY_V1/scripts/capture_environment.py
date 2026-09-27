#!/usr/bin/env python3
"""Capture immutable environment evidence before creating the isolated venv."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CONFIGS.mkdir(parents=True, exist_ok=True)
CANONICAL = Path("/data/ydai/miniconda3/envs/bitdecode/bin/python")
if Path(sys.executable).resolve() != CANONICAL.resolve():
    raise RuntimeError(f"Wrong snapshot interpreter: {sys.executable}")
target = CONFIGS / "canonical_environment_snapshot.txt"
if target.exists():
    raise FileExistsError(target)
version = subprocess.check_output([str(CANONICAL), "--version"], text=True).strip()
freeze = subprocess.check_output([str(CANONICAL), "-m", "pip", "freeze"], text=True)
for line in freeze.splitlines():
    if any(mark in line.lower() for mark in ("password", "token=", "api_key=", "apikey=")):
        raise RuntimeError("Secret-like pip freeze entry; manual redaction required")
target.write_text(f"python: {version}\nexecutable: {CANONICAL}\n\npip freeze:\n{freeze}")
record = {"python": version, "canonical_prefix": sys.prefix,
          "snapshot_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
(CONFIGS / "canonical_environment_snapshot.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
