#!/usr/bin/env python3
"""Refresh compact artifact hashes; exclude generated Python cache and logs."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'hashes/artifact_sha256.txt'
paths = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and 'logs' not in p.parts and '__pycache__' not in p.parts
               and p.suffix != '.pyc' and p != TARGET)
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n'
    for p in paths))
print(f'artifacts={len(paths)}')
