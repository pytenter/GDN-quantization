#!/usr/bin/env python3
"""Mechanically hash only compact, Git-safe artifacts of this task."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'hashes/artifact_sha256.txt'
EXCLUDED_SUFFIXES = {'.pt', '.pth', '.safetensors', '.bin'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted(path for path in ROOT.rglob('*') if path.is_file()
                   and path != OUTPUT and '__pycache__' not in path.parts
                   and path.suffix.lower() not in EXCLUDED_SUFFIXES)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8', newline='\n') as handle:
        for path in paths:
            handle.write(f'{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n')
    print(f'hashed {len(paths)} compact artifacts: {OUTPUT}')


if __name__ == '__main__':
    main()
