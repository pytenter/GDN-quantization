#!/usr/bin/env python3
"""Validate and hash the stopped V2 diagnostic artifacts without modifying them."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    verdict = json.loads((ROOT / 'analysis/final_verdict.json').read_text())
    if verdict['DISTRIBUTED_V2_STATUS'] != 'FAIL' or verdict['FORMAL_C5_C6_TRAINING'] != 'NOT_STARTED':
        raise RuntimeError('unexpected final state')
    required = (
        'analysis/hardware_topology.json', 'analysis/p2p_bandwidth.json',
        'analysis/deterministic_cumsum_micro.json',
        'analysis/deterministic_chunk_source_delta.json',
        'analysis/deterministic_chunk_function_gate.json',
        'analysis/deterministic_full_model_teacher_forward_gate.json',
        'analysis/deterministic_full_model_teacher_forward_gate.error.json',
        'analysis/final_verdict.json',
        'reports/DISTRIBUTED_V2_FINAL_REPORT.md',
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f'missing required artifact {relative}')
    files = sorted(path for directory in ('analysis', 'configs', 'reports', 'runtime')
                   for path in (ROOT / directory).rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts and path.suffix in ('.json', '.md', '.py', '.csv'))
    total = 0
    for path in files:
        size = path.stat().st_size
        if size >= 1_000_000_000:
            raise RuntimeError(f'artifact exceeds 1GB cap: {path}')
        total += size
        if path.suffix == '.json':
            json.loads(path.read_text())
        elif path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
    output = ROOT / 'hashes/artifact_sha256.txt'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        for path in files:
            handle.write(f'{digest(path)}  {path.relative_to(ROOT).as_posix()}\n')
    print(json.dumps({'DISTRIBUTED_V2_STATUS': verdict['DISTRIBUTED_V2_STATUS'],
                      'manifest_files': len(files), 'total_artifact_bytes': total,
                      'manifest_sha256': digest(output)}), flush=True)


if __name__ == '__main__':
    main()
