#!/usr/bin/env python3
"""Read-only provenance audit of frozen trajectory, geometry, and replay artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OUT = ROOT / 'experiments/QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1'
GEOM = ROOT / 'experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1'
SOURCE = Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def main() -> None:
    prereg = json.loads((OUT / 'preregistration.json').read_text(encoding='utf-8'))
    prior_audit = json.loads((GEOM / 'analysis/replay_audit.json').read_text(encoding='utf-8'))
    prior_parity = json.loads((GEOM / 'analysis/replay_parity.json').read_text(encoding='utf-8'))
    prior_result = json.loads((GEOM / 'results.json').read_text(encoding='utf-8'))
    manifest = json.loads((GEOM / 'run_manifest.json').read_text(encoding='utf-8'))
    if prior_audit['hash_verification'] != 'PASS' or prior_audit['pairing_status'] != 'PASS':
        raise RuntimeError('prior provenance/pairing gate failed')
    if prior_parity['REPLAY_PARITY_GATE'] != 'PASS' or manifest['status'] != 'COMPLETE':
        raise RuntimeError('prior parity/completion gate failed')
    if prior_result['phase'] != 'PHASE_1_COMPLETE':
        raise RuntimeError('prior geometry result incomplete')
    for name, meta in manifest['outputs'].items():
        path = GEOM / name
        if not path.is_file() or path.stat().st_size != meta['bytes'] or digest(path) != meta['sha256']:
            raise RuntimeError(f'geometry manifest hash mismatch: {name}')
    source_rows = []
    for path in sorted(SOURCE.glob('worker*/fp_state.jsonl')):
        with path.open(encoding='utf-8') as stream:
            source_rows.extend(json.loads(line) for line in stream if line.strip())
    source_rows.sort(key=lambda row: (int(row['canonical_index']), int(row['seed'])))
    if len(source_rows) != 60:
        raise RuntimeError(f'expected 60 source trajectories, got {len(source_rows)}')
    meta_files = sorted((GEOM / 'analysis').glob('shard_*/*_meta.json'))
    if len(meta_files) != 60:
        raise RuntimeError(f'expected 60 geometry units, got {len(meta_files)}')
    tokens = 0
    unit_ids = set()
    for path in meta_files:
        meta = json.loads(path.read_text(encoding='utf-8'))
        index = int(meta['global_index'])
        row = source_rows[index]
        key = (row['problem_id'], int(row['seed']))
        if key != (meta['problem_id'], int(meta['seed'])) or key in unit_ids:
            raise RuntimeError(f'unit ID mismatch or duplicate: {path}')
        unit_ids.add(key)
        token_ids = list(map(int, row['generated_token_ids']))
        token_hash = hashlib.sha256(json.dumps(token_ids, separators=(',', ':')).encode()).hexdigest()
        if token_hash != meta['token_ids_sha256'] or len(token_ids) != int(meta['generated_tokens']):
            raise RuntimeError(f'token/hash mismatch: {path}')
        n = len(token_ids) - 1
        if int(meta['rows']) != n * 24:
            raise RuntimeError(f'layer/time grid mismatch: {path}')
        data = path.with_name(path.name.replace('_meta.json', '.parquet'))
        if not data.is_file() or digest(data) != meta['parquet_sha256']:
            raise RuntimeError(f'geometry unit hash mismatch: {data}')
        tokens += n
    clean_path = GEOM / 'analysis/clean59_paired.csv'
    with clean_path.open(encoding='utf-8', newline='') as stream:
        clean = list(csv.DictReader(stream))
    clean_ids = {(row['problem_id'], int(row['seed'])) for row in clean}
    excluded = ('aime26_28', 2)
    if len(clean) != 59 or len(clean_ids) != 59 or excluded in clean_ids or clean_ids != unit_ids - {excluded}:
        raise RuntimeError('clean59 real-ID pairing failed')
    group_counts = {}
    for row in clean:
        group_counts[row['group']] = group_counts.get(row['group'], 0) + 1
    if group_counts != {'stable_correct': 18, 'hadamard_rescue': 23, 'stable_failure': 18}:
        raise RuntimeError(f'outcome groups mismatch: {group_counts}')
    if tokens != 1854391 or tokens * 24 != 44505384:
        raise RuntimeError(f'token-layer count mismatch: {tokens}')
    disk = shutil.disk_usage(OUT)
    result = {
        'task': prereg['task'], 'status': 'PASS', 'source_trajectories': 60,
        'clean59': 59, 'gdn_layers': 24, 'token_positions_per_condition': tokens,
        'paired_token_layer_rows': tokens * 24, 'source_token_hashes_verified': 60,
        'geometry_unit_parquet_hashes_verified': 60,
        'geometry_manifest_output_hashes_verified': len(manifest['outputs']),
        'exact_real_id_pairing': True, 'group_counts': group_counts,
        'frozen_exclusion': {'problem_id': excluded[0], 'seed': excluded[1], 'raw_capture_preserved': True},
        'prior_replay_parity': 'PASS',
        'frozen_checkpoint_sha256': prior_audit['frozen_checkpoint_sha256'],
        'preregistration_sha256': digest(OUT / 'preregistration.json'),
        'source_geometry_manifest_sha256': digest(GEOM / 'run_manifest.json'),
        'disk_free_gib': disk.free / 2**30, 'new_generation': False,
    }
    save(OUT / 'analysis/replay_audit.json', result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
