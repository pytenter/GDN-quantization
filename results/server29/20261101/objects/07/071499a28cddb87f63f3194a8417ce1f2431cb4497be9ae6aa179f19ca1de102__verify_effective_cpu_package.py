#!/usr/bin/env python3
"""Read-only stdlib verification of a CPU-only source overlay. Never uses GPU."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

PARENT_MANIFEST = '052be5c615ec13a83e7bf07c1c2a7f400b09977028c42b6215e95cb8cbdcace5'
PARENT_FREEZE = '6168ee5e98acc5935be056fdf0fdf94feccbc6f1fb202bcd62a323d8bd3814e5'
PARENT_VERIFICATION = '1ec66e25610c102d99b45ec410f2fc23f531e0d06bbb3df0a8d0197ce88ef9cf'
REPLACEMENTS = {'scripts/restoration_controller.py', 'worker/restoration_worker.py'}
TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'

def need(value, message):
    if not value:
        raise RuntimeError(message)

def raw(path):
    need(path.is_file() and not path.is_symlink(), 'Missing or symlink: ' + str(path))
    return path.read_bytes()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(raw(path))

def checked_path(value):
    path = PurePosixPath(value)
    need(not path.is_absolute() and '..' not in path.parts and str(path) == value, 'Unsafe path')
    return value

def check_row(directory, row):
    body = raw(directory / checked_path(row['path']))
    need(len(body) == row['bytes'] and sha(body) == row['sha256'], 'SHA/bytes mismatch: ' + row['path'])

def verify(root, parent):
    need(sha(raw(parent / 'SHA256_MANIFEST.json')) == PARENT_MANIFEST, 'Parent manifest changed')
    need(sha(raw(parent / 'SOURCE_FREEZE.json')) == PARENT_FREEZE, 'Parent source freeze changed')
    need(sha(raw(parent / 'LOCAL_DELIVERY_VERIFICATION.json')) == PARENT_VERIFICATION, 'Parent delivery verification changed')
    manifest = read(parent / 'SHA256_MANIFEST.json')
    need(manifest['files_count'] == len(manifest['files']) == 526, 'Parent roster count changed')
    seen = set()
    for row in manifest['files']:
        need(row['path'] not in seen, 'Duplicate parent path')
        seen.add(row['path'])
        check_row(parent, row)
    parent_receipt = read(parent / 'LOCAL_DELIVERY_VERIFICATION.json')
    for name, field in [('SHA256SUMS.txt', 'SHA256SUMS_sha256'), ('FINAL_SEAL_RECEIPT.json', 'FINAL_SEAL_RECEIPT_sha256')]:
        need(sha(raw(parent / name)) == parent_receipt[field], 'Parent external anchor changed: ' + name)
    base = read(parent / 'SOURCE_FREEZE.json')
    effective = read(root / 'EFFECTIVE_SOURCE_FREEZE.json')
    need(effective['task_id'] == TASK and effective['ALLOW_GPU_RESTORE'] == 'NO', 'Effective scope differs')
    need(effective['parent_source_freeze_sha256'] == PARENT_FREEZE, 'Effective parent binding differs')
    need(len(base['files']) == len(effective['files']) == 407, 'Effective source count differs')
    before = {r['path']: r for r in base['files']}
    after = {r['path']: r for r in effective['files']}
    need(len(before) == len(after) == 407 and set(before) == set(after), 'Effective source roster differs')
    changed = {name for name in before if before[name] != after[name]}
    need(changed == REPLACEMENTS, 'Only the two control sources may be replaced')
    for name, row in after.items():
        check_row(root if name in REPLACEMENTS else parent, row)
    need(effective['total_bytes'] == sum(r['bytes'] for r in after.values()), 'Effective bytes sum differs')
    for name in ['ROOT.json', 'HUMAN_GPU_RESTORE_AUTHORIZATION.json']:
        state = read(root / name)
        need(state['task_id'] == TASK and state['ALLOW_GPU_RESTORE'] == 'NO', 'CPU authorization changed')
        need(state.get('GPU_queries', 0) == state.get('GPU_dispatches', 0) == 0, 'Unexpected GPU usage')
    need(read(root / 'ROOT.json')['status'] == 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED', 'CPU state differs')
    return dict(status='PASS_CPU_ONLY_EFFECTIVE_407_AND_IMMUTABLE_PARENT_526', task_id=TASK,
                parent_indexed_files_verified=526, effective_sources_verified=407,
                immutable_parent_sources=405, control_source_replacements=sorted(REPLACEMENTS),
                parent_manifest_sha256=PARENT_MANIFEST, parent_source_freeze_sha256=PARENT_FREEZE,
                effective_source_freeze_sha256=sha(raw(root / 'EFFECTIVE_SOURCE_FREEZE.json')),
                GPU_queries=0, GPU_dispatches=0, GPU_qualification='NOT_RUN')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--parent-delivery', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    parent = args.parent_delivery or root.parent / 'delivery'
    print(json.dumps(verify(root, parent.resolve()), sort_keys=True))

if __name__ == '__main__':
    main()
