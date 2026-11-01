#!/usr/bin/env python3
"""Supplemental CPU delivery only; default preview. Never touches the parent seal.

--prepare copies this revision's bounded small payload once. --run uses the
byte-exact copied sealer prepare/run implementation after fresh local admission.
The original task-specific main/validate_scope and historical STOP main are never
called. No GPU engine, query, model, tensor or network operation occurs in preview.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
REVISION = 'CPU_CONTROL_REVISION_01'
STATUS = 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED'
REVISION_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = REVISION_ROOT.parent
OVERLAYS = {'scripts/restoration_controller.py', 'worker/restoration_worker.py'}
TOOL_SHA = {
    'seal_cpu_delivery.py': '1d23d720a58f23439da885b7f70bd1287290807e1d9f94a02c2355ad155d2335',
    'transfer_primitives.py': '0214931830433d9e9d7fab27f418d21b838e3464d1a4307b7375c5a535d67948',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stable(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Not a regular file: ' + str(p))
    a = p.stat()
    raw = p.read_bytes()
    b = p.stat()
    require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) ==
            (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns), 'File changed during read: ' + str(p))
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(stable(path))


def safe_relative(name):
    require(isinstance(name, str) and name and not any(c in name for c in '\n\r\x00'),
            'Invalid relative file name')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
            'Unsafe/noncanonical relative file name: ' + name)
    return p


def local_verifier(root, runner=subprocess.run):
    script = root / 'scripts/verify_effective_cpu_package.py'
    stable(script)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', ALLOW_GPU_RESTORE='NO')
    p = runner([sys.executable, '-B', str(script)], capture_output=True, text=True,
               env=env, timeout=120)
    require(p.returncode == 0, 'Revision CPU verifier failed: ' + p.stderr[-3000:])
    try:
        result = json.loads(p.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Revision verifier must emit one JSON object') from exc
    require(isinstance(result, dict) and str(result.get('status', '')).startswith('PASS_'),
            'Revision CPU verifier status is not PASS_')
    require(result.get('GPU_queries', 0) == result.get('GPU_dispatches', 0) == 0,
            'CPU verifier reports GPU use')
    return dict(result=result, script_sha256=sha(stable(script)),
                stdout_sha256=sha(p.stdout.encode()), GPU_queries=0, GPU_dispatches=0)


def validate_revision(root, verifier=local_verifier):
    root = Path(root).resolve()
    require(root == REVISION_ROOT and root.name == REVISION, 'Only this isolated revision root is permitted')
    # The checker is a CPU-only, read-only admission step. No original sealer main
    # or task-specific validator is used, even after it succeeds.
    verification = verifier(root)
    state = read(root / 'ROOT.json')
    auth = read(root / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json')
    require(state.get('task_id') == auth.get('task_id') == TASK, 'Wrong revision task identity')
    require(state.get('status') == STATUS, 'Revision must remain CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED')
    require(state.get('ALLOW_GPU_RESTORE') == auth.get('ALLOW_GPU_RESTORE') == 'NO', 'GPU authorization must remain NO')
    require(state.get('GPU_queries') == state.get('GPU_dispatches') == auth.get('GPU_dispatches') == 0,
            'Revision ROOT/authorization reports GPU use')
    require(auth.get('GPU_queries', 0) == 0, 'Authorization reports GPU queries')
    for key in ['remote_root', 'archive_root']:
        value = state.get(key)
        require(isinstance(value, str), 'Revision transport path missing: ' + key)
        remote = PurePosixPath(value)
        require(remote.is_absolute() and '..' not in remote.parts and remote.name == REVISION,
                'Transport path must be a separate CPU_CONTROL_REVISION_01 child: ' + key)
    for name, expected in TOOL_SHA.items():
        require(sha(stable(root / 'delivery_tools' / name)) == expected, 'Copied sealer tool bytes changed: ' + name)
    freeze_path = root / 'EFFECTIVE_SOURCE_FREEZE.json'
    freeze_raw = stable(freeze_path)
    require(state.get('effective_source_freeze_sha256') == sha(freeze_raw), 'ROOT effective freeze SHA differs')
    freeze = json.loads(freeze_raw)
    require(freeze.get('task_id') == TASK and freeze.get('ALLOW_GPU_RESTORE') == 'NO', 'Wrong effective CPU source freeze')
    rows = freeze.get('files')
    require(isinstance(rows, list) and len(rows) == 407, 'Effective source closure must have exactly 407 files')
    seen = set()
    for row in rows:
        name = row['path']
        safe_relative(name)
        require(name not in seen, 'Duplicate effective source path: ' + name)
        seen.add(name)
        # Only two reviewed control files are overlays. All 405 inherited bytes
        # are read from the immutable original delivery, never mutable base files.
        p = root / name if name in OVERLAYS else PARENT_ROOT / 'delivery' / name
        raw = stable(p)
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Effective source changed: ' + name)
    require(OVERLAYS <= seen, 'Two reviewed control overlays are not both in the effective freeze')
    report = read(root / 'analysis/final_review/FINAL_REPORT_REVIEW.json')
    require(str(report.get('status', '')).startswith('PASS_') and not report.get('findings'),
            'Revision independent report review is not PASS_')
    if 'checks_passed' in report or 'checks_total' in report:
        require(isinstance(report.get('checks_total'), int) and report['checks_total'] > 0 and
                report.get('checks_passed') == report['checks_total'], 'Revision review checks incomplete')
    review_rows = report.get('receipts')
    require(isinstance(review_rows, list), 'Revision report review receipts missing')
    reviewed = set()
    for row in review_rows:
        name = row['path']
        safe_relative(name)
        require(name not in reviewed, 'Duplicate review receipt path')
        reviewed.add(name)
        raw = stable(root / name)
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Reviewed revision bytes changed: ' + name)
    require({'FINAL_REPORT.md'} | OVERLAYS <= reviewed, 'Review must bind FINAL_REPORT and both control sources')
    descriptor = dict(path='EFFECTIVE_SOURCE_FREEZE.json', bytes=len(freeze_raw),
                      sha256=sha(freeze_raw), files_verified=407,
                      inherited_parent_delivery_files=405, revision_overlay_files=2,
                      status='PASS_CPU_EFFECTIVE_FROZEN_SOURCE_SHA_BYTES')
    return state, descriptor, verification


def copied_sealer(root):
    directory = root / 'delivery_tools'
    for filename, expected in TOOL_SHA.items():
        require(sha(stable(directory / filename)) == expected, 'Copied transfer tool SHA changed')
    # Explicit module loading avoids resolving any older task module from sys.path.
    spec = importlib.util.spec_from_file_location('transfer_primitives', directory / 'transfer_primitives.py')
    primitives = importlib.util.module_from_spec(spec)
    sys.modules['transfer_primitives'] = primitives
    spec.loader.exec_module(primitives)
    spec = importlib.util.spec_from_file_location('revision_copied_cpu_sealer', directory / 'seal_cpu_delivery.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(root, action='preview', validator=validate_revision, loader=copied_sealer):
    require(action in {'preview', 'prepare', 'run'}, 'Unknown supplemental delivery action')
    state, freeze, verification = validator(root)
    sealer = loader(root)
    rows = sealer.source_roster(root)
    if action == 'prepare':
        sealer.prepare(root, rows)
        status = 'PASS_PREPARED_REVISION_CPU_SMALL_DELIVERY_NOT_SEALED_NOT_TRANSFERRED'
        result = dict(indexed_files=len(rows), indexed_bytes=sum(x['bytes'] for x in rows))
    elif action == 'run':
        # The raw-copied implementation enforces prepared-copy equality, immutable
        # one-shot transfer state, and new remote paths. It does not call its main.
        result = sealer.run(root, state, freeze, rows)
        status = result['status']
    else:
        status = 'PREVIEW_REVISION_CPU_ONLY_NO_WRITES_NO_TRANSFER_NOT_SEALED'
        result = dict(indexed_files=len(rows), indexed_bytes=sum(x['bytes'] for x in rows),
                      delivery_exists=(root / 'delivery').exists())
    return dict(result, status=status, revision_id=REVISION, task_id=TASK,
                restoration_status=STATUS, EFFECTIVE_SOURCE_FREEZE=freeze,
                verifier=verification, GPU_queries=0, GPU_dispatches=0,
                parent_seal_or_delivery_mutated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REVISION_ROOT)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    args = parser.parse_args()
    result = dispatch(args.root.resolve(), 'prepare' if args.prepare else 'run' if args.run else 'preview')
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
