#!/usr/bin/env python3
"""Bounded CPU admission delivery, not a GPU runner. Default: read-only preview.

Only --prepare/--run use the copied, byte-bound CPU delivery implementation.
No old main/validate_scope, controller, worker, Torch, Driver or NVML is called.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
PARENT_TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
STATUS = 'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED'
MODE = 'HISTORICAL_PORT'
ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457'
PARENT_REVIEW_SHA = '6c965fb955012f3c01d07bd9dd86d866216f6127e64745c4dd1a75a1b9160c06'
CONTROL_SHA = {
    'scripts/restoration_controller.py': '1a71c44fb55f4d5a758625439f99c3a384b067cadbba6d2ffa56160b8717bbef',
    'worker/restoration_worker.py': '8b14e35fdeac501978b0e3f948eb888eacd0e1dce37487afed83181b10bb396e',
}
TOOLS = {
    'seal_cpu_delivery.py': '1d23d720a58f23439da885b7f70bd1287290807e1d9f94a02c2355ad155d2335',
    'transfer_primitives.py': '0214931830433d9e9d7fab27f418d21b838e3464d1a4307b7375c5a535d67948',
}
REVIEW_REQUIRED = {'FINAL_REPORT.md', 'ROOT.json', 'AUTHORIZATION_AND_PERMIT.json',
                   'EFFECTIVE_SOURCE_VERIFICATION.json'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def raw(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Not a regular file: ' + str(p))
    a = p.stat()
    b = p.read_bytes()
    z = p.stat()
    require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) ==
            (z.st_dev, z.st_ino, z.st_size, z.st_mtime_ns), 'File changed during read: ' + str(p))
    return b


def digest(b):
    return hashlib.sha256(b).hexdigest()


def read(path):
    return json.loads(raw(path))


def relative(name):
    require(isinstance(name, str) and name and not any(c in name for c in '\n\r\x00'), 'Invalid path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'Unsafe relative path: ' + name)
    return p


def matches(path, row):
    b = raw(path)
    return len(b) == row['bytes'] and digest(b) == row['sha256']


def no_gpu(j, label, required=True):
    for key in ['GPU_queries', 'GPU_dispatches']:
        require(j.get(key) == 0 if required else j.get(key, 0) == 0, label + ' reports GPU use: ' + key)
    for key in ['Driver_API_calls', 'Driver_calls', 'NVML_calls', 'Torch_GPU_initializations']:
        require(j.get(key, 0) == 0, label + ' reports device calls: ' + key)


def validate(root):
    root = Path(root).resolve()
    require(root == ROOT and root.name == TASK, 'Only this independent CPU admission root is permitted')
    state = read(root / 'ROOT.json')
    auth = read(root / 'AUTHORIZATION_AND_PERMIT.json')
    require(state.get('task_id') == auth.get('task_id') == TASK, 'Task identity differs')
    require(state.get('status') == STATUS and state.get('restoration_mode') == MODE, 'Not a final CPU-only HISTORICAL_PORT admission')
    require(state.get('ALLOW_GPU_RESTORE') == auth.get('ALLOW_GPU_RESTORE') == 'NO', 'GPU authorization must remain NO')
    require(auth.get('valid_execution_permit') is None and auth.get('permit_created') is False and
            auth.get('separate_human_GPU_authorization_received') is False, 'GPU permit or authorization exists')
    require(state.get('current_GPU_work_allowed') is False, 'Current GPU work must remain prohibited')
    no_gpu(state, 'ROOT')
    no_gpu(auth, 'AUTHORIZATION_AND_PERMIT')
    require(state.get('request_sha256') == digest(raw(root / 'USER_REQUEST.txt')), 'New request SHA differs')
    for key, namespace in [('remote_root', 'qwen_gdn_historical_runtime_gpu_recovery_v1'),
                           ('archive_root', 'qwen_historical_runtime_gpu_recovery_archive')]:
        value = state.get(key)
        require(isinstance(value, str), 'Missing independent transport path: ' + key)
        p = PurePosixPath(value)
        require(p.is_absolute() and '..' not in p.parts and namespace in p.parts and p.name != namespace,
                'Transport must use this new task independent child directory: ' + key)
    for filename, expected in TOOLS.items():
        require(digest(raw(root / 'delivery_tools' / filename)) == expected, 'Copied delivery tool changed: ' + filename)
    fp = root / 'execution_payload/SOURCE_FREEZE.json'
    fb = raw(fp)
    require(digest(fb) == FREEZE_SHA == state.get('effective_source_freeze_sha256'), 'Latest effective407 freeze identity differs')
    freeze = json.loads(fb)
    require(freeze.get('task_id') == PARENT_TASK and freeze.get('ALLOW_GPU_RESTORE') == 'NO', 'Frozen inherited source task/NO differs')
    rows = freeze.get('files')
    require(isinstance(rows, list) and len(rows) == 407, 'Exactly407 frozen sources required')
    source = read(root / 'EFFECTIVE_SOURCE_VERIFICATION.json')
    require(source.get('task_id') == TASK and str(source.get('status', '')).startswith('PASS_'), 'Effective source verification not PASS_')
    require(source.get('files_count') == 407 and source.get('immutable_parent_sources') == 405 and
            source.get('control_revision_sources') == 2 and source.get('source_code_changes') == 0 and
            source.get('effective_source_freeze_sha256') == FREEZE_SHA, 'Effective verification closure differs')
    no_gpu(source, 'Effective verification')
    proof_rows = source.get('files')
    require(isinstance(proof_rows, list) and len(proof_rows) == 407, 'Effective verification files missing')
    proofs = {}
    for row in proof_rows:
        relative(row['path'])
        require(row['path'] not in proofs, 'Duplicate effective verification path')
        proofs[row['path']] = row
    seen = set()
    for row in rows:
        name = row['path']
        relative(name)
        require(name not in seen, 'Duplicate frozen source path')
        seen.add(name)
        require(name in proofs and all(proofs[name][k] == row[k] for k in ['path', 'bytes', 'sha256']), 'Effective proof/file mismatch: ' + name)
        require(matches(root / 'execution_payload' / name, row), 'Actual407 source SHA/bytes changed: ' + name)
    require(seen == set(proofs) and CONTROL_SHA.keys() <= seen, '407 source roster differs')
    for name, expected in CONTROL_SHA.items():
        require(digest(raw(root / 'execution_payload' / name)) == expected, 'Latest revised control source not used: ' + name)
    require(source.get('actual_new_execution_controller_sha256') == CONTROL_SHA['scripts/restoration_controller.py'] and
            source.get('actual_new_execution_worker_sha256') == CONTROL_SHA['worker/restoration_worker.py'], 'Latest control source proof differs')
    parent_path = root / 'analysis/PARENT_SEALS_INDEPENDENT_REVIEW.json'
    require(digest(raw(parent_path)) == PARENT_REVIEW_SHA, 'Fixed independent parent seal review changed')
    parent = read(parent_path)
    require(parent.get('status') == 'PASS_CPU_INDEPENDENT_PARENT_SEALS_AND_EFFECTIVE407_REVIEW' and not parent.get('errors'), 'Parent seal closure not PASS')
    for key, configured in [('base_delivery', 'parent_delivery'), ('control_revision_delivery', 'control_revision_delivery')]:
        anchor = parent[key]
        directory = Path(state[configured])
        require(str(directory) == anchor['delivery_root'], 'Parent seal directory rebound')
        for filename, description in [('SHA256_MANIFEST.json', 'manifest'), ('SHA256SUMS.txt', 'SHA256SUMS'),
                                       ('FINAL_SEAL_RECEIPT.json', 'seal'), ('LOCAL_DELIVERY_VERIFICATION.json', 'local_delivery_verification')]:
            require(matches(directory / filename, anchor[description]), 'Prior sealed anchor changed: ' + key + '/' + filename)
    remote_name = state.get('current_remote_CPU_admission_receipt')
    remote_relative = relative(remote_name)
    require(len(remote_relative.parts) >= 2 and remote_relative.parts[0] == 'qualification' and
            remote_relative.suffix == '.json', 'Current remote CPU receipt must be a qualification/ JSON path')
    selected = root
    for component in remote_relative.parts:
        selected = selected / component
        require(not selected.is_symlink(), 'Current remote CPU receipt symlink prohibited')
    require(root in selected.resolve().parents, 'Current remote CPU receipt escapes independent root')
    remote = read(selected)
    require(remote.get('task_id') == TASK and str(remote.get('status', '')).startswith('PASS_'), 'Remote CPU admission not PASS_')
    no_gpu(remote, 'Remote CPU admission')
    require(raw(root / 'FINAL_REPORT.md') == raw(root / 'FINAL_SCIENTIFIC_REPORT.md'), 'Report aliases must have exact same bytes')
    review = read(root / 'analysis/final_review/FINAL_REPORT_REVIEW.json')
    require(str(review.get('status', '')).startswith('PASS_') and not review.get('findings'), 'Final independent report review not PASS_')
    require(isinstance(review.get('checks_total'), int) and review['checks_total'] > 0 and
            review.get('checks_passed') == review['checks_total'], 'Final review checks incomplete')
    no_gpu(review, 'Final review', required=False)
    receipts = review.get('receipts')
    require(isinstance(receipts, list), 'Final review receipts missing')
    reviewed = set()
    for row in receipts:
        name = row['path']
        relative(name)
        require(name not in reviewed, 'Duplicate review receipt path')
        reviewed.add(name)
        require(matches(root / name, row), 'Reviewed report/source bytes changed: ' + name)
    require(REVIEW_REQUIRED | {remote_name} <= reviewed, 'Final review must bind report/ROOT/AUTH/407proof/current remoteCPU')
    descriptor = dict(path='execution_payload/SOURCE_FREEZE.json', bytes=len(fb), sha256=FREEZE_SHA,
                      files_verified=407, inherited_parent_delivery_files=405, revision_overlay_files=2,
                      status='PASS_CPU_EFFECTIVE407_MATERIALIZED_SOURCE_SHA_BYTES')
    return state, descriptor


def copied_sealer(root):
    directory = root / 'delivery_tools'
    for name, expected in TOOLS.items():
        require(digest(raw(directory / name)) == expected, 'Copied delivery source changed')
    spec = importlib.util.spec_from_file_location('transfer_primitives', directory / 'transfer_primitives.py')
    primitives = importlib.util.module_from_spec(spec)
    sys.modules['transfer_primitives'] = primitives
    spec.loader.exec_module(primitives)
    spec = importlib.util.spec_from_file_location('gpu_recovery_cpu_delivery_only', directory / 'seal_cpu_delivery.py')
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    require(sealer.MODE == MODE and sealer.STATUS == STATUS, 'Copied CPU delivery scope differs')
    sealer.TASK = TASK
    return sealer


def dispatch(root, action='preview', validator=validate, loader=copied_sealer):
    require(action in {'preview', 'prepare', 'run'}, 'Unknown delivery action')
    state, freeze = validator(root)
    sealer = loader(root)
    rows = sealer.source_roster(root)
    if action == 'prepare':
        sealer.prepare(root, rows)
        result = dict(status='PASS_PREPARED_CPU_ADMISSION_SMALL_DELIVERY_NOT_SEALED_NOT_TRANSFERRED')
    elif action == 'run':
        result = sealer.run(root, state, freeze, rows)
    else:
        result = dict(status='PREVIEW_CPU_ADMISSION_ONLY_NO_WRITES_NO_TRANSFER_NOT_SEALED', delivery_exists=(root / 'delivery').exists())
    return dict(result, task_id=TASK, restoration_mode=MODE, restoration_status=STATUS,
                indexed_files=len(rows), indexed_bytes=sum(row['bytes'] for row in rows), SOURCE_FREEZE=freeze,
                GPU_queries=0, GPU_dispatches=0, old_parent_packages_mutated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--prepare', action='store_true')
    options.add_argument('--run', action='store_true')
    args = parser.parse_args()
    result = dispatch(args.root.resolve(), 'prepare' if args.prepare else 'run' if args.run else 'preview')
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
