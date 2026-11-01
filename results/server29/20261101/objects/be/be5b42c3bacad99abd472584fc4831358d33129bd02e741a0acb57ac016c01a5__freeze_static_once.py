"""Freeze one new scale-repair revision after two fresh CPU admissions.

This program does not call a GPU, stage files, launch a child or rewrite old seals.
"""
import hashlib
import json
from pathlib import Path
import sys

L = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(L / 'src'))
import pace_control as C

GRANT = 'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json'
GRANT_SHA = 'f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3'
REQUIRED_CPU_FILES = {
    'src/pace_quantizer.py', 'src/pace_buffer.py', 'src/pace_runtime.py',
    'src/pace_control.py', 'src/pace_continuation.py',
    'scripts/pace_worker.py', 'scripts/pace_controller.py',
    'scripts/offline_parent.py', 'scripts/finalize_cpu.py', 'scripts/run_offline.sh',
    'operations/pace_independent_delivery.py', 'operations/freeze_static_once.py',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')


def validate_final_cpu_admission(root):
    root = Path(root)
    report = read(root / 'operations/SCALE_REPAIR_FINAL_CPU_ADMISSION.json')
    C.require(report['status'] == 'PASS_TWO_FRESH_FINAL_SCALE_REPAIR_CPU_ADMISSION'
              and report['GPU_calls'] == 0 and len(report['runs']) == 2,
              'Two fresh final CPU admissions required')
    C.require(all(x['returncode'] == 0 and x['status'] == 'PASS' for x in report['runs']),
              'Both final CPU admissions must pass')
    rows = report['files']; names = [x['path'] for x in rows]
    C.require(len(names) == len(set(names)) and REQUIRED_CPU_FILES <= set(names),
              'Final CPU source coverage incomplete/duplicate')
    for row in rows:
        p = C.relative(root, row['path'])
        C.require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                  'Final CPU-qualified source changed: ' + row['path'])
    C.require(report['human_authorization_sha256'] == sha(root / GRANT) == GRANT_SHA,
              'New scale-repair human authorization differs')
    C.require(report['reuse_admission_path'] == 'analysis/PACE_REUSE3_DUAL_HOST_FRESH_REVIEW.json',
              'Exact fresh two-host fixed-origin review required')
    admission = C.relative(root, report['reuse_admission_path'])
    C.require(sha(admission) == report['reuse_admission_sha256']
              and report['reuse_admission_verified'] is True,
              'Fixed old qualification/calibration/mask/three admission differs')
    return report


def main():
    for name in ('source_manifest.json', 'GPU_PERMIT.json', 'SESSION_STARTED.json',
                 'OFFLINE_PARENT_STARTED.json', 'SCIENCE_FREEZE.json'):
        C.require(not (L / name).exists(), 'Already frozen/started: ' + name)
    report = validate_final_cpu_admission(L)
    grant = read(L / GRANT)
    C.require(grant['execution_revision'] == C.REVISION and grant['remote_root'] == C.REMOTE
              and grant['archive_root'] == C.ARCHIVE and grant['only_physical_gpu'] == 0
              and grant['only_UUID'] == C.UUID and grant['formal_total_unique'] == 60
              and grant['remaining_formal_if_reuse_admitted'] == 57,
              'One fixed new revision authorization required')
    C.require(read(L / 'FORMAL_JOB_MANIFEST.json') == C.build_jobs(read(L / 'AIME26_DATASET_FREEZE.json')),
              'Canonical original60 job roster differs')
    C.require(not (L / 'calibration').exists(),
              'Current calibration must be admitted from fixed origin after new repair gate')
    rows = []
    for path in sorted(L.rglob('*')):
        C.require(not path.is_symlink(), 'A source symlink is not allowed')
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        name = path.relative_to(L).as_posix()
        if name in ('source_manifest.json', 'GPU_PERMIT.json'):
            continue
        C.require(not any(x in name for x in ('\r', '\n', '\x00')), 'Unsafe source path')
        rows.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
    save(L / 'source_manifest.json', dict(task_id=C.TASK, execution_revision=C.REVISION,
         status='STATIC_SCALE_REPAIR_SOURCE_AND_FIXED_REUSE_PROVENANCE_FROZEN_BEFORE_GPU',
         count=len(rows), files=rows, code_or_input_changes_after_START_FORBIDDEN=True,
         prior_formal_reuse=3, current_formal_samples=57, unique_formal_target=60,
         final_CPU_admission_sha256=sha(L / 'operations/SCALE_REPAIR_FINAL_CPU_ADMISSION.json')))
    permit = dict(task_id=C.TASK, execution_revision=C.REVISION, ALLOW_GPU='YES',
                  execution_root=C.REMOTE, source_manifest_sha256=sha(L / 'source_manifest.json'),
                  method_spec_sha256=sha(L / 'method_spec.json'),
                  human_authorization_sha256=sha(L / GRANT),
                  reuse_admission_sha256=report['reuse_admission_sha256'],
                  physical_indices=[0], devices=[C.UUID], seed=1,
                  formal_new_samples=57, reused_formal_samples=3, formal_total_unique=60,
                  max_new_tokens=81920, watchdog_seconds=1800, cleanup_seconds=120,
                  GPUh_hard_cap=None, total_wall_cap=None, unit_elapsed_cap=None,
                  no_foreign_GPU_signals=True, server29_CPU_archive_only=True,
                  one_new_independent_execution=True,
                  stage_order=['fixed_origin_admission_CPU', 'SCALE_REPAIR_GATES_actual_CUDA',
                               'admit_original_once_calibration_same_mask', 'new_science_freeze',
                               'canonical57_remaining_fresh', 'once_CPU_finisher_union60_only'])
    save(L / 'GPU_PERMIT.json', permit)
    C.validate_permit_values(read(L / 'ROOT.json'), permit, grant,
                            sha(L / 'source_manifest.json'), sha(L / 'method_spec.json'), sha(L / GRANT))
    print(json.dumps(dict(status='FROZEN_ONCE_NO_GPU_OR_CHILD_STARTED', count=len(rows),
                          source_sha256=sha(L / 'source_manifest.json'),
                          permit_sha256=sha(L / 'GPU_PERMIT.json'))))


if __name__ == '__main__':
    main()
