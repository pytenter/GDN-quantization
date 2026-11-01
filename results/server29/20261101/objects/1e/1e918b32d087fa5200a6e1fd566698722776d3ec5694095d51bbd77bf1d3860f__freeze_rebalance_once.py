"""Freeze the CPU-admitted54+6 once, before any new GPU dispatch."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pace_control as C
import pace_four_gpu_admission as A


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def new(p, v):
    with p.open('x') as f:
        json.dump(v, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def main():
    for name in ('source_manifest.json', 'GPU_PERMIT.json', 'SESSION_STARTED.json', 'OFFLINE_PARENT_STARTED.json', 'SCIENCE_FREEZE.json'):
        assert not (ROOT / name).exists(), name
    report = read(ROOT / 'engineering/REBALANCE_FINAL_CPU_ADMITTED_TWO_FRESH/RESULT.json')
    assert len(report['children']) == 8 and all(x['actual_returncode'] == 0 and x['actual_reaped'] and x['proc_directory_gone'] for x in report['children'])
    for row in report['children']:
        script = Path(row['argv'][2])
        relative = script.relative_to(C.REMOTE)
        assert sha(ROOT / relative) == row['script_sha256']
        assert sha(ROOT / row['log']) == row['log_sha256']
    reuse = read(ROOT / 'REUSE_ADMISSION.json')
    plan = read(ROOT / 'FOUR_GPU_PLAN.json')
    grant = read(ROOT / 'operations/HUMAN_4GPU_AUTHORIZATION.json')
    assert len(reuse['reused_ids']) == 54 and len(plan['jobs']) == 6
    assert len({x['unit'] for x in plan['jobs']} | set(reuse['reused_ids'])) == 60
    assert grant['human_answer'] == '接受，等q24完成后重排四卡'
    admission = dict(status='PASS_TWO_FRESH_TARGET_CPU_REBALANCE_54_REUSE_6_FORMAL', GPU_calls=0,
                     report='engineering/REBALANCE_FINAL_CPU_ADMITTED_TWO_FRESH/RESULT.json',
                     report_sha256=sha(ROOT / 'engineering/REBALANCE_FINAL_CPU_ADMITTED_TWO_FRESH/RESULT.json'),
                     children=report['children'], reused=54, new_formal=6,
                     no_numeric_or_lifecycle_changes=True)
    new(ROOT / 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json', admission) if not (ROOT / 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json').exists() else None
    if (ROOT / 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json').exists():
        assert read(ROOT / 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json') == admission, 'Inherited admission must be moved to provenance before freeze'
    rows = []
    for p in sorted(ROOT.rglob('*')):
        assert not p.is_symlink(), str(p)
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
    source = dict(task_id=C.TASK, execution_revision=C.REVISION,
                  status='FROZEN_EXPLICIT_HUMAN_POST_Q24_FOUR_GPU_REBALANCE_54_REUSE_6_FRESH',
                  count=len(rows), files=rows, code_or_input_changes_after_START_FORBIDDEN=True,
                  prior_formal_reuse=54, current_formal_samples=6, unique_formal_target=60,
                  final_CPU_admission_sha256=sha(ROOT / 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json'),
                  method_variant=C.METHOD_VARIANT, state_scale_format=C.STATE_SCALE_FORMAT)
    new(ROOT / 'source_manifest.json', source)
    permit = dict(task_id=C.TASK, execution_revision=C.REVISION, ALLOW_GPU='YES', execution_root=C.REMOTE,
                  source_manifest_sha256=sha(ROOT / 'source_manifest.json'), method_spec_sha256=sha(ROOT / 'method_spec.json'),
                  method_variant=C.METHOD_VARIANT, state_scale_format=C.STATE_SCALE_FORMAT,
                  human_authorization_sha256=sha(ROOT / 'operations/HUMAN_4GPU_AUTHORIZATION.json'),
                  four_gpu_plan_sha256=sha(ROOT / 'FOUR_GPU_PLAN.json'), reuse_admission_sha256=sha(ROOT / 'REUSE_ADMISSION.json'),
                  physical_indices=[0, 1, 2, 3], devices=list(A.UUIDS.values()), seed=1,
                  formal_new_samples=6, reused_formal_samples=54, formal_total_unique=60,
                  max_new_tokens=81920, watchdog_seconds=1800, cleanup_seconds=120,
                  GPUh_hard_cap=None, total_wall_cap=None, unit_elapsed_cap=None,
                  no_foreign_GPU_signals=True, no_automatic_GPU_retry=True, server29_CPU_archive_only=True,
                  one_new_independent_execution=True)
    A.check_permit(permit, grant, sha(ROOT / 'source_manifest.json'), sha(ROOT / 'operations/HUMAN_4GPU_AUTHORIZATION.json'), sha(ROOT / 'FOUR_GPU_PLAN.json'))
    new(ROOT / 'GPU_PERMIT.json', permit)
    print(json.dumps(dict(status='FROZEN_ONCE_NO_GPU_STARTED', count=len(rows), bytes=sum(x['bytes'] for x in rows), source_sha256=sha(ROOT / 'source_manifest.json'), permit_sha256=sha(ROOT / 'GPU_PERMIT.json'))))


if __name__ == '__main__':
    main()
