"""Append-only correction of engineering evidence staging for Git recovery."""
import os
from pathlib import Path
import subprocess
import sys
import traceback

REPO = Path('/data/zypan/worktrees/qwen-gdn-fd-path-qdq-interaction-v1')
EXP = REPO / 'experiments/QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1'
ENGINEERING = EXP / 'engineering/git_identity_resume_001'
sys.path.insert(0, str(EXP / 'scripts'))
from common import PYTHON, read_json, sha256, write_json
from controller import verify_freeze
from qualification_controller import freeze_commit, run


def main():
    assert not (ENGINEERING / 'resume_v2_started.json').exists()
    assert not (EXP / 'PREMEASUREMENT_COMMIT.json').exists()
    assert not (EXP / 'discovery/logs/formal_controller.log').exists()
    assert not list((EXP / 'discovery/unit_results').glob('*.json'))
    assert all(v == 'PASS' for v in read_json(EXP / 'gate_status.json').values())
    assert read_json(EXP / 'qualification/qualification_report.json')['status'] == 'PASS'
    verify_freeze()
    original = read_json(ENGINEERING / 'engineering_amendment.json')
    for row in original['original_failure_preserved']:
        assert sha256(EXP / row['copy']) == row['sha256']
        if row['path'] != 'execution/status.json':
            assert sha256(EXP / row['path']) == row['sha256']
    assert "ignored by one of your .gitignore files" in (ENGINEERING / 'resume.log').read_text()
    assert read_json(ENGINEERING / 'recovery_error.json')['exception'] == 'CalledProcessError'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == original['parent_stop_commit_preserved']
    freeze = read_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json')
    prefix = str(EXP.relative_to(REPO))
    allowed = {prefix + '/' + r['path'] for r in freeze['frozen_files']}
    allowed |= {prefix + '/PREMEASUREMENT_SHA256SUMS', prefix + '/PREMEASUREMENT_FREEZE_MANIFEST.json'}
    allowed |= {str(p.relative_to(REPO)) for p in ENGINEERING.iterdir() if p.is_file()}
    staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).splitlines())
    assert staged <= allowed, 'unrelated staged file'
    write_json(ENGINEERING / 'resume_v2_started.json', {
        'classification': 'ENGINEERING_ONLY_GIT_IGNORED_EVIDENCE_STAGING',
        'correction': 'git add -f for the explicit, already-preserved historical evidence files only',
        'scientific_changes': 0, 'qualification_repeated': False,
        'formal_measurement_count_before_commit': 0,
        'previous_resume_log_sha256': sha256(ENGINEERING / 'resume.log'),
        'previous_recovery_error_sha256': sha256(ENGINEERING / 'recovery_error.json'),
        'frozen_files_unchanged': True,
    })
    os.environ.update({
        'GIT_AUTHOR_NAME': 'Codex', 'GIT_AUTHOR_EMAIL': 'codex@local',
        'GIT_COMMITTER_NAME': 'Codex', 'GIT_COMMITTER_EMAIL': 'codex@local',
    })
    names = [
        'REPORT.md', 'engineering_amendment.json', 'historical_FAIL_AND_STOP.json',
        'historical_execution_autorun.log', 'historical_execution_doc01_capture.exitcode',
        'historical_execution_status.json', 'resume.py', 'resume_started.json',
        'recovery_error.json', 'resume.log', 'resume_v2.py', 'resume_v2_started.json',
    ]
    subprocess.check_call(['git', 'add', '-f', '--'] + [str((ENGINEERING / n).relative_to(REPO)) for n in names], cwd=REPO)
    commit = freeze_commit()
    verify_freeze()
    write_json(ENGINEERING / 'resume_commit_result.json', {
        'ENGINEERING_FIX_GATE': 'PASS', 'PREMEASUREMENT_COMMIT_SHA': commit,
        'frozen_files_unchanged': True, 'historical_failure_evidence_preserved': True,
        'formal_measurement_count_before_commit': 0,
    })
    write_json(EXP / 'execution/status.json', {
        'stage': 'PREMEASUREMENT_COMMITTED_FORMAL_START',
        'PREMEASUREMENT_COMMIT_SHA': commit,
        'historical_failure_resolution': str((ENGINEERING / 'resume_commit_result.json').relative_to(EXP)),
    }, overwrite=True)
    run('formal_controller', [str(PYTHON), str(EXP / 'scripts/controller.py')])
    write_json(ENGINEERING / 'resume_v2_exit.json', {'exitcode': 0, 'stage': 'COMPLETE'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(ENGINEERING / 'recovery_v2_error.json', {
            'FINAL_ACTION': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(),
        })
        write_json(EXP / 'execution/status.json', {
            'stage': 'FAIL_AND_STOP', 'message': str(exc),
            'latest_error': str((ENGINEERING / 'recovery_v2_error.json').relative_to(EXP)),
        }, overwrite=True)
        raise
