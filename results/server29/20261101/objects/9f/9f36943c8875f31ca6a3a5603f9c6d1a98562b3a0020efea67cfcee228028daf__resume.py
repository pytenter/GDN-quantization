"""Engineering-only resume at the failed Git premeasurement commit boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path('/data/zypan/worktrees/qwen-gdn-fd-path-qdq-interaction-v1')
EXP = REPO / 'experiments/QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1'
ENGINEERING = EXP / 'engineering/git_identity_resume_001'
sys.path.insert(0, str(EXP / 'scripts'))
from common import PYTHON, read_json, sha256, write_json, write_text
from controller import verify_freeze
from qualification_controller import freeze_commit, run


def main():
    if (ENGINEERING / 'resume_started.json').exists():
        raise RuntimeError('write-once recovery already started')
    gates = read_json(EXP / 'gate_status.json')
    expected = {
        'PARENT_ARTIFACT_IDENTITY', 'SCIENTIFIC_ASSET_IDENTITY',
        'INTERVENTION_IDENTIFIABILITY', 'INTERVENTION_ACTIVATION',
        'QPARAM_BASIS_VALIDATION', 'EVERY_TOKEN_INT8_WRITEBACK',
        'SHORT_REPLAY_DETERMINISM', 'STORAGE',
    }
    assert set(gates) == expected and all(v == 'PASS' for v in gates.values())
    qualification = read_json(EXP / 'qualification/qualification_report.json')
    assert qualification['status'] == 'PASS'
    assert qualification['scientific_measurement_count'] == 0
    assert qualification['same_process_identities'] == qualification['fresh_process_identities']
    assert read_json(EXP / 'discovery/doc_01/drive/capture_seal.json')['DOCUMENT_DRIVE_SEAL_GATE'] == 'PASS'
    assert read_json(EXP / 'execution/status.json')['stage'] == 'FAIL_AND_STOP'
    failure = read_json(EXP / 'FAIL_AND_STOP.json')
    assert failure['exception'] == 'CalledProcessError'
    assert "git', 'commit'" in failure['message'] and '128' in failure['message']
    assert 'fatal: empty ident name' in (EXP / 'execution/autorun.log').read_text()
    assert not (EXP / 'PREMEASUREMENT_COMMIT.json').exists()
    assert not (EXP / 'discovery/logs/formal_controller.log').exists()
    assert not list((EXP / 'discovery/unit_results').glob('*.json'))
    assert subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=REPO, text=True).strip() == 'exp/qwen-gdn-fd-path-qdq-interaction-v1'
    old_commit = 'fcd1dff6d286743732c6d20305cfdd31d9de7751'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == old_commit
    assert subprocess.check_output(['git', 'log', '-1', '--format=%an <%ae>'], cwd=REPO, text=True).strip() == 'Codex <codex@local>'
    verify_freeze()
    freeze = read_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json')
    prefix = str(EXP.relative_to(REPO))
    expected_index = {prefix + '/' + row['path'] for row in freeze['frozen_files']}
    expected_index |= {prefix + '/PREMEASUREMENT_SHA256SUMS', prefix + '/PREMEASUREMENT_FREEZE_MANIFEST.json'}
    staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).splitlines())
    assert staged == expected_index, 'unrelated staged files or incomplete freeze staging'
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=REPO, text=True).strip(), 'tracked unstaged changes'

    ENGINEERING.mkdir(parents=True, exist_ok=True)
    preserved = []
    for relative in ['FAIL_AND_STOP.json', 'execution/status.json', 'execution/autorun.log', 'execution/doc01_capture.exitcode']:
        source = EXP / relative
        target = ENGINEERING / ('historical_' + relative.replace('/', '_'))
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha256(source) == sha256(target)
        preserved.append({'path': relative, 'sha256': sha256(source), 'copy': str(target.relative_to(EXP))})
    report = {
        'classification': 'ENGINEERING_ONLY_GIT_IDENTITY_MANIFEST_RESUME',
        'authorization': 'User task section 25: launcher/manifest/resume append-only fix, test, resume',
        'failing_path': 'scripts/qualification_controller.py:50 freeze_commit()',
        'observed_error': 'fatal: empty ident name (for <zypan@hostname.(none)>) not allowed',
        'correction': 'Process-local GIT_AUTHOR_NAME/GIT_COMMITTER_NAME=Codex and GIT_AUTHOR_EMAIL/GIT_COMMITTER_EMAIL=codex@local, matching previous commit',
        'global_or_repository_git_config_changed': False,
        'scientific_code_or_frozen_files_changed': False,
        'frozen_file_set_sha256': freeze['frozen_file_set_sha256'],
        'gates_reused': gates,
        'qualification_repeated': False,
        'capture_repeated': False,
        'measurement_count_before_resume': 0,
        'original_failure_preserved': preserved,
        'resume_script_sha256': sha256(Path(__file__)),
        'resume_from': 'freeze_commit(); existing formal_controller',
        'parent_stop_commit_preserved': old_commit,
    }
    write_json(ENGINEERING / 'engineering_amendment.json', report)
    write_text(ENGINEERING / 'REPORT.md', '# Git identity engineering recovery\n\nThe eight qualification gates passed before measurement. Git commit creation alone failed because the identity name was empty. This recovery sets the author and committer identity only in its process environment, using the identity in the previous commit, then calls the unchanged freeze_commit() and formal controller. No capture, anchor qualification, smoke, scientific computation, frozen file, or numerical runtime setting is changed or repeated. The original FAIL_AND_STOP.json and autorun.log remain untouched; exact copies and hashes are retained here.\n')
    write_json(ENGINEERING / 'resume_started.json', {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': 'GIT_COMMIT_ONLY_RESUME'})
    os.environ.update({
        'GIT_AUTHOR_NAME': 'Codex', 'GIT_AUTHOR_EMAIL': 'codex@local',
        'GIT_COMMITTER_NAME': 'Codex', 'GIT_COMMITTER_EMAIL': 'codex@local',
    })
    evidence_paths = [str(path.relative_to(REPO)) for path in sorted(ENGINEERING.iterdir()) if path.is_file() and path.name != 'resume.log']
    subprocess.check_call(['git', 'add', '--'] + evidence_paths, cwd=REPO)
    commit = freeze_commit()
    verify_freeze()
    for row in preserved:
        assert sha256(EXP / row['path']) == row['sha256']
    write_json(ENGINEERING / 'resume_commit_result.json', {
        'ENGINEERING_FIX_GATE': 'PASS', 'PREMEASUREMENT_COMMIT_SHA': commit,
        'frozen_files_unchanged': True, 'original_failure_and_logs_unchanged': True,
        'formal_measurement_count_before_commit': 0,
    })
    write_json(EXP / 'execution/status.json', {
        'stage': 'PREMEASUREMENT_COMMITTED_FORMAL_START',
        'PREMEASUREMENT_COMMIT_SHA': commit,
        'historical_failure_resolution': str((ENGINEERING / 'resume_commit_result.json').relative_to(EXP)),
    }, overwrite=True)
    run('formal_controller', [str(PYTHON), str(EXP / 'scripts/controller.py')])
    write_json(ENGINEERING / 'resume_exit.json', {'exitcode': 0, 'stage': 'COMPLETE'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(ENGINEERING / 'recovery_error.json', {
            'FINAL_ACTION': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(),
        })
        write_json(EXP / 'execution/status.json', {
            'stage': 'FAIL_AND_STOP', 'message': str(exc),
            'latest_error': str((ENGINEERING / 'recovery_error.json').relative_to(EXP)),
        }, overwrite=True)
        raise
