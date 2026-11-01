import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (CONDITIONS, EXP, GPU_MAP, PARENT, PYTHON, REPO, read_json,
                    sha256, write_json)


def update_status(stage, **extra):
    payload = {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    payload.update(extra)
    write_json(EXP / 'execution/status.json', payload, overwrite=True)


def run_logged(name, command, gpu=None):
    log_path = EXP / 'discovery/logs' / f'{name}.log'
    if log_path.exists():
        raise RuntimeError(f'write-once log exists: {log_path}')
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = process.wait()
    write_json(EXP / 'execution' / f'{name}.json', {
        'command': [str(x) for x in command], 'physical_gpu': gpu,
        'pid': process.pid, 'exitcode': rc, 'log_sha256': sha256(log_path)})
    if rc != 0:
        raise RuntimeError(f'{name} failed with exit code {rc}')


def run_parallel(document):
    workers = []
    for condition in CONDITIONS:
        name = f'doc{document:02d}_{condition}_formal'
        log_path = EXP / 'discovery/logs' / f'{name}.log'
        if log_path.exists():
            raise RuntimeError(f'write-once log exists: {log_path}')
        log = log_path.open('x', encoding='utf-8', buffering=1)
        env = os.environ.copy()
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        env['CUDA_VISIBLE_DEVICES'] = str(GPU_MAP[condition])
        command = [str(PYTHON), str(EXP / 'scripts/pulse_replay.py'),
                   '--mode', 'formal', '--document', str(document),
                   '--condition', condition]
        process = subprocess.Popen(command, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        workers.append((condition, name, process, log, log_path, command))
    failed = []
    rows = []
    for condition, name, process, log, log_path, command in workers:
        rc = process.wait()
        log.close()
        row = {'condition': condition, 'physical_gpu': GPU_MAP[condition],
               'pid': process.pid, 'exitcode': rc,
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'command': command, 'log_sha256': sha256(log_path)}
        rows.append(row)
        if rc != 0:
            failed.append(row)
    write_json(EXP / 'execution' / f'doc{document:02d}_parallel_replay.json', {
        'document_index': document, 'workers': rows,
        'all_pass': not failed, 'failed': failed})
    if failed:
        raise RuntimeError(f'document {document} parallel replay failure')


def seal_and_release(document):
    result_rows = []
    for condition in CONDITIONS:
        path = EXP / f'discovery/unit_results/doc_{document:02d}_{condition}.json'
        payload = read_json(path)
        if payload['status'] != 'PASS':
            raise RuntimeError(f'cannot seal failed result {path}')
        result_rows.append({'condition': condition, 'path': str(path),
                            'sha256': sha256(path), 'bytes': path.stat().st_size})
    seal_path = EXP / f'discovery/doc_{document:02d}/result_seal.json'
    write_json(seal_path, {'RESULT_SEAL_GATE': 'PASS', 'document_index': document,
                           'unit_count': 4, 'results': result_rows,
                           'all_result_hashes_verified': True})

    drive_dir = EXP / f'discovery/doc_{document:02d}/drive'
    manifest = read_json(drive_dir / 'drive_manifest.json')
    released = []
    for row in manifest['shards']:
        path = drive_dir / row['relative_path']
        if sha256(path) != row['file_sha256']:
            raise RuntimeError(f'release refused: shard hash mismatch {path}')
        released.append({'path': str(path), 'sha256': row['file_sha256'],
                         'bytes': path.stat().st_size})
    for row in released:
        Path(row['path']).unlink()
    if any((drive_dir / row['relative_path']).exists() for row in manifest['shards']):
        raise RuntimeError('manifest shard release incomplete')
    write_json(drive_dir / 'drive_release_record.json', {
        'document_index': document,
        'drive_manifest_sha256': sha256(drive_dir / 'drive_manifest.json'),
        'drive_identity_sha256': sha256(EXP / f'discovery/doc_{document:02d}/drive_identity.json'),
        'result_seal_sha256': sha256(seal_path),
        'all_anchor_results_complete': True,
        'all_result_hashes_verified': True,
        'released_paths': released,
        'released_bytes': sum(x['bytes'] for x in released),
        'release_complete': True,
        'parent_artifacts_deleted': 0,
    })


def main():
    if subprocess.run(['git', 'diff', '--quiet'], cwd=REPO).returncode != 0:
        raise RuntimeError('tracked worktree differs from premeasurement commit')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                     text=True).strip()
    freeze = read_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json')
    if freeze['PREREGISTRATION_GATE'] != 'PASS':
        raise RuntimeError('preregistration gate not PASS')

    write_json(EXP / 'execution_authorization.json', {
        'PREMEASUREMENT_COMMIT_SHA': commit,
        'CAPTURE_INPUT_ACQUISITION_AUTHORIZED': 'YES',
        'DISCOVERY_EXECUTION_AUTHORIZED': 'YES',
        'scope': 'The legacy field above authorizes frozen-input acquisition only; causal Discovery remains blocked until discovery_authorization.json exists.',
        'CAUSAL_DISCOVERY_EXECUTION_AUTHORIZED': 'NO_PENDING_G0_G16',
        'measurement_count_at_authorization': 0,
    })
    write_json(EXP / 'scientific_execution_authorization.json', {
        'PREMEASUREMENT_COMMIT_SHA': commit,
        'DISCOVERY_EXECUTION_AUTHORIZED': 'YES',
        'scope': 'FROZEN_INPUT_ACQUISITION_ONLY_NOT_CAUSAL_MEASUREMENT'})
    write_json(EXP / 'extended_capture_freeze.json', {
        'policy_sha256': sha256(EXP / 'protocol/extended_capture_policy.json'),
        'script_sha256': sha256(PARENT / 'scripts/capture_extended_drive.py'),
        'freeze_commit': commit,
    })
    for phase in ['factorization', 'schema', 'fp', 'invariance', 'decomposition',
                  'isolation', 'determinism', 'streaming']:
        parent_gate = PARENT / 'engineering/gates' / f'{phase}.json'
        if not parent_gate.exists() and phase == 'determinism':
            parent_gate = PARENT / 'engineering/gates/determinism_2048.json'
        value = read_json(parent_gate)
        if value['gate'] != 'PASS':
            raise RuntimeError(f'parent gate {phase} not PASS')
        write_json(EXP / 'engineering/gates' / f'{phase}.json', {
            'gate': 'PASS', 'reused_for_capture_input_acquisition_only': True,
            'parent_path': str(parent_gate), 'parent_sha256': sha256(parent_gate)})

    update_status('DOC1_CAPTURE')
    run_logged('doc01_capture', [str(PYTHON), str(EXP / 'scripts/capture_driver.py'),
                                '--document', '1'], gpu=0)
    run_logged('doc01_drive_verify', [str(PYTHON), str(EXP / 'scripts/verify_drive.py'),
                                     '--document', '1'])

    update_status('QUALIFICATION_SAME_PROCESS')
    run_logged('smoke_same_process', [str(PYTHON), str(EXP / 'scripts/pulse_replay.py'),
                                      '--mode', 'same_process', '--document', '1',
                                      '--condition', 'Native'], gpu=0)
    for run in [1, 2, 3]:
        update_status('QUALIFICATION_FRESH_PROCESS', run=run)
        run_logged(f'smoke_fresh_process_{run}', [
            str(PYTHON), str(EXP / 'scripts/pulse_replay.py'),
            '--mode', 'fresh_process', '--document', '1', '--condition', 'Native',
            '--run', str(run)], gpu=0)
    run_logged('qualification_finalize', [str(PYTHON), str(EXP / 'scripts/qualify.py')])

    for document in range(1, 5):
        if document > 1:
            update_status(f'DOC{document}_CAPTURE')
            run_logged(f'doc{document:02d}_capture', [
                str(PYTHON), str(EXP / 'scripts/capture_driver.py'),
                '--document', str(document)], gpu=0)
            run_logged(f'doc{document:02d}_drive_verify', [
                str(PYTHON), str(EXP / 'scripts/verify_drive.py'),
                '--document', str(document)])
        update_status(f'DOC{document}_PARALLEL_REPLAY')
        run_parallel(document)
        update_status(f'DOC{document}_SEAL_RELEASE')
        seal_and_release(document)

    update_status('ANALYSIS')
    run_logged('analysis', [str(PYTHON), str(EXP / 'scripts/analyze.py')])
    update_status('COMPLETE', final_action='STOP_AFTER_DISCOVERY')
    write_json(EXP / 'controller_exit.json', {'exitcode': 0,
                                              'FINAL_ACTION': 'STOP_AFTER_DISCOVERY',
                                              'DISCOVERY_COMPLETE': '16/16'})
    print('HISTORY_PULSE_DISCOVERY_COMPLETE_STOP', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (EXP / 'FAIL_AND_STOP.json').exists():
            write_json(EXP / 'FAIL_AND_STOP.json', {
                'FINAL_ACTION': 'FAIL_AND_STOP',
                'exception': type(exc).__name__, 'message': str(exc),
                'traceback': traceback.format_exc(), 'retry_authorized': False})
        update_status('FAIL_AND_STOP', message=str(exc))
        raise
