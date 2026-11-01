import os
import subprocess
import time
import traceback
from pathlib import Path

from common import CONDITIONS, EXP, PARENT, PYTHON, REPO, read_json, sha256, write_json


def status(stage, **extra):
    value = {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    value.update(extra)
    write_json(EXP/'execution/status.json', value, overwrite=True)


def run(name, command, gpu=None):
    log_path = EXP/'discovery/logs'/f'{name}.log'
    if log_path.exists():
        raise RuntimeError(f'write-once log exists: {log_path}')
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        p = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    write_json(EXP/'execution'/f'{name}.json', {
        'command': [str(x) for x in command], 'physical_gpu': gpu,
        'pid': p.pid, 'exitcode': rc, 'log_sha256': sha256(log_path)})
    if rc != 0:
        raise RuntimeError(f'{name} failed rc={rc}')


def seal_release(doc):
    results = []
    for condition in CONDITIONS:
        path = EXP/f'discovery/unit_results/doc_{doc:02d}_{condition}.json'
        value = read_json(path)
        if value['status'] != 'PASS':
            raise RuntimeError(f'cannot seal failed result: {path}')
        results.append({'condition': condition, 'path': str(path),
                        'sha256': sha256(path), 'bytes': path.stat().st_size})
    seal = EXP/f'discovery/doc_{doc:02d}/result_seal.json'
    write_json(seal, {'RESULT_SEAL_GATE': 'PASS', 'document_index': doc,
                      'unit_count': 4, 'results': results,
                      'all_result_hashes_verified': True})
    drive = EXP/f'discovery/doc_{doc:02d}/drive'
    manifest = read_json(drive/'drive_manifest.json')
    released = []
    for row in manifest['shards']:
        path = drive/row['relative_path']
        if sha256(path) != row['file_sha256']:
            raise RuntimeError(f'release refused for hash mismatch: {path}')
        released.append({'path': str(path), 'sha256': row['file_sha256'],
                         'bytes': path.stat().st_size})
    for row in released:
        Path(row['path']).unlink()
    anchor_released = []
    for condition in CONDITIONS:
        path = EXP/f'qualification/doc_{doc:02d}/anchors/{condition}.pt'
        meta = read_json(path.with_suffix('.json'))
        if sha256(path) != meta['file_sha256']:
            raise RuntimeError('anchor release refused for hash mismatch')
        anchor_released.append({'path': str(path), 'sha256': meta['file_sha256'],
                                'bytes': path.stat().st_size})
    for row in anchor_released:
        Path(row['path']).unlink()
    write_json(drive/'drive_release_record.json', {
        'document_index': doc, 'drive_manifest_sha256': sha256(drive/'drive_manifest.json'),
        'drive_identity_sha256': sha256(EXP/f'discovery/doc_{doc:02d}/drive_identity.json'),
        'result_seal_sha256': sha256(seal), 'all_anchor_results_complete': True,
        'all_result_hashes_verified': True, 'released_paths': released,
        'released_anchor_paths': anchor_released,
        'released_bytes': sum(x['bytes'] for x in released+anchor_released),
        'release_complete': True, 'parent_artifacts_deleted': 0})


def verify_freeze():
    freeze = read_json(EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json')
    for row in freeze['frozen_files']:
        if sha256(EXP/row['path']) != row['sha256']:
            raise RuntimeError(f'premeasurement frozen file changed: {row["path"]}')


def main():
    auth = read_json(EXP/'execution_authorization.json')
    if auth['FORMAL_MEASUREMENT_AUTHORIZED'] != 'YES' or not auth['PREMEASUREMENT_COMMIT_SHA']:
        raise RuntimeError('formal authorization missing')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() != auth['PREMEASUREMENT_COMMIT_SHA']:
        raise RuntimeError('HEAD is not premeasurement commit')
    verify_freeze()
    for doc in range(1,5):
        if doc > 1:
            status(f'DOC{doc}_CAPTURE')
            run(f'doc{doc:02d}_capture', [str(PYTHON), str(EXP/'scripts/capture_driver.py'),
                                          '--document', str(doc)], gpu=0)
            run(f'doc{doc:02d}_verify', [str(PYTHON), str(EXP/'scripts/verify_drive.py'),
                                         '--document', str(doc)])
            run(f'doc{doc:02d}_anchors', [str(PYTHON), str(EXP/'scripts/launch_group.py'),
                                          '--mode','build_anchors','--document',str(doc)])
        status(f'DOC{doc}_FORMAL')
        run(f'doc{doc:02d}_formal', [str(PYTHON), str(EXP/'scripts/launch_group.py'),
                                     '--mode','formal','--document',str(doc)])
        status(f'DOC{doc}_SEAL_RELEASE')
        seal_release(doc)
    status('ANALYSIS')
    run('analysis', [str(PYTHON), str(EXP/'scripts/analyze.py')])
    status('COMPLETE', final_action='STOP')
    write_json(EXP/'controller_exit.json', {'exitcode': 0, 'DISCOVERY_COMPLETE': '16/16',
                                            'FINAL_ACTION': 'STOP'})
    print('CLEAN_QPARAM_CAUSAL_INTERVENTION_COMPLETE_STOP', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (EXP/'FAIL_AND_STOP.json').exists():
            write_json(EXP/'FAIL_AND_STOP.json', {
                'FINAL_ACTION': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc(),
                'retry_authorized': False})
        status('FAIL_AND_STOP', message=str(exc))
        raise
