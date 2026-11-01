import json
import os
import subprocess
import time
import traceback

from common import EXP, PYTHON, REPO, read_json, sha256, write_json


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


def wait_capture():
    while True:
        probe = subprocess.run(['tmux','has-session','-t','qwen_clean_qparam_prepare'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode != 0:
            break
        time.sleep(30)
    seal = EXP/'discovery/doc_01/drive/capture_seal.json'
    manifest = EXP/'discovery/doc_01/drive/drive_manifest.json'
    if not seal.exists() or not manifest.exists():
        raise RuntimeError('Document 1 capture ended without sealed valid capture')
    if read_json(seal)['DOCUMENT_DRIVE_SEAL_GATE'] != 'PASS':
        raise RuntimeError('Document 1 capture seal not PASS')


def freeze_commit():
    freeze = read_json(EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json')
    paths = [str(EXP.relative_to(REPO)/row['path']) for row in freeze['frozen_files']]
    paths += [str((EXP/'PREMEASUREMENT_SHA256SUMS').relative_to(REPO)),
              str((EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json').relative_to(REPO))]
    subprocess.check_call(['git','add','--'] + paths, cwd=REPO)
    subprocess.check_call(['git','commit','-m',
                           'Freeze clean-qparam causal intervention protocol'], cwd=REPO)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    write_json(EXP/'PREMEASUREMENT_COMMIT.json', {
        'PREMEASUREMENT_COMMIT_SHA': commit,
        'frozen_manifest_sha256': sha256(EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json'),
        'formal_measurement_count_before_commit': 0})
    auth = read_json(EXP/'execution_authorization.json')
    auth['FORMAL_MEASUREMENT_AUTHORIZED'] = 'YES'
    auth['PREMEASUREMENT_COMMIT_SHA'] = commit
    write_json(EXP/'execution_authorization.json', auth, overwrite=True)
    scientific = read_json(EXP/'scientific_execution_authorization.json')
    scientific['PREMEASUREMENT_COMMIT_SHA'] = commit
    scientific['scope'] = 'FROZEN_INPUT_ACQUISITION_AND_CLEAN_QPARAM_FORMAL_MEASUREMENT'
    write_json(EXP/'scientific_execution_authorization.json', scientific, overwrite=True)
    return commit


def main():
    wait_capture()
    run('doc01_verify', [str(PYTHON), str(EXP/'scripts/verify_drive.py'),
                         '--document','1'])
    run('doc01_anchors', [str(PYTHON), str(EXP/'scripts/launch_group.py'),
                          '--mode','build_anchors','--document','1'])
    run('qualification_same_process', [str(PYTHON), str(EXP/'scripts/launch_group.py'),
                                       '--mode','same_process','--document','1','--end','8448'])
    run('qualification_fresh_1', [str(PYTHON), str(EXP/'scripts/launch_group.py'),
                                  '--mode','fresh','--document','1','--run','1','--end','8448'])
    run('qualification_finalize', [str(PYTHON), str(EXP/'scripts/qualify.py')])
    commit = freeze_commit()
    write_json(EXP/'execution/status.json', {
        'stage': 'PREMEASUREMENT_COMMITTED_FORMAL_START',
        'PREMEASUREMENT_COMMIT_SHA': commit}, overwrite=True)
    run('formal_controller', [str(PYTHON), str(EXP/'scripts/controller.py')])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (EXP/'FAIL_AND_STOP.json').exists():
            write_json(EXP/'FAIL_AND_STOP.json', {
                'FINAL_ACTION': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc(),
                'retry_authorized': False})
        write_json(EXP/'execution/status.json', {'stage': 'FAIL_AND_STOP',
                                                 'message': str(exc)}, overwrite=True)
        raise
