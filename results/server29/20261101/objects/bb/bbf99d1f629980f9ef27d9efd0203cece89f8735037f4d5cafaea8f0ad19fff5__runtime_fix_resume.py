import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (CONDITIONS, EXP, GPU_MAP, PYTHON, REPO, read_json, sha256,
                    write_json)
from controller import run_logged, run_parallel, seal_and_release, update_status


FIX_CLASS = 'ENGINEERING_ONLY_DETERMINISTIC_RUNTIME_FIX'
FIX_DIR = EXP / 'engineering/deterministic_runtime_fix'
ROTATED = ('H', 'C6R', 'C7')
WORKSPACE = ':4096:8'


def child_env(condition):
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
    env['CUDA_VISIBLE_DEVICES'] = str(GPU_MAP[condition])
    return env


def run_parallel_commands(stage, script, arguments, log_prefix):
    workers = []
    for condition in ROTATED:
        log_path = EXP / 'discovery/logs' / f'{log_prefix}_{condition}.log'
        if log_path.exists():
            raise RuntimeError(f'write-once recovery log exists: {log_path}')
        log = log_path.open('x', encoding='utf-8', buffering=1)
        command = [str(PYTHON), str(script)] + list(arguments(condition))
        process = subprocess.Popen(command, cwd=REPO, env=child_env(condition),
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        workers.append((condition, process, log, log_path, command))
    rows = []
    failed = []
    for condition, process, log, log_path, command in workers:
        rc = process.wait()
        log.close()
        row = {
            'condition': condition,
            'physical_gpu': GPU_MAP[condition],
            'pid': process.pid,
            'exitcode': rc,
            'command': command,
            'CUBLAS_WORKSPACE_CONFIG': WORKSPACE,
            'CUDA_VISIBLE_DEVICES': str(GPU_MAP[condition]),
            'log_path': str(log_path),
            'log_sha256': sha256(log_path),
        }
        rows.append(row)
        if rc != 0:
            failed.append(row)
    report = {
        'stage': stage,
        'classification': FIX_CLASS,
        'workers': rows,
        'all_pass': not failed,
        'failed': failed,
    }
    write_json(FIX_DIR / f'{stage}.json', report)
    if failed:
        raise RuntimeError(f'{stage} failed')
    return report


def preserve_and_validate_failure():
    failure = read_json(EXP / 'FAIL_AND_STOP.json')
    if failure.get('message') != 'document 1 parallel replay failure':
        raise RuntimeError('historical failure signature changed')
    parallel_path = EXP / 'execution/doc01_parallel_replay.json'
    parallel = read_json(parallel_path)
    if parallel.get('all_pass') is not False:
        raise RuntimeError('historical parallel failure evidence changed')
    failed = {row['condition'] for row in parallel['failed']}
    if failed != set(ROTATED):
        raise RuntimeError('historical affected-unit set changed')
    native_path = EXP / 'discovery/unit_results/doc_01_Native.json'
    native = read_json(native_path)
    if native.get('status') != 'PASS' or native.get('mode') != 'formal':
        raise RuntimeError('Document 1 Native PASS result missing')
    evidence = {
        'classification': FIX_CLASS,
        'historical_FAIL_AND_STOP_sha256': sha256(EXP / 'FAIL_AND_STOP.json'),
        'historical_parallel_report_sha256': sha256(parallel_path),
        'Document_1_Native_GPU0': 'PASS_PRESERVED_NOT_RERUN',
        'Document_1_H_GPU1': 'FAILED_STARTUP_PRESERVED',
        'Document_1_C6R_GPU2': 'FAILED_STARTUP_PRESERVED',
        'Document_1_C7_GPU3': 'FAILED_STARTUP_PRESERVED',
        'old_logs_preserved': {},
    }
    for condition in ROTATED:
        result_path = EXP / f'discovery/unit_results/doc_01_{condition}.json'
        if result_path.exists():
            raise RuntimeError(f'unexpected pre-recovery result exists: {result_path}')
        log_path = EXP / f'discovery/logs/doc01_{condition}_formal.log'
        error_path = EXP / f'discovery/logs/formal_doc1_{condition}.error.json'
        if not log_path.exists() or not error_path.exists():
            raise RuntimeError(f'historical failure artifact missing: {condition}')
        evidence['old_logs_preserved'][condition] = {
            'log_path': str(log_path),
            'log_sha256': sha256(log_path),
            'error_path': str(error_path),
            'error_sha256': sha256(error_path),
        }
    write_json(FIX_DIR / 'historical_failure_evidence.json', evidence)
    return evidence


def main():
    if subprocess.run(['git', 'diff', '--quiet'], cwd=REPO).returncode != 0:
        raise RuntimeError('tracked worktree is not clean at runtime-fix resume')
    fix_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                         text=True).strip()
    update_status('ENGINEERING_RUNTIME_FIX_EVIDENCE_PRESERVATION',
                  classification=FIX_CLASS)
    evidence = preserve_and_validate_failure()

    update_status('ENGINEERING_RUNTIME_FIX_STARTUP_SMOKE',
                  conditions=list(ROTATED))
    smoke = run_parallel_commands(
        'startup_smoke', EXP / 'scripts/deterministic_startup_smoke.py',
        lambda condition: ['--condition', condition], 'runtimefix_smoke')
    for condition in ROTATED:
        row = read_json(FIX_DIR / f'startup_smoke_{condition}.json')
        required = {
            'status': 'PASS',
            'CUBLAS_WORKSPACE_CONFIG': WORKSPACE,
            'CUDA_VISIBLE_DEVICES': str(GPU_MAP[condition]),
            'deterministic_algorithms_enabled': True,
            'cuda_matmul_allow_tf32': False,
            'cudnn_allow_tf32': False,
        }
        for key, expected in required.items():
            if row.get(key) != expected:
                raise RuntimeError(f'startup smoke runtime mismatch: {condition} {key}')

    update_status('DOC1_ROTATED_RUNTIME_FIX_REPLAY', conditions=list(ROTATED))
    rerun = run_parallel_commands(
        'doc01_rotated_rerun', EXP / 'scripts/pulse_replay.py',
        lambda condition: ['--mode', 'formal', '--document', '1',
                           '--condition', condition],
        'doc01_formal_runtimefix_001')
    for condition in ROTATED:
        result = read_json(EXP / f'discovery/unit_results/doc_01_{condition}.json')
        if result.get('status') != 'PASS' or result.get('mode') != 'formal':
            raise RuntimeError(f'recovered unit not PASS: {condition}')

    write_json(FIX_DIR / 'runtime_fix_closure.json', {
        'classification': FIX_CLASS,
        'status': 'PASS',
        'fix_commit': fix_commit,
        'CUBLAS_WORKSPACE_CONFIG': WORKSPACE,
        'scientific_protocol_amendment': False,
        'scientific_configuration_changed': False,
        'Document_1_Native_GPU0': 'PASS_PRESERVED_NOT_RERUN',
        'Document_1_H_GPU1': 'PASS_AFTER_AUTHORIZED_STARTUP_RETRY',
        'Document_1_C6R_GPU2': 'PASS_AFTER_AUTHORIZED_STARTUP_RETRY',
        'Document_1_C7_GPU3': 'PASS_AFTER_AUTHORIZED_STARTUP_RETRY',
        'historical_evidence_sha256': sha256(FIX_DIR / 'historical_failure_evidence.json'),
        'startup_smoke_report_sha256': sha256(FIX_DIR / 'startup_smoke.json'),
        'rerun_report_sha256': sha256(FIX_DIR / 'doc01_rotated_rerun.json'),
        'L9_TRAINING_EXECUTED': 'NO',
        'NEW_ROTATION_TRAINING_EXECUTED': 'NO',
        'AIME_EXECUTED': 'NO',
        'FREE_GENERATION_EXECUTED': 'NO',
    })

    update_status('DOC1_SEAL_RELEASE_AFTER_RUNTIME_FIX')
    seal_and_release(1)

    for document in range(2, 5):
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
    update_status('COMPLETE', final_action='STOP_AFTER_DISCOVERY',
                  historical_fail_and_stop_preserved=True,
                  engineering_runtime_fix=FIX_CLASS)
    write_json(EXP / 'controller_exit.json', {
        'exitcode': 0,
        'FINAL_ACTION': 'STOP_AFTER_DISCOVERY',
        'DISCOVERY_COMPLETE': '16/16',
        'historical_FAIL_AND_STOP_preserved': True,
        'engineering_runtime_fix': FIX_CLASS,
        'L9_TRAINING_EXECUTED': 'NO',
        'NEW_ROTATION_TRAINING_EXECUTED': 'NO',
        'AIME_EXECUTED': 'NO',
        'FREE_GENERATION_EXECUTED': 'NO',
    })
    print('HISTORY_PULSE_DISCOVERY_COMPLETE_AFTER_RUNTIME_FIX_STOP', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        output = FIX_DIR / 'runtime_fix_fail_and_stop.json'
        if not output.exists():
            write_json(output, {
                'status': 'FAIL_AND_STOP',
                'classification': FIX_CLASS,
                'exception': type(exc).__name__,
                'message': str(exc),
                'traceback': traceback.format_exc(),
                'retry_authorized': False,
            })
        update_status('FAIL_AND_STOP_AFTER_ENGINEERING_RUNTIME_FIX',
                      message=str(exc), classification=FIX_CLASS)
        raise
