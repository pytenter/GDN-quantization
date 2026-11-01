import argparse
import os
import socket
import subprocess

from common import CONDITIONS, EXP, PYTHON, REPO, sha256, write_json


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build_anchors','qualification','smoke','formal'], required=True)
    parser.add_argument('--document', type=int, required=True, choices=range(1,5))
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    port = free_port()
    workers = []
    for rank, condition in enumerate(CONDITIONS):
        name = f'{args.mode}_doc{args.document:02d}_run{args.run}_{condition}'
        log_path = EXP / 'discovery/logs' / f'{name}.log'
        if log_path.exists():
            raise RuntimeError(f'write-once log exists: {log_path}')
        log = log_path.open('x', encoding='utf-8', buffering=1)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': str(rank), 'MASTER_ADDR': '127.0.0.1',
                    'MASTER_PORT': str(port), 'RANK': str(rank), 'WORLD_SIZE': '4',
                    'LOCAL_RANK': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        cmd = [str(PYTHON), str(EXP/'scripts/direction_replay.py'),
               '--mode', args.mode, '--document', str(args.document),
               '--rank', str(rank), '--run', str(args.run)]
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True)
        workers.append((rank, condition, proc, log, log_path, cmd))
    rows, failed = [], []
    for rank, condition, proc, log, log_path, cmd in workers:
        rc = proc.wait()
        log.close()
        row = {'rank': rank, 'condition': condition, 'pid': proc.pid,
               'exitcode': rc, 'command': cmd, 'log_sha256': sha256(log_path)}
        rows.append(row)
        if rc != 0:
            failed.append(row)
    write_json(EXP / 'execution' / f'{args.mode}_doc{args.document:02d}_run{args.run}.json', {
        'mode': args.mode, 'document_index': args.document, 'run': args.run,
        'workers': rows, 'all_pass': not failed, 'failed': failed})
    if failed:
        raise RuntimeError(f'{args.mode} group failed: {failed}')
    print(f'SCALE_DIRECTION_{args.mode.upper()}_GROUP_PASS', flush=True)


if __name__ == '__main__':
    main()

