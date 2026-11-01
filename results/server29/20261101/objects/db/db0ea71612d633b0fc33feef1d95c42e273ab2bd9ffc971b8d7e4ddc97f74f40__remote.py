"""SSH transport only; credentials remain in the existing interactive masters."""
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH55 = ['ssh', '-S', '/tmp/qwen_gdn_replication_20261013.sock',
         '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ConnectTimeout=15',
         '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
         'zypan@202.38.247.55']
SSH29 = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock',
         '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ConnectTimeout=15',
         '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
         'zypan@202.38.247.29']


def command(role='29', python='/usr/bin/python3'):
    remote = 'exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(python) + ' -B -'
    if role == '29':
        remote = 'exec ' + shlex.join(SSH29 + [remote])
    return SSH55 + [remote]


def run(code, role='29', python='/usr/bin/python3', timeout=55):
    proc = subprocess.run(command(role, python), input=code, text=True,
                          capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError('SSH/CPU operation failed: ' + proc.stderr[-5000:] + proc.stdout[-1000:])
    return json.loads(proc.stdout)


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
