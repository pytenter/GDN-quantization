import hashlib
import json
import os
from pathlib import Path

REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
EXP = REPO / 'experiments/QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1'
PARENT_ROOT = REPO / 'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
PARENT = PARENT_ROOT / 'amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1'
PARENT_SCRIPTS = PARENT / 'scripts'
PYTHON = Path('/data/ydai/miniconda3/envs/bitdecode/bin/python')
CONDITIONS = ['Native', 'H', 'C6R', 'C7']
GPU_MAP = {'Native': 0, 'H': 1, 'C6R': 2, 'C7': 3}
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
          16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
CHECKPOINTS = [8192, 12288, 16384, 20480, 24576, 28672, 32768]
FIRST_TOKEN = 1024
ANCHOR_TOKEN = 8192
ENDPOINT = 32768
EXPLOSION_THRESHOLD = 1000.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f'write-once artifact exists: {path}')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text(path, text, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f'write-once artifact exists: {path}')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_sum_file(root, manifest):
    rows = []
    for line in Path(manifest).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, rel = line.split('  ', 1)
        target = Path(root) / rel
        actual = sha256(target)
        rows.append({'path': rel, 'expected': expected, 'actual': actual,
                     'pass': actual == expected})
        if actual != expected:
            raise RuntimeError(f'hash mismatch: {target}')
    return rows


def tensor_sha(tensor):
    import torch
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
