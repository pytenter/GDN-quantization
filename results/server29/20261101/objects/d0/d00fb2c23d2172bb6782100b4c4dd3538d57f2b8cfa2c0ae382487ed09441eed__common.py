import hashlib
import json
import os
from pathlib import Path


REPO = Path('/data/zypan/worktrees/qwen-gdn-fd-scale-feedback-direction-v1')
EXP = REPO / 'experiments/QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1'
SOURCE_REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
PARENT_HISTORY = SOURCE_REPO / 'experiments/QWEN_GDN_FD_HISTORY_PULSE_CAUSAL_DIAGNOSIS_V1'
PARENT_ROOT = SOURCE_REPO / 'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
PARENT = PARENT_ROOT / 'amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1'
PARENT_SCRIPTS = PARENT / 'scripts'
CLEAN_REPO = Path('/data/zypan/worktrees/qwen-gdn-fd-path-qdq-interaction-v1')
CLEAN_EXP = CLEAN_REPO / 'experiments/QWEN_GDN_FD_CLEAN_QPARAM_CAUSAL_INTERVENTION_V1'
PYTHON = Path('/data/ydai/miniconda3/envs/bitdecode/bin/python')
FORMAL = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')

CONDITIONS = ['Native', 'H', 'C6R', 'C7']
ARMS = ['NO_SCALE_INFLATION', 'NO_SCALE_DEFLATION']
GPU_MAP = {'Native': 0, 'H': 1, 'C6R': 2, 'C7': 3}
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
          16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
FOCUS_LAYERS = [6, 28]
READPOINTS = [12288, 16384, 20480, 24576, 28672, 32768]
ANCHOR_TOKEN = 8192
ENDPOINT = 32768
FIRST_TOKEN = 1024
EXPLOSION_THRESHOLD = 1000.0
HIST_MIN = -64.0
HIST_MAX = 64.0
HIST_BINS = 65536


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


def write_text(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f'write-once artifact exists: {path}')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tensor_sha(tensor):
    import torch
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def clean_result(document, condition):
    return read_json(CLEAN_EXP / f'discovery/unit_results/doc_{document:02d}_{condition}.json')

