#!/usr/bin/env python3
"""Read-only inspection of the explicitly authorized GPU 6/7 training host."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
PAIR = (6, 7)


def run(*args: str) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    inventory_text = run('nvidia-smi', '--query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader,nounits')
    rows = list(csv.reader(inventory_text.splitlines()))
    inventory = [{
        'index': int(row[0].strip()), 'name': row[1].strip(),
        'pci_bus_id': row[2].strip(), 'memory_total_mib': int(row[3].strip()),
        'memory_used_mib': int(row[4].strip()),
        'utilization_gpu_percent': int(row[5].strip()),
        'driver_version': row[6].strip(),
    } for row in rows]
    if len(inventory) != 8 or [x['index'] for x in inventory] != list(range(8)):
        raise RuntimeError('unexpected inventory; refusing GPU selection')
    topology = run('nvidia-smi', 'topo', '-m')
    nvlink = run('nvidia-smi', 'nvlink', '-s')
    gpu_rows = [line.split() for line in topology.splitlines() if line.startswith('GPU') and line.split()[0] in {f'GPU{i}' for i in range(8)}]
    pair_topology = gpu_rows[6][8]
    if pair_topology != 'PIX':
        raise RuntimeError(f'GPU 6/7 topology changed: {pair_topology}')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('set CUDA_VISIBLE_DEVICES=6,7 before audit')
    peer = {'6_to_7': torch.cuda.can_device_access_peer(0, 1),
            '7_to_6': torch.cuda.can_device_access_peer(1, 0)}
    free = run('df', '-B1', '/data')
    model_files = ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json')
    model_hashes = {name: sha256(MODEL / name) for name in model_files}
    selected = [inventory[i] for i in PAIR]
    idle = all(x['memory_used_mib'] < 1024 and x['utilization_gpu_percent'] < 10 for x in selected)
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'hostname': platform.node(),
        'purpose': 'training_host_only; original 4x3090 host retained for canonical inference',
        'selected_physical_gpu_pair': list(PAIR),
        'gpu_inventory': inventory,
        'selected_pair_topology': pair_topology,
        'nvidia_smi_topo_m_raw': topology,
        'nvidia_smi_nvlink_s_raw': nvlink,
        'selected_pair_nvlink_available': pair_topology.startswith('NV'),
        'selected_pair_direct_peer_access': peer,
        'GPU_RESOURCE_GATE': 'PASS' if idle else 'BLOCKED',
        'model_path': str(MODEL),
        'model_metadata_sha256': model_hashes,
        'disk_df_B1_raw': free,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'torch_distributed_available': torch.distributed.is_available(),
        'torch_distributed_nccl_available': torch.distributed.is_nccl_available(),
        'nccl_version': list(torch.cuda.nccl.version()),
        'transformers_version': version('transformers'),
        'triton_version': version('triton'),
        'fla_version': version('flash-linear-attention'),
        'environment_change_performed': False,
    }
    target = ROOT / 'analysis' / 'training_server_hardware_topology.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'GPU_RESOURCE_GATE': result['GPU_RESOURCE_GATE'],
                      'selected_physical_gpu_pair': list(PAIR),
                      'selected_pair_topology': pair_topology,
                      'direct_peer_access': peer}, sort_keys=True))


if __name__ == '__main__':
    main()
