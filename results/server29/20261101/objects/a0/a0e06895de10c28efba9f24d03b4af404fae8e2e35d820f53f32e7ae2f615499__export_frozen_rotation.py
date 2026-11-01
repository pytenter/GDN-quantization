#!/usr/bin/env python3
"""Fresh-process training-host export of the canonical GPU Cayley matrices."""
from __future__ import annotations

import json
import socket
import traceback
from pathlib import Path

import torch

import runtime


def main() -> None:
    if runtime.ORIGINAL_HOST or torch.cuda.device_count() != 1:
        raise RuntimeError('export must run on the single-vGPU training host')
    provenance = runtime.check_parent_sources()
    format_config = json.loads((runtime.ROOT / 'configs/deployment_checkpoint_format.json').read_text())
    checkpoint = Path(json.loads((runtime.ROOT / 'configs/parent_manifest.json').read_text())['training_theta_checkpoint'])
    target = Path(format_config['vgpu_absolute_path'])
    manifest_path = runtime.ROOT / 'analysis/frozen_rotation_matrix_manifest.json'
    if target.exists() or manifest_path.exists():
        raise RuntimeError('frozen rotation export already exists; refusing overwrite/retry')
    core = runtime.load_core()
    bank, _theta_payload, theta_sha = runtime.load_theta_bank(core, checkpoint, torch.device('cuda:0'))
    matrices = {}
    rows = {}
    identity = torch.eye(128, dtype=torch.float32)
    for layer in core.GDN_LAYERS:
        # This is the exact canonical Cayley matrix() on the training GPU.
        value = bank.layer(layer).matrix().detach().float().cpu().contiguous()
        if value.shape != (128, 128) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'nonfinite/wrong-shape Cayley matrix at layer {layer}')
        error = float((value.T @ value - identity).abs().max())
        sign, log_abs_det = torch.linalg.slogdet(value)
        rows[str(layer)] = {
            'layer_id': layer, 'shape': [128, 128], 'dtype': 'torch.float32',
            'sha256': runtime.tensor_sha256(value),
            'orthogonality_max_abs_rt_r_minus_i': error,
            'determinant_sign': float(sign), 'determinant_log_abs': float(log_abs_det),
            'all_finite': True,
        }
        if error > core.ORTHOGONALITY_THRESHOLD or float(sign) <= 0:
            raise RuntimeError(f'canonical orthogonality/proper-rotation gate failed layer {layer}')
        matrices[str(layer)] = value
    payload = {
        'format_version': format_config['format_version'],
        'layer_ids': tuple(core.GDN_LAYERS), 'matrices': matrices,
        'matrix_dtype': 'torch.float32', 'matrix_shape': [128, 128],
        'matrix_meaning': format_config['matrix_meaning'],
        'orientation': format_config['runtime_orientation'],
        'theta_checkpoint_sha256': theta_sha,
        'rotation_source_sha256': provenance['rotation_source']['actual_sha256'],
        'canonical_source_sha256': provenance['canonical_c5_c6_source']['actual_sha256'],
        'export_host': {'hostname': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0),
                        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    manifest = {
        'task': runtime.ROOT.name,
        'vgpu_artifact_path': str(target), 'artifact_sha256': runtime.file_sha256(target),
        'artifact_size_bytes': target.stat().st_size,
        'theta_checkpoint_sha256': theta_sha,
        'source_provenance': provenance,
        'layer_count': len(matrices), 'layer_ids': list(core.GDN_LAYERS),
        'matrix_meaning': format_config['matrix_meaning'],
        'runtime_orientation': format_config['runtime_orientation'],
        'per_layer': rows,
    }
    runtime.save_once(manifest_path, manifest)
    print(json.dumps({'artifact_sha256': manifest['artifact_sha256'],
                      'artifact_size_bytes': manifest['artifact_size_bytes'],
                      'layer_count': len(matrices), 'status': 'EXPORTED'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        detail = {'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc()}
        error_path = runtime.ROOT / 'analysis/frozen_rotation_export.error.json'
        if not error_path.exists():
            runtime.save_once(error_path, detail)
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}), flush=True)
        raise
