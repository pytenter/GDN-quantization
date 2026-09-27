#!/usr/bin/env python3
"""Exact theta-bank versus frozen-bank non-AIME probe on one vGPU."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

import frozen_probe
import runtime


def differing_paths(left, right, prefix='') -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [path for key in sorted(set(left) | set(right))
                for path in differing_paths(left.get(key), right.get(key), f'{prefix}.{key}')]
    return [] if left == right else [prefix]


def main() -> None:
    if runtime.ORIGINAL_HOST or torch.cuda.device_count() != 1:
        raise RuntimeError('same-host gate requires the single-vGPU training host')
    output = runtime.ROOT / 'analysis/frozen_rotation_same_host.json'
    if output.exists():
        raise RuntimeError('same-host gate already attempted')
    provenance = runtime.check_parent_sources()
    manifest = json.loads((runtime.ROOT / 'analysis/frozen_rotation_matrix_manifest.json').read_text())
    artifact = Path(manifest['vgpu_artifact_path'])
    artifact_sha = runtime.file_sha256(artifact)
    if artifact_sha != manifest['artifact_sha256']:
        raise RuntimeError('frozen artifact differs from export manifest')
    core = runtime.load_core()
    payload, frozen_bank = runtime.load_frozen_artifact(artifact, core)
    parent = json.loads((runtime.ROOT / 'configs/parent_manifest.json').read_text())
    theta_bank, _, theta_sha = runtime.load_theta_bank(
        core, Path(parent['training_theta_checkpoint']), torch.device('cuda:0'))
    frozen_bank.to('cuda:0')
    per_layer = {}
    for layer in core.GDN_LAYERS:
        theta_matrix = theta_bank.layer(layer).matrix().detach().cpu().contiguous()
        frozen_matrix = frozen_bank.layer(layer).matrix().detach().cpu().contiguous()
        theta_hash = runtime.tensor_sha256(theta_matrix)
        frozen_hash = runtime.tensor_sha256(frozen_matrix)
        expected_hash = manifest['per_layer'][str(layer)]['sha256']
        per_layer[str(layer)] = {
            'theta_matrix_sha256': theta_hash,
            'frozen_matrix_sha256': frozen_hash,
            'export_manifest_sha256': expected_hash,
            'exact': theta_hash == frozen_hash == expected_hash,
        }
    matrix_exact = all(row['exact'] for row in per_layer.values())
    if not matrix_exact:
        result = {'FROZEN_R_LOADER_SEMANTICS_GATE': 'FAIL_MATRIX_HASH',
                  'per_layer_matrix_hashes': per_layer, 'artifact_sha256': artifact_sha,
                  'theta_checkpoint_sha256': theta_sha, 'source_provenance': provenance,
                  'functional_probe': 'NOT_RUN_AFTER_MATRIX_HASH_FAILURE'}
        runtime.save_once(output, result)
        raise RuntimeError('same-host theta/frozen matrix hashes differ')
    model, tokenizer, _ = runtime.load_model_and_tokenizer(core)
    theta_probe = frozen_probe.run_fixed_probe(core, model, tokenizer, theta_bank)
    frozen_probe_result = frozen_probe.run_fixed_probe(core, model, tokenizer, frozen_bank)
    mismatches = differing_paths(theta_probe, frozen_probe_result)
    status = 'PASS' if not mismatches else 'FAIL_FUNCTIONAL_EXACTNESS'
    result = {
        'FROZEN_R_LOADER_SEMANTICS_GATE': status,
        'artifact_sha256': artifact_sha, 'theta_checkpoint_sha256': theta_sha,
        'source_provenance': provenance,
        'matrix_meaning': payload['matrix_meaning'], 'orientation': payload['orientation'],
        'per_layer_matrix_hashes': per_layer,
        'theta_probe': theta_probe, 'frozen_probe': frozen_probe_result,
        'exact_probe_mismatch_paths': mismatches,
        'all_24_matrix_hashes_exact': matrix_exact,
        'same_host_functional_exact': not mismatches,
        'formal_training_started': False, 'AIME_started': False,
    }
    runtime.save_once(output, result)
    print(json.dumps({'FROZEN_R_LOADER_SEMANTICS_GATE': status,
                      'all_24_matrix_hashes_exact': matrix_exact,
                      'exact_probe_mismatch_count': len(mismatches)}), flush=True)
    if mismatches:
        raise RuntimeError('same-host frozen loader functional exactness failed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = runtime.ROOT / 'analysis/frozen_rotation_same_host.error.json'
        if not path.exists():
            runtime.save_once(path, {'error_type': type(exc).__name__, 'error': str(exc),
                                     'traceback': traceback.format_exc()})
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}), flush=True)
        raise
