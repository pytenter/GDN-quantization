#!/usr/bin/env python3
"""Canonical original-host deployment check; loads frozen R without Cayley recomputation."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

import frozen_probe
import runtime


def main() -> None:
    if not runtime.ORIGINAL_HOST or torch.cuda.device_count() != 1:
        raise RuntimeError('original canonical server with CUDA_VISIBLE_DEVICES=0 required')
    file_output = runtime.ROOT / 'analysis/cross_host_file_integrity.json'
    probe_output = runtime.ROOT / 'analysis/original_server_deployment_gate.json'
    if file_output.exists() or probe_output.exists():
        raise RuntimeError('original-server deployment gate already attempted')
    provenance = runtime.check_parent_sources()
    manifest = json.loads((runtime.ROOT / 'analysis/frozen_rotation_matrix_manifest.json').read_text())
    format_config = json.loads((runtime.ROOT / 'configs/deployment_checkpoint_format.json').read_text())
    artifact = Path(format_config['original_server_absolute_path'])
    actual_sha = runtime.file_sha256(artifact)
    expected_sha = manifest['artifact_sha256']
    integrity = {
        'vgpu_artifact_path': manifest['vgpu_artifact_path'],
        'original_artifact_path': str(artifact),
        'expected_sha256': expected_sha, 'actual_sha256': actual_sha,
        'expected_size_bytes': manifest['artifact_size_bytes'],
        'actual_size_bytes': artifact.stat().st_size,
        'FROZEN_R_FILE_PORTABILITY_GATE': 'PASS' if actual_sha == expected_sha and
        artifact.stat().st_size == manifest['artifact_size_bytes'] else 'FAIL',
    }
    runtime.save_once(file_output, integrity)
    if integrity['FROZEN_R_FILE_PORTABILITY_GATE'] != 'PASS':
        raise RuntimeError('cross-host frozen-R artifact bytes differ')

    core = runtime.load_core()
    payload, bank = runtime.load_frozen_artifact(artifact, core)
    per_layer = {}
    for layer in core.GDN_LAYERS:
        value = bank.layer(layer).matrix()
        digest = runtime.tensor_sha256(value)
        expected = manifest['per_layer'][str(layer)]['sha256']
        identity = torch.eye(128, dtype=torch.float32)
        error = float((value.T @ value - identity).abs().max())
        per_layer[str(layer)] = {'loaded_sha256': digest, 'expected_sha256': expected,
                                 'exact': digest == expected,
                                 'orthogonality_max_abs_rt_r_minus_i': error,
                                 'finite': bool(torch.isfinite(value).all()),
                                 'orthogonality_pass': error <= core.ORTHOGONALITY_THRESHOLD}
    matrix_ok = all(row['exact'] and row['finite'] and row['orthogonality_pass']
                    for row in per_layer.values())
    if not matrix_ok:
        runtime.save_once(probe_output, {
            'ORIGINAL_SERVER_FROZEN_R_RUNTIME': 'FAIL_MATRIX_LOAD',
            'per_layer': per_layer, 'functional_probe': 'NOT_RUN',
            'V1_RECOMPUTED_MATRIX_HASH_PORTABILITY': 'FAIL',
        })
        raise RuntimeError('original direct-load frozen matrix gate failed')
    model, tokenizer, _ = runtime.load_model_and_tokenizer(core)
    bank.to('cuda:0')
    probe = frozen_probe.run_fixed_probe(core, model, tokenizer, bank)
    qdq_ok = probe['C128_QDQ_calls'] == 24 * 32 and -127 <= probe['C128_QDQ_code_min'] <= 0 and 0 <= probe['C128_QDQ_code_max'] <= 127
    writeback_ok = all(probe['writeback_exact_by_layer'].values())
    status = 'PASS' if qdq_ok and writeback_ok else 'FAIL_FUNCTIONAL'
    result = {
        'ORIGINAL_SERVER_FROZEN_R_RUNTIME': status,
        'VGPU_TO_CANONICAL_DEPLOYMENT_PORTABILITY_V2': status,
        'V1_RECOMPUTED_MATRIX_HASH_PORTABILITY': 'FAIL',
        'source_provenance': provenance,
        'artifact_sha256': actual_sha,
        'matrix_meaning': payload['matrix_meaning'],
        'orientation': payload['orientation'],
        'per_layer': per_layer,
        'all_24_loaded_matrix_hashes_exact': matrix_ok,
        'C128_QDQ_gate': qdq_ok,
        'recurrent_writeback_gate': writeback_ok,
        'fixed_non_AIME_probe': probe,
        'theta_recomputed_on_original_for_deployment': False,
        'formal_training_started': False, 'AIME_started': False,
    }
    runtime.save_once(probe_output, result)
    print(json.dumps({'ORIGINAL_SERVER_FROZEN_R_RUNTIME': status,
                      'FROZEN_R_FILE_PORTABILITY_GATE': integrity['FROZEN_R_FILE_PORTABILITY_GATE'],
                      'all_24_loaded_matrix_hashes_exact': matrix_ok,
                      'C128_QDQ_gate': qdq_ok, 'recurrent_writeback_gate': writeback_ok}), flush=True)
    if status != 'PASS':
        raise RuntimeError('original-server frozen-R functional gate failed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = runtime.ROOT / 'analysis/original_server_deployment.error.json'
        if not path.exists():
            runtime.save_once(path, {'error_type': type(exc).__name__, 'error': str(exc),
                                     'traceback': traceback.format_exc()})
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}), flush=True)
        raise
