#!/usr/bin/env python3
"""Single-GPU teacher endpoint comparison; no training or formal inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SOURCE_ROOT = Path('/data/zypan')
SOURCE = SOURCE_ROOT / 'worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('QWEN_TRANSFORMERS_SRC', str(SOURCE_ROOT / 'transformers-qwen35'))
ORIGINAL_MODEL = SOURCE_ROOT / 'modelscope_models/Qwen3.5-9B'
IS_ORIGINAL_HOST = ORIGINAL_MODEL.is_dir()
os.environ.setdefault('QWEN_MODEL_PATH', str(ORIGINAL_MODEL if IS_ORIGINAL_HOST
                                               else Path('/root/autodl-tmp/models/Qwen3.5-9B')))
os.environ.setdefault('HF_HOME', '/data/zypan/.cache/huggingface' if IS_ORIGINAL_HOST
                      else '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(SOURCE_ROOT / 'transformers-qwen35/src'))


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_summary(tensor: torch.Tensor) -> dict:
    f = tensor.detach().float()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': sha_tensor(tensor),
            'min': float(f.min()), 'max': float(f.max()), 'mean': float(f.mean()),
            'l2': float(torch.linalg.vector_norm(f)), 'all_finite': bool(torch.isfinite(f).all())}


def save_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load_core():
    spec = importlib.util.spec_from_file_location('qwen_48gb_forward_v1', SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError('canonical C5 source unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def endpoint() -> tuple[dict, dict]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA device required for teacher endpoint')
    core = load_core()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['QWEN_MODEL_PATH'], trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        os.environ['QWEN_MODEL_PATH'], torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to('cuda:0')
    device = model.get_input_embeddings().weight.device
    ids, _ = core.tokenize(tokenizer, core.corpus_rows('TRAIN')[0])
    ids = ids[:64]
    if len(ids) != 64:
        raise RuntimeError('TRAIN:first-row has fewer than 64 tokens')
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    boundary = {}
    def capture_boundary(_module, _args, output):
        boundary['hidden'] = output.detach().cpu().contiguous()
        return None

    handle = model.model.layers[15].register_forward_hook(capture_boundary)
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
        logits = output.logits[:, -1, :].detach().cpu().contiguous()
        loss = F.cross_entropy(output.logits[:, -1, :].float(), input_ids[:, -1])
        states = {f'state_{layer}': output.past_key_values.layers[layer].recurrent_states[0].detach().cpu().contiguous()
                  for layer in core.GDN_LAYERS}
    handle.remove()
    tensors = {'input_ids': input_ids.detach().cpu().contiguous(),
               'boundary_hidden': boundary['hidden'], 'logits': logits, **states}
    summary = {'sample': 'TRAIN:first-row', 'token_count': 64,
               'input_ids_sha256': sha_tensor(tensors['input_ids']),
               'target_token_sha256': sha_tensor(tensors['input_ids'][:, -1:]),
               'loss': float(loss.detach().cpu()),
               'tensors': {key: scalar_summary(value) for key, value in tensors.items()},
               'canonical_source_sha256': sha_file(SOURCE),
               'model_config_sha256': sha_file(Path(os.environ['QWEN_MODEL_PATH']) / 'config.json'),
               'model_index_sha256': sha_file(Path(os.environ['QWEN_MODEL_PATH']) / 'model.safetensors.index.json'),
               'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
    return tensors, summary


def numerical_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {'shape_dtype_match': False, 'reference_shape': list(reference.shape),
                'actual_shape': list(actual.shape), 'reference_dtype': str(reference.dtype),
                'actual_dtype': str(actual.dtype)}
    a = reference.float().flatten()
    b = actual.float().flatten()
    difference = b - a
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    denominator = max(float(norm_a), 1e-12)
    cosine = float(torch.dot(a, b) / (norm_a * norm_b).clamp_min(1e-12))
    return {'shape_dtype_match': True, 'sha256_exact': sha_tensor(reference) == sha_tensor(actual),
            'max_abs': float(difference.abs().max()),
            'relative_l2': float(torch.linalg.vector_norm(difference)) / denominator,
            'cosine': cosine, 'cosine_error': 1.0 - cosine,
            'reference_all_finite': bool(torch.isfinite(a).all()),
            'actual_all_finite': bool(torch.isfinite(b).all())}


def main(args) -> None:
    if args.mode == 'reference':
        prefix = Path(args.output_prefix)
        raw_path = prefix.with_suffix('.safetensors')
        json_path = prefix.with_suffix('.json')
        if raw_path.exists() or json_path.exists():
            raise RuntimeError('reference output already exists')
        tensors, summary = endpoint()
        save_file(tensors, str(raw_path))
        summary['temporary_tensor_file_sha256'] = sha_file(raw_path)
        summary['temporary_tensor_file_size_bytes'] = raw_path.stat().st_size
        save_once(json_path, summary)
        print(json.dumps({'mode': 'reference', 'input_ids_sha256': summary['input_ids_sha256'],
                          'loss': summary['loss'], 'temporary_tensor_file_sha256': summary['temporary_tensor_file_sha256']}), flush=True)
        return

    source_gate = json.loads((ROOT / 'analysis/source_parity.json').read_text())
    if source_gate['SOURCE_HASH_GATE'] != 'PASS':
        raise RuntimeError('SOURCE_HASH_GATE not PASS')
    reference_path = Path(args.reference_tensors)
    reference_info = json.loads(Path(args.reference_json).read_text())
    if sha_file(reference_path) != reference_info['temporary_tensor_file_sha256']:
        raise RuntimeError('transferred reference tensor hash mismatch')
    reference = load_file(str(reference_path), device='cpu')
    actual, actual_info = endpoint()
    metrics = {key: numerical_metrics(reference[key], actual[key]) for key in reference}
    historical = json.loads((ROOT / 'configs/historical_original_teacher_reference.json').read_text())
    history_exact = {
        'input_ids': actual_info['input_ids_sha256'] == historical['sample_token_hash'],
        'logits': reference_info['tensors']['logits']['sha256'] == historical['logits']['sha256'],
        'loss': reference_info['loss'] == historical['loss'],
        'selected_states': {str(layer): reference_info['tensors'][f'state_{layer}']['sha256'] ==
                            historical['gdn_recurrent_states'][str(layer)]['sha256'] for layer in (0, 16, 30)},
    }
    all_finite = all(row.get('actual_all_finite', False) and row.get('reference_all_finite', False)
                     for row in metrics.values())
    source_exact = (actual_info['canonical_source_sha256'] == reference_info['canonical_source_sha256']
                    and actual_info['model_config_sha256'] == reference_info['model_config_sha256']
                    and actual_info['model_index_sha256'] == reference_info['model_index_sha256'])
    result = {
        'sample': 'TRAIN:first-row', 'token_count': 64,
        'source_model_input_exact': source_exact and metrics['input_ids']['sha256_exact'],
        'same_operators_dtype_semantics': source_exact and all_finite,
        'cross_gpu_tolerance': 'NOT_PRE_REGISTERED; raw numerical differences only',
        'scientific_failure_from_nonexact_hash': 'NOT_INFERRED',
        'all_finite': all_finite,
        'fresh_original_reference_matches_historical': history_exact,
        'reference': reference_info, 'vgpu': actual_info,
        'numerical_metrics': metrics,
    }
    save_once(ROOT / 'analysis/forward_parity.json', result)
    print(json.dumps({'source_model_input_exact': result['source_model_input_exact'],
                      'all_finite': all_finite, 'historical_reference_exact': history_exact,
                      'logits': metrics['logits'], 'loss_difference': actual_info['loss'] - reference_info['loss']}), flush=True)
    if not result['source_model_input_exact'] or not all_finite:
        raise RuntimeError('structural forward parity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('reference', 'compare'), required=True)
    parser.add_argument('--output-prefix')
    parser.add_argument('--reference-tensors')
    parser.add_argument('--reference-json')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        detail = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        if args.mode == 'compare':
            save_once(ROOT / 'analysis/forward_parity.error.json', detail)
        else:
            print(json.dumps(detail), flush=True)
        raise
