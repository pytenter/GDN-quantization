#!/usr/bin/env python3
"""Frozen non-AIME full-model teacher reference; no update, no PP runtime."""
from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
OLD = ROOT.parent / 'QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1/analysis/single01_rank0.json'
GDN = json.loads((ROOT / 'configs/canonical_loss_definition.json').read_text())['gdn_layers']


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def scalar_summary(t):
    f = t.detach().float()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': sha(t),
            'min': float(f.min()), 'max': float(f.max()),
            'mean': float(f.mean()), 'l2': float(torch.linalg.vector_norm(f))}


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required for canonical reference')
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from run_stage_load import memory
        import importlib.util, sys
        v1_path = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
        spec = importlib.util.spec_from_file_location('pp2_read_only_v1', v1_path)
        v1 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = v1
        spec.loader.exec_module(v1)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                                  local_files_only=True)
        row = v1.corpus_rows('TRAIN')[0]
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        if len(ids) != 64:
            raise RuntimeError('Frozen reference token prefix shorter than 64')
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to('cuda:0')
        boundary = {}
        handle = model.model.layers[15].register_forward_hook(
            lambda _module, _args, output: boundary.setdefault('hidden', output.detach().clone()))
        input_ids = torch.tensor([ids], device='cuda:0', dtype=torch.long)
        before = memory()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
            logits = output.logits[:, -1, :].detach().clone()
            states = {str(layer): scalar_summary(output.past_key_values.layers[layer].recurrent_states[0])
                      for layer in GDN}
            loss = F.cross_entropy(logits.float(), torch.tensor([ids[-1]], device='cuda:0'))
        handle.remove()
        old = json.loads(OLD.read_text())
        result = {'sample_id': str(row.get('id', 'TRAIN:first-row')),
                  'sample_token_hash': sha(input_ids), 'boundary_after_block15': scalar_summary(boundary['hidden']),
                  'logits': scalar_summary(logits), 'loss': float(loss),
                  'gdn_recurrent_states': states, 'memory_before': before,
                  'memory_after': memory(),
                  'historical_reference_exact': {
                      'input_ids': sha(input_ids) == old['input_ids_sha256'],
                      'logits': sha(logits) == old['logits']['sha256'],
                      'loss': float(loss) == old['loss'],
                      'selected_states': {str(l): states[str(l)]['sha256'] == old['selected_states'][str(l)]['sha256']
                                          for l in (0,16,30)}}}
        save_once(ROOT / 'analysis/single_teacher_reference.json', result)
        print(json.dumps({'historical_reference_exact': result['historical_reference_exact'],
                          'boundary_sha256': result['boundary_after_block15']['sha256'],
                          'logits_sha256': result['logits']['sha256']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis/single_teacher_reference.error.json',
                  {'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
