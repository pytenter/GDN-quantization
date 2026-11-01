#!/usr/bin/env python3
"""H1 prerequisite: frozen teacher forward against historic canonical summary."""
from __future__ import annotations

import hashlib
import json
import os
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F

from run_zero3_load import ROOT, load_zero3, memory, save_once, sha, MODEL_SOURCE, V1_PATH


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def summary(t):
    f = t.detach().float()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': tensor_sha(t),
            'min': float(f.min()), 'max': float(f.max()), 'mean': float(f.mean()),
            'l2': float(torch.linalg.vector_norm(f))}


def main():
    rank = int(os.environ['LOCAL_RANK'])
    if os.environ.get('WORLD_SIZE') != '2' or torch.cuda.device_count() != 2:
        raise RuntimeError('two visible GPUs and ranks required')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    tag = f'zero3_h1_teacher_rank{rank}'
    try:
        engine, _, bank, tokenizer, v1, hf_ds_config = load_zero3(rank)
        row = v1.corpus_rows('TRAIN')[0]
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        if len(ids) != 64:
            raise RuntimeError('Frozen diagnostic sample changed')
        device = torch.device('cuda', rank)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        target = torch.tensor([ids[-1]], device=device, dtype=torch.long)
        before = memory()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = engine(input_ids=input_ids, use_cache=True)
            logits = output.logits[:, -1, :].detach().clone()
            selected = {layer: output.past_key_values.layers[layer].recurrent_states[0].detach().clone()
                        for layer in (0, 16, 30)}
            loss = F.cross_entropy(logits.float(), target)
        after = memory()
        historical = json.loads((ROOT.parent / 'QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1'
                                 / 'analysis/single01_rank0.json').read_text())
        matches = {
            'sample': tensor_sha(input_ids) == historical['input_ids_sha256'],
            'model_source': sha(MODEL_SOURCE) == historical['model_source_sha256'],
            'v1_source': sha(V1_PATH) == historical['v1_source_sha256'],
            'logits_sha256': tensor_sha(logits) == historical['logits']['sha256'],
            'loss_abs_error': abs(float(loss) - historical['loss']),
            'selected_state_sha256': {str(layer): tensor_sha(value) == historical['selected_states'][str(layer)]['sha256']
                                      for layer, value in selected.items()},
        }
        exact = (all(matches[key] for key in ('sample', 'model_source', 'v1_source', 'logits_sha256'))
                 and matches['loss_abs_error'] <= 1e-5
                 and all(matches['selected_state_sha256'].values()))
        result = {'rank': rank, 'sample_id': str(row.get('id', 'TRAIN:first-row')),
                  'input_ids_sha256': tensor_sha(input_ids), 'logits': summary(logits),
                  'loss': float(loss), 'selected_states': {str(k): summary(v) for k, v in selected.items()},
                  'historical_exact_comparison': matches,
                  'H1_CANONICAL_FORWARD_GATE': 'PASS' if exact else 'NEEDS_NUMERIC_COMPARISON',
                  'memory_before': before, 'memory_after': after}
        save_once(ROOT / 'analysis' / f'{tag}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            other = json.loads((ROOT / 'analysis/zero3_h1_teacher_rank1.json').read_text())
            overall = {'ranks': [result, other],
                       'H1_CANONICAL_FORWARD_GATE': 'PASS' if exact and other['H1_CANONICAL_FORWARD_GATE'] == 'PASS'
                       else 'NEEDS_NUMERIC_COMPARISON'}
            save_once(ROOT / 'analysis/h1_forward_semantics.json', overall)
            print(json.dumps({'H1_CANONICAL_FORWARD_GATE': overall['H1_CANONICAL_FORWARD_GATE'],
                              'logits_sha256': result['logits']['sha256'], 'loss': result['loss']}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'{tag}.error.json', {'rank': rank, 'type': type(exc).__name__,
                  'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
