#!/usr/bin/env python3
"""Two-rank static PP2 teacher parity; one boundary send, no training."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from stage import StaticQwenStage
from run_single_teacher_reference import sha, scalar_summary, save_once

MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
GDN = json.loads((ROOT / 'configs/canonical_loss_definition.json').read_text())['gdn_layers']


def main():
    if os.environ.get('WORLD_SIZE') != '2' or torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two ranks and visible GPUs required')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        stage = StaticQwenStage.load(rank, torch.device('cuda', rank))
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                                  local_files_only=True)
        import importlib.util
        v1_path = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
        spec = importlib.util.spec_from_file_location('pp2_teacher_v1', v1_path)
        v1 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = v1
        spec.loader.exec_module(v1)
        row = v1.corpus_rows('TRAIN')[0]
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        input_ids = torch.tensor([ids], device=f'cuda:{rank}', dtype=torch.long)
        rank_hashes = [None, None]
        dist.all_gather_object(rank_hashes, sha(input_ids))
        if rank_hashes[0] != rank_hashes[1]:
            raise RuntimeError('Replicated input IDs mismatch')
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            with torch.no_grad():
                hidden, _, cache = stage.forward_stage(input_ids=input_ids,
                                                         start_position=0)
            boundary = scalar_summary(hidden)
            dist.send(hidden.detach().contiguous(), dst=1)
            logits = None
            loss = None
        else:
            wire = torch.empty((1, len(ids), stage.config.hidden_size),
                               device='cuda:1', dtype=torch.bfloat16)
            dist.recv(wire, src=0)
            boundary = scalar_summary(wire)
            with torch.no_grad():
                _, full_logits, cache = stage.forward_stage(boundary_hidden=wire,
                                                              start_position=0)
                logits = scalar_summary(full_logits[:, -1, :])
                loss = float(F.cross_entropy(full_logits[:, -1, :].float(),
                                             torch.tensor([ids[-1]], device='cuda:1')))
        states = {str(layer): scalar_summary(cache.layers[layer].recurrent_states[0])
                  for layer in GDN if layer in stage.block_ids}
        result = {'rank': rank, 'physical_visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES'),
                  'input_ids_sha256': sha(input_ids), 'boundary_after_block15': boundary,
                  'logits': logits, 'loss': loss, 'local_gdn_recurrent_states': states,
                  'memory': {'allocated': torch.cuda.memory_allocated(),
                             'reserved': torch.cuda.memory_reserved(),
                             'peak_allocated': torch.cuda.max_memory_allocated(),
                             'peak_reserved': torch.cuda.max_memory_reserved()},
                  'forward_wire_bytes': int(hidden.numel() * hidden.element_size()) if rank == 0 else None}
        save_once(ROOT / 'analysis' / f'pp2_teacher_rank{rank}.json', result)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            other = json.loads((ROOT / 'analysis/pp2_teacher_rank1.json').read_text())
            ref = json.loads((ROOT / 'analysis/single_teacher_reference.json').read_text())
            matches = {'input_ids': result['input_ids_sha256'] == other['input_ids_sha256'] == ref['sample_token_hash'],
                       'boundary_hidden_rank0_vs_rank1': result['boundary_after_block15']['sha256'] == other['boundary_after_block15']['sha256'],
                       'boundary_hidden_vs_single': result['boundary_after_block15']['sha256'] == ref['boundary_after_block15']['sha256'],
                       'logits': other['logits']['sha256'] == ref['logits']['sha256'],
                       'loss': other['loss'] == ref['loss'],
                       'gdn_states': {str(layer): (result if layer < 16 else other)['local_gdn_recurrent_states'][str(layer)]['sha256']
                                      == ref['gdn_recurrent_states'][str(layer)]['sha256'] for layer in GDN}}
            exact = all(v for k, v in matches.items() if k != 'gdn_states') and all(matches['gdn_states'].values())
            summary = {'matches': matches, 'PP2_TEACHER_FORWARD_GATE': 'PASS' if exact else 'FAIL',
                       'single_reference_historical_exact': ref['historical_reference_exact']}
            save_once(ROOT / 'analysis/pp2_teacher_forward_parity.json', summary)
            print(json.dumps({'PP2_TEACHER_FORWARD_GATE': summary['PP2_TEACHER_FORWARD_GATE'],
                              'matches': matches}), flush=True)
    except Exception as exc:
        save_once(ROOT / 'analysis' / f'pp2_teacher_rank{rank}.error.json',
                  {'rank': rank, 'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
