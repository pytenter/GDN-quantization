#!/usr/bin/env python3
"""Finalize the preregistered early-stop verdict from immutable compact evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'analysis' / name).read_text())


def put(rel, value):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(value, encoding='utf-8')


def put_json(rel, value):
    put(rel, json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    shard = load('fsdp_load_gate_review.json')
    forward = load('forward_semantics.json')
    qdq = load('recurrent_qdq_semantics.json')
    repeat = load('same_topology_repeatability.json')
    fsdp_load = load('fsdp_load_memory.json')
    single_h1 = load('single_H1_sem_01_rank0.json')
    h1 = [load(f'fsdp_H1_update_01_rank{r}.json') for r in (0, 1)]
    h4 = [load(f'fsdp_H4_update_01_rank{r}.error.json') for r in (0, 1)]
    if not (shard['corrected_PARAMETER_SHARDING_GATE'] == 'PASS'
            and forward['CANONICAL_FORWARD_SEMANTICS_GATE'] == 'PASS'
            and qdq['gate'] == 'PASS' and repeat['status'] == 'PASS'
            and all(x['model_sample_unchanged'] and x['theta_changed_count'] == 24 for x in h1)
            and all(x['type'] == 'RuntimeError' and 'setStorage' in x['message'] for x in h4)):
        raise RuntimeError('source evidence does not support declared early stop')

    load_static = [x['memory_after_shard_and_rotation']['allocated_bytes']
                   for x in fsdp_load['ranks']]
    h1_peak = [x['memory_after_optimizer']['peak_allocated_bytes'] for x in h1]
    h1_reserved = [x['memory_after_optimizer']['peak_reserved_bytes'] for x in h1]
    rows = [
        ('static load', 17_907_635_200, load_static[0], load_static[1], 'measured; single is full-model load'),
        ('H1', single_h1['memory_after_forward']['peak_allocated_bytes'], h1_peak[0], h1_peak[1],
         'single forward-only; FSDP includes backward and Adam; not matched update workloads'),
        ('H4', None, None, None, 'FSDP backward setStorage failure on both ranks; peak not recorded'),
        ('H8', None, None, None, 'NOT_RUN after H4 framework error'),
        ('H16', None, None, None, 'NOT_RUN after H4 framework error'),
        ('H32', 'OOM in historical single-GPU audit', None, None, 'FSDP NOT_RUN after H4 framework error'),
        ('H64', None, None, None, 'NOT_AUTHORIZED'),
        ('H128', None, None, None, 'NOT_AUTHORIZED'),
    ]
    csv_path = ROOT / 'analysis/memory_scaling.csv'
    if csv_path.exists():
        raise FileExistsError(csv_path)
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'single_gpu_peak_or_static_bytes', 'fsdp_rank0_bytes',
                         'fsdp_rank1_bytes', 'interpretation'])
        writer.writerows(rows)

    put_json('analysis/memory_lifetime.json',
             {'gate': 'BLOCKED', 'reason': 'H4 backward failed before any H32 sustained updates',
              'H32_updates_completed': 0, 'leak_conclusion': 'NOT_DETERMINED'})
    put_json('analysis/saved_tensor_memory_by_family.json',
             {'status': 'NOT_RUN', 'reason': 'early stop on H4 FSDP2 backward compatibility failure',
              'largest_families': None, 'hooks_used': False})
    put_json('analysis/checkpoint_portability.json',
             {'gate': 'NOT_RUN', 'reason': 'zero sustained diagnostic updates; eight-update trigger unmet',
              'rotation_checkpoint': None})
    put_json('analysis/fsdp_h4_backward_failure.json',
             {'status': 'FRAMEWORK_COMPATIBILITY_BLOCK', 'condition': 'frozen C5, H4, two FSDP2 ranks',
              'phase': 'c5_loss.backward', 'ranks': [
                  {'rank': x['rank'], 'type': x['type'], 'message': x['message'],
                   'source': f'analysis/fsdp_H4_update_01_rank{x["rank"]}.error.json'} for x in h4],
              'OOM': False, 'root_cause_proven': False,
              'hypothesis_not_proof': 'Repeated FSDP-wrapped block invocation before backward may interact with resharded storage.',
              'retry_or_config_change': 'NONE', 'deepspeed_installed': False,
              'deepspeed_fallback': 'NOT_AVAILABLE_IN_CURRENT_ENV'})
    gates = {
        'FSDP_RUNTIME_GATE': 'PASS', 'PARAMETER_SHARDING_GATE': 'PASS',
        'TRAINABLE_PARAMETER_GATE': 'PASS', 'CANONICAL_FORWARD_SEMANTICS_GATE': 'PASS',
        'RECURRENT_QDQ_SEMANTICS_GATE': 'PASS', 'GRADIENT_GRAPH_GATE': 'PARTIAL',
        'FSDP2_SAME_TOPOLOGY_REPEATABILITY': 'PASS',
        'FROZEN_BACKBONE_IMMUTABILITY_GATE': 'PASS',
        'MEMORY_LIFETIME_GATE': 'BLOCKED', 'H32_FEASIBILITY_GATE': 'BLOCKED',
        'H64_FEASIBILITY_GATE': 'BLOCKED', 'H128_FEASIBILITY_GATE': 'BLOCKED',
        'CHECKPOINT_PORTABILITY_GATE': 'BLOCKED',
        'FSDP_H4_BACKWARD_COMPATIBILITY_GATE': 'FAIL',
    }
    verdict = {'experiment': 'QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1',
               'status': 'BLOCKED', 'gates': gates,
               'FSDP2_RECURRENT_TRAINING_FEASIBLE': 'NOT_ESTABLISHED',
               'FORMAL_C5_C6_TRAINING_AUTHORIZED': False, 'FORMAL_AIME_AUTHORIZED': False,
               'selected_physical_gpus': [0, 1], 'torch_version': '2.5.1+cu121',
               'fsdp_api': 'torch.distributed._composable.fsdp.fully_shard',
               'deepspeed_fallback_used': False, 'deepspeed_installed': False,
               'single_full_model_load_allocated_bytes': 17_907_635_200,
               'fsdp_static_load_allocated_bytes_by_rank': load_static,
               'single_H1_forward_only_peak_allocated_bytes': rows[1][1],
               'fsdp_H1_update_peak_allocated_bytes_by_rank': h1_peak,
               'fsdp_H1_update_peak_reserved_bytes_by_rank': h1_reserved,
               'first_failed_horizon': 4, 'failed_phase': 'backward',
               'H32_memory_feasible': 'NOT_DETERMINED',
               'max_completed_diagnostic_horizon': 1,
               'memory_leak': 'NOT_DETERMINED',
               'saved_tensor_largest_families': 'NOT_RUN',
               'checkpoint_portability': 'NOT_RUN',
               'formal_C5_C6_started': False, 'AIME_started': False,
               'PP_or_TP_used': False, 'other_user_process_interrupted': False}
    put_json('analysis/final_verdict.json', verdict)

    report = f'''# FSDP2 recurrent-aware training feasibility — early-stop report

Status: **BLOCKED**. The two-card FSDP2 implementation shards frozen Qwen parameters and preserves canonical teacher forward and H1 C128 QDQ/writeback. Five independent H1 two-rank processes yielded identical loss, gradients and one-step rotation updates. The first H4 update then failed on **both ranks during backward**, before optimizer step, with `setStorage ... storage of size 0`. This is a reproducible-in-rank framework/storage error in the single attempted H4 run, **not an OOM**. It was not retried or masked.

## Scope and provenance

- Base: `9f2df90d49d3b692fbc0da5286ce735de43fa1db` (verified latest causal-isolation branch at preregistration).
- Branch: `exp/qwen-gdn-fsdp2-recurrent-training-feasibility-v1`.
- Server: original four-RTX3090 host; selected GPU0/GPU1. Other user processes were not interrupted.
- Runtime: PyTorch 2.5.1+cu121, NCCL, `torch.distributed._composable.fsdp.fully_shard`; block-wise FSDP2, plus embedding, final norm, distinct lm_head and root. Rotation bank was replicated, small, and solely trainable.
- Canonical GDN `torch.cumsum`, C128 QDQ, Key-side rotation and recurrent writeback were unchanged. No fixed-prefix, `device_map` dual, PP, TP, activation checkpointing, formal C5/C6 or AIME was used.
- DeepSpeed was checked only after the H4 block, is absent in the frozen environment, and was not installed or run.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| FSDP runtime / parameter sharding / trainable inventory | PASS | 427 model parameter entries have `S(0)` and exactly half local numel on each rank; 8,953,803,264 logical model params, 4,476,901,632 local per rank; 195,072 trainable rotation params. Initial load checker falsely looked for `Shard` rather than actual `S(0)`; original artifact and separate correction retained. |
| Canonical teacher forward | PASS | Single vs 3 fresh two-rank runs: selected GDN states, logits and loss bitwise equal. |
| H1 QDQ/recurrent writeback | PASS | All 24 layers: pre-state, scale, qcodes, post-state hashes and cache writeback exact; loss exact. |
| H1 gradient/one Adam update | PASS at H1 | All 24 rotation gradients exist, finite and nonzero on both ranks; all 24 theta tensors change; sampled frozen model hashes unchanged. |
| Same-topology H1 repeatability | PASS | Five fresh two-rank processes: exact forward/loss, gradient norms (maximum range 0), and post-Adam theta hashes. |
| H4 backward compatibility | FAIL | Both ranks: `RuntimeError: setStorage: sizes [8192,1,1,4] ... storage of size 0` at `c5_loss.backward()`; no OOM. Root cause is unproven; repeated invocation/reshard interaction is only a hypothesis. |
| H32, lifetime, H64/H128, portability | BLOCKED | H4 error triggered preregistered stop. No extrapolation to H32, no sustained updates, no checkpoint export. |

## GPU memory (bytes)

| Condition | Single GPU | FSDP2 rank0 | FSDP2 rank1 |
|---|---:|---:|---:|
| Static full-model load / sharded load | 17,907,635,200 | {load_static[0]:,} | {load_static[1]:,} |
| H1 peak allocated | {rows[1][1]:,} (forward only) | {h1_peak[0]:,} (update) | {h1_peak[1]:,} (update) |
| H4 | — | backward storage error; peak not captured | backward storage error; peak not captured |
| H8 / H16 | — | NOT_RUN | NOT_RUN |
| H32 | historical single-GPU OOM | NOT_RUN | NOT_RUN |
| H64 / H128 | — | NOT_AUTHORIZED | NOT_AUTHORIZED |

FSDP2 cut measured static allocation by 8,949,886,976 bytes per GPU (~50%). H1 FSDP update peak reserved was {h1_reserved[0]:,} bytes per rank. The single H1 number is a forward-only diagnostic and **must not** be interpreted as an apples-to-apples update comparison. The old single-GPU memory audit found `NO_LEAK_H32_TOO_LARGE`; its H32 OOM is historical. Neither an H32 FSDP peak nor an H-dependent activation slope can be estimated from this stopped run. `STATIC_VS_ACTIVATION_CONTRIBUTION=PARTIALLY_IDENTIFIED`.

## Deferred audits and conclusion

The saved-tensor family audit was not run after the H4 backward failure; no largest-family claim is made. The 5 H1 probes were independent one-step runs, **not** 5 sustained updates; the eight-update checkpoint-portability trigger was not met. Memory leak status under FSDP2 is undetermined. DeepSpeed ZeRO-3 fallback was unavailable in this environment and no packages were changed.

`FSDP2_RECURRENT_TRAINING_FEASIBLE=NOT_ESTABLISHED`; `H32_FEASIBILITY_GATE=BLOCKED`. This negative feasibility result does not overturn prior frozen experiments. A separate protocol is needed to investigate the FSDP2 H4 backward/storage incompatibility before any H32 memory claim. Formal C5/C6 and AIME remain **NO**.

See `analysis/final_verdict.json`, `analysis/fsdp_h4_backward_failure.json`, per-rank H4 `.error.json`, `analysis/recurrent_qdq_semantics.json`, `analysis/same_topology_repeatability.json`, and `hashes/artifact_sha256.txt`.
'''
    put('reports/FINAL_REPORT.md', report)
    put('reports/MEMORY_SCALING.md', 'Memory scaling terminated at H4 backward. See `analysis/memory_scaling.csv` and `reports/FINAL_REPORT.md`. No M(H) regression is valid.\n')
    put('reports/SAVED_TENSOR_MEMORY_AUDIT.md', 'NOT_RUN: H4 FSDP2 backward compatibility failure required an early stop. No saved-tensor families were measured.\n')

    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.py')
                   and 'temporary' not in p.parts and '__pycache__' not in p.parts)
    lines = [hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(ROOT)) for p in paths]
    put('hashes/artifact_sha256.txt', '\n'.join(lines) + '\n')
    print(json.dumps({'status': 'BLOCKED', 'hash_entries': len(lines), 'H4_error': h4[0]['message']}))


if __name__ == '__main__':
    main()
