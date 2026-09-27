#!/usr/bin/env python3
"""Finalize frozen negative H1 evidence without further model execution."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import deepspeed
import torch
import transformers
import triton

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1'
ENV = Path('/data/zypan/worktrees/qwen-gdn-zero3-v1/.venv-qwen-zero3-recurrent-v1')
CANONICAL = Path('/data/ydai/miniconda3/envs/bitdecode/bin/python')
if Path(sys.prefix).resolve() != ENV.resolve():
    raise RuntimeError(f'Wrong finalizer runtime: {sys.prefix}')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        f.write('\n')


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


historical = read_json(PARENT / 'analysis/single01_rank0.json')
zero = read_json(ROOT / 'analysis/h1_forward_semantics.json')
load = read_json(ROOT / 'analysis/zero3_load_attempt02_memory.json')
env_gate = read_json(ROOT / 'analysis/environment_gate.json')
thresholds = read_json(PARENT / 'preregistration.json')['forward_thresholds']
ranks = zero['ranks']
if env_gate['DEEPSPEED_ENVIRONMENT_GATE'] != 'PASS' or load['ZERO3_LOAD_GATE'] != 'PASS':
    raise RuntimeError('Environment/load gate did not pass')
if len(ranks) != 2 or any(r['input_ids_sha256'] != historical['input_ids_sha256'] for r in ranks):
    raise RuntimeError('Diagnostic sample mismatch')

comparisons = []
for r in ranks:
    loss_abs = abs(r['loss'] - historical['loss'])
    loss_rel = loss_abs / abs(historical['loss'])
    logits_max_abs_lower_bound = abs(r['logits']['max'] - historical['logits']['max'])
    logits_rel_l2_lower_bound = abs(r['logits']['l2'] - historical['logits']['l2']) / historical['logits']['l2']
    states = {layer: {
        'max_abs_lower_bound': abs(r['selected_states'][layer]['max'] - historical['selected_states'][layer]['max']),
        'historical_sha256_equal': r['selected_states'][layer]['sha256'] == historical['selected_states'][layer]['sha256']}
        for layer in ('0', '16', '30')}
    comparisons.append({'rank': r['rank'], 'historical_loss': historical['loss'],
                        'zero3_loss': r['loss'], 'loss_abs_error': loss_abs,
                        'loss_relative_error': loss_rel,
                        'logits_max_abs_error_lower_bound': logits_max_abs_lower_bound,
                        'logits_relative_l2_error_lower_bound': logits_rel_l2_lower_bound,
                        'selected_state_bounds': states})

decisive = all(
    c['loss_relative_error'] > thresholds['loss_relative_error']
    and c['logits_max_abs_error_lower_bound'] > thresholds['logits_max_abs']
    and c['logits_relative_l2_error_lower_bound'] > thresholds['logits_relative_l2']
    and c['selected_state_bounds']['16']['max_abs_lower_bound'] > thresholds['state_max_abs']
    for c in comparisons)
if not decisive:
    raise RuntimeError('Summary bounds alone do not prove the pre-registered failure')

snapshot = ROOT / 'configs/canonical_environment_snapshot.txt'
version = subprocess.check_output([str(CANONICAL), '--version'], text=True).strip()
freeze = subprocess.check_output([str(CANONICAL), '-m', 'pip', 'freeze'], text=True)
current = f'python: {version}\nexecutable: {CANONICAL}\n\npip freeze:\n{freeze}'
canonical_unchanged = snapshot.read_text() == current
canonical_ds_absent = subprocess.run([str(CANONICAL), '-m', 'pip', 'show', 'deepspeed'],
                                     capture_output=True, text=True).returncode != 0

try:
    fla_version = metadata.version('flash-linear-attention')
except metadata.PackageNotFoundError:
    fla_version = 'UNKNOWN (not a registered distribution in this environment)'

write_json(ROOT / 'configs/isolated_environment.json', {
    'path': str(ENV), 'python': sys.version.split()[0], 'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda, 'nccl': list(torch.cuda.nccl.version()),
    'deepspeed': deepspeed.__version__, 'transformers': transformers.__version__,
    'triton': triton.__version__, 'fla': fla_version,
    'venv_uses_read_only_canonical_system_site_packages': True,
    'canonical_snapshot_sha256': sha(snapshot),
    'canonical_freeze_unchanged': canonical_unchanged,
    'canonical_deepspeed_absent': canonical_ds_absent})

write_json(ROOT / 'analysis/h1_forward_verdict.json', {
    'H1_CANONICAL_FORWARD_GATE': 'FAIL', 'source_and_sample_match': True,
    'historical_thresholds': thresholds, 'conservative_summary_comparisons': comparisons,
    'proof': 'For any vectors, max absolute difference is at least the difference of maxima; relative L2 error is at least the difference of L2 norms divided by reference L2.',
    'note': 'Raw historical tensors were unavailable; conservative lower bounds and scalar loss already exceed thresholds. No backward/QDQ/horizon retry is authorized.'})

write_json(ROOT / 'analysis/external_parameter_audit.json', {
    'scope': 'read-only static audit; H1 stopped before student forward; H4 not run',
    'rotation_bank': {
        'source': '/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py:255',
        'observation': 'patch calls self.bank.layer(layer).matrix() directly, outside CayleyDenseRotation.forward; bank was ZeRO-managed but separate from engine.module',
        'status': 'POTENTIAL_EXTERNAL_PARAMETER_ACCESS', 'runtime_verified': False},
    'conv_weight_views': {
        'source': '/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py:461',
        'observation': 'linear attention forward passes self.conv1d.weight.squeeze(1) into a functional convolution path rather than calling conv1d.forward',
        'status': 'POTENTIAL_EXTERNAL_PARAMETER_ACCESS', 'runtime_verified': False},
    'embedding_lm_head': {'observation': 'Qwen declares tied lm_head/embedding weights; ZeRO auto handling not individually proven',
                          'status': 'UNDETERMINED'},
    'registration_applied': False,
    'conclusion': 'POTENTIAL_ISSUES_ONLY; no H1 student or H4 execution, so no causal attribution'})

write_json(ROOT / 'analysis/final_verdict.json', {
    'verdict': 'FAIL', 'failure_stage': 'H1 canonical teacher forward',
    'DEEPSPEED_ENVIRONMENT_GATE': 'PASS', 'ZERO3_LOAD_GATE': 'PASS',
    'H1_CANONICAL_FORWARD_GATE': 'FAIL', 'H1_C128_QDQ': 'NOT_RUN',
    'H1_RECURRENT_WRITEBACK': 'NOT_RUN', 'H1_GRADIENT_GRAPH': 'NOT_RUN',
    'H1_ADAM_UPDATE': 'NOT_RUN', 'H1_REPEATABILITY': 'NOT_RUN',
    'H4_STATUS': 'NOT_RUN', 'H8_STATUS': 'NOT_RUN', 'H16_STATUS': 'NOT_RUN', 'H32_STATUS': 'NOT_RUN',
    'MAX_STABLE_HORIZON': 'NONE_ESTABLISHED', 'MEMORY_LEAK': 'UNKNOWN',
    'ZERO3_RECURRENT_TRAINING_FEASIBLE': 'NOT_DETERMINED', 'H32_FEASIBILITY': 'NOT_DETERMINED',
    'DEEPSPEED_ZERO3_RECURRENT_ROUTE': 'STOP_AT_H1',
    'EXTERNAL_PARAMETER_ISSUE': 'POTENTIAL_NOT_RUNTIME_VERIFIED',
    'ZERO3_ROTATION_CHECKPOINT_PORTABILITY': 'NOT_RUN',
    'FORMAL_C5_C6_TRAINING_STARTED': False, 'AIME_STARTED': False,
    'OTHER_USER_TASKS_INTERRUPTED': False,
    'CANONICAL_ENVIRONMENT_UNCHANGED': canonical_unchanged and canonical_ds_absent,
    'h4_root_cause_diagnostic': 'NOT_APPLICABLE_H4_NOT_RUN'})

c = comparisons[0]
write_text(ROOT / 'reports/H1_SEMANTICS_REPORT.md', f'''# H1 prerequisite forward gate: FAIL

The two ZeRO-3 ranks agree with each other, but not with the frozen canonical single-GPU result. Source and sample SHA256 match.

| Metric | Canonical | ZeRO-3 | Conservative error/bound | Frozen threshold |
| --- | ---: | ---: | ---: | ---: |
| Teacher loss | {c['historical_loss']:.9f} | {c['zero3_loss']:.9f} | relative {c['loss_relative_error']:.6g} | {thresholds['loss_relative_error']} |
| Logits max | {historical['logits']['max']:.6g} | {ranks[0]['logits']['max']:.6g} | max-abs ≥ {c['logits_max_abs_error_lower_bound']:.6g} | {thresholds['logits_max_abs']} |
| Logits L2 | {historical['logits']['l2']:.6g} | {ranks[0]['logits']['l2']:.6g} | relative L2 ≥ {c['logits_relative_l2_error_lower_bound']:.6g} | {thresholds['logits_relative_l2']} |
| Layer 16 state max | {historical['selected_states']['16']['max']:.6g} | {ranks[0]['selected_states']['16']['max']:.6g} | max-abs ≥ {c['selected_state_bounds']['16']['max_abs_lower_bound']:.6g} | {thresholds['state_max_abs']} |

These are lower bounds from retained summaries, not reconstructed tensor-level errors. Each exceeds its pre-registered threshold. Layer 0 state SHA is exact; layer 16/30 and logits SHA differ. Exact root cause is unknown. No recurrent QDQ, backward, Adam update, H4 or later horizon was attempted after this failure.
''')
write_text(ROOT / 'reports/H4_COMPATIBILITY_REPORT.md', '# H4 compatibility: NOT RUN\n\nThe H1 canonical teacher-forward gate failed. The frozen stop rule forbids H4 and any H4 root-cause diagnostic.\n')
write_text(ROOT / 'reports/MEMORY_SCALING.md', '# Memory scaling: NOT DETERMINED\n\nZeRO-3 loaded and partitioned 8,953,803,264 logical Qwen parameters. The H1 forward semantic gate failed, so no valid recurrent horizon memory scaling was measured. No H32 allocation or leak conclusion is available.\n')
write_text(ROOT / 'reports/ZERO3_DESIGN.md', '''# ZeRO-3 design and observed setup

Pure two-rank ZeRO Stage 3, BF16, no CPU/NVMe offload, PP, TP or activation checkpointing. Frozen canonical Qwen3.5-9B weights were loaded through the Transformers `HfDeepSpeedConfig` ZeRO-3 integration. Frozen model has 8,953,803,264 logical parameters. The 195,072-parameter Cayley rotation bank was created under `deepspeed.zero.Init` because the ZeRO-3 Adam optimizer requires partition metadata. It remained separate from `engine.module`; the optimizer was constructed from rotation parameters only. This is permitted by the protocol but introduces untested external-parameter lifecycle risk.

First load attempt failed before forward because the separately created bank lacked `partition_numel`; the exact error is preserved. The one setup correction moved bank creation into ZeRO initialization. Second load passed. H1 teacher forward subsequently failed and ended this route. No change was made to the frozen V1 model, QDQ, rotation formula, objective or source.
''')
write_text(ROOT / 'reports/FINAL_REPORT.md', f'''# QWEN_GDN_DEEPSPEED_ZERO3_RECURRENT_FEASIBILITY_V1 — FAIL at H1

Branch: `exp/qwen-gdn-deepspeed-zero3-recurrent-feasibility-v1`; base `c4fa1819633398696b3fed98e82b95fcc07473fa`; FSDP2 parent `ad5007de643f55d92fccdb37562318f22a2e734b`. HEAD and commit list are recorded by Git at delivery.

Isolated environment: `{ENV}`. Canonical environment unchanged: **{'YES' if canonical_unchanged and canonical_ds_absent else 'NO'}** (pip freeze byte-equal to pre-install snapshot; DeepSpeed absent there). Python {sys.version.split()[0]}, Torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL {torch.cuda.nccl.version()}, DeepSpeed {deepspeed.__version__}, Transformers {transformers.__version__}, Triton {triton.__version__}, FLA {fla_version}. GPUs 0/1, ZeRO stage 3, CPU/NVMe offload NO, PP NO, TP NO, activation checkpoint NO.

Trainable Qwen base parameters: **0**. Logical Qwen base parameters: **8,953,803,264**. Trainable rotation parameters: **195,072**. Optimizer initially contained rotation only; DeepSpeed ZeRO-managed the bank. Two-rank NCCL and load/sharding gates PASS. First load setup attempt failed with `AttributeError: 'Parameter' object has no attribute 'partition_numel'`; its evidence was preserved and one parameter-creation-scope correction allowed load.

H1 canonical teacher forward: **FAIL**. Historical loss {c['historical_loss']:.9f}; ZeRO-3 loss {c['zero3_loss']:.9f}; relative error {c['loss_relative_error']:.6g} vs frozen 1e-5 threshold. Logits max-abs lower bound {c['logits_max_abs_error_lower_bound']:.6g} > {thresholds['logits_max_abs']}; logits relative-L2 lower bound {c['logits_relative_l2_error_lower_bound']:.6g} > {thresholds['logits_relative_l2']}. Layer 16 state max-abs lower bound {c['selected_state_bounds']['16']['max_abs_lower_bound']:.6g} > {thresholds['state_max_abs']}. Exact cause **unknown**. No claim that recurrent BPTT itself failed is supported.

H1 C128 QDQ: NOT RUN; writeback: NOT RUN; gradient graph: NOT RUN; Adam update: NOT RUN; repeatability: NOT RUN. H4/H8/H16/H32: NOT RUN. H4 first error/root-cause diagnostic: NOT APPLICABLE. Max stable horizon: NONE ESTABLISHED. H32 peak memory: NOT RUN. Memory leak: UNKNOWN. External parameter audit: potential rotation `.matrix()` and conv-weight-view risks, not runtime verified; no registration was applied. Rotation checkpoint portability: NOT RUN. `ZERO3_RECURRENT_TRAINING_FEASIBLE=NOT_DETERMINED`; `H32_FEASIBILITY=NOT_DETERMINED`; route `STOP_AT_H1` by protocol.

Formal C5/C6 training started: **NO**. AIME started: **NO**. Other user tasks interrupted: **NO**. Old FSDP2 conclusions and artifacts remain unchanged. The result is a valid negative feasibility gate, not a comparative verdict about DeepSpeed versus FSDP2. Further investigation would require a separately authorized forward-parity protocol; this task stops here.
''')

paths = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and '/logs/' not in str(p) and '/.venv-' not in str(p)
               and '__pycache__' not in p.parts and p.suffix != '.pyc'
               and p != ROOT / 'hashes/artifact_sha256.txt')
write_text(ROOT / 'hashes/artifact_sha256.txt', ''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in paths))
print(json.dumps({'verdict': 'FAIL', 'stage': 'H1_CANONICAL_FORWARD_GATE',
                  'canonical_unchanged': canonical_unchanged and canonical_ds_absent,
                  'loss_relative_error': c['loss_relative_error'], 'artifacts': len(paths)}))
