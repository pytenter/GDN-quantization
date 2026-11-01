#!/usr/bin/env python3
"""Validate causal-prefix stop evidence and write compact immutable reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'analysis'
REPORTS = ROOT / 'reports'


def read(path):
    return json.loads(path.read_text())


def save_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def save_report(name, text):
    path = REPORTS / name
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + '\n')


def main():
    prereg = read(ROOT / 'preregistration.json')
    baseline = read(ANALYSIS / 'P0R0_summary.json')
    prefix = read(ANALYSIS / 'P1R0_summary.json')
    pair = read(ANALYSIS / 'P0R0_vs_P1R0.json')
    reps = [read(ANALYSIS / 'prefix_replay' / f'rep_{i:02d}.json') for i in range(1, 11)]
    if [x['replicate'] for x in reps] != list(range(1, 11)):
        raise RuntimeError('fresh-process replay count/index invalid')
    if baseline['status'] != 'COMPLETE' or prefix['status'] != 'COMPLETE':
        raise RuntimeError('factorial prefix pair incomplete')
    if baseline['source_sha256'] != prefix['source_sha256'] or baseline['frozen_input'] != prefix['frozen_input']:
        raise RuntimeError('factorial source or frozen input mismatch')
    if baseline['source_sha256']['modeling'] != prereg['canonical_modeling_source_sha256'] or baseline['source_sha256']['v1_c5'] != prereg['canonical_c5_source_sha256']:
        raise RuntimeError('source provenance changed after preregistration')
    if pair['first_divergent_node'] != 'gdn.prefix_output' or pair['previous_node'] != 'gdn.prefix_input':
        raise RuntimeError('unexpected first divergence; do not use this stop-path report')
    if not pair['prefix_input_exact'] or pair['prefix_output_exact']:
        raise RuntimeError('expected exact prefix input and divergent prefix output')
    if pair['full_model_forward_semantics_gate'] != 'FAIL':
        raise RuntimeError('expected preregistered prefix-only failure')
    if any((ANALYSIS / f'{c}_summary.json').exists() for c in ('P0R1', 'P1R1')):
        raise RuntimeError('strict-runtime conditions were run after the prefix stop')
    ref_input = baseline['captures']['gdn.prefix_input']['sha256']
    ref_canonical = baseline['captures']['gdn.prefix_output']['sha256']
    ref_candidate = prefix['captures']['gdn.prefix_output']['sha256']
    for rep in reps:
        if rep['input']['sha256'] != ref_input:
            raise RuntimeError('replay input does not match real full-model prefix input')
        if not rep['canonical_same_process_exact'] or not rep['candidate_same_process_exact']:
            raise RuntimeError('same-process operator nondeterminism')
        if set(rep['canonical_hashes']) != {ref_canonical} or set(rep['candidate_hashes']) != {ref_candidate}:
            raise RuntimeError('fresh-process operator output differs from full-model capture')
    replay = {'fresh_process_count': 10, 'same_process_repeats_per_process': 5,
              'input_sha256': ref_input,
              'canonical_output_sha256': ref_canonical,
              'candidate_output_sha256': ref_candidate,
              'canonical_same_process_repeatability': 'EXACT',
              'candidate_same_process_repeatability': 'EXACT',
              'canonical_fresh_process_repeatability': 'EXACT',
              'candidate_fresh_process_repeatability': 'EXACT',
              'input_metadata': reps[0]['input'],
              'canonical_vs_candidate': reps[0]['canonical_vs_candidate'],
              'canonical_vs_cpu_fp64_reference': reps[0]['canonical_vs_cpu_fp64_reference'],
              'candidate_vs_cpu_fp64_reference': reps[0]['candidate_vs_cpu_fp64_reference'],
              'cpu_fp64_reference_scope': 'mathematical diagnostic only; not a runtime candidate'}
    save_json(ANALYSIS / 'prefix_full_shape_replay.json', replay)
    divergence = {'first_divergent_layer': 0,
                  'first_divergent_node': pair['first_divergent_node'],
                  'previous_node': pair['previous_node'],
                  'previous_node_exact': pair['previous_node_exact'],
                  'prefix_input_exact': pair['prefix_input_exact'],
                  'prefix_output_exact': pair['prefix_output_exact'],
                  'prefix_output_metrics': pair['captures']['gdn.prefix_output'],
                  'source_delta_evidence': 'V2 deterministic_chunk_source_delta.json verified copied Torch chunk function differs only at g.cumsum replacement'}
    save_json(ANALYSIS / 'first_divergence.json', divergence)
    factorial = {'P0R0': 'COMPLETE', 'P1R0': 'COMPLETE',
                 'P0R1': 'NOT_RUN_PRE_REGISTERED_PREFIX_FAILURE_STOP',
                 'P1R1': 'NOT_RUN_PRE_REGISTERED_PREFIX_FAILURE_STOP',
                 'PREFIX_MAIN_EFFECT': 'PRESENT',
                 'STRICT_RUNTIME_MAIN_EFFECT': 'NOT_IDENTIFIABLE',
                 'INTERACTION_SIGNAL': 'NOT_IDENTIFIABLE',
                 'comparison_path': 'analysis/P0R0_vs_P1R0.json',
                 'V2_combined_comparison_is_confounded': True}
    save_json(ANALYSIS / 'factorial_comparison.json', factorial)
    verdict = {
        'task': ROOT.name, 'verdict': 'FAIL',
        'parent_v2_commit_sha': prereg['parent_v2_commit_sha'],
        'main_base_sha': prereg['main_base_sha'],
        'P0R0_status': factorial['P0R0'], 'P1R0_status': factorial['P1R0'],
        'P0R1_status': factorial['P0R1'], 'P1R1_status': factorial['P1R1'],
        'PREFIX_MAIN_EFFECT': 'PRESENT',
        'STRICT_RUNTIME_MAIN_EFFECT': 'NOT_IDENTIFIABLE',
        'INTERACTION_SIGNAL': 'NOT_IDENTIFIABLE',
        'FIRST_DIVERGENT_OPERATOR': 'torch.Tensor.cumsum(dim=-1) versus fixed_left_to_right_cumsum',
        'FIRST_DIVERGENT_LAYER': 0,
        'FIRST_DIVERGENT_NODE': 'gdn.prefix_output',
        'PREFIX_INPUT_EXACT': True, 'PREFIX_OUTPUT_EXACT': False,
        'CURRENT_DETERMINISTIC_PREFIX_CANDIDATE': 'REJECTED_FOR_CANONICAL_COMPATIBILITY',
        'canonical_prefix_fresh_process_repeatability': 'EXACT_10_OF_10',
        'candidate_prefix_fresh_process_repeatability': 'EXACT_10_OF_10',
        'STRICT_RUNTIME_FIRST_PROBLEMATIC_FACTOR': 'NOT_TESTED_PRE_REGISTERED_STOP',
        'full_model_logits_max_abs': pair['logits']['max_abs'],
        'full_model_logits_rel_l2': pair['logits']['relative_l2'],
        'C128_qcode_changed_count_by_layer': pair['qcode_changed_count_by_layer'],
        'FULL_MODEL_FORWARD_SEMANTICS_GATE': 'FAIL',
        'FRESH_PROCESS_FORWARD_REPEATABILITY': 'NOT_RUN_FORWARD_SEMANTICS_FAIL',
        'BACKWARD_REPEATABILITY_GATE': 'NOT_RUN_FORWARD_SEMANTICS_FAIL',
        'PP2_RETRY_AUTHORIZED': 'NO',
        'PP2_started': 'NO', 'H32_started': 'NO',
        'formal_C5_C6_training_started': 'NO', 'AIME_started': 'NO',
        'other_user_tasks_interrupted': 'NO',
        'limitations': ['P0R1/P1R1 not run after preregistered prefix-only failure; strict-runtime effect and interaction are not identifiable in this experiment',
                        'teacher-forward cache QDQ is a downstream diagnostic, not a substituted recurrent training runtime',
                        'CPU FP64 reference assesses numerical error only and is not a candidate runtime']}
    save_json(ANALYSIS / 'final_verdict.json', verdict)
    save_report('FACTORIAL_CAUSAL_ISOLATION.md', f'''# Factorial causal isolation

P0R0 and P1R0 completed on the same frozen V2 sample, model and normal runtime. P0R1 and P1R1 were **not run** under the preregistered prefix-failure stop rule. Thus a prefix main effect is present; strict-runtime main effect and interaction are **not identifiable**. The old V2 combined failure remains a confounded comparison.

Frozen input IDs SHA256: `{baseline['frozen_input']['input_ids_sha256']}`. Model source SHA256: `{baseline['source_sha256']['modeling']}`. Details: `analysis/factorial_comparison.json` and `analysis/P0R0_vs_P1R0.json`.
''')
    save_report('FIRST_DIVERGENCE_LOCALIZATION.md', f'''# First divergence

First GDN layer: **0**. Previous node `gdn.prefix_input` is bitwise exact; first non-identical node `gdn.prefix_output`. Real captured input is FP32, shape `{replay['input_metadata']['shape']}`, contiguous stride `{replay['input_metadata']['stride']}`. The only audited copied-function source delta is replacing `g.cumsum(dim=-1)` by fixed left-to-right accumulation.

Prefix output max abs difference: {divergence['prefix_output_metrics']['max_abs']:.10g}; relative L2: {divergence['prefix_output_metrics']['relative_l2']:.10g}; different elements: {divergence['prefix_output_metrics']['different_element_count']}. Full logits max abs: {pair['logits']['max_abs']}; relative L2: {pair['logits']['relative_l2']:.10g}. This localizes the numerical path change; it does not prove mathematical incorrectness.
''')
    save_report('PREFIX_OPERATOR_AUDIT.md', f'''# Full-shape prefix replay

The exact full-model prefix input was replayed in **10 fresh processes**, with **5 same-process repeats** each. Both operators were internally exact and reproducible across processes, but their output hashes differ consistently. Replay output hashes match the corresponding full-model captures.

Canonical versus candidate max abs: {replay['canonical_vs_candidate']['max_abs']:.10g}, relative L2: {replay['canonical_vs_candidate']['relative_l2']:.10g}. Against CPU FP64 mathematical reference, canonical relative L2 is {replay['canonical_vs_cpu_fp64_reference']['relative_l2']:.10g} and candidate is {replay['candidate_vs_cpu_fp64_reference']['relative_l2']:.10g}. The candidate is deterministic but not canonical-compatible for this model; CPU FP64 was diagnostic only.
''')
    save_report('STRICT_RUNTIME_FACTOR_AUDIT.md', '''# Strict runtime factor audit

**NOT RUN.** The prefix-only intervention already failed the frozen full-model semantics gate, so the preregistered stop rule prohibited continuing to P0R1, P1R1 or R1a–R1f. No strict-runtime causal claim is made.
''')
    save_report('FINAL_REPORT.md', f'''# QWEN_GDN_DETERMINISM_CAUSAL_ISOLATION_V1 — FAIL

The previous V2 numerical early stop remains intact and is archived at commit `{prereg['parent_v2_commit_sha']}`. This new experiment isolated the prefix intervention under normal runtime: P0R0 and P1R0 completed, with identical input through the first GDN prefix input and first divergence at its prefix output. P1R0 fails every frozen full-model numerical criterion; full-model logits max abs {pair['logits']['max_abs']}, relative L2 {pair['logits']['relative_l2']:.10g}. The prefix difference is small locally (max abs {divergence['prefix_output_metrics']['max_abs']:.10g}) but propagates to logits and C128 qcodes. Per-layer qcode counts are in `analysis/final_verdict.json`.

The exact prefix input was replayed 10 times in fresh processes (five repeats per process). Canonical and candidate each repeat exactly, yet differ from each other; the canonical result is closer to a CPU FP64 mathematical reference on this tensor. This demonstrates a reproducible arithmetic-path difference, not a claim that the candidate is mathematically invalid.

By the preregistered stop rule, P0R1/P1R1 and strict-runtime subfactors were not run. Therefore strict-runtime main effect and interaction remain unidentifiable. The current fixed left-to-right prefix candidate is rejected for canonical compatibility. Full-model forward semantics: **FAIL**; fresh-process full-model repeatability and backward: **NOT RUN**; PP2 retry authorization: **NO**.

PP2, H32/H64/H128, formal C5/C6 training and AIME were **not started**. No other user task was interrupted. All model tests used the original Qwen server, one idle GPU. Diagnostic tensors are temporary and excluded from Git; compact hashes and summaries are retained.
''')
    files = [p for p in ROOT.rglob('*') if p.is_file() and 'temporary' not in p.parts and '__pycache__' not in p.parts and p.name != 'artifact_sha256.txt' and p.suffix != '.pt']
    manifest = ROOT / 'hashes' / 'artifact_sha256.txt'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(files)))
    print(json.dumps({'verdict': 'FAIL', 'compact_file_count': len(files),
                      'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
