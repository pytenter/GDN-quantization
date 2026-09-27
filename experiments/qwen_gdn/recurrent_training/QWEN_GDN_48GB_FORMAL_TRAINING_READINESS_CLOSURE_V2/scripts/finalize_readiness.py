#!/usr/bin/env python3
"""Summarize immutable V2 diagnostic JSON; never launch training or inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'analysis'
R = ROOT / 'reports'


def read(name):
    path = A / name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'refusing to overwrite frozen artifact: {path}')
    with path.open('x', encoding='utf-8') as out:
        if isinstance(value, str):
            out.write(value)
        else:
            json.dump(value, out, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')


def gi(value):
    return f'{value / 2**30:.3f} GiB'


def gate(condition, h, sustained):
    name = (f'{condition}_stability.json' if sustained and h == 32 else
            f'{condition}_H{h}_stability.json' if sustained else
            f'{condition}_H{h}_fulldoc.json')
    value = read(name)
    if value is not None:
        return value['status'], value
    error = read(name.removesuffix('.json') + '.error.json')
    return (error['status'], error) if error else ('NOT_SCREENED', None)


parent = json.loads((ROOT / 'configs/parent_manifest.json').read_text(encoding='utf-8'))
policy = json.loads((ROOT / 'configs/horizon_policy_audit.json').read_text(encoding='utf-8'))
manifest = read('frozen_rotation_matrix_manifest.json')
same = read('frozen_rotation_same_host.json')
integrity = read('cross_host_file_integrity.json')
original = read('original_server_deployment_gate.json')
if not all((manifest, same, integrity, original)):
    raise RuntimeError('deployment evidence incomplete')
if manifest['layer_count'] != 24 or len(manifest['per_layer']) != 24:
    raise RuntimeError('matrix manifest must contain exactly 24 layers')
artifact_hash = manifest['artifact_sha256']
portability = (same['FROZEN_R_LOADER_SEMANTICS_GATE'] == 'PASS' and
               same['all_24_matrix_hashes_exact'] and
               integrity['FROZEN_R_FILE_PORTABILITY_GATE'] == 'PASS' and
               original['ORIGINAL_SERVER_FROZEN_R_RUNTIME'] == 'PASS' and
               original['C128_QDQ_gate'] and original['recurrent_writeback_gate'] and
               original['all_24_loaded_matrix_hashes_exact'] and
               artifact_hash == same['artifact_sha256'] == integrity['actual_sha256'] == original['artifact_sha256'])

checks = {}
for h in (32, 64, 128):
    checks[str(h)] = {}
    for condition in ('C5', 'C6'):
        one_status, one = gate(condition, h, False)
        sustained_status, sustained = gate(condition, h, True)
        checks[str(h)][condition] = {'one_update': one_status, 'sustained': sustained_status,
                                      'one_update_evidence': one is not None,
                                      'sustained_evidence': sustained is not None}

h32_ready = all(checks['32'][c]['one_update'] == 'PASS' and
                checks['32'][c]['sustained'] == 'PASS' for c in ('C5', 'C6'))
both_higher_screened = all(
    checks[str(h)][c]['one_update'] != 'NOT_SCREENED' and
    (checks[str(h)][c]['one_update'] != 'PASS' or
     checks[str(h)][c]['sustained'] != 'NOT_SCREENED')
    for h in (64, 128) for c in ('C5', 'C6'))
selected = None
if h32_ready and both_higher_screened:
    for h in (128, 64, 32):
        if all(checks[str(h)][c]['one_update'] == 'PASS' and
               checks[str(h)][c]['sustained'] == 'PASS' for c in ('C5', 'C6')):
            selected = h
            break

validation = {}
for condition in ('C5', 'C6'):
    _, item = gate(condition, 32, True)
    if item is not None and 'validation_memory' in item:
        validation[condition] = {'status': item['validation_return_to_baseline'],
                                 'documents': item['validation']['documents'],
                                 'samples': item['validation']['samples'],
                                 'memory': item['validation_memory'],
                                 'memory_creep': item['memory_creep'],
                                 'end_allocated_span_bytes': item['end_allocated_span_bytes']}

provenance = all(
    gate(condition, 32, False)[1] is not None and
    all(x['exact'] for x in gate(condition, 32, False)[1]['parent_provenance'].values())
    for condition in ('C5', 'C6'))
ready = (portability and h32_ready and selected is not None and provenance and
         len(validation) == 2 and all(validation[c]['status'] == 'PASS' and
         validation[c]['memory_creep'] == 'NO' for c in ('C5', 'C6')))
horizon = {'historical_policy': policy['later_frozen_selection_rule'],
           'historical_candidate_order': policy['later_frozen_candidate_order'],
           'screen_order_this_task': [32, 64, 128],
           'resource_only': True, 'checks': checks,
           'all_higher_one_update_screens_complete': both_higher_screened,
           'selected_formal_horizon': selected}
verdict = {'task': ROOT.name, 'parent_experiment': parent['parent_experiment'],
           'parent_sha': parent['parent_sha'],
           'V1_RECOMPUTED_MATRIX_HASH_PORTABILITY': 'FAIL',
           'V2_FROZEN_MATRIX_DEPLOYMENT_PORTABILITY': 'PASS' if portability else 'FAIL',
           'theta_checkpoint_sha256': parent['training_theta_checkpoint_sha256'],
           'frozen_R_artifact_sha256': artifact_hash, 'matrix_layers': manifest['layer_count'],
           'source_model_manifest_provenance': 'PASS' if provenance else 'FAIL',
           'C5_FULLDOC_H32_GATE': checks['32']['C5']['one_update'],
           'C6_FULLDOC_H32_GATE': checks['32']['C6']['one_update'],
           'C5_FULLDOC_MEMORY_LIFETIME': checks['32']['C5']['sustained'],
           'C6_FULLDOC_MEMORY_LIFETIME': checks['32']['C6']['sustained'],
           'validation_return_to_baseline': 'PASS' if len(validation) == 2 and
               all(validation[c]['status'] == 'PASS' for c in validation) else 'FAIL',
           'selected_formal_horizon': selected,
           'FORMAL_RECURRENT_TRAINING_READY': 'YES' if ready else 'NO',
           'FORMAL_C5_C6_TRAINING': 'NOT_STARTED', 'AIME': 'NOT_STARTED',
           'distributed_framework_used': False,
           'original_inference_runtime_modified': False}
write_once(A / 'horizon_policy.json', horizon)
write_once(A / 'validation_memory.json', validation)
write_once(A / 'final_verdict.json', verdict)

lines = ['# Full-document recurrent memory closure', '',
         'The unchanged canonical C5/C6 loop uses one 1024-token document per optimizer update, '
         'eight frozen captures, segment backward at each occupied H boundary, cache detach every H tokens, '
         'and one Adam step after the document. No training formula or graph-lifetime strategy was changed.', '']
for c in ('C5', 'C6'):
    one_status, one = gate(c, 32, False)
    sustained_status, sustained = gate(c, 32, True)
    lines.append(f'## {c}')
    lines.append('')
    if one and one.get('documents'):
        d = one['documents'][0]
        lines.append(f"H32 single full document: **{one_status}**; {d['document_tokens']} tokens, "
                     f"{d['captured']} captures, {d['backward_count']} occupied segment backprops; "
                     f"peak allocated {gi(d['peak_allocated_bytes'])}, reserved {gi(d['peak_reserved_bytes'])}; "
                     f"24/24 valid rotation gradients; orthogonality {one['orthogonality']['status']}.")
    else:
        lines.append(f'H32 single full document: **{one_status}**.')
    if sustained and sustained.get('documents'):
        peaks = ', '.join(gi(d['peak_allocated_bytes']) for d in sustained['documents'])
        lines.append(f"Three-update lifetime: **{sustained_status}**; peak allocated by document [{peaks}]; "
                     f"end-allocated span {sustained['end_allocated_span_bytes']} bytes; "
                     f"memory creep {sustained['memory_creep']}; "
                     f"validation {sustained['validation']['documents']} documents/"
                     f"{sustained['validation']['samples']} captures, return-to-baseline "
                     f"{sustained['validation_return_to_baseline']} "
                     f"(delta {sustained['validation_memory']['return_delta_allocated_bytes']} bytes).")
    else:
        lines.append(f'Three-update lifetime: **{sustained_status}**.')
    lines.append('')
lines += ['The hard peak-free margin is 5% of device VRAM, with 10% preferred. '
          'All horizon decisions are resource/stability-only; loss and validation score did not select a horizon.', '']
write_once(R / 'FULL_DOCUMENT_MEMORY_CLOSURE.md', '\n'.join(lines))

readiness = ['# Formal recurrent training readiness', '',
             f"Verdict: **{verdict['FORMAL_RECURRENT_TRAINING_READY']}**. "
             'This is only a prospective readiness closure; no formal C5/C6 checkpoint or AIME generation was started.', '',
             f"V1 independent matrix recomputation: **FAIL**. V2 frozen-matrix deployment: "
             f"**{verdict['V2_FROZEN_MATRIX_DEPLOYMENT_PORTABILITY']}**.",
             f"C5/C6 H32 one-document: **{checks['32']['C5']['one_update']}**/"
             f"**{checks['32']['C6']['one_update']}**; sustained memory: "
             f"**{checks['32']['C5']['sustained']}**/**{checks['32']['C6']['sustained']}**.",
             f"Frozen historical policy: {policy['later_frozen_selection_rule']}; "
             f"screened candidate set {policy['later_frozen_candidate_order']}; "
             f"selected horizon: {selected if selected is not None else 'not selected'}.",
             'The training host remains the 48GB single-vGPU. The original four-RTX3090 Qwen server '
             'remains the canonical inference/evaluation host; its source runtime was not modified. '
             'Future formal checkpoints would preserve theta as provenance and deploy exact frozen FP32 R matrices.',
             'Distributed frameworks, formal C5/C6 training, and AIME were not used in this task.', '']
write_once(R / 'FORMAL_TRAINING_READINESS.md', '\n\n'.join(readiness))

summary = ['# QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2 — final report', '',
           f"Parent: `{parent['parent_branch']}` at `{parent['parent_sha']}`. "
           f"Training host: NVIDIA vGPU-48GB. Canonical inference host: four RTX 3090 (24GB each).", '',
           f"V1 recomputed-R exact-hash portability: **FAIL** (unchanged). "
           f"V2 frozen-R deployment portability: **{verdict['V2_FROZEN_MATRIX_DEPLOYMENT_PORTABILITY']}**. "
           f"Theta SHA-256 `{parent['training_theta_checkpoint_sha256']}`; "
           f"frozen-R SHA-256 `{artifact_hash}`; 24/24 per-layer hashes exact.", '',
           'Full-document training has 1024 tokens and eight zero-based captures at '
           '625, 663, 813, 819, 881, 940, 957, 989. There are at most two simultaneously '
           'stored capture losses before a segment backward. Theta gradients accumulate over '
           'H segments; one optimizer step follows the document.', '']
for c in ('C5', 'C6'):
    s, o = gate(c, 32, False)
    t, st = gate(c, 32, True)
    if o and o.get('documents'):
        d = o['documents'][0]
        summary.append(f"{c} H32: **{s}**; peak allocated {gi(d['peak_allocated_bytes'])}, "
                       f"peak reserved {gi(d['peak_reserved_bytes'])}; sustained **{t}**, "
                       f"memory creep {st['memory_creep'] if st else 'unknown'}.")
    else:
        summary.append(f'{c} H32: **{s}**; sustained **{t}**.')
summary += ['', f"Historical horizon rule: {policy['later_frozen_selection_rule']}. "
            f"Selected horizon: {selected if selected is not None else 'not selected'}. "
            f"FORMAL_RECURRENT_TRAINING_READY={verdict['FORMAL_RECURRENT_TRAINING_READY']}.",
            'Formal C5/C6 training: NOT_STARTED. AIME: NOT_STARTED. Distributed framework: NO. '
            'Original inference source modification: NO. Other user processes were not interrupted.', '',
            'See `ROTATION_DEPLOYMENT_PORTABILITY_V2.md`, `FULL_DOCUMENT_GRAPH_LIFETIME_AUDIT.md`, '
            '`FULL_DOCUMENT_MEMORY_CLOSURE.md`, and `FORMAL_TRAINING_READINESS.md` for stage evidence.']
write_once(R / 'FINAL_REPORT.md', '\n'.join(summary) + '\n')

hashes = []
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.suffix in ('.pt', '.pyc') or 'logs' in path.parts or '__pycache__' in path.parts:
        continue
    if path.name == 'artifact_sha256.txt':
        continue
    rel = path.relative_to(ROOT).as_posix()
    hashes.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}')
write_once(ROOT / 'hashes/artifact_sha256.txt', '\n'.join(hashes) + '\n')
print(json.dumps(verdict, indent=2), flush=True)
