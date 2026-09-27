#!/usr/bin/env python3
"""Instrument the frozen C5/C6 document loop without changing its updates."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path

import torch

from runtime import ROOT, check_parent_sources, cuda_memory, load_core, load_model_and_tokenizer, save_once, tensor_sha256


def _model_sentinels(model):
    chosen = {}
    for name, parameter in model.named_parameters():
        if not chosen or 'embed_tokens' in name or 'lm_head' in name:
            if name not in chosen:
                chosen[name] = tensor_sha256(parameter.detach().flatten()[:1024])
        if len(chosen) >= 3:
            break
    return chosen


def _snap(device, label, *, token=None):
    value = cuda_memory(device)
    value['stage'] = label
    if token is not None:
        value['token_zero_based'] = token
    return value


def _finite_number(value):
    return math.isfinite(float(value))


def train_document(core, model, tokenizer, bank, patch, optimizer, device, row, condition, horizon, update):
    """Canonical train() lines 815-864 with observational memory checkpoints."""
    objective = 'DENSE_STATE' if condition == 'C5' else 'DENSE_FUNCTIONAL'
    ids, positions = core.tokenize(tokenizer, row)
    position_set = set(positions)
    doc = {'update': update, 'document_id': row['document_id'], 'document_tokens': len(ids),
           'capture_positions_zero_based': positions, 'horizon': horizon, 'memory': [],
           'capture_memory': [], 'backward_memory': [], 'elapsed_seconds': None}
    started = time.time()
    targets = core.collect_teacher_targets(model, patch, ids, positions)
    student_cache = None
    optimizer.zero_grad(set_to_none=True)
    doc_primary = doc_state = doc_functional = doc_combined = 0.0
    segment_losses = []
    captured = 0
    doc['memory'].append(_snap(device, 'before_document_student'))
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for index, token_id in enumerate(ids):
            capture = index in position_set
            if capture:
                core.install_teacher_target(patch, targets[index], device)
            student_cache = core.forward_student(model, patch, token_id, student_cache, capture)
            if capture:
                state, functional, c5, c6 = patch.captured_losses()
                primary = state if objective == 'DENSE_STATE' else functional
                combined = c5 if objective == 'DENSE_STATE' else c6
                segment_losses.append(combined / core.TARGETS_PER_DOCUMENT)
                doc_primary += float(primary.detach().cpu()) / core.TARGETS_PER_DOCUMENT
                doc_state += float(state.detach().cpu()) / core.TARGETS_PER_DOCUMENT
                doc_functional += float(functional.detach().cpu()) / core.TARGETS_PER_DOCUMENT
                doc_combined += float(combined.detach().cpu()) / core.TARGETS_PER_DOCUMENT
                captured += 1
                doc['capture_memory'].append(_snap(device, 'after_forward_capture', token=index))
                del state, functional, c5, c6, primary, combined
            boundary = (index + 1) % horizon == 0 or index + 1 == len(ids)
            if boundary:
                if segment_losses:
                    doc['backward_memory'].append(_snap(device, 'before_backward', token=index))
                    segment_loss = sum(segment_losses)
                    if not bool(torch.isfinite(segment_loss).detach().cpu()):
                        raise FloatingPointError(f'nonfinite loss update={update} token={index}')
                    segment_loss.backward()
                    segment_losses.clear()
                    patch.clear_capture()
                    del segment_loss
                    doc['backward_memory'].append(_snap(device, 'after_backward', token=index))
                core.detach_cache(student_cache)
                if (index + 1) % 128 == 0 or index + 1 == len(ids):
                    print(json.dumps({'event': 'document_progress', 'condition': condition,
                                      'update': update, 'token_count': index + 1,
                                      'capture_count': captured,
                                      'allocated_bytes': torch.cuda.memory_allocated(device),
                                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}), flush=True)
        if captured != core.TARGETS_PER_DOCUMENT:
            raise RuntimeError(f'captured {captured}, expected {core.TARGETS_PER_DOCUMENT}')
        params = list(bank.parameters())
        grads = [p.grad for p in params]
        if len(grads) != 24 or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError(f'invalid gradient at update {update}')
        before_step = [p.detach().clone() for p in params]
        raw_norms = [float(torch.linalg.vector_norm(g.detach().float()).cpu()) for g in grads]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        doc['memory'].append(_snap(device, 'after_optimizer_before_cleanup'))
        displacements = [float(torch.linalg.vector_norm((p.detach() - before).float()).cpu())
                         for p, before in zip(params, before_step)]
        del before_step, targets, student_cache
        patch.clear_capture()
        gc.collect()
        torch.cuda.empty_cache()
        doc['memory'].append(_snap(device, 'after_document_cleanup'))
        doc.update({'captured': captured, 'backward_count': len(doc['backward_memory']) // 2,
                    'train_combined': doc_combined, 'train_primary': doc_primary,
                    'train_state': doc_state, 'train_functional': doc_functional,
                    'gradient_norm_before_clip': float(grad_norm.detach().cpu()),
                    'per_layer_gradient_norm_before_clip': raw_norms,
                    'per_layer_theta_displacement': displacements,
                    'rotation_parameters_with_valid_gradients': len(grads),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'peak_minimum_free_bytes': min(x['device_free_bytes'] for x in
                        doc['memory'] + doc['capture_memory'] + doc['backward_memory']),
                    'device_total_bytes': torch.cuda.get_device_properties(device).total_memory,
                    'elapsed_seconds': time.time() - started})
        for key in ('train_combined', 'train_primary', 'train_state', 'train_functional',
                    'gradient_norm_before_clip'):
            if not _finite_number(doc[key]):
                raise FloatingPointError(f'nonfinite {key} update={update}')
        if any(not _finite_number(x) for x in raw_norms + displacements):
            raise FloatingPointError('nonfinite gradient/displacement')
        print(json.dumps({'event': 'document_complete', 'condition': condition,
                          'update': update, 'document_id': row['document_id'],
                          'elapsed_seconds': doc['elapsed_seconds'],
                          'peak_allocated_bytes': doc['peak_allocated_bytes'],
                          'peak_reserved_bytes': doc['peak_reserved_bytes']}), flush=True)
        return doc
    except BaseException as exc:
        doc['failure'] = {'type': type(exc).__name__, 'message': str(exc),
                          'traceback': traceback.format_exc(),
                          'last_memory': _snap(device, 'failure')}
        raise DocumentFailure(doc) from exc


class DocumentFailure(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(evidence['failure']['message'])


def run(condition: str, horizon: int, sustained: bool):
    condition = condition.upper()
    if condition not in ('C5', 'C6') or horizon not in (32, 64, 128):
        raise ValueError('condition/horizon outside frozen candidate set')
    target_name = (f'{condition}_stability.json' if sustained and horizon == 32 else
                   f'{condition}_H{horizon}_stability.json' if sustained else
                   f'{condition}_H{horizon}_fulldoc.json')
    target = ROOT / 'analysis' / target_name
    error_target = target.with_suffix('.error.json')
    if target.exists() or error_target.exists():
        raise RuntimeError(f'prior evidence exists; do not rerun {target}')
    protocol = json.loads((ROOT / 'configs/full_document_protocol.json').read_text())
    parent_checks = check_parent_sources()
    core = load_core()
    random.seed(core.TRAIN_SEED)
    torch.manual_seed(core.TRAIN_SEED)
    model, tokenizer, device = load_model_and_tokenizer(core)
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003 if condition == 'C5' else 0.001,
                                 weight_decay=0.0)
    train_rows = core.corpus_rows('TRAIN')
    validation_rows = core.corpus_rows('VALIDATION') if sustained else None
    if sustained:
        schedule = [train_rows[random.randrange(len(train_rows))] for _ in range(3)]
        expected = protocol['canonical_train_schedule_first_three_document_ids']
        if [row['document_id'] for row in schedule] != expected:
            raise RuntimeError('canonical training schedule mismatch')
    else:
        schedule = [train_rows[0]]
        if schedule[0]['document_id'] != protocol['fixed_first_document']:
            raise RuntimeError('first training document mismatch')
        ids, positions = core.tokenize(tokenizer, schedule[0])
        token_hash = hashlib.sha256(bytes().join(int(x).to_bytes(4, 'little') for x in ids)).hexdigest()
        if len(ids) != protocol['first_document_token_count'] or positions != protocol['first_document_capture_positions_zero_based']:
            raise RuntimeError('first training document token/positions mismatch')
        # Record this explicit LE-uint32 digest; the frozen audit digest used
        # a different serialization, while document identity/count/positions
        # are checked here against the same verified manifest and tokenizer.
    model_before = _model_sentinels(model)
    device_memory_before = _snap(device, 'after_model_load')
    h = core.hadamard(device)
    evidence = {'condition': condition, 'horizon': horizon, 'sustained': sustained,
                'parent_provenance': parent_checks,
                'model_sentinels_before': model_before,
                'after_model_load': device_memory_before,
                'documents': [], 'formal_training_started': False, 'AIME_used': False,
                'distributed_framework_used': False,
                'canonical_source_sha256': parent_checks['canonical_c5_c6_source']['actual_sha256']}
    if not sustained:
        evidence['first_document_token_sha256_le_uint32'] = token_hash
    print(json.dumps({'event': 'diagnostic_start', 'condition': condition, 'horizon': horizon,
                      'sustained': sustained, 'document_ids': [x['document_id'] for x in schedule]}), flush=True)
    try:
        with core.RecurrentExperimentPatch(model, bank, h) as patch:
            for update, row in enumerate(schedule, 1):
                doc = train_document(core, model, tokenizer, bank, patch, optimizer, device,
                                     row, condition, horizon, update)
                evidence['documents'].append(doc)
            ortho = bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
            evidence['orthogonality'] = {key: ortho[key] for key in
                ('status', 'threshold', 'max_abs_rt_r_minus_i', 'all_finite', 'all_proper')}
            if ortho['status'] != 'PASS':
                raise RuntimeError('ORTHOGONALITY_GATE_FAIL')
            if sustained:
                gc.collect(); torch.cuda.empty_cache()
                before_validation = _snap(device, 'before_validation')
                torch.cuda.reset_peak_memory_stats(device)
                objective = 'DENSE_STATE' if condition == 'C5' else 'DENSE_FUNCTIONAL'
                with torch.no_grad():
                    validation = core.evaluate_validation(model, tokenizer, bank, patch,
                                                          validation_rows, horizon, objective)
                after_validation = _snap(device, 'after_validation')
                evidence['validation'] = validation
                evidence['validation_memory'] = {'before': before_validation,
                                                  'after': after_validation,
                                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                                                  'return_delta_allocated_bytes': after_validation['allocated_bytes'] - before_validation['allocated_bytes']}
                evidence['validation_return_to_baseline'] = 'PASS' if abs(evidence['validation_memory']['return_delta_allocated_bytes']) <= protocol['sustained_memory_creep_stop_span_bytes'] else 'FAIL'
        model_after = _model_sentinels(model)
        evidence['model_sentinels_after'] = model_after
        evidence['base_model_unchanged'] = model_before == model_after and not any(p.requires_grad for p in model.parameters())
        if not evidence['base_model_unchanged']:
            raise RuntimeError('BASE_MODEL_CHANGED')
        total_bytes = evidence['documents'][0]['device_total_bytes']
        minimum_free = min(min(x['peak_minimum_free_bytes'],
                               total_bytes - x['peak_reserved_bytes'])
                           for x in evidence['documents'])
        evidence['minimum_observed_free_fraction'] = minimum_free / total_bytes
        evidence['preferred_10pct_margin_met'] = evidence['minimum_observed_free_fraction'] >= protocol['preferred_peak_free_fraction']
        hard_margin = evidence['minimum_observed_free_fraction'] >= protocol['hard_minimum_peak_free_fraction']
        evidence['hard_5pct_margin_met'] = hard_margin
        end_allocations = [x['memory'][-1]['allocated_bytes'] for x in evidence['documents']]
        evidence['end_allocated_span_bytes'] = max(end_allocations) - min(end_allocations)
        evidence['memory_creep'] = 'NO' if evidence['end_allocated_span_bytes'] <= protocol['sustained_memory_creep_stop_span_bytes'] else 'YES'
        gradients_nonzero = all(x['gradient_norm_before_clip'] > 0 and
                                any(value > 0 for value in x['per_layer_theta_displacement'])
                                for x in evidence['documents'])
        evidence['gradients_and_theta_update_nonzero'] = gradients_nonzero
        evidence['status'] = ('PASS' if hard_margin and gradients_nonzero and evidence['memory_creep'] == 'NO' and
                              (not sustained or evidence['validation_return_to_baseline'] == 'PASS')
                              else 'FAIL')
        evidence['gate'] = f'{condition}_FULLDOC_MEMORY_LIFETIME' if sustained else f'{condition}_FULLDOC_H{horizon}_GATE'
        save_once(target, evidence)
        print(json.dumps({'event': 'diagnostic_final', 'condition': condition, 'horizon': horizon,
                          'sustained': sustained, 'status': evidence['status'],
                          'minimum_observed_free_fraction': evidence['minimum_observed_free_fraction'],
                          'memory_creep': evidence['memory_creep']}), flush=True)
        if evidence['status'] != 'PASS':
            raise RuntimeError('DIAGNOSTIC_GATE_FAIL')
    except BaseException as exc:
        if isinstance(exc, DocumentFailure):
            evidence['failed_document'] = exc.evidence
        evidence['failure'] = {'type': type(exc).__name__, 'message': str(exc),
                               'traceback': traceback.format_exc()}
        evidence['status'] = 'BLOCKED' if 'out of memory' in str(exc).lower() else 'FAIL'
        if not target.exists():
            save_once(error_target, evidence)
        raise
