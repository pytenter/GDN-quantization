"""V2 input-recovery preflight and one-shot canonical deterministic entry gate.

This is not a surrogate-measurement implementation. It deliberately has no
training, generation, replacement scan, full-model load, or automatic retry.
"""
import argparse
import ast
import datetime as dt
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

TASK = 'QWEN_GDN_SCALE_SURROGATE_INPUT_RECOVERY_AND_VALIDATION_V2'
REPO = Path('/data/zypan/worktrees/qwen-gdn-scale-surrogate-input-recovery-v2')
ROOT = REPO / 'experiments' / TASK
V1 = Path('/data/zypan/experiment_runs/QWEN_GDN_SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION_V1/finalized_audit_002')
CLOSURE = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1')
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
ARCH_SHA = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
UUID0 = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(rel, value):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def prose(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(text)


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def resource():
    return {
        'utc': utc(), 'host': socket.gethostname(),
        'disk_available_bytes': shutil.disk_usage('/data').free,
        'gpu_inventory_csv': command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
        'compute_processes_csv': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader']),
    }


def seal_rows(manifest):
    rows = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rows[rel.lstrip('*')] = digest
    return rows


def prepare():
    check(not (ROOT / 'RECOVERY_PLAN.md').exists(), 'V2 preparation already exists; no overwrite')
    v = read(V1 / 'ASSET_AND_PROVENANCE_AUDIT.json')
    check(sha(V1 / 'SHA256SUMS.txt') == '0dc48c211bfbce01664f3b18a79a757e19f2b2bc392cb051b62b2d505dbb0fc2', 'V1 final manifest identity')
    old_rows = seal_rows(V1 / 'SHA256SUMS.txt')
    check(sha(V1 / 'ASSET_AND_PROVENANCE_AUDIT.json') == old_rows['ASSET_AND_PROVENANCE_AUDIT.json'], 'V1 audit identity')
    check(all(x['status'] == 'PASS' for x in v['audited_checks']), 'V1 asset identity was not PASS')
    roots = v['source_roots']
    parent_root = Path(roots['PARENT_ROOT'])
    fixture = parent_root / 'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1/engineering/capture_2048'
    fm = read(fixture / 'drive_manifest.json')
    shard = next(x for x in fm['shards'] if x['first_token'] == 1)
    check(sha(fixture / shard['relative_path']) == shard['file_sha256'], 'reused real Doc1 first drive shard identity')
    check(sha(ARCH) == ARCH_SHA, 'canonical architecture source changed')
    # Reuse all old heavy identity audits. Recheck only dependencies exercised
    # by the new entry gate and the newly consulted numerical closure reports.
    closure_seal = seal_rows(CLOSURE / 'FINAL_SHA256SUMS')
    consulted = [
        'FINAL_REPORT.md', 'provenance/final_verdict.json',
        'provenance/operator_parity.json', 'provenance/chunk_dtype_revalidation.json',
        'provenance/full_model_probe_parity.json',
        'provenance/fresh_process_determinism.json',
        'provenance/original_source_identity.json',
        'provenance/patched_source_identity.json',
        'protocol/numerical_parity_policy.json',
        'runtime_patch/patched_gdn.py', 'runtime_patch/deterministic_cumsum.py',
    ]
    checks = []
    for rel in consulted:
        actual = sha(CLOSURE / rel)
        check(rel in closure_seal and actual == closure_seal[rel], 'historical closure sealed dependency drift: ' + rel)
        checks.append({'path': str(CLOSURE / rel), 'sha256': actual, 'status': 'PASS'})
    full = read(CLOSURE / 'provenance/full_model_probe_parity.json')
    breakdown = {}
    for name in ['historical32', 'dev256']:
        rows = [x for x in full['rows'] if x['probe'] == name]
        logits = next(x for x in rows if x['object'] == 'logits')
        breakdown[name] = {'objects': len(rows), 'pass': sum(x['PASS'] for x in rows), 'logits': logits}
    core = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
    check(sha(core) == '48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e', 'C5/C6 helper changed')
    source = core.read_text()
    fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'collect_teacher_targets')
    doc_manifest = read(Path(roots['PARENT_HISTORY']) / 'document_manifest.json')
    rs = resource()
    audit = {
        'task': TASK, 'utc': utc(), 'status': 'SOURCE_PREFLIGHT_COMPLETE_RUNTIME_GATE_PENDING',
        'V1_audit_reused': True, 'V1_audit_path': str(V1 / 'ASSET_AND_PROVENANCE_AUDIT.json'),
        'V1_audit_sha256': sha(V1 / 'ASSET_AND_PROVENANCE_AUDIT.json'),
        'V1_asset_checks_reused': len(v['audited_checks']),
        'V1_Direction_POSTFINAL_checks_reused': v['direction_postfinal_manifest'],
        'heavy_model_or_archive_rehash_repeated': False,
        'source_roots': roots, 'formal_drives': v['formal_drives'],
        'student_tensor_files': v['student_tensor_files'],
        'doc1_engineering_2048': v['doc1_engineering_2048'],
        'parent_G_definition': v['parent_G_definition'],
        'parent_natural_labels_reused_as_historical_context_only': v['natural_parent_labels'],
        'historical_G_new_short_score_pairing_authorized': False,
        'document_manifest': doc_manifest, 'rotation_manifest': v['rotation_manifest'],
        'original_helpers': v['original_helpers'],
        'canonical_source_identity': {'path': str(ARCH), 'sha256': ARCH_SHA},
        'original_teacher_capture_source': ast.get_source_segment(source, fn),
        'original_teacher_prefix_semantics': 'First forward contains ids[0:first_capture_position]; original multi-token prefill must not silently become single-token teacher stepping.',
        'closure_selected_identity_checks': checks,
        'closure_overall_verdict': read(CLOSURE / 'provenance/final_verdict.json')['verdict'],
        'closure_full_model_breakdown': breakdown,
        'closure_fresh_process_determinism': read(CLOSURE / 'provenance/fresh_process_determinism.json'),
        'fixed_order_replacement_installed': False,
        'scoped_historical32_PASS_not_general_full_model_PASS': True,
        'engineering_fixture': {'directory': str(fixture), 'first_shard': str(fixture / shard['relative_path']),
                                'first_shard_sha256': shard['file_sha256'], 'document_id': fm['doc_id'],
                                'source_runtime': fm['runtime'], 'scientific_panel_unit': False},
        'runtime_resource_snapshot': rs,
    }
    save('SOURCE_AND_ARTIFACT_AUDIT.json', audit)
    save('RESOURCE_POLICY_V2.json', {
        'FIXED_GPU_HOURS_CAP': None, 'FIXED_GPU_HOURS_CAP_TEXT': 'NONE',
        'arbitrary_VRAM_percentage_cap': None, 'runtime_snapshot': rs,
        'safety_policy': ['Check actual free VRAM and disk before launch', 'Do not evict other task processes',
                          'Stop on OOM, uncontrolled growth, numerical/determinism/identity/pairing failure',
                          'No automatic scientific retry or implicit repair loop', 'Never delete old research assets'],
        'initial_entry_gate_expected_scratch_bytes': 16 * 1024 ** 2,
        'drive_retention_design_if_qualified': 'Bound one active document drive, seal and retain small results; no release without a frozen explicit new-artifact release manifest.',
        'physical_disk_capacity_not_GPU_hours_is_a_gate': True,
    })
    save('FROZEN_SURROGATE_SPEC.json', {
        'status': 'USER_DEFINITIONS_RECORDED_PRE_SCIENCE; EXECUTION_NOT_QUALIFIED',
        'measurement_authorized': False, 'independent_documents': 4,
        'documents_from_manifest': doc_manifest, 'roster': ['H', 'C6R', 'C7'], 'negative_control': 'Native',
        'state_shape': [1, 32, 128, 128], 'scale_shape': [1, 32, 1, 128], 'mask': 'all scale elements; no selection',
        'S1': 'mean_token_layer_scale_element(relu(log(s_student/s_clean))**2)',
        'D1': 'mean_token_layer_scale_element(relu(-log(s_student/s_clean))**2)',
        'D2': 'mean_token_layer_scale_element(log(s_student/s_clean)**2)',
        'S2': 'mean_token_layer(||formal_QDQ(u;s_student)-formal_QDQ(u;min(s_student,s_clean))||_F**2/(||u||_F**2+1e-12))',
        'S2_epsilon': 1e-12, 'S2_epsilon_design_note': 'New per-layer observational Frobenius ratio; epsilon fixed now, not a claim that it is original B0 denominator semantics.',
        'S2_observer': 'read-only clones; formal AST assignments; no recurrent writeback; no extra model call',
        'quantizer': 'frozen formal quantize_c128, round=torch.round, range=-127..127, zero_point=0, amax(dim=-2,keepdim=True).clamp_min(1e-12)/127',
        'B0': 'unchanged original relative_state_mse: per-head matrix ratio then head/layer mean, denominator clamp_min(1e-12)',
        'B1': 'actual out_proj output versus matched teacher output; unchanged original global relative_mse, then layer mean',
        'B2': 'B1+0.1*B0', 'scale_clipping_saturation': 'both original and cap grids, separately',
        'Track_A': {'anchor_consumed_1_based': 8192, 'H32_0_based': [8192, 8224], 'H64_0_based': [8192, 8256],
                    'H32_1_based_inclusive': [8193, 8224], 'H64_1_based_inclusive': [8193, 8256]},
        'Track_B': {'geometry_tokens': 1024, 'capture_positions': 'original sampled_positions helper with each frozen document raw_text_sha256; eight captures',
                    'teacher_forward': 'original collect_teacher_targets; no silent change of chunked prefill',
                    'H32_H64_window_execution_spec': 'NOT_IMPLEMENTED_PENDING_RUNTIME_GATE', '8192_warmup': False},
        'G': v['parent_G_definition'],
        'future_only_secondary_label': 'NOT_FROZEN_NOT_RUN; requires legal matched 8192/8224/8256/32768 cumulative boundaries before long-label execution',
        'statistical_unit': 'document, n=4 Discovery only', 'pre_onset_qualified': False,
        'gradient_or_KL_pilot_authorized': False,
    })
    prose('RECOVERY_PLAN.md', f'''# {TASK} — planned before execution

1. Reuse V1's {len(v['audited_checks'])} identity checks and Direction's sealed 269-entry audit. Recheck only newly exercised source/fixture/closure dependencies. No new model rehash or old-panel rerun.
2. Verify four GPU UUIDs and physical resources. No fixed GPUh cap or 85% VRAM rule.
3. First unresolved engineering gate: deterministic canonical capture entry. Use the real frozen Doc1 first-token, first-layer drive inputs on GPU0, unchanged production Torch chunk helper, deterministic algorithms ON, warn_only OFF, TF32 OFF, CUBLAS_WORKSPACE_CONFIG=:4096:8. This is an input/operator qualification, not a surrogate score or H/C6R ranking unit.
4. The existing fixed-order closure has a real full-model FAIL on multi-token prefill. Its single-token historical32 subset PASS is not a general authorization. Do not install that replacement in the original C5/C6 teacher collector or bypass its chunk-to-position forward semantics. No CUDA/Torch/FLA switch or new scan implementation.
5. If the canonical entry is unsupported or any numerical/scientific gate fails: preserve the first new error/traceback, STOP, do not start student H/C6R, and report all downstream gates NOT_RUN. No automatic retry.
6. Only if the entry and legally scoped deterministic source qualification PASS, implement the one-Doc1 H+C6R reconstruction/parity qualification (fresh processes, every-token writeback/next-token consumption, observer non-mutation, exact index checks). Then Track B with original matched B0/B1/B2 and true forwards.
7. Following all engineering gates, freeze the full four-document Track A roster and H32/H64 observer implementation, reconstruct to 8256, require bytewise historical trajectory identity or a preregistered legitimate equivalence basis before pairing old G. Matching cumulative J alone is insufficient.
8. PATH B is allowed only after an independently valid deterministic capture/teacher runtime is qualified. If required, freeze new matched long G and boundary accumulators BEFORE execution. Old G cannot be mixed into new numerical trajectories without qualification.
9. Complete all qualified Discovery units, describe n=4 ranking effects and all discordant pairs without outcome-based selection, seal actual artifacts. Do not generate empty score CSVs for unexecuted stages.

Historical deterministic-capture=False/replay=True is preserved as evidence. This V2 does not reinterpret any old COMPLETE/PASS/FAIL. No training, optimizer, gradient, C8, new rotation, AIME, free generation, or independent KL pilot.
''')
    save('QUALIFICATION_PREREGISTRATION.json', {
        'utc': utc(), 'gate_order': ['source_resource_preflight', 'canonical_deterministic_capture_entry',
                                   'Doc1_H_C6R_reconstruction_parity', 'Track_B_matched_helpers', 'Track_A', 'qualified_long_label_and_analysis'],
        'probe': 'unchanged production torch_chunk_gated_delta_rule on Doc1 token1 layer0, before any new model load or quantization',
        'source': str(ARCH), 'source_sha256': ARCH_SHA, 'fixture': audit['engineering_fixture'],
        'physical_gpu': 0, 'expected_GPU_UUID': UUID0, 'deterministic': True, 'deterministic_warn_only': False,
        'TF32': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'same_runtime': {'torch': '2.5.1+cu121', 'cuda': '12.1'},
        'failure_rule': 'Any source, numerical, deterministic or unsupported-operation failure seals first error and stops; no replacement or retry.',
        'historical_replacement_full_model_FAIL_preserved': True, 'replacement_scan_allowed_for_this_unqualified_panel': False,
    })
    print(json.dumps({'phase': 'PREPARED', 'root': str(ROOT), 'resource': rs}, ensure_ascii=False), flush=True)


def probe():
    a = read(ROOT / 'QUALIFICATION_PREREGISTRATION.json')
    check(command(['git', 'status', '--porcelain'], REPO) == '', 'qualification must start from a clean frozen worktree')
    commit = command(['git', 'rev-parse', 'HEAD'], REPO)
    receipt = {'started_utc': utc(), 'pid': os.getpid(), 'source_commit': commit,
               'preregistration_sha256': sha(ROOT / 'QUALIFICATION_PREREGISTRATION.json'),
               'scientific_measurements': 0, 'new_full_model_forwards': 0,
               'persistent_student_QDQ_calls': 0, 'source_modified': False,
               'replacement_scan_installed': False, 'retry': False}
    save('execution/STARTED.json', receipt)
    started = time.monotonic()
    torch = None
    op_started = None
    gate = 'FAIL'
    error = None
    try:
        check(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'fixed GPU0 gate mapping')
        check(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS runtime plumbing')
        rs = resource()
        check(not rs['compute_processes_csv'], 'another GPU task exists; do not compete')
        check(rs['disk_available_bytes'] >= read(ROOT / 'RESOURCE_POLICY_V2.json')['initial_entry_gate_expected_scratch_bytes'], 'physical storage insufficient')
        check(UUID0 in rs['gpu_inventory_csv'].splitlines()[0], 'GPU0 UUID identity')
        check(sha(ARCH) == a['source_sha256'], 'canonical source changed after freeze')
        f = a['fixture']
        check(sha(f['first_shard']) == f['first_shard_sha256'], 'fixture changed after freeze')
        import torch as torch_module
        torch = torch_module
        check(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'runtime identity drift')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        check(torch.are_deterministic_algorithms_enabled(), 'deterministic algorithms not enabled')
        check(not torch.is_deterministic_algorithms_warn_only_enabled(), 'warn-only mode forbidden')
        sys.path.insert(0, '/data/zypan/transformers-qwen35/src')
        qmod = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
        check(Path(qmod.__file__).resolve() == ARCH.resolve(), 'production module path mismatch')
        fn = qmod.torch_chunk_gated_delta_rule
        # Call the actual production binding, not a copied formula or scan.
        production = inspect.unwrap(fn)
        check(Path(inspect.getsourcefile(production)).resolve() == ARCH.resolve(), 'Torch production helper source mismatch')
        payload = torch.load(f['first_shard'], map_location='cpu', weights_only=True)
        check(payload['first_token'] == 1 and '0' in payload['layers'], 'real Doc1 first-token/layer coverage')
        fields = payload['layers']['0']
        inputs = {n: fields[n][:, :1].contiguous().to('cuda:0') for n in ['query', 'key', 'value', 'g', 'beta']}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        op_started = time.monotonic()
        receipt['input_tensors'] = {n: {'shape': list(x.shape), 'dtype': str(x.dtype),
                                      'sha256': hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                                    for n, x in inputs.items()}
        with torch.inference_mode():
            result, state = fn(**inputs, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        check(bool(torch.isfinite(result).all() and torch.isfinite(state).all()), 'nonfinite original production result')
        gate = 'PASS'
        receipt['original_chunk_result_shape'] = list(result.shape)
        receipt['original_chunk_state_shape'] = list(state.shape)
        receipt['warning'] = 'Entry support PASS alone does not qualify full capture, repeatability, H/C6R or Track B.'
    except Exception as exc:
        error = {'gate': 'CANONICAL_DETERMINISTIC_CAPTURE_ENTRY', 'status': 'FAIL_AND_STOP',
                 'exception_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                 'utc': utc(), 'retry_authorized': False, 'scientific_measurement_count': 0,
                 'replacement_scan_installed': False, 'old_evidence_unchanged': True}
        save('execution/FIRST_FAILURE.json', error)
    finally:
        total_elapsed = time.monotonic() - started
        receipt['finished_utc'] = utc()
        receipt['status'] = gate
        receipt['wall_seconds_including_imports'] = total_elapsed
        receipt['GPU_process_wall_seconds_upper_bound'] = total_elapsed
        receipt['GPU_hours_upper_bound'] = total_elapsed / 3600
        receipt['operator_elapsed_seconds'] = time.monotonic() - op_started if op_started is not None else 0
        receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else 0
        receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch is not None and torch.cuda.is_initialized() else 0
        receipt['new_full_model_tokens_per_second'] = None
        receipt['successful_original_chunk_calls'] = int(gate == 'PASS')
        receipt['runtime'] = {
            'torch': torch.__version__ if torch is not None else None,
            'cuda': torch.version.cuda if torch is not None else None,
            'deterministic': torch.are_deterministic_algorithms_enabled() if torch is not None else None,
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled() if torch is not None else None,
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32 if torch is not None else None,
            'TF32_cudnn': torch.backends.cudnn.allow_tf32 if torch is not None else None,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'GPU_UUID': UUID0,
        }
        save('execution/FINISHED.json', receipt)
        save('DETERMINISM_AND_CUMSUM_GATE.json', {
            'status': gate, 'failure': error, 'receipt': receipt,
            'historical_capture_deterministic': False, 'historical_replay_deterministic': True,
            'old_fixed_order_full_model_parity': 'FAIL', 'old_fixed_order_fresh_process_determinism': 'NOT_EVALUATED',
            'fixed_order_replacement_used': False, 'no_new_patch_or_runtime_version_change': True,
            'single_token_old_partial_PASS_is_not_full_panel_qualification': True,
        })
        print(json.dumps({'gate': gate, 'error': error['message'] if error else None, 'receipt': receipt}, ensure_ascii=False), flush=True)
    return 0 if gate == 'PASS' else 2


def finalize():
    receipt = read(ROOT / 'execution/FINISHED.json')
    check(receipt['status'] == 'FAIL', 'Cannot finalize stopped science when canonical entry passed; continue unfinished gates instead')
    error = read(ROOT / 'execution/FIRST_FAILURE.json')
    audit = read(ROOT / 'SOURCE_AND_ARTIFACT_AUDIT.json')
    reasons = {'status': 'NOT_RUN', 'blocked_by': 'DETERMINISM_AND_CUMSUM_GATE=FAIL',
               'first_failure_sha256': sha(ROOT / 'execution/FIRST_FAILURE.json'),
               'no_scientific_retry': True, 'scientific_measurements': 0}
    for name in ['ANCHOR_IDENTITY_GATE.json', 'FP_STUDENT_PAIRING_GATE.json',
                 'PERSISTENT_QDQ_PARITY_GATE.json', 'QUANTIZER_HOOK_PARITY_GATE.json']:
        save(name, dict(reasons, gate=name.removesuffix('.json'), historical_V1_identity_PASS_reused=True,
                        new_panel_qualification_PASS_claimed=False))
    save('CAPTURE_RECONSTRUCTION_REPORT.json', dict(reasons,
         PATH_A='NOT_QUALIFIED', PATH_B='NOT_STARTED_UNQUALIFIED_RUNTIME',
         historical_drive_shards_released=True, historical_student_8192_tensor_anchors_released=True,
         actual_reused_input='Doc1 first64-token shard from preserved 2048-token engineering fixture, only token1 layer0 exercised',
         reused_input_not_8192_anchor=True, new_surrogate_scores_produced=False,
         engineering_Doc1_H_C6R='NOT_RUN_TEACHER_ENTRY_GATE_FAILED'))
    rs = resource()
    artifact_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    save('GPU_COST_AND_MEMORY_LEDGER.json', {
        'status': 'ENGINEERING_GATE_EXECUTED_AND_STOPPED', 'qualification_receipts': [receipt],
        'fixed_gpu_hours_cap': None, 'science_GPU_hours': 0,
        'GPU_process_wall_hours_upper_bound': receipt['GPU_hours_upper_bound'],
        'wall_seconds': receipt['wall_seconds_including_imports'],
        'peak_allocated_bytes': receipt['peak_allocated_bytes'], 'peak_reserved_bytes': receipt['peak_reserved_bytes'],
        'full_model_throughput_tokens_per_second': None, 'not_applicable_reason': 'No full-model forward or scientific token measured',
        'temporary_drive_bytes_created': 0, 'pre_finalization_artifact_bytes': artifact_bytes,
        'post_gate_resources': rs, 'old_assets_deleted': 0,
    })
    save('FINAL_STATUS.json', {
        'task': TASK, 'PRIMARY_STATUS': 'ENGINEERING_OR_NUMERICAL_STOP', 'FINAL_ACTION': 'FAIL_AND_STOP',
        'FIRST_FAILED_GATE': 'CANONICAL_DETERMINISTIC_CAPTURE_ENTRY',
        'failure': error['message'], 'input_recovery_qualified': False,
        'Track_A': 'NOT_RUN', 'Track_B': 'NOT_RUN', 'S1_S2': 'NOT_MEASURED',
        'scientific_units_executed': 0, 'qualification_processes': 1,
        'H_C6R_full_reconstruction_started': False, 'new_long_G_units': 0,
        'training': False, 'rotation_optimization': False, 'C8': False,
        'AIME': False, 'free_generation': False, 'gradient_optimization': False, 'KL_pilot': False,
        'budget_stop': False, 'physical_disk_stop': False, 'arbitrary_VRAM_cap_stop': False,
        'old_experiment_statuses_modified': False, 'scientific_CSVs_created': [],
        'preflight_and_gate_commit': receipt['source_commit'],
    })
    prose('FINAL_SCIENTIFIC_REPORT.md', f'''# {TASK}

## 结论

PRIMARY_STATUS = ENGINEERING_OR_NUMERICAL_STOP；FINAL_ACTION = FAIL_AND_STOP。
第一项新运行门禁 CANONICAL_DETERMINISTIC_CAPTURE_ENTRY = FAIL。
V2已取消4 GPUh硬上限，没有85%显存门槛；本次停止与预算、磁盘不足均无关。

实际错误：`{error['message']}`。
完整原始 traceback 保存在 execution/FIRST_FAILURE.json；没有重试、关闭 deterministic、安装替代 scan 或修改科学配置。

## 1. V1缺少哪些输入？

四文档正式32768-token drive与8192学生锚点张量已按历史release释放。剩余SHA记录不是张量。clean FP端点、旧连续G及其原始记录、Doc1工程256/2048 drive仍存在。自然Parent的逐token paired scales、out_proj matched teacher输出和精确first-onset不在可复用资产中。

## 2. 找到可复用历史anchor/drive了吗？

复用V1的{audit['V1_asset_checks_reused']}项身份PASS和Direction269项审计；只新增检查本次使用的源码、真实Doc1工程首shard和历史cumsum闭环封存文件。Doc1工程2048可供资格测试，不是8192-anchor或8192后短窗。没有找到新的正式8192学生锚点/drive备份。

## 3. 如何处理历史确定性差异？

历史capture deterministic=False、replay=True明确保留。新进程 deterministic=True、warn_only=False、TF32=False、CUBLAS_WORKSPACE_CONFIG=:4096:8，Torch2.5.1+cu121/CUDA12.1未切换。对真实Doc1 token1/layer0输入调用原生产Torch chunk绑定；原CUDA cumsum入口失败。
历史fixed-order替代曾通过operator/chunk，single-token historical32有145/145对象字节一致，但multi-token dev256只有12/145对象PASS；logits max_abs=0.375、relative_L2=0.009980417299255454，全模型资格FAIL，fresh-process determinism未执行。不能把局部PASS转成整体PASS或直接用于原C5/C6多token teacher collector。

## 4. 历史输入重建是否合格？

否。原生产deterministic入口已失败，尚未进入H/C6R重建、8192累计J或状态身份比较。单个端点J接近不能证明整个旧trajectory等价。没有将新短程信号与旧G正式配对。

## 5. 新匹配面板是否建立？

NOT_STARTED。PATH B也需要科学资格合格的deterministic capture和真实teacher forward。不能借新面板名义绕过同一未合格源码/数值门禁。

## 6. Track B是否完成？

NOT_RUN。原helper collect_teacher_targets使用chunk-to-position teacher forward，原训练8-capture、1024-token语义不得静默改为逐token FP teacher。没有用WikiText targets冒充RepoBench配对，也没有用q/k/v驱动近似B1。

## 7. Track A是否完成？

NOT_RUN。8192-anchor之后的H32/H64窗口索引已明确记录：0-based [8192,8224)、[8192,8256)，对应1-based8193..8224、8193..8256。没有人工reset、学生轨迹或短窗测量。

## 8. 真实S1/S2分数？

H32/H64 S1/S2/D1/D2/B0/B1/B2全部NOT_MEASURED。定义和epsilon=1e-12记录在FROZEN_SURROGATE_SPEC.json，但执行资格未建立；没有空壳TRACK_A/B CSV，没有编造排名或相关性。

## 9. 自然旋转能区分吗？

新surrogate未测，不能回答。旧G已有Doc1/2 H<C6R<C7；Doc3 C6R<C7<H；Doc4 H<C7<C6R。这仅是历史Discovery背景，不是新预测成功。

## 10. S1能预测合法长期G吗？

NOT_EVALUATED。旧G定义保持1024..T各token与24层等权FP64累计，G=J32768/J8192；没有新matched G。短窗与旧G numerator重叠，不能宣称严格未来预测。

## 11. S1相对原C6/B2有增量吗？

NOT_EVALUATED；不能报有增量，也不能报NO_ELIGIBLE_INCREMENTAL_DISCORDANCE。B0仍是per-head matrix relative-MSE且分母clamp_min；B1必须是真实out_proj/teacher global relative-MSE；B2=B1+0.1B0。没有更改这些定义。

## 12. pre-onset资格？

NO。没有自然精确first-onset和新合法future-only边界观测。

## 13. 实际资源消耗？

新增正式science GPUh=0。执行一次GPU0工程入口检查，进程wall-clock（含导入）{receipt['wall_seconds_including_imports']:.6f}秒；保守GPU-process wall-hours上界{receipt['GPU_hours_upper_bound']:.9f}，不把它伪装成CUDA kernel activity精确积分。
峰值 allocated={receipt['peak_allocated_bytes']} bytes，reserved={receipt['peak_reserved_bytes']} bytes；没有加载9B模型、正式token吞吐或大型临时drive，新drive=0 bytes。封存前小型产物{artifact_bytes} bytes，结束/data可用{rs['disk_available_bytes']} bytes。最终交付字节数见RESULT_AND_DELIVERY_HASHES.json。

## 14. 是否值得进入独立full-model persistent KL？

本轮没有新科学依据支持进入，且用户未授权该pilot，因此NOT_RUN。
下一步所需的是单独解决并资格认证原multi-token teacher/capture的deterministic numerical closure（源码、parity和fresh-process determinism均须合格），或用户另行明确批准一种新的确定性前向语义与独立matched面板资格协议。不能在本轮静默替换teacher forward、放宽旧caps、修改quantizer或重新解释旧FAIL。恢复后才继续本V2尚未完成的H/C6R最小重建与Track B/A Discovery。

## 复现和身份

Gate source commit：{receipt['source_commit']}。
仅在独立新输出目录中执行本脚本prepare → git freeze → probe；原目录write-once，不能重复probe。
环境：CUDA_VISIBLE_DEVICES=0，CUBLAS_WORKSPACE_CONFIG=:4096:8；Python /data/ydai/miniconda3/envs/bitdecode/bin/python。
GPU0 UUID：{UUID0}。
旧source/checkpoint/rotation/文档及旧COMPLETE/PASS/FAIL均未改写。科学单位0，训练0，AIME0，自由生成0。
''')
    files = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(ROOT.rglob('*')) if p.is_file()]
    save('RESULT_AND_DELIVERY_HASHES.json', {
        'utc': utc(), 'result_files': files, 'data_bytes_before_this_index': sum(x['bytes'] for x in files),
        'self_and_SHA256SUMS_excluded_to_avoid_circular_hash': True,
        'old_V1_manifest_sha256': sha(V1 / 'SHA256SUMS.txt'),
        'old_cumsum_seal_sha256': sha(CLOSURE / 'FINAL_SHA256SUMS'),
        'V2_code_sha256': sha(Path(__file__)), 'qualification_commit': receipt['source_commit'],
        'no_unexecuted_scientific_CSV': True,
    })
    all_files = sorted(p for p in ROOT.rglob('*') if p.is_file())
    prose('SHA256SUMS.txt', ''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in all_files))
    rows = seal_rows(ROOT / 'SHA256SUMS.txt')
    check(all(sha(ROOT / rel) == digest for rel, digest in rows.items()), 'final manifest verification')
    print(json.dumps({'status': 'FAIL_AND_STOP', 'report': str(ROOT / 'FINAL_SCIENTIFIC_REPORT.md'),
                      'manifest_sha256': sha(ROOT / 'SHA256SUMS.txt'), 'files_verified': len(rows),
                      'delivery_bytes': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'probe', 'finalize'])
    args = parser.parse_args()
    if args.phase == 'probe':
        sys.exit(probe())
    elif args.phase == 'prepare':
        prepare()
    else:
        finalize()
