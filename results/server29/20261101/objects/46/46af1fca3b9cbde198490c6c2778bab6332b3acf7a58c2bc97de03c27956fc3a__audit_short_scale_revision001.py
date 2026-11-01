#!/usr/bin/env python3
"""CPU-only, write-once feasibility audit. Never imports production runners.

Old research assets are only read. Output is confined to a new audit directory.
No CUDA context, model forward, recurrence, quantization, or scientific score run.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone

TASK = 'QWEN_GDN_SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION_V1'
CONDITIONS = ['Native', 'H', 'C6R', 'C7']
FORMAL_SHA = '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
POSTFINAL_SHA = 'c7e1b74de7dc11099b198c5d7ae52767857dc3f20fcd1d4f7a23b6b0542d6856'
OLD_COMMIT = '9e302aa46cbbc0f621eb792c1af5429abb6acb27'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())


def identity(p, expected=None):
    p = Path(p)
    if not p.is_file():
        return dict(path=str(p), status='MISSING', expected_sha256=expected)
    s = p.stat()
    h = sha(p)
    unchanged = (s.st_size, s.st_mtime_ns) == (p.stat().st_size, p.stat().st_mtime_ns)
    return dict(path=str(p), bytes=s.st_size, sha256=h, expected_sha256=expected,
                unchanged_during_read=unchanged,
                status='PASS' if unchanged and (expected is None or h == expected) else 'FAIL')


def constants(source):
    """Evaluate only literal/Path/join assignments, without importing old code."""
    values = {}

    def evaluate(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.Name): return values[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Path':
            return Path(evaluate(n.args[0]))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return evaluate(n.left) / evaluate(n.right)
        return ast.literal_eval(n)

    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try: values[n.targets[0].id] = evaluate(n.value)
            except (ValueError, KeyError, TypeError): pass
    return values


def source_section(p, name):
    text = Path(p).read_text()
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    s = ast.get_source_segment(text, node)
    return dict(path=str(p), file_sha256=sha(p), name=name, first_line=node.lineno,
                last_line=node.end_lineno, source_sha256=hashlib.sha256(s.encode()).hexdigest(), source=s)


def manifest_check(root, name):
    p = root / name
    if not p.is_file(): return dict(path=str(p), status='MISSING')
    rows = []
    for line in p.read_text().splitlines():
        h, rel = line.split(None, 1)
        f = Path(rel.strip()) if Path(rel.strip()).is_absolute() else root / rel.strip()
        rows.append(identity(f, h))
    return dict(path=str(p), sha256=sha(p), files=len(rows),
                matched=sum(r['status'] == 'PASS' for r in rows),
                status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'REQUIRES_ROW_LEVEL_INTERPRETATION',
                nonmatching=[r for r in rows if r['status'] != 'PASS'], rows=rows)


def save(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    assert out.name == TASK
    assert not out.exists(), 'Write-once audit output already exists; do not overwrite.'
    started, t0 = utc(), time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    base = Path('/data/zypan/worktrees')
    found = sorted(base.glob('*/experiments/QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1'))
    assert len(found) == 1, 'Ambiguous direction experiment discovery.'
    direction = found[0]
    C = constants((direction / 'scripts/common.py').read_text())
    assert C['EXP'] == direction
    parent, proot, clean = C['PARENT'], C['PARENT_ROOT'], C['CLEAN_EXP']
    capture_policy = read(parent / 'protocol/capture_engineering_preregistration.json')
    policy = read(parent / 'protocol/remaining_engineering_gate_policy.json')
    extended = read(parent / 'protocol/extended_capture_policy.json')
    frozen_anchors = read(proot / 'protocol/frozen_anchor_conditions.json')
    formal = C['FORMAL']
    src = formal.parents[4]
    # Anchor manifest supplies the canonical model/checkpoint bindings.
    model_checks = [identity(r['path'], r['sha256']) for r in capture_policy['model_identity']['files']]
    source_checks = [identity(p, h) for p, h in capture_policy['source_sha256'].items()]
    formal_check = identity(formal, FORMAL_SHA)
    manifest_check_direction = manifest_check(direction, 'POSTFINAL_SHA256SUMS')
    postfinal_check = identity(direction / 'POSTFINAL_SHA256SUMS', POSTFINAL_SHA)
    commit_check = command(['git', '-C', str(C['REPO']), 'cat-file', '-t', OLD_COMMIT])
    git_before = command(['git', '-C', str(C['REPO']), 'status', '--porcelain', '--untracked-files=all'])
    # Derive original objective paths/hashes from the archived matched inventory.
    matched = sorted(base.glob('*/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/provenance/preclosure_training_artifact_inventory.json'))
    inventories = [(p, read(p)) for p in matched]
    assert inventories, 'Matched training inventory missing.'
    original = next((r for _, j in inventories for r in j['files']
                     if r['path'].endswith('/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')), None)
    assert original is not None
    core = src / 'experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
    assert core.is_file(), str(core)
    helper_check = identity(core, original['sha256'])
    corpus_record = next(r for _, j in inventories for r in j['files'] if r['path'].endswith('/CALIBRATION_RAW_TEXTS.jsonl'))
    corpus_check = identity(src / 'CALIBRATION_RAW_TEXTS.jsonl', corpus_record['sha256'])
    cayley_record = next(r for _, j in inventories for r in j['files'] if r['path'].endswith('/cayley_rotation.py'))
    cayley = src / 'experiments/shared/rotation/cayley_rotation.py'
    cayley_check = identity(cayley, cayley_record['sha256'])
    matched_engine = inventories[0][0].parents[1] / 'scripts/a800_matched.py'
    engine_check = identity(matched_engine, '0f85e40100070c051a01d05125341c1863b5c9808247803a725b1b218f54412d')
    sections = [source_section(core, n) for n in ['sampled_positions', 'relative_state_mse', 'captured_losses', 'detach_cache']]
    sections += [source_section(cayley, 'relative_mse'), source_section(formal, 'quantize_c128'), source_section(matched_engine, 'train_document')]
    # No training function is imported or executed.
    parent_hashes = {}
    parent_seal = parent / 'PARALLEL_FINAL_SHA256SUMS'
    for line in parent_seal.read_text().splitlines():
        h, rel = line.split(None, 1)
        parent_hashes[str((parent / rel.strip()).resolve())] = h
    parent_rows = []
    drives = []
    checkpoints = []
    for idx, doc in enumerate(extended['documents'], 1):
        assert doc['token_count'] == 32768
        token_file = Path(doc['token_file'])
        text_file = proot / f'inputs/discovery_{idx:02d}.txt'
        doc_checks = [identity(token_file, doc['token_file_sha256']), identity(text_file, doc['text_sha256'])]
        droot = Path(doc['output_directory'])
        j = read(droot / 'drive_manifest.json')
        present = sum((droot / r['relative_path']).is_file() for r in j['shards'])
        refs = [identity(droot / r['relative_path'], r['file_sha256']) for r in j['fp_reference_checkpoints']]
        checkpoints.extend(refs)
        release_paths = sorted(droot.glob('*release*.json'))
        drives.append(dict(document_index=idx, document=doc, document_checks=doc_checks,
                           manifest_identity=identity(droot / 'drive_manifest.json'),
                           original_shards=len(j['shards']), retained_shards=present,
                           historical_capture_seconds=j['elapsed_seconds'],
                           release_evidence=[identity(f) for f in release_paths],
                           fp_reference_checkpoints=refs))
        for condition in CONDITIONS:
            f = droot / 'results' / condition / 'replay_result.json'
            r = read(f)
            obs = r['identity']['outputs'][condition]
            a = obs['horizons']['8192']['normalized_mean'][0]
            z = obs['horizons']['32768']['normalized_mean'][0]
            g = z / a
            assert r['status'] == 'PASS' and a > 0 and all(math.isfinite(v) for v in [a, z, g])
            assert r['document_token_sha256'] == doc['token_file_sha256']
            assert r['source_sha256'][str(formal)] == FORMAL_SHA
            assert obs['horizons']['8192']['token_count'] == 7169
            assert obs['horizons']['32768']['token_count'] == 31745
            assert str(f.resolve()) in parent_hashes, 'Parent label not in historical seal.'
            parent_rows.append(dict(document_index=idx, condition=condition, J_ACC_FD_8192=a,
                                    J_ACC_FD_32768=z, G=g, source_identity=identity(f, parent_hashes[str(f.resolve())]),
                                    natural_parent=True, G_label_finite=True,
                                    intermediate_observation_endpoints=list(obs['horizons']),
                                    first_explosion_token='NOT_RETAINED_TOKEN_RESOLUTION',
                                    original_endpoint_stored_Q_hashes=obs['endpoints']['8192']))
    # Endpoint state hashes are not serialized student-state tensors.
    student_tensors = {p.name: [str(f) for f in p.rglob('*.pt') if 'fp_reference' not in f.parts]
                       for p in [direction, clean, C['PARENT_HISTORY']]}
    engineering = proot / 'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1/engineering/capture_2048'
    fixture = read(engineering / 'drive_manifest.json')
    fixture_checks = [identity(engineering / r['relative_path'], r['file_sha256']) for r in fixture['shards']]
    fixture_doc1 = fixture['doc_id'] == extended['documents'][0]['doc_id']
    # Existing bins are counterfactual, not a natural Parent scale label.
    scales = read(direction / 'analysis/signed_scale_trajectory.json')
    onset = read(direction / 'analysis/temporal_ordering.json')
    scale_arm_names = sorted(set(r['arm'] for r in scales))
    onset_arm_names = sorted(set(r['arm'] for r in onset))
    actual_capture_times = [read(direction / f'discovery/doc_{i:02d}/drive/drive_manifest.json')['elapsed_seconds'] for i in range(1, 5)]
    capture_only_gpuh = sum(actual_capture_times) / 32768 * 8256 / 3600
    reserve_gpuh = 1.25 * capture_only_gpuh
    all_checks = model_checks + source_checks + [formal_check, postfinal_check, helper_check, corpus_check, cayley_check, engine_check]
    rotation_checks = [identity(proot / 'protocol/frozen_anchor_conditions.json')]
    for a in frozen_anchors['anchors']:
        if 'export_path' in a:
            rotation_checks.append(identity(a['export_path'], a['export_file_sha256']))
            orig = a['original_provenance']
            rotation_checks.append(identity(orig['theta'], orig['theta_sha256']))
    all_checks += checkpoints + fixture_checks + rotation_checks + [x for d in drives for x in d['document_checks']]
    all_checks += [r['source_identity'] for r in parent_rows]
    identity_pass = all(r['status'] == 'PASS' for r in all_checks) and commit_check['stdout'] == 'commit' and manifest_check_direction['status'] == 'PASS'
    # Ranking support of retained G is an audit finding, not a surrogate result.
    h_rankings = {str(i): [r['condition'] for r in sorted((r for r in parent_rows if r['document_index'] == i and r['condition'] != 'Native'), key=lambda r: r['G'])] for i in range(1, 5)}
    retained_student = sum(len(v) for v in student_tensors.values())
    assert not retained_student, 'Unexpected retained student tensors: re-audit before choosing a stop.'
    assert all(d['retained_shards'] == 0 for d in drives), 'Unexpected retained formal drive: re-audit feasibility.'
    hardware = dict(gpus=command(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
                    processes=command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader']),
                    disk=command(['df', '-B1', '/data']), host=platform.node(),
                    python=platform.python_version())
    import torch  # CPU metadata only; CUDA devices were hidden before import.
    hardware.update(torch=torch.__version__, torch_cuda_build=torch.version.cuda,
                    cuda_context_initialized=torch.cuda.is_initialized(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    assert not hardware['cuda_context_initialized']
    # Identity construction on CPU calls the unchanged original hadamard source.
    formal_tree = ast.parse(formal.read_text())
    hadamard_node = next(n for n in formal_tree.body if isinstance(n,ast.FunctionDef) and n.name == 'hadamard')
    env = {'torch':torch, 'math':math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[hadamard_node],type_ignores=[])), str(formal), 'exec'),env)
    rotation_tensor_identity = []
    for a, value in zip(frozen_anchors['anchors'][:2], [torch.eye(128,dtype=torch.float32),env['hadamard'](128,device='cpu')]):
        h = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        rotation_tensor_identity.append(dict(condition=a['condition'], actual_sha256=h,
                                            expected_sha256=a['matrix_sha256'], status='PASS' if h == a['matrix_sha256'] else 'FAIL', device='cpu'))
    identity_pass = identity_pass and all(r['status']=='PASS' for r in rotation_tensor_identity)
    free = os.statvfs('/data').f_bavail * os.statvfs('/data').f_frsize
    historical_gates = {}
    for name, root in [('Direction', direction), ('CleanQParam', clean)]:
        q = read(root / 'qualification/qualification_report.json')
        historical_gates[name] = dict(status=q['status'], gates=q['gates'], source_identity=identity(root / 'qualification/qualification_report.json'))
    # Qualification status is historical only, never relabeled new PASS.
    stop = 'AUDIT_BLOCKED' if identity_pass else 'AUDIT_BLOCKED_IDENTITY'
    block_reasons = ['Natural Parent time-resolved adaptive scale/paired clean scale and pre-QDQ states not retained.',
                     'All four formal raw drives released; only hashes and FP endpoint tensor checkpoints remain.',
                     'All four-condition 8192 student anchors in direction/clean/history released; hashes are not recoverable tensors.',
                     'No token-resolution natural Parent first-onset or non-overlapping future target boundary at8256 retained.',
                     'Four-document prefix reacquisition historical estimate already exceeds4GPUh before replay/qualification.',
                     'Original B1 requires actual out_proj output/paired teacher output; q/k/v/g/beta-only drive cannot supply it.']
    finished = utc()
    git_after = command(['git', '-C', str(C['REPO']), 'status', '--porcelain', '--untracked-files=all'])
    assert git_before == git_after, 'Old checkout changed during read-only audit.'
    binding = dict(task=TASK, audit_started_utc=started, audit_finished_utc=finished,
                   audited_host='zypan@202.38.247.55:22', scientific_start_utc=None,
                   scientific_end_utc=None, premeasurement_commit='NOT_CREATED_P0_BLOCKED',
                   audit_code_sha256=sha(Path(__file__)), actual_gpu_hours=0.0,
                   scientific_units_executed=0, GPU_qualification_runs=0,
                   historical_premeasurement_commit=OLD_COMMIT, source_roots={k:str(C[k]) for k in ['EXP','PARENT_ROOT','PARENT','CLEAN_EXP','PARENT_HISTORY']})
    out.mkdir(parents=True, exist_ok=False)
    save(out, 'ASSET_AND_PROVENANCE_AUDIT.json', dict(binding, status=stop,
         identity_revalidation='PASS' if identity_pass else 'FAIL', audited_checks=all_checks,
         direction_postfinal_manifest=manifest_check_direction,
         discovery_method='Literal-only AST of discovered direction common.py; manifests/run records resolve documents, model and original training helper.',
         independent_documents=4, old_counterfactual_units=32,
         rotation_manifest=frozen_anchors, rotation_asset_checks=rotation_checks,
         Native_H_cpu_matrix_identity=rotation_tensor_identity,
         parent_label_seal_identity=identity(parent_seal),
         reusable_observer_sources=[source_section(direction/'scripts/direction_replay.py',n) for n in ['CanonicalScaleAdapter','SignedStreamingReducer']],
         observer_limitations='Historical reducer stores counterfactual histogram/bin summaries, not natural Parent r^2 mean or same-u S2. Reuse requires a new registered side-effect-free observer and parity; histogram medians are not exact S1.',
         parent_G_definition='Equal-layer mean of per-token ||SQ_native-FP_native||_F^2/(||FP_native||_F^2+1e-12), FP64 accumulation, inclusive1024..T. G=J_ACC_FD(32768)/J_ACC_FD(8192). Dimensionless.',
         natural_parent_labels=parent_rows, natural_label_ranking_status='FINITE_UNCLIPPED_RECORDED_G_RANKS_AVAILABLE',
         historical_discrimination_low_to_high_G=h_rankings,
         time_resolved_scale_available='COUNTERFACTUAL_BINS_ONLY_NOT_ELIGIBLE_PARENT',
         signed_scale_arms=scale_arm_names, onset_arms=onset_arm_names,
         formal_drives=drives, student_tensor_files=student_tensors,
         doc1_engineering_2048=dict(identity=identity(engineering/'drive_manifest.json'),
                                   same_doc1=fixture_doc1, shard_checks=fixture_checks,
                                   usable_for='Potential qualification/first1024 inputs ONLY; no8192 history or original out_proj targets.'),
         original_helpers=sections, historical_gates=historical_gates,
         persistent_semantics='Historical matched training used every-token QDQ/writeback, H64 TBPTT,1024tokens,8 captures,125updates/1000exposures. It did not lack recurrent feedback.',
         block_reasons=block_reasons, protected_checkout_unchanged=True))
    save(out, 'FROZEN_PREMEASUREMENT_SPEC.json', dict(binding,
         status='NOT_FROZEN_P0_BLOCKED', measurement_authorized=False,
         proposed_only_not_executed=True, user_metric_definitions={
             'S1':'mean(relu(log(s_student/s_clean))^2)',
             'S2':'mean(||QDQ(u;s_student)-QDQ(u;min(s_student,s_clean))||^2/(||u||^2+epsilon))',
             'D1':'mean(relu(-log(s_student/s_clean))^2)', 'D2':'mean(log(s_student/s_clean)^2)',
             'B0':'original relative_state_mse on physical student PRE-state vs matched teacher PRE-state; equal heads then24 layers; original capture aggregation',
             'B1':'original out_proj output global relative_mse; equal24layers; actual paired teacher output required',
             'B2':'original captured_losses functional+0.1state; never equate checkpoint C6R with C6_TOTAL'},
         epsilon_aggregation='NOT_FROZEN_NO_MEASUREMENT',
         candidate_roster_proposed=CONDITIONS, document_identities=[d['document'] for d in drives],
         track_A=dict(status='NOT_RUN', indexing_audited='Anchor is state AFTER consuming1-based8192; proposed0-based[8192,8224)/[8192,8256) consumes1-based8193..8224/8256.',
                      overlap='Historical G numerator includes short window; no8256 cumulative boundary retained to remove it exactly.', onset='Natural token-resolution onset unknown; no early-warning claim.'),
         track_B=dict(status='NOT_RUN', short_documents='Same four documents need prospective first1024 token matched capture registration; existing WikiText teacher caches cannot be paired with RepoBench G.',
                      semantics='Original sampled_positions hash seed;8 positions in[128,1024),detach each64steps; no8192 warmup; no backward.'),
         immutable_constraints=['No training','No gradients','No AIME','No free generation','No production quantizer changes','No alternative recurrence or cumsum closure','No counterfactual trajectory substituted for natural Parent'],
         analysis_proposed=['per-document Spearman','within-document concordance','H/C6R/C7-only','signed rank disagreements','pre-onset eligibility','finite/saturation audit','LODO descriptive; no confirmatory p-values'],
         S2_writeback='Forbidden; observation clone only; NOT_IMPLEMENTED_P0_BLOCKED'))
    save(out, 'QUANTIZER_PARITY_GATE.json', dict(binding, status='NOT_RUN_P0_BLOCKED',
         source_identity=formal_check, historical_source_identity_status='PASS' if formal_check['status']=='PASS' else 'FAIL',
         new_observer_parity='NOT_TESTED', same_u_S2_gate='NOT_TESTED', unchanged_production_quantizer=True))
    save(out, 'DETERMINISM_GATE.json', dict(binding, status='NOT_RUN_P0_BLOCKED',
         historical_gates=historical_gates,
         required_runtime=dict(deterministic=True, TF32=False, CUBLAS_WORKSPACE_CONFIG=':4096:8'),
         old_capture_runtime='Historical production input acquisition used deterministic_enforcement=False; replay used True. No new acquisition change or torch.cumsum substitution authorized/executed.',
         current_hardware=hardware))
    save(out, 'FP_PAIRING_GATE.json', dict(binding, status='NOT_RUN_P0_BLOCKED',
         FP_endpoint_identity_checks=checkpoints, missing='Same-token pre-QDQ paired clean/true student tensors and functional teacher targets forH32/H64.',
         historical_pairing_not_sufficient_for_new_observer=True))
    shard_bytes = read(direction/'discovery/doc_01/drive/drive_manifest.json')['shards'][0].get('file_size_bytes',38077314)
    sequential_8256_drive_bytes = math.ceil(8256/64)*shard_bytes
    save(out, 'RESOURCE_AND_DISK_BUDGET.json', dict(binding, status='BUDGET_STOP',
         available_data_bytes=free, current_hardware=hardware,
         GPU_cap_hours=4.0, prefix_capture_only_historical_estimate_hours=capture_only_gpuh,
         prefix_capture_plus25percent_estimate_hours=reserve_gpuh,
         basis=dict(capture_seconds_for_four32768_documents=actual_capture_times, tokens_per_document=32768, required_prefix_tokens=8256),
         estimate_not_new_benchmark=True, excludes=['student anchor rematerialization','S1/S2 instrumentation qualification','determinism processes','TrackB','original functional baseline reconstruction'],
         sequential_prefix_drive_estimate_bytes=sequential_8256_drive_bytes,
         disk_status='AVAILABLE_NOT_THE_PRIMARY_BLOCKER',
         no_new_model_runtime_memory_measurement=True, old2048_capture_peak_allocated=fixture['max_gpu_allocated_bytes'],
         effective_peak_storage_budget='NOT_QUALIFIED_NO_GPU_LAUNCH', reason='Cannot plan valid four-document TrackA/TrackB discovery inside4GPUh using available verified inputs. Do not silently reduce documents/windows or raise budget.'))
    save(out, 'GPU_COST_LEDGER.json', dict(binding, status='CPU_AUDIT_ONLY_STOP',
         GPU_cap_hours=4.0, qualification_gpu_hours=0.0, science_gpu_hours=0.0,
         remaining_gpu_hours=4.0, executed_gpu_runs=[], CPU_audit_elapsed_seconds=time.monotonic()-t0))
    save(out, 'DISCOVERY_VERDICT.json', dict(binding, P0=stop, P1='NOT_STARTED',
         PRIMARY_STATUS='ENGINEERING_OR_BUDGET_STOP', execution_gate='BUDGET_STOP',
         TRACK_A='NOT_RUN_INPUTS_RELEASED', TRACK_B='NOT_RUN_PREREQUISITES_NOT_QUALIFIED',
         S1_S2_RESULT='NOT_MEASURED', rotation_discrimination='NOT_EVALUATED_FOR_SURROGATE',
         C5_C6_COMPARISON='NOT_PERFORMED', incremental_discordance='NOT_EVALUATED',
         long_label_status='VALID_RECORDED_CONTINUOUS_G_NOT_THE_BLOCKER',
         full_model_KL_pilot='NOT_AUTHORIZED_OR_STARTED_NO_INCREMENTAL_EVIDENCE',
         FINAL_ACTION='AUDIT_FREEZE_REPORT_STOP', reasons=block_reasons))
    not_produced = ['SHORT_SCALE_SURROGATE_SCORES.csv','TRACK_A_HISTORY_CONDITIONED.csv','TRACK_B_TRAIN_ACCESSIBLE.csv','ROTATION_RANKING_COMPARISON.csv','C5_C6_INCREMENTAL_ANALYSIS.csv','EARLY_WARNING_ONSET_AUDIT.json']
    save(out, 'ARTIFACT_STATUS_MANIFEST.json', dict(binding,
         artifacts={n:'NOT_PRODUCED_UNEXECUTED' for n in not_produced},
         raw_G_location='ASSET_AND_PROVENANCE_AUDIT.json:natural_parent_labels',
         prohibition_status={n:'NOT_STARTED' for n in ['TRAINING','NEW_ROTATION','C8','GRADIENT_OPTIMIZATION','AIME26','FREE_GENERATION','FULL_MODEL_KL_2K_8K','NEW32K_GENERATION']}))
    report = f'''# {TASK}\n\n## 结论\n\nP0 = {stop}；P1 = NOT_STARTED；PRIMARY_STATUS = ENGINEERING_OR_BUDGET_STOP。\n这是输入资产/预算阻断，不是 S1/S2 的科学否定。旧实验 COMPLETE/PASS/FAIL 身份不变。\n\n## 1. P0 审计\n\n冻结 quantizer、model 文件、原始 helper 和文档身份重新核验：{'PASS' if identity_pass else 'FAIL'}。Direction POSTFINAL 原哈希及全部 {manifest_check_direction['files']} 个条目重新核验：{manifest_check_direction['status']}。\n独立文档4篇；旧32/32是四文档×四条件×两个干预，不是32篇独立文档。\n\n## 2. 历史资产可复用性\n\n原始 Parent 连续 G 和 FP 状态端点可复用；原始 G 是无量纲 J_ACC_FD(32768)/J_ACC_FD(8192)。J采用每层每token物理空间相对误差，再按24层、1024..T等权均值，FP64累计。G值有限，未对记录的G截断；是否发生内部动态饱和不能仅由G有限推出。\n正式 drive 及学生8192锚点张量已按旧release释放；仅保存SHA/资格记录不是保存张量。工程2048输入仅覆盖Doc1早期，不能替代8192后的窗。\n时间分辨 signed scale/onset 来自 NO_SCALE_INFLATION/NO_SCALE_DEFLATION 干预轨迹，不得充当自然 Parent 短期信号。原始 Parent只有2048/8192/32768累计观测，不能恢复精确首次onset。\n\n## 3. 实际执行\n\n新增正式科学单元0；GPU资格测试0；没有模型forward、replay或量化观测执行。所有科学CSV标记NOT_PRODUCED_UNEXECUTED，没有生成空壳CSV。\n\n## 4. S1/S2 H32/H64\n\n全部NOT_MEASURED。用户给出的S1/S2/D1/D2定义记录为候选；epsilon/aggregation尚未冻结，不把方案当作测量。S2必须用同一student pre-QDQ u，观察分支不得写回；本次未实现/验证该hook。\n\n## 5. 两个Track\n\nTrackA：NOT_RUN_INPUTS_RELEASED。8192状态是已消费1-based8192 token后的状态；0-based[8192,8224)/[8192,8256)对应1-based8193..8224/8256。原G numerator与短窗重叠；没有8256累计边界，不能精确剔除重叠，亦不能声称pre-onset预测。\nTrackB：NOT_RUN_PREREQUISITES_NOT_QUALIFIED。原helper确认1024tokens、8 captures、H64 TBPTT和every-token QDQ/writeback；不做8192 warmup。WikiText训练teacher不能错配RepoBench长期G；本次没有改变文档或capture规则。\n\n## 6. H/C6R/C7区分\n\n已有长期G的从小到大顺序：Doc1 H<C6R<C7；Doc2 H<C6R<C7；Doc3 C6R<C7<H；Doc4 H<C7<C6R。它说明旧连续label有排序信息，不说明surrogate能预测排序。Native仅作negativecontrol；未计算任何surrogate Spearman或显著性。\n\n## 7. 原C5/C6比较\n\n原始helper可定位并通过SHA核验。B0是每head矩阵相对误差均值，分母clamp_min(1e-12)，再层均值；不是旧Parent按整层Frobenius归一化的J。B1是实际out_proj output与teacher output的global relative-MSE，再层均值；B2=B1+0.1B0。只用drive内q/k/v/g/beta或out_proj权重不能冒充原functional loss。原matched training确实有recurrent feedback。未计算B0/B1/B2与S1的matched comparison。\n\n## 8. 预测分歧\n\nNOT_EVALUATED。既不能报有增量，也不能报NO_ELIGIBLE_INCREMENTAL_DISCORDANCE；没有短程分数。\n\n## 9. 成本与资源\n\n新增GPUh=0；4GPUh余额=4。当前/data可用{free}bytes（{free/1024**3:.2f}GiB）；4张3090状态见runtime记录。磁盘不是主要阻断。按四篇历史capture的实测速率，重建到8256tokens的capture-only估算{capture_only_gpuh:.4f}GPUh，加25%余量{reserve_gpuh:.4f}GPUh；尚未计入学生anchor重建、qualification、TrackB和functional baseline。四卡并行可缩短墙钟时间，但不降低总GPUh。该估算不是新microbenchmark或精确下界。\n\n## 10. 后续与授权边界\n\n没有证据支持直接启动独立full-model KL pilot；本次没有执行该pilot，也没有C8/训练/AIME/自由生成。恢复需先获得身份可核验的自然Parent短窗drive、学生锚点及原functional pairing，或另行授权一个预算与原生产capture确定性闭环均合格的最小输入重建方案。当前旧capture使用deterministic_enforcement=False、replay使用True；不能静默改cumsum或关闭本轮determinism来恢复输入。\n\n不能把本次工程停止解释为NO_STABLE_ROTATION_DISCRIMINATION、HISTORY_CONDITIONED_ONLY_NOT_TRAIN_READY或LONG_HORIZON_LABEL_INVALID_FOR_RANKING。\n'''
    with (out/'FINAL_SCIENTIFIC_REPORT.md').open('x',encoding='utf-8',newline='\n') as f: f.write(report)
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8',newline='\n') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt': f.write(f'{sha(p)}  {p.relative_to(out)}\n')
    print(json.dumps(dict(output=str(out), P0=stop, identity='PASS' if identity_pass else 'FAIL',
                          postfinal_files=manifest_check_direction['files'], postfinal_status=manifest_check_direction['status'],
                          capture_only_estimate_gpu_hours=capture_only_gpuh, new_gpu_hours=0,
                          files=len(list(out.glob('*')))), ensure_ascii=False))


if __name__ == '__main__':
    main()
