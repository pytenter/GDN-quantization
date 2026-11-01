"""Finite PACE admissions and payload contracts. Import is stdlib/CPU only."""
import ast
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from inherited_device_guards import (atomic_new, binding_rows, process_identity,
                                     query_gpu, read, require, same_process, sha)

TASK = 'QWEN_GDN_PACE48_AIME26_SEED1_60_V1'
REVISION = 'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_REPAIR_20261025'
REMOTE = '/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
ARCHIVE = '/data/zypan/qwen_gdn_pace48_archive/' + REVISION
UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
BASE_GRANT_SHA = 'd245f009d90b67b11f283179ae91e16f538ed3bc94146361d62b78df5f978d31'
GPU0_CLARIFICATION_SHA = '4a73f4c35afb21e4172c9b8c2abe6b7dec02937e24eeea568e0e58a94bda3794'
SCALE_REPAIR_AUTHORIZATION_SHA = 'f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3'
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
              16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
CONDITIONS = {'R2': 8, 'R3': 1}
SOURCE = 'source_manifest.json'
STAGES = ('math_gpu_gates', 'calibration', 'gate_d', 'scale_repair_gates', 'formal')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def objsha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def status(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()) + '.' + str(threading.get_ident()))
    with tmp.open('xb') as f:
        f.write(canonical(value) + b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def progress(target, phase, **values):
    status(Path(target) / 'PROGRESS.json', dict(phase=phase, monotonic=time.monotonic(), **values))


def relative(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name and name != '.',
            'Unsafe relative artifact path')
    q = Path(root) / p
    require(not q.is_symlink(), 'Symlink artifact forbidden')
    return q


def verify_sources(root):
    root = Path(root)
    m = read(root / SOURCE)
    rows = m['files']
    require(m['task_id'] == TASK and m['count'] == len(rows) > 0, 'Code freeze identity/count differs')
    seen = set()
    for x in rows:
        p = relative(root, x['path'])
        require(x['path'] not in seen and p.is_file() and
                p.stat().st_size == x['bytes'] and sha(p) == x['sha256'],
                'Frozen code/input differs: ' + x['path'])
        seen.add(x['path'])
    for directory in ('src', 'vendor', 'scripts', 'tests', 'protocol', 'inputs', 'research'):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            require(not path.is_symlink(), 'Static source/input symlink forbidden: ' + str(path))
            if path.is_file():
                require(path.relative_to(root).as_posix() in seen,
                        'Unbound file in static execution/input directory: ' + str(path))
    return sha(root / SOURCE)


def validate_permit_values(root_metadata, permit, grant, source_sha, method_sha, grant_sha):
    """Pure validation shared with CPU fixtures; no file/device side effects."""
    require(root_metadata['task_id'] == TASK and root_metadata['execution_revision'] == REVISION and
            root_metadata['remote_root'] == REMOTE and root_metadata['archive_root'] == ARCHIVE,
            'Exact new task/revision roots differ')
    require(permit['task_id'] == grant['canonical_task_id'] == TASK and permit['ALLOW_GPU'] == 'YES' and
            permit['execution_revision'] == grant['execution_revision'] == REVISION,
            'Task-specific human GPU authorization missing')
    require(permit['execution_root'] == grant['remote_root'] == REMOTE and
            permit['physical_indices'] == [grant['only_physical_gpu']] == [0] and
            permit['devices'] == [grant['only_UUID']] == [UUID], 'Only55 physicalGPU0 is authorized')
    require(grant['status'] == 'EXPLICIT_HUMAN_PACE_SCALE_CHECK_REPAIR_AND_CONTINUATION_AUTHORIZATION' and
            grant['formal_total_unique'] == 60 and grant['remaining_formal_if_reuse_admitted'] == 57 and
            grant['old_true_STOP_source_and_seals_readonly'] is True and
            grant['failed_partial_credit'] == 0 and grant['one_new_independent_finite_execution_no_unbounded_GPU_retry'] is True,
            'Explicit finite repair and reuse scope differs')
    require(permit['source_manifest_sha256'] == source_sha and
            permit['method_spec_sha256'] == method_sha and
            permit['human_authorization_sha256'] == grant_sha, 'Permit source/method/grant pins differ')
    require(grant_sha == SCALE_REPAIR_AUTHORIZATION_SHA and
            permit['formal_new_samples'] == 57 and permit['reused_formal_samples'] == 3 and
            permit['formal_total_unique'] == 60 and permit['seed'] == 1 and
            permit['max_new_tokens'] == 81920 and permit['watchdog_seconds'] == 1800 and
            permit['cleanup_seconds'] == 120, 'Finite sample/watchdog policy differs')
    require(all(permit[k] is None for k in ('GPUh_hard_cap', 'total_wall_cap', 'unit_elapsed_cap')),
            'No GPUh/elapsed hard cap is authorized')


def authorize(root):
    root = Path(root).resolve()
    require(str(root) == REMOTE, 'Only the exact new remote PACE root may execute')
    p = read(root / 'GPU_PERMIT.json')
    g = root / 'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json'
    require(sha(root / 'operations/HUMAN_STOP_GROUP64_AND_START_PACE48_AUTHORIZATION.json') == BASE_GRANT_SHA,
            'Original human switch authorization changed')
    require(sha(root / 'operations/HUMAN_GPU0_ONLY_CLARIFICATION.json') == GPU0_CLARIFICATION_SHA,
            'GPU0-only/29 GPU5+6 foreign clarification changed')
    validate_permit_values(read(root / 'ROOT.json'), p, read(g), verify_sources(root),
                           sha(root / 'method_spec.json'), sha(g))
    method = read(root / 'method_spec.json')
    require(method['common']['no_rotation'] is True and method['common']['group_size'] == 32 and
            method['generation']['deterministic_algorithms'] is False and
            method['configurations']['R2']['decode_period'] == 8 and
            method['configurations']['R3']['decode_period'] == 1,
            'Frozen PACE/protocol semantics differ')
    return p


def model_guard(root):
    rows = read(Path(root) / 'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']
    require(len(rows) == 17, 'Full model/tokenizer17 inventory required')
    for x in rows:
        s = Path(x['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (x['bytes'], x['inode'], x['mtime_ns']),
                'Model/tokenizer stat changed: ' + x['path'])
    return dict(status='PASS_MODEL17_STAT_AGAINST_FROZEN_FRESH_SHA_INVENTORY', files=17)


def bindings(root):
    root = Path(root)
    rows = []
    for name in ('EXTERNAL_NATIVE_BINDINGS.json', 'EXTERNAL_PYTHON_BINDINGS.json'):
        rows += read(root / 'analysis' / name)['files']
    result = {}
    for x in rows:
        p = str(Path(x['path']).resolve())
        old = result.setdefault(p, x)
        require((old['bytes'], old['sha256']) == (x['bytes'], x['sha256']),
                'Conflicting external dependency binding')
    require(result, 'External runtime inventory missing')
    return result


def verify_external(root, full_model=False):
    rows = bindings(root)
    models = read(Path(root) / 'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']
    model_paths = {str(Path(x['path']).resolve()) for x in models}
    if full_model:
        for x in models:
            rows[str(Path(x['path']).resolve())] = x
    count = total = 0
    for p, x in rows.items():
        if not full_model and p in model_paths:
            continue
        require(Path(p).is_file() and Path(p).stat().st_size == x['bytes'] and sha(p) == x['sha256'],
                'External runtime bytes changed: ' + p)
        count += 1; total += x['bytes']
    return dict(status='PASS_FRESH_BOUND_EXTERNAL_SHA_BYTES', files=count, bytes=total,
                includes_full_model17_SHA=bool(full_model))


def device(own_pid=None, minimum_free=23000):
    obs = query_gpu()
    rows = [x for x in obs['devices'] if x['physical_index'] == 0]
    require(len(rows) == 1 and rows[0]['uuid'] == UUID and '3090' in rows[0]['name'] and
            rows[0]['memory_total_MiB'] >= 24000, 'Actual physicalGPU0/UUID differs')
    require(not [x for x in obs['apps'] if x['uuid'] == UUID and x['pid'] != own_pid],
            'Foreign process on GPU0; never preempt it')
    if own_pid is None:
        require(rows[0]['memory_free_MiB'] >= minimum_free, 'GPU0 free VRAM insufficient')
    else:
        own = [x for x in obs['apps'] if x['pid'] == own_pid]
        require(own and all(x['uuid'] == UUID for x in own), 'Actual worker NVML ownership differs')
    return obs


def driver_identity(root):
    # Same qualified driver admission, only its visibility-string I/O changes.
    p = Path(root) / 'src/inherited_device_guards.py'
    fn = next(n for n in ast.parse(p.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'require_driver_ordinal_identity')
    changed = 0
    for n in ast.walk(fn):
        if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Call) and
            isinstance(n.left.func, ast.Attribute) and n.left.func.attr == 'get' and
            n.left.args and isinstance(n.left.args[0], ast.Constant) and
            n.left.args[0].value == 'CUDA_VISIBLE_DEVICES'):
            require(len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Constant) and
                    n.comparators[0].value == '0', 'Qualified driver AST changed')
            n.comparators[0] = ast.Constant(UUID); changed += 1
    require(changed == 1, 'Exactly one visibility comparison must be adapted')
    ns = dict(Path=Path, require=require, sha=sha, os=os, UUID=UUID)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(p), 'exec'), ns)
    drivers = [x for p, x in bindings(root).items() if Path(p).name.startswith('libcuda.so')]
    return ns['require_driver_ordinal_identity'](drivers)


def library_check(root):
    allowed = bindings(root)
    selected = {}
    missing = []
    maps = Path('/proc/self/maps').read_text()
    words = ('libtorch', 'libc10', 'libcudart', 'libcublas', 'libcudnn', 'libnvrtc', 'libcuda.')
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        name = fields[5]
        if not (('.so' in name and '/site-packages/torch/' in name) or
                any(w in Path(name).name for w in words)):
            continue
        key = str(Path(name).resolve())
        row = allowed.get(key)
        if row is None:
            missing.append(key); continue
        s = Path(key).stat()
        require(int(fields[4]) == s.st_ino, 'Mapped native library inode differs')
        if key not in selected:
            require(s.st_size == row['bytes'] and sha(key) == row['sha256'], 'Mapped native SHA differs')
            selected[key] = dict(path=key, bytes=s.st_size, sha256=row['sha256'], inode=s.st_ino)
    require(selected and not missing, 'Actual loaded native closure is incomplete: ' + repr(missing))
    return dict(status='PASS_ACTUAL_LOADED_NATIVE_SHA_INODE', selected=list(selected.values()), full_maps=maps)


def worker_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key in ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_DEVICE_ORDER',
                'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        env.pop(key, None)
    return env


def storage_guard(root):
    s = os.statvfs(root)
    require(s.f_bavail * s.f_frsize >= 4 * 1024**3, 'Output filesystem below4GiB headroom')
    return dict(free_bytes=s.f_bavail * s.f_frsize)


def build_jobs(dataset):
    records = dataset['records']
    require(len(records) == 30 and {r['problem_id'] for r in records} ==
            {'aime26_' + str(i).zfill(2) for i in range(1, 31)}, 'Canonical30 roster differs')
    jobs = []
    for r in records:
        for condition, period in CONDITIONS.items():
            jobs.append(dict(id=r['problem_id'] + '_' + condition + '_seed1',
                problem_id=r['problem_id'], condition=condition, seed=1, decode_period=period,
                input_ids_sha256=r['input_ids_sha256'], input_tokens=len(r['input_ids']),
                max_new_tokens=81920, gpu=0, gpu_uuid=UUID))
    return dict(task_id=TASK, count=60, jobs=jobs, order='frozen_record_order_R2_then_R3_per_question')


def validate_jobs(root):
    saved = read(Path(root) / 'FORMAL_JOB_MANIFEST.json')
    expected = build_jobs(read(Path(root) / 'AIME26_DATASET_FREEZE.json'))
    require(saved == expected, 'Exact fixed60 interleaved formal roster differs')
    return saved['jobs']


def verify_head_mask(root):
    root = Path(root)
    m = read(root / 'calibration/head_mask.json')
    cal = read(root / 'calibration/calibration_manifest.json')
    score_path = root / 'calibration/head_scores.csv'
    require(m['schema'] == 'PACE48_HEAD_MASK_V1' and m['selection_decode_period'] == 8 and
            m['head_count'] == 32 and m['int8_per_layer'] == 8 and
            m['tie_break'] == 'ascending_head_index', 'Head mask definition differs')
    require(set(m['layers']) == set(map(str, GDN_LAYERS)), 'Head mask layer roster differs')
    require(cal['status'] == 'PASS' and cal['actual_cuda'] is True and cal['windows'] == 8 and
            cal['tokens_per_window'] == 2048 and cal['error_samples_per_head'] == 64 and
            cal['decay_samples_per_head'] == 16384 and cal['AIME_used_for_calibration'] is False and
            cal['head_mask_sha256'] == sha(root / 'calibration/head_mask.json') and
            cal['head_scores_sha256'] == m['head_scores_sha256'] == sha(score_path),
            'Actual calibration identity/counts differ')
    require(cal['input_manifest_sha256'] == m['input_manifest_sha256'] ==
            sha(root / 'protocol/calibration_windows.json') and
            cal['state_boundaries'] == list(range(256, 2049, 256)),
            'Actual calibration input/boundaries differ')
    inputs = read(root / 'protocol/calibration_windows.json')['windows']
    require(len(inputs) == 8 and cal['window_token_sha256'] ==
            [objsha(x['input_ids']) for x in inputs] and
            all(len(x['input_ids']) == 2048 for x in inputs), 'Calibration token window SHA differs')
    with score_path.open() as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 24 * 32, 'Missing/duplicate calibrated head scores')
    by = {}
    for x in rows:
        layer, head = int(x['layer']), int(x['head'])
        require(layer in GDN_LAYERS and 0 <= head < 32 and (layer, head) not in by,
                'Score layer/head identity differs')
        e, a, E = float(x['e_h']), float(x['a_h']), float(x['E_h'])
        require(all(math.isfinite(v) for v in (e, a, E)) and e >= 0 and 0 < a < 1,
                'Nonfinite/unsupported error-persistence statistics')
        denominator = -math.expm1(16 * math.log(a))
        require(int(x['error_samples']) == 64 and int(x['decay_samples']) == 16384 and
                math.isclose(float(x['denominator']), denominator, rel_tol=1e-12, abs_tol=1e-12) and
                math.isclose(E, e / denominator, rel_tol=1e-12, abs_tol=1e-12),
                'Frozen e/(1-a**16) score arithmetic differs')
        by[layer, head] = E
    for layer in GDN_LAYERS:
        chosen = sorted(sorted(range(32), key=lambda h: (-by[layer, h], h))[:8])
        x = m['layers'][str(layer)]
        require(x['int8_heads'] == chosen and x['int4_heads'] == sorted(set(range(32)) - set(chosen)),
                'Actual mask is not fixed top8/32 by calibrated score')
        for head in range(32):
            row = next(r for r in rows if int(r['layer']) == layer and int(r['head']) == head)
            require(int(row['selected_bits']) == (8 if head in chosen else 4), 'CSV/mask bits differ')
    return dict(status='PASS_ACTUAL_CALIBRATION_MASK_SCORE_AND_SAME_ARM_CONTRACT',
                head_mask_sha256=sha(root / 'calibration/head_mask.json'), layers=24,
                INT8_heads_per_layer=8, INT4_heads_per_layer=24,
                numerical_or_GPU_qualification_credit=0)


def freeze_science(root):
    root = Path(root)
    mask = verify_head_mask(root)
    names = [SOURCE, 'method_spec.json', 'GPU_PERMIT.json', 'FORMAL_JOB_MANIFEST.json',
             'AIME26_DATASET_FREEZE.json', 'protocol/calibration_windows.json',
             'calibration/head_mask.json', 'calibration/head_scores.csv',
             'calibration/calibration_manifest.json', 'protocol/sampler.py',
             'protocol/common.py', 'protocol/aime26_scorer_v4.py', 'protocol/aime26_frozen.jsonl',
             'CONTINUATION_ADMISSION.json', 'REUSED_CALIBRATION_ADMISSION.json']
    value = dict(status='SCIENCE_FROZEN_AFTER_REPAIR_GATE_AND_ONCE_CALIBRATION_REUSE',
                 task_id=TASK, mask=mask, files=[dict(path=n, bytes=(root / n).stat().st_size,
                                                     sha256=sha(root / n)) for n in names])
    atomic_new(root / 'SCIENCE_FREEZE.json', value)
    return value


def verify_science(root):
    root = Path(root)
    f = read(root / 'SCIENCE_FREEZE.json')
    permitted = {'SCIENCE_FROZEN_AFTER_REPAIR_GATE_AND_ONCE_CALIBRATION_REUSE'}
    # Only the exact fixed provenance snapshot may carry its original status.
    if root.name == 'PRIOR_PACE_TRUE_STOP' and root.parent.name == 'provenance':
        permitted = {'SCIENCE_FROZEN_AFTER_ONE_CALIBRATION_BEFORE_GATE_D_AND_FORMAL'}
    require(f['status'] in permitted and f['task_id'] == TASK, 'Final science freeze missing')
    for x in f['files']:
        p = relative(root, x['path'])
        require(p.stat().st_size == x['bytes'] and sha(p) == x['sha256'], 'Frozen science changed: ' + x['path'])
    return sha(root / 'SCIENCE_FREEZE.json')


def validate_formal(row, job, root):
    root = Path(root)
    require(row['status'] == 'COMPLETE' and row['runtime_error'] is None and
            row['nonfinite'] is False and row['deterministic_algorithms'] is False,
            'Technical/nonfinite/protocol formal failure')
    for key in ('condition', 'problem_id', 'seed', 'input_ids_sha256', 'input_tokens'):
        require(row[key] == job[key], 'Formal identity differs: ' + key)
    require(row['head_mask_sha256'] == sha(root / 'calibration/head_mask.json'), 'Formal mask changed')
    tokens = row['generated_token_ids']; n = len(tokens)
    require(0 < n <= 81920 and row['generated_token_count'] == n and
            all(type(t) is int and 0 <= t < 248320 for t in tokens) and
            row['generated_token_ids_sha256'] == objsha(tokens), 'Formal token payload differs')
    eos = tokens[-1] in row['stop_token_ids']
    require(not any(t in row['stop_token_ids'] for t in tokens[:-1]) and
            row['finish_reason'] == ('eos' if eos else 'length') and (eos or n == 81920),
            'Illegal original EOS/length termination')
    total = row['input_tokens'] + n - 1
    require(row['actual_model_forward_calls'] == total and row['sampler_calls'] == n,
            'Original p+n-1 model/sampling count differs')
    audit = row['quantizer_audit']
    require(set(audit['layers']) == set(GDN_LAYERS) and len(audit['layers']) == 24 and
            audit['actual_layer_coverage'] == 24 and len(audit['helper_calls']) == 24 and
            all(v == total for v in audit['helper_calls'].values()) and
            audit['dense_recurrent_carrier'] is False and audit['decode_period'] == job['decode_period'] and
            audit['prefill_period'] == 256 and audit['output_before_state_writeback'] is True,
            'PACE true packed buffer/helper coverage differs')
    for key in ('native_first_chunk_compat_calls', 'native_first_chunk_state_checked'):
        require(set(audit[key]) == set(map(str, GDN_LAYERS)) and
                all(type(v) is int and v == 1 for v in audit[key].values()),
                'Each actual layer requires one native initial-chunk compatibility/state check')
    require(audit['native_first_chunk_state_retained'] is False and
            audit['native_first_chunk_state_discarded'] is True,
            'Native initial FP state must be transient and discarded')
    expected_checks = 24 * sum(t > 0 and (t < 128 or t % 2048 == 0) for t in range(total))
    require(audit['marker_checks'] == expected_checks, 'Actual sparse native marker checks differ')
    require(len(audit['storage_by_layer']) == 24, 'Packed storage accounting missing')
    for x in audit['storage_by_layer'].values():
        require(x['dense_state_persistent'] is False and x['fixture_only'] is False and
                x['tokens'] == total and x['prompt_length'] == row['input_tokens'] and
                x['decode_tokens'] == n - 1 and x['pending_tokens'] == (n - 1) % job['decode_period'] and
                x['writeback_count'] == math.ceil(row['input_tokens'] / 256) + (n - 1) // job['decode_period'] and
                x['quantization_count'] == x['writeback_count'] and
                x['finalization_policy'] == 'preserve', 'PACE boundary/persistent storage differs')
    for name, digest in row['protocol_identity'].items():
        require(digest == sha(root / 'protocol' / name), 'Frozen original protocol changed')
    require(set(row['protocol_identity']) == {'sampler.py', 'common.py', 'aime26_scorer_v4.py',
                                             'aime26_frozen.jsonl'}, 'Complete original protocol binding required')
    require(row['dataset_freeze_sha256'] == sha(root / 'AIME26_DATASET_FREEZE.json'),
            'Formal dataset identity differs')
    return row


def unit_target(root, stage, unit):
    require(stage in STAGES and isinstance(unit, str) and unit.replace('_', '').isalnum(), 'Unsafe stage/unit')
    if stage == 'formal':
        jobs = {x['id']: x for x in validate_jobs(root)}
        require(unit in jobs, 'Unknown formal unit')
        condition = jobs[unit]['condition']
        return Path(root) / (condition + '_PACE48_DP' + str(CONDITIONS[condition])) / 'trajectories' / unit
    expected = {'math_gpu_gates': 'GATES_ABC', 'calibration': 'CALIBRATION_ONCE', 'gate_d': 'FULL_MODEL_D',
                'scale_repair_gates': 'SCALE_REPAIR_GATES'}
    require(unit == expected[stage], 'Engineering phase identity differs')
    return Path(root) / 'execution' / unit


def release_state(proc_exited, observation, pid, identity=None, reader=process_identity):
    """Unknown NVML or a live child never means released. Other devices are foreign."""
    if not proc_exited or observation is None:
        return False
    if not any(x['pid'] == pid for x in observation['apps']):
        return True
    if identity is not None:
        try:
            current = reader(pid)
            # A different starttick AND actual exited Popen proves PID reuse;
            # this app is foreign and must never be signalled.
            return current['pid'] == pid and current['starttick'] != identity['starttick']
        except (OSError, ValueError):
            pass
    return False


def terminate_owned(proc, identity, reader=process_identity, kill=False):
    require(identity is not None and identity['pid'] == proc.pid and same_process(identity, reader),
            'Refuse signal: own PID/starttick/full argv identity is not exact')
    if proc.poll() is None:
        # Only this Popen PID. No name matching, no global/group GPU kill.
        proc.send_signal(signal.SIGKILL if kill else signal.SIGTERM)


def verify_result(root, stage, unit, expected_identity=None):
    target = unit_target(root, stage, unit)
    r = read(target / 'RESULT.json')
    digest = r.pop('payload_sha256')
    require(objsha(r) == digest, 'Canonical worker payload SHA differs')
    r['payload_sha256'] = digest
    require(r['status'] == 'PASS' and r['task_id'] == TASK and r['stage'] == stage and
            r['unit'] == unit and r['gpu'] == 0 and r['gpu_uuid'] == UUID and
            r['source_manifest_sha256'] == sha(Path(root) / SOURCE) and
            r['permit_sha256'] == sha(Path(root) / 'GPU_PERMIT.json'), 'Worker receipt binding differs')
    if expected_identity is not None:
        require(r['identity'] == expected_identity, 'Actual worker receipt PID/starttick/argv differs')
    if stage in ('gate_d', 'formal'):
        require(r['science_freeze_sha256'] == verify_science(root), 'Final frozen science SHA differs')
    if stage == 'formal':
        job = next(x for x in validate_jobs(root) if x['id'] == unit)
        validate_formal(r['result'], job, root)
    if stage == 'scale_repair_gates':
        value = r['result']
        require(value['status'] == 'PASS' and value['actual_cuda'] is True and
                value['no_AIME_or_calibration_or_formal_credit'] is True and
                value['original_gates']['status'] == 'PASS' and value['original_gates']['actual_cuda'] is True and
                all(value['original_gates']['gates'][g]['status'] == 'PASS' and
                    value['original_gates']['gates'][g]['actual_cuda'] is True for g in ('A','B','C')) and
                value['scale_boundary_tests']['status'] == 'PASS' and
                value['scale_boundary_tests']['actual_cuda'] is True,
                'New related CUDA ABC plus true FP16 boundary gates did not all pass')
    return r
