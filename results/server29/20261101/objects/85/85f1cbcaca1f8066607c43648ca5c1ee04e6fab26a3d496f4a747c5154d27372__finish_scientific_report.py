#!/usr/bin/env python3
"""Explicit post-release CPU reporting only. Default preview; never launches GPU/model.

--run requires a root-verified release proof, executes each retained-data CPU
reducer at most once, and preserves partial evidence/errors without retry.
No archive publication, deletion, model forward, Driver or NVML query exists.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
PARENT = '64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457'
WORKER = '1b709359fe4652a611814bc07648b8a42181259603917f29058e486ea86f9613'
FIXTURE = '18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d'
LAYERS = tuple(i for i in range(32) if i % 4 != 3)
NAMES = ['query', 'key', 'value', 'gate', 'beta', 'core_actual', 'state_actual', 'native_norm_query', 'native_norm_key']
UNITS = ['NUMERIC_FRESH0', 'NUMERIC_FRESH1', 'OLD_FP_R0', 'OLD_FP_R1', 'OLD_FP_R2', 'NEW_FP_R0']
REPLACED = {'scripts/restoration_controller.py', 'worker/restoration_worker.py', 'qualification/GPU_PLAN.json', 'USER_REQUEST.txt'}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def stable(p):
    p = Path(p); require(p.is_file() and not p.is_symlink(), 'Missing/unsafe evidence: ' + str(p))
    a = p.stat(); b = p.read_bytes(); z = p.stat()
    require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) == (z.st_dev, z.st_ino, z.st_size, z.st_mtime_ns), 'Evidence changed: ' + str(p))
    return b


def sha(p): return hashlib.sha256(stable(p)).hexdigest()
def read(p): return json.loads(stable(p))
def optional(p): return read(p) if Path(p).is_file() else None
def descriptor(p, root): return dict(path=str(Path(p).relative_to(root)), bytes=Path(p).stat().st_size, sha256=sha(p))


def safe(root, name):
    require(isinstance(name, str) and name and not any(c in name for c in '\n\r\x00'), 'Invalid evidence path')
    p = PurePosixPath(name); require(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'Unsafe evidence path')
    q = Path(root)
    for component in p.parts: q = q / component; require(not q.is_symlink(), 'Evidence ancestor symlink')
    require(Path(root).resolve() in q.resolve().parents, 'Evidence escapes root')
    return q


def verify_description(root, row):
    p = safe(root, row['path']); require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Evidence SHA/bytes differs: ' + row['path']); return p


def new_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def events(path):
    rows, errors = [], []
    if not path.is_file(): return rows, ['RESOURCE_EVENTS.jsonl missing']
    for number, line in enumerate(stable(path).splitlines(), 1):
        if not line: continue
        try: rows.append(json.loads(line))
        except (ValueError, UnicodeError) as exc: errors.append(dict(line=number, error=str(exc)))
    return rows, errors


def source_binding(payload):
    freeze = read(payload / 'SOURCE_FREEZE.json'); rows = freeze['files']; parent_path = payload / 'provenance/PARENT_EFFECTIVE_SOURCE_FREEZE.json'
    require(freeze.get('task_id') == TASK and len(rows) == 407 and sha(parent_path) == PARENT, 'New407/parent lineage differs')
    old = {x['path']: x for x in read(parent_path)['files']}; now = {}
    for row in rows:
        require(row['path'] not in now, 'Duplicate frozen path'); now[row['path']] = row; verify_description(payload, row)
    require(set(now) == set(old) and len(old) == 407 and all(now[n] == old[n] for n in set(now) - REPLACED), 'Inherited403 changed')
    require(all(now[n] != old[n] for n in REPLACED), 'Four amendments not explicit')
    require(sha(payload / 'worker/restoration_worker.py') == WORKER, 'Observation worker differs from approved1b709 source')
    require(sha(payload / 'candidate/benchmark_replay/fixtures/doc1_tokens.json') == FIXTURE, 'Fixture differs')
    plan = read(payload / 'qualification/GPU_PLAN.json'); auth = read(payload / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json')
    identifiers = dict(plan_sha256=sha(payload / 'qualification/GPU_PLAN.json'), source_freeze_sha256=sha(payload / 'SOURCE_FREEZE.json'), parent_effective_source_freeze_sha256=PARENT, request_sha256=sha(payload / 'USER_REQUEST.txt'), resource_authorization_sha256=sha(payload / 'HUMAN_GPU_RESOURCE_AUTHORIZATION.json'), authorization_sha256=sha(payload / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json'))
    require(auth.get('task_id') == TASK and auth.get('ALLOW_GPU_RESTORE') == 'YES' and auth.get('execution_root') == str(payload), 'Execution authorization binding differs')
    require(auth.get('device_uuid') == UUID and all(auth.get(k) == identifiers[k] for k in identifiers if k != 'authorization_sha256'), 'Authorization source/device differs')
    require(plan.get('numeric_tokens') == 32 and plan.get('warmup_tokens') == plan.get('measured_tokens') == 128 and plan.get('max_active_GPUs') == 1, 'Finite scientific protocol differs')
    require(all(k in plan and plan[k] is None for k in ['max_GPU_hours', 'hard_seconds', 'dispatch_max_seconds']), 'Superseded elapsed cap reintroduced')
    require(plan.get('units') == [dict(id=u, mode='numeric' if u.startswith('NUMERIC') else 'benchmark', optional=u == 'NEW_FP_R0') for u in UNITS], 'Finite six-unit roster differs')
    bindings = []
    def collect(obj):
        if isinstance(obj, dict):
            if {'path', 'bytes', 'sha256'} <= set(obj): bindings.append(obj)
            for value in obj.values(): collect(value)
        elif isinstance(obj, list):
            for value in obj: collect(value)
    for name in plan['binding_files']: collect(read(safe(payload, name)))
    models = {x.get('resolved_path', x['path']): x for x in bindings if x['path'].endswith('.safetensors')}
    require(len(models) == 4, 'Four model shard identities required')
    return dict(status='PASS_LOCAL_SOURCE403_PLUS4_SHA_BINDING', files=407, inherited=403, amended=4, identities=identifiers, worker_sha256=WORKER, GPU_UUID=UUID, fixture_sha256=FIXTURE, model_shards=list(models.values()), binding_metadata_files=[descriptor(safe(payload, n), payload) for n in plan['binding_files']], historical_binary_EXACT='UNKNOWN_NOT_CLAIMED', plan=plan)


def release_admission(run, proof_path, binding):
    run = Path(run).resolve(); payload = run / 'payload'; proof_path = Path(proof_path).resolve(); require(run in proof_path.parents, 'Release proof must be inside this run')
    proof = read(proof_path)
    require(proof.get('schema') == 'QWEN_RECOVERY_RELEASE_ADMISSION_V1' and proof.get('task_id') == TASK and proof.get('status') == 'PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED', 'Root release proof schema/status differs')
    require(proof.get('payload_root') == proof.get('execution_root') == str(payload), 'Release proof root differs')
    require(all(proof.get(k) is True for k in ['controller_exit_verified', 'controller_exited', 'all_owned_processes_exited', 'all_owned_GPU_resources_released']), 'Actual exit/release proof incomplete')
    freeze_values = [proof[k] for k in ['source_freeze_sha256', 'source_freeze_sha'] if k in proof]
    require(freeze_values and all(x == binding['identities']['source_freeze_sha256'] for x in freeze_values), 'Release/source freeze differs')
    for row in proof.get('evidence_receipts', []): verify_description(run, row)
    require(proof.get('evidence_receipts'), 'Release evidence receipts empty')
    start = optional(payload / 'execution/CONTROLLER_STARTED.json') or read(payload / 'execution/CONTROLLER_ATTEMPT_STARTED.json')
    require(proof['controller_identity'] == start['controller_identity'], 'Exited controller identity differs')
    dispatched = []
    for unit in UNITS:
        p = payload / 'execution' / unit / 'DISPATCH.json'
        if p.is_file(): dispatched.append(read(p)['worker_identity'])
    key = lambda x: (x['pid'], x['starttick'], tuple(x['cmdline']))
    require(sorted(map(key, proof['owned_worker_identities'])) == sorted(map(key, dispatched)), 'Release worker roster differs from actual dispatches')
    own = {key(x) for x in [proof['controller_identity']] + dispatched}; listed = {key(x): x for x in proof['own_processes']}
    require(own <= set(listed), 'Release process identities omitted')
    require(all(listed[k].get('process_exit_verified') is True and listed[k].get('all_UUID_NVML_release_verified') is True for k in own), 'Release process row incomplete')
    pids = {k[0] for k in own}; require(proof.get('final_owned_NVML_records') == [] and not any(x['pid'] in pids for x in proof['final_NVML_apps']), 'An owned PID remains in final actual NVML evidence')
    for field, name in [('controller_result', 'execution/CONTROLLER_RESULT.json'), ('resource_ledger', 'execution/RESOURCE_EVENTS.jsonl'), ('first_failure', 'FIRST_FAILURE.json')]:
        row = proof.get(field); exists = (payload / name).is_file()
        require(bool(row) == exists, 'Release proof terminal artifact presence differs: ' + field)
        if row: require(row['path'] == name, 'Terminal descriptor path differs'); verify_description(payload, row)
    return proof


def resource_summary(rows, parse_errors):
    result, issues = [], list(parse_errors)
    for unit in UNITS:
        def pick(name): return [x for x in rows if x.get('event') == name and (x.get('unit') or {}).get('id', x.get('unit_id')) == unit]
        dispatch, exits, releases, bounds = [pick(x) for x in ['FIRST_DISPATCH', 'FIRST_PROCESS_EXIT_OBSERVED', 'FIRST_ALL_UUID_NVML_RELEASE_OBSERVED', 'UNIT_RESERVATION_INTERVAL_BOUNDS']]
        if not dispatch: continue
        if any(len(x) != 1 for x in [dispatch, exits, releases, bounds]): issues.append(dict(unit=unit, issue='Missing/duplicate first dispatch/exit/release/cost events', counts=list(map(len, [dispatch, exits, releases, bounds]))))
        identity = dispatch[0]['worker_identity']; samples = []; bad_uuid = []; unknown = 0
        if exits and exits[0].get('worker_identity') != identity: issues.append(dict(unit=unit, issue='Exit identity differs', actual=exits[0]))
        if releases and releases[0].get('worker_identity') != identity: issues.append(dict(unit=unit, issue='Release identity differs'))
        for event in rows:
            if event.get('event') != 'NVML_OBSERVATION' or event.get('worker_identity') != identity: continue
            for app in event.get('observation', {}).get('apps', []):
                if app['pid'] != identity['pid']: continue
                if app['uuid'] != UUID: bad_uuid.append(app)
                try:
                    value = float(app['memory_MiB']); require(math.isfinite(value) and value >= 0, 'Invalid NVML memory sample'); samples.append(value)
                except (ValueError, TypeError, RuntimeError): unknown += 1
        if bad_uuid: issues.append(dict(unit=unit, issue='Own PID on another UUID', records=bad_uuid))
        cost = bounds[0] if len(bounds) == 1 else {}
        lo, hi = cost.get('reservation_seconds_lower'), cost.get('reservation_seconds_upper')
        if not (type(lo) in (int, float) and type(hi) in (int, float) and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi): issues.append(dict(unit=unit, issue='Reservation interval not finite nonnegative ordered'))
        result.append(dict(unit=unit, worker_identity=identity, first_dispatch=dispatch[0], first_exit=exits[0] if exits else None, first_all_UUID_release=releases[0] if releases else None, worker_returncode=exits[0].get('returncode') if exits else None, reservation_seconds_lower=cost.get('reservation_seconds_lower', 'UNRESOLVED'), reservation_seconds_upper=cost.get('reservation_seconds_upper', 'UNRESOLVED'), NVML_owned_PID_sample_peak_MiB=max(samples) if samples else None, NVML_sample_count=len(samples), unknown_memory_samples=unknown, NVML_peak_scope='SAMPLING_LOWER_BOUND_NOT_CONTINUOUS_EXACT_PEAK'))
    if not result: issues.append('No actual dispatch/resource intervals retained')
    lower = [x['reservation_seconds_lower'] for x in result]; upper = [x['reservation_seconds_upper'] for x in result]
    known = lambda xs: all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in xs)
    return dict(status='PASS_EVENT_DERIVED_INTERVALS' if not issues else 'UNRESOLVED_OR_INVALID_RESOURCE_EVENTS', units=result, issues=issues, cumulative_reservation_GPU_hours_lower=sum(lower) / 3600 if lower and known(lower) else 'UNRESOLVED', cumulative_reservation_GPU_hours_upper=sum(upper) / 3600 if upper and known(upper) else 'UNRESOLVED', actual_physical_GPU_hours='UNRESOLVED', meaning='Process-reserved interval bounds, including initialization/CPU work with GPU resident; not exact utilization GPUh. Empty observations are not zero-cost proof.', no_elapsed_GPU_or_dispatch_cap=True, no_progress_watchdog_seconds=1800)


def audit_unit(payload, unit, binding):
    directory = payload / 'execution' / unit; out = dict(unit=unit, status='NOT_RUN_NO_DISPATCH', issues=[], result=None)
    if not (directory / 'DISPATCH.json').is_file(): return out
    out['status'] = 'DISPATCHED_PARTIAL_OR_FAILED'
    def need(ok, text):
        if not ok: out['issues'].append(text)
    dispatch = read(directory / 'DISPATCH.json'); permit = read(directory / 'PERMIT.json'); result = optional(directory / 'RESULT.json'); out.update(result=result, dispatch=dispatch, first_failure=optional(directory / 'FIRST_FAILURE.json'), earliest_divergence=optional(directory / 'EARLIEST_DIVERGENCE.json'), earliest_numerical_failure=optional(directory / 'EARLIEST_NUMERICAL_FAILURE.json'))
    need(dispatch.get('permit_sha256') == sha(directory / 'PERMIT.json'), 'Dispatch/permit SHA differs')
    need(permit.get('task_id') == TASK and permit.get('root') == str(payload) and permit.get('unit', {}).get('id') == unit and permit.get('device', {}).get('uuid') == UUID, 'Permit task/root/unit/device differs')
    need(all(permit.get(k) == v for k, v in binding['identities'].items()), 'Permit runtime/source/model authorization binding differs')
    need(result is not None and result.get('status') == 'PASS', 'No completed PASS worker RESULT')
    if result: need(result.get('process_identity') == dispatch['worker_identity'] and result.get('permit_sha256') == sha(directory / 'PERMIT.json') and result.get('unit_id') == unit, 'Result own process/permit/unit differs')
    driver = optional(directory / 'DRIVER_ORDINAL_IDENTITY.json'); out['driver'] = driver
    need(driver is not None and driver.get('status') == 'PASS' and driver.get('visible_device_count') == 1 and driver.get('actual_uuid') == UUID, 'Driver actual ordinal/UUID not PASS')
    for phase in ['PREFLIGHT', 'POSTFLIGHT']:
        lib = optional(directory / (phase + '_LIBRARY_IDENTITY.json')); own = optional(directory / (phase + '_OWN_GPU_IDENTITY.json')); out[phase] = dict(library=lib, own_GPU=own)
        need(lib is not None and lib.get('status') == 'PASS' and lib.get('phase') == phase and not lib.get('missing') and bool(lib.get('selected')) and all(x.get('match') and x.get('stable') and x.get('mapped_inode') == x.get('disk_inode') for x in lib.get('selected', [])) and lib.get('selection_matches_same_phase_OLD_reference') is True, phase + ' bound library selection not PASS')
        need(own is not None and own.get('status') == 'PASS' and own.get('own_pid') == dispatch['worker_identity']['pid'] and own.get('actual_UUIDs') == [UUID] and bool(own.get('own_records')), phase + ' actual own PID/UUID not PASS')
        if lib and own: need(lib.get('own_gpu_identity') == own, phase + ' embedded own GPU identity differs')
    progress, progress_errors = events(directory / 'PROGRESS_EVENTS.jsonl'); completed = {}
    for event in progress:
        if event.get('process_identity') == dispatch['worker_identity'] and event.get('phase') == 'BOUND_FILE_HASH_COMPLETE': completed[event['details']['path']] = event['details']
    model_hash_ok = all(any(v.get('sha256') == row['sha256'] and v.get('read_bytes') == row['bytes'] for path, v in completed.items() if path in {row['path'], row.get('resolved_path', row['path'])}) for row in binding['model_shards'])
    need(model_hash_ok, 'Actual pre-runtime completed four-model-shard hash facts incomplete'); out['actual_model_hash_facts_complete'] = model_hash_ok
    out['loader_timing'] = optional(directory / 'MODEL_LOAD_TIMING.json'); out['allocator_observation'] = optional(directory / 'WORKER_CUDA_ALLOCATOR_OBSERVATION.json')
    need(out['loader_timing'] is not None and out['allocator_observation'] is not None, 'New load/allocator observation missing')
    if out['loader_timing']:
        timer = out['loader_timing']; a, b, wall = [timer.get(k) for k in ['loader_call_start_monotonic', 'loader_call_end_monotonic', 'loader_call_wall_seconds']]
        need(timer.get('unit') == unit and all(type(x) in (int, float) and math.isfinite(x) for x in [a, b, wall]) and b >= a and math.isclose(b-a, wall, abs_tol=1e-9), 'Actual loader monotonic/wall identity differs')
    if out['allocator_observation']:
        allocator = out['allocator_observation']; need(allocator.get('unit') == unit and bool(allocator.get('samples')) and allocator.get('original_benchmark_warmup_decode_peak_unchanged') is True, 'Allocator scope/epochs incomplete')
    if unit.startswith('NUMERIC'):
        math_audit = optional(directory / 'MATHEMATICAL_AUDIT.json'); capture = optional(directory / 'SUCCESS_ACTIVATION_MANIFEST.json'); trace = optional(directory / 'TRACE.json'); out.update(math_audit=math_audit, capture_summary=None, trace_tokens=len(trace.get('tokens', [])) if trace else 0)
        expected = {(t, layer) for t in range(32) for layer in LAYERS}; keys = set(); capture_issues = []
        if capture:
            need(capture.get('unit') == unit and capture.get('tokens') == 32 and capture.get('layers') == list(LAYERS) and capture.get('calls') == 768 and capture.get('status') == 'COMPLETE_SUCCESS_ACTIVATION_RAW_ROSTER_NOT_QUALIFICATION', 'Success capture summary contract differs')
            for item in capture.get('files', []):
                key = (item['token'], item['layer']); need(key not in keys, 'Duplicate capture key'); keys.add(key)
                index = safe(payload, item['index_path']); need(index.stat().st_size == item['index_bytes'] and sha(index) == item['index_sha256'], 'Capture index SHA/bytes differs'); rec = read(index)
                need(rec['unit'] == unit and (rec['token'], rec['layer']) == key and rec['kind'] == ('chunk' if key[0] == 0 else 'recurrent') and [x['name'] for x in rec['tensors']] == NAMES, 'Nine-tensor capture identity/roster differs')
                need(all(item[k] == rec[k] for k in ['raw_path', 'raw_bytes', 'raw_sha256']), 'Capture manifest/index raw identity differs')
                sb = rec.get('source_bindings', {}); need(sb.get('worker_sha256') == WORKER and sb.get('fixture_sha256') == FIXTURE and sb.get('source_freeze_sha256') == binding['identities']['source_freeze_sha256'] and sb.get('permit_sha256') == sha(directory / 'PERMIT.json') and sb.get('GPU_UUID') == UUID, 'Capture source/runtime/fixture/GPU binding differs')
            out['capture_summary'] = dict(calls=capture.get('calls'), actual_keys=len(keys), expected_keys=768, raw_bytes=capture.get('raw_bytes'), nine_fields=NAMES)
        need(capture is not None and keys == expected, 'Incomplete actual32x24 nine-field capture')
        need(math_audit is not None and math_audit.get('status') == 'PASS_PREFIX_ONLY_NATIVE_RECURRENCE_FP64_CHECK' and len(math_audit.get('checks', [])) == 768 and {(x['token'], x['layer']) for x in math_audit.get('checks', [])} == expected, 'Original768 online math checks incomplete')
        if math_audit: need(all(x['state_max_abs'] <= x['state_bound'] and x['core_max_abs'] <= x['core_bound'] and all(y['max_abs'] <= y['bound'] for y in x['normalization_checks']) for x in math_audit['checks']), 'Original online math bound failed')
        need(result is not None and result.get('tokens') == 32 and result.get('all_model_layers') == 32 and result.get('GDN_layers') == 24 and trace is not None and len(trace['tokens']) == 32 and trace.get('source_fixture_sha256') == FIXTURE, 'Original32-token cache/logits trace incomplete')
    else:
        legacy = result.get('historical_worker_result') if result else None; out['measurement'] = None
        if legacy:
            original = payload / 'candidate/benchmark_replay/execution' / unit / 'RESULT.json'
            need(original.is_file() and sha(original) == result['historical_worker_result_sha256'] and read(original) == legacy, 'Legacy actual measurement receipt differs')
            need(legacy.get('status') == 'PASS' and legacy.get('profile') is False and legacy.get('GPU_UUID') == UUID and legacy.get('fixture_sha256') == FIXTURE and legacy.get('warmup_tokens') == legacy.get('measured_tokens') == 128 and legacy.get('model_forward_calls') == 256 and legacy.get('cache_retained_at_measurement_start') == 128 and legacy.get('cache_final_length') == 256, 'Original128+128 full-call/cache/UUID/fixture contract differs')
            need(legacy.get('condition') == ('NEW_FP' if unit.startswith('NEW') else 'OLD_FP') and legacy.get('repeat') == int(unit.rsplit('R', 1)[1]) and legacy.get('scope') == 'MATCHED_NON_AIME_PERFORMANCE_NOT_NUMERICAL_EQUIVALENCE' and legacy.get('decoder_layer_calls') == 8192 and legacy.get('GDN_chunk_calls') == 24 and legacy.get('GDN_recurrent_calls') == 6120 and legacy.get('measured_GDN_recurrent_calls') == 3072 and legacy.get('prefill_tokens') == 1 and legacy.get('QDQ_calls') == 0 and legacy.get('hadamard_calls') == 0, 'FP call counts/condition/short-prefill/no-QDQ contract differs')
            need(legacy.get('source_manifest_sha256') == sha(payload / 'candidate/benchmark_replay/SOURCE_SHA256_MANIFEST.json') and legacy.get('plan_sha256') == sha(payload / 'candidate/benchmark_replay/GPU_DIAGNOSTIC_PLAN.json'), 'Legacy benchmark manifest/plan binding differs')
            need(legacy.get('actual_positions') == [[[[t]], [[t]], [[t]]] for t in range(256)] and legacy.get('actual_KV_growth') == [dict(token_index=t, before=t, after=t+1, all8attention_KV_lengths=[t+1]*8) for t in range(256)], 'Actual per-input rotary position or all8 KV lengths differs')
            wall = legacy.get('decode_wall_seconds'); tps = legacy.get('tokens_per_second'); ms = legacy.get('wall_ms_per_token')
            valid = all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in [wall, tps, ms])
            need(valid and math.isclose(tps, 128 / wall, rel_tol=1e-12) and math.isclose(ms, 1000 * wall / 128, rel_tol=1e-12), 'Actual timing arithmetic inconsistent')
            out['measurement'] = dict(unit=unit, tokens_per_second=tps, wall_ms_per_token=ms, decode_wall_seconds=wall, peak_allocated_bytes=legacy.get('peak_allocated_bytes'), peak_reserved_bytes=legacy.get('peak_reserved_bytes'), original_warmup_decode_peak_scope='ORIGINAL_RESULT_UNCHANGED', GPU_UUID=UUID, fixture_sha256=FIXTURE, worker_sha256=WORKER, source_freeze_sha256=binding['identities']['source_freeze_sha256'], parent_effective_source_freeze_sha256=PARENT)
    if not out['issues']: out['status'] = 'PASS_COMPLETE_UNIT_IDENTITY_AND_RETAINED_METADATA'
    return out


def reducer_once(command, output, expected_json, runner=subprocess.run):
    require(not output.exists(), 'Reducer output exists: no retry/overwrite')
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    record = dict(command=command, reducer_sha256=sha(Path(command[2])), status='REDUCER_ERROR', output=str(output), retry=False)
    try:
        proc = runner(command, env=env, capture_output=True, text=True)
        record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        result = optional(expected_json)
        if result is not None:
            record.update(result=result, result_sha256=sha(expected_json)); record['status'] = 'ACTUAL_REDUCER_RESULT' if proc.returncode == 0 or proc.returncode == 2 and result.get('failure') else 'REDUCER_ERROR'
    except BaseException as exc: record['error'] = type(exc).__name__ + ': ' + str(exc)
    return record


def run_reducers(run, runner=subprocess.run):
    payload = run / 'payload'; output = run / 'analysis'; result = {}
    units = {u: payload / 'execution' / u for u in UNITS[:2]}
    has_raw = lambda p: any((p / 'raw').glob('*.bin')) or any((p / 'intermediates').glob('*.bin'))
    if all(has_raw(p) for p in units.values()):
        destination = output / 'retained_pair'; command = [sys.executable, '-B', str(run / 'scripts/reduce_retained_pairs.py'), '--root', str(payload), '--output', str(destination)]
        result['pair'] = reducer_once(command, destination, destination / 'FRESH_PROCESS_PARITY.json', runner)
    else: result['pair'] = dict(status='NOT_RUN_NO_ACTUAL_TWO_UNIT_RAW', retry=False)
    for unit, path in units.items():
        if any((path / 'intermediates').glob('TOKEN_*_LAYER_*.json')) and any((path / 'intermediates').glob('*.bin')):
            destination = output / ('RETAINED_FP64_' + unit + '.json'); command = [sys.executable, '-B', str(run / 'scripts/replay_retained_fp64.py'), '--root', str(payload), '--unit', unit, '--output', str(destination)]
            result[unit] = reducer_once(command, destination, destination, runner)
        else: result[unit] = dict(status='NOT_RUN_NO_ACTUAL_COMMITTED_INTERMEDIATE_RAW', retry=False)
    return result


def decide(units, reducers, resources, controller, first_failure):
    completed = [x for x in units if x['status'] == 'PASS_COMPLETE_UNIT_IDENTITY_AND_RETAINED_METADATA']
    numeric = all(any(x['unit'] == u for x in completed) for u in UNITS[:2])
    pair = reducers['pair'].get('result', {}); parity = reducers['pair'].get('status') == 'ACTUAL_REDUCER_RESULT' and pair.get('status') == 'PASS_FULL_32TOKEN_RETAINED_CPU_PAIR_COMPARISON' and pair.get('complete_pair') is True and pair.get('all_byte_equal') is True and pair.get('all_torch_equal') is True and pair.get('actual_token_pairs') == 32 and pair.get('actual_intermediate_pairs') == 768
    replay = all(reducers[u].get('status') == 'ACTUAL_REDUCER_RESULT' and reducers[u].get('result', {}).get('status') == 'PASS_FULL_32TOKEN_24LAYER_RETAINED_FP64_REPLAY' and reducers[u]['result'].get('complete_32token_24layer_audit') is True and reducers[u]['result'].get('actual_records_audited') == 768 for u in UNITS[:2])
    legal = [x['measurement'] for x in completed if x['unit'].startswith('OLD_FP_') and x.get('measurement')]
    median = statistics.median(x['tokens_per_second'] for x in legal) if len(legal) == 3 else None
    median_ms = statistics.median(x['wall_ms_per_token'] for x in legal) if len(legal) == 3 else None
    controller_ok = controller is not None and controller.get('status') == 'COMPLETE_FINITE_RECOVERY_UNITS'
    clean_exits = all(any(r['unit'] == u and r.get('worker_returncode') == 0 for r in resources.get('units', [])) for u in UNITS[:5])
    numerical_failure = bool(pair.get('earliest_difference') or pair.get('earliest_metadata_difference')) or any(reducers[u].get('result', {}).get('failure') for u in UNITS[:2]) or any(x.get('earliest_divergence') or x.get('earliest_numerical_failure') for x in units)
    identity_failure = any(x.get('driver') and x['driver'].get('status') != 'PASS' or any(x.get(p, {}).get('own_GPU') and x[p]['own_GPU'].get('status') != 'PASS' or x.get(p, {}).get('library') and x[p]['library'].get('status') != 'PASS' for p in ['PREFLIGHT', 'POSTFLIGHT']) for x in units)
    qualifies = numeric and parity and replay and len(legal) == 3 and resources['status'] == 'PASS_EVENT_DERIVED_INTERVALS' and clean_exits and controller_ok and not first_failure
    if qualifies: status = 'HISTORICAL_PORT_FP_QUALIFIED_SPEED_RECOVERED' if median >= 13 else 'HISTORICAL_PORT_FP_QUALIFIED_PARTIAL_SPEED'
    elif numerical_failure: status = 'FP_NUMERICAL_QUALIFICATION_FAIL'
    elif identity_failure: status = 'GPU_RESOURCE_OR_IDENTITY_FAIL'
    elif first_failure or controller and controller.get('status') == 'FAIL_CLOSED_STOP': status = 'FAIL_CLOSED_STOP'
    else: status = 'PARTIAL_GPU_RESULTS_PRESERVED_AND_STOP'
    return dict(task_id=TASK, status=status, execution_stop=bool(first_failure or not controller_ok), restored_protocol='HISTORICAL_PORT', original_online_numeric_complete=numeric, full_retained_pair_pass=parity, full_retained_FP64_replay_pass=replay, mandatory_own_processes_exited_zero=clean_exits, legal_OLD_repeats=len(legal), current_OLD_three_repeat_median_TPS=median, current_OLD_three_repeat_median_ms_per_token=median_ms, engineering_target_TPS=13.0, engineering_target_met=bool(qualifies and median >= 13), FP_short_protocol_qualified=qualifies, INT8_GPU_QUALIFICATION='NOT_RUN', long_horizon_qualification='NOT_RUN_NOT_QUALIFIED', future_Group32_AIME_or_training_authorized=False, historical_OLD_FP_TPS_reference_only=13.742036038730332, historical_NEW_FP_TPS_reference_only=7.77671073754353, new_measurements_never_filled_from_history=True)


def scientific_report(decision, units, reducers, resources, binding, first_failure):
    rows = [x for x in units if x.get('measurement')]; median = decision['current_OLD_three_repeat_median_TPS']
    numerical = '完整32-token两fresh与保留raw复核通过' if decision['FP_short_protocol_qualified'] else '未满足完整资格；详见实际部分结果、首次失败与CPU reducer回执'
    lines = [f"本轮状态：**{decision['status']}**。恢复协议为 HISTORICAL_PORT，历史二进制 EXACT 身份仍 UNKNOWN。", '', '十个问题逐项回答：', '',
             '1. **是否使用最新有效源码？** 本轮源清单407项逐SHA/bytes核验，其中403项继承、控制器/观察worker/计划/新请求4项明示修订；父有效清单SHA为 '+PARENT+'。',
             '2. **真实Driver/NVML身份通过了吗？** 每个实际派发单位的Driver与PRE/POST own-PID/库身份逐项保留；未完成或失败单位保持未通过。全部自有进程退出与所有UUID释放由外部root证明准入，本工具不查询GPU。',
             '3. **是否执行真实9B FP？** '+('至少有实际完成且身份完整的FP单位。' if any(x['status'].startswith('PASS_COMPLETE') for x in units) else '没有可宣称完成资格的单位；不把调度或部分raw当完整9B成功。'),
             '4. **两次fresh是否逐token逐Tensor一致？** '+str(reducers['pair'].get('result', {}).get('status', reducers['pair']['status']))+'；torch.equal、byte、maxabs/maxrel、首token/layer/head来自CPU对实际保留raw的比较，signed-zero字节差与数值相等分开。',
             '5. **真实激活FP64递推是否合格？** '+numerical+'。每unit需32×24的9类真实中间态与独立FP64从token0 carry重放；不是全模型FP64或32K/81K误差证明。',
             '6. **当前128+128实际TPS？** '+('三次合法OLD中位 '+str(median)+' TPS，逐次数据见下表。' if median is not None else '未完成三次合法OLD测量；不报告三重复中位，实际合法部分单独保留。'),
             '7. **达到约13TPS了吗？** '+('达到预设工程目标。' if decision['engineering_target_met'] else '当前证据不满足完整“数值资格+三合法重复中位≥13”验收；不借用历史13.742。'),
             '8. **若未达到，原因是什么？** 首次失败、未完整单位及reducer错误见结构化回执；身份/数值/资源/观测不足分开，不能将未测或失败解释为速度确实慢。首失败：'+('有实际FIRST_FAILURE，原文原位保留。' if first_failure else '没有FIRST_FAILURE文件；不虚构失败原因。'),
             '9. **能否用于后续科研？** '+('仅本次已测FP短前缀与短缓存窗口具有对应证据，仍需后续独立授权和科学合同。' if decision['FP_short_protocol_qualified'] else '当前不批准将未完整/失败结果用于后续正式科学结论。'),
             '10. **Group32还缺什么？** H+C128/C32 Sym/Asym分组、scale/zero-point、prefill/QDQ/writeback/next-consumption及persistent shadow须独立冻结、同恢复协议FP基线匹配并实际GPU资格；本次INT8/AIME/长Future-KL/训练均未运行。', '', '实际普通短窗观察：', '', '| 单位 | 合法身份完整 | TPS | ms/token | 128-token wall秒 |', '|---|---|---:|---:|---:|']
    for unit in rows:
        m = unit['measurement']; lines.append('| '+unit['unit']+' | '+str(unit['status'].startswith('PASS_COMPLETE'))+' | '+str(m['tokens_per_second'])+' | '+str(m['wall_ms_per_token'])+' | '+str(m['decode_wall_seconds'])+' |')
    if not rows: lines.append('当前没有实际measurement回执，因此没有科学测量CSV。')
    lines += ['', '资源采用首次dispatch、首次退出、首次所有UUID释放事件；进程预留GPUh区间为 '+str(resources['cumulative_reservation_GPU_hours_lower'])+' 至 '+str(resources['cumulative_reservation_GPU_hours_upper'])+'。精确物理GPUh为 **UNRESOLVED**；NVML owned-PID峰仅为采样下界，不能称连续精确峰。当前无GPU时间硬上限，仍执行1800秒真实无进展watchdog及8GiB raw/10GiB磁盘保留。', '', 'MODEL_LOAD_TIMING记录原loader/reference调用wall，初始化/reset及同步边界分别列出；WORKER_CUDA_ALLOCATOR_OBSERVATION汇总显式reset前epochs及结束计数，原RESULT的warmup/decode peak保持原值。它覆盖进程内观察阶段，不冒充进程退出之后的遥测。', '', '成功中间态包括 query/key/value/gate/beta/core_actual/state_actual/native_norm_query/native_norm_key，原cache/logits raw另保留；捕获完成不是数学PASS，必须结合独立重放。', '', '历史13.742036038730332和7.77671073754353仅是历史同窗背景，不是本轮数据，也不证明INT8速度。原CPU530包与控制修订51包保持不变。本报告不封存、不删raw、不重试GPU或CPU reducer，不创建空科学CSV。', '']
    return '\n'.join(lines)


def finish(run, proof_path, runner=subprocess.run):
    run = Path(run).resolve(); payload = run / 'payload'; binding = source_binding(payload); proof = release_admission(run, proof_path, binding)
    start = run / 'analysis/SCIENTIFIC_FINISHER_STARTED.json'; require(not start.exists(), 'Finisher already attempted: inspect artifacts, never retry blindly')
    require(not any((run / n).exists() for n in ['FINAL_SCIENTIFIC_REPORT.md', 'RUNTIME_RESTORATION_DECISION.json', 'RESOURCE_LEDGER.json']), 'Scientific outputs exist: no overwrite')
    new_json(start, dict(task_id=TASK, status='CPU_RETAINED_DATA_FINISHER_STARTED_ONCE', release_proof=descriptor(proof_path, run), source_freeze_sha256=binding['identities']['source_freeze_sha256'], finisher_sha256=sha(Path(__file__)), GPU_dispatches=0, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    reducers = run_reducers(run, runner); new_json(run / 'analysis/RETAINED_REDUCER_INVOCATIONS.json', reducers)
    observations = []
    for unit in UNITS:
        try: observations.append(audit_unit(payload, unit, binding))
        except BaseException as exc: observations.append(dict(unit=unit, status='ERROR_READING_ACTUAL_UNIT_EVIDENCE', issues=[type(exc).__name__+': '+str(exc)], result=None))
    ledger_rows, parse_errors = events(payload / 'execution/RESOURCE_EVENTS.jsonl'); resources = resource_summary(ledger_rows, parse_errors)
    dispatched = {u: read(payload / 'execution' / u / 'DISPATCH.json')['worker_identity'] for u in UNITS if (payload / 'execution' / u / 'DISPATCH.json').is_file()}
    actual_intervals = {x['unit']: x['worker_identity'] for x in resources['units']}
    if dispatched != actual_intervals:
        resources['issues'].append(dict(issue='Actual DISPATCH roster and reservation ledger differ', dispatched=dispatched, intervals=actual_intervals)); resources['status'] = 'UNRESOLVED_OR_INVALID_RESOURCE_EVENTS'
    terminal = optional(payload / 'execution/CONTROLLER_RESULT.json'); failure = optional(payload / 'FIRST_FAILURE.json'); decision = decide(observations, reducers, resources, terminal, failure)
    new_json(run / 'analysis/SOURCE_AND_RUNTIME_BINDINGS.json', binding); new_json(run / 'analysis/ACTUAL_UNIT_EVIDENCE_AUDIT.json', observations); new_json(run / 'RESOURCE_LEDGER.json', resources); new_json(run / 'RUNTIME_RESTORATION_DECISION.json', decision)
    measurement_observations = [dict(unit=x['unit'], legal_identity_complete=x['status'].startswith('PASS_COMPLETE'), measurement=x.get('measurement'), issues=x['issues'], loader_timing=x.get('loader_timing'), allocator_observation=x.get('allocator_observation')) for x in observations if x.get('measurement')]
    new_json(run / 'benchmarks/BENCHMARK_OBSERVATIONS.json', measurement_observations)
    for prefix, filename in [('OLD_FP_', 'RESTORED_FP_BENCHMARK.csv'), ('NEW_FP_', 'CURRENT_NEW_REFERENCE.csv')]:
        measured = [x['measurement'] for x in observations if x['unit'].startswith(prefix) and x['status'].startswith('PASS_COMPLETE') and x.get('measurement')]
        if measured:
            p = run / 'benchmarks' / filename; p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('x', newline='') as f: writer=csv.DictWriter(f, fieldnames=list(measured[0])); writer.writeheader(); writer.writerows(measured)
    with (run / 'FINAL_SCIENTIFIC_REPORT.md').open('x') as f: f.write(scientific_report(decision, observations, reducers, resources, binding, failure))
    final = dict(task_id=TASK, status='CPU_ACTUAL_RETAINED_DATA_REPORT_COMPLETE_NOT_SEALED', decision=decision['status'], release_proof=descriptor(proof_path, run), finisher_sha256=sha(Path(__file__)), report=descriptor(run / 'FINAL_SCIENTIFIC_REPORT.md', run), GPU_dispatches=0, no_model_or_GPU_executed=True, no_archive_or_deletion=True)
    new_json(run / 'analysis/SCIENTIFIC_FINISHER_RESULT.json', final); return final


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True, type=Path); p.add_argument('--release-proof', type=Path); p.add_argument('--run', action='store_true'); args=p.parse_args(); root=args.root.resolve()
    if not args.run:
        print(json.dumps(dict(status='PREVIEW_ONLY_NO_REDUCER_NO_WRITES_NO_GPU', root=str(root), payload_exists=(root/'payload').is_dir(), release_proof_supplied=args.release_proof is not None, GPU_queries=0, GPU_dispatches=0))); return
    require(args.release_proof is not None, 'Explicit actual release proof is required before CPU reducers')
    print(json.dumps(finish(root,args.release_proof.resolve()), ensure_ascii=False, allow_nan=False))


if __name__=='__main__': main()
