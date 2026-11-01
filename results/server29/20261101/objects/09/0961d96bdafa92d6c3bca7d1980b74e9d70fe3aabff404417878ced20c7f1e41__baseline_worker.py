#!/usr/bin/env python3
"""Phase-0 fixed-token measurements of immutable, qualified 32K reference math.

No optimized codec, sampler, scorer or AIME input is present. The original
core/configure/load, Hadamard, C128, Dual-M and Observe AST remain unchanged.
Only orchestration and separately labelled measurement instrumentation are new.
"""
import argparse
import ast
import collections
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
CONDITIONS = ('FP_NATIVE', 'H_C128', 'H_DUAL_M_REFERENCE')
LENGTHS = (128, 512, 2048)
WARMUP = 128
GPU_UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
CORE_SHA = 'aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575'
WORKER_SHA = 'e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773'
CODEC_SHA = 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f'
ENCODE_SHA = 'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6'


def check(ok, why):
    if not bool(ok):
        raise RuntimeError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canon(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_new(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def manifest(root, path, digest):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    check(sha(path) == digest, 'Manifest identity changed: ' + str(path))
    obj = read(path)
    rows = obj.get('execution_sources', obj.get('files', [])) + obj.get('external_dependencies', [])
    check(rows, 'Empty dependency manifest: ' + str(path))
    for row in rows:
        p = Path(row['path'])
        if not p.is_absolute():
            p = root / p
        check(p.is_file() and sha(p) == row['sha256'], 'Source dependency changed: ' + str(p))
        if 'bytes' in row:
            check(p.stat().st_size == row['bytes'], 'Dependency bytes changed: ' + str(p))
    return obj


def baseline_schedule():
    rows = []
    for j, length in enumerate(LENGTHS):
        for repeat in range(3):
            offset = (j + repeat) % 3
            for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
                rows.append({'condition': condition, 'length': length, 'repeat': repeat})
    return rows


def admission(root, args):
    check(not (root / 'FIRST_FAILURE.json').exists(), 'Prior failure prohibits further GPU work')
    auth = read(args.authorization)
    check(auth.get('status') == 'PASS' and auth.get('task_id') == TASK, 'Explicit Phase0 PASS authorization required')
    check(args.mode in auth.get('allowed_modes', []), 'Mode not authorized')
    check(auth['gpu'] == 0 and auth['gpu_uuid'] == GPU_UUID and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Only fixed physical GPU0 allowed')
    check(sha(root / 'PLAN.json') == auth['plan_sha256'], 'Measurement plan changed')
    sources = manifest(root, 'SOURCE_MANIFEST.json', auth['source_manifest_sha256'])
    check(any((root / row['path']).resolve() == Path(__file__).resolve()
              for row in sources.get('execution_sources', [])), 'Baseline worker is not source-manifest bound')
    check(sha(root / 'REFERENCE_BINDING.json') == auth['reference_binding_sha256'], 'Reference binding changed')
    plan = read(root / 'PLAN.json'); binding = read(root / 'REFERENCE_BINDING.json')
    check(plan['task_id'] == TASK and plan['warmup_tokens'] == WARMUP and plan['lengths'] == list(LENGTHS) and plan['repeats'] == 3,
          'Fixed measurement lengths/repeats/warmup changed')
    check(plan['baseline_schedule'] == baseline_schedule(), 'Balanced baseline schedule changed')
    check(binding['status'] == 'PASS', 'Reference identity not PASS')
    for key, digest in (('core', CORE_SHA), ('worker', WORKER_SHA), ('codec', CODEC_SHA)):
        row = binding[key]; p = Path(row['path'])
        check(p.is_absolute() and row['sha256'] == digest and sha(p) == digest and p.stat().st_size == row['bytes'], 'Immutable 32K source changed: ' + key)
    check(binding['manifests'], 'Reference dependency manifest closure is required')
    for row in binding['manifests']:
        manifest(Path(row.get('root', str(Path(row['path']).parent))), row['path'], row['sha256'])
    fixture = binding['fixture']
    check(fixture['non_AIME'] is True and fixture['role'] == 'FROZEN_PROSPECTIVE_DOC1_INPUT_PREFIX', 'Legal non-AIME fixture missing')
    p = Path(fixture['tokens_path'])
    if not p.is_absolute():
        p = root / p
    check(sha(p) == fixture['tokens_file_sha256'], 'Fixed input file changed')
    ids = read(p)
    check(isinstance(ids, list) and len(ids) == fixture['length'] and len(ids) >= WARMUP + max(LENGTHS), 'Fixed-token fixture too short')
    check(all(type(x) is int and 0 <= x < 248320 for x in ids), 'Invalid fixed input tokens')
    check(hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == fixture['tokens_canonical_sha256'], 'Fixed token canonical identity changed')
    return auth, plan, binding, ids


def import_pinned(name, row):
    spec = importlib.util.spec_from_file_location(name, row['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(root, binding):
    # All GPU-capable imports occur after full admission. Historical paths are
    # read-only: bytecode writes are disabled globally, all save callbacks new.
    K = import_pinned('_speedup_reference_core', binding['core'])
    C = import_pinned('_speedup_reference_codec', binding['codec'])
    K.ROOT = root
    K.save = lambda rel, obj: atomic_new(root / 'reference_failures' / rel, obj)
    ns = {'K': K, 'C': C, 'torch': K.torch, 'collections': collections, 'CURRENT': {}}
    tree = ast.parse(Path(binding['worker']['path']).read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in {'Observe', 'dual'}]
    check({node.name for node in nodes} == {'Observe', 'dual'}, 'Missing original reference AST')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), binding['worker']['path'], 'exec'), ns)
    K.configure()
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    rows = [line.split(', ') for line in actual.strip().splitlines()]
    selected = [row for row in rows if row[0] == '0']
    check(len(selected) == 1 and selected[0][1] == GPU_UUID and '3090' in selected[0][2], 'Physical GPU UUID/class changed')
    model, qmod, env = K.load()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    own = [line.split(',') for line in active.strip().splitlines() if line.split(',')[0].strip() == str(os.getpid())]
    check(len(own) == 1 and own[0][1].strip() == GPU_UUID, 'Live PID is not on authorized physical GPU')
    check(env['parameter_dtypes'] == ['torch.bfloat16'], 'Reference model dtype changed')
    env.update(gpu_uuid=GPU_UUID, physical_gpu=0, CUDA_VISIBLE_DEVICES='0', deterministic_algorithms=True,
               TF32=False, chunk_binding='original qualified compensated', observer=False,
               no_new_global_synchronize_per_token=True, reference_internal_event_synchronize_preserved=True)
    return K, C, ns, model, qmod, env


def summary(values):
    check(values, 'No timing observations')
    ordered = sorted(values)
    return {'count': len(values), 'min': ordered[0], 'max': ordered[-1], 'mean': math.fsum(values) / len(values),
            'p50': statistics.median(values), 'p95': ordered[max(0, math.ceil(.95 * len(values)) - 1)]}


def profile_rows(profiler):
    rows = []
    for event in profiler.key_averages():
        rows.append({'name': event.key, 'count': event.count,
                     'cpu_total_us': event.cpu_time_total, 'self_cpu_us': event.self_cpu_time_total,
                     'device_total_us': getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0.)),
                     'self_device_us': getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0.)),
                     'cpu_memory_bytes': event.cpu_memory_usage,
                     'device_memory_bytes': getattr(event, 'device_memory_usage', getattr(event, 'cuda_memory_usage', 0))})
    activities = []
    for event in profiler.events():
        if str(event.device_type).endswith('CUDA'):
            activities.append({'name': event.name, 'start_us': event.time_range.start, 'end_us': event.time_range.end})
    intervals = sorted((row['start_us'], row['end_us']) for row in activities)
    union = 0.; end = None
    for start, finish in intervals:
        if end is None or start > end:
            union += finish - start
        else:
            union += max(0., finish - end)
        end = max(finish, end) if end is not None else finish
    kernels = [row for row in activities if not any(x in row['name'].lower() for x in ('memcpy', 'memset', '[memory]'))]
    grouped = collections.defaultdict(list)
    for row in activities:
        grouped[row['name']].append(row['end_us'] - row['start_us'])
    device_costs = [{'name': name, 'count': len(values), 'summed_duration_us': math.fsum(values),
                     'duration_summary_us': summary(values),
                     'classification': 'MEMORY_ACTIVITY' if any(x in name.lower() for x in ('memcpy', 'memset', '[memory]')) else 'KERNEL_ACTIVITY'}
                    for name, values in sorted(grouped.items())]
    return {'operators': rows, 'CUDA_device_activity_count': len(activities), 'CUDA_kernel_activity_count_excluding_memcpy_memset': len(kernels),
            'CUDA_device_activity_costs_by_name': device_costs,
            'CUDA_active_interval_union_us': union, 'CUDA_activity_duration_sum_us': math.fsum(row['end_us'] - row['start_us'] for row in activities),
            'CUDA_activity_sum_is_not_wall_time': True,
            'limitations': 'Kineto device intervals include kernels/copies; union is observed device-active time. CPU wall minus this union is not a pure CPU-dispatch attribution.'}


def replay(K, C, ref, model, qmod, gpu_ids, condition, length, repeat, instrumented, root, mode, stage):
    torch = K.torch; f = K.formal(True); cache = None; obs = ref['Observe'](model, qmod, f, False)
    hcontext = f.QwenKeyHadamardPatch(model) if condition != 'FP_NATIVE' else contextlib.nullcontext()
    wall_samples = []; event_samples = []; native_calls = 0; writebacks = 0; qdq_calls = 0
    scale_counts = collections.Counter(); codec_timings = []; fixture = None; measured = False; prof = None
    tag = condition + '_L' + str(length) + '_R' + str(repeat)
    random.seed(20261015); torch.manual_seed(20261015); torch.cuda.manual_seed_all(20261015)
    torch.cuda.reset_peak_memory_stats()
    begun = time.monotonic(); window_wall = None; stream_ms = None
    try:
        with hcontext as h, torch.inference_mode():
            for t in range(WARMUP + length):
                if t == WARMUP:
                    # Warmup state is retained; never rerun the measured prefix.
                    if instrumented and condition == 'H_DUAL_M_REFERENCE':
                        fixture = f.get_state(cache, f.GDN_LAYERS[0]).detach().clone()
                    torch.cuda.synchronize()
                    if instrumented:
                        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                                      record_shapes=False, profile_memory=True, with_stack=False)
                        prof.__enter__()
                        window_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        window_events[0].record()
                    window_start = time.monotonic(); measured = True
                obs.position = t; obs.details = {}; obs.finite = []
                ref['CURRENT'].update(tag=tag, condition=condition, token_index=t, consumed=t + 1, role='FIXED_TOKEN_REPLAY_PROFILE')
                tick = time.monotonic()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if measured and instrumented else None
                if events: events[0].record()
                scope = torch.profiler.record_function if measured and instrumented else lambda _name: contextlib.nullcontext()
                with scope('REFERENCE/model_forward'):
                    out = model(input_ids=gpu_ids[:, t:t + 1], past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                with scope('REFERENCE/finite_state_and_logits_gate'):
                    K.check(torch.stack(obs.finite).all() & torch.isfinite(out.logits).all(), 'Nonfinite state/logits')
                with scope('REFERENCE/QDQ_and_original_safety'):
                    if condition == 'H_C128':
                        audit = {'calls': 0, 'layers': set(), 'scale_shapes': {}, 'code_min': 0, 'code_max': 0}
                        f.quantize_c128(cache, audit)
                        K.check(audit['calls'] == 1 and audit['layers'] == set(f.GDN_LAYERS), 'Original C128 writeback audit failed')
                        writebacks += 24; qdq_calls += 1
                    elif condition == 'H_DUAL_M_REFERENCE':
                        _, stat, timing = ref['dual'](cache, f, trace=False, metrics=False, parity=False)
                        writebacks += 24; qdq_calls += 1
                        if measured:
                            codec_timings.append(timing)
                            for key in ('raw_row_bound_hits', 'raw_column_bound_hits', 'row_floor_hits', 'row_ceiling_hits', 'column_floor_hits', 'column_ceiling_hits'):
                                scale_counts[key] += stat[key]
                obs.previous = {l: (f.get_state(cache, l).data_ptr(), None) for l in f.GDN_LAYERS} if condition != 'FP_NATIVE' else {}
                for l in f.GDN_LAYERS:
                    K.check(obs.calls[(t, l)] == 1, 'Not exactly one original GDN call per token/layer')
                native_calls += 24
                del out
                if events:
                    events[1].record(); event_samples.append(events)
                if measured:
                    wall_samples.append(time.monotonic() - tick)
                if (t + 1) % 128 == 0:
                    print(json.dumps({'event': 'PROGRESS', 'mode': mode, 'measurement_stage': stage, 'condition': condition,
                                      'length': length, 'repeat': repeat, 'consumed_tokens': t + 1, 'measured_tokens': max(0, t + 1 - WARMUP),
                                      'elapsed_seconds': time.monotonic() - begun}), flush=True)
            if instrumented:
                window_events[1].record()
            torch.cuda.synchronize()
            window_wall = time.monotonic() - window_start
            if instrumented:
                prof.__exit__(None, None, None)
                stream_ms = window_events[0].elapsed_time(window_events[1])
            n = WARMUP + length
            K.check(len(obs.calls) == 24 * n and native_calls == 24 * n, 'Missing native forward calls')
            K.check(int(cache.get_seq_length()) == n, 'Full-attention cache cursor changed')
            ha = {'calls': h.calls, 'layers': sorted(h.layers), 'phase_counts': dict(h.phase_counts)} if condition != 'FP_NATIVE' else None
            if ha:
                K.check(ha['calls'] == 24 * n and ha['phase_counts'] == {'chunk_prefill': 24, 'recurrent_decode': 24 * (n - 1)}, 'Original H boundary changed')
            profile = profile_rows(prof) if instrumented else None
            result = {'status': 'PASS', 'task_id': TASK, 'role': 'PHASE0_DIAGNOSTIC_NOT_NEW_SCIENCE',
                      'condition': condition, 'length': length, 'repeat': repeat, 'instrumented': instrumented, 'measurement_stage': stage,
                      'warmup_tokens': WARMUP, 'consumed_tokens': n, 'measured_decode_tokens': length,
                      'prefill_protocol': 'Original single first input token; QDQ immediately after every forward including first. Retain128-token warmup cache; measure subsequent fixed tokens.',
                      'whole_replay_wall_seconds': time.monotonic() - begun,
                      'decode_window_wall_seconds': window_wall, 'decode_tokens_per_second': length / window_wall,
                      'per_token_wall_dispatch_seconds': wall_samples, 'per_token_wall_dispatch_summary_seconds': summary(wall_samples),
                      'per_token_wall_scope': 'Host iteration elapsed incl actual reference safety waits; not independently synchronized GPU-completion latency. Whole window synchronized only at its boundaries.',
                      'CUDA_stream_window_ms': stream_ms,
                      'per_token_CUDA_stream_ms': [events[0].elapsed_time(events[1]) for events in event_samples],
                      'CUDA_event_scope': 'Instrumented only; ordered same-stream elapsed intervals include device idle gaps between endpoints, not pure kernel-active duration.',
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'persistent_recurrent_state_tensor_bytes': sum(state.numel() * state.element_size() for state in K.states(cache, f)),
                      'native_core_calls': native_calls, 'QDQ_calls': qdq_calls, 'layer_writebacks': writebacks,
                      'measured_QDQ_calls': length if condition != 'FP_NATIVE' else 0,
                      'next_token_consumption_checks': 24 * (n - 1) if condition != 'FP_NATIVE' else 0,
                      'consumption_policy': 'Immutable original Observe(on=False) checks actual initial_state.data_ptr against preceding post-QDQ carrier on each layer/token; no per-token CPU tensor hash.',
                      'hadamard_audit': ha, 'scale_safety_totals_measured_window': dict(scale_counts),
                      'original_DualM_event_timings_per_token_ms': codec_timings,
                      'reference_internal_synchronization': 'Preserved: finite/safety scalar checks, original DualM events[4].synchronize. No new per-token torch.cuda.synchronize.',
                      'profile': profile, 'sampling': 'NOT_RUN_FIXED_TEACHER_FORCED_TOKENS', 'AIME': 'NOT_RUN'}
            if instrumented:
                trace_path = root / 'profiles' / (tag + '.trace.json')
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                check(not trace_path.exists(), 'Profiler trace already exists')
                prof.export_chrome_trace(str(trace_path))
                result['chrome_trace'] = {'path': str(trace_path.relative_to(root)), 'sha256': sha(trace_path), 'bytes': trace_path.stat().st_size}
            return result, fixture
    finally:
        if prof is not None and getattr(prof, 'profiler', None) is not None and window_wall is None:
            prof.__exit__(*sys.exc_info())
        obs.close()
        cache = None; obs = None
        gc.collect(); torch.cuda.empty_cache()


def codec_stage_profile(torch, C, fixture):
    """Observe original line boundaries; never replace a numerical statement."""
    source_path = Path(C.encode.__code__.co_filename)
    check(sha(source_path) == ENCODE_SHA, 'Original encode source changed before stage diagnosis')
    fn = next(node for node in ast.parse(source_path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'encode')
    check(hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == C.ENCODE_AST_SHA256,
          'Executed encode AST differs from original stage reference')
    tail = fn.body[-9:]
    names = [node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else 'return' if isinstance(node, ast.Return) else None for node in tail]
    check(names == ['r', 'scaled', 'c', 'z', 'v', 'den', 'c', 'z', 'return'], 'Frozen DualM stage boundaries changed')
    boundaries = {fn.body[0].lineno: 'INPUT_VALIDATION', tail[0].lineno: 'ROW', tail[1].lineno: 'INITIAL_COLUMN',
                  tail[3].lineno: 'INITIAL_LEVELS', tail[4].lineno: 'LS', tail[7].lineno: 'FINAL_LEVELS', tail[8].lineno: 'VALIDATE'}
    scope = [None]; reached = []
    def close():
        if scope[0] is not None:
            scope[0].__exit__(None, None, None); scope[0] = None
    def local(frame, event, arg):
        if event == 'line' and frame.f_lineno in boundaries:
            close(); label = boundaries[frame.f_lineno]; reached.append(label)
            scope[0] = torch.profiler.record_function('CODEC_STAGE/' + label); scope[0].__enter__()
        elif event in ('return', 'exception'):
            close()
        return local
    def globaltrace(frame, event, arg):
        return local if frame.f_code is C.encode.__code__ else None
    check(sys.gettrace() is None, 'Refuse nested Python trace instrumentation')
    activities = [torch.profiler.ProfilerActivity.CPU]
    if fixture.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA); torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True, with_stack=False) as prof:
        start = time.monotonic(); sys.settrace(globaltrace)
        try:
            encoded = C.encode(fixture, 'DUAL_M_INT8')
        finally:
            sys.settrace(None); close()
        with torch.profiler.record_function('CODEC_STAGE/DECODE'):
            decoded = encoded.decode()
        if fixture.is_cuda:
            torch.cuda.synchronize()
        wall = time.monotonic() - start
    expected_stages = ['INPUT_VALIDATION', 'ROW', 'INITIAL_COLUMN', 'INITIAL_LEVELS', 'LS', 'FINAL_LEVELS', 'VALIDATE']
    check(reached == expected_stages, 'Incomplete original encoder stage observation')
    costs = profile_rows(prof)
    stages = [row for row in costs['operators'] if row['name'].startswith('CODEC_STAGE/')]
    ls = next(row for row in stages if row['name'] == 'CODEC_STAGE/LS')
    ls_device = ls['device_total_us'] if fixture.is_cuda and ls['device_total_us'] > 0 else None
    report = {'status': 'PASS_INSTRUMENTATION_ONLY_PENDING_OUTPUT_PARITY', 'stage_boundaries_original_line_numbers': boundaries,
              'stages': stages, 'profile': costs, 'instrumented_wall_seconds': wall,
              'LS_device_attributed_us': ls_device,
              'LS_device_attribution_status': 'MEASURED_GROUPED_DIAGNOSTIC_NOT_WALL' if ls_device is not None else 'UNKNOWN_NOT_MEASURED',
              'stage_scopes': {'ROW': 'r=safe_half(abs.mean.clamp.sqrt)', 'INITIAL_COLUMN': 'scaled=x/r.float plus initial column safe_half',
                               'INITIAL_LEVELS': 'initial levels(scaled/c.float)',
                               'LS': 'v=r.float*z; denominator; numerator/guarded division/where; fitted FP16 safe_half, exactly original statements',
                               'FINAL_LEVELS': 'final levels(scaled/c.float)',
                               'VALIDATE': 'INT8 materialization/clone plus EncodedState.validate; not validation alone'},
              'limitations': 'sys.settrace and record_function add CPU overhead. Device attribution groups original operations and is not stage wall latency; missing CUDA attribution remains UNKNOWN. Do not sum nested stage/aten device times as overall wall.'}
    return encoded, decoded, report


def codec_micro(K, C, fixture):
    """Actual single-layer H/QDQ state, immutable encode/decode, no stage rewrite."""
    torch = K.torch
    check(fixture.dtype == torch.float32 and tuple(fixture.shape) == (1, 32, 128, 128), 'Real codec fixture shape/dtype changed')
    torch.cuda.synchronize()
    fixture_sha = K.tsha(fixture)
    rows = []
    for repeat in range(4):
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode():
            enc = C.encode(fixture, 'DUAL_M_INT8'); middle = time.monotonic(); qdq = enc.decode()
            K.check(torch.isfinite(qdq).all(), 'Nonfinite codec-only reconstruction')
        torch.cuda.synchronize(); end = time.monotonic()
        if repeat:
            rows.append({'repeat': repeat - 1, 'encode_host_seconds_including_reference_checks': middle - start,
                         'encode_decode_finite_complete_wall_seconds': end - start,
                         'encoded_tensor_bytes': enc.nbytes(), 'codes_dtype': str(enc.codes.dtype),
                         'rows_dtype': str(enc.rows.dtype), 'columns_dtype': str(enc.columns.dtype)})
        if repeat != 3:
            del enc, qdq
    observed, observed_qdq, diagnostic = codec_stage_profile(torch, C, fixture)
    parity = {}
    for name, expected, actual in (('codes', enc.codes, observed.codes), ('rows', enc.rows, observed.rows),
                                   ('columns', enc.columns, observed.columns), ('QDQ', qdq, observed_qdq)):
        expected_sha = K.tsha(expected); actual_sha = K.tsha(actual)
        check(expected.dtype == actual.dtype and tuple(expected.shape) == tuple(actual.shape) and expected_sha == actual_sha,
              'Stage instrumentation changed original encoder bytes: ' + name)
        parity[name] = {'status': 'BYTE_EXACT', 'sha256': actual_sha, 'dtype': str(actual.dtype), 'shape': list(actual.shape)}
    diagnostic.update(status='PASS_BYTE_EXACT_INSTRUMENTED_VS_ORIGINAL', output_parity=parity)
    del enc, qdq, observed, observed_qdq
    return {'status': 'PASS', 'role': 'REAL_STATE_SINGLE_LAYER_DIAGNOSTIC_CODEC_ONLY',
            'fixture_source': 'First H-only GDN layer after original QDQ at warmup token128 of this new non-AIME fixed replay; not an FP teacher or new scientific result',
            'fixture_shape': [1, 32, 128, 128], 'fixture_dtype': 'torch.float32', 'fixture_sha256': fixture_sha,
            'fixture_bytes': fixture.numel() * fixture.element_size(), 'rows': rows,
            'stage_diagnostic': diagnostic,
            'limitations': 'Single layer encode/decode excludes24-layer concatenation, scatter writebacks and worker-level raw-scale risk audit. Full-model reference measurements include all of them.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--mode', choices=('profile', 'baseline'), required=True)
    args = parser.parse_args(); root = args.root.resolve(); phase = root / 'execution' / args.mode
    try:
        auth, plan, binding, ids = admission(root, args)
        atomic_new(phase / 'STARTED.json', {'task_id': TASK, 'mode': args.mode, 'pid': os.getpid(), 'unix': time.time(),
                   'authorization_sha256': sha(args.authorization), 'plan_sha256': auth['plan_sha256']})
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
        K, C, ref, model, qmod, env = reference(root, binding)
        torch = K.torch
        gpu_ids = torch.tensor([ids[:WARMUP + max(LENGTHS)]], dtype=torch.long, device='cuda:0')
        atomic_new(phase / 'ENVIRONMENT.json', env)
        schedule = ([dict(unit, instrumented=False, stage='SHORT_NONINSTRUMENTED_BASELINE') for unit in baseline_schedule() if unit['length'] == 128]
                    + [{'condition': c, 'length': 128, 'repeat': 0, 'instrumented': True, 'stage': 'INSTRUMENTED_PROFILE'} for c in CONDITIONS]
                    if args.mode == 'profile' else [dict(unit, instrumented=False, stage='FULL_BASELINE') for unit in baseline_schedule()])
        completed = []; micro = None
        for unit in schedule:
            print(json.dumps({'event': 'UNIT_START', 'mode': args.mode, **unit}), flush=True)
            row, fixture = replay(K, C, ref, model, qmod, gpu_ids, unit['condition'], unit['length'], unit['repeat'], unit['instrumented'], root, args.mode, unit['stage'])
            row.update(reference_binding_sha256=auth['reference_binding_sha256'], source_manifest_sha256=auth['source_manifest_sha256'],
                       plan_sha256=auth['plan_sha256'], input_ids_sha256=binding['fixture']['tokens_canonical_sha256'])
            name = unit['condition'] + '_L' + str(unit['length']) + '_R' + str(unit['repeat']) + '.json'
            path = phase / ('short_baseline/records' if unit['stage'] == 'SHORT_NONINSTRUMENTED_BASELINE' else 'records') / name
            atomic_new(path, row)
            completed.append({'unit': unit, 'path': str(path.relative_to(root)), 'sha256': sha(path), 'bytes': path.stat().st_size})
            if unit['instrumented'] and fixture is not None:
                micro = codec_micro(K, C, fixture)
                atomic_new(phase / 'CODEC_MICRO.json', micro)
            del fixture
            print(json.dumps({'event': 'UNIT_COMPLETE', 'mode': args.mode, **unit,
                              'decode_wall_seconds': row['decode_window_wall_seconds'], 'tokens_per_second': row['decode_tokens_per_second']}), flush=True)
        del gpu_ids, model
        gc.collect(); torch.cuda.empty_cache()
        result = {'status': 'PASS', 'task_id': TASK, 'mode': args.mode, 'completed_units': completed,
                  'numerical_role': 'IMMUTABLE_REFERENCE_MEASUREMENT_ONLY_NO_OPTIMIZED_CANDIDATE_OR_SCIENCE',
                  'profiling_overhead': 'Instrumented profile is separate from non-instrumented baseline; never report profile TPS as baseline TPS.',
                  'authorization_sha256': sha(args.authorization), 'plan_sha256': auth['plan_sha256'],
                  'reference_binding_sha256': auth['reference_binding_sha256'], 'source_manifest_sha256': auth['source_manifest_sha256']}
        atomic_new(phase / 'RESULT.json', result)
    except BaseException:
        if not (root / 'FIRST_FAILURE.json').exists():
            atomic_new(root / 'FIRST_FAILURE.json', {'status': 'FAIL_CLOSED_STOP', 'task_id': TASK, 'mode': args.mode,
                       'pid': os.getpid(), 'unix': time.time(), 'traceback': traceback.format_exc(), 'GPU_retry': False})
        raise


if __name__ == '__main__':
    main()
