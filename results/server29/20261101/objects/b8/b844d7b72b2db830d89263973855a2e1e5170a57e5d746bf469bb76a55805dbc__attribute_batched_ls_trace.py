#!/usr/bin/env python3
"""Fail-closed CPU analysis of existing Kineto trace; never launches CUDA.

An independently recorded, SHA-bound original-code ATen template identifies
the two levels operations and LS group. Every measured QDQ must have exactly
one full encode+decode match. Attribution uses actual external IDs and launch
correlations, never kernel-name guesses or a single-layer times24 projection.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import statistics

TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
ENCODE_SHA = 'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6'
CODEC_SHA = 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f'


class Unknown(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise Unknown(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def interval(event):
    return float(event['ts']), float(event['ts']) + float(event.get('dur', 0.))


def contains(parent, child):
    a, b = interval(parent); c, d = interval(child)
    return parent.get('pid') == child.get('pid') and parent.get('tid') == child.get('tid') and a <= c and d <= b


def maximal_ops(events, region):
    ops = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op'
                  and e.get('name', '').startswith('aten::') and contains(region, e)),
                 key=lambda e: (float(e['ts']), -float(e.get('dur', 0.))))
    roots = []; outer = None
    for event in ops:
        if outer is not None and contains(outer, event):
            continue
        if outer is not None:
            require(interval(event)[0] >= interval(outer)[1], 'Partially overlapping non-nested CPU ATen roots')
        roots.append(event); outer = event
    return roots


def stats(values):
    require(values, 'Empty timing observations')
    s = sorted(values)
    return {'count': len(values), 'sum': math.fsum(values), 'mean': math.fsum(values) / len(values),
            'median': statistics.median(values), 'min': s[0], 'max': s[-1], 'p95': s[math.ceil(.95 * len(s)) - 1]}


def union_duration(events):
    spans = sorted(interval(e) for e in events); total = 0.; end = None
    for begin, finish in spans:
        total += finish - begin if end is None or begin > end else max(0., finish - end)
        end = finish if end is None else max(end, finish)
    return total


def external(event):
    return event.get('args', {}).get('External id')


def correlation(event):
    return event.get('args', {}).get('correlation')


def attribute(events, template, expected_regions=128):
    require(template['original_encode_sha256'] == ENCODE_SHA and template['adapter_sha256'] == CODEC_SHA,
            'Untrusted original-code template binding')
    require(template.get('SCALE_TRACE_enabled') is True and template.get('raw_scale_clone_count') == 3,
            'Template must include actual trace_encode raw-scale clones')
    pattern = template['encode_decode_maximal_ATen_names']; stages = template['stages_per_maximal_op']
    require(len(pattern) == len(stages) and pattern and stages.count('LS') > 0, 'Invalid fixed template')
    first_levels = [i for i, name in enumerate(stages) if name == 'INITIAL_LEVELS']
    last_levels = [i for i, name in enumerate(stages) if name == 'FINAL_LEVELS']
    require(first_levels and last_levels and max(first_levels) < min(i for i, name in enumerate(stages) if name == 'LS')
            and max(i for i, name in enumerate(stages) if name == 'LS') < min(last_levels), 'Two-level/LS boundaries invalid')
    require([pattern[i] for i in first_levels].count('aten::round') == 1
            and [pattern[i] for i in last_levels].count('aten::round') == 1, 'Levels boundaries require two exact round sites')
    regions = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                      and e.get('name') == 'REFERENCE/QDQ_and_original_safety'),
                     key=lambda e: float(e['ts']))
    require(len(regions) == expected_regions, 'Expected exactly128 measured QDQ regions; actual=' + str(len(regions)))
    require(len({(r.get('pid'), r.get('tid')) for r in regions}) == 1, 'QDQ regions cross threads/processes')
    for a, b in zip(regions, regions[1:]):
        require(interval(a)[1] <= interval(b)[0], 'Overlapping/reentrant QDQ regions')
    cpu_ops = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op' and external(e) is not None]
    ids = [external(e) for e in cpu_ops]
    require(len(ids) == len(set(ids)), 'Nonunique CPU external IDs')
    cpu_by_id = {external(e): e for e in cpu_ops}
    runtimes = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    devices = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    require(devices, 'No CUDA kernel/memory activity in existing trace')
    runtime_by_corr = {}
    for e in runtimes:
        corr = correlation(e)
        if corr is not None:
            runtime_by_corr.setdefault(corr, []).append(e)
    device_by_corr = {}
    device_by_external = {}
    for e in devices:
        corr = correlation(e)
        if corr is not None:
            device_by_corr.setdefault(corr, []).append(e)
        device_by_external.setdefault(external(e), []).append(e)
    runtime_by_external = {}
    for e in runtimes:
        runtime_by_external.setdefault(external(e), []).append(e)
    cpu_sorted = sorted(cpu_ops, key=lambda e: float(e['ts']))
    cpu_starts = [float(e['ts']) for e in cpu_sorted]
    runtime_sorted = sorted(runtimes, key=lambda e: float(e['ts']))
    runtime_starts = [float(e['ts']) for e in runtime_sorted]
    results = []
    for index, region in enumerate(regions):
        begin, end = interval(region)
        region_cpu_ops = [e for e in cpu_sorted[bisect.bisect_left(cpu_starts, begin):bisect.bisect_right(cpu_starts, end)] if contains(region, e)]
        roots = maximal_ops(region_cpu_ops, region); names = [e['name'] for e in roots]
        matches = [i for i in range(len(names) - len(pattern) + 1) if names[i:i + len(pattern)] == pattern]
        require(len(matches) == 1, 'QDQ' + str(index) + ' complete encode/decode template match count=' + str(len(matches)))
        matched = roots[matches[0]:matches[0] + len(pattern)]
        owners = {}
        for op, stage in zip(matched, stages):
            for e in region_cpu_ops:
                if contains(op, e):
                    owner = external(e)
                    require(owner not in owners or owners[owner] == stage, 'CPU operator assigned multiple stages')
                    owners[owner] = stage
        require(owners, 'Empty original encode/decode external-ID ownership')
        local_runtimes = {id(e): e for e in runtime_sorted[bisect.bisect_left(runtime_starts, begin):bisect.bisect_right(runtime_starts, end)] if contains(region, e)}
        candidates = {}
        for owner in owners:
            for event in device_by_external.get(owner, []):
                candidates[id(event)] = event
            for runtime in runtime_by_external.get(owner, []):
                local_runtimes[id(runtime)] = runtime
                for event in device_by_corr.get(correlation(runtime), []):
                    candidates[id(event)] = event
        attributed = []
        for event in candidates.values():
            owner = external(event)
            corr = correlation(event)
            links = runtime_by_corr.get(corr, []) if corr is not None else []
            require(corr is not None and len(links) == 1, 'Attributed CUDA activity lacks unique launch correlation')
            runtime_owner = external(links[0])
            require(runtime_owner in owners and runtime_owner in cpu_by_id, 'CUDA launch runtime lacks recognized original CPU external-ID owner')
            stage = owners[runtime_owner]
            if owner is not None:
                require(owner in cpu_by_id and owner in owners and owners[owner] == stage,
                        'Device external ID is unknown or disagrees with unique launch operator stage')
            attributed.append((event, stage))
        require(attributed, 'No matched encode/decode GPU activity')
        for runtime in local_runtimes.values():
            if external(runtime) not in owners and not contains(region, runtime):
                continue
            inside_matched = any(contains(op, runtime) for op in matched)
            if external(runtime) not in owners and not inside_matched:
                continue
            name = runtime.get('name', '')
            if ('LaunchKernel' in name or 'Memcpy' in name or 'Memset' in name) and 'Get' not in name:
                require(external(runtime) in owners, 'Launch inside matched original encode/decode lacks operator external-ID ownership')
                corr = correlation(runtime)
                require(corr is not None and device_by_corr.get(corr), 'Matched CUDA launch/copy has no device activity')
        ls_events = [e for e, stage in attributed if stage == 'LS']
        encode_events = [e for e, stage in attributed if stage != 'DECODE']
        require(ls_events and encode_events, 'Missing actual LS/encode activity')
        ls_ops = [op for op, stage in zip(matched, stages) if stage == 'LS']
        ls_cpu = interval(ls_ops[-1])[1] - interval(ls_ops[0])[0]
        encode_sum = math.fsum(float(e['dur']) for e in encode_events)
        ls_sum = math.fsum(float(e['dur']) for e in ls_events)
        results.append({'token_window_index': index, 'template_unique_matches': 1, 'encode_decode_maximal_op_count': len(matched),
                        'LS_CPU_dispatch_interval_us': ls_cpu, 'LS_device_activity_count': len(ls_events),
                        'LS_device_activity_duration_sum_us': ls_sum, 'LS_device_active_interval_union_us': union_duration(ls_events),
                        'encode_device_activity_duration_sum_us': encode_sum,
                        'LS_fraction_of_encode_summed_device_activity': ls_sum / encode_sum,
                        'all_device_activity_correlations_unique': True})
    return {'status': 'PASS_UNIQUE_EXISTING_TRACE_GPU_ACTIVITY_ATTRIBUTION_ONLY', 'matched_QDQ_regions': len(results),
            'actual_batch': [24, 32, 128, 128], 'batch_basis': 'Original immutable dual concatenates24 native GDN states; full-reference measured QDQ scope, not single-layer extrapolation.',
            'per_token': results,
            'LS_device_activity_duration_sum_us': stats([r['LS_device_activity_duration_sum_us'] for r in results]),
            'LS_CPU_dispatch_interval_us': stats([r['LS_CPU_dispatch_interval_us'] for r in results]),
            'LS_fraction_of_encode_summed_device_activity': stats([r['LS_fraction_of_encode_summed_device_activity'] for r in results]),
            'aggregate_LS_fraction_of_encode_summed_device_activity': math.fsum(r['LS_device_activity_duration_sum_us'] for r in results)
                                                                    / math.fsum(r['encode_device_activity_duration_sum_us'] for r in results),
            'LS_complete_wall_time': 'UNKNOWN_NOT_MEASURED_BY_STAGE_EVENT_BOUNDARIES',
            'limitations': ['GPU activity durations and their sum are not complete wall time; overlapping activities must not be added as wall.',
                           'CPU source template is only a strict ownership pattern; all128 actual GPU regions must uniquely match.',
                           'LS group includes v construction, denominator, numerator, guarded fit and fitted FP16 scale/raw trace clone.',
                           'This does not measure the attainable speedup from deleting LS, does not change reference math, and never multiplies one-layer timing by24.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--record-sha256', required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--template-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bindings = {'record_path': str(args.record), 'record_sha256': args.record_sha256,
                'template_path': str(args.template), 'template_sha256': args.template_sha256}
    try:
        require(sha(args.record) == args.record_sha256, 'Actual runtime record SHA mismatch')
        require(sha(args.template) == args.template_sha256, 'Fixed source template SHA mismatch')
        record = read(args.record)
        require(record['status'] == 'PASS' and record['task_id'] == TASK and record['condition'] == 'H_DUAL_M_REFERENCE'
                and record['instrumented'] is True and record['length'] == 128 and record['warmup_tokens'] == 128,
                'Not the completed fixed128 H/DualM reference profile')
        require(record['layer_writebacks'] == 24 * 256 and record['native_core_calls'] == 24 * 256,
                'Actual24-layer reference call/quantization identity failed')
        trace = args.root / record['chrome_trace']['path']
        require(sha(trace) == record['chrome_trace']['sha256'] and trace.stat().st_size == record['chrome_trace']['bytes'],
                'Existing Chrome trace SHA/bytes mismatch')
        bindings.update(trace_path=str(trace), trace_sha256=sha(trace), trace_bytes=trace.stat().st_size)
        obj = read(trace)
        result = attribute(obj['traceEvents'], read(args.template))
    except (Unknown, KeyError, ValueError, OSError, TypeError) as exc:
        result = {'status': 'UNKNOWN_FAIL_CLOSED_NO_NEW_GPU', 'reason': str(exc),
                  'LS_batched_GPU_activity_time': 'UNKNOWN', 'LS_complete_wall_time': 'UNKNOWN',
                  'no_kernel_name_guessing': True, 'no_single_layer_times24': True}
    result.update(task_id=TASK, analysis_role='CPU_ONLY_READONLY_EXISTING_TRACE', bindings=bindings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': sha(args.output), 'reason': result.get('reason')}))


if __name__ == '__main__':
    main()
