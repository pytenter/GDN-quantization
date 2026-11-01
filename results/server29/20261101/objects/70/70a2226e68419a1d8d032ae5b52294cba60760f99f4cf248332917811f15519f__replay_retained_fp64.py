#!/usr/bin/env python3
"""CPU FP64 replay of actually retained native GDN inputs; never invokes a model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
NAMES = ('query', 'key', 'value', 'gate', 'beta', 'core_actual',
         'state_actual', 'native_norm_query', 'native_norm_key')
MATH_GATE = dict(state_atol=1e-5, state_scale_rtol=1e-4,
                 output_atol=.001, output_scale_rtol=.01,
                 normalization_atol=.0001, normalization_scale_rtol=.02)
SERIALIZATION = 'contiguous_C_order_native_dtype'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stable(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe retained file: ' + str(path))
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'Retained file changed while reading: ' + str(path))
    return body


def sha(body):
    return hashlib.sha256(body).hexdigest()


def safe_file(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and str(name) == relative,
            'Unsafe retained raw path')
    path = root / name
    require(root.resolve() in path.resolve().parents and path.is_file() and not path.is_symlink(),
            'Retained raw path escapes root or is missing')
    ancestor = root
    for component in name.parts:
        ancestor = ancestor / component
        require(not ancestor.is_symlink(), 'Retained raw ancestor symlink prohibited')
    return path


def coverage(keys):
    expected = {(token, layer) for token in range(32) for layer in GDN_LAYERS}
    return set(keys) == expected and len(keys) == len(expected)


def decode_record(root, path, torch):
    metadata_body = stable(path)
    record = json.loads(metadata_body)
    token, layer = record['token'], record['layer']
    require(record['unit'] == path.parent.parent.name, 'Retained intermediate unit identity differs')
    require(type(token) is int and 0 <= token < 32 and type(layer) is int and layer in GDN_LAYERS,
            'Unexpected token/layer identity')
    require(path.name == 'TOKEN_%03d_LAYER_%02d.json' % (token, layer),
            'Intermediate filename differs from token/layer identity')
    require(record['kind'] == ('chunk' if token == 0 else 'recurrent'), 'Native helper kind differs')
    require(record['serialization'] == SERIALIZATION and
            record.get('source_stride_is_metadata_not_serialized_stride') is True,
            'Unknown intermediate serialization')
    expected_raw = str(path.with_suffix('.bin').relative_to(root))
    require(record['raw_path'] == expected_raw, 'Intermediate raw path/unit identity differs')
    raw_path = safe_file(root, record['raw_path'])
    body = stable(raw_path)
    require(0 < len(body) <= 16 * 1024**2 and len(body) == record['raw_bytes'] and
            sha(body) == record['raw_sha256'], 'Intermediate container SHA/bytes differ')
    require([row['name'] for row in record['tensors']] == list(NAMES),
            'Complete ordered nine-tensor intermediate roster required')
    tensors = {}
    offset = 0
    for row in record['tensors']:
        require(isinstance(row['shape'], list) and all(type(n) is int and n > 0 for n in row['shape']),
                'Invalid retained tensor shape')
        require(isinstance(row['stride'], list) and len(row['stride']) == len(row['shape']) and
                all(type(n) is int and n >= 0 for n in row['stride']), 'Invalid source stride metadata')
        require(row['dtype'] in ('torch.float32', 'torch.bfloat16', 'torch.float64'),
                'Unexpected actual native intermediate dtype')
        dtype = getattr(torch, row['dtype'][6:])
        count = math.prod(row['shape'])
        size = torch.empty((), dtype=dtype, device='cpu').element_size()
        require(type(row['offset']) is int and row['offset'] == offset and
                row['bytes'] == count * size and offset + row['bytes'] <= len(body),
                'Retained tensor bounds/contiguous serialization differs')
        data = body[offset:offset + row['bytes']]
        require(sha(data) == row['sha256'], 'Retained tensor SHA differs: ' + row['name'])
        tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(row['shape'])
        require(bool(torch.isfinite(tensor).all()), 'Nonfinite retained tensor: ' + row['name'])
        tensors[row['name']] = tensor
        offset += row['bytes']
    require(offset == len(body), 'Unclaimed/trailing retained tensor bytes')
    for name in ('query', 'key', 'value', 'core_actual', 'native_norm_query', 'native_norm_key'):
        require(list(tensors[name].shape) == [1, 1, 32, 128], 'Frozen native tensor geometry differs: ' + name)
    for name in ('gate', 'beta'):
        require(list(tensors[name].shape) == [1, 1, 32], 'Frozen scalar update geometry differs: ' + name)
    require(list(tensors['state_actual'].shape) == [1, 32, 128, 128] and
            tensors['state_actual'].dtype == torch.float32, 'Frozen FP32 native state geometry differs')
    evidence = dict(metadata_path=str(path.relative_to(root)), metadata_bytes=len(metadata_body),
                    metadata_sha256=sha(metadata_body), raw_path=record['raw_path'],
                    raw_bytes=len(body), raw_sha256=sha(body))
    return record, tensors, evidence


def evaluate(root, unit, torch):
    require(unit in ('NUMERIC_FRESH0', 'NUMERIC_FRESH1'), 'Unexpected numeric unit')
    plan_path = root / 'qualification/GPU_PLAN.json'
    plan_body = stable(plan_path)
    plan = json.loads(plan_body)
    require(plan['numeric_tokens'] == 32 and all(plan['math_gate'].get(k) == v for k, v in MATH_GATE.items()),
            'Frozen original FP64 gate changed')
    directory = root / 'execution' / unit / 'intermediates'
    paths = sorted(directory.glob('TOKEN_*_LAYER_*.json'))
    states, previous_actual_hash, seen, records, evidence = {}, {}, set(), [], []
    current = None
    failure = None
    try:
        for path in paths:
            current = dict(metadata_path=str(path.relative_to(root)))
            rec, tensors, proof = decode_record(root, path, torch)
            token, layer = rec['token'], rec['layer']
            current.update(token=token, layer=layer)
            require((token, layer) not in seen, 'Duplicate retained token/layer')
            require(token == sum(1 for t, l in seen if l == layer), 'Missing retained layer prefix')
            expected_previous = None if token == 0 else previous_actual_hash[layer]
            require(rec.get('previous_state_sha256') == expected_previous,
                    'Actual previous-state SHA lineage differs')
            seen.add((token, layer))
            evidence.append(proof)
            q0, k0 = tensors['native_norm_query'].double(), tensors['native_norm_key'].double()
            norm_checks = []
            for source_name, actual in [('query', q0), ('key', k0)]:
                source = tensors[source_name].double()
                reference = source * torch.rsqrt((source * source).sum(-1, keepdim=True) + 1e-6)
                error = float((actual - reference).abs().max())
                bound = MATH_GATE['normalization_atol'] + MATH_GATE['normalization_scale_rtol'] * float(reference.abs().max())
                norm_checks.append(dict(tensor=source_name, max_abs=error, bound=bound, passed=error <= bound))
            v0, gate, beta = (tensors[name].double() for name in ('value', 'gate', 'beta'))
            state = torch.zeros((1, 32, 128, 128), dtype=torch.float64, device='cpu') if token == 0 else states[layer]
            outputs = []
            # Same frozen activation-conditioned formula; the FP64 oracle carries
            # from token0, never resets from the rounded GPU state.
            for j in range(q0.shape[1]):
                state = state * gate[:, j].exp()[..., None, None]
                memory = (state * k0[:, j, ..., None]).sum(-2)
                state = state + k0[:, j, ..., None] * ((v0[:, j] - memory) * beta[:, j, ..., None])[..., None, :]
                outputs.append((state * (q0[:, j] / math.sqrt(128))[..., None]).sum(-2))
            output = torch.stack(outputs, 1)
            require(bool(torch.isfinite(state).all()) and bool(torch.isfinite(output).all()),
                    'Nonfinite FP64 recurrence reference')
            state_error = float((tensors['state_actual'].double() - state).abs().max())
            core_error = float((tensors['core_actual'].double() - output).abs().max())
            state_bound = MATH_GATE['state_atol'] + MATH_GATE['state_scale_rtol'] * float(state.abs().max())
            core_bound = MATH_GATE['output_atol'] + MATH_GATE['output_scale_rtol'] * float(output.abs().max())
            row = dict(token=token, layer=layer, kind=rec['kind'], state_max_abs=state_error,
                       state_bound=state_bound, core_max_abs=core_error, core_bound=core_bound,
                       normalization_checks=norm_checks,
                       passed=all(x['passed'] for x in norm_checks) and state_error <= state_bound and core_error <= core_bound)
            records.append(row)
            require(row['passed'], 'Frozen FP64 normalization/state/core bound failed')
            states[layer] = state
            previous_actual_hash[layer] = next(x['sha256'] for x in rec['tensors'] if x['name'] == 'state_actual')
    except (RuntimeError, KeyError, ValueError, OSError, TypeError) as exc:
        failure = dict(kind='INTEGRITY_OR_FROZEN_NUMERICAL_GATE_REJECTED', record=current,
                       error=type(exc).__name__ + ': ' + str(exc),
                       numerical_details=records[-1] if records and current and
                       all(records[-1].get(k) == current.get(k) for k in ('token', 'layer')) else None)
    complete = coverage(list(seen)) and len(records) == 768 and failure is None
    status = ('FAIL_CLOSED_RETAINED_FP64_REPLAY' if failure else
              'PASS_FULL_32TOKEN_24LAYER_RETAINED_FP64_REPLAY' if complete else
              'PASS_PARTIAL_ACTUAL_RETAINED_FP64_REPLAY_NOT_FULL_QUALIFICATION' if records else
              'NOT_RUN_NO_ACTUAL_RETAINED_INTERMEDIATES')
    return dict(status=status, unit=unit, actual_records_discovered=len(paths), actual_records_audited=len(records),
                expected_records=768, expected_tokens=32, expected_GDN_layers=list(GDN_LAYERS),
                complete_32token_24layer_audit=complete, failure=failure,
                actual_audit_rows=records, retained_input_evidence=evidence,
                plan_sha256=sha(plan_body), original_frozen_math_gate=MATH_GATE,
                reference_scope='CPU FP64 recurrence conditioned on actual retained native normalized q/k, with independent FP64 norm checks, actual v/g/b and state carry from token0. Not full-model FP64 or long-horizon/INT8 proof.',
                current_process_GPU_dispatches=0, current_process_CUDA_context_initialized=False)


def replay(root, unit, output):
    require(not output.exists(), 'Replay output exists; no overwrite')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    require(not torch.cuda.is_initialized(), 'CPU replay must not inherit a CUDA context')
    result = evaluate(root.resolve(), unit, torch)
    require(not torch.cuda.is_initialized(), 'CPU replay unexpectedly initialized CUDA')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--unit', choices=['NUMERIC_FRESH0', 'NUMERIC_FRESH1'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    value = replay(args.root, args.unit, args.output)
    print(json.dumps({k: v for k, v in value.items() if k not in ('actual_audit_rows', 'retained_input_evidence')}, allow_nan=False))
    raise SystemExit(2 if value['failure'] else 0)
