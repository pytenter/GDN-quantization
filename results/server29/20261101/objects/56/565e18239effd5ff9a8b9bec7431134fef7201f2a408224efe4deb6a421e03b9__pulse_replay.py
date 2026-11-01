import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from common import (ANCHOR_TOKEN, CHECKPOINTS, CONDITIONS, ENDPOINT, EXP,
                    FIRST_TOKEN, GPU_MAP, LAYERS, PARENT, PARENT_SCRIPTS,
                    canonical_hash, read_json, sha256, tensor_sha, write_json)


class PulseReducer:
    names = ['J_HIST_PULSE', 'J_PERSIST', 'J_QDQ_DEP', 'J_CROSS']

    def __init__(self):
        self.count = 0
        self.normalized = [[0.0] * 4 for _ in LAYERS]
        self.raw = [[0.0] * 4 for _ in LAYERS]

    def update(self, token, normalized, raw):
        if token < FIRST_TOKEN:
            return
        self.count += 1
        for i in range(len(LAYERS)):
            for j in range(4):
                value = normalized[i][j]
                raw_value = raw[i][j]
                if not math.isfinite(value) or not math.isfinite(raw_value):
                    raise RuntimeError('nonfinite pulse energy')
                self.normalized[i][j] += value
                self.raw[i][j] += raw_value

    def snapshot(self):
        if self.count <= 0:
            raise RuntimeError('empty reducer')
        per_layer = [[x / self.count for x in row] for row in self.normalized]
        raw_per_layer = [[x / self.count for x in row] for row in self.raw]
        global_values = [math.fsum(row[j] for row in per_layer) / len(LAYERS)
                         for j in range(4)]
        raw_global = [math.fsum(row[j] for row in raw_per_layer) / len(LAYERS)
                      for j in range(4)]
        return {'token_count': self.count, 'first_token': FIRST_TOKEN,
                'metric_names': self.names, 'normalized_mean': global_values,
                'per_layer': per_layer, 'raw_energy_mean': raw_global,
                'raw_per_layer': raw_per_layer}


def load_replay():
    sys.path.insert(0, str(PARENT_SCRIPTS))
    import capture_original_drive as capture_base
    capture_base.AMEND = PARENT
    import isolated_replay_engineering as engine
    return engine.Replay


def parent_result(document, condition):
    path = PARENT / f'scientific/doc_{document:02d}/results/{condition}/replay_result.json'
    value = read_json(path)
    return path, value, value['identity']['outputs'][condition]


def run_once(document, condition, capture_path, collect_streaming_fixture=False):
    Replay = load_replay()
    R = Replay()
    R.calls = {'blocked_cumsum': 0, 'full_model_forward': 0,
               'QDQ_path': 0, 'QDQ_reference': 0}
    torch = R.torch
    _, parent_payload, parent_output = parent_result(document, condition)
    reference_hashes = {x['token']: x['per_layer_sha256']
                        for x in read_json(capture_path / 'fp_state_hashes.json')}
    reducer = PulseReducer()
    fp = {l: None for l in LAYERS}
    persistent = {l: None for l in LAYERS}
    pulse = None
    snapshots = {}
    state_hashes = {}
    anchor_identity = None
    maximum_vector_residual = 0.0
    maximum_energy_residual = 0.0
    u64 = 2 ** -53
    n = 32 * 128 * 128
    gamma = n * u64 / (1 - n * u64)

    segment_lists = {l: [[] for _ in range(4)] for l in LAYERS}
    segment_sums = {l: [0.0] * 4 for l in LAYERS}
    segment_count = 0
    started = time.monotonic()

    with R.guard():
        for token, records in R.drive(ENDPOINT, capture_path):
            old_fp = fp
            fp = {l: R.step('Native', l, old_fp[l], records[l]) for l in LAYERS}
            persistent = {l: R.step(condition, l, persistent[l], records[l])
                          for l in LAYERS}
            R.qdq(persistent, 'path')
            physical_persistent = {l: R.physical(condition, l, persistent[l])
                                   for l in LAYERS}

            if token < ANCHOR_TOKEN:
                physical_pulse = physical_persistent
            elif token == ANCHOR_TOKEN:
                pulse = {l: physical_persistent[l].clone() for l in LAYERS}
                physical_pulse = pulse
            else:
                pulse = {l: R.step('Native', l, pulse[l], records[l]) for l in LAYERS}
                physical_pulse = pulse

            normalized = []
            raw = []
            for l in LAYERS:
                a = fp[l].double()
                b = physical_pulse[l].double()
                c = physical_persistent[l].double()
                e_hist = b - a
                e_persist = c - a
                e_qdq = c - b
                residual = float((e_persist - e_hist - e_qdq).abs().max())
                scale = max(1.0, float(a.abs().max()), float(b.abs().max()),
                            float(c.abs().max()))
                vector_bound = 8 * u64 * scale
                if residual > vector_bound:
                    raise RuntimeError('state-space identity closure failed')
                energies = torch.stack([(e_hist * e_hist).sum(),
                                        (e_persist * e_persist).sum(),
                                        (e_qdq * e_qdq).sum(),
                                        2 * (e_hist * e_qdq).sum()])
                values = energies.cpu().tolist()
                energy_residual = abs(values[1] - values[0] - values[2] - values[3])
                energy_bound = 16 * gamma * max(1.0, values[0], values[1],
                                                values[2], abs(values[3]))
                if energy_residual > energy_bound:
                    raise RuntimeError('pulse energy identity closure failed')
                denominator = float((a * a).sum()) + 1e-12
                row = [x / denominator for x in values]
                if not all(math.isfinite(x) for x in row + values):
                    raise RuntimeError('nonfinite pulse metric')
                normalized.append(row)
                raw.append(values)
                maximum_vector_residual = max(maximum_vector_residual, residual)
                maximum_energy_residual = max(maximum_energy_residual, energy_residual)

            reducer.update(token, normalized, raw)

            if collect_streaming_fixture and ANCHOR_TOKEN <= token < ANCHOR_TOKEN + 256:
                segment_count += 1
                for i, l in enumerate(LAYERS):
                    for j in range(4):
                        segment_lists[l][j].append(normalized[i][j])
                        segment_sums[l][j] += normalized[i][j]

            if token in reference_hashes:
                for l in LAYERS:
                    actual = tensor_sha(fp[l])
                    if actual != reference_hashes[token][str(l)]:
                        raise RuntimeError(f'FP reference replay mismatch token={token} layer={l}')

            if token in CHECKPOINTS:
                snap = reducer.snapshot()
                state_row = {}
                for l in LAYERS:
                    state_row[str(l)] = {
                        'A_FP': tensor_sha(fp[l]),
                        'B_HIST_PULSE': tensor_sha(physical_pulse[l]),
                        'C_PERSISTENT_STORED': tensor_sha(persistent[l]),
                        'C_PERSISTENT_PHYSICAL': tensor_sha(physical_persistent[l]),
                    }
                snapshots[str(token)] = snap
                state_hashes[str(token)] = state_row

                if token in (ANCHOR_TOKEN, ENDPOINT):
                    parent_endpoint = parent_output['endpoints'][str(token)]
                    for l in LAYERS:
                        if state_row[str(l)]['A_FP'] != parent_endpoint[str(l)]['FP']:
                            raise RuntimeError(f'parent FP endpoint mismatch token={token} layer={l}')
                        if state_row[str(l)]['C_PERSISTENT_STORED'] != parent_endpoint[str(l)]['stored_Q']:
                            raise RuntimeError(f'parent persistent endpoint mismatch token={token} layer={l}')

                if token == ANCHOR_TOKEN:
                    for l in LAYERS:
                        if state_row[str(l)]['B_HIST_PULSE'] != state_row[str(l)]['C_PERSISTENT_PHYSICAL']:
                            raise RuntimeError('B/C physical anchor state identity failed')
                    anchor_identity = state_row

                if token in (ANCHOR_TOKEN, ENDPOINT):
                    expected = parent_output['horizons'][str(token)]['normalized_mean'][0]
                    actual = snap['normalized_mean'][1]
                    tolerance = 16 * (256 * u64 / (1 - 256 * u64)) * max(1.0, abs(expected), abs(actual))
                    if abs(actual - expected) > tolerance:
                        raise RuntimeError(f'parent J_ACC parity failed at {token}: {actual} vs {expected}')

            if token % 512 == 0:
                print(f'DOC={document} CONDITION={condition} TOKEN={token}', flush=True)

    if anchor_identity is None or set(snapshots) != set(map(str, CHECKPOINTS)):
        raise RuntimeError('missing anchor or checkpoint')
    if R.calls['blocked_cumsum'] != 0 or R.calls['full_model_forward'] != 0:
        raise RuntimeError('replay isolation failed')
    if R.calls['QDQ_path'] != ENDPOINT:
        raise RuntimeError('persistent QDQ count mismatch')

    stream_report = None
    if collect_streaming_fixture:
        if segment_count != 256:
            raise RuntimeError('streaming fixture length mismatch')
        maximum = 0.0
        for l in LAYERS:
            for j in range(4):
                materialized = math.fsum(segment_lists[l][j]) / segment_count
                streaming = segment_sums[l][j] / segment_count
                diff = abs(materialized - streaming)
                bound = 16 * (256 * u64 / (1 - 256 * u64)) * max(
                    1.0, abs(materialized), abs(streaming))
                if diff > bound:
                    raise RuntimeError('streaming reducer parity failed')
                maximum = max(maximum, diff)
        stream_report = {'gate': 'PASS', 'tokens': 256,
                         'range': [ANCHOR_TOKEN, ANCHOR_TOKEN + 255],
                         'maximum_absolute_difference': maximum}

    scientific_identity = {
        'document_index': document,
        'condition': condition,
        'anchor_hashes': anchor_identity,
        'checkpoint_metrics': snapshots,
        'checkpoint_state_hashes': state_hashes,
        'maximum_vector_residual': maximum_vector_residual,
        'maximum_energy_residual': maximum_energy_residual,
        'calls': R.calls,
        'pulse_qdq_calls_after_8192': 0,
        'pulse_writeback_calls_after_8192': 0,
        'persistent_qdq_calls_after_8192': ENDPOINT - ANCHOR_TOKEN,
        'parent_result_sha256': sha256(parent_result(document, condition)[0]),
        'drive_identity_sha256': sha256(EXP / f'discovery/doc_{document:02d}/drive_identity.json'),
        'strict_deterministic_algorithms': True,
        'tf32': False,
    }
    scientific_identity['identity_sha256'] = canonical_hash(scientific_identity)
    return {
        'status': 'PASS',
        'scientific_identity': scientific_identity,
        'streaming_fixture': stream_report,
        'elapsed_seconds': time.monotonic() - started,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'physical_gpu': GPU_MAP[condition],
        'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_uuid': read_json(EXP / 'runtime_manifest.json')['gpu_inventory'][GPU_MAP[condition]]['uuid'],
        'training': False, 'optimizer_steps': 0, 'AIME': False,
        'free_generation': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['same_process', 'fresh_process', 'formal'], required=True)
    parser.add_argument('--document', type=int, required=True, choices=range(1, 5))
    parser.add_argument('--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--run', type=int)
    args = parser.parse_args()
    expected_gpu = GPU_MAP[args.condition]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(expected_gpu):
        raise RuntimeError(f'physical GPU mapping mismatch for {args.condition}')
    authorization = read_json(EXP / 'execution_authorization.json')
    if not authorization.get('PREMEASUREMENT_COMMIT_SHA'):
        raise RuntimeError('missing premeasurement commit')
    if args.mode == 'formal':
        discovery_auth = read_json(EXP / 'discovery_authorization.json')
        if discovery_auth['CAUSAL_DISCOVERY_EXECUTION_AUTHORIZED'] != 'YES':
            raise RuntimeError('formal causal Discovery is not authorized')

    capture_path = EXP / f'discovery/doc_{args.document:02d}/drive'
    drive_gate = read_json(EXP / f'discovery/doc_{args.document:02d}/drive_identity.json')
    if drive_gate['G5_FROZEN_DRIVE_IDENTITY_GATE'] != 'PASS':
        raise RuntimeError('drive identity gate not PASS')

    if args.mode == 'same_process':
        if (args.document, args.condition) != (1, 'Native') or args.run is not None:
            raise RuntimeError('frozen same-process smoke unit mismatch')
        output = EXP / 'smoke/same_process.json'
        runs = [run_once(1, 'Native', capture_path, collect_streaming_fixture=(i == 0))
                for i in range(3)]
        payload = {'status': 'PASS', 'mode': args.mode, 'runs': runs}
    elif args.mode == 'fresh_process':
        if (args.document, args.condition) != (1, 'Native') or args.run not in [1, 2, 3]:
            raise RuntimeError('frozen fresh-process smoke unit/run mismatch')
        output = EXP / f'smoke/fresh_process_{args.run}.json'
        payload = run_once(1, 'Native', capture_path,
                           collect_streaming_fixture=(args.run == 1))
        payload['mode'] = args.mode
        payload['run'] = args.run
    else:
        if args.run is not None:
            raise RuntimeError('formal run has no retry/run selector')
        output = EXP / f'discovery/unit_results/doc_{args.document:02d}_{args.condition}.json'
        payload = run_once(args.document, args.condition, capture_path)
        payload['mode'] = args.mode

    try:
        write_json(output, payload)
    except Exception:
        raise
    print(f'PULSE_REPLAY_{args.mode.upper()}_PASS {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            mode = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--mode'), 'unknown')
            document = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--document'), 'unknown')
            condition = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--condition'), 'unknown')
            path = EXP / f'discovery/logs/{mode}_doc{document}_{condition}.error.json'
            if not path.exists():
                write_json(path, {'status': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
                                  'message': str(exc), 'traceback': traceback.format_exc(),
                                  'retry_authorized': False})
        finally:
            raise
