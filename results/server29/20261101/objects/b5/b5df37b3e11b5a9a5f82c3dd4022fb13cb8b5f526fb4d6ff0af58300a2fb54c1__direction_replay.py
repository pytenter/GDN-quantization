import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import traceback

from common import (ANCHOR_TOKEN, ARMS, CONDITIONS, ENDPOINT, EXP,
                    FOCUS_LAYERS, FORMAL, HIST_BINS, HIST_MAX, HIST_MIN,
                    LAYERS, PARENT, PARENT_SCRIPTS, READPOINTS,
                    canonical_hash, clean_result, read_json, sha256,
                    tensor_sha, write_json)


def load_replay():
    sys.path.insert(0, str(PARENT_SCRIPTS))
    import capture_original_drive as capture_base
    capture_base.AMEND = PARENT
    import isolated_replay_engineering as engine
    return engine.Replay


class CanonicalScaleAdapter:
    """Execute the source/scale/codes/dequant nodes from the frozen formal AST."""
    def __init__(self, replay):
        self.R = replay
        source = FORMAL.read_text(encoding='utf-8')
        tree = ast.parse(source)
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and
                  x.name == 'quantize_c128')
        assignments = {}
        for node in fn.body[0].body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node
        if not all(x in assignments for x in ['source','scale','codes','dequant']):
            raise RuntimeError('frozen quantizer AST dataflow changed')
        self.qparam_code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignments['source'], assignments['scale']], type_ignores=[])),
            str(FORMAL), 'exec')
        self.source_code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignments['source']], type_ignores=[])), str(FORMAL), 'exec')
        self.apply_code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignments['source'], assignments['codes'], assignments['dequant']],
            type_ignores=[])), str(FORMAL), 'exec')
        self.formal_sha256 = sha256(FORMAL)

    def qparam(self, state):
        env = {'state': state, 'torch': self.R.torch, 'EPS': self.R.F.EPS}
        exec(self.qparam_code, env)
        return env['scale']

    def source(self, state):
        env = {'state': state, 'torch': self.R.torch, 'EPS': self.R.F.EPS}
        exec(self.source_code, env)
        return env['source']

    def apply(self, state, scale):
        env = {'state': state, 'scale': scale, 'torch': self.R.torch,
               'EPS': self.R.F.EPS}
        exec(self.apply_code, env)
        dequant = env['dequant']
        if not bool(self.R.torch.isfinite(dequant).all().item()):
            raise FloatingPointError('nonfinite C128 dequant under direction intervention')
        state.copy_(dequant.to(state.dtype))
        return env['codes'], dequant, env['source']


def parent_output(document, condition):
    path = PARENT / f'scientific/doc_{document:02d}/results/{condition}/replay_result.json'
    return read_json(path)['identity']['outputs'][condition]


def iter_drive_shards(R, capture_path, start, end):
    manifest = read_json(capture_path / 'drive_manifest.json')
    for row in manifest['shards']:
        if row['last_token'] < start:
            continue
        if row['first_token'] > end:
            break
        path = capture_path / row['relative_path']
        if sha256(path) != row['file_sha256']:
            raise RuntimeError(f'drive shard hash mismatch: {path}')
        blob = R.torch.load(path, map_location='cpu', weights_only=True)
        lo = max(start, row['first_token'])
        hi = min(end, row['last_token'])
        values = {layer: {
            name: tensor[:, lo-row['first_token']:hi-row['first_token']+1].to('cuda:0')
            for name, tensor in blob['layers'][str(layer)].items()}
            for layer in LAYERS}
        yield lo, hi, values


def record_at(values, offset):
    return {layer: {name: tensor[:, offset:offset+1]
                    for name, tensor in values[layer].items()} for layer in LAYERS}


def init_dist(rank):
    import torch
    import torch.distributed as dist
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(rank):
        raise RuntimeError('physical GPU mapping mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG missing')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, dist


def build_anchor(document, condition, rank):
    R = load_replay()()
    states = {layer: None for layer in LAYERS}
    capture = EXP / f'discovery/doc_{document:02d}/drive'
    started = time.monotonic()
    with R.guard():
        for lo, hi, values in iter_drive_shards(R, capture, 1, ANCHOR_TOKEN):
            for token in range(lo, hi+1):
                rec = record_at(values, token-lo)
                states = {layer: R.step(condition, layer, states[layer], rec[layer])
                          for layer in LAYERS}
                R.qdq(states, 'path')
            if hi % 512 == 0:
                print(f'ANCHOR CONDITION={condition} TOKEN={hi}', flush=True)
    output = parent_output(document, condition)
    hashes = {str(layer): tensor_sha(states[layer]) for layer in LAYERS}
    for layer in LAYERS:
        if hashes[str(layer)] != output['endpoints'][str(ANCHOR_TOKEN)][str(layer)]['stored_Q']:
            raise RuntimeError(f'parent anchor mismatch condition={condition} layer={layer}')
    target = EXP / f'qualification/doc_{document:02d}/anchors'
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'{condition}.pt'
    if path.exists():
        raise RuntimeError(f'write-once anchor exists: {path}')
    R.torch.save({str(layer): states[layer].detach().cpu() for layer in LAYERS}, path)
    write_json(path.with_suffix('.json'), {
        'status': 'PASS', 'document_index': document, 'condition': condition,
        'token': ANCHOR_TOKEN, 'state_sha256': hashes, 'file_sha256': sha256(path),
        'parent_endpoint_exact': True, 'qdq_calls': R.calls['QDQ_path'],
        'elapsed_seconds': time.monotonic()-started, 'physical_gpu': rank})


def load_anchor(R, document, condition):
    path = EXP / f'qualification/doc_{document:02d}/anchors/{condition}.pt'
    meta = read_json(path.with_suffix('.json'))
    if sha256(path) != meta['file_sha256']:
        raise RuntimeError('qualified anchor artifact mismatch')
    payload = R.torch.load(path, map_location='cpu', weights_only=True)
    return {layer: payload[str(layer)].to('cuda:0') for layer in LAYERS}


def load_fp_anchor(R, document):
    drive = EXP / f'discovery/doc_{document:02d}/drive'
    path = drive / 'fp_reference/token_08192.pt'
    manifest = read_json(drive / 'drive_manifest.json')
    row = next(x for x in manifest['fp_reference_checkpoints']
               if x['token'] == ANCHOR_TOKEN)
    if sha256(path) != row['file_sha256']:
        raise RuntimeError('clean FP anchor mismatch')
    payload = R.torch.load(path, map_location='cpu', weights_only=True)
    return {layer: payload[str(layer)].to('cuda:0') for layer in LAYERS}


def parent_prefix(document, condition):
    snap = parent_output(document, condition)['horizons'][str(ANCHOR_TOKEN)]
    count = int(snap['token_count'])
    sums = [float(row[0])*count for row in snap['per_layer']]
    return count, sums, float(snap['normalized_mean'][0])


def token_bin(token):
    return next(x for x in READPOINTS if token <= x)


class SignedStreamingReducer:
    def __init__(self, torch):
        self.torch = torch
        self.rows = {}
        self.buffers = {}
        self.parity_values = []
        self.parity_stream = torch.zeros(HIST_BINS, dtype=torch.int64)

    def _row(self, arm, layer, endpoint):
        key = (arm, layer, endpoint)
        if key not in self.rows:
            self.rows[key] = {
                'hist': self.torch.zeros(HIST_BINS, dtype=self.torch.int64),
                'count': 0, 'positive_count': 0, 'negative_count': 0,
                'max_positive': None, 'min_negative': None,
                'clamp_count': 0, 'saturation_count': 0,
                'saturation_total': 0}
            self.buffers[key] = []
        return key, self.rows[key]

    def add(self, arm, layer, token, signed, clamp_mask, saturation_count,
            saturation_total, parity=False):
        endpoint = token_bin(token)
        key, row = self._row(arm, layer, endpoint)
        flat = signed.detach().float().flatten()
        if not bool(self.torch.isfinite(flat).all().item()):
            raise FloatingPointError('nonfinite signed scale ratio')
        row['count'] += flat.numel()
        row['positive_count'] += int((flat > 0).sum().item())
        row['negative_count'] += int((flat < 0).sum().item())
        positive = flat[flat > 0]
        negative = flat[flat < 0]
        if positive.numel():
            value = float(positive.max().item())
            row['max_positive'] = value if row['max_positive'] is None else max(row['max_positive'], value)
        if negative.numel():
            value = float(negative.min().item())
            row['min_negative'] = value if row['min_negative'] is None else min(row['min_negative'], value)
        row['clamp_count'] += int(clamp_mask.sum().item())
        row['saturation_count'] += int(saturation_count)
        row['saturation_total'] += int(saturation_total)
        self.buffers[key].append(flat)
        if parity:
            cpu = flat.cpu()
            self.parity_values.append(cpu)
            self.parity_stream += self.torch.histc(
                cpu.clamp(HIST_MIN, HIST_MAX), bins=HIST_BINS,
                min=HIST_MIN, max=HIST_MAX).to(self.torch.int64)

    def flush(self):
        for key, values in self.buffers.items():
            if not values:
                continue
            # CUDA histc has no deterministic implementation.  Histogramming is
            # an observational reducer only, so transfer the already-computed
            # signed ratios to CPU and preserve the frozen bins exactly.
            flat = self.torch.cat(values).detach().float().cpu().clamp(
                HIST_MIN, HIST_MAX)
            counts = self.torch.histc(flat, bins=HIST_BINS,
                                      min=HIST_MIN, max=HIST_MAX)
            self.rows[key]['hist'] += counts.to(self.torch.int64)
            values.clear()

    def _quantile(self, hist, total, q):
        target = max(1, int(math.ceil(total*q)))
        index = int(self.torch.searchsorted(self.torch.cumsum(hist, 0),
                                            self.torch.tensor(target)).item())
        width = (HIST_MAX-HIST_MIN)/HIST_BINS
        return HIST_MIN + (index+0.5)*width

    def summary(self):
        self.flush()
        result = []
        for (arm, layer, endpoint), row in sorted(self.rows.items()):
            if int(row['hist'].sum().item()) != row['count']:
                raise RuntimeError('streaming histogram count mismatch')
            n = row['count']
            result.append({
                'arm': arm, 'layer': layer, 'token_bin_end': endpoint,
                'count': n,
                'median_signed_log_scale_ratio': self._quantile(row['hist'], n, .5),
                'p90_signed_log_scale_ratio': self._quantile(row['hist'], n, .9),
                'p99_signed_log_scale_ratio': self._quantile(row['hist'], n, .99),
                'max_positive_deviation': row['max_positive'],
                'min_negative_deviation': row['min_negative'],
                'fraction_scale_adapt_gt_clean': row['positive_count']/n,
                'fraction_scale_adapt_lt_clean': row['negative_count']/n,
                'clamp_count': row['clamp_count'],
                'clamp_fraction': row['clamp_count']/n,
                'saturation_count': row['saturation_count'],
                'saturation_fraction': row['saturation_count']/max(1, row['saturation_total'])})
        return result

    def parity_report(self):
        if not self.parity_values:
            return None
        exact = self.torch.histc(
            self.torch.cat(self.parity_values).clamp(HIST_MIN, HIST_MAX),
            bins=HIST_BINS, min=HIST_MIN, max=HIST_MAX).to(self.torch.int64)
        passed = self.torch.equal(exact, self.parity_stream)
        return {'G8_STREAMING_REDUCER_PARITY': 'PASS' if passed else 'FAIL',
                'oracle_count': int(exact.sum().item()),
                'stream_count': int(self.parity_stream.sum().item()),
                'integer_histogram_exact': bool(passed),
                'oracle_histogram_sha256': hashlib.sha256(exact.numpy().tobytes()).hexdigest(),
                'stream_histogram_sha256': hashlib.sha256(self.parity_stream.numpy().tobytes()).hexdigest()}


def ordinary_qdq_unit_test(R, adapter):
    torch = R.torch
    source = {layer: torch.linspace(-2.0, 2.0, 32*128*128,
              device='cuda:0', dtype=torch.float32).reshape(1,32,128,128)
              for layer in LAYERS}
    ordinary = {layer: x.clone() for layer, x in source.items()}
    replayed = {layer: x.clone() for layer, x in source.items()}
    scales = {layer: adapter.qparam(replayed[layer]) for layer in LAYERS}
    R.qdq(ordinary, 'reference')
    for layer in LAYERS:
        adapter.apply(replayed[layer], scales[layer])
        if not torch.equal(ordinary[layer], replayed[layer]):
            raise RuntimeError(f'identical-qparam replay mismatch layer={layer}')
    return {'IDENTICAL_QPARAM_REPLAY_GATE': 'PASS', 'exact_tensor_equality': True}


def runtime_gate(torch):
    value = {
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    expected = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'deterministic_algorithms': True,
                'tf32_matmul': False, 'tf32_cudnn': False}
    if value != expected:
        raise RuntimeError(f'deterministic runtime mismatch: {value}')
    return value


def run_replay(document, condition, rank, schedule_end, arm_end, mode, run_index):
    import torch.distributed as dist
    R = load_replay()()
    torch = R.torch
    adapter = CanonicalScaleAdapter(R)
    runtime = runtime_gate(torch)
    capture = EXP / f'discovery/doc_{document:02d}/drive'
    fp = load_fp_anchor(R, document) if rank == 0 else None
    schedule_hash = hashlib.sha256()
    schedule_entries = 0
    treatment = None
    counts = {}
    sums = {}
    j_anchor = {}
    checkpoints = {arm: {} for arm in ARMS}
    readpoint_diag = {arm: {} for arm in ARMS}
    qdq_calls = {arm: 0 for arm in ARMS}
    writeback_calls = {arm: 0 for arm in ARMS}
    positive_count = {arm: 0 for arm in ARMS}
    negative_count = {arm: 0 for arm in ARMS}
    clamp_count = {arm: 0 for arm in ARMS}
    basis_max = 0.0
    reducer = SignedStreamingReducer(torch)
    unit_test = ordinary_qdq_unit_test(R, adapter)
    if rank == 1 and arm_end:
        anchor = load_anchor(R, document, condition)
        treatment = {arm: {layer: anchor[layer].clone() for layer in LAYERS}
                     for arm in ARMS}
        for arm in ARMS:
            counts[arm], sums[arm], j_anchor[arm] = parent_prefix(document, condition)
            checkpoints[arm][str(ANCHOR_TOKEN)] = {
                'J': j_anchor[arm], 'token_count': counts[arm],
                'source': 'INHERITED_EXACT_PARENT_PREFIX'}
    elif mode == 'formal':
        anchor = load_anchor(R, document, condition)
        treatment = {arm: {layer: anchor[layer].clone() for layer in LAYERS}
                     for arm in ARMS}
        for arm in ARMS:
            counts[arm], sums[arm], j_anchor[arm] = parent_prefix(document, condition)
            checkpoints[arm][str(ANCHOR_TOKEN)] = {
                'J': j_anchor[arm], 'token_count': counts[arm],
                'source': 'INHERITED_EXACT_PARENT_PREFIX'}
    started = time.monotonic()
    with R.guard():
        for lo, hi, values in iter_drive_shards(R, capture, ANCHOR_TOKEN+1, schedule_end):
            n = hi-lo+1
            sample = load_anchor(R, document, condition)[LAYERS[0]] if False else None
            if treatment is not None:
                sample_shape = tuple(treatment[ARMS[0]][LAYERS[0]].shape)
                clean_dtype = treatment[ARMS[0]][LAYERS[0]].dtype
            else:
                # All canonical states have the frozen [1,32,128,128] FP32 carrier.
                sample_shape = (1,32,128,128)
                clean_dtype = torch.float32
            clean_batch = torch.empty((n, len(LAYERS))+sample_shape,
                                      device='cuda:0', dtype=clean_dtype)
            if rank == 0:
                for offset, token in enumerate(range(lo, hi+1)):
                    rec = record_at(values, offset)
                    fp = {layer: R.step('Native', layer, fp[layer], rec[layer])
                          for layer in LAYERS}
                    for li, layer in enumerate(LAYERS):
                        clean_batch[offset, li].copy_(fp[layer])
            dist.broadcast(clean_batch, src=0)
            shard_scales = []
            for offset, token in enumerate(range(lo, hi+1)):
                rec = record_at(values, offset)
                clean_scales = {}
                for li, layer in enumerate(LAYERS):
                    clean_physical = clean_batch[offset, li]
                    clean_basis = R.inverse(condition, layer, clean_physical)
                    if token == ANCHOR_TOKEN+1:
                        roundtrip = R.physical(condition, layer, clean_basis)
                        difference = float((roundtrip-clean_physical.float()).abs().max().item())
                        scale = max(1.0, float(clean_physical.float().abs().max().item()))
                        bound = 8*R.rules['NO_QDQ_ROTATION_INVARIANCE_GATE']['gamma128']*scale
                        if difference > bound:
                            raise RuntimeError(f'clean scale basis roundtrip failed layer={layer}')
                        basis_max = max(basis_max, difference)
                    clean_scale = adapter.qparam(clean_basis)
                    clean_scales[layer] = clean_scale
                    shard_scales.append(clean_scale.detach())
                    schedule_entries += clean_scale.numel()
                if treatment is not None and token <= arm_end:
                    for arm in ARMS:
                        pre = {layer: R.step(condition, layer, treatment[arm][layer], rec[layer])
                               for layer in LAYERS}
                        state_diag = {}
                        for li, layer in enumerate(LAYERS):
                            adaptive = adapter.qparam(pre[layer])
                            clean_scale = clean_scales[layer]
                            signed = torch.log2(adaptive/clean_scale)
                            pos = adaptive > clean_scale
                            neg = adaptive < clean_scale
                            positive_count[arm] += int(pos.sum().item())
                            negative_count[arm] += int(neg.sum().item())
                            if arm == 'NO_SCALE_INFLATION':
                                chosen = torch.minimum(adaptive, clean_scale)
                                clamp = pos
                            else:
                                chosen = torch.maximum(adaptive, clean_scale)
                                clamp = neg
                            clamp_count[arm] += int(clamp.sum().item())
                            source = adapter.source(pre[layer])
                            saturation = (source/chosen < -127) | (source/chosen > 127)
                            pre_physical = R.physical(condition, layer, pre[layer]).clone()
                            codes, _, _ = adapter.apply(pre[layer], chosen)
                            treatment[arm][layer] = pre[layer]
                            post_physical = R.physical(condition, layer, pre[layer])
                            clean_physical = clean_batch[offset, li]
                            error = post_physical.double()-clean_physical.double()
                            denom = float((clean_physical.double()**2).sum().item())+1e-12
                            value = float((error*error).sum().item())/denom
                            if not math.isfinite(value):
                                raise RuntimeError('nonfinite intervention metric')
                            sums[arm][li] += value
                            residual = post_physical.double()-pre_physical.double()
                            residual_norm = float(torch.linalg.vector_norm(residual).item())
                            pre_norm = float(torch.linalg.vector_norm(pre_physical.double()).item())
                            reducer.add(arm, layer, token, signed, clamp,
                                        int(saturation.sum().item()), saturation.numel(),
                                        parity=(mode != 'formal' and arm == ARMS[0] and
                                                layer == FOCUS_LAYERS[0] and
                                                token <= ANCHOR_TOKEN+4))
                            if token in READPOINTS:
                                state_diag[str(layer)] = {
                                    'signed_log_scale_ratio_median': float(torch.median(signed).item()),
                                    'signed_log_scale_ratio_max': float(signed.max().item()),
                                    'signed_log_scale_ratio_min': float(signed.min().item()),
                                    'QDQ_residual_norm': residual_norm,
                                    'normalized_QDQ_residual': residual_norm/max(pre_norm,1e-12),
                                    'state_error_norm': float(torch.linalg.vector_norm(error).item()),
                                    'saturation_count': int(saturation.sum().item()),
                                    'saturation_fraction': int(saturation.sum().item())/saturation.numel(),
                                    'clamp_count': int(clamp.sum().item()),
                                    'code_min': int(codes.min().item()),
                                    'code_max': int(codes.max().item())}
                        counts[arm] += 1
                        qdq_calls[arm] += 1
                        writeback_calls[arm] += 1
                        if token in READPOINTS:
                            per_layer = [x/counts[arm] for x in sums[arm]]
                            checkpoints[arm][str(token)] = {
                                'J': math.fsum(per_layer)/len(LAYERS),
                                'token_count': counts[arm], 'per_layer_J': per_layer}
                            readpoint_diag[arm][str(token)] = {
                                'per_layer': state_diag,
                                'focus_layers': {str(layer): state_diag[str(layer)]
                                                 for layer in FOCUS_LAYERS}}
            packed = torch.stack(shard_scales).float().cpu().contiguous()
            schedule_hash.update(packed.numpy().tobytes())
            reducer.flush()
            if hi % 512 == 0 or hi == schedule_end:
                print(f'MODE={mode} RUN={run_index} DOC={document} CONDITION={condition} TOKEN={hi}', flush=True)
            dist.barrier()
    schedule_digest = schedule_hash.hexdigest()
    expected_schedule = clean_result(document, condition)['qparam_schedule_sha256']
    schedule_pass = schedule_end == ENDPOINT and schedule_digest == expected_schedule
    if schedule_end == ENDPOINT and not schedule_pass:
        raise RuntimeError(f'clean scale schedule mismatch {condition}: {schedule_digest} != {expected_schedule}')
    schedule_payload = {
        'status': 'PASS' if schedule_pass else 'PARTIAL_NOT_IDENTITY_CLAIM',
        'document_index': document, 'condition': condition,
        'start_token': ANCHOR_TOKEN+1, 'end_token': schedule_end,
        'schedule_entries': schedule_entries,
        'rematerialized_qparam_schedule_sha256': schedule_digest,
        'clean_formal_qparam_schedule_sha256': expected_schedule,
        'byte_identity': bool(schedule_pass), 'basis_roundtrip_max_abs': basis_max,
        'formal_quantizer_sha256': adapter.formal_sha256}
    arm_payloads = {}
    if treatment is not None:
        summaries = reducer.summary()
        parity = reducer.parity_report()
        if parity and parity['G8_STREAMING_REDUCER_PARITY'] != 'PASS':
            raise RuntimeError('streaming reducer parity failure')
        for arm in ARMS:
            if qdq_calls[arm] != arm_end-ANCHOR_TOKEN or writeback_calls[arm] != arm_end-ANCHOR_TOKEN:
                raise RuntimeError(f'every-token QDQ/writeback count mismatch: {arm}')
            final_j = checkpoints[arm][str(arm_end)]['J']
            payload = {
                'status': 'PASS', 'mode': mode, 'run_index': run_index,
                'document_index': document, 'condition': condition, 'arm': arm,
                'anchor_token': ANCHOR_TOKEN, 'end_token': arm_end,
                'J_8192': j_anchor[arm], 'J_INTERVENTION': final_j,
                'G_INTERVENTION': final_j/j_anchor[arm],
                'explosive': final_j/j_anchor[arm] >= 1000.0,
                'checkpoints': checkpoints[arm],
                'readpoint_diagnostics': readpoint_diag[arm],
                'signed_scale_streaming_summary': [x for x in summaries if x['arm'] == arm],
                'positive_scale_deviation_count': positive_count[arm],
                'negative_scale_deviation_count': negative_count[arm],
                'clamp_count': clamp_count[arm],
                'qdq_calls_after_anchor': qdq_calls[arm],
                'int8_writeback_calls_after_anchor': writeback_calls[arm],
                'persistent_state_dtype': str(treatment[arm][LAYERS[0]].dtype),
                'paired_clean_scale_schedule': schedule_payload,
                'clean_FP_used_for_scale_only': True,
                'clean_FP_writeback_to_treatment': False,
                'runtime': runtime, 'basis_roundtrip_max_abs': basis_max,
                'software_unit_test': unit_test,
                'streaming_reducer_parity': parity,
                'state_hashes_final': {str(layer): tensor_sha(treatment[arm][layer])
                                       for layer in LAYERS},
                'elapsed_seconds': time.monotonic()-started,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                'physical_gpu': rank, 'training': False, 'optimizer_steps': 0,
                'AIME': False, 'free_generation': False}
            identity = dict(payload)
            for key in ['elapsed_seconds','peak_gpu_allocated_bytes','mode','run_index']:
                identity.pop(key)
            payload['deterministic_identity_sha256'] = canonical_hash(identity)
            arm_payloads[arm] = payload
    return schedule_payload, arm_payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build_anchors','qualification','smoke','formal'], required=True)
    parser.add_argument('--document', type=int, required=True, choices=range(1,5))
    parser.add_argument('--rank', type=int, required=True, choices=range(4))
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    condition = CONDITIONS[args.rank]
    torch, dist = init_dist(args.rank)
    try:
        if args.mode == 'build_anchors':
            build_anchor(args.document, condition, args.rank)
            dist.barrier()
            return
        gate = read_json(EXP / f'discovery/doc_{args.document:02d}/drive_identity.json')
        if gate['FROZEN_DRIVE_IDENTITY_GATE'] != 'PASS':
            raise RuntimeError('drive identity is not PASS')
        if args.mode == 'qualification':
            schedule, arms = run_replay(args.document, condition, args.rank,
                                        ENDPOINT, 12288 if args.rank == 1 else None,
                                        args.mode, args.run)
            write_json(EXP / f'qualification/schedule_identity_doc01_{condition}.json', schedule)
            if args.rank == 1:
                write_json(EXP / f'qualification/smoke_run{args.run}_H.json',
                           {'status': 'PASS', 'arms': arms})
        elif args.mode == 'smoke':
            schedule, arms = run_replay(args.document, condition, args.rank,
                                        12288, 12288 if args.rank == 1 else None,
                                        args.mode, args.run)
            if args.rank == 1:
                write_json(EXP / f'qualification/smoke_run{args.run}_H.json',
                           {'status': 'PASS', 'arms': arms})
        else:
            auth = read_json(EXP / 'execution_authorization.json')
            if auth['FORMAL_MEASUREMENT_AUTHORIZED'] != 'YES':
                raise RuntimeError('formal measurement not authorized')
            schedule, arms = run_replay(args.document, condition, args.rank,
                                        ENDPOINT, ENDPOINT, args.mode, args.run)
            for arm, payload in arms.items():
                write_json(EXP / f'discovery/unit_results/doc_{args.document:02d}_{condition}_{arm}.json', payload)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        rank = next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x == '--rank'), 'unknown')
        mode = next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x == '--mode'), 'unknown')
        run = next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x == '--run'), '1')
        path = EXP / f'discovery/logs/{mode}_run{run}_rank{rank}.error.json'
        if not path.exists():
            write_json(path, {'status': 'FAIL_AND_STOP',
                              'exception': type(exc).__name__, 'message': str(exc),
                              'traceback': traceback.format_exc(),
                              'retry_authorized': False})
        raise
