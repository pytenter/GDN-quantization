import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from common import (ANCHOR_TOKEN, CONDITIONS, ENDPOINT, EXP,
                    EXPLOSION_THRESHOLD, FIRST_TOKEN, FOCUS_LAYERS, GPU_MAP,
                    LAYERS, PARENT, PARENT_SCRIPTS, READPOINTS, canonical_hash,
                    read_json, sha256, tensor_sha, write_json)


FORMAL = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')


def load_replay():
    sys.path.insert(0, str(PARENT_SCRIPTS))
    import capture_original_drive as capture_base
    capture_base.AMEND = PARENT
    import isolated_replay_engineering as engine
    return engine.Replay


class CanonicalQParamAdapter:
    """Compile the qparam/code/dequant statements from the frozen formal AST."""
    def __init__(self, replay):
        self.R = replay
        source = FORMAL.read_text(encoding='utf-8')
        tree = ast.parse(source)
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and
                  x.name == 'quantize_c128')
        body = fn.body[0].body
        assignments = {}
        for node in body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node
        required = ['source', 'scale', 'codes', 'dequant']
        if not all(x in assignments for x in required):
            raise RuntimeError('frozen quantizer AST dataflow changed')
        self.qparam_code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignments['source'], assignments['scale']], type_ignores=[])),
            str(FORMAL), 'exec')
        self.apply_code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignments['source'], assignments['codes'], assignments['dequant']],
            type_ignores=[])), str(FORMAL), 'exec')
        self.formal_sha256 = sha256(FORMAL)

    def qparam(self, state):
        env = {'state': state, 'torch': self.R.torch, 'EPS': self.R.F.EPS}
        exec(self.qparam_code, env)
        return env['scale']

    def apply(self, state, scale):
        env = {'state': state, 'scale': scale, 'torch': self.R.torch,
               'EPS': self.R.F.EPS}
        exec(self.apply_code, env)
        dequant = env['dequant']
        if not bool(self.R.torch.isfinite(dequant).all().item()):
            raise FloatingPointError('nonfinite C128 dequant under clean qparam')
        state.copy_(dequant.to(state.dtype))
        return env['codes'], dequant


def parent_output(document, condition):
    p = PARENT / f'scientific/doc_{document:02d}/results/{condition}/replay_result.json'
    return p, read_json(p)['identity']['outputs'][condition]


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
        values = {l: {name: tensor[:, lo-row['first_token']:hi-row['first_token']+1].to('cuda:0')
                      for name, tensor in blob['layers'][str(l)].items()}
                  for l in LAYERS}
        yield lo, hi, values


def record_at(values, offset):
    return {l: {name: tensor[:, offset:offset+1]
                for name, tensor in values[l].items()} for l in LAYERS}


def init_dist(rank):
    import torch
    import torch.distributed as dist
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(rank):
        raise RuntimeError('physical GPU mapping mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG missing')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=4)
    return torch, dist


def build_anchor(document, condition, rank):
    R = load_replay()()
    torch = R.torch
    capture = EXP / f'discovery/doc_{document:02d}/drive'
    states = {l: None for l in LAYERS}
    started = time.monotonic()
    with R.guard():
        for lo, hi, values in iter_drive_shards(R, capture, 1, ANCHOR_TOKEN):
            for token in range(lo, hi + 1):
                rec = record_at(values, token-lo)
                states = {l: R.step(condition, l, states[l], rec[l]) for l in LAYERS}
                R.qdq(states, 'path')
            if hi % 512 == 0:
                print(f'ANCHOR CONDITION={condition} TOKEN={hi}', flush=True)
    _, output = parent_output(document, condition)
    hashes = {str(l): tensor_sha(states[l]) for l in LAYERS}
    for l in LAYERS:
        if hashes[str(l)] != output['endpoints'][str(ANCHOR_TOKEN)][str(l)]['stored_Q']:
            raise RuntimeError(f'parent anchor mismatch condition={condition} layer={l}')
    anchor_dir = EXP / f'qualification/doc_{document:02d}/anchors'
    anchor_dir.mkdir(parents=True, exist_ok=True)
    path = anchor_dir / f'{condition}.pt'
    if path.exists():
        raise RuntimeError(f'write-once anchor exists: {path}')
    torch.save({str(l): states[l].detach().cpu() for l in LAYERS}, path)
    write_json(anchor_dir / f'{condition}.json', {
        'status': 'PASS', 'document_index': document, 'condition': condition,
        'token': ANCHOR_TOKEN, 'state_sha256': hashes, 'file_sha256': sha256(path),
        'parent_endpoint_exact': True, 'qdq_calls': R.calls['QDQ_path'],
        'elapsed_seconds': time.monotonic()-started, 'physical_gpu': rank})
    return {'status': 'PASS', 'condition': condition, 'anchor_file_sha256': sha256(path)}


def load_anchor(R, document, condition):
    path = EXP / f'qualification/doc_{document:02d}/anchors/{condition}.pt'
    payload = R.torch.load(path, map_location='cpu', weights_only=True)
    meta = read_json(path.with_suffix('.json'))
    if sha256(path) != meta['file_sha256']:
        raise RuntimeError('qualified anchor artifact mismatch')
    return {l: payload[str(l)].to('cuda:0') for l in LAYERS}


def load_fp_anchor(R, document):
    path = EXP / f'discovery/doc_{document:02d}/drive/fp_reference/token_08192.pt'
    manifest = read_json(EXP / f'discovery/doc_{document:02d}/drive/drive_manifest.json')
    row = next(x for x in manifest['fp_reference_checkpoints'] if x['token'] == ANCHOR_TOKEN)
    if sha256(path) != row['file_sha256']:
        raise RuntimeError('clean FP anchor mismatch')
    payload = R.torch.load(path, map_location='cpu', weights_only=True)
    return {l: payload[str(l)].to('cuda:0') for l in LAYERS}


def ordinary_qdq_unit_test(R, adapter):
    torch = R.torch
    source = {l: torch.linspace(-2.0, 2.0, 32*128*128, device='cuda:0',
                                dtype=torch.float32).reshape(1,32,128,128).to(torch.bfloat16)
              for l in LAYERS}
    ordinary = {l: x.clone() for l, x in source.items()}
    replayed = {l: x.clone() for l, x in source.items()}
    scales = {l: adapter.qparam(replayed[l]) for l in LAYERS}
    R.qdq(ordinary, 'reference')
    for l in LAYERS:
        adapter.apply(replayed[l], scales[l])
        if not torch.equal(ordinary[l], replayed[l]):
            raise RuntimeError(f'identical-qparam replay mismatch layer={l}')
    return {'IDENTICAL_QPARAM_REPLAY_GATE': 'PASS', 'layers': LAYERS,
            'exact_tensor_equality': True, 'scientific_measurement': False}


def parent_prefix(document, condition):
    _, output = parent_output(document, condition)
    snap = output['horizons'][str(ANCHOR_TOKEN)]
    count = int(snap['token_count'])
    sums = [float(row[0]) * count for row in snap['per_layer']]
    return count, sums, float(snap['normalized_mean'][0])


def run_segment(document, condition, rank, end, mode, run_index):
    import torch.distributed as dist
    Replay = load_replay()
    R = Replay()
    torch = R.torch
    adapter = CanonicalQParamAdapter(R)
    capture = EXP / f'discovery/doc_{document:02d}/drive'
    treatment = load_anchor(R, document, condition)
    fp = load_fp_anchor(R, document) if rank == 0 else None
    count, sums, j_anchor = parent_prefix(document, condition)
    checkpoints = {str(ANCHOR_TOKEN): {'J_CLEAN_QPARAM': j_anchor,
                                      'token_count': count,
                                      'source': 'INHERITED_EXACT_PARENT_PREFIX'}}
    diagnostic = {}
    schedule_hash = hashlib.sha256()
    qdq_calls = 0
    writeback_calls = 0
    qparam_different_entries = 0
    qparam_max_abs_difference = 0.0
    basis_max_abs = 0.0
    started = time.monotonic()
    env = {'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
           'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
           'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    if env != {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'deterministic_algorithms': True,
               'tf32_matmul': False, 'tf32_cudnn': False}:
        raise RuntimeError(f'deterministic runtime mismatch: {env}')

    unit_test = ordinary_qdq_unit_test(R, adapter) if mode != 'formal' else None
    with R.guard():
        for lo, hi, values in iter_drive_shards(R, capture, ANCHOR_TOKEN+1, end):
            n = hi-lo+1
            sample_shape = tuple(treatment[LAYERS[0]].shape)
            clean_batch = torch.empty((n, len(LAYERS)) + sample_shape,
                                      device='cuda:0', dtype=treatment[LAYERS[0]].dtype)
            if rank == 0:
                for offset, token in enumerate(range(lo, hi+1)):
                    rec = record_at(values, offset)
                    fp = {l: R.step('Native', l, fp[l], rec[l]) for l in LAYERS}
                    for li, l in enumerate(LAYERS):
                        clean_batch[offset, li].copy_(fp[l])
            dist.broadcast(clean_batch, src=0)
            shard_scales = []
            for offset, token in enumerate(range(lo, hi+1)):
                rec = record_at(values, offset)
                pre = {l: R.step(condition, l, treatment[l], rec[l]) for l in LAYERS}
                state_diag = {}
                for li, l in enumerate(LAYERS):
                    clean_physical = clean_batch[offset, li]
                    clean_basis = R.inverse(condition, l, clean_physical)
                    if token == ANCHOR_TOKEN+1:
                        roundtrip = R.physical(condition, l, clean_basis)
                        basis_difference = float((roundtrip-clean_physical.float()).abs().max())
                        basis_scale = max(1.0, float(clean_physical.float().abs().max()))
                        basis_bound = 8 * R.rules['NO_QDQ_ROTATION_INVARIANCE_GATE']['gamma128'] * basis_scale
                        if basis_difference > basis_bound:
                            raise RuntimeError(f'clean qparam basis roundtrip failed layer={l}')
                        basis_max_abs = max(basis_max_abs, basis_difference)
                    clean_scale = adapter.qparam(clean_basis)
                    adaptive_scale = adapter.qparam(pre[l])
                    diff = adaptive_scale-clean_scale
                    qparam_different_entries += int((diff != 0).sum().item())
                    qparam_max_abs_difference = max(qparam_max_abs_difference,
                                                    float(diff.abs().max().item()))
                    pre_physical = R.physical(condition, l, pre[l]).clone()
                    codes, _ = adapter.apply(pre[l], clean_scale)
                    treatment[l] = pre[l]
                    post_physical = R.physical(condition, l, treatment[l])
                    fp64 = clean_physical.double()
                    err = post_physical.double()-fp64
                    denom = float((fp64*fp64).sum()) + 1e-12
                    value = float((err*err).sum())/denom
                    if not math.isfinite(value):
                        raise RuntimeError('nonfinite clean-qparam metric')
                    sums[li] += value
                    shard_scales.append(clean_scale.detach())
                    if token in READPOINTS:
                        residual = post_physical.double()-pre_physical.double()
                        state_diag[str(l)] = {
                            'state_error_norm': float(torch.linalg.vector_norm(err).item()),
                            'QDQ_residual_norm': float(torch.linalg.vector_norm(residual).item()),
                            'qparam_deviation_norm': float(torch.linalg.vector_norm(diff.double()).item()),
                            'qparam_deviation_max_abs': float(diff.abs().max().item()),
                            'code_min': int(codes.min().item()),
                            'code_max': int(codes.max().item())}
                count += 1
                qdq_calls += 1
                writeback_calls += 1
                if token in READPOINTS:
                    per_layer = [x/count for x in sums]
                    checkpoints[str(token)] = {
                        'J_CLEAN_QPARAM': math.fsum(per_layer)/len(LAYERS),
                        'token_count': count, 'per_layer_J': per_layer}
                    diagnostic[str(token)] = {
                        'per_layer': state_diag,
                        'focus_layers': {str(l): state_diag[str(l)] for l in FOCUS_LAYERS}}
            packed = torch.stack(shard_scales).float().cpu().contiguous()
            schedule_hash.update(packed.numpy().tobytes())
            if hi % 512 == 0 or hi == end:
                print(f'MODE={mode} RUN={run_index} DOC={document} CONDITION={condition} TOKEN={hi}', flush=True)
            dist.barrier()

    if qdq_calls != end-ANCHOR_TOKEN or writeback_calls != end-ANCHOR_TOKEN:
        raise RuntimeError('every-token QDQ/writeback count mismatch')
    final_j = checkpoints[str(end)]['J_CLEAN_QPARAM'] if str(end) in checkpoints else math.fsum(x/count for x in sums)/len(LAYERS)
    payload = {
        'status': 'PASS', 'mode': mode, 'run_index': run_index,
        'document_index': document, 'condition': condition,
        'anchor_token': ANCHOR_TOKEN, 'end_token': end,
        'J_8192': j_anchor, 'J_CLEAN_QPARAM': final_j,
        'G_CLEAN_QPARAM': final_j/j_anchor,
        'explosive': bool(final_j/j_anchor >= EXPLOSION_THRESHOLD),
        'checkpoints': checkpoints, 'readpoint_diagnostics': diagnostic,
        'qparam_bundle': ['scale'], 'qparam_schedule_sha256': schedule_hash.hexdigest(),
        'qparam_different_entries': qparam_different_entries,
        'qparam_max_abs_difference': qparam_max_abs_difference,
        'basis_roundtrip_max_abs': basis_max_abs,
        'qdq_calls_after_anchor': qdq_calls,
        'int8_writeback_calls_after_anchor': writeback_calls,
        'persistent_state_dtype': str(treatment[LAYERS[0]].dtype),
        'clean_FP_used_for_qparams_only': True,
        'clean_FP_writeback_to_treatment': False,
        'clean_FP_replay_owner_rank': 0,
        'clean_FP_replays_per_document_group': 1,
        'runtime': env, 'formal_quantizer_sha256': adapter.formal_sha256,
        'software_unit_test': unit_test,
        'state_hashes_final': {str(l): tensor_sha(treatment[l]) for l in LAYERS},
        'elapsed_seconds': time.monotonic()-started,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'physical_gpu': rank, 'training': False, 'optimizer_steps': 0,
        'AIME': False, 'free_generation': False,
    }
    identity = dict(payload)
    identity.pop('elapsed_seconds')
    identity.pop('peak_gpu_allocated_bytes')
    identity.pop('mode')
    identity.pop('run_index')
    payload['deterministic_identity_sha256'] = canonical_hash(identity)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build_anchors', 'same_process', 'fresh', 'formal'], required=True)
    parser.add_argument('--document', type=int, required=True, choices=range(1,5))
    parser.add_argument('--rank', type=int, required=True, choices=range(4))
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    condition = CONDITIONS[args.rank]
    torch, dist = init_dist(args.rank)
    try:
        if args.mode == 'build_anchors':
            result = build_anchor(args.document, condition, args.rank)
            dist.barrier()
            return
        gate = read_json(EXP / f'discovery/doc_{args.document:02d}/drive_identity.json')
        if gate['FROZEN_DRIVE_IDENTITY_GATE'] != 'PASS':
            raise RuntimeError('drive identity is not PASS')
        if args.mode == 'formal':
            auth = read_json(EXP / 'execution_authorization.json')
            if auth['FORMAL_MEASUREMENT_AUTHORIZED'] != 'YES':
                raise RuntimeError('formal measurement not authorized')
            end = ENDPOINT
            result = run_segment(args.document, condition, args.rank, end, args.mode, args.run)
            output = EXP / f'discovery/unit_results/doc_{args.document:02d}_{condition}.json'
            write_json(output, result)
        elif args.mode == 'same_process':
            end = args.end or 8448
            first = run_segment(args.document, condition, args.rank, end, args.mode, 1)
            dist.barrier()
            second = run_segment(args.document, condition, args.rank, end, args.mode, 2)
            if first['deterministic_identity_sha256'] != second['deterministic_identity_sha256']:
                raise RuntimeError('same-process deterministic identity mismatch')
            output = EXP / f'qualification/same_process_{condition}.json'
            write_json(output, {'status': 'PASS', 'condition': condition,
                                'runs': [first, second],
                                'identity_equal': True})
        else:
            end = args.end or 8448
            result = run_segment(args.document, condition, args.rank, end, args.mode, args.run)
            output = EXP / f'qualification/fresh_{args.run}_{condition}.json'
            write_json(output, result)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        rank = next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x == '--rank'), 'unknown')
        mode = next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x == '--mode'), 'unknown')
        path = EXP / f'discovery/logs/{mode}_rank{rank}.error.json'
        if not path.exists():
            write_json(path, {'status': 'FAIL_AND_STOP', 'exception': type(exc).__name__,
                              'message': str(exc), 'traceback': traceback.format_exc(),
                              'retry_authorized': False})
        raise
