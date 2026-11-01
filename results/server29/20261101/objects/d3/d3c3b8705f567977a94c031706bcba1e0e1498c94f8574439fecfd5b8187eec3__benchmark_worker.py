#!/usr/bin/env python3
"""One explicitly separate H/Dual-M throughput prefix; no scoring or retries.

The SHA-pinned original Components/generate implement all numerical operations.
This wrapper changes only authorization, output I/O and the authorized 2048 cap.
Import/compile is standard-library-only. CUDA is inaccessible until admission.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# All imported frozen modules remain read-only, including their pyc locations.
sys.dont_write_bytecode = True

TASK = 'QWEN_GDN_H_DUALM_AIME26_THROUGHPUT_20261017_V1'
OLD_ROOT = '/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1/repair_20261017_v1'
PINS = {
    'old_source_manifest_sha256': '3ad66d7fffe0687724ad0b89a99ce1f48ee9740f4a7323cae0081ec78b36cf4c',
    'old_prereg_sha256': '488e1f4fa58de02e18c9898c8d77b5963a8f7dadf17d5c3d488abc90c2e9888e',
    'old_runtime_manifest_sha256': 'fc72e7c6b55d5becd839f9a36e2f96c7793ce4c8b6ccebbcc5fee9abb2886a6f',
    'old_worker_sha256': 'f204487007fec03d95cc6a403fc12812c3d9ebc2d3ddc39464bd38f811f27358',
    'old_plan_sha256': 'a0613ebdf249f90f6ef128b283733ca4368baca8fbeef6e09b1f4db25cd0fa79',
    'old_model_identity_sha256': '1a9ad56c34ad33440e80994bf5927876483ef514c9fc43fe916bce83c45866de',
    'old_H_identity_sha256': 'bb0482f0b9cb1e084d09ba3d1f3e5964105264f1e9519f84d72ced76f68928a5',
}
UNITS_SHA = '230319f4eebd3261a2561847403294aa46be534baebb3453de2ea9e094f064d8'
GPU_MAP = {
    '0': 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422',
    '1': 'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
    '2': 'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
    '3': 'GPU-30b07aba-a728-7649-4899-e09d33f8bc75',
}


def check(ok, reason):
    if not bool(ok):
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def verify_new_manifest(root, digest):
    path = root / 'SOURCE_MANIFEST.json'
    check(sha(path) == digest, 'New source manifest SHA changed')
    manifest = read(path)
    rows = manifest.get('execution_sources', []) + manifest.get('external_dependencies', [])
    check(rows, 'Empty benchmark source manifest')
    found_self = False
    for row in rows:
        p = Path(row['path'])
        if not p.is_absolute():
            p = root / p
        check(p.is_file() and sha(p) == row['sha256'], 'Benchmark dependency changed: ' + str(p))
        check(p.stat().st_size == row['bytes'], 'Benchmark dependency bytes changed: ' + str(p))
        found_self |= p.resolve() == Path(__file__).resolve()
    check(found_self, 'Benchmark worker not bound by source manifest')
    return manifest


def admit(root, args):
    auth = read(args.authorization)
    check(auth.get('status') == 'PASS' and auth.get('task_id') == TASK, 'Benchmark authorization missing PASS/task')
    check(auth.get('allowed_mode') == 'throughput', 'Only explicit throughput authorization accepted')
    check(sha(root / 'PLAN.json') == auth['plan_sha256'], 'Benchmark PLAN SHA changed')
    verify_new_manifest(root, auth['SOURCE_MANIFEST_sha256'])
    plan = read(root / 'PLAN.json')
    check(plan['task_id'] == TASK and isinstance(plan['RUN_ID'], str) and plan['RUN_ID'], 'Benchmark task/run identity changed')
    check(plan['old_root'] == OLD_ROOT, 'Original qualified root changed')
    for key, expected in PINS.items():
        check(plan[key] == expected, 'Original source binding changed: ' + key)
    check(plan['gpu_uuid_map'] == GPU_MAP, 'Fixed physical GPU mapping changed')
    check(plan['output_cap'] == 2048 and plan['warmup_tokens'] == 128 and plan['original_formal_cap'] == 81920,
          'Explicit benchmark cap/window changed')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu), 'Visibility must be exactly authorized physical GPU')
    oldroot = Path(plan['old_root'])
    worker = oldroot / 'source/shared_runtime_worker.py'
    check(sha(worker) == PINS['old_worker_sha256'], 'Original numerical worker changed')
    spec = importlib.util.spec_from_file_location('frozen_h_dualm_worker', worker)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.verify_manifest(oldroot, 'SOURCE_MANIFEST.json', PINS['old_source_manifest_sha256'])
    old.verify_manifest(oldroot, 'RUNTIME_SOURCE_MANIFEST.json', PINS['old_runtime_manifest_sha256'])
    for rel, key in (
        ('PREREGISTRATION.json', 'old_prereg_sha256'),
        ('MODEL_TOKENIZER_FRESH_IDENTITY.json', 'old_model_identity_sha256'),
        ('H_DUAL_M_IDENTITY.json', 'old_H_identity_sha256'),
        ('calibration/qwen35_9b_stepquant6_plan.pt', 'old_plan_sha256'),
    ):
        check(sha(oldroot / rel) == PINS[key], 'Original artifact changed: ' + rel)
    check(read(oldroot / 'MODEL_TOKENIZER_FRESH_IDENTITY.json')['status'] == 'PASS', 'Original model identity not PASS')
    gate = read(oldroot / 'calibration/PLAN_CPU_QUALIFICATION.json')
    check(gate['status'] == 'PASS' and gate['plan_sha256'] == PINS['old_plan_sha256'], 'Actual plan CPU gate missing')
    check(sha(oldroot / 'FROZEN_UNITS.json') == UNITS_SHA, 'Original frozen AIME units changed')
    units = read(oldroot / 'FROZEN_UNITS.json')['units']
    check(len(units) == 60 and plan['units'] == units[:4], 'Benchmark must use exact first four original units')
    check([r['canonical_index'] for r in plan['units']] == [0, 1, 2, 3], 'Prefix question selection changed')
    for row in plan['units']:
        i = row['canonical_index']
        check(row['gpu'] == i and row['seed'] == [1, 2][i % 2] and row.get('gpu_uuid', GPU_MAP[str(i)]) == GPU_MAP[str(i)],
              'Benchmark seed/GPU identity changed')
    item = plan['units'][args.gpu]
    check(item['input_ids_hash'] and item['problem_id'], 'Frozen prompt identity missing')
    return auth, plan, old, oldroot, item


def validate_record(record, item, cap):
    check(record['status'] == 'COMPLETE', 'Original generation ended with technical failure')
    n = record['output_tokens']
    check(0 < n <= cap and n == len(record['generated_token_ids']), 'Actual output count mismatch')
    check(record['seed'] == item['seed'] and record['input_ids_hash'] == item['input_ids_hash'], 'Prompt/seed identity changed')
    q = record['quantizer_audit']; h = record['hadamard_audit']
    check(q['calls'] == n and q['prefill_calls'] == 1 and q['decode_calls'] == n - 1 and q['prefill_quantized'] is True,
          'Prefill/decode QDQ count mismatch')
    check(q['layer_writebacks'] == 24 * n and q['next_consumption_checks'] == 24 * (n - 1), 'Persistent next-consumption count mismatch')
    check(h['calls'] == 24 * n and h['phase_counts'] == {'chunk_prefill': 24, 'recurrent_decode': 24 * (n - 1)}, 'Original H call counts changed')
    check(record['generation_config']['max_new_tokens'] == cap and record['generation_config']['sampling_seed'] == item['seed'], 'Sampling cap/seed mismatch')
    check(record['finish_reason'] in ('length', 'eos'), 'Invalid completion reason')
    check(record['finish_reason'] != 'length' or n == cap, 'Short length completion')
    check(record['runtime_error'] is None and record['nonfinite'] is False, 'Technical/numeric failure in record')
    check(record['latency_s'] > 0, 'Invalid generation timing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    phase = root / 'outputs' / ('worker' + str(args.gpu))
    try:
        auth, plan, old, oldroot, item = admit(root, args)
        atomic_new(phase / 'STARTED.json', {'task_id': TASK, 'RUN_ID': plan['RUN_ID'], 'pid': os.getpid(),
                   'gpu': args.gpu, 'unix': time.time(), 'authorization_sha256': sha(args.authorization),
                   'plan_sha256': sha(root / 'PLAN.json'), 'SOURCE_MANIFEST_sha256': auth['SOURCE_MANIFEST_sha256']})
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        c = old.Components(oldroot)
        # I/O-only rebinding: a safety STOP must never write into old sealed root.
        c.dual.__globals__['K'].save = lambda rel, obj: atomic_new(phase / 'failures' / rel, obj)
        torch = c.torch
        gpu = old.gpu_identity(torch, args, {'gpu_uuid_map': plan['gpu_uuid_map']})
        model, tokenizer, qmod, metadata = c.load()
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
        own = [line.split(',') for line in active.strip().splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        check(len(own) == 1 and own[0][1].strip() == gpu['UUID'], 'Live process is not on authorized GPU UUID')
        artifact = torch.load(oldroot / 'calibration/qwen35_9b_stepquant6_plan.pt', map_location='cpu', weights_only=False)
        check(artifact['format_version'] == 2 and artifact['settings']['nominal_bits'] == 6, 'Actual qualified plan format changed')
        bindings = {**PINS, 'units_sha256': UNITS_SHA, 'benchmark_plan_sha256': sha(root / 'PLAN.json'),
                    'benchmark_source_manifest_sha256': auth['SOURCE_MANIFEST_sha256'], 'authorization_sha256': sha(args.authorization)}
        record = old.generate(c, model, tokenizer, qmod, item, 'H_DUALM', artifact, gpu, metadata, plan['output_cap'], trace=False)
        original_status = record['status']
        record.update(task=TASK, task_id=TASK, RUN_ID=plan['RUN_ID'], original_generation_status=original_status,
                      status='BENCHMARK_COMPLETE_PREFIX' if original_status == 'COMPLETE' else 'BENCHMARK_INCOMPLETE',
                      role='THROUGHPUT_ONLY_NOT_FORMAL_AIME_RESULT', scoring_status='NOT_RUN_THROUGHPUT_ONLY',
                      original_formal_cap=81920, benchmark_output_cap=plan['output_cap'], warmup_tokens=128,
                      output_coverage='CAP_REACHED' if record['output_tokens'] == plan['output_cap'] else 'EARLY_EOS' if original_status == 'COMPLETE' else 'TECHNICAL_FAILURE',
                      **bindings)
        record.update(old.identity_aliases(oldroot))
        record.update(physical_gpu_uuid=gpu['UUID'], prompt_token_sha=record.get('input_ids_hash'),
                      generated_text=record['response'], wall_time=record['latency_s'],
                      peak_cuda_allocated=record.get('peak_allocated_bytes'), peak_cuda_reserved=record.get('peak_reserved_bytes'),
                      sampling_config=record.get('generation_config'), top_k_audit=record.get('sampling_audit'),
                      q_d_q_writeback_audit=record.get('quantizer_audit'), codec_or_plan_sha=PINS['old_H_identity_sha256'])
        record['output_payload_sha256'] = old.canonical_sha(record)
        atomic_new(phase / 'RAW_BENCHMARK_PREFIX.json', record)
        checked = dict(record, status=original_status)
        validate_record(checked, item, plan['output_cap'])
        result = {'status': 'PASS', 'task_id': TASK, 'RUN_ID': plan['RUN_ID'], 'mode': 'throughput', 'method': 'H_DUALM',
                  'gpu': gpu, 'canonical_index': item['canonical_index'], 'seed': item['seed'], 'prompt_token_sha': item['input_ids_hash'],
                  'actual_output_tokens': record['output_tokens'], 'output_cap': plan['output_cap'], 'warmup_tokens': 128,
                  'coverage': record['output_coverage'], 'steady_window_coverage': 'AVAILABLE' if record['output_tokens'] > 128 else 'INSUFFICIENT_EARLY_EOS',
                  'latency_s': record['latency_s'], 'overall_tokens_per_s': record['tokens_per_s'],
                  'forward_wall_seconds': record['forward_wall_seconds'], 'qdq_and_consumption_audit_wall_seconds': record['qdq_and_consumption_audit_wall_seconds'],
                  'peak_allocated_bytes': record['peak_allocated_bytes'], 'peak_reserved_bytes': record['peak_reserved_bytes'],
                  'sampling_config': record['generation_config'], 'sampling_audit': record['sampling_audit'],
                  'quantizer_audit': record['quantizer_audit'], 'hadamard_audit': record['hadamard_audit'],
                  'raw_sha256': sha(phase / 'RAW_BENCHMARK_PREFIX.json'), 'raw_bytes': (phase / 'RAW_BENCHMARK_PREFIX.json').stat().st_size,
                  'scoring_status': 'NOT_RUN', 'original_formal_units_completed_by_this_benchmark': 0,
                  'timing_scope': 'Original generate wall: seed/render/prefill/sampling/forward/QDQ/audits/final decode; excludes model load and post-record cleanup. Root reducer uses unchanged PROGRESS t128 difference for steady window.',
                  **bindings}
        atomic_new(phase / 'RESULT.json', result)
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        failure = {'status': 'INCOMPLETE', 'task_id': TASK, 'pid': os.getpid(), 'gpu': args.gpu,
                   'unix': time.time(), 'traceback': traceback.format_exc(), 'retry_authorized': False}
        if not (phase / 'FIRST_FAILURE.json').exists():
            atomic_new(phase / 'FIRST_FAILURE.json', failure)
        raise


if __name__ == '__main__':
    main()
