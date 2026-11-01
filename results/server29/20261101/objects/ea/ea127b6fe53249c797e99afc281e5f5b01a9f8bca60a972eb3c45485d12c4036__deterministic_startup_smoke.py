import argparse
import os
import sys
import time

from common import EXP, GPU_MAP, LAYERS, write_json
from pulse_replay import load_replay


EXPECTED_WORKSPACE = ':4096:8'
ROTATED_CONDITIONS = ('H', 'C6R', 'C7')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=ROTATED_CONDITIONS, required=True)
    args = parser.parse_args()

    expected_gpu = str(GPU_MAP[args.condition])
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != EXPECTED_WORKSPACE:
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG did not enter child environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_gpu:
        raise RuntimeError('CUDA_VISIBLE_DEVICES differs from frozen GPU mapping')

    Replay = load_replay()
    replay = Replay()
    torch = replay.torch
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('PyTorch deterministic algorithms are not enabled')
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('CUDA matmul TF32 unexpectedly enabled')
    if torch.backends.cudnn.allow_tf32:
        raise RuntimeError('cuDNN TF32 unexpectedly enabled')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('startup smoke must see exactly one mapped GPU')

    capture_path = EXP / 'discovery/doc_01/drive'
    layer = LAYERS[0]
    started = time.monotonic()
    with replay.guard():
        token, records = next(replay.drive(1, capture_path))
        record = records[layer]
        state = replay.step(args.condition, layer, None, record)
    if not bool(torch.isfinite(state).all()):
        raise RuntimeError('startup smoke produced non-finite recurrent state')

    payload = {
        'status': 'PASS',
        'classification': 'ENGINEERING_ONLY_DETERMINISTIC_RUNTIME_FIX',
        'condition': args.condition,
        'physical_gpu': GPU_MAP[args.condition],
        'visible_device_count': torch.cuda.device_count(),
        'visible_device_name': torch.cuda.get_device_name(0),
        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'token': token,
        'layer': layer,
        'query_dtype': str(record['query'].dtype),
        'key_dtype': str(record['key'].dtype),
        'value_dtype': str(record['value'].dtype),
        'state_dtype': str(state.dtype),
        'state_shape': list(state.shape),
        'state_finite': True,
        'elapsed_seconds': time.monotonic() - started,
        'scientific_measurement': False,
        'optimizer_steps': 0,
    }
    output = (EXP / 'engineering/deterministic_runtime_fix' /
              f'startup_smoke_{args.condition}.json')
    write_json(output, payload)
    print(f'DETERMINISTIC_STARTUP_SMOKE_PASS {args.condition} {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        condition = next((sys.argv[i + 1] for i, item in enumerate(sys.argv)
                          if item == '--condition'), 'unknown')
        output = (EXP / 'engineering/deterministic_runtime_fix' /
                  f'startup_smoke_{condition}.error.json')
        if not output.exists():
            write_json(output, {
                'status': 'FAIL_AND_STOP',
                'classification': 'ENGINEERING_ONLY_DETERMINISTIC_RUNTIME_FIX',
                'condition': condition,
                'exception': type(exc).__name__,
                'message': str(exc),
            })
        raise
