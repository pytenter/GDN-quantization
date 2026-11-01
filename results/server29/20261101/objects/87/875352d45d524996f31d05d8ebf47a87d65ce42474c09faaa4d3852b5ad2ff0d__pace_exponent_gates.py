"""Finite CPU/CUDA engineering gates for the explicitly authorized exponent format.

No model, calibration, answers, score or formal completion credit. The actual
worker chooses the device only after frozen permit/dispatch/UUID admission.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

METHOD_VARIANT = 'PACE48_FP16_EXPONENT_V1'
STATE_SCALE_FORMAT = 'FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT'


def run_exponent_format_tests(root, device='cuda:0', progress=None):
    root = Path(root).resolve()
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Only explicit CPU or logical cuda:0 is supported')
    path = root / 'tests/test_pace_exponent_format.py'
    spec = importlib.util.spec_from_file_location('pace_current_exponent_format_tests', path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    if progress is not None:
        progress('EXPONENT_FORMAT_CPU_REFERENCE')
    reference = tests.run_device_tests(device='cpu', root=root)
    if device == 'cuda:0':
        if progress is not None:
            progress('EXPONENT_FORMAT_ACTUAL_CUDA_CASES')
        actual = tests.run_device_tests(device=device, root=root, progress=progress)
    else:
        actual = reference
    if reference['status'] != 'PASS' or actual['status'] != 'PASS':
        raise RuntimeError('Exponent format gate failed')
    cpu_payload = reference.pop('parity_payload')
    device_payload = actual.pop('parity_payload') if actual is not reference else cpu_payload
    if cpu_payload != device_payload:
        raise RuntimeError('Actual CPU/GPU packed exponent codec payload differs')
    raw = json.dumps(cpu_payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    parity = dict(status='PASS', actual_cuda=device == 'cuda:0',
                  scope='EXACT_CODEC_CODES_FP16_MANTISSA_INT8_EXPONENT_INTEGER_ZP_AND_FP32_DECODE_BYTES',
                  fixtures=len(cpu_payload), matched_payload_sha256=hashlib.sha256(raw).hexdigest(),
                  platform_neighbor_casts_are_authoritative_separate_cases=True)
    return dict(status='PASS', actual_cuda=device == 'cuda:0', actual_device=device,
                method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT,
                accepted_domain_exact=True, exponent_dtype='int8', key_buffer_extension=False,
                CPU_GPU_parity=parity, CPU_reference=reference, actual_device_tests=actual,
                fixed_mask_period_cases=actual['fixed_mask_period_cases'],
                physical_exponent_metadata_counted=True,
                no_AIME_or_calibration_or_formal_credit=True,
                no_long_model_dynamics_stability_claim=True,
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                               ('src/pace_quantizer.py','src/pace_buffer.py','src/pace_runtime.py',
                                'src/pace_exponent_gates.py','tests/test_pace_exponent_format.py')})
