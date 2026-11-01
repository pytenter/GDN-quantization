"""Finite tests for the explicitly authorized exponent format.

Default CLI is CPU-only. The admitted worker calls explicit cuda:0 gates;
no model, AIME or formal completion credit is awarded by these tests.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import types
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL_PATH = ROOT / 'provenance/CPU_ONLY_SCALE_EXPONENT_CANDIDATE/fp16_exponent_cpu_only_20261025/ORIGINAL_FROZEN_QUANTIZER.py'
METHOD_VARIANT = 'PACE48_FP16_EXPONENT_V1'
STATE_SCALE_FORMAT = 'FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def half(value):
    try:
        return struct.unpack('<e', struct.pack('<e', value))[0]
    except OverflowError:
        return math.inf

def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)

def oracle(values, bits, allow=True):
    assert len(values) == 32 and bits in (4, 8)
    if not all((math.isfinite(x) for x in values)):
        raise ValueError('nonfinite input')
    (lo, hi) = (min(min(values), 0.0), max(max(values), 0.0))
    qmax = (1 << bits) - 1
    raw = (hi - lo) / qmax
    if not math.isfinite(raw):
        raise ValueError('nonfinite raw span')
    mantissa = half(1.0 if raw == 0 else max(raw, 2.0 ** (-24)))
    exponent = 0
    if not math.isfinite(mantissa) or mantissa <= 0:
        if not allow:
            raise ValueError('original FP16 range')
        exponent = max(1, math.frexp(raw)[1] - 16)
        if exponent > 127:
            raise ValueError('INT8 exponent range')
        mantissa = half(math.ldexp(raw, -exponent))
        if not math.isfinite(mantissa) or mantissa <= 0:
            exponent += 1
        if exponent > 127:
            raise ValueError('INT8 exponent range')
        mantissa = half(math.ldexp(raw, -exponent))
        if not math.isfinite(mantissa) or mantissa <= 0:
            raise ValueError('mantissa')
    effective = math.ldexp(mantissa, exponent)
    if not math.isfinite(f32(effective)) or f32(effective) <= 0:
        raise ValueError('FP32 scale range')
    zp = min(qmax, max(0, round(-lo / effective)))
    codes = [min(qmax, max(0, round(x / effective) + zp)) for x in values]
    decoded = [f32((q - zp) * f32(effective)) for q in codes]
    if not all((math.isfinite(x) for x in decoded)):
        raise ValueError('FP32 carrier range')
    return dict(lo=lo, hi=hi, raw=raw, mantissa=mantissa, exponent=exponent, effective=effective, zp=zp, codes=codes, decoded=decoded)

def run_stdlib_tests():
    rows = []
    for bits in (4, 8):
        qmax = (1 << bits) - 1
        for raw in (65504.0, 65505.0, 65512.0, 65519.99609375, 65520.0, 65944.02083333333, 131008.0, 131040.0, 10000000000.0, 1e+30):
            z = oracle([0.0, raw * qmax] + [0.0] * 30, bits)
            expected_zero = math.isfinite(half(raw))
            assert (z['exponent'] == 0) == expected_zero
            assert z['exponent'] <= 127 and math.isfinite(z['effective'])
            assert all((math.isfinite(x) for x in z['decoded']))
            if raw == 131040.0:
                assert z['exponent'] == 2
            rows.append(dict(bits=bits, raw=raw, exponent=z['exponent'], mantissa=z['mantissa'], effective=z['effective']))
    actual = oracle([-587500.1875, 401660.125] + [0.0] * 30, 4)
    assert actual['raw'] == 65944.02083333333
    assert (actual['exponent'], actual['mantissa'], actual['effective']) == (1, 32960.0, 65920.0)
    assert actual['zp'] == 9 and actual['codes'][:2] == [0, 15]
    assert struct.unpack('b', struct.pack('b', 127))[0] == 127
    assert 8 / 32 == 0.25
    for path in (ROOT / 'src/pace_quantizer.py', ORIGINAL_PATH, Path(__file__)):
        compile(path.read_text(), str(path), 'exec')
    tree = ast.parse((ROOT / 'src/pace_quantizer.py').read_text())
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'encode'))
    call = fn.body[-1].value
    assert any((x.arg == 'allow_scale_exponents' and x.value.value is True for x in call.keywords))
    return dict(status='PASS_STDLIB_ORACLE_AND_COMPILE', actual_torch=False, boundary_rows=rows, actual_saved_extrema_oracle=actual, extra_state_bits_per_value_when_all_groups_allocate_exponents=0.25, all_accepted_initial_FP16_casts_take_exponent_zero=True, human_authorized_new_format=True, GPU_calls=0, model_loads=0, formal_credit=0)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out

def check_storage(torch, a, b):
    assert a.bits == b.bits and a.shape == b.shape
    for name in ('codes', 'scales', 'zero_points'):
        (x, y) = (getattr(a, name), getattr(b, name))
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8))
    assert b.scale_exponents is None
    assert a.storage_nbytes() == b.storage_nbytes()
    assert torch.equal(a.integer_codes(), b.integer_codes())
    assert torch.equal(a.integer_zero_points(), b.integer_zero_points())
    assert torch.equal(a.dequantize(), b.dequantize())

def run_device_tests(device='cpu', root=None, progress=None, original_text=None, candidate_text=None):
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Only explicit CPU or logical cuda:0 is admitted')
    import torch
    actual_cuda = device == 'cuda:0'
    if root is not None and Path(root).resolve()!=ROOT.resolve():
        raise ValueError('Exact source-bound test root required')
    if actual_cuda:
        assert torch.cuda.current_device()==0

    def from_text(text, name):
        obj = types.ModuleType(name)
        obj.__file__ = '<CPU_ONLY_INJECTED_' + name + '>'
        sys.modules[name] = obj
        exec(compile(text, obj.__file__, 'exec'), obj.__dict__)
        return obj
    old = module(ORIGINAL_PATH, 'exponent_candidate_original_quantizer') if original_text is None else from_text(original_text, 'exponent_candidate_original_quantizer')
    new = module(ROOT / 'src/pace_quantizer.py', 'exponent_candidate_quantizer') if candidate_text is None else from_text(candidate_text, 'exponent_candidate_quantizer')
    tests = []
    parity_payload = []
    for bits in (4, 8):
        qmax = (1 << bits) - 1
        groups = [[0.0] * 32, [3.0] * 32, [-3.0] * 32, [2.0 ** (-40)] * 32, [-2.0 ** (-22), 2.0 ** (-22)] + [0.0] * 30, [float(i * 17 % 31 - 15) / 7 for i in range(32)], [0.0, 65504.0 * qmax] + [0.0] * 30, [0.0, 65505.0 * qmax] + [0.0] * 30, [0.0, 65512.0 * qmax] + [0.0] * 30, [0.0, 65519.99609375 * qmax] + [0.0] * 30]
        x = torch.tensor(groups, dtype=torch.float64, device=device)
        a = old.PackedAffineGroups.from_tensor(x, bits)
        b = new.PackedAffineGroups.from_tensor(x, bits, allow_scale_exponents=True)
        check_storage(torch, a, b)
        assert b.scales.device.type==('cuda' if actual_cuda else 'cpu')
        parity_payload.append(_packed_signature(b))
        a32 = old.PackedAffineGroups.from_tensor(x.float(), bits)
        b32 = new.PackedAffineGroups.from_tensor(x.float(), bits, allow_scale_exponents=True)
        check_storage(torch, a32, b32)
        tests.append(dict(test='accepted_domain_exact_storage', bits=bits, groups=len(groups), input_dtypes=['float32', 'float64']))
        for raw in (65520.0, 65944.02083333333, 131008.0, 131040.0, 10000000000.0, 1e+30):
            values = [0.0, raw * qmax] + [0.0] * 30
            expected = oracle(values, bits)
            z = new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64, device=device), bits, allow_scale_exponents=True)
            assert z.scale_exponents.dtype == torch.int8 and z.scale_exponents.shape == z.scales.shape
            assert int(z.scale_exponents.item()) == expected['exponent']
            assert float(z.scales.item()) == expected['mantissa']
            assert int(z.integer_zero_points().item()) == expected['zp']
            parity_payload.append(_packed_signature(z))
            assert z.integer_codes().tolist() == expected['codes']
            assert z.dequantize().tolist() == expected['decoded']
            assert z.resident_tensors()['scale_exponents'].numel() == 1
            assert z.storage_nbytes() == sum((t.numel() * t.element_size() for t in z.resident_tensors().values()))
            try:
                new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64, device=device), bits)
            except new.FP16ScaleOverflowError:
                pass
            else:
                raise AssertionError('Default buffer-key codec must still reject scale overflow')
            tests.append(dict(test='extension_matches_independent_oracle', bits=bits, raw=raw, exponent=expected['exponent']))
        for (name, values) in [('negative_only', [-65944.02083333333 * qmax, 0.0] + [0.0] * 30), ('mixed_sign', [-65944.02083333333 * qmax * 0.4, 65944.02083333333 * qmax * 0.6] + [0.0] * 30)]:
            expected = oracle(values, bits)
            z = new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64, device=device), bits, allow_scale_exponents=True)
            assert z.integer_codes().tolist() == expected['codes']
            assert int(z.integer_zero_points()) == expected['zp']
            assert z.dequantize().tolist() == expected['decoded']
            tests.append(dict(test='signed_extended_oracle', bits=bits, case=name))
        for center in (65520.0, 131040.0):
            raw64 = math.nextafter(center, -math.inf)
            values64 = torch.tensor([0.0, raw64 * qmax] + [0.0] * 30, dtype=torch.float64, device=device)
            actual_raw = (values64.amax().clamp(min=0) - values64.amin().clamp(max=0)) / qmax
            initial_stored = actual_raw.to(torch.float16)
            z = new.PackedAffineGroups.from_tensor(values64, bits, allow_scale_exponents=True)
            if bool(torch.isfinite(initial_stored)) and bool(initial_stored > 0):
                assert z.scale_exponents is None
                check_storage(torch, old.PackedAffineGroups.from_tensor(values64, bits), z)
            else:
                assert z.scale_exponents is not None and int(z.scale_exponents) > 0
                e = int(z.scale_exponents)
                authoritative_mantissa = torch.ldexp(actual_raw, torch.tensor(-e, dtype=torch.int32, device=device)).to(torch.float16)
                assert torch.equal(z.scales, authoritative_mantissa)
                if e > 1:
                    previous = torch.ldexp(actual_raw, torch.tensor(-(e - 1), dtype=torch.int32, device=device)).to(torch.float16)
                    assert not bool(torch.isfinite(previous))
            assert bool(torch.isfinite(z.dequantize()).all())
            tests.append(dict(test='actual_platform_initial_and_mantissa_cast_authoritative', bits=bits, center=center, raw=float(actual_raw), direct_half_prediction_finite=math.isfinite(half(float(actual_raw))), initial_finite=bool(torch.isfinite(initial_stored)), exponent=0 if z.scale_exponents is None else int(z.scale_exponents), no_universal_direct_double_to_half_assumption=True))
        mixed_values = [[0.0, 2.0] + [0.0] * 30, [0.0, 65944.02083333333 * qmax] + [0.0] * 30]
        mixed = torch.tensor(mixed_values, dtype=torch.float64, device=device)
        z = new.PackedAffineGroups.from_tensor(mixed, bits, allow_scale_exponents=True)
        ordinary = old.PackedAffineGroups.from_tensor(mixed[:1], bits)
        assert z.scale_exponents.tolist() == [0, 1]
        assert float(z.scales[0]) == float(ordinary.scales[0])
        assert torch.equal(z.integer_codes()[0], ordinary.integer_codes()[0])
        assert int(z.integer_zero_points()[0]) == int(ordinary.integer_zero_points()[0])
        assert torch.equal(z.dequantize()[0], ordinary.dequantize()[0])
        assert z.scale_exponents.numel() * z.scale_exponents.element_size() == 2
        expected_bytes = (32 if bits == 4 else 64) + 4 + (1 if bits == 4 else 2) + 2
        assert z.storage_nbytes() == expected_bytes
        tests.append(dict(test='mixed_zero_exponent_exact_and_physical_metadata', bits=bits, actual_exponent_bytes=2, actual_total_bytes=expected_bytes))
    saved = torch.tensor([-587500.1875, 401660.125] + [0.0] * 30, dtype=torch.float32, device=device)
    z = new.PackedAffineGroups.from_tensor(saved, 4, allow_scale_exponents=True)
    expected = oracle(saved.tolist(), 4)
    assert int(z.scale_exponents) == 1 and float(z.scales) == 32960.0
    assert z.integer_codes().tolist() == expected['codes'] and z.dequantize().tolist() == expected['decoded']
    tests.append(dict(test='recorded_FP32_extrema_range_fixture_not_actual32values_replay', exponent=1, mantissa=32960.0, effective=65920.0))
    high = {5, 11, 16, 17, 18, 19, 22, 23}
    mask = [i in high for i in range(32)]
    state = torch.zeros((1, 32, 128, 128), dtype=torch.float32, device=device)
    state[0, 10, 32, 51] = -587500.1875
    state[0, 10, 33, 51] = 401660.125
    packed = new.PackedAffineState.from_tensor(state, mask)
    assert packed.int4.scale_exponents.shape == (1, 24, 128, 4)
    assert int(packed.int4.scale_exponents[0, 9, 51, 1]) == 1
    assert int(torch.count_nonzero(packed.int4.scale_exponents)) == 1
    assert packed.int8.scale_exponents is None
    assert bool(torch.isfinite(packed.dequantize()).all())
    exponent_bytes = sum((t.numel() * t.element_size() for (name, t) in packed.resident_tensors().items() if name.endswith('scale_exponents')))
    assert exponent_bytes == 24 * 128 * 4 == 12288
    assert packed.storage_nbytes() == sum((t.numel() * t.element_size() for t in packed.resident_tensors().values()))
    tests.append(dict(test='fullshape_fixedmask_recorded_range_fixture_and_physical_storage', exponent_bytes=exponent_bytes, worst_case_all_state_groups_bytes=16384, key_codec_exponents_used=False))
    for (values, bits, reason) in [([0.0, 1e+101] + [0.0] * 30, 4, 'exponent_bound'), ([0.0, 1e+42] + [0.0] * 30, 4, 'FP32_scale_guard'), ([0.0, 6e+38] + [0.0] * 30, 4, 'FP32_code_times_scale_guard'), ([math.inf] + [0.0] * 31, 4, 'nonfinite_input_guard')]:
        try:
            new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64, device=device), bits, allow_scale_exponents=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing failure guard ' + reason)
        tests.append(dict(test=reason, rejected=True))
    clean = new.PackedAffineGroups.from_tensor(saved, 4, allow_scale_exponents=True)
    for bad in (torch.tensor(-1, dtype=torch.int8, device=device), torch.tensor(1, dtype=torch.int16, device=device), torch.ones((2,), dtype=torch.int8, device=device), torch.tensor(127, dtype=torch.int8, device=device)):
        malformed = new.PackedAffineGroups(clean.bits, clean.shape, clean.codes, clean.scales, clean.zero_points, bad)
        try:
            malformed.dequantize()
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed exponent metadata accepted')
    tests.append(dict(test='negative_dtype_shape_or_FP32_range_metadata_rejected', cases=4))
    mask_cases = _fixed_mask_period_cases(ROOT, new, torch, device, progress)
    return dict(status='PASS', tests=tests, count=len(tests), torch_version=torch.__version__, actual_cuda=actual_cuda, actual_device=device, model_loads=0, AIME_inputs=0, formal_credit=0, no_production_source_changed=True, old_three_formal_reuse='CONDITIONAL_ON_FINAL_ACCEPTED_PATH_AND_FIXED_PROVENANCE_PROOF', no_long_model_dynamics_stability_claim=True, method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT, accepted_domain_exact=True, exponent_dtype='int8', key_buffer_extension=False, parity_payload=parity_payload, fixed_mask_period_cases=mask_cases, guard_cases=8, recorded_extrema_are_range_fixture_not_actual_group_replay=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stdlib-only',action='store_true')
    parser.add_argument('--with-torch',action='store_true')
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--output')
    args=parser.parse_args()
    report=dict(status='PASS_STDLIB_EXPONENT_FORMAT_TESTS',stdlib=run_stdlib_tests(),
                actual_Torch='NOT_RUN',GPU_calls=0,formal_credit=0)
    if not args.stdlib_only:
        report['actual_Torch']=run_device_tests(device='cpu',root=ROOT)
        report['status']='PASS_ACTUAL_CPU_EXPONENT_FORMAT_TESTS'
    if args.output:
        with Path(args.output).open('x')as stream:
            json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(report,sort_keys=True,allow_nan=False))


def _packed_signature(encoded):
    """Bit-level codec payload only, independent of report/timing fields."""
    def byte_values(tensor):
        return tensor.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().tolist()
    return dict(bits=encoded.bits,shape=list(encoded.shape),
                tensors={k:dict(dtype=str(t.dtype),shape=list(t.shape),bytes=byte_values(t))
                         for k,t in encoded.resident_tensors().items()},
                decoded=byte_values(encoded.dequantize()))


def _fixed_mask_period_cases(root, quantizer, torch, device, progress):
    """Related new-format engineering cases, not a rerun of old qualification credit."""
    sys.path.insert(0,str(root/'src'))
    from pace_buffer import PACEBuffer
    mask_data=json.loads((root/'provenance/PRIOR_PACE_TRUE_STOP/calibration/head_mask.json').read_text())
    cases=[]
    for layer,definition in mask_data['layers'].items():
        mask=[h in definition['int8_heads'] for h in range(32)]
        h4,h8=definition['int4_heads'][0],definition['int8_heads'][0]
        for period in (1,8):
            state=torch.zeros((1,32,32,2),dtype=torch.float32,device=device)
            state[0,h4,0,0]=-587500.1875;state[0,h4,1,0]=401660.125
            state[0,h8,0,1]=-1e8;state[0,h8,1,1]=1e8
            packed=quantizer.PackedAffineState.from_tensor(state,mask)
            assert packed.int4.scale_exponents is not None and packed.int8.scale_exponents is not None
            assert sum(t.numel()*t.element_size() for n,t in packed.resident_tensors().items()
                       if n.endswith('scale_exponents'))==64
            buffer=PACEBuffer(period,mask,prefill_period=256,key_heads=16)
            buffer.set_prompt_length(0);buffer._shape=tuple(state.shape);buffer.packed_state=packed
            key=torch.zeros((1,16,32),dtype=torch.bfloat16,device=device);key[:,:,0]=1
            query=key.clone();value=torch.zeros((1,32,2),dtype=torch.bfloat16,device=device)
            beta=torch.zeros((1,32),dtype=torch.bfloat16,device=device)
            decay=torch.full((1,32),-.125,dtype=torch.float32,device=device)
            for _ in range(period):
                output=buffer.step(query,key,value,beta,decay)
                assert bool(torch.isfinite(output).all())
            assert buffer.tokens==period and buffer.pending_tokens==0
            assert buffer.quantization_count==buffer.writeback_count==1 and buffer._fp_state is None
            report=buffer.storage_report()
            assert report['dense_state_persistent'] is False and report['decode_period']==period
            assert report['total_resident_tensor_nbytes']==sum(t.numel()*t.element_size()
                   for t in buffer.resident_tensors().values())
            cases.append(dict(layer=int(layer),decode_period=period,status='PASS',actual_cuda=device=='cuda:0',
                              initial_exponent_bytes=64,actual_resident_nbytes=report['total_resident_tensor_nbytes'],
                              dense_state_persistent=False,key_buffer_extension=False))
            if progress is not None:progress('EXPONENT_FORMAT_MASK_PERIOD_CASE',layer=int(layer),decode_period=period)
    assert len(cases)==48
    return cases

if __name__ == '__main__':
    main()
