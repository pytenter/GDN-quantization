"""CPU-only candidate tests. Default stdlib; --with-torch enables real CPU tests.

No GPU, model, AIME or formal credit. This proposed format requires a new policy.
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
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite input')
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    qmax = (1 << bits) - 1
    raw = (hi-lo)/qmax
    if not math.isfinite(raw):
        raise ValueError('nonfinite raw span')
    mantissa = half(1.0 if raw == 0 else max(raw, 2.0**-24))
    exponent = 0
    if not math.isfinite(mantissa) or mantissa <= 0:
        if not allow:
            raise ValueError('original FP16 range')
        exponent = max(1, math.frexp(raw)[1]-16)
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
    zp = min(qmax, max(0, round(-lo/effective)))
    codes = [min(qmax, max(0, round(x/effective)+zp)) for x in values]
    decoded = [f32((q-zp)*f32(effective)) for q in codes]
    if not all(math.isfinite(x) for x in decoded):
        raise ValueError('FP32 carrier range')
    return dict(lo=lo, hi=hi, raw=raw, mantissa=mantissa, exponent=exponent,
                effective=effective, zp=zp, codes=codes, decoded=decoded)


def run_stdlib_tests():
    rows = []
    for bits in (4, 8):
        qmax = (1 << bits)-1
        for raw in (65504., 65505., 65512., 65519.99609375, 65520.,
                    65944.02083333333, 131008., 131040., 1e10, 1e30):
            z = oracle([0., raw*qmax]+[0.]*30, bits)
            expected_zero = math.isfinite(half(raw))
            assert (z['exponent'] == 0) == expected_zero
            assert z['exponent'] <= 127 and math.isfinite(z['effective'])
            assert all(math.isfinite(x) for x in z['decoded'])
            if raw == 131040.:
                assert z['exponent'] == 2
            rows.append(dict(bits=bits, raw=raw, exponent=z['exponent'],
                             mantissa=z['mantissa'], effective=z['effective']))
    actual = oracle([-587500.1875, 401660.125]+[0.]*30, 4)
    assert actual['raw'] == 65944.02083333333
    assert (actual['exponent'], actual['mantissa'], actual['effective']) == (1, 32960., 65920.)
    assert actual['zp'] == 9 and actual['codes'][:2] == [0, 15]
    assert struct.unpack('b', struct.pack('b', 127))[0] == 127
    assert 8/32 == .25
    for path in (HERE/'pace_quantizer_candidate.py', HERE/'ORIGINAL_FROZEN_QUANTIZER.py', Path(__file__)):
        compile(path.read_text(), str(path), 'exec')
    tree = ast.parse((HERE/'pace_quantizer_candidate.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'encode')
    call = fn.body[-1].value
    assert any(x.arg == 'allow_scale_exponents' and x.value.value is True for x in call.keywords)
    return dict(status='PASS_STDLIB_ORACLE_AND_COMPILE', actual_torch=False,
                boundary_rows=rows, actual_saved_extrema_oracle=actual,
                extra_state_bits_per_value_when_all_groups_allocate_exponents=.25,
                all_accepted_initial_FP16_casts_take_exponent_zero=True,
                proposed_format_only=True, GPU_calls=0, model_loads=0, formal_credit=0)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


def check_storage(torch, a, b):
    assert a.bits == b.bits and a.shape == b.shape
    for name in ('codes', 'scales', 'zero_points'):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x.contiguous().reshape(-1).view(torch.uint8),
                           y.contiguous().reshape(-1).view(torch.uint8))
    assert b.scale_exponents is None
    assert a.storage_nbytes() == b.storage_nbytes()
    assert torch.equal(a.integer_codes(), b.integer_codes())
    assert torch.equal(a.integer_zero_points(), b.integer_zero_points())
    assert torch.equal(a.dequantize(), b.dequantize())


def run_torch_tests(device='cpu', original_text=None, candidate_text=None):
    if device != 'cpu':
        raise ValueError('This candidate is currently CPU-only, never launch CUDA')
    import torch
    assert not torch.cuda.is_initialized()
    def from_text(text, name):
        obj = types.ModuleType(name)
        obj.__file__ = '<CPU_ONLY_INJECTED_' + name + '>'
        sys.modules[name] = obj
        exec(compile(text, obj.__file__, 'exec'), obj.__dict__)
        return obj
    old = (module(HERE/'ORIGINAL_FROZEN_QUANTIZER.py', 'exponent_candidate_original_quantizer')
           if original_text is None else from_text(original_text, 'exponent_candidate_original_quantizer'))
    new = (module(HERE/'pace_quantizer_candidate.py', 'exponent_candidate_quantizer')
           if candidate_text is None else from_text(candidate_text, 'exponent_candidate_quantizer'))
    tests = []
    for bits in (4, 8):
        qmax = (1 << bits)-1
        groups = [[0.]*32, [3.]*32, [-3.]*32, [2.**-40]*32,
                  [-2.**-22, 2.**-22]+[0.]*30,
                  [float((i*17)%31-15)/7 for i in range(32)],
                  [0., 65504.*qmax]+[0.]*30,
                  [0., 65505.*qmax]+[0.]*30,
                  [0., 65512.*qmax]+[0.]*30,
                  [0., 65519.99609375*qmax]+[0.]*30]
        x = torch.tensor(groups, dtype=torch.float64)
        a = old.PackedAffineGroups.from_tensor(x, bits)
        b = new.PackedAffineGroups.from_tensor(x, bits, allow_scale_exponents=True)
        check_storage(torch, a, b)
        a32 = old.PackedAffineGroups.from_tensor(x.float(), bits)
        b32 = new.PackedAffineGroups.from_tensor(x.float(), bits, allow_scale_exponents=True)
        check_storage(torch, a32, b32)
        tests.append(dict(test='accepted_domain_exact_storage', bits=bits, groups=len(groups), input_dtypes=['float32','float64']))
        for raw in (65520., 65944.02083333333, 131008., 131040., 1e10, 1e30):
            values = [0., raw*qmax]+[0.]*30
            expected = oracle(values, bits)
            z = new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64),
                                                  bits, allow_scale_exponents=True)
            assert z.scale_exponents.dtype == torch.int8 and z.scale_exponents.shape == z.scales.shape
            assert int(z.scale_exponents.item()) == expected['exponent']
            assert float(z.scales.item()) == expected['mantissa']
            assert int(z.integer_zero_points().item()) == expected['zp']
            assert z.integer_codes().tolist() == expected['codes']
            assert z.dequantize().tolist() == expected['decoded']
            assert z.resident_tensors()['scale_exponents'].numel() == 1
            assert z.storage_nbytes() == sum(t.numel()*t.element_size() for t in z.resident_tensors().values())
            try:
                new.PackedAffineGroups.from_tensor(torch.tensor(values, dtype=torch.float64), bits)
            except new.FP16ScaleOverflowError:
                pass
            else:
                raise AssertionError('Default buffer-key codec must still reject scale overflow')
            tests.append(dict(test='extension_matches_independent_oracle', bits=bits,
                              raw=raw, exponent=expected['exponent']))
        for name, values in [('negative_only',[-65944.02083333333*qmax,0.]+[0.]*30),
                             ('mixed_sign',[-65944.02083333333*qmax*.4,
                                            65944.02083333333*qmax*.6]+[0.]*30)]:
            expected = oracle(values,bits)
            z = new.PackedAffineGroups.from_tensor(torch.tensor(values,dtype=torch.float64),bits,
                                                   allow_scale_exponents=True)
            assert z.integer_codes().tolist()==expected['codes']
            assert int(z.integer_zero_points())==expected['zp']
            assert z.dequantize().tolist()==expected['decoded']
            tests.append(dict(test='signed_extended_oracle',bits=bits,case=name))
        for center in (65520., 131040.):
            raw64 = math.nextafter(center,-math.inf)
            values64 = torch.tensor([0.,raw64*qmax]+[0.]*30,dtype=torch.float64)
            actual_raw = (values64.amax().clamp(min=0)-values64.amin().clamp(max=0))/qmax
            initial_stored = actual_raw.to(torch.float16)
            z = new.PackedAffineGroups.from_tensor(values64,bits,allow_scale_exponents=True)
            if bool(torch.isfinite(initial_stored)) and bool(initial_stored>0):
                assert z.scale_exponents is None
                check_storage(torch,old.PackedAffineGroups.from_tensor(values64,bits),z)
            else:
                assert z.scale_exponents is not None and int(z.scale_exponents)>0
                e = int(z.scale_exponents)
                authoritative_mantissa = torch.ldexp(actual_raw,torch.tensor(-e,dtype=torch.int32)).to(torch.float16)
                assert torch.equal(z.scales,authoritative_mantissa)
                if e > 1:
                    previous = torch.ldexp(actual_raw,torch.tensor(-(e-1),dtype=torch.int32)).to(torch.float16)
                    assert not bool(torch.isfinite(previous))
            assert bool(torch.isfinite(z.dequantize()).all())
            tests.append(dict(test='actual_platform_initial_and_mantissa_cast_authoritative',bits=bits,
                              center=center,raw=float(actual_raw),
                              direct_half_prediction_finite=math.isfinite(half(float(actual_raw))),
                              initial_finite=bool(torch.isfinite(initial_stored)),
                              exponent=0 if z.scale_exponents is None else int(z.scale_exponents),
                              no_universal_direct_double_to_half_assumption=True))
        mixed_values = [[0., 2.]+[0.]*30, [0., 65944.02083333333*qmax]+[0.]*30]
        mixed = torch.tensor(mixed_values, dtype=torch.float64)
        z = new.PackedAffineGroups.from_tensor(mixed, bits, allow_scale_exponents=True)
        ordinary = old.PackedAffineGroups.from_tensor(mixed[:1], bits)
        assert z.scale_exponents.tolist() == [0, 1]
        assert float(z.scales[0]) == float(ordinary.scales[0])
        assert torch.equal(z.integer_codes()[0], ordinary.integer_codes()[0])
        assert int(z.integer_zero_points()[0]) == int(ordinary.integer_zero_points()[0])
        assert torch.equal(z.dequantize()[0], ordinary.dequantize()[0])
        assert z.scale_exponents.numel()*z.scale_exponents.element_size() == 2
        expected_bytes = (32 if bits == 4 else 64)+4+(1 if bits == 4 else 2)+2
        assert z.storage_nbytes() == expected_bytes
        tests.append(dict(test='mixed_zero_exponent_exact_and_physical_metadata', bits=bits,
                          actual_exponent_bytes=2, actual_total_bytes=expected_bytes))
    saved = torch.tensor([-587500.1875, 401660.125]+[0.]*30, dtype=torch.float32)
    z = new.PackedAffineGroups.from_tensor(saved, 4, allow_scale_exponents=True)
    expected = oracle(saved.tolist(), 4)
    assert int(z.scale_exponents)==1 and float(z.scales)==32960.
    assert z.integer_codes().tolist()==expected['codes'] and z.dequantize().tolist()==expected['decoded']
    tests.append(dict(test='recorded_FP32_extrema_range_fixture_not_actual32values_replay', exponent=1, mantissa=32960., effective=65920.))
    # Original fixed layer0 mask; use recorded extrema plus synthetic zeros.
    # The other30 original values were not saved; this is not actualgroup replay.
    high = {5, 11, 16, 17, 18, 19, 22, 23}
    mask = [i in high for i in range(32)]
    state = torch.zeros((1, 32, 128, 128), dtype=torch.float32)
    state[0, 10, 32, 51] = -587500.1875
    state[0, 10, 33, 51] = 401660.125
    packed = new.PackedAffineState.from_tensor(state, mask)
    assert packed.int4.scale_exponents.shape == (1, 24, 128, 4)
    assert int(packed.int4.scale_exponents[0, 9, 51, 1]) == 1
    assert int(torch.count_nonzero(packed.int4.scale_exponents)) == 1
    assert packed.int8.scale_exponents is None
    assert bool(torch.isfinite(packed.dequantize()).all())
    exponent_bytes = sum(t.numel()*t.element_size() for name,t in packed.resident_tensors().items()
                         if name.endswith('scale_exponents'))
    assert exponent_bytes == 24*128*4 == 12288
    assert packed.storage_nbytes() == sum(t.numel()*t.element_size() for t in packed.resident_tensors().values())
    tests.append(dict(test='fullshape_fixedmask_recorded_range_fixture_and_physical_storage',
                      exponent_bytes=exponent_bytes, worst_case_all_state_groups_bytes=16384,
                      key_codec_exponents_used=False))
    for values, bits, reason in [([0., 1e101]+[0.]*30, 4, 'exponent_bound'),
                                 ([0., 1e42]+[0.]*30, 4, 'FP32_scale_guard'),
                                 ([0., 6e38]+[0.]*30, 4, 'FP32_code_times_scale_guard'),
                                 ([math.inf]+[0.]*31, 4, 'nonfinite_input_guard')]:
        try:
            new.PackedAffineGroups.from_tensor(torch.tensor(values,dtype=torch.float64),
                                              bits,allow_scale_exponents=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing failure guard '+reason)
        tests.append(dict(test=reason, rejected=True))
    # Tampered metadata must never decode silently.
    clean = new.PackedAffineGroups.from_tensor(saved,4,allow_scale_exponents=True)
    for bad in (torch.tensor(-1,dtype=torch.int8),torch.tensor(1,dtype=torch.int16),
                torch.ones((2,),dtype=torch.int8),torch.tensor(127,dtype=torch.int8)):
        malformed = new.PackedAffineGroups(clean.bits,clean.shape,clean.codes,clean.scales,
                                          clean.zero_points,bad)
        try:malformed.dequantize()
        except ValueError:pass
        else:raise AssertionError('Malformed exponent metadata accepted')
    tests.append(dict(test='negative_dtype_shape_or_FP32_range_metadata_rejected', cases=4))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_ACTUAL_CPU_TORCH_CANDIDATE_TESTS',tests=tests,count=len(tests),
                torch_version=torch.__version__,actual_cuda=False,CUDA_context_initialized=False,
                model_loads=0,AIME_inputs=0,formal_credit=0,no_production_source_changed=True,
                old_three_formal_reuse='CONDITIONAL_ON_FINAL_ACCEPTED_PATH_AND_FIXED_PROVENANCE_PROOF',
                no_long_model_dynamics_stability_claim=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-torch',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args()
    report=dict(status='PASS_CPU_ONLY_CANDIDATE_STDLIB_TESTS',candidate_sha256=sha(HERE/'pace_quantizer_candidate.py'),
                original_sha256=sha(HERE/'ORIGINAL_FROZEN_QUANTIZER.py'),stdlib=run_stdlib_tests(),
                actual_Torch='NOT_RUN',GPU='NOT_RUN_NO_AUTHORIZED_FORMAT_POLICY',formal='NOT_RUN')
    if args.with_torch:
        report['actual_Torch']=run_torch_tests()
        report['status']='PASS_CPU_ONLY_PROPOSED_FORMAT_CANDIDATE'
    if args.output:
        with Path(args.output).open('x')as stream:
            json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
