"""Finite IEEE-half boundary checks and small-shape CPU persistence stress.

The scalar oracle and AST proof require only Python's standard library.
Production tensor tests import Torch lazily; CUDA is used only when the caller
explicitly supplies a CUDA device. No model, AIME text, scorer or controller runs.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD_QUANTIZER_SHA256 = "e14ddd708f047f0b3a095f70082c0f1612cecda90a972d74d7242dfb776a25b9"
STRESS_TOKENS = 16384
sys.dont_write_bytecode = True


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _old_path(explicit=None):
    candidates = ([Path(explicit)] if explicit is not None else [
        ROOT.parent / "QWEN_GDN_PACE48_AIME26_SEED1_60_V1/src/pace_quantizer.py",
        Path("/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1/src/pace_quantizer.py"),
    ])
    for path in candidates:
        if path.is_file():
            if _sha(path) != OLD_QUANTIZER_SHA256:
                raise AssertionError("Original frozen quantizer SHA differs: " + str(path))
            return path
    raise FileNotFoundError("Supply the immutable original quantizer with --old-quantizer")


def _half(value):
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _ieee_edges():
    largest = struct.unpack("<e", bytes.fromhex("ff7b"))[0]
    previous = struct.unpack("<e", bytes.fromhex("fe7b"))[0]
    overflow_midpoint = largest + (largest - previous) / 2
    assert largest == 65504 and overflow_midpoint == 65520
    return largest, overflow_midpoint


def _fp32_previous(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits - 1))[0]


def _scalar_group(values, bits):
    if len(values) != 32 or not all(math.isfinite(x) for x in values):
        raise ValueError("Oracle requires32 finite scalar inputs")
    qmax = (1 << bits) - 1
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    raw = (hi - lo) / qmax
    scale = _half(1.0 if raw == 0 else max(raw, 2.0 ** -24))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Oracle stored half scale is invalid")
    zp = max(0, min(qmax, round(-lo / scale)))
    codes = [max(0, min(qmax, round(x / scale) + zp)) for x in values]
    return codes, scale, zp


def run_struct_oracle():
    largest, midpoint = _ieee_edges()
    cases = []
    for bits in (4, 8):
        qmax = (1 << bits) - 1
        for label, high, accepted in [
            ("largest_finite_half", largest * qmax, True),
            ("first_rejected_but_finite_65505", 65505.0 * qmax, True),
            ("rejected_but_finite_65512", 65512.0 * qmax, True),
            ("FP32_adjacent_below_overflow_midpoint", _fp32_previous(midpoint) * qmax, True),
            ("overflow_midpoint_65520", midpoint * qmax, False),
        ]:
            raw = high / qmax
            try:
                _, stored, _ = _scalar_group([0.0, high] + [0.0] * 30, bits)
                assert accepted and stored == largest
            except ValueError:
                assert not accepted
                stored = "+Infinity"
            cases.append(dict(bits=bits, case=label, high=high, raw_scale=raw,
                              old_precast_rejected=raw > largest,
                              accepted_after_actual_half_cast=accepted, stored_scale=stored))
    raw64 = math.nextafter(midpoint, -math.inf)
    direct = _half(raw64)
    via32 = _half(struct.unpack("<f", struct.pack("<f", raw64))[0])
    assert direct == largest and math.isinf(via32)
    return dict(status="PASS", independent_reference="Python struct IEEE754 binary16 and scalar arithmetic",
                cases=cases, FP64_adjacent_boundary=dict(raw_scale=raw64,
                    direct_IEEE_half=direct, IEEE_FP32_then_half="+Infinity",
                    platform_Torch_cast_policy_not_assumed=True),
                source_or_science_mutations=0, GPU_calls=0)


def run_ast_equivalence_proof(old_quantizer_path=None):
    old_path = _old_path(old_quantizer_path)
    new_path = ROOT / "src/pace_quantizer.py"
    old, new = ast.parse(old_path.read_bytes()), ast.parse(new_path.read_bytes())
    before = copy.deepcopy(old)
    after = copy.deepcopy(new)
    old_class = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "PackedAffineGroups")
    new_class = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == "PackedAffineGroups")
    old_fn = next(n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name == "from_tensor")
    new_fn = next(n for n in new_class.body if isinstance(n, ast.FunctionDef) and n.name == "from_tensor")
    guard = ast.parse("bool((raw_scale > FP16_MAX).any().item())", mode="eval").body
    guards = [n for n in old_fn.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(guard)]
    assert len(guards) == 1 and not guards[0].orelse
    assert ast.dump(guards[0].body[0]) == ast.dump(ast.parse(
        'raise ValueError("affine group scale exceeds finite FP16 range")').body[0])
    old_fn.body.remove(guards[0])
    predicate = ast.parse("not bool(torch.isfinite(effective).all().item()) or bool((effective <= 0).any().item())", mode="eval").body
    old_error = [n for n in old_fn.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(predicate)]
    new_error = [n for n in new_fn.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(predicate)]
    assert len(old_error) == len(new_error) == 1
    expected_raise = ast.parse("raise FP16ScaleOverflowError(_invalid_scale_diagnostics(groups, bits, lo, hi, raw_scale, effective))").body[0]
    assert ast.dump(new_error[0].body[0]) == ast.dump(expected_raise)
    assert len(new_error[0].body) == 1 and not new_error[0].orelse
    new_error[0].body = copy.deepcopy(old_error[0].body)
    added = {"FP16ScaleOverflowError", "_scale_diagnostic_value", "_invalid_scale_diagnostics"}
    extra = [n for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in added]
    assert {n.name for n in extra} == added
    math_import = [n for n in after.body if isinstance(n, ast.Import) and ast.dump(n) == ast.dump(ast.parse("import math").body[0])]
    assert len(math_import) == 1
    after.body = [n for n in after.body if n not in extra and n not in math_import]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    return dict(status="PASS", old_quantizer_path=str(old_path), old_quantizer_sha256=_sha(old_path),
                new_quantizer_sha256=_sha(new_path), deleted_premature_guard_nodes=1,
                actual_postcast_finite_positive_predicate_identical=True,
                all_remaining_preexisting_module_AST_identical=True,
                new_failure_only_helpers=sorted(added), old_successful_mapping_exact=True,
                equivalence_domain="Every previously successful finite input: the removed branch did not execute; all original encoding/decoding/packing operations remain identical.",
                old_failure_not_reclassified=True, proof_kind="AST identity after only explicit admission/error-path normalization")


def _production():
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import pace_quantizer as quantizer
    return torch, quantizer


def _old_module(path):
    name = "pace48_immutable_old_quantizer_scale_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _same_storage(torch, old, new):
    assert old.bits == new.bits and old.shape == new.shape
    for key in ("codes", "scales", "zero_points"):
        a, b = getattr(old, key), getattr(new, key)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8))
    assert torch.equal(old.integer_codes(), new.integer_codes())
    assert torch.equal(old.integer_zero_points(), new.integer_zero_points())
    assert torch.equal(old.dequantize(), new.dequantize())


def run_boundary_tests(device="cpu", old_quantizer_path=None, progress=None):
    torch, Q = _production()
    old_path = _old_path(old_quantizer_path)
    old = _old_module(old_path)
    proof = run_ast_equivalence_proof(old_path)
    largest, midpoint = _ieee_edges()
    cases, accepted_checks, diagnostic_checks, platform_cast_checks = [], [], [], []
    for bits in (4, 8):
        qmax = (1 << bits) - 1
        for label, high, accepted in [
            ("largest_finite_half", largest * qmax, True),
            ("65505_finite_rounding_band", 65505.0 * qmax, True),
            ("65512_finite_rounding_band", 65512.0 * qmax, True),
            ("FP32_next_below_65520", _fp32_previous(midpoint) * qmax, True),
            ("65520_true_half_overflow", midpoint * qmax, False),
        ]:
            values = [0.0, high] + [0.0] * 30
            x = torch.tensor(values, dtype=torch.float64, device=device)
            if accepted:
                encoded = Q.PackedAffineGroups.from_tensor(x, bits)
                codes, scale, zp = _scalar_group(values, bits)
                assert encoded.scales.dtype == torch.float16 and float(encoded.scales.item()) == scale
                assert encoded.integer_codes().cpu().tolist() == codes
                assert int(encoded.integer_zero_points().item()) == zp
                assert bool(torch.isfinite(encoded.dequantize()).all().item())
                if high / qmax > largest:
                    try:
                        old.PackedAffineGroups.from_tensor(x, bits)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Original premature rejection was not reproduced")
            else:
                try:
                    Q.PackedAffineGroups.from_tensor(x, bits)
                except Q.FP16ScaleOverflowError as exc:
                    d = exc.diagnostics
                    assert isinstance(exc, ValueError) and d["bits"] == bits
                    assert d["group_shape"] == [32] and d["scale_shape"] == []
                    assert d["first_bad_group_indices"] == [[]] and d["bad_group_count"] == 1
                    assert d["groups"][0]["effective_scale"] == "+Infinity"
                    assert d["input_classification"] == "FINITE_FLOATING_INPUT_ALREADY_VALIDATED"
                    json.dumps(d, allow_nan=False)
                    diagnostic_checks.append(dict(bits=bits, scalar_group_diagnostic=d))
                else:
                    raise AssertionError("True FP16 overflow was silently accepted")
            cases.append(dict(bits=bits, case=label, accepted=accepted))
        # FP64's immediate predecessor can round via FP32 to the half overflow
        # midpoint. Observe that platform conversion rather than assuming a
        # direct binary64-to-binary16 rounding path. The production postcast
        # finite gate must exactly follow the actually stored representation.
        high64 = math.nextafter(midpoint * qmax, -math.inf)
        raw64 = high64 / qmax
        direct = _half(raw64)
        via32 = _half(struct.unpack("<f", struct.pack("<f", raw64))[0])
        assert direct == largest and math.isinf(via32)
        raw_tensor = torch.tensor(raw64, dtype=torch.float64, device=device)
        actual_stored = float(raw_tensor.to(torch.float16).item())
        values64 = torch.tensor([0.0, high64] + [0.0] * 30, dtype=torch.float64, device=device)
        if math.isfinite(actual_stored):
            assert actual_stored == direct
            encoded64 = Q.PackedAffineGroups.from_tensor(values64, bits)
            assert float(encoded64.scales.item()) == direct
            observed = "DIRECT_IEEE_HALF_FINITE"
        else:
            assert math.isinf(actual_stored)
            try:
                Q.PackedAffineGroups.from_tensor(values64, bits)
            except Q.FP16ScaleOverflowError as exc:
                assert exc.diagnostics["groups"][0]["effective_scale"] == "+Infinity"
                json.dumps(exc.diagnostics, allow_nan=False)
            else:
                raise AssertionError("Actual nonfinite FP16 cast was accepted")
            observed = "MATCHES_IEEE_FP32_THEN_HALF_OVERFLOW"
        platform_cast_checks.append(dict(bits=bits, raw_scale=raw64,
                                         independent_direct_half=direct,
                                         independent_FP32_then_half="+Infinity",
                                         observed_cast=observed,
                                         actual_postcast_gate_matches=True))
        rng = random.Random(20261025 + bits)
        fixtures = [[0.0] * 32, [3.25] * 32, [-4.5] * 32,
                    [-7.0, 8.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5] + [0.0] * 24,
                    [1e-30, -1e-30] * 16,
                    [0.0, largest * qmax] + [1.0] * 30,
                    [-largest * qmax, 0.0] + [-1.0] * 30]
        fixtures += [[rng.gauss(0, 1) * (10.0 ** exponent) for _ in range(32)]
                     for exponent in (-7, -3, 0, 3)]
        x = torch.tensor(fixtures, dtype=torch.float64, device=device)
        a, b = old.PackedAffineGroups.from_tensor(x, bits), Q.PackedAffineGroups.from_tensor(x, bits)
        _same_storage(torch, a, b)
        # Compare both implementations for an actual second writeback too.
        _same_storage(torch, old.PackedAffineGroups.from_tensor(a.dequantize(), bits),
                      Q.PackedAffineGroups.from_tensor(b.dequantize(), bits))
        for i, values in enumerate(fixtures):
            codes, scale, zp = _scalar_group(values, bits)
            assert b.integer_codes()[i].cpu().tolist() == codes
            assert float(b.scales[i].item()) == scale and int(b.integer_zero_points()[i].item()) == zp
        grid = torch.tensor(([0.0, largest * qmax] + [0.0] * 30), dtype=torch.float64, device=device)
        once = Q.PackedAffineGroups.from_tensor(grid, bits)
        twice = Q.PackedAffineGroups.from_tensor(once.dequantize(), bits)
        _same_storage(torch, once, twice)
        accepted_checks.append(dict(bits=bits, groups=len(fixtures), old_new_storage_bytes_exact=True,
                                    second_writeback_old_new_exact=True, endpoint_grid_idempotence_exact=True))
        invalid = torch.zeros((12, 32), dtype=torch.float64, device=device)
        invalid[:, 1] = midpoint * qmax
        try:
            Q.PackedAffineGroups.from_tensor(invalid, bits)
        except Q.FP16ScaleOverflowError as exc:
            d = exc.diagnostics
            assert d["bad_group_count"] == 12 and d["first_bad_group_indices"] == [[i] for i in range(8)]
            assert len(d["groups"]) == d["diagnostic_group_limit"] == 8
            json.dumps(d, allow_nan=False)
        else:
            raise AssertionError("Batched real overflow did not fail closed")
    # Input nonfinite validation remains the original gate, not a scale recovery.
    for value in (math.inf, -math.inf, math.nan):
        try:
            Q.PackedAffineGroups.from_tensor(torch.full((32,), value, dtype=torch.float64, device=device), 4)
        except ValueError as exc:
            assert not isinstance(exc, Q.FP16ScaleOverflowError)
        else:
            raise AssertionError("Nonfinite input was accepted")
    # A finite FP64 input can overflow the FP64 span itself; diagnostics stay JSON-safe.
    try:
        Q.PackedAffineGroups.from_tensor(torch.tensor([-1e308, 1e308] + [0.0] * 30,
                                                    dtype=torch.float64, device=device), 4)
    except Q.FP16ScaleOverflowError as exc:
        assert exc.diagnostics["groups"][0]["raw_scale"] == "+Infinity"
        json.dumps(exc.diagnostics, allow_nan=False)
    else:
        raise AssertionError("Finite-input span overflow was accepted")
    result = dict(status="PASS", actual_device=str(device), actual_cuda=torch.device(device).type == "cuda",
                  boundary_cases=cases, accepted_domain_checks=accepted_checks,
                  scalar_and_batched_failure_diagnostics_passed=True,
                  diagnostic_cases=diagnostic_checks, platform_cast_checks=platform_cast_checks,
                  allzero_tiny_packing_integer_ZP_preserved=True,
                  accepted_mapping_AST_proof=proof, no_precision_fallback=True,
                  model_loads=0, AIME_inputs=0, formal_credit=0)
    if progress:
        progress("SCALE_BOUNDARY_TESTS_COMPLETE", cases=len(cases))
    return result


def run_cpu_stress(tokens=STRESS_TOKENS, progress=None):
    """Two finite16k non-AIME small-shape streams, with retention near one."""
    if tokens != STRESS_TOKENS:
        raise ValueError("The planned CPU stress length is exactly16384 per period")
    torch, _ = _production()
    from src.pace_buffer import PACEBuffer
    generator = torch.Generator(device="cpu").manual_seed(20261025)
    # Real production buffer/quantizer, B1/H4/K32/V1, including25% INT8.
    queries = torch.randn((tokens, 1, 2, 32), generator=generator).to(torch.bfloat16)
    keys = torch.randn((tokens, 1, 2, 32), generator=generator).to(torch.bfloat16)
    values = (torch.randn((tokens, 1, 4, 1), generator=generator) * 0.25).to(torch.bfloat16)
    beta = torch.tensor([[0.0, 0.25, 0.5, 1.0]], dtype=torch.bfloat16)
    decay = torch.full((1, 4), -1e-4, dtype=torch.float32)
    prompt = 17
    cases = []
    for period in (1, 8):
        buffer = PACEBuffer(period, [False, False, True, False], key_heads=2)
        buffer.set_prompt_length(prompt)
        started = time.monotonic()
        max_state = 0.0
        max_scale = 0.0
        for token in range(tokens):
            out = buffer.step(queries[token], keys[token], values[token], beta, decay)
            assert out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all().item())
            assert buffer._fp_state is None
            if token % 512 == 0 or token + 1 == tokens:
                scratch = buffer.state_view_for_test()
                assert bool(torch.isfinite(scratch).all().item())
                max_state = max(max_state, float(scratch.abs().max().item()))
                del scratch
                for part in (buffer.packed_state.int4, buffer.packed_state.int8):
                    assert part.scales.dtype == torch.float16
                    assert bool(torch.isfinite(part.scales).all().item())
                    max_scale = max(max_scale, float(part.scales.max().item()))
                if progress:
                    progress("CPU_SCALE_STRESS_PROGRESS", decode_period=period, tokens=token + 1,
                             planned=tokens, model_loads=0)
        report = buffer.finalize("preserve")
        assert report["tokens"] == tokens and report["pending_tokens"] == (tokens - prompt) % period
        assert report["quantization_count"] == 1 + (tokens - prompt) // period
        assert report["dense_state_persistent"] is False and report["fixture_only"] is False
        cases.append(dict(decode_period=period, tokens=tokens, prompt_tokens=prompt,
                          log_decay=-1e-4, decay_near_one=math.exp(-1e-4),
                          packed_INT4_INT8_FP16_scales_integer_ZP=True,
                          sampled_max_abs_recurrent_state=max_state, sampled_max_FP16_scale=max_scale,
                          sampled_statistics_not_global_extrema=True, buffer_report=report,
                          seconds=time.monotonic() - started))
    return dict(status="PASS", actual_device="cpu", actual_cuda=False,
                cases=cases, purpose="Finite long small-shape retention/stored-scale stress; no9B or long-AIME stability claim",
                model_loads=0, AIME_inputs=0, GPU_calls=0, formal_credit=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--old-quantizer")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    def progress(phase, **values):
        print(json.dumps(dict(phase=phase, **values), sort_keys=True), flush=True)
    result = dict(status="PASS", independent_struct_oracle=run_struct_oracle(),
                  accepted_mapping_AST_proof=run_ast_equivalence_proof(args.old_quantizer))
    if not args.oracle_only:
        result["boundary"] = run_boundary_tests(args.device, args.old_quantizer, progress)
        if not args.skip_stress:
            result["cpu_stress"] = run_cpu_stress(progress=progress)
    result["Torch_tests_run"] = not args.oracle_only
    result["GPU_tests_run"] = not args.oracle_only and args.device.startswith("cuda")
    if args.out:
        with args.out.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(status=result["status"], Torch_tests_run=result["Torch_tests_run"],
                          GPU_tests_run=result["GPU_tests_run"]), sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
