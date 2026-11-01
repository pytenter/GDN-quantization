"""Finite CPU/CUDA engineering gates; never formal AIME configurations.

CPU scalar codecs use Python arithmetic and IEEE binary16 via struct, not the
production tensor encoder. The recurrent oracle consumes tokens sequentially;
PACE reconstructs each prefix from packed base and the bounded pending ring.
An actual native-source callback is mandatory for CUDA Gate B qualification.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import struct
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pace_quantizer import (PackedAffineGroups, PackedAffineState,
                                pack_nibbles, unpack_nibbles)
from src.pace_buffer import PACEBuffer


TOLERANCES = {"state_atol": 1e-5, "state_rtol": 1e-5,
              "native_bf16_output_exact": True,
              "scalar_affine_codes_zero_points_scales_exact": True,
              "quantized_buffer_bf16_output_exact": True}


def _scalar_group(values, bits):
    """Independent scalar affine convention, including stored binary16 scale."""
    if len(values) != 32 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid scalar reference group")
    maximum = (1 << bits) - 1
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    raw = (hi - lo) / maximum
    if raw > 65504.0:
        raise ValueError("unrepresentable scalar scale")
    scale = struct.unpack("e", struct.pack("e", 1.0 if raw == 0 else max(raw, 2.0 ** -24)))[0]
    zero = min(max(round(-lo / scale), 0), maximum)
    codes = [min(max(round(x / scale) + zero, 0), maximum) for x in values]
    return codes, scale, zero, [(c - zero) * scale for c in codes]


def _scalar_state(state, mask):
    """Loop B/H/V/key groups independently of production tensor reshaping."""
    original = state.detach().to("cpu", torch.float64)
    b, h, k, v = original.shape
    output = torch.empty_like(original)
    for bi in range(b):
        for hi in range(h):
            for vi in range(v):
                for start in range(0, k, 32):
                    values = [float(original[bi, hi, ki, vi]) for ki in range(start, start + 32)]
                    _, _, _, decoded = _scalar_group(values, 8 if mask[hi] else 4)
                    output[bi, hi, start:start + 32, vi] = torch.tensor(decoded, dtype=torch.float64)
    return output.to(device=state.device, dtype=torch.float32)


def _scalar_key(key):
    original = key.detach().to("cpu", torch.float64)
    out = torch.empty_like(original)
    for b in range(key.shape[0]):
        for h in range(key.shape[1]):
            for start in range(0, key.shape[-1], 32):
                values = original[b, h, start:start + 32].tolist()
                out[b, h, start:start + 32] = torch.tensor(_scalar_group(values, 4)[3], dtype=torch.float64)
    return out.to(device=key.device, dtype=torch.bfloat16)


def _direct_reference(q, k, v, beta, g, state):
    """Separate sequential native-order oracle for CPU engineering fallback."""
    q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q = q.float() * (1.0 / math.sqrt(q.shape[-1]))
    k = k.float()
    decayed = state * torch.exp(g.float())[..., None, None]
    prediction = torch.sum(decayed * k[..., None], dim=-2)
    delta = beta.float()[..., None] * (v.float() - prediction)
    updated = decayed + torch.einsum("bhk,bhv->bhkv", k, delta)
    out = torch.sum(updated * q[..., None], dim=-2).to(torch.bfloat16)
    return out, updated


def load_native_reference_from_source(path, expected_sha256=None):
    """Read-only AST extraction executes only the two original Torch helpers."""
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("native source SHA256 mismatch")
    tree = ast.parse(source)
    names = {"l2norm", "torch_recurrent_gated_delta_rule"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError("native source lacks required helpers")
    for node in selected:
        node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    native = namespace["torch_recurrent_gated_delta_rule"]

    def reference(q, k, v, beta, g, state):
        out, final = native(q[:, None], k[:, None], v[:, None], g[:, None], beta[:, None],
                            state, True, use_qk_l2norm_in_kernel=True)
        return out[:, 0], final
    reference.source_sha256 = digest
    reference.source_path = str(path)
    return reference


def _tokens(count, *, b=1, h=4, u=2, k=32, v=3, seed=817, device="cpu"):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    data = []
    for i in range(count):
        q = torch.randn(b, u, k, generator=rng).to(torch.bfloat16).repeat_interleave(h // u, dim=1)
        key = torch.randn(b, u, k, generator=rng).to(torch.bfloat16).repeat_interleave(h // u, dim=1)
        value = torch.randn(b, h, v, generator=rng).to(torch.bfloat16)
        beta = torch.rand(b, h, generator=rng).to(torch.bfloat16)
        g = -torch.rand(b, h, generator=rng) * 0.7
        if i % 7 == 0:
            beta[:, 0] = 0
            beta[:, -1] = 1
            g[:, 0] = 0
            g[:, -1] = -80  # Strong decay, not merely average random gates.
        data.append(tuple(x.to(device) for x in (q, key, value, beta, g)))
    return data


def _state_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=TOLERANCES["state_atol"],
                               rtol=TOLERANCES["state_rtol"])


def _all_owned_tensors(obj, seen=None):
    """Inspect actual object graph, not just the codec's claimed inventory."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple)):
        values = obj
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        values = vars(obj).values()
    else:
        return []
    return [tensor for value in values for tensor in _all_owned_tensors(value, seen)]


def gate_a(device="cpu", progress=None):
    completed = []
    q = torch.arange(16, dtype=torch.uint8, device=device).repeat(2)
    assert torch.equal(unpack_nibbles(pack_nibbles(q), q.numel()), q)
    odd = q[:17]
    assert torch.equal(unpack_nibbles(pack_nibbles(odd), 17), odd)
    assert pack_nibbles(torch.tensor([1, 2, 15], dtype=torch.uint8, device=device)).cpu().tolist() == [33, 15]
    completed.append("physical_nibble_order_odd_padding_and_all_codes")

    fixtures = [[0.] * 32, [3.25] * 32, [-4.5] * 32,
                [-7., 8., -2.5, -1.5, -0.5, 0., .5, 1.5, 2.5] + [0.] * 23,
                [1e-30, -1e-30] * 16, [0., 65504. * 15.] + [1.] * 30,
                [0., 1., .49995, .499995] + [0.] * 28]
    for bits in (4, 8):
        x = torch.tensor(fixtures, dtype=torch.float64, device=device)
        encoded = PackedAffineGroups.from_tensor(x, bits)
        for index, fixture in enumerate(fixtures):
            codes, scale, zp, decoded = _scalar_group(fixture, bits)
            assert encoded.integer_codes()[index].cpu().tolist() == codes
            assert float(encoded.scales[index].item()) == scale
            assert int(encoded.integer_zero_points()[index].item()) == zp
            assert encoded.dequantize(torch.float64)[index].cpu().tolist() == decoded
        assert encoded.codes.dtype == torch.uint8 and encoded.zero_points.dtype == torch.uint8
        assert encoded.scales.dtype == torch.float16
        assert bool(torch.isfinite(encoded.dequantize()).all())
    completed.append("independent_scalar_int4_int8_zeros_constants_ties_saturation_underflow_maxscale")

    rng = torch.Generator().manual_seed(37)
    state = torch.randn(2, 4, 64, 3, generator=rng).to(device)
    state[..., 0] *= .01
    state[..., 1] += 9
    state[..., 2] *= 100
    mask = [False, True, False, False]
    encoded = PackedAffineState.from_tensor(state, mask)
    assert torch.equal(encoded.dequantize(), _scalar_state(state, mask))
    modified = state.clone()
    modified[0] *= 1000
    assert torch.equal(PackedAffineState.from_tensor(modified, mask).dequantize()[1], encoded.dequantize()[1])
    # Force a noncontiguous K/V layout without changing the logical matrix.
    noncontiguous = state.transpose(2, 3).contiguous().transpose(2, 3)
    assert not noncontiguous.is_contiguous()
    assert torch.equal(PackedAffineState.from_tensor(noncontiguous, mask).dequantize(), encoded.dequantize())
    completed.append("fixed_value_column_key_groups_noncontiguous_batch_isolation")

    full = PackedAffineState.from_tensor(torch.zeros(1, 32, 128, 128, device=device), [False] * 24 + [True] * 8)
    assert full.storage_nbytes() == 370688
    assert full.storage_nbytes() * 8 / (32 * 128 * 128) == 5.65625
    assert torch.count_nonzero(full.dequantize()).item() == 0
    completed.append("32_head_actual_packed_storage_payload_and_metadata_accounting")
    for bad in (float("nan"), float("inf"), 1e20):
        values = torch.full((1, 32), bad, dtype=torch.float64, device=device)
        try:
            PackedAffineGroups.from_tensor(values, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite/unrepresentable scale was accepted")
    completed.append("fail_closed_nonfinite_and_unrepresentable_fp16_scale")
    return completed


def gate_b(device="cpu", native_reference=None, progress=None):
    oracle = native_reference or _direct_reference
    if str(device).startswith("cuda") and native_reference is None:
        raise RuntimeError("actual native helper callback is mandatory for CUDA Gate B")
    completed = []
    # Toy shapes cover each token and boundary; actual model shapes cover native
    # head sharing, BF16 normalization/reduction and FP32 recurrence precision.
    for period in (1, 8):
        for prompt, count, h, u, k, v in ((0, 19, 4, 2, 32, 3), (7, 24, 4, 2, 32, 3),
                                         (0, 9, 32, 16, 128, 128)):
            mask = [i >= 3 * h // 4 for i in range(h)]
            buffer = PACEBuffer(period, mask, key_heads=u, state_quantization=False,
                                key_quantization=False, _validation_only=True)
            buffer.set_prompt_length(prompt)
            state = torch.zeros(1, h, k, v, dtype=torch.float32, device=device)
            for values in _tokens(count, h=h, u=u, k=k, v=v, device=device):
                expected, state = oracle(*values, state)
                actual = buffer.step(*values)
                assert torch.equal(actual, expected), "native BF16 output mismatch"
                _state_close(buffer.state_view_for_test(), state)
            completed.append(f"sequential_native_parity_dp{period}_prompt{prompt}_shape{h}x{k}x{v}")
            if progress:
                progress(completed[-1])
    # Independent Eq2 matrix expression catches the decay/delta coefficient
    # placement; it does not reuse the implementation's delta arithmetic.
    buffer = PACEBuffer(8, [False, False, False, True], key_heads=2,
                        state_quantization=False, key_quantization=False, _validation_only=True)
    buffer.set_prompt_length(0)
    matrix_state = torch.zeros(1, 4, 32, 3, dtype=torch.float64, device=device)
    eye = torch.eye(32, dtype=torch.float64, device=device)[None, None]
    for q, key, value, beta, g in _tokens(17, device=device, seed=18):
        normalized = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        nk = normalized.double()
        transition = torch.exp(g.double())[..., None, None] * (
            eye - beta.double()[..., None, None] * (nk[..., :, None] @ nk[..., None, :]))
        matrix_state = transition @ matrix_state + beta.double()[..., None, None] * (
            nk[..., :, None] @ value.double()[..., None, :])
        buffer.step(q, key, value, beta, g)
        _state_close(buffer.state_view_for_test().double(), matrix_state)
    completed.append("independent_double_matrix_eq2_vs_replayed_delta_rule")
    return completed


def gate_c(device="cpu", progress=None):
    completed = []
    mask = [False, False, False, True]
    for period in (1, 8):
        for prompt, count in ((0, 19), (7, 24), (255, 257), (256, 257), (257, 266)):
            buffer = PACEBuffer(period, mask, key_heads=2)
            buffer.set_prompt_length(prompt)
            state = torch.zeros(1, 4, 32, 3, dtype=torch.float32, device=device)
            for index, values in enumerate(_tokens(count, device=device, seed=83)):
                q, key, value, beta, g = values
                recovered = key if index < prompt else _scalar_key(key[:, ::2]).repeat_interleave(2, dim=1)
                expected, state = _direct_reference(q, recovered, value, beta, g, state)
                actual = buffer.step(*values)
                assert torch.equal(actual, expected), "packed-buffer output differs from scalar-codec golden"
                forwarded = index + 1
                boundary = (forwarded <= prompt and (forwarded % 256 == 0 or forwarded == prompt)) or (
                    forwarded > prompt and (forwarded - prompt) % period == 0)
                if boundary:
                    state = _scalar_state(state, mask)
                _state_close(buffer.state_view_for_test(), state)
                assert buffer.pending_tokens < (256 if forwarded < prompt else period) or forwarded == 0
                assert not any(t.dtype == torch.float32 and tuple(t.shape) == (1, 4, 32, 3)
                               for t in _all_owned_tensors(buffer)), "production retains dense FP32 state"
                if progress and forwarded % 128 == 0:
                    progress(f"packed golden dp{period} prefill{prompt}: token{forwarded}/{count}")
            expected_commits = (math.ceil(prompt / 256) if prompt else 0) + (count - prompt) // period
            assert buffer.quantization_count == expected_commits
            assert buffer.pending_tokens == (count - prompt) % period
            before = buffer.state_view_for_test()
            before_tokens, before_count = buffer.tokens, buffer.quantization_count
            buffer.finalize("preserve")
            buffer.finalize("preserve")
            assert torch.equal(buffer.state_view_for_test(), before)
            assert buffer.tokens == before_tokens and buffer.quantization_count == before_count
            pending = buffer.pending_tokens
            buffer.finalize("flush")
            buffer.finalize("flush")
            assert buffer.tokens == before_tokens
            assert buffer.quantization_count == before_count + bool(pending)
            assert buffer.pending_tokens == 0
            report = buffer.storage_report()
            assert report["beta_storage_included"] and not report["dense_state_persistent"]
            assert report["total_resident_tensor_nbytes"] == sum(
                x.numel() * x.element_size() for x in buffer.resident_tensors().values())
            completed.append(f"packed_golden_dp{period}_prefill{prompt}_tokens{count}_tail_once")
            if progress:
                progress(completed[-1])
    # Physical state/key grouping must isolate concurrent sequence batches, not
    # merely pass a single-sequence recurrence fixture.
    batched = PACEBuffer(8, mask, key_heads=2)
    batched.set_prompt_length(0)
    independent = [PACEBuffer(8, mask, key_heads=2) for _ in range(2)]
    for instance in independent:
        instance.set_prompt_length(0)
    for values in _tokens(11, b=2, device=device, seed=418):
        actual = batched.step(*values)
        expected = torch.cat([instance.step(*(x[i:i + 1] for x in values))
                              for i, instance in enumerate(independent)], dim=0)
        assert torch.equal(actual, expected)
        assert torch.equal(batched.state_view_for_test(), torch.cat(
            [instance.state_view_for_test() for instance in independent], dim=0))
    completed.append("batch_two_matches_two_independent_packed_sequences")
    # A preserved partial ring is the complete resumable state. Restoring that
    # object must apply each pending update once, even after preserve-finalize.
    batched.finalize("preserve")
    restored = copy.deepcopy(batched)
    for values in _tokens(9, b=2, device=device, seed=419):
        assert torch.equal(batched.step(*values), restored.step(*values))
        assert torch.equal(batched.state_view_for_test(), restored.state_view_for_test())
    completed.append("preserved_packed_base_plus_pending_ring_restart_continuation")
    # A flush-token output must precede state QDQ. Choose a forced small
    # representational range and verify it is distinguishable from rounded read.
    buffer = PACEBuffer(1, mask, key_heads=2)
    buffer.set_prompt_length(0)
    values = _tokens(1, seed=999, device=device)[0]
    out = buffer.step(*values)
    q = values[0]
    nq = (q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)).float() / math.sqrt(32)
    rounded_read = (buffer.packed_state.dequantize() * nq[..., None]).sum(-2).to(torch.bfloat16)
    assert not torch.equal(out, rounded_read), "fixture did not expose output/writeback order"
    completed.append("flush_output_before_state_quantization_is_observable")
    # Distinct paired keys must be rejected, rather than silently losing heads.
    corrupted = list(_tokens(1, device=device)[0])
    corrupted[1] = corrupted[1].clone()
    corrupted[1][:, 1] += 1
    invalid = PACEBuffer(8, mask, key_heads=2)
    invalid.set_prompt_length(0)
    try:
        invalid.step(*corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid repeated-key mapping accepted")
    completed.append("repeated_key_mapping_validation")
    return completed


def _run_gates(report_dir, device, progress, native_reference):
    reports = {}
    actual_cuda = str(device).startswith("cuda")
    source_identity = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ("src/pace_quantizer.py", "src/pace_buffer.py", "tests/test_pace_math.py")}
    process_identity = {"pid": os.getpid(), "executable": sys.executable,
                        "platform": platform.platform(),
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    definitions = (("A", "quantizer_tests.json", lambda: gate_a(device, progress)),
                   ("B", "fp_buffer_parity.json", lambda: gate_b(device, native_reference, progress)),
                   ("C", "quantized_buffer_tests.json", lambda: gate_c(device, progress)))
    failed = False
    for gate, filename, function in definitions:
        if progress:
            progress(f"PACE gate {gate} start on {device}")
        start = time.monotonic()
        report = {"gate": gate, "actual_cuda": actual_cuda, "device": str(device),
                  "scope": "engineering_gate_only_not_AIME", "tolerances": TOLERANCES,
                  "source_sha256": source_identity, "process": process_identity,
                  "native_source_sha256": getattr(native_reference, "source_sha256", None),
                  "native_callback_provided": native_reference is not None}
        if failed:
            report.update(status="NOT_RUN", reason="preceding numerical gate failed")
        else:
            try:
                cases = function()
                report.update(status="PASS", tests=len(cases), cases=cases)
            except Exception as exc:
                failed = True
                report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        report["elapsed_seconds"] = time.monotonic() - start
        reports[gate] = report
        if report_dir is not None:
            destination = Path(report_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / filename).write_text(json.dumps(report, indent=2) + "\n")
        if progress:
            progress(f"PACE gate {gate}: {report['status']}")
    return {"status": "PASS" if not failed else "FAIL", "actual_cuda": actual_cuda,
            "GPU_qualification_credit": actual_cuda and not failed,
            "torch_version": torch.__version__, "source_sha256": source_identity,
            "process": process_identity, "AIME_samples": 0, "model_loads": 0,
            "gates": reports}


def run_cpu_gates(report_dir=None, progress=None, native_reference=None):
    return _run_gates(report_dir, "cpu", progress, native_reference)


def run_device_gates(report_dir, device="cuda:0", progress=None, native_reference=None):
    if not str(device).startswith("cuda"):
        raise ValueError("run_device_gates requires an explicitly selected CUDA device")
    return _run_gates(report_dir, device, progress, native_reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-source")
    parser.add_argument("--native-source-sha256")
    args = parser.parse_args()
    reference = (load_native_reference_from_source(args.native_source, args.native_source_sha256)
                 if args.native_source else None)
    result = _run_gates(args.report_dir, args.device, lambda message: print(message, flush=True), reference)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
