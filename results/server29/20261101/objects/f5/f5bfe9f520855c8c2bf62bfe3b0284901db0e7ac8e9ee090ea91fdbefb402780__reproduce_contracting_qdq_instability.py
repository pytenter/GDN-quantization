"""Finite, stdlib-only numerical audit; never a production or GPU qualifier.

This oracle follows the declared affine codec and native input precisions, but
does not claim exact Torch/CUDA reduction, exp, or FP64-to-FP16 cast parity.
It demonstrates that the unrounded recurrence's contraction is insufficient
to prove stability after state-dependent affine rounding.
"""
from __future__ import annotations

import json
import math
import random
import struct


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def bf16(value):
    bits = struct.unpack("I", struct.pack("f", fp32(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def fp16(value):
    return struct.unpack("e", struct.pack("e", value))[0]


class ScaleOverflow(Exception):
    def __init__(self, lo, hi, scale):
        self.numeric = dict(lo=lo, hi=hi, raw_scale=scale)
        super().__init__("true positive-finite FP16 scale is unrepresentable")


def affine_g32_int4(values):
    """Scalar independent copy of the declared stored-scale affine convention."""
    assert len(values) == 32 and all(math.isfinite(v) for v in values)
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    raw = (hi - lo) / 15
    # IEEE round-to-nearest overflow midpoint, not the premature 65504 guard.
    if raw >= 65520:
        raise ScaleOverflow(lo, hi, raw)
    scale = fp16(1.0 if raw == 0 else max(raw, 2.0 ** -24))
    assert scale > 0 and math.isfinite(scale)
    zero = min(15, max(0, round(-lo / scale)))
    codes = [min(15, max(0, round(v / scale) + zero)) for v in values]
    decoded = [fp32((code - zero) * scale) for code in codes]
    return decoded, scale, zero


def pairwise_fp32_sum(values):
    assert len(values) and len(values) & (len(values) - 1) == 0
    while len(values) > 1:
        values = [fp32(values[i] + values[i + 1])
                  for i in range(0, len(values), 2)]
    return values[0]


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def run():
    small = [-1.0, 0.01] + [0.0] * 30
    recovered, scale, zero = affine_g32_int4(small)
    assert norm(recovered) > norm(small)
    result = dict(
        status="PASS_COUNTEREXAMPLE_TO_CONTRACTION_IMPLIES_ROUNDED_STABILITY",
        GPU_calls=0, model_loads=0, Torch_imports=0, AIME_inputs=0,
        scope="Independent CPU mathematical oracle; no scientific qualification credit",
        finite_roster=dict(random_seed=171, case_count_limit=10, steps_per_case_limit=2000),
        simple_quantizer_expansion=dict(
            before=small, after=recovered, stored_fp16_scale=scale,
            integer_zero_point=zero, before_l2=norm(small), after_l2=norm(recovered)),
        precision_limits=[
            "The explicit pairwise FP32 sum need not equal the GPU reduction tree.",
            "libm exp followed by FP32 rounding need not equal torch.exp on CUDA.",
            "struct binary16 conversion need not match a Torch FP64 intermediary cast at ties.",
            "Synthetic fixed-key zero-value recurrence is not the actual failing model head.",
            "This is a witness of possible numerical instability, not its frequency or accuracy effect.",
        ],
    )
    rng = random.Random(171)
    beta = bf16(0.6)
    log_decay = fp32(math.log(0.9985))
    decay = fp32(math.exp(log_decay))
    prior_cases = []
    for case in range(10):
        raw_key = [bf16(rng.gauss(0, 1)) for _ in range(32)]
        rounded_key, _, _ = affine_g32_int4(raw_key)
        key = [bf16(v) for v in rounded_key] + [0.0] * 96
        # Native BF16 arithmetic placement: square -> BF16 reduction -> add
        # epsilon -> rsqrt -> multiply. Zero-padding embeds one G32 group in K128.
        denominator = bf16(bf16(sum(bf16(v * v) for v in key)) + 1e-6)
        inverse = bf16(1.0 / math.sqrt(denominator))
        key = [bf16(v * inverse) for v in key]
        initial_input = [fp32(rng.gauss(0, 1)) for _ in range(32)]
        state = affine_g32_int4(initial_input)[0] + [0.0] * 96
        initial_state = list(state)
        traces = {"0": norm(state)}
        key_norm_squared = sum(v * v for v in key)
        # Eq2 with fixed key and v=0 has these eigenvalues: decay on the
        # orthogonal subspace and decay*(1-beta*||k||^2) along the key.
        spectral_norm = max(decay, abs(decay * (1.0 - beta * key_norm_squared)))
        assert spectral_norm < 1.0
        for step in range(1, 2001):
            decayed = [fp32(v * decay) for v in state]
            prediction = pairwise_fp32_sum([fp32(s * k) for s, k in zip(decayed, key)])
            delta = fp32(fp32(-prediction) * beta)
            prequantized = [fp32(s + fp32(k * delta)) for s, k in zip(decayed, key)]
            try:
                state = affine_g32_int4(prequantized[:32])[0] + [0.0] * 96
            except ScaleOverflow as failure:
                result["bounded_contractive_recurrence_witness"] = dict(
                    case=case, failure_step=step, initial_l2=norm(initial_state),
                    pre_overflow_l2=norm(prequantized), trace_l2=traces,
                    failure_numeric=failure.numeric,
                    key_norm=math.sqrt(key_norm_squared),
                    beta_bf16=beta, log_decay_fp32=log_decay, decay_fp32=decay,
                    unrounded_transition_spectral_norm=spectral_norm,
                    raw_key_bf16=raw_key, normalized_quantized_key_bf16=key,
                    initial_state_fp32=initial_state,
                    current_state_fp32_before_overflow=state,
                    value_vector="exact zeros for every step",
                    quantizer_bits=4, group_size=32, key_dim=128,
                    preceding_cases=prior_cases,
                )
                assert case == 9 and step == 913
                assert failure.numeric["raw_scale"] == 65545.9625
                return result
            if step in (8, 32, 128, 256, 512, 1024):
                traces[str(step)] = norm(state)
        prior_cases.append(dict(case=case, completed_steps=2000,
                                initial_l2=norm(initial_state), final_l2=norm(state)))
    raise AssertionError("the fixed finite counterexample was not reproduced")


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2, allow_nan=False))
