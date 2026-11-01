"""Frozen full-vocabulary FP32 KL on CPU with an independent extended reference.

No GPU, model state, clipping, logit truncation or empirical forward-error bound.
The reference-based radius describes scoring arithmetic on identical FP32 logits.
"""
import math
import numpy as np
import torch

torch.set_num_threads(2)

def original_full_kl(torch, reference, student):
    ref, other = reference[:, -1, :].float(), student[:, -1, :].float()
    logp, logq = torch.log_softmax(ref, -1), torch.log_softmax(other, -1)
    p = torch.softmax(ref, -1)
    return float(torch.sum(p * (logp - logq), -1).mean().item())

def score_pair(reference, student):
    a = np.asarray(reference, dtype=np.float32).reshape(-1)
    b = np.asarray(student, dtype=np.float32).reshape(-1)
    assert a.shape == b.shape and a.size > 1
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # Preserve original full_kl operation sequence, FP32 dtype and KL direction.
    # A copy avoids a read-only memory-map tensor alias and never enters a model.
    value = original_full_kl(torch, torch.from_numpy(a.copy()).reshape(1, 1, -1),
                             torch.from_numpy(b.copy()).reshape(1, 1, -1))
    assert math.isfinite(value)
    dtype = np.longdouble
    assert np.finfo(dtype).eps < np.finfo(np.float64).eps, 'EXTENDED_CPU_REFERENCE_UNAVAILABLE'
    x, y = a.astype(dtype), b.astype(dtype)
    x -= x.max(); y -= y.max()
    lp = x - np.log(np.sum(np.exp(x), dtype=dtype))
    lq = y - np.log(np.sum(np.exp(y), dtype=dtype))
    terms = np.exp(lp) * (lp - lq)
    independent = np.sum(terms, dtype=dtype)
    assert np.isfinite(independent)
    eps = np.finfo(dtype).eps
    n_eps = dtype(a.size) * eps
    assert n_eps < dtype('0.001')
    gamma = n_eps / (1 - n_eps)
    # Prospectively fixed extended-reference rounding allowance. This is a
    # scoring-resolution estimate, not certified interval math for libm and
    # never an uncertainty bound for the full-model forward computation.
    allowance = 64 * gamma * (1 + np.sum(np.abs(terms), dtype=dtype)
                                + np.max(np.abs(lp)) + np.max(np.abs(lq)))
    reference_float = float(independent)
    radius = float(abs(dtype(value) - independent) + allowance) + 4 * math.ulp(reference_float)
    assert math.isfinite(radius) and radius >= 0
    return dict(kl_fp32=value, kl_reference_extended=reference_float,
                kl_reference_extended_decimal=np.format_float_scientific(independent, precision=24),
                arithmetic_error_radius=radius,
                reference_rounding_allowance=float(allowance), vocabulary_size=a.size,
                extended_reference_mantissa_bits=int(np.finfo(dtype).nmant),
                negative_fp32_value_preserved=value < 0,
                arithmetic_scope='same_FP32_logits_scoring_only_not_forward_error_bound')

def aggregate(rows):
    assert len(rows) == 256
    value = math.fsum(float(x['kl_fp32']) for x in rows) / 256
    reference = math.fsum(float(x['kl_reference_extended']) for x in rows) / 256
    radius = math.fsum(float(x['arithmetic_error_radius']) for x in rows) / 256
    radius += 4 * math.ulp(value) + 4 * math.ulp(reference)
    return dict(K=value, reference_K=reference, arithmetic_error_radius=radius,
                position_count=256, reducer='math.fsum(FP32_fullV_KL_values)/256',
                original_per_position_KL_unclipped=True,
                arithmetic_scope='same_FP32_logits_scoring_only_not_forward_error_bound')

def resolved_difference(left, right):
    value = left['K'] - right['K']
    radius = left['arithmetic_error_radius'] + right['arithmetic_error_radius'] + 4 * math.ulp(value)
    return dict(value=value, arithmetic_error_radius=radius,
                label='POSITIVE' if value > radius else 'NEGATIVE' if value < -radius else 'NUMERICALLY_UNRESOLVED')

def cpu_tests():
    rng = np.random.default_rng(0)
    cases = []
    for name, n, scale, shift in [('fullV_random', 157184, 3, .03),
                                  ('fullV_near_equal', 157184, 1, 1e-7),
                                  ('small_extreme', 31, 1000, .1),
                                  ('small_uniform', 17, 0, 0)]:
        a = (rng.normal(size=n) * scale).astype(np.float32)
        b = (a + rng.normal(size=n) * shift).astype(np.float32)
        result = score_pair(a, b)
        assert abs(result['kl_fp32'] - result['kl_reference_extended']) <= result['arithmetic_error_radius']
        self_result = score_pair(a, a)
        assert self_result['kl_fp32'] == self_result['kl_reference_extended'] == 0
        cases.append(dict(name=name, status='PASS', result=result, self_KL_exact_zero=True))
    rows = [dict(kl_fp32=1.0 / (i + 1), kl_reference_extended=1.0 / (i + 1),
                 arithmetic_error_radius=1e-10) for i in range(256)]
    out = aggregate(rows)
    assert out['K'] == math.fsum(x['kl_fp32'] for x in rows) / 256
    assert resolved_difference(out, out)['label'] == 'NUMERICALLY_UNRESOLVED'
    return dict(status='PASS_CPU_SCORING_ONLY', cases=cases, reducer_and_resolution='PASS',
                GPU_initialized=torch.cuda.is_initialized(),
                original_FP32_full_kl_operation_sequence=True, no_fixed_Qwen_or_old_Ling_resolution=True)

if __name__ == '__main__':
    import json
    print(json.dumps(cpu_tests(), ensure_ascii=False, allow_nan=False))
