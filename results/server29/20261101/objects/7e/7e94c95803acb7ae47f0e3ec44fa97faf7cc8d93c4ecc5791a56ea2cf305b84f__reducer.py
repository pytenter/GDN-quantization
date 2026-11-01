"""Prospectively fixed full-vocabulary scoring for the direct 32K pilot.

Primary: CPU FP32 log_softmax/exp/product/torch.sum, with one CPU thread.
Every real score is independently audited by scalar binary64 math.exp/log/fsum.
The resolution is a pre-fixed function of arithmetic audit errors, never Delta.
It concerns reduction of the *same saved logits*, not model-forward uncertainty.
"""
import hashlib
import json
import math
import pathlib
import struct
import time

import torch

SCORE_POSITIONS = tuple(8192 + 64 * k for k in range(384))
CONSUMED_INPUT_TOKENS = 32768
PREDICTION_CONVENTION = 'zero-based t: after consuming ids[t], logits predict ids[t+1]'
WINDOWS = {'Early': (8192, 16384), 'Middle': (16384, 24576), 'Late': (24576, 32768)}
REFERENCE_ASSUMPTIONS = (
    'IEEE-754 binary64 round-to-nearest basic operations; Python libm exp/log '
    'error at most 1 ulp; fsum error covered conservatively by sequential gamma_n; '
    'finite FP32 source logits. This is a conditional conservative numerical '
    'resolution, not a formally verified transcendental interval enclosure.'
)
REDUCTION_POLICY = {
    'primary_device': 'cpu', 'primary_dtype': 'torch.float32', 'cpu_threads': 1,
    'primary_sequence': ['contiguous.flatten FP32', 'torch.log_softmax(a,-1)',
                         'torch.log_softmax(b,-1)', 'exp(logp)',
                         'subtract(logp,logq)', 'multiply', 'torch.sum(dtype=FP32)'],
    'primary_value': 'retain signed raw KL; never clamp negative roundoff to zero',
    'time_mean': 'math.fsum(per-position raw FP32 KL in increasing t)/384',
    'reference': 'independent scalar Python binary64 stable logsumexp, math.exp/log/math.fsum',
    'reference_roundoff_formula': 'gamma(16*V+128)*(1+range(a)+range(b)+sum(abs(weighted_log_ratio))) + exp_underflow_term',
    'reference_roundoff_gamma': 'n*u/(1-n*u), u=2^-53',
    'reference_assumptions': REFERENCE_ASSUMPTIONS,
    'underflow_term': '8*V*smallest_binary64_subnormal*(1+range(a)+range(b))',
    'per_arm_error': 'abs(KL32-KL64_scalar)+binary64_reference_roundoff',
    'paired_resolution': 'mean384(per_arm_error_A+per_arm_error_D)+binary64_time_mean_and_subtraction_roundoff',
    'decision': 'positive if Delta > resolution; negative if Delta < -resolution; otherwise NUMERICALLY_UNRESOLVED',
    'relative_gain': 'Delta/K_A only if K_A > its mean absolute arithmetic error; else null',
    'resolution_scope': 'fixed identical saved logits; does not establish native/H equality or bound forward nondeterminism',
    'positions': list(SCORE_POSITIONS), 'consumed_tokens': CONSUMED_INPUT_TOKENS,
    'position_convention': PREDICTION_CONVENTION,
    'vocab_size': 'inferred from actual model logits; qualification fixtures use 248320 only',
}
REDUCTION_POLICY_SHA256 = hashlib.sha256(json.dumps(REDUCTION_POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError('fixed one-thread CPU reducer contract not active')


def tensor_sha256(x):
    y = x.detach().to(device='cpu').contiguous().reshape(-1)
    return hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()


def _pair(fp_logits, student_logits):
    a = fp_logits.detach().to(device='cpu', dtype=torch.float32).contiguous().flatten()
    b = student_logits.detach().to(device='cpu', dtype=torch.float32).contiguous().flatten()
    if a.numel() < 2 or a.shape != b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError('finite matching full-vocabulary logits required')
    return a, b


def kl_fp32(fp_logits, student_logits):
    configure_cpu()
    a, b = _pair(fp_logits, student_logits)
    lp = torch.log_softmax(a, dim=-1)
    lq = torch.log_softmax(b, dim=-1)
    return float(torch.sum(torch.exp(lp) * (lp - lq), dtype=torch.float32))


def gamma64(n):
    nu = int(n) * 2.0 ** -53
    if not 0 <= nu < 0.01:
        raise ValueError('unsupported binary64 operation-count bound')
    return nu / (1.0 - nu)


def kl_fp64_independent(fp_logits, student_logits):
    """No Torch normalization, exponential, product, or sum in the reference."""
    a, b = _pair(fp_logits, student_logits)
    av, bv = a.tolist(), b.tolist()
    ma, mb = max(av), max(bv)
    ea = [math.exp(x - ma) for x in av]
    eb = [math.exp(x - mb) for x in bv]
    za, zb = math.fsum(ea), math.fsum(eb)
    lza, lzb = math.log(za), math.log(zb)
    # The shifted form avoids subtracting a large common logit offset.
    terms = [(e / za) * (((x - ma) - lza) - ((y - mb) - lzb))
             for e, x, y in zip(ea, av, bv)]
    value = math.fsum(terms)
    absolute_terms = math.fsum(abs(z) for z in terms)
    ra, rb = ma - min(av), mb - min(bv)
    underflow = 8 * len(av) * math.ulp(0.0) * (1.0 + ra + rb)
    bound = gamma64(16 * len(av) + 128) * (1.0 + ra + rb + absolute_terms) + underflow
    if not all(map(math.isfinite, (value, bound, absolute_terms))):
        raise FloatingPointError('nonfinite independent KL audit')
    return {'KL_FP64': value, 'FP64_reference_roundoff': bound,
            'sum_abs_weighted_terms': absolute_terms,
            'reference_dtype': 'binary64', 'reference_algorithm': 'scalar_math_fsum_stable_logsumexp',
            'reference_roundoff_assumptions': REFERENCE_ASSUMPTIONS}


def kl_fp64_torch_crosscheck(fp_logits, student_logits):
    configure_cpu()
    a, b = _pair(fp_logits, student_logits)
    lp, lq = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    return float(torch.sum(torch.exp(lp) * (lp - lq), dtype=torch.float64))


def score_pair(fp_logits, student_logits, token_index, next_token_id, identity):
    """Caller must supply full matching vocab vectors and frozen identity dict.

    identity requires doc_id and input_ids_sha256. The coordinator should also
    include source/model/runtime/GPU/reference-trajectory identities in it.
    """
    t = int(token_index)
    if t not in SCORE_POSITIONS:
        raise ValueError('not a frozen zero-based scoring token')
    if not isinstance(identity, dict) or not identity.get('doc_id') or not identity.get('input_ids_sha256'):
        raise ValueError('frozen document/input identity is required')
    started = time.monotonic()
    a, b = _pair(fp_logits, student_logits)
    p = kl_fp32(a, b)
    audit = kl_fp64_independent(a, b)
    cross = kl_fp64_torch_crosscheck(a, b)
    cross_error = abs(cross - audit['KL_FP64'])
    if cross_error > audit['FP64_reference_roundoff']:
        raise ArithmeticError('independent binary64 crosscheck exceeds fixed reference roundoff model')
    error = abs(p - audit['KL_FP64'])
    if not math.isfinite(p):
        raise FloatingPointError('nonfinite primary KL')
    return dict(identity=identity, token_index=t, consumed_tokens=t + 1,
                predicted_token_index=t + 1, next_token_id=int(next_token_id),
                full_vocabulary_size=a.numel(), KL_FP32=p, **audit,
                FP32_vs_FP64_abs_error=error,
                per_arm_arithmetic_error=error + audit['FP64_reference_roundoff'],
                FP64_torch_crosscheck=cross, FP64_crosscheck_abs_error=cross_error,
                reference_logits_sha256=tensor_sha256(a), student_logits_sha256=tensor_sha256(b),
                reducer_policy_sha256=REDUCTION_POLICY_SHA256,
                scoring_wall_seconds=time.monotonic() - started)


def _validate_rows(rows):
    if len(rows) != 384 or tuple(int(r['token_index']) for r in rows) != SCORE_POSITIONS:
        raise ValueError('a complete increasing 384-position document is mandatory')
    first = rows[0]
    for r in rows:
        if r['identity'] != first['identity'] or r['reducer_policy_sha256'] != REDUCTION_POLICY_SHA256:
            raise ValueError('identity/reducer mismatch within trajectory')
        if r['consumed_tokens'] != r['token_index'] + 1 or r['predicted_token_index'] != r['token_index'] + 1:
            raise ValueError('score cursor/prediction off by one')
        if r['full_vocabulary_size'] != first['full_vocabulary_size']:
            raise ValueError('vocabulary mismatch')
    return rows


def _aggregate(a, d):
    n = len(a)
    ka = math.fsum(r['KL_FP32'] for r in a) / n
    kd = math.fsum(r['KL_FP32'] for r in d) / n
    delta = ka - kd
    ea = math.fsum(r['per_arm_arithmetic_error'] for r in a) / n
    ed = math.fsum(r['per_arm_arithmetic_error'] for r in d) / n
    # Covers fsum final rounding, division, two means, their subtraction, and
    # the rounded summation of audit error magnitudes under the stated model.
    temporal = gamma64(8 * n + 32) * (1.0 + math.fsum(abs(r['KL_FP32']) for r in a + d) / n + ea + ed)
    resolution = ea + ed + temporal
    winner = 'H_DUAL_M' if delta > resolution else 'H_C128' if delta < -resolution else 'NUMERICALLY_UNRESOLVED'
    return {'K_A': ka, 'K_D': kd, 'Delta': delta,
            'RelativeGain': delta / ka if ka > ea + temporal else None,
            'Winner': winner, 'Delta_resolution': resolution,
            'K_A_arithmetic_error': ea, 'K_D_arithmetic_error': ed,
            'time_mean_subtraction_roundoff': temporal, 'score_count': n}


def summarize_document(a_rows, d_rows):
    a, d = _validate_rows(list(a_rows)), _validate_rows(list(d_rows))
    # Codec identity may differ, but all shared input/reference identities must
    # be identical. Store codec in a separate field, outside identity.
    for ra, rd in zip(a, d):
        for key in ('identity', 'token_index', 'consumed_tokens', 'predicted_token_index',
                    'next_token_id', 'full_vocabulary_size', 'reference_logits_sha256', 'reducer_policy_sha256'):
            if ra[key] != rd[key]:
                raise ValueError('unmatched paired score identity: ' + key)
    result = _aggregate(a, d)
    result.update(identity=a[0]['identity'], complete=True,
                  reducer_policy_sha256=REDUCTION_POLICY_SHA256,
                  resolution_scope=REDUCTION_POLICY['resolution_scope'],
                  reference_roundoff_assumptions=REFERENCE_ASSUMPTIONS,
                  windows={name: _aggregate([r for r in a if lo <= r['token_index'] < hi],
                                             [r for r in d if lo <= r['token_index'] < hi])
                           for name, (lo, hi) in WINDOWS.items()})
    return result


def identity_record():
    configure_cpu()
    source = pathlib.Path(__file__)
    return {'policy': REDUCTION_POLICY, 'fixed_reduce_order_sha256': REDUCTION_POLICY_SHA256,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'torch_version': torch.__version__, 'CPU_threads': torch.get_num_threads(),
            'CPU_interop_threads': torch.get_num_interop_threads(),
            'CUDA_initialized': torch.cuda.is_initialized()}
