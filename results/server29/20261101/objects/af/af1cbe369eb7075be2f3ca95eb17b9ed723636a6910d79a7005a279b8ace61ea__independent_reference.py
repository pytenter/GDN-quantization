"""Independent ASYM32_CANONICAL_V2 oracle: Fraction and integer IEEE bits only.

No implementation module, tensor library, model or device API is imported.
Each FP32 node rounds its exact rational result once, ties to even; binary16
storage is then rounded independently from that already-rounded FP32 value.
Inputs/outputs are integer bit patterns, not host floating-point expressions.
"""
from fractions import Fraction


class CanonicalArithmeticError(ValueError):
    pass


def round_integer_ties_even(value):
    """Exact rational to mathematical integer, preserving no integer zero sign."""
    if not isinstance(value, (int, Fraction)):
        raise TypeError("Use an integer or exact Fraction, never host float")
    value = Fraction(value)
    sign = -1 if value < 0 else 1
    n, d = abs(value.numerator), value.denominator
    whole, remainder = divmod(n, d)
    twice = remainder * 2
    if twice > d or (twice == d and whole & 1):
        whole += 1
    return sign * whole


def bits_to_fraction(bits, exp_bits=8, frac_bits=23):
    """Decode a finite IEEE binary value exactly; signed zeros both map to 0."""
    width = 1 + exp_bits + frac_bits
    if type(bits) is not int or not 0 <= bits < 1 << width:
        raise CanonicalArithmeticError("INVALID_IEEE_BIT_PATTERN")
    fraction_mask = (1 << frac_bits) - 1
    exponent_mask = (1 << exp_bits) - 1
    exponent = (bits >> frac_bits) & exponent_mask
    fraction = bits & fraction_mask
    if exponent == exponent_mask:
        raise CanonicalArithmeticError("NONFINITE_IEEE_VALUE")
    bias = (1 << (exp_bits - 1)) - 1
    significand = fraction if exponent == 0 else (1 << frac_bits) + fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - frac_bits
    value = (Fraction(significand << power) if power >= 0
             else Fraction(significand, 1 << -power))
    return -value if bits >> (width - 1) else value


def _scaled(value, power):
    return value * (1 << power) if power >= 0 else value / (1 << -power)


def round_fraction(value, exp_bits=8, frac_bits=23, zero_sign=0):
    """Exact rational -> IEEE bits with RN-even, gradual underflow and overflow.

    Overflow returns IEEE infinity bits. Canonical codec nodes reject these
    immediately. zero_sign applies only when the exact value is zero; a tiny
    nonzero value rounded to zero retains that nonzero value's sign.
    """
    if not isinstance(value, (int, Fraction)):
        raise TypeError("Use an integer or exact Fraction, never host float")
    value = Fraction(value)
    sign = int(value < 0) if value else int(bool(zero_sign))
    sign_bits = sign << (exp_bits + frac_bits)
    value = abs(value)
    if not value:
        return sign_bits
    bias = (1 << (exp_bits - 1)) - 1
    emin, emax = 1 - bias, bias
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    below_power = n < (d << exponent) if exponent >= 0 else (n << -exponent) < d
    if below_power:
        exponent -= 1
    if exponent < emin:
        mantissa = round_integer_ties_even(_scaled(value, frac_bits - emin))
        # A rounded largest subnormal can become the smallest normal.
        return sign_bits | mantissa
    mantissa = round_integer_ties_even(_scaled(value, frac_bits - exponent))
    if mantissa == 1 << (frac_bits + 1):
        mantissa >>= 1
        exponent += 1
    if exponent > emax:
        return sign_bits | (((1 << exp_bits) - 1) << frac_bits)
    return sign_bits | ((exponent + bias) << frac_bits) | (mantissa - (1 << frac_bits))


def rn32_add(a, b):
    av, bv = bits_to_fraction(a), bits_to_fraction(b)
    return round_fraction(av + bv, zero_sign=bool(a >> 31 and b >> 31))


def rn32_sub(a, b):
    return rn32_add(a, b ^ 0x80000000)


def rn32_mul(a, b):
    return round_fraction(bits_to_fraction(a) * bits_to_fraction(b),
                          zero_sign=(a ^ b) >> 31)


def rn32_div(a, b):
    denominator = bits_to_fraction(b)
    if not denominator:
        raise CanonicalArithmeticError("DIVISION_BY_ZERO")
    return round_fraction(bits_to_fraction(a) / denominator,
                          zero_sign=(a ^ b) >> 31)


def fp32_to_fp16(bits):
    return round_fraction(bits_to_fraction(bits), exp_bits=5, frac_bits=10,
                          zero_sign=bits >> 31)


def fp16_to_fp32(bits):
    return round_fraction(bits_to_fraction(bits, exp_bits=5, frac_bits=10),
                          zero_sign=bits >> 15)


def is_finite32(bits):
    return type(bits) is int and 0 <= bits < 1 << 32 and bits & 0x7F800000 != 0x7F800000


def _finite_node(bits, stage):
    if not is_finite32(bits):
        raise CanonicalArithmeticError("NONFINITE_CANONICAL_NODE:" + stage)
    return bits


ZERO32 = 0
EPS32 = round_fraction(Fraction(1, 10**12))
LOW32 = round_fraction(Fraction(1, 1 << 24))
HIGH32 = round_fraction(Fraction(65504))
DIVISOR32 = round_fraction(Fraction(255))


def canonical_group(input_bits32):
    """Canonical stages for exactly one logical contiguous Value group of 32.

    S0 preserves the caller's signed-zero bits. S2 includes +0 and canonicalizes
    min/max zero to +0. Other arithmetic follows IEEE RN signed-zero behavior.
    q's integer addition is exact and saturating; it cannot overflow int32.
    H transforms and native-cache writeback lie outside this scalar oracle.
    """
    values = list(input_bits32)
    if len(values) != 32 or not all(is_finite32(x) for x in values):
        raise CanonicalArithmeticError("REQUIRE_32_FINITE_BINARY32_INPUT_BITS")
    exact = [bits_to_fraction(x) for x in values]
    lower = round_fraction(min([Fraction(0)] + exact))
    upper = round_fraction(max([Fraction(0)] + exact))
    span = _finite_node(rn32_sub(upper, lower), "S3_span")
    fitted = EPS32 if bits_to_fraction(span) < bits_to_fraction(EPS32) else span
    raw = _finite_node(rn32_div(fitted, DIVISOR32), "S4_raw_scale")
    clamped = LOW32 if bits_to_fraction(raw) < bits_to_fraction(LOW32) else raw
    if bits_to_fraction(clamped) > bits_to_fraction(HIGH32):
        clamped = HIGH32
    stored = fp32_to_fp16(clamped)
    scale = _finite_node(fp16_to_fp32(stored), "S5_scale")
    ratio = _finite_node(rn32_div(lower ^ 0x80000000, scale), "S6_ratio")
    z = min(255, max(0, round_integer_ties_even(bits_to_fraction(ratio))))
    levels = [_finite_node(rn32_div(x, scale), "S7_levels") for x in values]
    integers = [round_integer_ties_even(bits_to_fraction(x)) for x in levels]
    rounded = [round_fraction(Fraction(n), zero_sign=x >> 31)
               for x, n in zip(levels, integers)]
    codes = [min(255, max(0, n + z)) for n in integers]
    centered_bits = [round_fraction(Fraction(q - z)) for q in codes]
    decoded = [_finite_node(rn32_mul(x, scale), "S9_decode") for x in centered_bits]
    return dict(S0_bits32=values, S1_bits32=values.copy(), S2_min_bits32=lower,
                S2_max_bits32=upper, S3_span_bits32=span, S3_fit_bits32=fitted,
                S4_raw_bits32=raw, S5_stored_bits16=stored, S5_scale_bits32=scale,
                S6_ratio_bits32=ratio, S6_z=z, S7_levels_bits32=levels,
                S7_round_bits32=rounded, S8_q=codes, S9_decode_bits32=decoded)


def require_exact_stages(actual, expected):
    """Reject any missing/changed canonical field; paths/timing are not stages."""
    if actual.keys() != expected.keys():
        raise AssertionError("CANONICAL_STAGE_KEY_MISMATCH")
    for stage in expected:
        if actual[stage] != expected[stage]:
            raise AssertionError("CANONICAL_STAGE_MISMATCH:" + stage)
    return True
