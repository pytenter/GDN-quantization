def abs(self, x0):
                        return self._default('abs', (x0, ), {})

def acos(self, x0):
                        return self._default('acos', (x0, ), {})

def acosh(self, x0):
                        return self._default('acosh', (x0, ), {})

def add(self, x0, x1):
                        return self._default('add', (x0, x1, ), {})

def airy_ai(self, x):
                        return self._default('airy_ai', (x, ), {})

def and_(self, x0, x1):
                        return self._default('and_', (x0, x1, ), {})

def asin(self, x0):
                        return self._default('asin', (x0, ), {})

def asinh(self, x0):
                        return self._default('asinh', (x0, ), {})

def atan(self, x0):
                        return self._default('atan', (x0, ), {})

def atan2(self, x0, x1):
                        return self._default('atan2', (x0, x1, ), {})

def atanh(self, x0):
                        return self._default('atanh', (x0, ), {})

def bessel_j0(self, x):
                        return self._default('bessel_j0', (x, ), {})

def bessel_j1(self, x):
                        return self._default('bessel_j1', (x, ), {})

def bessel_y0(self, x):
                        return self._default('bessel_y0', (x, ), {})

def bessel_y1(self, x):
                        return self._default('bessel_y1', (x, ), {})

def bitwise_and(self, x0, x1):
                        return self._default('bitwise_and', (x0, x1, ), {})

def bitwise_left_shift(self, x0, x1):
                        return self._default('bitwise_left_shift', (x0, x1, ), {})

def bitwise_not(self, x0):
                        return self._default('bitwise_not', (x0, ), {})

def bitwise_or(self, x0, x1):
                        return self._default('bitwise_or', (x0, x1, ), {})

def bitwise_right_shift(self, x0, x1):
                        return self._default('bitwise_right_shift', (x0, x1, ), {})

def bitwise_xor(self, x0, x1):
                        return self._default('bitwise_xor', (x0, x1, ), {})

def ceil(self, x0):
                        return self._default('ceil', (x0, ), {})

def ceil_to_int(self, x, dtype):
                        return self._default('ceil_to_int', (x, dtype, ), {})

def chebyshev_polynomial_t(self, x, y):
                        return self._default('chebyshev_polynomial_t', (x, y, ), {})

def chebyshev_polynomial_u(self, x, y):
                        return self._default('chebyshev_polynomial_u', (x, y, ), {})

def chebyshev_polynomial_v(self, x, y):
                        return self._default('chebyshev_polynomial_v', (x, y, ), {})

def chebyshev_polynomial_w(self, x, y):
                        return self._default('chebyshev_polynomial_w', (x, y, ), {})

def check_bounds(self, expr, size, lower, upper):
                        return self._default('check_bounds', (expr, size, lower, upper, ), {})

def constant(self, value, dtype):
                        return self._default('constant', (value, dtype, ), {})

def copysign(self, x0, x1):
                        return self._default('copysign', (x0, x1, ), {})

def cos(self, x0):
                        return self._default('cos', (x0, ), {})

def cosh(self, x0):
                        return self._default('cosh', (x0, ), {})

def digamma(self, x):
                        return self._default('digamma', (x, ), {})

def eq(self, x0, x1):
                        return self._default('eq', (x0, x1, ), {})

def erf(self, x0):
                        return self._default('erf', (x0, ), {})

def erfc(self, x0):
                        return self._default('erfc', (x0, ), {})

def erfcx(self, x):
                        return self._default('erfcx', (x, ), {})

def erfinv(self, x0):
                        return self._default('erfinv', (x0, ), {})

def exp(self, x0):
                        return self._default('exp', (x0, ), {})

def exp2(self, x0):
                        return self._default('exp2', (x0, ), {})

def expm1(self, x0):
                        return self._default('expm1', (x0, ), {})

def floor(self, x0):
                        return self._default('floor', (x0, ), {})

def floor_to_int(self, x, dtype):
                        return self._default('floor_to_int', (x, dtype, ), {})

def floordiv(self, x0, x1):
                        return self._default('floordiv', (x0, x1, ), {})

def fma(self, x, y, z):
                        return self._default('fma', (x, y, z, ), {})

def fmod(self, x0, x1):
                        return self._default('fmod', (x0, x1, ), {})

def frexp(self, x0):
                        return self._default('frexp', (x0, ), {})

def gammainc(self, x, y):
                        return self._default('gammainc', (x, y, ), {})

def gammaincc(self, x, y):
                        return self._default('gammaincc', (x, y, ), {})

def ge(self, x0, x1):
                        return self._default('ge', (x0, x1, ), {})

def gt(self, x0, x1):
                        return self._default('gt', (x0, x1, ), {})

def halide_clamp(self, value, size, check):
                        return self._default('halide_clamp', (value, size, check, ), {})

def hermite_polynomial_h(self, x, y):
                        return self._default('hermite_polynomial_h', (x, y, ), {})

def hermite_polynomial_he(self, x, y):
                        return self._default('hermite_polynomial_he', (x, y, ), {})

def hypot(self, x0, x1):
                        return self._default('hypot', (x0, x1, ), {})

def i0(self, x):
                        return self._default('i0', (x, ), {})

def i0e(self, x):
                        return self._default('i0e', (x, ), {})

def i1(self, x):
                        return self._default('i1', (x, ), {})

def i1e(self, x):
                        return self._default('i1e', (x, ), {})

def identity(self, x):
                        return self._default('identity', (x, ), {})

def igamma(self, x, y):
                        return self._default('igamma', (x, y, ), {})

def igammac(self, x, y):
                        return self._default('igammac', (x, y, ), {})

def index_expr(self, expr, dtype):
                        return self._default('index_expr', (expr, dtype, ), {})

def int_truediv(self, x0, x1):
                        return self._default('int_truediv', (x0, x1, ), {})

def isinf(self, x0):
                        return self._default('isinf', (x0, ), {})

def isnan(self, x0):
                        return self._default('isnan', (x0, ), {})

def laguerre_polynomial_l(self, x, y):
                        return self._default('laguerre_polynomial_l', (x, y, ), {})

def le(self, x0, x1):
                        return self._default('le', (x0, x1, ), {})

def legendre_polynomial_p(self, x, y):
                        return self._default('legendre_polynomial_p', (x, y, ), {})

def lgamma(self, x0):
                        return self._default('lgamma', (x0, ), {})

def libdevice_abs(self, x0):
                        return self._default('libdevice_abs', (x0, ), {})

def libdevice_cos(self, x0):
                        return self._default('libdevice_cos', (x0, ), {})

def libdevice_exp(self, x0):
                        return self._default('libdevice_exp', (x0, ), {})

def libdevice_log(self, x0):
                        return self._default('libdevice_log', (x0, ), {})

def libdevice_sigmoid(self, x0):
                        return self._default('libdevice_sigmoid', (x0, ), {})

def libdevice_sin(self, x0):
                        return self._default('libdevice_sin', (x0, ), {})

def libdevice_sqrt(self, x0):
                        return self._default('libdevice_sqrt', (x0, ), {})

def load(self, name, index):
                        return self._default('load', (name, index, ), {})

def load_seed(self, name, offset):
                        return self._default('load_seed', (name, offset, ), {})

def log(self, x0):
                        return self._default('log', (x0, ), {})

def log10(self, x0):
                        return self._default('log10', (x0, ), {})

def log1p(self, x0):
                        return self._default('log1p', (x0, ), {})

def log2(self, x0):
                        return self._default('log2', (x0, ), {})

def log_ndtr(self, x):
                        return self._default('log_ndtr', (x, ), {})

def logical_and(self, x0, x1):
                        return self._default('logical_and', (x0, x1, ), {})

def logical_not(self, x0):
                        return self._default('logical_not', (x0, ), {})

def logical_or(self, x0, x1):
                        return self._default('logical_or', (x0, x1, ), {})

def logical_xor(self, x0, x1):
                        return self._default('logical_xor', (x0, x1, ), {})

def lshift(self, x0, x1):
                        return self._default('lshift', (x0, x1, ), {})

def lt(self, x0, x1):
                        return self._default('lt', (x0, x1, ), {})

def masked(self, mask, body, other):
                        return self._default('masked', (mask, body, other, ), {})

def maximum(self, x0, x1):
                        return self._default('maximum', (x0, x1, ), {})

def minimum(self, x0, x1):
                        return self._default('minimum', (x0, x1, ), {})

def mod(self, x0, x1):
                        return self._default('mod', (x0, x1, ), {})

def modified_bessel_i0(self, x):
                        return self._default('modified_bessel_i0', (x, ), {})

def modified_bessel_i1(self, x):
                        return self._default('modified_bessel_i1', (x, ), {})

def modified_bessel_k0(self, x):
                        return self._default('modified_bessel_k0', (x, ), {})

def modified_bessel_k1(self, x):
                        return self._default('modified_bessel_k1', (x, ), {})

def mul(self, x0, x1):
                        return self._default('mul', (x0, x1, ), {})

def ndtr(self, x):
                        return self._default('ndtr', (x, ), {})

def ndtri(self, x):
                        return self._default('ndtri', (x, ), {})

def ne(self, x0, x1):
                        return self._default('ne', (x0, x1, ), {})

def neg(self, x0):
                        return self._default('neg', (x0, ), {})

def nextafter(self, x0, x1):
                        return self._default('nextafter', (x0, x1, ), {})

def or_(self, x0, x1):
                        return self._default('or_', (x0, x1, ), {})

def placeholder(self, index):
                        return self._default('placeholder', (index, ), {})

def polygamma(self, x, y):
                        return self._default('polygamma', (x, y, ), {})

def pow(self, x0, x1):
                        return self._default('pow', (x0, x1, ), {})

def rand(self, seed, offset):
                        return self._default('rand', (seed, offset, ), {})

def randint64(self, seed, offset, low, high):
                        return self._default('randint64', (seed, offset, low, high, ), {})

def randn(self, seed, offset):
                        return self._default('randn', (seed, offset, ), {})

def reciprocal(self, x0):
                        return self._default('reciprocal', (x0, ), {})

def reduction(self, dtype, src_dtype, reduction_type, value):
                        return self._default('reduction', (dtype, src_dtype, reduction_type, value, ), {})

def relu(self, x0):
                        return self._default('relu', (x0, ), {})

def remainder(self, x0, x1):
                        return self._default('remainder', (x0, x1, ), {})

def round(self, x0):
                        return self._default('round', (x0, ), {})

def round_to_int(self, x, dtype):
                        return self._default('round_to_int', (x, dtype, ), {})

def rshift(self, x0, x1):
                        return self._default('rshift', (x0, x1, ), {})

def rsqrt(self, x0):
                        return self._default('rsqrt', (x0, ), {})

def scaled_modified_bessel_k0(self, x):
                        return self._default('scaled_modified_bessel_k0', (x, ), {})

def scaled_modified_bessel_k1(self, x):
                        return self._default('scaled_modified_bessel_k1', (x, ), {})

def scan(self, dtypes, combine_fn, values):
                        return self._default('scan', (dtypes, combine_fn, values, ), {})

def shifted_chebyshev_polynomial_t(self, x, y):
                        return self._default('shifted_chebyshev_polynomial_t', (x, y, ), {})

def shifted_chebyshev_polynomial_u(self, x, y):
                        return self._default('shifted_chebyshev_polynomial_u', (x, y, ), {})

def shifted_chebyshev_polynomial_v(self, x, y):
                        return self._default('shifted_chebyshev_polynomial_v', (x, y, ), {})

def shifted_chebyshev_polynomial_w(self, x, y):
                        return self._default('shifted_chebyshev_polynomial_w', (x, y, ), {})

def sigmoid(self, x0):
                        return self._default('sigmoid', (x0, ), {})

def sign(self, x0):
                        return self._default('sign', (x0, ), {})

def signbit(self, x0):
                        return self._default('signbit', (x0, ), {})

def sin(self, x0):
                        return self._default('sin', (x0, ), {})

def sinh(self, x0):
                        return self._default('sinh', (x0, ), {})

def sort(self, dtypes, values, stable, descending):
                        return self._default('sort', (dtypes, values, stable, descending, ), {})

def spherical_bessel_j0(self, x):
                        return self._default('spherical_bessel_j0', (x, ), {})

def sqrt(self, x0):
                        return self._default('sqrt', (x0, ), {})

def square(self, x0):
                        return self._default('square', (x0, ), {})

def store_reduction(self, name, index, value):
                        return self._default('store_reduction', (name, index, value, ), {})

def sub(self, x0, x1):
                        return self._default('sub', (x0, x1, ), {})

def tan(self, x0):
                        return self._default('tan', (x0, ), {})

def tanh(self, x0):
                        return self._default('tanh', (x0, ), {})

def to_dtype_bitcast(self, x, dtype, src_dtype):
                        return self._default('to_dtype_bitcast', (x, dtype, src_dtype, ), {})

def truediv(self, x0, x1):
                        return self._default('truediv', (x0, x1, ), {})

def trunc(self, x0):
                        return self._default('trunc', (x0, ), {})

def trunc_to_int(self, x, dtype):
                        return self._default('trunc_to_int', (x, dtype, ), {})

def truncdiv(self, x0, x1):
                        return self._default('truncdiv', (x0, x1, ), {})

def where(self, condition, input, other):
                        return self._default('where', (condition, input, other, ), {})

def xor(self, x0, x1):
                        return self._default('xor', (x0, x1, ), {})

def zeta(self, x, y):
                        return self._default('zeta', (x, y, ), {})

