
def __mul__(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        val = mpf_mul(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    ttype = type(other)
    if ttype in int_types:
        val = mpf_mul_int(sval, other, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif ttype is float:
        tval = from_float(other)
        val = mpf_mul(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        val = mpc_mul_mpf(tval, sval, prec, rounding); obj = new(mpc); obj._mpc_ = val; return obj
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        val = mpc_mul_mpf(tval, sval, prec, rounding); obj = new(mpc); obj._mpc_ = val; return obj
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.__mul__(other)
