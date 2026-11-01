
def __add__(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        val = mpf_add(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    ttype = type(other)
    if ttype in int_types:
        val = mpf_add(sval, from_int(other), prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif ttype is float:
        tval = from_float(other)
        val = mpf_add(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        val = mpc_add_mpf(tval, sval, prec, rounding); obj = new(mpc); obj._mpc_ = val; return obj
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        val = mpc_add_mpf(tval, sval, prec, rounding); obj = new(mpc); obj._mpc_ = val; return obj
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.__add__(other)
