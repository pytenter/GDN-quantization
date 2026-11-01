
def __mod__(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        val = mpf_mod(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    ttype = type(other)
    if ttype in int_types:
        val = mpf_mod(sval, from_int(other), prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif ttype is float:
        tval = from_float(other)
        val = mpf_mod(sval, tval, prec, rounding); obj = new(mpf); obj._mpf_ = val; return obj
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        raise NotImplementedError("complex modulo")
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        raise NotImplementedError("complex modulo")
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.__mod__(other)
