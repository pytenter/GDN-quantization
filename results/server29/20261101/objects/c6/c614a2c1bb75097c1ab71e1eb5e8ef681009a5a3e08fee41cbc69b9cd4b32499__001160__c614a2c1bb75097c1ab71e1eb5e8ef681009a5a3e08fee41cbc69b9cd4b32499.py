
def __eq__(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        return mpf_eq(sval, tval)
    ttype = type(other)
    if ttype in int_types:
        return mpf_eq(sval, from_int(other))
    elif ttype is float:
        tval = from_float(other)
        return mpf_eq(sval, tval)
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        return (tval[1] == fzero) and mpf_eq(tval[0], sval)
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        return (tval[1] == fzero) and mpf_eq(tval[0], sval)
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.__eq__(other)
