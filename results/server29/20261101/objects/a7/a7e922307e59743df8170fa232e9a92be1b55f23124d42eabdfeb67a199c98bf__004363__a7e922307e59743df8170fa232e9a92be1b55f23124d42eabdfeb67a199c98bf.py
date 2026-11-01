def sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    # Pick a default dtype for the reduction if one was not specified.
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)
