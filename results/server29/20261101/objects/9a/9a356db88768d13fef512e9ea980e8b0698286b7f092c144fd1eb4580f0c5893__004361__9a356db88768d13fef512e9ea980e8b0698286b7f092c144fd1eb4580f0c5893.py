def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,  # batch stride
    stride_x_d,  # dim stride
    stride_y_n,  # batch stride
    stride_y_d,  # dim stride
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    # [BD]
    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if USE_INITIAL_STATE:
        # 2. Shift Cache (Read [1:])
        p_cache_read = cache + i_n * D*W + o_d[:, None] * W + (o_w + 1)[None, :]
        b_cache = tl.load(p_cache_read, mask=m_d[:, None] & ((o_w + 1) < W)[None, :], other=0.0).to(tl.float32)

        # 3. Fill x to the last position
        m_update = o_w == (W - 1)
        b_cache = tl.where(m_update[None, :], b_x[:, None], b_cache)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(y + i_n * stride_y_n + o_d * stride_y_d, tl.cast(b_y,
             dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if USE_INITIAL_STATE:
        p_cache_write = cache + i_n * D*W + o_d[:, None] * W + o_w[None, :]
        tl.store(p_cache_write, tl.cast(b_cache, dtype=cache.dtype.element_ty,
                 fp_downcast_rounding='rtne'), mask=m_d[:, None] & m_w[None, :])
