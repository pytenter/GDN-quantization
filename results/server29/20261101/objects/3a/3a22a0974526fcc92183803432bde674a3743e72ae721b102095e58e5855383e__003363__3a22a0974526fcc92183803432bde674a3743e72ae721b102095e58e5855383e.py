from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = None

def root(context, missing=missing, environment=environment):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_def_kernel = resolve('def_kernel')
    l_0_size = resolve('size')
    l_0_stride = resolve('stride')
    l_0_GROUPS = resolve('GROUPS')
    l_0_UNROLL = resolve('UNROLL')
    l_0_range = resolve('range')
    l_0_KERNEL_D = resolve('KERNEL_D')
    l_0_store_output = resolve('store_output')
    pass
    yield '\n'
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'X', 'W'))
    yield '\n    # Tensor dimensions\n    BATCH = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'X', 0))
    yield '\n    IN_C = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'X', 1))
    yield '\n    IN_D = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'X', 2))
    yield '\n    IN_H = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'X', 3))
    yield '\n    IN_W = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'X', 4))
    yield '\n    OUT_C = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), None, 1))
    yield '\n    OUT_D = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), None, 2))
    yield '\n    OUT_H = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), None, 3))
    yield '\n    OUT_W = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), None, 4))
    yield '\n\n    # Strides:\n    stride_xn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'X', 0))
    yield '\n    stride_xc = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'X', 1))
    yield '\n    stride_xd = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'X', 2))
    yield '\n    stride_xh = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'X', 3))
    yield '\n    stride_xw = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'X', 4))
    yield '\n    stride_wc_out = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'W', 0))
    yield '\n    stride_wc_in = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'W', 1))
    yield '\n    stride_wd = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'W', 2))
    yield '\n    stride_wh = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'W', 3))
    yield '\n    stride_ww = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'W', 4))
    yield '\n\n    ndhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)\n    idx_y_w = ndhw % OUT_W\n    ndh = ndhw // OUT_W\n    idx_y_h = ndh % OUT_H\n    nd = ndh // OUT_H\n    idx_y_d = nd % OUT_D\n    idx_n = nd // OUT_D\n    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)\n\n'
    if ((undefined(name='GROUPS') if l_0_GROUPS is missing else l_0_GROUPS) == 1):
        pass
        yield '\n    group = 0\n    GROUP_IN_C = IN_C\n    GROUP_OUT_C = OUT_C\n'
    else:
        pass
        yield '\n    group = tl.program_id(2)\n    GROUP_IN_C = IN_C // GROUPS\n    GROUP_OUT_C = OUT_C // GROUPS\n'
    yield '\n\n    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]\n    w_base = (\n        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]\n    )\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n\n'
    if (undefined(name='UNROLL') if l_0_UNROLL is missing else l_0_UNROLL):
        pass
        yield '\n'
        for l_1_d in context.call((undefined(name='range') if l_0_range is missing else l_0_range), (undefined(name='KERNEL_D') if l_0_KERNEL_D is missing else l_0_KERNEL_D)):
            l_1_KERNEL_H = resolve('KERNEL_H')
            _loop_vars = {}
            pass
            yield '\n'
            for l_2_i in context.call((undefined(name='range') if l_0_range is missing else l_0_range), (undefined(name='KERNEL_H') if l_1_KERNEL_H is missing else l_1_KERNEL_H), _loop_vars=_loop_vars):
                l_2_KERNEL_W = resolve('KERNEL_W')
                _loop_vars = {}
                pass
                yield '\n'
                for l_3_j in context.call((undefined(name='range') if l_0_range is missing else l_0_range), (undefined(name='KERNEL_W') if l_2_KERNEL_W is missing else l_2_KERNEL_W), _loop_vars=_loop_vars):
                    _loop_vars = {}
                    pass
                    yield '\n    d = '
                    yield str(l_1_d)
                    yield '\n    i = '
                    yield str(l_2_i)
                    yield '\n    j = '
                    yield str(l_3_j)
                    yield '\n    for k in range(0, GROUP_IN_C, BLOCK_K):\n        \n        idx_x_d = d - PADDING_D + idx_y_d * STRIDE_D\n        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H\n        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W\n        idx_x_c = tl.arange(0, BLOCK_K) + k\n\n        x_ptrs = x_base + (\n            (idx_x_d * stride_xd)[:, None]\n            + (idx_x_h * stride_xh)[:, None]\n            + (idx_x_w * stride_xw)[:, None]\n            + (idx_x_c * stride_xc)[None, :]\n        )\n        mask_x = (\n            (idx_n < BATCH)[:, None]\n            & (idx_x_d >= 0)[:, None]\n            & (idx_x_d < IN_D)[:, None]\n            & (idx_x_h >= 0)[:, None]\n            & (idx_x_h < IN_H)[:, None]\n            & (idx_x_w >= 0)[:, None]\n            & (idx_x_w < IN_W)[:, None]\n            & (idx_x_c < GROUP_IN_C)[None, :]\n        )\n        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)\n\n        w_ptrs = w_base + (\n            (idx_x_c * stride_wc_in)[:, None] +\n            (d * stride_wd) + (i * stride_wh) + (j * stride_ww)\n        )\n        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)\n        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)\n        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)\n\n'
                l_3_j = missing
                yield '\n'
            l_2_i = l_2_KERNEL_W = missing
            yield '\n'
        l_1_d = l_1_KERNEL_H = missing
        yield '\n'
    else:
        pass
        yield '\n    # Could be simplified, but slightly slower:\n    # for d in range(KERNEL_D):\n    #   for i in range(KERNEL_H):\n    #     for j in range(KERNEL_W):\n    #         for k in range(0, GROUP_IN_C, BLOCK_K):\n    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K\n    for dijk in range(KERNEL_D * KERNEL_H * KERNEL_W * BLOCK_K_COUNT):\n        k = (dijk % BLOCK_K_COUNT) * BLOCK_K\n        dij = dijk // BLOCK_K_COUNT\n        j = dij % KERNEL_W\n        di = dij // KERNEL_W\n        i = di % KERNEL_H\n        d = di // KERNEL_H\n        \n        idx_x_d = d - PADDING_D + idx_y_d * STRIDE_D\n        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H\n        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W\n        idx_x_c = tl.arange(0, BLOCK_K) + k\n\n        x_ptrs = x_base + (\n            (idx_x_d * stride_xd)[:, None]\n            + (idx_x_h * stride_xh)[:, None]\n            + (idx_x_w * stride_xw)[:, None]\n            + (idx_x_c * stride_xc)[None, :]\n        )\n        mask_x = (\n            (idx_n < BATCH)[:, None]\n            & (idx_x_d >= 0)[:, None]\n            & (idx_x_d < IN_D)[:, None]\n            & (idx_x_h >= 0)[:, None]\n            & (idx_x_h < IN_H)[:, None]\n            & (idx_x_w >= 0)[:, None]\n            & (idx_x_w < IN_W)[:, None]\n            & (idx_x_c < GROUP_IN_C)[None, :]\n        )\n        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)\n\n        w_ptrs = w_base + (\n            (idx_x_c * stride_wc_in)[:, None] +\n            (d * stride_wd) + (i * stride_wh) + (j * stride_ww)\n        )\n        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)\n        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)\n        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)\n\n'
    yield '\n\n    mask = (\n        (idx_n < BATCH)[:, None]\n        & (idx_y_d < OUT_D)[:, None]\n        & (idx_y_h < OUT_H)[:, None]\n        & (idx_y_w < OUT_W)[:, None]\n        & (idx_y_c < GROUP_OUT_C)[None, :]\n    )\n    idx_n = idx_n[:, None]\n    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C\n    idx_d = idx_y_d[:, None]\n    idx_h = idx_y_h[:, None]\n    idx_w = idx_y_w[:, None]\n\n    # inductor generates a suffix\n    '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_n', 'idx_c', 'idx_d', 'idx_h', 'idx_w'), 'acc', 'mask'))

blocks = {}
debug_info = '2=20&4=22&5=24&6=26&7=28&8=30&9=32&10=34&11=36&12=38&15=40&16=42&17=44&18=46&19=48&20=50&21=52&22=54&23=56&24=58&35=60&52=67&53=70&54=75&55=80&56=84&57=86&58=88&157=100'