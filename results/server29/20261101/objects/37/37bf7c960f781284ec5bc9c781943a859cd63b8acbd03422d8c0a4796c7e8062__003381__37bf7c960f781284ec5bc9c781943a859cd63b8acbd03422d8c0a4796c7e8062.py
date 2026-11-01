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
    l_0_store_output = resolve('store_output')
    pass
    yield '\n'
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'A', 'B', 'A_inverse_scale', 'B_inverse_scale', 'bias_ptr'))
    yield '\n    M = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 0))
    yield '\n    N = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'B', 1))
    yield '\n    K = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 1))
    yield '\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n    stride_am = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 0))
    yield '\n    stride_ak = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 1))
    yield '\n    stride_bk = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 0))
    yield '\n    stride_bn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 1))
    yield '\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    rk = tl.arange(0, BLOCK_K)\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)\n    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k in range(K, 0, -BLOCK_K):\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k, other=0.)\n        if USE_FAST_ACCUM:\n            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)\n        else:\n            acc += tl.dot(a, b, out_dtype=ACC_TYPE)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K * stride_bk\n\n    if SCALING_ROWWISE:\n        inv_a_scale_row = tl.load(A_inverse_scale + rm, mask=rm < M)\n        inv_b_scale_row = tl.load(B_inverse_scale + rn, mask=rn < N)\n        inv_scale_row = inv_a_scale_row[:, None] * inv_b_scale_row[None, :]\n        acc *= inv_scale_row\n    else:\n        # for tensor-wise scaling, the scales are scalars\n        inv_a_scale = tl.load(A_inverse_scale)\n        inv_b_scale = tl.load(B_inverse_scale)\n        inv_scale = inv_a_scale * inv_b_scale\n        acc *= inv_scale\n\n    # rematerialize rm and rn to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n\n    # bias\n    bias = tl.load(bias_ptr + rn, mask=rn < N)\n    acc += bias\n\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_n'), 'acc', 'mask'))

blocks = {}
debug_info = '2=16&3=18&4=20&5=22&9=24&10=26&11=28&12=30&74=32'