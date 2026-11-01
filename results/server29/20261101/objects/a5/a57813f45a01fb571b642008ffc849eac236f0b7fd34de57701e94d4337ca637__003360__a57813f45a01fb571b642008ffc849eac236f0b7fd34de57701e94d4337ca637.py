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
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'A', 'B', 'C', 'D'))
    yield '\n    M = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 0))
    yield '\n    N = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'B', 1))
    yield '\n    K1 = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 1))
    yield '\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n    # K2 = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'C', 1))
    yield '\n    stride_am = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 0))
    yield '\n    stride_ak = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 1))
    yield '\n    stride_bk = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 0))
    yield '\n    stride_bn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 1))
    yield '\n    stride_cm = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'C', 0))
    yield '\n    stride_ck = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'C', 1))
    yield '\n    stride_dk = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'D', 0))
    yield '\n    stride_dn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'D', 1))
    yield '\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n\n    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))\n        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):\n        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    else:\n        ram = rm % M\n\n    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))\n        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):\n        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    else:\n        rbn = rn % N\n\n    rk = tl.arange(0, BLOCK_K)\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)\n    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)\n    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)\n    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k1 in range(K1, 0, -BLOCK_K):\n        # First matmul with A @ B\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k1, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k1, other=0.)\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K * stride_bk\n\n    for k2 in range(K1, 0, -BLOCK_K):\n\n        # Second matmul with C @ D\n        if EVEN_K:\n            c = tl.load(C)\n            d = tl.load(D)\n        else:\n            c = tl.load(C, mask=rk[None, :] < k2, other=0.)\n            d = tl.load(D, mask=rk[:, None] < k2, other=0.)\n        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)\n        C += BLOCK_K * stride_ck\n        D += BLOCK_K * stride_dk\n\n\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_n'), 'acc', 'mask'))

blocks = {}
debug_info = '2=16&3=18&4=20&5=22&9=24&10=26&11=28&12=30&13=32&14=34&15=36&16=38&17=40&84=42'