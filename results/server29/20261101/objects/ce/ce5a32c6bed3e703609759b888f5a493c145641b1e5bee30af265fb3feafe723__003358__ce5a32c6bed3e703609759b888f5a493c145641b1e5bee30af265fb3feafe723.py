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
    l_0_EVEN_K = resolve('EVEN_K')
    l_0_load_input = resolve('load_input')
    l_0_store_output = resolve('store_output')
    pass
    yield '\n'
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'A', 'B'))
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
    yield '\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and M >= BLOCK_M:\n        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    else:\n        offs_a_m = rm % M\n    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and N >= BLOCK_N:\n        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    else:\n        offs_b_n = rn % N\n    offs_k = tl.arange(0, BLOCK_K)\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n\n    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):\n        '
    if (not (undefined(name='EVEN_K') if l_0_EVEN_K is missing else l_0_EVEN_K)):
        pass
        yield '\n        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)\n        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)\n        '
    yield '\n        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)\n        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)\n\n        idx_m = offs_a_m[:, None]\n        idx_n = a_k_idx_vals\n        '
    yield str(context.call((undefined(name='load_input') if l_0_load_input is missing else l_0_load_input), 'A', 'a', ('idx_m', 'idx_n'), mask=(None if (undefined(name='EVEN_K') if l_0_EVEN_K is missing else l_0_EVEN_K) else 'a_mask'), indent_width=8))
    yield '\n\n        idx_m = b_k_idx_vals\n        idx_n = offs_b_n[None, :]\n        '
    yield str(context.call((undefined(name='load_input') if l_0_load_input is missing else l_0_load_input), 'B', 'b', ('idx_m', 'idx_n'), mask=(None if (undefined(name='EVEN_K') if l_0_EVEN_K is missing else l_0_EVEN_K) else 'b_mask'), indent_width=8))
    yield '\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n\n    # rematerialize rm and rn to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_n'), 'acc', 'mask'))

blocks = {}
debug_info = '2=18&3=20&4=22&5=24&9=26&10=28&11=30&12=32&40=34&49=38&53=40&64=42'