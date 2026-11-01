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
    l_0_modification = resolve('modification')
    l_0_store_output = resolve('store_output')
    try:
        t_1 = environment.filters['indent_except_first']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No filter named 'indent_except_first' found.")
    pass
    yield '\n'
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'A', 'B', 'C'))
    yield '\n\n\n    # B2B_GEMM_RIGHT_TRITON_ENTRANCE\n\n    # dynamic shapes\n    M = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 0))
    yield '\n    N = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 1))
    yield '\n    O = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'C', 0))
    yield '\n    P = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'C', 1))
    yield '\n\n    # dynamic strides\n    stride_am = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 0))
    yield '\n    stride_an = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 1))
    yield '\n    stride_bn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 0))
    yield '\n    stride_bo = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 1))
    yield '\n    stride_co = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'C', 0))
    yield '\n    stride_cp = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'C', 1))
    yield '\n\n    # output block counts\n    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)\n    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)\n\n    # internal block counts\n    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)\n    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)\n\n    # output block ids\n    pid = tl.program_id(axis=0)\n    m_block_id = pid // num_p_block\n    p_block_id = pid % num_p_block\n\n    # accumulator\n    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)\n\n    # main loop (two cases)\n    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))\n    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))\n    # (A @ subgraph(B @ C))\n    offs_n = tl.arange(0, BLOCK_SIZE_N)\n    for _ in range(num_n_block):\n        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)\n        a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)\n        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N\n        acc_bc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)\n        offs_o = tl.arange(0, BLOCK_SIZE_O)\n        for __ in range(num_o_block):\n            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)\n            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)\n            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O\n            c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)\n            c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)\n            c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P\n            acc_bc += tl.dot(b, c, out_dtype=tl.float32)\n            offs_o += BLOCK_SIZE_O\n        # apply the subgraph\n        '
    yield str(t_1(context.call((undefined(name='modification') if l_0_modification is missing else l_0_modification), subgraph_number=0, output_name='post_subgraph_acc_bc', inner_mm='acc_bc'), 2))
    yield '\n        acc += tl.dot(a, post_subgraph_acc_bc, out_dtype=tl.float32)\n        offs_n += BLOCK_SIZE_N\n\n    # type conversion\n    acc = acc.to(tl.float16)\n\n    # store preparation\n    idx_m = offs_m[:, None]\n    idx_p = offs_p[None, :]\n    out_mask = (idx_m < M) & (idx_p < P)\n\n    '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_p'), 'acc', 'out_mask'))

blocks = {}
debug_info = '2=23&8=25&9=27&10=29&11=31&14=33&15=35&16=37&17=39&18=41&19=43&62=45&74=47'