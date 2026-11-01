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
    yield str(context.call((undefined(name='def_kernel') if l_0_def_kernel is missing else l_0_def_kernel), 'A', 'B', 'A_inverse_scale', 'B_inverse_scale'))
    yield '\n    M = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 0))
    yield '\n    N = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'B', 1))
    yield '\n    K = '
    yield str(context.call((undefined(name='size') if l_0_size is missing else l_0_size), 'A', 1))
    yield '\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n\n    stride_am = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 0))
    yield '\n    stride_ak = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'A', 1))
    yield '\n    stride_bk = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 0))
    yield '\n    stride_bn = '
    yield str(context.call((undefined(name='stride') if l_0_stride is missing else l_0_stride), 'B', 1))
    yield '\n\n    if SCALING_ROWWISE:\n        stride_a_scale_m = 1\n        stride_b_scale_n = 1\n    else:\n        stride_a_scale_m = 0\n        stride_b_scale_n = 0\n\n    start_pid = tl.program_id(axis=0)\n    num_pid_m = tl.cdiv(M, BLOCK_M)\n    num_pid_n = tl.cdiv(N, BLOCK_N)\n    k_tiles = tl.cdiv(K, BLOCK_K)\n    num_tiles = num_pid_m * num_pid_n\n\n    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE\n    a_desc_ptr = workspace_base\n    b_desc_ptr = workspace_base + TMA_SIZE\n\n    triton.language.extra.cuda.experimental_device_tensormap_create2d(\n        desc_ptr=a_desc_ptr,\n        global_address=A,\n        load_size=[BLOCK_M, BLOCK_K],\n        global_size=[M, K],\n        element_ty=A.dtype.element_ty,\n    )\n    triton.language.extra.cuda.experimental_device_tensormap_create2d(\n        desc_ptr=b_desc_ptr,\n        global_address=B,\n        load_size=[BLOCK_N, BLOCK_K],\n        global_size=[N, K],\n        element_ty=B.dtype.element_ty,\n    )\n\n    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)\n    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)\n\n    tiles_per_SM = num_tiles // NUM_SMS\n    if start_pid < num_tiles % NUM_SMS:\n        tiles_per_SM += 1\n\n    tile_id = start_pid - NUM_SMS\n    ki = -1\n\n    pid_m = 0\n    pid_n = 0\n    offs_am = 0\n    offs_bn = 0\n\n    num_pid_in_group = GROUP_M * num_pid_n\n    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    a_scale, b_scale = load_scales(A_inverse_scale, B_inverse_scale, SCALING_ROWWISE)\n\n    for _ in range(0, k_tiles * tiles_per_SM):\n        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)\n        if ki == 0:\n            tile_id += NUM_SMS\n            group_id = tile_id // num_pid_in_group\n            first_pid_m = group_id * GROUP_M\n            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)\n            pid_m = first_pid_m + (tile_id % group_size_m)\n            pid_n = (tile_id % num_pid_in_group) // group_size_m\n\n            offs_am = pid_m * BLOCK_M\n            offs_bn = pid_n * BLOCK_N\n\n        offs_k = ki * BLOCK_K\n\n        a = tl._experimental_descriptor_load(\n            a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K],  A.dtype.element_ty\n        )\n        b = tl._experimental_descriptor_load(\n            b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],  B.dtype.element_ty\n        )\n        if USE_FAST_ACCUM:\n            accumulator = tl.dot(a, b.T, accumulator)\n        else:\n            accumulator += tl.dot(a, b.T)\n\n        if ki == k_tiles - 1:\n            # Apply inverse scaling\n            offs_cm = offs_am + tl.arange(0, BLOCK_M)\n            offs_cn = offs_bn + tl.arange(0, BLOCK_N)\n            # Apply scaling\n            accumulator = apply_scaling(\n                accumulator,\n                a_scale,\n                b_scale,\n                SCALING_ROWWISE,\n                offs_cm,\n                offs_cn,\n                M,\n                N,\n                stride_a_scale_m,\n                stride_b_scale_n,\n            )\n\n            idx_m = offs_cm[:, None]\n            idx_n = offs_cn[None, :]\n            mask = (idx_m < M) & (idx_n < N)\n            # inductor generates a suffix\n            '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_n'), 'accumulator', 'mask', indent_width=12))
    yield "\n            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n\n@triton.jit\ndef load_scales(a_scale_ptr, b_scale_ptr, SCALING_ROWWISE: tl.constexpr):\n    if SCALING_ROWWISE:\n        # For row-wise scaling, we'll return the pointers\n        return a_scale_ptr, b_scale_ptr\n    else:\n        # For per-tensor scaling, we'll load the scalar values\n        a_scale = tl.load(a_scale_ptr)\n        b_scale = tl.load(b_scale_ptr)\n        return a_scale, b_scale\n\n@triton.jit\ndef apply_scaling(\n    accumulator,\n    a_scale,\n    b_scale,\n    SCALING_ROWWISE: tl.constexpr,\n    offs_cm,\n    offs_cn,\n    M,\n    N,\n    stride_a_scale_m,\n    stride_b_scale_n,\n):\n    if SCALING_ROWWISE:\n        # For row-wise scaling, we need to load the scales for each row/column\n        a_scales = tl.load(\n            a_scale + (offs_cm * stride_a_scale_m),\n            mask=offs_cm < M,\n            other=0.0,\n        )\n        b_scales = tl.load(\n            b_scale + (offs_cn * stride_b_scale_n),\n            mask=offs_cn < N,\n            other=0.0,\n        )\n        acc_scale = a_scales[:, None] * b_scales[None, :]\n    else:\n        # For per-tensor scaling, we can directly use the loaded scalar values\n        acc_scale = a_scale * b_scale\n\n    return accumulator * acc_scale"

blocks = {}
debug_info = '2=16&3=18&4=20&5=22&10=24&11=26&12=28&13=30&114=32'