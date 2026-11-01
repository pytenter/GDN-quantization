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
    yield '\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n\n    start_pid = tl.program_id(0)\n    grid_m = tl.cdiv(M, BLOCK_M)\n    grid_n = tl.cdiv(N, BLOCK_N)\n    k_tiles = tl.cdiv(K, BLOCK_K)\n    num_tiles = grid_m * grid_n\n    tiles_per_SM = num_tiles // NUM_SMS\n    if start_pid < num_tiles % NUM_SMS:\n        tiles_per_SM += 1\n\n    tile_id = start_pid - NUM_SMS\n    ki = -1\n\n    width = GROUP_M * grid_n\n    rk_for_mask = tl.arange(0, BLOCK_K)\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n\n    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE\n    a_desc_ptr = workspace_base\n    b_desc_ptr = workspace_base + TMA_SIZE\n\n    triton.language.extra.cuda.experimental_device_tensormap_create2d(\n        desc_ptr=a_desc_ptr,\n        global_address=A,\n        load_size=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],\n        global_size=[M, K] if A_ROW_MAJOR else [K, M],\n        element_ty=A.dtype.element_ty,\n    )\n    triton.language.extra.cuda.experimental_device_tensormap_create2d(\n        desc_ptr=b_desc_ptr,\n        global_address=B,\n        load_size=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],\n        global_size=[K, N] if B_ROW_MAJOR else [N, K],\n        element_ty=B.dtype.element_ty,\n    )\n\n    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)\n    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)\n\n    pid_m = 0\n    pid_n = 0\n    rm = 0\n    rn = 0\n\n    for _ in range(0, k_tiles * tiles_per_SM):\n        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)\n        if ki == 0:\n            tile_id += NUM_SMS\n            # re-order program ID for better L2 performance\n            group_id = tile_id // width\n            group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n            pid_m = group_id * GROUP_M + (tile_id % group_size)\n            pid_n = (tile_id % width) // (group_size)\n\n            rm = pid_m * BLOCK_M\n            rn = pid_n * BLOCK_N\n\n        rk = ki * BLOCK_K\n\n        a = tl._experimental_descriptor_load(\n            a_desc_ptr,\n            [rm, rk] if A_ROW_MAJOR else [rk, rm],\n            [BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],\n            A.dtype.element_ty,\n        )\n        b = tl._experimental_descriptor_load(\n            b_desc_ptr,\n            [rk, rn] if B_ROW_MAJOR else [rn, rk],\n            [BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],\n            B.dtype.element_ty,\n        )\n        acc += tl.dot(\n            a if A_ROW_MAJOR else a.T,\n            b if B_ROW_MAJOR else b.T,\n            allow_tf32=ALLOW_TF32,\n        )\n\n        if ki == k_tiles - 1:\n            # rematerialize rm and rn to save registers\n            rcm = rm + tl.arange(0, BLOCK_M)\n            rcn = rn + tl.arange(0, BLOCK_N)\n            idx_m = rcm[:, None]\n            idx_n = rcn[None, :]\n            mask = (idx_m < M) & (idx_n < N)\n\n            # inductor generates a suffix\n            '
    yield str(context.call((undefined(name='store_output') if l_0_store_output is missing else l_0_store_output), ('idx_m', 'idx_n'), 'acc', 'mask', indent_width=12))
    yield '\n            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)'

blocks = {}
debug_info = '2=15&3=17&4=19&5=21&95=23'