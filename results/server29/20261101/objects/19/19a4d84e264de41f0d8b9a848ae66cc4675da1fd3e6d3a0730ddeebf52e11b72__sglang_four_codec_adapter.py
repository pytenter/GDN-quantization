"""Single-request persistent-QDQ hooks around original SGLang KDA methods.

No torch, model, SGLang, device query or compilation runs on import. Production
math lives in the separately frozen FourCodecBoundary and original KDA decode.
All four arms use the same original unpacked Triton recurrent kernel. Prefill
calls that kernel once per token, after the original causal convolution.
"""


class SingleRequestKDAAdapter:
    def __init__(self, boundary, *, probe_positions=(0, 31, 127, 511)):
        self.boundary = boundary
        self.torch = boundary.torch
        self.probes = frozenset(probe_positions)
        self.current_layer = None
        self.installed = False
        self.actual_caches = {}
        self.prefill_writes = 0
        self.decode_writes = 0
        self.gate_probes = []
        self.calls = []

    def _layer(self):
        if self.current_layer is None:
            raise RuntimeError('ACTUAL_BACKEND_LAYER_CONTEXT_REQUIRED')
        return self.current_layer

    def _position(self, layer):
        return self.boundary.cursor.get(layer, -1) + 1

    def _native(self, original, dispatcher, kw, phase):
        torch = self.torch
        layer = self._layer()
        pos = self._position(layer)
        pool, indices = kw['ssm_states'], kw['cache_indices']
        if indices.numel() != 1:
            raise ValueError('BATCH_ONE_REAL_CACHE_SLOT_REQUIRED')
        if kw['q'].shape[:2] != (1, 1):
            raise ValueError('ONE_NATIVE_RECURRENT_TOKEN_REQUIRED')
        if kw['query_start_loc'].tolist() != [0, 1]:
            raise ValueError('ONE_TOKEN_QUERY_LOCATION_REQUIRED')
        if any(kw.get(k) is not None for k in
               ('replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_write_pos')):
            raise ValueError('BUFFERED_OR_DELAYED_STATE_UPDATE_FORBIDDEN')
        active = pool.index_select(0, indices.to(torch.int64))
        self.boundary.before_native_kernel(active, layer, pos)
        self.actual_caches[layer] = (pool, indices)
        before = None
        if pos in self.probes:
            before = {k: kw[k].detach().clone() for k in ('q', 'k', 'a', 'b', 'A_log', 'dt_bias')}
        native = dict(kw)
        native['v'] = self.boundary.enter_value(kw['v'])
        result = original(dispatcher, **native)
        if before is not None:
            checks = {k: bool(torch.equal(kw[k].contiguous().view(torch.uint8),
                                         v.contiguous().view(torch.uint8)))
                      for k, v in before.items()}
            if not all(checks.values()):
                raise AssertionError(('KEY_QUERY_GATE_OR_PARAMETERS_MODIFIED', checks))
            self.gate_probes.append(dict(layer=layer, position=pos, unchanged=checks))
        self.boundary.after_native_kernel(pool, indices, layer, pos, probe=pos in self.probes)
        if phase == 'prefill':
            self.prefill_writes += 1
        else:
            self.decode_writes += 1
        return self.boundary.restore_core(result)

    def install(self, backend_class=None, dispatcher_class=None):
        if self.installed:
            raise RuntimeError('DUPLICATE_ADAPTER_INSTALL')
        if backend_class is None or dispatcher_class is None:
            from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend, KDAKernelDispatcher
            backend_class, dispatcher_class = KDAAttnBackend, KDAKernelDispatcher
        if getattr(dispatcher_class, '_four_codec_persistent_adapter', False):
            raise RuntimeError('ANOTHER_FOUR_CODEC_ADAPTER_ALREADY_INSTALLED')
        adapter = self
        old_decode = dispatcher_class.decode
        old_extend = dispatcher_class.extend
        old_backend_decode = backend_class.forward_decode
        old_backend_extend = backend_class.forward_extend
        self.originals = (backend_class, dispatcher_class, old_decode, old_extend,
                          old_backend_decode, old_backend_extend)

        def call_backend(original, backend, layer, args, kwargs, phase):
            if adapter.current_layer is not None:
                raise RuntimeError('NESTED_OR_CONCURRENT_KDA_REQUEST_FORBIDDEN')
            adapter.current_layer = int(layer.layer_id)
            batch = kwargs.get('forward_batch')
            if batch is None:
                batch = args[0]
            if (layer.head_k_dim, layer.head_v_dim) != (128, 128):
                raise ValueError('FROZEN_KDA_K128_V128_REQUIRED')
            # Select an existing original kernel, shared by all four arms and
            # stepwise prefill. No kernel equation, gate or sampler is replaced.
            backend.kernel_dispatcher.supports_packed_decode = False
            if getattr(backend.forward_metadata, 'has_mamba_track_mask', False):
                raise ValueError('PREFIX_TRACK_OR_SPECULATIVE_STATE_PATH_UNQUALIFIED')
            expected = int(batch.input_ids.numel()) if phase == 'prefill' else 1
            writes = adapter.boundary.writes
            try:
                result = original(backend, layer, *args, **kwargs)
                actual = adapter.boundary.writes - writes
                if actual != expected:
                    raise AssertionError(('REAL_KDA_UPDATE_BYPASSED_QDQ', phase, actual, expected))
                adapter.calls.append(dict(layer=adapter.current_layer, phase=phase,
                                          tokens=expected, QDQ_writes=actual))
                return result
            finally:
                adapter.current_layer = None

        def backend_decode(backend, layer, *args, **kwargs):
            return call_backend(old_backend_decode, backend, layer, args, kwargs, 'decode')

        def backend_extend(backend, layer, *args, **kwargs):
            return call_backend(old_backend_extend, backend, layer, args, kwargs, 'prefill')

        def decode(dispatcher, *args, **kwargs):
            if args:
                raise ValueError('UNQUALIFIED_POSITIONAL_DISPATCH')
            return adapter._native(old_decode, dispatcher, kwargs, 'decode')

        def extend(dispatcher, *args, **kwargs):
            if args or kwargs.get('return_intermediate_states') or kwargs.get('is_spec_decode'):
                raise ValueError('SINGLE_REQUEST_NO_ROLLBACK_STEPWISE_PREFILL_REQUIRED')
            q, k, v = kwargs['q'], kwargs['k'], kwargs['v']
            if q.ndim != 4 or q.shape[0] != 1 or kwargs['cache_indices'].numel() != 1:
                raise ValueError('ONE_REQUEST_PREFILL_REQUIRED')
            count = q.shape[1]
            if kwargs['query_start_loc'].tolist() != [0, count]:
                raise ValueError('ACTUAL_SINGLE_REQUEST_PREFILL_BOUNDS_REQUIRED')

            def gate_slice(x, t):
                if x.shape[0] == count:
                    return x[t:t+1].contiguous()
                if x.ndim >= 2 and x.shape[0] == 1 and x.shape[1] == count:
                    return x[:, t:t+1].contiguous()
                raise ValueError('UNQUALIFIED_GATE_TOKEN_LAYOUT')

            loc = adapter.torch.tensor([0, 1], dtype=adapter.torch.int32, device=q.device)
            result = []
            for t in range(count):
                native = dict(q=q[:, t:t+1].contiguous(), k=k[:, t:t+1].contiguous(),
                              v=v[:, t:t+1].contiguous(),
                              a=gate_slice(kwargs['g'], t), b=gate_slice(kwargs['beta'], t),
                              A_log=kwargs['A_log'], dt_bias=kwargs['dt_bias'],
                              ssm_states=kwargs['ssm_states'], cache_indices=kwargs['cache_indices'],
                              query_start_loc=loc, lower_bound=kwargs.get('lower_bound'))
                result.append(adapter._native(old_decode, dispatcher, native, 'prefill'))
            return adapter.torch.cat(result, dim=1)

        backend_class.forward_decode = backend_decode
        backend_class.forward_extend = backend_extend
        dispatcher_class.decode = decode
        dispatcher_class.extend = extend
        dispatcher_class._four_codec_persistent_adapter = True
        self.installed = True

    def close(self):
        if self.installed:
            backend, dispatcher, dec, ext, bdec, bext = self.originals
            backend.forward_decode, backend.forward_extend = bdec, bext
            dispatcher.decode, dispatcher.extend = dec, ext
            delattr(dispatcher, '_four_codec_persistent_adapter')
            self.installed = False
        self.actual_caches.clear()

