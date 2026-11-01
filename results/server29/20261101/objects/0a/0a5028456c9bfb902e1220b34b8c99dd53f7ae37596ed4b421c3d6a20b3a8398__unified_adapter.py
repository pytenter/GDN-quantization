"""Original H128/helper timing with a layer-batched replacement of QDQ only."""
import json
from pathlib import Path
import torch
from canonical_h128 import QwenKeyHadamardPatch, get_state
from unified_codec import encode, decode, require_consumed, slice_encoded


class UnifiedQDQ:
    def __init__(self, model, condition, layers, observe=True, callback=None):
        if condition not in ('base', 'W', 'R', 'RW'):
            raise ValueError('Unknown frozen quantizer condition')
        self.model = model
        self.condition = condition
        self.method = condition
        self.layers = tuple(layers)
        self.observe = observe
        self.callback = callback
        self.h = QwenKeyHadamardPatch(model)
        self.previous = {}
        self.qdq_calls = 0
        self.consumption_calls = 0
        self.consumption_layers = set()
        self.current_token = 0
        self.saved = []
        root = Path(__file__).resolve().parents[1]
        weights = json.loads((root / 'TRANSPORTED_QUERY_OMEGA.json').read_text())
        selected = json.loads((root / 'CALIBRATION_PARAMETER_SELECTION.json').read_text())
        if weights['layer_ids'] != list(self.layers) or weights['shape'] != [24, 32, 128]:
            raise RuntimeError('Frozen actual layer/head omega map differs')
        if weights['AIME_used'] is not False:
            raise RuntimeError('AIME-derived sensitivity is forbidden')
        self.a = selected['selected']['a']
        self.b = selected['selected']['b']
        self.omega = torch.tensor(weights['omega_normalized_fp32'], dtype=torch.float32,
                                  device='cuda:0').contiguous()
        if tuple(self.omega.shape) != (24, 32, 128):
            raise RuntimeError('Shared sensitivity tensor shape differs')
        if not bool((torch.isfinite(self.omega) & (self.omega >= 1/16) & (self.omega <= 16)).all()):
            raise RuntimeError('Frozen sensitivity must be finite within [1/16,16]')
        from unified_codec import PARAMETER_GRID
        if (self.a, self.b) not in PARAMETER_GRID:
            raise RuntimeError('CAL-selected row parameters outside frozen grid')
        self.metadata = {}

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        self.h.__enter__()
        for name in ('torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule'):
            original = getattr(Q, name)
            self.saved.append((Q, name, original))

            def wrap(fn=original):
                def call(query, key, value, *args, **kwargs):
                    if query.shape[1] != 1:
                        raise RuntimeError('Multi-token chunk violates per-token QDQ')
                    layer = int(self.h.current_layer)
                    state = kwargs.get('initial_state', args[2] if len(args) > 2 else None)
                    if self.current_token > 0 and self.observe:
                        if state is None or layer not in self.previous:
                            raise RuntimeError('Missing persistent state')
                        try:
                            require_consumed(state, self.previous[layer])
                        except BaseException:
                            if self.callback:
                                self.callback('NEXT_CONSUMPTION_FAILURE', self.current_token, layer,
                                              dict(actual=state, expected=decode(self.previous[layer])))
                            raise
                        self.consumption_calls += 1
                        self.consumption_layers.add(layer)
                        if self.callback:
                            self.callback('KERNEL_INITIAL', self.current_token, layer, dict(actual=state))
                    return fn(query, key, value, *args, **kwargs)
                return call

            setattr(Q, name, wrap())
        return self

    def after_token(self, cache):
        if self.current_token != self.qdq_calls:
            raise RuntimeError('QDQ sequence mismatch')
        states = [get_state(cache, layer) for layer in self.layers]
        if any(s.dtype != torch.float32 or tuple(s.shape) != (1, 32, 128, 128)
               or s.device != self.omega.device for s in states):
            raise RuntimeError('Actual native FP32 recurrent carrier shape/device differs')
        if self.callback:
            for layer, state in zip(self.layers, states):
                self.callback('PRE_QDQ', self.current_token, layer, dict(state=state))
        # This temporary tensor is the current updated state only. It is discarded
        # after immediate decoded writeback, never used as a continuation cache.
        with torch.profiler.record_function('WSCALE_ROWBAL_QDQ'):
            current = torch.cat(states, dim=0)
            encoded = encode(current, self.condition, self.omega,
                             a=self.a, b=self.b, check_finite=True)
            rebuilt = decode(encoded)
            # The codec checks finite/positive/legal fields once for all24 layers
            # before any cache is written. No Python head/group/candidate loop.
            for i, state in enumerate(states):
                state.copy_(rebuilt[i:i+1])
            expected = encoded.cpu() if self.observe else None
            if self.observe:
                for i, (layer, state) in enumerate(zip(self.layers, states)):
                    self.previous[layer] = slice_encoded(expected, i)
                    require_consumed(state, self.previous[layer])
            if self.callback:
                for i, (layer, state) in enumerate(zip(self.layers, states)):
                    e = slice_encoded(encoded, i)
                    self.callback('POST_QDQ', self.current_token, layer,
                                  dict(codes=e.codes, scales=e.scales, decoded=rebuilt[i:i+1],
                                       writeback=state, negative_zero_flags=e.negative_zero_flags,
                                       row_factors=e.row_factors, lambda_indices=e.lambda_indices))
            self.metadata = dict(omega_bytes=self.omega.numel()*self.omega.element_size(),
                codes_bytes=encoded.codes.numel()*encoded.codes.element_size(),
                scales_bytes=encoded.scales.numel()*encoded.scales.element_size(),
                row_factors_bytes=encoded.row_factors.numel()*encoded.row_factors.element_size(),
                lambda_indices_bytes=encoded.lambda_indices.numel()*encoded.lambda_indices.element_size(),
                negative_zero_flags_bytes=encoded.negative_zero_flags.numel()*encoded.negative_zero_flags.element_size(),
                weighted_loss_workspace_bytes=encoded.weighted_losses.numel()*encoded.weighted_losses.element_size(),
                FP32_carrier_bytes=sum(s.numel()*s.element_size() for s in states),
                temporary_current_FP32_bytes=current.numel()*current.element_size(),
                temporary_decoded_FP32_bytes=rebuilt.numel()*rebuilt.element_size())
        self.qdq_calls += 1
        self.current_token += 1

    def __exit__(self, *args):
        for module, name, fn in self.saved:
            setattr(module, name, fn)
        self.h.__exit__(*args)

    def audit(self):
        return dict(method=self.method, condition=self.condition, coordinate='H128',
                    tokens=self.qdq_calls, layer_count=len(self.layers),
                    qdq_layer_updates=self.qdq_calls*len(self.layers), token0_QDQ=self.qdq_calls > 0,
                    next_consumption_checks=self.consumption_calls,
                    consumption_layers=sorted(self.consumption_layers),
                    expected_checks=max(0,self.qdq_calls-1)*len(self.layers),
                    actual_values_and_bytes_compared=self.observe,
                    unquantized_shadow=False, recurrent_carrier='native_FP32_post_QDQ',
                    packed_INT8_cache=False, group_axis='Key', group_size=32,
                    batch_layer_count=len(self.layers), finite_check_per_token='one_all_layer_GPU_flag_sync',
                    shared_CAL_omega=True, shared_row_parameters=dict(a=self.a,b=self.b),
                    dynamic_metadata_and_workspace=self.metadata)
