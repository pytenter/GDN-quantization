"""Read-only official STEPQuant statistics beside the unchanged native GDN.

Native kernels are the only source of model outputs/cache. Official FP32
RecurrentKernel runs a discarded shadow on native-normalized q/k, with no
double normalization/scaling. Shadow states are calibration references, not a
claim of byte equality to every native chunk intermediate state.
"""
import inspect
import torch
from stepquant.adapters import RecurrentKernel
from stepquant.calibration import LayerStatistics


class NativeShadowStatistics:
    def __init__(self, model, native_module, sample_every=8, max_snapshots=64):
        self.model = model
        self.native = native_module
        self.statistics = {}
        self.layer_name_to_index = {}
        self.current_name = None
        self.handles = []
        self.originals = {}
        self.calls = {}
        self.sample_every = sample_every
        self.max_snapshots = max_snapshots

    def wrap(self, function, name):
        signature = inspect.signature(function)
        observer = self.statistics[name]
        shadow = RecurrentKernel('gdn', observer)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            a = arguments.arguments
            q, k, v = a['query'], a['key'], a['value']
            g, beta = a['g'], a['beta']
            if q.shape[0] != 1 or not a['output_final_state']:
                raise ValueError('Calibration requires one unpadded request and final state')
            extra = a.get('kwargs', {})
            if set(extra) - {'cu_seqlens'}:
                raise ValueError('Unaudited calibration kernel kwargs: ' + str(sorted(extra)))
            initial = a['initial_state']
            initial_shadow = None if initial is None else initial.detach().clone()
            use_norm = a['use_qk_l2norm_in_kernel']
            # Exactly native l2norm at its input dtype, then the original
            # RecurrentKernel applies only its single 1/sqrt(key_dim) scale.
            q_shadow = self.native.l2norm(q, dim=-1, eps=1e-6) if use_norm else q
            k_shadow = self.native.l2norm(k, dim=-1, eps=1e-6) if use_norm else k
            native_result = function(*args, **kwargs)
            with torch.no_grad():
                shadow(q_shadow.detach(), k_shadow.detach(), v.detach(),
                       g=g.detach(), beta=beta.detach(), initial_state=initial_shadow,
                       output_final_state=True, use_qk_l2norm_in_kernel=False,
                       cu_seqlens=extra.get('cu_seqlens'))
            self.calls[name] = self.calls.get(name, 0) + 1
            # The original tuple and tensors are returned. No shadow value is
            # copied to native output, original initial state, or model cache.
            return native_result
        return wrapped

    def __enter__(self):
        for name, module in self.model.named_modules():
            if type(module).__name__ != 'Qwen3_5GatedDeltaNet':
                continue
            self.statistics[name] = LayerStatistics('gdn', self.sample_every, self.max_snapshots)
            self.layer_name_to_index[name] = int(module.layer_idx)
            self.handles.append(module.register_forward_pre_hook(
                lambda mod, args, n=name: setattr(self, 'current_name', n)))
        if not self.statistics:
            raise ValueError('No actual Qwen3_5GatedDeltaNet modules')
        for symbol in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            original = getattr(self.native, symbol)
            self.originals[symbol] = original
            signature = inspect.signature(original)
            if not {'query', 'key', 'value', 'g', 'beta', 'initial_state', 'output_final_state'} <= set(signature.parameters):
                raise ValueError('Native kernel interface not qualified: ' + symbol)
            def dispatch(*args, original=original, **kwargs):
                if self.current_name not in self.statistics:
                    raise ValueError('Kernel without known GDN layer identity')
                return self.wrap(original, self.current_name)(*args, **kwargs)
            setattr(self.native, symbol, dispatch)
        return self

    def __exit__(self, *exc):
        for symbol, original in self.originals.items():
            setattr(self.native, symbol, original)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.originals.clear()

    def export_statistics(self):
        values = {name: observer.export() for name, observer in self.statistics.items()}
        if set(values) != set(self.layer_name_to_index):
            raise ValueError('Statistics/layer identity mismatch')
        return values
