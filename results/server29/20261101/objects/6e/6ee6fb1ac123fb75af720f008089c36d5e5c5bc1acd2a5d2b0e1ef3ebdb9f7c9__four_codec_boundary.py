"""Four-arm SGLang physical-cache boundary candidate; GPU qualification pending.

No model, device discovery, CUDA compilation, or execution happens on import.
SGLang stores [B,H,V,K]; inherited codecs consume logical [B,H,K,V]. H arms
rotate incoming post-conv Value once and restore the returned core once. Their
cache stays in H coordinates, including across the prefill/decode boundary.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for sub in (HERE / 'inherited/sym', HERE / 'inherited/asym'):
    sys.path.insert(0, str(sub))
from group32_gpu_codec import encode as inherited_sym_encode
from canonical_codec import encode_torch as canonical_cpu_encode


@dataclass(frozen=True)
class Arm:
    identifier: str
    hadamard: bool
    asymmetric: bool


ARMS = {
    'B0': Arm('B0', False, False),
    'B1': Arm('B1', False, True),
    'B2': Arm('B2', True, False),
    'B3': Arm('B3', True, True),
}


class FourCodecBoundary:
    def __init__(self, torch, arm, h, *, gpu_authorized=False,
                 canonical_cuda_backend=None, strict_comparator=None,
                 durable_evidence=None):
        if arm not in ARMS:
            raise ValueError('EXACT_FOUR_ARM_ID_REQUIRED')
        self.torch, self.arm, self.h = torch, ARMS[arm], h
        if h.dtype != torch.float32 or tuple(h.shape) != (128, 128):
            raise ValueError('FROZEN_FP32_VALUE_H128_REQUIRED')
        self.authorized = gpu_authorized is True
        self.asym_cuda = canonical_cuda_backend
        self.compare = strict_comparator
        self.durable = durable_evidence
        self.previous = {}
        self.cursor = {}
        self.writes = self.consumptions = 0

    def _guard(self, x):
        if x.device.type == 'cuda' and not self.authorized:
            raise PermissionError('GPU_STAGE_NOT_AUTHORIZED')

    def enter_value(self, v):
        self._guard(v)
        if not self.arm.hadamard:
            return v  # No Identity GEMM, no inherited H cache or signed-zero change.
        return (v.float() @ self.h).to(v.dtype)

    def restore_core(self, core):
        if isinstance(core, tuple):
            return (self.restore_core(core[0]),) + core[1:]
        self._guard(core)
        if not self.arm.hadamard:
            return core
        return (core.float() @ self.h.T).to(core.dtype)

    def before_native_kernel(self, physical_active, layer, position):
        self._guard(physical_active)
        if position != self.cursor.get(layer, -1) + 1:
            raise RuntimeError('CONTIGUOUS_REQUEST_OWN_HISTORY_REQUIRED')
        if layer in self.previous:
            a = physical_active.contiguous().view(self.torch.uint8)
            b = self.previous[layer].contiguous().view(self.torch.uint8)
            if a.shape != b.shape or not self.torch.equal(a, b):
                raise AssertionError('REAL_NEXT_KERNEL_POST_QDQ_BYTES_DIFFER')
            self.consumptions += 1
        elif bool(physical_active.count_nonzero()):
            raise AssertionError('FRESH_REQUEST_ZERO_STATE_REQUIRED')

    def after_native_kernel(self, pool, indices, layer, position, *, probe=False):
        torch = self.torch
        self._guard(pool)
        if position != self.cursor.get(layer, -1) + 1:
            raise RuntimeError('DUPLICATE_OR_NONCONTIGUOUS_WRITE')
        if pool.dtype != torch.float32 or pool.ndim != 4 or tuple(pool.shape[-2:]) != (128, 128):
            raise ValueError('NATIVE_PHYSICAL_FP32_CACHE_REQUIRED')
        idx = indices.to(torch.int64)
        physical = pool.index_select(0, idx)
        logical = physical.transpose(-1, -2).contiguous()
        if not bool(torch.isfinite(logical).all()):
            raise ArithmeticError('NONFINITE_PRE_QDQ_INPUT')
        cuda = logical.device.type == 'cuda'
        # Before the first actual Asym CUDA encode/comparison, retain real S0.
        strict = self.arm.asymmetric and (probe or layer not in self.previous)
        context = dict(arm=self.arm.identifier, layer=layer, position=position,
                       logical_layout='B,H,K,V', physical_layout='B,H,V,K')
        host_input = None
        if self.arm.asymmetric and cuda and strict:
            if self.durable is None or self.compare is None:
                raise RuntimeError('ASYM_STRICT_REQUIRES_DURABLE_S0_AND_COMPARATOR')
            host_input = logical.detach().to(device='cpu', copy=True)
            self.durable('PRE_ENCODE_INPUT', context, host_input, None)
        encoded = reference = None
        try:
            if self.arm.asymmetric:
                if cuda:
                    if self.asym_cuda is None:
                        raise RuntimeError('CANONICAL_V2_AUTHORIZED_CUDA_BACKEND_REQUIRED')
                    encoded = self.asym_cuda.encode(logical, capture_stages=strict)
                else:
                    encoded = canonical_cpu_encode(torch, logical, capture_stages=strict)
                if cuda and strict:
                    reference = canonical_cpu_encode(torch, host_input, capture_stages=True)
                    self.compare(encoded, reference)
                    self.durable('ACTUAL_CANONICAL_STAGES', context, host_input, encoded.stages)
            else:
                # Old Pilot B1 means H+Sym32. Only its codec is reused here;
                # this task's B0/B1/B2/B3 identifiers are never passed through.
                encoded = inherited_sym_encode(torch, logical, 'B1', self.authorized)
        except Exception as error:
            if self.arm.asymmetric and cuda and self.durable is not None:
                if host_input is None:
                    host_input = logical.detach().to(device='cpu', copy=True)
                self.durable('FIRST_NUMERICAL_FAILURE', dict(context, error=repr(error)), host_input,
                             dict(actual=None if encoded is None else encoded.stages,
                                  cpu_reference=None if reference is None else reference.stages))
            raise
        decoded = encoded.decode(torch)
        restored = decoded.transpose(-1, -2).contiguous()
        if not bool(torch.isfinite(restored).all()):
            raise ArithmeticError('NONFINITE_POST_QDQ_CACHE')
        pool.index_copy_(0, idx, restored)
        actual = pool.index_select(0, idx)
        if not torch.equal(actual.view(torch.uint8), restored.view(torch.uint8)):
            raise AssertionError('PHYSICAL_QDQ_WRITEBACK_BYTES_DIFFER')
        self.previous[layer] = actual.detach().clone()  # Audit only, never continuation.
        self.cursor[layer] = position
        self.writes += 1
        return encoded

    def close(self):
        self.previous.clear()


def stepwise_prefill(*, torch, q, k, v, g, beta, A_log, dt_bias,
                     ssm_states, cache_indices, original_decode, boundary,
                     layer, start_position=0, lower_bound=None):
    """Candidate common stepwise KDA prefill, not chunk-end QDQ.

    Caller supplies native post-conv q/k/v and raw Key-channel gate/beta.
    Original recurrent decode kernel computes all gates and updates. No gate
    formula or readout is reimplemented. Only batch=1 is accepted. Real GPU
    shape/functional/sampler qualification is mandatory before formal use.
    """
    if q.ndim != 4 or q.shape[0] != 1 or cache_indices.numel() != 1:
        raise ValueError('STEPWISE_PREFILL_REQUIRES_SINGLE_REQUEST')
    count = q.shape[1]
    if k.shape[1] != count or v.shape[1] != count:
        raise ValueError('MATCHED_QKV_TOKEN_COUNTS_REQUIRED')
    loc = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    def gate_slice(x, t):
        if x.shape[0] == count:
            return x[t:t + 1].contiguous()
        if x.ndim >= 2 and x.shape[0] == 1 and x.shape[1] == count:
            return x[:, t:t + 1].contiguous()
        raise ValueError('UNQUALIFIED_GATE_TOKEN_LAYOUT')
    outputs = []
    for t in range(count):
        pos = start_position + t
        boundary.before_native_kernel(ssm_states.index_select(0, cache_indices.to(torch.int64)), layer, pos)
        kwargs = dict(q=q[:, t:t + 1].contiguous(), k=k[:, t:t + 1].contiguous(),
                      v=boundary.enter_value(v[:, t:t + 1].contiguous()),
                      a=gate_slice(g, t), b=gate_slice(beta, t), A_log=A_log,
                      dt_bias=dt_bias, ssm_states=ssm_states,
                      cache_indices=cache_indices, query_start_loc=loc,
                      lower_bound=lower_bound)
        out = original_decode(**kwargs)
        boundary.after_native_kernel(ssm_states, cache_indices, layer, pos, probe=(pos == 0))
        outputs.append(boundary.restore_core(out))
    return torch.cat(outputs, dim=1)
