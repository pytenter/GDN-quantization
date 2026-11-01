"""Decode-only LQ-style direct read; unchanged PACE prefill and state codec."""
from dataclasses import dataclass
import math
import torch
from pace_buffer import PACEBuffer, _BufferedToken, native_l2norm
from pace_quantizer import PackedAffineGroups, PackedAffineState
from packed_direct_read import packed_readout, torch_readout


@dataclass(frozen=True)
class DeltaRecord:
    key: PackedAffineGroups
    u: torch.Tensor
    log_decay: torch.Tensor


class LQUnrollBuffer(PACEBuffer):
    def __init__(self, decode_period, int8_mask, *, prefill_period=256,
                 validation_cpu=False, capture_validation=False,
                 verify_batched_keys=False, **kwargs):
        if decode_period not in (8, 16, 32):
            raise ValueError("Only dP8, dP16 and throughput-only dP32 are authorized")
        super().__init__(8, int8_mask, prefill_period=prefill_period, **kwargs)
        if self.fixture_only:
            raise ValueError("Unroll production always uses original packed state and K4")
        self.decode_period = decode_period
        self._slots = [None] * decode_period
        self.validation_cpu = validation_cpu
        self.capture_validation = capture_validation
        self.last_delta = None
        self.last_boundary_state = None
        self.decode_direct_calls = 0
        self.decode_dense_materializations = 0
        self.validation_dense_materializations = 0
        self.prefill_replay_calls = 0
        self.initial_dense_materializations = 0
        self.packed_kernel_launches = 0
        self._head_maps = {}
        self.peak_vector_scratch_bytes = 0
        self.verify_batched_keys = verify_batched_keys
        self.batched_key_exact_checks = 0

    def _batched_keys(self, records):
        """Batch identical codec operations across records, then native BF16 norm.

        No precision/formula changes: original packed decoder, BF16 conversion,
        repeated-head layout, and native_l2norm. Only the outer batch axis grows.
        All tensors here are temporary; packed K remains the resident format.
        """
        b,h,k,_ = self._shape
        t=len(records)
        assert all(x.key.bits==4 and x.key.scale_exponents is None for x in records)
        codes=torch.stack([x.key.codes for x in records]).reshape(-1)
        scales=torch.stack([x.key.scales for x in records]).reshape(t*b,self.key_heads,k//32)
        zeros=torch.stack([x.key.zero_points for x in records]).reshape(-1)
        packet=PackedAffineGroups(4,(t*b,self.key_heads,k//32,32),codes,scales,zeros)
        decoded=packet.dequantize().reshape(t*b,self.key_heads,k).to(torch.bfloat16)
        decoded=decoded.repeat_interleave(h//self.key_heads,dim=1)
        keys=native_l2norm(decoded).float().reshape(t,b,h,k)
        if self.verify_batched_keys:
            for i,record in enumerate(records):
                assert torch.equal(keys[i],self._token_keys(record)), 'Batched key codec/norm is not byte-equivalent'
                self.batched_key_exact_checks += 1
        return keys

    def _delta_state(self, *, validation=False):
        if validation:
            self.validation_dense_materializations += 1
        else:
            self.decode_dense_materializations += 1
        state = self.packed_state.dequantize()
        for i in range(self.pending_tokens):
            token = self._slots[i]
            k = self._token_keys(token)
            state = state * token.log_decay.exp().unsqueeze(-1).unsqueeze(-1)
            state = state + k.unsqueeze(-1) * token.u.unsqueeze(-2)
        return state

    def _replay(self):
        if not self.pending_tokens or isinstance(self._slots[0], _BufferedToken):
            self.prefill_replay_calls += 1
            return super()._replay()
        return self._delta_state(validation=True)

    @torch.no_grad()
    def step(self, q, k, v, beta, g, *, scale=None):
        if self.tokens < self.prompt_length:
            return super().step(q, k, v, beta, g, scale=scale)
        if not self._prompt_length_set:
            raise ValueError("Prompt length must be declared")
        if not (q.ndim == k.ndim == v.ndim == 3 and beta.ndim == g.ndim == 2):
            raise ValueError("Invalid recurrent input ranks")
        b, h, dv = v.shape
        dk = k.shape[-1]
        if (dk != 128 or dv != 128 or h != len(self.int8_mask)
                or tuple(beta.shape) != (b,h) or tuple(g.shape) != (b,h)
                or q.shape[0] != b or k.shape[0] != b or q.shape[-1] != dk
                or q.shape[1] not in (h,self.key_heads)):
            raise ValueError("Invalid recurrent input shape")
        for name, x, dtype in (("q",q,torch.bfloat16),("k",k,torch.bfloat16),
                               ("v",v,torch.bfloat16),("beta",beta,torch.bfloat16),
                               ("g",g,torch.float32)):
            if x.dtype != dtype or x.device != v.device or not bool(torch.isfinite(x).all()):
                raise ValueError(name + " dtype/device/finite violation")
        if bool(((beta < 0) | (beta > 1)).any()) or bool((g > 0).any()):
            raise ValueError("Invalid beta or positive log decay")
        if v.device.type != "cuda" and not self.validation_cpu:
            raise RuntimeError("CPU direct prototype is validation-only")
        shape = (b,h,dk,dv)
        if self._shape is None:
            self._shape = shape
            self.packed_state = PackedAffineState.from_tensor(
                torch.zeros(shape,dtype=torch.float32,device=v.device),self.int8_mask)
            self.initial_dense_materializations += 1
        elif self._shape != shape:
            raise ValueError("State shape changed")
        unique = self._unique_keys(k,h)
        encoded = PackedAffineGroups.from_tensor(unique.reshape(b,self.key_heads,dk//32,32),4)
        current = DeltaRecord(encoded, None, g)
        records = self._slots[:self.pending_tokens]
        all_keys = self._batched_keys(records+[current])
        key,keys = all_keys[-1],all_keys[:-1]
        query = q if q.shape[1] == h else q.repeat_interleave(h//self.key_heads,dim=1)
        query = native_l2norm(query).float()
        effective_scale = 1/math.sqrt(dk) if scale is None else float(scale)
        if not math.isfinite(effective_scale) or effective_scale <= 0:
            raise ValueError("Invalid query scale")
        query = query * effective_scale
        if records:
            updates = torch.stack([t.u for t in records])
        else:
            updates = torch.empty((0,b,h,dv),device=v.device,dtype=torch.float32)
        alphas = torch.stack([t.log_decay for t in records]+[g]).exp().contiguous()
        self.peak_vector_scratch_bytes = max(self.peak_vector_scratch_bytes,
            sum(t.numel()*t.element_size() for t in (keys,updates,alphas,key,query)))
        if v.device.type == "cuda":
            if not self._head_maps:
                self._head_maps = {bits:torch.tensor(heads,device=v.device,dtype=torch.int32)
                    for bits,heads in ((4,self.packed_state.int4_heads),(8,self.packed_state.int8_heads))}
            u,out = packed_readout(self.packed_state,keys,updates,alphas,query,key,v,beta,self._head_maps)
            self.packed_kernel_launches += int(self.packed_state.int4 is not None)+int(self.packed_state.int8 is not None)
        else:
            u,out = torch_readout(self.packed_state,keys,updates,alphas,query,key,v,beta)
            self.validation_dense_materializations += 1
        if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(out).all()):
            raise FloatingPointError("Nonfinite direct delta/output")
        if self.pending_tokens >= len(self._slots):
            raise RuntimeError("Delta ring overflow")
        self._slots[self.pending_tokens] = DeltaRecord(encoded,u,g.clone())
        self.pending_tokens += 1
        self.decode_direct_calls += 1
        if self.capture_validation:
            self.last_delta = u.clone()
        self.tokens += 1
        self.peak_decode_buffer_nbytes = max(self.peak_decode_buffer_nbytes,
            self.pending_tokens*self._token_capacity_nbytes(False))
        if self.decode_tokens % self.decode_period == 0:
            state = self._delta_state()
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError("Nonfinite boundary state")
            if self.capture_validation:
                self.last_boundary_state = state.clone()
            self._commit(state,"decode_period")
        return out

    def resident_tensors(self):
        result = {}
        if self.packed_state is not None:
            result.update({"base."+n:t for n,t in self.packed_state.resident_tensors().items()})
        for i,t in enumerate(self._slots[:self.pending_tokens]):
            if isinstance(t.key,PackedAffineGroups):
                result.update({f"buffer.{i}.key.{n}":x for n,x in t.key.resident_tensors().items()})
            else:
                result[f"buffer.{i}.raw_bf16_key"] = t.key
            if isinstance(t,DeltaRecord):
                result[f"buffer.{i}.u"] = t.u
            else:
                result[f"buffer.{i}.value"] = t.value
                result[f"buffer.{i}.beta"] = t.beta
            result[f"buffer.{i}.log_decay"] = t.log_decay
        result.update({f"runtime.head_map.{b}":t for b,t in self._head_maps.items()})
        if self.last_delta is not None:
            result['VALIDATION_ONLY.last_delta'] = self.last_delta
        return result

    def _token_capacity_nbytes(self,prefill):
        if prefill:
            return super()._token_capacity_nbytes(True)
        if self._shape is None:
            return 0
        b,h,k,v = self._shape
        count = b*self.key_heads*(k//32)
        return b*self.key_heads*k//2+2*count+(count+1)//2+4*b*h*v+4*b*h

    def storage_report(self):
        row = super().storage_report()
        row.update(delta_record='packed_K4,u_FP32,log_decay_FP32',beta_storage_included=False,
            decode_direct_calls=self.decode_direct_calls,
            decode_dense_materializations=self.decode_dense_materializations,
            initial_dense_materializations=self.initial_dense_materializations,
            validation_dense_materializations=self.validation_dense_materializations,
            prefill_replay_calls=self.prefill_replay_calls,
            packed_kernel_launches=self.packed_kernel_launches,
            vector_scratch_peak_payload_bytes=self.peak_vector_scratch_bytes,
            runtime_head_map_bytes=sum(t.numel()*t.element_size() for t in self._head_maps.values()),
            batched_key_exact_checks=self.batched_key_exact_checks,
            key_preparation='BATCHED_ORIGINAL_CODEC_AND_NATIVE_BF16_NORM',
            production_backend='TRITON_PACKED_DIRECT' if self._head_maps else 'CPU_VALIDATION_PROTOTYPE',
            no_dense_state_between_decode_calls=True)
        return row
