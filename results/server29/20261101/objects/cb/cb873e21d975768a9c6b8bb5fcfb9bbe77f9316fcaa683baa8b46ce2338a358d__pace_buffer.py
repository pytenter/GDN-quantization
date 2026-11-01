"""Periodic PACE replay with a packed base and a bounded token ring.

Production persists no dense recurrent state. Each call reconstructs temporary
FP32 scratch, replays pending raw-v updates, computes output, and releases it.
FP state/key modes are explicit validation fixtures, never formal configurations.
"""
from dataclasses import dataclass
import math
from typing import Dict

import torch

try:
    from .pace_quantizer import PackedAffineGroups, PackedAffineState
except ImportError:
    from pace_quantizer import PackedAffineGroups, PackedAffineState


def native_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Keep BF16 arithmetic/reduction placement identical to native helper."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


@dataclass(frozen=True)
class _BufferedToken:
    key: object
    value: torch.Tensor
    beta: torch.Tensor
    log_decay: torch.Tensor


class PACEBuffer:
    def __init__(self, decode_period, int8_mask, *, prefill_period=256,
                 state_quantization=True, key_quantization=True, key_heads=16,
                 _validation_only=False):
        if decode_period not in (1, 8) or prefill_period != 256:
            raise ValueError("formal periods are decode=1/8 and prefill=256")
        if isinstance(int8_mask, torch.Tensor):
            if int8_mask.dtype != torch.bool or int8_mask.ndim != 1:
                raise ValueError("int8_mask must be a bool vector")
            int8_mask = int8_mask.detach().cpu().tolist()
        self.int8_mask = tuple(int8_mask)
        if not self.int8_mask or any(type(x) is not bool for x in self.int8_mask):
            raise ValueError("invalid head mask")
        if len(self.int8_mask) % 4 or sum(self.int8_mask) != len(self.int8_mask) // 4:
            raise ValueError("PACE-4/8 requires exactly 25% INT8 heads")
        if key_heads <= 0 or len(self.int8_mask) % key_heads:
            raise ValueError("key head sharing must be integral")
        self.decode_period = int(decode_period)
        self.prefill_period = int(prefill_period)
        self.key_heads = int(key_heads)
        self.state_quantization = bool(state_quantization)
        self.key_quantization = bool(key_quantization)
        self.fixture_only = not (self.state_quantization and self.key_quantization)
        self._validation_only = bool(_validation_only)
        self.tokens = 0
        self.quantization_count = 0
        self.writeback_count = 0
        self.prompt_length = 0
        self._prompt_length_set = False
        self.packed_state = None
        self._fp_state = None  # Used solely by explicit CPU fixture mode.
        self._shape = None
        self._slots = [None] * self.decode_period
        self.pending_tokens = 0
        self.last_writeback = None
        self.finalization_policy = None
        self.peak_prefill_buffer_nbytes = 0
        self.peak_decode_buffer_nbytes = 0

    def set_prompt_length(self, n):
        if self.tokens or self._shape is not None or type(n) is not int or n < 0:
            raise ValueError("set a nonnegative prompt length before the first step")
        self.prompt_length = n
        self._prompt_length_set = True
        self._slots = [None] * (min(n, self.prefill_period) if n else self.decode_period)

    @property
    def decode_tokens(self):
        return max(0, self.tokens - self.prompt_length)

    def _unique_keys(self, k, value_heads):
        if k.shape[1] == self.key_heads:
            return k
        if k.shape[1] != value_heads:
            raise ValueError("key head count must be unique-key or value-head count")
        repeat = value_heads // self.key_heads
        unique = k[:, ::repeat]
        if not torch.equal(k, unique.repeat_interleave(repeat, dim=1)):
            raise ValueError("already-repeated keys disagree within native head-sharing pairs")
        return unique

    def _token_keys(self, token):
        if isinstance(token.key, PackedAffineGroups):
            b, u, groups, _ = token.key.shape
            key = token.key.dequantize().reshape(b, u, groups * 32).to(torch.bfloat16)
        else:
            key = token.key
        # Reconstruct native repeated head layout BEFORE the single BF16 norm.
        key = key.repeat_interleave(self._shape[1] // self.key_heads, dim=1)
        return native_l2norm(key).to(torch.float32)

    def _base(self):
        return (self.packed_state.dequantize() if self.state_quantization
                else self._fp_state.clone())

    def _replay(self):
        state = self._base()
        for i in range(self.pending_tokens):
            token = self._slots[i]
            key = self._token_keys(token)
            state = state * token.log_decay.exp().unsqueeze(-1).unsqueeze(-1)
            delta = (token.value.to(torch.float32) - (state * key.unsqueeze(-1)).sum(dim=-2))
            delta = delta * token.beta.to(torch.float32).unsqueeze(-1)
            state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        return state

    def _commit(self, state, reason):
        if self.state_quantization:
            self.packed_state = PackedAffineState.from_tensor(state, self.int8_mask)
            self.quantization_count += 1
        else:
            self._fp_state = state.clone()
        self.writeback_count += 1
        self.last_writeback = {"tokens": self.tokens, "reason": reason,
                               "applied_pending": self.pending_tokens}
        self.pending_tokens = 0
        capacity = (min(self.prefill_period, self.prompt_length - self.tokens)
                    if self.tokens < self.prompt_length else self.decode_period)
        self._slots = [None] * capacity

    @torch.no_grad()
    def step(self, q, k, v, beta, g, *, scale=None):
        """Consume exactly one FORWARDED token; final sampled EOS is not one."""
        if not self._prompt_length_set:
            raise ValueError("set_prompt_length must be called before step")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or beta.ndim != 2 or g.ndim != 2:
            raise ValueError("expected q/k/v[B,H,D] and beta/g[B,H]")
        b, h, dv = v.shape
        dk = k.shape[-1]
        if (h != len(self.int8_mask) or dk % 32 or q.shape[0] != b or k.shape[0] != b
                or q.shape[-1] != dk or tuple(beta.shape) != (b, h) or tuple(g.shape) != (b, h)):
            raise ValueError("incompatible PACE tensor shapes")
        if q.shape[1] not in (self.key_heads, h):
            raise ValueError("query head count mismatch")
        for name, x, dtype in (("q", q, torch.bfloat16), ("k", k, torch.bfloat16),
                               ("v", v, torch.bfloat16), ("beta", beta, torch.bfloat16),
                               ("g", g, torch.float32)):
            if x.dtype != dtype or x.device != v.device or not bool(torch.isfinite(x).all().item()):
                raise ValueError(f"{name} must be finite {dtype} on the common device")
        if bool(((beta < 0) | (beta > 1)).any().item()) or bool((g > 0).any().item()):
            raise ValueError("native beta must be [0,1] and log-decay nonpositive")
        if self.fixture_only and v.device.type != "cpu" and not self._validation_only:
            raise ValueError("unquantized PACE fixture modes are CPU-only")
        current_shape = (b, h, dk, dv)
        if self._shape is None:
            self._shape = current_shape
            zeros = torch.zeros(current_shape, dtype=torch.float32, device=v.device)
            if self.state_quantization:
                self.packed_state = PackedAffineState.from_tensor(zeros, self.int8_mask)
            else:
                self._fp_state = zeros
        elif self._shape != current_shape:
            raise ValueError("batch/head/state shape cannot change within a sequence")
        unique = self._unique_keys(k, h)
        # Prefill is native BF16 input arithmetic, with state quantization only
        # at chunk boundaries. INT4 keys belong exclusively to decode buffers.
        in_prefill = self.tokens < self.prompt_length
        if self.key_quantization and not in_prefill:
            encoded = PackedAffineGroups.from_tensor(unique.reshape(b, self.key_heads, dk // 32, 32), 4)
        else:
            encoded = unique.clone()
        token = _BufferedToken(encoded, v.clone(), beta.clone(), g.clone())
        if self.pending_tokens >= len(self._slots):
            raise RuntimeError("pending token ring overflow")
        self._slots[self.pending_tokens] = token
        self.pending_tokens += 1
        key_bytes = (encoded.storage_nbytes() if isinstance(encoded, PackedAffineGroups)
                     else encoded.numel() * encoded.element_size())
        token_bytes = key_bytes + sum(t.numel() * t.element_size()
                                     for t in (token.value, token.beta, token.log_decay))
        if in_prefill:
            self.peak_prefill_buffer_nbytes = max(self.peak_prefill_buffer_nbytes,
                                                self.pending_tokens * token_bytes)
        else:
            self.peak_decode_buffer_nbytes = max(self.peak_decode_buffer_nbytes,
                                               self.pending_tokens * token_bytes)
        state = self._replay()
        query = q if q.shape[1] == h else q.repeat_interleave(h // self.key_heads, dim=1)
        query = native_l2norm(query).to(torch.float32)
        effective_scale = 1.0 / math.sqrt(dk) if scale is None else float(scale)
        if not math.isfinite(effective_scale) or effective_scale <= 0:
            raise ValueError("query scale must be positive finite")
        query = query * effective_scale
        out = (state * query.unsqueeze(-1)).sum(dim=-2).to(torch.bfloat16)
        if not bool(torch.isfinite(state).all().item()) or not bool(torch.isfinite(out).all().item()):
            raise FloatingPointError("nonfinite recurrent state/output")
        self.tokens += 1
        if self.tokens <= self.prompt_length:
            if self.tokens % self.prefill_period == 0 or self.tokens == self.prompt_length:
                self._commit(state, "prefill_end" if self.tokens == self.prompt_length else "prefill_256")
        elif self.decode_tokens % self.decode_period == 0:
            self._commit(state, "decode_period")
        # No tensor holding state is assigned to self in production; only packed
        # base, token ring and scalar counters survive the return.
        return out

    @torch.no_grad()
    def state_view_for_test(self):
        if self._shape is None:
            return None
        return self._replay()

    @torch.no_grad()
    def finalize(self, policy="preserve"):
        if policy not in ("preserve", "flush"):
            raise ValueError("tail policy must be preserve or flush")
        self.finalization_policy = policy
        if policy == "flush" and self.pending_tokens:
            self._commit(self._replay(), "final_tail_without_forward")
        return self.storage_report()

    def resident_tensors(self) -> Dict[str, torch.Tensor]:
        result = {}
        if self.packed_state is not None:
            result.update({"base." + name: t for name, t in self.packed_state.resident_tensors().items()})
        if self._fp_state is not None:
            result["CPU_FIXTURE_ONLY.dense_state"] = self._fp_state
        for i in range(self.pending_tokens):
            token = self._slots[i]
            if isinstance(token.key, PackedAffineGroups):
                result.update({f"buffer.{i}.key.{name}": t for name, t in token.key.resident_tensors().items()})
            else:
                result[f"buffer.{i}.raw_bf16_key"] = token.key
            result.update({f"buffer.{i}.value": token.value, f"buffer.{i}.beta": token.beta,
                           f"buffer.{i}.log_decay": token.log_decay})
        return result

    def tensor_inventory(self):
        return [{"name": n, "shape": list(t.shape), "dtype": str(t.dtype),
                 "nbytes": t.numel() * t.element_size()} for n, t in self.resident_tensors().items()]

    def _token_capacity_nbytes(self, prefill):
        if self._shape is None:
            return 0
        b, h, k, v = self._shape
        if self.key_quantization and not prefill:
            group_count = b * self.key_heads * (k // 32)
            key_bytes = b * self.key_heads * k // 2 + 2 * group_count + (group_count + 1) // 2
        else:
            key_bytes = 2 * b * self.key_heads * k
        return key_bytes + 2 * b * h * v + 2 * b * h + 4 * b * h

    def storage_report(self):
        inventory = self.tensor_inventory()
        base_bytes = sum(t["nbytes"] for t in inventory if t["name"].startswith("base."))
        buffer_bytes = sum(t["nbytes"] for t in inventory if t["name"].startswith("buffer."))
        state_values = math.prod(self._shape) if self._shape else 0
        total_bytes = sum(t["nbytes"] for t in inventory)
        return {"tokens": self.tokens, "prompt_length": self.prompt_length,
                "decode_tokens": self.decode_tokens, "pending_tokens": self.pending_tokens,
                "ring_capacity_tokens": len(self._slots), "decode_period": self.decode_period,
                "prefill_period": self.prefill_period, "quantization_count": self.quantization_count,
                "writeback_count": self.writeback_count, "last_writeback": self.last_writeback,
                "finalization_policy": self.finalization_policy, "fixture_only": self.fixture_only,
                "logical_state_shape": list(self._shape) if self._shape else None,
                "base_nbytes": base_bytes, "buffer_nbytes": buffer_bytes,
                "total_resident_tensor_nbytes": total_bytes,
                "actual_resident_tensor_bpv": total_bytes * 8 / state_values if state_values else None,
                "base_bpv": base_bytes * 8 / state_values if state_values else None,
                "beta_storage_included": True, "dense_state_persistent": self._fp_state is not None,
                "peak_prefill_buffer_nbytes": self.peak_prefill_buffer_nbytes,
                "peak_decode_buffer_nbytes": self.peak_decode_buffer_nbytes,
                "prefill_buffer_capacity_nbytes": min(self.prompt_length, self.prefill_period) *
                                                   self._token_capacity_nbytes(True),
                "decode_buffer_capacity_nbytes": self.decode_period * self._token_capacity_nbytes(False),
                "resident_accounting_scope": "live_tensor_payload_only_excludes_allocator_reserved_and_transient_scratch",
                "inventory": inventory}
