# Protocol

This document separates current formal protocols from historical mechanism experiments. Protocols with different context length, deterministic settings, cache behavior, sampling, model revision, quantization axis, state dtype, or scorer version must not be merged automatically.

## Models and recurrent states

- Qwen3.5-9B uses Gated DeltaNet (GDN).
- Ling-3.0-tiny uses Kimi Delta Attention (KDA).
- Quantization targets recurrent state storage, not model weights.
- `FP_STATE` means state quantization is disabled. It does not by itself imply that all model compute or state storage is FP32; actual storage and accumulation dtypes require runtime capture.

## Current INT8 semantics

The current Ling `INT8_R128` implementation is audited as:

```text
group axis = V
group size = 128
scale shape = [B,H,K,1]
scale = amax / 127
rounding = ties-to-even
clamp = [-127,127]
```

These values document the observed canonical implementation; this consolidation does not modify numerical behavior.

## Current rotation semantics

Qwen/GDN uses a Key-side Hadamard line with `INT8_C128`.

Ling/KDA uses a Value-side Hadamard line with `INT8_R128` under `CORRECTED_PREFILL_ENDPOINT_V2`:

1. The KDA kernel returns recurrent state already in rotated Value coordinates.
2. That state is written directly to the recurrent cache.
3. The first decode step consumes the same rotated-basis cached state.
4. No extra H or H-transpose is applied at the prefill endpoint.
5. The KDA core/readout output is mapped back to native Value coordinates before RMSNorm, learned scale/dynamic gate, head merge, and `o_proj`.

Outputs produced through the historical redundant endpoint rotation path are `INVALID_OLD_ROTATION_SEMANTICS` for formal mixing.

## AIME26 81,920 final protocol

```text
questions = 30
seeds = 2
samples_per_condition = 60
max_new_tokens = 81920
scorer = AIME26_STRICT_V4_CANDIDATE
```

Formal accuracy is correct / 60. `Abstain` is an extraction-status subset of incorrect samples, not an additional denominator category.

## Ling 256K protocol boundary

The running helper protocol is a separate length-sensitivity run:

```text
max_new_tokens = 262144
context_length = 262144
model_native_context_length = 131072
rope_scaling = YaRN factor 2.0
seed = 1
condition = INT8_R128 + Value-Hadamard
execution = two independent TP=1 instances
deterministic inference = enabled
radix cache = disabled
mamba radix cache = disabled
CUDA graph = disabled
```

Because the YaRN/context intervention and execution layout differ from the 81,920 protocol, this run must not be represented as a simple continuation of the 81,920 baseline without an explicit matched-baseline audit.

## Historical mechanism protocol

Earlier Qwen mechanism experiments commonly used FP32 prefill followed by teacher-forced quantized/intervened recurrent continuation on a small prompt set. Those results remain useful mechanism evidence but are not interchangeable with AIME26 free generation.
