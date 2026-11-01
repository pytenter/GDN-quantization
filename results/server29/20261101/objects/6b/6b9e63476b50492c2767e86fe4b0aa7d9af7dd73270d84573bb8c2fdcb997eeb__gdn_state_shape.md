# Qwen3.5 GDN/KDA Recurrent State Shape Report

## Static Findings
- GDN/KDA layers are config entries with `layer_types[i] == "linear_attention"`.
- GDN/KDA layer count: `24`
- Full attention layer count: `8`
- Linear key heads: `16`
- Linear value heads: `32`
- Linear key head dim: `128`
- Linear value head dim: `128`
- Linear convolution kernel dim: `4`

## Expected Recurrent State Tensor
- Expected semantic form: state maps accumulated key features to value features for each linear-attention state group.
- Expected shape family: `(batch, state_heads, key_dim, value_dim)`.
- From config, the most likely shape is `(batch, 16, 128, 128)` or an implementation-specific grouped variant derived from `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_key_head_dim=128`, and `linear_value_head_dim=128`.
- Exact runtime shape must be confirmed by `quant_state.py --mode inspect-state` once PyTorch/Transformers are installed.

## State Lifecycle
- Generation: recurrent state is created by the linear attention/GDN cache path when `use_cache=True` and the first GDN layer needs a recurrent cache entry.
- Prefill initialization: for the first prompt chunk, recurrent state starts empty/zero or absent in the cache, then the full prompt updates it token by token or through a fused parallel/chunk scan kernel.
- Update location: inside the GDN/KDA linear attention forward path, where the recurrence has the form `S_t = alpha_t * S_{t-1} + DeltaS_t`; `DeltaS_t` is produced from current-token key/value features and gates.
- Decode persistence: after prefill, the updated recurrent state is stored in the model cache, commonly exposed as `recurrent_states` in a dynamic cache object; each decode step reads `S_{t-1}`, applies the update, and writes new `S_t` back to cache.
- Attention layers use KV cache, while GDN/KDA layers use recurrent state cache.

## Gate/Activation Tensors To Trace
- q: linear attention query projection output.
- k: linear attention key projection output.
- v: linear attention value projection output.
- gate: output or recurrent gate when present.
- alpha: recurrent decay/retention gate in `S_t = alpha_t * S_{t-1} + DeltaS_t`.
- beta: update/input gate when present in the implementation.

## Limitation
- No local `modeling_qwen3_5.py` or installed Transformers package was available at inspection time, so exact function names and tensor rank are deferred to runtime hooks.
