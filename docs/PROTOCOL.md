# Canonical protocol

## Formal AIME26 at 81,920 tokens

Protocol identifier: `GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2`.

- `max_new_tokens = 81920`
- seeds `[1, 2]`
- 30 AIME26 problems, giving 60 samples per condition
- sampling enabled
- temperature `1.0`
- top-p `0.95`
- top-k `20`
- min-p `0`
- presence penalty `1.5`
- repetition penalty `1.0`
- thinking enabled

Qwen uses the manual Hugging Face runtime. SGLang-only flags must not be added to Qwen. Ling uses the archived SGLang/KDA integration. Historical 65,536-token smoke runs are legacy and must not be mixed into formal results.

## Qwen/GDN state quantization

- State shape: `[B,H,K,V]`.
- INT8_C128 groups on K / `dim=-2`, group size 128.
- Scale shape: `[B,H,1,V]`.
- Scale source: `state.detach().float()`.
- Scale: `amax(abs(state), dim=-2) / 127`, epsilon floor `1e-12`.
- Rounding: `torch.round` (ties-to-even), clamp `[-127,127]`.
- Dequantization and stored recurrent state: FP32.
- Prefill is not quantized. During decode: kernel output -> C128 QDQ -> stored state -> next decode.

The canonical formal runner SHA256 is `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`.

## Fixed rotations

Qwen Key-Hadamard applies normalized H128 after q/k normalization and before the GDN core in both prefill and recurrent decode. q and the recurrent state use the same Key-rotated basis; Value is not rotated.

Ling Value-Hadamard uses `CORRECTED_PREFILL_ENDPOINT_V2` semantics. The KDA kernel already returns recurrent state in rotated Value coordinates. That state is written directly to recurrent cache and consumed unchanged by first decode. Only the KDA core/readout output is mapped back to native Value coordinates before RMSNorm, learned scale, dynamic gate, merge, and `o_proj`.

`REDUNDANT_PREFILL_ENDPOINT_ROTATION = NO`.

## Scoring and abstain

The frozen scorer is documented in `docs/SCORER_PROTOCOL.md`. Gold is not used for candidate selection. `Abstain` is a subset of `Incorrect`, so the only valid accounting identity is `Correct + Incorrect = N`.

## 256K follow-up

The completed 256K offline length-sensitivity analysis uses seed 1, `max_new_tokens=262144`, SGLang/Triton, and YaRN factor 2 with original maximum positions 131072. Its compact frozen scoring and verification evidence is archived under `experiments/ling_kda/long_horizon/LING_256K_LONG_HORIZON_V1/`. It remains supporting length-sensitivity evidence and does not replace the 81,920 two-seed frozen formal protocol.

## Artifact policy

Only compact code, summaries, reports, manifests, and tests are tracked. Checkpoints, model weights, datasets, caches, logs, raw generation JSONL, token/layer/head traces, and large per-horizon tables remain on their provenance hosts and are represented by manifests.
