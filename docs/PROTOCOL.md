# Evaluation protocols

## Ling/KDA current canonical protocol: 256K long horizon

The current canonical Ling-3.0-tiny/KDA evaluation uses:

- `context_length = 262144`
- `server_max_total_tokens = 262144`
- `safety_margin = 512`
- `max_new_tokens = 262144 - prompt_tokens - 512`
- seeds `[1, 2]`
- sampling enabled with temperature `1.0`, top-p `0.95`, top-k `20`, and repetition penalty `1.0`
- thinking enabled

The full frozen runtime and decoding specification is in `docs/LING_CANONICAL_256K_PROTOCOL.md`. The machine-readable source of truth is `experiments/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/configs/frozen_eval_config.json` on branch `exp/ling-l6-l7-canonical-60-confirmatory-v1`, frozen at commit `69b23bd999dc23a9c852497b98df0f56abdbdff0`.

The 81,920-token Ling protocol below is historical reference only. Its outputs must not be merged with 256K dynamic-budget outputs.

## Historical formal AIME26 at 81,920 tokens

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

Qwen uses the manual Hugging Face runtime. SGLang-only flags must not be added to Qwen. Ling uses the archived SGLang/KDA integration. For Ling, this protocol is a historical baseline only. Historical 65,536-token smoke runs are legacy and must not be mixed into formal results.

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

## 256K evidence boundary

The completed seed-1 256K offline length-sensitivity analysis used a fixed `max_new_tokens=262144`, SGLang/Triton, and YaRN factor 2 with original maximum positions 131072. Its compact frozen scoring and verification evidence is archived under `experiments/ling_kda/long_horizon/LING_256K_LONG_HORIZON_V1/`. It is precursor evidence, not interchangeable with the current 60-sample 256K canonical evaluation, whose generation budget is dynamic and reserves a 512-token safety margin.

## Artifact policy

Only compact code, summaries, reports, manifests, and tests are tracked. Checkpoints, model weights, datasets, caches, logs, raw generation JSONL, token/layer/head traces, and large per-horizon tables remain on their provenance hosts and are represented by manifests.
