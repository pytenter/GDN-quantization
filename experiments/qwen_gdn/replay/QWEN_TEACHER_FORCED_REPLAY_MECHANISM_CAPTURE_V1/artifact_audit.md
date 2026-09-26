# Artifact audit — QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1

## Existing artifacts reused

- 60 FP_STATE AIME26 trajectories with complete token IDs: 1,854,451 teacher-forced continuation tokens.
- Paired correctness labels for FP_STATE, INT8_C128, and INT8_C128 + Key-Hadamard: A=18, B=23, C=11, D=8.
- Canonical Qwen INT8-C128 implementation and key-side normalized H128 implementation from the existing AIME runner.
- Frozen Qwen M5 operator semantics from the existing propagation experiments.
- Existing model and tokenizer at `/data/zypan/modelscope_models/Qwen3.5-9B`.

## Initially missing

- No AIME trajectory had saved per-step logits, recurrent states, state error, M5, persistence, or decision-margin sensitivity.
- Dense Oracle artifacts exist but were not used: this task requires only FP, Native INT8, and Key-Hadamard and forbids retraining.

## New capture

- Four RTX3090 shards completed on the required server.
- Full states were reduced online; no TB-scale state dump was created.
- Full-attention KV caches were losslessly offloaded to CPU to fit the longest 81,920-token trajectories; GDN recurrent states and all mechanism calculations remained on GPU.
- Per-layer/per-token M0 and M5 were computed for all 24 GDN layers. The CSV retains token-level cross-layer summaries; the JSON retains trajectory-level per-layer summaries.
- Token-condition rows: 3,708,902.
- No sampling, generation, scorer invocation, gold-conditioned metric construction, training, or parameter update occurred.

## Numerical overflow disclosure

- 2 trajectory-condition streams contain non-finite values, all from `aime26_28`, seed 2, Group D (FP wrong), after very long teacher-forced prefixes.
- The raw CSV preserves `inf`/`nan`; `token_level_metrics.json` records exact counts and first affected timesteps.
- Group D aggregate values are finite-prefix summaries. The primary failure population (A/B/C) and rescue comparison (B/C) contain no non-finite values and are unaffected.

## Integrity

- Capture script SHA256: `3e4eb01be6c970ca404572d7b1bd904443be5b629545039c598ff17da3c7b6bf`.
- Native and FP prefill rows were numerically identical in every trajectory; Hadamard finite-precision differences are recorded per trajectory.
- Output group counts reproduce the frozen benchmark labels exactly.
