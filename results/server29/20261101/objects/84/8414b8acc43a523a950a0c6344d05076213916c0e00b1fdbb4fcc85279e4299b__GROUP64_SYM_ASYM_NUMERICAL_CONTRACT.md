# Group64 numerical contract

One original Group32 encode/decode graph, group_size parameter32/64. Formal4 conditions only G64-0 native/B1, G64-1 native/B2, G64-2 H128/B1, G64-3 H128/B2. State actual[B,32,128Key,128Value] FP32; reshape[B,H,2,64,128]; no cross Value/Head/Layer mixing.

Sym unchanged: amax(abs) on Key, floor1e-12, original CUDA scalar /127 operation, RNE, clip[-127,127], int8 plus negative-zero flags, FP32 decode. Asym unchanged include-zero extrema/canonical+0, Tensor/Tensor /255, RNE(-a/s), clip zp UINT8; clip(round(x/s)+zp), UINT8 codes; int16 subtraction then FP32 multiplication and canonical+0. Independent NumPy has explicitly labelled CPU vs CUDA legacy emulation. Group32 short regression never AIME regeneration.

H128 matrix uses original tensor operations and normalization, cached once per dtype/device/process. BF16 Q/K original32-head normalization before FP32 H; original GDN recurrence/gate/beta/value unchanged. Native does not transform Q/K. No PBR, dense OSCAR, resident16-head optimization, whole-state rotate eachtoken, FP continuation shadow, changed sampler or scorer.

Every actual forward token0 onward: all24 GDN updated states encoded/decode then immediate copy_ writeback; next kernel consumes this. 512canary checks complete actual tensors/bytes before every next call and actual post-writeback, all24layers. Fresh R0/R1 compare complete logits/cache/integer fields at frozen0/31/127/511 checkpoints, not an all-step fresh-byte trajectory claim. Free64 checks complete next-consumption plus all sampledtokens and initial/final RNG/termination/prompt IDs. References never recurrent inputs. Formal observes first127 next calls and every2048, same old sparse roster; everyforward still actualQDQ. Final sampledtoken not forwarded: p+n-1 forwards,24writebacks each.

FP32 decoded carrier48MiB remains resident; codes/scales/zp are temporary, not physical packedINT8 persistent storage. No arbitrary FP2e-6/4H/MSE/accuracy gate. Mathematical/codec/QDQ/nonfinite/actualidentity/resource error stops; no automatic GPU retry or frozen-source hotpatch. Synthetic82293 slots are memory feasibility only.
