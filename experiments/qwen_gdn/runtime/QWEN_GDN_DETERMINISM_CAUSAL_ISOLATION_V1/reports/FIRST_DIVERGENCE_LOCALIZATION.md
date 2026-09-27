# First divergence

First GDN layer: **0**. Previous node `gdn.prefix_input` is bitwise exact; first non-identical node `gdn.prefix_output`. Real captured input is FP32, shape `[1, 32, 1, 64]`, contiguous stride `[2048, 64, 64, 1]`. The only audited copied-function source delta is replacing `g.cumsum(dim=-1)` by fixed left-to-right accumulation.

Prefix output max abs difference: 1.525878906e-05; relative L2: 1.125205311e-07; different elements: 1232. Full logits max abs: 0.203125; relative L2: 0.01116291292. This localizes the numerical path change; it does not prove mathematical incorrectness.
