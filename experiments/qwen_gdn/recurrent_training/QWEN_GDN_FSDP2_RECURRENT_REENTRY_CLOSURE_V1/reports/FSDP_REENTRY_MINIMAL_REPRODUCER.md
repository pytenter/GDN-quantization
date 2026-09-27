# Minimal real-path re-entry diagnostics

All tests used two ranks on physical GPUs 0/1, exact Qwen3.5 checkpoint tensors, one fresh process per condition and one backward after N invocations of the same FSDP2-wrapped module. Inputs were synthetic, not AIME. `analysis/minimal_reentry_matrix.json` links all 60 rank-level records.

The preregistered real GDN, no-cache factorial passed N1/N2/N3 with either reshard setting. N4 completed backward but failed the diagnostic nonzero input-gradient gate under the synthetic squared-output scalar loss on **both** settings; this is not the parent's zero-storage error. An exploratory raw `DynamicCache` N2 failed a different autograd in-place version check and is not the frozen V1 differentiable writeback. With the exact V1 differentiable cache, N1/N2/N4 passed. Adding canonical C128 QDQ and Hadamard theta=0 Key-side rotation also passed N1/N2/N4. Real full decoder layer 0 passed N1/N2/N4. Nested FSDP2 stacks of two and three real GDN decoder blocks passed N1/N4. N4 reshard=True and reshard=False both passed in every canonical cache/decoder/stack comparison.

No reduced condition reproduced `setStorage ... storage size 0`; the minimum parent-style failing re-entry count is **not found**. This negative result cannot establish that the full 9B/H4 parent failure was not an FSDP lifecycle issue.
