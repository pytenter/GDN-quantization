# LING_DETERMINISTIC_16DOC_REPLICATION_V2 runtime manifest

- Host: `nlpg-SYS-4029GP-TRT`
- System GPUs: `8 x NVIDIA GeForce RTX 3090`
- Selected physical GPUs: `2,3` only (`CUDA_VISIBLE_DEVICES=2,3`)
- Actual repository: `exp/ling-deterministic-16doc-replication-v2-3090` at `506071224db4d091c398a0fc96b0c58c67b296f4` (dirty working tree preserved)
- Recorded closure commit `d757a35` present: `False`
- Runtime manifest hash: `414a130b9822d37521def5ca988b19532f74e3a3eadfe0eada712aa6463b63cf`
- Kernel configuration hash: `a9d5ffa5e13716f19a659facec84bbb0c655038aa38e0c2fef86595334f69023`
- Generation configuration hash: `332ae114a76ac46a7d3480c20e4e121a600cdceb208b3609478f2bbcda77310e`
- Panel hash: `80133307bb1fecba49ddc8929368a532e083e9c8314a9b9d3b4b05adc41ceb40`
- Closure verdict: `REPEATABILITY_CLOSED`
- AUC noise after closure: p95 `0.0`, max `0.0`

The recorded closure commit is absent from the server object database. The historical
working tree was not reset or modified. This replication is instead pinned to exact
audited code, kernel, panel, checkpoint, model-index, and closure-artifact hashes.
