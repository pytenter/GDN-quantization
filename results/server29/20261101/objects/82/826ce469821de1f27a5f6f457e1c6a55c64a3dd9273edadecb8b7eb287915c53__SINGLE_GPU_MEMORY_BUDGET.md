# Historical single-GPU memory budget

The frozen C5 memory-lifetime audit concluded `NO_LEAK_H32_TOO_LARGE`: 11 H32 updates completed, then update 12 OOMed. End-of-update allocated memory was byte-identical at 17,957,448,192 bytes across the 11 completed updates. The OOM was therefore a peak-capacity problem, not observed cross-update retention.

The four checkpoint safetensors contain 9,653,104,368 parameters and 19,306,216,416 raw bytes (17.98 GiB); this includes all checkpoint contents and is **not** asserted to equal the CUDA-loaded text-model static bytes. The first audited update started at 17,911,608,320 allocated bytes and reached 24,161,069,056 after loss construction. The failed update measured 23,592.3 MiB allocated, 23,896.0 MiB reserved, and 3.7 MiB free before a further allocation failed. H1/H4/H8/H16 single-GPU peaks were not established in this audit and are not invented here.

Halving all checkpoint parameter bytes would save at most about 8.99 GiB before all-gather and workspace costs. This gives H32 a *plausible* opportunity under 24 GB, but not a proof: loaded text-model bytes, FSDP per-block all-gather peaks and activation slope remain to be measured. `STATIC_VS_ACTIVATION_CONTRIBUTION=PARTIALLY_IDENTIFIED`. See `analysis/single_gpu_memory_budget.json` for raw evidence and source hashes.
