# Parent H4 failure provenance

Parent `ad5007de643f55d92fccdb37562318f22a2e734b` is frozen. Both rank error JSON files show `setStorage ... [8192,1,1,4] ... storage of size 0` at `c5_loss.backward()`, not OOM. Their full saved Python tracebacks stop at the autograd engine; no failing model or FSDP frame is exposed. See `configs/parent_failure_manifest.json` for exact hashes and environment.

The 32,768-element shape matches layer-0 GDN depthwise `conv1d.weight` inventory, and the installed Qwen source uses `weight.squeeze(1)` and later `weight.unsqueeze(1)` in the causal convolution path. This is a **candidate only**. `FAILED_STORAGE_OWNER=UNKNOWN` until a hook/trace identifies the actual storage owner.
