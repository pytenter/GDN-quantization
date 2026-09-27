# FSDP2 diagnostic design

Use the installed PyTorch 2.5.1 composable `fully_shard` API with two NCCL ranks and one idle RTX3090 per rank. Load the canonical BF16 Qwen backbone without a dual-GPU `device_map`, freeze every base parameter, then shard bottom-up by the 32 decoder blocks, embedding, final norm and separate `lm_head`, and finally the root for any residual parameter. Keep the ~195k-parameter rotation bank unsharded and trainable. `reshard_after_forward=True`; activation checkpointing remains off.

FSDP shards **parameters only**. It does not split GDN heads, recurrent state `[1,32,128,128]`, QDQ groups or layer computations. The original Qwen source, `torch.cumsum`, C128 Key-axis QDQ and recurrent writeback must remain unchanged. Load-only sharding and trainable-inventory checks precede all forward/backward tests. Failure at a semantic gate stops before H32. DeepSpeed is not a parallel first choice and is not installed in the current environment.
