# QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1

## Verdict

```text
{
  "QWEN_C5_MEMORY_LIFETIME_AUDIT": "PARTIAL",
  "FINAL_VERDICT": "UNRESOLVED",
  "MEMORY_BASELINE_GROWTH": "NO",
  "PRIMARY_RETENTION_SOURCE": "UNRESOLVED",
  "MEMORY_LIFETIME_STABILITY_GATE": "FAIL",
  "VALIDATION_MEMORY_RETURN_GATE": "PASS",
  "LOSS_GRAPH_RETENTION_GATE": "PASS",
  "QDQ_AUTOGRAD_LIFETIME_GATE": "FAIL",
  "RECURRENT_CACHE_LIFETIME_GATE": "PASS",
  "HOOK_LIFETIME_GATE": "PASS",
  "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
  "C5_TRAINING_STATUS": "BLOCKED_MEMORY_FEASIBILITY",
  "C5_formal_AIME": "NOT_STARTED",
  "C6_formal_AIME": "NOT_STARTED"
}
```

## Evidence summary

- Completed diagnostic optimizer updates: 11/30.
- Diagnostic run status: OOM.
- End-of-update baseline samples: 11.
- Post-warmup allocated-memory slope: 0.000 bytes/update (R²=1.000000).
- Validation checkpoints observed: 1.
- Recurrent cache graph-bearing tensors after detach: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
- OOM/failure: `{"document_id": "wikitext2raw-train-036", "error": "CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 23.69 GiB of which 3.69 MiB is free. Including non-PyTorch memory, this process has 23.68 GiB memory in use. Of the allocated memory 23.04 GiB is allocated by PyTorch, and 301.75 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)", "exposure": 96, "memory": {"allocated_bytes": 24738268160, "exposure": 96, "free_bytes": 3866624, "max_allocated_bytes": 24742462464, "max_reserved_bytes": 25056772096, "reserved_bytes": 25056772096, "stage": "OOM", "total_bytes": 25438126080, "update": 12}, "target_positions": [146, 154, 409, 630, 729, 730, 733, 814], "token_count": 1024, "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1/run_memory_lifetime_audit.py\", line 474, in memory_audit\n    student_cache = BASE.forward_student(model, patch, token_id, student_cache, capture)\n  File \"/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py\", line 371, in forward_student\n    output = model(input_ids=token, past_key_values=cache, use_cache=True)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1736, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1747, in _call_impl\n    return forward_call(*args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/utils/generic.py\", line 939, in wrapper\n    output = func(self, *args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py\", line 1629, in forward\n    outputs: BaseModelOutputWithPast = self.model(\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1736, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1747, in _call_impl\n    return forward_call(*args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/utils/generic.py\", line 1068, in wrapper\n    output = func(self, *args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/utils/output_capturing.py\", line 262, in wrapper\n    outputs = func(self, *args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py\", line 1204, in forward\n    hidden_states = decoder_layer(\n  File \"/data/zypan/transformers-qwen35/src/transformers/modeling_layers.py\", line 114, in __call__\n    return super().__call__(*args, **kwargs)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1736, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1747, in _call_impl\n    return forward_call(*args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py\", line 769, in forward\n    hidden_states = self.linear_attn(\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1736, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1844, in _call_impl\n    return inner()\n  File \"/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/nn/modules/module.py\", line 1790, in inner\n    result = forward_call(*args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/integrations/accelerate.py\", line 949, in wrapped\n    output = forward_func(self, *args, **kwargs)\n  File \"/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py\", line 507, in forward\n    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(\n  File \"/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py\", line 262, in wrapped\n    if not bool(torch.isfinite(quant.dequant).all().detach().cpu()):\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 23.69 GiB of which 3.69 MiB is free. Including non-PyTorch memory, this process has 23.68 GiB memory in use. Of the allocated memory 23.04 GiB is allocated by PyTorch, and 301.75 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)\n", "update": 12}`.

## Required questions

1. Baseline growth: **NO**.
2. Validation as a secondary factor: **validation returned to baseline**.
3. Recurrent cache old-graph retention: **not observed after detach**.
4. QDQ/STE cross-update retention: **observed**.
5. Python container/loss/history retention: **not observed for the probed loss graph**.
6. Patch parity: **NOT_RUN; no lifetime patch was applied in this diagnostic run.**
7. Sustained 20–30 update stability: **FAIL**.
8. H=32 sustained feasibility on 24GB: **FAIL**.
9. C5 restart allowed: **NO under this audit stop point.**
10. Horizon amendment needed: **not determined by this run**.

C5 recurrent-aware training passed semantic/numerical gates but has not yet passed sustained 24GB memory feasibility.
