# Backward source/determinism audit

Five independent single-GPU runs had bitwise-identical forward/QDQ/teacher targets/loss but non-identical raw rotation gradients. The first divergent trainable parameter is `bank.layer(0).theta`; 21 of 24 GDN layers varied in at least one pair.

A diagnostic runtime enabled PyTorch deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, disabled TF32, and selected deterministic cuDNN. It failed during teacher-target forward at `g.cumsum(dim=-1)` (`cumsum_cuda_kernel`) in Qwen3.5's gated-delta fallback, before backward. No operator was bypassed/replaced and these settings were not adopted for formal training. This flags a nondeterministic operator but does not prove it caused the observed gradient variation; backward source classification remains UNRESOLVED. The QDQ STE is a detach-based identity gradient, not a custom autograd Function. See analysis JSON for source/runtime hashes and other candidates.
