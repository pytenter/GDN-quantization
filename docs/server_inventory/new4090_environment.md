# new4090 environment audit

Audited 2026-09-21 without changing the environment or active processes.

| Component | Actual value |
|---|---|
| Hostname | `nlpg-SYS-4029GP-TRT` |
| GPU | 8 × NVIDIA GeForce RTX 3090 |
| NVIDIA driver | 580.173.02 |
| Python | 3.11.16 |
| PyTorch | 2.9.1+cu128 |
| PyTorch CUDA build | 12.8 |
| Triton | 3.5.1 |
| FLA | `fla-core` 0.5.2 |
| Transformers | 5.12.1 |
| SGLang | 0.5.19 |
| flash-attn | NOT_INSTALLED |
| nvcc | unavailable to the audit shell |

The task description's possible 2×RTX 4090 / PyTorch 2.7.1 environment does not match this host. No package was installed, removed, or upgraded.
