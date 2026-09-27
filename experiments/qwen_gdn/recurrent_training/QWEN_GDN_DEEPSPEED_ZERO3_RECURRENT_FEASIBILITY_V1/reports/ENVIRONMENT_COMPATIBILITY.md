# DeepSpeed isolated environment compatibility

Decision frozen before installation: Python 3.10.18; reuse read-only canonical Torch 2.5.1+cu121, CUDA runtime 12.1, NCCL 2.21.5, Transformers 5.16.0.dev0 and Triton 3.1.0 through a new `--system-site-packages` venv. Install DeepSpeed 0.16.4 and only its missing runtime dependencies inside that venv. This is a lightweight overlay, not a fully copied environment; `sys.prefix` and `pip` must resolve to the venv before installation. Canonical site-packages must remain unchanged.

DeepSpeed 0.16.4 metadata lists Python 3.10 support and unpinned `torch` as a dependency; that is *not* a compatibility guarantee. The version is a contemporary patch release with a BF16 optimizer correction. Import and two-rank NCCL smoke tests are mandatory before model work. NVIDIA driver is 535.183.01; four RTX 3090s are currently idle. FLA package version is unconfirmed and must not be fabricated. The Qwen custom source, V1 quantizer and rotation modules must import successfully in the isolated runtime.

Primary sources: [DeepSpeed 0.16.4 release](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.16.4), [PyPI 0.16.4 metadata](https://pypi.org/pypi/deepspeed/0.16.4/json), [ZeRO documentation](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero.md).

Resource note: `/data` had about 20 GB free before venv creation. The venv must reuse canonical large packages read-only; no Torch/CUDA upgrade, no build cache or large checkpoint in Git.
