# Canonical L6/L7 training environment

This file records the environment used by the frozen L6/L7 trainer. It is distinct from the later formal-evaluation environment.

## Training environment

- Python 3.11.16
- PyTorch 2.7.1+cu128
- CUDA 12.8; cuDNN 90701
- Triton 3.3.1
- FLA 0.5.2
- Transformers 4.57.6
- Accelerate 1.14.0
- huggingface_hub 0.36.2
- safetensors 0.8.0
- GCC 13.3.0
- NVIDIA driver 580.173.02

The full `pip freeze` captured from `/data/zypan/envs/ling-kda` is in `environment_lock.txt`. No package was upgraded or installed to produce it.

## Formal evaluation environment (separate)

- Python 3.11.16
- PyTorch 2.9.1+cu128
- Triton 3.5.1
- FLA 0.5.2
- SGLang 0.5.19
- SGLang source commit `0bcd822377da7b5718e674eaf9c870d349424dd1`

The training scripts should be run in an environment compatible with the training lock, not silently substituted with the evaluation environment.

`ENVIRONMENT_PROVENANCE_GATE = PASS`
