#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU PORT [METHOD]" >&2
  exit 2
fi
PHYSICAL_GPU="$1"
PORT="$2"
METHOD="${3:-fp_state}"
case "$PHYSICAL_GPU" in 4) ;; *) echo "GPU must be physical 2 or 3" >&2; exit 2;; esac
case "$METHOD" in fp_state|int8_r128|int8_r128_value_h) ;; *) echo "invalid method" >&2; exit 2;; esac

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
RUNTIME_ROOT=/data/zypan/runtime_compat/ling256_exact
ENV_DIR="$RUNTIME_ROOT/.conda/envs/ling-sglang-aime26"
CUDA_TOOLKIT="$RUNTIME_ROOT/tools/cuda-nvcc-12.8.93"
COMPILER_ENV="$RUNTIME_ROOT/.conda/envs/spatialcoc"
SGLANG_SRC=/data/zypan/src/sglang-v0.5.19-ling256/python
MODEL=/data/zypan/models/Ling-3.0-tiny
NVIDIA_PY="$ENV_DIR/lib/python3.11/site-packages/nvidia"
YARN_JSON='{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":131072}}'

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PATH="$CUDA_TOOLKIT/bin:$ENV_DIR/bin:$PATH"
export CPLUS_INCLUDE_PATH="$NVIDIA_PY/cuda_nvcc/include:$NVIDIA_PY/cuda_runtime/include:$NVIDIA_PY/cuda_cccl/include:$NVIDIA_PY/cublas/include:$NVIDIA_PY/curand/include"
export LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$CUDA_TOOLKIT"
export NVCC_CCBIN="$COMPILER_ENV/bin/x86_64-conda-linux-gnu-g++"
export CC="$COMPILER_ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$NVCC_CCBIN"
export PYTHONPATH="$SGLANG_SRC:$REPO/experiments/aime26"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
export TRITON_CACHE_DIR="/data/zypan/runtime_compat/ling256_exact/cache/tp1_gpu${PHYSICAL_GPU}/triton"
export TORCHINDUCTOR_CACHE_DIR="/data/zypan/runtime_compat/ling256_exact/cache/tp1_gpu${PHYSICAL_GPU}/torchinductor"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export AIME26_KDA_MODE="$METHOD"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

"$ENV_DIR/bin/python" - <<PY
import os, torch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "$PHYSICAL_GPU"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print("GPU_ISOLATION_GATE=PASS physical=$PHYSICAL_GPU visible=1")
PY

exec "$ENV_DIR/bin/python" -u "$REPO/experiments/aime26/launch_ling_sglang.py" \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --tp-size 1 --dtype bfloat16 \
  --context-length 262144 --max-total-tokens 262144 \
  --json-model-override-args "$YARN_JSON" \
  --mem-fraction-static 0.80 --sampling-backend pytorch --attention-backend triton \
  --linear-attn-backend triton --moe-runner-backend triton --disable-cuda-graph \
  --disable-radix-cache --trust-remote-code --max-running-requests 1 \
  --enable-deterministic-inference
