#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODE GPU PORT TAG" >&2
  exit 2
fi

MODE="$1"
GPU="$2"
PORT="$3"
TAG="$4"
REPO="/data01/user2/worktrees/aime26-sglang-rotation-v1"
ENV_DIR="/data01/user2/.conda/envs/ling-sglang-aime26"
CUDA_TOOLKIT="/data01/user2/tools/cuda-nvcc-12.8.93"
NVIDIA_PY="$ENV_DIR/lib/python3.11/site-packages/nvidia"

export PATH="$CUDA_TOOLKIT/bin:$ENV_DIR/bin:$PATH"
export CPLUS_INCLUDE_PATH="$NVIDIA_PY/cuda_nvcc/include:$NVIDIA_PY/cuda_runtime/include:$NVIDIA_PY/cuda_cccl/include:$NVIDIA_PY/cublas/include:$NVIDIA_PY/curand/include"
export LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LD_LIBRARY_PATH:-}"
export NVCC_CCBIN="/data01/user2/.conda/envs/spatialcoc/bin/x86_64-conda-linux-gnu-g++"
export CC="/data01/user2/.conda/envs/spatialcoc/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$NVCC_CCBIN"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="$CUDA_TOOLKIT"
export PYTHONPATH="/data01/user2/src/sglang-v0.5.19/python:$REPO/experiments/aime26"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export AIME26_KDA_MODE="$MODE"
export AIME26_KDA_AUDIT_JSONL="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/parity/${TAG}_audit.jsonl"
export AIME26_KDA_FORCE_JSON="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/parity/${TAG}_force.json"
export AIME26_KDA_GATE_C_DUMP_DIR="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/parity/${TAG}_gate_c_dumps"

exec "$ENV_DIR/bin/python" -u "$REPO/experiments/aime26/launch_ling_sglang.py" \
  --model-path /data01/user2/models/Ling-3.0-tiny \
  --host 127.0.0.1 --port "$PORT" --tp-size 1 --dtype bfloat16 \
  --mem-fraction-static 0.80 --sampling-backend pytorch --attention-backend triton \
  --linear-attn-backend triton --moe-runner-backend triton --disable-cuda-graph \
  --disable-radix-cache \
  --trust-remote-code --max-running-requests 1 --enable-deterministic-inference
