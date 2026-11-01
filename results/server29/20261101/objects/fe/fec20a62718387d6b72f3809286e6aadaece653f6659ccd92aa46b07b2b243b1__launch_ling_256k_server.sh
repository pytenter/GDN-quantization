#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODE PORT" >&2
  exit 2
fi

MODE="$1"
PORT="$2"
case "$MODE" in fp_state|int8_r128|int8_r128_value_h) ;; *) echo "invalid mode" >&2; exit 2;; esac

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
RUNTIME_ROOT=/data/zypan/runtime_compat/ling256_exact
ENV_DIR="$RUNTIME_ROOT/.conda/envs/ling-sglang-aime26"
CUDA_TOOLKIT="$RUNTIME_ROOT/tools/cuda-nvcc-12.8.93"
COMPILER_ENV="$RUNTIME_ROOT/.conda/envs/spatialcoc"
SGLANG_SRC=/data/zypan/src/sglang-v0.5.19-ling256/python
MODEL=/data/zypan/models/Ling-3.0-tiny
OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1"
NVIDIA_PY="$ENV_DIR/lib/python3.11/site-packages/nvidia"
YARN_JSON='{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":131072}}'

export CUDA_VISIBLE_DEVICES=2,3
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
# FlashInfer's fused DeepSeek grouped-topk kernel does not support SM86
# (RTX 3090). Select SGLang's opt-in Triton JIT grouped router instead.
export SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export AIME26_KDA_MODE="$MODE"
export AIME26_KDA_AUDIT_JSONL="$OUT/runtime/${MODE}_audit.jsonl"
export AIME26_KDA_FORCE_JSON="$OUT/runtime/${MODE}_force.json"
export AIME26_KDA_GATE_C_DUMP_DIR="$OUT/runtime/${MODE}_gate_c_dumps"
mkdir -p "$OUT/runtime"

"$ENV_DIR/bin/python" - <<'PY'
import os, torch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2,3"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
print("GPU_ISOLATION_GATE=PASS visible=2 physical=2,3")
PY

exec "$ENV_DIR/bin/python" -u "$REPO/experiments/aime26/launch_ling_sglang.py" \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tp-size 2 --dtype bfloat16 \
  --context-length 262144 --max-total-tokens 262144 \
  --json-model-override-args "$YARN_JSON" \
  --mem-fraction-static 0.80 --sampling-backend pytorch --attention-backend triton \
  --linear-attn-backend triton --moe-runner-backend triton --disable-cuda-graph \
  --disable-radix-cache --trust-remote-code --max-running-requests 1 \
  --enable-deterministic-inference
