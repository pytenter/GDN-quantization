#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PHYSICAL_GPU PORT MODE RUN_DIR PATCH FINAL_ROTATION_OR_EMPTY" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
PORT="$2"
MODE="$3"
RUN_DIR="$4"
PATCH="$5"
FINAL_ROTATION="$6"
EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
ENV_DIR=/data/zypan/envs/ling-sglang-aime26
CUDA_TOOLKIT=/data/zypan/tools/cuda-nvcc-12.8.93
COMPILER_ENV=/data/zypan/envs/spatialcoc
SGLANG_SRC=/data/zypan/src/sglang-v0.5.19-frozen/python
MODEL=/data/zypan/models/Ling-3.0-tiny
NVIDIA_PY="$ENV_DIR/lib/python3.11/site-packages/nvidia"
YARN_JSON='{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":131072}}'

[[ "$PHYSICAL_GPU" == 3 || "$PHYSICAL_GPU" == 4 ]] || { echo "GPU must be 3 or 4" >&2; exit 2; }
[[ -x "$ENV_DIR/bin/python" ]] || { echo "missing frozen environment" >&2; exit 2; }
[[ -f "$PATCH" ]] || { echo "missing patch: $PATCH" >&2; exit 2; }
[[ -z "$FINAL_ROTATION" || -f "$FINAL_ROTATION" ]] || { echo "missing final rotation: $FINAL_ROTATION" >&2; exit 2; }
mkdir -p "$RUN_DIR" "$RUN_DIR/forced_dumps" "$RUN_DIR/gate_c"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PATH="$CUDA_TOOLKIT/bin:$ENV_DIR/bin:$PATH"
export CPLUS_INCLUDE_PATH="$NVIDIA_PY/cuda_nvcc/include:$NVIDIA_PY/cuda_runtime/include:$NVIDIA_PY/cuda_cccl/include:$NVIDIA_PY/cublas/include:$NVIDIA_PY/curand/include"
export LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_PY/cuda_runtime/lib:$NVIDIA_PY/cublas/lib:$NVIDIA_PY/curand/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$CUDA_TOOLKIT"
export NVCC_CCBIN="$COMPILER_ENV/bin/x86_64-conda-linux-gnu-g++"
export CC="$COMPILER_ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$NVCC_CCBIN"
export PYTHONPATH="$SGLANG_SRC:$EXP/scripts"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
export TRITON_CACHE_DIR="/data/zypan/cache/recurrent_formal_gpu${PHYSICAL_GPU}/triton"
export TORCHINDUCTOR_CACHE_DIR="/data/zypan/cache/recurrent_formal_gpu${PHYSICAL_GPU}/torchinductor"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export LING_KDA_PATCH_PATH="$PATCH"
export AIME26_KDA_MODE="$MODE"
export AIME26_KDA_AUDIT_JSONL="$RUN_DIR/runtime_audit.jsonl"
export AIME26_KDA_FORCE_JSON="$RUN_DIR/force.json"
export AIME26_KDA_GATE_C_DUMP_DIR="$RUN_DIR/gate_c"
export AIME26_KDA_FINAL_ROTATION="$FINAL_ROTATION"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cat > "$RUN_DIR/effective_launch.json" <<EOF
{"host_class":"RTX3090","physical_gpu":$PHYSICAL_GPU,"port":$PORT,"mode":"$MODE","patch":"$PATCH","final_rotation":"$FINAL_ROTATION","context_length":262144,"tp_size":1,"dtype":"bfloat16","max_running_requests":1,"deterministic":true,"cuda_graph":false,"radix_cache":false}
EOF

exec "$ENV_DIR/bin/python" -u "$EXP/scripts/launch_ling_sglang_from_patch.py" \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --tp-size 1 --dtype bfloat16 --context-length 262144 --max-total-tokens 262144 \
  --json-model-override-args "$YARN_JSON" --mem-fraction-static 0.80 \
  --sampling-backend pytorch --attention-backend triton --linear-attn-backend triton \
  --moe-runner-backend triton --disable-cuda-graph --disable-radix-cache \
  --trust-remote-code --max-running-requests 1 --enable-deterministic-inference
