#!/bin/bash
set -euo pipefail
REPO=/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1
BUG=$REPO/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1/amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/zypan/transformers-qwen35/src CUBLAS_WORKSPACE_CONFIG=:4096:8
stage=G11_CAPTURE2048
on_exit() {
    rc=$?
    printf "%s\n" "$rc" > "$BUG/logs/bugfix_workflow.exitcode"
    if [ "$rc" -ne 0 ] && [ ! -e "$BUG/FINAL_REPORT.md" ]; then
        CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage stop --failed-stage "$stage" > "$BUG/logs/failure_finalize.log" 2>&1 || true
    fi
}
trap on_exit EXIT
gpu_idle() {
    local occupants
    occupants="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
    if [ -n "$occupants" ]; then
        printf 'GPU0 occupied before %s; STOP without signalling any process: %s\n' "$stage" "$occupants" >&2
        exit 3
    fi
}
gpu_idle
"$PY" "$BUG/scripts/run_repaired_stage.py" --stage capture2048 > "$BUG/logs/capture2048_retest.log" 2>&1
for run in 01 02; do
    stage=G11_REPLAY2048_${run}
    gpu_idle
    "$PY" "$BUG/scripts/run_repaired_stage.py" --stage replay2048 --run "$run" > "$BUG/logs/replay2048_${run}.log" 2>&1
done
stage=G11_COMPARE2048
CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage compare2048 > "$BUG/logs/determinism2048.log" 2>&1
stage=G12_STREAMING_PARITY
CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage streaming > "$BUG/logs/streaming.log" 2>&1
stage=FULL_PREMEASUREMENT_FREEZE
CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage preregister > "$BUG/logs/preregistration.log" 2>&1
for doc in 1 2 3 4; do
    stage=DOC_${doc}_CAPTURE
    gpu_idle
    "$PY" "$BUG/scripts/run_repaired_stage.py" --stage capture_discovery --document "$doc" > "$BUG/logs/doc${doc}_capture.log" 2>&1
    for condition in Native H C6R C7; do
        stage=DOC_${doc}_${condition}_REPLAY
        gpu_idle
        "$PY" "$BUG/scripts/run_repaired_stage.py" --stage replay_discovery --document "$doc" --condition "$condition" > "$BUG/logs/doc${doc}_${condition}_replay.log" 2>&1
    done
    stage=DOC_${doc}_RESULT_SEAL_RELEASE
    CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage release --document "$doc" > "$BUG/logs/doc${doc}_release.log" 2>&1
done
stage=DISCOVERY_FINALIZE
CUDA_VISIBLE_DEVICES= "$PY" "$BUG/scripts/run_repaired_stage.py" --stage finalize > "$BUG/logs/finalize.log" 2>&1
