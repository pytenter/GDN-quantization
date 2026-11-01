#!/bin/bash
# Independent tmux controller: waits without touching the active GPU job.
set -euo pipefail
REPO=/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1
ROOT=$REPO/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1
AMEND=$ROOT/amendments/STORAGE_ACCOUNTING_CORRECTION_V1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/zypan/transformers-qwen35/src CUBLAS_WORKSPACE_CONFIG=:4096:8
stage=WAIT_LONG_QUALIFICATION
failure_finalize() {
    rc=$?
    if [ "$rc" -ne 0 ] && [ ! -e "$AMEND/FINAL_REPORT.md" ]; then
        CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/scientific_lifecycle.py" --phase stop --failed-phase "$stage" > "$AMEND/logs/failure_finalize.log" 2>&1 || true
    fi
    printf "%s\n" "$rc" > "$AMEND/logs/continuation.exitcode"
}
trap failure_finalize EXIT
gpu_idle() {
    local occupants
    occupants="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
    if [ -n "$occupants" ]; then
        printf 'GPU0 has another computing process before %s; STOP, do not signal it: %s\n' "$stage" "$occupants" >&2
        exit 3
    fi
}
while [ ! -e "$AMEND/logs/long_qualification.exitcode" ]; do
    if ! tmux has-session -t qwen_fd_long_qualification 2>/dev/null; then
        printf '%s\n' 'Long qualification session disappeared without an exitcode; STOP' >&2
        exit 2
    fi
    sleep 30
done
[ "$(tr -d '\r\n' < "$AMEND/logs/long_qualification.exitcode")" = 0 ]
stage=FULL_PREMEASUREMENT_FREEZE
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/scientific_lifecycle.py" --phase preregister > "$AMEND/logs/scientific_preregistration.log" 2>&1
for doc in 1 2 3 4; do
    stage=DOC_${doc}_CAPTURE
    gpu_idle
    "$PY" "$AMEND/scripts/capture_extended_drive.py" --mode discovery --document "$doc" > "$AMEND/logs/doc${doc}_capture.log" 2>&1
    for condition in Native H C6R C7; do
        stage=DOC_${doc}_${condition}_REPLAY
        gpu_idle
        "$PY" "$AMEND/scripts/persistent_anchor_replay.py" --mode discovery --length 32768 --document "$doc" --condition "$condition" > "$AMEND/logs/doc${doc}_${condition}_replay.log" 2>&1
    done
    stage=DOC_${doc}_RESULT_SEAL_AND_RELEASE
    CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/scientific_lifecycle.py" --phase release --document "$doc" > "$AMEND/logs/doc${doc}_release.log" 2>&1
done
stage=DISCOVERY_FINALIZE
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/scientific_lifecycle.py" --phase finalize > "$AMEND/logs/scientific_finalize.log" 2>&1
