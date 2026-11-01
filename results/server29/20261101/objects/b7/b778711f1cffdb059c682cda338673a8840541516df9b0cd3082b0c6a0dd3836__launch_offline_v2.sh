#!/bin/sh
set -u
printf '%s\n' 'Qwen direct32K continuation: remaining five trajectories; original FP is retained.' 'Execution logs are saved on this server; detaching SSH does not stop tmux.'
TASK_ROOT=/data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/continuation_01/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1
mkdir -p "$TASK_ROOT/logs"
exec >> "$TASK_ROOT/logs/TMUX_LAUNCH.log" 2>&1
/data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/continuation_01_coordinator.py --config /data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/CONTINUATION_CONTROL_CONFIG.json --run
controller_rc=$?
printf 'controller_exit=%s\n' "$controller_rc"
finish_rc=95
if [ -f "$TASK_ROOT/GATE_STATUS_FINAL.json" ]; then
 CUDA_VISIBLE_DEVICES='' /data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/continuation_01/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1/analysis_scripts/offline_continuation_finish.py --root /data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/continuation_01/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1 --old-parent /data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1 > "$TASK_ROOT/logs/OFFLINE_FINISH.log" 2>&1
 finish_rc=$?
fi
printf 'CPU_finish_exit=%s\n' "$finish_rc"
if [ "$finish_rc" -ne 0 ]; then exit "$finish_rc"; fi
exit "$controller_rc"
