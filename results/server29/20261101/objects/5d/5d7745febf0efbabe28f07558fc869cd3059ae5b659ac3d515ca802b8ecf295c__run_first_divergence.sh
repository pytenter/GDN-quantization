#!/usr/bin/env bash
set -u
R=/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_C6R_C7_FIRST_DIVERGENCE_PREFIX_REPLAY_V1
P=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$R" || exit 90
exec 9>logs/controller.lock
flock -n 9 || exit 91
if [ -e logs/controller.started ]; then exit 92; fi
date -u +%FT%TZ > logs/controller.started
for gpu in 0 1 2 3; do
  name=qwen_prefix_replay_$gpu
  if tmux has-session -t "$name" 2>/dev/null; then exit 93; fi
  tmux new-session -d -s "$name" "env CUDA_VISIBLE_DEVICES=$gpu $P -u $R/scripts/first_divergence_replay.py --phase worker --gpu $gpu > $R/logs/worker$gpu.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $R/logs/worker$gpu.exitcode"
done
# Detached process waits only for these workers; never launches a retry or next experiment.
while true; do
  if [ -e logs/STOP.json ]; then printf 'Worker failure: downstream stopped\n'; exit 94; fi
  n=0
  for gpu in 0 1 2 3; do if [ -e logs/worker$gpu.exitcode ]; then n=$((n+1)); fi; done
  if [ "$n" = 4 ]; then break; fi
  sleep 15
done
for gpu in 0 1 2 3; do
  if [ "$(tr -d '\r\n' < logs/worker$gpu.exitcode)" != 0 ]; then exit 95; fi
done
nice -n 19 ionice -c 3 "$P" -u scripts/analyze_first_divergence.py > logs/analysis.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/analysis.exitcode
printf 'Analysis exit %s; FINAL_ACTION STOP\n' "$rc"
exit "$rc"
