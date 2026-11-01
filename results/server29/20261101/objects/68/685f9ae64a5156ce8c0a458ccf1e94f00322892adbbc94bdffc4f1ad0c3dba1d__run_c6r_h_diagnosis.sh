#!/usr/bin/env bash
set -eu
R=/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_C6R_H_DISCORDANT_MECHANISM_DIAGNOSIS_V1
P=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$R"
exec 9>logs/controller.lock
flock -n 9 || exit 91
test ! -e logs/controller.started || exit 92
date -u +%FT%TZ > logs/controller.started
for gpu in 0 1 2 3; do
  name=qwen_c6r_h_diag_$gpu
  if tmux has-session -t "$name" 2>/dev/null; then exit 93; fi
  env -u TMUX tmux new-session -d -s "$name" "env CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 $P -u $R/scripts/c6r_h_diagnosis.py --phase worker --gpu $gpu > $R/logs/worker$gpu.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $R/logs/worker$gpu.exitcode"
done
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
set +e
nice -n 19 ionice -c 3 "$P" -u scripts/c6r_h_analysis.py > logs/analysis.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/analysis.exitcode
printf 'Analysis exit %s; FINAL_ACTION STOP\n' "$rc"
exit "$rc"
