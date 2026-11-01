#!/usr/bin/env bash
set -u

experiment_pgid="$1"
output_dir="$2"
log="$output_dir/logs/gpu37_watchdog.log"
printf '%s watchdog_start pgid=%s selected_gpus=3,7\n' \
  "$(date --iso-8601=seconds)" "$experiment_pgid" >> "$log"

while kill -0 "$experiment_pgid" 2>/dev/null; do
  mapfile -t pids < <(
    nvidia-smi -i 3,7 --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d'
  )
  for pid in "${pids[@]}"; do
    observed_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -z "$observed_pgid" ]] && continue
    if [[ "$observed_pgid" != "$experiment_pgid" ]]; then
      {
        printf '%s foreign_gpu_process_detected pid=%s pgid=%s\n' \
          "$(date --iso-8601=seconds)" "$pid" "$observed_pgid"
        ps -o user,pid,ppid,pgid,lstart,cmd -p "$pid"
        printf 'action=terminate_experiment_group_only\n'
      } >> "$log"
      kill -TERM -- "-$experiment_pgid" 2>/dev/null || true
      exit 99
    fi
  done
  sleep 10
done

printf '%s watchdog_clean_end\n' "$(date --iso-8601=seconds)" >> "$log"
