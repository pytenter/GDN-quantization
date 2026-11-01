#!/usr/bin/env bash
set -u
ROOT=/data/zypan
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=$ROOT/experiments/qwen35_gdn_quant/run_multi_timescale_stability_validation.py
LOG=$ROOT/results/gdn_multi_timescale_stability_v1_offline_orchestrator.log
stamp(){ date '+%Y-%m-%d %H:%M:%S %z'; }
log(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }
wait_group(){
  prefix="$1"
  log "waiting for tmux group $prefix"
  while true; do
    running=0
    for i in 0 1 2 3; do
      if tmux has-session -t "${prefix}_${i}" 2>/dev/null; then running=$((running+1)); fi
    done
    log "$prefix running sessions: $running"
    [ "$running" -eq 0 ] && break
    sleep 600
  done
}
run_analyze(){
  log "running analyze"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" --stage analyze --max-steps 32 >> "$LOG" 2>&1
  log "analyze exit code $?"
}
launch_8192_shadow(){
  if tmux ls 2>/dev/null | grep -q 'gdn_mts_8192_'; then
    log "8192 shadow sessions already exist; not launching duplicates"
    return 0
  fi
  log "launching 8192 shadow-decode funnel: 2 SHA-policy prompts, INT8-column/INT4-row/INT4-column"
  for i in 0 1 2 3; do
    tmux new-session -d -s "gdn_mts_8192_${i}" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=$i $PY $SCRIPT --stage shadow-decode --quantizers int8_column,int4_row,int4_column --max-steps 8192 --checkpoints 2048,3072,4096,6144,8192 --shard-id $i --num-shards 4 --record-every 512 --continue-after-eos > $ROOT/results/gdn_multi_timescale_stability_v1_8192_shard_${i}.log 2>&1"
  done
}
log "offline orchestrator started"
wait_group gdn_mts_4096
run_analyze
launch_8192_shadow
wait_group gdn_mts_8192
run_analyze
log "offline orchestrator complete"
