#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1"
GATE="$OUT/stage0/STAGE0_PASS"

if [[ ! -f "$GATE" ]] || ! grep -qx 'STAGE_0_STATUS=PASS' "$GATE"; then
  echo "formal launch blocked: Stage 0 PASS gate is absent" >&2
  exit 1
fi

# Because the old 4090 formal used native RoPE without YaRN, first produce the
# required matched 81920 baseline on this exact TP=2/YaRN runtime.
"$EXP/run_ling_256k_condition.sh" fp_state 31000 baseline81920 formal256k
"$EXP/run_ling_256k_condition.sh" int8_r128 31000 formal256k
"$EXP/run_ling_256k_condition.sh" int8_r128_value_h 31000 formal256k

