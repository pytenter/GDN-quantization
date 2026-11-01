#!/usr/bin/env bash
set -euo pipefail
cd /data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1
export PYTHONDONTWRITEBYTECODE=1
export LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env/bin/python experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/experiment.py controller
