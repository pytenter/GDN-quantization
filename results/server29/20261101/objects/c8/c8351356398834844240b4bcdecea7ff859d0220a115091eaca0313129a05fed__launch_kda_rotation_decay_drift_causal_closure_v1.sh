#!/usr/bin/env bash
set -euo pipefail

repo=/data01/user2/repos/GDN-quantization
python_bin=/data01/user2/.conda/envs/ling-kda/bin/python
runner=experiments/rotation/run_kda_rotation_decay_drift_causal_closure_v1.py
test_file=tests/rotation/test_kda_rotation_decay_drift_causal_closure_v1.py
session=kda_decay_closure_v1
final_dir=results/kda_rotation_decay_drift_causal_closure_v1
shard0_dir=results/kda_rotation_decay_drift_causal_closure_v1_shard0_gpu0
shard1_dir=results/kda_rotation_decay_drift_causal_closure_v1_shard1_gpu1
timestamp=$(date +%Y%m%d_%H%M%S)
log_dir=results/kda_rotation_decay_drift_causal_closure_v1_tmux_logs_${timestamp}

cd "$repo"
if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
fi
if [[ -e "$final_dir/summary.json" || -e "$shard0_dir/unit_level/horizon_results.jsonl" || -e "$shard1_dir/unit_level/horizon_results.jsonl" ]]; then
    echo "formal output already exists; refusing to append" >&2
    exit 2
fi

mkdir -p "$final_dir/tests" "$log_dir"
python -m pytest -q "$test_file" | tee "$final_dir/tests/pytest.txt"
printf '%s\n' "$log_dir" > results/kda_rotation_decay_drift_causal_closure_v1_latest_tmux_logdir.txt

tmux new-session -d -s "$session" -n shard0_gpu0
tmux set-option -t "$session" remain-on-exit on
tmux pipe-pane -t "$session:shard0_gpu0" -o "cat >> '$repo/$log_dir/shard0_gpu0.log'"
tmux send-keys -t "$session:shard0_gpu0" "cd '$repo' && CUDA_VISIBLE_DEVICES=0 '$python_bin' '$runner' --scope formal --horizon 64 --unit-shard-index 0 --unit-shard-count 2 --output-dir '$shard0_dir' --keep-going" C-m

tmux new-window -t "$session" -n shard1_gpu1
tmux pipe-pane -t "$session:shard1_gpu1" -o "cat >> '$repo/$log_dir/shard1_gpu1.log'"
tmux send-keys -t "$session:shard1_gpu1" "cd '$repo' && CUDA_VISIBLE_DEVICES=1 '$python_bin' '$runner' --scope formal --horizon 64 --unit-shard-index 1 --unit-shard-count 2 --output-dir '$shard1_dir' --keep-going" C-m

tmux new-window -t "$session" -n merge
tmux pipe-pane -t "$session:merge" -o "cat >> '$repo/$log_dir/merge.log'"
tmux send-keys -t "$session:merge" "cd '$repo'; while [[ ! -f '$shard0_dir/summary.json' || ! -f '$shard1_dir/summary.json' ]]; do sleep 30; done; '$python_bin' '$runner' --scope formal --merge-shard-dirs '$shard0_dir' '$shard1_dir' --output-dir '$final_dir'" C-m

echo "SESSION=$session"
echo "LOG_DIR=$log_dir"
echo "FINAL_DIR=$final_dir"
