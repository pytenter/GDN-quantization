#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/repos/GDN-quantization
PY=/data/zypan/envs/ling-kda/bin/python3.11
RUNNER=experiments/rotation/run_kda_value_hadamard_postfix_scientific_rebaseline_v1.py
OUT=results/kda_value_hadamard_postfix_scientific_rebaseline_v1

rm -f "$OUT/TMUX_COMPLETE" "$OUT/TMUX_FAILED"
"$PY" "$RUNNER" --phase finalize >> "$OUT/run.log" 2>&1
"$PY" "$RUNNER" --phase validate >> "$OUT/run.log" 2>&1

set +e
"$PY" -m pytest -q tests/test_kda_rotation_*.py \
  tests/test_kda_value_hadamard_postfix_scientific_rebaseline_v1.py \
  2>&1 | tee "$OUT/pytest.log"
pytest_status=${PIPESTATUS[0]}
set -e

if [[ $pytest_status -eq 0 ]]; then
  printf '{"PYTEST":"PASS","exit_code":0}\n' > "$OUT/pytest_summary.json"
else
  printf '{"PYTEST":"FAIL","exit_code":%d}\n' "$pytest_status" > "$OUT/pytest_summary.json"
fi

"$PY" "$RUNNER" --phase finalize >> "$OUT/run.log" 2>&1

if [[ $pytest_status -ne 0 ]]; then
  touch "$OUT/TMUX_FAILED"
  exit "$pytest_status"
fi

touch "$OUT/TMUX_COMPLETE"
