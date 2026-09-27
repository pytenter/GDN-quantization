#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/root/autodl-tmp/GDN-formal-c5-c6-h64-v1"
EXP_REL="experiments/qwen_gdn/recurrent_training/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1"
EXP="$REPO/$EXP_REL"
PY="/root/miniconda3/envs/qwen-recurrent/bin/python"
PIPELINE="$EXP/scripts/formal_pipeline.py"
BRANCH="exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1"
STATE_DIR="/root/autodl-tmp/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1_runner"
RUN_LOG="$STATE_DIR/runner.log"
STATUS_FILE="$STATE_DIR/status.env"
LOCK_FILE="$STATE_DIR/runner.lock"

mkdir -p "$STATE_DIR" "$EXP/logs"
touch "$RUN_LOG"
exec >>"$RUN_LOG" 2>&1
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s runner already active\n' "$(date -Is)"
  exit 73
fi

CURRENT_GATE="bootstrap"
LAST_VALID_ARTIFACT="C5_preflight"
LAST_VALID_COMMIT="$(git -C "$REPO" rev-parse HEAD)"

write_status() {
  local state="$1"
  local why="${2:-}"
  local tmp="$STATUS_FILE.tmp.$$"
  {
    printf 'STATE=%q\n' "$state"
    printf 'CURRENT_GATE=%q\n' "$CURRENT_GATE"
    printf 'WHY=%q\n' "$why"
    printf 'LAST_VALID_ARTIFACT=%q\n' "$LAST_VALID_ARTIFACT"
    printf 'LAST_VALID_COMMIT=%q\n' "$LAST_VALID_COMMIT"
    printf 'UPDATED_AT=%q\n' "$(date -Is)"
  } >"$tmp"
  mv -f "$tmp" "$STATUS_FILE"
}

fail() {
  local why="$1"
  printf '%s FAIL gate=%s why=%s\n' "$(date -Is)" "$CURRENT_GATE" "$why"
  write_status FAILED "$why"
  exit 1
}

on_error() {
  local rc=$?
  local line="${BASH_LINENO[0]:-unknown}"
  fail "unexpected command failure rc=$rc line=$line"
}
trap on_error ERR

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

json_gate() {
  "$PY" - "$@" <<'PY'
import json
import sys

path, *checks = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
for check in checks:
    key, expected = check.split("=", 1)
    value = payload
    for part in key.split("."):
        value = value[part]
    if str(value) != expected:
        raise SystemExit(f"{path}: {key}={value!r}, expected {expected!r}")
PY
}

run_logged() {
  local stage="$1"
  shift
  local console="$EXP/logs/${stage}.console.log"
  local exitcode="$EXP/logs/${stage}.exitcode"
  rm -f "$exitcode"
  log "START $stage"
  set +e
  "$@" >"$console" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$exitcode"
  if (( rc != 0 )); then
    fail "$stage exited with rc=$rc; see $console"
  fi
  log "PASS $stage"
}

run_logged_external() {
  local stage="$1"
  shift
  local console="$STATE_DIR/${stage}.console.log"
  local exitcode="$STATE_DIR/${stage}.exitcode"
  rm -f "$exitcode"
  log "START $stage"
  set +e
  "$@" >"$console" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$exitcode"
  if (( rc != 0 )); then
    fail "$stage exited with rc=$rc; see $console"
  fi
  log "PASS $stage"
}

commit_stage() {
  local message="$1"
  shift
  git -C "$REPO" add -f -- "$@"
  if git -C "$REPO" diff --cached --quiet; then
    fail "no staged artifacts for commit: $message"
  fi
  git -C "$REPO" commit -m "$message"
  LAST_VALID_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
  log "COMMIT $LAST_VALID_COMMIT $message"
}

cd "$REPO"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || fail "wrong git branch"
[[ "$(git status --porcelain --untracked-files=no)" == "" ]] || fail "tracked worktree is dirty"
[[ -x "$PY" && -f "$PIPELINE" ]] || fail "formal runtime is missing"
write_status RUNNING

CURRENT_GATE="C5_formal_training"
write_status RUNNING
log "waiting for detached C5 formal training"
while [[ ! -f "$EXP/logs/C5_formal.exitcode" ]]; do
  if ! pgrep -f '[f]ormal_pipeline.py train --condition C5' >/dev/null; then
    fail "C5 process disappeared before writing an exit code"
  fi
  sleep 60
done
[[ "$(tr -d '[:space:]' < "$EXP/logs/C5_formal.exitcode")" == "0" ]] || \
  fail "C5 formal training returned nonzero"
json_gate "$EXP/analysis/C5_training_summary.json" status=PASS total_updates=125 total_target_exposures=1000
json_gate "$EXP/analysis/C5_validation_summary.json" status=PASS candidate_count=20
json_gate "$EXP/analysis/C5_checkpoint_selection.json" status=PASS
LAST_VALID_ARTIFACT="C5_formal_training"
commit_stage "qwen: complete formal recurrent C5 training" \
  "$EXP_REL/analysis/C5_training_summary.json" \
  "$EXP_REL/analysis/C5_validation_summary.json" \
  "$EXP_REL/analysis/C5_checkpoint_selection.json" \
  "$EXP_REL/checkpoints/C5" \
  "$EXP_REL/logs/C5_updates.jsonl" \
  "$EXP_REL/logs/C5_formal.console.log" \
  "$EXP_REL/logs/C5_formal.exitcode" \
  "$EXP_REL/scripts/run_remaining_pipeline.sh"
write_status RUNNING

CURRENT_GATE="C5_frozen_export"
write_status RUNNING
run_logged C5_export "$PY" "$PIPELINE" export --condition C5
json_gate "$EXP/analysis/C5_export_summary.json" status=PASS C5_DEPLOYMENT_ARTIFACT_READY=YES functional_sanity=PASS
json_gate "$EXP/analysis/C5_frozen_functional_sanity.json" status=PASS
LAST_VALID_ARTIFACT="C5_frozen_export"
commit_stage "qwen: freeze and validate formal C5 deployment artifacts" \
  "$EXP_REL/analysis/C5_export_summary.json" \
  "$EXP_REL/analysis/C5_frozen_functional_sanity.json" \
  "$EXP_REL/artifacts/C5" \
  "$EXP_REL/deployment/C5" \
  "$EXP_REL/logs/C5_export.console.log" \
  "$EXP_REL/logs/C5_export.exitcode"
write_status RUNNING

CURRENT_GATE="C6_preflight"
write_status RUNNING
run_logged C6_preflight "$PY" "$PIPELINE" preflight --condition C6
json_gate "$EXP/analysis/C6_preflight.json" status=PASS formal_training_started=False H=64
LAST_VALID_ARTIFACT="C6_preflight"
commit_stage "qwen: pass formal C6 no-update preflight" \
  "$EXP_REL/analysis/C6_preflight.json" \
  "$EXP_REL/logs/C6_preflight.console.log" \
  "$EXP_REL/logs/C6_preflight.exitcode"
write_status RUNNING

CURRENT_GATE="C6_formal_training"
write_status RUNNING
run_logged C6_formal "$PY" "$PIPELINE" train --condition C6
json_gate "$EXP/analysis/C6_training_summary.json" status=PASS total_updates=125 total_target_exposures=1000
json_gate "$EXP/analysis/C6_validation_summary.json" status=PASS candidate_count=20
json_gate "$EXP/analysis/C6_checkpoint_selection.json" status=PASS
LAST_VALID_ARTIFACT="C6_formal_training"
commit_stage "qwen: complete formal recurrent C6 training" \
  "$EXP_REL/analysis/C6_training_summary.json" \
  "$EXP_REL/analysis/C6_validation_summary.json" \
  "$EXP_REL/analysis/C6_checkpoint_selection.json" \
  "$EXP_REL/checkpoints/C6" \
  "$EXP_REL/logs/C6_updates.jsonl" \
  "$EXP_REL/logs/C6_formal.console.log" \
  "$EXP_REL/logs/C6_formal.exitcode"
write_status RUNNING

CURRENT_GATE="C6_frozen_export"
write_status RUNNING
run_logged C6_export "$PY" "$PIPELINE" export --condition C6
json_gate "$EXP/analysis/C6_export_summary.json" status=PASS C6_DEPLOYMENT_ARTIFACT_READY=YES functional_sanity=PASS
json_gate "$EXP/analysis/C6_frozen_functional_sanity.json" status=PASS
LAST_VALID_ARTIFACT="C6_frozen_export"
commit_stage "qwen: freeze and validate formal C6 deployment artifacts" \
  "$EXP_REL/analysis/C6_export_summary.json" \
  "$EXP_REL/analysis/C6_frozen_functional_sanity.json" \
  "$EXP_REL/artifacts/C6" \
  "$EXP_REL/deployment/C6" \
  "$EXP_REL/logs/C6_export.console.log" \
  "$EXP_REL/logs/C6_export.exitcode"
write_status RUNNING

CURRENT_GATE="final_reports_and_hashes"
write_status RUNNING
# Keep this command's own stdout and exit-code files outside EXP: final-report
# inventories every file under EXP, so in-tree logging would mutate an entry
# after its digest was computed.
run_logged_external final_report "$PY" "$PIPELINE" final-report
json_gate "$EXP/hashes/inventory_provenance.json" status=PASS
grep -q '^HASH_VERIFICATION = PASS$' "$EXP/reports/FORMAL_TRAINING_RESULT.md" || \
  fail "final result does not report HASH_VERIFICATION=PASS"
LAST_VALID_ARTIFACT="final_reports_and_hashes"
commit_stage "qwen: finalize recurrent C5 C6 H64 reports and hashes" \
  "$EXP_REL/reports" \
  "$EXP_REL/hashes"
write_status RUNNING

CURRENT_GATE="push_final_branch"
write_status RUNNING
git push -u origin "$BRANCH"
remote_head="$(git ls-remote --heads origin "$BRANCH" | awk '{print $1}')"
local_head="$(git rev-parse HEAD)"
[[ -n "$remote_head" && "$remote_head" == "$local_head" ]] || fail "remote branch HEAD verification failed"
LAST_VALID_ARTIFACT="remote_branch"
LAST_VALID_COMMIT="$local_head"
CURRENT_GATE="complete"
write_status COMPLETE
log "COMPLETE branch=$BRANCH head=$local_head"
