#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT_ROOT" ]] || { echo "--output-root is required" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_ROOT"
hf download pytenter/ling-kda-artifacts \
  --type dataset \
  --include 'LING_RECURRENT_DENSE_L6_L7_V1/*' \
  --local-dir "$OUTPUT_ROOT"
python3 "$SCRIPT_DIR/verify_hf_artifacts.py" --root "$OUTPUT_ROOT"
