#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
SOCKET=/tmp/qwen_a800_source_control
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/transfer.exitcode"' EXIT
ssh -S "$SOCKET" -O check -p 12222 root@connect.bjb1.seetacloud.com
for directory in canonical_source GDN-formal-c5-c6-h64-v1 wheelhouse models; do
    rsync -a --partial --append-verify --info=progress2 \
      -e "ssh -S $SOCKET -p 12222 -o BatchMode=yes" \
      "root@connect.bjb1.seetacloud.com:/root/autodl-tmp/$directory" /root/autodl-tmp/
done
printf 'TRANSFER_COMPLETE\n'
