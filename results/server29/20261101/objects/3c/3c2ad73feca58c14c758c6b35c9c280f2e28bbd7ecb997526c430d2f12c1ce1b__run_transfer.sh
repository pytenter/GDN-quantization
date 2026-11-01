#!/usr/bin/env bash
set -euo pipefail

task_root=/data/zypan/research_archive/qwen_gdn/QWEN_SCIENTIFIC_RETENTION_AND_3090_MIGRATION_V1
target_root="$task_root/scientific_data"
identity_file=/tmp/qwen_retention_transfer_ed25519
ssh_transport="ssh -i $identity_file -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p 12222"

date -u +'%Y-%m-%dT%H:%M:%SZ transfer_start'

ionice -c3 nice -n 19 rsync \
  -a --partial --append-verify --bwlimit=20000 --info=progress2 \
  --rsync-path='ionice -c3 nice -n 19 rsync' \
  -e "$ssh_transport" \
  --files-from=/tmp/rsync_files_archive_A.txt \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1/ \
  "$target_root/"

date -u +'%Y-%m-%dT%H:%M:%SZ archive_A_complete'

ionice -c3 nice -n 19 rsync \
  -a --partial --append-verify --bwlimit=20000 --info=progress2 \
  --rsync-path='ionice -c3 nice -n 19 rsync' \
  -e "$ssh_transport" \
  --files-from=/tmp/rsync_files_archive_B.txt \
  root@connect.bjb1.seetacloud.com:/root/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1/ \
  "$target_root/"

date -u +'%Y-%m-%dT%H:%M:%SZ archive_B_complete'
df -B1 /data
date -u +'%Y-%m-%dT%H:%M:%SZ transfer_complete'
