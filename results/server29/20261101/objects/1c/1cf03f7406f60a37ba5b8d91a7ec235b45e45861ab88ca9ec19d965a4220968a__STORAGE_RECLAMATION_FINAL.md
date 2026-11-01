# Storage Reclamation Final Report

## Verdict

`DATA_RECLAMATION_GATE = PASS`

- `/data` free before: `20641325056` bytes
- `/data` free after: `59346919424` bytes (`56G` from `df -h`)
- Filesystem free-space delta: `38705594368` bytes
- Explicit verified source bytes removed: `38705015311` bytes
- `/data/zypan` after: `23726983427` bytes
- Files archived: `350`
- Hot-storage source files removed: `350`

## Relocated groups

- Endpoint probes: `16161539267` bytes, `15` files; `ENDPOINT_PROBE_ARCHIVE_HASH_GATE = PASS`
- M5 component replay: `8878073608` bytes, `124` files; `M5_COMPONENT_REPLAY_ARCHIVE_HASH_GATE = PASS`
- Large legacy results: `7027486573` bytes, `81` files; destination hash gate `PASS`
- Completed old raw analysis/trajectory/capture: `6637915863` bytes, `130` files; destination hash gate `PASS`

## Archive destinations

- `ssh://root@connect.bjb1.seetacloud.com:12222/root/autodl-tmp/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1`
- `ssh://root@connect.bjb1.seetacloud.com:12222/root/qwen_archive/QWEN_DATA_STORAGE_SAFE_RECLAMATION_V1`

All removed hot-storage files have byte-identical verified archives. Original scientific manifests were not rewritten. The relocation manifests retain original paths and archive paths.

## Integrity checks

- Qwen model exists and recorded shard sizes match the pre-cleanup audit: `PASS`
- Original endpoint and M5 manifests preserved: `PASS`
- Relocation rows have matching source/destination SHA256: `PASS`
- Archived source copies removed only after verification: `PASS`
- Deleted-but-open regular files under `/data`: `NONE`
- Unknown remaining files over 500 MB: `0`
- Pre-existing files outside `/data/zypan` modified or deleted on the original server: `NO`
- Task-only helper scripts staged under `/tmp`: `5` (removed after final report generation)
- Active experiments interrupted: `NO`

The target was met, so large-scale cleanup stopped immediately. No training or AIME generation was started.
