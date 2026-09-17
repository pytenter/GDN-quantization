# Server A storage audit

- Audit mode: read-only; no existing cache, environment, model, or result was deleted or modified.
- `/data/zypan`: writable, but `/data` has only 23.6 GB available.
- `/home`: 63.5 GB available; `/home/zypan` does not exist and mount root is not writable.
- `/scratch` and `/work`: missing.
- `/mnt`: root filesystem, 9.7 GB available, not writable.
- `/dev/shm`: ~126 GB writable, but volatile RAM-backed tmpfs and therefore rejected for a persistent SGLang/model environment.

`QWEN_SGLANG_STORAGE_CANDIDATE = NONE`

Status: **NO_PERSISTENT_WRITABLE_MOUNT_WITH_GT_80_GIB_AVAILABLE**. Wait for a persistent mount or an explicit capacity-management decision.
