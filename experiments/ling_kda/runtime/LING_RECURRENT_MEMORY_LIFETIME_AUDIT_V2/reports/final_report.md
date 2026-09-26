# LING_RECURRENT_MEMORY_LIFETIME_AUDIT_V2

## Verdict

`LING_RECURRENT_MEMORY_LIFETIME_AUDIT = PARTIAL`

`LING_MEMORY_BASELINE_GROWTH = UNRESOLVED`

`LING_MEMORY_LIFETIME_STABILITY_GATE = UNKNOWN`

`LING_RECURRENT_CACHE_LIFETIME_GATE = UNKNOWN`

`LING_QDQ_AUTOGRAD_LIFETIME_GATE = UNKNOWN`

`LING_LOSS_GRAPH_RETENTION_GATE = PASS`

`LING_VALIDATION_MEMORY_RETURN_GATE = UNKNOWN`

`LING_SUSTAINED_RECURRENT_MEMORY_FEASIBILITY = UNKNOWN`

Classification: `LING_MEMORY_LIFETIME_UNRESOLVED`.

## Required questions

1. **Does completed L6 training prove a stable baseline?** No. It proves completion at horizon 128, but the 100-step curve has no per-update allocated/reserved memory telemetry.
2. **Does end-of-update allocated memory grow?** Unresolved until the independent 20–30 update harness runs.
3. **Does recurrent cache retain old BPTT graphs?** No accumulating pattern was found statically. Student states are detached at every 128-token boundary and numerically reused, but the dynamic gate remains UNKNOWN.
4. **Does QDQ/STE retain across updates?** The implementation has local QDQ intermediates and detach-based STE with no custom autograd context/cache. Dynamic lifetime remains UNKNOWN.
5. **Do loss/metrics retain graph-bearing CUDA tensors?** No. Persistent history is populated from detached CPU/Python numbers; static gate PASS.
6. **Does validation return to baseline?** Validation uses `torch.no_grad()` and explicit per-path cleanup, but no memory telemetry exists; UNKNOWN.
7. **Baseline / peak / headroom?** Baseline and headroom are unresolved. The selected horizon-128 feasibility probe peak was 19.804682 GiB allocated.
8. **Growth without OOM?** Unresolved. Horizon 512 and 256 OOMed; selected 128 completed both 100-step trainings.
9. **Does L7 share potential risk?** Yes: it uses the same recurrent lifetime structure and has a larger functional graph. Static evidence is favorable, but dynamic measurement is required.
10. **Is a patch needed?** No patch is authorized or justified before dynamic evidence. A non-invasive audit harness is the next step when GPUs are idle.

## Resource deferral

The dynamic harness, weakref probes, per-layer CUDA inventories and memory snapshots were not run because live formal Ling generation was detected. Existing processes were not modified, interrupted or profiled.
