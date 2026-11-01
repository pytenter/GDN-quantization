# QWEN_GDN_STAGE_B_FULL60_MECHANISM_CAPTURE_DESIGN_V1 report

## Verdict

`FULL60_CAPTURE_DESIGN_READY = YES`

`FULL60_EXECUTION_READY = NO`

No Full60 generation, replay, model load, scoring, or GPU workload was started.

## Main finding

The current Stage-B reducer already gives the exact tail evidence needed to
test the Hadamard tail-suppression hypothesis, but it cannot by itself explain
recurrent failure: its branches are reset to a common FP state each token. The
minimum sufficient experiment therefore has two explicitly separate planes:

1. controlled common-FP capture for rotation and direct quantization error;
2. persistent-writeback recurrent capture for state drift and function-space
   degradation.

The two outputs are joined with frozen result metadata and frozen output-token
divergence. This yields the complete requested chain without saving full states
or logits and without changing reducer semantics.

## Feasibility

- Exact reducer plane: approximately 1,298 base GPU-hours; 14.9–16.2 days on
  four GPUs after the sidecar overhead allowance.
- Recurrent/function capture: approximately 29.9 GPU-hours and 8.1–10 h wall.
- Total planning envelope: 1,457–1,587 GPU-hours and 15.2–16.7 days wall.
- Conservative final storage: 9.5 GiB; temporary peak: 13 GiB.
- Current `/data` free: 30.68 GiB versus a 28 GiB required launch threshold.
- GPU memory requires two sequential three-row recurrent bundles; five rows are
  forbidden.

## Required boundary before execution

Implementation must first pass exact reducer parity, scalar reference parity,
repeated-FP bundle parity, artifact integrity, 2×64-token integrated smoke, and
longest-prefix resource smoke. The implementation task must then stop for
separate user authorization. This design does not authorize Full60.

See `DESIGN_DOCUMENT.md`, `STORAGE_ESTIMATE.md`, `RUNTIME_ESTIMATE.md`,
`IMPLEMENTATION_PLAN.md`, and `READINESS_AUDIT.md` for the complete specification.
