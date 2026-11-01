# QWEN_GDN_STAGE_B_STREAMING_ACCELERATION_V1 report

## Final verdict

All required gates passed:

- `STREAMING_REDUCER_ACCEL_P0_PARITY = PASS`
- `STREAMING_REDUCER_ACCEL_P1_PARITY = PASS`
- `GPU_STREAMING_SMOKE_ACCEL = PASS`

The existing formal mechanism analysis is materially faster without changing
the model, sample/seed order, token order, common-FP fan-out, frozen rotations,
formal quantizer, metric definitions, histogram boundaries, tail definitions,
or TopK ordering.

## Phase 1: histogram/moments path

The accelerated multiplicative histogram vectorizes candidate-bin extraction
with NumPy, applies deterministic boundary correction, and falls back to the
frozen scalar oracle within 16 float64 epsilon of a bin boundary. Integer
histogram updates remain exact.

Formal parity covered two frozen samples and 64 total tokens. Full canonical
reducer states were byte-identical for both samples. Histogram integer counts,
p99, CVaR99, max, threshold counts, moments, TopK identity, cursor, and token
counts all matched exactly.

- Oracle reducer time: 315.973959 s
- Accelerated P0 reducer time: 200.280967 s
- P0 reducer speedup: 1.577653x

## Phase 2: deterministic TopK path

Per-token GPU results are reduced to deterministic local candidates using the
frozen total order (value descending, then group id ascending). The CPU global
merge uses the unchanged oracle event key (value descending, token ascending,
group ascending), preserving Top-32 identity, Top-42 sufficiency, duplicate
identity checks, and tie breaking.

Formal parity again covered the same two samples and 64 total tokens. Both
canonical reducer-state SHA256 values matched the V3 oracle exactly:

- `aime26_01_seed1`: `100070b37481d7bdb164b64be465f2664145cd68951d28aa509de3a91d831ab6`
- `aime26_01_seed2`: `21ce0a7b4c5b049af02a27162fadc92d551b62506d27506dbef507e79b733629`

All histogram, tail, threshold, moments, TopK, cursor, common-FP, and formal
quantizer checks passed.

- Oracle reducer time: 312.829636 s
- Accelerated P1 reducer time: 140.262901 s
- P1 reducer speedup: 2.230309x

## End-to-end accelerated smoke

The final smoke ran the two frozen teacher-forced samples for 32 tokens each
(64 total) on one RTX 3090 through the unchanged formal common-FP fan-out.

- Baseline: 0.201873 token/s
- Accelerated: 0.396935 token/s
- End-to-end speedup: 1.966262x
- Decode wall time: 161.235366 s
- GPU utilization: 2.61% mean, 25.00% max
- Process CPU utilization: 100.09% of one CPU core
- Sampled GPU memory maximum: 17,710 MiB
- PyTorch peak allocated: 16.940 GiB
- PyTorch peak reserved: 16.964 GiB
- End-of-sample memory spread: 0.469 MiB (no memory creep; 16 MiB limit)
- Atomic artifact readback: PASS for both samples
- Common-FP state checks: PASS
- Formal quantizer shared-path checks: PASS
- Runtime improvement gate (at least 1.25x): PASS

The low mean GPU-utilization figure is expected for this CPU-bound reducer
workload: profiling identified reducer bookkeeping as the dominant cost, while
the accelerated run consumed approximately one full CPU core.

## Launch incident

The first smoke launch stopped before model loading because two manually copied
prefix SHA256 literals contained one duplicated hexadecimal character each.
It processed zero tokens and created no scientific output. The wrapper was
repaired only by copying the tracked 64-character values from the frozen
`smoke_identity.json` and pinning that file's SHA256. Model, data, protocol,
metrics, reducer semantics, and formal paths were not changed.

## Scope compliance

The following remained false throughout: FULL60, recurrent replay,
multi-horizon analysis, pre-divergence analysis, layer intervention, C7,
new AIME generation, new training, new loss, and new rotation training.

The task stops at this report.
