# Orthogonal Rotation Interface Audit

Task: `ORTHOGONAL_NUMERICAL_CLOSURE_V1`

## Verdict

- Generic `R`/`R.T` interface: **PASS**.
- Historical Hadamard self-inverse dependency: **found only in the historical interface and audit helpers; generalized code uses explicit forward/inverse operations**.
- Unnecessary inference roundtrip: **none found**.
- Audit path side effect: **NO**; runtime outputs were bitwise identical with detached recovery enabled and disabled.

## Qwen3.5-9B / GDN

| Location | Purpose | Inference | Audit only | Before | After |
|---|---|---:|---:|---|---|
| `QwenKeyOrthogonalPatch._wrap`, q | Key-basis forward rotation | yes | no | native model dtype | FP32 rotated q |
| `QwenKeyOrthogonalPatch._wrap`, k | Key-basis forward rotation | yes | no | native model dtype | FP32 rotated k |
| recurrent GDN state | stay in rotated Key basis | yes | no | rotated state | rotated state |
| `recover_key_state` | compare state in native coordinates | no | yes | detached rotated state | detached FP32 native view |
| `first_decode_capture`, q/k recovery | localization only | no | yes | detached rotated q/k | detached FP32 native view |

There is no functional Key-side inverse after the GDN core because the Value/readout dimension remains native. Qwen inference therefore contains forward q/k rotations but no artificial `R.T -> R` cycle.

## Ling-3.0-tiny / KDA

| Location | Purpose | Inference | Audit only | Before | After |
|---|---|---:|---:|---|---|
| `FullPathForensicProbe._wrap`, v | Value-basis forward rotation | yes | no | BF16 native Value | rotated Value |
| recurrent KDA state | stay in rotated Value basis | yes | no | rotated state | rotated state |
| KDA raw readout mapback | functional native-basis boundary before norm/gate/o_proj | yes | no | rotated core output | native model dtype |
| `state_gap` / `semantic_stack` | compare state in native coordinates | no | yes | detached rotated state | detached FP32 native view |
| `first_decode_capture`, v/state recovery | localization only | no | yes | detached rotated tensor | detached FP32 native view |

Ling uses `CORRECTED_PREFILL_ENDPOINT_V2`: the prefill kernel state is already in rotated Value coordinates, is written directly to cache, and is consumed unchanged by the first decode step. No endpoint rotation is applied.

## Required interface

- Identity: `forward(x) = x`, `inverse(x) = x`.
- Hadamard: `forward(x) = x @ H`, `inverse(x) = x @ H`.
- Dense orthogonal: `forward(x) = x @ R`, `inverse(x) = x @ R.T`.

Only the Hadamard backend may exploit symmetry/self-inverse behavior. Generic model, cache, recovery, and logging code must never assume `R == R.T`.

## Precision boundary

- Keep the rotation matrix master copy and dense rotation/recovery matmuls in FP32 or higher.
- Treat every BF16 storage boundary as explicit metadata.
- Audit recovery must operate on detached tensors and must never be copied back into inference caches.
- Ling must cast the native-basis readout at its existing functional boundary; moving the recurrent state to FP32 is a separate policy decision, not an audit operation.
