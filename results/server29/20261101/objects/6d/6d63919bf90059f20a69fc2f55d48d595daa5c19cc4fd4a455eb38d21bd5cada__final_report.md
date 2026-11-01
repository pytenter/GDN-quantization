# Qwen dual-GPU trajectory equivalence closure

- Verdict: **FAIL**.
- Forward exact gate: **PASS**; trajectory envelope gate: **FAIL**; fixed non-AIME endpoint probe: **FAIL**.
- Single-GPU bitwise backward repeatability: **FAIL** (established in prior preserved closure).
- 10 fresh-process single trajectories and 5 fresh-process dual trajectories; 8 H32 Adam updates each from theta=0; no formal C5/C6 training or AIME generation.
- Single reference: all 45 pairwise differences, per update and per layer, frozen as strict observed maxima without multiplier before dual comparison.
- Cross comparisons: 50 single-versus-dual pairs; 106550 trajectory metric cells; 4303 outside the frozen envelope.
- Endpoint probe: 96800 metric cells; 4885 outside the frozen 45-pair single endpoint envelope.

## First out-of-envelope evidence

- Trajectory: `{"envelope": 9.465344701031775e-06, "family": "rotation", "metric": "cosine_error", "run_a": "single_02", "run_b": "dual_01", "scope": "17", "update": 1, "value": 9.585602435135954e-06, "within": false, "zero_case": "neither_zero"}`
- Functional probe: `{"dual": "dual_01", "envelope": 0.08111376039423988, "excess": 0.002495284892320579, "family": "scale", "metric": "relative_l2", "single": "single_01", "tensor": "scale_31_22", "value": 0.08360904528656046}`

## Per-update loss range

| Update | Single-GPU min–max | Dual-GPU min–max | Cross-envelope failures |
|---:|---:|---:|---:|
| 1 | 0.0118267038–0.0118267038 | 0.0118267038–0.0118267038 | 224 |
| 2 | 0.00686402834–0.0129296395 | 0.00689945763–0.0112506584 | 718 |
| 3 | 0.00686640688–0.00815987977 | 0.00697572483–0.00727176989 | 625 |
| 4 | 0.00807748234–0.00920204091 | 0.00806227804–0.00879053571 | 544 |
| 5 | 0.0101044087–0.0117410795 | 0.0103170497–0.0138355776 | 682 |
| 6 | 0.00989490311–0.011763904 | 0.00970853181–0.0111435215 | 543 |
| 7 | 0.00974361534–0.0115349377 | 0.00938237738–0.0112833223 | 550 |
| 8 | 0.00968081027–0.0152143415 | 0.0103753208–0.0128909142 | 417 |

## Provenance and stop point

- Frozen protocol SHA-256: `746974233485b5deef79f945bac961ebb93c4d4f64144a876575f8e97d59d237`.
- Frozen trajectory envelope SHA-256: `9fcb49d1951c0a333bd2f2cab667c5e082ec34160cb5fe070d9f8ab168efa638`.
- Frozen functional envelope SHA-256: `0a5be9801c8b560e085a4a2d81c73900a846249a6d05524d5b0609a29c095825`.
- Full update-level, layer-level, raw forward/QDQ, gradient and rotation artifacts, pairwise trajectory comparisons, probe data and hashes are retained under this closure directory.
- Gradient and parameter hashes are provenance only, not equality gates, because the single-GPU reference is not bitwise backward deterministic.
- NEXT_STAGE_AUTHORIZED = **NO**. No next-stage work was started; wait for a separate decision even if verified.
