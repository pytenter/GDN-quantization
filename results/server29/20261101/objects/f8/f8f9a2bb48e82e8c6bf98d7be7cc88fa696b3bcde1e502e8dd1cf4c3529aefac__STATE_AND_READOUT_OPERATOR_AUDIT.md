# State and Readout Operator Audit

Status: premeasurement frozen; formal results pending.

## State path

The actual model cache is retained as a full model-specific cache object. Its tensor inventory contains 84 tensors in the mechanical test, covering the hybrid KDA/MLA incremental state rather than only the KDA matrix. The fork is serialized/restored as the complete cache; branch caches are deep-copied and checked for storage aliasing.

The KDA state blocks are shaped `[batch=1, heads=16, query_dim=128, value_dim=128]` at layers `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22`. The fixed intervention support is layer 20, all heads.

## Physical and rotated coordinates

State comparison and intervention are in physical cache coordinates. L6 writeback uses the frozen rotation `R`: `rotated = S @ R`; original rowwise `qdq_r128` is applied over the value dimension; physical writeback is `S_qdq = qdq(rotated) @ R.T`. Every continuation token performs the same operation for all 18 KDA layers.

## Readout operator

For each head, normalized prepared FP queries from the independent 512-token calibration window form `C_h = mean(q_h q_h^T)`. For a state error `E_h`, frozen readout sensitivity is `trace(E_h^T C_h E_h)`. No scored Future-KL or downstream logit target enters direction construction.

VISIBLE uses the maximum-eigenvalue query direction of each `C_h`; LOW uses its minimum-eigenvalue direction. Both use the leading value-space direction from that document's natural target-block error and are globally Frobenius-normalized. LOW is explicitly **low-sensitivity**, not claimed to be a strict nullspace. The construction is future-query-aware and is a retrospective mechanism diagnostic, not a training-time surrogate.

## Equal-magnitude construction

`epsilon = ||S_INT8 - S_FP||_F` at the frozen target block. Each signed candidate is `S_FP ± epsilon*u`, then passed through real L6 QDQ. An outcome-free search compensates only for finite-grid norm distortion. It is accepted solely by the frozen mechanical gates in `FROZEN_CONFIG.json`.

## Asset identities

- L6 artifact SHA256: `51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964`.
- Capture implementation SHA256: `23972a3d6b86547c56f9eb5df2762af4325d627fdd0d7b34c064fd88e1ea37a0`.
- Common implementation SHA256: `4bf5e7b61d670eae37875afff1b97776e0919782967fbc9b2ffa39b19ce9bfa6`.
- Formal runner SHA256: `fb49e377b6b2786deedd694c4489115f08d4cb0bb95cb7f45a93b0a44914ed20`.
## Formal outcome audit

All three formal runs preserved the frozen physical-coordinate intervention and original L6 rotated-QDQ/writeback path. The post-QDQ norm, signed-direction, sensitivity-separation, finite-value, branch-identity, and fork-parity gates all passed.

| Document | Natural target norm | Visible/low sensitivity ratio | Mean visible endpoint target norm | Mean low endpoint target norm |
|---|---:|---:|---:|---:|
| pg19-test-37702 | 73.204887 | 14088376.173x | 26.714661 | 68.003311 |
| pg19-test-28988 | 89.253441 | 8045351.788x | 30.568223 | 87.630905 |
| pg19-test-26493 | 81.990967 | 11206070.989x | 27.262639 | 84.261864 |

The visible intervention produced *smaller*, not larger, endpoint target-state error than LOW on every document while still producing larger Future-KL AUC. Thus larger endpoint error magnitude cannot explain the observed contrast. The permitted claim remains trajectory-level: initial error direction changes the subsequent recurrent/quantized trajectory and functional damage. These runs do not isolate a readout-only direct effect from mediated recurrent and quantizer feedback.
