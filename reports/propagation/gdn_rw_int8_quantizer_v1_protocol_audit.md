# Protocol Audit

```json
{
  "C128_group_definition": "per-column scale over K axis; broadcast [B,H,1,V]",
  "CALIBRATION_EVAL_DISJOINT": "YES",
  "HEAD": "7c2a1c977e412ba8fc0b71c0e15e8816a46bdbec",
  "TASK": "GDN_READ_WRITE_AWARE_INT8_QUANTIZER_FEASIBILITY_V1",
  "branch": "research-sync-2026-09-02",
  "calibration_ids": [
    "test/algebra/1332.json|64",
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|256",
    "test/counting_and_probability/119.json|64",
    "test/counting_and_probability/119.json|128",
    "test/counting_and_probability/119.json|256"
  ],
  "clipping_rule": "clamp [-127,127]",
  "dequantization_timing": "fake quantized state is stored in cache for next token",
  "evaluation_ids": [
    "test/geometry/477.json|64",
    "test/geometry/477.json|128",
    "test/geometry/477.json|256"
  ],
  "gamma_grid": [
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.2
  ],
  "git_status_start": "?? experiments/propagation/run_gdn_read_write_aware_int8_quantizer_feasibility.py",
  "k_first_write(E)": "same next single-token recurrent rule memory term S^T k",
  "q_first_read(E)": "next single-token recurrent rule readout after quantized cache becomes initial_state",
  "quantization_insertion_point": "after each decode forward, cached recurrent_states[0] are rewritten",
  "rounding_rule": "torch.round for RTN; deterministic Bernoulli for SR",
  "scale_definition": "absmax(K)/127",
  "sr_seeds": [
    1729,
    2718,
    3141
  ],
  "state_tensor_shape": "[B,H,K,V] = [1,32,128,128]",
  "teacher_forced_horizon": 32,
  "timestamp": "2026-09-07 12:17:53 +0800",
  "zero_point": "symmetric zero"
}
```
