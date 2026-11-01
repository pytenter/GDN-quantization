# Quantizer Semantics

```json
{
  "QWEN_R128_SEMANTICS_VERIFIED": true,
  "QWEN_C128_SEMANTICS_VERIFIED": true,
  "LING_R128_SEMANTICS_VERIFIED": true,
  "LING_C128_SEMANTICS_VERIFIED": true,
  "state_shape": "[B,H,K,V]",
  "key_axis": -2,
  "value_axis": -1,
  "R128": {
    "scale_shape": "[B,H,K,1]",
    "group": "fixed key row, all 128 value entries",
    "scale_formula": "amax(abs(group)).clamp_min(1e-12)/127"
  },
  "C128": {
    "scale_shape": "[B,H,1,V]",
    "group": "fixed value column, all 128 key entries",
    "scale_formula": "amax(abs(group)).clamp_min(1e-12)/127"
  },
  "bits": 8,
  "qrange": [
    -127,
    127
  ],
  "symmetric": true,
  "zero_point": 0,
  "rounding": "torch.round",
  "clipping": "clamp(-127,127)",
  "scale_recomputation_timing": "dynamic recompute on every quantize/dequantize event"
}
```
