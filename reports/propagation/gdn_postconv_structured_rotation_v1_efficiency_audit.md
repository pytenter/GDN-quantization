# Efficiency Audit

```json
{
  "ONLINE_FULL_STATE_ROTATION": "NO",
  "QK_FWHT_REQUIRED": "YES_FOR_NON_IDENTITY_CANDIDATES",
  "d_k": 128,
  "fwht_additions_per_vector": 896,
  "num_value_heads": 32,
  "rotation_extra_flops_per_layer_token_additions": 57344,
  "rough_gdn_core_flops_per_layer_token": 2113536,
  "ROTATION_EXTRA_FLOPS_PERCENT": 2.7131782945736433,
  "POSTCONV_FWHT_FUSABLE": "CONDITIONAL_BUT_NOT_USEFUL_WITH_FAILED_FP_GATE",
  "STATE_BITS_PER_VALUE": 8,
  "ROTATION_METADATA_BYTES": 0
}
```
