# Efficiency Audit

```json
{
  "DYNAMIC_GATHER_SCATTER": "NO in static packed layout; YES in prototype mask implementation",
  "EXTRA_FULL_STATE_PASS": 0,
  "EXTRA_STATE_READ_BYTES": "prototype reads FP branch for causal intervention; algorithm would keep protected segment natively",
  "FUSABLE_SINGLE_PASS": "YES_CONDITIONAL",
  "ONLINE_FUNCTIONAL_SCORE": "NO",
  "ONLINE_RANKING": "NO",
  "STATIC_LAYOUT_FEASIBLE": "CONDITIONAL",
  "algorithmic_runtime_overhead": "same recurrent update plus fixed mixed-precision state write; no online scoring",
  "future_optimized_kernel_feasibility": "STATIC_LAYOUT_CONDITIONAL_FEASIBLE",
  "prototype_implementation_overhead": "Python branch replay and torch.where masks are not kernel latency",
  "rows": [
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 16,
      "protection_ratio": 0.125,
      "raw_data_bytes_per_full_gdn_state_estimate": 14155776.0,
      "raw_state_bits_per_value": 9.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.0009765625
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 32,
      "protection_ratio": 0.25,
      "raw_data_bytes_per_full_gdn_state_estimate": 15728640.0,
      "raw_state_bits_per_value": 10.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.001953125
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 48,
      "protection_ratio": 0.375,
      "raw_data_bytes_per_full_gdn_state_estimate": 17301504.0,
      "raw_state_bits_per_value": 11.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.0029296875
    },
    {
      "permutation_metadata_bytes_per_layer_head": 256,
      "protected_columns_per_head": 64,
      "protection_ratio": 0.5,
      "raw_data_bytes_per_full_gdn_state_estimate": 18874368.0,
      "raw_state_bits_per_value": 12.0,
      "scale_metadata_bytes_per_layer_head": 256,
      "selector_metadata_bits_per_value_estimate": 0.00390625
    }
  ]
}
```
