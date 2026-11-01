# Static Reorder / Pack Feasibility

```json
{
  "STATIC_REORDER_FEASIBLE": "CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED",
  "must_permute": [
    "value projection output/value state V axis",
    "recurrent state Value dimension",
    "out_proj input slices aligned by head/value"
  ],
  "note": "Formal task audited static packing feasibility but did not implement CUDA kernel or final quantizer.",
  "permutation_identity_test": "NOT_RUN_NO_KERNEL_IMPLEMENTATION"
}
```
