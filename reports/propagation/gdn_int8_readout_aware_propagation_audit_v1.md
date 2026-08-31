# GDN INT8 Readout-Aware Propagation Audit V1

## Gates
- FORMAL_STATUS: `NOT_RUN`
- PROTOCOL_GATE: `PASS`
- READOUT_SENSITIVITY_MECHANISM: `NOT_EVALUATED`
- PILOT_CLASSIFICATION: `READOUT_AWARE_FIDELITY_SIGNAL_INCONCLUSIVE`
- FINAL_CLASSIFICATION: `READOUT_AWARE_FIDELITY_SIGNAL_INCONCLUSIVE`
- MECHANISM_CLOSURE_CANDIDATE: `NO`
- METHOD_DESIGN_READY_CANDIDATE: `NO`
- METHOD_DESIGN_READY: `NO`

## Predictor Ladder
```json
{
  "median_delta_rho_vs_state": {
    "actual_readout": 0.2117647058823523,
    "frozen_readout": 0.2617647058823521,
    "frozen_visibility": 0.17058823529411715,
    "postproj": 0.14705882352941135,
    "postproj_relative": 0.14705882352941135,
    "residual_stream": 0.017647058823529363
  },
  "median_rho": {
    "actual_readout": 0.2999999999999991,
    "frozen_readout": 0.3499999999999989,
    "frozen_visibility": 0.25882352941176395,
    "postproj": 0.23529411764705813,
    "postproj_relative": 0.23529411764705813,
    "residual_stream": 0.10588235294117615,
    "state": 0.08823529411764679
  },
  "metric_rho_greater_than_state_count": {
    "actual_readout": 1,
    "frozen_readout": 1,
    "frozen_visibility": 1,
    "postproj": 1,
    "postproj_relative": 1,
    "residual_stream": 1
  }
}
```
