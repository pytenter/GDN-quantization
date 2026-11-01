# GDN_STATE_CHANGE_PRESERVATION_INSTRUMENTED_REPLAY_V1

- FINAL_CLASSIFICATION: `STATE_CHANGE_SWAMPING_SIGNAL_IDENTIFIED`
- METHOD_DESIGN_READY: `NO`
- METRIC_GATE: `PASS`
- MECHANISM_SIGNAL_READY_FOR_CAUSAL_INTERVENTION: `YES`
- STATE_CHANGE_PRESERVATION_STATUS: `SWAMPING_DOMINANT`
- L25_H14_H15_STATUS: `EARLY_RISK_SIGNATURE`
- SCALE_FAILURE_STATUS: `INCONCLUSIVE`

## Quantizer Comparison

{"failure_tokens": [1395, null, 1521, null], "final_KL": 4.660127282142639, "lost_energy_fraction": 0.5374210210438525, "mean_KL": 3.1663212044613247, "prompt_count": 4, "quantizer": "INT4-column", "state_norm": 3.1630646809693616e+16, "survival_fraction": 0.13648198157031696, "update_cosine": 0.5175575769074625, "update_distortion": 0.6090612880138138, "update_norm_ratio": 0.6266749757219329}
{"failure_tokens": [null, null, null, null], "final_KL": 0.37834865297554643, "lost_energy_fraction": 0.3995689085725618, "mean_KL": 0.3611431432850647, "prompt_count": 4, "quantizer": "INT4-row", "state_norm": 168.88383970666717, "survival_fraction": 0.07371840894736677, "update_cosine": 0.7052655008843879, "update_distortion": 0.501385262273071, "update_norm_ratio": 0.831596143277324}
{"failure_tokens": [null, null, null, null], "final_KL": 0.0008308506853040853, "lost_energy_fraction": 0.0426357223027951, "mean_KL": 0.0013213664229555612, "prompt_count": 4, "quantizer": "INT8-column", "state_norm": 16.693238158511175, "survival_fraction": 0.4955341026169959, "update_cosine": 0.9635512317820328, "update_distortion": 0.07111006187666925, "update_norm_ratio": 0.9563775744462912}

## Conclusion

See quantizer summaries and L25 head map in JSON.
