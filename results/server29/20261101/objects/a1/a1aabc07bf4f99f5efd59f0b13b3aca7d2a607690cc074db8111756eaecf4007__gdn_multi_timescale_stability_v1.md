# GDN_MULTI_TIMESCALE_STABILITY_VALIDATION_V1

FINAL_CLASSIFICATION: `SHORT_LONG_OBJECTIVE_SEPARATION_PARTIALLY_SUPPORTED`
METHOD_DESIGN_READY: `NO`
MULTI_TIMESCALE_GATE: `PASS`

## Row/Column Comparison
|track|checkpoint|row KL|column KL|column-row|row failures|column failures|
|-|-|-|-|-|-|-|
|teacher_forced|128|0.4222424483718781|0.025136747059578413|-0.39710570131229966|0|0|
|teacher_forced|256|0.5604408737939734|0.057696472571629215|-0.5027444012223442|0|0|
|teacher_forced|512|0.5562386110952018|0.36285828411549303|-0.19338032697970875|0|0|
|teacher_forced|1024|0.49162545635766863|0.4531005600439016|-0.03852489631376704|0|0|
|teacher_forced|2048|0.5967996976554583|3.6308088228665887|3.0340091252111305|0|0|
|teacher_forced|3072|0.9053549494027111|5.178940664993573|4.273585715590862|0|0|
|shadow_decode|128|0.3320448710994751|0.0209543425638301|-0.311090528535645|0|0|
|shadow_decode|256|0.2783730119042223|0.0399068747855076|-0.23846613711871473|0|0|
|shadow_decode|512|0.18316497387389344|0.10602906459042663|-0.07713590928346681|0|0|
|shadow_decode|1024|0.12032733940736089|0.529426050933298|0.4090987115259371|0|0|
|shadow_decode|2048|0.1951016314960688|1.3282321608880439|1.1331305293919751|0|0|
|shadow_decode|3072|0.2488377753580677|1.0437670232051708|0.7949292478471031|0|0|
|shadow_decode|4096|0.3722087561693546|1.1991265810205918|0.8269178248512372|0|0|
|shadow_decode|6144|0.5807166619168788|1.803006373403708|1.2222897114868294|0|0|

## Hypotheses
```json
{
  "H1_short_term_column_advantage": "SUPPORTED",
  "H2_short_long_objective_separation": "PARTIALLY_SUPPORTED",
  "H3_repeated_injection_driven_horizon_growth": "PARTIALLY_SUPPORTED",
  "H4_state_change_swamping": "SUPPORTED",
  "H5_layer25_failure_criticality": "INCONCLUSIVE",
  "H6_hybrid_position_structure": "INCONCLUSIVE"
}
```

## Files
```json
{
  "failure_traces": "/data/zypan/results/gdn_multi_timescale_stability_v1_failure_traces.jsonl.gz",
  "figures": "/data/zypan/figures/gdn_multi_timescale_stability_v1",
  "head_records": "/data/zypan/results/gdn_multi_timescale_stability_v1_head_records.jsonl.gz",
  "layer_records": "/data/zypan/results/gdn_multi_timescale_stability_v1_layer_records.jsonl.gz",
  "protocols": "/data/zypan/results/gdn_multi_timescale_stability_v1_protocols.json",
  "records": "/data/zypan/results/gdn_multi_timescale_stability_v1_records.jsonl",
  "report": "/data/zypan/reports/gdn_multi_timescale_stability_v1.md",
  "result": "/data/zypan/results/gdn_multi_timescale_stability_v1.json",
  "script": "/data/zypan/experiments/qwen35_gdn_quant/run_multi_timescale_stability_validation.py",
  "token_records": "/data/zypan/results/gdn_multi_timescale_stability_v1_token_records.jsonl.gz"
}
```
