# GDN INT8 Repeated Accumulation Formal V1

## Answers
1. Robust across 6 prompts: `SUPPORTED`.
2. Robust across t0={64,128,256}: `{'64': {'valid_units': 6, 'median_rho_KL': 0.9999999999999429, 'median_rho_state': 0.9999999999999429, 'median_KL_adjacent_count': 6.0, 'median_state_adjacent_count': 6.0}, '128': {'valid_units': 6, 'median_rho_KL': 0.971428571428516, 'median_rho_state': 0.9999999999999429, 'median_KL_adjacent_count': 5.5, 'median_state_adjacent_count': 6.0}, '256': {'valid_units': 6, 'median_rho_KL': 0.9428571428570891, 'median_rho_state': 0.9999999999999429, 'median_KL_adjacent_count': 5.0, 'median_state_adjacent_count': 6.0}}`.
3. State accumulation monotonicity median: `6.0 / 6`.
4. KL degradation monotonicity median: `6.0 / 6`.
5. Readout-visible monotonicity is diagnostic; median rho frozen/actual: `0.9999999999999429`, `0.9999999999999429`.
6. Failure-containing units: `0`.
7. R128 repeated accumulation formally supported: `SUPPORTED`.

## Formal Gate
- valid formal units: `18 / 18`
- median rho(freq, state AUC): `0.9999999999999429`
- median rho(freq, KL AUC): `0.9999999999999429`
- PHASE_B_READY: `YES`
- PROJECT_STAGE: `MECHANISM_VALIDATION`
- METHOD_DESIGN_READY: `NO`
