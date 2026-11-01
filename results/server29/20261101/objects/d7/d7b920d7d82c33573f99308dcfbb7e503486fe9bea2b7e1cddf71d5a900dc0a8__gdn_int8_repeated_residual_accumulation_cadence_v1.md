# GDN INT8 Repeated Residual Accumulation / Cadence Causal Experiment V1

## 1. Task
`GDN_INT8_REPEATED_RESIDUAL_ACCUMULATION_CADENCE_CAUSAL_EXPERIMENT_V1`

## 2. Scientific Question
Does increasing the cadence of natural canonical R128 recurrent-state quantization residuals cause systematic state drift, readout-visible drift, and future KL degradation?

## 3. Prior Evidence
Prior results establish the R128/C128 orientation effect, R128 degradation, R16 rescue, invalidity of old same-codebook lost-update proxies, runtime effective-update relevance, residual strength/direction causality, broad directional sensitivity, low-gain-biased natural R128 residuals, and insufficiency of single-pulse propagation/readout metrics. These are not re-proved here.

## 4. Intervention Definition
At each scheduled token the model first computes `S_pre`, then writes `S_post = Q_R128(S_pre)` into `DynamicCache.layers[layer_idx].recurrent_states[0]`. The residual is the natural `S_post - S_pre`; no rescaling, replacement, randomization, or fixed residual replay is used.

## 5. Protocol
- model: `Qwen3.5-9B`
- t0: `128`
- continuation_horizon: `128`
- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`
- exact_original_generation_token_ids_available: `false`
- causal variable: `R128 intervention cadence`

## 6. Stage-0 Gates
See `gdn_int8_repeated_residual_accumulation_cadence_v1_stage0.json`; all gates must pass before Pilot.

## 7. Smoke
Smoke ran FP_STATE / SINGLE / EVERY_8 / EVERY_1 and checks crash-free execution, hook stability, finite metrics, serialization, and checkpoint writes.

## 8. Pilot Configuration
- prompts: `3` / 3
- cadence conditions: `SINGLE / EVERY_32 / EVERY_16 / EVERY_8 / EVERY_4 / EVERY_2 / EVERY_1`

## 9. Injection Counts
Injection count is an exposure description, not a separate causal variable.

## 10. State Accumulation Results
Median rho(freq, state AUC): `0.9999999999999429`

## 11. Readout-Visible Accumulation
Median rho(freq, frozen readout AUC): `0.9999999999999429`
Median rho(freq, actual readout AUC): `0.9999999999999429`

## 12. KL Dose Response
Median rho(freq, KL AUC): `0.9999999999999429`

## 13. Per-Prompt Results
| Prompt | Cadence | Injection Count | AUC State Error | AUC Frozen Readout Error | AUC Actual Readout Error | AUC KL | Terminal KL | Top-1 Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test/algebra/1332.json | SINGLE | 1 | 0.014506627260725206 | 2.2289203794004777 | 3.097733865261265 | 0.0005514973365127371 | 9.490953800650459e-09 | 1.0 |
| test/algebra/1332.json | EVERY_32 | 4 | 0.020547323644664533 | 3.019720521973629 | 4.25199766355259 | 0.0008960602997546774 | 4.7652051193836087e-07 | 1.0 |
| test/algebra/1332.json | EVERY_16 | 8 | 0.023740908825426295 | 4.117213125210253 | 5.671545606638036 | 0.0025271082369043027 | 1.8225157418783056e-06 | 1.0 |
| test/algebra/1332.json | EVERY_8 | 16 | 0.026831051206851846 | 5.3809204234075425 | 7.493349997722267 | 0.003609804953156981 | 1.5513714970438741e-06 | 0.9921875 |
| test/algebra/1332.json | EVERY_4 | 32 | 0.028817711881765025 | 5.823403601151064 | 8.007645377010427 | 0.006373900971567811 | 1.3262155107440776e-06 | 0.9921875 |
| test/algebra/1332.json | EVERY_2 | 64 | 0.031620512681335235 | 7.067926350508066 | 9.493042676401831 | 0.009074665021120032 | 2.9725656531809364e-06 | 0.9921875 |
| test/algebra/1332.json | EVERY_1 | 128 | 0.0348374249263509 | 8.313076381518801 | 11.104510743250332 | 0.009483456678197218 | 4.856396571994992e-06 | 0.984375 |
| test/counting_and_probability/119.json | SINGLE | 1 | 0.016552381886880695 | 3.3045092427590967 | 4.494102571464426 | 0.0005779328986039627 | 3.4282886645087274e-06 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_32 | 4 | 0.021693212249775422 | 4.402243439750785 | 6.0491360603533995 | 0.0008464544854002036 | 7.662216376047581e-06 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_16 | 8 | 0.02412574071384165 | 5.004699894297973 | 6.9642509498683705 | 0.001377564710658411 | 3.6332041872810805e-06 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_8 | 16 | 0.025950152432214463 | 5.538095949495712 | 7.634675443303192 | 0.003270022553420895 | 4.284716851543635e-06 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_4 | 32 | 0.028541643422325215 | 6.885539180770916 | 9.66940847237855 | 0.005620383338187682 | 9.913879694067873e-06 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_2 | 64 | 0.03110103012920895 | 8.131365284979555 | 11.228531339100268 | 0.009611083972839569 | 0.0006497848662547767 | 0.9921875 |
| test/counting_and_probability/119.json | EVERY_1 | 128 | 0.034147602914416694 | 9.558694969758633 | 13.193223329801134 | 0.013891563203434296 | 0.0014691109536215663 | 0.984375 |
| test/geometry/477.json | SINGLE | 1 | 0.014945953494150501 | 2.364116741257619 | 3.0957398529466107 | 0.0010467427923719665 | 0.0001145062706200406 | 1.0 |
| test/geometry/477.json | EVERY_32 | 4 | 0.020496079770063236 | 3.2429285262740595 | 4.2616981298069865 | 0.0021285917645804155 | 0.00026944957789964974 | 0.984375 |
| test/geometry/477.json | EVERY_16 | 8 | 0.023337574142172652 | 3.829068287670652 | 5.007149883930189 | 0.002881071051229936 | 0.0007134731858968735 | 0.9921875 |
| test/geometry/477.json | EVERY_8 | 16 | 0.026002452137584845 | 4.640813952772259 | 6.034753344552027 | 0.005988310050938042 | 0.004868532530963421 | 0.984375 |
| test/geometry/477.json | EVERY_4 | 32 | 0.0283260164874896 | 5.193563028942287 | 6.654018282332598 | 0.006923291700269368 | 0.0023598838597536087 | 0.9921875 |
| test/geometry/477.json | EVERY_2 | 64 | 0.03145645431494965 | 6.392190154999646 | 8.116786496650075 | 0.01005064821338128 | 0.0007159532979130745 | 0.9609375 |
| test/geometry/477.json | EVERY_1 | 128 | 0.03477216603843417 | 7.637231585891671 | 9.651292729471344 | 0.01447524412054435 | 0.00027046288596466184 | 0.96875 |

## 14. Adjacent Dose Response
| Prompt | rho(freq, state AUC) | rho(freq, frozen-readout AUC) | rho(freq, actual-readout AUC) | rho(freq, KL AUC) | KL adjacent monotonic count | state adjacent monotonic count |
|---|---:|---:|---:|---:|---:|---:|
| test/algebra/1332.json | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 6/6 | 6/6 |
| test/counting_and_probability/119.json | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 6/6 | 6/6 |
| test/geometry/477.json | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 0.9999999999999429 | 6/6 | 6/6 |

## 15. State Accumulation vs KL
This is reported descriptively only; correlations among state AUC, readout AUC, and KL are not labeled as causal relations.

## 16. Negative / Corrective Findings
Any noisy, heterogeneous, or non-monotone prompt behavior is preserved in JSON and not smoothed away.

## 17. Pilot Classification
`POSITIVE`; FORMAL_READY = `YES`

## 18. What Is Supported
`REPEATED_RESIDUAL_ACCUMULATION_CADENCE_SIGNAL = SUPPORTED_IN_PILOT` only if Pilot is POSITIVE; otherwise no positive cadence conclusion is claimed.

## 19. What Is NOT Supported
This does not prove the full mechanism, network feedback, linear error addition, method readiness, bit allocation, mixed precision, high-gain direction protection, or quantization-frequency reduction as a final method.

## 20. Next Step
Formal repeated-accumulation validation
