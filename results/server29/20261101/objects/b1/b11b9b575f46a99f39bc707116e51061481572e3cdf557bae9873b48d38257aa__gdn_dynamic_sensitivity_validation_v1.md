# GDN Dynamic Sensitivity Validation V1

## Goal
Validate whether same-norm artificial recurrent-state perturbation sensitivity predicts real fake-quantizer degradation.

## Protocol
Teacher-forced FP32 greedy paths; recurrent state only; no model/source/runtime changes.
Layers: `[1, 9, 17, 25, 29]`; heads: `[0, 8]`.

## Correctness
```json
{
  "fp32_reference_reproducible": true,
  "fp32_repro_agreement": 1.0,
  "full_attention_layers_with_recurrent_state": [],
  "full_attention_recurrent_state_record_count": 0,
  "gdn_state_count": 24,
  "one_shot_norm_mismatch_max": 4.781006547224122e-08,
  "state_device": "cuda:0",
  "state_dtype": "torch.float32",
  "state_location": "DynamicCache.layers[layer_idx].recurrent_states[0]",
  "state_shape": [
    1,
    32,
    128,
    128
  ],
  "teacher_forced_token_path_length": 384
}
```

## Multi-Prompt Propagation Map
Top sensitive entries: `[{'layer_idx': 17, 'head_idx': 0, 'mean_G128': 0.8967128926783338, 'std_G128': 0.13335004247549392, 'mean_AUC': 0.9773032307131295, 'label': 'consistently_sensitive', 'prompt_count': 8}, {'layer_idx': 25, 'head_idx': 8, 'mean_G128': 0.832998615022733, 'std_G128': 0.08519255343828858, 'mean_AUC': 0.9625242548863997, 'label': 'consistently_sensitive', 'prompt_count': 8}, {'layer_idx': 29, 'head_idx': 8, 'mean_G128': 0.6957747467365168, 'std_G128': 0.2505536495028608, 'mean_AUC': 0.9276220539557734, 'label': 'trajectory_dependent', 'prompt_count': 8}, {'layer_idx': 25, 'head_idx': 0, 'mean_G128': 0.6456094716848235, 'std_G128': 0.2551181758489294, 'mean_AUC': 0.9287608899585809, 'label': 'trajectory_dependent', 'prompt_count': 8}, {'layer_idx': 29, 'head_idx': 0, 'mean_G128': 0.5115569512347942, 'std_G128': 0.23289323644702187, 'mean_AUC': 0.8901657018228823, 'label': 'trajectory_dependent', 'prompt_count': 8}, {'layer_idx': 17, 'head_idx': 8, 'mean_G128': 0.17437204543982868, 'std_G128': 0.1150241399971368, 'mean_AUC': 0.7722256417500962, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 9, 'head_idx': 8, 'mean_G128': 0.0005964014526441874, 'std_G128': 0.0009510425778416926, 'mean_AUC': 0.541513504883411, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 9, 'head_idx': 0, 'mean_G128': 0.0004323534649390647, 'std_G128': 0.0008825342707282641, 'mean_AUC': 0.5267857627215726, 'label': 'consistently_damping', 'prompt_count': 8}]`.

Strong damping entries: `[{'layer_idx': 1, 'head_idx': 0, 'mean_G128': 0.0, 'std_G128': 0.0, 'mean_AUC': 0.1427761091337848, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 1, 'head_idx': 8, 'mean_G128': 0.0, 'std_G128': 0.0, 'mean_AUC': 0.21830151776959858, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 9, 'head_idx': 0, 'mean_G128': 0.0004323534649390647, 'std_G128': 0.0008825342707282641, 'mean_AUC': 0.5267857627215726, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 9, 'head_idx': 8, 'mean_G128': 0.0005964014526441874, 'std_G128': 0.0009510425778416926, 'mean_AUC': 0.541513504883411, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 17, 'head_idx': 8, 'mean_G128': 0.17437204543982868, 'std_G128': 0.1150241399971368, 'mean_AUC': 0.7722256417500962, 'label': 'consistently_damping', 'prompt_count': 8}, {'layer_idx': 29, 'head_idx': 0, 'mean_G128': 0.5115569512347942, 'std_G128': 0.23289323644702187, 'mean_AUC': 0.8901657018228823, 'label': 'trajectory_dependent', 'prompt_count': 8}, {'layer_idx': 25, 'head_idx': 0, 'mean_G128': 0.6456094716848235, 'std_G128': 0.2551181758489294, 'mean_AUC': 0.9287608899585809, 'label': 'trajectory_dependent', 'prompt_count': 8}, {'layer_idx': 29, 'head_idx': 8, 'mean_G128': 0.6957747467365168, 'std_G128': 0.2505536495028608, 'mean_AUC': 0.9276220539557734, 'label': 'trajectory_dependent', 'prompt_count': 8}]`.

## Real Quantizer Error Profiles
```json
[
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.007752745705010615,
    "mean_logit_kl": 0.0003229378050898066,
    "mean_relative_error": 0.00921999532665706,
    "prompt_id": 0,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.05460327309583541,
    "mean_logit_kl": 0.022152693271754426,
    "mean_relative_error": 0.06452544515860459,
    "prompt_id": 0,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.96875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.07983986751655762,
    "mean_logit_kl": 0.055901263145458825,
    "mean_relative_error": 0.10346664712613507,
    "prompt_id": 0,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.96875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.14458369454394987,
    "mean_logit_kl": 0.07358785901505344,
    "mean_relative_error": 0.1584459594656954,
    "prompt_id": 0,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.953125
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06774881385850795,
    "mean_logit_kl": 0.0036856016783150215,
    "mean_relative_error": 0.08491998360770567,
    "prompt_id": 0,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.006501093230481741,
    "mean_logit_kl": 0.00018605226023609106,
    "mean_relative_error": 0.008842997923273083,
    "prompt_id": 1,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 0.984375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.041566201584748705,
    "mean_logit_kl": 0.00513794340018886,
    "mean_relative_error": 0.06325902185492174,
    "prompt_id": 1,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06047892429145663,
    "mean_logit_kl": 0.010887913048702957,
    "mean_relative_error": 0.10157307290777436,
    "prompt_id": 1,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.1120783974210514,
    "mean_logit_kl": 0.018774655148698693,
    "mean_relative_error": 0.1521406055728303,
    "prompt_id": 1,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.1219154293836214,
    "mean_logit_kl": 0.0017509143930490266,
    "mean_relative_error": 0.07903060784417087,
    "prompt_id": 1,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.007035407365631644,
    "mean_logit_kl": 0.00020460641405332936,
    "mean_relative_error": 0.009213423468767547,
    "prompt_id": 2,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06451757105828168,
    "mean_logit_kl": 0.032673719148045066,
    "mean_relative_error": 0.06580114511604125,
    "prompt_id": 2,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.9609375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.08548163661987962,
    "mean_logit_kl": 0.054366966740605704,
    "mean_relative_error": 0.1041752952006572,
    "prompt_id": 2,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.953125
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.1386574105591639,
    "mean_logit_kl": 0.06352732033931119,
    "mean_relative_error": 0.15682523986295335,
    "prompt_id": 2,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.953125
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.16318956362608927,
    "mean_logit_kl": 0.0018536794795378925,
    "mean_relative_error": 0.08130001504077493,
    "prompt_id": 2,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.00755135165424966,
    "mean_logit_kl": 0.00013891654318878442,
    "mean_relative_error": 0.009147424433427134,
    "prompt_id": 3,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.05037451369383615,
    "mean_logit_kl": 0.01147641799983402,
    "mean_relative_error": 0.06388865100863085,
    "prompt_id": 3,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.984375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.0761492510283434,
    "mean_logit_kl": 0.020208835259186514,
    "mean_relative_error": 0.10234916606593969,
    "prompt_id": 3,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.96875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.13773034282199823,
    "mean_logit_kl": 0.03651363003994733,
    "mean_relative_error": 0.1556688159224645,
    "prompt_id": 3,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.984375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06413548417317964,
    "mean_logit_kl": 0.003525200600307192,
    "mean_relative_error": 0.08437410208137598,
    "prompt_id": 3,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.006838943373720911,
    "mean_logit_kl": 0.000263959934891578,
    "mean_relative_error": 0.009006617430442374,
    "prompt_id": 4,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.05143394299480384,
    "mean_logit_kl": 0.024020552676173864,
    "mean_relative_error": 0.0652478337962398,
    "prompt_id": 4,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.07088078325150268,
    "mean_logit_kl": 0.018849862476563534,
    "mean_relative_error": 0.10444883734309109,
    "prompt_id": 4,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.11205926855328473,
    "mean_logit_kl": 0.02546794182922285,
    "mean_relative_error": 0.15952745074645772,
    "prompt_id": 4,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.984375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.05528272675274938,
    "mean_logit_kl": 0.001332530556169647,
    "mean_relative_error": 0.08396510716522283,
    "prompt_id": 4,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.007103471228436957,
    "mean_logit_kl": 0.00021638310165235548,
    "mean_relative_error": 0.009149664517556833,
    "prompt_id": 5,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.0625072517553174,
    "mean_logit_kl": 0.058378410606918396,
    "mean_relative_error": 0.06384447495287794,
    "prompt_id": 5,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.9375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.08292583156909306,
    "mean_logit_kl": 0.06077482375631238,
    "mean_relative_error": 0.10266194630174123,
    "prompt_id": 5,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.12620373248708028,
    "mean_logit_kl": 0.08021557651342015,
    "mean_relative_error": 0.1526745590340914,
    "prompt_id": 5,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06125778932537728,
    "mean_logit_kl": 0.0027595524427110885,
    "mean_relative_error": 0.08021714599711516,
    "prompt_id": 5,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.007350076918295424,
    "mean_logit_kl": 0.0008130177260232824,
    "mean_relative_error": 0.009309917250794877,
    "prompt_id": 6,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.07056017882038171,
    "mean_logit_kl": 0.1721065742923611,
    "mean_relative_error": 0.06782184425462157,
    "prompt_id": 6,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.08930970464524646,
    "mean_logit_kl": 0.19559976212705932,
    "mean_relative_error": 0.10695563261876007,
    "prompt_id": 6,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.8359375
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.15465228927470717,
    "mean_logit_kl": 0.45422567544635495,
    "mean_relative_error": 0.15888715287095498,
    "prompt_id": 6,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.7421875
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.10859914220977358,
    "mean_logit_kl": 0.015991878727834252,
    "mean_relative_error": 0.08391316618667458,
    "prompt_id": 6,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.0075322034619132,
    "mean_logit_kl": 0.0001817980454039362,
    "mean_relative_error": 0.008900495701758623,
    "prompt_id": 7,
    "quantizer": "INT8-column",
    "steps_completed": 128,
    "top1_agreement": 1.0
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.050996602173436024,
    "mean_logit_kl": 0.019659349326733827,
    "mean_relative_error": 0.06466487135965626,
    "prompt_id": 7,
    "quantizer": "INT5-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.06949388665726615,
    "mean_logit_kl": 0.021646192146460908,
    "mean_relative_error": 0.10344460892484794,
    "prompt_id": 7,
    "quantizer": "INT4-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.1074911502754503,
    "mean_logit_kl": 0.027339482248088576,
    "mean_relative_error": 0.15791569766041788,
    "prompt_id": 7,
    "quantizer": "INT3-row",
    "steps_completed": 128,
    "top1_agreement": 0.9765625
  },
  {
    "first_failure": null,
    "mean_global_state_divergence": 0.05343307152321665,
    "mean_logit_kl": 0.0030764048107695136,
    "mean_relative_error": 0.0829616645758608,
    "prompt_id": 7,
    "quantizer": "INT4-column",
    "steps_completed": 128,
    "top1_agreement": 0.9921875
  }
]
```

## Raw vs Propagation-Weighted Prediction
```json
[
  {
    "n": 40,
    "pearson": 0.40484900170581184,
    "predictor": "raw_mean_error",
    "spearman": 0.7170731707317073,
    "target": "target_mean_KL"
  },
  {
    "n": 40,
    "pearson": 0.4093401141410076,
    "predictor": "raw_mean_error",
    "spearman": 0.5790582705006347,
    "target": "target_top1_disagreement"
  },
  {
    "n": 40,
    "pearson": 0.878358093447952,
    "predictor": "raw_mean_error",
    "spearman": 0.8634146341463415,
    "target": "target_global_divergence"
  },
  {
    "n": 40,
    "pearson": 0.8515360444805551,
    "predictor": "raw_mean_error",
    "spearman": 0.8350844277673546,
    "target": "target_hidden_divergence"
  },
  {
    "n": 40,
    "pearson": 0.4084028447483181,
    "predictor": "raw_p95_error",
    "spearman": 0.673733583489681,
    "target": "target_mean_KL"
  },
  {
    "n": 40,
    "pearson": 0.40623525213352135,
    "predictor": "raw_p95_error",
    "spearman": 0.5653931785714162,
    "target": "target_top1_disagreement"
  },
  {
    "n": 40,
    "pearson": 0.8600441559471067,
    "predictor": "raw_p95_error",
    "spearman": 0.8604127579737336,
    "target": "target_global_divergence"
  },
  {
    "n": 40,
    "pearson": 0.8481793404857498,
    "predictor": "raw_p95_error",
    "spearman": 0.8191369606003752,
    "target": "target_hidden_divergence"
  },
  {
    "n": 40,
    "pearson": 0.11730667939161969,
    "predictor": "pw_G128_mean",
    "spearman": 0.43696060037523454,
    "target": "target_mean_KL"
  },
  {
    "n": 40,
    "pearson": 0.10456647145683766,
    "predictor": "pw_G128_mean",
    "spearman": 0.2936096835347368,
    "target": "target_top1_disagreement"
  },
  {
    "n": 40,
    "pearson": 0.7134955124645663,
    "predictor": "pw_G128_mean",
    "spearman": 0.6153846153846154,
    "target": "target_global_divergence"
  },
  {
    "n": 40,
    "pearson": 0.7096508441187583,
    "predictor": "pw_G128_mean",
    "spearman": 0.6232645403377111,
    "target": "target_hidden_divergence"
  },
  {
    "n": 40,
    "pearson": 0.4084489389565101,
    "predictor": "pw_AUC_mean",
    "spearman": 0.7056285178236398,
    "target": "target_mean_KL"
  },
  {
    "n": 40,
    "pearson": 0.4330425548711502,
    "predictor": "pw_AUC_mean",
    "spearman": 0.6074323155481093,
    "target": "target_top1_disagreement"
  },
  {
    "n": 40,
    "pearson": 0.8640222176168314,
    "predictor": "pw_AUC_mean",
    "spearman": 0.8519699812382739,
    "target": "target_global_divergence"
  },
  {
    "n": 40,
    "pearson": 0.8520658246107284,
    "predictor": "pw_AUC_mean",
    "spearman": 0.8326454033771107,
    "target": "target_hidden_divergence"
  }
]
```

## Row vs Column Explanation
Row and column differences are reported as measured real fake-quantization error distributions plus KL/top1/global divergence. No new algorithm is proposed.

## Cadence Multi-Prompt Confirmation
```json
[
  {
    "cadence": 1,
    "failures": 0,
    "mean_KL": 0.2474213619729797,
    "mean_global_divergence": 0.15712814899953378,
    "mean_top1_agreement": 0.875,
    "prompt_count": 8
  },
  {
    "cadence": 4,
    "failures": 0,
    "mean_KL": 0.18387492184304696,
    "mean_global_divergence": 0.12909128703570524,
    "mean_top1_agreement": 0.892578125,
    "prompt_count": 8
  },
  {
    "cadence": 16,
    "failures": 0,
    "mean_KL": 0.13861891438253443,
    "mean_global_divergence": 0.09878978464037433,
    "mean_top1_agreement": 0.916015625,
    "prompt_count": 8
  },
  {
    "cadence": 32,
    "failures": 0,
    "mean_KL": 0.11404950290890382,
    "mean_global_divergence": 0.08399014364262529,
    "mean_top1_agreement": 0.916015625,
    "prompt_count": 8
  }
]
```

## Static-Metric Negative Results
```json
[
  {
    "metric": "state_norm",
    "n": 80,
    "pearson": 0.629593014883155,
    "spearman": 0.7304365358405499,
    "target": "G128"
  },
  {
    "metric": "rms",
    "n": 80,
    "pearson": 0.6295930117276247,
    "spearman": 0.7304365358405499,
    "target": "G128"
  },
  {
    "metric": "max_abs",
    "n": 80,
    "pearson": 0.4917702544951916,
    "spearman": 0.6579365439389899,
    "target": "G128"
  },
  {
    "metric": "mean_abs",
    "n": 80,
    "pearson": 0.6301852091499854,
    "spearman": 0.7640917828521488,
    "target": "G128"
  },
  {
    "metric": "sv_top1_energy",
    "n": 80,
    "pearson": 0.09119623044173891,
    "spearman": -0.019157602145064043,
    "target": "G128"
  },
  {
    "metric": "sv_top4_energy",
    "n": 80,
    "pearson": -0.22889905339326722,
    "spearman": -0.4955323257546971,
    "target": "G128"
  },
  {
    "metric": "sv_top8_energy",
    "n": 80,
    "pearson": -0.27234468991167887,
    "spearman": -0.5833702814086688,
    "target": "G128"
  },
  {
    "metric": "effective_rank_entropy",
    "n": 80,
    "pearson": 0.1646075422398861,
    "spearman": 0.2428473190833456,
    "target": "G128"
  }
]
```

## Hybrid Attention Summary
```json
{
  "available": true,
  "hybrid_attention_summary": null,
  "source": "/data/zypan/results/gdn_state_sensitivity_structure_v1.json"
}
```

## Limitations
This is a validation-stage run. It may use a sampled head grid when the full 8 prompt x 5 layer x 32 head map is too expensive. No LongBench is run.

## Research Decision
```json
{
  "current_strongest_research_conclusion": "This run validates whether artificial propagation sensitivity explains real quantizer degradation better than raw reconstruction error. It is evidence gathering, not method design.",
  "final_classification": "DYNAMIC_SENSITIVITY_PARTIALLY_VALIDATED",
  "method_design_ready": "NO",
  "propagation_weighted_prediction_result": 0.43696060037523454,
  "propagation_weighting_improves_KL_prediction": false,
  "raw_error_prediction_result": 0.7170731707317073,
  "recommended_next_step": "Repeat with a denser head grid or longer paths before designing adaptive precision."
}
```
