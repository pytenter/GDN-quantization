# GDN INT8 Effective Update Small Prompt Validation V1

## Observation
- FORMAL_STATUS: `COMPLETE`
- PROTOCOL_GATE: `PASS`; METRIC_GATE: `PASS`
- METHOD_DESIGN_READY: `NO`
- FINAL_CLASSIFICATION: `EFFECTIVE_UPDATE_METRIC_MULTI_PROMPT_SUPPORTED`

## Prompt Manifest
```json
[
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 1909,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/algebra/1332.json",
    "role": "INT8-row truncated pathological candidate",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 914,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/algebra/1214.json",
    "role": "INT8-row non-truncated termination control",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 4536,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/counting_and_probability/119.json",
    "role": "INT8-row truncated pathological candidate",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 5139,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/intermediate_algebra/207.json",
    "role": "INT8-row non-truncated termination control",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 5118,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/geometry/477.json",
    "role": "INT8-row truncated pathological candidate",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "continuation_token_count_available": 18813,
    "exact_original_generation_token_ids_available": false,
    "exact_replay_claimed": false,
    "prompt_id": "test/geometry/702.json",
    "role": "INT8-row truncated pathological candidate",
    "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
    "source_path": "<GDN_DATA_ROOT>/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
  }
]
```

## Consistency Table
```json
{
  "C128_to_C16": {
    "D_effective": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "KL": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "R_distortion_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "R_lost_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "cos_effective": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "cosine_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "orthogonal_ratio": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "state_reconstruction_relative_error_E_S": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    }
  },
  "R128_to_R16": {
    "D_effective": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "KL": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "R_distortion_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "R_lost_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "cos_effective": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "cosine_same_codebook": {
      "rate": 0.0,
      "same_direction_count": 0,
      "valid_prompt_count": 6
    },
    "orthogonal_ratio": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    },
    "state_reconstruction_relative_error_E_S": {
      "rate": 1.0,
      "same_direction_count": 6,
      "valid_prompt_count": 6
    }
  }
}
```

## Pairwise Rescue
```json
[
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.02944987610621578,
        "Delta_E_S": -0.00337350603536493,
        "Delta_KL": -0.00041316661565083096,
        "Delta_abs_alpha": -0.0013071642981562486,
        "Delta_abs_norm_ratio_minus_1": -0.0008053539124703857,
        "Delta_cos_effective": 0.002105727840388316,
        "Delta_orthogonal_ratio": -0.029408167737290825
      },
      "metrics": {
        "D_effective": {
          "C128": 0.0722012530874067,
          "C16": 0.04275137698119092,
          "C16_minus_C128": -0.02944987610621578,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.0009054763947956287,
          "C16": 0.0004923097791447977,
          "C16_minus_C128": -0.00041316661565083096,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.06124043507003784,
          "C16": 0.13863714631869709,
          "C16_minus_C128": 0.07739671124865924,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.03607998771443544,
          "C16": 0.10009919258881021,
          "C16_minus_C128": 0.06401920487437476,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.5768330506227105,
          "C16": 0.6574036173002437,
          "C16_minus_C128": 0.08057056667753315,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.0017833629956718836,
          "C16": -0.00047619869751563504,
          "C16_minus_C128": 0.0013071642981562486,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9966958431260923,
          "C16": 0.9988015709664806,
          "C16_minus_C128": 0.002105727840388316,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9686186634106185,
          "C16": 0.9268837593257698,
          "C16_minus_C128": -0.0417349040848487,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.9982166370043276,
          "C16": 0.9995238013024841,
          "C16_minus_C128": 0.0013071642981564713,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.0015297082436365,
          "C16": 1.0007243543311661,
          "C16_minus_C128": -0.0008053539124703857,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.963346796323126,
          "C16": 0.9074456516302101,
          "C16_minus_C128": -0.05590114469291596,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.0721512320917072,
          "C16": 0.04274306435441637,
          "C16_minus_C128": -0.029408167737290825,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.0722012530874067,
          "C16": 0.04275137698119092,
          "C16_minus_C128": -0.02944987610621578,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.008128743745961396,
          "C16": 0.004755237710596466,
          "C16_minus_C128": -0.00337350603536493,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.99609375,
          "C16": 0.998046875,
          "C16_minus_C128": 0.001953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/algebra/1332.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.047998119785081554,
        "Delta_E_S": -0.006025446673020473,
        "Delta_KL": -0.041861184017958804,
        "Delta_abs_alpha": -0.002019304053237907,
        "Delta_abs_norm_ratio_minus_1": -0.0017135364050426816,
        "Delta_cos_effective": 0.0037130638859818044,
        "Delta_orthogonal_ratio": -0.047950624518430185
      },
      "metrics": {
        "D_effective": {
          "R128": 0.08803333445056155,
          "R16": 0.04003521466548,
          "R16_minus_R128": -0.047998119785081554,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.04548890240825434,
          "R16": 0.0036277183902955336,
          "R16_minus_R128": -0.041861184017958804,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.16819663661361292,
          "R16": 0.21969030151399557,
          "R16_minus_R128": 0.05149366490038265,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.12790876524815967,
          "R16": 0.17810101441753484,
          "R16_minus_R128": 0.050192249169375175,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.46001865516981105,
          "R16": 0.6412383929935067,
          "R16_minus_R128": 0.18121973782369566,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.0024689195578258384,
          "R16": -0.0004496155045879314,
          "R16_minus_R128": 0.002019304053237907,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9952373231304912,
          "R16": 0.998950387016473,
          "R16_minus_R128": 0.0037130638859818044,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9094661787876448,
          "R16": 0.8798198125070594,
          "R16_minus_R128": -0.029646366280585368,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.9975310804421749,
          "R16": 0.9995503844954127,
          "R16_minus_R128": 0.0020193040532378026,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.002314997368261,
          "R16": 1.0006014609632183,
          "R16_minus_R128": -0.0017135364050426816,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.9011130339783663,
          "R16": 0.8581968780807796,
          "R16_minus_R128": -0.04291615589758668,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.08798146614240358,
          "R16": 0.0400308416239734,
          "R16_minus_R128": -0.047950624518430185,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.08803333445056155,
          "R16": 0.04003521466548,
          "R16_minus_R128": -0.047998119785081554,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.010535141899127196,
          "R16": 0.004509695226106723,
          "R16_minus_R128": -0.006025446673020473,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.955078125,
          "R16": 0.986328125,
          "R16_minus_R128": 0.03125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/algebra/1332.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  },
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.029475060627673008,
        "Delta_E_S": -0.0034122741496746553,
        "Delta_KL": -0.0004607120213389423,
        "Delta_abs_alpha": -0.00123732437986492,
        "Delta_abs_norm_ratio_minus_1": -0.0008697166043509785,
        "Delta_cos_effective": 0.0020991983906707423,
        "Delta_orthogonal_ratio": -0.029438585658607
      },
      "metrics": {
        "D_effective": {
          "C128": 0.07215482003878274,
          "C16": 0.04267975941110973,
          "C16_minus_C128": -0.029475060627673008,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.0007783030138381004,
          "C16": 0.00031759099249915814,
          "C16_minus_C128": -0.0004607120213389423,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.05988769761916158,
          "C16": 0.13550308619158752,
          "C16_minus_C128": 0.07561538857242595,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.035299365097021675,
          "C16": 0.09694481633891841,
          "C16_minus_C128": 0.061645451241896736,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.5782347927665843,
          "C16": 0.6590276003972583,
          "C16_minus_C128": 0.08079280763067398,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.0016829053515592402,
          "C16": -0.00044558097169432017,
          "C16_minus_C128": 0.00123732437986492,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9967212954078013,
          "C16": 0.998820493798472,
          "C16_minus_C128": 0.0020991983906707423,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9693434861745628,
          "C16": 0.9287311731116571,
          "C16_minus_C128": -0.04061231306290569,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.9983170946484401,
          "C16": 0.9995544190283063,
          "C16_minus_C128": 0.0012373243798662337,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.00160587523147,
          "C16": 1.000736158627119,
          "C16_minus_C128": -0.0008697166043509785,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.9643604836994539,
          "C16": 0.9089923341061633,
          "C16_minus_C128": -0.055368149593290616,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.07211181222109572,
          "C16": 0.042673226562488724,
          "C16_minus_C128": -0.029438585658607,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.07215482003878274,
          "C16": 0.04267975941110973,
          "C16_minus_C128": -0.029475060627673008,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.008217233788793155,
          "C16": 0.0048049596391185,
          "C16_minus_C128": -0.0034122741496746553,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.994140625,
          "C16": 0.99609375,
          "C16_minus_C128": 0.001953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/counting_and_probability/119.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.047835935097723395,
        "Delta_E_S": -0.006145223559745285,
        "Delta_KL": -0.02294984811158313,
        "Delta_abs_alpha": -0.00202990248681497,
        "Delta_abs_norm_ratio_minus_1": -0.0016818597769758803,
        "Delta_cos_effective": 0.0036935873198411517,
        "Delta_orthogonal_ratio": -0.04778872427164779
      },
      "metrics": {
        "D_effective": {
          "R128": 0.08788233674338106,
          "R16": 0.04004640164565766,
          "R16_minus_R128": -0.047835935097723395,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.025130137566665805,
          "R16": 0.0021802894550826753,
          "R16_minus_R128": -0.02294984811158313,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.16353702056859556,
          "R16": 0.21540829910495132,
          "R16_minus_R128": 0.05187127853635576,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.1219357805736717,
          "R16": 0.17259961985307506,
          "R16_minus_R128": 0.050663839279403364,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.4635912697064825,
          "R16": 0.6425168738206528,
          "R16_minus_R128": 0.1789256041141703,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.0024697588416406247,
          "R16": -0.0004398563548256547,
          "R16_minus_R128": 0.00202990248681497,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9952675894835927,
          "R16": 0.9989611768034339,
          "R16_minus_R128": 0.0036935873198411517,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9121991262199524,
          "R16": 0.8823999398365489,
          "R16_minus_R128": -0.02979918638340351,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.99753024115836,
          "R16": 0.9995601436451742,
          "R16_minus_R128": 0.002029902486814228,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.002282182120473,
          "R16": 1.0006003223434972,
          "R16_minus_R128": -0.0016818597769758803,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.9027380098752897,
          "R16": 0.8599325548159289,
          "R16_minus_R128": -0.04280545505936084,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.08783118115136383,
          "R16": 0.04004245687971604,
          "R16_minus_R128": -0.04778872427164779,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.08788233674338106,
          "R16": 0.04004640164565766,
          "R16_minus_R128": -0.047835935097723395,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.010684527245495129,
          "R16": 0.004539303685749844,
          "R16_minus_R128": -0.006145223559745285,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.97265625,
          "R16": 0.984375,
          "R16_minus_R128": 0.01171875,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/counting_and_probability/119.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  },
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.027326707987394054,
        "Delta_E_S": -0.0036252047445083646,
        "Delta_KL": -0.0007552045787667799,
        "Delta_abs_alpha": -0.0009467975077903845,
        "Delta_abs_norm_ratio_minus_1": -0.0008081698830937345,
        "Delta_cos_effective": 0.001748763204631243,
        "Delta_orthogonal_ratio": -0.02730155610800577
      },
      "metrics": {
        "D_effective": {
          "C128": 0.06535783621184635,
          "C16": 0.038031128224452294,
          "C16_minus_C128": -0.027326707987394054,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.0014488834232867305,
          "C16": 0.0006936788445199506,
          "C16_minus_C128": -0.0007552045787667799,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.05963331609891871,
          "C16": 0.1379934831978864,
          "C16_minus_C128": 0.07836016709896769,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.034289028048348026,
          "C16": 0.09861407022994564,
          "C16_minus_C128": 0.06432504218159762,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.6149157308241319,
          "C16": 0.6916834790179408,
          "C16_minus_C128": 0.07676774819380894,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.0012392558285259153,
          "C16": -0.00029245832073553074,
          "C16_minus_C128": 0.0009467975077903845,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9973443667346653,
          "C16": 0.9990931299392966,
          "C16_minus_C128": 0.001748763204631243,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9694625980184892,
          "C16": 0.9272750533888784,
          "C16_minus_C128": -0.04218754462961083,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.9987607441714751,
          "C16": 0.9997075416792639,
          "C16_minus_C128": 0.0009467975077888102,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.0014238759754122,
          "C16": 1.0006157060923184,
          "C16_minus_C128": -0.0008081698830937345,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.9623605791214842,
          "C16": 0.9065490180382764,
          "C16_minus_C128": -0.05581156108320784,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.06532862652773083,
          "C16": 0.03802707041972506,
          "C16_minus_C128": -0.02730155610800577,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.06535783621184635,
          "C16": 0.038031128224452294,
          "C16_minus_C128": -0.027326707987394054,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.008597451172558144,
          "C16": 0.004972246428049779,
          "C16_minus_C128": -0.0036252047445083646,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.98828125,
          "C16": 0.990234375,
          "C16_minus_C128": 0.001953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/geometry/477.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.04470238171088075,
        "Delta_E_S": -0.006566089814774073,
        "Delta_KL": -0.030707537143337908,
        "Delta_abs_alpha": -0.0015087155129827798,
        "Delta_abs_norm_ratio_minus_1": -0.0015694591553880866,
        "Delta_cos_effective": 0.003065417487288502,
        "Delta_orthogonal_ratio": -0.04467567699779352
      },
      "metrics": {
        "D_effective": {
          "R128": 0.08079930294243322,
          "R16": 0.03609692123155247,
          "R16_minus_R128": -0.04470238171088075,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.03448620815562148,
          "R16": 0.003778671012283571,
          "R16_minus_R128": -0.030707537143337908,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.17137632757578483,
          "R16": 0.22528054231620387,
          "R16_minus_R128": 0.05390421474041904,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.12915805790853488,
          "R16": 0.18179679217211703,
          "R16_minus_R128": 0.05263873426358215,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.49096460896073324,
          "R16": 0.6726116158957589,
          "R16_minus_R128": 0.18164700693502567,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.001819227363222647,
          "R16": -0.000310511850239867,
          "R16_minus_R128": 0.0015087155129827798,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9961113536126539,
          "R16": 0.9991767710999424,
          "R16_minus_R128": 0.003065417487288502,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9081379852834205,
          "R16": 0.8769096054734545,
          "R16_minus_R128": -0.031228379809965956,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.9981807726367771,
          "R16": 0.9996894881497601,
          "R16_minus_R128": 0.0015087155129830387,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.0020830141978325,
          "R16": 1.0005135550424444,
          "R16_minus_R128": -0.0015694591553880866,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.8966801639430105,
          "R16": 0.8534532650680091,
          "R16_minus_R128": -0.04322689887500142,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.08077045651073213,
          "R16": 0.03609477951293861,
          "R16_minus_R128": -0.04467567699779352,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.08079930294243322,
          "R16": 0.03609692123155247,
          "R16_minus_R128": -0.04470238171088075,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.011321012614897789,
          "R16": 0.004754922800123716,
          "R16_minus_R128": -0.006566089814774073,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.947265625,
          "R16": 0.98828125,
          "R16_minus_R128": 0.041015625,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/geometry/477.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  },
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.027354103450473614,
        "Delta_E_S": -0.003595820160280599,
        "Delta_KL": -0.0007917616707781376,
        "Delta_abs_alpha": -0.0009643004623967047,
        "Delta_abs_norm_ratio_minus_1": -0.0008073182339991547,
        "Delta_cos_effective": 0.0017652492836580658,
        "Delta_orthogonal_ratio": -0.027329223710984285
      },
      "metrics": {
        "D_effective": {
          "C128": 0.0658266306600779,
          "C16": 0.03847252720960429,
          "C16_minus_C128": -0.027354103450473614,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.0011744850072797144,
          "C16": 0.0003827233365015769,
          "C16_minus_C128": -0.0007917616707781376,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.05944417247652481,
          "C16": 0.1365609364617752,
          "C16_minus_C128": 0.07711676398525039,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.034273192289657306,
          "C16": 0.09717956151652103,
          "C16_minus_C128": 0.06290636922686373,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.6151172550223811,
          "C16": 0.6926669469473803,
          "C16_minus_C128": 0.07754969192499916,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.0012780631818946773,
          "C16": -0.00031376271949797274,
          "C16_minus_C128": 0.0009643004623967047,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9972959929969051,
          "C16": 0.9990612422805631,
          "C16_minus_C128": 0.0017652492836580658,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9695703349667318,
          "C16": 0.9281040655068724,
          "C16_minus_C128": -0.04146626945985943,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.9987219368181055,
          "C16": 0.999686237280502,
          "C16_minus_C128": 0.0009643004623964835,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.0014337303168495,
          "C16": 1.0006264120828503,
          "C16_minus_C128": -0.0008073182339991547,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.9626561826915124,
          "C16": 0.9071542637318798,
          "C16_minus_C128": -0.055501918959632635,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.06579732878733152,
          "C16": 0.038468105076347237,
          "C16_minus_C128": -0.027329223710984285,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.0658266306600779,
          "C16": 0.03847252720960429,
          "C16_minus_C128": -0.027354103450473614,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.008576368873974624,
          "C16": 0.004980548713694025,
          "C16_minus_C128": -0.003595820160280599,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.98828125,
          "C16": 0.998046875,
          "C16_minus_C128": 0.009765625,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/algebra/1214.json",
      "role": "INT8-row non-truncated termination control",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.044724477620943975,
        "Delta_E_S": -0.006443706487943017,
        "Delta_KL": -0.03346307866287211,
        "Delta_abs_alpha": -0.0015640839596850827,
        "Delta_abs_norm_ratio_minus_1": -0.0015688583885646867,
        "Delta_cos_effective": 0.0031191175153496564,
        "Delta_orthogonal_ratio": -0.044695105088257345
      },
      "metrics": {
        "D_effective": {
          "R128": 0.0809866407143991,
          "R16": 0.03626216309345512,
          "R16_minus_R128": -0.044724477620943975,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.03788858313139271,
          "R16": 0.0044255044685205976,
          "R16_minus_R128": -0.03346307866287211,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.16723298481450483,
          "R16": 0.2198194213799348,
          "R16_minus_R128": 0.05258643656542997,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.12535969924054566,
          "R16": 0.17628428561117881,
          "R16_minus_R128": 0.05092458637063316,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.4939106392533813,
          "R16": 0.6764307683072526,
          "R16_minus_R128": 0.18252012905387127,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.0018857730063752223,
          "R16": -0.00032168904669013955,
          "R16_minus_R128": 0.0015640839596850827,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9960431120779418,
          "R16": 0.9991622295932915,
          "R16_minus_R128": 0.0031191175153496564,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9101424845177328,
          "R16": 0.8797607662208466,
          "R16_minus_R128": -0.030381718296886184,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.9981142269936251,
          "R16": 0.9996783109533098,
          "R16_minus_R128": 0.00156408395968477,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.0020858391369345,
          "R16": 1.0005169807483698,
          "R16_minus_R128": -0.0015688583885646867,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.8982407242215904,
          "R16": 0.8559828679560101,
          "R16_minus_R128": -0.04225785626558032,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.0809548792451648,
          "R16": 0.036259774156907454,
          "R16_minus_R128": -0.044695105088257345,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.0809866407143991,
          "R16": 0.03626216309345512,
          "R16_minus_R128": -0.044724477620943975,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.011183638887445538,
          "R16": 0.004739932399502521,
          "R16_minus_R128": -0.006443706487943017,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.95703125,
          "R16": 0.9765625,
          "R16_minus_R128": 0.01953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/algebra/1214.json",
      "role": "INT8-row non-truncated termination control",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  },
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.02923066181904877,
        "Delta_E_S": -0.0034111005197101094,
        "Delta_KL": -0.0006875099352505519,
        "Delta_abs_alpha": -0.0011706991405187337,
        "Delta_abs_norm_ratio_minus_1": -0.000879103589912722,
        "Delta_cos_effective": 0.0020418852295505108,
        "Delta_orthogonal_ratio": -0.029198929520391062
      },
      "metrics": {
        "D_effective": {
          "C128": 0.07121236892722531,
          "C16": 0.04198170710817654,
          "C16_minus_C128": -0.02923066181904877,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.001114190329250089,
          "C16": 0.000426680393999537,
          "C16_minus_C128": -0.0006875099352505519,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.06010849960911793,
          "C16": 0.13552686797239857,
          "C16_minus_C128": 0.07541836836328064,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.0357051944916254,
          "C16": 0.09804271373568019,
          "C16_minus_C128": 0.062337519244054784,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.5825160237130041,
          "C16": 0.6638764711431017,
          "C16_minus_C128": 0.08136044743009752,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.001587176374723294,
          "C16": -0.0004164772342045603,
          "C16_minus_C128": 0.0011706991405187337,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9968270668535166,
          "C16": 0.9988689520830671,
          "C16_minus_C128": 0.0020418852295505108,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9692249012757429,
          "C16": 0.9286515281085511,
          "C16_minus_C128": -0.040573373167191784,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.9984128236252764,
          "C16": 0.9995835227657951,
          "C16_minus_C128": 0.001170699140518705,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.001595664766674,
          "C16": 1.0007165611767612,
          "C16_minus_C128": -0.000879103589912722,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.9642320550101664,
          "C16": 0.9091983395511187,
          "C16_minus_C128": -0.05503371545904767,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.07117454143450055,
          "C16": 0.04197561191410949,
          "C16_minus_C128": -0.029198929520391062,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.07121236892722531,
          "C16": 0.04198170710817654,
          "C16_minus_C128": -0.02923066181904877,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.008222085633030748,
          "C16": 0.004810985113320639,
          "C16_minus_C128": -0.0034111005197101094,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.99609375,
          "C16": 0.99609375,
          "C16_minus_C128": 0.0,
          "metric_improved_with_KL_rescue": "NO"
        }
      },
      "problem_id": "test/intermediate_algebra/207.json",
      "role": "INT8-row non-truncated termination control",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.04764269708880945,
        "Delta_E_S": -0.006130052455941287,
        "Delta_KL": -0.02861710457191615,
        "Delta_abs_alpha": -0.0019739116292272992,
        "Delta_abs_norm_ratio_minus_1": -0.001694003125518817,
        "Delta_cos_effective": 0.003649166856258934,
        "Delta_orthogonal_ratio": -0.04759776376822096
      },
      "metrics": {
        "D_effective": {
          "R128": 0.08725541694591879,
          "R16": 0.03961271985710934,
          "R16_minus_R128": -0.04764269708880945,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.031137635541929137,
          "R16": 0.0025205309700129853,
          "R16_minus_R128": -0.02861710457191615,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.16732076144320535,
          "R16": 0.21929244373782336,
          "R16_minus_R128": 0.051971682294618016,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.12581273909876806,
          "R16": 0.17716518640600393,
          "R16_minus_R128": 0.05135244730723587,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.4631619952662816,
          "R16": 0.6466246631047496,
          "R16_minus_R128": 0.18346266783846804,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.0023885386553356243,
          "R16": -0.00041462702610832507,
          "R16_minus_R128": 0.0019739116292272992,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9953380536144655,
          "R16": 0.9989872204707244,
          "R16_minus_R128": 0.003649166856258934,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9101761648211509,
          "R16": 0.8802139913688789,
          "R16_minus_R128": -0.029962173452272034,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.9976114613446648,
          "R16": 0.9995853729738916,
          "R16_minus_R128": 0.0019739116292267767,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.002293462752024,
          "R16": 1.000599459626505,
          "R16_minus_R128": -0.001694003125518817,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.9008912438036891,
          "R16": 0.8582321171338941,
          "R16_minus_R128": -0.04265912666979499,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.08720701174751028,
          "R16": 0.039609247979289317,
          "R16_minus_R128": -0.04759776376822096,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.08725541694591879,
          "R16": 0.03961271985710934,
          "R16_minus_R128": -0.04764269708880945,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.01069320601647537,
          "R16": 0.004563153560534083,
          "R16_minus_R128": -0.006130052455941287,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.96484375,
          "R16": 0.994140625,
          "R16_minus_R128": 0.029296875,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/intermediate_algebra/207.json",
      "role": "INT8-row non-truncated termination control",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  },
  {
    "C128_to_C16": {
      "deltas": {
        "Delta_D_effective": -0.027969898291808452,
        "Delta_E_S": -0.0032278926571065805,
        "Delta_KL": -0.0003420691108164867,
        "Delta_abs_alpha": -0.0011567184069902789,
        "Delta_abs_norm_ratio_minus_1": -0.0008023306419431275,
        "Delta_cos_effective": 0.0019516862749033859,
        "Delta_orthogonal_ratio": -0.027936039748767252
      },
      "metrics": {
        "D_effective": {
          "C128": 0.06936660940868282,
          "C16": 0.04139671111687437,
          "C16_minus_C128": -0.027969898291808452,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "C128": 0.0007064274114932748,
          "C16": 0.0003643583006767881,
          "C16_minus_C128": -0.0003420691108164867,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "C128": 0.06117335326727746,
          "C16": 0.13750434643348694,
          "C16_minus_C128": 0.07633099316620948,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "C128": 0.0353473986390123,
          "C16": 0.09786852161243946,
          "C16_minus_C128": 0.06252112297342716,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "C128": 0.5652972157298544,
          "C16": 0.6446222106537942,
          "C16_minus_C128": 0.07932499492393985,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "C128": -0.001595090026982201,
          "C16": -0.0004383716199919221,
          "C16_minus_C128": 0.0011567184069902789,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "C128": 0.9969240446068932,
          "C16": 0.9988757308817966,
          "C16_minus_C128": 0.0019516862749033859,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "C128": 0.9686905034688366,
          "C16": 0.9276321220217154,
          "C16_minus_C128": -0.04105838144712115,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "C128": 0.998404909973018,
          "C16": 0.9995616283800082,
          "C16_minus_C128": 0.0011567184069901648,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "C128": 1.0014902535879975,
          "C16": 1.0006879229460544,
          "C16_minus_C128": -0.0008023306419431275,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "C128": 0.96239968637212,
          "C16": 0.9070470506816558,
          "C16_minus_C128": -0.055352635690464225,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "C128": 0.06932591903218935,
          "C16": 0.041389879283422096,
          "C16_minus_C128": -0.027936039748767252,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "C128": 0.06936660940868282,
          "C16": 0.04139671111687437,
          "C16_minus_C128": -0.027969898291808452,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "C128": 0.007897671269901662,
          "C16": 0.0046697786127950815,
          "C16_minus_C128": -0.0032278926571065805,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "C128": 0.99609375,
          "C16": 0.998046875,
          "C16_minus_C128": 0.001953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/geometry/702.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "C128": 511,
        "C16": 511
      }
    },
    "R128_to_R16": {
      "deltas": {
        "Delta_D_effective": -0.046562466372439104,
        "Delta_E_S": -0.005888903996551249,
        "Delta_KL": -0.019164766671436864,
        "Delta_abs_alpha": -0.0018944019268287487,
        "Delta_abs_norm_ratio_minus_1": -0.001663168589529862,
        "Delta_cos_effective": 0.003538746588462538,
        "Delta_orthogonal_ratio": -0.04652075267844284
      },
      "metrics": {
        "D_effective": {
          "R128": 0.08607485743880681,
          "R16": 0.03951239106636771,
          "R16_minus_R128": -0.046562466372439104,
          "metric_improved_with_KL_rescue": "YES"
        },
        "KL": {
          "R128": 0.021591667956042598,
          "R16": 0.0024269012846057317,
          "R16_minus_R128": -0.019164766671436864,
          "metric_improved_with_KL_rescue": "YES"
        },
        "R_distortion_same_codebook": {
          "R128": 0.16599892091711435,
          "R16": 0.21919748829115226,
          "R16_minus_R128": 0.053198567374037914,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_lost_same_codebook": {
          "R128": 0.12342810433078553,
          "R16": 0.17569904121527952,
          "R16_minus_R128": 0.05227093688449398,
          "metric_improved_with_KL_rescue": "NO"
        },
        "R_survival_same_codebook": {
          "R128": 0.4477260593332189,
          "R16": 0.6264077604633486,
          "R16_minus_R128": 0.17868170113012966,
          "metric_improved_with_KL_rescue": "YES"
        },
        "alpha": {
          "R128": -0.002322031942367059,
          "R16": -0.0004276300155383102,
          "R16_minus_R128": 0.0018944019268287487,
          "metric_improved_with_KL_rescue": "NA"
        },
        "cos_effective": {
          "R128": 0.9954416989570569,
          "R16": 0.9989804455455195,
          "R16_minus_R128": 0.003538746588462538,
          "metric_improved_with_KL_rescue": "YES"
        },
        "cosine_same_codebook": {
          "R128": 0.9109051301273462,
          "R16": 0.8804015600197236,
          "R16_minus_R128": -0.030503570107622613,
          "metric_improved_with_KL_rescue": "NO"
        },
        "effective_parallel_gain": {
          "R128": 0.9976779680576336,
          "R16": 0.9995723699844622,
          "R16_minus_R128": 0.0018944019268285484,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_effective": {
          "R128": 1.0022564748722331,
          "R16": 1.0005933062827033,
          "R16_minus_R128": -0.001663168589529862,
          "metric_improved_with_KL_rescue": "CLOSER_TO_IDEAL"
        },
        "norm_ratio_same_codebook": {
          "R128": 0.9005017558506714,
          "R16": 0.8575171361476774,
          "R16_minus_R128": -0.04298461970299394,
          "metric_improved_with_KL_rescue": "NA"
        },
        "orthogonal_ratio": {
          "R128": 0.08602941572670893,
          "R16": 0.039508663048266085,
          "R16_minus_R128": -0.04652075267844284,
          "metric_improved_with_KL_rescue": "YES"
        },
        "residual_to_update": {
          "R128": 0.08607485743880681,
          "R16": 0.03951239106636771,
          "R16_minus_R128": -0.046562466372439104,
          "metric_improved_with_KL_rescue": "YES"
        },
        "state_reconstruction_relative_error_E_S": {
          "R128": 0.010375736477605039,
          "R16": 0.00448683248105379,
          "R16_minus_R128": -0.005888903996551249,
          "metric_improved_with_KL_rescue": "YES"
        },
        "top1_agreement": {
          "R128": 0.9765625,
          "R16": 0.99609375,
          "R16_minus_R128": 0.01953125,
          "metric_improved_with_KL_rescue": "YES"
        }
      },
      "problem_id": "test/geometry/702.json",
      "role": "INT8-row truncated pathological candidate",
      "valid_token_count": {
        "R128": 511,
        "R16": 511
      }
    }
  }
]
```

## Across-Prompt Association
```json
{
  "C128_to_C16": {
    "Delta_KL_vs_D_effective": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": -0.5428571428571428
    },
    "Delta_KL_vs_E_S": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.8857142857142857
    },
    "Delta_KL_vs_abs_alpha": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": -0.6
    },
    "Delta_KL_vs_abs_norm_ratio_minus_1": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.4857142857142857
    },
    "Delta_KL_vs_cos_effective": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.6
    },
    "Delta_KL_vs_orthogonal_ratio": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": -0.5428571428571428
    }
  },
  "R128_to_R16": {
    "Delta_KL_vs_D_effective": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.08571428571428572
    },
    "Delta_KL_vs_E_S": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.3142857142857143
    },
    "Delta_KL_vs_abs_alpha": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": -0.14285714285714285
    },
    "Delta_KL_vs_abs_norm_ratio_minus_1": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.08571428571428572
    },
    "Delta_KL_vs_cos_effective": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": -0.08571428571428572
    },
    "Delta_KL_vs_orthogonal_ratio": {
      "N": 6,
      "label": "descriptive small-N association",
      "spearman": 0.08571428571428572
    }
  }
}
```

## Final Classification
```json
{
  "EFFECTIVE_UPDATE_METRIC_READY_FOR_CAUSAL_INTERVENTION": "YES",
  "FINAL_CLASSIFICATION": "EFFECTIVE_UPDATE_METRIC_MULTI_PROMPT_SUPPORTED",
  "METHOD_DESIGN_READY": "NO",
  "METHOD_DESIGN_READY_CANDIDATE": "NO",
  "table": {
    "EFFECTIVE_UPDATE_METRICS_OUTPERFORM_SAME_CODEBOOK_METRICS_AS_QUALITY_PROXY": "YES",
    "EFFECTIVE_UPDATE_METRICS_TRACK_COLUMN_KL_RESCUE": "YES",
    "EFFECTIVE_UPDATE_METRICS_TRACK_ROW_KL_RESCUE": "YES",
    "METHOD_DESIGN_READY": "NO",
    "ORTHOGONAL_RESIDUAL_DOMINANCE_STABLE": "YES",
    "PROMPT_MANIFEST_RECOVERED_FROM_CANONICAL_ARTIFACTS": "YES"
  }
}
```

## Hypothesis
- Runtime effective-update fidelity may be a more stable mechanism metric candidate than same-codebook representability.

## Supported Conclusion
- Small-N descriptive validation only; no causal proof or method design is claimed.

## Negative Result
- No new quantizer, causal intervention, residual propagation, replay, cadence, INT4, kernel, or end-to-end generation was run.

## Unresolved
- Whether effective-update metrics remain predictive under causal intervention is not tested here.
