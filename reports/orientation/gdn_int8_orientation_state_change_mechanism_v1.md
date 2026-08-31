# GDN INT8 Orientation State-Change Mechanism V1

## OBSERVATION
- FORMAL_STATUS: `COMPLETE`
- PROTOCOL_GATE: `PASS`; METRIC_GATE: `PASS`
- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`; exact original generation replay is not claimed.
- METHOD_DESIGN_READY: `NO`

## Table 1: Global INT8-row vs INT8-column
```json
{
  "INT8-column": {
    "KL": {
      "mean": 0.002287848176101701,
      "median": 0.001978345296106701,
      "p90": 0.0035942350352027785,
      "p95": 0.004145655566115493
    },
    "delta_norm": {
      "mean": 2.707832506577089,
      "median": 2.672793158748063,
      "p90": 2.8562313779866857,
      "p95": 2.861456694181542
    },
    "lost_update_fraction": {
      "mean": 0.03602562938844226,
      "median": 0.0363477862227223,
      "p90": 0.03689392705338315,
      "p95": 0.03689431546698135
    },
    "represented_update_norm_ratio": {
      "mean": 0.9626224295944859,
      "median": 0.9623227313523792,
      "p90": 0.9635518078387773,
      "p95": 0.9636737143954336
    },
    "residual_update_cosine": {
      "mean": -0.017267630101151243,
      "median": -0.01770661957364112,
      "p90": -0.015159573573807034,
      "p95": -0.01458639570828208
    },
    "residual_update_projection_coeff": {
      "mean": -0.001746061414352747,
      "median": -0.0018162570592515906,
      "p90": -0.0014024971640955444,
      "p95": -0.0013203066690432607
    },
    "saturation_fraction": {
      "mean": 0.007921596358944622,
      "median": 0.007921317292563658,
      "p90": 0.007922695642417571,
      "p95": 0.007923045884007985
    },
    "scale_max": {
      "mean": 0.030240371428780086,
      "median": 0.03070018812832375,
      "p90": 0.031533140074977914,
      "p95": 0.03153842121315038
    },
    "scale_mean": {
      "mean": 0.00030003658963169926,
      "median": 0.0003041451809066112,
      "p90": 0.00030740215153756605,
      "p95": 0.00030804356216486157
    },
    "scale_median": {
      "mean": 0.00013374840600091748,
      "median": 0.0001337259310460958,
      "p90": 0.00013709101504820917,
      "p95": 0.00013730599118500357
    },
    "scale_p95_over_median": {
      "mean": 6.464458077419242,
      "median": 6.473251826878675,
      "p90": 6.6162379716054005,
      "p95": 6.657140172510397
    },
    "state_change_error_E_Delta": {
      "mean": 0.22693185067593083,
      "median": 0.22762026957034132,
      "p90": 0.22864832133500534,
      "p95": 0.22881303572329292
    },
    "state_norm": {
      "mean": 20.333190174455876,
      "median": 20.658140638532277,
      "p90": 20.797892460604075,
      "p95": 20.851780053194755
    },
    "state_reconstruction_relative_error_E_S": {
      "mean": 0.007959056963755156,
      "median": 0.007906996034538531,
      "p90": 0.008372351754991262,
      "p95": 0.008446247439977354
    },
    "survival_fraction": {
      "mean": 0.5732338678346095,
      "median": 0.5694378839338838,
      "p90": 0.6045380423620388,
      "p95": 0.6097539768222809
    },
    "top1_agreement": {
      "mean": 0.9900895493946317,
      "median": 0.9912243077423329,
      "p90": 0.992431640625,
      "p95": 0.9925537109375
    },
    "update_cosine": {
      "mean": 0.9684830758989419,
      "median": 0.9683323772646892,
      "p90": 0.9690220796937705,
      "p95": 0.9691556243686941
    },
    "update_distortion_ratio": {
      "mean": 0.06154846283925151,
      "median": 0.06184979876363546,
      "p90": 0.06230112189326099,
      "p95": 0.06250951464644981
    },
    "within_group_dynamic_range_p95": {
      "mean": 32.54151234058204,
      "median": 32.692808515728046,
      "p90": 33.081620014098675,
      "p95": 33.14749856330447
    }
  },
  "INT8-row": {
    "KL": {
      "mean": 0.06735320581445192,
      "median": 0.07008501332433197,
      "p90": 0.0887525258257469,
      "p95": 0.09426197066902461
    },
    "delta_norm": {
      "mean": 2.731770098243826,
      "median": 2.6959102206215872,
      "p90": 2.87826883733054,
      "p95": 2.882767174990122
    },
    "lost_update_fraction": {
      "mean": 0.12748431073918537,
      "median": 0.12632028359342526,
      "p90": 0.13080797338355732,
      "p95": 0.13084555411346763
    },
    "represented_update_norm_ratio": {
      "mean": 0.8989076749096561,
      "median": 0.8991673794051358,
      "p90": 0.9009445134567992,
      "p95": 0.9012843141378002
    },
    "residual_update_cosine": {
      "mean": -0.02143481223733862,
      "median": -0.022057944420264083,
      "p90": -0.019022564476992625,
      "p95": -0.0186702122502164
    },
    "residual_update_projection_coeff": {
      "mean": -0.002403009452297841,
      "median": -0.0025332773240584524,
      "p90": -0.0019397124251319276,
      "p95": -0.0018897459162949951
    },
    "saturation_fraction": {
      "mean": 0.007856196381562519,
      "median": 0.007856158781136255,
      "p90": 0.007857177277338118,
      "p95": 0.007857310240052467
    },
    "scale_max": {
      "mean": 0.030371004782343933,
      "median": 0.030898560720262345,
      "p90": 0.03145896332637303,
      "p95": 0.031507172829192284
    },
    "scale_mean": {
      "mean": 0.0011047537391797428,
      "median": 0.00111509947482554,
      "p90": 0.0011303036129488584,
      "p95": 0.0011317229769872506
    },
    "scale_median": {
      "mean": 0.0005300410225923328,
      "median": 0.0005355610865783154,
      "p90": 0.0005466892234738352,
      "p95": 0.0005480841985053492
    },
    "scale_p95_over_median": {
      "mean": 8.333035919173321,
      "median": 8.334160705318759,
      "p90": 8.463738642123634,
      "p95": 8.479523702278641
    },
    "state_change_error_E_Delta": {
      "mean": 0.38822289772331486,
      "median": 0.38662650693832185,
      "p90": 0.3929648627739926,
      "p95": 0.3937997389257235
    },
    "state_norm": {
      "mean": 20.17466013415,
      "median": 20.459617441197693,
      "p90": 20.583583895046083,
      "p95": 20.637845734095187
    },
    "state_reconstruction_relative_error_E_S": {
      "mean": 0.010527537354616071,
      "median": 0.010446796475406804,
      "p90": 0.011075681414221696,
      "p95": 0.011141589167347902
    },
    "survival_fraction": {
      "mean": 0.4585950439944131,
      "median": 0.45506841597409375,
      "p90": 0.48551509271727666,
      "p95": 0.49048842655009
    },
    "top1_agreement": {
      "mean": 0.9393826213109536,
      "median": 0.9424036700780394,
      "p90": 0.9521484375,
      "p95": 0.95263671875
    },
    "update_cosine": {
      "mean": 0.908855285775006,
      "median": 0.9093961056000385,
      "p90": 0.9100135030769974,
      "p95": 0.9101144285357557
    },
    "update_distortion_ratio": {
      "mean": 0.16959594496442285,
      "median": 0.1685688067013172,
      "p90": 0.172746934219698,
      "p95": 0.17298581399161134
    },
    "within_group_dynamic_range_p95": {
      "mean": 4903.718713633861,
      "median": 4891.365389412175,
      "p90": 5286.100199728657,
      "p95": 5369.83866777086
    }
  }
}
```

## Table 2: 6 Prompts Paired Values
```json
[
  {
    "KL_column": 0.001074108306315155,
    "KL_row": 0.051120189860976106,
    "KL_row_minus_column": 0.05004608155466095,
    "KL_row_over_column": 47.59314266579826,
    "actual_token_count": 913,
    "lost_update_fraction_column": 0.03431434723224641,
    "lost_update_fraction_row": 0.12597160714915126,
    "lost_update_fraction_row_minus_column": 0.09165725991690485,
    "lost_update_fraction_row_over_column": 3.6711060331863536,
    "problem_id": "test/algebra/1214.json",
    "represented_update_norm_ratio_column": 0.9620109010145687,
    "represented_update_norm_ratio_row": 0.8970825160614555,
    "represented_update_norm_ratio_row_minus_column": -0.06492838495311315,
    "represented_update_norm_ratio_row_over_column": 0.932507641145607,
    "residual_update_cosine_column": -0.014013217842757126,
    "residual_update_cosine_row": -0.018317860023440182,
    "residual_update_cosine_row_minus_column": -0.004304642180683056,
    "residual_update_cosine_row_over_column": 1.3071844189525645,
    "role": "INT8-row non-truncated termination control",
    "scale_p95_over_median_column": 6.253698987144517,
    "scale_p95_over_median_row": 8.316032328132156,
    "scale_p95_over_median_row_minus_column": 2.0623333409876388,
    "scale_p95_over_median_row_over_column": 1.3297781593305171,
    "state_change_error_E_Delta_column": 0.22374278516191773,
    "state_change_error_E_Delta_row": 0.38610572593303955,
    "state_change_error_E_Delta_row_minus_column": 0.16236294077112182,
    "state_change_error_E_Delta_row_over_column": 1.7256678272491484,
    "state_reconstruction_relative_error_E_S_column": 0.008520143124963443,
    "state_reconstruction_relative_error_E_S_row": 0.011207496920474107,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.0026873537955106643,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.315411813639245,
    "survival_fraction_column": 0.6149699112825231,
    "survival_fraction_row": 0.4954617603829033,
    "survival_fraction_row_minus_column": -0.1195081508996198,
    "survival_fraction_row_over_column": 0.8056682957863989,
    "top1_agreement_column": 0.9912376779846659,
    "top1_agreement_row": 0.9463307776560789,
    "top1_agreement_row_minus_column": -0.04490690032858702,
    "top1_agreement_row_over_column": 0.9546961325966852,
    "update_cosine_column": 0.9692891690436178,
    "update_cosine_row": 0.9093899973082678,
    "update_cosine_row_minus_column": -0.05989917173535009,
    "update_cosine_row_over_column": 0.9382029907602788,
    "update_distortion_ratio_column": 0.05999624442836486,
    "update_distortion_ratio_row": 0.16850428775402068,
    "update_distortion_ratio_row_minus_column": 0.10850804332565582,
    "update_distortion_ratio_row_over_column": 2.8085805929938457,
    "within_group_dynamic_range_p95_column": 31.809234200067255,
    "within_group_dynamic_range_p95_row": 4774.546706745955,
    "within_group_dynamic_range_p95_row_minus_column": 4742.737472545888,
    "within_group_dynamic_range_p95_row_over_column": 150.09939179032042
  },
  {
    "KL_column": 0.002115199083873334,
    "KL_row": 0.06821841753513605,
    "KL_row_minus_column": 0.06610321845126271,
    "KL_row_over_column": 32.25153511801597,
    "actual_token_count": 1908,
    "lost_update_fraction_column": 0.036122581179905806,
    "lost_update_fraction_row": 0.13073281192373673,
    "lost_update_fraction_row_minus_column": 0.09461023074383093,
    "lost_update_fraction_row_over_column": 3.619143694982143,
    "problem_id": "test/algebra/1332.json",
    "represented_update_norm_ratio_column": 0.9625046006319298,
    "represented_update_norm_ratio_row": 0.8981227043759376,
    "represented_update_norm_ratio_row_minus_column": -0.06438189625599222,
    "represented_update_norm_ratio_row_over_column": 0.9331100379014059,
    "residual_update_cosine_column": -0.01742990468594276,
    "residual_update_cosine_row": -0.021459666277166823,
    "residual_update_cosine_row_minus_column": -0.0040297615912240625,
    "residual_update_cosine_row_over_column": 1.2311981427227237,
    "role": "INT8-row truncated pathological candidate",
    "scale_p95_over_median_column": 6.534433569795407,
    "scale_p95_over_median_row": 8.43216852181362,
    "scale_p95_over_median_row_minus_column": 1.8977349520182134,
    "scale_p95_over_median_row_over_column": 1.290420727634336,
    "state_change_error_E_Delta_column": 0.22793885545924295,
    "state_change_error_E_Delta_row": 0.3912951104705308,
    "state_change_error_E_Delta_row_minus_column": 0.16335625501128784,
    "state_change_error_E_Delta_row_over_column": 1.7166669968670483,
    "state_reconstruction_relative_error_E_S_column": 0.008061988054552082,
    "state_reconstruction_relative_error_E_S_row": 0.010593967731017269,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.002531979676465186,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.314063933031449,
    "survival_fraction_column": 0.5813338243550841,
    "survival_fraction_row": 0.46418676032946726,
    "survival_fraction_row_minus_column": -0.11714706402561681,
    "survival_fraction_row_over_column": 0.7984857252103358,
    "top1_agreement_column": 0.9868972746331237,
    "top1_agreement_row": 0.9360587002096437,
    "top1_agreement_row_minus_column": -0.050838574423480054,
    "top1_agreement_row_over_column": 0.9484864577801381,
    "update_cosine_column": 0.9683110236280297,
    "update_cosine_row": 0.9072304146871909,
    "update_cosine_row_minus_column": -0.061080608940838776,
    "update_cosine_row_over_column": 0.9369204651703908,
    "update_distortion_ratio_column": 0.06188433638688334,
    "update_distortion_ratio_row": 0.17226917467587136,
    "update_distortion_ratio_row_minus_column": 0.11038483828898801,
    "update_distortion_ratio_row_over_column": 2.7837282377707546,
    "within_group_dynamic_range_p95_column": 32.69240778592932,
    "within_group_dynamic_range_p95_row": 5008.184072078396,
    "within_group_dynamic_range_p95_row_minus_column": 4975.491664292466,
    "within_group_dynamic_range_p95_row_over_column": 153.19104376992073
  },
  {
    "KL_column": 0.0015078200876760908,
    "KL_row": 0.07773363613919146,
    "KL_row_minus_column": 0.07622581605151536,
    "KL_row_over_column": 51.553654692979634,
    "actual_token_count": 2048,
    "lost_update_fraction_column": 0.03689470388057955,
    "lost_update_fraction_row": 0.12561524222081588,
    "lost_update_fraction_row_minus_column": 0.08872053834023633,
    "lost_update_fraction_row_over_column": 3.4046957695447615,
    "problem_id": "test/counting_and_probability/119.json",
    "represented_update_norm_ratio_column": 0.9637956209520899,
    "represented_update_norm_ratio_row": 0.9016241148188011,
    "represented_update_norm_ratio_row_minus_column": -0.06217150613328881,
    "represented_update_norm_ratio_row_over_column": 0.9354930601657305,
    "residual_update_cosine_column": -0.019466759667039167,
    "residual_update_cosine_row": -0.02370163937425336,
    "residual_update_cosine_row_minus_column": -0.004234879707214194,
    "residual_update_cosine_row_over_column": 1.2175441511400908,
    "role": "INT8-row truncated pathological candidate",
    "scale_p95_over_median_column": 6.354069880402791,
    "scale_p95_over_median_row": 8.232031876352108,
    "scale_p95_over_median_row_minus_column": 1.8779619959493177,
    "scale_p95_over_median_row_over_column": 1.2955526192340636,
    "state_change_error_E_Delta_column": 0.22831889255843024,
    "state_change_error_E_Delta_row": 0.38481635078016585,
    "state_change_error_E_Delta_row_minus_column": 0.1564974582217356,
    "state_change_error_E_Delta_row_over_column": 1.6854336777307448,
    "state_reconstruction_relative_error_E_S_column": 0.007599795209098329,
    "state_reconstruction_relative_error_E_S_row": 0.010095172458301556,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.0024953772492032266,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.328347959457619,
    "survival_fraction_column": 0.5450503569514298,
    "survival_fraction_row": 0.43802959607218567,
    "survival_fraction_row_minus_column": -0.10702076087924411,
    "survival_fraction_row_over_column": 0.8036497737973578,
    "top1_agreement_column": 0.9921875,
    "top1_agreement_row": 0.953125,
    "top1_agreement_row_minus_column": -0.0390625,
    "top1_agreement_row_over_column": 0.9606299212598425,
    "update_cosine_column": 0.968349171767521,
    "update_cosine_row": 0.9102153539945139,
    "update_cosine_row_minus_column": -0.05813381777300708,
    "update_cosine_row_over_column": 0.9399660582484974,
    "update_distortion_ratio_column": 0.06184325936756178,
    "update_distortion_ratio_row": 0.16727065684295037,
    "update_distortion_ratio_row_minus_column": 0.1054273974753886,
    "update_distortion_ratio_row_over_column": 2.704751634269259,
    "within_group_dynamic_range_p95_column": 31.89098278377152,
    "within_group_dynamic_range_p95_row": 4538.744350095085,
    "within_group_dynamic_range_p95_row_minus_column": 4506.853367311313,
    "within_group_dynamic_range_p95_row_over_column": 142.3206171120174
  },
  {
    "KL_column": 0.004697076097028206,
    "KL_row": 0.09977141551230234,
    "KL_row_minus_column": 0.09507433941527413,
    "KL_row_over_column": 21.24117503129803,
    "actual_token_count": 2048,
    "lost_update_fraction_column": 0.03535600254619627,
    "lost_update_fraction_row": 0.13088313484337794,
    "lost_update_fraction_row_minus_column": 0.09552713229718165,
    "lost_update_fraction_row_over_column": 3.701864617538863,
    "problem_id": "test/geometry/477.json",
    "represented_update_norm_ratio_column": 0.9621408620728288,
    "represented_update_norm_ratio_row": 0.8961397476726107,
    "represented_update_norm_ratio_row_minus_column": -0.06600111440021805,
    "represented_update_norm_ratio_row_over_column": 0.931401817548809,
    "residual_update_cosine_column": -0.016305929304856942,
    "residual_update_cosine_row": -0.01972726893054507,
    "residual_update_cosine_row_minus_column": -0.0034213396256881287,
    "residual_update_cosine_row_over_column": 1.209821811546125,
    "role": "INT8-row truncated pathological candidate",
    "scale_p95_over_median_column": 6.698042373415394,
    "scale_p95_over_median_row": 8.495308762433647,
    "scale_p95_over_median_row_minus_column": 1.7972663890182528,
    "scale_p95_over_median_row_over_column": 1.2683271154198164,
    "state_change_error_E_Delta_column": 0.2253111370829741,
    "state_change_error_E_Delta_row": 0.39463461507745445,
    "state_change_error_E_Delta_row_minus_column": 0.16932347799448036,
    "state_change_error_E_Delta_row_over_column": 1.7515095799819453,
    "state_reconstruction_relative_error_E_S_column": 0.008224560385019081,
    "state_reconstruction_relative_error_E_S_row": 0.010943865907969285,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.002719305522950204,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.3306323250910017,
    "survival_fraction_column": 0.5941061734415544,
    "survival_fraction_row": 0.47556842505165003,
    "survival_fraction_row_minus_column": -0.11853774838990433,
    "survival_fraction_row_over_column": 0.800477164370746,
    "top1_agreement_column": 0.986328125,
    "top1_agreement_row": 0.9111328125,
    "top1_agreement_row_minus_column": -0.0751953125,
    "top1_agreement_row_over_column": 0.9237623762376238,
    "update_cosine_column": 0.9687549903439232,
    "update_cosine_row": 0.9070820826087735,
    "update_cosine_row_minus_column": -0.06167290773514966,
    "update_cosine_row_over_column": 0.9363379715718886,
    "update_distortion_ratio_column": 0.06099269129335131,
    "update_distortion_ratio_row": 0.17322469376352467,
    "update_distortion_ratio_row_minus_column": 0.11223200247017337,
    "update_distortion_ratio_row_over_column": 2.84008936300877,
    "within_group_dynamic_range_p95_column": 33.21337711251026,
    "within_group_dynamic_range_p95_row": 4528.636753426416,
    "within_group_dynamic_range_p95_row_minus_column": 4495.423376313906,
    "within_group_dynamic_range_p95_row_over_column": 136.34978274222664
  },
  {
    "KL_column": 0.0018414915083400682,
    "KL_row": 0.03532396672557771,
    "KL_row_minus_column": 0.03348247521723764,
    "KL_row_over_column": 19.182258818787034,
    "actual_token_count": 2048,
    "lost_update_fraction_column": 0.03657299126553879,
    "lost_update_fraction_row": 0.12503410826033126,
    "lost_update_fraction_row_minus_column": 0.08846111699479248,
    "lost_update_fraction_row_over_column": 3.4187553146123353,
    "problem_id": "test/geometry/702.json",
    "represented_update_norm_ratio_column": 0.961974598170033,
    "represented_update_norm_ratio_row": 0.9002120544343342,
    "represented_update_norm_ratio_row_minus_column": -0.06176254373569878,
    "represented_update_norm_ratio_row_over_column": 0.9357960762652259,
    "residual_update_cosine_column": -0.017983334461339477,
    "residual_update_cosine_row": -0.022656222563361347,
    "residual_update_cosine_row_minus_column": -0.00467288810202187,
    "residual_update_cosine_row_over_column": 1.2598454759359357,
    "role": "INT8-row truncated pathological candidate",
    "scale_p95_over_median_column": 6.44823949581049,
    "scale_p95_over_median_row": 8.17038494380304,
    "scale_p95_over_median_row_minus_column": 1.72214544799255,
    "scale_p95_over_median_row_over_column": 1.2670721906516424,
    "state_change_error_E_Delta_column": 0.22897775011158047,
    "state_change_error_E_Delta_row": 0.38533829613509435,
    "state_change_error_E_Delta_row_minus_column": 0.15636054602351387,
    "state_change_error_E_Delta_row_over_column": 1.6828634919651349,
    "state_reconstruction_relative_error_E_S_column": 0.007595850994373026,
    "state_reconstruction_relative_error_E_S_row": 0.01002509589013787,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.002429244895764843,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.3198120786682648,
    "survival_fraction_column": 0.5464009974643819,
    "survival_fraction_row": 0.43237365051155197,
    "survival_fraction_row_minus_column": -0.1140273469528299,
    "survival_fraction_row_over_column": 0.7913119714605518,
    "top1_agreement_column": 0.9912109375,
    "top1_agreement_row": 0.951171875,
    "top1_agreement_row_minus_column": -0.0400390625,
    "top1_agreement_row_over_column": 0.9596059113300492,
    "update_cosine_column": 0.9678785178487036,
    "update_cosine_row": 0.9098116521594809,
    "update_cosine_row_minus_column": -0.0580668656892227,
    "update_cosine_row_over_column": 0.9400060393753883,
    "update_distortion_ratio_column": 0.06271790739963863,
    "update_distortion_ratio_row": 0.1676735311015561,
    "update_distortion_ratio_row_minus_column": 0.10495562370191748,
    "update_distortion_ratio_row_over_column": 2.67345544603617,
    "within_group_dynamic_range_p95_column": 32.69320924552678,
    "within_group_dynamic_range_p95_row": 5453.577135813063,
    "within_group_dynamic_range_p95_row_minus_column": 5420.883926567536,
    "within_group_dynamic_range_p95_row_over_column": 166.81070049919447
  },
  {
    "KL_column": 0.002491393973377351,
    "KL_row": 0.07195160911352788,
    "KL_row_minus_column": 0.06946021514015054,
    "KL_row_over_column": 28.880060673820196,
    "actual_token_count": 2048,
    "lost_update_fraction_column": 0.03689315022618676,
    "lost_update_fraction_row": 0.12666896003769926,
    "lost_update_fraction_row_minus_column": 0.0897758098115125,
    "lost_update_fraction_row_over_column": 3.433400489280788,
    "problem_id": "test/intermediate_algebra/207.json",
    "represented_update_norm_ratio_column": 0.9633079947254647,
    "represented_update_norm_ratio_row": 0.9002649120947972,
    "represented_update_norm_ratio_row_minus_column": -0.06304308263066749,
    "represented_update_norm_ratio_row_over_column": 0.9345556322839049,
    "residual_update_cosine_column": -0.018406634644971987,
    "residual_update_cosine_row": -0.022746216255264923,
    "residual_update_cosine_row_minus_column": -0.004339581610292936,
    "residual_update_cosine_row_over_column": 1.2357618159970567,
    "role": "INT8-row non-truncated termination control",
    "scale_p95_over_median_column": 6.49826415794686,
    "scale_p95_over_median_row": 8.35228908250536,
    "scale_p95_over_median_row_minus_column": 1.8540249245584999,
    "scale_p95_over_median_row_over_column": 1.2853107967750395,
    "state_change_error_E_Delta_column": 0.22730168368143966,
    "state_change_error_E_Delta_row": 0.38714728794360415,
    "state_change_error_E_Delta_row_minus_column": 0.1598456042621645,
    "state_change_error_E_Delta_row_over_column": 1.7032310613509842,
    "state_reconstruction_relative_error_E_S_column": 0.007752004014524978,
    "state_reconstruction_relative_error_E_S_row": 0.010299625219796339,
    "state_reconstruction_relative_error_E_S_row_minus_column": 0.0025476212052713606,
    "state_reconstruction_relative_error_E_S_row_over_column": 1.328640336163122,
    "survival_fraction_column": 0.5575419435126835,
    "survival_fraction_row": 0.4459500716187202,
    "survival_fraction_row_minus_column": -0.1115918718939633,
    "survival_fraction_row_over_column": 0.7998502656304194,
    "top1_agreement_column": 0.99267578125,
    "top1_agreement_row": 0.9384765625,
    "top1_agreement_row_minus_column": -0.05419921875,
    "top1_agreement_row_over_column": 0.9454008853910477,
    "update_cosine_column": 0.9683155827618575,
    "update_cosine_row": 0.9094022138918092,
    "update_cosine_row_minus_column": -0.05891336887004828,
    "update_cosine_row_over_column": 0.9391589168667369,
    "update_distortion_ratio_column": 0.061856338159709146,
    "update_distortion_ratio_row": 0.1686333256486137,
    "update_distortion_ratio_row_minus_column": 0.10677698748890455,
    "update_distortion_ratio_row_over_column": 2.7262093209141014,
    "within_group_dynamic_range_p95_column": 32.9498629156871,
    "within_group_dynamic_range_p95_row": 5118.623263644249,
    "within_group_dynamic_range_p95_row_minus_column": 5085.673400728562,
    "within_group_dynamic_range_p95_row_over_column": 155.34581363029963
  }
]
```

## Table 3: Temporal Windows
```json
{
  "by_quantizer": {
    "INT8-column": {
      "1-128": {
        "KL": 0.00039915671418464443,
        "delta_norm": 2.929718060824534,
        "lost_update_fraction": 0.03263659444891815,
        "represented_update_norm_ratio": 0.96445074500728,
        "residual_update_cosine": -0.01350120803393016,
        "residual_update_projection_coeff": -0.0011896936495453918,
        "saturation_fraction": 0.007918946073436987,
        "scale_max": 0.02610548784580405,
        "scale_mean": 0.00027181025140316336,
        "scale_median": 0.00012680473792888976,
        "scale_p95_over_median": 6.161920323985135,
        "state_change_error_E_Delta": 0.2189642133152682,
        "state_norm": 17.970787311750133,
        "state_reconstruction_relative_error_E_S": 0.008971334856247362,
        "survival_fraction": 0.6333174532375517,
        "top1_agreement": 0.9986979166666666,
        "update_cosine": 0.9707586614485124,
        "update_distortion_ratio": 0.057166999169706516,
        "within_group_dynamic_range_p95": 31.60743204598769
      },
      "1025-1536": {
        "KL": 0.003139242185151616,
        "delta_norm": 2.666399936322719,
        "lost_update_fraction": 0.03659746033165271,
        "represented_update_norm_ratio": 0.9625654249085391,
        "residual_update_cosine": -0.018227028733369365,
        "residual_update_projection_coeff": -0.001906546545161012,
        "saturation_fraction": 0.007922027384241422,
        "scale_max": 0.03202070868447132,
        "scale_mean": 0.000308542981103565,
        "scale_median": 0.00013449977516197812,
        "scale_p95_over_median": 6.627609412451214,
        "state_change_error_E_Delta": 0.22835807791360013,
        "state_norm": 21.061663300714763,
        "state_reconstruction_relative_error_E_S": 0.007784011748204198,
        "survival_fraction": 0.5634516756050292,
        "top1_agreement": 0.983984375,
        "update_cosine": 0.9681291462499457,
        "update_distortion_ratio": 0.06223680171929221,
        "within_group_dynamic_range_p95": 32.8843616338602
      },
      "129-256": {
        "KL": 0.0008876593899181432,
        "delta_norm": 2.789360257900969,
        "lost_update_fraction": 0.035603777369319585,
        "represented_update_norm_ratio": 0.9627530613901233,
        "residual_update_cosine": -0.016084396674885846,
        "residual_update_projection_coeff": -0.0015121009555422247,
        "saturation_fraction": 0.007920686879919635,
        "scale_max": 0.027481932233942838,
        "scale_mean": 0.00028876241583464414,
        "scale_median": 0.0001329458933554406,
        "scale_p95_over_median": 6.258299760782993,
        "state_change_error_E_Delta": 0.225429413594751,
        "state_norm": 19.33988828250828,
        "state_reconstruction_relative_error_E_S": 0.008221557006332909,
        "survival_fraction": 0.5847079417564802,
        "top1_agreement": 0.9947916666666666,
        "update_cosine": 0.9687778551176761,
        "update_distortion_ratio": 0.06095829189238911,
        "within_group_dynamic_range_p95": 32.44967948528937
      },
      "1537-2048": {
        "KL": 0.003890616095424032,
        "delta_norm": 2.7125786078905243,
        "lost_update_fraction": 0.036392514420459106,
        "represented_update_norm_ratio": 0.9625433265512406,
        "residual_update_cosine": -0.018278893388165164,
        "residual_update_projection_coeff": -0.001942094501692506,
        "saturation_fraction": 0.007922981441434886,
        "scale_max": 0.0332511540148765,
        "scale_mean": 0.0003108201621418227,
        "scale_median": 0.00013522312747642372,
        "scale_p95_over_median": 6.594731341610226,
        "state_change_error_E_Delta": 0.22821995618876528,
        "state_norm": 21.40057576729007,
        "state_reconstruction_relative_error_E_S": 0.007790007132968959,
        "survival_fraction": 0.5644627573469312,
        "top1_agreement": 0.9892557123655914,
        "update_cosine": 0.9681497081714797,
        "update_distortion_ratio": 0.06220665108198994,
        "within_group_dynamic_range_p95": 32.701752888451416
      },
      "257-512": {
        "KL": 0.0013991804745964523,
        "delta_norm": 2.696109421687737,
        "lost_update_fraction": 0.036201323554308634,
        "represented_update_norm_ratio": 0.9628548088419054,
        "residual_update_cosine": -0.01718796519513433,
        "residual_update_projection_coeff": -0.0017030671720192174,
        "saturation_fraction": 0.00792134144446916,
        "scale_max": 0.02840684556878159,
        "scale_mean": 0.00029694188615923454,
        "scale_median": 0.00013478705992017414,
        "scale_p95_over_median": 6.369813654130016,
        "state_change_error_E_Delta": 0.22666260299802318,
        "state_norm": 19.881732747230366,
        "state_reconstruction_relative_error_E_S": 0.007952799088614662,
        "survival_fraction": 0.5687991474341186,
        "top1_agreement": 0.9895833333333334,
        "update_cosine": 0.9685415144946695,
        "update_distortion_ratio": 0.0614211443654944,
        "within_group_dynamic_range_p95": 32.56589819859557
      },
      "513-1024": {
        "KL": 0.0018820100461999093,
        "delta_norm": 2.6188645756254387,
        "lost_update_fraction": 0.03663886828581387,
        "represented_update_norm_ratio": 0.9622862451640808,
        "residual_update_cosine": -0.01793462286281407,
        "residual_update_projection_coeff": -0.001831990619333114,
        "saturation_fraction": 0.00792122033621314,
        "scale_max": 0.030023145272443496,
        "scale_mean": 0.00030123834279297126,
        "scale_median": 0.0001343485685618296,
        "scale_p95_over_median": 6.4564759895581005,
        "state_change_error_E_Delta": 0.22826721674709094,
        "state_norm": 20.367839537604834,
        "state_reconstruction_relative_error_E_S": 0.007745195329704445,
        "survival_fraction": 0.5593883115427927,
        "top1_agreement": 0.9926583281379884,
        "update_cosine": 0.9681101447715177,
        "update_distortion_ratio": 0.062263266577295695,
        "within_group_dynamic_range_p95": 32.566885334931975
      }
    },
    "INT8-row": {
      "1-128": {
        "KL": 0.020625739954812157,
        "delta_norm": 2.9456428515070026,
        "lost_update_fraction": 0.12164866473730833,
        "represented_update_norm_ratio": 0.9019849142078987,
        "residual_update_cosine": -0.018512266597271242,
        "residual_update_projection_coeff": -0.001899068016897646,
        "saturation_fraction": 0.007855881126116791,
        "scale_max": 0.02602998117901721,
        "scale_mean": 0.0009951826779297434,
        "scale_median": 0.0004772537729308383,
        "scale_p95_over_median": 8.39204590836871,
        "state_change_error_E_Delta": 0.37934159202764933,
        "state_norm": 17.904881141173767,
        "state_reconstruction_relative_error_E_S": 0.011553357736997541,
        "survival_fraction": 0.5076042137329556,
        "top1_agreement": 0.984375,
        "update_cosine": 0.9128041563120491,
        "update_distortion_ratio": 0.16277228505794653,
        "within_group_dynamic_range_p95": 3901.392824499156
      },
      "1025-1536": {
        "KL": 0.10163641978950991,
        "delta_norm": 2.692124443507555,
        "lost_update_fraction": 0.1284195176548845,
        "represented_update_norm_ratio": 0.8987495700561727,
        "residual_update_cosine": -0.02223747382314846,
        "residual_update_projection_coeff": -0.002557647354547145,
        "saturation_fraction": 0.007856333535164595,
        "scale_max": 0.03217841056417683,
        "scale_mean": 0.0011311663892119444,
        "scale_median": 0.0005413684920560453,
        "scale_p95_over_median": 8.353025724688743,
        "state_change_error_E_Delta": 0.3900302672890896,
        "state_norm": 20.83707288786536,
        "state_reconstruction_relative_error_E_S": 0.010363789457064903,
        "survival_fraction": 0.4503902926109732,
        "top1_agreement": 0.923046875,
        "update_cosine": 0.9083020720296189,
        "update_distortion_ratio": 0.17066277775087313,
        "within_group_dynamic_range_p95": 4748.163575906663
      },
      "129-256": {
        "KL": 0.024141445581138288,
        "delta_norm": 2.8083820175398646,
        "lost_update_fraction": 0.12695256707033528,
        "represented_update_norm_ratio": 0.8991120769601136,
        "residual_update_cosine": -0.020805396864902705,
        "residual_update_projection_coeff": -0.0022830880885424698,
        "saturation_fraction": 0.007855297066271303,
        "scale_max": 0.027410040604890027,
        "scale_mean": 0.0010741680918328422,
        "scale_median": 0.0005176970784616961,
        "scale_p95_over_median": 8.353509297098585,
        "state_change_error_E_Delta": 0.3870521435873404,
        "state_norm": 19.250738994218413,
        "state_reconstruction_relative_error_E_S": 0.010702999158690154,
        "survival_fraction": 0.4641147592208452,
        "top1_agreement": 0.9596354166666666,
        "update_cosine": 0.9092320432424054,
        "update_distortion_ratio": 0.16886339906227732,
        "within_group_dynamic_range_p95": 4995.89785322935
      },
      "1537-2048": {
        "KL": 0.08772425530586288,
        "delta_norm": 2.741584095139231,
        "lost_update_fraction": 0.12929616089363638,
        "represented_update_norm_ratio": 0.8978520905470466,
        "residual_update_cosine": -0.022177235564260316,
        "residual_update_projection_coeff": -0.0025443697779766483,
        "saturation_fraction": 0.007856560916402864,
        "scale_max": 0.03364136765656465,
        "scale_mean": 0.0011428777753189222,
        "scale_median": 0.0005471109686930201,
        "scale_p95_over_median": 8.310534647805346,
        "state_change_error_E_Delta": 0.3919595981132482,
        "state_norm": 21.20791484395841,
        "state_reconstruction_relative_error_E_S": 0.010397978206230524,
        "survival_fraction": 0.4524234750321452,
        "top1_agreement": 0.92606686827957,
        "update_cosine": 0.9075447777316686,
        "update_distortion_ratio": 0.1720247126967264,
        "within_group_dynamic_range_p95": 4402.7166951893405
      },
      "257-512": {
        "KL": 0.042857452151993464,
        "delta_norm": 2.7200551903948766,
        "lost_update_fraction": 0.12688499594452565,
        "represented_update_norm_ratio": 0.899514127833933,
        "residual_update_cosine": -0.021311476420738672,
        "residual_update_projection_coeff": -0.002359062464545133,
        "saturation_fraction": 0.007855937350541351,
        "scale_max": 0.028429668867696975,
        "scale_mean": 0.0010966243084989998,
        "scale_median": 0.000528886205170467,
        "scale_p95_over_median": 8.32505715192866,
        "state_change_error_E_Delta": 0.3869264277386906,
        "state_norm": 19.753246082206413,
        "state_reconstruction_relative_error_E_S": 0.010472523122140871,
        "survival_fraction": 0.4540788889345196,
        "top1_agreement": 0.9524739583333334,
        "update_cosine": 0.9093345418789087,
        "update_distortion_ratio": 0.16871877828364965,
        "within_group_dynamic_range_p95": 5578.998279678634
      },
      "513-1024": {
        "KL": 0.0610165073747598,
        "delta_norm": 2.641812578886831,
        "lost_update_fraction": 0.1275459293464806,
        "represented_update_norm_ratio": 0.8993380883521659,
        "residual_update_cosine": -0.022106398304453833,
        "residual_update_projection_coeff": -0.002518343077390571,
        "saturation_fraction": 0.007856430925055204,
        "scale_max": 0.030066039325442745,
        "scale_mean": 0.001108759369967521,
        "scale_median": 0.0005323590520228337,
        "scale_p95_over_median": 8.315248889461762,
        "state_change_error_E_Delta": 0.387484019937881,
        "state_norm": 20.19567362987378,
        "state_reconstruction_relative_error_E_S": 0.01028371373261122,
        "survival_fraction": 0.44729180609706476,
        "top1_agreement": 0.9396244090295095,
        "update_cosine": 0.9088940489213077,
        "update_distortion_ratio": 0.1693931082079361,
        "within_group_dynamic_range_p95": 5343.18805018215
      }
    }
  },
  "row_minus_column": {
    "1-128": {
      "KL_row_minus_column": 0.020226583240627514,
      "KL_row_over_column": 51.673288264596174,
      "lost_update_fraction_row_minus_column": 0.08901207028839017,
      "lost_update_fraction_row_over_column": 3.7273700516672865,
      "state_change_error_E_Delta_row_minus_column": 0.16037737871238114,
      "state_change_error_E_Delta_row_over_column": 1.732436484867357,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.002582022880750179,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.2878081046046495,
      "survival_fraction_row_minus_column": -0.12571323950459612,
      "survival_fraction_row_over_column": 0.8015004341630828,
      "update_cosine_row_minus_column": -0.057954505136463275,
      "update_cosine_row_over_column": 0.9402997805345493,
      "update_distortion_ratio_row_minus_column": 0.10560528588824,
      "update_distortion_ratio_row_over_column": 2.8473120405487635,
      "within_group_dynamic_range_p95_row_minus_column": 3869.7853924531687,
      "within_group_dynamic_range_p95_row_over_column": 123.43276792694732
    },
    "1025-1536": {
      "KL_row_minus_column": 0.09849717760435829,
      "KL_row_over_column": 32.376100279947394,
      "lost_update_fraction_row_minus_column": 0.09182205732323179,
      "lost_update_fraction_row_over_column": 3.5089734776982864,
      "state_change_error_E_Delta_row_minus_column": 0.1616721893754895,
      "state_change_error_E_Delta_row_over_column": 1.7079766603949897,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.0025797777088607055,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.3314200687654243,
      "survival_fraction_row_minus_column": -0.11306138299405599,
      "survival_fraction_row_over_column": 0.7993414734765821,
      "update_cosine_row_minus_column": -0.05982707422032685,
      "update_cosine_row_over_column": 0.938203415885094,
      "update_distortion_ratio_row_minus_column": 0.10842597603158091,
      "update_distortion_ratio_row_over_column": 2.7421521195870024,
      "within_group_dynamic_range_p95_row_minus_column": 4715.279214272803,
      "within_group_dynamic_range_p95_row_over_column": 144.38971413748223
    },
    "129-256": {
      "KL_row_minus_column": 0.023253786191220146,
      "KL_row_over_column": 27.196744444245137,
      "lost_update_fraction_row_minus_column": 0.0913487897010157,
      "lost_update_fraction_row_over_column": 3.5657050024060815,
      "state_change_error_E_Delta_row_minus_column": 0.16162272999258942,
      "state_change_error_E_Delta_row_over_column": 1.7169549324345699,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.0024814421523572452,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.3018214372832102,
      "survival_fraction_row_minus_column": -0.120593182535635,
      "survival_fraction_row_over_column": 0.7937548407956125,
      "update_cosine_row_minus_column": -0.05954581187527075,
      "update_cosine_row_over_column": 0.9385351228243777,
      "update_distortion_ratio_row_minus_column": 0.10790510716988821,
      "update_distortion_ratio_row_over_column": 2.770146502142403,
      "within_group_dynamic_range_p95_row_minus_column": 4963.448173744061,
      "within_group_dynamic_range_p95_row_over_column": 153.95831122135348
    },
    "1537-2048": {
      "KL_row_minus_column": 0.08383363921043885,
      "KL_row_over_column": 22.547651362734094,
      "lost_update_fraction_row_minus_column": 0.09290364647317728,
      "lost_update_fraction_row_over_column": 3.5528229624317684,
      "state_change_error_E_Delta_row_minus_column": 0.16373964192448293,
      "state_change_error_E_Delta_row_over_column": 1.7174641721035586,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.002607971073261566,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.334784170122782,
      "survival_fraction_row_minus_column": -0.11203928231478599,
      "survival_fraction_row_over_column": 0.8015116482770462,
      "update_cosine_row_minus_column": -0.06060493043981108,
      "update_cosine_row_over_column": 0.9374012821278704,
      "update_distortion_ratio_row_minus_column": 0.10981806161473645,
      "update_distortion_ratio_row_over_column": 2.7653749189936856,
      "within_group_dynamic_range_p95_row_minus_column": 4370.014942300889,
      "within_group_dynamic_range_p95_row_over_column": 134.63243729494863
    },
    "257-512": {
      "KL_row_minus_column": 0.04145827167739701,
      "KL_row_over_column": 30.630396099798556,
      "lost_update_fraction_row_minus_column": 0.09068367239021702,
      "lost_update_fraction_row_over_column": 3.504982235088031,
      "state_change_error_E_Delta_row_minus_column": 0.16026382474066744,
      "state_change_error_E_Delta_row_over_column": 1.7070589617382323,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.002519724033526209,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.3168348659949778,
      "survival_fraction_row_minus_column": -0.11472025849959905,
      "survival_fraction_row_over_column": 0.7983114795141522,
      "update_cosine_row_minus_column": -0.0592069726157608,
      "update_cosine_row_over_column": 0.9388699691962593,
      "update_distortion_ratio_row_minus_column": 0.10729763391815525,
      "update_distortion_ratio_row_over_column": 2.746916880604941,
      "within_group_dynamic_range_p95_row_minus_column": 5546.432381480039,
      "within_group_dynamic_range_p95_row_over_column": 171.31412269535477
    },
    "513-1024": {
      "KL_row_minus_column": 0.05913449732855989,
      "KL_row_over_column": 32.42092543446421,
      "lost_update_fraction_row_minus_column": 0.09090706106066673,
      "lost_update_fraction_row_over_column": 3.481164547756101,
      "state_change_error_E_Delta_row_minus_column": 0.15921680319079007,
      "state_change_error_E_Delta_row_over_column": 1.6975018378008901,
      "state_reconstruction_relative_error_E_S_row_minus_column": 0.002538518402906775,
      "state_reconstruction_relative_error_E_S_row_over_column": 1.3277539551741226,
      "survival_fraction_row_minus_column": -0.11209650544572791,
      "survival_fraction_row_over_column": 0.799608781355181,
      "update_cosine_row_minus_column": -0.05921609585021004,
      "update_cosine_row_over_column": 0.9388333071706572,
      "update_distortion_ratio_row_minus_column": 0.1071298416306404,
      "update_distortion_ratio_row_over_column": 2.720594622154073,
      "within_group_dynamic_range_p95_row_minus_column": 5310.621164847218,
      "within_group_dynamic_range_p95_row_over_column": 164.0681322524548
    }
  }
}
```

## Table 4: Truncated vs Non-Truncated Termination Controls
```json
{
  "INT8-row non-truncated termination controls": {
    "metrics": {
      "KL": 0.061535899487251994,
      "delta_norm": 2.801318902002367,
      "lost_update_fraction": 0.12632028359342526,
      "represented_update_norm_ratio": 0.8986737140781264,
      "residual_update_cosine": -0.020532038139352553,
      "residual_update_projection_coeff": -0.002239798949506911,
      "saturation_fraction": 0.007856007767787825,
      "scale_max": 0.028755490300396418,
      "scale_mean": 0.0010885142806920693,
      "scale_median": 0.0005227366986342595,
      "scale_p95_over_median": 8.334160705318759,
      "state_change_error_E_Delta": 0.38662650693832185,
      "state_norm": 19.71145849211262,
      "state_reconstruction_relative_error_E_S": 0.010753561070135222,
      "survival_fraction": 0.47070591600081174,
      "top1_agreement": 0.9424036700780394,
      "update_cosine": 0.9093961056000385,
      "update_distortion_ratio": 0.1685688067013172,
      "within_group_dynamic_range_p95": 4946.5849851951025
    },
    "prompt_count": 2
  },
  "INT8-row truncated pathological candidates": {
    "metrics": {
      "KL": 0.07026185897805189,
      "delta_norm": 2.6969956963645547,
      "lost_update_fraction": 0.12806632431206547,
      "represented_update_norm_ratio": 0.8990246553254209,
      "residual_update_cosine": -0.02188619928633165,
      "residual_update_projection_coeff": -0.002484614703693306,
      "saturation_fraction": 0.007856290688449865,
      "scale_max": 0.031178762023317687,
      "scale_mean": 0.0011128734684235793,
      "scale_median": 0.0005336931845713696,
      "scale_p95_over_median": 8.332473526100603,
      "state_change_error_E_Delta": 0.3890210931158114,
      "state_norm": 20.40626095516869,
      "state_reconstruction_relative_error_E_S": 0.010414525496856496,
      "survival_fraction": 0.45253960799121373,
      "top1_agreement": 0.9378720969274109,
      "update_cosine": 0.9085848758624898,
      "update_distortion_ratio": 0.17010951409597563,
      "within_group_dynamic_range_p95": 4882.28557785324
    },
    "prompt_count": 4
  }
}
```

## Table 5: Layer-Level Strongest Row-Column Gaps
```json
{
  "E_Delta_gap": [
    {
      "column_mean": 0.15546604784057966,
      "location": [
        "0"
      ],
      "prompt_count": 6,
      "row_mean": 0.5413475501427562,
      "row_minus_column": 0.3858815023021765,
      "score": 0.3858815023021765
    },
    {
      "column_mean": 0.1020264735600518,
      "location": [
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.46621408978917117,
      "row_minus_column": 0.3641876162291194,
      "score": 0.3641876162291194
    },
    {
      "column_mean": 0.1877001959794128,
      "location": [
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 0.5352356681034598,
      "row_minus_column": 0.347535472124047,
      "score": 0.347535472124047
    },
    {
      "column_mean": 0.1413137493049194,
      "location": [
        "14"
      ],
      "prompt_count": 6,
      "row_mean": 0.47087812328892564,
      "row_minus_column": 0.32956437398400623,
      "score": 0.32956437398400623
    },
    {
      "column_mean": 0.19403185486453192,
      "location": [
        "2"
      ],
      "prompt_count": 6,
      "row_mean": 0.5197414976416797,
      "row_minus_column": 0.32570964277714776,
      "score": 0.32570964277714776
    },
    {
      "column_mean": 0.16313676244968497,
      "location": [
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.46924688288535177,
      "row_minus_column": 0.30611012043566677,
      "score": 0.30611012043566677
    },
    {
      "column_mean": 0.21808829647280603,
      "location": [
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.46951711837372984,
      "row_minus_column": 0.2514288219009238,
      "score": 0.2514288219009238
    },
    {
      "column_mean": 0.21265906200204995,
      "location": [
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 0.4428721440242737,
      "row_minus_column": 0.23021308202222376,
      "score": 0.23021308202222376
    },
    {
      "column_mean": 0.27235933853665384,
      "location": [
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.48758475504079407,
      "row_minus_column": 0.21522541650414023,
      "score": 0.21522541650414023
    },
    {
      "column_mean": 0.19187775846703548,
      "location": [
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 0.39247403381073925,
      "row_minus_column": 0.20059627534370378,
      "score": 0.20059627534370378
    },
    {
      "column_mean": 0.20859777535294347,
      "location": [
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.382240814937721,
      "row_minus_column": 0.17364303958477753,
      "score": 0.17364303958477753
    },
    {
      "column_mean": 0.19842761973657208,
      "location": [
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.36151780057282346,
      "row_minus_column": 0.16309018083625137,
      "score": 0.16309018083625137
    },
    {
      "column_mean": 0.21712743044291008,
      "location": [
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 0.36862770829744207,
      "row_minus_column": 0.151500277854532,
      "score": 0.151500277854532
    },
    {
      "column_mean": 0.2009915903006716,
      "location": [
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.3512342352338093,
      "row_minus_column": 0.15024264493313771,
      "score": 0.15024264493313771
    },
    {
      "column_mean": 0.22528496149972846,
      "location": [
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.36824298542973066,
      "row_minus_column": 0.1429580239300022,
      "score": 0.1429580239300022
    }
  ],
  "R_distortion_gap": [
    {
      "column_mean": 0.03152672213726908,
      "location": [
        "0"
      ],
      "prompt_count": 6,
      "row_mean": 0.30120943450590554,
      "row_minus_column": 0.26968271236863645,
      "score": 0.26968271236863645
    },
    {
      "column_mean": 0.042210542075224955,
      "location": [
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 0.2952720661669895,
      "row_minus_column": 0.2530615240917645,
      "score": 0.2530615240917645
    },
    {
      "column_mean": 0.05014731974506945,
      "location": [
        "2"
      ],
      "prompt_count": 6,
      "row_mean": 0.2890027318967759,
      "row_minus_column": 0.23885541215170644,
      "score": 0.23885541215170644
    },
    {
      "column_mean": 0.011930926274557387,
      "location": [
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.22145522772875795,
      "row_minus_column": 0.20952430145420056,
      "score": 0.20952430145420056
    },
    {
      "column_mean": 0.02231983693968496,
      "location": [
        "14"
      ],
      "prompt_count": 6,
      "row_mean": 0.22787561823486727,
      "row_minus_column": 0.20555578129518232,
      "score": 0.20555578129518232
    },
    {
      "column_mean": 0.03311907487242806,
      "location": [
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.23709857425777991,
      "row_minus_column": 0.20397949938535187,
      "score": 0.20397949938535187
    },
    {
      "column_mean": 0.06437789119747095,
      "location": [
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.24401770923314928,
      "row_minus_column": 0.17963981803567833,
      "score": 0.17963981803567833
    },
    {
      "column_mean": 0.0974494119575141,
      "location": [
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.2701256372731166,
      "row_minus_column": 0.1726762253156025,
      "score": 0.1726762253156025
    },
    {
      "column_mean": 0.05491982444334506,
      "location": [
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 0.2151259548367135,
      "row_minus_column": 0.16020613039336845,
      "score": 0.16020613039336845
    },
    {
      "column_mean": 0.04076451377362886,
      "location": [
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 0.16359846163541683,
      "row_minus_column": 0.12283394786178797,
      "score": 0.12283394786178797
    },
    {
      "column_mean": 0.05030260200078778,
      "location": [
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.1623069655872613,
      "row_minus_column": 0.1120043635864735,
      "score": 0.1120043635864735
    },
    {
      "column_mean": 0.04529918241086011,
      "location": [
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.14034162594423136,
      "row_minus_column": 0.09504244353337124,
      "score": 0.09504244353337124
    },
    {
      "column_mean": 0.05057297838949442,
      "location": [
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 0.141921234490198,
      "row_minus_column": 0.09134825610070357,
      "score": 0.09134825610070357
    },
    {
      "column_mean": 0.054660898064288865,
      "location": [
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.14419472723598184,
      "row_minus_column": 0.08953382917169297,
      "score": 0.08953382917169297
    },
    {
      "column_mean": 0.04352958993525982,
      "location": [
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.1311534351017298,
      "row_minus_column": 0.08762384516646998,
      "score": 0.08762384516646998
    }
  ],
  "R_lost_gap": [
    {
      "column_mean": 0.01603739935882087,
      "location": [
        "0"
      ],
      "prompt_count": 6,
      "row_mean": 0.22856235577893122,
      "row_minus_column": 0.21252495642011035,
      "score": 0.21252495642011035
    },
    {
      "column_mean": 0.02151601045142401,
      "location": [
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 0.23031654409464244,
      "row_minus_column": 0.20880053364321843,
      "score": 0.20880053364321843
    },
    {
      "column_mean": 0.02932992460174774,
      "location": [
        "2"
      ],
      "prompt_count": 6,
      "row_mean": 0.2325926877762382,
      "row_minus_column": 0.20326276317449046,
      "score": 0.20326276317449046
    },
    {
      "column_mean": 0.047719019575205864,
      "location": [
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.22122004028513373,
      "row_minus_column": 0.17350102070992787,
      "score": 0.17350102070992787
    },
    {
      "column_mean": 0.012189913853829877,
      "location": [
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.18147914576242152,
      "row_minus_column": 0.16928923190859166,
      "score": 0.16928923190859166
    },
    {
      "column_mean": 0.028177148032602396,
      "location": [
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.1832798244562692,
      "row_minus_column": 0.1551026764236668,
      "score": 0.1551026764236668
    },
    {
      "column_mean": 0.009251718714689905,
      "location": [
        "14"
      ],
      "prompt_count": 6,
      "row_mean": 0.15699730493913322,
      "row_minus_column": 0.1477455862244433,
      "score": 0.1477455862244433
    },
    {
      "column_mean": 0.026899052286630365,
      "location": [
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 0.16672795611308455,
      "row_minus_column": 0.1398289038264542,
      "score": 0.1398289038264542
    },
    {
      "column_mean": 0.005166241323213731,
      "location": [
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.1333707511088937,
      "row_minus_column": 0.12820450978568,
      "score": 0.12820450978568
    },
    {
      "column_mean": 0.03243543892006616,
      "location": [
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.1398613330832149,
      "row_minus_column": 0.10742589416314875,
      "score": 0.10742589416314875
    },
    {
      "column_mean": 0.025126933420797165,
      "location": [
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 0.1303181348116953,
      "row_minus_column": 0.10519120139089813,
      "score": 0.10519120139089813
    },
    {
      "column_mean": 0.027304589069480953,
      "location": [
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.10426075129124822,
      "row_minus_column": 0.07695616222176727,
      "score": 0.07695616222176727
    },
    {
      "column_mean": 0.028981662830805572,
      "location": [
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.10288193808634637,
      "row_minus_column": 0.0739002752555408,
      "score": 0.0739002752555408
    },
    {
      "column_mean": 0.036406338105697066,
      "location": [
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.10677630292544764,
      "row_minus_column": 0.07036996481975058,
      "score": 0.07036996481975058
    },
    {
      "column_mean": 0.035382105540974824,
      "location": [
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 0.10260186546327861,
      "row_minus_column": 0.06721975992230378,
      "score": 0.06721975992230378
    }
  ],
  "scale_geometry_gap": [
    {
      "column_mean": 115.34234505968327,
      "location": [
        "0"
      ],
      "prompt_count": 6,
      "row_mean": 115935.29992524609,
      "row_minus_column": 115819.9575801864,
      "score": 115819.9575801864
    },
    {
      "column_mean": 31.34063914651556,
      "location": [
        "2"
      ],
      "prompt_count": 6,
      "row_mean": 154.89936363839112,
      "row_minus_column": 123.55872449187557,
      "score": 123.55872449187557
    },
    {
      "column_mean": 29.036020134873798,
      "location": [
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 135.64412373111838,
      "row_minus_column": 106.60810359624459,
      "score": 106.60810359624459
    },
    {
      "column_mean": 22.754489765728206,
      "location": [
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 126.43086743131043,
      "row_minus_column": 103.67637766558222,
      "score": 103.67637766558222
    },
    {
      "column_mean": 24.790682653310057,
      "location": [
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 122.24898824688775,
      "row_minus_column": 97.45830559357769,
      "score": 97.45830559357769
    },
    {
      "column_mean": 25.715366261336683,
      "location": [
        "14"
      ],
      "prompt_count": 6,
      "row_mean": 121.57253136012103,
      "row_minus_column": 95.85716509878435,
      "score": 95.85716509878435
    },
    {
      "column_mean": 36.887312554416575,
      "location": [
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 129.3571561760505,
      "row_minus_column": 92.46984362163391,
      "score": 92.46984362163391
    },
    {
      "column_mean": 26.07190594163001,
      "location": [
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 106.85966661076276,
      "row_minus_column": 80.78776066913275,
      "score": 80.78776066913275
    },
    {
      "column_mean": 22.647858078260885,
      "location": [
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 92.53571316612151,
      "row_minus_column": 69.88785508786063,
      "score": 69.88785508786063
    },
    {
      "column_mean": 39.31329584160974,
      "location": [
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 100.06423631745757,
      "row_minus_column": 60.750940475847834,
      "score": 60.750940475847834
    },
    {
      "column_mean": 27.23890656909914,
      "location": [
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 81.90259818706015,
      "row_minus_column": 54.66369161796101,
      "score": 54.66369161796101
    },
    {
      "column_mean": 21.76954241312471,
      "location": [
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 74.51882166198968,
      "row_minus_column": 52.749279248864966,
      "score": 52.749279248864966
    },
    {
      "column_mean": 21.096428683415386,
      "location": [
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 71.01986060688932,
      "row_minus_column": 49.92343192347394,
      "score": 49.92343192347394
    },
    {
      "column_mean": 30.606559873309223,
      "location": [
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 75.0622399233934,
      "row_minus_column": 44.45568005008418,
      "score": 44.45568005008418
    },
    {
      "column_mean": 30.314141225090538,
      "location": [
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 74.467158500865,
      "row_minus_column": 44.15301727577446,
      "score": 44.15301727577446
    }
  ],
  "update_cosine_gap": [
    {
      "column_mean": 0.9840533140418057,
      "location": [
        "0"
      ],
      "prompt_count": 6,
      "row_mean": 0.8345510765320765,
      "row_minus_column": -0.14950223750972924,
      "score": 0.14950223750972924
    },
    {
      "column_mean": 0.9786284679524462,
      "location": [
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 0.8382707804737347,
      "row_minus_column": -0.14035768747871147,
      "score": 0.14035768747871147
    },
    {
      "column_mean": 0.9743130733213133,
      "location": [
        "2"
      ],
      "prompt_count": 6,
      "row_mean": 0.8375680483664684,
      "row_minus_column": -0.13674502495484497,
      "score": 0.13674502495484497
    },
    {
      "column_mean": 0.9834078159490599,
      "location": [
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.8706965483361255,
      "row_minus_column": -0.11271126761293437,
      "score": 0.11271126761293437
    },
    {
      "column_mean": 0.9940685681455813,
      "location": [
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.8837427462111407,
      "row_minus_column": -0.11032582193444063,
      "score": 0.11032582193444063
    },
    {
      "column_mean": 0.9887959034159839,
      "location": [
        "14"
      ],
      "prompt_count": 6,
      "row_mean": 0.8787639340676011,
      "row_minus_column": -0.11003196934838277,
      "score": 0.11003196934838277
    },
    {
      "column_mean": 0.9490211899400153,
      "location": [
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.8453210129006404,
      "row_minus_column": -0.10370017703937484,
      "score": 0.10370017703937484
    },
    {
      "column_mean": 0.9670945580985273,
      "location": [
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.8651869716007935,
      "row_minus_column": -0.1019075864977338,
      "score": 0.1019075864977338
    },
    {
      "column_mean": 0.9720661725814498,
      "location": [
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 0.8828047338976198,
      "row_minus_column": -0.08926143868382996,
      "score": 0.08926143868382996
    },
    {
      "column_mean": 0.9793504308904845,
      "location": [
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 0.9134321273414924,
      "row_minus_column": -0.06591830354899209,
      "score": 0.06591830354899209
    },
    {
      "column_mean": 0.9743656510984117,
      "location": [
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.913176771406541,
      "row_minus_column": -0.0611888796918707,
      "score": 0.0611888796918707
    },
    {
      "column_mean": 0.9769380752148719,
      "location": [
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.9264418765382461,
      "row_minus_column": -0.0504961986766258,
      "score": 0.0504961986766258
    },
    {
      "column_mean": 0.9743180400688555,
      "location": [
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 0.9258935471025348,
      "row_minus_column": -0.048424492966320676,
      "score": 0.048424492966320676
    },
    {
      "column_mean": 0.9721673353552639,
      "location": [
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.9242444790628528,
      "row_minus_column": -0.04792285629241111,
      "score": 0.04792285629241111
    },
    {
      "column_mean": 0.9779867850382674,
      "location": [
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.9315422204647721,
      "row_minus_column": -0.04644456457349533,
      "score": 0.04644456457349533
    }
  ]
}
```

## Table 6: Head-Level Strongest Row-Column Gaps
```json
{
  "E_Delta_gap": [
    {
      "column_mean": 0.07635172841218738,
      "location": [
        "0",
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.5511567709767303,
      "row_minus_column": 0.4748050425645429,
      "score": 0.4748050425645429
    },
    {
      "column_mean": 0.05221060372527162,
      "location": [
        "0",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.5004095435920464,
      "row_minus_column": 0.4481989398667747,
      "score": 0.4481989398667747
    },
    {
      "column_mean": 0.08948365718351503,
      "location": [
        "2",
        "22"
      ],
      "prompt_count": 6,
      "row_mean": 0.5294815085034728,
      "row_minus_column": 0.4399978513199577,
      "score": 0.4399978513199577
    },
    {
      "column_mean": 0.1636399582435986,
      "location": [
        "5",
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.5895675998822596,
      "row_minus_column": 0.42592764163866104,
      "score": 0.42592764163866104
    },
    {
      "column_mean": 0.2142343372815433,
      "location": [
        "4",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.6388166103100666,
      "row_minus_column": 0.42458227302852336,
      "score": 0.42458227302852336
    },
    {
      "column_mean": 0.14845537817943932,
      "location": [
        "8",
        "30"
      ],
      "prompt_count": 6,
      "row_mean": 0.5729207148116046,
      "row_minus_column": 0.42446533663216524,
      "score": 0.42446533663216524
    },
    {
      "column_mean": 0.14609890250983584,
      "location": [
        "0",
        "7"
      ],
      "prompt_count": 6,
      "row_mean": 0.5689625262685464,
      "row_minus_column": 0.42286362375871056,
      "score": 0.42286362375871056
    },
    {
      "column_mean": 0.15208446222527675,
      "location": [
        "0",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.5629491391952044,
      "row_minus_column": 0.4108646769699277,
      "score": 0.4108646769699277
    },
    {
      "column_mean": 0.2155761140448933,
      "location": [
        "1",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.626322124169077,
      "row_minus_column": 0.41074601012418366,
      "score": 0.41074601012418366
    },
    {
      "column_mean": 0.1589028448712185,
      "location": [
        "0",
        "20"
      ],
      "prompt_count": 6,
      "row_mean": 0.5682749583342791,
      "row_minus_column": 0.40937211346306057,
      "score": 0.40937211346306057
    },
    {
      "column_mean": 0.1368348807124086,
      "location": [
        "10",
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.5437500907627039,
      "row_minus_column": 0.4069152100502953,
      "score": 0.4069152100502953
    },
    {
      "column_mean": 0.18238225169228586,
      "location": [
        "2",
        "24"
      ],
      "prompt_count": 6,
      "row_mean": 0.5889957891880031,
      "row_minus_column": 0.40661353749571727,
      "score": 0.40661353749571727
    },
    {
      "column_mean": 0.11976485563546402,
      "location": [
        "2",
        "17"
      ],
      "prompt_count": 6,
      "row_mean": 0.5234949959214065,
      "row_minus_column": 0.40373014028594245,
      "score": 0.40373014028594245
    },
    {
      "column_mean": 0.16034407893816283,
      "location": [
        "10",
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.5624664165340082,
      "row_minus_column": 0.40212233759584537,
      "score": 0.40212233759584537
    },
    {
      "column_mean": 0.14429276693325124,
      "location": [
        "2",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.5432448656248474,
      "row_minus_column": 0.3989520986915962,
      "score": 0.3989520986915962
    }
  ],
  "R_distortion_gap": [
    {
      "column_mean": 0.07043864996851519,
      "location": [
        "4",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.47017468345779334,
      "row_minus_column": 0.39973603348927816,
      "score": 0.39973603348927816
    },
    {
      "column_mean": 0.04128610206056496,
      "location": [
        "5",
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.39422719227278996,
      "row_minus_column": 0.352941090212225,
      "score": 0.352941090212225
    },
    {
      "column_mean": 0.06664652861722233,
      "location": [
        "1",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.40786555126521806,
      "row_minus_column": 0.34121902264799575,
      "score": 0.34121902264799575
    },
    {
      "column_mean": 0.05991114686278387,
      "location": [
        "2",
        "24"
      ],
      "prompt_count": 6,
      "row_mean": 0.4000702462377794,
      "row_minus_column": 0.34015909937499555,
      "score": 0.34015909937499555
    },
    {
      "column_mean": 0.09354515178504313,
      "location": [
        "4",
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.4106908232718598,
      "row_minus_column": 0.31714567148681666,
      "score": 0.31714567148681666
    },
    {
      "column_mean": 0.04483207288952793,
      "location": [
        "0",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.356187263041873,
      "row_minus_column": 0.31135519015234503,
      "score": 0.31135519015234503
    },
    {
      "column_mean": 0.012667175676390655,
      "location": [
        "0",
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.3219288066195474,
      "row_minus_column": 0.3092616309431568,
      "score": 0.3092616309431568
    },
    {
      "column_mean": 0.06013572474809297,
      "location": [
        "0",
        "20"
      ],
      "prompt_count": 6,
      "row_mean": 0.3684929944364488,
      "row_minus_column": 0.3083572696883558,
      "score": 0.3083572696883558
    },
    {
      "column_mean": 0.07268529325340585,
      "location": [
        "4",
        "15"
      ],
      "prompt_count": 6,
      "row_mean": 0.3800084248398059,
      "row_minus_column": 0.30732313158640007,
      "score": 0.30732313158640007
    },
    {
      "column_mean": 0.03541917979470895,
      "location": [
        "8",
        "30"
      ],
      "prompt_count": 6,
      "row_mean": 0.3393341291806478,
      "row_minus_column": 0.3039149493859389,
      "score": 0.3039149493859389
    },
    {
      "column_mean": 0.04511912780310338,
      "location": [
        "0",
        "7"
      ],
      "prompt_count": 6,
      "row_mean": 0.3486591871195332,
      "row_minus_column": 0.3035400593164298,
      "score": 0.3035400593164298
    },
    {
      "column_mean": 0.16365895611599535,
      "location": [
        "0",
        "28"
      ],
      "prompt_count": 6,
      "row_mean": 0.46642684389305183,
      "row_minus_column": 0.3027678877770565,
      "score": 0.3027678877770565
    },
    {
      "column_mean": 0.04782841555403475,
      "location": [
        "2",
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.3500497312180529,
      "row_minus_column": 0.3022213156640181,
      "score": 0.3022213156640181
    },
    {
      "column_mean": 0.03572791419543777,
      "location": [
        "10",
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.33728638887808665,
      "row_minus_column": 0.3015584746826489,
      "score": 0.3015584746826489
    },
    {
      "column_mean": 0.041929657874589014,
      "location": [
        "2",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.33862013368732713,
      "row_minus_column": 0.2966904758127381,
      "score": 0.2966904758127381
    }
  ],
  "R_lost_gap": [
    {
      "column_mean": 0.05182903902597164,
      "location": [
        "4",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.44114970962887706,
      "row_minus_column": 0.3893206706029054,
      "score": 0.3893206706029054
    },
    {
      "column_mean": 0.11594447774843901,
      "location": [
        "0",
        "28"
      ],
      "prompt_count": 6,
      "row_mean": 0.45422041940826324,
      "row_minus_column": 0.3382759416598242,
      "score": 0.3382759416598242
    },
    {
      "column_mean": 0.020328569851910578,
      "location": [
        "5",
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.34207067228372745,
      "row_minus_column": 0.3217421024318169,
      "score": 0.3217421024318169
    },
    {
      "column_mean": 0.04203007227382841,
      "location": [
        "1",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.3543979420085112,
      "row_minus_column": 0.3123678697346828,
      "score": 0.3123678697346828
    },
    {
      "column_mean": 0.044060549682087724,
      "location": [
        "2",
        "24"
      ],
      "prompt_count": 6,
      "row_mean": 0.3543470153282496,
      "row_minus_column": 0.31028646564616186,
      "score": 0.31028646564616186
    },
    {
      "column_mean": 0.02251995572355202,
      "location": [
        "0",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.3312063518904818,
      "row_minus_column": 0.3086863961669298,
      "score": 0.3086863961669298
    },
    {
      "column_mean": 0.04426742027845793,
      "location": [
        "0",
        "25"
      ],
      "prompt_count": 6,
      "row_mean": 0.34746268902016103,
      "row_minus_column": 0.3031952687417031,
      "score": 0.3031952687417031
    },
    {
      "column_mean": 0.02654629625310857,
      "location": [
        "0",
        "7"
      ],
      "prompt_count": 6,
      "row_mean": 0.3283194720347579,
      "row_minus_column": 0.30177317578164936,
      "score": 0.30177317578164936
    },
    {
      "column_mean": 0.07342363411392393,
      "location": [
        "4",
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.37371956989090865,
      "row_minus_column": 0.3002959357769847,
      "score": 0.3002959357769847
    },
    {
      "column_mean": 0.04083530479172263,
      "location": [
        "0",
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.33984319831742926,
      "row_minus_column": 0.2990078935257066,
      "score": 0.2990078935257066
    },
    {
      "column_mean": 0.05279334927122769,
      "location": [
        "4",
        "15"
      ],
      "prompt_count": 6,
      "row_mean": 0.3457634914906758,
      "row_minus_column": 0.2929701422194481,
      "score": 0.2929701422194481
    },
    {
      "column_mean": 0.06230341468289836,
      "location": [
        "2",
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.3517859686209586,
      "row_minus_column": 0.28948255393806027,
      "score": 0.28948255393806027
    },
    {
      "column_mean": 0.07460772555341054,
      "location": [
        "6",
        "26"
      ],
      "prompt_count": 6,
      "row_mean": 0.3530231129684262,
      "row_minus_column": 0.27841538741501565,
      "score": 0.27841538741501565
    },
    {
      "column_mean": 0.06381288536066342,
      "location": [
        "5",
        "21"
      ],
      "prompt_count": 6,
      "row_mean": 0.341658907396779,
      "row_minus_column": 0.2778460220361156,
      "score": 0.2778460220361156
    },
    {
      "column_mean": 0.006266738104861398,
      "location": [
        "0",
        "6"
      ],
      "prompt_count": 6,
      "row_mean": 0.2830043616421482,
      "row_minus_column": 0.2767376235372868,
      "score": 0.2767376235372868
    }
  ],
  "scale_geometry_gap": [
    {
      "column_mean": 10.100422742063499,
      "location": [
        "0",
        "11"
      ],
      "prompt_count": 6,
      "row_mean": 1609790.9032670485,
      "row_minus_column": 1609780.8028443065,
      "score": 1609780.8028443065
    },
    {
      "column_mean": 7.950390539678124,
      "location": [
        "0",
        "19"
      ],
      "prompt_count": 6,
      "row_mean": 344972.32120736875,
      "row_minus_column": 344964.3708168291,
      "score": 344964.3708168291
    },
    {
      "column_mean": 8.620440130481624,
      "location": [
        "0",
        "18"
      ],
      "prompt_count": 6,
      "row_mean": 321872.53524566937,
      "row_minus_column": 321863.9148055389,
      "score": 321863.9148055389
    },
    {
      "column_mean": 11.188950351189307,
      "location": [
        "0",
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 281937.72853171243,
      "row_minus_column": 281926.5395813612,
      "score": 281926.5395813612
    },
    {
      "column_mean": 8.942488582053963,
      "location": [
        "0",
        "22"
      ],
      "prompt_count": 6,
      "row_mean": 244397.67793264517,
      "row_minus_column": 244388.73544406312,
      "score": 244388.73544406312
    },
    {
      "column_mean": 9.002681182584531,
      "location": [
        "0",
        "23"
      ],
      "prompt_count": 6,
      "row_mean": 225545.20479853323,
      "row_minus_column": 225536.20211735065,
      "score": 225536.20211735065
    },
    {
      "column_mean": 10.763607067677398,
      "location": [
        "0",
        "26"
      ],
      "prompt_count": 6,
      "row_mean": 205051.6129102571,
      "row_minus_column": 205040.84930318943,
      "score": 205040.84930318943
    },
    {
      "column_mean": 9.337566486728766,
      "location": [
        "0",
        "29"
      ],
      "prompt_count": 6,
      "row_mean": 191756.79547771232,
      "row_minus_column": 191747.4579112256,
      "score": 191747.4579112256
    },
    {
      "column_mean": 10.798584562206509,
      "location": [
        "0",
        "27"
      ],
      "prompt_count": 6,
      "row_mean": 169019.79719346968,
      "row_minus_column": 169008.99860890748,
      "score": 169008.99860890748
    },
    {
      "column_mean": 12.393074187869159,
      "location": [
        "0",
        "5"
      ],
      "prompt_count": 6,
      "row_mean": 152550.98253764192,
      "row_minus_column": 152538.58946345406,
      "score": 152538.58946345406
    },
    {
      "column_mean": 10.532491354882561,
      "location": [
        "0",
        "9"
      ],
      "prompt_count": 6,
      "row_mean": 140657.7513740708,
      "row_minus_column": 140647.2188827159,
      "score": 140647.2188827159
    },
    {
      "column_mean": 215.04198188143286,
      "location": [
        "0",
        "24"
      ],
      "prompt_count": 6,
      "row_mean": 101363.86805552442,
      "row_minus_column": 101148.826073643,
      "score": 101148.826073643
    },
    {
      "column_mean": 197.78478644245902,
      "location": [
        "0",
        "25"
      ],
      "prompt_count": 6,
      "row_mean": 99682.73077724323,
      "row_minus_column": 99484.94599080077,
      "score": 99484.94599080077
    },
    {
      "column_mean": 9.909312113747236,
      "location": [
        "0",
        "28"
      ],
      "prompt_count": 6,
      "row_mean": 97402.20993188616,
      "row_minus_column": 97392.30061977242,
      "score": 97392.30061977242
    },
    {
      "column_mean": 9.973122346942686,
      "location": [
        "0",
        "8"
      ],
      "prompt_count": 6,
      "row_mean": 79536.89407976491,
      "row_minus_column": 79526.92095741797,
      "score": 79526.92095741797
    }
  ],
  "update_cosine_gap": [
    {
      "column_mean": 0.9635485935224278,
      "location": [
        "4",
        "4"
      ],
      "prompt_count": 6,
      "row_mean": 0.6895340613796325,
      "row_minus_column": -0.27401453214279536,
      "score": 0.27401453214279536
    },
    {
      "column_mean": 0.9127760201601713,
      "location": [
        "0",
        "28"
      ],
      "prompt_count": 6,
      "row_mean": 0.6882779474175414,
      "row_minus_column": -0.2244980727426299,
      "score": 0.2244980727426299
    },
    {
      "column_mean": 0.9659694220986658,
      "location": [
        "0",
        "25"
      ],
      "prompt_count": 6,
      "row_mean": 0.7422443682175611,
      "row_minus_column": -0.22372505388110464,
      "score": 0.22372505388110464
    },
    {
      "column_mean": 0.9682586155694842,
      "location": [
        "2",
        "24"
      ],
      "prompt_count": 6,
      "row_mean": 0.7475010604969206,
      "row_minus_column": -0.22075755507256367,
      "score": 0.22075755507256367
    },
    {
      "column_mean": 0.9793968583361877,
      "location": [
        "5",
        "16"
      ],
      "prompt_count": 6,
      "row_mean": 0.7627223954523409,
      "row_minus_column": -0.21667446288384684,
      "score": 0.21667446288384684
    },
    {
      "column_mean": 0.9664200943739423,
      "location": [
        "1",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.7609294243470942,
      "row_minus_column": -0.20549067002684807,
      "score": 0.20549067002684807
    },
    {
      "column_mean": 0.6440471516045666,
      "location": [
        "30",
        "22"
      ],
      "prompt_count": 6,
      "row_mean": 0.44060135053909893,
      "row_minus_column": -0.20344580106546767,
      "score": 0.20344580106546767
    },
    {
      "column_mean": 0.9502549810703694,
      "location": [
        "4",
        "10"
      ],
      "prompt_count": 6,
      "row_mean": 0.7473162426611184,
      "row_minus_column": -0.20293873840925103,
      "score": 0.20293873840925103
    },
    {
      "column_mean": 0.9620308459096399,
      "location": [
        "4",
        "15"
      ],
      "prompt_count": 6,
      "row_mean": 0.7674641731891619,
      "row_minus_column": -0.194566672720478,
      "score": 0.194566672720478
    },
    {
      "column_mean": 0.9544009869328081,
      "location": [
        "2",
        "13"
      ],
      "prompt_count": 6,
      "row_mean": 0.7615054325552149,
      "row_minus_column": -0.1928955543775932,
      "score": 0.1928955543775932
    },
    {
      "column_mean": 0.9771381664432203,
      "location": [
        "0",
        "1"
      ],
      "prompt_count": 6,
      "row_mean": 0.7843276167134915,
      "row_minus_column": -0.19281054972972878,
      "score": 0.19281054972972878
    },
    {
      "column_mean": 0.9680731061061939,
      "location": [
        "0",
        "20"
      ],
      "prompt_count": 6,
      "row_mean": 0.7792169725927133,
      "row_minus_column": -0.18885613351348063,
      "score": 0.18885613351348063
    },
    {
      "column_mean": 0.9767653922424349,
      "location": [
        "0",
        "7"
      ],
      "prompt_count": 6,
      "row_mean": 0.7887013392884903,
      "row_minus_column": -0.18806405295394457,
      "score": 0.18806405295394457
    },
    {
      "column_mean": 0.9551108240915536,
      "location": [
        "5",
        "21"
      ],
      "prompt_count": 6,
      "row_mean": 0.7700016157072378,
      "row_minus_column": -0.18510920838431577,
      "score": 0.18510920838431577
    },
    {
      "column_mean": 0.9750352185050817,
      "location": [
        "2",
        "12"
      ],
      "prompt_count": 6,
      "row_mean": 0.7900719875697811,
      "row_minus_column": -0.18496323093530054,
      "score": 0.18496323093530054
    }
  ]
}
```

## Table 7: Snapshot vs State-Change Separation
```json
{
  "E_Delta_row_over_column": 1.710746625328126,
  "E_S_row_over_column": 1.3227116481962056,
  "R_distortion_row_over_column": 2.755486280906204,
  "R_lost_row_over_column": 3.5387115479538265
}
```

## Final Classification
```json
{
  "FINAL_CLASSIFICATION": "INT8_ORIENTATION_X_RECURRENCE_INTERACTION",
  "table": {
    "INT8_AXIS_SCALE_GEOMETRY_DIFFERENCE_REPLICATED": "YES",
    "INT8_ORIENTATION_X_RECURRENCE_INTERACTION_SUPPORTED": "PARTIAL",
    "INT8_PROGRESSIVE_RECURRENT_ACCUMULATION_SIGNAL": "YES",
    "INT8_ROW_TRUNCATION_PHENOTYPE_ASSOCIATED_WITH_STATE_CHANGE_METRICS": "YES",
    "INT8_STATE_CHANGE_PRESERVATION_DIFFERENCE_REPLICATED": "YES",
    "STATE_CHANGE_SEPARATION_EXCEEDS_SNAPSHOT_SEPARATION": "YES"
  }
}
```

## HYPOTHESIS
- Axis scale geometry may be associated with state-change preservation differences under recurrent feedback.

## SUPPORTED CONCLUSION
- Formal evidence supports mechanistic association only; no causal intervention was run.

## NEGATIVE RESULT
- This experiment does not validate method design, replay readiness, or termination causality.

## PROPOSED FOLLOW-UP
- GDN_INT8_AXIS_RESCUE_CAUSAL_DIAGNOSTIC_V1
